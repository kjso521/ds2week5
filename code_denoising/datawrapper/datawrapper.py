import glob
import random
from dataclasses import dataclass
from enum import IntEnum
from typing import Literal, TypedDict
from pathlib import Path
import re
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import itertools
from torch import Tensor

from .noise_simulator import NoiseSimulator, NoisyType
from dataset.forward_simulator import ForwardSimulator
from ..common.logger import logger


prob_flip: float = 0.5


class DataKey(IntEnum):
    image_gt = 0
    image_noise = 1
    name = 2


class LoaderConfig(TypedDict):
    data_type: str
    batch: int
    num_workers: int
    shuffle: bool
    augmentation_mode: str
    training_phase: str
    noise_type: str
    noise_levels: list[float]
    conv_directions: list[tuple[float, float]]


# Base class for all data wrappers to share common utilities
class BaseDataWrapper(Dataset):
    def _load_from_npy(self, npy_path: str) -> np.ndarray:
        """Loads a numpy array from a file."""
        try:
            return np.load(npy_path)
        except Exception as e:
            logger.error(f"Error loading npy file at {npy_path}: {e}")
            return np.array([]) # Return empty array on failure

    def _augment(self, img: np.ndarray) -> np.ndarray:
        """Applies random flips for data augmentation."""
        if random.random() < prob_flip:
            img = np.fliplr(img)
        if random.random() < prob_flip:
            img = np.flipud(img)
        return img.copy()

    def __getitem__(self, index):
        raise NotImplementedError("Each child class must implement its own __getitem__ method.")

    def __len__(self):
        raise NotImplementedError("Each child class must implement its own __len__ method.")


class RandomDataWrapper(BaseDataWrapper):
    file_list: list[str]
    training_mode: bool
    
    # --- 재설계: 실시간 증강을 위한 멤버 변수 ---
    augmentation_mode: Literal['none', 'noise_only', 'conv_only', 'both']
    noise_levels: list[float]
    conv_directions: list[tuple[float, float]]
    noise_type: NoisyType
    noise_simulator: NoiseSimulator
    forward_simulator: ForwardSimulator

    def __init__(
        self,
        file_path: list[str],
        data_type: str,
        training_mode: bool,
        # --- 재설계: 실시간 증강 파라미터를 직접 받음 ---
        augmentation_mode: Literal['none', 'noise_only', 'conv_only', 'both'],
        noise_type: NoisyType,
        noise_levels: list[float],
        conv_directions: list[tuple[float, float]],
    ):
        super().__init__()
        self.training_mode = training_mode
        self.augmentation_mode = augmentation_mode
        self.noise_type = noise_type
        self.noise_levels = noise_levels
        self.conv_directions = conv_directions

        # 시뮬레이터들을 미리 초기화해둡니다.
        self.noise_simulator = NoiseSimulator(noise_type=self.noise_type, noise_sigma=0.0) # sigma는 __getitem__에서 매번 덮어씀
        self.forward_simulator = ForwardSimulator()
        self.num_augmentations = 1

        # Calculate the total number of augmentation combinations
        num_noise = len(self.noise_levels) if self.noise_levels else 1
        num_conv = len(self.conv_directions) if self.conv_directions else 1
        
        if self.augmentation_mode == 'both':
            self.num_augmentations = num_noise * num_conv
        elif self.augmentation_mode == 'noise_only':
            self.num_augmentations = num_noise
        elif self.augmentation_mode == 'conv_only':
            self.num_augmentations = num_conv
            
    def get_params_for_index(self, index: int) -> dict:
        """
        Determines the augmentation parameters for a given virtual index.
        This enables the "Hybrid On-the-fly" strategy.
        """
        original_index = index // self.num_augmentations
        augmentation_index = index % self.num_augmentations

        num_noise_levels = len(self.noise_levels) if self.noise_levels else 1
        
        params = {"noise_level": 0.0, "conv_dir": (0.0, 0.0)}

        if self.augmentation_mode in ["noise_only", "both"]:
            noise_idx = augmentation_index % num_noise_levels
            params["noise_level"] = self.noise_levels[noise_idx]

        if self.augmentation_mode in ["conv_only", "both"]:
            conv_idx = augmentation_index // num_noise_levels if self.augmentation_mode == 'both' else augmentation_index
            params["conv_dir"] = self.conv_directions[conv_idx]
            
        return params, original_index

    def __len__(self) -> int:
        # Virtual dataset length is original length multiplied by number of augmentations
        if self.training_mode:
            return len(self.file_list) * self.num_augmentations
        else:
            # For validation/test, we don't need to augment
            return len(self.file_list)

    def __getitem__(self, index: int) -> dict[DataKey, Tensor | str]:
        if self.training_mode:
            params, original_index = self.get_params_for_index(index)
            gt_path = self.file_list[original_index]
        else:
            # For validation/test, use original index and no augmentation
            gt_path = self.file_list[index]
            params = {"noise_level": 0.0, "conv_dir": (0.0, 0.0)}
            # NOTE: For validation, we still need to apply a *consistent* degradation
            # to compare apples-to-apples. Let's use the first combination.
            if self.augmentation_mode in ["noise_only", "both"]:
                 params["noise_level"] = self.noise_levels[0]
            if self.augmentation_mode in ["conv_only", "both"]:
                 params["conv_dir"] = self.conv_directions[0]

        image_gt_np = self._load_from_npy(gt_path)
        image_gt = torch.from_numpy(image_gt_np).unsqueeze(0).float()

        image_noise = self.forward_simulator(
            image_gt, B0_dir=params["conv_dir"], noise_std=params["noise_level"]
        )

        return {
            DataKey.image_gt: image_gt.squeeze(0),
            DataKey.image_noise: image_noise.squeeze(0),
            DataKey.name: gt_path.name,
        }

class ControlledDataWrapper(BaseDataWrapper):
    def __init__(self, file_path: list, training_mode: bool, data_type: str, 
                 augmentation_mode: str, noise_type: str, noise_levels: list, 
                 conv_directions: list, **kwargs):
        super().__init__()
        self.training_mode = training_mode

        # Find all files matching the pattern
        self.file_list = []
        for path in file_path:
            self.file_list.extend(glob.glob(os.path.join(path, kwargs.get("data_type", "*.npy"))))
        
        if not self.file_list:
            raise FileNotFoundError(f"No data files found in {file_path}")

        # Store augmentation mode for use in __getitem__
        self.augmentation_mode = augmentation_mode

        # Augmentation settings
        self.num_augmentations = 1
        # --- 💡 수정된 부분: 모든 증강 모드를 올바르게 처리하도록 로직 변경 ---
        if self.training_mode:
            num_noise = len(noise_levels) if noise_levels else 1
            num_conv = len(conv_directions) if conv_directions else 1

            if self.augmentation_mode == 'both':
                self.num_augmentations = num_noise * num_conv
            elif self.augmentation_mode == 'noise_only':
                self.num_augmentations = num_noise
            elif self.augmentation_mode == 'conv_only':
                self.num_augmentations = num_conv

        # Store noise/conv parameters needed for __getitem__
        self.noise_levels = noise_levels
        self.conv_directions = conv_directions
        self.noise_conv_combinations = list(itertools.product(self.noise_levels, self.conv_directions))
        # Store total combinations for use in __getitem__
        self.total_combinations = len(self.noise_conv_combinations)

        # Initialize simulators
        self.noise_simulator = NoiseSimulator(noise_type=NoisyType.from_string(noise_type), noise_sigma=0.0)
        self.forward_simulator = ForwardSimulator()
        self.current_epoch = 0 # Initialize epoch counter

    def __len__(self):
        return len(self.file_list) * self.num_augmentations

    def set_epoch(self, epoch: int):
        self.current_epoch = epoch

    def __getitem__(self, index: int) -> dict[DataKey, Tensor | str]:
        # Determine the actual file index and augmentation index
        file_idx = index // self.num_augmentations
        aug_idx = index % self.num_augmentations

        image_np = self._load_from_npy(self.file_list[file_idx])
        _name = Path(self.file_list[file_idx]).name

        # In training mode, we load a clean GT and create a noisy version
        if self.training_mode:
            image_gt_np = self._augment(image_np)
            image_gt_tensor = torch.from_numpy(image_gt_np.copy()).unsqueeze(0).float() # Shape: [1, H, W]
            
            # --- 💡 수정된 부분: 시뮬레이터 입력 형태를 [1, 1, H, W]로 통일 ---
            # 모든 시뮬레이터는 [Batch, Channel, H, W] 형태의 4D 텐서를 기대함
            image_noise_tensor_4d = image_gt_tensor.unsqueeze(0) # Shape: [1, 1, H, W]

            # Apply on-the-fly degradation
            if self.augmentation_mode == 'conv_only':
                conv_direction = self.conv_directions[(self.current_epoch + index) % len(self.conv_directions)]
                # forward_simulator는 [2, H, W]를 반환
                image_noise_tensor = self.forward_simulator(image_noise_tensor_4d, conv_direction)
            
            elif self.augmentation_mode == 'noise_only':
                noise_level = self.noise_levels[(self.current_epoch + index) % len(self.noise_levels)]
                self.noise_simulator.noise_sigma = noise_level
                # noise_simulator는 [1, 1, H, W]를 반환하므로, 채널 차원을 보존하기 위해 squeeze(0)만 적용
                image_noise_tensor = self.noise_simulator(image_noise_tensor_4d).squeeze(0)

            elif self.augmentation_mode == 'both':
                combination_idx = (self.current_epoch + index) % self.total_combinations
                noise_level, conv_direction = self.noise_conv_combinations[combination_idx]
                
                # forward_simulator가 먼저 [2, H, W]를 만들고, 
                conv_output_3d = self.forward_simulator(image_noise_tensor_4d, conv_direction)
                # --- 💡 수정된 부분: 이를 다시 4D로 변환하여 noise_simulator에 전달 ---
                conv_output_4d = conv_output_3d.unsqueeze(0) 

                self.noise_simulator.noise_sigma = noise_level
                # noise_simulator는 [1, 2, H, W]를 반환하므로, squeeze(0)만 적용
                image_noise_tensor = self.noise_simulator(conv_output_4d).squeeze(0)

            else: # 'none' mode
                image_noise_tensor = image_gt_tensor


            return {
                DataKey.image_gt: image_gt_tensor,
                DataKey.image_noise: image_noise_tensor, # 최종 형태는 [C, H, W]
                DataKey.name: _name,
            }
        # In evaluation mode, the loaded image is already the noisy input
        else:
            image_noise_tensor = torch.from_numpy(image_np).unsqueeze(0).float()
            return {
                # In a real test set, we might not have the GT. 
                # We can return the noisy image as a placeholder for GT.
                DataKey.image_gt: image_noise_tensor,
                DataKey.image_noise: image_noise_tensor,
                DataKey.name: _name,
            }


def get_data_wrapper_loader(
    file_path: list[str],
    training_mode: bool,
    data_wrapper_class: str = "random",
    **kwargs,
) -> tuple[DataLoader, Dataset | None] | tuple[None, None]:
    """
    Creates a DataLoader instance for a given dataset configuration.
    """
    wrapper_map = {
        'random': RandomDataWrapper,
        'controlled': ControlledDataWrapper,
    }
    DataWrapperClass = wrapper_map[data_wrapper_class]
    
    try:
        dataset = DataWrapperClass(
            file_path=file_path, training_mode=training_mode, **kwargs
        )

        if not len(dataset):
            return (None, None)

        # Ensure num_workers is an integer before comparison
        num_workers = kwargs.get("num_workers", 0)

        loader = DataLoader(
            dataset,
            batch_size=kwargs.get("batch", 1),
            shuffle=kwargs.get("shuffle", False),
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=num_workers > 0,
        )

        return (
            loader,
            dataset,
        )
    except Exception as e:
        logger.error(f"Error creating data loader for {data_wrapper_class}: {e}")
        return (None, None)
