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


class RandomDataWrapper(Dataset):
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

class ControlledDataWrapper(Dataset):
    def __init__(self, file_path: list, training_mode: bool, data_type: str, 
                 augmentation_mode: str, noise_type: str, noise_levels: list, 
                 conv_directions: list, **kwargs):
        super().__init__()
        self.training_mode = training_mode

        # Find all files matching the pattern
        self.file_list = []
        for path in file_path:
            self.file_list.extend(glob.glob(os.path.join(path, data_type)))

        if not self.file_list:
            logger.error(f"No files found for pattern {data_type} in paths {file_path}")
            raise FileNotFoundError(f"No data files found in {file_path}")

        # Augmentation settings
        self.num_augmentations = 1
        if self.training_mode and augmentation_mode in ['noise', 'both']:
            num_noise = len(noise_levels) if noise_levels else 1
            num_conv = len(conv_directions) if conv_directions else 1
            
            if augmentation_mode == 'both':
                self.num_augmentations = num_noise * num_conv
            elif augmentation_mode == 'noise_only':
                self.num_augmentations = num_noise
            elif augmentation_mode == 'conv_only':
                self.num_augmentations = num_conv

        # Store noise/conv parameters needed for __getitem__
        self.noise_levels = noise_levels
        self.conv_directions = conv_directions
        self.noise_conv_combinations = list(itertools.product(self.noise_levels, self.conv_directions))

        # Initialize simulators
        self.noise_simulator = NoiseSimulator(noise_type=NoisyType.from_string(noise_type), noise_sigma=0.0)
        self.forward_simulator = ForwardSimulator()

    def set_epoch(self, epoch: int):
        self.current_epoch = epoch

    def __getitem__(self, idx: int) -> dict[DataKey, Tensor | str]:
        image_gt_np = self._load_from_npy(self.file_list[idx])
        _name = Path(self.file_list[idx]).name

        if self.training_mode:
            image_gt_np = self._augment(image_gt_np)

        # Convert NumPy array to a 4D Torch Tensor for simulators
        image_noise_tensor = torch.from_numpy(image_gt_np.copy()).unsqueeze(0).unsqueeze(0)

        if self.augmentation_mode == 'noise_only':
            if len(self.noise_levels) > 0:
                noise_level = self.noise_levels[(self.current_epoch + idx) % len(self.noise_levels)]
                self.noise_simulator.noise_sigma = noise_level
                image_noise_tensor = self.noise_simulator(image_noise_tensor)
        elif self.augmentation_mode == 'conv_only':
            if len(self.conv_directions) > 0:
                conv_direction = self.conv_directions[(self.current_epoch + idx) % len(self.conv_directions)]
                image_noise_tensor = self.forward_simulator(image_noise_tensor, conv_direction)
        elif self.augmentation_mode == 'both':
            if self.total_combinations > 0:
                combination_idx = (self.current_epoch + idx) % self.total_combinations
                noise_level, conv_direction = self.noise_conv_combinations[combination_idx]
                
                image_noise_tensor = self.forward_simulator(image_noise_tensor, conv_direction)
                self.noise_simulator.noise_sigma = noise_level
                image_noise_tensor = self.noise_simulator(image_noise_tensor)

        return {
            DataKey.image_gt: torch.from_numpy(image_gt_np.copy()).unsqueeze(0),
            DataKey.image_noise: image_noise_tensor.squeeze(0),
            DataKey.name: _name,
        }


def get_data_wrapper_loader(
    file_path: list[str],
    loader_cfg: LoaderConfig,
    training_mode: bool,
    data_wrapper_class: Literal['random', 'controlled'] = 'random',
) -> tuple[
    DataLoader,
    Dataset,
]:
    wrapper_map = {
        'random': RandomDataWrapper,
        'controlled': ControlledDataWrapper,
    }
    DataWrapperClass = wrapper_map[data_wrapper_class]
    
    dataset = DataWrapperClass(
        file_path=file_path,
        data_type=loader_cfg["data_type"],
        training_mode=training_mode,
        augmentation_mode=loader_cfg["augmentation_mode"] if training_mode else 'none',
        noise_type=NoisyType.from_string(loader_cfg["noise_type"]),
        noise_levels=loader_cfg["noise_levels"],
        conv_directions=loader_cfg["conv_directions"],
    )
    
    if not len(dataset):
        return (None, None)

    dataloader = DataLoader(
        dataset,
        batch_size=loader_cfg["batch"],
        num_workers=loader_cfg["num_workers"],
        pin_memory=True,
        persistent_workers=loader_cfg["num_workers"] > 0,
        shuffle=loader_cfg["shuffle"],
    )

    return (
        dataloader,
        dataset,
    )
