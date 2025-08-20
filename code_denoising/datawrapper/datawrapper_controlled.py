import glob
import random
from dataclasses import dataclass
from enum import IntEnum
from typing import Literal
from pathlib import Path
import re

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import os
import sys
import itertools

# 프로젝트 최상위 경로를 시스템 경로에 추가하여 다른 패키지(dataset)를 찾을 수 있도록 합니다.
sys.path.append(str(Path(__file__).resolve().parents[2]))

# 실시간 증강을 위해 시뮬레이터들을 임포트합니다.
# `.noise_simulator`: 현재 패키지(datawrapper) 내의 모듈 (상대 경로)
# `dataset.forward_simulator`: 다른 패키지(dataset) 내의 모듈 (절대 경로)
from .noise_simulator import NoiseSimulator, NoisyType
from dataset.forward_simulator import ForwardSimulator


prob_flip: float = 0.5


class DataKey(IntEnum):
    Label = 0
    Noisy = 1
    Name = 2


@dataclass
class LoaderConfig:
    # --- 재설계: 실시간 증강을 위한 설정들로 변경 ---
    data_type: str
    batch: int
    num_workers: int
    shuffle: bool
    augmentation_mode: Literal['none', 'noise_only', 'conv_only', 'both']
    noise_type: Literal["gaussian"]
    noise_levels: list[float]
    conv_directions: list[tuple[float, float]]


class DataWrapper(Dataset):
    file_list: list[str]
    training_mode: bool
    
    # --- 재설계: 실시간 증강을 위한 멤버 변수 ---
    augmentation_mode: Literal['none', 'noise_only', 'conv_only', 'both']
    noise_levels: list[float]
    conv_directions: list[tuple[float, float]]
    noise_type: NoisyType
    noise_simulator: NoiseSimulator
    forward_simulator: ForwardSimulator
    current_epoch: int = 0 # 현재 에폭 번호를 저장할 변수 추가

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
        # noise_sigma는 __getitem__에서 매번 덮어쓰므로, 여기서는 임의의 초기값(0.0)을 사용합니다.
        self.noise_simulator = NoiseSimulator(noise_sigma=0.0)
        self.forward_simulator = ForwardSimulator()

        # For controlled augmentation
        self.current_epoch = 0
        self.noise_conv_combinations = list(itertools.product(self.noise_levels, self.conv_directions))
        self.total_combinations = len(self.noise_conv_combinations)

        # 이제 file_path는 항상 원본 'train' 또는 'val' 폴더입니다.
        total_list: list[str] = []
        for _file_path in file_path:
            p = Path(_file_path)
            total_list += [str(f) for f in p.glob(data_type)]
        self.file_list = sorted(total_list)

        # 증강 조합의 총 개수를 계산합니다.
        self.num_noise_levels = len(self.noise_levels)
        self.num_conv_directions = len(self.conv_directions)
        self.total_combinations = self.num_noise_levels * self.num_conv_directions


    @staticmethod
    def _load_from_npy(
        file_npy: str,
    ) -> np.ndarray: # --- torch.Tensor에서 np.ndarray로 반환 타입 변경 ---
        # astype(np.float32)를 추가하여 타입 안정성 확보
        img = np.load(file_npy).astype(np.float32)
        return img

    def _augment(
        self,
        img_np: np.ndarray, # --- 입력 타입을 np.ndarray로 변경 ---
    ) -> np.ndarray:
        if random.random() > prob_flip:
            img_np = np.ascontiguousarray(np.flip(img_np, axis=0))
        if random.random() > prob_flip:
            img_np = np.ascontiguousarray(np.flip(img_np, axis=1))
        return img_np

    def set_epoch(self, epoch: int):
        """Sets the current epoch for deterministic augmentation."""
        self.current_epoch = epoch

    def __getitem__(
        self,
        idx: int,
    ):
        # 1. 원본(label) 이미지를 numpy 배열로 로드합니다.
        label_np = self._load_from_npy(self.file_list[idx])
        _name = Path(self.file_list[idx]).name

        # 2. 학습 모드일 때만 데이터 증강(flip)을 적용합니다.
        if self.training_mode:
            label_np = self._augment(label_np)
        
        # 3. 실시간 손상(corruption)을 적용하여 noisy 이미지를 생성합니다.
        #    validation 모드에서는 augmentation_mode='none'으로 설정하여 원본을 그대로 사용합니다.
        noisy_np = label_np.copy()
        
        # NumPy 배열을 Tensor로 변환 (시뮬레이터 입력용)
        noisy_tensor = torch.from_numpy(noisy_np).unsqueeze(0).unsqueeze(0)
        
        # --- 핵심 로직 수정: '에폭별 순환' 방식 적용 ---
        if self.augmentation_mode != 'none' and self.total_combinations > 0:
            # (현재 에폭 + 이미지 인덱스)를 기반으로 조합을 결정
            # 이렇게 하면 에폭이 바뀔 때마다 다른 조합이 적용됨
            combination_idx = (self.current_epoch + idx) % self.total_combinations
            
            noise_idx = combination_idx // self.num_conv_directions
            conv_idx = combination_idx % self.num_conv_directions

            if self.augmentation_mode == 'conv_only' or self.augmentation_mode == 'both':
                direction = self.conv_directions[conv_idx]
                noisy_tensor = self.forward_simulator(noisy_tensor, B0_dir=direction)
                
            if self.augmentation_mode == 'noise_only' or self.augmentation_mode == 'both':
                sigma = self.noise_levels[noise_idx]
                self.noise_simulator.noise_sigma = sigma
                noisy_tensor = self.noise_simulator(noisy_tensor)
        
        # 최종 결과를 다시 NumPy 배열로 변환
        noisy_np = noisy_tensor.squeeze().cpu().numpy()

        # 4. 최종적으로 모든 NumPy 배열을 Tensor로 변환하여 반환합니다.
        return (
            torch.from_numpy(label_np.copy()).unsqueeze(0),
            torch.from_numpy(noisy_np.copy()).unsqueeze(0),
            _name,
        )

    def __len__(self) -> int:
        return len(self.file_list)


def get_data_wrapper_loader(
    file_path: list[str],
    training_mode: bool,
    loader_cfg: LoaderConfig,
) -> tuple[
    DataLoader,
    DataWrapper,
    int,
]:
    dataset = DataWrapper(
        file_path=file_path,
        data_type=loader_cfg.data_type,
        training_mode=training_mode,
        # --- 재설계: loader_cfg에서 실시간 증강 파라미터 전달 ---
        augmentation_mode=loader_cfg.augmentation_mode if training_mode else 'none',
        noise_type=NoisyType.from_string(loader_cfg.noise_type),
        noise_levels=loader_cfg.noise_levels,
        conv_directions=loader_cfg.conv_directions,
    )
    len_data = len(dataset)
    if not len_data:
        return (None, None, len_data)

    _ = dataset[0]

    dataloader = DataLoader(
        dataset,
        batch_size=loader_cfg.batch,
        num_workers=loader_cfg.num_workers,
        pin_memory=True,
        persistent_workers=loader_cfg.num_workers > 0,
        shuffle=loader_cfg.shuffle,
    )

    return (
        dataloader,
        dataset,
        len(dataset),
    )
