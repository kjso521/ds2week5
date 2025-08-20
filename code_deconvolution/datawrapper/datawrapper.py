import glob
import random
from dataclasses import dataclass
from enum import IntEnum

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from datawrapper.forward_simulator import ForwardSimulator

prob_flip: float = 0.5


class DataKey(IntEnum):
    Label = 0
    Measure = 1
    Name = 2


@dataclass
class LoaderConfig:
    data_type: str
    batch: int
    num_workers: int
    shuffle: bool


class DataWrapper(Dataset):
    file_list: list[str]
    training_mode: bool
    forward_simulator: ForwardSimulator

    def __init__(
        self,
        file_path: list[str],
        data_type: str,
        training_mode: bool,
    ):
        self.training_mode = training_mode
        self.forward_simulator = ForwardSimulator()

        super().__init__()
        total_list: list[str] = []
        for _file_path in file_path:
            total_list += glob.glob(f"{_file_path}/{data_type}")

        self.file_list = total_list

    @staticmethod
    def _load_from_npy(
        file_npy: str,
    ) -> torch.Tensor:
        img = torch.from_numpy(np.load(file_npy)).type(torch.float)
        if len(img.shape) == 2:
            img = img.unsqueeze(0)
        return img

    def _augment(
        self,
        label: torch.Tensor,
    ) -> torch.Tensor:
        if random.random() > prob_flip:
            label = torch.flip(label, dims=[1])
        if random.random() > prob_flip:
            label = torch.flip(label, dims=[2])

        return label

    def __getitem__(
        self,
        idx: int,
    ):
        label = self._load_from_npy(self.file_list[idx])
        if self.training_mode:
            label = self._augment(label)

        measure = self.forward_simulator(label)

        _name = self.file_list[idx].split("/")[-1]

        return (
            label,
            measure,
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
    )

    _ = dataset[0]

    dataloader = DataLoader(
        dataset,
        batch_size=loader_cfg.batch,
        num_workers=loader_cfg.num_workers,
        pin_memory=True,
        persistent_workers=True,
        shuffle=loader_cfg.shuffle,
    )

    return (
        dataloader,
        dataset,
        len(dataset),
    )
