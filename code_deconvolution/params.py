"""
#  Copyright Juhyung Park
#  Laboratory for Imaging Science and Technology
#  Seoul National University
#  email : jack0878@snu.ac.kr
#
"""

import argparse
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

import torch

default_root: str = "/fast_storage/juhyung/dataset"
DATA_ROOT: str = os.environ.get("DATA_ROOT", default_root)

TRAIN_DATASET: list[str] = [
    DATA_ROOT + "/train",
]
VALID_DATASET: list[str] = [
    DATA_ROOT + "/val",
]
TEST_DATASET: list[str] = [
    DATA_ROOT + "/val",
]

default_run_dir: str = "../logs_deconvolution"
RUN_DIR: str = os.environ.get("RUN_DIR", default_run_dir)


@dataclass
class GeneralConfig:
    # Dataset
    train_dataset: list[str] = field(default_factory=lambda: TRAIN_DATASET)
    valid_dataset: list[str] = field(default_factory=lambda: VALID_DATASET)
    test_dataset: list[str] = field(default_factory=lambda: TEST_DATASET)
    data_type: str = "*.npy"

    # Logging
    log_lv: str = "INFO"
    run_dir: Path = Path(RUN_DIR)
    init_time: float = 0.0

    # Model experiment
    model_type: Literal["unet"] = "unet"

    # Optimizer
    optimizer: Literal["adam", "adamw"] = "adam"
    loss_model: Literal["l1", "l2", "model_loss"] = "l2"
    lr: float = 1e-4
    lr_decay: float = 0.88
    lr_tol: int = 1

    # Train params
    gpu: str = "0"
    train_batch: int = 16
    valid_batch: int = 8
    train_epoch: int = 100
    logging_density: int = 4
    valid_interval: int = 2
    valid_tol: int = 2
    num_workers: int = 4
    save_val: bool = True
    parallel: bool = True
    device: torch.device | None = None
    save_max_idx: int = 500

    tag: str = ""

    # Model loss
    model_loss_weight: float = 0.8


@dataclass
class NetworkConfig:
    # Model architecture
    in_chans: int = 1
    out_chans: int = 1
    chans: int = 16
    num_pool_layers: int = 4


@dataclass
class TestConfig:
    # Dataset
    trained_checkpoints: str = ""


# Argparser
parser = argparse.ArgumentParser(description="Training Configuration")
general_config_dict = asdict(GeneralConfig())
network_config_dict = asdict(NetworkConfig())
test_config_dict = asdict(TestConfig())

for key, default_value in general_config_dict.items():
    if isinstance(default_value, bool):
        parser.add_argument(
            f"--{key}",
            type=lambda x: x.lower() in ("true", "t", "yes", "y", "1"),
            default=None,
            help=f"Set {key} (true/false, default: {default_value})",
        )
    else:
        parser.add_argument(
            f"--{key}",
            type=type(default_value),
            default=None,
            help=f"Override for {key}",
        )

for key, default_value in network_config_dict.items():
    if isinstance(default_value, bool):
        parser.add_argument(
            f"--{key}",
            type=lambda x: x.lower() in ("true", "t", "yes", "y", "1"),
            default=None,
            help=f"Set {key} (true/false, default: {default_value})",
        )
    else:
        parser.add_argument(
            f"--{key}",
            type=type(default_value),
            default=None,
            help=f"Override for {key}",
        )

for key, default_value in test_config_dict.items():
    if isinstance(default_value, bool):
        parser.add_argument(
            f"--{key}",
            type=lambda x: x.lower() in ("true", "t", "yes", "y", "1"),
            default=None,
            help=f"Set {key} (true/false, default: {default_value})",
        )
    else:
        parser.add_argument(
            f"--{key}",
            type=type(default_value),
            default=None,
            help=f"Override for {key}",
        )

# Apply argparser
config = GeneralConfig()
networkconfig = NetworkConfig()
args = parser.parse_args()

for key, value in vars(args).items():
    if value is not None:
        if hasattr(config, key):
            if isinstance(getattr(config, key), bool):
                setattr(config, key, bool(value))
            else:
                setattr(config, key, value)

        if hasattr(networkconfig, key):
            if isinstance(getattr(networkconfig, key), bool):
                setattr(networkconfig, key, bool(value))
            else:
                setattr(networkconfig, key, value)
