"""
#  Copyright Juhyung Park
#  Laboratory for Imaging Science and Technology
#  Seoul National University
#  email : jack0878@snu.ac.kr
#
"""

import argparse
import os
import time
import warnings
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Literal

import torch
from filter.mean_filter import mean_filter
from filter.median_filter import median_filter
from scipy.io import savemat
from torch import Tensor
from torch.utils.data import DataLoader

from common.logger import logger, logger_add_handler
from common.metric import calculate_psnr, calculate_ssim
from common.utils import (
    call_next_id,
    separator,
    validate_tensor_dimensions,
    validate_tensors,
)
from common.wrapper import error_wrap
from components.metriccontroller import MetricController
from core_funcs import log_summary
from datawrapper.datawrapper import DataKey, LoaderConfig, get_data_wrapper_loader
from model.dncnn import DnCNN
from params import DnCNNConfig

warnings.filterwarnings("ignore")


default_root: str = "../dataset"
default_run_dir: str = "../logs_denoising"
checkpoint_dir: str = "../checkpoint_best.ckpt"

DATA_ROOT: str = os.environ.get("DATA_ROOT", default_root)
RUN_DIR: str = os.environ.get("RUN_DIR", default_run_dir)
CHECKPOINT_DIR: str = os.environ.get("CHECKPOINT_DIR", checkpoint_dir)

TEST_DATASET: list[str] = [
    DATA_ROOT + "/test_y",
]


class ModelType(str, Enum):
    DnCNN = "dncnn"
    MeanFilter = "mean_filter"
    MedianFilter = "median_filter"

    @classmethod
    def from_string(cls, value: str) -> "ModelType":
        try:
            return cls(value)
        except ValueError as err:
            raise ValueError(
                f"Invalid ModelType value: {value}. Must be one of {list(cls)} : {err}"
            ) from err


@dataclass
class TestConfig:
    # Dataset
    trained_checkpoints: str = CHECKPOINT_DIR

    test_dataset: list[str] = field(default_factory=lambda: TEST_DATASET)
    data_type: str = "*.npy"

    # Logging
    log_lv: str = "INFO"
    run_dir: Path = Path(RUN_DIR)
    init_time: float = 0.0

    # Model experiment
    model_type: Literal["dncnn", "mean_filter", "median_filter"] = "mean_filter"

    # Test params
    gpu: str = "0"
    valid_batch: int = 8
    num_workers: int = 4
    device: torch.device | None = None

    # Experiment
    noise_type: Literal["gaussian", "rician", "uniform", "salt_and_pepper"] = "gaussian"
    noise_sigma: float = 0.0


parser = argparse.ArgumentParser(description="Test Configuration")
test_dict = asdict(TestConfig())
for key, default_value in test_dict.items():
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
args = parser.parse_args()

NET_PREDICTION = DnCNN | torch.nn.DataParallel[DnCNN]


def test_part_prediction(
    _data: dict[DataKey, Tensor | str],
    test_dir: Path,
    network_prediction: NET_PREDICTION,
    save_val: bool,
    test_state: MetricController,
    config: TestConfig,
) -> None:
    noisy: Tensor = _data[DataKey.Noisy].to(config.device)
    label: Tensor = _data[DataKey.Label].to(config.device)
    name: str = _data[DataKey.Name]

    validate_tensors([noisy, label])
    validate_tensor_dimensions([noisy, label], 4)

    output = network_prediction(noisy)

    validate_tensors([output])
    validate_tensor_dimensions([output], 4)

    if not save_val:
        return

    for idx in range(output.shape[0]):
        test_state.add(
            "psnr",
            calculate_psnr(output[idx : idx + 1, ...], label[idx : idx + 1, ...]),
        )
        test_state.add(
            "ssim",
            calculate_ssim(output[idx : idx + 1, ...], label[idx : idx + 1, ...]),
        )

        save_dict = {
            "noisy": noisy.cpu().numpy()[idx, ...],
            "output": output.cpu().numpy()[idx, ...],
            "label": label.cpu().numpy()[idx, ...],
        }

        save_path = test_dir / "test"
        os.makedirs(save_path, exist_ok=True)
        savemat(save_path / name[idx].replace(".npy", ".mat"), save_dict)


def test_part_mean_filter(
    _data: dict[DataKey, Tensor | str],
    test_dir: Path,
    test_state: MetricController,
) -> None:
    noisy: Tensor = _data[DataKey.Noisy]
    label: Tensor = _data[DataKey.Label]
    name: str = _data[DataKey.Name]

    validate_tensors([noisy, label])
    validate_tensor_dimensions([noisy, label], 4)

    for idx in range(noisy.shape[0]):
        output = mean_filter(noisy[idx : idx + 1, ...], kernel_size=3)
        test_state.add("psnr", calculate_psnr(output, label[idx : idx + 1, ...]))
        test_state.add("ssim", calculate_ssim(output, label[idx : idx + 1, ...]))

        save_dict = {
            "noisy": noisy.cpu().numpy()[idx, ...],
            "output": output.cpu().numpy()[0, ...],
            "label": label.cpu().numpy()[idx, ...],
        }

        save_path = test_dir / "test"
        os.makedirs(save_path, exist_ok=True)
        savemat(save_path / name[idx].replace(".npy", ".mat"), save_dict)


def test_part_median_filter(
    _data: dict[DataKey, Tensor | str],
    test_dir: Path,
    test_state: MetricController,
) -> None:
    noisy: Tensor = _data[DataKey.Noisy]
    label: Tensor = _data[DataKey.Label]
    name: str = _data[DataKey.Name]

    validate_tensors([noisy, label])
    validate_tensor_dimensions([noisy, label], 4)

    for idx in range(noisy.shape[0]):
        output = median_filter(noisy[idx : idx + 1, ...], kernel_size=3)
        test_state.add("psnr", calculate_psnr(output, label[idx : idx + 1, ...]))
        test_state.add("ssim", calculate_ssim(output, label[idx : idx + 1, ...]))

        save_dict = {
            "noisy": noisy.cpu().numpy()[idx, ...],
            "output": output.cpu().numpy()[0, ...],
            "label": label.cpu().numpy()[idx, ...],
        }

        save_path = test_dir / "test"
        os.makedirs(save_path, exist_ok=True)
        savemat(save_path / name[idx].replace(".npy", ".mat"), save_dict)


def test_part(
    valid_state: MetricController,
    valid_loader: DataLoader,
    network_prediction: NET_PREDICTION | None,
    run_dir: Path,
    save_val: bool,
    config: TestConfig,
) -> float:
    if config.device is None:
        raise TypeError("device is not to be None")

    if network_prediction is not None:
        network_prediction.eval()

    for _data in valid_loader:
        if ModelType.from_string(config.model_type) == ModelType.DnCNN:
            test_part_prediction(
                _data=_data,
                test_dir=run_dir,
                network_prediction=network_prediction,
                save_val=save_val,
                test_state=valid_state,
                config=config,
            )
        elif ModelType.from_string(config.model_type) == ModelType.MeanFilter:
            test_part_mean_filter(
                _data=_data,
                test_dir=run_dir,
                test_state=valid_state,
            )
        elif ModelType.from_string(config.model_type) == ModelType.MedianFilter:
            test_part_median_filter(
                _data=_data,
                test_dir=run_dir,
                test_state=valid_state,
            )
        else:
            raise KeyError("model type not matched")

    log_summary(state=valid_state, log_std=True, init_time=config.init_time)

    primary_metric = valid_state.mean("psnr")
    return primary_metric


class Tester:
    run_dir: Path
    network_prediction: NET_PREDICTION | None
    test_loader: DataLoader
    config: TestConfig
    modelconfig: DnCNNConfig

    def __init__(
        self,
    ) -> None:
        self.config = TestConfig()
        for key, value in vars(args).items():
            if value is not None and hasattr(self.config, key):
                if isinstance(getattr(self.config, key), bool):
                    setattr(self.config, key, bool(value))
                else:
                    setattr(self.config, key, value)

        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = self.config.gpu

        self.config.init_time = time.time()
        self.config.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # dir setting
        self.run_dir = (
            self.config.run_dir / f"{call_next_id(self.config.run_dir):05d}_test"
        )
        logger_add_handler(logger, f"{self.run_dir/'log.log'}", self.config.log_lv)
        logger.info(separator())
        logger.info(f"Run dir: {self.run_dir}")
        os.makedirs(self.run_dir, exist_ok=True)

        # log config
        logger.info(separator())
        logger.info("Text Config")
        config_dict = asdict(self.config)
        for k in config_dict:
            logger.info(f"{k}:{config_dict[k]}")

    def __call__(
        self,
    ) -> None:
        self._set_data()
        self._set_network()
        self._test()

    @error_wrap
    def _set_data(
        self,
    ) -> None:
        logger.info(separator())
        test_loader_cfg = LoaderConfig(
            data_type=self.config.data_type,
            batch=self.config.valid_batch,
            num_workers=self.config.num_workers,
            shuffle=False,
            noisy_type=self.config.noise_type,
            noise_sigma=self.config.noise_sigma,
        )
        self.test_loader, _, test_len = get_data_wrapper_loader(
            file_path=self.config.test_dataset,
            training_mode=False,
            loader_cfg=test_loader_cfg,
        )
        logger.info(f"Test dataset length : {test_len}")

    @error_wrap
    def _set_network(
        self,
    ) -> None:
        if ModelType.from_string(self.config.model_type) != ModelType.DnCNN:
            self.network_prediction = None
            return

        longitudinal_checkpoint_data = torch.load(
            self.config.trained_checkpoints,
            map_location="cpu",
            weights_only=True,
        )

        if not (
            ("model_state_dict" in longitudinal_checkpoint_data)
            and ("model_config" in longitudinal_checkpoint_data)
        ):
            logger.error("Invalid Checkpoint")
            raise KeyError("Invalid Checkpoint")

        self.modelconfig = DnCNNConfig(**longitudinal_checkpoint_data["model_config"])
        self.network_prediction = DnCNN(
            channels=self.modelconfig.channels,
            num_of_layers=self.modelconfig.num_of_layers,
            kernel_size=self.modelconfig.kernel_size,
            padding=self.modelconfig.padding,
            features=self.modelconfig.features,
        )
        load_state_dict = longitudinal_checkpoint_data["model_state_dict"]

        _state_dict = {}
        for key, value in load_state_dict.items():
            new_key = key.replace("module.", "")
            _state_dict[new_key] = value

        try:
            self.network_prediction.load_state_dict(_state_dict, strict=True)
        except Exception as err:
            logger.warning(
                f"Strict load failure. Trying to load weights available: {err}"
            )
            self.network_prediction.load_state_dict(_state_dict, strict=False)

        logger.info(separator())
        logger.info("Model Config")
        config_dict = asdict(self.modelconfig)
        for k in config_dict:
            logger.info(f"{k}:{config_dict[k]}")

        self.network_prediction = self.network_prediction.to(self.config.device)

    @error_wrap
    def _test(self) -> None:
        test_state = MetricController()
        test_state.reset()
        logger.info(separator())
        logger.info("Test")
        with torch.no_grad():
            test_part(
                valid_state=test_state,
                valid_loader=self.test_loader,
                network_prediction=self.network_prediction,
                run_dir=self.run_dir,
                save_val=True,
                config=self.config,
            )


if __name__ == "__main__":
    test = Tester()
    test()
