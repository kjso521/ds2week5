"""
#  Copyright Juhyung Park
#  Laboratory for Imaging Science and Technology
#  Seoul National University
#  email : jack0878@snu.ac.kr
#
"""

import os
import sys
import dataclasses
from pathlib import Path
from copy import deepcopy

# --- 중요: 모든 import 이전에 프로젝트 루트 경로를 시스템 경로에 추가 ---
# 이 스크립트가 실행되는 위치를 기준으로, 상위 2단계 폴더(week5)를 경로에 추가합니다.
# 이렇게 하면 'dataset' 폴더를 항상 찾을 수 있습니다.
sys.path.append(str(Path(__file__).resolve().parents[1]))


import time
import warnings
from enum import Enum

import torch
from torch.nn import DataParallel
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from torch import Tensor

from code_denoising.datawrapper.datawrapper import DataKey, get_data_wrapper_loader, LoaderConfig, BaseDataWrapper
from code_denoising.core_funcs import get_model, get_optimizer, get_loss_model, save_checkpoint, test_part, ModelType
from code_denoising.common.utils import call_next_id, separator
from code_denoising.common.logger import logger, logger_add_handler
from code_denoising.common.wrapper import error_wrap
from params import config, dncnnconfig, unetconfig, parse_args_for_train_script
from code_denoising.common.metric import calculate_psnr


warnings.filterwarnings("ignore")


class Trainer:
    """Trainer"""

    def __init__(self) -> None:
        """__init__"""
        self.config = deepcopy(config)
        
        if ModelType.from_string(self.config.model_type) == ModelType.Unet:
            self.config.model_config = deepcopy(unetconfig)
        elif ModelType.from_string(self.config.model_type) == ModelType.DnCNN:
            self.config.model_config = deepcopy(dncnnconfig)

        if self.config.augmentation_mode in ['conv_only', 'both']:
            self.config.model_config.in_chans = 2
            self.config.model_config.out_chans = 2 # Deconvolution requires 2-channel output (real/imaginary)
            logger.info("Setting model input and output channels to 2 for deconvolution.")

        self.run_dir = Path(self.config.run_dir) / f"{call_next_id(Path(self.config.run_dir)):05d}_{self.config.tag or 'train'}"
        self.save_dir = self.run_dir / "checkpoints"
        self.log_file = self.run_dir / "training.log"
        self.writer = SummaryWriter(log_dir=str(self.run_dir))
        
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        logger_add_handler(logger, f"{self.run_dir / 'log.log'}", self.config.log_lv)
        logger.info(separator())
        logger.info(f"Run dir: {self.run_dir}")
        logger.info(separator())
        self._logging_config()

        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        self.config.init_time = time.time()
        self.config.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        os.environ["CUDA_VISIBLE_DEVICES"] = str(self.config.gpu)

        self.device = self.config.device
        self.best_metric: float = 0.0
        self.best_epoch: int = 0
        self.epoch: int = 0
        self.primary_metric: float = 0.0
        self.tol_count: int = 0
        self.global_step: int = 0

        self._init_essential()

    def _init_essential(self):
        self._set_data()
        self._set_network()
        self._set_optimizer()
        self._set_loss()

    def run(self) -> None:
        self._train()
        self._test("best")

    def _logging_config(self):
        logger.info("General Config")
        for k, v in dataclasses.asdict(self.config).items():
            if k not in ['dncnn_config', 'unet_config', 'model_config']:
                logger.info(f"{k}:{v}")
        logger.info(separator())
        
        logger.info(f"Model Config ({self.config.model_type})")
        for k, v in dataclasses.asdict(self.config.model_config).items():
            logger.info(f"{k}:{v}")

    @error_wrap
    def _set_network(self) -> None:
        self.model = get_model(self.config).to(self.config.device)
        if self.config.parallel:
            self.model = DataParallel(self.model)

    @error_wrap
    def _set_optimizer(self) -> None:
        self.optimizer = get_optimizer(self.config, self.model.parameters())
        self.scheduler = ReduceLROnPlateau(self.optimizer, 'max', factor=self.config.lr_decay, patience=self.config.lr_tol)

    @error_wrap
    def _set_loss(self) -> None:
        self.loss_model = get_loss_model(self.config).to(self.config.device)

    @error_wrap
    def _set_data(self) -> None:
        loader_cfg: LoaderConfig = {
            "data_type": self.config.data_type,
            "batch": self.config.train_batch,
            "num_workers": self.config.num_workers,
            "shuffle": True,
            "augmentation_mode": self.config.augmentation_mode,
            "training_phase": self.config.training_phase, # Pass parameter
            "noise_type": self.config.noise_type,
            "noise_levels": self.config.noise_levels,
            "conv_directions": self.config.conv_directions
        }
        self.train_loader, self.train_dataset_obj = get_data_wrapper_loader(
            file_path=self.config.train_dataset,
            training_mode=True,
            data_wrapper_class='controlled',
            **loader_cfg
        )
        logger.info(f"Train dataset length : {len(self.train_dataset_obj)}")

        # Create a new, separate config for the validation set using dict.copy()
        valid_loader_cfg = loader_cfg.copy()
        valid_loader_cfg['batch'] = self.config.valid_batch
        valid_loader_cfg['shuffle'] = False
        self.valid_loader, self.valid_dataset_obj = get_data_wrapper_loader(
            file_path=self.config.valid_dataset,
            training_mode=True, # Augmentation is controlled by mode, not just training_mode
            data_wrapper_class='controlled',
            **valid_loader_cfg
        )
        logger.info(f"Valid dataset length : {len(self.valid_dataset_obj)}")

        # Create a separate config for the test set as well
        test_loader_cfg = loader_cfg.copy()
        test_loader_cfg['batch'] = 1  # Test batch size is always 1
        test_loader_cfg['shuffle'] = False
        test_loader_cfg['augmentation_mode'] = 'none'
        self.test_loader, self.test_dataset_obj = get_data_wrapper_loader(
            file_path=self.config.test_dataset,
            training_mode=False,
            data_wrapper_class='controlled',
            **test_loader_cfg
        )
        logger.info(f"Test dataset length : {len(self.test_loader.dataset)}")

    @error_wrap
    def _train(self) -> None:
        """train entry point"""
        logger.info("####################################################################################################")
        logger.info("Train start")
        
        for epoch in range(self.config.train_epoch):
            self.epoch = epoch
            logger.info(f"Epoch: {epoch}")
            logger.info(f"Learning rate: {self.optimizer.param_groups[0]['lr']:.3e}")
            self.global_step = 0
            self.primary_metric = 0.0

            # Set the current epoch on the dataset object to ensure varied augmentations
            if hasattr(self.train_dataset_obj, 'set_epoch'):
                self.train_dataset_obj.set_epoch(epoch)

            self.model.train()
            train_loss = 0
            for i, data in enumerate(tqdm(self.train_loader, leave=False)):
                self.global_step += 1
                image_gt = data[DataKey.image_gt].to(self.device)
                image_noise = data[DataKey.image_noise].to(self.device)

                # Model prediction
                image_pred = self.model(image_noise)

                # Loss calculation
                self.optimizer.zero_grad()
                loss = self.loss_model(image_pred, image_gt)

                # Loss backward
                loss.backward()
                self.optimizer.step()

                train_loss += loss.item()
                self.writer.add_scalar("Loss/train", loss.item(), self.epoch * len(self.train_loader) + i)

            logger.info(
                f"Epoch {epoch}: train loss {train_loss / len(self.train_loader):.4f}"
            )

            if epoch % self.config.valid_interval == 0:
                valid_psnr = test_part(
                    data_loader=self.valid_loader,
                    network=self.model,
                    run_dir=self.run_dir,
                    save_val=self.config.save_val,
                    epoch=epoch,
                    test_mode=False
                )
                self.scheduler.step(valid_psnr)

                if valid_psnr > self.best_metric:
                    self.best_metric = valid_psnr
                    self.best_epoch = epoch
                    self.tol_count = 0
                    logger.info("Best model renewed")
                    self._save_checkpoint("checkpoint_best.ckpt")
                else:
                    self.tol_count += 1
                    if self.tol_count >= self.config.valid_tol:
                        logger.info(f"Early stop at epoch {epoch}")
                        break
                # Save checkpoint every validation interval regardless of performance
                # self._save_checkpoint(f"checkpoint_epoch_{epoch}.ckpt")

    @error_wrap
    def _test(self, mode: str) -> None:
        """Test"""
        
        if not (self.save_dir / "checkpoint_best.ckpt").exists():
            logger.warning(
                f"Best checkpoint not found in {self.save_dir / 'checkpoint_best.ckpt'}. Skipping test."
            )
            return

        checkpoint = torch.load(self.save_dir / "checkpoint_best.ckpt")
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

        with torch.no_grad():
            for i, data in enumerate(self.test_loader):
                image_gt = data[DataKey.image_gt].to(self.device)
                image_noise = data[DataKey.image_noise].to(self.device)
                image_pred = self.model(image_noise)

                if i < self.config.save_max_idx:
                    self._save_image(
                        image_pred,
                        f"test_epoch{self.best_epoch}_{Path(data[DataKey.name][0]).stem}.png",
                        self.run_dir / "test_images"
                    )

    def _save_checkpoint(self, filename: str):
        """Saves model checkpoint."""
        save_checkpoint(
            network=self.model,
            run_dir=self.save_dir,
            epoch=self.epoch,
            model_type=self.config.model_type
        )

    def _save_image(self, tensor: torch.Tensor, filename: str, directory: Path):
        """Saves a tensor as an image."""
        directory.mkdir(parents=True, exist_ok=True)
        # Assuming core_funcs.py has the save_numpy_as_image function
        from code_denoising.common.utils import save_numpy_as_image
        save_numpy_as_image(tensor.cpu().numpy(), directory / filename)


def main() -> None:
    """execution entry point"""
    parse_args_for_train_script()
    trainer = Trainer()
    trainer.run()


if __name__ == "__main__":
    main()
