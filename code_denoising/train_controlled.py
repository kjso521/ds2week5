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

# --- 중요: 모든 import 이전에 프로젝트 루트 경로를 시스템 경로에 추가 ---
# 이 스크립트가 실행되는 위치를 기준으로, 상위 2단계 폴더(week5)를 경로에 추가합니다.
# 이렇게 하면 'dataset' 폴더를 항상 찾을 수 있습니다.
sys.path.append(str(Path(__file__).resolve().parents[1]))


import time
import warnings
from dataclasses import asdict
from enum import Enum

import torch
from torch.nn import DataParallel
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm

from code_denoising.datawrapper.datawrapper import DataKey, get_data_wrapper_loader, LoaderConfig
from code_denoising.core_funcs import get_model, get_optimizer, get_loss_model, save_checkpoint, test_part
from code_denoising.common.utils import call_next_id, separator
from code_denoising.common.logger import logger, logger_add_handler
from code_denoising.common.wrapper import error_wrap
from params import config, dncnnconfig, unetconfig, parse_args_for_train_script

warnings.filterwarnings("ignore")


class Trainer:
    """Trainer"""

    def __init__(self) -> None:
        """__init__"""
        self.run_dir = Path(config.run_dir) / f"{call_next_id(Path(config.run_dir)):05d}_{config.tag or 'train'}"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        
        logger_add_handler(logger, f"{self.run_dir / 'log.log'}", config.log_lv)
        logger.info(separator())
        logger.info(f"Run dir: {self.run_dir}")
        logger.info(separator())

        # Log configurations
        logger.info("General Config")
        for k, v in asdict(config).items():
            logger.info(f"{k}:{v}")
        logger.info(separator())
        
        # This part depends on the model_type, so we log it after parsing args
        if config.model_type == "dncnn":
            logger.info("Model Config (DnCNN)")
            for k, v in asdict(dncnnconfig).items():
                logger.info(f"{k}:{v}")
        elif config.model_type == "unet":
            logger.info("Model Config (U-Net)")
            for k, v in asdict(unetconfig).items():
                logger.info(f"{k}:{v}")

        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        config.init_time = time.time()
        config.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        os.environ["CUDA_VISIBLE_DEVICES"] = str(config.gpu)

        self.device = config.device
        self.best_metric: float = 0.0
        self.best_epoch: int = 0
        self.epoch: int = 0
        self.primary_metric: float = 0.0
        self.tol_count: int = 0
        self.global_step: int = 0

    def run(self) -> None:
        self._set_data()
        self._set_network()
        self._train()
        self._test("best")  # Call test after training is finished

    @error_wrap
    def _set_data(self) -> None:
        loader_cfg = LoaderConfig(
            data_type=config.data_type,
            batch=config.train_batch,
            num_workers=config.num_workers,
            shuffle=True,
            augmentation_mode=config.augmentation_mode,
            training_phase=config.training_phase, # Pass parameter
            noise_type=config.noise_type,
            noise_levels=config.noise_levels,
            conv_directions=config.conv_directions,
        )
        self.train_loader, self.train_dataset_obj = get_data_wrapper_loader(
            file_path=config.train_dataset,
            training_mode=True,
            data_wrapper_class='controlled',
            **dataclasses.asdict(loader_cfg)
        )
        logger.info(f"Train dataset length : {len(self.train_dataset_obj)}")

        # Create a new, separate config for the validation set to ensure immutability
        valid_loader_cfg = dataclasses.replace(loader_cfg, batch=config.valid_batch, shuffle=False)
        
        self.valid_loader, self.valid_dataset_obj = get_data_wrapper_loader(
            file_path=config.valid_dataset,
            training_mode=True, # Augmentation is controlled by mode, not just training_mode
            data_wrapper_class='controlled',
            **dataclasses.asdict(valid_loader_cfg)
        )
        logger.info(f"Valid dataset length : {len(self.valid_dataset_obj)}")

        # Create a separate config for the test set as well
        test_loader_cfg = dataclasses.replace(loader_cfg, batch=1, shuffle=False, augmentation_mode='none')

        self.test_loader, _ = get_data_wrapper_loader(
            file_path=config.test_dataset,
            training_mode=False,
            data_wrapper_class='controlled',
            **dataclasses.asdict(test_loader_cfg)
        )
        logger.info(f"Test dataset length : {len(self.test_loader.dataset)}")

    @error_wrap
    def _set_network(self) -> None:
        self.model = get_model(config).to(config.device)
        self.optimizer = get_optimizer(config, self.model.parameters())
        self.scheduler = ReduceLROnPlateau(self.optimizer, 'max', factor=config.lr_decay, patience=config.lr_tol)
        self.loss_model = get_loss_model(config).to(config.device)

        if config.parallel:
            self.model = DataParallel(self.model)

    @error_wrap
    def _train(self) -> None:
        """train entry point"""
        logger.info("####################################################################################################")
        logger.info("Train start")
        
        for epoch in range(config.train_epoch):
            self.epoch = epoch
            logger.info(f"Epoch: {epoch}")
            logger.info(f"Learning rate: {self.optimizer.param_groups[0]['lr']:.3e}")
            self.global_step = 0
            self.primary_metric = 0.0

            # --- REMOVED ---
            # No longer need to set epoch on the dataset, as augmentation
            # is now based on index.
            # if hasattr(self.train_dataset_obj, 'set_epoch'):
            #     self.train_dataset_obj.set_epoch(epoch)

            self.model.train()
            for i, data in enumerate(tqdm(self.train_loader, leave=False)):
                self.global_step += 1
                image_noise = data[DataKey.image_noise].to(config.device)
                image_gt = data[DataKey.image_gt].to(config.device)

                # Model prediction
                image_pred = self.model(image_noise)

                # Loss calculation
                self.optimizer.zero_grad()
                total_loss = self.loss_model(image_pred, image_gt)

                # Loss backward
                total_loss.backward()
                self.optimizer.step()

            # --- Save checkpoint for every epoch ---
            save_checkpoint(self.model, self.run_dir, epoch=self.epoch, model_type=config.model_type)

            if epoch % config.valid_interval == 0:
                is_best = self._valid()
                if is_best:
                    self.best_metric = self.primary_metric
                    self.tol_count = 0
                else:
                    self.tol_count += 1
                
                # Step the scheduler based on validation metric
                self.scheduler.step(self.primary_metric)

            if self.tol_count > config.valid_tol:
                logger.info("Early stopping triggered")
                break

    @error_wrap
    def _valid(self) -> bool:
        """Validation"""
        logger.info("Valid")
        primary_metric = test_part(
            data_loader=self.valid_loader, 
            network=self.model, 
            run_dir=self.run_dir, 
            save_val=config.save_val, 
            epoch=self.epoch
        )

        self.primary_metric = primary_metric

        if primary_metric > self.best_metric:
            logger.success("Best model renewed")
            self.best_metric = primary_metric
            self.best_epoch = self.epoch
            # Save as the best checkpoint with a fixed name, no epoch number
            save_checkpoint(self.model, self.run_dir, model_type=config.model_type)
            return True
        return False

    @error_wrap
    def _test(self, tag: str) -> None:
        """Test"""
        
        if tag == "best":
            # Load the best checkpoint directly by its fixed name
            checkpoint_path = self.run_dir / "checkpoints" / "checkpoint_best.ckpt"
            if not checkpoint_path.exists():
                logger.warning(f"Best checkpoint not found in {checkpoint_path}. Skipping test.")
                return

            checkpoint = torch.load(checkpoint_path)
            best_epoch_info = checkpoint.get('epoch', self.best_epoch) # Get epoch from checkpoint for logging
            logger.info(f"Test with '{tag}' model from epoch {best_epoch_info}")
            
            state_dict = checkpoint.get('model_state_dict')
            if state_dict:
                model_to_load = self.model.module if isinstance(self.model, torch.nn.DataParallel) else self.model
                model_to_load.load_state_dict(state_dict)
            else:
                logger.error("Could not find a valid state_dict in the checkpoint.")
                return
        else:
            logger.info(f"Test with current model at epoch {self.epoch}")


        test_part(
            data_loader=self.test_loader,
            network=self.model,
            run_dir=self.run_dir, 
            save_val=True, 
            epoch=self.epoch, 
            test_mode=True
        )


def main() -> None:
    """execution entry point"""
    parse_args_for_train_script()
    trainer = Trainer()
    trainer.run()


if __name__ == "__main__":
    main()
