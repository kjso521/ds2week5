import sys
from pathlib import Path

# --- Add project root to system path ---
sys.path.append(str(Path(__file__).resolve().parents[2]))

import time
import warnings
from dataclasses import asdict
import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import os
from tqdm import tqdm

from code_denoising.datawrapper.datawrapper import DataKey, get_data_wrapper_loader, LoaderConfig
from code_denoising.model.unet import Unet
from code_denoising.model.ddpm import DiffusionModel
from code_denoising.core_funcs import get_optimizer, get_loss_model, save_checkpoint
from code_denoising.common.utils import call_next_id, logger_add_handler, separator, error_wrap
from code_denoising.common.logger import logger
from params import config, unetconfig, parse_args_for_train_script


class DiffusionTrainer:
    def __init__(self) -> None:
        self.run_dir = Path(config.run_dir) / f"{call_next_id(Path(config.run_dir)):05d}_diffusion"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        
        logger_add_handler(logger, f"{self.run_dir / 'log.log'}", config.log_lv)
        
        config.device = torch.device(f"cuda:{config.gpu}" if torch.cuda.is_available() else "cpu")
        os.environ["CUDA_VISIBLE_DEVICES"] = str(config.gpu)
        self.device = config.device

        self.best_metric = 0.0
        self.epoch = 0

    def run(self) -> None:
        self._set_data()
        self._set_network()
        self._train()

    def _set_data(self) -> None:
        loader_cfg = LoaderConfig(
            data_type=config.data_type,
            batch=config.train_batch,
            num_workers=config.num_workers,
            shuffle=True,
            augmentation_mode='none', # Diffusion trains on clean images
        )
        self.train_loader, _ = get_data_wrapper_loader(file_path=config.train_dataset, loader_cfg=loader_cfg, training_mode=True)
        
        loader_cfg.shuffle = False
        self.valid_loader, _ = get_data_wrapper_loader(file_path=config.valid_dataset, loader_cfg=loader_cfg, training_mode=False)

    def _set_network(self) -> None:
        # NOTE: For diffusion, Unet is the typical choice for the noise predictor
        unet = Unet(in_chans=unetconfig.in_chans, out_chans=unetconfig.out_chans, chans=unetconfig.chans, num_pool_layers=unetconfig.num_pool_layers, time_emb_dim=32).to(self.device)
        self.model = DiffusionModel(network=unet, n_steps=1000, device=self.device)
        
        self.optimizer = get_optimizer(self.model, config.optimizer, config.lr)
        self.scheduler = ReduceLROnPlateau(self.optimizer, 'max', patience=config.lr_tol, factor=config.lr_decay, verbose=True)
        self.loss_model = get_loss_model(config.loss_model)

    def _train(self) -> None:
        logger.info("Train start")
        for epoch in range(config.train_epoch):
            self.epoch = epoch
            self.model.train()
            pbar = tqdm(self.train_loader)
            for data in pbar:
                self.optimizer.zero_grad()
                
                clean_images = data[DataKey.image_gt].to(self.device)
                t = torch.randint(0, self.model.n_steps, (clean_images.shape[0],)).to(self.device)

                noisy_images, noise = self.model(clean_images, t)
                predicted_noise = self.model.network(noisy_images, t)
                
                loss = self.loss_model(predicted_noise, noise)
                loss.backward()
                self.optimizer.step()
                
                pbar.set_description(f"Epoch {epoch} Loss: {loss.item():.4f}")
            
            if epoch % config.valid_interval == 0:
                self._valid()

    def _valid(self) -> None:
        self.model.eval()
        # TODO: Implement a proper validation logic for diffusion by running the reverse process
        # This is a placeholder as the full reverse process is slow.
        logger.info("Validation step is a placeholder. Saving checkpoint.")
        save_checkpoint(self.model.network, self.run_dir, epoch=self.epoch, model_type="diffusion_unet")


if __name__ == "__main__":
    parse_args_for_train_script()
    config.run_dir = "logs_diffusion"
    config.model_type = "unet" # Noise predictor is Unet
    
    trainer = DiffusionTrainer()
    trainer.run()
