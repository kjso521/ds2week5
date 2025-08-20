"""
최종 평가를 위한 결과물(.npy 파일) 생성 스크립트.

이 스크립트는 학습된 DnCNN 모델(.ckpt)을 불러와 `test_y` 데이터셋의 모든 이미지를 복원하고,
`evaluate.ipynb`가 요구하는 형식에 맞춰 지정된 폴더에 .npy 파일로 저장합니다.
"""

import argparse
import os
from pathlib import Path
import sys
import warnings

# --- 중요: 모든 import 이전에 프로젝트 루트 경로를 시스템 경로에 추가 ---
# 이 스크립트가 실행되는 위치를 기준으로, 상위 1단계 폴더(week5)를 경로에 추가합니다.
sys.path.append(str(Path(__file__).resolve().parents[1]))

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

# 프로젝트 루트를 기준으로 필요한 모듈 import
from code_denoising.datawrapper.datawrapper import DataKey, get_data_wrapper_loader, LoaderConfig
from code_denoising.core_funcs import get_model
from params import config
from code_denoising.common.logger import logger

warnings.filterwarnings("ignore")

def create_sbs_results(
    denoising_network: torch.nn.Module,
    deconv_network: torch.nn.Module,
    data_loader: DataLoader,
    result_dir: str,
):
    """
    Generate and save model outputs from a step-by-step pipeline.
    """
    result_path = Path(result_dir)
    result_path.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Saving SBS results to {result_path}")

    with torch.no_grad():
        for data in tqdm(data_loader, leave=False):
            image_noise = data[DataKey.image_noise].to(config.device)
            filenames = data[DataKey.name]

            # Step 1: Denoising
            denoised_image = denoising_network(image_noise)
            # Step 2: Deconvolution
            final_image = deconv_network(denoised_image)


            for i in range(final_image.shape[0]):
                pred_np = final_image[i, 0, :, :].cpu().numpy()
                filename = filenames[i]
                
                base_filename = Path(filename).stem
                
                np.save(result_path / f"{base_filename}.npy", pred_np)
    
    logger.info("Finished creating SBS result files.")


def main():
    parser = argparse.ArgumentParser(description="Create Step-by-Step evaluation results from two checkpoints.")
    parser.add_argument("--denoising_ckpt_path", type=str, required=True, help="Path to the Denoising model checkpoint file.")
    parser.add_argument("--deconv_ckpt_path", type=str, required=True, help="Path to the Deconvolution model checkpoint file.")
    parser.add_argument("--data_root", type=str, default="dataset/test_y", help="Path to the test data directory.")
    parser.add_argument("--result_dir", type=str, default="result_sbs", help="Directory to save the result .npy files.")
    args = parser.parse_args()

    # --- 1. Denoising 모델 로드 ---
    if not os.path.exists(args.denoising_ckpt_path):
        raise FileNotFoundError(f"Denoising checkpoint not found at {args.denoising_ckpt_path}")

    logger.info(f"Loading Denoising checkpoint from: {args.denoising_ckpt_path}")
    denoising_checkpoint = torch.load(args.denoising_ckpt_path, map_location=config.device)
    
    config.model_type = denoising_checkpoint.get("model_type", "dncnn")
    logger.info(f"Loading Denoising model type: {config.model_type}")
    denoising_network = get_model(config).to(config.device)
    
    denoising_state_dict = denoising_checkpoint['model_state_dict']
    if isinstance(denoising_network, torch.nn.DataParallel):
        denoising_network.module.load_state_dict(denoising_state_dict)
    else:
        denoising_network.load_state_dict(denoising_state_dict)
    denoising_network.eval()

    # --- 2. Deconvolution 모델 로드 ---
    if not os.path.exists(args.deconv_ckpt_path):
        raise FileNotFoundError(f"Deconvolution checkpoint not found at {args.deconv_ckpt_path}")

    logger.info(f"Loading Deconvolution checkpoint from: {args.deconv_ckpt_path}")
    deconv_checkpoint = torch.load(args.deconv_ckpt_path, map_location=config.device)

    config.model_type = deconv_checkpoint.get("model_type", "unet")
    logger.info(f"Loading Deconvolution model type: {config.model_type}")
    deconv_network = get_model(config).to(config.device)

    deconv_state_dict = deconv_checkpoint['model_state_dict']
    if isinstance(deconv_network, torch.nn.DataParallel):
        deconv_network.module.load_state_dict(deconv_state_dict)
    else:
        deconv_network.load_state_dict(deconv_state_dict)
    deconv_network.eval()

    # --- 데이터 로더 설정 ---
    loader_cfg = LoaderConfig(
        data_type=config.data_type,
        batch=8,
        num_workers=0,
        shuffle=False,
        augmentation_mode='none',
        noise_type=config.noise_type,
        noise_levels=config.noise_levels,
        conv_directions=config.conv_directions
    )
    data_loader, _ = get_data_wrapper_loader(
        file_path=[args.data_root],
        loader_cfg=loader_cfg,
        training_mode=False,
    )

    if not data_loader:
        logger.error(f"Failed to create data loader from {args.data_root}. No data found?")
        return
        
    # --- 결과 생성 ---
    create_sbs_results(denoising_network, deconv_network, data_loader, args.result_dir)

if __name__ == "__main__":
    main()
