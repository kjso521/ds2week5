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

def create_results(
    network: torch.nn.Module,
    data_loader: DataLoader,
    result_dir: str,
):
    """
    Generate and save model outputs.
    """
    result_path = Path(result_dir)
    result_path.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Saving results to {result_path}")

    total_psnr = 0.0
    total_ssim = 0.0
    count = 0

    with torch.no_grad():
        for data in tqdm(data_loader, leave=False):
            image_noise = data[DataKey.image_noise].to(config.device)
            filenames = data[DataKey.name]

            image_pred = network(image_noise)

            for i in range(image_pred.shape[0]):
                pred_np = image_pred[i, 0, :, :].cpu().numpy()
                filename = filenames[i]
                
                # Ensure the saved file has the original extension stripped
                base_filename = Path(filename).stem
                
                np.save(result_path / f"{base_filename}.npy", pred_np)
    
    logger.info("Finished creating result files.")


def main():
    parser = argparse.ArgumentParser(description="Create evaluation results from a checkpoint.")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to the model checkpoint file.")
    parser.add_argument("--data_root", type=str, default="dataset/test_y", help="Path to the test data directory.")
    parser.add_argument("--result_dir", type=str, default="result", help="Directory to save the result .npy files.")
    args = parser.parse_args()

    # --- 체크포인트 로드 ---
    if not os.path.exists(args.checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found at {args.checkpoint_path}")

    logger.info(f"Loading checkpoint from: {args.checkpoint_path}")
    checkpoint = torch.load(args.checkpoint_path, map_location=config.device)
    
    # 체크포인트에 저장된 model_type을 사용하여 모델 로드
    config.model_type = checkpoint.get("model_type", "dncnn") # 이전 버전 호환성을 위해 dncnn을 기본값으로
    logger.info(f"Loading model type: {config.model_type}")

    network = get_model(config).to(config.device)
    
    state_dict = checkpoint['model_state_dict']
    
    # DataParallel 래핑 핸들링
    if isinstance(network, torch.nn.DataParallel):
        network.module.load_state_dict(state_dict)
    else:
        network.load_state_dict(state_dict)
    network.eval()

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
    create_results(network, data_loader, args.result_dir)

if __name__ == "__main__":
    main()
