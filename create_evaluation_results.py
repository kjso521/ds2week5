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
from params import config, parse_args_for_eval_script  # Import the new parsing function
from code_denoising.common.utils import logger

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
    """
    Main function to run the evaluation.
    """
    # 1. Parse arguments and update the global config
    parse_args_for_eval_script()

    # 2. Setup paths and device
    checkpoint_path = Path(config.checkpoint_path)
    result_dir = Path(config.result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 3. Load model from checkpoint
    logger.info(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Infer model type from checkpoint if not overridden
    if not config.model_type:
        model_type_from_ckpt = checkpoint.get('model_type')
        if not model_type_from_ckpt:
            raise ValueError("Model type not found in checkpoint and not provided as an argument.")
        config.model_type = model_type_from_ckpt
    
    logger.info(f"Using model type: {config.model_type}")
    model = get_model(config).to(device)
    state_dict = checkpoint.get('model_state_dict')
    if not state_dict:
        raise ValueError("Could not find a valid 'model_state_dict' in the checkpoint.")
    model.load_state_dict(state_dict)
    model.eval()

    # 4. Setup data loader for the test dataset
    loader_cfg = LoaderConfig(
        data_type='*.npy',
        batch=1,  # Process one image at a time
        num_workers=0,
        shuffle=False,
        augmentation_mode='none', # No augmentation during evaluation
        training_phase='end_to_end', # This doesn't matter for eval but needs a value
        noise_type="gaussian",
        noise_levels=[],
        conv_directions=[]
    )
    test_loader, _ = get_data_wrapper_loader(
        file_path=config.test_dataset,
        loader_cfg=loader_cfg,
        training_mode=False,
        data_wrapper_class='controlled' # Use the flexible datawrapper
    )

    # 5. Run inference and save results
    logger.info(f"Starting inference on {len(test_loader.dataset)} images...")
    with torch.no_grad():
        for data in tqdm(test_loader):
            input_tensor = data[DataKey.image_noise].to(device)
            filename = data[DataKey.filename]

            output_tensor = model(input_tensor)

            output_np = output_tensor.squeeze().cpu().numpy()
            save_path = result_dir / f"{Path(filename).stem}.npy"
            np.save(save_path, output_np)
    
    logger.info(f"Inference complete. Results saved to: {result_dir}")

if __name__ == "__main__":
    main()
