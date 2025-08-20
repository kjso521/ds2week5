"""
최종 평가를 위한 결과물(.npy 파일) 생성 스크립트.

이 스크립트는 학습된 DnCNN 모델(.ckpt)을 불러와 `test_y` 데이터셋의 모든 이미지를 복원하고,
`evaluate.ipynb`가 요구하는 형식에 맞춰 지정된 폴더에 .npy 파일로 저장합니다.
"""

import sys
from pathlib import Path
import warnings
import argparse
import os

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# --- 중요: 모든 import 이전에 프로젝트 루트 경로를 시스템 경로에 추가 ---
sys.path.append(str(Path(__file__).resolve().parents[1]))

from code_denoising.common.logger import logger
from code_denoising.core_funcs import get_model, ModelType
from code_denoising.datawrapper.datawrapper import BaseDataWrapper, DataKey, get_data_wrapper_loader
from code_denoising.common.utils import save_numpy_as_image
from params import LoaderConfig, config, parse_args_for_eval_script, unetconfig, dncnnconfig

warnings.filterwarnings("ignore")


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
    
    # --- 💡 수정된 모델 타입 결정 로직 ---
    # sys.argv를 직접 확인하여 사용자가 명령줄에서 명시했는지 체크
    user_overrode_model_type = any(arg.startswith('--model_type') for arg in sys.argv)

    if user_overrode_model_type:
        # 사용자가 직접 지정했다면, parse_args_for_eval_script가 이미 config에 설정했으므로 그대로 사용
        logger.info(f"User explicitly specified model type: {config.model_type}")
    else:
        # 사용자가 지정하지 않았다면, 체크포인트에서 정보를 가져옴
        logger.info("Inferring model type from checkpoint...")
        model_type_from_ckpt = checkpoint.get('model_type')
        if model_type_from_ckpt:
            config.model_type = model_type_from_ckpt
        else:
            # 체크포인트에도 정보가 없으면, 마지막 수단으로 기본값을 사용 (이 경우 config의 기본값)
            logger.warning(f"Model type not found in checkpoint. Falling back to default: {config.model_type}")

    logger.info(f"Using model type: {config.model_type}")
    
    # 원상 복구: Trainer.__init__ 로직과 동일하게 model_config를 동적으로 추가
    if ModelType.from_string(config.model_type) == ModelType.Unet:
        config.model_config = unetconfig
    elif ModelType.from_string(config.model_type) == ModelType.DnCNN:
        config.model_config = dncnnconfig

    # Deconvolution 모드일 때만 입력 채널 수를 2로 변경 (효과가 있었던 최소 수정)
    if config.augmentation_mode in ['conv_only', 'both']:
        config.model_config.in_chans = 2
        logger.info("Setting model input channels to 2 for deconvolution.")

    model = get_model(config).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # 4. Setup data loader for the test dataset
    loader_cfg = {
        "data_type": '*.npy',
        "batch": 1,  # Process one image at a time
        "num_workers": 0,
        "shuffle": False,
        "augmentation_mode": 'none', # No augmentation during evaluation
        "training_phase": 'end_to_end', # This doesn't matter for eval but needs a value
        "noise_type": "gaussian",
        "noise_levels": [],
        "conv_directions": []
    }
    test_loader, _ = get_data_wrapper_loader(
        file_path=config.test_dataset,
        training_mode=False,
        data_wrapper_class='controlled',
        **loader_cfg
    )

    # 5. Run inference and save results
    logger.info(f"Starting inference on {len(test_loader.dataset)} images...")
    with torch.no_grad():
        for data in tqdm(test_loader):
            input_tensor = data[DataKey.image_noise].to(device)
            # 파일명을 리스트로 감싸서 오는 경우가 있을 수 있으므로 첫 번째 요소를 사용
            filename = data[DataKey.name][0] if isinstance(data[DataKey.name], list) else data[DataKey.name]

            output_tensor = model(input_tensor)

            output_np = output_tensor.squeeze().cpu().numpy()
            save_path = result_dir / f"{Path(filename).stem}.npy"
            np.save(save_path, output_np)
    
    logger.info(f"Inference complete. Results saved to: {result_dir}")

if __name__ == "__main__":
    main()
