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
from code_denoising.datawrapper.datawrapper import BaseDataWrapper, DataKey, get_data_wrapper_loader, LoaderConfig
from code_denoising.common.utils import save_numpy_as_image
from params import config, parse_args_for_eval_script, unetconfig, dncnnconfig

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

    # Load Checkpoint
    logger.info(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # 평가 시에는 augmentation_mode를 'none'으로 강제하여 채널 불일치 방지
    config.augmentation_mode = 'none'
    logger.info("Setting augmentation_mode to 'none' for evaluation.")

    # Get model type from checkpoint if not provided
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
    
    # 💡 --- 모델 설정 복원 로직 --- 💡
    # 1. 체크포인트에서 model_config를 직접 가져오기 (신규 체크포인트 방식)
    model_config_from_ckpt = checkpoint.get('model_config')
    
    if model_config_from_ckpt:
        logger.info("Found model_config in checkpoint. Using it to build the model.")
        config.model_config = model_config_from_ckpt
    else:
        # 2. model_config가 없는 경우, 기존 방식으로 설정 (하위 호환성)
        logger.warning("model_config not found in checkpoint. Falling back to default config from params.py.")
        if ModelType.from_string(config.model_type) == ModelType.Unet:
            config.model_config = unetconfig
        elif ModelType.from_string(config.model_type) == ModelType.DnCNN:
            config.model_config = dncnnconfig

    # 💡 --- 수동 채널 설정 로직 --- 💡
    # 사용자가 --model_channels 인자를 사용했다면, 모든 설정을 무시하고 채널 수를 강제로 덮어쓴다.
    if config.model_channels_override is not None:
        logger.warning(f"User is manually overriding model channels to: {config.model_channels_override}")
        if ModelType.from_string(config.model_type) == ModelType.Unet:
            config.model_config.in_chans = config.model_channels_override
            config.model_config.out_chans = config.model_channels_override
        elif ModelType.from_string(config.model_type) == ModelType.DnCNN:
            config.model_config.channels = config.model_channels_override

    # 모델 생성
    model = get_model(config.model_config, config.model_type).to(device)
    
    # 모델 가중치 불러오기 (Size Mismatch 오류 방지)
    try:
        model.load_state_dict(checkpoint['model_state_dict'])
    except RuntimeError as e:
        logger.error(f"Failed to load state_dict, likely due to a model architecture mismatch: {e}")
        logger.error("Please ensure the checkpoint was trained with a compatible architecture.")
        sys.exit(1) # 오류 발생 시 스크립트 중단
        
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

            # 💡 --- 입력 채널 동적 맞춤 로직 ---
            # 모델이 기대하는 입력 채널 수를 확인 (Unet: in_chans, DnCNN: channels)
            model_in_channels = getattr(config.model_config, 'channels', getattr(config.model_config, 'in_chans', 1))

            # 모델은 2채널을 원하는데 입력이 1채널인 경우, 0으로 채워진 두 번째 채널을 추가
            if model_in_channels > 1 and input_tensor.shape[1] == 1:
                zeros = torch.zeros_like(input_tensor)
                input_tensor = torch.cat((input_tensor, zeros), dim=1)

            output_tensor = model(input_tensor)

            # 💡 --- 출력 채널 동적 맞춤 로직 ---
            # 모델 출력이 2채널(complex)인 경우, 평가를 위해 1채널 크기(magnitude)로 변환
            if output_tensor.shape[1] == 2:
                output_tensor = torch.sqrt(output_tensor[:, 0, :, :]**2 + output_tensor[:, 1, :, :]**2).unsqueeze(1)

            output_np = output_tensor.squeeze().cpu().numpy()
            save_path = result_dir / f"{Path(filename).stem}.npy"
            np.save(save_path, output_np)
    
    logger.info(f"Inference complete. Results saved to: {result_dir}")

if __name__ == "__main__":
    main()
    # Force reload in Colab environment
