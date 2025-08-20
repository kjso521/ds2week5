"""
[Sample Generation Script]

이 스크립트는 데이터 증강(Data Augmentation) 파이프라인을 수정하기 전,
컨볼루션(Convolution)과 노이즈(Noise) 추가 로직을 안전하게 테스트하기 위해 제작되었습니다.

[사용법]
1. 프로젝트 최상위 폴더에서 아래 명령어를 실행합니다.
   `python generate_samples.py`

2. 실행이 완료되면, 프로젝트 최상위 폴더에 `IMAGES`라는 새 폴더가 생성됩니다.
   이 폴더 안에서 원본 이미지와 다양하게 오염된 결과 이미지들을 .png 파일로 직접 확인할 수 있습니다.

[설정 변경]
스크립트 상단의 'Configuration' 섹션에서 아래 값들을 변경하여 실험해볼 수 있습니다.
- NUM_SAMPLES: 처리할 샘플 이미지의 개수
- SIGMA_L1, SIGMA_L2: 적용할 노이즈의 강도 (표준편차)
- CONV_DIRECTIONS: 테스트해볼 컨볼루션의 방향 벡터
"""
import numpy as np
import torch
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
import sys

# Add project directories to the python path to import simulators
# This allows the script to be run from the root directory
sys.path.append(str(Path(__file__).parent / 'code_denoising/datawrapper'))
sys.path.append(str(Path(__file__).parent / 'dataset'))

from noise_simulator import NoiseSimulator, NoisyType
from forward_simulator import ForwardSimulator

# --- Configuration ---
# You can modify these parameters for your experiments
NUM_SAMPLES = 3  # Number of random train images to process
OUTPUT_DIR = Path("IMAGES")
TRAIN_DIR = Path("dataset/train")

# Noise levels from our final analysis (robust_noise_analyzer.py)
SIGMA_L1 = 0.070
SIGMA_L2 = 0.132

# A few sample convolution directions to test
CONV_DIRECTIONS = [
    (0.0, 1.0),   # Vertical
    (1.0, 0.0),   # Horizontal
    (0.707, 0.707) # Diagonal
]

# CONV_DIRECTIONS = []

# --- End of Configuration ---

def load_npy_as_tensor(path: Path) -> torch.Tensor:
    """Loads a .npy file and converts it to a PyTorch tensor."""
    img = torch.from_numpy(np.load(path)).float()
    if len(img.shape) == 2:
        img = img.unsqueeze(0)
    return img

def save_tensor_as_png(tensor: torch.Tensor, path: Path):
    """Saves a PyTorch tensor as a grayscale PNG image."""
    # Ensure the directory exists
    path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert tensor to numpy array for saving
    img_np = tensor.squeeze().cpu().numpy()
    
    plt.imsave(path, img_np, cmap='gray')

def main():
    """Main function to generate and save sample corrupted images."""
    print("--- Starting Sample Generation ---")

    # --- Print current settings ---
    print("\n[Current Settings]")
    print(f"- Number of samples: {NUM_SAMPLES}")
    print(f"- Noise Sigma L1: {SIGMA_L1}")
    print(f"- Noise Sigma L2: {SIGMA_L2}")
    if not CONV_DIRECTIONS:
        print("- Convolution: [DISABLED]")
    else:
        print(f"- Convolution Directions: {len(CONV_DIRECTIONS)} enabled -> {CONV_DIRECTIONS}")
    print("-" * 20)
    
    # 1. Check for train directory and create output directory
    if not TRAIN_DIR.is_dir():
        print(f"Error: Training data directory not found at '{TRAIN_DIR}'")
        return
    OUTPUT_DIR.mkdir(exist_ok=True)
    print(f"Output will be saved to '{OUTPUT_DIR}' directory.")

    # 2. Get a list of all training images and select random samples
    all_train_files = list(TRAIN_DIR.glob('*.npy'))
    if len(all_train_files) < NUM_SAMPLES:
        print(f"Error: Not enough images in '{TRAIN_DIR}' to generate {NUM_SAMPLES} samples.")
        return
    sample_files = random.sample(all_train_files, NUM_SAMPLES)
    print(f"Selected {len(sample_files)} random images to process.\n")

    # 3. Prepare the simulators
    forward_sim = ForwardSimulator()
    noise_sim_l1 = NoiseSimulator(noise_type=NoisyType.Gaussian, noise_sigma=SIGMA_L1)
    noise_sim_l2 = NoiseSimulator(noise_type=NoisyType.Gaussian, noise_sigma=SIGMA_L2)

    total_generated_count = 0
    # 4. Process each sample file
    for i, file_path in enumerate(sample_files):
        base_name = file_path.stem
        print(f"[{i+1}/{NUM_SAMPLES}] Processing: {file_path.name}")
        
        # Load the original clean image
        original_img = load_npy_as_tensor(file_path)
        
        # --- Save Original ---
        save_path = OUTPUT_DIR / f"{base_name}_0_original.png"
        save_tensor_as_png(original_img, save_path)
        print(f"  > Saved: {save_path.name}")
        total_generated_count += 1

        # --- Generate and Save Noisy Samples ---
        noisy_l1 = noise_sim_l1(original_img)
        noisy_l2 = noise_sim_l2(original_img)
        save_path_l1 = OUTPUT_DIR / f"{base_name}_1_noise_L1.png"
        save_path_l2 = OUTPUT_DIR / f"{base_name}_1_noise_L2.png"
        save_tensor_as_png(noisy_l1, save_path_l1)
        print(f"  > Saved: {save_path_l1.name}")
        save_tensor_as_png(noisy_l2, save_path_l2)
        print(f"  > Saved: {save_path_l2.name}")
        total_generated_count += 2
        
        # --- Generate and Save Convolved and Noisy Samples ---
        for direction in CONV_DIRECTIONS:
            dir_str = f"{direction[0]}_{direction[1]}"

            # Only convolution
            convolved_img = forward_sim(original_img, B0_dir=direction)
            save_path_conv = OUTPUT_DIR / f"{base_name}_2_conv_{dir_str}.png"
            save_tensor_as_png(convolved_img, save_path_conv)
            print(f"  > Saved: {save_path_conv.name}")
            total_generated_count += 1

            # Convolution + Noise L1
            conv_noisy_l1 = noise_sim_l1(convolved_img)
            save_path_cnl1 = OUTPUT_DIR / f"{base_name}_3_conv_{dir_str}_noise_L1.png"
            save_tensor_as_png(conv_noisy_l1, save_path_cnl1)
            print(f"  > Saved: {save_path_cnl1.name}")
            total_generated_count += 1
            
            # Convolution + Noise L2
            conv_noisy_l2 = noise_sim_l2(convolved_img)
            save_path_cnl2 = OUTPUT_DIR / f"{base_name}_3_conv_{dir_str}_noise_L2.png"
            save_tensor_as_png(conv_noisy_l2, save_path_cnl2)
            print(f"  > Saved: {save_path_cnl2.name}")
            total_generated_count += 1
        
        print("-" * 20)


    print("\n--- Sample Generation Complete! ---")
    print(f"Processed {NUM_SAMPLES} original images.")
    print(f"Generated a total of {total_generated_count} sample images in '{OUTPUT_DIR}'.")


if __name__ == "__main__":
    main()
