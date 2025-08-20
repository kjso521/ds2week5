# Temporary script to run tests from run_diffusion_tests.ipynb locally
import sys
from pathlib import Path
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

# This is necessary to avoid a bug in matplotlib in some environments
plt.switch_backend('agg')

print("--- Test Runner Initialized ---")

# Add project root to system path
ROOT = Path.cwd()
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

# Import the implemented modules
try:
    from code_denoising.classical_methods.deconvolution import ClippedInverseFilter
    from code_denoising.diffusion_methods.hf_denoiser import HuggingFace_Denoiser
    from code_denoising.diffusion_methods.hf_diffpir import DiffPIR_Pipeline
    from diffusers import DDPMScheduler, UNet2DModel
    print("Imports successful!")
except ImportError as e:
    print(f"Error during import: {e}")
    sys.exit(1)

# --- 1. Load Sample Data ---
print("\n--- 1. Loading Sample Data ---")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# Using the CORRECT filename confirmed via `dir` command
sample_path = ROOT / "dataset/test_y/L1_000d090392623f8046ebe84a1b345bf7.npy"
sample_image_np = np.load(sample_path)
sample_image_torch = torch.from_numpy(sample_image_np).unsqueeze(0).unsqueeze(0).float().to(DEVICE)

B0_DIRS = [(-0.809, -0.5878), (-0.809, 0.5878), (0.309, -0.9511), (0.309, 0.9511), (1.0, 0.0)]

def plot_image(tensor, title=""):
    plt.imshow(tensor.squeeze().cpu().numpy(), cmap='gray')
    plt.title(title)
    plt.axis('off')

# --- 2. Test ClippedInverseFilter ---
print("\n--- 2. Testing ClippedInverseFilter ---")
deconv_filter = ClippedInverseFilter()
restored_images_deconv = deconv_filter.run_on_all_directions(sample_image_torch, B0_DIRS)

plt.figure(figsize=(20, 4))
plt.subplot(1, 6, 1)
plot_image(sample_image_torch, title="Input")
for i, (img, b0_dir) in enumerate(zip(restored_images_deconv, B0_DIRS)):
    plt.subplot(1, 6, i + 2)
    plot_image(img, title=f"Deconv Dir {i+1}")
plt.suptitle("ClippedInverseFilter Results", fontsize=16)
plt.savefig("test_result_ClippedInverseFilter.png")
print("Saved ClippedInverseFilter results to test_result_ClippedInverseFilter.png")

# --- 3. Test HuggingFace_Denoiser ---
print("\n--- 3. Testing HuggingFace_Denoiser ---")
model_save_path = ROOT / "hf_models/ddpm-celebahq-256"
model_name = "google/ddpm-celebahq-256"
denoiser = None

if os.path.exists(model_save_path):
    print(f"Loading model from local path: {model_save_path}")
    denoiser = HuggingFace_Denoiser(model_name=str(model_save_path), device=DEVICE)
else:
    print(f"Downloading model from Hugging Face Hub: {model_name}")
    denoiser = HuggingFace_Denoiser(model_name=model_name, device=DEVICE)
    print(f"Saving model to local path for future use: {model_save_path}")
    denoiser.model.save_pretrained(model_save_path)
    denoiser.scheduler.save_pretrained(model_save_path)

normalized_input = sample_image_torch * 2.0 - 1.0
normalized_input_3ch = normalized_input.repeat(1, 3, 1, 1)
denoised_image = denoiser.denoise(normalized_input_3ch, noise_level=150)
denoised_image_gray = denoised_image.mean(dim=1, keepdim=True)
denoised_image_final = (denoised_image_gray + 1.0) / 2.0

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plot_image(sample_image_torch, title="Input")
plt.subplot(1, 2, 2)
plot_image(denoised_image_final, title="Denoised Output")
plt.suptitle("HuggingFace_Denoiser Result", fontsize=16)
plt.savefig("test_result_HuggingFace_Denoiser.png")
print("Saved HuggingFace_Denoiser results to test_result_HuggingFace_Denoiser.png")

# --- 4. Test DiffPIR_Pipeline ---
print("\n--- 4. Testing DiffPIR_Pipeline ---")
unet = UNet2DModel.from_pretrained(str(model_save_path)).to(DEVICE)
scheduler = DDPMScheduler.from_pretrained(str(model_save_path))
diffpir_pipeline = DiffPIR_Pipeline(unet=unet, scheduler=scheduler)

# --- New: Test DiffPIR for all 5 directions ---
print("--- 4a. Testing DiffPIR for all 5 directions to find the correct one ---")
normalized_input_3ch_diffpir = sample_image_torch.repeat(1, 3, 1, 1)

plt.figure(figsize=(20, 4))
plt.subplot(1, 6, 1)
plot_image(sample_image_torch, title="Input")

for i, b0_dir in enumerate(B0_DIRS):
    print(f"  - Running DiffPIR for direction {i+1}...")
    restored_image_diffpir = diffpir_pipeline(
        degraded_image=normalized_input_3ch_diffpir,
        B0_dir=b0_dir,
        guidance_scale=0.1,  # Lower guidance scale to prevent black images on wrong directions
        num_inference_steps=50
    )
    restored_image_gray = restored_image_diffpir.mean(dim=1, keepdim=True)
    restored_image_final_diffpir = (restored_image_gray + 1.0) / 2.0
    
    plt.subplot(1, 6, i + 2)
    plot_image(restored_image_final_diffpir, title=f"DiffPIR Dir {i+1}")

plt.suptitle("DiffPIR Results for All Directions", fontsize=16)
plt.savefig("test_result_DiffPIR_All_Directions.png")
print("Saved DiffPIR all-direction results to test_result_DiffPIR_All_Directions.png")


# --- Original single-direction test is no longer needed ---
# chosen_b0_dir = B0_DIRS[0]
# ... (rest of the old code is removed)

print("\n--- All tests completed successfully. ---")
