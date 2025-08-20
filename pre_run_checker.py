import sys
from pathlib import Path
import os

print("--- Starting Pre-run Environment Check ---")
success = True

# 1. Check Custom Modules and System Path
print("\n[1/4] Checking project structure and module imports...")
try:
    ROOT = Path.cwd()
    if str(ROOT) not in sys.path:
        sys.path.append(str(ROOT))
    
    from code_denoising.classical_methods.deconvolution import ClippedInverseFilter
    from code_denoising.diffusion_methods.hf_denoiser import HuggingFace_Denoiser
    from code_denoising.diffusion_methods.hf_diffpir import DiffPIR_Pipeline
    print("  - [OK] Custom modules imported successfully.")
except ImportError as e:
    print(f"  - [FAIL] Could not import custom modules. Error: {e}")
    success = False

# 2. Check Essential Libraries
print("\n[2/4] Checking essential libraries...")
try:
    import numpy as np
    import matplotlib
    from PIL import Image
    from diffusers import DDPMScheduler, UNet2DModel
    import torch
    print("  - [OK] All essential libraries are installed and imported.")
except ImportError as e:
    print(f"  - [FAIL] A required library is missing. Error: {e}")
    print("  - Please run 'pip install numpy matplotlib Pillow diffusers transformers accelerate torch'")
    success = False

# 3. Check PyTorch and GPU status
if 'torch' in sys.modules:
    print("\n[3/4] Checking PyTorch and GPU status...")
    try:
        torch_version = torch.__version__
        print(f"  - [OK] PyTorch version: {torch_version}")
        
        is_cuda_available = torch.cuda.is_available()
        if is_cuda_available:
            gpu_name = torch.cuda.get_device_name(0)
            print(f"  - [OK] GPU detected: {gpu_name}")
        else:
            print("  - [WARNING] No CUDA-enabled GPU detected. Tests will run on CPU and will be extremely slow.")
            
    except Exception as e:
        print(f"  - [FAIL] PyTorch or CUDA check failed. Error: {e}")
        success = False

# 4. Check Data Path
print("\n[4/4] Checking for sample data file...")
try:
    sample_path = ROOT / "dataset/test_y/00000.npy"
    if os.path.exists(sample_path):
        print(f"  - [OK] Sample data found at: {sample_path}")
    else:
        print(f"  - [FAIL] Sample data file not found at: {sample_path}")
        success = False
except Exception as e:
    print(f"  - [FAIL] Error checking data path. Error: {e}")
    success = False


# Final Result
print("\n--- Check Complete ---")
if success:
    print("Environment check passed successfully. You can now run the main test notebook.")
else:
    print("Environment check failed. Please resolve the issues above before proceeding.")
    sys.exit(1)
