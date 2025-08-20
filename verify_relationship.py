import numpy as np
import torch
from pathlib import Path
import re
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
from scipy.stats import skew, kurtosis, norm
import random
import sys

# Add project directories to the python path to import simulators
sys.path.append(str(Path(__file__).parent / 'dataset'))
from forward_simulator import ForwardSimulator

def parse_v2_filename(path):
    """Parses the filename to extract the base name, direction, and noise level."""
    match = re.search(r'(.+)_dir_(-?\d+\.\d+)_(-?\d+\.\d+)_noise_lv(\d+)\.npy', path.name)
    if not match:
        return None, None, None
    base_name = match.group(1)
    direction = (float(match.group(2)), float(match.group(3)))
    noise_level = int(match.group(4))
    return base_name, direction, noise_level

def verify_hypothesis_2():
    """
    Verifies Hypothesis 2: test_y_v2 = Conv(test_y) + Noise.
    Calculates Residual = test_y_v2 - Conv(test_y) and checks if it's pure Gaussian noise.
    """
    base_dir = Path(__file__).parent
    test_y_dir = base_dir / 'dataset' / 'test_y'
    test_y_v2_dir = base_dir / 'dataset' / 'test_y_v2'
    output_dir = base_dir / "relationship_verification_results"
    output_dir.mkdir(exist_ok=True)

    if not test_y_dir.exists() or not test_y_v2_dir.exists():
        print("Error: Dataset directories not found.")
        return

    residuals = {1: [], 2: []}
    forward_sim = ForwardSimulator()
    v2_files = list(test_y_v2_dir.glob('*.npy'))

    print("Verifying Hypothesis 2 by calculating residuals...")
    for v2_path in tqdm(v2_files, desc="Processing image pairs"):
        base_name, direction, noise_level = parse_v2_filename(v2_path)
        if base_name is None: continue

        original_y_path = test_y_dir / f'{base_name}.npy'
        if not original_y_path.exists(): continue

        # Load images
        y_v2_np = np.load(v2_path)
        y_np = np.load(original_y_path)
        y_tensor = torch.from_numpy(y_np).float().unsqueeze(0)

        # Apply the same convolution to test_y
        convolved_y_tensor = forward_sim(y_tensor, B0_dir=direction)
        convolved_y_np = convolved_y_tensor.squeeze().cpu().numpy()

        # Calculate the residual
        residual = y_v2_np - convolved_y_np
        residuals[noise_level].append(residual.flatten())

    print("\n--- Analysis of Residuals (test_y_v2 - Conv(test_y)) ---")
    
    # Perform statistical analysis and judgment
    for level in sorted(residuals.keys()):
        if not residuals[level]:
            print(f"\n--- Level {level} ---")
            print("No valid data pairs found.")
            continue

        all_residuals = np.concatenate(residuals[level])
        
        # Statistical tests
        mean_val = np.mean(all_residuals)
        std_val = np.std(all_residuals)
        skew_val = skew(all_residuals)
        kurt_val = kurtosis(all_residuals) # Fisher's kurtosis (normal = 0)

        # Judgment
        is_gaussian_like = (abs(mean_val) < 1e-3) and (abs(skew_val) < 0.1) and (abs(kurt_val) < 0.1)
        
        print(f"\n--- Level {level} ---")
        print(f"Mean                  : {mean_val:.6f}")
        print(f"Std Dev (σ)           : {std_val:.6f}")
        print(f"Skewness              : {skew_val:.6f} (Gaussian ideal: 0)")
        print(f"Kurtosis (Fisher)     : {kurt_val:.6f} (Gaussian ideal: 0)")
        
        print("\n[Judgment]")
        if is_gaussian_like:
            print("✅ The residual appears to be pure, zero-mean Gaussian noise.")
            print("   This STRONGLY SUPPORTS Hypothesis 2.")
            print(f"   The true Ground-Truth noise sigma for this level is ≈ {std_val:.4f}")
        else:
            print("❌ The residual does NOT appear to be pure Gaussian noise.")
            print("   This REJECTS Hypothesis 2.")

    # --- Comprehensive Visualization for a random sample ---
    print("\nGenerating comprehensive visual reports for 5 random samples...")
    
    # Ensure we don't pick the same sample multiple times if the list is small
    num_reports = 5
    if len(v2_files) < num_reports:
        sample_paths = v2_files
    else:
        sample_paths = random.sample(v2_files, num_reports)

    for i, random_v2_path in enumerate(sample_paths):
        print(f"[{i+1}/{num_reports}] Generating report for {random_v2_path.name}...")
        base_name, direction, noise_level = parse_v2_filename(random_v2_path)
        original_y_path = test_y_dir / f'{base_name}.npy'

        if not original_y_path.exists():
            print(f"  > Skipping, corresponding test_y file not found.")
            continue

        y_v2_np = np.load(random_v2_path)
        y_np = np.load(original_y_path)
        y_tensor = torch.from_numpy(y_np).float().unsqueeze(0)
        convolved_y_tensor = forward_sim(y_tensor, B0_dir=direction)
        convolved_y_np = convolved_y_tensor.squeeze().cpu().numpy()
        residual_np = y_v2_np - convolved_y_np

        # Plotting
        fig = plt.figure(figsize=(18, 10))
        fig.suptitle(f'Hypothesis 2 Verification: Sample "{base_name}" (Noise Lv {noise_level})', fontsize=20)

        # Images
        ax1 = plt.subplot2grid((2, 3), (0, 0))
        ax1.imshow(y_np, cmap='gray')
        ax1.set_title('Original `test_y`', fontsize=16)
        ax1.axis('off')

        ax2 = plt.subplot2grid((2, 3), (0, 1))
        ax2.imshow(convolved_y_np, cmap='gray')
        ax2.set_title('`Conv(test_y)`\n(Simulated)', fontsize=16)
        ax2.axis('off')
        
        ax3 = plt.subplot2grid((2, 3), (0, 2))
        ax3.imshow(y_v2_np, cmap='gray')
        ax3.set_title('Actual `test_y_v2`', fontsize=16)
        ax3.axis('off')

        # Residual Image
        ax4 = plt.subplot2grid((2, 3), (1, 0))
        im = ax4.imshow(residual_np, cmap='gray')
        ax4.set_title('Residual Image (`test_y_v2 - Conv(test_y)`)', fontsize=16)
        ax4.axis('off')
        plt.colorbar(im, ax=ax4)

        # Histogram and Gaussian Fit
        ax5 = plt.subplot2grid((2, 3), (1, 1), colspan=2)
        all_residuals_level = np.concatenate(residuals[noise_level])
        mu, std = norm.fit(all_residuals_level)
        ax5.hist(all_residuals_level, bins=256, density=True, alpha=0.8, label='Residual Histogram', range=(-0.5, 0.5))
        xmin, xmax = ax5.get_xlim()
        x = np.linspace(xmin, xmax, 100)
        p = norm.pdf(x, mu, std)
        ax5.plot(x, p, 'r', linewidth=2, label='Fitted Gaussian Curve')
        ax5.set_title(f'Residual Distribution vs. Ideal Gaussian (Level {noise_level})', fontsize=16)
        ax5.legend()
        ax5.grid(True)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        report_path = output_dir / f'verification_report_lv{noise_level}_sample_{base_name}_{i+1}.png'
        plt.savefig(report_path)
        print(f"  > Visual report saved to: {report_path}")

if __name__ == '__main__':
    verify_hypothesis_2()
