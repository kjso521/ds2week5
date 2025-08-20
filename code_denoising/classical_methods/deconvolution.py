import torch
import numpy as np

from dataset.forward_simulator import dipole_kernel as dipole_kernel_np


class ClippedInverseFilter:
    """
    Performs deconvolution using a clipped inverse filter in the frequency domain,
    based on the specific method required by the project.
    """
    def __init__(self, clip_min: float = -5.0, clip_max: float = 5.0):
        """
        Initializes the clipped inverse filter.
        Args:
            clip_min (float): The minimum value for the inverse filter.
            clip_max (float): The maximum value for the inverse filter.
        """
        self.clip_min = clip_min
        self.clip_max = clip_max

    def run(self, degraded_image: torch.Tensor, B0_dir: tuple[float, float]) -> torch.Tensor:
        """
        Performs deconvolution on the degraded image tensor.
        Args:
            degraded_image (torch.Tensor): The blurry image tensor (shape: [C, H, W] or [B, C, H, W]).
            B0_dir (tuple[float, float]): The direction vector for generating the dipole kernel.
        Returns:
            torch.Tensor: The restored image tensor.
        """
        if not isinstance(degraded_image, torch.Tensor):
            raise TypeError("Input image must be a torch.Tensor.")

        img_shape = degraded_image.shape[-2:]
        device = degraded_image.device

        # Generate kernel using the numpy version from forward_simulator and convert to torch
        kernel_np = dipole_kernel_np(matrix_size=img_shape, B0_dir=B0_dir)
        kernel = torch.from_numpy(np.fft.fftshift(kernel_np)).to(device)

        # Calculate the clipped inverse filter
        kernel_inv = 1 / kernel
        kernel_inv = torch.clip(kernel_inv, min=self.clip_min, max=self.clip_max)

        # Apply the filter in frequency domain
        img_k = torch.fft.fftn(degraded_image, dim=(-2, -1))
        restored_k = img_k * kernel_inv
        restored_image = torch.fft.ifftn(restored_k, dim=(-2, -1)).real
        
        return torch.clip(restored_image, 0, 1)

    def run_on_all_directions(self, degraded_image: torch.Tensor, all_B0_dirs: list[tuple[float, float]]) -> list[torch.Tensor]:
        """
        Runs the deconvolution for all 5 possible directions and returns all results.
        """
        results = []
        for b0_dir in all_B0_dirs:
            results.append(self.run(degraded_image, b0_dir))
        return results
