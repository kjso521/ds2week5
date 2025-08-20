import torch
from tqdm import tqdm
import numpy as np

from code_denoising.diffusion_methods.hf_denoiser import HuggingFace_Denoiser
from dataset.forward_simulator import dipole_kernel as dipole_kernel_np

class PnP_Restoration:
    """
    Implements the Plug-and-Play ADMM restoration algorithm by combining a
    data-fidelity step (deconvolution) and a prior step (denoising).
    """
    def __init__(self, denoiser: HuggingFace_Denoiser):
        """
        Initializes the PnP algorithm.
        Args:
            denoiser (HuggingFace_Denoiser): The pre-loaded denoising model.
        """
        self.denoiser = denoiser
        self.device = denoiser.device

    def _get_kernel_fft(self, image: torch.Tensor, B0_dir: tuple[float, float]) -> torch.Tensor:
        """Generates the dipole kernel and returns its FFT."""
        img_shape = image.shape[-2:]
        kernel_np = dipole_kernel_np(matrix_size=img_shape, B0_dir=B0_dir)
        kernel = torch.from_numpy(np.fft.fftshift(kernel_np)).to(self.device)
        return torch.fft.fftn(kernel, dim=(-2, -1))

    def run(
        self,
        degraded_image: torch.Tensor,
        B0_dir: tuple[float, float],
        max_iter: int = 10,
        rho: float = 1.0,
        denoiser_noise_level: int = 50,
    ) -> torch.Tensor:
        """
        Runs the PnP-ADMM restoration loop.
        Args:
            degraded_image (torch.Tensor): The input blurry and noisy image ([0, 1] range).
            B0_dir (tuple[float, float]): The assumed convolution direction.
            max_iter (int): Number of PnP iterations.
            rho (float): ADMM penalty parameter.
            denoiser_noise_level (int): The noise level to pass to the denoiser.
        Returns:
            torch.Tensor: The restored image.
        """
        # Initializations
        x = degraded_image.clone()
        z = degraded_image.clone()
        u = torch.zeros_like(degraded_image)

        # Pre-calculations for x-update (data-fidelity)
        H_fft = self._get_kernel_fft(degraded_image, B0_dir)
        H_t_H = torch.abs(H_fft)**2
        G_fft = torch.fft.fftn(degraded_image, dim=(-2, -1))
        H_t_G = torch.conj(H_fft) * G_fft

        pbar = tqdm(range(max_iter), desc="PnP Restoration")
        for _ in pbar:
            # 1. x-update (Data-Fidelity Step)
            numerator = H_t_G + rho * torch.fft.fftn(z - u, dim=(-2, -1))
            denominator = H_t_H + rho
            x = torch.fft.ifftn(numerator / denominator, dim=(-2, -1)).real
            x = torch.clip(x, 0, 1)

            # 2. z-update (Denoising Step)
            # Denoiser expects input in [-1, 1] and 3 channels
            x_plus_u_norm = (x + u) * 2.0 - 1.0
            x_plus_u_3ch = x_plus_u_norm.repeat(1, 3, 1, 1)

            denoised_3ch = self.denoiser.denoise(x_plus_u_3ch, noise_level=denoiser_noise_level)
            
            # Convert back to grayscale and [0, 1] range
            denoised_gray = denoised_3ch.mean(dim=1, keepdim=True)
            z = (denoised_gray + 1.0) / 2.0
            
            # 3. u-update
            u = u + x - z
        
        return torch.clip(z, 0, 1)
