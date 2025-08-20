import numpy as np
import torch
from numpy.fft import fft2, ifft2
from tqdm import tqdm

from code_denoising.model.ddpm import DiffusionModel

class PnP_ADMM_Restoration:
    """
    Plug-and-Play ADMM for Image Restoration (Deconvolution + Denoising).
    """
    def __init__(self, denoiser: DiffusionModel, rho: float = 1.0, max_iter: int = 10):
        """
        Initializes the PnP-ADMM algorithm.
        Args:
            denoiser (DiffusionModel): A pre-trained diffusion model for the denoising step.
            rho (float): ADMM penalty parameter.
            max_iter (int): Number of iterations to run.
        """
        self.denoiser = denoiser
        self.rho = rho
        self.max_iter = max_iter

    def _get_padded_kernel_fft(self, image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        padded_kernel = np.zeros_like(image, dtype=np.float32)
        k_h, k_w = kernel.shape
        im_h, im_w = image.shape
        start_h, start_w = (im_h - k_h) // 2, (im_w - k_w) // 2
        padded_kernel[start_h : start_h + k_h, start_w : start_w + k_w] = kernel
        return fft2(np.fft.fftshift(padded_kernel))

    def run(self, degraded_image_np: np.ndarray, kernel_np: np.ndarray) -> np.ndarray:
        # Initializations
        x = degraded_image_np.copy()
        z = degraded_image_np.copy()
        u = np.zeros_like(degraded_image_np)

        # Pre-calculations for x-update
        H = self._get_padded_kernel_fft(degraded_image_np, kernel_np)
        H_t_H = np.abs(H)**2
        G_fft = fft2(degraded_image_np)
        H_t_G = np.conj(H) * G_fft

        device = self.denoiser.device

        for _ in tqdm(range(self.max_iter), desc="PnP-ADMM Iterations"):
            # 1. x-update (Data-Fidelity)
            numerator = H_t_G + self.rho * fft2(z - u)
            denominator = H_t_H + self.rho
            x = np.real(ifft2(numerator / denominator))
            x = np.clip(x, 0, 1)

            # 2. z-update (Denoising)
            with torch.no_grad():
                x_plus_u_tensor = torch.from_numpy(x + u).unsqueeze(0).unsqueeze(0).float().to(device)
                # Run the full reverse diffusion process for denoising
                denoised_tensor_list = self.denoiser.reverse(x_plus_u_tensor)
                z_tensor = denoised_tensor_list[-1] # Get the final denoised image
                z = z_tensor.squeeze().cpu().numpy()
            
            # 3. u-update
            u = u + x - z
            
        return np.clip(z, 0, 1)
