import torch
from diffusers import DDPMPipeline
from tqdm.auto import tqdm
import numpy as np

from dataset.forward_simulator import dipole_kernel as dipole_kernel_np

class DiffPIR_Pipeline(DDPMPipeline):
    """
    A custom DiffPIR pipeline for deconvolution, inheriting from the base DDPMPipeline.
    """
    @torch.no_grad()
    def __call__(
        self,
        degraded_image: torch.Tensor,
        B0_dir: tuple[float, float],
        guidance_scale: float = 1.0,
        num_inference_steps: int = 100,
    ):
        # 0. Initial setup
        self.scheduler.set_timesteps(num_inference_steps)
        batch_size = degraded_image.shape[0]
        device = self.device
        
        # 1. Prepare kernel for guidance
        img_shape = degraded_image.shape[-2:]
        kernel_np = dipole_kernel_np(matrix_size=img_shape, B0_dir=B0_dir)
        kernel = torch.from_numpy(np.fft.fftshift(kernel_np)).to(device)
        kernel_fft = torch.fft.fftn(kernel, dim=(-2, -1))
        
        # 2. Start with random noise (standard DDPM reverse process)
        image = torch.randn(degraded_image.shape, device=device)
        
        for t in self.progress_bar(self.scheduler.timesteps):
            # --- Inside the reverse diffusion loop ---
            
            # A. Standard denoising step
            with torch.no_grad():
                model_output = self.unet(image, t).sample
            
            # B. Guidance calculation (the core of DiffPIR)
            # Estimate x_0 from x_t and the predicted noise
            x_0_pred = self.scheduler.step(model_output, t, image).pred_original_sample
            
            # C. Calculate the guidance gradient
            # This requires enabling gradients temporarily
            with torch.enable_grad():
                x_0_pred_grad = x_0_pred.detach().requires_grad_(True)
                
                # Apply convolution in frequency domain
                x_0_fft = torch.fft.fftn(x_0_pred_grad, dim=(-2, -1))
                y_pred_fft = x_0_fft * kernel_fft
                y_pred = torch.fft.ifftn(y_pred_fft, dim=(-2, -1)).real
                
                # Calculate the loss (data fidelity term)
                loss = torch.nn.functional.mse_loss(y_pred, degraded_image)
                
                # Get the gradient of the loss w.r.t. the predicted clean image
                gradient = torch.autograd.grad(loss, x_0_pred_grad)[0]

            # D. Update the model output with the guidance
            # The gradient tells us how to change the output to better match the degraded image
            model_output = model_output - guidance_scale * gradient

            # E. Perform the actual scheduler step to get x_{t-1}
            image = self.scheduler.step(model_output, t, image).prev_sample

        return image
