import torch
from diffusers import DDPMScheduler, UNet2DModel
from PIL import Image
import numpy as np

class HuggingFace_Denoiser:
    """
    A wrapper for a pre-trained Hugging Face Diffusers model to be used as a denoiser.
    This is intended to be a "plug-in" for the PnP algorithm.
    """
    def __init__(self, model_name: str = "google/ddpm-celebahq-256", device: str = "cuda"):
        """
        Initializes the denoiser by loading a pre-trained model from Hugging Face Hub.
        Args:
            model_name (str): The name of the pre-trained model on the Hub.
            device (str): The device to run the model on ('cuda' or 'cpu').
        """
        self.scheduler = DDPMScheduler.from_pretrained(model_name)
        self.model = UNet2DModel.from_pretrained(model_name).to(device)
        self.device = device

    @torch.no_grad()
    def denoise(self, noisy_image: torch.Tensor, noise_level: int = 50) -> torch.Tensor:
        """
        Denoises an image using the DDPM reverse process.
        Args:
            noisy_image (torch.Tensor): The input noisy image tensor (B, C, H, W),
                                        with values expected to be in the range [-1, 1].
            noise_level (int): The starting timestep for the reverse diffusion process.
                               A higher value means more noise is assumed. Range: 0-999.
        Returns:
            torch.Tensor: The denoised image tensor.
        """
        if noise_level < 0 or noise_level >= self.scheduler.config.num_train_timesteps:
            raise ValueError(f"noise_level must be between 0 and {self.scheduler.config.num_train_timesteps - 1}")

        self.scheduler.set_timesteps(self.scheduler.config.num_train_timesteps)
        
        # The denoising process starts from the specified noise_level
        timesteps_to_run = self.scheduler.timesteps[self.scheduler.config.num_train_timesteps - noise_level:]
        
        denoised_image = noisy_image.to(self.device)

        for t in timesteps_to_run:
            # 1. predict noise model_output
            model_output = self.model(denoised_image, t).sample

            # 2. compute previous image: x_t -> x_t-1
            denoised_image = self.scheduler.step(model_output, t, denoised_image).prev_sample
            
        return denoised_image

# Example Usage (for testing)
if __name__ == '__main__':
    # This is a simple test to ensure the class works.
    # It requires a CUDA device to run as is.
    denoiser = HuggingFace_Denoiser()
    
    # Create a dummy noisy image tensor
    dummy_image = torch.randn(1, 3, 256, 256) # B, C, H, W
    
    print("Denoising a dummy image...")
    denoised_result = denoiser.denoise(dummy_image, noise_level=200)
    
    print("Denoising complete.")
    print("Input shape:", dummy_image.shape)
    print("Output shape:", denoised_result.shape)

    # Convert to PIL Image to visualize or save
    denoised_np = denoised_result.squeeze(0).cpu().numpy().transpose(1, 2, 0)
    denoised_np = (denoised_np + 1) / 2 * 255 # Denormalize from [-1, 1] to [0, 255]
    denoised_np = denoised_np.clip(0, 255).astype(np.uint8)
    
    img = Image.fromarray(denoised_np)
    # img.save("denoised_example.png")
    print("Example usage finished successfully.")
