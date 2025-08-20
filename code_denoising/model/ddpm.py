import torch
import torch.nn as nn
import math


class TimeEmbedding(nn.Module):
    """
    Sinusoidal time embedding module.
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        device = t.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = t[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class DiffusionModel(nn.Module):
    """
    Denoising Diffusion Probabilistic Model (DDPM).
    """
    def __init__(self, network: nn.Module, n_steps: int, device: torch.device):
        super().__init__()
        self.network = network
        self.n_steps = n_steps
        self.device = device
        self.beta = torch.linspace(1e-4, 0.02, n_steps).to(device)
        self.alpha = 1. - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)

    def forward(self, x0: torch.Tensor, t: torch.Tensor):
        """
        Forward process (noising).
        """
        noise = torch.randn_like(x0).to(self.device)
        alpha_bar_t = self.alpha_bar[t].reshape(-1, 1, 1, 1)
        xt = torch.sqrt(alpha_bar_t) * x0 + torch.sqrt(1 - alpha_bar_t) * noise
        return xt, noise

    @torch.no_grad()
    def reverse(self, xt: torch.Tensor):
        """
        Reverse process (denoising).
        """
        images = []
        for t in reversed(range(self.n_steps)):
            t_tensor = torch.full((xt.shape[0],), t, device=self.device, dtype=torch.long)
            predicted_noise = self.network(xt, t_tensor)
            
            alpha_t = self.alpha[t]
            alpha_bar_t = self.alpha_bar[t]
            
            xt = (1 / torch.sqrt(alpha_t)) * (xt - ((1 - alpha_t) / torch.sqrt(1 - alpha_bar_t)) * predicted_noise)
            
            if t > 0:
                z = torch.randn_like(xt)
                sigma_t = torch.sqrt(self.beta[t])
                xt += sigma_t * z
            
            images.append(xt.cpu())
        return images
