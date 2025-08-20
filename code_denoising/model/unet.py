from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as f

from code_denoising.model.ddpm import TimeEmbedding


def create_down_sample_layers(
    in_chans: int,
    chans: int,
    num_pool_layers: int,
) -> nn.ModuleList:
    layers = nn.ModuleList([ConvBlock(in_chans, chans)])
    ch = chans
    for _ in range(num_pool_layers - 1):
        layers.append(ConvBlock(ch, ch * 2))
        ch *= 2
    return layers


def create_up_sample_layers(
    chans: int,
    num_pool_layers: int,
) -> nn.ModuleList:
    layers = nn.ModuleList()
    ch = chans * (2 ** (num_pool_layers - 1))
    for _ in range(num_pool_layers - 1):
        layers.append(ConvBlock(ch * 2, ch // 2))
        ch //= 2
    layers.append(ConvBlock(ch * 2, ch))
    return layers


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_chans: int,
        out_chans: int,
    ) -> None:
        super().__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(in_chans, out_chans, kernel_size=3, padding=1),
            nn.GroupNorm(4, out_chans),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_chans, out_chans, kernel_size=3, padding=1),
            nn.GroupNorm(4, out_chans),
            nn.SiLU(inplace=True),
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        return self.layers(x)


class Block(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, time_emb_dim: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.relu = nn.SiLU(inplace=True)
        self.time_mlp = nn.Linear(time_emb_dim, out_ch)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        h = self.conv1(x)
        time_emb = self.relu(self.time_mlp(t))
        h += time_emb[(...,) + (None,) * 2]
        h = self.conv2(h)
        return self.relu(h)


class Unet(nn.Module):
    def __init__(
        self,
        in_chans: int,
        out_chans: int,
        chans: int = 32,
        num_pool_layers: int = 4,
        time_emb_dim: int = 32,
    ):
        super().__init__()
        self.time_mlp = TimeEmbedding(time_emb_dim)

        self.down_layers = nn.ModuleList()
        self.up_layers = nn.ModuleList()
        self.pool = nn.MaxPool2d(2)

        prev_chans = in_chans
        for i in range(num_pool_layers):
            out_ch = chans * (2**i)
            self.down_layers.append(Block(prev_chans, out_ch, time_emb_dim))
            prev_chans = out_ch

        self.conv = Block(prev_chans, prev_chans, time_emb_dim)

        for i in reversed(range(num_pool_layers)):
            out_ch = chans * (2**i)
            self.up_layers.append(nn.ConvTranspose2d(prev_chans, out_ch, 2, 2))
            self.up_layers.append(Block(out_ch * 2, out_ch, time_emb_dim))
            prev_chans = out_ch

        self.conv_last = nn.Conv2d(prev_chans, out_chans, 1)

    def forward(self, x: torch.Tensor, t: torch.Tensor | None = None) -> torch.Tensor:
        if t is None:
            # For standard (non-diffusion) Unet usage
            t = torch.zeros(x.shape[0], device=x.device)
        
        t = self.time_mlp(t)
        
        residual_inputs = []
        for i, layer in enumerate(self.down_layers):
            x = layer(x, t)
            residual_inputs.append(x)
            x = self.pool(x)

        x = self.conv(x, t)

        for i in range(0, len(self.up_layers), 2):
            x = self.up_layers[i](x)
            residual_x = residual_inputs.pop()
            x = torch.cat([x, residual_x], dim=1)
            x = self.up_layers[i + 1](x, t)

        return self.conv_last(x)


# Example usage:
if __name__ == "__main__":
    print("Testing U-Net...")
    input_tensor = torch.randn(1, 1, 64, 64)  # Example input tensor
    print(f"Input shape: {input_tensor.shape}")

    model = Unet(
        in_chans=1,
        out_chans=1,
        chans=24,
        num_pool_layers=4,
    )
    output_tensor = model(input_tensor)

    print(f"Output shape: {output_tensor.shape}")
