import torch
from torch import Tensor, nn


class DnCNN(nn.Module):
    def __init__(
        self,
        channels: int,
        num_of_layers: int,
        kernel_size: int,
        padding: int,
        features: int,
    ):
        super().__init__()

        layers = []
        layers.append(nn.Conv2d(in_channels=channels, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False))
        layers.append(nn.SiLU(inplace=True))

        for _ in range(num_of_layers - 1):
            layers.append(nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False))
            layers.append(nn.GroupNorm(4, features))
            layers.append(nn.SiLU(inplace=True))

        layers.append(nn.Conv2d(in_channels=features, out_channels=channels, kernel_size=kernel_size, padding=padding, bias=False))

        self.dncnn = nn.Sequential(*layers)

    def forward(
        self,
        x: Tensor,
    ) -> Tensor:
        if len(x.shape) != 4:
            raise ValueError(f"Input tensor must be 4D, but got {len(x.shape)}D tensor.")

        out = x + self.dncnn(x)
        return out


if __name__ == "__main__":
    model = DnCNN(
        channels=3,
        num_of_layers=17,
        kernel_size=3,
        padding=1,
        features=64,
    )
    input_tensor = torch.randn(1, 3, 64, 64)
    print(f"Input shape: {input_tensor.shape}")
    output_tensor = model.forward(input_tensor)
    print(f"Output shape: {output_tensor.shape}")
