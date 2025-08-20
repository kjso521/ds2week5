from functools import lru_cache

import numpy as np
import torch


@lru_cache(maxsize=16)
def dipole_kernel(
    matrix_size: tuple[int, int],
    voxel_size: tuple[float, float] = (1.0, 1.0),
    B0_dir: tuple[float, float] = (0.0, 1.0),
) -> torch.Tensor:
    y = np.arange(-matrix_size[1] / 2, matrix_size[1] / 2, 1)
    x = np.arange(-matrix_size[0] / 2, matrix_size[0] / 2, 1)
    Y, X = np.meshgrid(y, x)

    X = X / (matrix_size[0] * voxel_size[0])
    Y = Y / (matrix_size[1] * voxel_size[1])

    D = 1 / 3 - (X * B0_dir[0] + Y * B0_dir[1]) ** 2 / (X**2 + Y**2 + 1e-8)
    D = np.fft.fftshift(D)
    D = torch.tensor(D, dtype=torch.float32)
    return D


class ForwardSimulator:
    def __init__(self):
        pass

    def __call__(
        self,
        img: torch.Tensor,
        B0_dir: tuple[float, float] = (0.0, 1.0),
    ) -> torch.Tensor:
        if not isinstance(img, torch.Tensor):
            raise TypeError("Input must be a torch.Tensor")

        img_dim = img.dim()
        if img_dim == 3:
            img = img.squeeze(0)  # img to 2D

        kernel = dipole_kernel(
            matrix_size=img.shape[-2:],
            voxel_size=(1.0, 1.0),
            B0_dir=B0_dir,
        )

        img_k = torch.fft.fftn(img, dim=(-2, -1))
        img_dipole_complex = torch.fft.ifftn(img_k * kernel, dim=(-2, -1))
        
        # .real 대신 실수부와 허수부를 채널로 쌓아 반환
        img_dipole = torch.stack([img_dipole_complex.real, img_dipole_complex.imag], dim=-3)

        # 입력 차원에 관계없이 항상 [C, H, W] 형태의 3D 텐서를 반환하도록 보장
        if img_dipole.dim() > 3:
            img_dipole = img_dipole.squeeze(0)

        return img_dipole
