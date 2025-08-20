import torch

from common.dipole import dipole_kernel


class ForwardSimulator:
    def __init__(self):
        pass

    def __call__(
        self,
        img: torch.Tensor,
    ) -> torch.Tensor:
        if not isinstance(img, torch.Tensor):
            raise TypeError("Input must be a torch.Tensor")

        img_dim = img.dim()
        if img_dim == 3:
            img = img.squeeze(0)  # img to 2D

        kernel = dipole_kernel(
            matrix_size=img.shape[-2:],
            voxel_size=(1.0, 1.0),
            B0_dir=(0.0, 1.0),
        )

        img_k = torch.fft.fftn(img, dim=(-2, -1))
        img_dipole = torch.fft.ifftn(img_k * kernel, dim=(-2, -1)).real

        if img_dim == 3:
            img_dipole = img_dipole.unsqueeze(0)

        return img_dipole
