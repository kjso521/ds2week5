import torch

from common.dipole import dipole_kernel


def tkd(
    img: torch.Tensor,
) -> torch.Tensor:

    img_dim = img.dim()
    if img_dim == 3:
        img = img.squeeze(0)
    elif img_dim == 4:
        img = img.squeeze(0).squeeze(0)

    kernel = dipole_kernel(
        matrix_size=img.shape[-2:],
        voxel_size=(1.0, 1.0),
        B0_dir=(0.0, 1.0),
    )
    kernel_inv = 1 / kernel
    kernel_inv = torch.clip(kernel_inv, min=-5, max=5)

    img_k = torch.fft.fftn(img, dim=(-2, -1))
    img_tkd = torch.fft.ifftn(img_k * kernel_inv, dim=(-2, -1)).real

    if img_dim == 3:
        img_tkd = img_tkd.unsqueeze(0)
    elif img_dim == 4:
        img_tkd = img_tkd.unsqueeze(0).unsqueeze(0)

    return img_tkd
