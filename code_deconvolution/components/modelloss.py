import torch
from torch import Tensor, nn

from common.dipole import dipole_kernel


def dipole_model_forward(
    pred: Tensor,
) -> Tensor:
    if pred.dim() != 4:
        raise ValueError("Input tensor pred must be 4-dimensional (B, C, H, W)")

    B, C, H, W = pred.shape

    kernel = dipole_kernel(
        matrix_size=(H, W),
        voxel_size=(1.0, 1.0),
        B0_dir=(0.0, 1.0),
    ).to(pred.device)
    kernel = kernel.unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, H, W)
    kernel = kernel.repeat(B, 1, 1, 1)  # Repeat for batch size

    pred_k = torch.fft.fftn(pred, dim=(-2, -1))

    pred_dipole = torch.real(torch.fft.ifftn(pred_k * kernel, dim=(-2, -1)))
    return pred_dipole


class ModelLoss(nn.Module):
    def __init__(
        self,
        model_loss_weight: float = 0.5,
    ) -> None:
        super().__init__()
        self.l1 = nn.L1Loss(reduction="none")
        self.l2 = nn.MSELoss(reduction="none")

        self.model_loss_weight = model_loss_weight

    def forward(
        self,
        output: Tensor,
        target: Tensor,
        measure: Tensor,
    ) -> Tensor:
        if output.shape != target.shape:
            raise ValueError("Input and target tensors must have the same shape")

        forward_output = dipole_model_forward(output)

        md_loss = self.l1(forward_output, measure)
        l2_loss = self.l2(output, target)

        total_loss = (1 - self.model_loss_weight) * l2_loss + self.model_loss_weight * md_loss
        return total_loss
