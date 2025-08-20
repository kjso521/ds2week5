import torch


def mean_filter(
    img: torch.Tensor,
    kernel_size=3,
) -> torch.Tensor:

    img_dim = img.dim()
    if img_dim == 2:
        img = img.unsqueeze(0)
    elif img_dim == 4:
        img = img.squeeze(0)

    padding = kernel_size // 2
    padded_img = torch.nn.functional.pad(img, (padding, padding, padding, padding), mode="reflect")

    filtered_img = torch.zeros_like(img)

    for i in range(img.shape[-2]):
        for j in range(img.shape[-1]):
            region = padded_img[
                :,
                i : i + kernel_size,
                j : j + kernel_size,
            ]
            filtered_img[:, i, j] = region.mean(dim=(-2, -1))

    if img_dim == 2:
        filtered_img = filtered_img.squeeze(0)
    elif img_dim == 4:
        filtered_img = filtered_img.unsqueeze(0)
    return filtered_img
