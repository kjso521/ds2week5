from enum import Enum

import torch


class NoisyType(str, Enum):
    Gaussian = "gaussian"
    Rician = "rician"
    Uniform = "uniform"
    SaltAndPepper = "salt_and_pepper"

    @classmethod
    def from_string(cls, value: str) -> "NoisyType":
        try:
            return cls(value)
        except ValueError as err:
            raise ValueError(f"Invalid NoisyType value: {value}. Must be one of {list(cls)} : {err}") from err


def gaussian_noise(
    img,
    sigma: float,
) -> torch.Tensor:
    noise = torch.randn_like(img) * sigma
    noisy_img = img + noise
    return noisy_img


def rician_noise(
    img,
    sigma: float,
) -> torch.Tensor:
    noise_real = torch.randn_like(img) * sigma
    noise_imag = torch.randn_like(img) * sigma
    noisy_img = torch.abs(img + noise_real + 1j * noise_imag)
    return noisy_img


def uniform_noise(
    img,
    sigma: float,
) -> torch.Tensor:
    noise = torch.randint(-100, 100, img.shape, dtype=torch.float32) / 50.0 * sigma
    return img + noise


def salt_and_pepper_noise(
    img,
    sigma: float,
) -> torch.Tensor:
    salt_prob = sigma / 2
    pepper_prob = sigma / 2
    noisy_img = img.clone()
    total_pixels = img.numel()

    # Salt noise
    num_salt = int(total_pixels * salt_prob)
    coords = [torch.randint(0, dim, (num_salt,)) for dim in img.shape]
    noisy_img[coords] = img.max()
    # Pepper noise
    num_pepper = int(total_pixels * pepper_prob)
    coords = [torch.randint(0, dim, (num_pepper,)) for dim in img.shape]
    noisy_img[coords] = 0  # Set to black

    return noisy_img


class NoiseSimulator:
    def __init__(
        self,
        noise_type: NoisyType,
        noise_sigma: float,
    ) -> None:
        self.noise_type = noise_type
        self.noise_sigma = noise_sigma

    def __call__(
        self,
        img: torch.Tensor,
    ) -> torch.Tensor:
        if self.noise_type == NoisyType.Gaussian:
            noisy_img = gaussian_noise(img, self.noise_sigma)
        elif self.noise_type == NoisyType.Rician:
            noisy_img = rician_noise(img, self.noise_sigma)
        elif self.noise_type == NoisyType.Uniform:
            noisy_img = uniform_noise(img, self.noise_sigma)
        elif self.noise_type == NoisyType.SaltAndPepper:
            noisy_img = salt_and_pepper_noise(img, self.noise_sigma)
        else:
            raise ValueError(f"Unknown noise type: {self.noise_type}")

        return noisy_img
