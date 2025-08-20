import os
import time
from datetime import datetime
from pathlib import Path
import json
import numpy as np
from PIL import Image

from torch import Tensor


def timestamp() -> str:
    return datetime.fromtimestamp(time.time()).strftime("%Y%m%d-%H%M%S")


def separator(cols: int = 100) -> str:
    return "#" * cols


def seconds_to_dhms(seconds: float) -> str:
    s = seconds % 60
    m = (seconds // 60) % 60
    h = seconds // (60 * 60) % 1000
    return f"{int(h):02}h {int(m):02}m {int(s):02}s"


def call_next_id(run_dir: Path) -> int:
    run_ids = []
    os.makedirs(run_dir, exist_ok=True)
    for entry in os.listdir(run_dir):
        if (run_dir / entry).is_dir():
            try:
                run_ids.append(int(entry.split("_")[0]))
            except ValueError:
                continue
    return max(run_ids, default=-1) + 1


def validate_tensors(tensors: list[Tensor]) -> None:
    for i, t in enumerate(tensors):
        if not isinstance(t, Tensor):
            raise TypeError(f"Tensor at index {i} is not a torch.Tensor, got {type(t)} instead.")


def validate_tensor_dimensions(tensors: list[Tensor], expected_dim: int) -> None:
    for i, t in enumerate(tensors):
        if t.dim() != expected_dim:
            raise ValueError(f"Tensor at index {i} has {t.dim()} dimensions, expected {expected_dim} dimensions.")


def validate_tensor_channels(tensor: Tensor, expected_channels: int) -> None:
    if tensor.shape[1] != expected_channels:
        raise ValueError(f"Expected tensor with {expected_channels} channels, but got {tensor.shape[1]} channels.")


def get_config_from_json(json_file: str) -> dict:
    """
    Get the config from a json file
    """
    # parse the configurations from the config json file provided
    with open(json_file, 'r') as config_file:
        config_dict = json.load(config_file)
    return config_dict


def save_numpy_as_image(array: np.ndarray, file_name: str) -> None:
    """
    Saves a numpy array as a grayscale image, handling tensor-specific formats.
    """
    # Squeeze singleton dimensions for batch and channels (e.g., [1, 1, H, W] -> [H, W])
    array_2d = np.squeeze(array)
    
    # Clip, scale to 0-255, and convert to uint8
    # The model output is typically in the range [0, 1]
    image_data = np.clip(array_2d * 255.0, 0, 255).astype(np.uint8)

    img = Image.fromarray(image_data)
    img.save(file_name)
