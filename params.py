import os
from dataclasses import dataclass, field, fields, MISSING
from enum import Enum
from argparse import ArgumentParser
import torch
import typing

# ... (Previous dataclasses: GeneralConfig, DnCNNConfig, UnetConfig) ...
# I will rewrite the entire file content as it was corrupted.

class NoisyType(str, Enum):
    GAUSSIAN = "gaussian"
    RICIAN = "rician"
    UNIFORM = "uniform"
    SP = "s&p"

    @classmethod
    def from_string(cls, s: str) -> "NoisyType":
        if s == "gaussian":
            return NoisyType.GAUSSIAN
        if s == "rician":
            return NoisyType.RICIAN
        if s == "uniform":
            return NoisyType.UNIFORM
        if s == "s&p":
            return NoisyType.SP
        raise ValueError(f"Unknown value {s} for NoisyType")

@dataclass
class GeneralConfig:
    # --- Dataset parameters ---
    if 'COLAB_GPU' in os.environ:
        default_root: str = "/content/dataset"
    else:
        default_root: str = "dataset"
    
    DATA_ROOT: str = default_root
    train_dataset: list[str] = field(default_factory=lambda: [os.path.join(GeneralConfig.DATA_ROOT, "train")])
    valid_dataset: list[str] = field(default_factory=lambda: [os.path.join(GeneralConfig.DATA_ROOT, "val")])
    test_dataset: list[str] = field(default_factory=lambda: [os.path.join(GeneralConfig.DATA_ROOT, "val")])
    data_type: str = "*.npy"

    # --- Logging parameters ---
    log_lv: str = "INFO"
    run_dir: str = "logs"
    init_time: float = 0.0

    # --- Training parameters ---
    augmentation_mode: str = "both"
    training_phase: str = "end_to_end"  # New parameter: 'denoising', 'deconvolution', or 'end_to_end'
    noise_levels: list[float] = field(default_factory=lambda: [0.07, 0.132])
    conv_directions: list[tuple[float, float]] = field(default_factory=lambda: [(-0.809, -0.5878), (-0.809, 0.5878), (0.309, -0.9511), (0.309, 0.9511), (1.0, 0.0)])
    model_type: str = "dncnn"
    optimizer: str = "adam"
    loss_model: str = "l2"
    lr: float = 1e-4
    # lr_decay_step: int = 10 # NOTE: Not used in ReduceLROnPlateau scheduler
    lr_decay: float = 0.88
    lr_tol: int = 5
    gpu: int = 0
    train_batch: int = 2
    valid_batch: int = 8
    train_epoch: int = 100
    logging_density: int = 4
    valid_interval: int = 2
    valid_tol: int = 10
    num_workers: int = 4
    save_val: bool = True
    parallel: bool = True
    device: torch.device = torch.device("cpu")
    save_max_idx: int = 500
    noise_type: str = "gaussian"
    tag: str = ""

@dataclass
class DnCNNConfig:
    channels: int = 1
    num_of_layers: int = 17
    kernel_size: int = 3
    padding: int = 1
    features: int = 64

@dataclass
class UnetConfig:
    in_chans: int = 1
    out_chans: int = 1
    chans: int = 32
    num_pool_layers: int = 4

config = GeneralConfig()
dncnnconfig = DnCNNConfig()
unetconfig = UnetConfig()

def update_config_from_args(config_obj, args_obj):
    for f in fields(config_obj):
        if hasattr(args_obj, f.name):
            setattr(config_obj, f.name, getattr(args_obj, f.name))

def parse_args_for_train_script():
    parser = ArgumentParser()
    for cfg_class in [GeneralConfig, DnCNNConfig, UnetConfig]:
        for f in fields(cfg_class):
            default_val = f.default
            if f.default_factory is not MISSING:
                default_val = f.default_factory()

            if typing.get_origin(f.type) is list:
                # This correctly handles types like list[float] and list[tuple]
                parser.add_argument(f"--{f.name}", nargs='+', default=default_val)
            elif f.type is bool:
                # Simplified boolean handling
                parser.add_argument(f"--{f.name}", type=bool, default=default_val)
            elif f.type is not torch.device:
                parser.add_argument(f"--{f.name}", type=f.type, default=default_val)

    parser.add_argument("--augmentation_mode", type=str, default="both", help="Augmentation mode: none, noise_only, conv_only, both")
    parser.add_argument("--training_phase", type=str, default="end_to_end", help="Training phase: denoising, deconvolution, end_to_end")

    args = parser.parse_args()

    update_config_from_args(config, args)
    update_config_from_args(dncnnconfig, args)
    update_config_from_args(unetconfig, args)
    
    config.device = torch.device(f"cuda:{config.gpu}" if torch.cuda.is_available() else "cpu")
    if not torch.cuda.is_available() or torch.cuda.device_count() <= 1:
        config.parallel = False
    
    config.augmentation_mode = args.augmentation_mode
    config.training_phase = args.training_phase

    if config.tag is None:
        config.tag = f"{config.model_type}_{config.training_phase}"

    if config.DATA_ROOT:
        config.train_dataset = [os.path.join(config.DATA_ROOT, "train")]
        config.valid_dataset = [os.path.join(config.DATA_ROOT, "val")]
        config.test_dataset = [os.path.join(config.DATA_ROOT, "val")]

if __name__ == "__main__":
    pass
