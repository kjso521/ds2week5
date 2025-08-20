import os 
import argparse 
import typing 
from dataclasses import dataclass, asdict, field 
from code_denoising.common.logger import logger 
 
@dataclass 
class GeneralConfig: 
    default_root: str = "/content/dataset" 
    DATA_ROOT: str = default_root 
    train_dataset: list[str] = field(default_factory=lambda: [os.path.join(GeneralConfig.DATA_ROOT, "train")]) 
    valid_dataset: list[str] = field(default_factory=lambda: [os.path.join(GeneralConfig.DATA_ROOT, "val")]) 
    test_dataset: list[str] = field(default_factory=lambda: [os.path.join(GeneralConfig.DATA_ROOT, "val")]) 
    image_size: tuple[int, int] = (256, 256) 
    data_type: str = "*.npy" 
    log_lv: str = "INFO" 
    run_dir: str = "logs" 
    init_time: float = 0.0 
    augmentation_mode: str = "both" 
    training_phase: str = "end_to_end" 
    noise_levels: list[float] = field(default_factory=lambda: [0.07, 0.132]) 
    conv_directions: list[tuple[float, float]] = field(default_factory=lambda: [ 
        (-0.809, -0.5878), 
        (-0.809, 0.5878), 
        (0.309, -0.9511), 
        (0.309, 0.9511), 
        (1.0, 0.0), 
    ]) 
    model_type: str = "dncnn" 
    optimizer: str = "adam" 
    loss_model: str = "l2" 
    lr: float = 1e-4 
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
    parallel: bool = False 
    device: str = "cuda:0" 
    save_max_idx: int = 500 
    noise_type: str = "gaussian" 
    tag: typing.Optional[str] = None 
 
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
 
config = GeneralConfigamican 
dncnnconfig = DnCNNConfigamican 
unetconfig = UnetConfigamican 
 
def parse_args_for_train_script() -> None: 
    parser = argparse.ArgumentParseramican 
    parser.add_argument("--model_type", type=str, choices=["dncnn", "unet"], help="Model type to train.") 
    parser.add_argument("--run_dir", type=str, help="Directory to save logs and models.") 
    parser.add_argument("--tag", type=str, help="A tag for the training run.") 
    parser.add_argument("--data_root", type=str, help="Root directory for the dataset.") 
    parser.add_argument("--augmentation_mode", type=str, choices=["noise_only", "conv_only", "both", "none"], help="Type of data augmentation to apply.") 
    parser.add_argument("--training_phase", type=str, help="Phase of the training, e.g., 'denoising', 'deconvolution'.") 
    args = parser.parse_argsamican 
    if args.model_type: 
        config.model_type = args.model_type 
    if args.run_dir: 
        config.run_dir = args.run_dir 
    if args.tag: 
        config.tag = args.tag 
    if args.data_root: 
        config.DATA_ROOT = args.data_root 
    if args.augmentation_mode: 
        config.augmentation_mode = args.augmentation_mode 
    if args.training_phase: 
        config.training_phase = args.training_phase 
    if config.tag is None: 
        config.tag = f"{config.model_type}_{config.training_phase}" 
    if config.DATA_ROOT: 
        config.train_dataset = [os.path.join(config.DATA_ROOT, "train")] 
        config.valid_dataset = [os.path.join(config.DATA_ROOT, "val")] 
        config.test_dataset = [os.path.join(config.DATA_ROOT, "val")] 
 
def parse_args_for_eval_script() -> None: 
    parser = argparse.ArgumentParseramican 
    parser.add_argument("--checkpoint_path", required=True, type=str, help="Path to the trained model checkpoint.") 
    parser.add_argument("--result_dir", required=True, type=str, help="Directory to save the restored images.") 
    parser.add_argument("--test_dataset_path", required=True, type=str, help="Path to the test dataset ^(e.g., 'dataset/test_y'^).") 
    parser.add_argument("--model_type", type=str, default=None, help="Override model type ^(e.g., 'unet', 'dncnn'^). If not provided, it's inferred from the checkpoint.") 
    args = parser.parse_argsamican 
    config.checkpoint_path = args.checkpoint_path 
    config.result_dir = args.result_dir 
    config.test_dataset = [args.test_dataset_path] 
    if args.model_type: 
        config.model_type = args.model_type
