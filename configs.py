from dataclasses import dataclass
from typing import Optional, Tuple, List


@dataclass
class ModelConfig:
    log_dir: str = "./logs/"
    ckpt_path: str = "./ckpts/"
    saved_ckpt_path: str = ""
    world_size: int = 1
    lr: float = 0.0002
    num_epochs: int = 500
    log_interval: int = 400
    num_encoded_bits: int = 100
    image_shape: Tuple[int, int] = (256, 256)
    num_down_levels: int = 4
    num_initial_channels: int = 32
    batch_size: int = 32
    beta_min: float = 0.0001
    beta_max: float = 40.0
    beta_start_epoch: int = 3
    beta_epochs: int = 20
    warmup_epochs: int = 3
    discriminator_feature_dim: int = 16
    num_discriminator_layers: int = 4
    watermark_hidden_dim: int = 16
    psnr_threshold: float = 80.0
    enc_mode: str = "uuid"  # "ecc"
    ecc_t: int = 16
    ecc_m: int = 8
    num_classes: int = 2
    beta_transform: float = 0.5
    num_noises: int = 2
    noise_start_epoch: int = 20
    num_repeats: int = 1
    enc_arch: str = "swin" # "resnet"
    
    # DWT model architecture options: "unet", "swin", "convnext", "resnet50", "efficientnet"
    dwt_encoder_arch: str = "swin"
    dwt_decoder_arch: str = "convnext_base"

    enabled_attacks: Optional[List[str]] = None
    num_geo_noises_per_step: int = 1
    num_pert_noises_per_step: int = 0
    ai_attack_ratio: float = 0.0
    encoder_freeze_epoch: Optional[int] = None

    use_dct: bool = False
    use_dwt: bool = False
    use_dct_dwt: bool = False
    
    decoder_name: str = "convnext_base"

    # Swin-UNet encoder config
    use_swin_encoder: bool = False
    swin_config_path: str = "Swin_Unet/swin_tiny_patch4_window7_224_lite.yaml"
    swin_pretrained_path: str = "Swin_Unet/model.pth"
    swin_img_size: int = 224
    swin_freeze_encoder: bool = False