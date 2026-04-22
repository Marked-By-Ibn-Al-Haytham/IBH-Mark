from argparse import ArgumentParser
import logging
import sys
import os
import socket

import train
import configs

import torch
import torchvision.datasets as dset
import torchvision.transforms as transforms
from PIL import Image


logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger(__name__)


def load_dataset(path, batch_size, image_size, num_workers=8):
    dataset = dset.ImageFolder(
        root=path,
        transform=transforms.Compose(
            [
                # transforms.Lambda(lambda img: img.convert("RGB")),
                transforms.Resize(image_size),
                # transforms.CenterCrop((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        ),
    )
    return torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )


def main(args, device):
    torch.manual_seed(42)

    # Parse enabled_attacks from comma-separated string, or None for all attacks
    enabled_attacks = None
    if args.enabled_attacks:
        enabled_attacks = [a.strip() for a in args.enabled_attacks.split(",")]
    cfg_kwargs = {}
    if args.lr is not None:
        cfg_kwargs["lr"] = args.lr
    if args.num_epochs is not None:
        cfg_kwargs["num_epochs"] = args.num_epochs
    if args.ckpt_path is not None:
        cfg_kwargs["ckpt_path"] = args.ckpt_path
    if args.saved_ckpt_path is not None:
        cfg_kwargs["saved_ckpt_path"] = args.saved_ckpt_path
    if args.log_dir is not None:
        cfg_kwargs["log_dir"] = args.log_dir
    if args.num_bits is not None:
        cfg_kwargs["num_encoded_bits"] = args.num_bits
    if args.image_size is not None:
        cfg_kwargs["image_shape"] = (args.image_size, args.image_size)
    if args.batch_size is not None:
        cfg_kwargs["batch_size"] = args.batch_size
    if args.beta_epochs is not None:
        cfg_kwargs["beta_epochs"] = args.beta_epochs
    if args.beta_max is not None:
        cfg_kwargs["beta_max"] = args.beta_max
    if args.num_noises is not None:
        cfg_kwargs["num_noises"] = args.num_noises
    if args.noise_start_epoch is not None:
        cfg_kwargs["noise_start_epoch"] = args.noise_start_epoch
    if enabled_attacks is not None:
        cfg_kwargs["enabled_attacks"] = enabled_attacks
    if args.num_geo_noises_per_step is not None:
        cfg_kwargs["num_geo_noises_per_step"] = args.num_geo_noises_per_step
    if args.num_pert_noises_per_step is not None:
        cfg_kwargs["num_pert_noises_per_step"] = args.num_pert_noises_per_step
    if args.decoder_name is not None:
        cfg_kwargs["decoder_name"] = args.decoder_name
    if args.use_dct_dwt:
        cfg_kwargs["use_dct_dwt"] = True
    elif args.use_dwt:
        cfg_kwargs["use_dwt"] = True
    elif args.use_dct:
        cfg_kwargs["use_dct"] = True

    if args.enc_arch is not None:
        cfg_kwargs["enc_arch"] = args.enc_arch
    
    if args.dwt_encoder_arch is not None:
        cfg_kwargs["dwt_encoder_arch"] = args.dwt_encoder_arch

    cfg = configs.ModelConfig(**cfg_kwargs)
    print(cfg)

    image_size = cfg.image_shape[0]
    train_data = load_dataset(args.train_path, cfg.batch_size, image_size, num_workers=0)
    eval_data = load_dataset(args.eval_path, 1, image_size, num_workers=0)

    run_name = (
        args.name if args.name else os.path.basename(os.path.normpath(cfg.log_dir))
    )
    wandb_options = {
        "enabled": args.wandb,
        "api_key": args.wandb_api_key,
        "project": args.wandb_project,
        "name": run_name,
        "entity": args.wandb_entity,
        "dir": cfg.log_dir,
        "config": {
            "model_config": vars(cfg),
            "train_options": vars(args),
            "hostname": socket.gethostname(),
            "command": " ".join(sys.argv),
        },
    }

    wm_model = train.Watermark(cfg, device=device, wandb_options=wandb_options)
    wm_model.train(train_data, eval_data, args.saved_ckpt_path)


if __name__ == "__main__":
    parser = ArgumentParser()

    # ── Paths ──────────────────────────────────────────────────────────
    parser.add_argument("--train_path", type=str, required=True)
    parser.add_argument("--eval_path", type=str, required=True)
    parser.add_argument("--log_dir", type=str, default=None)
    parser.add_argument("--ckpt_path", type=str, default=None)
    # to continue training from a checkpoint, set this to the checkpoint file path
    parser.add_argument("--saved_ckpt_path", type=str, default=None)

    # ── Training ───────────────────────────────────────────────────────
    parser.add_argument("--num_epochs", type=int, default=None)
    parser.add_argument("--num_bits", type=int, default=None)
    parser.add_argument("--image_size", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--enc_arch", type=str,   default=None)
    parser.add_argument(
        "--dwt_encoder_arch",
        type=str,
        default=None,
        help="DWT encoder architecture: unet, swin, convnext, resnet50, efficientnet",
    )

    parser.add_argument(
        "--decoder_name",
        type=str,
        default=None,
        help="Decoder backbone registry key (e.g. convnext_base, efficientnet_v2_m).",
    )

    # ── Beta schedule ──────────────────────────────────────────────────
    parser.add_argument("--beta_max", type=float, default=None)
    parser.add_argument("--beta_epochs", type=int, default=None)

    # ── Challenge attack curriculum ────────────────────────────────────
    parser.add_argument(
        "--num_noises",
        type=int,
        default=None,
        help="Top-k worst-performing attacks added to the loss function each step.",
    )
    parser.add_argument(
        "--noise_start_epoch",
        type=int,
        default=None,
        help="Epoch at which to start applying attacks in the loss.",
    )
    parser.add_argument(
        "--enabled_attacks",
        type=str,
        default=None,
        help=(
            "Comma-separated list of challenge attack names to use. "
            "Example: 'JPEG50,Crop,GaussNoise,Blur5'. "
            "Leave unset to enable ALL available attacks."
        ),
    )
    parser.add_argument(
        "--num_geo_noises_per_step",
        type=int,
        default=None,
        help="Number of random geometric attacks sampled per training step.",
    )
    parser.add_argument(
        "--num_pert_noises_per_step",
        type=int,
        default=None,
        help="Number of random perturbation attacks sampled per training step.",
    )
    freq_mode_group = parser.add_mutually_exclusive_group()
    freq_mode_group.add_argument(
        "--use_dct",
        action="store_true",
        help="Enable frequency-domain encoding: DCT -> encoder -> IDCT.",
    )
    freq_mode_group.add_argument(
        "--use_dwt",
        action="store_true",
        help="Enable wavelet-domain encoding: DWT -> encoder -> IDWT.",
    )
    freq_mode_group.add_argument(
        "--use_dct_dwt",
        action="store_true",
        help="Enable hybrid DCT+DWT encoding with adaptive residual fusion.",
    )

    # ── Logging ────────────────────────────────────────────────────────
    parser.add_argument("--name", type=str, default="")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="invismark")
    parser.add_argument("--wandb_entity", type=str, default="")
    parser.add_argument("--wandb_api_key", type=str, default=None)

    torch.cuda.empty_cache()
    command_args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    main(command_args, device)
