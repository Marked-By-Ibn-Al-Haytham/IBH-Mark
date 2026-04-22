from argparse import ArgumentParser
from collections import defaultdict
import logging
import os
import sys

import torch
import torchvision.datasets as dset
import torchvision.transforms as transforms

import configs
import train


logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger(__name__)




def _validate_eval_args(args):
    ckpt_path = args.ckpt_path.strip()
    if not ckpt_path:
        raise ValueError("--ckpt_path is required.")

    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint file not found: {ckpt_path}")

    decoder_name = args.decoder_name.strip()
    if not decoder_name:
        raise ValueError("--decoder_name is required.")

    encoder_name = args.encoder_name.strip()
    if not encoder_name:
        raise ValueError("--encoder_name is required.")

    return ckpt_path, encoder_name, decoder_name


def load_dataset_vary_size(path, batch_size, image_size, num_workers=4):
    dataset = dset.ImageFolder(
        root=path,
        transform=transforms.Compose(
            [
                transforms.Lambda(lambda img: img.convert("RGB")),
                # transforms.Resize(image_size),
                # transforms.CenterCrop((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        ),
    )
    return torch.utils.data.DataLoader(
        dataset,
        # batch_size=batch_size, 
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
    )

def load_dataset(path, batch_size, image_size, num_workers=4):
    dataset = dset.ImageFolder(
        root=path,
        transform=transforms.Compose(
            [
                transforms.Lambda(lambda img: img.convert("RGB")),
                transforms.Resize(image_size),
                transforms.CenterCrop((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        ),
    )
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )


def main(args):
    device = torch.device(args.device if args.device else ("cuda:0" if torch.cuda.is_available() else "cpu"))
    ckpt_path, encoder_name, decoder_name = _validate_eval_args(args)

    print("use_dct:", args.use_dct)
    print("use_dwt:", args.use_dwt)
    print("use_dct_dwt:", args.use_dct_dwt)
    print("encoder_name:", encoder_name)
    print("decoder_name:", decoder_name)
    cfg = configs.ModelConfig(
        # image_shape=(args.image_size, args.image_size),
        num_geo_noises_per_step=1,
        num_pert_noises_per_step=1,
        num_encoded_bits=args.num_bits,
        enc_arch=encoder_name,
        dwt_encoder_arch=encoder_name,
        decoder_name=decoder_name,
        use_dwt=args.use_dwt,
        use_dct=args.use_dct,
        use_dct_dwt=args.use_dct_dwt,
        # ai_attack_ratio=1.0,

    )

    wm_model = train.Watermark(cfg, device=device, wandb_options={"enabled": False})
    wm_model.load_model(ckpt_path)
    wm_model.alpha = 0.0
    print("#"*40)
    print("eval_path used ", args.eval_path)

    eval_loader = load_dataset_vary_size(
    # eval_loader = load_dataset(
        args.eval_path,
        batch_size=args.batch_size,
        image_size=args.image_size,
        num_workers=args.num_workers,
    )

    avg_metrics = defaultdict(float)
    seen = 0
    for idx, batch in enumerate(eval_loader):
        if args.max_batches > 0 and idx >= args.max_batches:
            break

        images = batch[0]
        secrets = wm_model._generate_secret(images.shape[0], wm_model.device)
        metrics = wm_model._calculate_metric(images, secrets, prefix="Eval")

        for k, v in metrics.items():
            avg_metrics[k] += float(v)
        seen += 1

        logger.info(f"Processed eval batch {seen}")

    if seen == 0:
        raise RuntimeError("No evaluation batches were processed.")

    for k in list(avg_metrics.keys()):
        avg_metrics[k] /= seen

    print("=== AI Attack Evaluation Summary ===")
    print(f"checkpoint: {ckpt_path}")
    print(f"batches: {seen}")
    for key in sorted(avg_metrics.keys()):
        print(f"{key}: {avg_metrics[key]:.6f}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--encoder_name", type=str, required=True)
    parser.add_argument("--decoder_name", type=str, required=True)
    parser.add_argument("--eval_path", type=str, required=True)

    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_bits", type=int, default=100)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument(
        "--max_batches",
        type=int,
        default=100,
        help="Maximum number of eval batches to process. Use -1 for full dataset.",
    )
    parser.add_argument("--device", type=str, default="")
    freq_mode_group = parser.add_mutually_exclusive_group()
    freq_mode_group.add_argument("--use_dwt", action="store_true", default=False)
    freq_mode_group.add_argument("--use_dct", action="store_true", default=False)
    freq_mode_group.add_argument("--use_dct_dwt", action="store_true", default=False)
    main(parser.parse_args())
