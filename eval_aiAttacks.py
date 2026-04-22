from argparse import ArgumentParser
from collections import defaultdict
import logging
import sys

import torch
import torchvision.datasets as dset
import torchvision.transforms as transforms

import configs
import train


logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger(__name__)


AI_ATTACKS = ["JPEGAI", "RemoveAI", "ReplaceAI", "CreateAI"]


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

    print("use_dct:", args.use_dct)
    print("use_dwt:", args.use_dwt)
    print("use_dct_dwt:", args.use_dct_dwt)

    device = torch.device(args.device if args.device else ("cuda:0" if torch.cuda.is_available() else "cpu"))

    cfg = configs.ModelConfig(
        enabled_attacks=AI_ATTACKS,
        image_shape=(args.image_size, args.image_size),
        num_geo_noises_per_step=1,
        num_pert_noises_per_step=1,
        ai_attack_ratio=1.0,
        num_encoded_bits=args.num_bits,
        
        use_dwt=args.use_dwt,
        use_dct=args.use_dct,
        use_dct_dwt=args.use_dct_dwt,
    )

    wm_model = train.Watermark(cfg, device=device, wandb_options={"enabled": False})
    wm_model.load_model(args.ckpt_path)

    print("#"*40)
    print("eval_path used ", args.eval_path)


    eval_loader = load_dataset_vary_size(
        args.eval_path,
        batch_size=args.batch_size,
        image_size=args.image_size,
        # num_workers=args.num_workers,
        num_workers=0,
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
        logger.info("metrics ",metrics)

    if seen == 0:
        raise RuntimeError("No evaluation batches were processed.")

    for k in list(avg_metrics.keys()):
        avg_metrics[k] /= seen

    print("=== AI Attack Evaluation Summary ===")
    print(f"checkpoint: {args.ckpt_path}")
    print(f"batches: {seen}")
    print(f"enabled_attacks: {AI_ATTACKS}")
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
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument(
        "--max_batches",
        type=int,
        default=100,
        help="Maximum number of eval batches to process. Use -1 for full dataset.",
    )
    parser.add_argument("--device", type=str, default="")
    
    parser.add_argument("--use_dwt", action="store_true", default=False)  
    parser.add_argument("--use_dct", action="store_true" , default=False)
    parser.add_argument("--use_dct_dwt", action="store_true", default=False)

    main(parser.parse_args())
