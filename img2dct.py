import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from dct import DCT2D


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}


def _list_images(root: Path) -> list[Path]:
    return sorted(
        p
        for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def _load_rgb_tensor(path: Path, device: torch.device) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    arr = np.asarray(img, dtype=np.float32) / 255.0
    x = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    return x.to(device)


def _save_dct_as_image(path: Path, coeffs: torch.Tensor) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Log-scale magnitude makes DCT coefficients visible in image form.
    x = coeffs.squeeze(0).detach().cpu()
    x = torch.sign(x) * torch.log1p(torch.abs(x))

    c_min = x.amin(dim=(1, 2), keepdim=True)
    c_max = x.amax(dim=(1, 2), keepdim=True)
    x = (x - c_min) / (c_max - c_min + 1e-8)

    arr = (x.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
    Image.fromarray(arr, mode="RGB").save(path)


def convert_dataset(
    src_dir: Path,
    dst_dir: Path,
    device: torch.device,
    progress_every: int,
    out_ext: str,
) -> None:
    dct = DCT2D().to(device)
    dct.eval()

    images = _list_images(src_dir)
    if not images:
        raise RuntimeError(f"No images found in: {src_dir}")

    print(f"Found {len(images)} images in {src_dir}")
    print(f"Writing DCT coefficients to {dst_dir}")

    with torch.no_grad():
        for idx, in_path in enumerate(images, start=1):
            rel = in_path.relative_to(src_dir)
            out_path = (dst_dir / rel).with_suffix(out_ext)

            x = _load_rgb_tensor(in_path, device)
            coeffs = dct(x)
            _save_dct_as_image(out_path, coeffs)

            if idx % progress_every == 0 or idx == len(images):
                print(f"Processed {idx}/{len(images)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert RGB images to DCT-domain visualizations and save as images."
    )
    parser.add_argument("--src", type=Path, required=True, help="Source image folder")
    parser.add_argument(
        "--dst", type=Path, required=True, help="Destination folder for output images"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run DCT on (cpu or cuda)",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=100,
        help="Print progress every N images",
    )
    parser.add_argument(
        "--out-ext",
        type=str,
        default=".jpg",
        help="Output image extension (e.g., .jpg, .png)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    src_dir = args.src.resolve()
    dst_dir = args.dst.resolve()

    if not src_dir.exists() or not src_dir.is_dir():
        raise FileNotFoundError(f"Source directory not found: {src_dir}")

    dst_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    out_ext = args.out_ext if args.out_ext.startswith(".") else f".{args.out_ext}"
    convert_dataset(src_dir, dst_dir, device, max(1, args.progress_every), out_ext)

    print("Done.")


if __name__ == "__main__":
    main()
