from __future__ import annotations

import logging
import io
import math
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import subprocess
import tempfile
import shutil
from pathlib import Path
from ._weights import ensure_sam_checkpoint
from torchvision.transforms import InterpolationMode
from PIL import Image, ImageFilter
import torchvision.io as tvio

logger = logging.getLogger(__name__)

try:
    import pillow_jxl
except ImportError:
    pass

from typing import Optional, Tuple, Any

try:
    import kornia
    import kornia.filters as Kf
    import kornia.enhance as Ke

    _HAS_KORNIA = True
except Exception:
    _HAS_KORNIA = False

try:
    from compressai.zoo import cheng2020_anchor

    _HAS_COMPRESSAI = True
except Exception:
    _HAS_COMPRESSAI = False


# ============================================================
# 1) Shape + Range Adapters
# ============================================================
def _detect_layout(x: torch.Tensor) -> str:
    # Infer tensor layout string (HW, CHW, HWC, BCHW, BHWC) from its shape
    if x.dim() == 2:
        return "HW"
    if x.dim() == 3:
        return "CHW" if x.shape[0] <= 4 else "HWC"
    if x.dim() == 4:
        return "BCHW" if x.shape[1] <= 4 else "BHWC"
    raise ValueError(f"Unsupported tensor dim={x.dim()} with shape={tuple(x.shape)}")


def _to_bchw(x: torch.Tensor):
    # Convert an image tensor of various layouts into BCHW and return metadata to restore later
    layout = _detect_layout(x)
    meta = {
        "layout": layout,
        "dtype": x.dtype,
        "device": x.device,
    }

    if layout == "HW":
        x_bchw = x.unsqueeze(0).unsqueeze(0)
    elif layout == "CHW":
        x_bchw = x.unsqueeze(0)
    elif layout == "HWC":
        x_bchw = x.permute(2, 0, 1).unsqueeze(0)
    elif layout == "BCHW":
        x_bchw = x
    elif layout == "BHWC":
        x_bchw = x.permute(0, 3, 1, 2)
    else:
        raise ValueError(f"Unsupported layout={layout}")

    return x_bchw, meta


def _from_bchw(x_bchw: torch.Tensor, meta: dict) -> torch.Tensor:
    # Restore a BCHW tensor back to the original layout recorded in meta
    layout = meta["layout"]

    if layout == "HW":
        return x_bchw[0, 0]
    if layout == "CHW":
        return x_bchw[0]
    if layout == "HWC":
        return x_bchw[0].permute(1, 2, 0)
    if layout == "BCHW":
        return x_bchw
    if layout == "BHWC":
        return x_bchw.permute(0, 2, 3, 1)

    raise ValueError(f"Unsupported layout={layout}")


def _detect_range_mode(x: torch.Tensor) -> str:
    # Detect value/range convention (uint8_255, float_0_1, float_0_255, float_-1_1)
    if not torch.is_floating_point(x):
        return "uint8_255"

    safe = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    x_min = float(safe.min().item())
    x_max = float(safe.max().item())

    if x_min >= -1.0001 and x_max <= 1.0001:
        return "float_0_1" if x_min >= -0.0001 else "float_-1_1"

    if x_min >= -0.0001 and x_max <= 255.0001:
        return "float_0_1" if x_max <= 1.0001 else "float_0_255"

    return "float_-1_1"


def _to_minus1_1(x: torch.Tensor, mode: str) -> torch.Tensor:
    # Convert input tensor from detected mode into float range [-1, 1]
    if mode == "uint8_255":
        x01 = (x.to(torch.float32) / 255.0).clamp(0.0, 1.0)
        return (x01 * 2.0 - 1.0).clamp(-1.0, 1.0)

    if mode == "float_0_1":
        x01 = x.to(torch.float32).clamp(0.0, 1.0)
        return (x01 * 2.0 - 1.0).clamp(-1.0, 1.0)

    if mode == "float_0_255":
        x01 = (x.to(torch.float32) / 255.0).clamp(0.0, 1.0)
        return (x01 * 2.0 - 1.0).clamp(-1.0, 1.0)

    return x.to(torch.float32).clamp(-1.0, 1.0)


def _from_minus1_1(
    x_m11: torch.Tensor, mode: str, orig_dtype: torch.dtype
) -> torch.Tensor:
    # Convert a float [-1, 1] tensor back into the original numeric mode/dtype
    x_m11 = x_m11.clamp(-1.0, 1.0)

    if mode == "uint8_255":
        x01 = (x_m11 + 1.0) / 2.0
        x255 = (x01 * 255.0).round().clamp(0.0, 255.0)
        return x255.to(torch.uint8)

    if mode == "float_0_1":
        x01 = (x_m11 + 1.0) / 2.0
        return x01.clamp(0.0, 1.0).to(orig_dtype)

    if mode == "float_0_255":
        x01 = (x_m11 + 1.0) / 2.0
        x255 = (x01 * 255.0).clamp(0.0, 255.0)
        return x255.to(orig_dtype)

    return x_m11.to(orig_dtype)


def _apply_attack_preserve(x: torch.Tensor, attack_fn, *args, **kwargs) -> torch.Tensor:
    # Run an attack in a normalized BCHW, [-1,1] space and then restore original layout/range
    x_bchw, meta = _to_bchw(x)
    range_mode = _detect_range_mode(x_bchw)

    x_m11 = _to_minus1_1(x_bchw, range_mode).to(meta["device"])
    y_m11 = attack_fn(x_m11, *args, **kwargs).clamp(-1.0, 1.0)

    y_bchw = _from_minus1_1(y_m11, range_mode, meta["dtype"]).to(meta["device"])
    y = _from_bchw(y_bchw, meta)
    return y


# ============================================================
# 2) Geometric Attacks
# ============================================================
def rotate_tensor(x: torch.Tensor, angle: Optional[float] = None) -> torch.Tensor:
    # Rotate an image tensor by angle degrees (sample angle when None)
    if angle is None:
        angle = float(torch.empty(1).uniform_(-25.0, 25.0).item())

    def _core(z: torch.Tensor):
        return TF.rotate(
            z,
            angle=float(angle),
            interpolation=InterpolationMode.BILINEAR,
            expand=False,
            fill=-1.0,
        )

    return _apply_attack_preserve(x, _core)

def rotate_tensor_inverse(x: torch.Tensor, angle: Optional[float] = None) -> torch.Tensor:
    # Rotate an image tensor by angle degrees (sample angle when None)
    if angle is None:
        angle = float(torch.empty(1).uniform_(-25.0, 25.0).item())

    def _core(z: torch.Tensor):
        z = TF.rotate(
            z,
            angle=float(angle),
            interpolation=InterpolationMode.BILINEAR,
            expand=False,
            fill=-1.0,
        )
        return TF.rotate(
            z,
            angle=float(-angle),
            interpolation=InterpolationMode.BILINEAR,
            expand=False,
            fill=-1.0,
        )

    return _apply_attack_preserve(x, _core)

def rotate_tensor_keep_all(x, angle, fill=-1.0):
    # Step 1: rotate with expansion (no cropping)
    def _core(z: torch.Tensor):
        return TF.rotate(
            z,
            angle=float(angle),
            interpolation=InterpolationMode.BILINEAR,
            expand=True,
            fill=-1.0,
        )

    return _apply_attack_preserve(x, _core)

    rotated = TF.rotate(
        z,
        angle=float(angle),
        interpolation=InterpolationMode.BILINEAR,
        expand=True,
        fill=fill,
    )

    # Original size
    orig_h, orig_w = z.shape[-2:]

    # New size after rotation
    new_h, new_w = rotated.shape[-2:]

    # Step 2: compute scale to fit back into original size
    scale = min(orig_h / new_h, orig_w / new_w)

    resized = TF.resize(
        rotated,
        [int(new_h * scale), int(new_w * scale)],
        interpolation=InterpolationMode.BILINEAR,
    )

    # Step 3: pad to original size (centered)
    pad_h = orig_h - resized.shape[-2]
    pad_w = orig_w - resized.shape[-1]

    padding = [
        pad_w // 2,                 # left
        pad_h // 2,                 # top
        pad_w - pad_w // 2,         # right
        pad_h - pad_h // 2          # bottom
    ]

    output = TF.pad(resized, padding, fill=fill)

    return output

def crop(x: torch.Tensor, pct: Optional[float] = None) -> torch.Tensor:
    # Center-crop by a percentage of the image size, then resize back to original resolution
    if pct is None:
        pct = float(torch.empty(1).uniform_(5.0, 25.0).item())

    def _core(z: torch.Tensor):
        _, _, H, W = z.shape

        dy = int(round(H * (float(pct) / 100.0)))
        dx = int(round(W * (float(pct) / 100.0)))

        dy = max(0, min(dy, (H - 2) // 2))
        dx = max(0, min(dx, (W - 2) // 2))

        if dy == 0 and dx == 0:
            return z

        new_h = H - 2 * dy
        new_w = W - 2 * dx

        cropped = TF.crop(z, top=dy, left=dx, height=new_h, width=new_w)
        resized = TF.resize(
            cropped,
            size=[H, W],
            interpolation=InterpolationMode.BILINEAR,
            antialias=False,
        )
        return resized

    return _apply_attack_preserve(x, _core)


def scaled(x: torch.Tensor, scale: Optional[float] = None) -> torch.Tensor:
    # Resize image by a scale factor (or percentage if >3), then resize back to original size
    if scale is None:
        scale = float(torch.empty(1).uniform_(0.75, 1.5).item())

    def _core(z: torch.Tensor):
        _, _, H, W = z.shape
        s = float(scale)

        if s > 3.0:
            s = 1.0 + (s / 100.0)

        if s <= 0:
            return z

        new_h = max(1, int(round(H * s)))
        new_w = max(1, int(round(W * s)))

        bigger = TF.resize(
            z, [new_h, new_w], interpolation=InterpolationMode.BILINEAR, antialias=False
        )
        back = TF.resize(
            bigger, [H, W], interpolation=InterpolationMode.BILINEAR, antialias=False
        )
        return back

    return _apply_attack_preserve(x, _core)


def flipping(x: torch.Tensor, mode: str) -> torch.Tensor:
    # Flip the image horizontally (H), vertically (V), both (B)
    def _core(z: torch.Tensor):
        m = str(mode).upper()
        if m == "H":
            return torch.flip(z, dims=[3])
        if m == "V":
            return torch.flip(z, dims=[2])
        if m == "B":
            return torch.flip(z, dims=[2, 3])
        return z

    return _apply_attack_preserve(x, _core)


def resized(x: torch.Tensor, pct: Optional[int] = None) -> torch.Tensor:
    # Downsample by a percentage amount, then upsample back to original size
    if pct is None:
        pct = int(torch.randint(low=10, high=41, size=(1,)).item())
    def _core(z: torch.Tensor):
        _, _, H, W = z.shape
        level_ratio = int(pct) / 100.0
        down = max(0.2, 1.0 - level_ratio)

        new_h = max(1, int(round(H * down)))
        new_w = max(1, int(round(W * down)))

        small = TF.resize(
            z,
            size=[new_h, new_w],
            interpolation=InterpolationMode.BILINEAR,
            antialias=False,
        )
        back = TF.resize(
            small,
            size=[H, W],
            interpolation=InterpolationMode.BILINEAR,
            antialias=False,
        )
        return back

    return _apply_attack_preserve(x, _core)


# ============================================================
# 3) Signal Processing Attacks
# ============================================================
def jpeg_compression(x: torch.Tensor, quality: int) -> torch.Tensor:
    # Apply JPEG encode/decode at a given quality using torchvision's JPEG routines.
    def _core(z: torch.Tensor):
        device = z.device
        z_cpu = z.detach().cpu().clamp(-1.0, 1.0)

        B, C, H, W = z_cpu.shape
        outs = []
        q = int(quality)

        for i in range(B):
            x01 = (z_cpu[i] + 1.0) / 2.0
            img_u8 = (x01 * 255.0).round().clamp(0, 255).to(torch.uint8)

            jpeg_bytes = tvio.encode_jpeg(img_u8, quality=q)
            dec_u8 = tvio.decode_jpeg(jpeg_bytes)

            dec_f = dec_u8.to(torch.float32) / 255.0
            dec_m11 = (dec_f * 2.0 - 1.0).clamp(-1.0, 1.0)
            outs.append(dec_m11)

        return torch.stack(outs, dim=0).to(device)

    return _apply_attack_preserve(x, _core)


def _jpeg_like_train_fast(
    x: torch.Tensor,
    *,
    quality: int = 50,
    block_size: int = 8,
    block_strength: float = 0.30,
    chroma_subsampling: bool = True,
    add_jitter: bool = False,
) -> torch.Tensor:
    # Fast, GPU-friendly JPEG-style artifact simulation for training only.
    def _core(z: torch.Tensor):
        z01 = ((z.clamp(-1.0, 1.0) + 1.0) / 2.0).clamp(0.0, 1.0)

        q = max(1, min(100, int(quality)))
        strength = (100.0 - float(q)) / 100.0

        B, C, H, W = z01.shape

        # Approximate 4:2:0 chroma subsampling in YCbCr for RGB images.
        if chroma_subsampling and C == 3:
            r = z01[:, 0:1]
            g = z01[:, 1:2]
            b = z01[:, 2:3]

            y = 0.299 * r + 0.587 * g + 0.114 * b
            cb = -0.168736 * r - 0.331264 * g + 0.5 * b + 0.5
            cr = 0.5 * r - 0.418688 * g - 0.081312 * b + 0.5

            if H > 1 and W > 1:
                cb = F.interpolate(
                    cb,
                    size=(max(1, H // 2), max(1, W // 2)),
                    mode="bilinear",
                    align_corners=False,
                )
                cr = F.interpolate(
                    cr,
                    size=(max(1, H // 2), max(1, W // 2)),
                    mode="bilinear",
                    align_corners=False,
                )
                cb = F.interpolate(
                    cb, size=(H, W), mode="bilinear", align_corners=False
                )
                cr = F.interpolate(
                    cr, size=(H, W), mode="bilinear", align_corners=False
                )

            r = y + 1.402 * (cr - 0.5)
            g = y - 0.344136 * (cb - 0.5) - 0.714136 * (cr - 0.5)
            b = y + 1.772 * (cb - 0.5)
            z01 = torch.cat([r, g, b], dim=1).clamp(0.0, 1.0)

        # Introduce blockiness by mixing with block averages.
        bs = max(2, int(block_size))
        pooled = F.avg_pool2d(z01, kernel_size=bs, stride=bs, ceil_mode=True)
        pooled_up = F.interpolate(pooled, size=(H, W), mode="nearest")
        mix = max(0.0, float(block_strength)) * (0.15 + 0.85 * strength)
        z01 = (z01 * (1.0 - mix) + pooled_up * mix).clamp(0.0, 1.0)

        # Quantization strength follows quality: lower quality -> coarser bins.
        step = (1.0 / 255.0) * (1.0 + 24.0 * strength)
        if add_jitter and strength > 0.0:
            noise = torch.empty_like(z01).uniform_(-0.5 * step, 0.5 * step)
            z01 = (z01 + noise).clamp(0.0, 1.0)
        z01 = torch.round(z01 / step) * step
        z01 = z01.clamp(0.0, 1.0)

        return (z01 * 2.0 - 1.0).clamp(-1.0, 1.0)

    return _apply_attack_preserve(x, _core)


def jpeg_compression_train_fast(x: torch.Tensor, quality: int = 50) -> torch.Tensor:
    # Training-only fast approximation of JPEG compression artifacts.
    return _jpeg_like_train_fast(
        x,
        quality=quality,
        block_size=8,
        block_strength=0.35,
        chroma_subsampling=True,
        add_jitter=True,
    )


def jpeg2000_compression_train_fast(x: torch.Tensor, quality: int = 40) -> torch.Tensor:
    # Training-only fast approximation of JPEG2000-like compression artifacts.
    return _jpeg_like_train_fast(
        x,
        quality=quality,
        block_size=16,
        block_strength=0.12,
        chroma_subsampling=False,
        add_jitter=False,
    )


def jpegxl_compression_train_fast(x: torch.Tensor, quality: int = 50) -> torch.Tensor:
    # Training-only fast approximation of JPEG-XL-like compression artifacts.
    return _jpeg_like_train_fast(
        x,
        quality=quality,
        block_size=8,
        block_strength=0.10,
        chroma_subsampling=True,
        add_jitter=False,
    )


def jpegxs_compression_train_fast(x: torch.Tensor, quality: int = 55) -> torch.Tensor:
    # Training-only fast approximation of JPEG-XS-like compression artifacts.
    return _jpeg_like_train_fast(
        x,
        quality=quality,
        block_size=8,
        block_strength=0.18,
        chroma_subsampling=False,
        add_jitter=False,
    )


def jpeg2000_compression(
    x: torch.Tensor,
    quality_layers=(20,),
    quality_mode="rates",
    irreversible=True,
    ext="jp2",
) -> torch.Tensor:
    # Apply JPEG2000 encode/decode using Pillow (quality controlled via quality_layers)
    def _core(z: torch.Tensor):
        device = z.device
        z_cpu = z.detach().cpu().clamp(-1.0, 1.0)
        B, C, H, W = z_cpu.shape
        outs = []

        ql = quality_layers
        if isinstance(ql, (int, float, np.integer, np.floating)):
            ql = (int(ql),)

        for i in range(B):
            x01 = (z_cpu[i] + 1.0) / 2.0
            img_u8 = (x01 * 255.0).round().clamp(0, 255).to(torch.uint8)

            if img_u8.shape[0] == 1:
                img_np = img_u8[0].numpy()
                img_pil = Image.fromarray(img_np, mode="L")
            else:
                img_np = img_u8.permute(1, 2, 0).numpy()
                img_pil = Image.fromarray(img_np, mode="RGB")

            buf = io.BytesIO()
            img_pil.save(
                buf,
                format="JPEG2000",
                quality_mode=quality_mode,
                quality_layers=list(ql),
                irreversible=bool(irreversible),
            )
            buf.seek(0)

            dec_pil = Image.open(buf)
            if img_u8.shape[0] == 1:
                dec_pil = dec_pil.convert("L")
                dec_np = np.array(dec_pil, dtype=np.uint8)
                dec_u8 = torch.from_numpy(dec_np).unsqueeze(0)
            else:
                dec_pil = dec_pil.convert("RGB")
                dec_np = np.array(dec_pil, dtype=np.uint8)
                dec_u8 = torch.from_numpy(dec_np).permute(2, 0, 1).contiguous()

            dec_f = dec_u8.to(torch.float32) / 255.0
            dec_m11 = (dec_f * 2.0 - 1.0).clamp(-1.0, 1.0)
            outs.append(dec_m11)

        return torch.stack(outs, dim=0).to(device)

    return _apply_attack_preserve(x, _core)


# ------------------------------------------------------------
# Helpers for JPEG-AI (CompressAI)
# ------------------------------------------------------------
_COMPRESSAI_MODEL_CACHE = {}


def _get_cheng2020_anchor(device: torch.device, quality: int):
    if not _HAS_COMPRESSAI:
        raise ImportError(
            "compressai is not installed. Install it to use jpegai_compression."
        )

    q = int(quality)
    if q < 1 or q > 8:
        raise ValueError("jpegai quality must be in [1..8] for cheng2020_anchor")

    key = (str(device), q)
    if key in _COMPRESSAI_MODEL_CACHE:
        return _COMPRESSAI_MODEL_CACHE[key]

    model = cheng2020_anchor(quality=q, pretrained=True).eval().to(device)
    _COMPRESSAI_MODEL_CACHE[key] = model
    return model


def _pad_to_multiple_of_64(x01_bchw: torch.Tensor) -> tuple[torch.Tensor, int, int]:
    _, _, h, w = x01_bchw.shape
    pad_h = (64 - (h % 64)) % 64
    pad_w = (64 - (w % 64)) % 64
    x_pad = F.pad(x01_bchw, (0, pad_w, 0, pad_h), mode="constant", value=0.0)
    return x_pad, h, w


def jpegai_compression(x: torch.Tensor, quality: int = 4) -> torch.Tensor:
    # Apply compression via CompressAI (cheng2020_anchor) and reconstruct the image.
    def _core(z_m11: torch.Tensor):
        device = z_m11.device

        z01 = ((z_m11 + 1.0) / 2.0).clamp(0.0, 1.0)

        B, C, H, W = z01.shape

        if C == 1:
            z01_in = z01.repeat(1, 3, 1, 1)
            single_channel = True
        elif C == 3:
            z01_in = z01
            single_channel = False
        else:
            raise ValueError(f"jpegai_compression supports only C=1 or C=3, got C={C}")

        model = _get_cheng2020_anchor(device, int(quality))

        x_pad, orig_h, orig_w = _pad_to_multiple_of_64(z01_in)

        with torch.no_grad():
            out = model(x_pad)
            x_hat = out["x_hat"].clamp(0.0, 1.0)

        x_hat = x_hat[:, :, :orig_h, :orig_w]

        if single_channel:
            x_hat = x_hat[:, :1, :, :]

        y_m11 = (x_hat * 2.0 - 1.0).clamp(-1.0, 1.0)
        return y_m11

    return _apply_attack_preserve(x, _core)


def jpegxl_compression(x: torch.Tensor, quality: int = 50) -> torch.Tensor:
    # Apply JPEG-XL encode/decode via Pillow (requires Pillow built with JXL support)
    def _core(z: torch.Tensor):
        device = z.device
        z_cpu = z.detach().cpu().clamp(-1.0, 1.0)

        B, C, H, W = z_cpu.shape
        q = int(quality)
        outs = []

        for i in range(B):
            x01 = (z_cpu[i] + 1.0) / 2.0
            img_u8 = (x01 * 255.0).round().clamp(0, 255).to(torch.uint8)

            if img_u8.shape[0] == 1:
                img_np = img_u8[0].numpy()
                img_pil = Image.fromarray(img_np, mode="L")
            else:
                img_np = img_u8.permute(1, 2, 0).numpy()
                img_pil = Image.fromarray(img_np, mode="RGB")

            buf = io.BytesIO()

            try:
                img_pil.save(buf, format="JXL", quality=q)
            except Exception as e:
                raise RuntimeError(
                    "JPEG-XL save failed. Your Pillow likely lacks JXL support. "
                    "Install/build Pillow with JXL support (or install pillow-jxl), "
                    f"then retry. Original error: {e}"
                )

            buf.seek(0)

            dec_pil = Image.open(buf)
            if img_u8.shape[0] == 1:
                dec_pil = dec_pil.convert("L")
                dec_np = np.array(dec_pil, dtype=np.uint8)
                dec_u8 = torch.from_numpy(dec_np).unsqueeze(0)
            else:
                dec_pil = dec_pil.convert("RGB")
                dec_np = np.array(dec_pil, dtype=np.uint8)
                dec_u8 = torch.from_numpy(dec_np).permute(2, 0, 1).contiguous()

            dec_f = dec_u8.to(torch.float32) / 255.0
            dec_m11 = (dec_f * 2.0 - 1.0).clamp(-1.0, 1.0)
            outs.append(dec_m11)

        return torch.stack(outs, dim=0).to(device)

    return _apply_attack_preserve(x, _core)


# def jpegxs_compression(
#     x: torch.Tensor,
#     bitrate: str = "40M",
#     pix_fmt: str = "yuv444p10le",
# ) -> torch.Tensor:
#     # Applies JPEG-XS encode/decode via ffmpeg and returns the reconstructed tensor
#     def _run_cmd(cmd, timeout: int, label: str, retry_timeout: Optional[int] = None):
#         # Retry once with a larger timeout for transient ffmpeg stalls under heavy load.
#         try:
#             proc = subprocess.run(
#                 cmd,
#                 capture_output=True,
#                 text=True,
#                 timeout=timeout,
#             )
#         except subprocess.TimeoutExpired as e:
#             if retry_timeout is None:
#                 raise RuntimeError(
#                     f"{label} timed out after {timeout}s.\nCommand: {' '.join(cmd)}"
#                 ) from e
#             logger.warning(
#                 "%s timed out after %ss; retrying once with %ss",
#                 label,
#                 timeout,
#                 retry_timeout,
#             )
#             try:
#                 proc = subprocess.run(
#                     cmd,
#                     capture_output=True,
#                     text=True,
#                     timeout=retry_timeout,
#                 )
#             except subprocess.TimeoutExpired as e2:
#                 raise RuntimeError(
#                     f"{label} timed out after retry ({retry_timeout}s).\n"
#                     f"Command: {' '.join(cmd)}"
#                 ) from e2

#         if proc.returncode != 0:
#             raise RuntimeError(
#                 f"{label} failed (code={proc.returncode}).\n"
#                 f"Command: {' '.join(cmd)}\n"
#                 f"stderr:\n{proc.stderr}"
#             )
#         return proc

#     def _core(z: torch.Tensor):
#         enc_app = shutil.which("SvtJpegxsEncApp")
#         dec_app = shutil.which("SvtJpegxsDecApp")
#         if not enc_app or not dec_app:
#             raise RuntimeError(
#                 "SvtJpegxsEncApp / SvtJpegxsDecApp not found on PATH."
#             )

#         device = z.device
#         z_cpu = z.detach().cpu().clamp(-1.0, 1.0)
#         B, C, H, W = z_cpu.shape
#         outs = []

#         # SVT-JPEG-XS expects --bpp. Keep "bitrate" API for compatibility and map it.
#         # If value looks like bits/sec (e.g., 40M), convert to bpp with a 30 FPS assumption.
#         br_str = str(bitrate).strip()
#         fps_assumed = 30.0
#         if br_str.upper().endswith("M"):
#             br_bps = float(br_str[:-1]) * 1_000_000.0
#             bpp = br_bps / (float(H) * float(W) * fps_assumed)
#         elif br_str.upper().endswith("K"):
#             br_bps = float(br_str[:-1]) * 1_000.0
#             bpp = br_bps / (float(H) * float(W) * fps_assumed)
#         else:
#             v = float(br_str)
#             # Small numeric values are treated as direct bpp; larger ones as bits/sec.
#             bpp = v if v <= 64.0 else (v / (float(H) * float(W) * fps_assumed))
#         bpp = max(0.01, float(bpp))

#         # Map pix_fmt → SVT app format string and bit depth
#         fmt_map = {
#             "yuv444p10le": ("yuv444", 10),
#             "yuv422p10le": ("yuv422", 10),
#             "yuv420p10le": ("yuv420", 10),
#             "yuv444p":     ("yuv444",  8),
#             "yuv422p":     ("yuv422",  8),
#             "yuv420p":     ("yuv420",  8),
#         }
#         svt_fmt, bit_depth = fmt_map.get(str(pix_fmt), ("yuv444", 10))

#         for i in range(B):
#             x01 = (z_cpu[i] + 1.0) / 2.0
#             img_u8 = (x01 * 255.0).round().clamp(0, 255).to(torch.uint8)

#             if img_u8.shape[0] == 1:
#                 img_np = img_u8[0].numpy()
#                 img_pil = Image.fromarray(img_np, mode="L").convert("RGB")
#             else:
#                 img_np = img_u8.permute(1, 2, 0).numpy()
#                 img_pil = Image.fromarray(img_np, mode="RGB")

#             with tempfile.TemporaryDirectory() as td:
#                 td = Path(td)
#                 in_png  = td / "in.png"
#                 raw_yuv = td / "in.yuv"
#                 out_jxs = td / "out.jxs"
#                 out_yuv = td / "out.yuv"
#                 out_png = td / "out.png"

#                 img_pil.save(in_png)

#                 try:
#                     # 1) PNG → raw YUV via ffmpeg
#                     _run_cmd([
#                         "ffmpeg", "-y",
#                         "-i", str(in_png),
#                         "-f", "rawvideo",
#                         "-pix_fmt", str(pix_fmt),
#                         str(raw_yuv),
#                     ], timeout=30, label="PNG→YUV")

#                     # 2) raw YUV → JXS via SvtJpegxsEncApp
#                     _run_cmd([
#                         enc_app,
#                         "-i", str(raw_yuv),
#                         "-b", str(out_jxs),
#                         "-w", str(W),
#                         "-h", str(H),
#                         "-n", "1",
#                         "--input-depth", str(bit_depth),
#                         "--colour-format", svt_fmt,
#                         "--bpp", f"{bpp:.4f}",
#                     ], timeout=45, label="JXS encode")

#                     # 3) JXS → raw YUV via SvtJpegxsDecApp
#                     _run_cmd([
#                         dec_app,
#                         "-i", str(out_jxs),
#                         "-o", str(out_yuv),
#                     ], timeout=60, label="JXS decode")

#                     # 4) raw YUV → PNG via ffmpeg (this is where timeout spikes commonly happen)
#                     _run_cmd([
#                         "ffmpeg", "-y",
#                         "-f", "rawvideo",
#                         "-pix_fmt", str(pix_fmt),
#                         "-s", f"{W}x{H}",
#                         "-i", str(out_yuv),
#                         "-pix_fmt", "rgb24",
#                         str(out_png),
#                     ], timeout=30, retry_timeout=90, label="YUV→PNG")

#                     dec_pil = Image.open(out_png)
#                     if img_u8.shape[0] == 1:
#                         dec_pil = dec_pil.convert("L")
#                         dec_np = np.array(dec_pil, dtype=np.uint8)
#                         dec_u8 = torch.from_numpy(dec_np).unsqueeze(0)
#                     else:
#                         dec_pil = dec_pil.convert("RGB")
#                         dec_np = np.array(dec_pil, dtype=np.uint8)
#                         dec_u8 = torch.from_numpy(dec_np).permute(2, 0, 1).contiguous()
#                 except Exception as e:
#                     # Never let a single eval sample kill training; pass-through keeps validation running.
#                     logger.warning("JPEGXS sample fallback (using input unchanged): %s", e)
#                     dec_u8 = img_u8

#             dec_f = dec_u8.to(torch.float32) / 255.0
#             dec_m11 = (dec_f * 2.0 - 1.0).clamp(-1.0, 1.0)
#             outs.append(dec_m11)

#         return torch.stack(outs, dim=0).to(device)

#     return _apply_attack_preserve(x, _core)



# def jpegxs_compression(
#     x: torch.Tensor,
#     bitrate: str = "40M",
#     pix_fmt: str = "yuv444p10le",
# ) -> torch.Tensor:
#     # Applies JPEG-XS encode/decode via ffmpeg and returns the reconstructed tensor
#     def _core(z: torch.Tensor):
#         if shutil.which("ffmpeg") is None:
#             raise RuntimeError(
#                 "ffmpeg not found on PATH. Install ffmpeg built with libsvtjpegxs "
#                 "(--enable-libsvtjpegxs) to use jpegxs_compression."
#             )

#         device = z.device
#         z_cpu = z.detach().cpu().clamp(-1.0, 1.0)

#         B, C, H, W = z_cpu.shape
#         outs = []

#         br = bitrate
#         if isinstance(br, (int, float, np.integer, np.floating)):
#             br = f"{int(br)}M"
#         br = str(br)

#         for i in range(B):
#             x01 = (z_cpu[i] + 1.0) / 2.0
#             img_u8 = (x01 * 255.0).round().clamp(0, 255).to(torch.uint8)

#             if img_u8.shape[0] == 1:
#                 img_np = img_u8[0].numpy()
#                 img_pil = Image.fromarray(img_np, mode="L")
#             else:
#                 img_np = img_u8.permute(1, 2, 0).numpy()
#                 img_pil = Image.fromarray(img_np, mode="RGB")

#             with tempfile.TemporaryDirectory() as td:
#                 td = Path(td)
#                 in_png  = td / "in.png"
#                 out_mxf = td / "out.mxf"        # ← MXF instead of .jxs
#                 out_png = td / "out.png"

#                 img_pil.save(in_png)

#                 encode_cmd = [
#                     "ffmpeg", "-y",
#                     "-i", str(in_png),           # ← drop `-loop 1`
#                     "-frames:v", "1",
#                     "-c:v", "libsvtjpegxs",
#                     "-pix_fmt", str(pix_fmt),
#                     "-b:v", br,
#                     str(out_mxf),                # ← .mxf
#                 ]

#                 enc = subprocess.run(encode_cmd, capture_output=True, text=True, timeout=45)
#                 if enc.returncode != 0:
#                     raise RuntimeError(
#                         "JPEG-XS encode failed.\n"
#                         f"Command: {' '.join(encode_cmd)}\n"
#                         f"stderr:\n{enc.stderr}"
#                     )

#                 decode_cmd = [
#                     "ffmpeg", "-y",
#                     "-i", str(out_mxf),          # ← .mxf
#                     "-frames:v", "1",
#                     "-pix_fmt", "rgb24",
#                     str(out_png),
#                 ]

#                 dec = subprocess.run(decode_cmd, capture_output=True, text=True, timeout=45)
#                 if dec.returncode != 0:
#                     raise RuntimeError(
#                         "JPEG-XS decode failed.\n"
#                         f"Command: {' '.join(decode_cmd)}\n"
#                         f"stderr:\n{dec.stderr}"
#                     )

#                 dec_pil = Image.open(out_png)
#                 if img_u8.shape[0] == 1:
#                     dec_pil = dec_pil.convert("L")
#                     dec_np = np.array(dec_pil, dtype=np.uint8)
#                     dec_u8 = torch.from_numpy(dec_np).unsqueeze(0)
#                 else:
#                     dec_pil = dec_pil.convert("RGB")
#                     dec_np = np.array(dec_pil, dtype=np.uint8)
#                     dec_u8 = torch.from_numpy(dec_np).permute(2, 0, 1).contiguous()

#             dec_f = dec_u8.to(torch.float32) / 255.0
#             dec_m11 = (dec_f * 2.0 - 1.0).clamp(-1.0, 1.0)
#             outs.append(dec_m11)

#         return torch.stack(outs, dim=0).to(device)

#     return _apply_attack_preserve(x, _core)


# def jpegxs_compression(
#     x: torch.Tensor,
#     bitrate: str = "40M",
#     pix_fmt: str = "yuv444p10le",
# ) -> torch.Tensor:
#     # Applies JPEG-XS encode/decode via ffmpeg and returns the reconstructed tensor
#     def _core(z: torch.Tensor):
#         if shutil.which("ffmpeg") is None:
#             raise RuntimeError(
#                 "ffmpeg not found on PATH. Install ffmpeg built with libsvtjpegxs "
#                 "(--enable-libsvtjpegxs) to use jpegxs_compression."
#             )

#         device = z.device
#         z_cpu = z.detach().cpu().clamp(-1.0, 1.0)

#         B, C, H, W = z_cpu.shape
#         outs = []

        
#         br = bitrate
#         if isinstance(br, (int, float, np.integer, np.floating)):
#             br = f"{int(br)}M"
#         br = str(br)
        
#         # check for image size, if it lower than 256x256 use br 500M to avoid ffmpeg stalling on very small inputs with low bitrate
#         if H < 512 or W < 512:
#             br = "500M"

#         for i in range(B):
#             x01 = (z_cpu[i] + 1.0) / 2.0
#             img_u8 = (x01 * 255.0).round().clamp(0, 255).to(torch.uint8)

#             if img_u8.shape[0] == 1:
#                 img_np = img_u8[0].numpy()
#                 img_pil = Image.fromarray(img_np, mode="L")
#             else:
#                 img_np = img_u8.permute(1, 2, 0).numpy()
#                 img_pil = Image.fromarray(img_np, mode="RGB")

#             with tempfile.TemporaryDirectory() as td:
#                 td = Path(td)
#                 in_png  = td / "in.png"
#                 out_jxs = td / "out.jxs"
#                 out_png = td / "out.png"

#                 img_pil.save(in_png)

#                 # ENCODE: Remove -f rawvideo, use default muxer for .jxs
#                 # We keep the high bitrate for small images
#                 encode_cmd = [
#                     "ffmpeg", "-y",
#                     "-i", str(in_png),
#                     "-frames:v", "1",
#                     "-c:v", "libsvtjpegxs",
#                     "-pix_fmt", str(pix_fmt),
#                     "-b:v", br,
#                     str(out_jxs),
#                 ]

#                 enc = subprocess.run(encode_cmd, capture_output=True, text=True, timeout=45)
#                 if enc.returncode != 0:
#                     raise RuntimeError(f"JPEG-XS encode failed.\n{enc.stderr}")

#                 # DECODE: Help the decoder by defining the pixel format it's about to see
#                 decode_cmd = [
#                     "ffmpeg", "-y",
#                     "-probesize", "15M",
#                     "-analyzeduration", "15M",
#                     "-c:v", "libsvtjpegxs", # Explicitly load the codec
#                     "-f", "jpegxs_pipe",
#                     "-i", str(out_jxs),
#                     "-frames:v", "1",
#                     "-pix_fmt", "rgb24",
#                     str(out_png),
#                 ]

#                 dec = subprocess.run(decode_cmd, capture_output=True, text=True, timeout=45)
#                 if dec.returncode != 0:
#                     # If it still fails, it's likely a header size issue with SVT
#                     raise RuntimeError(f"JPEG-XS decode failed.\n{dec.stderr}")

#                 dec_pil = Image.open(out_png)
#                 if img_u8.shape[0] == 1:
#                     dec_pil = dec_pil.convert("L")
#                     dec_np = np.array(dec_pil, dtype=np.uint8)
#                     dec_u8 = torch.from_numpy(dec_np).unsqueeze(0)
#                 else:
#                     dec_pil = dec_pil.convert("RGB")
#                     dec_np = np.array(dec_pil, dtype=np.uint8)
#                     dec_u8 = torch.from_numpy(dec_np).permute(2, 0, 1).contiguous()

#             dec_f = dec_u8.to(torch.float32) / 255.0
#             dec_m11 = (dec_f * 2.0 - 1.0).clamp(-1.0, 1.0)
#             outs.append(dec_m11)

#         return torch.stack(outs, dim=0).to(device)

#     return _apply_attack_preserve(x, _core)
def _jpegxs_single(img_pil, num_channels, pix_fmt, br, max_retries=3):
    """Encode/decode a single image via JPEG-XS. Returns uint8 tensor or None on failure."""
    for attempt in range(max_retries):
        try:
            with tempfile.TemporaryDirectory() as td:
                td = Path(td)
                in_png  = td / "in.png"
                out_jxs = td / "out.jxs"
                out_png = td / "out.png"

                img_pil.save(in_png)

                # Encode
                encode_cmd = [
                    "ffmpeg", "-y",
                    "-i", str(in_png),
                    "-frames:v", "1",
                    "-c:v", "libsvtjpegxs",
                    "-pix_fmt", str(pix_fmt),
                    "-b:v", br,
                    str(out_jxs),
                ]

                enc = subprocess.run(encode_cmd, capture_output=True, text=True, timeout=30)
                if enc.returncode != 0:
                    continue

                if not out_jxs.exists() or out_jxs.stat().st_size == 0:
                    continue

                # Small delay to ensure file is fully flushed
                import time
                time.sleep(0.05)

                # Decode
                decode_cmd = [
                    "ffmpeg", "-y",
                    "-probesize", "50M",
                    "-analyzeduration", "50M",
                    "-i", str(out_jxs),
                    "-frames:v", "1",
                    "-update", "1",
                    str(out_png),
                ]

                dec = subprocess.run(decode_cmd, capture_output=True, text=True, timeout=30)
                if dec.returncode != 0:
                    continue

                if not out_png.exists():
                    continue

                dec_pil = Image.open(out_png)
                if num_channels == 1:
                    dec_pil = dec_pil.convert("L")
                    dec_np = np.array(dec_pil, dtype=np.uint8)
                    return torch.from_numpy(dec_np).unsqueeze(0)
                else:
                    dec_pil = dec_pil.convert("RGB")
                    dec_np = np.array(dec_pil, dtype=np.uint8)
                    return torch.from_numpy(dec_np).permute(2, 0, 1).contiguous()

        except Exception as e:
            print(f"[JPEGXS] Attempt {attempt+1}/{max_retries} exception: {e}")
            continue

    print(f"[JPEGXS] All {max_retries} attempts failed, using fallback (passthrough).")
    return None


def jpegxs_compression(
    x: torch.Tensor,
    bitrate: str = "40M",
    pix_fmt: str = "yuv444p10le",
) -> torch.Tensor:
    def _core(z: torch.Tensor):
        if shutil.which("ffmpeg") is None:
            raise RuntimeError(
                "ffmpeg not found on PATH. Install ffmpeg built with libsvtjpegxs."
            )

        device = z.device
        z_cpu = z.detach().cpu().clamp(-1.0, 1.0)

        B, C, H, W = z_cpu.shape
        outs = []

        br = bitrate
        if isinstance(br, (int, float, np.integer, np.floating)):
            br = f"{int(br)}M"
        br = str(br)

        for i in range(B):
            x01 = (z_cpu[i] + 1.0) / 2.0
            img_u8 = (x01 * 255.0).round().clamp(0, 255).to(torch.uint8)

            if img_u8.shape[0] == 1:
                img_np = img_u8[0].numpy()
                img_pil = Image.fromarray(img_np, mode="L")
            else:
                img_np = img_u8.permute(1, 2, 0).numpy()
                img_pil = Image.fromarray(img_np, mode="RGB")

            decoded = _jpegxs_single(img_pil, img_u8.shape[0], pix_fmt, br)

            if decoded is None:
                # Fallback: return original unchanged (no crash)
                decoded = img_u8

            dec_f = decoded.to(torch.float32) / 255.0
            dec_m11 = (dec_f * 2.0 - 1.0).clamp(-1.0, 1.0)
            outs.append(dec_m11)

        return torch.stack(outs, dim=0).to(device)

    return _apply_attack_preserve(x, _core)




def gaussian_noise(x: torch.Tensor, var: float = 0.01) -> torch.Tensor:
    # Add zero-mean Gaussian noise (variance in 0–1 scale)
    def _core(z: torch.Tensor):
        z = z.clamp(-1.0, 1.0)
        v = float(var)

        if v > 1.0:
            sigma01 = v / 255.0
            v = sigma01 * sigma01

        sigma01 = math.sqrt(max(v, 0.0))
        sigma_m11 = 2.0 * sigma01

        noise = torch.normal(
            mean=0.0,
            std=sigma_m11,
            size=z.shape,
            device=z.device,
            dtype=z.dtype,
        )
        return (z + noise).clamp(-1.0, 1.0)

    return _apply_attack_preserve(x, _core)


def speckle_noise(x: torch.Tensor, sigma: float) -> torch.Tensor:
    # Add multiplicative (speckle) noise
    def _core(z: torch.Tensor):
        z = z.clamp(-1.0, 1.0)
        noise = torch.randn_like(z) * float(sigma)
        return (z + z * noise).clamp(-1.0, 1.0)

    return _apply_attack_preserve(x, _core)


def blurring(x: torch.Tensor, k: int) -> torch.Tensor:
    # Apply Gaussian blur with kernel size k
    def _core(z: torch.Tensor):
        kk = int(k)
        if kk % 2 == 0:
            kk += 1

        if _HAS_KORNIA:
            sigma = float(kk) / 6.0
            out = Kf.gaussian_blur2d(z, (kk, kk), (sigma, sigma))
            return out.clamp(-1.0, 1.0)

        sigma = float(kk) / 6.0
        out = TF.gaussian_blur(z, kernel_size=[kk, kk], sigma=[sigma, sigma])
        return out.clamp(-1.0, 1.0)

    return _apply_attack_preserve(x, _core)


def brightness(x: torch.Tensor, factor: float) -> torch.Tensor:
    # Scale image brightness by multiplying values in [0,1] space, then convert back
    def _core(z: torch.Tensor) -> torch.Tensor:
        z = z.clamp(-1.0, 1.0)
        z01 = (z + 1.0) / 2.0
        out01 = (z01 * float(factor)).clamp(0.0, 1.0)
        return (out01 * 2.0 - 1.0).clamp(-1.0, 1.0)

    return _apply_attack_preserve(x, _core)


# def histogram_equalization(x: torch.Tensor) -> torch.Tensor:
#     # Equalize image histogram per sample using PIL's equalize implementation.
#     def _core(z: torch.Tensor) -> torch.Tensor:
#         device = z.device
#         z_cpu = z.detach().cpu().clamp(-1.0, 1.0)

#         outs = []
#         for i in range(z_cpu.shape[0]):
#             x01 = (z_cpu[i] + 1.0) / 2.0
#             img_u8 = (x01 * 255.0).round().clamp(0, 255).to(torch.uint8)

#             if img_u8.shape[0] == 1:
#                 pil = Image.fromarray(img_u8[0].numpy(), mode="L")
#                 eq = ImageOps.equalize(pil)
#                 out_u8 = torch.from_numpy(np.array(eq, dtype=np.uint8)).unsqueeze(0)
#             else:
#                 pil = Image.fromarray(img_u8.permute(1, 2, 0).numpy(), mode="RGB")
#                 eq = ImageOps.equalize(pil)
#                 out_u8 = (
#                     torch.from_numpy(np.array(eq, dtype=np.uint8))
#                     .permute(2, 0, 1)
#                     .contiguous()
#                 )

#             out_f = out_u8.to(torch.float32) / 255.0
#             outs.append((out_f * 2.0 - 1.0).clamp(-1.0, 1.0))

#         return torch.stack(outs, dim=0).to(device)

#     return _apply_attack_preserve(x, _core)

def histogram_equalization(x: torch.Tensor, strength: float = 1.0) -> torch.Tensor:
    # Apply per-channel histogram equalization blended with the original.
    # strength=1.0 → fully equalized; strength=0.0 → original unchanged.
    def _core(z: torch.Tensor) -> torch.Tensor:
        device = z.device
        z_cpu = z.detach().cpu().clamp(-1.0, 1.0)
        B, C, H, W = z_cpu.shape
        s = float(strength)
        outs = []

        for i in range(B):
            z01 = (z_cpu[i] + 1.0) / 2.0
            img_u8 = (z01 * 255.0).round().clamp(0, 255).to(torch.uint8).numpy()
            out_channels = []
            for c in range(C):
                ch = img_u8[c]  # HxW uint8
                hist, _ = np.histogram(ch.flatten(), bins=256, range=[0, 256])
                cdf = hist.cumsum()
                cdf_min = int(cdf[cdf > 0].min())
                n_pixels = H * W
                denom = n_pixels - cdf_min
                if denom == 0:
                    out_channels.append(torch.from_numpy(ch.copy()))
                    continue
                lut = np.clip(
                    np.round((cdf - cdf_min) / denom * 255.0), 0, 255
                ).astype(np.uint8)
                equalized = lut[ch]
                blended = np.clip(
                    np.round((1.0 - s) * ch + s * equalized), 0, 255
                ).astype(np.uint8)
                out_channels.append(torch.from_numpy(blended))

            out_u8 = torch.stack(out_channels, dim=0)  # CHW uint8
            out01 = out_u8.to(torch.float32) / 255.0
            outs.append((out01 * 2.0 - 1.0).clamp(-1.0, 1.0))

        return torch.stack(outs, dim=0).to(device)

    return _apply_attack_preserve(x, _core)

# def gamma_correction(x: torch.Tensor, gamma: Optional[float] = None) -> torch.Tensor:
#     # Apply gamma mapping in [0,1] space (gamma<1 brightens, gamma>1 darkens).
#     if gamma is None:
#         gamma = float(torch.empty(1).uniform_(0.7, 1.6).item())

#     def _core(z: torch.Tensor) -> torch.Tensor:
#         z = z.clamp(-1.0, 1.0)
#         z01 = (z + 1.0) / 2.0
#         out01 = torch.pow(z01.clamp(0.0, 1.0), float(gamma))
#         return (out01 * 2.0 - 1.0).clamp(-1.0, 1.0)

#     return _apply_attack_preserve(x, _core)

def gamma_correction(x: torch.Tensor, gamma: float = 1.5) -> torch.Tensor:
    # Apply gamma correction: out = in^gamma, where in/out are in [0,1] space
    def _core(z: torch.Tensor) -> torch.Tensor:
        z01 = ((z.clamp(-1.0, 1.0) + 1.0) / 2.0).clamp(0.0, 1.0)
        out01 = z01.pow(float(gamma)).clamp(0.0, 1.0)
        return (out01 * 2.0 - 1.0).clamp(-1.0, 1.0)
    return _apply_attack_preserve(x, _core)



def sharpness(x: torch.Tensor, amount: float = 1.0) -> torch.Tensor:
    # Apply simple unsharp masking: add back (original - blurred) scaled by amount
    def _core(z: torch.Tensor) -> torch.Tensor:
        z = z.clamp(-1.0, 1.0)
        blur = F.avg_pool2d(z, kernel_size=3, stride=1, padding=1)
        out = z + float(amount) * (z - blur)
        return out.clamp(-1.0, 1.0)

    return _apply_attack_preserve(x, _core)


def median_filtering(x: torch.Tensor, k: int) -> torch.Tensor:
    # Apply median filtering with kernel size k
    def _core(z: torch.Tensor):
        kk = int(k)
        if kk % 2 == 0:
            kk += 1

        z = z.clamp(-1.0, 1.0)

        if _HAS_KORNIA:
            out = Kf.median_blur(z, (kk, kk))
            return out.clamp(-1.0, 1.0)

        B, C, H, W = z.shape
        pad = kk // 2
        z_pad = F.pad(z, (pad, pad, pad, pad), mode="reflect")
        patches = z_pad.unfold(2, kk, 1).unfold(3, kk, 1)
        patches = patches.contiguous().view(B, C, H, W, kk * kk)
        median_vals = patches.median(dim=-1).values
        return median_vals.clamp(-1.0, 1.0)

    return _apply_attack_preserve(x, _core)


# ============================================================
# 4) AI pipeline
# ============================================================

import requests
import openai
from io import BytesIO

_AI_CACHE: dict[str, Any] = {}


def _ensure_openai_key():
    # Ensure OpenAI API key is available from OPENAI_API_KEY env var
    import os

    key = os.environ.get("OPENAI_API_KEY", "dummy")
    if not key:
        raise RuntimeError(
            "OpenAI API key not set. Set the OPENAI_API_KEY environment variable "
            "before calling replace_ai/create_ai."
        )


def _get_openai_client():
    import os
    import openai as _openai
    
    api_key = os.environ.get("OPENAI_API_KEY", "dummy")
    base_url = os.environ.get("OPENAI_BASE_URL", "http://localhost:8005/v1")
    logger.info("Creating OpenAI-compatible client with base_url=%s", base_url)
    
    return _openai.OpenAI(
        api_key=api_key,
        base_url=base_url
    )


def _get_device(device: Optional[str] = None) -> str:
    return device or ("cuda" if torch.cuda.is_available() else "cpu")


def _chw_u8_to_pil_rgb(x_chw_u8: torch.Tensor) -> Image.Image:
    # Convert a CHW uint8 tensor (1 or 3 channels) into a PIL RGB image
    x = x_chw_u8.detach().cpu()
    if x.dtype != torch.uint8:
        x = x.to(torch.uint8)
    if x.dim() != 3:
        raise ValueError(f"Expected CHW tensor, got {tuple(x.shape)}")
    if x.shape[0] == 1:
        return Image.fromarray(x[0].numpy(), mode="L").convert("RGB")
    return Image.fromarray(x.permute(1, 2, 0).numpy(), mode="RGB")


def _pil_rgb_to_chw_u8(pil: Image.Image, like: torch.Tensor) -> torch.Tensor:
    # Convert a PIL image back to CHW uint8
    if like.shape[0] == 1:
        arr = np.array(pil.convert("L"), dtype=np.uint8)
        return torch.from_numpy(arr).unsqueeze(0)
    arr = np.array(pil.convert("RGB"), dtype=np.uint8)
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


def _get_yolo_and_sam(
    *,
    yolo_weights: str = "yolov10x.pt",
    sam_checkpoint: str = "sam_vit_h_4b8939.pth",
    sam_model_type: str = "vit_h",
    device: Optional[str] = None,
):
    # Load and cache YOLO detector + SAM segmenter for a given device and checkpoint config
    dev = _get_device(device)

    sam_path = Path(sam_checkpoint)
    if sam_path.exists():
        ckpt_path = str(sam_path)
    else:
        ckpt_path = ensure_sam_checkpoint(sam_path.name)

    key = f"yolo_sam::{yolo_weights}::{ckpt_path}::{sam_model_type}::{dev}"
    if key in _AI_CACHE:
        return _AI_CACHE[key]

    from ultralytics import YOLO
    from segment_anything import sam_model_registry, SamPredictor

    yolo = YOLO(yolo_weights)

    sam = sam_model_registry[sam_model_type](checkpoint=ckpt_path)
    sam.to(device=dev)
    predictor = SamPredictor(sam)

    _AI_CACHE[key] = (yolo, predictor, dev)
    return yolo, predictor, dev


def _get_blip(*, device: Optional[str] = None):
    # Load and cache BLIP image captioning model + processor
    dev = _get_device(device)
    key = f"blip::{dev}"
    if key in _AI_CACHE:
        return _AI_CACHE[key]
    from transformers import BlipProcessor, BlipForConditionalGeneration

    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-base"
    ).to(dev)
    _AI_CACHE[key] = (processor, model, dev)
    return processor, model, dev


def _get_instruct_pix2pix(
    *, model_name="paint-by-inpaint/general-finetuned-mb", device: Optional[str] = None
):
    # Load and cache an InstructPix2Pix diffusion pipeline for masked editing
    dev = _get_device(device)
    key = f"ip2p::{model_name}::{dev}"
    if key in _AI_CACHE:
        return _AI_CACHE[key]
    from diffusers import (
        StableDiffusionInstructPix2PixPipeline,
        EulerAncestralDiscreteScheduler,
    )

    dtype = torch.float16 if dev.startswith("cuda") else torch.float32
    pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
        model_name, torch_dtype=dtype
    ).to(dev)
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
    _AI_CACHE[key] = (pipe, dev)
    return pipe, dev


def _get_qwen_image_edit_plus(
    *,
    model_name: str = "black-forest-labs/FLUX.2-klein-9B",
    device: Optional[str] = None,
    torch_dtype: Optional[torch.dtype] = None,
):
    # Load and cache Qwen Image Edit Plus pipeline for local image editing.
    dev = _get_device(device)
    if torch_dtype is None:
        if dev.startswith("cuda") and torch.cuda.is_bf16_supported():
            dtype = torch.bfloat16
        elif dev.startswith("cuda"):
            dtype = torch.float16
        else:
            dtype = torch.float32
    else:
        dtype = torch_dtype

    key = f"qwen_image_edit_plus::{model_name}::{dev}::{str(dtype)}"
    if key in _AI_CACHE:
        return _AI_CACHE[key]

    try:
        from diffusers import QwenImageEditPlusPipeline
    except Exception as ex:
        raise ImportError(
            "QwenImageEditPlusPipeline is unavailable. Update/install diffusers with "
            "Qwen image editing support."
        ) from ex

    pipe = QwenImageEditPlusPipeline.from_pretrained(model_name, torch_dtype=dtype)
    pipe.to(dev)
    pipe.set_progress_bar_config(disable=None)

    _AI_CACHE[key] = (pipe, dev)
    return pipe, dev


def _get_flux2_klein(
    *,
    model_name: str = "black-forest-labs/FLUX.2-klein-9B",
    device: Optional[str] = None,
    torch_dtype: Optional[torch.dtype] = None,
):
    # Load and cache FLUX.2-klein pipeline for local image generation.
    dev = _get_device(device)
    if torch_dtype is None:
        if dev.startswith("cuda") and torch.cuda.is_bf16_supported():
            dtype = torch.bfloat16
        elif dev.startswith("cuda"):
            dtype = torch.float16
        else:
            dtype = torch.float32
    else:
        dtype = torch_dtype

    key = f"flux2_klein::{model_name}::{dev}::{str(dtype)}"
    if key in _AI_CACHE:
        return _AI_CACHE[key]

    try:
        from diffusers import Flux2KleinPipeline
    except Exception as ex:
        raise ImportError(
            "Flux2KleinPipeline is unavailable. Update/install diffusers with "
            "FLUX.2-klein support."
        ) from ex

    pipe = Flux2KleinPipeline.from_pretrained(model_name, torch_dtype=dtype)
    if dev.startswith("cuda"):
        pipe.enable_model_cpu_offload()
    else:
        pipe.to(dev)
    pipe.set_progress_bar_config(disable=None)

    _AI_CACHE[key] = (pipe, dev)
    return pipe, dev


def _get_z_image_turbo(
    *,
    model_name: str = "Tongyi-MAI/Z-Image-Turbo",
    device: Optional[str] = None,
    torch_dtype: Optional[torch.dtype] = None,
):
    # Load and cache Z-Image-Turbo pipeline for local image generation.
    dev = _get_device(device)
    if torch_dtype is None:
        if dev.startswith("cuda") and torch.cuda.is_bf16_supported():
            dtype = torch.bfloat16
        elif dev.startswith("cuda"):
            dtype = torch.float16
        else:
            dtype = torch.float32
    else:
        dtype = torch_dtype

    key = f"z_image_turbo::{model_name}::{dev}::{str(dtype)}"
    if key in _AI_CACHE:
        return _AI_CACHE[key]

    try:
        from diffusers import ZImagePipeline
    except Exception as ex:
        raise ImportError(
            "ZImagePipeline is unavailable. Update/install diffusers with "
            "Z-Image support."
        ) from ex

    pipe = ZImagePipeline.from_pretrained(
        model_name,
        torch_dtype=dtype,
        low_cpu_mem_usage=False,
    )
    pipe.to(dev)
    pipe.set_progress_bar_config(disable=None)

    _AI_CACHE[key] = (pipe, dev)
    return pipe, dev


def _get_replace_ai_pipeline(
    *,
    model_name: str,
    model_family: str = "auto",
    device: Optional[str] = None,
    torch_dtype: Optional[torch.dtype] = None,
):
    # Select and load a replace-attack pipeline by family (qwen/flux2klein/zimage/auto).
    family = str(model_family or "auto").strip().lower()
    if family == "auto":
        model_name_l = str(model_name).lower()
        if "z-image" in model_name_l or "zimage" in model_name_l:
            family = "zimage"
        elif "flux.2-klein" in model_name_l or "flux2-klein" in model_name_l:
            family = "flux2klein"
        else:
            family = "qwen"

    if family == "qwen":
        return _get_qwen_image_edit_plus(
            model_name=model_name,
            device=device,
            torch_dtype=torch_dtype,
        )
    if family == "flux2klein":
        return _get_flux2_klein(
            model_name=model_name,
            device=device,
            torch_dtype=torch_dtype,
        )
    if family == "zimage":
        return _get_z_image_turbo(
            model_name=model_name,
            device=device,
            torch_dtype=torch_dtype,
        )

    raise ValueError(
        f"Unsupported replace model_family='{model_family}'. "
        "Use one of: auto, qwen, flux2klein, zimage"
    )


def _yolov10_detection(model, image_batch: list[np.ndarray]):
    # Run YOLOv10 on numpy RGB images and return boxes + string labels per image
    results = model(image_batch)
    batch_boxes, batch_labels = [], []
    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()
        labels = [result.names[int(cls.cpu().numpy())] for cls in result.boxes.cls]
        batch_boxes.append(boxes)
        batch_labels.append(labels)
    return batch_boxes, batch_labels


def _make_primary_mask_yolo_sam(
    image_rgb: Image.Image,
    *,
    threshold_area: int = 500,
    yolo_weights: str = "yolov10x.pt",
    sam_checkpoint: str = "sam_vit_h_4b8939.pth",
    sam_model_type: str = "vit_h",
    device: Optional[str] = None,
) -> Image.Image:
    # Detect objects with YOLO and segment with SAM - return the largest mask above threshold_area
    yolo, predictor, _ = _get_yolo_and_sam(
        yolo_weights=yolo_weights,
        sam_checkpoint=sam_checkpoint,
        sam_model_type=sam_model_type,
        device=device,
    )

    img_np = np.array(image_rgb.convert("RGB"))
    boxes_batch, labels_batch = _yolov10_detection(yolo, [img_np])
    boxes = boxes_batch[0]
    labels = labels_batch[0]

    predictor.set_image(img_np)

    all_masks_with_area = []
    for box, label in zip(boxes, labels):
        input_box = np.array(box)
        masks, _, _ = predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_box[None, :],
            multimask_output=True,
        )
        for mask in masks:
            area = int(mask.sum())
            if area > threshold_area:
                all_masks_with_area.append((mask, area))

    all_masks_with_area.sort(key=lambda x: x[1], reverse=True)

    if len(all_masks_with_area) < 1:
        if len(boxes) == 0:
            return Image.fromarray(
                np.zeros((img_np.shape[0], img_np.shape[1]), dtype=np.uint8), mode="L"
            )
        x0, y0, x1, y1 = map(int, boxes[0])
        rect = np.zeros((img_np.shape[0], img_np.shape[1]), dtype=np.uint8)
        rect[y0:y1, x0:x1] = 255
        return Image.fromarray(rect, mode="L")

    mask_bool = all_masks_with_area[0][0]
    return Image.fromarray((mask_bool.astype(np.uint8) * 255), mode="L")


def _masked_crop(image_rgb: Image.Image, mask_l: Image.Image) -> Image.Image:
    # Zero-out everything outside the mask (keeps only masked region)
    img = np.array(image_rgb.convert("RGB"))
    m = np.array(mask_l.convert("L")) > 0
    out = img.copy()
    out[~m] = 0
    return Image.fromarray(out, mode="RGB")


def _blip_caption(pil_rgb: Image.Image, *, device: Optional[str] = None) -> str:
    # Generate a caption for an image using BLIP
    # try:
    processor, model, dev = _get_blip(device=device)
    inputs = processor(images=pil_rgb, return_tensors="pt").to(dev, torch.float32)
    out = model.generate(**inputs)
    return processor.decode(out[0], skip_special_tokens=True)
    # except Exception as ex:
    #     raise RuntimeError(f"BLIP captioning failed {str(ex)}") from ex




def _openai_prompt_for_replace(image_rgb: Image.Image, crop_rgb: Image.Image) -> str:
    # Use BLIP captions + OpenAI chat model to produce a DALL·E edit prompt for replacement
    _ensure_openai_key()
    import os

    model_name = os.environ.get("OPENAI_MODEL_NAME", "gemma4")
    logger.info("Generating replacement prompt with LLM model=%s", model_name)
    image_description = _blip_caption(image_rgb)
    masked_object_description = _blip_caption(crop_rgb)
    user_message = (
        f"Generate a prompt to replace the {masked_object_description} in an image similar to "
        f"'{image_description}', focusing on the areas defined by the provided masks. "
        f"Ensure the objects fit seamlessly into the scene."
    )
    try:
        completion = _get_openai_client().chat.completions.create(
            model=model_name,
            messages=[
                {
                    "role": "system",
                    "content": "You are skilled in creating prompts for DALL-E 2 image editing.",
                },
                {"role": "user", "content": user_message},
            ],
        )
        logger.info("Replacement prompt generated successfully")
        return completion.choices[0].message.content.strip()
    except Exception as ex:
        raise RuntimeError(f"LLM replacement prompt generation failed: {str(ex)}") from ex


def _openai_prompt_for_create(image_rgb_512: Image.Image) -> str:
    # Use BLIP + OpenAI chat model to suggest a simple object to add, returning an edit instruction
    _ensure_openai_key()
    import os

    model_name = os.environ.get("OPENAI_MODEL_NAME", "gemma4")
    logger.info("Generating creation prompt with LLM model=%s", model_name)
    image_description = _blip_caption(image_rgb_512)
    chatgpt_prompt = (
        f"Given the image description: '{image_description}', suggest a specific object "
        f"that would enhance the image. The object should be easily recognizable and should "
        f"not introduce complexity to the image."
    )
    try:
        response = _get_openai_client().chat.completions.create(
            model=model_name,
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert in suggesting simple, specific objects to enhance images based on descriptions.",
                },
                {"role": "user", "content": chatgpt_prompt},
            ],
            max_tokens=50,
            temperature=0.7,
        )
        logger.info("Creation prompt generated successfully")
        suggestion = response.choices[0].message.content.strip()
    except Exception as ex:
        raise RuntimeError("LLM creation prompt generation failed") from ex
    return f"Add '{suggestion}' to the image."


def _make_dalle_edit_mask(image_rgb: Image.Image, mask_l: Image.Image) -> Image.Image:
    # Create an RGBA mask image suitable for DALL·E edits (transparent where edits should happen)
    img = np.array(image_rgb.convert("RGB"), dtype=np.uint8)
    m = np.array(mask_l.convert("L"), dtype=np.uint8) > 0

    blended = img.copy()
    blended[m] = 0

    rgba = np.dstack([blended, np.full(blended.shape[:2], 255, dtype=np.uint8)])
    black = (rgba[..., 0] == 0) & (rgba[..., 1] == 0) & (rgba[..., 2] == 0)
    rgba[black, 3] = 0

    return Image.fromarray(rgba, mode="RGBA")


def _pil_to_png_bytes(pil_img: Image.Image) -> BytesIO:
    # Serialize a PIL image to PNG bytes in-memory.
    buf = BytesIO()
    pil_img.save(buf, format="PNG")
    buf.seek(0)
    return buf


# -------------------- Attacks --------------------


def replace_ai(
    x_chw_u8: torch.Tensor,
    strength: Any = None,
    *,
    threshold_area: int = 5000,
    feather_radius: int = 0,
    model_name: str = "black-forest-labs/FLUX.2-klein-9B",
    model_family: str = "auto",
    num_inference_steps: int = 40,
    true_cfg_scale: float = 4.0,
    guidance_scale: float = 1.0,
    negative_prompt: str = " ",
    seed: int = 0,
    **kwargs,
) -> torch.Tensor:
    
    print("replace_ai called with parameters: uses model=%s, num_inference_steps=%s, true_cfg_scale=%s, guidance_scale=%s, negative_prompt=%s, seed=%s", model_name, num_inference_steps, true_cfg_scale, guidance_scale, negative_prompt, seed)
    # Detect a main object, build a prompt, and replace it using local Qwen image editing.
    logger.info("replace_ai started: threshold_area=%s feather_radius=%s", threshold_area, feather_radius)

    image = _chw_u8_to_pil_rgb(x_chw_u8)
    mask_l = _make_primary_mask_yolo_sam(image, threshold_area=threshold_area)

    if feather_radius and feather_radius > 0:
        mask_l = mask_l.filter(ImageFilter.GaussianBlur(radius=float(feather_radius)))

    crop = _masked_crop(image, mask_l)
    try:
        prompt = _openai_prompt_for_replace(image, crop)
    except Exception as ex:
        logger.warning(
            "Prompt helper failed, using fallback replacement prompt: %s", str(ex)
        )
        prompt = "Replace the selected foreground object with a similar object that fits the scene naturally."

    pipe, dev = _get_replace_ai_pipeline(
        model_name=model_name,
        model_family=model_family,
    )
    gen_device = dev if dev.startswith("cuda") else "cpu"
    generator = torch.Generator(device=gen_device).manual_seed(int(seed))

    active_family = str(model_family or "auto").strip().lower()
    if active_family == "auto":
        model_name_l = str(model_name).lower()
        if "z-image" in model_name_l or "zimage" in model_name_l:
            active_family = "zimage"
        else:
            active_family = (
                "flux2klein"
                if "flux.2-klein" in model_name_l or "flux2-klein" in model_name_l
                else "qwen"
            )

    with torch.inference_mode():

        if active_family in ("flux2klein", "zimage"):
            family_guidance_scale = float(guidance_scale)
            if active_family == "zimage" and abs(family_guidance_scale - 1.0) < 1e-8:
                # Z-Image-Turbo recommends guidance_scale=0; auto-adjust default value.
                family_guidance_scale = 0.0
            out_images = pipe(
                prompt=str(prompt),
                height=int(image.size[1]),
                width=int(image.size[0]),
                guidance_scale=family_guidance_scale,
                num_inference_steps=int(num_inference_steps),
                generator=generator,
            ).images
        else:
            out_images = pipe(
                image=[image.convert("RGB"), crop.convert("RGB")],
                prompt=str(prompt),
                generator=generator,
                true_cfg_scale=float(true_cfg_scale),
                negative_prompt=str(negative_prompt),
                num_inference_steps=int(num_inference_steps),
                guidance_scale=float(guidance_scale),
                num_images_per_prompt=1,
            ).images

    if not out_images:
        raise RuntimeError("replace_ai pipeline returned no images")

    out = out_images[0].convert("RGB").resize(image.size, Image.LANCZOS)
    if active_family in ("flux2klein", "zimage"):
        # FLUX and Z-Image generate full images; keep edits local by compositing with the detected mask.
        out = Image.composite(out, image.convert("RGB"), mask_l.convert("L"))

    logger.info("replace_ai completed successfully")
    return _pil_rgb_to_chw_u8(out, x_chw_u8)


def remove_ai(
    x_chw_u8: torch.Tensor,
    strength: Any = None,
    *,
    threshold_area: int = 5000,
    feather_radius: int = 2,
    median_kernel: int = 1,
    upscale_factor: float = 1.0,
    **kwargs,
) -> torch.Tensor:
    # Detect a main object and remove it via inpainting (SimpleLama)
    image = _chw_u8_to_pil_rgb(x_chw_u8)

    mask_l = _make_primary_mask_yolo_sam(image, threshold_area=threshold_area)

    if feather_radius and feather_radius > 0:
        mask_l = mask_l.filter(ImageFilter.GaussianBlur(radius=float(feather_radius)))

    if upscale_factor and float(upscale_factor) != 1.0:
        s = float(upscale_factor)
        new_w = max(1, int(round(image.size[0] * s)))
        new_h = max(1, int(round(image.size[1] * s)))
        image_in = image.resize((new_w, new_h), Image.LANCZOS)
        mask_in = mask_l.resize((new_w, new_h), Image.LANCZOS)
    else:
        image_in = image
        mask_in = mask_l

    from simple_lama_inpainting import SimpleLama
    import cv2

    lama = SimpleLama()
    result = lama(image_in, mask_in)

    result_bgr = cv2.cvtColor(np.array(result), cv2.COLOR_RGB2BGR)

    k = int(median_kernel) if median_kernel is not None else 0
    if k > 1:
        if k % 2 == 0:
            k += 1
        result_bgr = cv2.medianBlur(result_bgr, k)

    result = Image.fromarray(cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB))

    if result.size != image.size:
        result = result.resize(image.size, Image.LANCZOS)

    return _pil_rgb_to_chw_u8(result, x_chw_u8)


def create_ai(
    x_chw_u8: torch.Tensor,
    strength: Any = None,
    *,
    threshold_area: int = 5000,
    model_name: str = "paint-by-inpaint/general-finetuned-mb",
    diffusion_steps: int = 50,
    guidance_scale: float = 7.0,
    image_guidance_scale: float = 1.5,
    **kwargs,
) -> torch.Tensor:

    # Detect a main region, create a prompt, then use InstructPix2Pix to add an object into the image
    _ensure_openai_key()
    logger.info("create_ai started: threshold_area=%s diffusion_steps=%s", threshold_area, diffusion_steps)

    image = _chw_u8_to_pil_rgb(x_chw_u8)
    mask_l = _make_primary_mask_yolo_sam(image, threshold_area=threshold_area).convert(
        "L"
    )

    image_512 = image.resize((512, 512))
    mask_512 = mask_l.resize((512, 512))

    prompt = _openai_prompt_for_create(image_512)

    pipe, _ = _get_instruct_pix2pix(model_name=model_name)
    out_images = pipe(
        prompt,
        image=image_512,
        mask_image=mask_512,
        guidance_scale=float(guidance_scale),
        image_guidance_scale=float(image_guidance_scale),
        num_inference_steps=int(diffusion_steps),
        num_images_per_prompt=1,
    ).images
    out = out_images[0].resize(image.size, Image.LANCZOS)

    logger.info("create_ai completed successfully")
    return _pil_rgb_to_chw_u8(out, x_chw_u8)



__all__ = [
    "rotate_tensor",
    "rotate_tensor_inverse"
    "rotate_tensor_keep_all",
    "crop",
    "scaled",
    "flipping",
    "resized",
    "jpeg_compression",
    "jpeg_compression_train_fast",
    "jpeg2000_compression",
    "jpeg2000_compression_train_fast",
    "jpegai_compression",
    "jpegxl_compression",
    "jpegxl_compression_train_fast",
    "jpegxs_compression",
    "jpegxs_compression_train_fast",
    "gaussian_noise",
    "speckle_noise",
    "blurring",
    "brightness",
    "histogram_equalization",
    "gamma_correction",
    "sharpness",
    "median_filtering",
    "remove_ai",
    "replace_ai",
    "create_ai",
]
