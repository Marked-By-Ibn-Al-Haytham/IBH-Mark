import logging
import random
from typing import Optional, List, Dict

import torch
from torch import nn

logger = logging.getLogger(__name__)


def _apply_ai_attack_batch(x: torch.Tensor, attack_fn, attack_name: str) -> torch.Tensor:
    """
    Apply an AI attack that expects CHW uint8 tensors to each image in a BCHW batch.
    Input/Output remain BCHW in [-1, 1]. Failures are handled per-sample.
    """
    if x.dim() != 4:
        raise ValueError(f"Expected BCHW tensor, got shape={tuple(x.shape)}")

    device = x.device
    dtype = x.dtype

    x_u8 = (
        ((x.detach().cpu().clamp(-1.0, 1.0) + 1.0) / 2.0) * 255.0
    ).round().to(torch.uint8)

    outs = []
    for i in range(x_u8.shape[0]):
        sample = x_u8[i]
        try:
            attacked = attack_fn(sample)
            if not isinstance(attacked, torch.Tensor):
                raise TypeError(
                    f"AI attack '{attack_name}' returned type {type(attacked)}"
                )
            if attacked.dtype != torch.uint8:
                attacked = attacked.clamp(0, 255).to(torch.uint8)
            if attacked.shape != sample.shape:
                raise ValueError(
                    f"AI attack '{attack_name}' changed shape from "
                    f"{tuple(sample.shape)} to {tuple(attacked.shape)}"
                )
            outs.append(attacked)
        except Exception as ex:
            logger.warning(
                f"AI attack '{attack_name}' failed on sample {i}: {ex}. "
                "Using original sample."
            )
            outs.append(sample)

    y_u8 = torch.stack(outs, dim=0)
    y = (y_u8.to(torch.float32) / 255.0) * 2.0 - 1.0
    return y.to(device=device, dtype=dtype).clamp(-1.0, 1.0)


# ============================================================
# Challenge attack registry
# Each entry: attack_name → (callable, default_strength)
# Strengths chosen to match realistic challenge evaluation levels.
# ============================================================
def _build_attack_registry(device: str) -> Dict:
    """
    Lazily import and register challenge attacks.
    Heavy/optional attacks (AI, compressai, ffmpeg) are wrapped in try/except
    so missing dependencies only disable those specific attacks.
    """
    from attack.attacks import (
        rotate_tensor,
        rotate_tensor_keep_all,
        crop,
        scaled,
        flipping,
        resized,
        jpeg_compression,
        jpeg2000_compression,
        jpegxl_compression,
        jpegxs_compression,
        gaussian_noise,
        speckle_noise,
        blurring,
        brightness,
        histogram_equalization,
        gamma_correction,
        sharpness,
        median_filtering,
        jpeg_compression_train_fast,
        jpeg2000_compression_train_fast,
        jpegxl_compression_train_fast,
        jpegxs_compression_train_fast,
    )

    ai_attacks = {}
    try:
        from attack.attacks import (
            jpegai_compression,
            remove_ai,
            replace_ai,
            create_ai,
        )

        ai_attacks = {
            "JPEGAI": (
                lambda x: _apply_ai_attack_batch(
                    x, lambda s: jpegai_compression(s, quality=4), "JPEGAI"
                ),
                "ai",
            ),
            "RemoveAI": (
                lambda x: _apply_ai_attack_batch(x, remove_ai, "RemoveAI"),
                "ai",
            ),
            # "ReplaceAI": (
            #     lambda x: _apply_ai_attack_batch(x, replace_ai, "ReplaceAI"),
            #     "ai",
            # ),
            # "CreateAI": (
            #     lambda x: _apply_ai_attack_batch(x, create_ai, "CreateAI"),
            #     "ai",
            # ),
        }
    except Exception as ex:
        logger.warning(f"AI attacks unavailable and will be disabled: {ex}")


    eval_registry = {
        # ── Geometric ─────────────────────────────────────────
        "Rotate": (lambda x: rotate_tensor(x, angle=15.0), "geo"),
        "Resized": (lambda x: resized(x, pct=20), "geo"),
        "Scaled": (lambda x: scaled(x, scale=0.5), "geo"),
        "Crop": (lambda x: crop(x, pct=10.0), "geo"),
        "HFlip": (lambda x: flipping(x, mode="H"), "geo"),
        # # ── Signal / Perturbation ──────────────────────────────
        "GaussNoise": (lambda x: gaussian_noise(x, var= 0.01), "pert"),
        "SpeckleNoise": (lambda x: speckle_noise(x, sigma=0.3), "pert"),
        "Blur5": (lambda x: blurring(x, k=5), "pert"),
        "Bright13": (lambda x: brightness(x, factor=1.3), "pert"),
        "Sharpness": (lambda x: sharpness(x, amount=1.25), "pert"),
        "HistEq": (lambda x: histogram_equalization(x), "pert"),
        "Gamma": (lambda x: gamma_correction(x, gamma=1.5), "pert"),
        "Median5": (lambda x: median_filtering(x, k=5), "pert"),
        # compression attacks
        "JPEG50": (lambda x: jpeg_compression(x, quality=50), "pert"),
        "JPEG2000": (lambda x: jpeg2000_compression(x, quality_layers=(10,)), "pert"),
        "JPEGXS": (lambda x: jpegxs_compression(x), "pert"),
        "JPEGXL": (lambda x: jpegxl_compression(x, quality=12), "pert"),
        **ai_attacks,
    }


    # eval_registry = {
    #     # ── Geometric ─────────────────────────────────────────
    #     "Rotate": (lambda x: rotate_tensor(x, angle=None), "geo"),
    #     "Crop": (lambda x: crop(x, pct=None), "geo"),
    #     "Scaled": (lambda x: scaled(x, scale=None), "geo"),
    #     "HFlip": (lambda x: flipping(x, mode="H"), "geo"),
    #     "VFlip": (lambda x: flipping(x, mode="V"), "geo"),
    #     "Resized": (lambda x: resized(x, pct=None), "geo"),
    #     # ── Signal / Perturbation ──────────────────────────────
    #     "JPEG50": (lambda x: jpeg_compression(x, quality=50), "pert"),
    #     "JPEG75": (lambda x: jpeg_compression(x, quality=75), "pert"),
    #     "JPEG2000": (lambda x: jpeg2000_compression(x), "pert"),
    #     "JPEGXS": (lambda x: jpegxs_compression(x), "pert"),
    #     "JPEGXL": (lambda x: jpegxl_compression(x, quality=50), "pert"),
    #     "GaussNoise": (lambda x: gaussian_noise(x, var=0.01), "pert"),
    #     "SpeckleNoise": (lambda x: speckle_noise(x, sigma=0.1), "pert"),
    #     "Blur3": (lambda x: blurring(x, k=3), "pert"),
    #     "Blur5": (lambda x: blurring(x, k=5), "pert"),
    #     "Bright125": (lambda x: brightness(x, factor=1.25), "pert"),
    #     "Bright075": (lambda x: brightness(x, factor=0.75), "pert"),
    #     "Sharpness": (lambda x: sharpness(x, amount=1.0), "pert"),
    #     "Median3": (lambda x: median_filtering(x, k=3), "pert"),
    #     # **ai_attacks,
    # }

    train_registry = eval_registry  # For simplicity, use the same attacks for training and evaluation

    return train_registry, eval_registry


class ChallengeNoiser(nn.Module):
    """
    Drop-in replacement for the original Noiser class.
    Uses the challenge's official attack functions instead of kornia augmentations.

    Interface is identical to the original Noiser:
      - forward(input)            → random geo + pert attacks
      - forward(input, [key])     → apply specific named attack(s)

    The curriculum in train.py (sorted by bit accuracy) works unchanged
    because attack names from get_attack_names() replace the old kornia keys.
    """

    def __init__(
        self,
        num_transforms: int,
        device: str,
        enabled_attacks: Optional[List[str]] = None,
        num_pert_transforms: Optional[int] = None,
        ai_attack_ratio: float = 0.0,
    ):
        super().__init__()
        self.device = device
        self.num_transforms = num_transforms
        self.num_pert_transforms = (
            num_pert_transforms if num_pert_transforms is not None else num_transforms
        )
        self.ai_attack_ratio = min(max(float(ai_attack_ratio), 0.0), 1.0)

        self._train_registry, self._eval_registry = _build_attack_registry(device)

        # Filter to only enabled attacks if specified, otherwise use all
        if enabled_attacks:
            self._train_registry = {
                k: v for k, v in self._train_registry.items() if k in enabled_attacks
            }
            self._eval_registry = {
                k: v for k, v in self._eval_registry.items() if k in enabled_attacks
            }
            available = set(self._train_registry.keys()) | set(self._eval_registry.keys())
            missing = set(enabled_attacks) - available
            if missing:
                logger.warning(f"Requested attacks not available: {missing}")

        self.train_geo_attacks = [
            k for k, (_, cat) in self._train_registry.items() if cat == "geo"
        ]
        self.train_pert_attacks = [
            k for k, (_, cat) in self._train_registry.items() if cat == "pert"
        ]
        self.train_ai_attacks = [
            k for k, (_, cat) in self._train_registry.items() if cat == "ai"
        ]

        logger.info(
            "ChallengeNoiser initialized. "
            f"Geo: {self.train_geo_attacks} | "
            f"Pert: {self.train_pert_attacks} | "
            f"AI: {self.train_ai_attacks} | "
            f"ai_attack_ratio={self.ai_attack_ratio:.2f}"
        )

    def get_train_attack_names(self) -> List[str]:
        """Returns all registered attack names — used by _calculate_metric in train.py."""
        return list(self._train_registry.keys())
    
    def get_eval_attack_names(self) -> List[str]:
        """Returns all registered attack names — used by _calculate_metric in train.py."""
        return list(self._eval_registry.keys())

    def forward(
        self, x: torch.Tensor, noises: Optional[List[str]] = None, train: bool = True
    ) -> torch.Tensor:
        """
        Args:
            x:      Image tensor in [-1, 1], shape (B, C, H, W).
            noises: If provided, apply exactly these named attacks in sequence.
                    If None, randomly sample num_transforms geo + num_transforms pert attacks.
        Returns:
            Attacked image tensor, same shape and range as input.
        """
        if noises is not None:
            for key in noises:
                if train:
                    x = self._apply_train(x, key)
                else:
                    x = self._apply_eval(x, key)
            return x
        if not train:
            logger.warning(
                "No attack keys provided to forward() in eval mode. Returning input unchanged."
            )
            return x

        # Random curriculum sampling with optional AI/classic ratio.
        classic_pool = self.train_geo_attacks + self.train_pert_attacks
        total_attacks = max(0, int(self.num_transforms) + int(self.num_pert_transforms))
        if total_attacks == 0:
            return x

        ai_k = 0
        if self.train_ai_attacks and self.ai_attack_ratio > 0.0:
            ai_k = int(round(total_attacks * self.ai_attack_ratio))
            ai_k = min(total_attacks, ai_k)

        classic_k = total_attacks - ai_k
        if classic_k > 0 and not classic_pool and self.train_ai_attacks:
            ai_k = total_attacks
            classic_k = 0

        if ai_k > 0 and not self.train_ai_attacks:
            ai_k = 0
            classic_k = total_attacks

        sampled = []
        if classic_k > 0 and classic_pool:
            sampled.extend(random.choices(classic_pool, k=classic_k))
        if ai_k > 0 and self.train_ai_attacks:
            sampled.extend(random.choices(self.train_ai_attacks, k=ai_k))

        random.shuffle(sampled)
        for key in sampled:
            print("key", key)
            x = self._apply_train(x, key)
        return x

    def _apply_train(self, x: torch.Tensor, key: str) -> torch.Tensor:
        if key not in self._train_registry:
            raise ValueError(
                f"Unknown attack '{key}'. Available: {list(self._train_registry.keys())}"
            )
        fn, _ = self._train_registry[key]
        try:
            return fn(x)
        except Exception as e:
            logger.warning(f"Attack '{key}' failed ({e}), returning input unchanged.")
            return x
    
    def _apply_eval(self, x: torch.Tensor, key: str) -> torch.Tensor:
        if key not in self._eval_registry:
            raise ValueError(
                f"Unknown attack '{key}'. Available: {list(self._eval_registry.keys())}"
            )
        fn, _ = self._eval_registry[key]
        try:
            return fn(x)
        except Exception as e:
            logger.warning(f"Attack '{key}' failed ({e}), returning input unchanged.")
            return x


# ============================================================
# Legacy compatibility shim
# train.py calls noise.supported_transforms() to get attack names
# for metric evaluation. We replace this with the registry keys.
# ============================================================
_SHARED_NOISER: Optional[ChallengeNoiser] = None


def init_shared_noiser(
    num_transforms: int,
    device: str,
    enabled_attacks: Optional[List[str]] = None,
    num_pert_transforms: Optional[int] = None,
    ai_attack_ratio: float = 0.0,
):
    """Call this once from train.py's Watermark.__init__ after creating the noiser."""
    global _SHARED_NOISER
    _SHARED_NOISER = ChallengeNoiser(
        num_transforms=num_transforms,
        device=device,
        enabled_attacks=enabled_attacks,
        num_pert_transforms=num_pert_transforms,
        ai_attack_ratio=ai_attack_ratio,
    )
    return _SHARED_NOISER


def get_train_attack_names(image_size=None) -> List[str]:
    """Replacement for supported_transforms() — returns list of attack names."""
    if _SHARED_NOISER is None:
        raise RuntimeError("Call init_shared_noiser() before get_train_attack_names().")
    return _SHARED_NOISER.get_train_attack_names()

def get_eval_attack_names(image_size=None) -> List[str]:
    """Replacement for supported_transforms() — returns list of attack names."""
    if _SHARED_NOISER is None:
        raise RuntimeError("Call init_shared_noiser() before get_eval_attack_names().")
    return _SHARED_NOISER.get_eval_attack_names()