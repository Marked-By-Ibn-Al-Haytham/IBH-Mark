"""
Decoder (Extractor) backbone registry.

Each backbone wraps a pretrained torchvision model and exposes the same
interface:  backbone(image) -> (B, num_encoded_bits)  logits in [0,1].

Registry key  →  torchvision model / description
------------------------------------------------------
convnext_base     ConvNeXt-Base  (original baseline)
convnext_large    ConvNeXt-Large  (~2× param of base)
efficientnet_v2_m EfficientNetV2-M  (fast, accurate)
efficientnet_v2_l EfficientNetV2-L  (larger variant)
swin_b            Swin Transformer-Base
swin_v2_b         Swin Transformer V2-Base
vit_b_16          ViT-B/16 (patch 16)
vit_l_16          ViT-L/16 (patch 16)
maxvit_t          MaxViT-Tiny  (multi-axis attention)
resnext101        ResNeXt-101 32×8d
regnet_y_16gf     RegNetY-16GF
densenet201       DenseNet-201
"""

import logging
from typing import Dict, Callable, Type

import torch
from torch import nn, Tensor
import torchvision
import torchvision.transforms as transforms

import configs

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _LayerNorm2d(nn.LayerNorm):
    """Channel-last LayerNorm for ConvNeXt-style heads."""
    def forward(self, x: Tensor) -> Tensor:
        x = x.permute(0, 2, 3, 1)
        x = torch.nn.functional.layer_norm(
            x, self.normalized_shape, self.weight, self.bias, self.eps
        )
        x = x.permute(0, 3, 1, 2)
        return x


def _make_head(in_features: int, num_bits: int) -> nn.Sequential:
    """Shared two-layer MLP head used by most backbones."""
    return nn.Sequential(
        nn.Linear(in_features, max(in_features // 2, num_bits * 4)),
        nn.GELU(),
        nn.Dropout(0.2),
        nn.Linear(max(in_features // 2, num_bits * 4), num_bits),
    )


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class BaseDecoder(nn.Module):
    """
    Common wrapper: resizes input if needed, runs backbone, applies Sigmoid.
    Subclasses set  self.backbone  and  self.out_features.
    """

    def __init__(self, config: configs.ModelConfig):
        super().__init__()
        self.config = config
        self.image_shape = config.image_shape          # (H, W)
        self.num_encoded_bits = config.num_encoded_bits
        # subclass must define self.backbone and call _build_head()

    def _build_head(self, in_features: int):
        self.head = _make_head(in_features, self.num_encoded_bits)
        self.sigmoid = nn.Sigmoid()

    def _maybe_resize(self, image: Tensor) -> Tensor:
        H, W = self.image_shape
        if image.shape[-2] != H or image.shape[-1] != W:
            logger.debug(
                "Decoder resizing image from %s to %s", image.shape[-2:], (H, W)
            )
            image = transforms.Resize((H, W))(image)
        return image

    def forward(self, image: Tensor) -> Tensor:
        image = self._maybe_resize(image)
        feat = self.backbone(image)             # (B, in_features)
        out = self.head(feat)                   # (B, num_encoded_bits)
        return self.sigmoid(out)


# ---------------------------------------------------------------------------
# ConvNeXt-Base  (original)
# ---------------------------------------------------------------------------

class ConvNeXtBaseDecoder(BaseDecoder):
    NAME = "convnext_base"

    def __init__(self, config: configs.ModelConfig):
        super().__init__(config)
        net = torchvision.models.convnext_base(weights="IMAGENET1K_V1")
        # Discover classifier input features
        in_features = None
        for name, child in net.named_children():
            if name == "classifier":
                for sub_name, sub_child in child.named_children():
                    if sub_name == "2":
                        in_features = sub_child.in_features
        net.classifier = nn.Sequential(
            _LayerNorm2d(in_features, eps=1e-6),
            nn.Flatten(1),
        )
        self.backbone = net
        self._build_head(in_features)


# ---------------------------------------------------------------------------
# ConvNeXt-Large
# ---------------------------------------------------------------------------

class ConvNeXtLargeDecoder(BaseDecoder):
    NAME = "convnext_large"

    def __init__(self, config: configs.ModelConfig):
        super().__init__(config)
        net = torchvision.models.convnext_large(weights="IMAGENET1K_V1")
        in_features = None
        for name, child in net.named_children():
            if name == "classifier":
                for sub_name, sub_child in child.named_children():
                    if sub_name == "2":
                        in_features = sub_child.in_features
        net.classifier = nn.Sequential(
            _LayerNorm2d(in_features, eps=1e-6),
            nn.Flatten(1),
        )
        self.backbone = net
        self._build_head(in_features)


# ---------------------------------------------------------------------------
# EfficientNetV2-M
# ---------------------------------------------------------------------------

class EfficientNetV2MDecoder(BaseDecoder):
    NAME = "efficientnet_v2_m"

    def __init__(self, config: configs.ModelConfig):
        super().__init__(config)
        net = torchvision.models.efficientnet_v2_m(weights="IMAGENET1K_V1")
        in_features = net.classifier[-1].in_features
        net.classifier = nn.Identity()
        self.backbone = net
        self._build_head(in_features)


# ---------------------------------------------------------------------------
# EfficientNetV2-L
# ---------------------------------------------------------------------------

class EfficientNetV2LDecoder(BaseDecoder):
    NAME = "efficientnet_v2_l"

    def __init__(self, config: configs.ModelConfig):
        super().__init__(config)
        net = torchvision.models.efficientnet_v2_l(weights="IMAGENET1K_V1")
        in_features = net.classifier[-1].in_features
        net.classifier = nn.Identity()
        self.backbone = net
        self._build_head(in_features)


# ---------------------------------------------------------------------------
# Swin Transformer-Base
# ---------------------------------------------------------------------------

class SwinBDecoder(BaseDecoder):
    NAME = "swin_b"

    def __init__(self, config: configs.ModelConfig):
        super().__init__(config)
        net = torchvision.models.swin_b(weights="IMAGENET1K_V1")
        in_features = net.head.in_features
        net.head = nn.Identity()
        self.backbone = net
        self._build_head(in_features)


# ---------------------------------------------------------------------------
# Swin Transformer V2-Base
# ---------------------------------------------------------------------------

class SwinV2BDecoder(BaseDecoder):
    NAME = "swin_v2_b"

    def __init__(self, config: configs.ModelConfig):
        super().__init__(config)
        net = torchvision.models.swin_v2_b(weights="IMAGENET1K_V1")
        in_features = net.head.in_features
        net.head = nn.Identity()
        self.backbone = net
        self._build_head(in_features)


# ---------------------------------------------------------------------------
# ViT-B/16
# ---------------------------------------------------------------------------

class ViTB16Decoder(BaseDecoder):
    NAME = "vit_b_16"

    def __init__(self, config: configs.ModelConfig):
        super().__init__(config)
        net = torchvision.models.vit_b_16(weights="IMAGENET1K_V1")
        in_features = net.heads.head.in_features
        net.heads = nn.Identity()
        self.backbone = net
        self._build_head(in_features)


# ---------------------------------------------------------------------------
# ViT-L/16
# ---------------------------------------------------------------------------

class ViTL16Decoder(BaseDecoder):
    NAME = "vit_l_16"

    def __init__(self, config: configs.ModelConfig):
        super().__init__(config)
        net = torchvision.models.vit_l_16(weights="IMAGENET1K_V1")
        in_features = net.heads.head.in_features
        net.heads = nn.Identity()
        self.backbone = net
        self._build_head(in_features)


# ---------------------------------------------------------------------------
# MaxViT-Tiny
# ---------------------------------------------------------------------------

class MaxViTTDecoder(BaseDecoder):
    NAME = "maxvit_t"

    def __init__(self, config: configs.ModelConfig):
        super().__init__(config)
        net = torchvision.models.maxvit_t(weights="IMAGENET1K_V1")
        in_features = net.classifier[-1].in_features
        net.classifier = nn.Sequential(
            net.classifier[0],   # AdaptiveAvgPool2d
            net.classifier[1],   # Flatten / squeeze
            net.classifier[2],   # Tanh
        )
        # Replace only the final linear
        in_features_actual = net.classifier[-1].in_features if hasattr(
            net.classifier[-1], "in_features"
        ) else in_features
        # Safer: strip the last Linear and use adaptive pool + flatten output
        net.classifier = nn.Sequential(*list(net.classifier.children())[:-1])
        self.backbone = net
        self._build_head(in_features)


# ---------------------------------------------------------------------------
# ResNeXt-101 32×8d
# ---------------------------------------------------------------------------

class ResNeXt101Decoder(BaseDecoder):
    NAME = "resnext101"

    def __init__(self, config: configs.ModelConfig):
        super().__init__(config)
        net = torchvision.models.resnext101_32x8d(weights="IMAGENET1K_V1")
        in_features = net.fc.in_features
        net.fc = nn.Identity()
        self.backbone = net
        self._build_head(in_features)


# ---------------------------------------------------------------------------
# RegNetY-16GF
# ---------------------------------------------------------------------------

class RegNetY16GFDecoder(BaseDecoder):
    NAME = "regnet_y_16gf"

    def __init__(self, config: configs.ModelConfig):
        super().__init__(config)
        net = torchvision.models.regnet_y_16gf(weights="IMAGENET1K_V1")
        in_features = net.fc.in_features
        net.fc = nn.Identity()
        self.backbone = net
        self._build_head(in_features)


# ---------------------------------------------------------------------------
# DenseNet-201
# ---------------------------------------------------------------------------

class DenseNet201Decoder(BaseDecoder):
    NAME = "densenet201"

    def __init__(self, config: configs.ModelConfig):
        super().__init__(config)
        net = torchvision.models.densenet201(weights="IMAGENET1K_V1")
        in_features = net.classifier.in_features
        net.classifier = nn.Identity()
        # DenseNet outputs (B, C, H, W) after features, needs pooling
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self._features = net.features
        self.backbone = nn.Sequential(
            net.features,
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(1),
        )
        self._build_head(in_features)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

DECODER_REGISTRY: Dict[str, Type[BaseDecoder]] = {
    cls.NAME: cls
    for cls in [
        ConvNeXtBaseDecoder,
        ConvNeXtLargeDecoder,
        EfficientNetV2MDecoder,
        EfficientNetV2LDecoder,
        SwinBDecoder,
        SwinV2BDecoder,
        ViTB16Decoder,
        ViTL16Decoder,
        MaxViTTDecoder,
        ResNeXt101Decoder,
        RegNetY16GFDecoder,
        DenseNet201Decoder,
    ]
}


def build_decoder(name: str, config: configs.ModelConfig) -> BaseDecoder:
    """
    Instantiate a decoder by registry name.

    Args:
        name:   One of the keys in DECODER_REGISTRY.
        config: ModelConfig instance.

    Returns:
        An instantiated BaseDecoder subclass.

    Raises:
        ValueError: If ``name`` is not in the registry.
    """
    if name not in DECODER_REGISTRY:
        available = ", ".join(sorted(DECODER_REGISTRY.keys()))
        raise ValueError(
            f"Unknown decoder '{name}'. Available decoders: {available}"
        )
    logger.info("Building decoder: %s", name)
    return DECODER_REGISTRY[name](config)


def list_decoders():
    """Print all registered decoder names."""
    print("Registered decoders:")
    for name, cls in sorted(DECODER_REGISTRY.items()):
        print(f"  {name:<22}  ({cls.__module__}.{cls.__name__})")
