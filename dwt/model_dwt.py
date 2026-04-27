import logging

import configs
import utils
import torch
from torch import nn, Tensor
import torchvision
from torch.nn import functional as thf
import torchvision.transforms as transforms
import bchlib
from torchvision.models import (
    EfficientNet_B0_Weights,
    ResNet50_Weights,
    Swin_S_Weights,
    convnext_base,
    efficientnet_b0,
    resnet50,
    swin_b,
    swin_s,
    swin_t,
)
from torchvision.models.feature_extraction import create_feature_extractor
from torchvision.models import Swin_T_Weights
from Swin_Unet.swin_transformer_unet_skip_expand_decoder_sys import SwinTransformerSys
from convnext_unet import ConvNeXtUnet
logger = logging.getLogger(__name__)


class LayerNorm2d(nn.LayerNorm):
    def forward(self, x: Tensor) -> Tensor:
        x = x.permute(0, 2, 3, 1)
        x = thf.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        x = x.permute(0, 3, 1, 2)
        return x


class ImageViewLayer(nn.Module):
    def __init__(self, hidden_dim=16, channel=3):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.channel = channel 

    def forward(self, x):
        return x.view(-1, self.channel, self.hidden_dim, self.hidden_dim)


class ImageRepeatLayer(nn.Module):
    def __init__(self, num_repeats):
        super().__init__()
        self.num_repeats = num_repeats

    def forward(self, x):
        return x.repeat(1, 1, self.num_repeats, self.num_repeats)


class Watermark2Image(nn.Module):
    def __init__(self, watermark_len, resolution=256, hidden_dim=16, num_repeats=2, channel=3):
        super().__init__()
        assert resolution % hidden_dim == 0, "Resolution should be divisible by hidden_dim"
        pad_length = resolution // 4
        self.transform = nn.Sequential(
            nn.Linear(watermark_len, hidden_dim*hidden_dim*channel),
            ImageViewLayer(hidden_dim),
            nn.Upsample(scale_factor=(resolution//hidden_dim//num_repeats//2, resolution//hidden_dim//num_repeats//2)),
            ImageRepeatLayer(num_repeats),
            transforms.Pad(pad_length),
            nn.Tanh()
        )

    def forward(self, x):
        return self.transform(x)


class Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True, activ='relu', norm=None):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        if activ == 'relu':
            self.activ = nn.ReLU(inplace=True)
        elif activ == 'silu':
            self.activ = nn.SiLU(inplace=True)
        elif activ == 'tanh':
            self.activ = nn.Tanh()
        elif activ == 'leaky_relu':
            self.activ =  nn.LeakyReLU(0.2, inplace=True)
        else:
            self.activ = None

        norm_dim = out_channels
        if norm == 'bn':
            self.norm = nn.BatchNorm2d(norm_dim)
        else:
            self.norm = None
    
    def forward(self, x):
        x = self.conv(x)
        if self.norm:
            x = self.norm(x)
        if self.activ:
            x = self.activ(x)
        return x


class DecBlock(nn.Module):
    def __init__(self, in_channels, skip_channels='default', out_channels='default', activ='relu', norm=None):
        super().__init__()
        if skip_channels == 'default':
            skip_channels = in_channels//2
        if out_channels == 'default':
            out_channels = in_channels//2
        self.up = nn.Upsample(scale_factor=(2,2))
        self.pad = nn.ZeroPad2d((0, 1, 0, 1))
        self.conv1 = Conv2d(in_channels, out_channels, 2, 1, 0, activ=activ, norm=norm)
        self.conv2 = Conv2d(out_channels + skip_channels, out_channels, 3, 1, 1, activ=activ, norm=norm)
    
    def forward(self, x, skip):
        x = self.conv1(self.pad(self.up(x)))
        x = torch.cat([x, skip], dim=1)
        x = self.conv2(x)
        return x
    

class Encoder(nn.Module):
    def __init__(self, config: configs.ModelConfig):
        super().__init__()
        self.config = config
        # self.watermark2image = Watermark2Image(config.num_encoded_bits, config.image_shape[0], 
        #                                                 config.watermark_hidden_dim, num_repeats=self.config.num_repeats)
        # input_channel: 3 from image + 3 from watermark
        dwt_channels = 21   # 7 subbands × 3 RGB
        wm_channels  = 3    # watermark projected to 3ch
        in_ch = dwt_channels + wm_channels  # 24
        
        self.watermark2image = Watermark2Image(
                config.num_encoded_bits,
                resolution=config.image_shape[0] // 4,   # ← was image_shape[0]
                hidden_dim=config.watermark_hidden_dim,
                num_repeats=self.config.num_repeats
        )
        self.pre = Conv2d(in_ch, config.num_initial_channels, 3, 1, 1)
        # All encoder/decoder U-Net layers are IDENTICAL to original Encoder
        # (copy them verbatim — channel progression is internal)
        self.enc = nn.ModuleList()
        input_channel = config.num_initial_channels
        for _ in range(config.num_down_levels):
            self.enc.append(Conv2d(input_channel, input_channel * 2, 3, 2, 1))
            input_channel *= 2

        self.dec = nn.ModuleList()
        for i in range(config.num_down_levels):
            skip_ch = input_channel // 2 if i < config.num_down_levels - 1 \
                      else input_channel // 2 + in_ch
            self.dec.append(DecBlock(input_channel, skip_ch, activ='relu', norm='none'))
            input_channel //= 2

        # Output: residuals for the 6 embeddable subband groups (HF only)
        # LL2 is NOT modified → output 18 channels (6 subbands × 3 RGB)
        # Plus pass-through for LL2 (3ch) → total output 21ch
        self.post = nn.Sequential(
            Conv2d(input_channel, input_channel, 3, 1, 1, activ='None'),
            Conv2d(input_channel, input_channel // 2, 1, 1, 0, activ='silu'),
            Conv2d(input_channel // 2, 18, 1, 1, 0, activ='tanh')  # 18ch residuals
        )

    def forward(self, image: torch.Tensor, watermark=None, alpha=1.0):
        if watermark == None:
            logger.info("Watermark is not provided. Use zero bits as test watermark.")
            watermark = torch.zeros(image.shape[0], self.config.num_encoded_bits, device = image.device)
        watermark_image = self.watermark2image(watermark)
        inputs = torch.cat((image, watermark_image), dim=1)

        enc = []
        x = self.pre(inputs)
        for layer in self.enc:
            enc.append(x)
            x = layer(x)
        
        enc = enc[::-1]
        for i, (layer, skip) in enumerate(zip(self.dec, enc)):
            if i < self.config.num_down_levels - 1:
                x = layer(x, skip)
            else:
                x = layer(x, torch.cat([skip, inputs], dim=1))
        residuals = self.post(x)
        # Expand 3ch watermark prior to 18ch so it can be blended with residual channels.
        watermark_prior = watermark_image.repeat(1, 6, 1, 1)
        # print(alpha)
        return residuals * (1 - alpha) + alpha * watermark_prior
        # return self.post(x) 

class EncoderSwin(nn.Module):
    """
    Swin Transformer encoder for DWT watermarking.
    Works with 21-channel DWT subbands + 3-channel watermark image.
    """
    def __init__(self, config: configs.ModelConfig):
        super().__init__()
        self.config = config
        dwt_channels = 21
        wm_channels = 3
        in_ch = dwt_channels + wm_channels  # 24
        
        self.watermark2image = Watermark2Image(
            config.num_encoded_bits,
            resolution=config.image_shape[0] // 4,
            hidden_dim=config.watermark_hidden_dim,
            num_repeats=self.config.num_repeats
        )
        
        # Stem: 24 -> 3 channels for Swin compatibility
        self.stem = nn.Sequential(
            Conv2d(in_ch, 16, 3, 1, 1, activ='relu'),
            Conv2d(16, 8, 3, 1, 1, activ='relu'),
            Conv2d(8, 3, 1, 1, 0, activ='None')
        )
        
        # Swin-T backbone with feature extraction
        backbone = swin_t(weights=Swin_T_Weights.IMAGENET1K_V1)
        self.encoder = create_feature_extractor(
            backbone,
            return_nodes={
                "features.1": "f1",
                "features.3": "f2",
                "features.5": "f3",
                "features.7": "f4",
            }
        )
        
        # Decoder with upsampling
        self.decoder_up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.decoder_fuse1 = Conv2d(768 + 384, 384, 1, 1, 0, activ='relu')
        
        self.decoder_up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.decoder_fuse2 = Conv2d(384 + 192, 192, 1, 1, 0, activ='relu')
        
        self.decoder_up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.decoder_fuse3 = Conv2d(192 + 96, 96, 1, 1, 0, activ='relu')
        
        self.decoder_up4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)
        self.decoder_fuse4 = Conv2d(96, 48, 1, 1, 0, activ='relu')
        
        # Output: 18-channel residuals
        self.post = nn.Sequential(
            Conv2d(48, 24, 3, 1, 1, activ='relu'),
            Conv2d(24, 18, 1, 1, 0, activ='tanh')
        )
    
    def _to_nchw(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 4:
            x = x.permute(0, 3, 1, 2).contiguous()
        return x
    
    def forward(self, image: torch.Tensor, watermark=None, alpha=1.0):
        if watermark is None:
            logger.info("Watermark is not provided. Use zero bits as test watermark.")
            watermark = torch.zeros(image.shape[0], self.config.num_encoded_bits, device=image.device)
        
        watermark_image = self.watermark2image(watermark)
        inputs = torch.cat((image, watermark_image), dim=1)
        
        # 24 -> 3 channels
        x = self.stem(inputs)
        
        # Extract multiscale features
        feats = self.encoder(x)
        f1 = self._to_nchw(feats["f1"])
        f2 = self._to_nchw(feats["f2"])
        f3 = self._to_nchw(feats["f3"])
        f4 = self._to_nchw(feats["f4"])
        
        # Decoder
        x = self.decoder_up1(f4)
        x = torch.cat([x, f3], dim=1)
        x = self.decoder_fuse1(x)
        
        x = self.decoder_up2(x)
        x = torch.cat([x, f2], dim=1)
        x = self.decoder_fuse2(x)
        
        x = self.decoder_up3(x)
        x = torch.cat([x, f1], dim=1)
        x = self.decoder_fuse3(x)
        
        x = self.decoder_up4(x)
        x = self.decoder_fuse4(x)
        
        residuals = self.post(x)
        watermark_prior = watermark_image.repeat(1, 6, 1, 1)
        return residuals * (1 - alpha) + alpha * watermark_prior


class EncoderSwinUNet(nn.Module):
 
    def __init__(self, config: configs.ModelConfig):
        super().__init__()
        self.config = config
        self.debug_shapes = True
        self._debug_printed_once = False
        dwt_channels = 21 #+ 3 # without image

        wm_channels = 3
        in_ch = dwt_channels + wm_channels  # 24
        
        self.watermark2image = Watermark2Image(
            config.num_encoded_bits,
            resolution=config.image_shape[0] // 4,
            hidden_dim=config.watermark_hidden_dim,
            num_repeats=self.config.num_repeats
        )
        
        # Stem: 24 -> 3 channels for Swin compatibility
        self.stem = nn.Sequential(
            Conv2d(in_ch, 16, 3, 1, 1, activ='relu'),
            Conv2d(16, 12, 3, 1, 1, activ='relu'),
            Conv2d(12, 3, 1, 1, 0, activ='None')
        )
        
        # Swin U-Net model for residual prediction

        self.swin_unet = SwinTransformerSys(
            img_size=config.image_shape[0] // 4,  # DWT downscales by 4
            patch_size=4,
            in_chans=3,
            num_classes=18,  # 18 residual channels (6 subbands × 3 RGB)
            embed_dim=96,
            depths=[2, 2, 18, 2],
            depths_decoder=[1, 2, 2, 2],
            num_heads=[3, 6, 12, 24],
            window_size=8,
            mlp_ratio=4.,
            qkv_bias=True,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=0.1,
            norm_layer=nn.LayerNorm,
            ape=False,
            patch_norm=True,
            use_checkpoint=False,
            final_upsample="expand_first"
        )
        print("Initialized Swin U-Net with Swin-T backbone for DWT watermarking.")

    def forward(self, image: torch.Tensor, watermark=None, alpha=1.0, org_image=None):
        """
        image: [B, 21, H/4, W/4] - DWT subbands
        watermark: [B, num_bits]
        returns: [B, 18, H/4, W/4] - residual predictions for HF subbands
        """
        if watermark == None:
            logger.info("Watermark is not provided. Use zero bits as test watermark.")
            watermark = torch.zeros(image.shape[0], self.config.num_encoded_bits, device=image.device)

        if self.debug_shapes and not self._debug_printed_once:
            print("[DWT-SWIN-UNET DEBUG] image shape:", tuple(image.shape))
            print("[DWT-SWIN-UNET DEBUG] watermark bits shape:", tuple(watermark.shape))
            print("[DWT-SWIN-UNET DEBUG] expected DWT spatial:", self.config.image_shape[0] // 4, self.config.image_shape[1] // 4)
        
        watermark_image = self.watermark2image(watermark)
        if self.debug_shapes and not self._debug_printed_once:
            print("[DWT-SWIN-UNET DEBUG] watermark_image shape:", tuple(watermark_image.shape))
        inputs = torch.cat((image, watermark_image), dim=1)  # [B, 24, H/4, W/4]
        
        # inputs = torch.cat((inputs, org_image), dim=1) # without image
        
        if self.debug_shapes and not self._debug_printed_once:
            print("[DWT-SWIN-UNET DEBUG] concat inputs shape:", tuple(inputs.shape))
        
        # 24 -> 3
        x = self.stem(inputs)  # [B, 3, H/4, W/4]
        if self.debug_shapes and not self._debug_printed_once:
            print("[DWT-SWIN-UNET DEBUG] stem output shape:", tuple(x.shape))
            print("[DWT-SWIN-UNET DEBUG] swin_unet expected img_size:", self.config.image_shape[0] // 4)
        
        # Swin U-Net processes and outputs residuals directly
        try:
            residuals = self.swin_unet(x)  # [B, 18, H/4, W/4]
        except Exception as ex:
            print("[DWT-SWIN-UNET DEBUG] swin_unet forward failed")
            print("[DWT-SWIN-UNET DEBUG] x fed to swin_unet:", tuple(x.shape))
            print("[DWT-SWIN-UNET DEBUG] configured img_size:", self.config.image_shape[0] // 4)
            raise ex
        if self.debug_shapes and not self._debug_printed_once:
            print("[DWT-SWIN-UNET DEBUG] residuals shape:", tuple(residuals.shape))
        
        # Expand 3ch watermark prior to 18ch for blending
        watermark_prior = watermark_image.repeat(1, 6, 1, 1)
        if self.debug_shapes and not self._debug_printed_once:
            print("[DWT-SWIN-UNET DEBUG] watermark_prior shape:", tuple(watermark_prior.shape))
            print("[DWT-SWIN-UNET DEBUG] final output shape:", tuple((residuals * (1 - alpha) + alpha * watermark_prior).shape))
            self._debug_printed_once = True
        return residuals * (1 - alpha) + alpha * watermark_prior


class EncoderConvNeXT(nn.Module):
    """
    ConvNeXT encoder for DWT watermarking.
    Works with 21-channel DWT subbands + 3-channel watermark image.
    """

    def __init__(self, config: configs.ModelConfig):
        super().__init__()
        self.config = config
        dwt_channels = 21
        wm_channels = 3
        in_ch = dwt_channels + wm_channels  # 24

        self.watermark2image = Watermark2Image(
            config.num_encoded_bits,
            resolution=config.image_shape[0] // 4,
            hidden_dim=config.watermark_hidden_dim,
            num_repeats=self.config.num_repeats
        )

        # 24 -> 3 so ConvNeXtUnet gets standard RGB-like tensors.
        self.stem = nn.Sequential(
            Conv2d(in_ch, 16, 3, 1, 1, activ='relu'),
            Conv2d(16, 12, 3, 1, 1, activ='relu'),
            Conv2d(12, 3, 1, 1, 0, activ='None')
        )

        # Match the other DWT encoders: predict 18 HF residual channels.
        self.convnext_unet = ConvNeXtUnet(
            num_classes=18,
            encoder_name='convnext_base',
            activation=None,
            pretrained=True,
            in_22k=False
        )



    def forward(self, image: torch.Tensor, watermark=None, alpha=1.0):
        if watermark is None:
            logger.info("Watermark is not provided. Use zero bits as test watermark.")
            watermark = torch.zeros(image.shape[0], self.config.num_encoded_bits, device=image.device)

       
        watermark_image = self.watermark2image(watermark)
        
        inputs = torch.cat((image, watermark_image), dim=1)
        

        x = self.stem(inputs)
       
        residuals = torch.tanh(self.convnext_unet(x))
        
        watermark_prior = watermark_image.repeat(1, 6, 1, 1)
         
        return  residuals * (1 - alpha) + alpha * watermark_prior


class EncoderResNet50(nn.Module):
    """
    ResNet50 encoder for DWT watermarking.
    Uses a pretrained ResNet50 backbone with a lightweight decoder to predict
    the 18 residual channels for the embeddable HF wavelet bands.
    """

    def __init__(self, config: configs.ModelConfig):
        super().__init__()
        self.config = config
        dwt_channels = 21
        wm_channels = 3
        in_ch = dwt_channels + wm_channels

        self.watermark2image = Watermark2Image(
            config.num_encoded_bits,
            resolution=config.image_shape[0] // 4,
            hidden_dim=config.watermark_hidden_dim,
            num_repeats=self.config.num_repeats,
        )

        self.stem = nn.Sequential(
            Conv2d(in_ch, 24, 3, 1, 1, activ='relu'),
            Conv2d(24, 3, 1, 1, 0, activ='None'),
        )

        backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        self.encoder = create_feature_extractor(
            backbone,
            return_nodes={
                "relu": "f0",
                "layer1": "f1",
                "layer2": "f2",
                "layer3": "f3",
                "layer4": "f4",
            },
        )

        self.decoder_up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.decoder_fuse1 = Conv2d(2048 + 1024, 512, 1, 1, 0, activ='relu')

        self.decoder_up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.decoder_fuse2 = Conv2d(512 + 512, 256, 1, 1, 0, activ='relu')

        self.decoder_up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.decoder_fuse3 = Conv2d(256 + 256, 128, 1, 1, 0, activ='relu')

        self.decoder_up4 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.decoder_fuse4 = Conv2d(128 + 64, 64, 1, 1, 0, activ='relu')

        self.decoder_up5 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.decoder_fuse5 = Conv2d(64, 32, 1, 1, 0, activ='relu')

        self.post = nn.Sequential(
            Conv2d(32, 16, 3, 1, 1, activ='relu'),
            Conv2d(16, 18, 1, 1, 0, activ='tanh'),
        )

    def forward(self, image: torch.Tensor, watermark=None, alpha=1.0):
        if watermark is None:
            logger.info("Watermark is not provided. Use zero bits as test watermark.")
            watermark = torch.zeros(image.shape[0], self.config.num_encoded_bits, device=image.device)

        watermark_image = self.watermark2image(watermark)
        inputs = torch.cat((image, watermark_image), dim=1)
        x = self.stem(inputs)

        feats = self.encoder(x)
        f0 = feats["f0"]
        f1 = feats["f1"]
        f2 = feats["f2"]
        f3 = feats["f3"]
        f4 = feats["f4"]

        x = self.decoder_up1(f4)
        x = torch.cat([x, f3], dim=1)
        x = self.decoder_fuse1(x)

        x = self.decoder_up2(x)
        x = torch.cat([x, f2], dim=1)
        x = self.decoder_fuse2(x)

        x = self.decoder_up3(x)
        x = torch.cat([x, f1], dim=1)
        x = self.decoder_fuse3(x)

        x = self.decoder_up4(x)
        x = torch.cat([x, f0], dim=1)
        x = self.decoder_fuse4(x)

        x = self.decoder_up5(x)
        x = self.decoder_fuse5(x)

        residuals = self.post(x)
        watermark_prior = watermark_image.repeat(1, 6, 1, 1)
        return residuals * (1 - alpha) + alpha * watermark_prior


class EncoderEfficientNetB0(nn.Module):
    """
    EfficientNet-B0 encoder for DWT watermarking.
    Uses a pretrained EfficientNet-B0 backbone with a lightweight decoder to
    predict the 18 residual channels for the embeddable HF wavelet bands.
    """

    def __init__(self, config: configs.ModelConfig):
        super().__init__()
        self.config = config
        dwt_channels = 21
        wm_channels = 3
        in_ch = dwt_channels + wm_channels

        self.watermark2image = Watermark2Image(
            config.num_encoded_bits,
            resolution=config.image_shape[0] // 4,
            hidden_dim=config.watermark_hidden_dim,
            num_repeats=self.config.num_repeats,
        )

        self.stem = nn.Sequential(
            Conv2d(in_ch, 24, 3, 1, 1, activ='relu'),
            Conv2d(24, 3, 1, 1, 0, activ='None'),
        )

        backbone = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        self.encoder = create_feature_extractor(
            backbone,
            return_nodes={
                "features.0": "f0",
                "features.2": "f1",
                "features.3": "f2",
                "features.5": "f3",
                "features.8": "f4",
            },
        )

        self.decoder_up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.decoder_fuse1 = Conv2d(1280 + 112, 256, 1, 1, 0, activ='relu')

        self.decoder_up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.decoder_fuse2 = Conv2d(256 + 40, 128, 1, 1, 0, activ='relu')

        self.decoder_up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.decoder_fuse3 = Conv2d(128 + 24, 64, 1, 1, 0, activ='relu')

        self.decoder_up4 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.decoder_fuse4 = Conv2d(64 + 32, 32, 1, 1, 0, activ='relu')

        self.decoder_up5 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.decoder_fuse5 = Conv2d(32, 16, 1, 1, 0, activ='relu')

        self.post = nn.Sequential(
            Conv2d(16, 16, 3, 1, 1, activ='relu'),
            Conv2d(16, 18, 1, 1, 0, activ='tanh'),
        )

    def forward(self, image: torch.Tensor, watermark=None, alpha=1.0):
        if watermark is None:
            logger.info("Watermark is not provided. Use zero bits as test watermark.")
            watermark = torch.zeros(image.shape[0], self.config.num_encoded_bits, device=image.device)

        watermark_image = self.watermark2image(watermark)
        inputs = torch.cat((image, watermark_image), dim=1)
        x = self.stem(inputs)

        feats = self.encoder(x)
        f0 = feats["f0"]
        f1 = feats["f1"]
        f2 = feats["f2"]
        f3 = feats["f3"]
        f4 = feats["f4"]

        x = self.decoder_up1(f4)
        x = torch.cat([x, f3], dim=1)
        x = self.decoder_fuse1(x)

        x = self.decoder_up2(x)
        x = torch.cat([x, f2], dim=1)
        x = self.decoder_fuse2(x)

        x = self.decoder_up3(x)
        x = torch.cat([x, f1], dim=1)
        x = self.decoder_fuse3(x)

        x = self.decoder_up4(x)
        x = torch.cat([x, f0], dim=1)
        x = self.decoder_fuse4(x)

        x = self.decoder_up5(x)
        x = self.decoder_fuse5(x)

        residuals = self.post(x)
        watermark_prior = watermark_image.repeat(1, 6, 1, 1)
        return residuals * (1 - alpha) + alpha * watermark_prior

class DisResNet(nn.Module):
    def __init__(self, config: configs.ModelConfig):
        super().__init__()
        self.config = config
        self.extractor = torchvision.models.resnet18(weights='ResNet18_Weights.IMAGENET1K_V1')
        self.extractor.fc = nn.Linear(self.extractor.fc.in_features, config.num_classes - 1)
        self.main = nn.Sequential(
            self.extractor,
            nn.Sigmoid()
        )

    def forward(self, image: torch.Tensor):
        if image.shape[-1] != self.config.image_shape[-1] or image.shape[-2] != self.config.image_shape[-2]:
            logger.debug(f"Image shape should be {self.config.image_shape} but got {image.shape}")
            image = transforms.Resize(self.config.image_shape)(image)
        return self.main(image)


class Extractor(nn.Module):
    # def __init__(self, config: configs.ModelConfig):
    #     super().__init__()
    #     self.config = config

    #     self.extractor = torchvision.models.convnext_base(weights='IMAGENET1K_V1')
    #     n_inputs = None
    #     for name, child in self.extractor.named_children():
    #         if name == 'classifier':
    #             for sub_name, sub_child in child.named_children():
    #                 if sub_name == '2':
    #                     n_inputs = sub_child.in_features
    
    #     self.extractor.classifier = nn.Sequential(
    #                 LayerNorm2d(n_inputs, eps=1e-6),
    #                 nn.Flatten(1),
    #                 nn.Linear(in_features=n_inputs, out_features=config.num_encoded_bits),
    #             )

    #     self.main = nn.Sequential(
    #         self.extractor,
    #         nn.Sigmoid()
    #     )

    # def forward(self, image: torch.Tensor):
    #     if image.shape[-1] != self.config.image_shape[-1] or image.shape[-2] != self.config.image_shape[-2]:
    #         logger.debug(f"Image shape should be {self.config.image_shape} but got {image.shape}")
    #         image = transforms.Resize(self.config.image_shape)(image)
    #     return self.main(image)
    def __init__(self, config):
        super().__init__()
        self.config = config
        dwt_channels = 21 #+ 3  # 7 subbands × 3 RGB

        # Load pretrained ConvNeXT-B
        backbone = torchvision.models.convnext_base(weights='IMAGENET1K_V1')

        # ── Pretrained weight inflation ──────────────────────────────────────
        # ConvNeXT-B stem: features[0][0] is a Conv2d(3, 128, kernel=4, stride=4)
        # Weight shape: (128, 3, 4, 4)
        # We need: (128, 21, 4, 4) — inflate by repeating 3ch weights 7 times
        # and scaling by 1/7 to preserve output magnitude
        orig_weight = backbone.features[0][0].weight.data   # (128, 3, 4, 4)
        new_weight = orig_weight.repeat(1, 7, 1, 1) / 7.0  # (128, 21, 4, 4)
        
        new_conv = nn.Conv2d(dwt_channels, 128, kernel_size=4, stride=4, bias=False)
        new_conv.weight = nn.Parameter(new_weight)
        backbone.features[0][0] = new_conv
        # ── End inflation ────────────────────────────────────────────────────

        # Replace classifier head (same as original Extractor)
        n_inputs = 1024  # ConvNeXT-B final feature dim
        backbone.classifier = nn.Sequential(
            LayerNorm2d(n_inputs, eps=1e-6),
            nn.Flatten(1),
            nn.Linear(n_inputs, config.num_encoded_bits),
        )
        self.main = nn.Sequential(backbone, nn.Sigmoid())

    def forward(self, subbands: torch.Tensor, image=None):
        """
        subbands: (B, 21, H/4, W/4) — output of HaarDWT2D
        ConvNeXT stem stride-4 brings this to (B, 128, H/16, W/16)
        which for H=256 gives 4×4 feature maps — OK for classification.
        """
        # print("Extractor input image shape:", image.shape)
        # print("Extractor input subbands shape:", subbands.shape)
        # x = torch.cat([subbands,image], dim=1)  # Add 3 zero channels for watermark prior
        x= subbands
        return self.main(x)


class ExtractorSwin(nn.Module):
    """
    Swin Transformer extractor for DWT watermarking.
    Extracts watermark bits from 21-channel DWT subbands.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        dwt_channels = 21
        
        # Stem: 21 -> 3 channels for Swin compatibility
        self.stem = nn.Sequential(
            Conv2d(dwt_channels, 16, 3, 1, 1, activ='relu'),
            Conv2d(16, 8, 3, 1, 1, activ='relu'),
            Conv2d(8, 3, 1, 1, 0, activ='None')
        )
        
        # Swin-T backbone
        # backbone = swin_t(weights=Swin_T_Weights.IMAGENET1K_V1)
        backbone=swin_s(weights=Swin_S_Weights.IMAGENET1K_V1)

        
        # Replace classifier
        n_inputs = 768  # Swin-T final feature dim
        backbone.head = nn.Sequential(
            nn.LayerNorm(n_inputs),
            nn.Linear(n_inputs, config.num_encoded_bits)
        )
        
        self.encoder = backbone
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, subbands: torch.Tensor):
        """
        subbands: (B, 21, H/4, W/4) — output of HaarDWT2D
        """
        x = self.stem(subbands)
        x = self.encoder(x)
        return self.sigmoid(x)


class ExtractorConvNeXT(nn.Module):
    """
    ConvNeXT extractor for DWT watermarking.
    Extracts watermark bits from 21-channel DWT subbands.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        dwt_channels = 21
        
        # Load pretrained ConvNeXT-B
        backbone = torchvision.models.convnext_base(weights='IMAGENET1K_V1')
        
        # Inflate input layer from 3 to 21 channels
        orig_weight = backbone.features[0][0].weight.data
        new_weight = orig_weight.repeat(1, 7, 1, 1) / 7.0
        
        new_conv = nn.Conv2d(21, 128, kernel_size=4, stride=4, bias=False)
        new_conv.weight = nn.Parameter(new_weight)
        backbone.features[0][0] = new_conv
        
        # Replace classifier head
        n_inputs = 1024  # ConvNeXT-B final feature dim
        backbone.classifier = nn.Sequential(
            LayerNorm2d(n_inputs, eps=1e-6),
            nn.Flatten(1),
            nn.Linear(n_inputs, config.num_encoded_bits),
        )
        
        self.main = nn.Sequential(backbone, nn.Sigmoid())
    
    def forward(self, subbands: torch.Tensor):
        """
        subbands: (B, 21, H/4, W/4) — output of HaarDWT2D
        """
        return self.main(subbands)


class BCHECC:

    def __init__(self, t, m):
        self.t = t # number of errors to be corrected
        self.m = m # total of bits n is 2^m
        self.bch = bchlib.BCH(t, m=m)
        self.data_bytes = (self.bch.n + 7) // 8 - self.bch.ecc_bytes

    def batch_encode(self, batch_size):
        secrets = []
        uuid_bytes = utils.uuid_to_bytes(batch_size)
        for input in uuid_bytes:
            ecc = self.bch.encode(input)
            secrets += [torch.Tensor([int(i) for i in ''.join(format(x, '08b') for x in input + ecc)])]
            assert len(secrets[-1]) == 2**self.m, f"Encoded secret bits length should be {2**self.m}"
        return torch.vstack(secrets).type(torch.float32)

    def batch_decode_ecc(self, secrets: torch.Tensor, threshold: float = 0.5):
        res = []
        for i in range(len(secrets)):
            packet = self._bch_correct(secrets[i], threshold)
            data_bits = [int(k) for k in ''.join(format(x, '08b') for x in packet)]
            res.append(torch.Tensor(data_bits).type(torch.float32))
        return  torch.vstack(res)

    def encode_str(self, input: str):
        assert len(input) == self.data_bytes, f"Input str length should be {self.data_bytes}"
        input_bytes = bytearray(input, 'utf-8')
        ecc = self.bch.encode(input_bytes)
        packet = input_bytes + ecc
        secret = [int(i) for i in ''.join(format(x, '08b') for x in packet)]
        assert len(secret) == 2**self.m, f"Encoded secret bits length should be {2**self.m}"
        return torch.Tensor(secret).type(torch.float32).unsqueeze(0)

    def decode_str(self, secrets: torch.Tensor, threshold: float = 0.5):
        n_errs, res = [], []
        for i in range(len(secrets)):
            bit_string = ''.join(str(int(k >= threshold)) for k in secrets[i])
            packet = self._bitstring_to_bytes(bit_string)
            data, ecc = packet[:-self.bch.ecc_bytes], packet[-self.bch.ecc_bytes:]
            n_err = self.bch.decode(data, ecc)
            if n_err < 0: 
                n_errs.append(n_err)
                res.append([])
                continue
            self.bch.correct(data, ecc)
            packet = data + ecc
            try:
                n_errs.append(n_err)
                res.append(packet[:-self.bch.ecc_bytes].decode('utf-8'))
            except:
                n_errs.append(-1)
                res.append([])
        return n_errs, res

    def _bch_correct(self, secret: torch.Tensor, threshold: float = 0.5):
        bitstring = ''.join(str(int(x >= threshold)) for x in secret)
        packet = self._bitstring_to_bytes(bitstring)
        data, ecc = packet[:-self.bch.ecc_bytes], packet[-self.bch.ecc_bytes:]
        n_err = self.bch.decode(data, ecc)
        if n_err < 0:
            logger.info("n_err < 0. Cannot accurately decode the message.")
            return packet
        self.bch.correct(data, ecc)
        return bytes(data  + ecc)

    def _decode_data_bits(self, secrets: torch.Tensor, threshold: float = 0.5):
        return self.batch_decode_ecc(secrets, threshold)[:, :-self.bch.ecc_bytes*8]
        
    def _bitstring_to_bytes(self, s):
        return bytearray(int(s, 2).to_bytes((len(s) + 7) // 8, byteorder='big'))