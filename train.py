import logging
import sys
import os
import time
from collections import defaultdict
from itertools import chain
from torchvision.models import resnet18, convnext_base,swin_t, swin_b, swin_s,Swin_T_Weights

import model
import noise
import metrics
import utils
import uuid
import re
from attack.attacks import rotate_tensor
try:
    from metrics_challange import fidelity
except Exception as ex:
    fidelity = None
    _FIDELITY_IMPORT_ERROR = ex
else:
    _FIDELITY_IMPORT_ERROR = None

try:
    from metrics_challange import robustness
except Exception as ex:
    robustness = None
    _ROBUSTNESS_IMPORT_ERROR = ex
else:
    _ROBUSTNESS_IMPORT_ERROR = None

import numpy as np
from focal_frequency_loss import FocalFrequencyLoss as ffl
import lpips
import dct as dct_module
from dwt import dwt as dwt_module
from dwt import model_dwt
from PIL import Image, ImageFile
import torch
from torch import nn
from torch.nn import functional as thf
from torch.utils.tensorboard import SummaryWriter
import torchvision.utils as vutils
import torchvision.transforms as transforms

# Necessary for avoiding reading truncated images from dataloader.
ImageFile.LOAD_TRUNCATED_IMAGES = True
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger(__name__)


class Watermark(nn.Module):
    def __init__(self, config, device="cuda:0", wandb_options=None):
        super().__init__()

        self.config = config
        self.device = device
        self.use_dct = bool(getattr(self.config, "use_dct", False))
        self.use_dwt = bool(getattr(self.config, "use_dwt", False))
        self.use_dct_dwt = bool(getattr(self.config, "use_dct_dwt", False))

        if self.use_dct_dwt:
            self.use_dct = True
            self.use_dwt = True

        if self.use_dwt and self.use_dct and not self.use_dct_dwt:
            raise ValueError("Use --use_dct_dwt to enable both DCT and DWT together.")

        self.encoder_dwt = None
        self.decoder_dwt = None
        print("Initializing DWT encoder and decoder")
        print(self.use_dwt ,config.dwt_encoder_arch)
        print("#"*100)
        if self.use_dwt and not self.use_dct_dwt:
            if config.dwt_encoder_arch == "unet":
                self.encoder = model_dwt.Encoder(config).to(device)
            elif config.dwt_encoder_arch == "swin":
                print("Using SwinUNet as DWT encoder")
                self.encoder = model_dwt.EncoderSwinUNet(config).to(device)
            elif config.dwt_encoder_arch == "convnext":
                print("Using ConvNeXT as DWT encoder")
                self.encoder = model_dwt.EncoderConvNeXT(config).to(device)

            elif config.dwt_encoder_arch == "resnet50":
                print("Using ResNet50 as DWT encoder")
                self.encoder = model_dwt.EncoderResNet50(config).to(device)
            elif config.dwt_encoder_arch == "efficientnet":
                print("Using EfficientNet-B0 as DWT encoder")
                self.encoder = model_dwt.EncoderEfficientNet(config).to(device)

            self.decoder = model_dwt.Extractor(config).to(device)
        elif self.use_dct_dwt:
            self.encoder = model.Encoder(config).to(device)
            self.decoder = model.Extractor(config).to(device)
            self.encoder_dwt = model_dwt.Encoder(config).to(device)
            
            self.decoder_dwt = model_dwt.Extractor(config).to(device)
        else:
            self.encoder = model.Encoder(config).to(device)
            self.decoder = model.Extractor(config).to(device)

        if self.use_dct_dwt:
            logger.info(
                "Hybrid branch setup: DCT encoder/decoder=%s/%s, DWT encoder/decoder=%s/%s",
                self.encoder.__class__.__name__,
                self.decoder.__class__.__name__,
                self.encoder_dwt.__class__.__name__,
                self.decoder_dwt.__class__.__name__,
            )
        else:
            logger.info(
                "Single branch setup: encoder/decoder=%s/%s",
                self.encoder.__class__.__name__,
                self.decoder.__class__.__name__,
            )

        self.discriminator = model.DisResNet(config).to(device)
        self.alpha = 1.0

        self.noiser = noise.init_shared_noiser(
            num_transforms=config.num_geo_noises_per_step,
            device=str(device),
            enabled_attacks=config.enabled_attacks,
        )
        # Note: current noiser implementation still uses one shared transform count.
        self.num_pert_noises_per_step = getattr(
            config, "num_pert_noises_per_step", None
        )

        encoder_params = self.encoder.parameters()
        decoder_params = self.decoder.parameters()
        if self.use_dct_dwt:
            encoder_params = chain(encoder_params, self.encoder_dwt.parameters())
            decoder_params = chain(decoder_params, self.decoder_dwt.parameters())

        self.opt_encoder = torch.optim.AdamW(encoder_params, lr=config.lr)
        self.opt_decoder = torch.optim.AdamW(decoder_params, lr=config.lr)
        self.opt_discriminator = torch.optim.AdamW(
            self.discriminator.parameters(), lr=config.lr
        )

        if self.config.enc_mode == "ecc":
            self.bchecc = model.BCHECC(t=config.ecc_t, m=config.ecc_m)
            logger.info(
                f"enc_bits: {self.config.num_encoded_bits}, data_bytes: {self.bchecc.data_bytes}"
            )

        self.lpips_loss_fn = lpips.LPIPS(net="vgg").to(self.device)
        self.ffl_fn = ffl(loss_weight=1.0, alpha=1.0)
        self.bce_loss_fn = nn.BCELoss()
        self.cur_epoch = 0
        self.cur_step = 0
        self.train_encoder = True

        self.wandb_options = wandb_options or {}
        self.wandb_run = None
        self.wandb = None
        self._init_wandb()

        self.writer = SummaryWriter(log_dir=config.log_dir)
        self.losses = defaultdict(float)
        self.eval_psnr = 0.0
        self.train_bit_accuracy = defaultdict(float)

        self.transform = transforms.Compose(
            [
                transforms.Resize(self.config.image_shape),
            ]
        )
        
        self.transform2 =  transforms.Compose(
            [
                transforms.Resize(64),
            ]
        )

        self.transform3 =  transforms.Compose(
            [
                transforms.Resize(256),

            ]
        )

        self.image_enhancer = model_dwt.ImageEnhancer().to(device)

        if self.use_dct:
            self.dct2d = dct_module.DCT2D().to(device)
            self.idct2d = dct_module.IDCT2D().to(device)
            logger.info("DCT mode enabled: encoder operates on DCT coefficients.")

        if self.use_dwt:
            self.dwt2d = dwt_module.HaarDWT2D().to(device)
            self.idwt2d = dwt_module.HaarIDWT2D().to(device)
            logger.info("DWT mode enabled: encoder operates on Haar wavelet subbands.")

        if self.use_dct_dwt:
            logger.info(
                "Hybrid DCT+DWT mode enabled: adaptive residual fusion with dual-domain decoding."
            )
            logger.info(
                "Hybrid decode rule: final_bits = 0.5 * (DCT_decoder + DWT_decoder)"
            )

        if fidelity is None:
            logger.warning(
                f"fidelity.py could not be imported. Fidelity benchmark metrics will be skipped. Error: {_FIDELITY_IMPORT_ERROR}"
            )
        if robustness is None:
            logger.warning(
                f"robustness.py could not be imported. Robustness BER metrics will be skipped. Error: {_ROBUSTNESS_IMPORT_ERROR}"
            )

    def _init_wandb(self):
        if not self.wandb_options.get("enabled", False):
            return

        try:
            import wandb

            api_key = self.wandb_options.get("api_key")
            if api_key:
                wandb.login(key=api_key)

            wandb_init_kwargs = {
                "project": self.wandb_options.get("project", "invismark"),
                "name": self.wandb_options.get("name"),
                "dir": self.wandb_options.get("dir", self.config.log_dir),
                "config": self.wandb_options.get("config", {}),
                "reinit": True,
            }
            if self.wandb_options.get("entity"):
                wandb_init_kwargs["entity"] = self.wandb_options["entity"]

            self.wandb_run = wandb.init(**wandb_init_kwargs)
            self.wandb = wandb
            logging.info("Weights & Biases logging is enabled.")
        except Exception as ex:
            logging.warning(
                f"Failed to initialize wandb. Continuing without wandb logging. Error: {ex}"
            )

    @staticmethod
    def _to_scalar(value):
        if isinstance(value, torch.Tensor):
            if value.numel() == 1:
                return float(value.detach().cpu().item())
            return float(value.detach().cpu().mean().item())
        return float(value)

    def _as_float_dict(self, values):
        return {k: self._to_scalar(v) for k, v in values.items()}

    @staticmethod
    def _format_seconds(seconds):
        seconds = max(0, int(seconds))
        h, rem = divmod(seconds, 3600)
        m, s = divmod(rem, 60)
        if h > 0:
            return f"{h:02d}:{m:02d}:{s:02d}"
        return f"{m:02d}:{s:02d}"

    def _wandb_log(self, payload, step):
        if self.wandb_run is None:
            return
        self.wandb.log(payload, step=step)

    @staticmethod
    def _to_uint8_numpy_image(x: torch.Tensor) -> np.ndarray:
        """
        Convert a single image tensor in CHW format to HWC uint8 for fidelity metrics.
        Supports tensors in [-1,1], [0,1], or [0,255].
        """
        if x.dim() != 3:
            raise ValueError(f"Expected CHW tensor, got shape={tuple(x.shape)}")

        y = x.detach().cpu().float()
        if y.shape[0] == 1:
            y = y.repeat(3, 1, 1)
        elif y.shape[0] > 3:
            y = y[:3]

        y_min = float(y.min())
        y_max = float(y.max())

        if y_min >= -1.1 and y_max <= 1.1:
            y = (y + 1.0) / 2.0
        elif y_min >= 0.0 and y_max <= 1.1:
            pass
        else:
            y = y / 255.0

        y = y.clamp(0.0, 1.0)
        y = (y * 255.0).round().to(torch.uint8)
        return y.permute(1, 2, 0).numpy()

    @staticmethod
    def _threshold_bits_to_numpy(
        bits: torch.Tensor, threshold: float = 0.5
    ) -> np.ndarray:
        return (bits.detach().cpu().float() >= threshold).to(torch.uint8).numpy()

    def _compute_clean_fidelity_metrics(self, orig_images, encoded_images):
        metric = {}
        if fidelity is None:
            return metric

        values = defaultdict(list)
        fid_refs = []
        fid_tsts = []
        batch_size = orig_images.shape[0]
        for idx in range(batch_size):
            ref = self._to_uint8_numpy_image(orig_images[idx])
            tst = self._to_uint8_numpy_image(encoded_images[idx])
            fid_refs.append(ref)
            fid_tsts.append(tst)

            try:
                values["fidelity_mse"].append(fidelity.MSE(ref, tst))
                values["fidelity_psnr"].append(fidelity.PSNR(ref, tst))
                values["fidelity_wpsnr"].append(fidelity.WPSNR(ref, tst))
                values["fidelity_ssim"].append(fidelity.SSIM(ref, tst))
                values["fidelity_jnd_pass_rate"].append(fidelity.JNDPassRate(ref, tst))
            except Exception as ex:
                logger.warning(
                    f"Failed to compute fidelity metrics for sample {idx}: {ex}"
                )

        try:
            
            # values["fidelity_fid"].append(
            #     10
            # )
            values["fidelity_fid"].append(
                fidelity.FID(fid_refs, fid_tsts, device=str(self.device))
            )
        except Exception as ex:
            logger.warning(f"Failed to compute batch FID metric: {ex}")

        for key, vals in values.items():
            if vals:
                metric[key] = float(np.mean(vals))
        return metric

    def _compute_robustness_metrics(self, secrets, extracted_secret, prefix=""):
        metric = {}
        if robustness is None:
            return metric

        gt = self._threshold_bits_to_numpy(secrets)
        ex = self._threshold_bits_to_numpy(extracted_secret)
        try:
            ber = robustness.BER(gt, ex)
        except Exception as ex_ber:
            logger.warning(f"Failed to compute robustness BER: {ex_ber}")
            ber = float(np.mean(gt != ex))
        metric[f"{prefix}robustness_ber"] = float(ber)
        return metric

    def _log_step_to_wandb(self, step_metrics):
        payload = {
            **{
                f"step/train/metric/{k}": self._to_scalar(v)
                for k, v in step_metrics.items()
            },
            **{
                f"step/train/loss/{k}": self._to_scalar(v)
                for k, v in self.losses.items()
            },
            "step/global_step": self.cur_step,
            "step/epoch": self.cur_epoch,
        }
        self._wandb_log(payload, step=self.cur_step)

    def train(self, train_data, eval_data=None, ckpt_path=None):
        if ckpt_path:
            logger.info(f"Loading model from ckpt: {ckpt_path}")
            self.load_model(ckpt_path)
            logger.info(
                f"Loaded model from epoch num_noises:{self.config.num_noises, self.config.beta_transform}"
            )

        fixed_batch = next(iter(train_data))
        for i in range(self.config.num_epochs):
            logger.info(
                f"Training for epoch: {self.cur_epoch}, beta_quality: {self._update_beta()}, train_encoder: {self.train_encoder}"
            )
            if self.eval_psnr > self.config.psnr_threshold:
                self.train_encoder = False

            if i < self.config.warmup_epochs:
                train_losses, train_metrics, train_images = self._train_one_epoch(
                    train_data, fixed_batch=fixed_batch
                )
            else:
                train_losses, train_metrics, train_images = self._train_one_epoch(
                    train_data
                )

            self._log_metrics(train_losses, "TrainEpoch/Loss", self.cur_epoch)
            self._log_metrics(train_metrics, "TrainEpoch/Metric", self.cur_epoch)
            self._log_images(*train_images, prefix="TrainEpoch", step=self.cur_epoch)
            self._wandb_log(
                {
                    **{f"train/loss/{k}": v for k, v in train_losses.items()},
                    **{f"train/metric/{k}": v for k, v in train_metrics.items()},
                    "train/psnr": train_metrics.get("psnr", 0.0),
                    "train/ber": train_metrics.get("ber", 0.0),
                    "train/bit_accuracy": train_metrics.get("bit_accuracy", 0.0),
                    "train/fidelity_psnr": train_metrics.get("fidelity_psnr", 0.0),
                    "train/fidelity_wpsnr": train_metrics.get("fidelity_wpsnr", 0.0),
                    "train/fidelity_fid": train_metrics.get("fidelity_fid", 0.0),
                    "train/fidelity_jnd_pass_rate": train_metrics.get(
                        "fidelity_jnd_pass_rate", 0.0
                    ),
                    "train/robustness_ber": train_metrics.get("robustness_ber", 0.0),
                },
                step=self.cur_epoch,
            )

            if self.wandb_run is not None and train_images[0] is not None:
                train_orig, train_secret = train_images
                with torch.no_grad():
                    train_encoded, _, _ = self._encode(train_orig, train_secret)
                self._wandb_log(
                    {
                        "train/images/input": self.wandb.Image(
                            transforms.ToPILImage()(
                                torch.clamp((train_orig[0].cpu() + 1.0) / 2.0, 0.0, 1.0)
                            )
                        ),
                        "train/images/encoded": self.wandb.Image(
                            transforms.ToPILImage()(
                                torch.clamp(
                                    (train_encoded[0].detach().cpu() + 1.0) / 2.0,
                                    0.0,
                                    1.0,
                                )
                            )
                        ),
                    },
                    step=self.cur_epoch,
                )

            if eval_data:
                eval_metrics, eval_images = self._validate(eval_data)
                self._log_metrics(eval_metrics, "EvalEpoch/Metric", self.cur_epoch)
                self._log_images(*eval_images, prefix="EvalEpoch", step=self.cur_epoch)
                self._wandb_log(
                    {
                        **{f"eval/metric/{k}": v for k, v in eval_metrics.items()},
                        "eval/psnr": eval_metrics.get("psnr", 0.0),
                        "eval/ber": eval_metrics.get("ber", 0.0),
                        "eval/bit_accuracy": eval_metrics.get("bit_accuracy", 0.0),
                        "eval/fidelity_psnr": eval_metrics.get("fidelity_psnr", 0.0),
                        "eval/fidelity_wpsnr": eval_metrics.get("fidelity_wpsnr", 0.0),
                        "eval/fidelity_fid": eval_metrics.get("fidelity_fid", 0.0),
                        "eval/fidelity_jnd_pass_rate": eval_metrics.get(
                            "fidelity_jnd_pass_rate", 0.0
                        ),
                        "eval/robustness_ber": eval_metrics.get("robustness_ber", 0.0),
                    },
                    step=self.cur_epoch,
                )

                if self.wandb_run is not None and eval_images[0] is not None:
                    eval_orig, eval_secret = eval_images
                    with torch.no_grad():
                        eval_encoded, _, _ = self._encode(eval_orig, eval_secret)
                    self._wandb_log(
                        {
                            "eval/images/input": self.wandb.Image(
                                transforms.ToPILImage()(
                                    torch.clamp(
                                        (eval_orig[0].cpu() + 1.0) / 2.0, 0.0, 1.0
                                    )
                                )
                            ),
                            "eval/images/encoded": self.wandb.Image(
                                transforms.ToPILImage()(
                                    torch.clamp(
                                        (eval_encoded[0].detach().cpu() + 1.0) / 2.0,
                                        0.0,
                                        1.0,
                                    )
                                )
                            ),
                        },
                        step=self.cur_epoch,
                    )

                self.eval_psnr = eval_metrics.get("psnr", self.eval_psnr)

            logger.info(f"Epoch {self.cur_epoch} train losses: {train_losses}")
            logger.info(f"Epoch {self.cur_epoch} train metrics: {train_metrics}")
            if eval_data:
                logger.info(f"Epoch {self.cur_epoch} eval metrics: {eval_metrics}")

            self._save_model()
            self.cur_epoch += 1

        if self.wandb_run is not None:
            self.wandb_run.finish()

    def eval(self, ckpt_path, input_images, secrets=None):
        self.load_model(ckpt_path)
        if secrets is not None:
            assert (
                secrets.shape[0] == input_images.shape[0]
            ), "Secrets and inputs need share the same batch dim."
        else:
            secrets = torch.randint(
                0,
                2,
                (input_images.shape[0], self.config.num_encoded_bits),
                device=self.device,
            ).type(torch.float32)
        enc_images, _, _ = self._encode(input_images, secrets)
        dec_secrets = self._decode(enc_images)
        return enc_images, secrets, dec_secrets

    def _encode_dwt(self, resize_inputs, secret):
        stacked_subbands, lv2_dict, lv1_hf = self.dwt2d(resize_inputs)
        dwt_encoder = self.encoder_dwt if self.use_dct_dwt else self.encoder

        # residuals_18ch = dwt_encoder(stacked_subbands, secret, self.alpha, self.transform2(resize_inputs))
        residuals_18ch = dwt_encoder(stacked_subbands, secret, self.alpha)

        lv2_lh, lv2_hl, lv2_hh = (
            residuals_18ch[:, 0:3],
            residuals_18ch[:, 3:6],
            residuals_18ch[:, 6:9],
        )
        lv1_lh, lv1_hl, lv1_hh = (
            residuals_18ch[:, 9:12],
            residuals_18ch[:, 12:15],
            residuals_18ch[:, 15:18],
        )

        lv1_target_hw = lv1_hf.shape[-2:]
        if lv1_lh.shape[-2:] != lv1_target_hw:
            lv1_lh = thf.interpolate(
                lv1_lh, size=lv1_target_hw, mode="bilinear", align_corners=False
            )
            lv1_hl = thf.interpolate(
                lv1_hl, size=lv1_target_hw, mode="bilinear", align_corners=False
            )
            lv1_hh = thf.interpolate(
                lv1_hh, size=lv1_target_hw, mode="bilinear", align_corners=False
            )

        modified_lv2 = {
            "LL": lv2_dict["LL"],
            "LH": torch.clamp(lv2_dict["LH"] + lv2_lh, -4, 4),
            "HL": torch.clamp(lv2_dict["HL"] + lv2_hl, -4, 4),
            "HH": torch.clamp(lv2_dict["HH"] + lv2_hh, -4, 4),
        }
        modified_lv1_hf = torch.cat(
            [
                torch.clamp(lv1_hf[:, 0:3] + lv1_lh, -4, 4),
                torch.clamp(lv1_hf[:, 3:6] + lv1_hl, -4, 4),
                torch.clamp(lv1_hf[:, 6:9] + lv1_hh, -4, 4),
            ],
            dim=1,
        )
        return torch.clamp(self.idwt2d(modified_lv2, modified_lv1_hf), -1.0, 1.0)

    @staticmethod
    def _adaptive_fuse_residuals(dct_residual, dwt_residual):
        # Energy-adaptive fusion: favor lower-energy residual branch per sample.
        dct_energy = dct_residual.pow(2).mean(dim=[1, 2, 3], keepdim=True)
        dwt_energy = dwt_residual.pow(2).mean(dim=[1, 2, 3], keepdim=True)
        mix_dct = dwt_energy / (dct_energy + dwt_energy + 1e-8)
        return mix_dct * dct_residual + (1.0 - mix_dct) * dwt_residual

    def _encode(self, inputs, secret):
        resize_inputs = self.transform(inputs).to(self.device)

        if self.use_dct_dwt:
            dct_inputs = self.dct2d(resize_inputs)
            dct_std = dct_inputs.std(dim=[2, 3], keepdim=True).clamp(min=1e-5)
            dct_norm = dct_inputs / dct_std
            dct_residual = self.encoder(dct_norm, secret, self.alpha)
            dct_encoded = torch.clamp(
                self.idct2d(dct_inputs + dct_residual), min=-1.0, max=1.0
            )

            dwt_encoded = self._encode_dwt(resize_inputs, secret)

            fused_residual = self._adaptive_fuse_residuals(
                dct_encoded - resize_inputs, dwt_encoded - resize_inputs
            )
            encoded_output = torch.clamp(resize_inputs + fused_residual, -1.0, 1.0)
        elif self.use_dwt:
            encoded_output = self._encode_dwt(resize_inputs, secret)
        elif self.use_dct:
            dct_inputs = self.dct2d(resize_inputs)
            dct_std = dct_inputs.std(dim=[2, 3], keepdim=True).clamp(min=1e-5)
            dct_norm = dct_inputs / dct_std
            dct_residual = self.encoder(dct_norm, secret, self.alpha)
            encoded_output = torch.clamp(
                self.idct2d(dct_inputs + dct_residual), min=-1.0, max=1.0
            )
        else:
            encoded_output = self.encoder(resize_inputs, secret, self.alpha)

        orig_diff = transforms.Resize(inputs.shape[-2:])(
            encoded_output - resize_inputs
        ).to("cpu")
        output = torch.clamp(inputs + orig_diff, min=-1.0, max=1.0)
        return output.to(self.device), resize_inputs, encoded_output

    def _decode(self, images):
        
        angle = self.image_enhancer(self.transform3(images))
        if angle > 10:
            noiser_angle = lambda x: rotate_tensor(x, angle=-15)
            images = noiser_angle(images)
   

        trans_images = self.transform(images)
        # trans_images = images
        
        
        if self.use_dct_dwt:
            dct_pred = self.decoder(trans_images)
            stacked_subbands, _, _ = self.dwt2d(trans_images)
            dwt_pred = self.decoder_dwt(stacked_subbands)
            return 0.5 * (dct_pred + dwt_pred)
        if self.use_dwt:
            stacked_subbands, _, _ = self.dwt2d(trans_images)
            return self.decoder(stacked_subbands, self.transform2(trans_images))
        return self.decoder(trans_images)

    def _update_beta(self):
        cur_beta_epoch = min(
            max(0, self.cur_epoch - self.config.beta_start_epoch),
            self.config.beta_epochs - 1,
        )
        beta_schedule = np.logspace(
            np.log10(self.config.beta_min),
            np.log10(self.config.beta_max),
            self.config.beta_epochs,
        )
        return beta_schedule[cur_beta_epoch]

    def _train_one_epoch(self, dataloader, fixed_batch=None):
        avg_losses = defaultdict(float)
        num_steps = 0
        epoch_total_steps = len(dataloader)
        epoch_start_time = time.time()
        example_images, example_secret = None, None

        for data in dataloader:
            iter_start_time = time.time()
            self.encoder.train()
            self.decoder.train()
            if self.use_dct_dwt:
                self.encoder_dwt.train()
                self.decoder_dwt.train()
            if self.train_encoder:
                self.opt_encoder.zero_grad()
            self.opt_decoder.zero_grad()
            self.opt_discriminator.zero_grad()
            self.cur_step += 1
            if fixed_batch is not None:
                data = fixed_batch

            secret = self._generate_secret(data[0].shape[0], self.device)
            total_loss = self._loss_fn(data[0], secret)
            total_loss.backward()
            num_steps += 1

            if self.train_encoder:
                self.opt_encoder.step()
            self.opt_decoder.step()
            self.opt_discriminator.step()

            avg_losses["total_loss"] += self._to_scalar(total_loss)
            for loss_name, loss_val in self.losses.items():
                avg_losses[loss_name] += self._to_scalar(loss_val)

            example_images = data[0]
            example_secret = secret

            iter_time_sec = time.time() - iter_start_time
            elapsed_epoch_sec = time.time() - epoch_start_time
            avg_iter_time_sec = elapsed_epoch_sec / max(1, num_steps)
            remaining_steps = max(0, epoch_total_steps - num_steps)
            eta_epoch_sec = remaining_steps * avg_iter_time_sec

            logger.info(
                f"epoch: {self.cur_epoch}, epoch_step: {num_steps}/{epoch_total_steps}, "
                f"global_step: {self.cur_step}, iter_time: {iter_time_sec:.2f}s, "
                f"eta_epoch: {self._format_seconds(eta_epoch_sec)}, total_loss: {total_loss}"
            )
            # if self.cur_step % self.config.log_interval == 0:
            #     step_metrics = self._calculate_metric(data[0], secret, prefix="Train")
            #     self._log_metrics(step_metrics, "Train")
            #     self._log_metrics(self.losses, "Train")
            #     self._log_step_to_wandb(step_metrics)

        if num_steps == 0:
            return {}, {}, (None, None)

        avg_losses = {k: v / num_steps for k, v in avg_losses.items()}
        train_metrics = self._calculate_metric(example_images, example_secret, "Train")
        return avg_losses, train_metrics, (example_images, example_secret)

    def _loss_fn(self, data, secret):
        if self.train_encoder:
            final_output, enc_inputs, enc_output = self._encode(data, secret)
        else:
            with torch.no_grad():
                final_output, enc_inputs, enc_output = self._encode(data, secret)
        extracted_secret = self._decode(final_output)

        self.losses["mse_loss"] = utils.compute_reconstruction_loss(
            enc_inputs, enc_output, self.device, recon_type="yuv"
        ).mean()
        self.losses["lpips_loss"] = self.lpips_loss_fn(enc_inputs, enc_output).mean()
        self.losses["bce_loss"] = self.bce_loss_fn(extracted_secret, secret)
        self.losses["ffl_loss"] = self.ffl_fn(enc_inputs, enc_output).mean()
        self.losses["discriminator_loss"] = torch.ones(1, device=self.device)

        if self.cur_epoch >= self.config.beta_start_epoch:
            real_loss = -torch.mean(self.discriminator(enc_inputs))
            fake_loss = torch.mean(self.discriminator(enc_output))
            self.losses["discriminator_loss"] += real_loss + fake_loss

        if not self.train_encoder or self.cur_epoch >= self.config.noise_start_epoch:
            sorted_keys = [
                [k]
                for k, _ in sorted(self.train_bit_accuracy.items(), key=lambda x: x[1])
            ]

            for key in [None] + sorted_keys[: self.config.num_noises]:
                if self.config.num_noises == 0:
                    break
                trans_output = self.noiser(final_output, key, train=True)
                if key is None:
                    logger.info("Adding noise: [random geo + random pert]")
                else:
                    logger.info(f"Adding noise: {key}")
                extracted_secret = self._decode(trans_output)
                self.losses[
                    "bce_loss"
                ] += self.config.beta_transform * self.bce_loss_fn(
                    extracted_secret, secret
                )
                print("lose after noise", self.losses["bce_loss"].item())

        if self.losses["bce_loss"] < 0.35:
            self.alpha = max(0.0, self.alpha - 0.05)

        if self.train_encoder:  # and self.cur_epoch > 0 :
            beta_quality = self._update_beta()
            return (
                beta_quality
                * (
                    self.losses["mse_loss"]
                    + self.losses["lpips_loss"]
                    + self.losses["discriminator_loss"]
                    + self.losses["ffl_loss"]
                )
                + self.losses["bce_loss"]
            )
        return self.losses["bce_loss"]

    def _generate_secret(self, batch_size, device):
        if self.config.enc_mode == "uuid":
            bits, _ = utils.uuid_to_bits(batch_size)
        elif self.config.enc_mode == "ecc":
            assert self.config.num_encoded_bits == 256, "Encode 256 bits in ecc mode"
            bits = self.bchecc.batch_encode(batch_size)
        else:
            raise ValueError(
                "secret enc_mode is not supported! choose between uuid and ecc."
            )
        return bits[:, : self.config.num_encoded_bits].to(device)

    @torch.no_grad()
    def _calculate_metric(self, orig_images, secrets, prefix="Train", log_img =False):
        self.encoder.eval()
        self.decoder.eval()
        if self.use_dct_dwt:
            self.encoder_dwt.eval()
            self.decoder_dwt.eval()
        final_output, _, _ = self._encode(orig_images, secrets)
        metric = defaultdict(float)
        
        if log_img is True:
            try:
                self._save_metric_pngs(orig_images, final_output, prefix=prefix)
            except Exception as ex:
                logger.warning(f"Failed to save metric PNGs: {ex}")

        metric["psnr"] = self._to_scalar(
            metrics.image_psnr(orig_images, final_output.cpu())
        )
        metric["ssim"] = self._to_scalar(
            metrics.image_ssim(final_output.cpu(), orig_images)
        )

        extracted_secret_clean = self._decode(final_output)
        bit_accuracy_clean = self._to_scalar(
            metrics.bit_accuracy(secrets, extracted_secret_clean)
        )
        metric["bit_accuracy"] = bit_accuracy_clean
        metric["ber"] = 1.0 - bit_accuracy_clean

        metric.update(self._compute_clean_fidelity_metrics(orig_images, final_output))
        metric.update(
            self._compute_robustness_metrics(secrets, extracted_secret_clean, prefix="")
        )

        if prefix == "Train":
            noise_keys = noise.get_train_attack_names()
            train = True
        elif prefix == "Eval":
            noise_keys = noise.get_eval_attack_names()
            train = False

        for key in noise_keys:
            trans_output = self.noiser(final_output, [key], train=(prefix == "Train"))
            extracted_secret = self._decode(trans_output)
            bit_accuracy_trans = self._to_scalar(
                metrics.bit_accuracy(secrets, extracted_secret)
            )
            
            if log_img is True:
                try:
                    self._save_metric_pngs(orig_images, trans_output, prefix=prefix)
                except Exception as ex:
                    logger.warning(f"Failed to save metric PNGs: {ex}")
                
            # metric[f"BitAcc-{key}"] = bit_accuracy_trans
            metric[f"BER-{key}"] = 1.0 - bit_accuracy_trans
            metric.update(
                self._compute_robustness_metrics(
                    secrets, extracted_secret, prefix=f"{key}/"
                )
            )
            if prefix == "Train":
                self.train_bit_accuracy[key] = bit_accuracy_trans
            if self.config.enc_mode == "ecc":
                cor_secret = self.bchecc.batch_decode_ecc(extracted_secret).cpu()
                data_bit_acc = self._to_scalar(
                    metrics.bit_accuracy(
                        cor_secret[:, : -self.bchecc.bch.ecc_bytes * 8],
                        secrets[:, : -self.bchecc.bch.ecc_bytes * 8].cpu(),
                    )
                )
                # metric[f"DataBitAcc-{key}"] = data_bit_acc
                metric[f"DataBER-{key}"] = 1.0 - data_bit_acc

        self.encoder.train()
        self.decoder.train()
        if self.use_dct_dwt:
            self.encoder_dwt.train()
            self.decoder_dwt.train()
        return dict(metric)

    def _log_metrics(self, metrics_dict, prefix="Train", step=None):
        current_step = self.cur_step if step is None else step
        for key in metrics_dict:
            self.writer.add_scalar(
                f"{prefix}/{key}", self._to_scalar(metrics_dict[key]), current_step
            )

    def _log_images(self, orig_images, secrets, prefix="Train", step=None):
        if orig_images is None:
            return
        current_step = self.cur_step if step is None else step
        with torch.no_grad():
            final_output, _, _ = self._encode(orig_images, secrets)
        grid = vutils.make_grid(orig_images[0], normalize=True, value_range=(-1.0, 1.0))
        self.writer.add_image(f"{prefix}/input_images", grid, current_step)
        grid = vutils.make_grid(
            final_output[0], normalize=True, value_range=(-1.0, 1.0)
        )
        self.writer.add_image(f"{prefix}/encoded_images", grid, current_step)
        grid = vutils.make_grid(
            10.0 * (orig_images[0] - final_output[0].cpu()),
            normalize=True,
            value_range=(-1.0, 1.0),
        )
        self.writer.add_image(f"{prefix}/image_diff_x10", grid, current_step)

    @torch.no_grad()
    def _validate(self, eval_data, num_batches=100):
        avg_metrics = defaultdict(float)
        eval_batch = None
        secrets = None
        num_batches_seen = 0
        for i, eval_batch in enumerate(eval_data):
            secrets = self._generate_secret(eval_batch[0].shape[0], self.device)
            batch_metric = self._calculate_metric(eval_batch[0], secrets, "Eval")
            for k, v in batch_metric.items():
                avg_metrics[k] += self._to_scalar(v)
            num_batches_seen += 1
            if (i + 1) >= num_batches:
                break

        if num_batches_seen == 0:
            return {}, (None, None)

        for k in avg_metrics:
            avg_metrics[k] = avg_metrics[k] / num_batches_seen
        return dict(avg_metrics), (
            eval_batch[0] if eval_batch is not None else None,
            secrets,
        )
        
    def _save_metric_pngs(self, orig_images, final_output, prefix="Train"):
        """Save paired input/output PNGs to a random folder for metric inspection."""
        base_dir = os.path.join(self.config.log_dir, "metric_pngs")
        rand_dir = f"{prefix.lower()}_{self.cur_epoch}_{self.cur_step}_{uuid.uuid4().hex[:8]}"
        out_dir = os.path.join(base_dir, rand_dir)
        os.makedirs(out_dir, exist_ok=True)

        for idx in range(orig_images.shape[0]):
            input_path = os.path.join(out_dir, f"{idx:04d}_input.png")
            output_path = os.path.join(out_dir, f"{idx:04d}_output.png")
            input_arr = self._to_uint8_numpy_image(orig_images[idx])
            output_arr = self._to_uint8_numpy_image(final_output[idx])
            Image.fromarray(input_arr, mode="RGB").save(input_path, format="PNG")
            Image.fromarray(output_arr, mode="RGB").save(output_path, format="PNG")

        logger.info(f"Saved metric PNG pairs to {out_dir}")

    def load_model(self, ckpt_path):
        state_dict = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        logger.info(f"Loading model from epoch:{state_dict['cur_epoch']}")
        self.encoder.load_state_dict(state_dict["encoder_state_dict"])
        self.encoder.train()
        if self.use_dct_dwt and "encoder_dwt_state_dict" in state_dict:
            self.encoder_dwt.load_state_dict(state_dict["encoder_dwt_state_dict"])
            self.encoder_dwt.train()
        self.decoder.load_state_dict(state_dict["decoder_state_dict"])
        self.decoder.train()
        if self.use_dct_dwt and "decoder_dwt_state_dict" in state_dict:
            self.decoder_dwt.load_state_dict(state_dict["decoder_dwt_state_dict"])
            self.decoder_dwt.train()
        self.discriminator.load_state_dict(state_dict["discriminator_state_dict"])
        self.discriminator.train()
        self.opt_encoder.load_state_dict(state_dict["opt_encoder_state_dict"])
        self.opt_decoder.load_state_dict(state_dict["opt_decoder_state_dict"])
        self.opt_discriminator.load_state_dict(
            state_dict["opt_discriminator_state_dict"]
        )
        self.cur_epoch = state_dict["cur_epoch"]
        self.cur_step = state_dict["cur_step"]
        self.config = state_dict["config"]
        self.image_enhancer.load_state_dict(state_dict["image_enhancer_state_dict"])

    # def _save_model(self):
    #     if not os.path.exists(self.config.ckpt_path):
    #         os.makedirs(self.config.ckpt_path)
    #     state = {
    #         "encoder_state_dict": self.encoder.state_dict(),
    #         "decoder_state_dict": self.decoder.state_dict(),
    #         "discriminator_state_dict": self.discriminator.state_dict(),
    #         "opt_encoder_state_dict": self.opt_encoder.state_dict(),
    #         "opt_decoder_state_dict": self.opt_decoder.state_dict(),
    #         "opt_discriminator_state_dict": self.opt_discriminator.state_dict(),
    #         "cur_epoch": self.cur_epoch,
    #         "cur_step": self.cur_step,
    #         "config": self.config,
    #     }
    #     if self.use_dct_dwt:
    #         state["encoder_dwt_state_dict"] = self.encoder_dwt.state_dict()
    #         state["decoder_dwt_state_dict"] = self.decoder_dwt.state_dict()
    #     torch.save(
    #         state,
    #         f"{self.config.ckpt_path}/model-{self.cur_epoch:04d}.ckpt",
    #     )

    def _save_model(self):
        if not os.path.exists(self.config.ckpt_path):
            os.makedirs(self.config.ckpt_path)
        torch.save(
            {
                "encoder_state_dict": self.encoder.state_dict(),
                "decoder_state_dict": self.decoder.state_dict(),
                "discriminator_state_dict": self.discriminator.state_dict(),
                "opt_encoder_state_dict": self.opt_encoder.state_dict(),
                "opt_decoder_state_dict": self.opt_decoder.state_dict(),
                "opt_discriminator_state_dict": self.opt_discriminator.state_dict(),
                "image_enhancer_state_dict": self.image_enhancer.state_dict(),
                "cur_epoch": self.cur_epoch,
                "cur_step": self.cur_step,
                "config": self.config,
            },
            f"{self.config.ckpt_path}/model-{self.cur_epoch:04d}.ckpt",
        )

        # Keep only the most recent checkpoints by epoch number.
        max_keep = 5
        ckpt_pattern = re.compile(r"^model-(\d+)\.ckpt$")
        ckpt_files = []
        for name in os.listdir(self.config.ckpt_path):
            match = ckpt_pattern.match(name)
            if match:
                ckpt_files.append((int(match.group(1)), name))

        if len(ckpt_files) > max_keep:
            ckpt_files.sort(key=lambda x: x[0])
            for _, name in ckpt_files[:-max_keep]:
                try:
                    os.remove(os.path.join(self.config.ckpt_path, name))
                except OSError as ex:
                    logger.warning(f"Failed to remove old checkpoint {name}: {ex}")