# dwt_utils.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class HaarDWT2D(nn.Module):
    """
    Differentiable 2-level Haar DWT.
    Returns a flat tensor of stacked subbands for direct use as encoder input.
    Output shape: (B, 21, H/4, W/4)
      - 7 subbands: [LL2, LH2, HL2, HH2, LH1, HL1, HH1]
      - each subband has 3 RGB channels → 7×3 = 21
    All operations are conv-based, so gradients flow cleanly.
    """

    def __init__(self):
        super().__init__()
        # Haar filter kernels — fixed, not learned
        ll = torch.tensor([[1, 1], [1, 1]], dtype=torch.float32) / 2.0
        lh = torch.tensor([[-1, -1], [1, 1]], dtype=torch.float32) / 2.0
        hl = torch.tensor([[-1, 1], [-1, 1]], dtype=torch.float32) / 2.0
        hh = torch.tensor([[1, -1], [-1, 1]], dtype=torch.float32) / 2.0

        filters = torch.stack([ll, lh, hl, hh]).unsqueeze(1)
        self.register_buffer("filters", filters)

    def _dwt_one_level(self, x: torch.Tensor):
        """Single DWT level: (B,C,H,W) → dict of 4 subbands at (B,C,H/2,W/2)"""
        B, C, H, W = x.shape
        x_grouped = x.reshape(B * C, 1, H, W)
        out = F.conv2d(x_grouped, self.filters, stride=2, padding=0)
        out = out.reshape(B, C, 4, H // 2, W // 2)
        return {
            "LL": out[:, :, 0],
            "LH": out[:, :, 1],
            "HL": out[:, :, 2],
            "HH": out[:, :, 3],
        }

    def forward(self, x: torch.Tensor):
        """
        x: (B, 3, H, W), values in [-1, 1]
        Returns: stacked subbands (B, 21, H/4, W/4)
        Order: [LL2(3), LH2(3), HL2(3), HH2(3), LH1(3), HL1(3), HH1(3)]
        Also returns level-2 dict and full-resolution level-1 HF for IDWT.
        """
        lv1 = self._dwt_one_level(x)  # level-1 decomp
        lv2 = self._dwt_one_level(lv1["LL"])  # level-2 on LL only

        lv1_lh_ds = F.avg_pool2d(lv1["LH"], kernel_size=2, stride=2)
        lv1_hl_ds = F.avg_pool2d(lv1["HL"], kernel_size=2, stride=2)
        lv1_hh_ds = F.avg_pool2d(lv1["HH"], kernel_size=2, stride=2)

        # Stack: 7 subbands × 3 channels = 21ch at H/4, W/4
        stacked = torch.cat(
            [
                lv2["LL"],
                lv2["LH"],
                lv2["HL"],
                lv2["HH"],
                lv1_lh_ds,
                lv1_hl_ds,
                lv1_hh_ds,
            ],
            dim=1,
        )  # (B, 21, H/4, W/4)

        lv1_hf = torch.cat([lv1["LH"], lv1["HL"], lv1["HH"]], dim=1)  # (B,9,H/2,W/2)
        return stacked, lv2, lv1_hf


class HaarIDWT2D(nn.Module):
    """
    Reconstruct spatial image from modified DWT subbands.
    Takes modified lv2 subbands and lv1 HF bands → spatial (B,3,H,W).
    """

    def __init__(self):
        super().__init__()
        ll = torch.tensor([[1, 1], [1, 1]], dtype=torch.float32) / 2.0
        lh = torch.tensor([[-1, -1], [1, 1]], dtype=torch.float32) / 2.0
        hl = torch.tensor([[-1, 1], [-1, 1]], dtype=torch.float32) / 2.0
        hh = torch.tensor([[1, -1], [-1, 1]], dtype=torch.float32) / 2.0
        filters = torch.stack([ll, lh, hl, hh]).unsqueeze(1)
        self.register_buffer("filters", filters)

    def _idwt_one_level(self, LL, LH, HL, HH):
        B, C, H, W = LL.shape
        combined = torch.stack([LL, LH, HL, HH], dim=2)  # (B,C,4,H,W)
        combined = combined.reshape(B * C, 4, H, W)
        out = F.conv_transpose2d(combined, self.filters, stride=2, padding=0)
        return out.reshape(B, C, H * 2, W * 2)

    def forward(self, lv2_dict, lv1_hf):
        """
        lv2_dict: dict with modified LL,LH,HL,HH at level-2 resolution
        lv1_hf: (B,9,H/2,W/2) — possibly modified LH1,HL1,HH1
        """
        recon_lv1_LL = self._idwt_one_level(
            lv2_dict["LL"], lv2_dict["LH"], lv2_dict["HL"], lv2_dict["HH"]
        )
        LH1, HL1, HH1 = lv1_hf.chunk(3, dim=1)
        return self._idwt_one_level(recon_lv1_LL, LH1, HL1, HH1)
