import math

import torch
import torch.nn as nn


def _orthonormal_dct_matrix(
    n: int, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    """Build orthonormal DCT-II matrix C with shape (n, n)."""
    k = torch.arange(n, device=device, dtype=dtype).unsqueeze(1)
    i = torch.arange(n, device=device, dtype=dtype).unsqueeze(0)

    mat = torch.cos((math.pi / n) * (i + 0.5) * k)
    mat[0] = mat[0] * math.sqrt(1.0 / n)
    if n > 1:
        mat[1:] = mat[1:] * math.sqrt(2.0 / n)
    return mat


class DCT2D(nn.Module):
    """
    Differentiable orthonormal 2D DCT (type-II) over spatial dimensions.
    Input:  (B, C, H, W)
    Output: (B, C, H, W)
    """

    def __init__(self):
        super().__init__()
        self._dct_cache = {}

    def _get_dct_mats(self, h: int, w: int, device: torch.device, dtype: torch.dtype):
        key = (h, w, str(device), str(dtype))
        mats = self._dct_cache.get(key)
        if mats is None:
            c_h = _orthonormal_dct_matrix(h, device, dtype)
            c_w = _orthonormal_dct_matrix(w, device, dtype)
            mats = (c_h, c_w)
            self._dct_cache[key] = mats
        return mats

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"Expected x shape (B,C,H,W), got {tuple(x.shape)}")

        _, _, h, w = x.shape
        c_h, c_w = self._get_dct_mats(h, w, x.device, x.dtype)

        # Y = C_h @ X @ C_w^T, applied independently for each batch/channel.
        tmp = torch.einsum("nh,bchw->bcnw", c_h, x)
        out = torch.einsum("mw,bcnw->bcnm", c_w, tmp)
        return out


class IDCT2D(nn.Module):
    """
    Inverse of orthonormal 2D DCT. Reconstructs spatial image.
    Input:  (B, C, H, W)
    Output: (B, C, H, W)
    """

    def __init__(self):
        super().__init__()
        self._dct_cache = {}

    def _get_dct_mats(self, h: int, w: int, device: torch.device, dtype: torch.dtype):
        key = (h, w, str(device), str(dtype))
        mats = self._dct_cache.get(key)
        if mats is None:
            c_h = _orthonormal_dct_matrix(h, device, dtype)
            c_w = _orthonormal_dct_matrix(w, device, dtype)
            mats = (c_h, c_w)
            self._dct_cache[key] = mats
        return mats

    def forward(self, coeffs: torch.Tensor) -> torch.Tensor:
        if coeffs.ndim != 4:
            raise ValueError(
                f"Expected coeffs shape (B,C,H,W), got {tuple(coeffs.shape)}"
            )

        _, _, h, w = coeffs.shape
        c_h, c_w = self._get_dct_mats(h, w, coeffs.device, coeffs.dtype)

        # X = C_h^T @ Y @ C_w for orthonormal C.
        tmp = torch.einsum("kh,bckw->bchw", c_h, coeffs)
        out = torch.einsum("bchw,wm->bchm", tmp, c_w)
        return out
