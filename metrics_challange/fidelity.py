from __future__ import annotations

from typing import Union
import os
import tempfile
import numpy as np
import cv2
from PIL import Image
from skimage.metrics import structural_similarity as ssim_fn

from ._io import imread_rgb, PathLike

ArrayLike = Union[np.ndarray, PathLike]

_FID_MODELS: dict[tuple[str, int], object] = {}


def _as_rgb(img: ArrayLike) -> np.ndarray:
    # Load/convert input into an RGB numpy array (H, W, 3)
    if isinstance(img, (str, bytes)) or hasattr(img, "__fspath__"):
        return imread_rgb(img)
    arr = np.asarray(img)
    if arr.ndim == 2:
        return np.stack([arr, arr, arr], axis=-1)
    if arr.ndim == 3 and arr.shape[2] >= 3:
        return arr[..., :3]
    raise ValueError(f"Unexpected image shape: {arr.shape}")


def _to_gray01(img: np.ndarray) -> np.ndarray:
    # Convert image to grayscale, handling uint8 and float inputs
    arr = np.asarray(img)
    if arr.dtype == np.uint8:
        arrf = arr.astype(np.float64) / 255.0
    else:
        arrf = arr.astype(np.float64)
        if arrf.max() > 1.5:
            arrf = np.clip(arrf, 0.0, 255.0) / 255.0
        else:
            arrf = np.clip(arrf, 0.0, 1.0)

    if arrf.ndim == 2:
        gray = arrf
    else:
        r, g, b = arrf[..., 0], arrf[..., 1], arrf[..., 2]
        gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return np.clip(gray, 0.0, 1.0).astype(np.float64)


def _match_shapes_center_crop(
    a: np.ndarray, b: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    # Center-crop two RGB images to the same (minimum) height/width
    if a.shape == b.shape:
        return a, b
    h = min(a.shape[0], b.shape[0])
    w = min(a.shape[1], b.shape[1])

    def crop(x):
        y0 = (x.shape[0] - h) // 2
        x0 = (x.shape[1] - w) // 2
        return x[y0 : y0 + h, x0 : x0 + w, :]

    return crop(a), crop(b)


def MSE(ref: ArrayLike, tst: ArrayLike) -> float:
    # Mean Squared Error between two RGB images
    a = _as_rgb(ref).astype(np.float64)
    b = _as_rgb(tst).astype(np.float64)
    if a.shape != b.shape:
        a, b = _match_shapes_center_crop(a, b)
    return float(np.mean((a - b) ** 2))


def PSNR(ref: ArrayLike, tst: ArrayLike, max_val: float = 255.0) -> float:
    # Peak Signal-to-Noise Ratio (dB) from RGB MSE
    m = MSE(ref, tst)
    if m == 0:
        return float("inf")
    return float(10.0 * np.log10((max_val**2) / m))


def WPSNR(ref: ArrayLike, tst: ArrayLike, eps: float = 1e-6) -> float:
    # Weighted PSNR: luminance-based weight map, RGB error
    ref_rgb = _as_rgb(ref).astype(np.float64)
    tst_rgb = _as_rgb(tst).astype(np.float64)
    if ref_rgb.shape != tst_rgb.shape:
        ref_rgb, tst_rgb = _match_shapes_center_crop(ref_rgb, tst_rgb)

    ref_y = _to_gray01(ref_rgb)
    ref8 = (np.clip(ref_y * 255, 0, 255)).astype(np.uint8)
    blur = cv2.GaussianBlur(ref8, (7, 7), 1.5).astype(np.float64) / 255.0

    w = 1.0 / (1.0 + np.abs(ref_y - blur) + eps)
    wmse = np.sum(w[..., None] * (ref_rgb - tst_rgb) ** 2) / np.sum(w[..., None])

    if wmse == 0:
        return float("inf")
    return float(10.0 * np.log10((255.0**2) / wmse))


def SSIM(ref: ArrayLike, tst: ArrayLike) -> float:
    # Structural Similarity Index (SSIM) on grayscale
    ref_rgb = _as_rgb(ref)
    tst_rgb = _as_rgb(tst)
    if ref_rgb.shape != tst_rgb.shape:
        ref_rgb, tst_rgb = _match_shapes_center_crop(ref_rgb, tst_rgb)

    a = _to_gray01(ref_rgb)
    b = _to_gray01(tst_rgb)
    return float(ssim_fn(a, b, data_range=1.0))


def FID(
    refs: list[ArrayLike],
    tsts: list[ArrayLike],
    device: str | None = None,
    dims: int = 2048,
    batch_size: int | None = None,
    num_workers: int = 0,
) -> float:
    # Frechet Inception Distance between two batches of images.
    if len(refs) != len(tsts):
        raise ValueError(
            f"refs and tsts must have the same length, got {len(refs)} and {len(tsts)}"
        )
    if len(refs) == 0:
        raise ValueError("refs and tsts must be non-empty")

    try:
        import torch
        from .fid_score.fid_score import (
            InceptionV3,
            calculate_activation_statistics,
            calculate_frechet_distance,
        )
    except Exception as ex:
        raise ImportError(
            "FID metric requires torch and metrics_challange/fid_score dependencies"
        ) from ex

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if batch_size is None:
        batch_size = min(50, len(refs))

    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
    model_key = (str(device), int(dims))
    model = _FID_MODELS.get(model_key)
    if model is None:
        model = InceptionV3([block_idx]).to(device)
        _FID_MODELS[model_key] = model

    with tempfile.TemporaryDirectory(prefix="fid_metric_") as tmpdir:
        ref_dir = os.path.join(tmpdir, "ref")
        tst_dir = os.path.join(tmpdir, "tst")
        os.makedirs(ref_dir, exist_ok=True)
        os.makedirs(tst_dir, exist_ok=True)

        ref_files: list[str] = []
        tst_files: list[str] = []
        for i, (ref, tst) in enumerate(zip(refs, tsts)):
            ref_rgb = _as_rgb(ref)
            tst_rgb = _as_rgb(tst)
            if ref_rgb.shape != tst_rgb.shape:
                ref_rgb, tst_rgb = _match_shapes_center_crop(ref_rgb, tst_rgb)

            ref_u8 = np.clip(ref_rgb, 0, 255).astype(np.uint8, copy=False)
            tst_u8 = np.clip(tst_rgb, 0, 255).astype(np.uint8, copy=False)

            ref_path = os.path.join(ref_dir, f"{i:06d}.png")
            tst_path = os.path.join(tst_dir, f"{i:06d}.png")
            Image.fromarray(ref_u8, mode="RGB").save(ref_path, format="PNG")
            Image.fromarray(tst_u8, mode="RGB").save(tst_path, format="PNG")
            ref_files.append(ref_path)
            tst_files.append(tst_path)

        m1, s1 = calculate_activation_statistics(
            ref_files,
            model,
            batch_size=batch_size,
            dims=dims,
            device=device,
            num_workers=num_workers,
        )
        m2, s2 = calculate_activation_statistics(
            tst_files,
            model,
            batch_size=batch_size,
            dims=dims,
            device=device,
            num_workers=num_workers,
        )
        return float(calculate_frechet_distance(m1, s1, m2, s2))


# need to update and change to mays version
def JNDPassRate(ref: ArrayLike, tst: ArrayLike) -> float:
    # Full JND map pipeline (bg luminance, contrast masking, content complexity, edge protection).
    ref_rgb = _as_rgb(ref)
    tst_rgb = _as_rgb(tst)
    if ref_rgb.shape != tst_rgb.shape:
        ref_rgb, tst_rgb = _match_shapes_center_crop(ref_rgb, tst_rgb)

    a = (_to_gray01(ref_rgb) * 255.0).astype(np.float64, copy=False)
    b = (_to_gray01(tst_rgb) * 255.0).astype(np.float64, copy=False)

    eps = 1e-6

    def _jnd_conv(inp, filt, padding="SAME", rotate=False):
        inp = np.asarray(inp, dtype=np.float64)
        ker = np.asarray(filt, dtype=np.float64)
        if rotate:
            ker = np.flipud(np.fliplr(ker))
        if padding == "SAME":
            res = cv2.filter2D(
                inp, ddepth=-1, kernel=ker, borderType=cv2.BORDER_REFLECT
            )
        elif padding == "FULL":
            fh, fw = ker.shape
            inp_p = cv2.copyMakeBorder(
                inp, fh - 1, fh - 1, fw - 1, fw - 1, cv2.BORDER_REFLECT
            )
            res = cv2.filter2D(
                inp_p, ddepth=-1, kernel=ker, borderType=cv2.BORDER_CONSTANT
            )
        elif padding == "VALID":
            fh, fw = ker.shape
            tmp = cv2.filter2D(
                inp, ddepth=-1, kernel=ker, borderType=cv2.BORDER_CONSTANT
            )
            top = fh // 2
            left = fw // 2
            res = tmp[
                top : tmp.shape[0] - (fh - 1 - top),
                left : tmp.shape[1] - (fw - 1 - left),
            ]
        else:
            raise ValueError("Unsupported padding mode.")
        return res.astype(np.float64, copy=False)

    def _jnd_bg_adjust(bg_lum, min_lum):
        adapt_bg = np.round(min_lum + bg_lum * (127 - min_lum) / 127 + eps)
        bg_lum = np.where(bg_lum <= 127, adapt_bg, bg_lum)
        return bg_lum

    def _jnd_lum_table():
        bg_jnd = {}
        T0 = 17
        gamma = 3 / 128
        for k in range(256):
            if k < 127:
                bg_jnd[k] = T0 * (1 - np.sqrt(k / 127)) + 3
            else:
                bg_jnd[k] = gamma * (k - 127) + 3
        return bg_jnd

    def _jnd_func_bg_lum(img):
        min_lum = 32
        alpha = 0.7
        B = np.array(
            [
                [1, 1, 1, 1, 1],
                [1, 2, 2, 2, 1],
                [1, 2, 0, 2, 1],
                [1, 2, 2, 2, 1],
                [1, 1, 1, 1, 1],
            ],
            dtype=np.float64,
        )
        bg_lum = np.floor(_jnd_conv(img, B) / 32)
        bg_lum = _jnd_bg_adjust(bg_lum, min_lum)
        table = _jnd_lum_table()
        jnd_lum = np.vectorize(table.get)(bg_lum)
        jnd_lum_adapt = alpha * jnd_lum
        return jnd_lum_adapt

    def _jnd_gkern(kernlen=21, nsig=3.0):
        x = np.linspace(-nsig, nsig, kernlen)
        kern1d = np.exp(-0.5 * x**2)
        kern1d /= kern1d.sum() + eps
        ker = np.outer(kern1d, kern1d)
        ker /= ker.sum() + eps
        return ker

    def _jnd_edge_height(img):
        G1 = np.array(
            [
                [0, 0, 0, 0, 0],
                [1, 3, 8, 3, 1],
                [0, 0, 0, 0, 0],
                [-1, -3, -8, -3, -1],
                [0, 0, 0, 0, 0],
            ],
            dtype=np.float64,
        )
        G2 = np.array(
            [
                [0, 0, 1, 0, 0],
                [0, 8, 3, 0, 0],
                [1, 3, 0, -3, -1],
                [0, 0, -3, -8, 0],
                [0, 0, -1, 0, 0],
            ],
            dtype=np.float64,
        )
        G3 = np.array(
            [
                [0, 0, 1, 0, 0],
                [0, 0, 3, 8, 0],
                [-1, -3, 0, 3, 1],
                [0, -8, -3, 0, 0],
                [0, 0, -1, 0, 0],
            ],
            dtype=np.float64,
        )
        G4 = np.array(
            [
                [0, 1, 0, -1, 0],
                [0, 3, 0, -3, 0],
                [0, 8, 0, -8, 0],
                [0, 3, 0, -3, 0],
                [0, 1, 0, -1, 0],
            ],
            dtype=np.float64,
        )
        grad = np.zeros((img.shape[0], img.shape[1], 4), dtype=np.float64)
        grad[:, :, 0] = _jnd_conv(img, G1) / 16
        grad[:, :, 1] = _jnd_conv(img, G2) / 16
        grad[:, :, 2] = _jnd_conv(img, G3) / 16
        grad[:, :, 3] = _jnd_conv(img, G4) / 16
        max_g = np.max(np.abs(grad), axis=2)
        core = max_g[2:-2, 2:-2]
        edge_height = np.pad(core, ((2, 2), (2, 2)), mode="symmetric")
        return edge_height

    def _jnd_edge_protect(img):
        try:
            from skimage import feature as sk_feature, morphology as sk_morph
        except Exception as e:
            raise ImportError(
                "scikit-image is required for JND pass-rate: pip install scikit-image"
            ) from e

        edge_h = 60.0
        edge_height = _jnd_edge_height(img)
        max_val = float(np.max(edge_height) + eps)
        edge_threshold = min(edge_h / max_val, 0.8) if max_val > 0 else 0.8

        edge_region = sk_feature.canny(
            img,
            sigma=np.sqrt(2.0),
            low_threshold=0.4 * edge_threshold * 255.0,
            high_threshold=edge_threshold * 255.0,
        ).astype(np.float32)

        kernel = sk_morph.disk(3)
        img_edge = sk_morph.dilation(edge_region, kernel)
        img_supedge = 1.0 - 1.0 * img_edge.astype(np.float64)

        gaussian_kernel = _jnd_gkern(5, 0.8)
        edge_protect = _jnd_conv(img_supedge, gaussian_kernel)
        return edge_protect

    def _jnd_luminance_contrast(img):
        R = 2
        ker = np.ones((2 * R + 1, 2 * R + 1), dtype=np.float64) / float(
            (2 * R + 1) ** 2
        )
        mean_mask = _jnd_conv(img, ker)
        mean_img_sqr = mean_mask**2
        img_sqr = img**2
        mean_sqr_img = _jnd_conv(img_sqr, ker)
        var_mask = mean_sqr_img - mean_img_sqr
        var_mask[var_mask < 0] = 0
        valid = np.zeros_like(img)
        valid[R:-R, R:-R] = 1
        var_mask *= valid
        return np.sqrt(var_mask)

    def _jnd_cmlx_num(img):
        r = 1
        nb = r * 8
        otr = 6
        kx = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float64) / 3.0
        ky = kx.T

        sps = np.zeros((nb, 2))
        at = 2 * np.pi / nb
        idx = np.arange(nb)
        sps[:, 0] = -r * np.sin(idx * at)
        sps[:, 1] = r * np.cos(idx * at)

        imgd = np.pad(img, ((r, r), (r, r)), mode="symmetric")
        h, w = imgd.shape

        Gx = _jnd_conv(imgd, kx)
        Gy = _jnd_conv(imgd, ky)

        Cimg = np.sqrt(Gx**2 + Gy**2)
        Cvimg = np.zeros_like(imgd)
        Cvimg[Cimg >= 5] = 1

        Oimg = np.round(np.arctan2(Gy, Gx) / np.pi * 180.0 + eps)
        Oimg[Oimg > 90] -= 180
        Oimg[Oimg < -90] += 180
        Oimg += 90
        Oimg[Cvimg == 0] = 180 + 2 * otr

        Oimgc = Oimg[r:-r, r:-r]
        Cvimgc = Cvimg[r:-r, r:-r]

        Oimg_norm = np.round(Oimg / (2 * otr) + eps)
        Oimgc_norm = np.round(Oimgc / (2 * otr) + eps)

        onum = int(np.round(180 / (2 * otr)) + 1 + eps)
        ssr_val = np.zeros((h - 2 * r, w - 2 * r, onum + 1))

        for i in range(onum + 1):
            ssr_val[:, :, i] += Oimgc_norm == i

        for i in range(nb):
            dx = int(np.round(r + sps[i, 0]) + eps)
            dy = int(np.round(r + sps[i, 1]) + eps)
            Oimgn = Oimg_norm[dx : h - 2 * r + dx, dy : w - 2 * r + dy]
            for j in range(onum + 1):
                ssr_val[:, :, j] += Oimgn == j

        ssr_no_zero = ssr_val != 0
        cmlx = np.sum(ssr_no_zero, axis=2)

        cmlx[Cvimgc == 0] = 1
        cmlx[:r, :] = 1
        cmlx[-r:, :] = 1
        cmlx[:, :r] = 1
        cmlx[:, -r:] = 1
        return cmlx

    def _jnd_ori_cmlx(img):
        cmlx_map = _jnd_cmlx_num(img)
        r = 3
        sig = 1.0
        fker = _jnd_gkern(r, sig)
        return _jnd_conv(cmlx_map, fker)

    img = a

    jnd_LA = _jnd_func_bg_lum(img)

    L_c = _jnd_luminance_contrast(img)
    alpha = 0.115 * 16
    beta = 26.0
    jnd_LC = (alpha * np.power(L_c, 2.4)) / (np.power(L_c, 2.0) + beta**2)

    P_c = _jnd_ori_cmlx(img)
    a1, a2, a3 = 0.3, 2.7, 1.0
    C_t = (a1 * np.power(P_c, a2)) / (np.power(P_c, 2.0) + a3**2)
    jnd_PM = L_c * C_t

    edge_prot = _jnd_edge_protect(img)
    jnd_PM_p = jnd_PM * edge_prot

    jnd_VM = np.where(jnd_LC > jnd_PM_p, jnd_LC, jnd_PM_p)
    jnd_map = jnd_LA + jnd_VM - 0.3 * np.minimum(jnd_LA, jnd_VM)

    err = np.abs(a - b)
    pass_map = err < jnd_map
    return float(np.mean(pass_map))
