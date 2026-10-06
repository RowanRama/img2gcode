"""Edge style: noise-robust line art from a flow-aligned difference of Gaussians.

The DoG is smoothed along the local edge direction (line-integral convolution
over a structure-tensor tangent field), so noise averages out while edges stay
sharp. Only the dark side of each edge responds, so an edge gives one line and
a thin dark line gives one line, not two. Thresholds are absolute and raised to
the measured residual noise, never percentiles.

Tuned on images/ (spec values in brackets): the DoG has no DC term, tau = 1 [0.98],
since 0.02 * L on flat areas hid light-on-white edges such as a yellow logo; LIC
runs over +-10 sigma [3] on a tangent field smoothed over 4 sigma [2], which turns
grass and hair into a few flowing strokes instead of many dashes; the base
threshold is 0.025 - 0.02 * detail [0.03 - ...].
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np

from ..geometry import deoverlap, polyline_length
from ..model import Artwork, Stroke
from ..raster import denoise, fit_page, foreground_mask, palette_kmeans, sample_colors, working_image
from ..skeleton import skeleton_strokes

_INK = (30, 30, 30)  # single-tool palette


def make(rgb, cfg) -> Artwork:
    lw = cfg.extrusion.effective_line_width
    n = max(1, cfg.tools.num_tools)
    detail = float(np.clip(cfg.edge.detail, 0.0, 1.0))
    H, W = rgb.shape[:2]
    width, height, s = fit_page(W, H, cfg)
    img, mmpp = working_image(rgb, s, lw / 3)
    lw_px = lw / mmpp
    mask = _line_map(_lightness(img, cfg.edge.contrast), lw_px, detail)
    if cfg.edge.debug_edges_path:
        p = Path(cfg.edge.debug_edges_path)
        cv2.imencode(p.suffix or ".png", mask.astype(np.uint8) * 255)[1].tofile(str(p))

    min_len = cfg.edge.min_length_factor * lw
    polys = skeleton_strokes(mask, mmpp, spur_mm=1.5 * lw, smooth_mm=0.5 * lw,
                             simplify_mm=0.05 * lw, min_length_mm=min_len)
    polys.sort(key=polyline_length, reverse=True)
    polys = deoverlap(polys, width, height, min_sep=0.9 * lw, min_length=min_len)

    if n == 1:
        palette = [_INK]
        tools = [0] * len(polys)
    else:
        img = denoise(img)
        palette = _palette(img, foreground_mask(img, cfg.image.white_threshold), n)
        tools = _nearest_tools(img, polys, mmpp, palette)
    strokes = [Stroke(p, t, "line") for p, t in zip(polys, tools)]
    return Artwork(width, height, [], strokes, palette, (W, H), rgb, "edge")


def _line_map(lum: np.ndarray, lw_px: float, detail: float) -> np.ndarray:
    """Binary map (bool) of the dark-side DoG response that survives hysteresis and cleanup."""
    sigma = max(0.7, lw_px * (0.6 - 0.35 * detail))
    dog = cv2.GaussianBlur(lum, (0, 0), sigma) - cv2.GaussianBlur(lum, (0, 0), 1.6 * sigma)
    dog = _lic(dog, *_tangent_field(lum, 4 * sigma), 10 * sigma)
    resp = np.maximum(0.0, -dog)
    hi = max(0.025 - 0.02 * detail, 4 * _residual_noise(dog, lw_px))
    lo = 0.4 * hi

    # Hysteresis: keep the weak components that hold a strong pixel and are not specks.
    _, labels, stats, _ = cv2.connectedComponentsWithStats((resp > lo).astype(np.uint8), connectivity=8)
    ok = np.zeros(len(stats), bool)
    ok[labels[resp > hi]] = True
    ok &= stats[:, cv2.CC_STAT_AREA] >= (2 * lw_px) ** 2
    ok[0] = False                                       # background
    keep = ok[labels].astype(np.uint8)

    k = max(1, int(round(lw_px)))
    keep = cv2.morphologyEx(keep, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    return keep > 0


def _residual_noise(dog: np.ndarray, lw_px: float) -> float:
    """Robust sigma (MAD) of *dog* over its flattest fifth, so edges and texture do not count as noise."""
    energy = cv2.GaussianBlur(dog * dog, (0, 0), 4 * lw_px)
    flat = dog[energy <= np.quantile(energy, 0.2)]
    return 1.4826 * float(np.median(np.abs(flat - np.median(flat))))


def _lightness(img: np.ndarray, contrast: float) -> np.ndarray:
    """Lab lightness in [0, 1] (float32): CLAHE-equalised when *contrast* > 0, then denoised.

    Denoising after CLAHE measures (and removes) the noise CLAHE amplified; denoising
    the colour image first leaves dark areas noisy, since L* stretches the darks.
    """
    L = cv2.cvtColor(img, cv2.COLOR_RGB2Lab)[..., 0]
    if contrast > 0:
        L = cv2.createCLAHE(clipLimit=contrast, tileGridSize=(8, 8)).apply(L)
    return denoise(L).astype(np.float32) / 255.0


def _tangent_field(g: np.ndarray, sigma: float) -> Tuple[np.ndarray, np.ndarray]:
    """Unit vectors along the edges: the structure tensor's minor eigenvector, smoothed over *sigma*."""
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    jxx = cv2.GaussianBlur(gx * gx, (0, 0), sigma)
    jxy = cv2.GaussianBlur(gx * gy, (0, 0), sigma)
    jyy = cv2.GaussianBlur(gy * gy, (0, 0), sigma)
    theta = 0.5 * np.arctan2(2 * jxy, jxx - jyy)       # gradient direction; the tangent is 90° off
    return (-np.sin(theta)).astype(np.float32), np.cos(theta).astype(np.float32)


def _lic(img: np.ndarray, tx: np.ndarray, ty: np.ndarray, length: float) -> np.ndarray:
    """Average *img* along the tangent field over ±*length* px (Gaussian weights)."""
    H, W = img.shape
    ys, xs = np.mgrid[0:H, 0:W].astype(np.float32)
    acc, wsum = img.copy(), np.float32(1.0)
    for sign in (1.0, -1.0):
        px, py = xs.copy(), ys.copy()
        dx, dy = tx * sign, ty * sign
        for i in range(1, int(np.ceil(length)) + 1):
            ltx = cv2.remap(tx, px, py, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            lty = cv2.remap(ty, px, py, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            flip = (ltx * dx + lty * dy) < 0                # keep walking the same way
            ltx[flip] *= -1
            lty[flip] *= -1
            dx, dy = ltx, lty
            px, py = px + dx, py + dy
            w = np.float32(np.exp(-(i * i) / (2 * (length / 2.0) ** 2)))
            acc += w * cv2.remap(img, px, py, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
            wsum += w
    return acc / wsum


def _palette(img: np.ndarray, fg: np.ndarray, n: int) -> List[Tuple[int, int, int]]:
    """Image colours sorted dark -> light, padded with greys to n entries."""
    pal, _ = palette_kmeans(img, fg, n)
    cols = [tuple(int(v) for v in c) for c in pal]
    cols += [(g, g, g) for g in np.linspace(40, 200, n).astype(int).tolist()[len(cols):]]
    lab = cv2.cvtColor(np.array([cols], np.uint8), cv2.COLOR_RGB2Lab)[0]
    return [cols[k] for k in np.argsort(lab[:, 0], kind="stable")]


def _nearest_tools(img, polys, mmpp, palette) -> List[int]:
    """Per stroke: the palette entry nearest (in Lab) to the median colour under it."""
    if not polys:
        return []
    med = np.array([np.median(sample_colors(img, p, mmpp), axis=0) for p in polys], np.uint8)
    lab = cv2.cvtColor(med[None], cv2.COLOR_RGB2Lab)[0].astype(np.float32)
    pal = cv2.cvtColor(np.array([palette], np.uint8), cv2.COLOR_RGB2Lab)[0].astype(np.float32)
    return ((lab[:, None] - pal[None]) ** 2).sum(axis=2).argmin(axis=1).tolist()
