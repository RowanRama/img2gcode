"""Edge style: noise-robust line art from a flow-aligned difference of Gaussians.

The DoG is smoothed along the local edge direction (line-integral convolution
over a structure-tensor tangent field), so noise averages out while edges stay
sharp. Only the dark side of each edge responds, so an edge gives one line and
a thin dark line gives one line, not two. Thresholds are absolute, never percentiles.

Noise is removed where it is: non-local means runs with a per-pixel strength from
a tile-wise noise map (see _noise_map), so noise confined to shadows or a noisy
photo inside a clean border is denoised as hard as uniform noise would be.

Tuned on images/ (spec values in brackets): the DoG has no DC term, tau = 1 [0.98],
since 0.02 * L on flat areas hid light-on-white edges such as a yellow logo; LIC
runs over +-10 sigma [3] on a tangent field smoothed over 4 sigma [2], which turns
grass and hair into a few flowing strokes instead of many dashes; the base
threshold is 0.025 - 0.02 * detail [0.03 - ...]; sigma_resid (3x in the spec, 4x here)
is measured on the image's flattest fifth, see _residual_noise.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np

from ..geometry import deoverlap, polyline_length
from ..model import Artwork, Stroke
from ..raster import denoise, estimate_noise, fit_page, foreground_mask, palette_kmeans, sample_colors, working_image
from ..skeleton import skeleton_strokes

_INK = (30, 30, 30)  # single-tool palette
_IMMERKAER = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], np.float32)


def make(rgb, cfg) -> Artwork:
    lw = cfg.extrusion.effective_line_width
    n = max(1, cfg.tools.num_tools)
    detail = float(np.clip(cfg.edge.detail, 0.0, 1.0))
    H, W = rgb.shape[:2]
    width, height, s = fit_page(W, H, cfg)
    img, mmpp = working_image(rgb, s, lw / 3)
    lw_px = lw / mmpp
    mask = _line_map(_lightness(img, cfg.edge.contrast, lw_px), lw_px, detail)
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
    """Robust sigma (MAD) of *dog* over its flattest fifth, so edges and most texture do not count.

    This is not what keeps noise out: after _lightness has denoised each area to its own
    noise level, 4x it stays below the base threshold on noisy inputs (at most about 0.007
    against 0.015 at the default detail). It only engages on images textured all over, where
    even the flattest fifth is texture (crowd about 0.08, grass and feathers 0.02-0.03).
    """
    energy = cv2.GaussianBlur(dog * dog, (0, 0), 4 * lw_px)
    flat = dog[energy <= np.quantile(energy, 0.2)]
    return 1.4826 * float(np.median(np.abs(flat - np.median(flat))))


def _lightness(img: np.ndarray, contrast: float, lw_px: float) -> np.ndarray:
    """Lab lightness in [0, 1] (float32): CLAHE-equalised when *contrast* > 0, then denoised.

    Denoising after CLAHE measures (and removes) the noise CLAHE amplified; denoising
    the colour image first leaves dark areas noisy, since L* stretches the darks.
    The strength follows _noise_map: NLM runs at steps of x1.5 from the lowest to the
    highest sigma in the map, and each pixel interpolates between its two nearest steps.
    """
    L = cv2.cvtColor(img, cv2.COLOR_RGB2Lab)[..., 0]
    if contrast > 0:
        L = cv2.createCLAHE(clipLimit=contrast, tileGridSize=(8, 8)).apply(L)
    sig = _noise_map(L, lw_px)
    base = max(1.5, float(sig.min()))                   # denoise() leaves sigma <= 1.5 alone
    u = np.log(np.maximum(sig, base) / base) / np.log(1.5)
    out = sum(np.maximum(0.0, 1.0 - np.abs(u - k)) * denoise(L, base * 1.5 ** k).astype(np.float32)
              for k in range(int(np.ceil(u.max())) + 1))
    return (out / 255.0).astype(np.float32)


def _noise_map(L: np.ndarray, lw_px: float) -> np.ndarray:
    """Per-pixel noise sigma (0-255) of uint8 *L*: the global estimate, raised where a tile is noisier.

    Each tile (10 lw_px) gets a median-based Immerkaer estimate, which edges barely move.
    Fine oriented structure (engraving hatching, grass, hair) reads as noise at this
    scale too, so a tile counts only as far as its structure tensor is isotropic: median
    coherence <= 0.1 counts fully, >= 0.2 not at all (pure noise measures about 0.07,
    Duerer hatching 0.15-0.45). The tile grid is dilated by one tile, so the rim of a
    noisy area is covered, and interpolated bilinearly.
    """
    g = L.astype(np.float32)
    jxx, jxy, jyy = _structure_tensor(g, 2 * lw_px)
    coherence = np.sqrt((jxx - jyy) ** 2 + 4 * jxy ** 2) / (jxx + jyy + 1e-6)
    lap = np.abs(cv2.filter2D(g, -1, _IMMERKAER, borderType=cv2.BORDER_REFLECT))
    tile = 10 * lw_px
    sig = _tile_median(lap, tile) / (0.6745 * 6)       # median |N(0, 6 sigma)| = 0.6745 * 6 sigma
    sig *= np.clip((0.2 - _tile_median(coherence, tile)) / 0.1, 0.0, 1.0)
    sig = cv2.dilate(np.maximum(sig, estimate_noise(L)), np.ones((3, 3), np.uint8))
    return cv2.resize(sig, (L.shape[1], L.shape[0]), interpolation=cv2.INTER_LINEAR)


def _tile_median(a: np.ndarray, tile: float) -> np.ndarray:
    """Median of *a* over a grid of tiles about *tile* px square (float32 grid)."""
    H, W = a.shape
    ys = np.linspace(0, H, max(1, round(H / tile)) + 1).astype(int)
    xs = np.linspace(0, W, max(1, round(W / tile)) + 1).astype(int)
    return np.array([[np.median(a[y0:y1, x0:x1]) for x0, x1 in zip(xs, xs[1:])]
                     for y0, y1 in zip(ys, ys[1:])], np.float32)


def _structure_tensor(g: np.ndarray, sigma: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(Jxx, Jxy, Jyy): Sobel gradient products smoothed over *sigma*."""
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    return tuple(cv2.GaussianBlur(a, (0, 0), sigma) for a in (gx * gx, gx * gy, gy * gy))


def _tangent_field(g: np.ndarray, sigma: float) -> Tuple[np.ndarray, np.ndarray]:
    """Unit vectors along the edges: the structure tensor's minor eigenvector, smoothed over *sigma*."""
    jxx, jxy, jyy = _structure_tensor(g, sigma)
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
