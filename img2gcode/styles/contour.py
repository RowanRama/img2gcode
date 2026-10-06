"""Contour style: topographic lines along iso-levels of smoothed luminance."""

from __future__ import annotations

from typing import List, Tuple

import cv2
import numpy as np
import shapely
from skimage.measure import find_contours

from ..geometry import deoverlap, polyline_length, simplify, smooth
from ..model import Artwork, Stroke
from ..raster import denoise, fit_page, foreground_mask, luminance, palette_kmeans, working_image

_MIN_STEP = 0.01    # levels closer than this (luminance in [0, 1]) count as one tone
_FLAT = 1e-4        # luminance change per pixel below which the blurred image counts as flat


def make(rgb, cfg) -> Artwork:
    lw = cfg.extrusion.effective_line_width
    n = max(1, cfg.tools.num_tools)
    H, W = rgb.shape[:2]
    width, height, s = fit_page(W, H, cfg)
    img, mmpp = working_image(rgb, s, lw / 3)
    lw_px = lw / mmpp
    img = denoise(img)
    lum = cv2.GaussianBlur(luminance(img), (0, 0), max(0.1, cfg.contour.smoothing * lw_px))
    fg = foreground_mask(img, cfg.image.white_threshold)
    r = max(1, int(round(lw_px)))
    near_fg = cv2.dilate(fg.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1,) * 2)) > 0

    # Flat areas (e.g. a logo's solid colours) would put levels exactly on their value,
    # where the iso-line traces rounding noise, so they do not vote on the levels.
    gy, gx = np.gradient(lum)
    levels = _levels(lum[fg & (np.hypot(gx, gy) > _FLAT)], cfg.contour.levels)
    polys: List[np.ndarray] = []
    tools: List[int] = []
    for i, level in enumerate(levels):             # dark -> light
        lines = []
        for c in find_contours(lum, level):
            # Keep or drop whole lines: clipping at the background edge leaves dashes.
            if near_fg[np.round(c[:, 0]).astype(int), np.round(c[:, 1]).astype(int)].mean() < 0.5:
                continue
            p = simplify(smooth((c[:, ::-1] + 0.5) * mmpp, 0.5 * lw), 0.05 * lw)
            if polyline_length(p) >= 3 * lw:
                lines.append(p)
        lines.sort(key=polyline_length, reverse=True)
        polys += lines
        tools += [i * n // len(levels)] * len(lines)

    strokes = _separate(polys, tools, width, height, lw)
    return Artwork(width, height, [], strokes, _palette(img, fg, n), (W, H), rgb, "contour")


def _levels(values: np.ndarray, count: int) -> List[float]:
    """Ascending levels at quantiles (i + 0.5) / count, dropping repeats; a flat image has none."""
    if values.size == 0:
        return []
    lo, hi = float(values.min()), float(values.max())
    out, last = [], lo
    for v in np.quantile(values, (np.arange(count) + 0.5) / count):
        if v - last >= _MIN_STEP and hi - v >= _MIN_STEP:
            out.append(float(v))
            last = v
    return out


def _separate(polys, tools, width, height, lw) -> List[Stroke]:
    """deoverlap in priority order; a line cut to less than half its length is dropped
    whole (its leftovers would be dashes) and the rest are separated again."""
    kept, owner = _deoverlap_owned(polys, width, height, lw)
    if not kept:
        return []
    got = np.bincount(owner, [polyline_length(k) for k in kept], len(polys))
    whole = got >= 0.5 * np.array([polyline_length(p) for p in polys])
    if not whole.all():
        polys = [p for p, w in zip(polys, whole) if w]
        tools = [t for t, w in zip(tools, whole) if w]
        kept, owner = _deoverlap_owned(polys, width, height, lw)
    return [Stroke(k, tools[o]) for k, o in zip(kept, owner)]


def _deoverlap_owned(polys, width, height, lw):
    """deoverlap, plus the index of the source line each kept piece came from."""
    kept = deoverlap(polys, width, height, min_sep=0.9 * lw, min_length=3 * lw)
    if not kept:
        return [], np.empty(0, int)
    # Every vertex deoverlap keeps lies on its source line, so the nearest source owns it.
    tree = shapely.STRtree([shapely.LineString(p) for p in polys])
    src = tree.query_nearest(shapely.points([k[len(k) // 2] for k in kept]), all_matches=False)
    owner = np.empty(len(kept), int)
    owner[src[0]] = src[1]
    return kept, owner


def _palette(img: np.ndarray, fg: np.ndarray, n: int) -> List[Tuple[int, int, int]]:
    """Image colours sorted dark -> light, padded with greys to n entries."""
    pal, _ = palette_kmeans(img, fg, n)
    cols = [tuple(int(v) for v in c) for c in pal]
    cols += [(g, g, g) for g in np.linspace(40, 200, n).astype(int).tolist()[len(cols):]]
    order = np.argsort(luminance(np.array([cols], np.uint8))[0], kind="stable")
    return [cols[k] for k in order]
