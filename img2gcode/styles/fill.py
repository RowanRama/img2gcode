"""fill: flat colour regions, robust to noise.

Denoise and flatten texture, cluster colours in Lab, smooth the label map by
majority vote, turn faint colour components back into paper, then reassign
every blob or hole smaller than the minimum feature to its nearest surviving
neighbour before vectorizing.

Faint components (not in the spec's fill section yet): a colour component
whose mean depth below ``image.white_threshold`` (threshold minus the darkest
channel) is under the measured noise sigma becomes paper. Noisy light paper
dips just under the threshold in blotches that survive majority smoothing and
the minimum feature; real ink is far deeper. On clean images sigma is about 0,
so the step does nothing.
"""

from __future__ import annotations

import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt
from shapely.geometry import box
from shapely.ops import unary_union

from ..model import Artwork, Region
from ..raster import (
    denoise, estimate_noise, fit_page, foreground_mask, mask_to_geometry, palette_kmeans, working_image,
)

EMPTY_PALETTE = [(32, 32, 32)]


def make(rgb, cfg) -> Artwork:
    H, W = rgb.shape[:2]
    width, height, s = fit_page(W, H, cfg)
    lw = cfg.extrusion.effective_line_width
    img, mmpp = working_image(rgb, s, lw / 4)
    lw_px = lw / mmpp

    noise = estimate_noise(img)
    img = denoise(img, noise)
    img = cv2.pyrMeanShiftFiltering(img, sp=max(2, int(round(lw_px))), sr=18)

    white = cfg.image.white_threshold
    fg = foreground_mask(img, white)
    if not fg.any():
        return Artwork(width, height, [], [], list(EMPTY_PALETTE), (W, H), rgb, "fill")

    palette, labels = palette_kmeans(img, fg, cfg.tools.num_tools)
    labels = _majority(labels, len(palette), cfg.fill.smoothing * lw_px)
    labels = _drop_faint(labels, len(palette), white - img.min(axis=2).astype(np.float32), noise)
    labels = _drop_small(labels, len(palette), (cfg.fill.min_feature_factor * lw_px) ** 2)

    page = box(0, 0, width, height)
    regions = []
    for t in range(len(palette)):
        geom = mask_to_geometry(labels == t, mmpp, simplify_mm=0.1 * lw,
                                min_area_mm2=(cfg.fill.min_feature_factor * lw) ** 2)
        geom = _polygonal(geom.intersection(page))
        if not geom.is_empty:
            regions.append(Region(geom, t))
    return Artwork(width, height, regions, [], [tuple(map(int, c)) for c in palette], (W, H), rgb, "fill")


def _majority(labels, k, sigma):
    """Blur each class's one-hot map (background = -1 included) and take the argmax."""
    if sigma <= 0:
        return labels
    votes = np.stack([cv2.GaussianBlur((labels == c).astype(np.float32), (0, 0), sigma)
                      for c in range(-1, k)])
    return (votes.argmax(axis=0) - 1).astype(np.int32)


def _drop_faint(labels, k, depth, margin):
    """Make paper of every colour component whose mean *depth* below the white threshold is under *margin*.

    On a noisy light background the paper dips just under the threshold in
    blotches; real ink is far darker than the noise level.
    """
    for c in range(k):
        n, cc = cv2.connectedComponents((labels == c).astype(np.uint8), connectivity=4)
        mean = np.bincount(cc.ravel(), depth.ravel(), n) / np.bincount(cc.ravel(), minlength=n)
        labels[np.r_[False, mean[1:] < margin][cc]] = -1
    return labels


def _drop_small(labels, k, min_px):
    """Give every component (of any class) smaller than *min_px* the label of its nearest kept pixel."""
    unknown = np.zeros(labels.shape, bool)
    for c in range(-1, k):
        n, cc, stats, _ = cv2.connectedComponentsWithStats((labels == c).astype(np.uint8), connectivity=4)
        small = np.r_[False, stats[1:, cv2.CC_STAT_AREA] < min_px]
        unknown |= small[cc]
    if not unknown.any() or unknown.all():
        return labels
    _, (rows, cols) = distance_transform_edt(unknown, return_indices=True)
    return labels[rows, cols]


def _polygonal(geom):
    """Drop any points or lines a clip left behind."""
    if geom.geom_type in ("Polygon", "MultiPolygon"):
        return geom
    return unary_union([g for g in getattr(geom, "geoms", []) if g.geom_type in ("Polygon", "MultiPolygon")])
