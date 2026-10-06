"""Engraving-style hatching: darker tones get more line directions.

Foreground luminance is split into `hatch.bands` tone bands. Nested masks,
lightest first, are each hatched in their own direction, so the darkest band
ends up cross-hatched in every direction.
"""

from __future__ import annotations

import cv2
import numpy as np

from ..geometry import hatch_lines
from ..model import Artwork, Stroke
from ..raster import (
    denoise, fit_page, foreground_mask, luminance, mask_to_geometry, palette_kmeans,
    tone_levels, working_image,
)


def make(rgb, cfg) -> Artwork:
    lw = cfg.extrusion.effective_line_width
    n = cfg.tools.num_tools
    H, W = rgb.shape[:2]
    width, height, mm_per_src_px = fit_page(W, H, cfg)
    img, mmpp = working_image(rgb, mm_per_src_px, lw / 3)
    img = denoise(img)
    lum = cv2.GaussianBlur(luminance(img), (0, 0), lw / mmpp)
    fg = foreground_mask(img, cfg.image.white_threshold)

    palette = [tuple(int(v) for v in c) for c in palette_kmeans(img, fg, n)[0]]
    palette += [(g, g, g) for g in np.linspace(40, 200, n).astype(int).tolist()[len(palette):]]

    bands = cfg.hatch.bands
    # A level often lands on a flat-colour plateau (logos, flags). Resampling and
    # blur leave values a hair either side of it, so without the tolerance the
    # plateau would split into speckle instead of falling wholly inside the mask.
    t = tone_levels(lum, fg, bands) + 1e-3
    angles = [float(a) for a in cfg.hatch.angles.split(",") if a.strip()]
    strokes = []
    for a in range(min(bands - 1, len(angles))):
        mask = fg & (lum <= t[bands - 2 - a])  # a = 0 is the lightest, most inclusive mask
        geom = mask_to_geometry(mask, mmpp, simplify_mm=0.1 * lw, min_area_mm2=(1.5 * lw) ** 2)
        for line in hatch_lines(geom, cfg.hatch.spacing * lw, angles[a], connect=False):
            strokes.append(Stroke(line, a % n, "line"))
    return Artwork(width, height, [], strokes, palette, (W, H), rgb, "hatch")
