"""Load an SVG file and produce the same (label_map, cluster_colors, fg_mask)
tuple as `segmenter.segment()` for raster images.

Each distinct fill colour in the SVG becomes its own tool — K-Means is skipped
because the SVG already has explicit per-shape colours.  The vector paths are
sampled at high resolution and rasterised with an even-odd fill rule so that
compound paths (e.g. letters with internal holes) are preserved.

The output is drop-in compatible with the rest of the pipeline: perimeter
tracing, infill, ROI exclusion, and GCode writing all work unchanged.
"""

from __future__ import annotations

from pathlib import Path as _Path
from typing import Dict, List, Tuple

import numpy as np
import cv2


def _subpath_points(subpath, scale: float, step_px: float) -> np.ndarray:
    """Sample a svgelements subpath into an Nx2 int array of pixel coords."""
    from svgelements import Path as SvgPath

    p = SvgPath(subpath)
    length = p.length(error=1e-3)
    if length <= 0:
        return np.empty((0, 2), dtype=np.int32)

    # Choose sample count so samples are roughly `step_px` pixels apart
    length_px = length * scale
    n = max(8, int(length_px / max(step_px, 0.5)))
    pts = np.empty((n + 1, 2), dtype=np.float32)
    for i in range(n + 1):
        pt = p.point(i / n)
        pts[i, 0] = float(pt.x)
        pts[i, 1] = float(pt.y)
    pts *= scale
    return pts.astype(np.int32)


def _shape_to_mask(shape, scale: float, out_w: int, out_h: int,
                   step_px: float = 1.0) -> np.ndarray:
    """Rasterise a single svgelements Shape into a binary mask.

    Compound paths are handled via even-odd alternation: the first subpath
    fills with 1, the second (if any) with 0 (treated as a hole), the third
    with 1 (island inside the hole), etc.  This matches SVG's `fill-rule:
    evenodd` and is a reasonable approximation of the default `nonzero` rule
    for typical logos.
    """
    from svgelements import Path as SvgPath

    path = SvgPath(shape)
    mask = np.zeros((out_h, out_w), dtype=np.uint8)

    fill_val = 1
    for sub in path.as_subpaths():
        pts = _subpath_points(sub, scale, step_px)
        if len(pts) >= 3:
            cv2.fillPoly(mask, [pts], int(fill_val))
        fill_val = 1 - fill_val  # alternate for even-odd
    return mask


def load_svg(
    svg_path: str | _Path,
    max_dim_px: int = 2000,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load an SVG and return (label_map, cluster_colors, fg_mask).

    Parameters
    ----------
    svg_path    : path to the .svg file
    max_dim_px  : the longest dimension of the rasterised output (pixels).
                  Higher values preserve more contour detail at the cost of
                  a bigger label map.

    Returns
    -------
    label_map      : (H, W) int32 — -1 = background, 0…n-1 = tool index
    cluster_colors : (n, 3) uint8 — one RGB row per tool/fill colour
    fg_mask        : (H, W) bool — foreground mask (label_map >= 0)
    """
    try:
        from svgelements import SVG, Shape, Color
    except ImportError as e:
        raise ImportError(
            "SVG support requires the 'svgelements' package. "
            "Install it with: pip install svgelements"
        ) from e

    svg = SVG.parse(str(svg_path))

    # Prefer viewBox, fall back to width/height attributes
    vb = svg.viewbox
    if vb is not None:
        svg_w, svg_h = float(vb.width), float(vb.height)
    else:
        svg_w, svg_h = float(svg.width or 0), float(svg.height or 0)

    if svg_w <= 0 or svg_h <= 0:
        raise ValueError(f"SVG has invalid dimensions: {svg_w}x{svg_h}")

    scale = max_dim_px / max(svg_w, svg_h)
    out_w = max(1, int(round(svg_w * scale)))
    out_h = max(1, int(round(svg_h * scale)))

    # Group all filled shapes by their resolved fill colour
    groups: Dict[str, List] = {}
    for elem in svg.elements():
        if not isinstance(elem, Shape):
            continue
        fill = getattr(elem, "fill", None)
        if fill is None:
            continue
        # Skip 'fill: none' and fully transparent fills
        if getattr(fill, "value", None) == "none":
            continue
        alpha = getattr(fill, "alpha", 255)
        if alpha == 0:
            continue
        try:
            key = fill.hex  # '#rrggbb'
        except Exception:
            continue
        groups.setdefault(key, []).append(elem)

    if not groups:
        raise ValueError(f"No filled shapes found in {svg_path}")

    label_map = np.full((out_h, out_w), -1, dtype=np.int32)
    cluster_colors = np.zeros((len(groups), 3), dtype=np.uint8)

    for i, (hex_color, elems) in enumerate(groups.items()):
        combined = np.zeros((out_h, out_w), dtype=np.uint8)
        for elem in elems:
            m = _shape_to_mask(elem, scale, out_w, out_h)
            combined = np.where(m > 0, m, combined)
        label_map[combined > 0] = i
        c = Color(hex_color)
        cluster_colors[i] = [c.red, c.green, c.blue]

    fg_mask = label_map >= 0
    return label_map, cluster_colors, fg_mask
