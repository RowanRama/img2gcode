"""Artwork → Toolpaths: perimeters, thin-feature centrelines, infill, ordering, layers."""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np
from shapely.geometry import box
from shapely.ops import unary_union

from .geometry import geometry_rings, hatch_lines, order_strokes, polyline_length
from .model import Artwork, Stroke, ToolPass, Toolpaths
from .skeleton import skeleton_strokes

Rect = Tuple[float, float, float, float]


def plan(art: Artwork, cfg, exclusion_zones: Optional[Sequence[Rect]] = None) -> Toolpaths:
    """Plan every (layer, tool) pass. Exclusion zones (page-mm rectangles) block infill only."""
    lw = cfg.extrusion.effective_line_width
    loops = cfg.layers.perimeter_loops
    density = max(1.0, min(100.0, cfg.infill.density)) / 100.0
    spacing = cfg.infill.line_spacing if cfg.infill.line_spacing > 0 else lw / density
    blocked = unary_union([box(*z) for z in exclusion_zones]) if exclusion_zones else None

    per_tool = []  # per tool: (perimeters, centrelines + artwork strokes, infill area or None)
    for t in range(art.n_tools):
        R = unary_union([r.shape for r in art.regions if r.tool == t]).buffer(0)
        perims: List[Stroke] = []
        lines: List[Stroke] = []
        inner = None
        if not R.is_empty:
            # Offsets are chained (erosions compose), which is ~10x faster than offsetting R
            # by the full distance each time on noisy outlines.
            core = R.buffer(-lw / 2)  # loop 0's path and the core of the lw-wide opening
            offset = core
            for k in range(loops):
                if k:
                    offset = offset.buffer(-lw)  # loop k sits (k + 0.5)·lw inside R
                perims += [Stroke(ring, t, "perimeter") for ring in geometry_rings(offset)]
            lines = _thin_features(R, core, lw, t)
            # (loops·lw + lw/2 − 0.15·lw) inside R: infill overlaps the last loop's bead by 0.15·lw
            inner = offset.buffer(-0.85 * lw) if loops > 0 else core
            if blocked is not None:
                inner = inner.difference(blocked)
        lines += [s for s in art.strokes if s.tool == t]
        per_tool.append((perims, lines, inner))

    passes = []
    by_parity = {}  # (tool, layer % 2) -> ordered strokes; layers of equal parity print the same
    for layer in range(cfg.layers.num_layers):
        angle = cfg.infill.angle + (90.0 if layer % 2 else 0.0)
        for t, (perims, lines, inner) in enumerate(per_tool):
            key = (t, layer % 2)
            if key not in by_parity:
                infill = [Stroke(p, t, "infill")
                          for p in hatch_lines(inner, spacing, angle, cfg.infill.connect_lines, 0.05 * lw)]
                by_parity[key] = _order((perims, lines, infill), 0.25 * lw)
            passes.append(ToolPass(t, layer, list(by_parity[key])))
    return Toolpaths(art.width_mm, art.height_mm, lw, cfg.layers.num_layers, passes,
                     list(art.palette), art.source_px, art.underlay, art.style)


def _order(groups, min_length: float) -> List[Stroke]:
    """Groups in the given order, each nearest-neighbour ordered from where the last one ended."""
    ordered: List[Stroke] = []
    cursor = None
    for group in groups:
        group = [s for s in group if polyline_length(s.points) >= min_length]
        if group:
            group = order_strokes(group, cursor)
            ordered += group
            cursor = tuple(group[-1].points[-1])
    return ordered


def _thin_features(R, core, lw: float, tool: int) -> List[Stroke]:
    """Centrelines of the parts of R too narrow for a perimeter (what an lw-wide opening removes).

    *core* is R.buffer(-lw/2). The opening is grown by 0.05·lw before subtracting: otherwise every
    slightly convex vertex of a jagged outline leaves a hairline sliver, and fillPoly draws chains
    of those as 1-px lines.
    """
    residue = R.difference(core.buffer(lw / 2 + 0.05 * lw))
    if residue.is_empty or residue.area < lw * lw / 4:
        return []
    px = lw / 4
    minx, miny, maxx, maxy = residue.bounds
    x0, y0 = minx - 2 * px, miny - 2 * px
    w = int(np.ceil((maxx - minx) / px)) + 5
    h = int(np.ceil((maxy - miny) / px)) + 5
    mask = np.zeros((h, w), np.uint8)
    # Pixel (c, r) is centred on (x0 + (c + 0.5) px, y0 + (r + 0.5) px); all rings in one call fill evenodd.
    rings = [np.round(((ring - (x0, y0)) / px - 0.5) * 16).astype(np.int32).reshape(-1, 1, 2)
             for ring in geometry_rings(residue)]
    cv2.fillPoly(mask, rings, 1, shift=4)
    strokes = skeleton_strokes(mask, px, spur_mm=lw, smooth_mm=lw / 2, simplify_mm=0.05 * lw, min_length_mm=lw)
    return [Stroke(p + (x0, y0), tool, "line") for p in strokes]
