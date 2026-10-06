"""Artwork SVG (logos, flags, icons) to Artwork for the planner.

Shapes are walked in paint order: each filled shape covers what was painted
before it, white counts as paper, and every remaining colour becomes a tool.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from shapely.geometry import LineString, box
from shapely.ops import unary_union
from svgelements import SVG, Close, Color, Line, Move, Path, Shape

from .geometry import polyline_length, rings_to_geometry
from .model import RGB, Artwork, Region, Stroke

STEP_MM = 0.25            # curve sampling on the page
MERGE_DELTA_E = 5.0       # colours closer than this (Lab) are one tool
GREY = (128, 128, 128)    # paint we cannot resolve
_URL = re.compile(r"url\(\s*['\"]?#([^'\")\s]+)")
_XLINK_HREF = "{http://www.w3.org/1999/xlink}href"


def load_svg_artwork(path, cfg, n_tools: Optional[int] = None) -> Artwork:
    """Read an artwork SVG fitted into the print area.

    One tool per distinct colour; with *n_tools*, colours are merged down to that many.
    """
    svg = SVG.parse(str(path), reify=True)     # shapes come out in width x height space
    doc_w, doc_h = float(svg.width), float(svg.height)
    k = min(cfg.machine.print_width / doc_w, cfg.machine.print_height / doc_h)
    width, height = doc_w * k, doc_h * k
    page = box(0, 0, width, height)
    lw = cfg.extrusion.effective_line_width
    gradients = _gradient_colours(path)

    paint: List[list] = []                     # [geometry, rgb] in paint order
    lines: List[Tuple[np.ndarray, RGB]] = []
    for elem in svg.elements():
        if not isinstance(elem, Shape) or _hidden(elem):
            continue
        subs = _subpaths(elem, k)
        fill = _paint(elem, "fill", gradients)
        if fill is not None:
            rule = str(elem.values.get("fill-rule", "nonzero")).strip()
            _cover(paint, rings_to_geometry(subs, rule), fill)
        stroke = _paint(elem, "stroke", gradients)
        stroke_width = (elem.implicit_stroke_width or 0.0) * k    # through any transform reify left
        if stroke is not None and stroke_width > 0:
            if stroke_width >= 1.5 * lw:
                _cover(paint, _stroke_outline(subs, stroke_width, elem.values), stroke)
            else:
                lines.extend((s, stroke) for s in subs)

    white = cfg.image.white_threshold
    paint = [(g.intersection(page), c) for g, c in paint if min(c) < white]   # paper only occludes
    paint = [(g, c) for g, c in paint if g.area > 0]
    lines = [(p, c) for p, c in lines if min(c) < white]
    if not paint and not lines:
        raise ValueError(f"No printable shapes in {path}")

    colours: List[RGB] = []                    # first seen in paint order
    weight: Dict[RGB, float] = {}
    for c, w in [(c, g.area) for g, c in paint] + [(c, polyline_length(p) * lw) for p, c in lines]:
        if c not in weight:
            colours.append(c)
            weight[c] = 0.0
        weight[c] += w
    palette, tool_of = _assign_tools(colours, [weight[c] for c in colours], n_tools)

    regions = []
    for t in range(len(palette)):
        geoms = [g for g, c in paint if tool_of[c] == t]
        if geoms:
            regions.append(Region(shape=unary_union(geoms).buffer(0), tool=t))
    vb = svg.viewbox
    src_w, src_h = (vb.width, vb.height) if vb is not None else (doc_w, doc_h)
    return Artwork(
        width_mm=width, height_mm=height,
        regions=regions,
        strokes=[Stroke(points=p, tool=tool_of[c], kind="line") for p, c in lines],
        palette=palette,
        source_px=(max(1, int(round(src_w))), max(1, int(round(src_h)))),
    )


def _hidden(elem) -> bool:
    v = elem.values
    return (v.get("display") == "none" or v.get("visibility") in ("hidden", "collapse")
            or _number(v.get("opacity"), 1.0) <= 0)


def _number(value, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _paint(elem, attr: str, gradients: Dict[str, RGB]) -> Optional[RGB]:
    """RGB of the shape's fill or stroke, or None when it paints nothing."""
    if _number(elem.values.get(f"{attr}-opacity"), 1.0) <= 0:
        return None
    url = _URL.search(str(elem.values.get(attr, "")))
    if url:
        return gradients.get(url.group(1), GREY)
    c = elem.fill if attr == "fill" else elem.stroke
    if c is None or c.value is None or c.alpha == 0:
        return None
    return (c.red, c.green, c.blue)


def _gradient_colours(path) -> Dict[str, RGB]:
    """Mean stop colour of every gradient by id (stops may be inherited through href)."""
    ids = {e.get("id"): e for e in ET.parse(str(path)).iter() if e.get("id")}

    def stops(node, depth=0):
        found = [s for s in node if _local(s.tag) == "stop"]
        ref = (node.get("href") or node.get(_XLINK_HREF) or "").lstrip("#")
        if found or depth > 8 or ref not in ids:
            return found
        return stops(ids[ref], depth + 1)

    out: Dict[str, RGB] = {}
    for gid, node in ids.items():
        if _local(node.tag) in ("linearGradient", "radialGradient"):
            cols = [c for c in (_stop_colour(s) for s in stops(node)) if c is not None]
            if cols:
                out[gid] = tuple(int(round(v)) for v in np.mean(cols, axis=0))
    return out


def _local(tag) -> str:
    return str(tag).rsplit("}", 1)[-1]


def _stop_colour(stop) -> Optional[RGB]:
    style = dict(kv.split(":", 1) for kv in (stop.get("style") or "").split(";") if ":" in kv)
    style = {k.strip(): v.strip() for k, v in style.items()}
    c = Color(style.get("stop-color", stop.get("stop-color", "black")))
    return None if c.value is None else (c.red, c.green, c.blue)


def _subpaths(elem, k: float) -> List[np.ndarray]:
    """Each subpath in page mm: line ends exact, curves sampled at most STEP_MM apart."""
    out = []
    path = Path(elem)
    path.reify()            # rotated, skewed or mirrored rects/ellipses keep their transform after parsing
    for sub in path.as_subpaths():
        pts = []
        for seg in sub:
            if isinstance(seg, (Move, Line, Close)):
                pts.append([[seg.end.x, seg.end.y]])
            else:
                rough = polyline_length(seg.npoint(np.linspace(0, 1, 17))) * k
                n = max(1, int(np.ceil(rough / STEP_MM)))
                pts.append(np.asarray(seg.npoint(np.linspace(0, 1, n + 1)[1:])))
        p = np.vstack(pts).astype(np.float64) * k
        p = p[np.r_[True, np.any(np.diff(p, axis=0) != 0, axis=1)]]   # drop repeated points
        if len(p) >= 2 and np.isfinite(p).all():
            out.append(p)
    return out


def _stroke_outline(subs: List[np.ndarray], width: float, values):
    """The area a stroke paints, with the SVG's own caps and joins (default butt and miter)."""
    cap = {"round": "round", "square": "square"}.get(str(values.get("stroke-linecap", "")).strip(), "flat")
    join = {"round": "round", "bevel": "bevel"}.get(str(values.get("stroke-linejoin", "")).strip(), "mitre")
    limit = max(1.0, _number(values.get("stroke-miterlimit"), 4.0))
    return unary_union([LineString(s).buffer(width / 2, cap_style=cap, join_style=join, mitre_limit=limit)
                        for s in subs])


def _cover(paint: List[list], geom, rgb: RGB) -> None:
    """Paint *geom* on top: cut it out of everything painted earlier."""
    if geom.is_empty:
        return
    for entry in paint:
        if entry[0].intersects(geom):
            entry[0] = entry[0].difference(geom)
    paint.append([geom, rgb])


def _lab(rgb) -> np.ndarray:
    a = np.asarray(rgb, np.float32).reshape(-1, 1, 3) / 255.0
    return cv2.cvtColor(a, cv2.COLOR_RGB2Lab).reshape(-1, 3)


def _assign_tools(colours: List[RGB], weights: List[float],
                  n_tools: Optional[int]) -> Tuple[List[RGB], Dict[RGB, int]]:
    """Group colours into tools: palette (dark to light) and the tool of each colour."""
    lab = _lab(colours)
    reps: List[int] = []                       # first colour seen of each group
    group = np.empty(len(colours), int)
    for i in range(len(colours)):
        near = [j for j, r in enumerate(reps) if np.linalg.norm(lab[i] - lab[r]) < MERGE_DELTA_E]
        group[i] = near[0] if near else len(reps)
        if not near:
            reps.append(i)
    palette_lab = lab[reps]
    palette = [colours[r] for r in reps]
    if n_tools is not None and len(reps) > n_tools:
        w = np.bincount(group, weights=weights)
        samples = np.repeat(palette_lab, np.maximum(1, np.round(5000 * w / w.sum())).astype(int), axis=0)
        cv2.setRNGSeed(0)
        crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.01)
        _, _, centres = cv2.kmeans(samples, n_tools, None, crit, 4, cv2.KMEANS_PP_CENTERS)
        label = np.argmin(((palette_lab[:, None] - centres[None]) ** 2).sum(-1), axis=1)
        used = np.unique(label)
        group = np.searchsorted(used, label)[group]
        palette_lab = centres[used]
        rgb = cv2.cvtColor(palette_lab.reshape(-1, 1, 3).astype(np.float32), cv2.COLOR_Lab2RGB)
        palette = [tuple(int(v) for v in row) for row in np.clip(np.round(rgb.reshape(-1, 3) * 255), 0, 255)]
    order = np.argsort(palette_lab[:, 0], kind="stable")
    rank = np.empty(len(order), int)
    rank[order] = np.arange(len(order))
    return [palette[i] for i in order], {c: int(rank[group[i]]) for i, c in enumerate(colours)}
