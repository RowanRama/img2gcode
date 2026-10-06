"""toolpaths.svg: the editable intermediate between planning and G-code.

The file holds the exact toolpaths in page millimetres (viewBox user units),
one Inkscape layer per print layer with one sub-layer per tool, plus a
hidden travel layer and a faint source-image underlay. The reader accepts
the file after Inkscape edits: transforms, polylines turned into paths,
hidden layers and stray shapes outside tool groups.
"""

from __future__ import annotations

import base64
import math
import re
import xml.etree.ElementTree as ET
from pathlib import Path as FilePath
from xml.sax.saxutils import quoteattr

import cv2
import numpy as np
from svgelements import Circle, Close, Color, Ellipse, Line, Matrix, Move, Path, Rect

from .model import Stroke, ToolPass, Toolpaths

SVG_NS = "http://www.w3.org/2000/svg"
XLINK_NS = "http://www.w3.org/1999/xlink"
INKSCAPE_NS = "http://www.inkscape.org/namespaces/inkscape"
SODIPODI_NS = "http://sodipodi.sourceforge.net/DTD/sodipodi-0.dtd"

UNDERLAY_MAX_PX = 1200
_SHAPES = ("polyline", "polygon", "line", "path", "rect", "circle", "ellipse")
_NUMBER = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")


# ---------------------------------------------------------------- writer

def _num(v) -> str:
    """Attribute number: 3 decimals, trailing zeros dropped."""
    return f"{v:.3f}".rstrip("0").rstrip(".")


def _points(pts: np.ndarray) -> str:
    return " ".join(f"{x:.3f},{y:.3f}" for x, y in np.asarray(pts, float).tolist())


def _underlay_href(rgb: np.ndarray) -> str:
    h, w = rgb.shape[:2]
    scale = UNDERLAY_MAX_PX / max(h, w)
    if scale < 1:
        rgb = cv2.resize(rgb, (max(1, round(w * scale)), max(1, round(h * scale))), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 80])
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode("ascii")


def write_toolpaths_svg(tp: Toolpaths, path) -> None:
    w, h, lw = _num(tp.width_mm), _num(tp.height_mm), _num(tp.line_width_mm)
    n_tools = tp.n_tools
    src_w, src_h = (int(round(v)) for v in tp.source_px)
    out = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="{SVG_NS}" xmlns:xlink="{XLINK_NS}"',
        f'     xmlns:inkscape="{INKSCAPE_NS}"',
        f'     xmlns:sodipodi="{SODIPODI_NS}"',
        f'     width="{w}mm" height="{h}mm" viewBox="0 0 {w} {h}"',
        f'     data-img2gcode="toolpaths" data-line-width="{lw}" data-layers="{tp.n_layers}" data-tools="{n_tools}"',
        f'     data-source-px="{src_w}x{src_h}" data-style={quoteattr(tp.style)}>',
        f'  <rect data-role="background" width="{w}" height="{h}" fill="#ffffff"/>',
    ]
    if tp.underlay is not None:
        out += [
            '  <g inkscape:groupmode="layer" inkscape:label="Source image" sodipodi:insensitive="true"',
            '     data-role="source" opacity="0.25">',
            f'    <image x="0" y="0" width="{w}" height="{h}" preserveAspectRatio="none"'
            f' xlink:href="{_underlay_href(tp.underlay)}"/>',
            '  </g>',
        ]

    by_key = {(p.layer, p.tool): p.strokes for p in tp.passes}
    travel, cursor = [], None
    for layer in range(tp.n_layers):
        out.append(f'  <g inkscape:groupmode="layer" inkscape:label="Layer {layer}" data-layer="{layer}">')
        for tool in range(n_tools):
            colour = "#%02x%02x%02x" % tuple(int(c) for c in tp.palette[tool])
            out += [
                f'    <g inkscape:groupmode="layer" inkscape:label="Layer {layer} · T{tool}" data-tool="{tool}"',
                f'       fill="none" stroke="{colour}" stroke-width="{lw}" stroke-linecap="round"'
                ' stroke-linejoin="round">',
            ]
            for s in by_key.get((layer, tool), []):
                out.append(f'      <polyline data-kind={quoteattr(s.kind)} points="{_points(s.points)}"/>')
                if cursor is not None:
                    travel.append(f"M {_points(cursor)} L {_points(s.points[:1])}")
                cursor = s.points[-1:]
            out.append('    </g>')
        out.append('  </g>')

    out += [
        '  <g inkscape:groupmode="layer" inkscape:label="Travel moves" data-role="travel" style="display:none"',
        f'     fill="none" stroke="#888888" stroke-width="{_num(tp.line_width_mm / 6)}">',
    ]
    if travel:
        out.append(f'    <path d="{" ".join(travel)}"/>')
    out += ['  </g>', '</svg>', '']
    FilePath(path).write_text("\n".join(out), encoding="utf-8")


# ---------------------------------------------------------------- reader

def _local(tag) -> str:
    return tag[len(SVG_NS) + 2:] if tag.startswith(f"{{{SVG_NS}}}") else tag


def _style(elem) -> dict:
    items = (item.split(":", 1) for item in (elem.get("style") or "").split(";") if ":" in item)
    return {k.strip(): v.strip() for k, v in items}


def _hidden(elem, style: dict) -> bool:
    return (style.get("display", elem.get("display")) == "none"
            or style.get("visibility", elem.get("visibility")) in ("hidden", "collapse"))


def _apply(m: Matrix, pts: np.ndarray) -> np.ndarray:
    return pts @ np.array([[m.a, m.b], [m.c, m.d]]) + [m.e, m.f]


def _attr(elem, name: str) -> float:
    return float(elem.get(name) or 0)


def _path_points(path: Path, step: float) -> list:
    """One (N, 2) array per subpath; curves sampled at <= step spacing."""
    subpaths, cur = [], None
    for seg in path:
        if isinstance(seg, Move):
            cur = [(seg.end.x, seg.end.y)]
            subpaths.append(cur)
        elif isinstance(seg, Close):
            if cur[-1] != (seg.end.x, seg.end.y):
                cur.append((seg.end.x, seg.end.y))
        elif isinstance(seg, Line):
            cur.append((seg.end.x, seg.end.y))
        else:  # CubicBezier, QuadraticBezier, Arc
            n = max(1, math.ceil(seg.length() / step))
            sampled = seg.npoint(np.linspace(0, 1, n + 1)[1:])
            cur.extend(map(tuple, sampled[:-1].tolist()))
            cur.append((seg.end.x, seg.end.y))
    return [np.array(p, float) for p in subpaths]


def _shape_points(elem, tag: str, m: Matrix, step: float) -> list:
    if tag in ("polyline", "polygon"):
        vals = [float(v) for v in _NUMBER.findall(elem.get("points") or "")]
        pts = np.array(vals[: len(vals) // 2 * 2], float).reshape(-1, 2)
        if tag == "polygon" and len(pts) and not np.array_equal(pts[0], pts[-1]):
            pts = np.vstack([pts, pts[:1]])
        return [_apply(m, pts)]
    if tag == "line":
        pts = np.array([[_attr(elem, "x1"), _attr(elem, "y1")], [_attr(elem, "x2"), _attr(elem, "y2")]])
        return [_apply(m, pts)]
    if tag == "path":
        shape = Path(elem.get("d") or "")
    else:
        keys = {"rect": ("x", "y", "width", "height", "rx", "ry"), "circle": ("cx", "cy", "r"),
                "ellipse": ("cx", "cy", "rx", "ry")}[tag]
        cls = {"rect": Rect, "circle": Circle, "ellipse": Ellipse}[tag]
        shape = Path(cls(**{k: elem.get(k) for k in keys if elem.get(k) is not None}))
    return _path_points(abs(shape * m), step)


def _read_underlay(root):
    for group in root.iter(f"{{{SVG_NS}}}g"):
        if group.get("data-role") != "source":
            continue
        for img in group.iter(f"{{{SVG_NS}}}image"):
            href = img.get(f"{{{XLINK_NS}}}href") or img.get("href") or ""
            if "base64," in href:
                data = np.frombuffer(base64.b64decode(href.split("base64,", 1)[1]), np.uint8)
                bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
                if bgr is not None:
                    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return None


def read_toolpaths_svg(path) -> tuple[Toolpaths, int]:
    """Read a toolpaths SVG. Returns (toolpaths, number of shapes ignored
    because they sit outside a tool group)."""
    root = ET.parse(path).getroot()
    vb = [float(v) for v in root.get("viewBox").replace(",", " ").split()]
    width, height = vb[2], vb[3]
    lw = float(root.get("data-line-width", "0") or 0)
    step = (lw or 4.0) / 8
    n_layers = int(root.get("data-layers", "1")); n_tools = int(root.get("data-tools", "1"))
    src = root.get("data-source-px", "0x0").lower().split("x")
    found = {}            # (layer, tool) -> list[Stroke]
    palette = {}          # tool -> RGB
    ignored = 0

    def walk(elem, matrix, layer, tool):
        nonlocal ignored
        for child in elem:
            tag = _local(child.tag)
            style = _style(child)
            if tag == "g" and child.get("data-tool") is not None:
                colour = Color(style.get("stroke") or child.get("stroke"))
                if colour.value is not None:
                    palette.setdefault(int(child.get("data-tool")), (colour.red, colour.green, colour.blue))
            if child.get("data-role") is not None or _hidden(child, style):
                continue
            m = Matrix(child.get("transform")) * matrix if child.get("transform") else matrix
            if tag == "g":
                walk(child, m,
                     int(child.get("data-layer")) if child.get("data-layer") is not None else layer,
                     int(child.get("data-tool")) if child.get("data-tool") is not None else tool)
            elif tag in _SHAPES:
                if layer is None or tool is None:
                    ignored += 1
                    continue
                kind = child.get("data-kind") or "line"
                found.setdefault((layer, tool), []).extend(
                    Stroke(pts, tool, kind) for pts in _shape_points(child, tag, m, step) if len(pts) >= 2)

    walk(root, Matrix(f"translate({-vb[0]},{-vb[1]})"), None, None)

    n_layers = max([n_layers] + [l + 1 for l, _ in found]); n_tools = max([n_tools] + [t + 1 for _, t in found])
    passes = [ToolPass(t, l, found.get((l, t), [])) for l in range(n_layers) for t in range(n_tools)]
    pal = [palette.get(t, (128, 128, 128)) for t in range(n_tools)]
    return Toolpaths(width, height, lw, n_layers, passes, pal, (int(src[0]), int(src[1])), _read_underlay(root),
                     root.get("data-style", "")), ignored


def is_toolpaths_svg(path) -> bool:
    try:
        with open(path, "rb") as fh:
            for _, elem in ET.iterparse(fh, events=("start",)):
                return elem.get("data-img2gcode") == "toolpaths"
    except ET.ParseError:
        return False
    return False
