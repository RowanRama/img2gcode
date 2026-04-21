"""Minimal GCode parser for the visualiser.

Produces a list of segments suitable for rendering:
  [{'type': 'travel'|'print', 'tool': int, 'layer': int,
    'x0': float, 'y0': float, 'x1': float, 'y1': float}, ...]

Also returns metadata dict with keys like 'line_width_mm', 'print_width', etc.
parsed from the header comments written by the GCode writer.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional


_G1_RE = re.compile(
    r"G1"
    r"(?:\s+X(-?[\d.]+))?"
    r"(?:\s+Y(-?[\d.]+))?"
    r"(?:\s+Z(-?[\d.]+))?"
    r"(?:\s+E(-?[\d.]+))?"
    r"(?:\s+F(-?[\d.]+))?",
    re.IGNORECASE,
)
_T_RE = re.compile(r"^T(\d+)", re.IGNORECASE)
_LAYER_Z_RE = re.compile(r"^;Z:([\d.]+)")
# Header comment patterns written by the GCode writer
_SCALE_RE = re.compile(r"; image size:.*?scale=([\d.]+)")
_SIZE_RE = re.compile(r"; image size:.*?→\s*([\d.]+)x([\d.]+)mm")
_LINEWIDTH_RE = re.compile(r"; line_width:([\d.]+)")


def parse_gcode(path: str | Path) -> tuple[List[Dict], Dict]:
    """Parse a GCode file and return (segments, metadata).

    metadata keys: 'scale', 'print_width_mm', 'print_height_mm',
                   'line_width_mm', 'n_layers'
    """
    segments: List[Dict] = []
    meta: Dict = {
        "scale": None,
        "print_width_mm": None,
        "print_height_mm": None,
        "line_width_mm": 0.4,   # fallback default
    }

    cur_x = 0.0
    cur_y = 0.0
    cur_z = 0.0
    cur_tool = 0
    cur_layer = -1
    layer_z_map: Dict[float, int] = {}

    lines = Path(path).read_text(errors="replace").splitlines()

    for raw_line in lines:
        line = raw_line.strip()

        # Parse header metadata from comments
        m = _SCALE_RE.match(line)
        if m:
            meta["scale"] = float(m.group(1))

        m = _SIZE_RE.match(line)
        if m:
            meta["print_width_mm"] = float(m.group(1))
            meta["print_height_mm"] = float(m.group(2))

        m = _LINEWIDTH_RE.match(line)
        if m:
            meta["line_width_mm"] = float(m.group(1))

        # Layer annotation
        m = _LAYER_Z_RE.match(line)
        if m:
            z_val = float(m.group(1))
            if z_val not in layer_z_map:
                cur_layer += 1
                layer_z_map[z_val] = cur_layer
            else:
                cur_layer = layer_z_map[z_val]
            continue

        # Tool change
        m = _T_RE.match(line)
        if m:
            cur_tool = int(m.group(1))
            continue

        # G1 move
        if line.upper().startswith("G1"):
            m = _G1_RE.match(line)
            if not m:
                continue
            x_s, y_s, z_s, e_s, f_s = m.groups()

            new_x = float(x_s) if x_s is not None else cur_x
            new_y = float(y_s) if y_s is not None else cur_y
            new_z = float(z_s) if z_s is not None else cur_z

            is_print = e_s is not None and float(e_s) > 0
            move_type = "print" if is_print else "travel"

            if (new_x != cur_x or new_y != cur_y) and cur_layer >= 0:
                segments.append({
                    "type": move_type,
                    "tool": cur_tool,
                    "layer": cur_layer,
                    "x0": cur_x,
                    "y0": cur_y,
                    "x1": new_x,
                    "y1": new_y,
                    "z": new_z,
                })

            cur_x, cur_y, cur_z = new_x, new_y, new_z

    meta["n_layers"] = cur_layer + 1
    return segments, meta
