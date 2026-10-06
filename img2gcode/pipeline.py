"""Conversion pipeline: input → toolpaths.svg → G-code.

The G-code is always made from the SVG as read back from disk, so the two
files can never disagree.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Tuple

from .config import Config
from .model import Artwork, Toolpaths
from .planner import plan
from .raster import load_rgb
from .styles import get_style
from .svg_input import load_svg_artwork
from .svg_io import is_toolpaths_svg, read_toolpaths_svg, write_toolpaths_svg
from .writer import PrintStats, compute_stats, write_gcode

Zone = Tuple[float, float, float, float]


@dataclass
class Result:
    toolpaths: Toolpaths
    svg_path: Optional[Path]      # None when no SVG was written (toolpaths SVG input)
    gcode_path: Optional[Path]    # None for SVG-only output
    stats: PrintStats
    ignored_shapes: int           # shapes in the SVG outside tool layers


def _is_toolpaths_input(path: Path) -> bool:
    return path.suffix.lower() == ".svg" and is_toolpaths_svg(path)


def make_artwork(input_path, cfg: Config, n_tools_explicit: bool = False) -> Optional[Artwork]:
    """Artwork for an image or artwork SVG; None for a toolpaths SVG."""
    path = Path(input_path)
    if _is_toolpaths_input(path):
        return None
    if path.suffix.lower() == ".svg":
        return load_svg_artwork(path, cfg, n_tools=cfg.tools.num_tools if n_tools_explicit else None)
    return get_style(cfg.style)(load_rgb(path), cfg)


def convert(input_path, output_path, cfg: Config, *, svg_path=None, svg_only: bool = False,
            exclusion_zones: Optional[Sequence[Zone]] = None, n_tools_explicit: bool = False,
            artwork: Optional[Artwork] = None) -> Result:
    """Write the toolpaths SVG, then the G-code made from it.

    An *output_path* ending in .svg (or *svg_only*) writes only the SVG. Otherwise
    the SVG goes to *svg_path*, or next to the G-code with the same stem.
    """
    input_path, output_path = Path(input_path), Path(output_path)
    if output_path.suffix.lower() == ".svg":
        svg_out, gcode_out = output_path, None
    else:
        svg_out = Path(svg_path) if svg_path else output_path.with_suffix(".svg")
        gcode_out = None if svg_only else output_path

    if _is_toolpaths_input(input_path):
        tp, ignored = read_toolpaths_svg(input_path)
        written_svg = None
        if gcode_out is None and svg_out.resolve() != input_path.resolve():
            svg_out.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(input_path, svg_out)
            written_svg = svg_out
    else:
        art = artwork if artwork is not None else make_artwork(input_path, cfg, n_tools_explicit)
        svg_out.parent.mkdir(parents=True, exist_ok=True)
        write_toolpaths_svg(plan(art, cfg, exclusion_zones), svg_out)
        tp, ignored = read_toolpaths_svg(svg_out)
        written_svg = svg_out

    stats = write_gcode(tp, cfg, gcode_out) if gcode_out else compute_stats(tp, cfg)
    return Result(tp, written_svg, gcode_out, stats, ignored)
