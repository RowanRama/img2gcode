"""High-level pipeline: image → GCode."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from .config import Config
from .segmenter import segment
from .toolpath import build_toolpaths
from .writer import GCodeWriter


def run(
    image_path: str,
    output_path: str,
    config_path: Optional[str] = None,
    headless: bool = False,
    verbose: bool = True,
) -> None:
    """Convert *image_path* to GCode and save to *output_path*.

    Parameters
    ----------
    image_path  : path to input image (PNG, JPG, BMP, …)
    output_path : path for the output .gcode file
    config_path : optional path to a .cfg override file
    headless    : if False, open the GUI after generation
    verbose     : print progress messages
    """
    if verbose:
        print(f"[img2gcode] Loading config…")
    cfg = Config.load(config_path)

    if verbose:
        print(f"[img2gcode] Segmenting '{image_path}' into {cfg.tools.num_tools} tool(s)…")
    label_map, cluster_colors, fg_mask = segment(
        image_path,
        n_tools=cfg.tools.num_tools,
        white_threshold=cfg.image.white_threshold,
        min_cluster_area=cfg.image.min_cluster_area,
    )

    if verbose:
        for i, color in enumerate(cluster_colors):
            mask_px = int((label_map == i).sum())
            print(f"  Tool {i}: RGB≈{tuple(color)}  area={mask_px}px")

    if verbose:
        print(f"[img2gcode] Building toolpaths…")
    layers = build_toolpaths(label_map, cfg)

    if verbose:
        total_moves = sum(len(tl.moves) for tl in layers)
        print(f"[img2gcode] Generated {total_moves} moves across {len(layers)} tool-layers.")

    writer = GCodeWriter(cfg, label_map.shape)
    writer.write(layers, output_path)

    if not headless:
        # Import here to avoid pulling in matplotlib when running headless
        from .gcode_parser import parse_gcode
        from .gui import launch_gui
        segments, meta = parse_gcode(output_path)
        launch_gui(segments, meta=meta, title=f"img2gcode – {Path(image_path).name}")
