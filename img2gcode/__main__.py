"""CLI entry point.

Usage
-----
# Convert image to GCode (opens GUI by default):
python -m img2gcode -i logo.png -o output/logo.gcode

# Headless (no GUI):
python -m img2gcode -i logo.png -o output/logo.gcode --headless

# Custom config:
python -m img2gcode -i logo.png -o output/logo.gcode -c my_config.cfg

# Visualise an existing GCode file:
python -m img2gcode --visualise output/logo.gcode

# Shorthand script (if installed via pip):
img2gcode -i logo.png -o output/logo.gcode
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="img2gcode",
        description="Convert raster images to multi-tool GCode toolpaths.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "-i", "--input",
        metavar="IMAGE",
        help="Input image path (PNG, JPG, BMP, …).",
    )
    p.add_argument(
        "-o", "--output",
        metavar="GCODE",
        help="Output GCode file path.",
    )
    p.add_argument(
        "-c", "--config",
        metavar="CFG",
        default=None,
        help="Path to a .cfg config file (overrides defaults).",
    )
    p.add_argument(
        "--headless",
        action="store_true",
        help="Skip the GUI visualiser after generation.",
    )
    p.add_argument(
        "--visualise",
        metavar="GCODE",
        help="Open the GUI for an already-generated GCode file.",
    )
    p.add_argument(
        "--n-tools",
        type=int,
        default=None,
        metavar="N",
        help="Override the number of tools/colours (overrides config value).",
    )
    p.add_argument(
        "--select-roi",
        action="store_true",
        help="Open an interactive window after segmentation to draw rectangular "
             "regions that will be excluded from infill (perimeters are unaffected).",
    )
    p.add_argument(
        "-v", "--verbose",
        action="store_true",
        default=True,
        help="Print progress messages (default: on).",
    )
    p.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Suppress progress messages.",
    )
    return p


def main(argv=None):
    parser = _build_parser()
    args = parser.parse_args(argv)

    verbose = args.verbose and not args.quiet

    # ------------------------------------------------------------------ #
    # Mode: visualise existing GCode                                        #
    # ------------------------------------------------------------------ #
    if args.visualise:
        gcode_path = Path(args.visualise)
        if not gcode_path.exists():
            print(f"[error] File not found: {gcode_path}", file=sys.stderr)
            sys.exit(1)
        from .gcode_parser import parse_gcode
        from .gui import launch_gui
        if verbose:
            print(f"[img2gcode] Parsing '{gcode_path}'…")
        segments, meta = parse_gcode(gcode_path)
        if verbose:
            print(f"[img2gcode] {len(segments)} segments loaded.")
        launch_gui(segments, meta=meta, title=f"img2gcode – {gcode_path.name}")
        return

    # ------------------------------------------------------------------ #
    # Mode: convert image → GCode                                          #
    # ------------------------------------------------------------------ #
    if not args.input:
        parser.error("--input/-i is required unless using --visualise.")
    if not args.output:
        parser.error("--output/-o is required unless using --visualise.")

    image_path = Path(args.input)
    if not image_path.exists():
        print(f"[error] Image not found: {image_path}", file=sys.stderr)
        sys.exit(1)

    from .config import Config

    cfg = Config.load(args.config)

    # CLI override for num_tools
    if args.n_tools is not None:
        cfg.tools.num_tools = args.n_tools

    from .segmenter import segment
    from .toolpath import build_toolpaths
    from .writer import GCodeWriter

    if verbose:
        print(f"[img2gcode] Segmenting '{image_path}' into {cfg.tools.num_tools} tool(s)…")

    label_map, cluster_colors, fg_mask = segment(
        str(image_path),
        n_tools=cfg.tools.num_tools,
        white_threshold=cfg.image.white_threshold,
        min_cluster_area=cfg.image.min_cluster_area,
    )

    if verbose:
        for i, color in enumerate(cluster_colors):
            mask_px = int((label_map == i).sum())
            print(f"  Tool {i}: RGB≈{tuple(color)}  area={mask_px}px")

    exclusion_zones = None
    if args.select_roi:
        from .roi_selector import select_exclusion_zones
        if verbose:
            print("[img2gcode] Opening ROI selector — draw rectangles to exclude from infill…")
        exclusion_zones = select_exclusion_zones(label_map, cluster_colors)
        if verbose:
            print(f"[img2gcode] {len(exclusion_zones)} exclusion zone(s) selected.")

    if verbose:
        print("[img2gcode] Building toolpaths…")
    layers = build_toolpaths(label_map, cfg, exclusion_zones=exclusion_zones)

    if verbose:
        total_moves = sum(len(tl.moves) for tl in layers)
        print(f"[img2gcode] Generated {total_moves} moves across {len(layers)} tool-layers.")

    writer = GCodeWriter(cfg, label_map.shape)
    writer.write(layers, args.output)

    if not args.headless:
        from .gcode_parser import parse_gcode
        from .gui import launch_gui
        segments, meta = parse_gcode(args.output)
        launch_gui(segments, meta=meta, title=f"img2gcode – {image_path.name}")


if __name__ == "__main__":
    main()
