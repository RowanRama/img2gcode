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

import cv2


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
        "--mode",
        choices=("fill", "edge"),
        default=None,
        help="Generation mode: 'fill' (default — segment + perimeters + infill) "
             "or 'edge' (Canny edge tracing for portraits / line drawings).",
    )
    p.add_argument(
        "--debug-edges",
        metavar="PATH",
        default=None,
        help="Edge mode only: save the pre-trace edge map (post CLAHE / "
             "multi-scale Canny / fg-mask) as a PNG at PATH. Useful for "
             "tuning detection params before running the full pipeline.",
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
    if args.mode is not None:
        cfg.mode = args.mode
    if args.debug_edges is not None:
        cfg.edge.debug_edges_path = args.debug_edges

    from .writer import GCodeWriter

    is_svg = image_path.suffix.lower() == ".svg"
    if is_svg:
        from .svg_loader import load_svg
        if verbose:
            print(f"[img2gcode] Loading SVG '{image_path}' (one tool per fill colour)…")
        label_map, cluster_colors, fg_mask = load_svg(str(image_path))
        # SVG defines its own set of tools via distinct fill colours — override cfg
        cfg.tools.num_tools = len(cluster_colors)
    else:
        from .segmenter import segment
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
    if args.select_roi and cfg.mode == "fill":
        from .roi_selector import select_exclusion_zones
        if verbose:
            print("[img2gcode] Opening ROI selector — draw rectangles to exclude from infill…")
        exclusion_zones = select_exclusion_zones(label_map, cluster_colors)
        if verbose:
            print(f"[img2gcode] {len(exclusion_zones)} exclusion zone(s) selected.")

    if verbose:
        print(f"[img2gcode] Building toolpaths in '{cfg.mode}' mode…")
    if cfg.mode == "edge":
        from .edge_pipeline import build_edge_toolpaths
        from .segmenter import load_image
        rgb = load_image(str(image_path))
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        layers = build_edge_toolpaths(
            gray, fg_mask, cfg, tool_idx=cfg.edge.tool_idx,
        )
    else:
        from .toolpath import build_toolpaths
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
