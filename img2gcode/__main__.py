"""CLI entry point.

Usage
-----
# Image → toolpaths SVG + G-code (opens the viewer by default):
python -m img2gcode -i logo.png -o output/logo.gcode

# Pick a drawing style, no viewer:
python -m img2gcode -i photo.jpg -o output/photo.gcode --style contour --headless

# Only the toolpaths SVG (inspect in a browser, edit in Inkscape):
python -m img2gcode -i logo.png -o output/logo.svg

# G-code from an edited toolpaths SVG:
python -m img2gcode -i output/logo.svg -o output/logo.gcode

# View an existing G-code file or toolpaths SVG:
python -m img2gcode --visualise output/logo.gcode
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .config import Config
from .gcode_parser import parse_gcode, parse_gcode_lines
from .pipeline import convert, make_artwork
from .styles import ALIASES, STYLES, resolve_style
from .svg_io import read_toolpaths_svg
from .writer import gcode_lines

_STYLE_CHOICES = list(STYLES) + list(ALIASES)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="img2gcode",
        description="Convert images and SVGs to a toolpaths SVG and multi-tool G-code.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("-i", "--input", metavar="INPUT",
                   help="Image (PNG, JPG, …), artwork SVG or toolpaths SVG.")
    p.add_argument("-o", "--output", metavar="OUTPUT",
                   help="Output .gcode (plus a sibling .svg), or .svg for the toolpaths SVG only.")
    p.add_argument("-c", "--config", metavar="CFG", default=None,
                   help="Path to a .cfg config file (overrides defaults).")
    p.add_argument("--style", choices=_STYLE_CHOICES, default=None,
                   help="Drawing style (default: [mode] type in the config).")
    p.add_argument("--mode", choices=_STYLE_CHOICES, default=None,
                   help="Same as --style (kept for old scripts).")
    p.add_argument("--n-tools", type=int, default=None, metavar="N",
                   help="Number of tools/colours (overrides the config value).")
    p.add_argument("--svg", metavar="PATH", default=None,
                   help="Where to write the toolpaths SVG (default: next to the G-code).")
    p.add_argument("--svg-only", action="store_true",
                   help="Write the toolpaths SVG only, no G-code.")
    p.add_argument("--select-roi", action="store_true",
                   help="Draw rectangles to exclude from infill before planning (perimeters are unaffected).")
    p.add_argument("--debug-edges", metavar="PATH", default=None,
                   help="Edge style only: save the binary line map as a PNG at PATH.")
    p.add_argument("--headless", action="store_true",
                   help="Do not open the viewer after writing.")
    p.add_argument("--visualise", metavar="FILE",
                   help="Open the viewer for an existing .gcode file or toolpaths .svg.")
    p.add_argument("-v", "--verbose", action="store_true",
                   help="Kept for old scripts; messages are on unless -q.")
    p.add_argument("-q", "--quiet", action="store_true",
                   help="Suppress progress messages.")
    return p


def load_segments(path, cfg: Config):
    """Viewer segments and metadata for a .gcode file or a toolpaths .svg."""
    path = Path(path)
    if path.suffix.lower() == ".svg":
        tp, _ = read_toolpaths_svg(path)
        return parse_gcode_lines(gcode_lines(tp, cfg)[0])
    return parse_gcode(path)


def _report(res, cfg: Config) -> None:
    if res.svg_path is not None:
        print(f"[img2gcode] SVG written → {res.svg_path}")
    if res.gcode_path is not None:
        n = len(res.gcode_path.read_text(encoding="utf-8").splitlines())
        print(f"[img2gcode] GCode written → {res.gcode_path}  ({n} lines)")
    s = res.stats
    grams = s.filament_grams(cfg.extrusion.filament_diameter)
    print(
        f"[img2gcode] Estimated: {s.total_time_min:.1f} min  "
        f"({s.print_time_min:.1f} print + {s.travel_time_min:.1f} travel)  "
        f"|  filament {s.filament_mm:.0f} mm ({grams:.2f} g PLA)  "
        f"|  travel {s.travel_distance_mm:.0f} mm / "
        f"print {s.print_distance_mm:.0f} mm"
    )
    if res.ignored_shapes:
        print(f"[img2gcode] warning: {res.ignored_shapes} shape(s) outside tool layers were ignored")
    lw = cfg.extrusion.effective_line_width
    if abs(res.toolpaths.line_width_mm - lw) > 1e-3:
        print(f"[img2gcode] warning: the SVG was planned for {res.toolpaths.line_width_mm:g} mm lines "
              f"but the config's line width is {lw:g} mm")


def main(argv=None):
    parser = _build_parser()
    args = parser.parse_args(argv)
    verbose = not args.quiet
    # Piped output on Windows is cp1252, which has no "→"; print "?" instead of crashing.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    cfg = Config.load(args.config)

    if args.visualise:
        path = Path(args.visualise)
        if not path.exists():
            sys.exit(f"[error] File not found: {path}")
        from .gui import launch_gui
        launch_gui(*load_segments(path, cfg), title=f"img2gcode – {path.name}")
        return

    if not args.input:
        parser.error("--input/-i is required unless using --visualise.")
    if not args.output:
        parser.error("--output/-o is required unless using --visualise.")
    input_path = Path(args.input)
    if not input_path.exists():
        sys.exit(f"[error] Input not found: {input_path}")

    if args.style or args.mode:
        cfg.style = args.style or args.mode
    try:
        cfg.style = resolve_style(cfg.style)
    except ValueError as e:
        sys.exit(f"[error] {e} (check [mode] type in the config)")
    if args.n_tools is not None:
        cfg.tools.num_tools = args.n_tools
    if args.debug_edges is not None:
        cfg.edge.debug_edges_path = args.debug_edges
    n_tools_explicit = args.n_tools is not None

    art, zones = None, None
    if args.select_roi:
        art = make_artwork(input_path, cfg, n_tools_explicit)
        if art is not None and art.regions:
            from .roi_selector import select_exclusion_zones
            zones = select_exclusion_zones(art)
            if verbose:
                print(f"[img2gcode] {len(zones)} exclusion zone(s) selected.")
        else:
            print("[img2gcode] --select-roi needs an artwork with filled regions; skipping it")

    res = convert(input_path, args.output, cfg, svg_path=args.svg, svg_only=args.svg_only,
                  exclusion_zones=zones, n_tools_explicit=n_tools_explicit, artwork=art)
    if verbose:
        _report(res, cfg)

    if not args.headless:
        from .gui import launch_gui
        target = res.gcode_path or res.svg_path or input_path
        launch_gui(*load_segments(target, cfg), title=f"img2gcode – {input_path.name}")


if __name__ == "__main__":
    main()
