"""Run drawing styles over a set of images and build contact sheets.

    python scripts/style_gallery.py [IMAGES...] --out DIR [--styles fill,edge,...] [-c CFG] [-j JOBS]

IMAGES are files, folders or glob patterns (default: images/). For each style
it writes DIR/<style>/<image>.svg (the toolpaths SVG), DIR/<style>/<image>.png
(the toolpaths drawn at the real line width on white), DIR/<style>_sheet.jpg
(source next to rendering for every image) and DIR/timings.csv. An image that
fails gets seconds = -1 in the CSV and the run carries on.
"""

from __future__ import annotations

import argparse
import copy
import csv
import glob
import math
import platform
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _no_wmi(*args, **kwargs):
    raise OSError("WMI query skipped")


# numpy asks platform.machine() at import; on Windows that runs a WMI query which
# can close an unrelated handle and break the --jobs worker pool (WinError 6).
# platform falls back to sys.getwindowsversion() when the query fails.
if sys.platform == "win32" and hasattr(platform, "_wmi_query"):
    pass  # TEMP

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from svgelements import SVG, Shape  # noqa: E402

from img2gcode.config import Config  # noqa: E402
from img2gcode.pipeline import convert  # noqa: E402
from img2gcode.raster import load_rgb  # noqa: E402
from img2gcode.styles import STYLES, resolve_style  # noqa: E402
from img2gcode.svg_input import _gradient_colours, _hidden, _paint, _subpaths  # noqa: E402

SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tif", ".tiff", ".webp", ".svg"}
PX_PER_MM, MAX_WIDTH = 4.0, 1200      # rendering scale
TILE, LABEL, GAP = 300, 24, 8         # contact sheet layout (px)
FONT = cv2.FONT_HERSHEY_SIMPLEX


def render(tp) -> np.ndarray:
    """RGB picture of the toolpaths: every stroke at the line width, in print order."""
    s = min(PX_PER_MM, MAX_WIDTH / tp.width_mm)
    img = np.full((max(1, round(tp.height_mm * s)), max(1, round(tp.width_mm * s)), 3), 255, np.uint8)
    thickness = max(1, round(tp.line_width_mm * s))
    for ps in tp.passes:
        lines = [np.round(st.points * s * 16).astype(np.int32) for st in ps.strokes if len(st.points) >= 2]
        if lines:
            colour = tuple(int(v) for v in tp.palette[ps.tool])
            cv2.polylines(img, lines, False, colour, thickness, cv2.LINE_AA, shift=4)
    return img


def fit(img: np.ndarray, size: int) -> np.ndarray:
    h, w = img.shape[:2]
    f = size / max(h, w)
    interp = cv2.INTER_AREA if f < 1 else cv2.INTER_LINEAR
    return cv2.resize(img, (max(1, round(w * f)), max(1, round(h * f))), interpolation=interp)


def source_thumbnail(path: Path) -> np.ndarray:
    """RGB thumbnail of the input. SVGs are drawn shape by shape in paint order."""
    if path.suffix.lower() != ".svg":
        return fit(load_rgb(path), TILE)
    svg = SVG.parse(str(path), reify=True)
    w, h = float(svg.width), float(svg.height)
    k = TILE / max(w, h)
    img = np.full((max(1, round(h * k)), max(1, round(w * k)), 3), 255, np.uint8)
    gradients = _gradient_colours(path)
    for elem in svg.elements():
        if not isinstance(elem, Shape) or _hidden(elem):
            continue
        rings = [np.round(r * 16).astype(np.int32) for r in _subpaths(elem, k)]
        if not rings:
            continue
        fill, stroke = _paint(elem, "fill", gradients), _paint(elem, "stroke", gradients)
        if fill is not None:
            cv2.fillPoly(img, rings, fill, cv2.LINE_AA, shift=4)
        width = (elem.implicit_stroke_width or 0.0) * k
        if stroke is not None and width > 0:
            cv2.polylines(img, rings, False, stroke, max(1, round(width)), cv2.LINE_AA, shift=4)
    return img


def run_one(task):
    """Convert one image with one style. Returns (seconds, strokes, print_mm, tile, error)."""
    style, path, out, cfg = task
    cfg = copy.deepcopy(cfg)
    cfg.style = style
    svg_path = out / style / f"{path.stem}.svg"
    try:
        t0 = time.perf_counter()
        res = convert(path, svg_path, cfg)
        seconds = time.perf_counter() - t0
        img = render(res.toolpaths)
        cv2.imwrite(str(svg_path.with_suffix(".png")), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        strokes = sum(len(ps.strokes) for ps in res.toolpaths.passes)
        return seconds, strokes, res.stats.print_distance_mm, fit(img, TILE), ""
    except Exception:
        return -1.0, None, None, None, traceback.format_exc()


def contact_sheet(cells) -> np.ndarray:
    """Grid of [source | rendering] pairs, each under a label. cells: (label, colour, source, render)."""
    cols = max(1, math.ceil(math.sqrt(len(cells) / 2)))
    rows = math.ceil(len(cells) / cols)
    cw, ch = 2 * TILE + 3 * GAP, LABEL + TILE + GAP
    sheet = np.full((rows * ch + GAP, cols * cw, 3), 255, np.uint8)
    for i, (label, colour, *tiles) in enumerate(cells):
        x0, y0 = (i % cols) * cw + GAP, (i // cols) * ch + GAP
        cv2.putText(sheet, label, (x0, y0 + LABEL - 9), FONT, 0.5, colour, 1, cv2.LINE_AA)
        for j, tile in enumerate(tiles):
            if tile is None:
                continue
            x, y = x0 + j * (TILE + GAP), y0 + LABEL
            th, tw = tile.shape[:2]
            sheet[y:y + th, x:x + tw] = tile
            cv2.rectangle(sheet, (x - 1, y - 1), (x + tw, y + th), (200, 200, 200), 1)
    return sheet


def find_images(args) -> list:
    if not args:
        args = [str(ROOT / "images")]
    found = []
    for arg in args:
        p = Path(arg)
        if p.is_dir():
            found += sorted(f for f in p.iterdir() if f.suffix.lower() in SUFFIXES)
        elif any(c in arg for c in "*?[") and not p.exists():     # shells that do not expand globs
            found += sorted(Path(f) for f in glob.glob(arg) if Path(f).suffix.lower() in SUFFIXES)
        else:
            found.append(p)
    return found


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Run drawing styles over images and build contact sheets.")
    ap.add_argument("images", nargs="*", help="Image or SVG files, folders or globs (default: images/).")
    ap.add_argument("--out", required=True, help="Output folder.")
    ap.add_argument("--styles", default=",".join(STYLES), help="Comma-separated styles (default: all).")
    ap.add_argument("-c", "--config", default=None, help="Config file (default: configs/default.cfg).")
    ap.add_argument("-j", "--jobs", type=int, default=1, help="Images converted in parallel (default: 1).")
    args = ap.parse_args(argv)

    styles = list(dict.fromkeys(resolve_style(s) for s in args.styles.split(",") if s.strip()))
    images = find_images(args.images)
    cfg = Config.load(args.config)
    out = Path(args.out)
    for style in styles:
        (out / style).mkdir(parents=True, exist_ok=True)

    thumbs = {}
    for path in images:
        try:
            thumbs[path] = source_thumbnail(path)
        except Exception:
            thumbs[path] = None

    tasks = [(style, path, out, cfg) for style in styles for path in images]
    if args.jobs > 1:
        pool = ProcessPoolExecutor(args.jobs)
        results = pool.map(run_one, tasks)
    else:
        pool, results = None, map(run_one, tasks)

    cells = {style: [] for style in styles}
    with open(out / "timings.csv", "w", newline="", encoding="utf-8") as fh:
        table = csv.writer(fh, lineterminator="\n")
        table.writerow(["style", "image", "seconds", "strokes", "print_mm"])
        for n, ((style, path, _, _), (seconds, strokes, print_mm, tile, error)) in enumerate(zip(tasks, results), 1):
            if error:
                table.writerow([style, path.name, -1, "", ""])
                reason = error.strip().splitlines()[-1]
                print(f"[{n}/{len(tasks)}] {style:<8} {path.name}  FAILED: {reason}", flush=True)
                print(error, file=sys.stderr, flush=True)
                cells[style].append((f"{path.name}  FAILED: {reason}"[:60], (220, 0, 0), thumbs[path], None))
            else:
                table.writerow([style, path.name, f"{seconds:.2f}", strokes, f"{print_mm:.0f}"])
                print(f"[{n}/{len(tasks)}] {style:<8} {path.name}  {seconds:.1f}s  {strokes} strokes", flush=True)
                cells[style].append((f"{path.name}  {seconds:.1f}s  {strokes} strokes", (0, 0, 0), thumbs[path], tile))
            fh.flush()
            if len(cells[style]) == len(images):
                sheet = cv2.cvtColor(contact_sheet(cells.pop(style)), cv2.COLOR_RGB2BGR)
                cv2.imwrite(str(out / f"{style}_sheet.jpg"), sheet, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if pool is not None:
        pool.shutdown()


if __name__ == "__main__":
    main()
