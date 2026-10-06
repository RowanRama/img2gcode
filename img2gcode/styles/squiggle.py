"""Squiggle: horizontal rows that wiggle harder where the image is darker.

Each row is a sine wave around its centreline. Darkness raises both its
frequency and its amplitude, so tone reads as ink density. Rows stop where the
image is paper-white and alternate direction, so the end of one row joins the
start of the next into a serpentine stroke.
"""

from __future__ import annotations

from typing import List, Tuple

import cv2
import numpy as np
from scipy.ndimage import map_coordinates, median_filter

from ..geometry import simplify
from ..model import Artwork, Stroke
from ..raster import denoise, fit_page, foreground_mask, luminance, palette_kmeans, working_image

SINGLE_TOOL_PALETTE = [(30, 30, 30)]


def make(rgb, cfg) -> Artwork:
    lw = cfg.extrusion.effective_line_width
    sp = cfg.squiggle.row_spacing * lw
    step = lw / 4
    H, W = rgb.shape[:2]
    width, height, mm_per_src = fit_page(W, H, cfg)
    img, mmpp = working_image(rgb, mm_per_src, step)
    img = denoise(img)
    # A light blur keeps leftover noise in paper areas from breaking it into specks.
    fg = foreground_mask(cv2.GaussianBlur(img, (0, 0), lw / 2 / mmpp), cfg.image.white_threshold)
    sigma = sp / 2 / mmpp
    dark = _tone(np.clip(1.0 - cv2.GaussianBlur(luminance(img), (0, 0), sigma), 0.0, 1.0), fg)

    # Samples every lw/4 along each row centreline; rows k = 0, 1, ... at y = (k + 0.5) * sp.
    xs = (np.arange(int(width / step)) + 0.5) * step
    ys = (np.arange(int(height / sp)) + 0.5) * sp
    grid = np.ix_(np.minimum((ys / mmpp).astype(int), img.shape[0] - 1),
                  np.minimum((xs / mmpp).astype(int), img.shape[1] - 1))
    present = fg[grid]
    d = map_coordinates(dark, np.meshgrid(ys / mmpp - 0.5, xs / mmpp - 0.5, indexing="ij"), order=1, mode="nearest")
    phase = 2 * np.pi * step * np.cumsum(cfg.squiggle.max_frequency / (2.5 * lw) * d, axis=1)
    y = ys[:, None] + d * (sp - lw) / 2 * 0.95 * np.sin(phase)

    palette = list(SINGLE_TOOL_PALETTE)
    tools = np.zeros(present.shape, int)
    if cfg.tools.num_tools > 1 and fg.any():
        pal, _ = palette_kmeans(img, fg, cfg.tools.num_tools)
        palette = [tuple(int(v) for v in c) for c in pal]
        if len(pal) > 1 and present.size:
            colour = _lab(_foreground_colour(img, fg, sigma)[grid])
            tools = ((colour[..., None, :] - _lab(pal[None])[0]) ** 2).sum(axis=-1).argmin(axis=-1)

    max_gap = int(round(2 * lw / step))
    median_size = int(round(4 * lw / step)) | 1
    rows: List[List[Tuple[np.ndarray, int]]] = []
    for k in range(len(ys)):
        pieces = []
        for s, e in _pieces(present[k], max_gap):
            for a, b, t in _runs(median_filter(tools[k, s:e], median_size, mode="nearest")):
                if b - a >= 2:
                    pieces.append((np.c_[xs[s + a:s + b], y[k, s + a:s + b]], t))
        if k % 2:
            pieces = [(p[::-1], t) for p, t in reversed(pieces)]
        rows.append(pieces)

    strokes = [Stroke(simplify(p, 0.02 * lw), tool=t) for p, t in _serpentine(rows, sp)]
    return Artwork(width, height, [], strokes, palette, (W, H), rgb, "squiggle")


def _tone(dark: np.ndarray, fg: np.ndarray) -> np.ndarray:
    """Stretch the foreground's darkness so its darkest 2 % wiggle fully, then take the root.

    Frequency and amplitude both scale with D, so the extra ink grows with D²; the
    root makes it grow about linearly with tone. Without the stretch, photos (mostly
    mid-tones) come out as near-uniform rows.
    """
    if not fg.any():
        return dark
    lo, hi = np.percentile(dark[fg], [2, 98])
    return np.sqrt(np.clip(1.0 - (hi - dark) / max(hi - lo, 0.3), 0.0, 1.0))


def _pieces(present: np.ndarray, max_gap: int) -> List[Tuple[int, int]]:
    """(start, end) sample ranges of the row, bridging gaps of at most max_gap samples."""
    out: List[Tuple[int, int]] = []
    for s, e, on in _runs(present.astype(np.int8)):
        if not on:
            continue
        if out and s - out[-1][1] <= max_gap:
            out[-1] = (out[-1][0], e)
        else:
            out.append((s, e))
    return out


def _runs(values: np.ndarray) -> List[Tuple[int, int, int]]:
    """(start, end, value) for each run of equal values."""
    cut = np.flatnonzero(np.diff(values)) + 1
    starts, ends = np.r_[0, cut], np.r_[cut, len(values)]
    return [(int(s), int(e), int(values[s])) for s, e in zip(starts, ends)]


def _serpentine(rows, sp: float) -> List[Tuple[np.ndarray, int]]:
    """Join each piece to a same-tool stroke that ended on the previous row nearby.

    Rows are in travel order (alternating direction). The gap is measured between
    centreline ends, so the wiggle's phase does not matter. Pieces on one row are
    never joined (that would bridge paper or another tool's piece), and a
    connector may not reach over the neighbouring piece on either row.
    """
    chains: List[List[np.ndarray]] = []
    chain_tool: List[int] = []
    tails: List[Tuple[int, float, float]] = []       # previous row: (chain, end x, free space after it)
    for k, pieces in enumerate(rows):
        direction = 1 if k % 2 else -1               # travel direction of the previous row
        starts = [p[0, 0] for p, _ in pieces]
        ends = [p[-1, 0] for p, _ in pieces]
        before = [np.inf] + [abs(s - e) for s, e in zip(starts[1:], ends[:-1])]
        after = before[1:] + [np.inf]
        new_tails = []
        for i, (pts, t) in enumerate(pieces):
            best, best_gap = None, 1.5 * sp
            for j, (c, end_x, room) in enumerate(tails):
                over = (starts[i] - end_x) * direction   # > 0: past the tail's end, < 0: past this start
                gap = np.hypot(starts[i] - end_x, sp)
                if chain_tool[c] == t and gap <= best_gap and over < room and -over < before[i]:
                    best, best_gap = j, gap
            if best is not None:
                c = tails.pop(best)[0]
                chains[c].append(pts)
            else:
                c = len(chains)
                chains.append([pts])
                chain_tool.append(t)
            new_tails.append((c, ends[i], after[i]))
        tails = new_tails
    return [(np.vstack(parts), t) for parts, t in zip(chains, chain_tool)]


def _lab(rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(np.asarray(rgb, np.float32) / 255.0, cv2.COLOR_RGB2Lab)


def _foreground_colour(img: np.ndarray, fg: np.ndarray, sigma: float) -> np.ndarray:
    """RGB of the nearby foreground (paper ignored), blurred with sigma px; white where there is none."""
    w = cv2.GaussianBlur(fg.astype(np.float32), (0, 0), sigma)[..., None]
    rgb = cv2.GaussianBlur(img.astype(np.float32) * fg[..., None], (0, 0), sigma)
    return np.where(w > 1e-3, np.clip(rgb / np.maximum(w, 1e-3), 0, 255), 255.0).astype(np.float32)
