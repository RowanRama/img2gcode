"""oneline: weighted Voronoi stippling joined into one continuous line per tool.

Darkness decides where points go (Lloyd relaxation weighted by darkness), a
nearest-neighbour tour improved by 2-opt joins them, and Chaikin smoothing
turns the tour into a flowing line whose density reads as tone.
"""

from __future__ import annotations

import math
import time
from collections import deque

import cv2
import numpy as np
import shapely
from scipy.spatial import cKDTree

from ..geometry import chaikin, simplify
from ..model import Artwork, Stroke
from ..raster import fit_page, foreground_mask, luminance, palette_kmeans, sample_colors, working_image

SPACING_FACTOR = 2.5          # auto point spacing in fully dark areas, in line widths
MIN_POINTS, MAX_POINTS = 200, 20000
LLOYD_ITERATIONS = 15
NEIGHBOURS = 8                # 2-opt candidate list size
DARK = (30, 30, 30)           # single-tool palette


def make(rgb, cfg) -> Artwork:
    """One continuous line per tool, denser where the image is darker."""
    lw = cfg.extrusion.effective_line_width
    H, W = rgb.shape[:2]
    width_mm, height_mm, mm_per_src_px = fit_page(W, H, cfg)
    img, mmpp = working_image(rgb, mm_per_src_px, lw / 2)
    art = Artwork(width_mm, height_mm, [], [], [DARK], (W, H), rgb, "oneline")

    fg = foreground_mask(img, cfg.image.white_threshold)
    dark = (1.0 - luminance(img).astype(np.float64)) ** cfg.oneline.gamma
    dark[~fg] = 0.0
    if dark.sum() <= 0:
        return art

    n = cfg.oneline.points
    if n <= 0:
        n = int(np.clip(dark.sum() * mmpp ** 2 / (SPACING_FACTOR * lw) ** 2, MIN_POINTS, MAX_POINTS))
    points = _stipple(dark, n, mmpp)

    tools = np.zeros(len(points), int)
    if cfg.tools.num_tools > 1:
        palette, _ = palette_kmeans(img, fg, cfg.tools.num_tools)
        tools = _nearest_colour(sample_colors(img, points, mmpp), palette)
        art.palette = [tuple(int(v) for v in c) for c in palette]

    for t in range(len(art.palette)):
        pts = points[tools == t]
        if len(pts) < 2:
            continue
        order = _tour(pts, cfg.oneline.tsp_seconds * len(pts) / len(points))   # tools share the budget
        line = simplify(chaikin(pts[order], 2), 0.05 * lw)
        art.strokes.append(Stroke(line, tool=t, kind="line"))
    return art


def _stipple(dark: np.ndarray, n: int, mmpp: float) -> np.ndarray:
    """*n* points in page mm spread with density proportional to *dark* (weighted Lloyd)."""
    rng = np.random.default_rng(0)
    H, W = dark.shape
    flat = dark.ravel()
    rows, cols = np.divmod(rng.choice(flat.size, n, p=flat / flat.sum()), W)
    points = (np.c_[cols, rows] + rng.random((n, 2))) * mmpp

    rr, cc = np.nonzero(dark > 0.01)
    pix = (np.c_[cc, rr] + 0.5) * mmpp
    w = dark[rr, cc]
    for _ in range(LLOYD_ITERATIONS):
        _, near = cKDTree(points).query(pix)
        mass = np.bincount(near, w, n)
        ok = mass > 0
        for axis in (0, 1):
            points[ok, axis] = np.bincount(near, w * pix[:, axis], n)[ok] / mass[ok]
    return points


def _nearest_colour(colours: np.ndarray, palette: np.ndarray) -> np.ndarray:
    """Index of the palette entry nearest (in Lab) to each RGB colour."""
    return cKDTree(_lab(palette)).query(_lab(colours))[1]


def _lab(rgb) -> np.ndarray:
    return cv2.cvtColor(np.asarray(rgb, np.uint8)[None], cv2.COLOR_RGB2Lab)[0].astype(np.float32)


def _tour(points: np.ndarray, seconds: float) -> np.ndarray:
    """Open path through every point: nearest-neighbour start, then 2-opt until
    no move improves or *seconds* run out. Returns a permutation of indices."""
    p = np.asarray(points, np.float64)
    if len(p) < 4:
        return np.arange(len(p))
    order = _nearest_neighbour_path(p, int(np.argmin(p[:, 0])))
    return _two_opt(p, order, time.perf_counter() + seconds)


def _nearest_neighbour_path(p: np.ndarray, start: int) -> np.ndarray:
    n = len(p)
    visited = np.zeros(n, bool)
    path = np.empty(n, np.int64)
    ids, tree, stale = np.arange(n), cKDTree(p), 0   # stale = visited points still in the tree
    cur = start
    for step in range(n):
        visited[cur] = True
        path[step] = cur
        stale += 1
        if step == n - 1:
            break
        if 2 * stale > len(ids):                     # rebuild over the unvisited points
            ids = np.flatnonzero(~visited)
            tree, stale = cKDTree(p[ids]), 0
        k = NEIGHBOURS
        while True:
            _, j = tree.query(p[cur], min(k, len(ids)))
            cand = ids[np.atleast_1d(j)]
            free = cand[~visited[cand]]
            if len(free):
                break
            k *= 2
        cur = int(free[0])
    return path


def _two_opt(p: np.ndarray, order: np.ndarray, deadline: float) -> np.ndarray:
    """2-opt on an open path with nearest-neighbour candidate lists and don't-look bits.

    For city a with path neighbour b (successor, then predecessor) and a
    candidate c closer to a than b, the move swaps edges (a, b) and (c, d) for
    (a, c) and (b, d), d being c's neighbour on the same side. When c is a path
    end, d does not exist and the move just re-attaches that end.

    Candidate lists miss crossings made by long nearest-neighbour jumps, so
    whenever the list search runs dry, crossing edge pairs are found
    geometrically and uncrossed (always a shorter path), then the search resumes.
    """
    n = len(order)
    xs, ys = p[:, 0].tolist(), p[:, 1].tolist()
    _, nb = cKDTree(p).query(p, min(NEIGHBOURS, n - 1) + 1)
    nb = nb[:, 1:].tolist()
    o = order.copy()
    pos = np.empty(n, np.int64)
    pos[o] = np.arange(n)
    queue, queued = deque(o.tolist()), [True] * n

    def dist(u, v):
        return math.hypot(xs[u] - xs[v], ys[u] - ys[v])

    def reverse(lo, hi, *cities):
        o[lo:hi + 1] = o[lo:hi + 1][::-1].copy()
        pos[o[lo:hi + 1]] = np.arange(lo, hi + 1)
        for city in cities:
            if city >= 0 and not queued[city]:
                queued[city] = True
                queue.append(city)

    def improve(a):
        """Make the first improving move around city a, if any."""
        i = int(pos[a])
        for step in (1, -1):
            if not 0 <= i + step < n:
                continue
            b = int(o[i + step])
            d_ab = dist(a, b)
            for c in nb[a]:
                d_ac = dist(a, c)
                if d_ac >= d_ab:
                    break
                j = int(pos[c])
                if 0 <= j + step < n:
                    d = int(o[j + step])
                    gain = d_ab + dist(c, d) - d_ac - dist(b, d)
                else:
                    d, gain = -1, d_ab - d_ac
                if gain > 1e-9:
                    if step == 1:
                        lo, hi = (i + 1, j) if j > i else (j + 1, i)
                    else:
                        lo, hi = (i, j - 1) if j > i else (j, i - 1)
                    reverse(lo, hi, a, b, c, d)
                    return

    while time.perf_counter() < deadline:
        if queue:
            a = queue.popleft()
            queued[a] = False
            improve(a)
            continue
        for a, b, c, d in _crossing_edges(p, o).tolist():
            i, j = int(pos[a]), int(pos[c])
            if pos[b] == i + 1 and pos[d] == j + 1:              # both edges still in the path
                reverse(min(i, j) + 1, max(i, j), a, b, c, d)
        if not queue:
            break
    return o


def _crossing_edges(p: np.ndarray, o: np.ndarray) -> np.ndarray:
    """(m, 4) cities (a, b, c, d) of path edges a->b and c->d that properly cross."""
    a, b = p[o[:-1]], p[o[1:]]
    segs = shapely.linestrings(np.stack([a, b], axis=1))
    i, j = shapely.STRtree(segs).query(segs)                     # bounding boxes overlap
    keep = i < j
    i, j = i[keep], j[keep]

    def side(p0, p1, q):
        u, v = p1 - p0, q - p0
        return np.sign(u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0])

    cross = ((side(a[i], b[i], a[j]) * side(a[i], b[i], b[j]) < 0)
             & (side(a[j], b[j], a[i]) * side(a[j], b[j], b[i]) < 0))
    i, j = i[cross], j[cross]
    return np.c_[o[i], o[i + 1], o[j], o[j + 1]]
