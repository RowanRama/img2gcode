"""Edge-detection toolpath generation.

Alternate to the fill pipeline. Designed for inputs (e.g. portraits) where
internal features matter and the segment-and-infill approach loses detail.

Pipeline
--------
1. Grayscale + bilateral filter (preserve edges, drop noise).
2. Canny edge detection (auto thresholds via the median heuristic).
3. Restrict edges to the foreground mask so background texture is ignored.
4. Collapse parallel edges that would print on top of each other:
   dilate the binary edge map by line_width/2, then thin back to a 1-px
   medial axis. Edges further apart than line_width survive intact;
   edges within line_width merge into a single centerline.
5. Trace the skeleton into polylines (junction-aware walk).
6. Simplify each polyline (Douglas-Peucker).
7. Greedy emission with a "drawn mask" guard: rasterised candidates that
   would overlap already-emitted strokes by more than the tolerance get
   split or dropped.

All output is a list of ToolLayer objects compatible with GCodeWriter.
"""

from __future__ import annotations

from typing import List, Tuple

import cv2
import numpy as np

from .config import Config
from .toolpath import Move, ToolLayer, order_polylines_greedy


# ------------------------------------------------------------------ #
# Skeletonization (Zhang-Suen, vectorised)                              #
# ------------------------------------------------------------------ #

def _zhang_suen_thin(binary: np.ndarray) -> np.ndarray:
    """Iterative Zhang-Suen thinning. Returns a 1-pixel-wide skeleton.

    Vectorised over the whole image per sub-iteration. Converges in
    O(max_thickness) sweeps which is fine for typical Canny outputs.
    """
    img = (binary > 0).astype(np.uint8)
    while True:
        changed = False
        for sub in (0, 1):
            # Build the 8 neighbour planes (P2..P9 going clockwise from N).
            P = img
            P2 = np.roll(P, 1, axis=0)
            P3 = np.roll(np.roll(P, 1, axis=0), -1, axis=1)
            P4 = np.roll(P, -1, axis=1)
            P5 = np.roll(np.roll(P, -1, axis=0), -1, axis=1)
            P6 = np.roll(P, -1, axis=0)
            P7 = np.roll(np.roll(P, -1, axis=0), 1, axis=1)
            P8 = np.roll(P, 1, axis=1)
            P9 = np.roll(np.roll(P, 1, axis=0), 1, axis=1)
            # Zero the rolled-around borders so wrap-around can't form skeletons.
            for arr in (P2, P3, P4, P5, P6, P7, P8, P9):
                arr[0, :] = 0; arr[-1, :] = 0; arr[:, 0] = 0; arr[:, -1] = 0

            B = P2 + P3 + P4 + P5 + P6 + P7 + P8 + P9  # neighbour count
            # A = number of 0→1 transitions in the sequence P2..P9,P2.
            seq = np.stack([P2, P3, P4, P5, P6, P7, P8, P9, P2], axis=0)
            A = np.sum((seq[:-1] == 0) & (seq[1:] == 1), axis=0)

            cond = (P == 1) & (B >= 2) & (B <= 6) & (A == 1)
            if sub == 0:
                cond &= (P2 * P4 * P6 == 0) & (P4 * P6 * P8 == 0)
            else:
                cond &= (P2 * P4 * P8 == 0) & (P2 * P6 * P8 == 0)

            if cond.any():
                img = np.where(cond, 0, img).astype(np.uint8)
                changed = True
        if not changed:
            break
    return img


# ------------------------------------------------------------------ #
# Skeleton → polyline tracing                                           #
# ------------------------------------------------------------------ #

_NEIGH = [(-1, -1), (-1, 0), (-1, 1),
          (0, -1),           (0, 1),
          (1, -1),  (1, 0),  (1, 1)]


def _neighbour_count(skel: np.ndarray) -> np.ndarray:
    """Per-pixel count of 8-connected skeleton neighbours."""
    k = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.uint8)
    return cv2.filter2D(skel, -1, k, borderType=cv2.BORDER_CONSTANT)


def _trace_polylines(skel: np.ndarray) -> List[List[Tuple[int, int]]]:
    """Walk a 1-px skeleton into polylines.

    Strategy: remove junction pixels (8-connectivity count >= 3) from a
    working copy first — that turns the skeleton into a set of simple
    chains (each pixel has <=2 skeleton neighbours), which we walk linearly.
    Junction pixels are then re-attached as 1-step extensions to the chains
    they touch. This avoids the classical failure mode where adjacent
    junction pixels fragment every walk into 2-vertex stubs.

    Coordinates are returned as (x, y) tuples.
    """
    skel = (skel > 0).astype(np.uint8)
    H, W = skel.shape
    counts = _neighbour_count(skel) * skel
    junctions = (counts >= 3) & (skel > 0)

    work = skel.copy()
    work[junctions] = 0
    work_counts = _neighbour_count(work) * work

    visited = np.zeros_like(work, dtype=bool)
    polylines: List[List[Tuple[int, int]]] = []

    def walk_chain(sy: int, sx: int) -> List[Tuple[int, int]]:
        """Walk a simple chain starting at (sy,sx). Pixels have <=2 neighbours."""
        path = [(sx, sy)]
        visited[sy, sx] = True
        prev_y, prev_x = -1, -1
        cy, cx = sy, sx
        while True:
            nxt = None
            for dy, dx in _NEIGH:
                ny, nx = cy + dy, cx + dx
                if 0 <= ny < H and 0 <= nx < W and work[ny, nx] and not visited[ny, nx]:
                    if (ny, nx) == (prev_y, prev_x):
                        continue
                    nxt = (ny, nx)
                    break
            if nxt is None:
                return path
            prev_y, prev_x = cy, cx
            cy, cx = nxt
            visited[cy, cx] = True
            path.append((cx, cy))

    # Pass 1: chains that start at endpoints (work_count == 1).
    ys, xs = np.where((work_counts == 1) & ~visited)
    for y, x in zip(ys, xs):
        if visited[y, x]:
            continue
        polylines.append(walk_chain(int(y), int(x)))

    # Pass 2: closed loops (work_count == 2 everywhere on the loop).
    ys, xs = np.where((work > 0) & ~visited)
    for y, x in zip(ys, xs):
        if visited[y, x]:
            continue
        path = walk_chain(int(y), int(x))
        if path and path[0] != path[-1]:
            path.append(path[0])
        polylines.append(path)

    # Pass 3: re-attach junctions. For each junction pixel, find the two
    # nearest chain endpoints among its 8-neighbours and stitch the junction
    # into the polyline so the printed path actually connects.
    if junctions.any():
        # Build an endpoint -> polyline-index lookup.
        ep_lookup: dict = {}
        for i, p in enumerate(polylines):
            if not p:
                continue
            ep_lookup.setdefault(p[0], []).append((i, "head"))
            ep_lookup.setdefault(p[-1], []).append((i, "tail"))

        jy, jx = np.where(junctions)
        for y, x in zip(jy, jx):
            for dy, dx in _NEIGH:
                ny, nx = int(y + dy), int(x + dx)
                if (nx, ny) in ep_lookup:
                    for idx, end in ep_lookup[(nx, ny)]:
                        if end == "head":
                            polylines[idx].insert(0, (int(x), int(y)))
                        else:
                            polylines[idx].append((int(x), int(y)))
                    # Update lookup so further junctions still find this end.
                    # (cheap rebuild for just this entry)
                    ep_lookup.pop((nx, ny), None)
                    if polylines[idx]:
                        ep_lookup.setdefault(polylines[idx][0], []).append((idx, "head"))
                        ep_lookup.setdefault(polylines[idx][-1], []).append((idx, "tail"))

    return polylines


# ------------------------------------------------------------------ #
# Polyline simplification                                               #
# ------------------------------------------------------------------ #

def _simplify(poly: List[Tuple[int, int]], epsilon: float) -> List[Tuple[float, float]]:
    """Douglas-Peucker simplify via cv2.approxPolyDP."""
    if epsilon <= 0 or len(poly) < 3:
        return [(float(x), float(y)) for x, y in poly]
    pts = np.asarray(poly, dtype=np.float32).reshape(-1, 1, 2)
    simplified = cv2.approxPolyDP(pts, float(epsilon), closed=False)
    return [(float(p[0][0]), float(p[0][1])) for p in simplified]


# ------------------------------------------------------------------ #
# Drawn-mask guard                                                      #
# ------------------------------------------------------------------ #

def _stamp_polyline(
    shape: Tuple[int, int], poly: List[Tuple[float, float]], radius_px: int
) -> np.ndarray:
    """Rasterise *poly* at *radius_px* thickness onto a fresh mask."""
    mask = np.zeros(shape, dtype=np.uint8)
    if len(poly) < 2:
        return mask
    pts = np.array([[int(round(x)), int(round(y))] for x, y in poly], dtype=np.int32)
    cv2.polylines(mask, [pts], isClosed=False, color=1,
                  thickness=max(1, 2 * radius_px + 1))
    return mask


def _split_polyline_by_mask(
    poly: List[Tuple[float, float]],
    drawn_mask: np.ndarray,
    radius_px: int,
) -> List[List[Tuple[float, float]]]:
    """Split a polyline into sub-polylines that clear a dilated *drawn_mask*.

    Walks every pixel along each segment (not just the vertices) so that a
    long simplified segment whose endpoints happen to land in the forbidden
    zone — e.g. an edge that meets two already-emitted edges at its
    corners — still emits its uncovered middle portion. Forbidden runs at
    each end of a segment are clipped, and a run that crosses through a
    forbidden region is split into two valid sub-polylines.
    """
    H, W = drawn_mask.shape
    if radius_px > 0:
        k = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * radius_px + 1, 2 * radius_px + 1)
        )
        forbidden = cv2.dilate(drawn_mask, k)
    else:
        forbidden = drawn_mask

    if len(poly) < 2:
        return []

    def is_blocked(px: float, py: float) -> bool:
        ix = int(round(px)); iy = int(round(py))
        if not (0 <= iy < H and 0 <= ix < W):
            return False
        return bool(forbidden[iy, ix])

    runs: List[List[Tuple[float, float]]] = []
    current: List[Tuple[float, float]] = []

    def flush():
        nonlocal current
        if len(current) >= 2:
            runs.append(current)
        current = []

    # Walk every pixel along each segment. Segment density is one sample per
    # unit pixel of segment length, which is enough for the dilated guard.
    prev_blocked = is_blocked(*poly[0])
    if not prev_blocked:
        current.append(poly[0])

    for i in range(1, len(poly)):
        x0, y0 = poly[i - 1]
        x1, y1 = poly[i]
        dx, dy = x1 - x0, y1 - y0
        seg_len = max(1, int(np.ceil(np.hypot(dx, dy))))
        for s in range(1, seg_len + 1):
            t = s / seg_len
            sx = x0 + dx * t
            sy = y0 + dy * t
            blocked = is_blocked(sx, sy)
            if blocked and not prev_blocked:
                flush()
            if not blocked:
                # Only retain the actual polyline vertices and the entry/exit
                # points on a forbidden boundary — interior dense samples on
                # the same segment would just bloat the move list.
                if prev_blocked:
                    # First clear sample after a blocked stretch — this is
                    # roughly where the polyline re-enters open territory.
                    current.append((sx, sy))
                if s == seg_len:
                    # Reached the original vertex — keep it as the chain pivot.
                    current.append((x1, y1))
            prev_blocked = blocked

    flush()
    return runs


# ------------------------------------------------------------------ #
# Edge extraction                                                       #
# ------------------------------------------------------------------ #

def _auto_canny(gray: np.ndarray, sigma: float = 0.33) -> np.ndarray:
    """Canny with thresholds derived from the image median.

    Kept for backward compatibility. New code should prefer
    `_detect_edges_multiscale` which is far more robust on portraits where
    a bright background skews the median.
    """
    v = float(np.median(gray))
    lo = int(max(0, (1.0 - sigma) * v))
    hi = int(min(255, (1.0 + sigma) * v))
    if hi <= lo:
        hi = min(255, lo + 1)
    return cv2.Canny(gray, lo, hi)


def _parse_scales(raw: str | List[float]) -> List[float]:
    """Parse the detection_scales config field (comma-separated floats)."""
    if isinstance(raw, list):
        return [float(s) for s in raw]
    parts = [s.strip() for s in str(raw).split(",") if s.strip()]
    out: List[float] = []
    for p in parts:
        try:
            out.append(float(p))
        except ValueError:
            continue
    return out or [0.0, 1.0, 2.0]


def _detect_edges_multiscale(
    gray: np.ndarray, eparams,
) -> np.ndarray:
    """Multi-scale Canny with optional CLAHE preprocessing.

    Designed for portraits where a single Canny pass either over-smooths
    (loses hair / beard texture) or under-smooths (picks up skin noise).
    The pipeline:

      1. CLAHE (Contrast Limited Adaptive Histogram Equalisation) — brings
         out faint features in shadowed/highlighted regions without the
         global contrast blowout of plain histogram equalisation. Disable
         with clahe_clip <= 0.
      2. Optional bilateral filter — denoise while keeping edges crisp.
      3. Otsu auto-thresholding — far more robust than the median heuristic
         when the background is mostly white (median ≈ 254). Falls back to
         pinned thresholds if both canny_low and canny_high are set.
      4. Run Canny at every blur scale in detection_scales and OR the
         results. A scale of 0 means "no blur"; larger sigmas pick up the
         structural edges (jaw, hairline) while smaller ones catch the
         fine texture (hair strands, eyelashes).
    """
    work = gray
    if getattr(eparams, "clahe_clip", 0.0) > 0:
        clahe = cv2.createCLAHE(
            clipLimit=float(eparams.clahe_clip),
            tileGridSize=(int(eparams.clahe_grid), int(eparams.clahe_grid)),
        )
        work = clahe.apply(work)

    if eparams.bilateral_d > 0:
        work = cv2.bilateralFilter(
            work, int(eparams.bilateral_d),
            float(eparams.bilateral_sigma_color),
            float(eparams.bilateral_sigma_space),
        )

    if eparams.canny_low > 0 and eparams.canny_high > 0:
        lo, hi = int(eparams.canny_low), int(eparams.canny_high)
    else:
        otsu_t, _ = cv2.threshold(
            work, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        hi = int(otsu_t)
        lo = int(otsu_t * 0.5)
        if hi <= lo:
            hi = min(255, lo + 1)

    scales = _parse_scales(getattr(eparams, "detection_scales", [0.0, 1.0, 2.0]))
    edges = np.zeros_like(work)
    for sigma in scales:
        if sigma <= 0:
            blurred = work
        else:
            blurred = cv2.GaussianBlur(work, (0, 0), float(sigma))
        e = cv2.Canny(blurred, lo, hi)
        edges = np.maximum(edges, e)
    return edges


def extract_edge_polylines(
    gray: np.ndarray,
    fg_mask: np.ndarray | None,
    line_width_px: float,
    cfg: Config,
) -> List[List[Tuple[float, float]]]:
    """Run the full Canny → collapse → trace → simplify pipeline.

    Returned polylines are in pixel space (x, y) and have not yet been
    filtered by the drawn-mask guard — that step happens at emission time
    so we can order strokes by length first.
    """
    eparams = cfg.edge

    # Multi-scale CLAHE + Otsu Canny — handles portraits, hair texture, and
    # bright-background photos where the median-based auto-Canny breaks down.
    edges = _detect_edges_multiscale(gray, eparams)

    # Optional foreground restriction. fg_mask is the K-means foreground mask
    # which sits exactly *inside* the subject — Canny edges sit at the
    # subject↔background boundary and almost always fall a pixel outside the
    # mask, so we dilate the mask before AND-ing or we lose the silhouette.
    if fg_mask is not None and eparams.restrict_to_foreground:
        dilate_px = max(1, int(eparams.foreground_dilate_px))
        kfg = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * dilate_px + 1, 2 * dilate_px + 1)
        )
        fg_dilated = cv2.dilate(fg_mask.astype(np.uint8), kfg)
        edges = edges & (fg_dilated * 255)

    # Optional debug dump of the pre-trace edge map.
    if getattr(eparams, "debug_edges_path", ""):
        cv2.imwrite(eparams.debug_edges_path, edges)

    if edges.sum() == 0:
        return []

    # Parallel-edge collapse: dilate then thin. Anything closer than
    # collapse_radius_px merges into one centerline.
    collapse_r = max(0, int(round(line_width_px * eparams.collapse_factor / 2.0)))
    if collapse_r > 0:
        k = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * collapse_r + 1, 2 * collapse_r + 1)
        )
        thick = cv2.dilate((edges > 0).astype(np.uint8), k)
    else:
        thick = (edges > 0).astype(np.uint8)

    skeleton = _zhang_suen_thin(thick)

    # Trace into polylines.
    raw = _trace_polylines(skeleton)

    # Simplify and drop short noise. min_polyline_length is interpreted as
    # a geometric pixel length so it survives Douglas-Peucker (which can
    # legitimately collapse a long straight segment to 2 vertices).
    min_len_px = float(eparams.min_polyline_length)
    epsilon = float(eparams.simplify_epsilon)
    polys: List[List[Tuple[float, float]]] = []
    for p in raw:
        if len(p) < 2:
            continue
        if min_len_px > 0 and _polyline_length([(float(x), float(y)) for x, y in p]) < min_len_px:
            continue
        polys.append(_simplify(p, epsilon))

    # Order longest first so the guard prefers structural strokes.
    polys.sort(key=lambda p: -_polyline_length(p))
    return polys


def _polyline_length(poly: List[Tuple[float, float]]) -> float:
    if len(poly) < 2:
        return 0.0
    pts = np.asarray(poly, dtype=np.float64)
    d = np.diff(pts, axis=0)
    return float(np.sum(np.hypot(d[:, 0], d[:, 1])))


def _chain_polylines(
    polys: List[List[Tuple[float, float]]], max_gap_px: float,
) -> List[List[Tuple[float, float]]]:
    """Greedy nearest-endpoint chaining.

    The tracer fragments the skeleton at every junction: a four-spoke star
    becomes four short polylines whose endpoints all sit on the junction
    pixel. This pass walks the longest unchained polyline and keeps
    appending other polylines whose head or tail lands within max_gap_px
    of the current chain's tail (reversing them if their tail is closer).
    Result: drastically fewer polylines, each one much longer, so the
    rendered output reads as continuous strokes rather than dashes.
    """
    if max_gap_px <= 0 or len(polys) < 2:
        return [list(p) for p in polys if len(p) >= 2]

    pieces = [list(p) for p in polys if len(p) >= 2]
    if not pieces:
        return []

    n = len(pieces)
    heads = np.empty((n, 2), dtype=np.float64)
    tails = np.empty((n, 2), dtype=np.float64)
    for i, p in enumerate(pieces):
        heads[i] = p[0]
        tails[i] = p[-1]
    used = np.zeros(n, dtype=bool)
    threshold = float(max_gap_px)

    # Order chains longest-first so we extend structural strokes first.
    order = sorted(range(n), key=lambda i: -_polyline_length(pieces[i]))
    chained: List[List[Tuple[float, float]]] = []

    for start in order:
        if used[start]:
            continue
        used[start] = True
        chain = list(pieces[start])
        # Repeatedly try to attach an unused piece to the chain's tail.
        while True:
            ex, ey = chain[-1]
            cand = np.flatnonzero(~used)
            if cand.size == 0:
                break
            d_head = np.hypot(heads[cand, 0] - ex, heads[cand, 1] - ey)
            d_tail = np.hypot(tails[cand, 0] - ex, tails[cand, 1] - ey)
            best_h = int(np.argmin(d_head))
            best_t = int(np.argmin(d_tail))
            if d_head[best_h] <= d_tail[best_t]:
                best_dist = float(d_head[best_h])
                idx = int(cand[best_h])
                reverse = False
            else:
                best_dist = float(d_tail[best_t])
                idx = int(cand[best_t])
                reverse = True
            if best_dist > threshold:
                break
            used[idx] = True
            seg = pieces[idx]
            if reverse:
                seg = list(reversed(seg))
            # Skip the duplicate vertex if tail and seg start coincide.
            if seg and chain[-1] == seg[0]:
                chain.extend(seg[1:])
            else:
                chain.extend(seg)
        chained.append(chain)
    return chained


# ------------------------------------------------------------------ #
# Top-level: build ToolLayers from edges                                #
# ------------------------------------------------------------------ #

def build_edge_toolpaths(
    gray: np.ndarray,
    fg_mask: np.ndarray | None,
    cfg: Config,
    tool_idx: int = 0,
) -> List[ToolLayer]:
    """Generate edge-mode ToolLayers.

    Edges all go to a single tool (default 0). Multi-tool inputs still
    benefit from the segmenter's foreground mask via *fg_mask*.
    """
    H, W = gray.shape
    n_layers = cfg.layers.num_layers
    line_width_mm = cfg.extrusion.effective_line_width
    scale_x = cfg.machine.print_width / W
    scale_y = cfg.machine.print_height / H
    scale = min(scale_x, scale_y)
    line_width_px = line_width_mm / scale

    polys = extract_edge_polylines(gray, fg_mask, line_width_px, cfg)

    # Drawn-mask guard: rasterise emitted strokes at line_width thickness and
    # split incoming candidates around any pixels they'd overlap.
    guard_radius = max(1, int(round(line_width_px / 2.0)))
    min_sep_px = cfg.edge.min_separation_factor * line_width_px
    forbid_radius = max(1, int(round(min_sep_px / 2.0)))

    drawn_mask = np.zeros((H, W), dtype=np.uint8)
    survived: List[List[Tuple[float, float]]] = []
    min_len_px = float(cfg.edge.min_polyline_length)
    for poly in polys:
        for sub in _split_polyline_by_mask(poly, drawn_mask, forbid_radius):
            if len(sub) < 2:
                continue
            if min_len_px > 0 and _polyline_length(sub) < min_len_px:
                continue
            stamp = _stamp_polyline((H, W), sub, guard_radius)
            drawn_mask = np.maximum(drawn_mask, stamp)
            survived.append(sub)

    # Greedy chaining merges polylines whose endpoints meet — this undoes
    # most of the fragmentation the tracer's junction removal introduces,
    # so the rendered output reads as continuous strokes.
    chain_gap_px = float(cfg.edge.chain_gap_factor) * line_width_px
    emitted = _chain_polylines(survived, chain_gap_px)
    # Greedy nearest-neighbour ordering of the surviving polylines cuts
    # total travel distance (often 30-50%) without changing what's printed.
    emitted = order_polylines_greedy(emitted)

    # Wrap in ToolLayer(s). Edge mode only uses one tool, so other tools get
    # empty layers — keeps the writer's tool-change accounting consistent.
    all_layers: List[ToolLayer] = []
    n_tools = cfg.tools.num_tools
    for layer_idx in range(n_layers):
        for t in range(n_tools):
            tl = ToolLayer(tool_idx=t, layer_idx=layer_idx)
            if t == tool_idx and layer_idx == 0:
                # Only the first layer carries the strokes — re-tracing every
                # layer would just print the same lines on top of each other.
                for poly in emitted:
                    if len(poly) < 2:
                        continue
                    tl.moves.append(Move(x=poly[0][0], y=poly[0][1], extrude=False))
                    for x, y in poly[1:]:
                        tl.moves.append(Move(x=x, y=y, extrude=True))
            all_layers.append(tl)
    return all_layers
