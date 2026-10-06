"""Polyline and polygon helpers in page millimetres."""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np
import shapely
from scipy.spatial import cKDTree
from shapely import affinity
from shapely.geometry import LineString, Polygon
from shapely.ops import polygonize, unary_union

from .model import Stroke


def _arr(points) -> np.ndarray:
    return np.asarray(points, dtype=np.float64).reshape(-1, 2)


def _is_closed(p: np.ndarray) -> bool:
    return len(p) > 2 and bool(np.allclose(p[0], p[-1]))


def polyline_length(points) -> float:
    p = _arr(points)
    if len(p) < 2:
        return 0.0
    return float(np.hypot(*np.diff(p, axis=0).T).sum())


def resample(points, step: float) -> np.ndarray:
    """Evenly spaced points at most *step* apart; endpoints kept, so closed stays closed."""
    p = _arr(points)
    if len(p) < 2 or step <= 0:
        return p.copy()
    s = np.r_[0.0, np.cumsum(np.hypot(*np.diff(p, axis=0).T))]
    if s[-1] == 0:
        return p[[0, -1]].copy()
    t = np.linspace(0.0, s[-1], int(np.ceil(s[-1] / step)) + 1)
    return np.c_[np.interp(t, s, p[:, 0]), np.interp(t, s, p[:, 1])]


def smooth(points, window: float) -> np.ndarray:
    """Moving average over about *window* mm of arc length.

    Open polylines keep their endpoints; closed ones wrap around and stay closed.
    """
    p = _arr(points)
    length = polyline_length(p)
    if window <= 0 or len(p) < 3 or length == 0:
        return p.copy()
    step = min(window / 4.0, length / 8.0)
    q = resample(p, step)
    k = max(3, int(round(window / step)) | 1)
    h = k // 2
    kernel = np.ones(k) / k
    closed = _is_closed(p)
    if closed:
        ring = q[:-1]
        pad = np.take(ring, np.arange(-h, len(ring) + h), axis=0, mode="wrap")
    else:
        pad = np.vstack([np.repeat(q[:1], h, 0), q, np.repeat(q[-1:], h, 0)])
    out = np.c_[np.convolve(pad[:, 0], kernel, "valid"), np.convolve(pad[:, 1], kernel, "valid")]
    if closed:
        return np.vstack([out, out[:1]])
    out[0], out[-1] = q[0], q[-1]
    return out


def simplify(points, tol: float) -> np.ndarray:
    """Douglas-Peucker; endpoints kept."""
    p = _arr(points)
    if tol <= 0 or len(p) < 3:
        return p.copy()
    out = np.asarray(LineString(p).simplify(tol, preserve_topology=False).coords)
    return out if len(out) >= 2 else p[[0, -1]].copy()


def chaikin(points, iterations: int = 2) -> np.ndarray:
    """Chaikin corner cutting. Open polylines keep their endpoints."""
    p = _arr(points)
    for _ in range(iterations):
        if len(p) < 3:
            break
        closed = _is_closed(p)
        a = p[:-1]
        b = np.roll(a, -1, axis=0) if closed else p[1:]
        new = np.empty((2 * len(a), 2))
        new[0::2], new[1::2] = 0.75 * a + 0.25 * b, 0.25 * a + 0.75 * b
        p = np.vstack([new, new[:1]]) if closed else np.vstack([p[:1], new, p[-1:]])
    return p


def _winding(ring: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Winding number of a closed ring around each point (vectorised)."""
    r = np.vstack([ring, ring[:1]])
    x0, y0 = r[:-1, 0][:, None], r[:-1, 1][:, None]
    x1, y1 = r[1:, 0][:, None], r[1:, 1][:, None]
    px, py = pts[:, 0][None, :], pts[:, 1][None, :]
    cross = (x1 - x0) * (py - y0) - (px - x0) * (y1 - y0)
    up = (y0 <= py) & (y1 > py) & (cross > 0)
    down = (y0 > py) & (y1 <= py) & (cross < 0)
    return up.sum(axis=0) - down.sum(axis=0)


def rings_to_geometry(rings: Sequence[np.ndarray], rule: str = "nonzero"):
    """Fill rings with an SVG fill rule ("nonzero" or "evenodd").

    Rings may self-intersect, overlap or nest: they are noded into faces and each
    face is kept by the winding number at an interior point.
    """
    rings = [_arr(r) for r in rings]
    rings = [r[:-1] if _is_closed(r) else r for r in rings]
    rings = [r for r in rings if len(r) >= 3]
    if not rings:
        return Polygon()
    faces = list(polygonize(unary_union([LineString(np.vstack([r, r[:1]])) for r in rings])))
    if not faces:
        return Polygon()
    pts = np.array([f.representative_point().coords[0] for f in faces])
    w = sum(_winding(r, pts) for r in rings)
    keep = (w % 2 != 0) if rule == "evenodd" else (w != 0)
    kept = [f for f, k in zip(faces, keep) if k]
    return unary_union(kept) if kept else Polygon()


def geometry_rings(geom) -> List[np.ndarray]:
    """Every exterior and interior ring of a (Multi)Polygon as a closed (N, 2) array."""
    out: List[np.ndarray] = []
    for poly in getattr(geom, "geoms", [geom]):
        if poly.is_empty or poly.geom_type != "Polygon":
            continue
        out.append(np.asarray(poly.exterior.coords))
        out.extend(np.asarray(i.coords) for i in poly.interiors)
    return out


def _iter_lines(g):
    if g.is_empty:
        return
    if g.geom_type == "LineString":
        yield g
    elif hasattr(g, "geoms"):
        for sub in g.geoms:
            yield from _iter_lines(sub)


def hatch_lines(geom, spacing: float, angle_deg: float, connect: bool = False,
                connect_tol: float = 0.0) -> List[np.ndarray]:
    """Parallel scan lines clipped to *geom*, in serpentine order.

    Rows sit at (k + 0.5) * spacing in the rotated frame, so neighbouring
    regions hatched at the same angle line up. With *connect*, consecutive
    segments are joined when the connector is at most 2.5 * spacing long and
    stays inside geom grown by *connect_tol*.
    """
    if geom is None or geom.is_empty or spacing <= 0:
        return []
    rot = affinity.rotate(geom, -angle_deg, origin=(0, 0))
    minx, miny, maxx, maxy = rot.bounds
    ks = np.arange(int(np.floor(miny / spacing)) - 1, int(np.ceil(maxy / spacing)) + 1)
    ys = (ks + 0.5) * spacing
    ys = ys[(ys > miny) & (ys < maxy)]
    if not len(ys):
        return []
    scan = shapely.linestrings([[(minx - 1, y), (maxx + 1, y)] for y in ys])
    ordered: List[np.ndarray] = []
    flip = False
    for g in shapely.intersection(scan, rot):
        segs = []
        for ls in _iter_lines(g):
            c = np.asarray(ls.coords)[[0, -1]]
            if c[0, 0] > c[1, 0]:
                c = c[::-1]
            if c[1, 0] - c[0, 0] > 1e-9:
                segs.append(c)
        if not segs:
            continue
        segs.sort(key=lambda c: c[0, 0])
        if flip:
            segs = [c[::-1] for c in reversed(segs)]
        ordered.extend(segs)
        flip = not flip
    if connect and ordered:
        zone = rot.buffer(connect_tol)
        shapely.prepare(zone)
        merged = [ordered[0]]
        for c in ordered[1:]:
            a, b = merged[-1][-1], c[0]
            if np.hypot(*(b - a)) <= 2.5 * spacing and zone.covers(LineString([a, b])):
                merged[-1] = np.vstack([merged[-1], c])
            else:
                merged.append(c)
        ordered = merged
    th = np.radians(angle_deg)
    rmat = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    return [c @ rmat.T for c in ordered]


def order_strokes(strokes: Sequence[Stroke], start: Optional[Tuple[float, float]] = None) -> List[Stroke]:
    """Greedy nearest-neighbour order from *start* (default: the first stroke's start).

    Open strokes may reverse; closed strokes may start at any of up to 32 evenly
    spaced vertices and are rotated to start there.
    """
    if not strokes:
        return []
    pts, owner, vert = [], [], []
    for i, s in enumerate(strokes):
        p = s.points
        if s.closed:
            n = len(p) - 1
            idx = np.unique(np.linspace(0, n - 1, min(n, 32)).round().astype(int))
        else:
            idx = np.array([0, len(p) - 1])
        pts.append(p[idx])
        owner.append(np.full(len(idx), i))
        vert.append(idx)
    pts = np.vstack(pts)
    owner = np.concatenate(owner)
    vert = np.concatenate(vert)
    tree = cKDTree(pts)
    used = np.zeros(len(strokes), dtype=bool)
    cursor = np.asarray(start if start is not None else strokes[0].points[0], float)
    out: List[Stroke] = []
    for _ in range(len(strokes)):
        k = 8
        while True:                              # widen the search until a free end shows up
            kk = min(k, len(pts))
            _, j = tree.query(cursor, k=kk)
            j = np.atleast_1d(j)
            free = j[~used[owner[j]]]
            if len(free) or kk == len(pts):
                break
            k *= 4
        c = int(free[0])
        i = int(owner[c])
        used[i] = True
        s = strokes[i]
        p = s.points
        if s.closed:
            v = int(vert[c])
            ring = p[:-1]
            p = np.vstack([np.roll(ring, -v, axis=0), ring[v:v + 1]])
        elif vert[c] != 0:
            p = p[::-1]
        out.append(Stroke(points=p, tool=s.tool, kind=s.kind))
        cursor = p[-1]
    return out


def chain_polylines(polys: Sequence[np.ndarray], max_gap: float) -> List[np.ndarray]:
    """Join open polylines whose ends lie within *max_gap*, starting from the longest."""
    pieces = [_arr(p) for p in polys if len(p) >= 2]
    if max_gap <= 0 or len(pieces) < 2:
        return pieces
    heads = np.array([p[0] for p in pieces])
    tails = np.array([p[-1] for p in pieces])
    closed = np.array([_is_closed(p) for p in pieces])
    used = closed.copy()                         # closed pieces never join anything
    out = [p for p, c in zip(pieces, closed) if c]
    for start in sorted(range(len(pieces)), key=lambda i: -polyline_length(pieces[i])):
        if used[start]:
            continue
        used[start] = True
        chain = pieces[start]
        for _ in range(2):                       # extend the tail, then (reversed) the head
            while True:
                cand = np.flatnonzero(~used)
                if not len(cand):
                    break
                end = chain[-1]
                dh = np.hypot(*(heads[cand] - end).T)
                dt = np.hypot(*(tails[cand] - end).T)
                bh, bt = int(np.argmin(dh)), int(np.argmin(dt))
                if min(dh[bh], dt[bt]) > max_gap:
                    break
                if dh[bh] <= dt[bt]:
                    j = int(cand[bh])
                    seg = pieces[j]
                else:
                    j = int(cand[bt])
                    seg = pieces[j][::-1]
                used[j] = True
                chain = np.vstack([chain, seg[1:] if np.allclose(chain[-1], seg[0]) else seg])
            chain = chain[::-1]
        out.append(chain)
    return out


def deoverlap(polys: Sequence[np.ndarray], width_mm: float, height_mm: float,
              min_sep: float, min_length: float) -> List[np.ndarray]:
    """Keep polylines in the given priority order, cutting away any part that
    comes within *min_sep* of a polyline already kept. Pieces shorter than
    *min_length* are dropped."""
    px = min_sep / 4.0                           # occupancy grid cell size
    W = int(np.ceil(width_mm / px)) + 2
    H = int(np.ceil(height_mm / px)) + 2
    mask = np.zeros((H, W), np.uint8)
    radius = max(1, int(round(min_sep / px)))
    out: List[np.ndarray] = []
    for p in polys:
        p = _arr(p)
        q = resample(p, px / 2.0)
        if len(q) < 2:
            continue
        ij = np.clip(np.floor(q / px).astype(int), 0, [W - 1, H - 1])
        free = mask[ij[:, 1], ij[:, 0]] == 0
        runs, start = [], None
        for i, f in enumerate(free):
            if f and start is None:
                start = i
            if (not f or i == len(free) - 1) and start is not None:
                runs.append(q[start:i + 1 if f else i])
                start = None
        if _is_closed(p) and len(runs) > 1 and free[0] and free[-1]:
            runs[0] = np.vstack([runs.pop(), runs[0][1:]])   # rejoin across the seam
        kept = [simplify(r, px / 4.0) for r in runs if len(r) >= 2 and polyline_length(r) >= min_length]
        for seg in kept:                         # cell (i, j) is centred on pixel (i, j) for cv2
            pts = np.round((seg / px - 0.5) * 16).astype(np.int32).reshape(-1, 1, 2)
            cv2.polylines(mask, [pts], False, 1, thickness=2 * radius + 1, shift=4)
        out.extend(kept)
    return out
