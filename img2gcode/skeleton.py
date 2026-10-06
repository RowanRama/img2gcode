"""Trace a binary mask's skeleton into smooth centreline strokes in page mm.

The skeleton becomes a graph: nodes are endpoints and junction clusters, edges are
the pixel runs between them. Short spurs are pruned, edges are joined through
junctions by straightest continuation, and the resulting chains are smoothed.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import cv2
import numpy as np
from skimage.morphology import skeletonize

from .geometry import polyline_length, simplify, smooth

_OFFSETS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

Edge = Tuple[int, int, np.ndarray]  # (node_a, node_b, (N, 2) xy path in px, node_a → node_b)
End = Tuple[int, int]  # (edge index, side): side 0 is the node_a end, side 1 the node_b end


def skeleton_strokes(mask, mm_per_px: float, *, spur_mm: float, smooth_mm: float,
                     simplify_mm: float, min_length_mm: float,
                     max_turn_deg: float = 60.0) -> List[np.ndarray]:
    """Centreline strokes of *mask* in page mm; closed loops have first == last.

    Spurs shorter than *spur_mm* are pruned; at junctions, edges continue through
    when they turn by less than *max_turn_deg*. Chains shorter than
    *min_length_mm* are dropped, the rest smoothed over *smooth_mm* and simplified.
    """
    skel = skeletonize(np.asarray(mask, dtype=bool))
    pos, edges, loops = _graph(skel)
    spur_px = spur_mm / mm_per_px
    edges = _prune_spurs(len(pos), edges, spur_px)
    chains = _chains(edges, max(3.0, spur_px), max_turn_deg) + loops
    out = []
    for c in chains:
        p = (c + 0.5) * mm_per_px
        if polyline_length(p) >= min_length_mm:
            out.append(simplify(smooth(p, smooth_mm), simplify_mm))
    return out


def _graph(skel: np.ndarray) -> Tuple[np.ndarray, List[Edge], List[np.ndarray]]:
    """Skeleton → (node positions, edges, node-less closed loops), all in pixel xy."""
    ys, xs = np.nonzero(skel)
    n = len(ys)
    xy = np.c_[xs, ys].astype(np.float64)
    idx = np.full((skel.shape[0] + 2, skel.shape[1] + 2), -1, np.int32)
    idx[ys + 1, xs + 1] = np.arange(n)
    nb = np.stack([idx[ys + 1 + dy, xs + 1 + dx] for dy, dx in _OFFSETS], axis=1)
    count = (nb >= 0).sum(axis=1)
    nbrs = [[j for j in row if j >= 0] for row in nb.tolist()]

    # Nodes: every endpoint, plus one node per 8-connected cluster of junction pixels.
    node_of = np.full(n, -1, np.int64)
    ends = np.flatnonzero(count == 1)
    node_of[ends] = np.arange(len(ends))
    junc = np.flatnonzero(count >= 3)
    jmap = np.zeros(skel.shape, np.uint8)
    jmap[ys[junc], xs[junc]] = 1
    n_lab, labels = cv2.connectedComponents(jmap, connectivity=8)
    lab = labels[ys[junc], xs[junc]] - 1
    node_of[junc] = len(ends) + lab
    size = np.bincount(lab, minlength=n_lab - 1)
    centroids = np.c_[np.bincount(lab, xs[junc], n_lab - 1),
                      np.bincount(lab, ys[junc], n_lab - 1)] / np.maximum(size, 1)[:, None]
    pos = np.vstack([xy[ends], centroids])

    node_px = np.flatnonzero(node_of >= 0).tolist()
    node_of = node_of.tolist()
    visited = [False] * n
    edges: List[Edge] = []
    for p in node_px:
        a = node_of[p]
        for q in nbrs[p]:
            if node_of[q] >= 0:  # two nodes touching: an edge with no interior
                if node_of[q] != a and p < q:
                    edges.append((a, node_of[q], pos[[a, node_of[q]]]))
                continue
            if visited[q]:
                continue
            path = [pos[a]]
            prev, cur = p, q
            while node_of[cur] < 0:
                visited[cur] = True
                path.append(xy[cur])
                prev, cur = cur, _next(nbrs[cur], prev)
            path.append(pos[node_of[cur]])
            edges.append((a, node_of[cur], np.array(path)))

    # Components with no nodes at all are closed rings of count-2 pixels.
    loops = []
    for s in np.flatnonzero(count == 2).tolist():
        if visited[s]:
            continue
        visited[s] = True
        path = [xy[s]]
        prev, cur = s, nbrs[s][0]
        while cur != s:
            visited[cur] = True
            path.append(xy[cur])
            prev, cur = cur, _next(nbrs[cur], prev)
        path.append(xy[s])
        loops.append(np.array(path))
    return pos, edges, loops


def _next(nbrs: List[int], prev: int) -> int:
    """The neighbour of a count-2 pixel that is not *prev*."""
    return nbrs[1] if nbrs[0] == prev else nbrs[0]


def _prune_spurs(n_nodes: int, edges: List[Edge], spur_px: float) -> List[Edge]:
    """Drop edges shorter than *spur_px* that join an endpoint to a junction.

    Shortest first, with degrees updated as we go, so a junction that drops to
    degree 2 keeps its remaining edges (a small Y loses one arm, not all three).
    Removing a spur never creates a new one, so a single pass is stable.
    """
    deg = np.zeros(n_nodes, np.int64)
    for a, b, _ in edges:
        deg[a] += 1
        deg[b] += 1
    lengths = [polyline_length(path) for _, _, path in edges]
    keep = [True] * len(edges)
    for i in sorted(range(len(edges)), key=lengths.__getitem__):
        if lengths[i] >= spur_px:
            break
        a, b, _ = edges[i]
        if min(deg[a], deg[b]) == 1 and max(deg[a], deg[b]) >= 3:
            keep[i] = False
            deg[a] -= 1
            deg[b] -= 1
    return [e for e, k in zip(edges, keep) if k]


def _direction(path: np.ndarray, side: int, look_px: float) -> np.ndarray:
    """Unit vector from the node at *side* of *path* to the point *look_px* along it."""
    p = path if side == 0 else path[::-1]
    s = np.r_[0.0, np.cumsum(np.hypot(*np.diff(p, axis=0).T))]
    k = min(int(np.searchsorted(s, look_px)), len(p) - 1)
    v = p[k] - p[0]
    norm = np.hypot(*v)
    return v / norm if norm > 0 else v


def _chains(edges: List[Edge], look_px: float, max_turn_deg: float) -> List[np.ndarray]:
    """Join edges through nodes into chains (pixel xy); cycles come back closed."""
    incident: Dict[int, List[End]] = {}
    for i, (a, b, _) in enumerate(edges):
        incident.setdefault(a, []).append((i, 0))
        incident.setdefault(b, []).append((i, 1))

    partner: Dict[End, End] = {}
    for ends in incident.values():
        if len(ends) == 2:
            partner[ends[0]], partner[ends[1]] = ends[1], ends[0]
        elif len(ends) >= 3:
            d = [_direction(edges[e][2], side, look_px) for e, side in ends]
            pairs = sorted(
                (np.degrees(np.arccos(np.clip(-d[i] @ d[j], -1.0, 1.0))), i, j)
                for i in range(len(ends)) for j in range(i + 1, len(ends)))
            for dev, i, j in pairs:
                if dev >= max_turn_deg:
                    break
                if ends[i] not in partner and ends[j] not in partner:
                    partner[ends[i]], partner[ends[j]] = ends[j], ends[i]

    used = [False] * len(edges)

    def follow(e: int, side: int) -> np.ndarray:
        parts = []
        while True:
            used[e] = True
            path = edges[e][2] if side == 0 else edges[e][2][::-1]
            parts.append(path if not parts else path[1:])
            nxt = partner.get((e, 1 - side))
            if nxt is None or used[nxt[0]]:
                return np.vstack(parts)
            e, side = nxt

    chains = []
    for e in range(len(edges)):  # open chains start at unpaired ends
        for side in (0, 1):
            if not used[e] and (e, side) not in partner:
                chains.append(follow(e, side))
    for e in range(len(edges)):  # every edge left over lies on a cycle
        if not used[e]:
            chains.append(follow(e, 0))
    return chains
