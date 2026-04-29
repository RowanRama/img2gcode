"""Toolpath generation from a segmented label map.

For each tool:
  1. Extract the binary mask for that tool.
  2. Trace the outer contours (perimeter loops).
  3. Fill the interior with a zigzag (raster) infill pattern.

All coordinates are in *pixel* space here; the GCode writer applies the
scaling transform from pixel → mm.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np
import cv2

from .config import Config


# ------------------------------------------------------------------ #
# Data structures                                                       #
# ------------------------------------------------------------------ #

@dataclass
class Move:
    """A single XY move in pixel coordinates."""
    x: float
    y: float
    extrude: bool = True   # False = travel


@dataclass
class ToolLayer:
    """All moves for a single tool on a single layer."""
    tool_idx: int
    layer_idx: int
    moves: List[Move] = field(default_factory=list)


# ------------------------------------------------------------------ #
# Travel-order optimisation                                             #
# ------------------------------------------------------------------ #

def order_polylines_greedy(
    polys: List[List[Tuple[float, float]]],
    start_xy: Tuple[float, float] | None = None,
) -> List[List[Tuple[float, float]]]:
    """Greedy nearest-neighbour ordering of polylines.

    For each step, picks the unvisited polyline whose head OR tail is
    closest to the current cursor position; the polyline is reversed if
    its tail was the closer end so the cursor advances naturally. Used
    to cut total travel distance — naive emission order can do double
    or triple the necessary travel on busy edge / infill outputs.

    Each input polyline is treated as an indivisible unit (we never
    split them mid-stroke), so the print sequence is preserved.
    """
    if len(polys) < 2:
        return [list(p) for p in polys]

    pieces = [list(p) for p in polys]
    n = len(pieces)
    heads = np.empty((n, 2), dtype=np.float64)
    tails = np.empty((n, 2), dtype=np.float64)
    for i, p in enumerate(pieces):
        heads[i] = p[0]
        tails[i] = p[-1]

    used = np.zeros(n, dtype=bool)
    ordered: List[List[Tuple[float, float]]] = []

    if start_xy is None:
        # Seed with polyline 0 in its given orientation.
        used[0] = True
        ordered.append(pieces[0])
        cursor = (float(pieces[0][-1][0]), float(pieces[0][-1][1]))
    else:
        cursor = (float(start_xy[0]), float(start_xy[1]))

    while not used.all():
        cand = np.flatnonzero(~used)
        d_head = np.hypot(heads[cand, 0] - cursor[0], heads[cand, 1] - cursor[1])
        d_tail = np.hypot(tails[cand, 0] - cursor[0], tails[cand, 1] - cursor[1])
        bh = int(np.argmin(d_head))
        bt = int(np.argmin(d_tail))
        if d_head[bh] <= d_tail[bt]:
            idx = int(cand[bh])
            seg = pieces[idx]
        else:
            idx = int(cand[bt])
            seg = list(reversed(pieces[idx]))
        used[idx] = True
        ordered.append(seg)
        cursor = (float(seg[-1][0]), float(seg[-1][1]))

    return ordered


def _polyline_travel_total(polys: List[List[Tuple[float, float]]]) -> float:
    """Sum of inter-polyline travel distances (start_i+1 - end_i)."""
    if len(polys) < 2:
        return 0.0
    total = 0.0
    prev_end = polys[0][-1]
    for p in polys[1:]:
        total += float(np.hypot(p[0][0] - prev_end[0], p[0][1] - prev_end[1]))
        prev_end = p[-1]
    return total


# ------------------------------------------------------------------ #
# Contour / perimeter extraction                                        #
# ------------------------------------------------------------------ #

def _extract_contours(binary_mask: np.ndarray) -> List[Tuple[np.ndarray, bool]]:
    """Return contours as (points, is_hole) tuples.

    Outer contours have is_hole=False; inner hole contours have is_hole=True.
    No morphological dilation is applied so letter gaps and interior holes are preserved.
    """
    mask = binary_mask.astype(np.uint8)
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    result = []
    if hierarchy is None:
        return result
    hierarchy = hierarchy[0]  # (N, 4): [next, prev, child, parent]
    for i, c in enumerate(contours):
        pts = c.reshape(-1, 2).astype(float)
        if len(pts) < 3:
            continue
        is_hole = hierarchy[i][3] != -1  # has a parent → it's a hole
        result.append((pts, is_hole))
    return result


def _offset_contour(
    contour: np.ndarray, offset_px: float
) -> List[np.ndarray]:
    """Inward-offset a contour by *offset_px* pixels using cv2 polygon clipping."""
    # Use the polygon erode approach via cv2 with a temporary mask
    if len(contour) < 3:
        return []
    pts = contour.astype(np.int32).reshape((-1, 1, 2))
    # We create a temporary mask, erode it, and re-extract contours
    xs, ys = contour[:, 0], contour[:, 1]
    x_min, y_min = int(xs.min()) - 4, int(ys.min()) - 4
    x_max, y_max = int(xs.max()) + 4, int(ys.max()) + 4
    w, h = x_max - x_min + 1, y_max - y_min + 1

    tmp_mask = np.zeros((h, w), dtype=np.uint8)
    shifted = pts - np.array([[[x_min, y_min]]])
    cv2.fillPoly(tmp_mask, [shifted.astype(np.int32)], 1)

    k = max(1, int(offset_px))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1))
    eroded = cv2.erode(tmp_mask, kernel)

    inner_contours, _ = cv2.findContours(
        eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    result = []
    for c in inner_contours:
        c_shifted = c.reshape(-1, 2).astype(float)
        c_shifted[:, 0] += x_min
        c_shifted[:, 1] += y_min
        result.append(c_shifted)
    return result


# ------------------------------------------------------------------ #
# Zigzag infill                                                         #
# ------------------------------------------------------------------ #

def _zigzag_infill(
    binary_mask: np.ndarray,
    line_spacing_px: float,
    angle_deg: float,
    max_connect_factor: float = 2.0,
    connect_lines: bool = True,
) -> List[List[Tuple[float, float]]]:
    """Generate infill lines clipped to *binary_mask*.

    When ``connect_lines`` is True, segments from consecutive scan rows are
    joined into one continuous serpentine polyline whenever the endpoint-to-
    endpoint distance is within ``max_connect_factor * line_spacing_px`` —
    PrusaSlicer-style.  When False, each scan-line segment is returned as its
    own polyline (the writer emits a travel between them), giving an evenly-
    spaced parallel hatch without edge connector artefacts.

    Returns a list of polylines; each polyline is a list of (x, y) tuples.
    """
    H, W = binary_mask.shape

    # Rotate into a canvas large enough to hold the whole mask after rotation —
    # otherwise content near the corners is clipped by warpAffine and never
    # gets scanned, leaving infill voids in the final output.
    center = (W / 2, H / 2)
    rot_mat = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    abs_cos = abs(rot_mat[0, 0])
    abs_sin = abs(rot_mat[0, 1])
    new_W = int(np.ceil(H * abs_sin + W * abs_cos))
    new_H = int(np.ceil(H * abs_cos + W * abs_sin))
    rot_mat[0, 2] += (new_W - W) / 2.0
    rot_mat[1, 2] += (new_H - H) / 2.0
    inv_rot = cv2.invertAffineTransform(rot_mat)
    rotated_mask = cv2.warpAffine(
        binary_mask.astype(np.uint8), rot_mat, (new_W, new_H),
        flags=cv2.INTER_NEAREST,
    )

    # --- Pass 1: collect every segment in scan order ----------------------
    # Direction only flips when a row actually has segments so that consecutive
    # filled rows always alternate sides — empty rows must not consume a flip.
    # row_data accumulates (row_y, starts_array, ends_array) for rows with content.
    seg_rows: List[int] = []      # rotated-y for each segment, in scan order
    seg_starts_x: List[int] = []  # rotated-x at the "left" end of each segment
    seg_ends_x: List[int] = []    # rotated-x at the "right" end of each segment
    seg_dir: List[int] = []       # +1 = traverse L→R, -1 = R→L

    y = line_spacing_px / 2.0
    direction = 1
    last_row_y = -1
    while y < new_H:
        row_y = int(round(y))
        # Skip rows already scanned (sub-pixel spacing collapses iterations)
        if 0 <= row_y < new_H and row_y != last_row_y:
            last_row_y = row_y
            row = rotated_mask[row_y]
            # Vectorised run-length scan: pad with 0, diff, locate ±1 transitions
            padded = np.empty(row.size + 2, dtype=np.int8)
            padded[0] = 0
            padded[-1] = 0
            padded[1:-1] = row
            diffs = np.diff(padded)
            row_starts = np.flatnonzero(diffs == 1)
            row_ends = np.flatnonzero(diffs == -1) - 1

            if row_starts.size:
                if direction == 1:
                    order = range(row_starts.size)
                else:
                    # Right-to-left: emit segments right→left, swap start/end
                    order = range(row_starts.size - 1, -1, -1)
                for i in order:
                    seg_rows.append(row_y)
                    seg_starts_x.append(int(row_starts[i]))
                    seg_ends_x.append(int(row_ends[i]))
                    seg_dir.append(direction)
                direction = -direction
        y += line_spacing_px

    n_segs = len(seg_rows)
    if n_segs == 0:
        return []

    # --- Vectorised inverse rotation -------------------------------------
    # Build a (2*N, 3) homogeneous matrix of rotated-space endpoints, transform
    # all in one matmul, then split back into start/end arrays in original space.
    rot_pts = np.empty((2 * n_segs, 3), dtype=np.float64)
    rot_pts[:, 2] = 1.0
    s_x = np.asarray(seg_starts_x, dtype=np.float64)
    e_x = np.asarray(seg_ends_x, dtype=np.float64)
    rows_y = np.asarray(seg_rows, dtype=np.float64)
    dirs = np.asarray(seg_dir, dtype=np.int8)
    # Even rows = "from" point, odd rows = "to" point
    # Direction encodes which physical end is "from": +1 → s_x first, -1 → e_x first
    from_x = np.where(dirs == 1, s_x, e_x)
    to_x = np.where(dirs == 1, e_x, s_x)
    rot_pts[0::2, 0] = from_x
    rot_pts[0::2, 1] = rows_y
    rot_pts[1::2, 0] = to_x
    rot_pts[1::2, 1] = rows_y
    orig_pts = rot_pts @ inv_rot.T  # (2N, 2)

    starts_arr = orig_pts[0::2]      # (N, 2): "from" endpoint per segment
    ends_arr = orig_pts[1::2]        # (N, 2): "to" endpoint per segment
    raw_segs: List[List[Tuple[float, float]]] = [
        [(float(starts_arr[i, 0]), float(starts_arr[i, 1])),
         (float(ends_arr[i, 0]),   float(ends_arr[i, 1]))]
        for i in range(n_segs)
    ]

    # Fast path: caller wants evenly-spaced parallel hatch — skip chaining.
    if not connect_lines:
        return raw_segs

    # --- Pass 2: greedy nearest-neighbour chaining -----------------------
    # Walk one connected "column" of parallel infill lines completely before
    # jumping to another column.  Uses a boolean ``visited`` mask so each step
    # is a vectorised argmin over the unvisited subset rather than rebuilding
    # numpy arrays / list-removing on every iteration.
    threshold = line_spacing_px * max_connect_factor
    polylines: List[List[Tuple[float, float]]] = []
    visited = np.zeros(n_segs, dtype=bool)

    visited[0] = True
    chain: List[Tuple[float, float]] = list(raw_segs[0])
    n_remaining = n_segs - 1

    while n_remaining > 0:
        ex, ey = chain[-1]
        unv_idx = np.flatnonzero(~visited)
        d_start = np.hypot(starts_arr[unv_idx, 0] - ex, starts_arr[unv_idx, 1] - ey)
        d_end = np.hypot(ends_arr[unv_idx, 0] - ex, ends_arr[unv_idx, 1] - ey)

        bs = int(np.argmin(d_start))
        be = int(np.argmin(d_end))
        if d_start[bs] <= d_end[be]:
            best_dist = float(d_start[bs])
            best_seg_idx = int(unv_idx[bs])
            reverse_it = False
        else:
            best_dist = float(d_end[be])
            best_seg_idx = int(unv_idx[be])
            reverse_it = True

        if best_dist > threshold:
            polylines.append(chain)
            next_idx = int(unv_idx[0])
            visited[next_idx] = True
            n_remaining -= 1
            chain = list(raw_segs[next_idx])
            continue

        visited[best_seg_idx] = True
        n_remaining -= 1
        seg = raw_segs[best_seg_idx]
        if reverse_it:
            chain.append(seg[1])
            chain.append(seg[0])
        else:
            chain.append(seg[0])
            chain.append(seg[1])

    polylines.append(chain)
    return polylines


# ------------------------------------------------------------------ #
# Debug visualisation                                                   #
# ------------------------------------------------------------------ #

TOOL_COLORS_MPL = ["#e63946", "#2a9d8f", "#f4a261", "#457b9d", "#8ecae6"]


def show_toolpath_debug(
    binary: np.ndarray,
    contours: List[Tuple[np.ndarray, bool]],
    offset_contours: List[List[np.ndarray]],
    fill_mask: np.ndarray,
    infill_lines: List[List[Tuple[float, float]]],
    tool_idx: int,
    layer_idx: int,
) -> None:
    """Plot binary mask, perimeter contours, eroded fill mask, and infill lines."""
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    color = TOOL_COLORS_MPL[tool_idx % len(TOOL_COLORS_MPL)]
    hole_color = "#ffbe0b"
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(f"Tool {tool_idx}  –  Layer {layer_idx}", fontweight="bold")

    # Panel 1: binary mask + outer contours (solid) and holes (yellow dashed)
    ax = axes[0]
    ax.set_title("Binary mask + perimeter contours")
    ax.imshow(np.flipud(binary), cmap="gray", origin="upper")
    H = binary.shape[0]
    for c, is_hole in contours:
        pts = np.array(c)
        closed = np.vstack([pts, pts[0]])
        c_color = hole_color if is_hole else color
        ax.plot(closed[:, 0], H - 1 - closed[:, 1], color=c_color, lw=1.5,
                linestyle="--" if is_hole else "-")
    for loop_contours in offset_contours:
        for c in loop_contours:
            pts = np.array(c)
            closed = np.vstack([pts, pts[0]])
            ax.plot(closed[:, 0], H - 1 - closed[:, 1], color=color, lw=1, linestyle="--", alpha=0.6)

    # Panel 2: eroded fill mask
    ax = axes[1]
    ax.set_title("Eroded fill mask")
    ax.imshow(np.flipud(fill_mask), cmap="gray", origin="upper")

    # Panel 3: infill lines
    ax = axes[2]
    ax.set_title("Infill lines")
    ax.set_facecolor("#111")
    ax.set_xlim(0, binary.shape[1])
    ax.set_ylim(0, binary.shape[0])
    ax.set_aspect("equal")
    segs = []
    for poly in infill_lines:
        for i in range(len(poly) - 1):
            x0, y0 = poly[i]
            x1, y1 = poly[i + 1]
            segs.append([(x0, H - 1 - y0), (x1, H - 1 - y1)])
    lc = LineCollection(segs, colors=color, linewidths=0.8)
    ax.add_collection(lc)

    for ax in axes:
        ax.axis("off")
    plt.tight_layout()
    plt.show()


# ------------------------------------------------------------------ #
# Main toolpath builder                                                 #
# ------------------------------------------------------------------ #

def build_toolpaths(
    label_map: np.ndarray,
    cfg: Config,
    exclusion_zones: List[Tuple[int, int, int, int]] | None = None,
) -> List[ToolLayer]:
    """Build toolpath moves for all tools and all layers.

    Parameters
    ----------
    label_map       : (H, W) int array, -1 = background, 0…n-1 = tool
    cfg             : loaded Config object
    exclusion_zones : optional list of (x1, y1, x2, y2) pixel rectangles.
                      Infill is suppressed inside these regions (perimeters
                      are unaffected).

    Returns
    -------
    List of ToolLayer objects ready for GCode writing.
    """
    H, W = label_map.shape
    n_tools = cfg.tools.num_tools
    n_layers = cfg.layers.num_layers
    nozzle_mm = cfg.extrusion.nozzle_diameter
    line_width_mm = cfg.extrusion.effective_line_width

    # Resolve line spacing: explicit override takes priority, else derive from density
    if cfg.infill.line_spacing > 0:
        spacing_mm = cfg.infill.line_spacing
    else:
        density = max(1.0, min(100.0, cfg.infill.density))
        spacing_mm = line_width_mm / (density / 100.0)

    # pixel → mm scaling (image maps onto print_width × print_height)
    scale_x = cfg.machine.print_width / W
    scale_y = cfg.machine.print_height / H
    scale = min(scale_x, scale_y)
    spacing_px = spacing_mm / scale
    nozzle_px = nozzle_mm / scale
    line_width_px = line_width_mm / scale
    perimeter_loops = cfg.layers.perimeter_loops

    all_layers: List[ToolLayer] = []

    for layer_idx in range(n_layers):
        angle = cfg.infill.angle
        if layer_idx % 2 == 1:
            angle = (angle + 90) % 180  # alternate direction each layer

        for tool_idx in range(n_tools):
            tl = ToolLayer(tool_idx=tool_idx, layer_idx=layer_idx)
            binary = (label_map == tool_idx).astype(np.uint8)

            if binary.sum() == 0:
                all_layers.append(tl)
                continue

            # Inset against neighbouring tools so adjacent perimeters don't
            # overlap on the shared boundary. We dilate the *other* tools'
            # mask by half the line width and subtract it from this tool's
            # binary — this pulls the perimeter back only where it touches
            # another tool, leaving boundaries against background untouched.
            other_tools = ((label_map != tool_idx) & (label_map != -1)).astype(np.uint8)
            if other_tools.sum() > 0:
                inset_px = max(1, int(round(line_width_px / 2.0)))
                inset_kernel = cv2.getStructuringElement(
                    cv2.MORPH_ELLIPSE, (2 * inset_px + 1, 2 * inset_px + 1)
                )
                encroach = cv2.dilate(other_tools, inset_kernel)
                binary = (binary & ~encroach).astype(np.uint8)
                if binary.sum() == 0:
                    all_layers.append(tl)
                    continue

            # --- Perimeter loops -------------------------------------------
            # Collect each loop iteration as a list of polylines, order them
            # with greedy nearest-neighbour, then emit. Closed loops are
            # represented by appending the start vertex to the end so the
            # ordering pass treats them as ordinary polylines.
            contours = _extract_contours(binary)
            offset_contours: List[List[np.ndarray]] = []
            cursor = (tl.moves[-1].x, tl.moves[-1].y) if tl.moves else None
            for loop in range(perimeter_loops):
                offset = loop * line_width_px
                loop_polys: List[List[Tuple[float, float]]] = []
                for c, is_hole in contours:
                    effective_offset = -offset if is_hole else offset
                    if loop == 0:
                        loop_contour = c[::-1] if is_hole else c
                    else:
                        offset_result = _offset_contour(c, effective_offset)
                        if not offset_result:
                            continue
                        offset_contours.append(offset_result)
                        loop_contour = offset_result[0]
                        if is_hole:
                            loop_contour = loop_contour[::-1]
                    if len(loop_contour) < 2:
                        continue
                    pts = [(float(p[0]), float(p[1])) for p in loop_contour]
                    pts.append(pts[0])  # close the loop
                    loop_polys.append(pts)

                if not loop_polys:
                    continue
                ordered_loops = order_polylines_greedy(loop_polys, start_xy=cursor)
                for poly in ordered_loops:
                    tl.moves.append(Move(x=poly[0][0], y=poly[0][1], extrude=False))
                    for pt in poly[1:]:
                        tl.moves.append(Move(x=pt[0], y=pt[1], extrude=True))
                cursor = (ordered_loops[-1][-1][0], ordered_loops[-1][-1][1])

            # --- Infill ------------------------------------------------------
            # Erode mask by perimeter thickness before filling
            erode_px = max(1, int(perimeter_loops * line_width_px))
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (2 * erode_px + 1, 2 * erode_px + 1)
            )
            fill_mask = cv2.erode(binary, kernel)

            # Zero out any user-selected exclusion rectangles
            if exclusion_zones:
                for x1, y1, x2, y2 in exclusion_zones:
                    fill_mask[y1:y2+1, x1:x2+1] = 0

            infill_lines = _zigzag_infill(
                fill_mask, spacing_px, angle,
                connect_lines=cfg.infill.connect_lines,
            )
            infill_lines = [
                [(float(x), float(y)) for x, y in p] for p in infill_lines if len(p) >= 2
            ]
            if infill_lines:
                ordered_infill = order_polylines_greedy(infill_lines, start_xy=cursor)
                for poly in ordered_infill:
                    tl.moves.append(Move(x=poly[0][0], y=poly[0][1], extrude=False))
                    for pt in poly[1:]:
                        tl.moves.append(Move(x=pt[0], y=pt[1], extrude=True))

            # show_toolpath_debug(
            #     binary, contours, offset_contours, fill_mask, infill_lines,
            #     tool_idx, layer_idx,
            # )

            all_layers.append(tl)

    return all_layers
