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

    center = (W / 2, H / 2)
    rot_mat = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    inv_rot = cv2.getRotationMatrix2D(center, -angle_deg, 1.0)
    rotated_mask = cv2.warpAffine(
        binary_mask.astype(np.uint8), rot_mat, (W, H),
        flags=cv2.INTER_NEAREST
    )

    # --- Pass 1: collect every segment in scan order ----------------------
    # Direction only flips when a row actually has segments so that consecutive
    # filled rows always alternate sides — empty rows must not consume a flip.
    raw_segs: List[List[Tuple[float, float]]] = []
    y = line_spacing_px / 2.0
    direction = 1
    while y < H:
        row_y = int(round(y))
        if 0 <= row_y < H:
            row = rotated_mask[row_y, :]
            starts, ends = [], []
            in_seg = False
            for x in range(W):
                if row[x] and not in_seg:
                    starts.append(x)
                    in_seg = True
                elif not row[x] and in_seg:
                    ends.append(x - 1)
                    in_seg = False
            if in_seg:
                ends.append(W - 1)

            if starts:
                row_segs: List[List[Tuple[float, float]]] = []
                for s, e in zip(starts, ends):
                    pts_rot = (
                        [(float(s), float(row_y)), (float(e), float(row_y))]
                        if direction == 1
                        else [(float(e), float(row_y)), (float(s), float(row_y))]
                    )
                    pts_orig: List[Tuple[float, float]] = []
                    for px, py in pts_rot:
                        r = (inv_rot @ np.array([[px, py, 1.0]]).T).flatten()
                        pts_orig.append((r[0], r[1]))
                    if pts_orig:
                        row_segs.append(pts_orig)

                # For right-to-left rows, reverse segment order so the nearest
                # endpoint (right side) is encountered first during chaining.
                if direction == -1:
                    row_segs.reverse()

                raw_segs.extend(row_segs)
                direction = -direction  # only flip on rows that had segments

        y += line_spacing_px

    if not raw_segs:
        return []

    # Fast path: caller wants evenly-spaced parallel hatch — skip chaining.
    if not connect_lines:
        return raw_segs

    # --- Pass 2: greedy nearest-neighbour chaining -----------------------
    # For each segment, pick the *closest* unvisited segment (either endpoint
    # may match).  This walks one connected "column" of parallel infill lines
    # completely before jumping to another column — far more robust than
    # scanning in strict row order, which constantly switches columns when a
    # row has multiple segments and fragments the serpentine.
    threshold = line_spacing_px * max_connect_factor
    polylines: List[List[Tuple[float, float]]] = []
    unvisited = list(range(len(raw_segs)))

    # Precompute endpoint arrays for fast distance queries
    starts_arr = np.array([seg[0] for seg in raw_segs], dtype=float)
    ends_arr = np.array([seg[-1] for seg in raw_segs], dtype=float)

    first = unvisited.pop(0)
    chain = list(raw_segs[first])

    while unvisited:
        ex, ey = chain[-1]
        remaining = np.array(unvisited)
        d_start = np.hypot(starts_arr[remaining, 0] - ex, starts_arr[remaining, 1] - ey)
        d_end = np.hypot(ends_arr[remaining, 0] - ex, ends_arr[remaining, 1] - ey)

        best_start_pos = int(np.argmin(d_start))
        best_end_pos = int(np.argmin(d_end))

        if d_start[best_start_pos] <= d_end[best_end_pos]:
            best_dist = float(d_start[best_start_pos])
            best_seg_idx = int(remaining[best_start_pos])
            reverse_it = False
        else:
            best_dist = float(d_end[best_end_pos])
            best_seg_idx = int(remaining[best_end_pos])
            reverse_it = True

        if best_dist > threshold:
            # Nothing close — finalise current polyline, start new one
            polylines.append(chain)
            next_idx = unvisited.pop(0)
            chain = list(raw_segs[next_idx])
            continue

        unvisited.remove(best_seg_idx)
        seg = raw_segs[best_seg_idx]
        chain.extend(reversed(seg) if reverse_it else seg)

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

            # --- Perimeter loops -------------------------------------------
            contours = _extract_contours(binary)
            offset_contours: List[List[np.ndarray]] = []
            for loop in range(perimeter_loops):
                offset = loop * line_width_px
                for c, is_hole in contours:
                    # Holes offset outward (expanding into the hole) not inward
                    effective_offset = -offset if is_hole else offset
                    if loop == 0:
                        loop_contour = c
                        # Reverse hole winding so the nozzle traces it correctly
                        if is_hole:
                            loop_contour = c[::-1]
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
                    # Travel to start
                    tl.moves.append(Move(x=loop_contour[0][0], y=loop_contour[0][1], extrude=False))
                    # Draw loop
                    for pt in loop_contour[1:]:
                        tl.moves.append(Move(x=pt[0], y=pt[1], extrude=True))
                    # Close loop
                    tl.moves.append(Move(x=loop_contour[0][0], y=loop_contour[0][1], extrude=True))

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
            for poly in infill_lines:
                if len(poly) < 2:
                    continue
                # Travel to start of line
                tl.moves.append(Move(x=poly[0][0], y=poly[0][1], extrude=False))
                for pt in poly[1:]:
                    tl.moves.append(Move(x=pt[0], y=pt[1], extrude=True))

            # show_toolpath_debug(
            #     binary, contours, offset_contours, fill_mask, infill_lines,
            #     tool_idx, layer_idx,
            # )

            all_layers.append(tl)

    return all_layers
