"""Interactive infill exclusion zone selector.

Displays the segmented label map and lets the user draw rectangular regions
that will be masked out from infill generation.  Perimeter contours are
unaffected — only the fill pattern is excluded inside the drawn rectangles.

Usage
-----
    from img2gcode.roi_selector import select_exclusion_zones

    zones = select_exclusion_zones(label_map, cluster_colors)
    # zones is a list of (x1, y1, x2, y2) tuples in pixel coordinates

    # Pass into build_toolpaths:
    layers = build_toolpaths(label_map, cfg, exclusion_zones=zones)
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.widgets import RectangleSelector, Button


# Colour palette matching the GUI (index = tool index)
_TOOL_COLORS = [
    "#e63946", "#2a9d8f", "#f4a261", "#457b9d", "#8ecae6", "#a8dadc",
]


def _build_display_image(
    label_map: np.ndarray,
    cluster_colors: Optional[np.ndarray],
) -> np.ndarray:
    """Convert label map to an RGB display image."""
    H, W = label_map.shape
    img = np.full((H, W, 3), 240, dtype=np.uint8)  # light-grey background

    n_tools = int(label_map.max()) + 1 if label_map.max() >= 0 else 0
    for t in range(n_tools):
        mask = label_map == t
        if cluster_colors is not None and t < len(cluster_colors):
            color = cluster_colors[t]
        else:
            # Fall back to the GUI palette
            hex_c = _TOOL_COLORS[t % len(_TOOL_COLORS)].lstrip("#")
            color = np.array([int(hex_c[i:i+2], 16) for i in (0, 2, 4)], dtype=np.uint8)
        img[mask] = color

    return img


def select_exclusion_zones(
    label_map: np.ndarray,
    cluster_colors: Optional[np.ndarray] = None,
) -> List[Tuple[int, int, int, int]]:
    """Show an interactive window for selecting infill exclusion rectangles.

    The user draws rectangles by clicking and dragging.  Each completed
    rectangle is added to the exclusion list and shown as a red overlay.
    The 'Undo' button removes the last rectangle; 'Done' closes the window.

    Parameters
    ----------
    label_map      : (H, W) int array from segmentation (-1 = background)
    cluster_colors : (n, 3) uint8 array of representative tool colours

    Returns
    -------
    List of (x1, y1, x2, y2) rectangles in pixel coordinates (origin top-left,
    y increases downward — same as label_map row/col convention).
    """
    H, W = label_map.shape
    display_img = _build_display_image(label_map, cluster_colors)

    zones: List[Tuple[int, int, int, int]] = []
    overlay_patches: List[mpatches.Rectangle] = []

    fig, ax = plt.subplots(figsize=(11, 8))
    fig.patch.set_facecolor("#111")
    fig.suptitle(
        "Draw rectangles to exclude from infill  ·  drag to draw  ·  Undo / Done",
        color="white", fontsize=11,
    )

    ax.imshow(display_img, origin="upper")
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)  # origin upper
    ax.set_facecolor("#1a1a2e")
    ax.set_title(f"0 exclusion zone(s)", color="white", fontsize=9)
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#444")

    def _refresh_title():
        ax.set_title(f"{len(zones)} exclusion zone(s)", color="white", fontsize=9)
        fig.canvas.draw_idle()

    def _on_select(eclick, erelease):
        # eclick/erelease give data coordinates (column, row in image space)
        x1 = max(0, int(min(eclick.xdata, erelease.xdata)))
        y1 = max(0, int(min(eclick.ydata, erelease.ydata)))
        x2 = min(W - 1, int(max(eclick.xdata, erelease.xdata)))
        y2 = min(H - 1, int(max(eclick.ydata, erelease.ydata)))

        if x2 <= x1 or y2 <= y1:
            return

        zones.append((x1, y1, x2, y2))

        patch = mpatches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=1.5, edgecolor="#ff4444", facecolor="#ff000033",
        )
        overlay_patches.append(patch)
        ax.add_patch(patch)
        _refresh_title()

    rs = RectangleSelector(
        ax, _on_select,
        useblit=True,
        button=[1],
        minspanx=2, minspany=2,
        spancoords="pixels",
        interactive=False,
        props=dict(edgecolor="#ff4444", facecolor="#ff000022", linewidth=1.5),
    )

    # --- Undo button ----------------------------------------------------------
    ax_undo = fig.add_axes([0.72, 0.02, 0.10, 0.05])
    ax_undo.set_facecolor("#333")
    btn_undo = Button(ax_undo, "Undo", color="#333", hovercolor="#555")
    btn_undo.label.set_color("white")

    def _on_undo(_):
        if zones:
            zones.pop()
            patch = overlay_patches.pop()
            patch.remove()
            _refresh_title()

    btn_undo.on_clicked(_on_undo)

    # --- Done button ----------------------------------------------------------
    ax_done = fig.add_axes([0.83, 0.02, 0.10, 0.05])
    ax_done.set_facecolor("#2a9d8f")
    btn_done = Button(ax_done, "Done", color="#2a9d8f", hovercolor="#1f7a6e")
    btn_done.label.set_color("white")

    def _on_done(_):
        plt.close(fig)

    btn_done.on_clicked(_on_done)

    plt.tight_layout(rect=[0, 0.08, 1, 0.95])
    plt.show()

    return zones
