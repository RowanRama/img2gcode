"""Interactive infill exclusion zone selector.

Shows the source image and the artwork's region outlines on the page and lets
the user draw rectangles that will be excluded from infill. Perimeters are
unaffected; only the fill pattern is skipped inside the rectangles.

Usage
-----
    from img2gcode.roi_selector import select_exclusion_zones

    zones = select_exclusion_zones(art)   # [(x0, y0, x1, y1), ...] in page mm
    tp = plan(art, cfg, exclusion_zones=zones)
"""

from __future__ import annotations

from typing import List, Tuple

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, RectangleSelector

from .geometry import geometry_rings
from .model import Artwork

Zone = Tuple[float, float, float, float]


def select_exclusion_zones(art: Artwork) -> List[Zone]:
    """Open a window for drawing infill exclusion rectangles.

    Drag to draw a rectangle; 'Undo' removes the last one; 'Done' closes the
    window. Returns (x0, y0, x1, y1) rectangles in page millimetres (origin
    top-left, y down), clipped to the page.
    """
    w, h = art.width_mm, art.height_mm
    zones: List[Zone] = []
    overlay_patches: List[mpatches.Rectangle] = []

    fig, ax = plt.subplots(figsize=(11, 8))
    fig.subplots_adjust(left=0.06, right=0.98, bottom=0.10, top=0.90)
    fig.patch.set_facecolor("#111")
    fig.suptitle(
        "Draw rectangles to exclude from infill  ·  drag to draw  ·  Undo / Done",
        color="white", fontsize=11,
    )

    ax.set_facecolor("white")
    if art.underlay is not None:
        ax.imshow(art.underlay, extent=(0, w, h, 0), alpha=0.5)
    for region in art.regions:
        colour = "#%02x%02x%02x" % tuple(int(c) for c in art.palette[region.tool])
        for ring in geometry_rings(region.shape):
            ax.plot(ring[:, 0], ring[:, 1], color=colour, linewidth=1.5)
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)  # y down, like the page
    ax.set_aspect("equal")
    ax.set_title("0 exclusion zone(s)", color="white", fontsize=9)
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#444")

    def _refresh_title():
        ax.set_title(f"{len(zones)} exclusion zone(s)", color="white", fontsize=9)
        fig.canvas.draw_idle()

    def _on_select(eclick, erelease):
        x0 = max(0.0, float(min(eclick.xdata, erelease.xdata)))
        y0 = max(0.0, float(min(eclick.ydata, erelease.ydata)))
        x1 = min(w, float(max(eclick.xdata, erelease.xdata)))
        y1 = min(h, float(max(eclick.ydata, erelease.ydata)))
        if x1 <= x0 or y1 <= y0:
            return

        zones.append((x0, y0, x1, y1))
        patch = mpatches.Rectangle(
            (x0, y0), x1 - x0, y1 - y0,
            linewidth=1.5, edgecolor="#ff4444", facecolor="#ff000033",
        )
        overlay_patches.append(patch)
        ax.add_patch(patch)
        _refresh_title()

    selector = RectangleSelector(  # noqa: F841 - a widget without a reference is garbage-collected
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
            overlay_patches.pop().remove()
            _refresh_title()

    btn_undo.on_clicked(_on_undo)

    # --- Done button ----------------------------------------------------------
    ax_done = fig.add_axes([0.83, 0.02, 0.10, 0.05])
    ax_done.set_facecolor("#2a9d8f")
    btn_done = Button(ax_done, "Done", color="#2a9d8f", hovercolor="#1f7a6e")
    btn_done.label.set_color("white")
    btn_done.on_clicked(lambda _: plt.close(fig))

    plt.show()
    return zones
