"""Interactive GCode visualiser using matplotlib.

Features
--------
- Layer selector slider
- Toggle: show travel moves / print moves
- Toggle: animate path (step-by-step playback with speed control)
- Per-tool colour coding
- Works both from a raw .gcode file and from in-memory segments
"""

from __future__ import annotations

import itertools
from typing import Dict, List, Optional

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.widgets import Slider, CheckButtons, Button
import numpy as np

# Colour palette per tool (extend as needed)
TOOL_COLORS = [
    "#e63946",  # T0 – red
    "#2a9d8f",  # T1 – teal
    "#f4a261",  # T2 – orange
    "#457b9d",  # T3 – blue
    "#8ecae6",  # T4 – light blue
    "#a8dadc",  # T5 –
]

TRAVEL_COLOR = "#aaaaaa"


def _build_layer_data(segments: List[Dict]) -> Dict[int, List[Dict]]:
    """Group segments by layer index."""
    layers: Dict[int, List[Dict]] = {}
    for seg in segments:
        lyr = seg["layer"]
        layers.setdefault(lyr, []).append(seg)
    return layers


def _build_render_groups(
    segments: List[Dict],
) -> Dict[int, Dict[tuple, np.ndarray]]:
    """Pre-group segments by (layer, tool, type) into (N, 2, 2) arrays.

    LineCollection accepts an array of shape (N, 2, 2) — N line segments,
    each with two endpoints in (x, y). Building these once at startup means
    each redraw just hands a few prepared arrays to matplotlib instead of
    iterating per-segment.
    """
    buckets: Dict[int, Dict[tuple, list]] = {}
    for s in segments:
        key = (s["tool"], s["type"])
        layer_buckets = buckets.setdefault(s["layer"], {})
        layer_buckets.setdefault(key, []).append(
            ((s["x0"], s["y0"]), (s["x1"], s["y1"]))
        )

    groups: Dict[int, Dict[tuple, np.ndarray]] = {}
    for lyr, by_key in buckets.items():
        groups[lyr] = {
            key: np.asarray(seg_list, dtype=np.float32)
            for key, seg_list in by_key.items()
        }
    return groups


def _compute_extents(segments: List[Dict]) -> tuple:
    """Tight (xmin, xmax, ymin, ymax) over all segments. Vectorised."""
    if not segments:
        return (0.0, 1.0, 0.0, 1.0)
    pts = np.empty((len(segments), 4), dtype=np.float32)
    for i, s in enumerate(segments):
        pts[i, 0] = s["x0"]; pts[i, 1] = s["x1"]
        pts[i, 2] = s["y0"]; pts[i, 3] = s["y1"]
    return (
        float(pts[:, :2].min()), float(pts[:, :2].max()),
        float(pts[:, 2:].min()), float(pts[:, 2:].max()),
    )


def launch_gui(
    segments: List[Dict],
    meta: Optional[Dict] = None,
    title: str = "img2gcode visualiser",
) -> None:
    """Launch the interactive matplotlib GUI.

    Parameters
    ----------
    segments : move segment list from parse_gcode()
    meta     : metadata dict from parse_gcode() with print_width_mm, line_width_mm, etc.
    title    : window title string
    """
    if meta is None:
        meta = {}
    line_width_mm = meta.get("line_width_mm", 0.4)
    print_width_mm = meta.get("print_width_mm") or 120.0

    layer_data = _build_layer_data(segments)
    if not layer_data:
        print("[gui] No segments to display.")
        return

    n_layers = max(layer_data.keys()) + 1
    n_tools = max(s["tool"] for s in segments) + 1

    # Pre-group every segment into LineCollection-ready arrays once. All
    # subsequent draws read from these instead of iterating segments.
    render_groups = _build_render_groups(segments)
    xmin, xmax, ymin, ymax = _compute_extents(segments)

    fig = plt.figure(figsize=(12, 8))
    fig.suptitle(title, fontsize=13, fontweight="bold")

    # Layout: main axes + controls strip on the right
    ax = fig.add_axes([0.05, 0.15, 0.70, 0.78])
    ax.set_aspect("equal")
    ax.set_facecolor("#1a1a2e")
    ax.set_title("Layer 0", fontsize=10)
    # Pin axis limits so they don't have to be re-derived from the artists
    # on every redraw — autoscale on a 15k-segment LineCollection is itself
    # a noticeable cost.
    pad = 0.02 * max(xmax - xmin, ymax - ymin, 1.0)
    ax.set_xlim(xmin - pad, xmax + pad)
    ax.set_ylim(ymin - pad, ymax + pad)

    # ------------------------------------------------------------------ #
    # State                                                                #
    # ------------------------------------------------------------------ #
    state = {
        "layer": 0,
        "show_travel": False,
        "show_all_layers": False,
    }

    # ------------------------------------------------------------------ #
    # Draw function                                                         #
    # ------------------------------------------------------------------ #
    def _compute_print_lw() -> float:
        """Compute matplotlib linewidth (pts) that represents one physical line.

        We map the axis data range → figure inches → points, so the rendered
        line width tracks the actual nozzle width relative to the print area.
        """
        try:
            xlim = ax.get_xlim()
            data_range = xlim[1] - xlim[0]
            if data_range <= 0:
                return 1.0
            ax_width_inches = ax.get_position().width * fig.get_figwidth()
            mm_per_data_unit = print_width_mm / data_range
            pts_per_mm = (ax_width_inches * 72) / print_width_mm
            return max(0.4, line_width_mm * pts_per_mm)
        except Exception:
            return 1.0

    def draw(layer_idx: int):
        # Strip prior collections without ax.cla() — cla() rebuilds the entire
        # axes (spines, title, ticks, limits) which is the slow part.
        for coll in list(ax.collections):
            coll.remove()

        layers_to_draw = (
            range(layer_idx + 1) if state["show_all_layers"] else [layer_idx]
        )

        # Compute the print line-width once per draw (it depends on axis
        # state, not per-segment).
        print_lw = _compute_print_lw()

        for lyr in layers_to_draw:
            by_key = render_groups.get(lyr)
            if not by_key:
                continue
            alpha = 1.0 if lyr == layer_idx else 0.3
            for (tool, mtype), segs_arr in by_key.items():
                if mtype == "travel" and not state["show_travel"]:
                    continue
                color = (
                    TRAVEL_COLOR if mtype == "travel"
                    else TOOL_COLORS[tool % len(TOOL_COLORS)]
                )
                lw = 0.5 if mtype == "travel" else print_lw
                lc = LineCollection(
                    segs_arr, colors=color, linewidths=lw,
                    alpha=alpha, capstyle="round",
                )
                ax.add_collection(lc)

        ax.set_title(f"Layer {layer_idx} / {n_layers - 1}", fontsize=10, color="white")
        ax.tick_params(colors="white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#444")

        # Build the legend only once per draw (cheap, but skip rebuilding on
        # every check-button toggle if nothing visible changed).
        handles = [
            Line2D([0], [0], color=TOOL_COLORS[t % len(TOOL_COLORS)],
                   linewidth=2, label=f"T{t}")
            for t in range(n_tools)
        ]
        if state["show_travel"]:
            handles.append(
                Line2D([0], [0], color=TRAVEL_COLOR, linewidth=1, linestyle="--",
                       label="Travel")
            )
        # Remove prior legend (if any) before adding the new one.
        leg = ax.get_legend()
        if leg is not None:
            leg.remove()
        ax.legend(handles=handles, loc="upper right", facecolor="#222", labelcolor="white",
                  fontsize=8, framealpha=0.8)

        fig.canvas.draw_idle()

    # ------------------------------------------------------------------ #
    # Widgets                                                              #
    # ------------------------------------------------------------------ #
    # Layer slider
    ax_slider = fig.add_axes([0.10, 0.04, 0.55, 0.03])
    slider = Slider(
        ax_slider, "Layer", 0, max(0, n_layers - 1),
        valinit=0, valstep=1, color="#2a9d8f"
    )
    ax_slider.set_facecolor("#222")
    slider.label.set_color("white")
    slider.valtext.set_color("white")

    def on_slider(val):
        state["layer"] = int(slider.val)
        draw(state["layer"])

    slider.on_changed(on_slider)

    # Checkboxes (right panel)
    ax_check = fig.add_axes([0.78, 0.60, 0.20, 0.15])
    ax_check.set_facecolor("#222")
    check = CheckButtons(
        ax_check,
        ["Show travel", "Show all layers"],
        [state["show_travel"], state["show_all_layers"]],
    )
    for txt in check.labels:
        txt.set_color("white")
    check.ax.set_facecolor("#222")

    def on_check(label):
        if label == "Show travel":
            state["show_travel"] = not state["show_travel"]
        elif label == "Show all layers":
            state["show_all_layers"] = not state["show_all_layers"]
        draw(state["layer"])

    check.on_clicked(on_check)

    # Prev / Next buttons
    ax_prev = fig.add_axes([0.68, 0.04, 0.07, 0.04])
    ax_next = fig.add_axes([0.76, 0.04, 0.07, 0.04])
    btn_prev = Button(ax_prev, "◀ Prev", color="#333", hovercolor="#555")
    btn_next = Button(ax_next, "Next ▶", color="#333", hovercolor="#555")
    btn_prev.label.set_color("white")
    btn_next.label.set_color("white")

    def on_prev(_):
        new_val = max(0, int(slider.val) - 1)
        slider.set_val(new_val)

    def on_next(_):
        new_val = min(n_layers - 1, int(slider.val) + 1)
        slider.set_val(new_val)

    btn_prev.on_clicked(on_prev)
    btn_next.on_clicked(on_next)

    # Layer info panel
    ax_info = fig.add_axes([0.78, 0.20, 0.20, 0.35])
    ax_info.set_facecolor("#222")
    ax_info.axis("off")

    # Pull whole-print estimates from the gcode header (set by the writer).
    total_time_min = meta.get("total_time_min")
    print_time_min = meta.get("print_time_min")
    travel_time_min = meta.get("travel_time_min")
    filament_mm = meta.get("filament_mm")
    filament_g = meta.get("filament_g")

    # Cache per-layer stats so slider scrubbing across already-visited
    # layers doesn't re-scan the segment list.
    stats_cache: Dict[int, tuple] = {}

    def _layer_stats(layer_idx: int) -> tuple:
        cached = stats_cache.get(layer_idx)
        if cached is not None:
            return cached
        by_key = render_groups.get(layer_idx, {})
        n_print = 0
        n_travel = 0
        total_len = 0.0
        for (tool, mtype), arr in by_key.items():
            if mtype == "print":
                n_print += arr.shape[0]
                # arr is (N, 2, 2): segments × (start/end) × (x/y)
                d = arr[:, 1, :] - arr[:, 0, :]
                total_len += float(np.hypot(d[:, 0], d[:, 1]).sum())
            else:
                n_travel += arr.shape[0]
        stats_cache[layer_idx] = (n_print, n_travel, total_len)
        return stats_cache[layer_idx]

    def update_info(layer_idx):
        ax_info.cla()
        ax_info.set_facecolor("#222")
        ax_info.axis("off")
        n_print, n_travel, total_len = _layer_stats(layer_idx)
        info_lines = [
            f"Layer:   {layer_idx}",
            f"Print:   {n_print}",
            f"Travel:  {n_travel}",
            f"Length:  {total_len:.1f} mm",
        ]
        # Whole-print estimate block (when present in the gcode header).
        if total_time_min is not None:
            info_lines.append("")
            info_lines.append(f"Total:   {total_time_min:.1f} min")
            if print_time_min is not None and travel_time_min is not None:
                info_lines.append(f" print:  {print_time_min:.1f} min")
                info_lines.append(f" travel: {travel_time_min:.1f} min")
        if filament_mm is not None:
            grams = f" ({filament_g:.1f}g)" if filament_g is not None else ""
            info_lines.append(f"Filament:{filament_mm:.0f} mm{grams}")

        # Allocate vertical space proportional to the number of rows so longer
        # lists don't overflow the panel.
        n = max(1, len(info_lines))
        step = min(0.13, 0.85 / n)
        for i, txt in enumerate(info_lines):
            ax_info.text(
                0.05, 0.92 - i * step, txt,
                transform=ax_info.transAxes,
                color="white", fontsize=8, fontfamily="monospace",
            )
        ax_info.set_title("Stats", color="white", fontsize=9, pad=4)
        fig.canvas.draw_idle()

    original_on_slider = on_slider

    def on_slider_with_info(val):
        original_on_slider(val)
        update_info(state["layer"])

    slider.on_changed(on_slider_with_info)

    fig.patch.set_facecolor("#111")

    draw(0)
    update_info(0)
    plt.show()
