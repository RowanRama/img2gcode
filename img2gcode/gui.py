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

    fig = plt.figure(figsize=(12, 8))
    fig.suptitle(title, fontsize=13, fontweight="bold")

    # Layout: main axes + controls strip on the right
    ax = fig.add_axes([0.05, 0.15, 0.70, 0.78])
    ax.set_aspect("equal")
    ax.set_facecolor("#1a1a2e")
    ax.set_title("Layer 0", fontsize=10)

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
        ax.cla()
        ax.set_facecolor("#1a1a2e")
        ax.set_aspect("equal")

        layers_to_draw = (
            range(layer_idx + 1) if state["show_all_layers"] else [layer_idx]
        )

        # Draw first pass to establish axis limits, then recompute lw
        for lyr in layers_to_draw:
            segs = layer_data.get(lyr, [])
            alpha = 1.0 if lyr == layer_idx else 0.3
            for seg in segs:
                if seg["type"] == "travel" and not state["show_travel"]:
                    continue
                color = (
                    TRAVEL_COLOR
                    if seg["type"] == "travel"
                    else TOOL_COLORS[seg["tool"] % len(TOOL_COLORS)]
                )
                # Travel lines are always thin; print lines use physical width
                if seg["type"] == "travel":
                    lw = 0.5
                else:
                    lw = _compute_print_lw()
                ax.plot(
                    [seg["x0"], seg["x1"]],
                    [seg["y0"], seg["y1"]],
                    color=color,
                    linewidth=lw,
                    alpha=alpha,
                    solid_capstyle="round",
                )

        ax.set_title(f"Layer {layer_idx} / {n_layers - 1}", fontsize=10, color="white")
        ax.tick_params(colors="white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#444")

        # Legend
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

    def update_info(layer_idx):
        ax_info.cla()
        ax_info.set_facecolor("#222")
        ax_info.axis("off")
        segs = layer_data.get(layer_idx, [])
        n_print = sum(1 for s in segs if s["type"] == "print")
        n_travel = sum(1 for s in segs if s["type"] == "travel")
        total_len = sum(
            np.hypot(s["x1"] - s["x0"], s["y1"] - s["y0"])
            for s in segs if s["type"] == "print"
        )
        info_lines = [
            f"Layer:   {layer_idx}",
            f"Print:   {n_print}",
            f"Travel:  {n_travel}",
            f"Length:  {total_len:.1f} mm",
        ]
        for i, txt in enumerate(info_lines):
            ax_info.text(
                0.05, 0.85 - i * 0.18, txt,
                transform=ax_info.transAxes,
                color="white", fontsize=9, fontfamily="monospace",
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
