"""GCode writer.

Converts ToolLayer move lists into valid GCode with:
  - Startup sequence (homing, temperature, units)
  - Layer change annotations
  - Tool change commands
  - Extrusion calculated from move length
  - Travel lifts
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

import numpy as np

from .config import Config
from .toolpath import Move, ToolLayer


# Default density for filament weight estimates. PLA = 1.24 g/cm^3 is the
# common case; if you swap to PETG / ABS / TPU the gram figure is just a
# linear rescale of this constant.
_PLA_DENSITY_G_CM3 = 1.24


@dataclass
class PrintStats:
    """Aggregated estimates for a generated GCode file."""
    print_distance_mm: float
    travel_distance_mm: float
    filament_mm: float
    print_time_min: float
    travel_time_min: float

    @property
    def total_time_min(self) -> float:
        return self.print_time_min + self.travel_time_min

    def filament_grams(
        self, filament_diameter_mm: float, density_g_cm3: float = _PLA_DENSITY_G_CM3,
    ) -> float:
        """Convert filament length (mm) to mass using the given density."""
        radius = filament_diameter_mm / 2.0
        cross_mm2 = math.pi * radius * radius
        # mm^3 → cm^3 via /1000, then × density (g/cm^3)
        return cross_mm2 * self.filament_mm * density_g_cm3 / 1000.0


# ------------------------------------------------------------------ #
# Extrusion maths                                                       #
# ------------------------------------------------------------------ #

def _extrusion_length(
    dist_mm: float,
    cfg: Config,
) -> float:
    """Compute filament extrusion length for a given move distance (mm)."""
    line_w = cfg.extrusion.effective_line_width
    layer_h = cfg.extrusion.layer_height
    fil_r = cfg.extrusion.filament_diameter / 2
    # Volume = line_width * layer_height * distance (rectangular cross section)
    volume = line_w * layer_h * dist_mm
    # Filament length = volume / (pi * r^2)
    fil_length = volume / (math.pi * fil_r**2)
    return fil_length * cfg.extrusion.extrusion_multiplier


# ------------------------------------------------------------------ #
# Writer                                                                #
# ------------------------------------------------------------------ #

class GCodeWriter:
    def __init__(self, cfg: Config, image_shape: tuple):
        self.cfg = cfg
        self.H, self.W = image_shape[:2]
        # Scale: pixel → mm
        scale_x = cfg.machine.print_width / self.W
        scale_y = cfg.machine.print_height / self.H
        self.scale = min(scale_x, scale_y)

    def _px_to_mm(self, x_px: float, y_px: float):
        # Centre the print at (0, 0): image spans ±(print_dim/2) in each axis
        half_w = self.W * self.scale / 2.0
        half_h = self.H * self.scale / 2.0
        x_mm = x_px * self.scale - half_w
        y_mm = (self.H - 1 - y_px) * self.scale - half_h
        return round(x_mm, 4), round(y_mm, 4)

    def compute_stats(self, layers: List[ToolLayer]) -> PrintStats:
        """Walk every move and tally distances, time, and filament use.

        The writer also computes these inline as it serialises GCode, but
        having them broken out in a single pass keeps the header generation
        clean and lets callers (pipeline / GUI) read the numbers without
        parsing the output file.
        """
        cfg = self.cfg
        print_dist = 0.0
        travel_dist = 0.0
        filament_total = 0.0

        for tl in layers:
            if not tl.moves:
                continue
            prev = None
            for move in tl.moves:
                x_mm, y_mm = self._px_to_mm(move.x, move.y)
                if prev is None:
                    prev = (x_mm, y_mm)
                    continue
                d = math.hypot(x_mm - prev[0], y_mm - prev[1])
                if move.extrude:
                    print_dist += d
                    filament_total += _extrusion_length(d, cfg)
                else:
                    travel_dist += d
                prev = (x_mm, y_mm)

        print_speed = max(1.0, cfg.machine.print_speed)
        travel_speed = max(1.0, cfg.machine.travel_speed)
        return PrintStats(
            print_distance_mm=print_dist,
            travel_distance_mm=travel_dist,
            filament_mm=filament_total,
            print_time_min=print_dist / print_speed,
            travel_time_min=travel_dist / travel_speed,
        )

    def write(self, layers: List[ToolLayer], output_path: str | Path) -> None:
        cfg = self.cfg

        # Stats first so we can put them in the header.
        stats = self.compute_stats(layers)
        grams = stats.filament_grams(cfg.extrusion.filament_diameter)

        lines: List[str] = []

        ts = datetime.now(timezone.utc).strftime("%Y-%m-%d at %H:%M:%S UTC")
        lines.append(f"; generated by img2gcode on {ts}")
        lines.append(f"; image size: {self.W}x{self.H}px → "
                     f"{self.W * self.scale:.1f}x{self.H * self.scale:.1f}mm "
                     f"(scale={self.scale:.4f})")
        lines.append(f"; tools: {cfg.tools.num_tools}  layers: {cfg.layers.num_layers}")
        # Resolve effective spacing for the header comment
        lw = cfg.extrusion.effective_line_width
        if cfg.infill.line_spacing > 0:
            eff_spacing = cfg.infill.line_spacing
        else:
            eff_spacing = lw / (max(1.0, cfg.infill.density) / 100.0)
        lines.append(f"; infill: {cfg.infill.pattern}  density: {cfg.infill.density:.0f}%  "
                     f"spacing: {eff_spacing:.3f}mm  angle: {cfg.infill.angle}°")
        lines.append(f"; line_width:{lw:.4f}  perimeters: {cfg.layers.perimeter_loops}")
        lines.append(";")
        # Estimated cost lines — keep keys parser-stable so the GUI can read them.
        lines.append(
            f"; estimated_print_time: {stats.print_time_min:.1f} min  "
            f"travel_time: {stats.travel_time_min:.1f} min  "
            f"total_time: {stats.total_time_min:.1f} min"
        )
        lines.append(
            f"; print_distance: {stats.print_distance_mm:.1f} mm  "
            f"travel_distance: {stats.travel_distance_mm:.1f} mm"
        )
        lines.append(
            f"; filament: {stats.filament_mm:.1f} mm  "
            f"({grams:.2f} g @ PLA {_PLA_DENSITY_G_CM3} g/cm³)"
        )
        lines.append(";")
        lines.append("")

        # Startup
        lines.append("M104 S200 ; set temperature")
        lines.append(";TYPE:Custom")
        lines.append("G28 ; home all axes")
        lines.append(f"G1 Z5 F{int(cfg.machine.travel_speed)} ; lift nozzle")
        lines.append("M109 S200 ; set temperature and wait")
        lines.append("G21 ; set units to millimeters")
        lines.append("G90 ; use absolute coordinates")
        lines.append("M82 ; use absolute distances for extrusion")
        lines.append("G92 E0")
        lines.append("M107")
        lines.append("")

        current_tool = -1
        E = 0.0  # running extrusion counter

        # Group by layer
        layer_indices = sorted(set(tl.layer_idx for tl in layers))
        for layer_idx in layer_indices:
            z = cfg.layers.layer_height * (layer_idx + 1)
            lines.append(";LAYER_CHANGE")
            lines.append(f";Z:{z:.3f}")
            lines.append(f";HEIGHT:{cfg.layers.layer_height:.3f}")
            lines.append("G92 E0")
            E = 0.0

            layer_tls = [tl for tl in layers if tl.layer_idx == layer_idx]
            for tl in layer_tls:
                if not tl.moves:
                    continue

                # Tool change
                if tl.tool_idx != current_tool:
                    lines.append(f"")
                    lines.append(f"; --- Tool {tl.tool_idx} ---")
                    lines.append(f"T{tl.tool_idx}")
                    lines.append("G92 E0")
                    E = 0.0
                    current_tool = tl.tool_idx

                lines.append(f";TYPE:External perimeter" if cfg.layers.perimeter_loops > 0 else ";TYPE:Infill")

                prev_x_mm: Optional[float] = None
                prev_y_mm: Optional[float] = None
                z_at_layer = z
                is_lifted = False

                for move in tl.moves:
                    x_mm, y_mm = self._px_to_mm(move.x, move.y)

                    if not move.extrude:
                        # Travel move — lift Z if not already lifted
                        if not is_lifted:
                            lines.append(
                                f"G1 Z{z_at_layer + cfg.machine.z_lift:.3f} "
                                f"F{int(cfg.machine.travel_speed)}"
                            )
                            is_lifted = True
                        lines.append(
                            f"G1 X{x_mm} Y{y_mm} F{int(cfg.machine.travel_speed)}"
                        )
                        prev_x_mm, prev_y_mm = x_mm, y_mm
                    else:
                        # Print move
                        if is_lifted:
                            lines.append(f"G1 Z{z_at_layer:.3f} F{int(cfg.machine.travel_speed)}")
                            lines.append(f"G1 F{int(cfg.machine.print_speed)}")
                            is_lifted = False

                        if prev_x_mm is not None:
                            dist_mm = math.hypot(
                                x_mm - prev_x_mm, y_mm - prev_y_mm
                            )
                            E += _extrusion_length(dist_mm, cfg)
                            lines.append(
                                f"G1 X{x_mm} Y{y_mm} E{E:.5f}"
                            )
                        else:
                            # First print move — just position, no extrusion
                            lines.append(
                                f"G1 X{x_mm} Y{y_mm} F{int(cfg.machine.print_speed)}"
                            )
                        prev_x_mm, prev_y_mm = x_mm, y_mm

                lines.append("G92 E0")
                E = 0.0

        # End sequence
        lines.append("")
        lines.append("; End of print")
        lines.append("G1 Z10 F3000 ; raise Z")
        lines.append("G28 X0 Y0 ; home X Y")
        lines.append("M104 S0 ; turn off temperature")
        lines.append("M84 ; disable motors")

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("\n".join(lines))
        print(f"[img2gcode] GCode written → {output_path}  ({len(lines)} lines)")
        print(
            f"[img2gcode] Estimated: {stats.total_time_min:.1f} min  "
            f"({stats.print_time_min:.1f} print + {stats.travel_time_min:.1f} travel)  "
            f"|  filament {stats.filament_mm:.0f} mm ({grams:.2f} g PLA)  "
            f"|  travel {stats.travel_distance_mm:.0f} mm / "
            f"print {stats.print_distance_mm:.0f} mm"
        )
