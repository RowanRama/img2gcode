"""Configuration loader – reads .cfg files in PrusaSlicer-style INI format."""

from __future__ import annotations

import configparser
from dataclasses import dataclass, field
from pathlib import Path
from typing import List


DEFAULT_CFG = Path(__file__).parent.parent / "configs" / "default.cfg"


@dataclass
class MachineConfig:
    bed_size_x: float = 200.0
    bed_size_y: float = 200.0
    origin_x: float = 10.0
    origin_y: float = 10.0
    print_width: float = 120.0
    print_height: float = 120.0
    travel_speed: float = 7800.0
    print_speed: float = 1800.0
    z_lift: float = 1.0


@dataclass
class ToolsConfig:
    num_tools: int = 2
    tool_change_gcode: List[str] = field(default_factory=lambda: ["T{tool}\nG92 E0"])


@dataclass
class ImageConfig:
    white_threshold: int = 220
    min_cluster_area: int = 50


@dataclass
class LayersConfig:
    num_layers: int = 1
    layer_height: float = 0.2
    perimeter_loops: int = 1
    horizontal_shell_layers: int = 0


@dataclass
class InfillConfig:
    pattern: str = "zigzag"
    # density is the primary knob (0–100 %).  line_spacing is derived from it
    # at build time using: spacing = nozzle_diameter / (density / 100)
    # If line_spacing is set explicitly in the config it takes precedence.
    density: float = 40.0       # infill density %
    line_spacing: float = 0.0   # 0 = derive from density + nozzle diameter
    angle: float = 45.0
    # True → join adjacent scan lines into a single serpentine polyline
    # (PrusaSlicer-style).  False → emit each scan line separately so the
    # rendered infill shows evenly-spaced parallel lines.
    connect_lines: bool = True


@dataclass
class ExtrusionConfig:
    filament_diameter: float = 1.75
    nozzle_diameter: float = 0.4
    # line_width_mm: physical width of a printed line.
    # 0 = use nozzle_diameter as line width (typical default).
    line_width_mm: float = 0.0
    extrusion_multiplier: float = 1.0
    layer_height: float = 0.2

    @property
    def effective_line_width(self) -> float:
        """Return the actual line width to use (mm)."""
        return self.line_width_mm if self.line_width_mm > 0 else self.nozzle_diameter


@dataclass
class Config:
    machine: MachineConfig = field(default_factory=MachineConfig)
    tools: ToolsConfig = field(default_factory=ToolsConfig)
    image: ImageConfig = field(default_factory=ImageConfig)
    layers: LayersConfig = field(default_factory=LayersConfig)
    infill: InfillConfig = field(default_factory=InfillConfig)
    extrusion: ExtrusionConfig = field(default_factory=ExtrusionConfig)

    # ------------------------------------------------------------------ #
    # Factory                                                              #
    # ------------------------------------------------------------------ #
    @classmethod
    def load(cls, path: str | Path | None = None) -> "Config":
        """Load config from *path*, falling back to defaults for missing keys."""
        cfg = configparser.ConfigParser()
        # Always load defaults first
        cfg.read(DEFAULT_CFG)
        if path is not None:
            cfg.read(Path(path))

        def _f(section: str, key: str) -> float:
            return cfg.getfloat(section, key)

        def _i(section: str, key: str) -> int:
            return cfg.getint(section, key)

        def _s(section: str, key: str) -> str:
            return cfg.get(section, key).strip()

        machine = MachineConfig(
            bed_size_x=_f("machine", "bed_size_x"),
            bed_size_y=_f("machine", "bed_size_y"),
            origin_x=_f("machine", "origin_x"),
            origin_y=_f("machine", "origin_y"),
            print_width=_f("machine", "print_width"),
            print_height=_f("machine", "print_height"),
            travel_speed=_f("machine", "travel_speed"),
            print_speed=_f("machine", "print_speed"),
            z_lift=_f("machine", "z_lift"),
        )
        tools = ToolsConfig(
            num_tools=_i("tools", "num_tools"),
        )
        image = ImageConfig(
            white_threshold=_i("image", "white_threshold"),
            min_cluster_area=_i("image", "min_cluster_area"),
        )
        layers = LayersConfig(
            num_layers=_i("layers", "num_layers"),
            layer_height=_f("layers", "layer_height"),
            perimeter_loops=_i("layers", "perimeter_loops"),
            horizontal_shell_layers=_i("layers", "horizontal_shell_layers"),
        )
        infill = InfillConfig(
            pattern=_s("infill", "pattern"),
            density=_f("infill", "density"),
            line_spacing=_f("infill", "line_spacing"),
            angle=_f("infill", "angle"),
            connect_lines=cfg.getboolean("infill", "connect_lines", fallback=True),
        )
        extrusion = ExtrusionConfig(
            filament_diameter=_f("extrusion", "filament_diameter"),
            nozzle_diameter=_f("extrusion", "nozzle_diameter"),
            line_width_mm=_f("extrusion", "line_width_mm"),
            extrusion_multiplier=_f("extrusion", "extrusion_multiplier"),
            layer_height=_f("extrusion", "layer_height"),
        )
        return cls(
            machine=machine,
            tools=tools,
            image=image,
            layers=layers,
            infill=infill,
            extrusion=extrusion,
        )
