"""Configuration: one dataclass per .cfg section.

The dataclass defaults are the single source of truth; configs/default.cfg
documents the same values. Unknown keys are ignored, so old config files
keep loading.
"""

from __future__ import annotations

import configparser
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Optional, Union, get_type_hints

DEFAULT_CFG = Path(__file__).parent.parent / "configs" / "default.cfg"


@dataclass
class MachineConfig:
    bed_size_x: float = 1200.0
    bed_size_y: float = 1200.0
    origin_x: float = 10.0
    origin_y: float = 10.0
    print_width: float = 1000.0
    print_height: float = 700.0
    travel_speed: float = 7800.0
    print_speed: float = 1800.0
    z_lift: float = 1.0


@dataclass
class ToolsConfig:
    num_tools: int = 2


@dataclass
class ImageConfig:
    white_threshold: int = 220


@dataclass
class LayersConfig:
    num_layers: int = 1
    layer_height: float = 3.0
    perimeter_loops: int = 1


@dataclass
class InfillConfig:
    pattern: str = "zigzag"
    density: float = 100.0
    line_spacing: float = 0.0
    angle: float = -45.0
    connect_lines: bool = True


@dataclass
class ExtrusionConfig:
    filament_diameter: float = 1.75
    nozzle_diameter: float = 4.0
    line_width_mm: float = 0.0
    extrusion_multiplier: float = 1.0
    layer_height: float = 3.0

    @property
    def effective_line_width(self) -> float:
        return self.line_width_mm if self.line_width_mm > 0 else self.nozzle_diameter


@dataclass
class FillConfig:
    min_feature_factor: float = 1.5
    smoothing: float = 0.5


@dataclass
class EdgeConfig:
    detail: float = 0.5
    contrast: float = 1.5
    min_length_factor: float = 2.0
    debug_edges_path: str = ""


@dataclass
class ContourConfig:
    levels: int = 8
    smoothing: float = 1.5


@dataclass
class OnelineConfig:
    points: int = 0
    gamma: float = 1.0
    tsp_seconds: float = 15.0


@dataclass
class SquiggleConfig:
    row_spacing: float = 3.0
    max_frequency: float = 1.0


@dataclass
class HatchConfig:
    bands: int = 4
    angles: str = "45,-45,0"
    spacing: float = 2.5


@dataclass
class Config:
    style: str = "fill"
    machine: MachineConfig = field(default_factory=MachineConfig)
    tools: ToolsConfig = field(default_factory=ToolsConfig)
    image: ImageConfig = field(default_factory=ImageConfig)
    layers: LayersConfig = field(default_factory=LayersConfig)
    infill: InfillConfig = field(default_factory=InfillConfig)
    extrusion: ExtrusionConfig = field(default_factory=ExtrusionConfig)
    fill: FillConfig = field(default_factory=FillConfig)
    edge: EdgeConfig = field(default_factory=EdgeConfig)
    contour: ContourConfig = field(default_factory=ContourConfig)
    oneline: OnelineConfig = field(default_factory=OnelineConfig)
    squiggle: SquiggleConfig = field(default_factory=SquiggleConfig)
    hatch: HatchConfig = field(default_factory=HatchConfig)

    @classmethod
    def load(cls, path: Optional[Union[str, Path]] = None) -> "Config":
        """Defaults, then configs/default.cfg, then *path* (if given)."""
        parser = configparser.ConfigParser()
        parser.read(DEFAULT_CFG, encoding="utf-8")
        if path is not None:
            path = Path(path)
            if not path.exists():
                raise FileNotFoundError(f"Config file not found: {path}")
            parser.read(path, encoding="utf-8")

        cfg = cls()
        for f in fields(cls):
            if f.name != "style":
                _read_section(parser, f.name, getattr(cfg, f.name))
        if parser.has_option("mode", "type"):
            cfg.style = parser.get("mode", "type").strip().lower()
        return cfg


def _read_section(parser: configparser.ConfigParser, name: str, obj) -> None:
    if not parser.has_section(name):
        return
    hints = get_type_hints(type(obj))
    for f in fields(obj):
        if not parser.has_option(name, f.name):
            continue
        kind = hints[f.name]
        if kind is bool:
            value = parser.getboolean(name, f.name)
        elif kind is int:
            value = int(parser.getfloat(name, f.name))
        elif kind is float:
            value = parser.getfloat(name, f.name)
        else:
            value = parser.get(name, f.name).strip()
        setattr(obj, f.name, value)
