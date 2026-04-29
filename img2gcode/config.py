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
class EdgeConfig:
    """Edge-detection mode parameters."""
    # CLAHE preprocessing — boosts local contrast so faint features (hair
    # strands, faint eyebrows) become detectable. Set clahe_clip <= 0 to
    # disable. clip ≈ 2.0 is a good portrait default; raise to 3-4 for
    # very flat / low-contrast images.
    clahe_clip: float = 2.0
    clahe_grid: int = 8
    # Bilateral pre-filter (after CLAHE) — set d=0 to disable. Keep this
    # gentle; the multi-scale Canny already handles noise.
    bilateral_d: int = 5
    bilateral_sigma_color: float = 30.0
    bilateral_sigma_space: float = 30.0
    # Canny pinned thresholds. Both 0 → auto via Otsu (recommended;
    # the median-based heuristic breaks on white-background portraits).
    canny_low: int = 0
    canny_high: int = 0
    canny_sigma: float = 0.33  # legacy median heuristic, only used by _auto_canny
    # Multi-scale Canny: comma-separated Gaussian blur sigmas. 0 = no blur
    # (catches finest texture), larger values catch structural edges.
    # "0,1.0,2.0" works well for portraits.
    detection_scales: str = "0,1.0,2.0"
    # Foreground masking. When True, edges are AND-ed with the segmenter
    # fg_mask dilated by foreground_dilate_px so silhouette edges survive.
    # Disable for full-frame photos with no clean background.
    restrict_to_foreground: bool = False
    foreground_dilate_px: int = 3
    # Parallel-edge collapse: edges within (line_width * factor) merge into
    # one centerline. 1.0 = collapse anything closer than line width.
    collapse_factor: float = 1.0
    # Drawn-mask guard: candidate strokes that would land within
    # (line_width * factor) of an emitted stroke get split or dropped.
    min_separation_factor: float = 1.0
    # Polyline chaining: after the guard splits polylines, any two whose
    # endpoints sit within (line_width * chain_gap_factor) get concatenated
    # into a single longer polyline. Counteracts the fragmentation that the
    # junction-removing tracer introduces. 0 disables chaining.
    chain_gap_factor: float = 1.5
    # Drop polylines whose total geometric length (px) falls below this.
    # Applied both pre-simplification and after the drawn-mask split so
    # short noise is suppressed without dropping long straight segments
    # that simplify to 2 vertices.
    min_polyline_length: float = 5.0
    # Douglas-Peucker tolerance in pixels (0 = no simplification)
    simplify_epsilon: float = 1.0
    # Which tool index carries the edge strokes (multi-tool inputs only)
    tool_idx: int = 0
    # If non-empty, save the intermediate edge map (pre-trace) here as PNG.
    # Useful for tuning detection params without running the full pipeline.
    debug_edges_path: str = ""


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
    # Top-level generation mode: "fill" (segment + perimeter + infill) or
    # "edge" (Canny + skeleton + traced polylines).
    mode: str = "fill"
    machine: MachineConfig = field(default_factory=MachineConfig)
    tools: ToolsConfig = field(default_factory=ToolsConfig)
    image: ImageConfig = field(default_factory=ImageConfig)
    layers: LayersConfig = field(default_factory=LayersConfig)
    infill: InfillConfig = field(default_factory=InfillConfig)
    extrusion: ExtrusionConfig = field(default_factory=ExtrusionConfig)
    edge: EdgeConfig = field(default_factory=EdgeConfig)

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

        edge = EdgeConfig()
        if cfg.has_section("edge"):
            edge = EdgeConfig(
                clahe_clip=cfg.getfloat("edge", "clahe_clip", fallback=edge.clahe_clip),
                clahe_grid=cfg.getint("edge", "clahe_grid", fallback=edge.clahe_grid),
                bilateral_d=cfg.getint("edge", "bilateral_d", fallback=edge.bilateral_d),
                bilateral_sigma_color=cfg.getfloat("edge", "bilateral_sigma_color", fallback=edge.bilateral_sigma_color),
                bilateral_sigma_space=cfg.getfloat("edge", "bilateral_sigma_space", fallback=edge.bilateral_sigma_space),
                canny_low=cfg.getint("edge", "canny_low", fallback=edge.canny_low),
                canny_high=cfg.getint("edge", "canny_high", fallback=edge.canny_high),
                canny_sigma=cfg.getfloat("edge", "canny_sigma", fallback=edge.canny_sigma),
                detection_scales=cfg.get("edge", "detection_scales", fallback=edge.detection_scales).strip(),
                restrict_to_foreground=cfg.getboolean("edge", "restrict_to_foreground", fallback=edge.restrict_to_foreground),
                foreground_dilate_px=cfg.getint("edge", "foreground_dilate_px", fallback=edge.foreground_dilate_px),
                collapse_factor=cfg.getfloat("edge", "collapse_factor", fallback=edge.collapse_factor),
                min_separation_factor=cfg.getfloat("edge", "min_separation_factor", fallback=edge.min_separation_factor),
                chain_gap_factor=cfg.getfloat("edge", "chain_gap_factor", fallback=edge.chain_gap_factor),
                min_polyline_length=cfg.getfloat("edge", "min_polyline_length", fallback=edge.min_polyline_length),
                simplify_epsilon=cfg.getfloat("edge", "simplify_epsilon", fallback=edge.simplify_epsilon),
                tool_idx=cfg.getint("edge", "tool_idx", fallback=edge.tool_idx),
                debug_edges_path=cfg.get("edge", "debug_edges_path", fallback=edge.debug_edges_path).strip(),
            )

        mode = "fill"
        if cfg.has_section("mode"):
            mode = cfg.get("mode", "type", fallback="fill").strip().lower()

        return cls(
            mode=mode,
            machine=machine,
            tools=tools,
            image=image,
            layers=layers,
            infill=infill,
            extrusion=extrusion,
            edge=edge,
        )
