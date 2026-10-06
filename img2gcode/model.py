"""Data passed between pipeline stages.

All coordinates are page millimetres: origin top-left, x right, y down
(the SVG convention). The G-code writer converts to machine coordinates.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

RGB = Tuple[int, int, int]


@dataclass(eq=False)
class Stroke:
    """One continuous extrusion path. Closed when first point == last point."""
    points: np.ndarray            # (N, 2) float64
    tool: int = 0
    kind: str = "line"            # "perimeter" | "infill" | "line"

    @property
    def closed(self) -> bool:
        p = self.points
        return len(p) > 2 and bool(np.allclose(p[0], p[-1]))


@dataclass
class Region:
    """A filled area printed by one tool (shapely Polygon or MultiPolygon)."""
    shape: object
    tool: int


@dataclass(eq=False)
class Artwork:
    """What a style produces: vector art on the page, not yet planned."""
    width_mm: float
    height_mm: float
    regions: List[Region] = field(default_factory=list)
    strokes: List[Stroke] = field(default_factory=list)
    palette: List[RGB] = field(default_factory=list)
    source_px: Tuple[int, int] = (0, 0)
    underlay: Optional[np.ndarray] = None
    style: str = ""

    @property
    def n_tools(self) -> int:
        return len(self.palette)


@dataclass(eq=False)
class ToolPass:
    """Everything one tool prints on one layer, in print order."""
    tool: int
    layer: int
    strokes: List[Stroke] = field(default_factory=list)


@dataclass(eq=False)
class Toolpaths:
    """Planned output: one ToolPass per (layer, tool), layer-major."""
    width_mm: float
    height_mm: float
    line_width_mm: float
    n_layers: int
    passes: List[ToolPass] = field(default_factory=list)
    palette: List[RGB] = field(default_factory=list)
    source_px: Tuple[int, int] = (0, 0)
    underlay: Optional[np.ndarray] = None
    style: str = ""

    @property
    def n_tools(self) -> int:
        return len(self.palette)
