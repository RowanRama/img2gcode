# img2gcode SVG Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rebuild img2gcode so every conversion goes image → artwork → planned toolpaths → `toolpaths.svg` → G-code (format unchanged), with noise-robust line art and four new abstract styles.

**Architecture:** Styles turn a raster into an `Artwork` (filled shapely regions + polyline strokes in page millimetres). A planner turns artwork into `Toolpaths` (perimeters, thin-feature centrelines, infill, travel order, layers). Toolpaths are written to an SVG that is read back and serialized to G-code by a port of today's writer, pinned by a golden file.

**Tech Stack:** Python ≥ 3.9, numpy, OpenCV (headless), scikit-image, SciPy, shapely 2, svgelements, Pillow, matplotlib (viewer only), pytest.

**Spec:** `docs/superpowers/specs/2026-10-06-svg-pipeline-design.md` (read it first; this plan argues from it).

## Global Constraints

- Python ≥ 3.9 syntax: `from __future__ import annotations` at the top of every module; no `match`, no runtime `X | Y` unions (use `Optional`/`Union` where evaluated at runtime, e.g. dataclass fields read by `get_type_hints`).
- Dependencies (exact list in `requirements.txt` and `setup.py`): `numpy>=1.24`, `Pillow>=10.0`, `opencv-python-headless>=4.8`, `matplotlib>=3.7`, `svgelements>=1.9`, `shapely>=2.0`, `scikit-image>=0.22`, `scipy>=1.10`. Nothing else. `scikit-learn` is removed.
- `img2gcode/__init__.py` calls `cv2.setNumThreads(1)` before anything else (OpenCV's pool throws "Unknown C++ exception" on Windows once SciPy is loaded).
- Page coordinates: millimetres, origin top-left, x right, y down. Raster pixel `(col, row)` centre ↦ `((col + 0.5)·mm_per_px, (row + 0.5)·mm_per_px)`.
- `lw` = `cfg.extrusion.effective_line_width` (4.0 mm with the default config).
- All text files are written with `encoding="utf-8"`.
- Determinism: `cv2.setRNGSeed(0)` before k-means, `np.random.default_rng(0)` for sampling.
- Styles and the pipeline never import matplotlib; only `gui.py` and `roi_selector.py` do.
- Test command (Windows, from repo root): `.venv/Scripts/python.exe -m pytest -q`. On POSIX use `.venv/bin/python`. Slow runtime tests: add `-m slow`.
- File ownership: each task edits only the files it lists. Files from Tasks 1–5 (`config.py`, `configs/default.cfg`, `model.py`, `styles/__init__.py`, `tests/synth.py`, `tests/conftest.py`, `geometry.py`, `raster.py`, `writer.py`, `svg_io.py`) are frozen for later tasks: if one needs a change, add a private helper in your own module or report the needed change in your final summary instead of editing.
- Commits: one per task, short subject, no `Co-Authored-By` or any Claude attribution. If `git` reports `index.lock`, wait a few seconds and retry; stage only your own files (`git add <paths>`, never `git add -A`).
- Tests are the contract. If a threshold in a test turns out to be wrong on reflection (not merely inconvenient), change it and justify the change in your report.

## Review Focus

1. **Tiny and huge sources** (a 24-unit SVG icon, a 32×24 PNG, a 6000 px photo): sensible page size, no crash, working image capped at 2500 px. Pinned by Task 3 `test_working_image_caps_size` and Task 15 `test_tiny_image`.
2. **Blank or single-colour images** (all white, all one grey): every style returns empty artwork without raising, and the pipeline still writes valid G-code with no extrusion. Pinned by each style's `test_blank_image` and Task 15 `test_blank_image_gives_valid_empty_gcode`.
3. **Unusual raster modes** (palette PNG, 16-bit greyscale, alpha, EXIF-rotated JPEG): `load_rgb` returns upright `(H, W, 3) uint8`. Pinned by Task 3 `test_load_rgb_modes` and `test_load_rgb_exif_orientation`.
4. **Toolpaths SVG after an Inkscape edit** (transforms on groups, polylines turned into paths, hidden layers, stray shapes): read correctly, hidden layers skipped, strays counted. Pinned by Task 5 `test_inkscape_style_edits` and `test_nested_transforms`.
5. **More tools than colours** (`--n-tools 6` on a two-colour logo): no crash, unused tools produce empty passes that the writer skips. Pinned by Task 3 `test_palette_kmeans_more_clusters_than_colours` and Task 9 `test_more_tools_than_colours`.

## File Structure

```
img2gcode/
  __init__.py        cv2 thread pin, version                             (Task 1)
  model.py           Stroke, Region, Artwork, ToolPass, Toolpaths          (Task 1)
  config.py          dataclass sections + generic loader                  (Task 1)
  styles/__init__.py STYLES, ALIASES, resolve_style, get_style            (Task 1)
  geometry.py        polyline maths, rings→geometry, hatch, order, deoverlap (Task 2)
  raster.py          load/fit/resample/noise/denoise/palette/mask→geometry  (Task 3)
  writer.py          Toolpaths → G-code (format unchanged)                 (Task 4)
  svg_io.py          toolpaths.svg write/read                              (Task 5)
  skeleton.py        mask → smooth centreline strokes                      (Task 6)
  planner.py         Artwork → Toolpaths                                   (Task 7)
  svg_input.py       artwork SVG → Artwork                                 (Task 8)
  styles/fill.py · edge.py · contour.py · oneline.py · squiggle.py · hatch.py   (Tasks 9–14)
  pipeline.py        convert(), make_artwork()                             (Task 15)
  __main__.py        CLI                                                   (Task 15)
  gcode_parser.py    + parse_gcode_lines()                                 (Task 15)
  gui.py             duplicate-handler fix                                 (Task 15)
  roi_selector.py    page-mm version                                       (Task 15)
  (deleted)          toolpath.py, edge_pipeline.py, segmenter.py, svg_loader.py   (Task 15)
configs/default.cfg  rewritten                                              (Task 1)
scripts/style_gallery.py   contact sheets for visual QA                    (Task 16)
tests/conftest.py, tests/synth.py, pytest.ini                              (Task 1)
tests/test_*.py      one per module / style                                (each task)
README.md                                                                  (Task 16)
```

Dependency order: Task 1 → {Task 2 → Task 3, Task 4, Task 5} → {Task 6, Task 8, Tasks 9, 11, 12, 13, 14} → {Task 7, Task 10 (need Task 6)} → Task 15 → Task 16 → Task 17.

---

### Task 1: Foundation (env, model, config, style registry, test helpers)

**Files:**
- Modify: `requirements.txt`, `setup.py`, `img2gcode/__init__.py`, `configs/default.cfg` (full rewrite), `img2gcode/config.py` (full rewrite)
- Create: `img2gcode/model.py`, `img2gcode/styles/__init__.py`, `pytest.ini`, `tests/__init__.py` (empty), `tests/conftest.py`, `tests/synth.py`, `tests/test_config.py`

**Interfaces:**
- Produces: everything in `model.py`; `Config` with sections `machine tools image layers infill extrusion fill edge contour oneline squiggle hatch` and attribute `style`; `Config.load(path=None)`; `ExtrusionConfig.effective_line_width`; `STYLES`, `ALIASES`, `resolve_style(name) -> str`, `get_style(name) -> Callable[[np.ndarray, Config], Artwork]`; `tests.synth` helpers listed below; `cfg` pytest fixture.

- [ ] **Step 1: Create the project venv**

```bash
uv venv .venv --python 3.12
uv pip install --python .venv/Scripts/python.exe -e . pytest
```
(`.venv/` is already gitignored.) Run this after Step 3's dependency edits; re-run the install whenever `setup.py` changes.

- [ ] **Step 2: Write the failing tests** — `tests/test_config.py`

```python
import pytest

from img2gcode.config import Config


def test_default_cfg_file_matches_dataclass_defaults():
    assert Config.load() == Config()


def test_key_values():
    cfg = Config.load()
    assert cfg.style == "fill"
    assert (cfg.machine.print_width, cfg.machine.print_height) == (1000.0, 700.0)
    assert cfg.extrusion.effective_line_width == 4.0
    assert cfg.edge.detail == 0.5
    assert cfg.oneline.tsp_seconds == 15.0
    assert cfg.hatch.angles == "45,-45,0"


def test_user_file_overrides_and_ignores_unknown_keys(tmp_path):
    p = tmp_path / "u.cfg"
    p.write_text(
        "[mode]\ntype = Contour\n"
        "[edge]\nclahe_clip = 3\ndetail = 0.8\n"
        "[infill]\nconnect_lines = false\n"
        "[tools]\nnum_tools = 3\n",
        encoding="utf-8",
    )
    cfg = Config.load(p)
    assert cfg.style == "contour"
    assert cfg.edge.detail == 0.8
    assert cfg.infill.connect_lines is False
    assert cfg.tools.num_tools == 3


def test_missing_user_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        Config.load(tmp_path / "nope.cfg")


def test_line_width_override():
    cfg = Config()
    cfg.extrusion.line_width_mm = 4.5
    assert cfg.extrusion.effective_line_width == 4.5


def test_style_registry():
    from img2gcode.styles import ALIASES, STYLES, resolve_style

    assert STYLES == ("fill", "edge", "contour", "oneline", "squiggle", "hatch")
    assert ALIASES == {"lines": "edge"}
    assert resolve_style("Lines") == "edge"
    assert resolve_style("hatch") == "hatch"
    with pytest.raises(ValueError):
        resolve_style("nope")


def test_cv2_threads_pinned():
    import cv2

    import img2gcode  # noqa: F401

    assert cv2.getNumThreads() == 1
```

- [ ] **Step 3: Run to verify failure**

Run: `.venv/Scripts/python.exe -m pytest tests/test_config.py -q`
Expected: FAIL (import errors: `Config()` sections missing, `img2gcode.styles` missing).

- [ ] **Step 4: Implement**

`requirements.txt`:
```
numpy>=1.24
Pillow>=10.0
opencv-python-headless>=4.8
matplotlib>=3.7
svgelements>=1.9
shapely>=2.0
scikit-image>=0.22
scipy>=1.10
```
`setup.py`: same list in `install_requires`, `version="0.2.0"`, description "Convert images to SVG toolpaths and multi-tool GCode". Keep `find_packages()` and the console script.

`img2gcode/__init__.py`:
```python
"""img2gcode – convert images to SVG toolpaths and multi-tool GCode."""

import cv2

# OpenCV's thread pool breaks ("Unknown C++ exception") on Windows once SciPy
# has been loaded in the same process. The images here are small enough that
# a single thread costs little.
cv2.setNumThreads(1)

__version__ = "0.2.0"
```

`img2gcode/model.py`:
```python
"""Data passed between pipeline stages.

All coordinates are page millimetres: origin top-left, x right, y down
(the SVG convention). The G-code writer converts to machine coordinates.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

RGB = Tuple[int, int, int]


@dataclass
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


@dataclass
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


@dataclass
class ToolPass:
    """Everything one tool prints on one layer, in print order."""
    tool: int
    layer: int
    strokes: List[Stroke] = field(default_factory=list)


@dataclass
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
```

`img2gcode/config.py`:
```python
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
```

`configs/default.cfg` (values must equal the dataclass defaults; `test_default_cfg_file_matches_dataclass_defaults` enforces it):
```ini
# img2gcode configuration. Copy this file and pass it with -c to override.
# Every key is optional; missing keys keep the values shown here.

[mode]
# Drawing style: fill | edge | contour | oneline | squiggle | hatch
#   fill     – flat colour regions, perimeters + infill, one tool per colour
#   edge     – clean line art (noise-robust)
#   contour  – topographic lines of brightness
#   oneline  – the whole image as one unbroken line
#   squiggle – rows of waves, taller and tighter where the image is dark
#   hatch    – engraving-style hatching, more directions in darker areas
type = fill

[machine]
# Bed size (mm)
bed_size_x = 1200
bed_size_y = 1200
# Starting corner offset (mm)
origin_x = 10
origin_y = 10
# The image is scaled to fit inside print_width x print_height (mm), centred on X0 Y0
print_width = 1000
print_height = 700
# Speeds (mm/min)
travel_speed = 7800
print_speed = 1800
# Z lift on travel moves (mm)
z_lift = 1.0

[tools]
# Number of tools / colours
num_tools = 2

[image]
# Pixels with R, G and B all at or above this value are background (paper), 0-255
white_threshold = 220

[layers]
num_layers = 1
# Layer height (mm)
layer_height = 3
# Perimeter loops around every filled region
perimeter_loops = 1

[infill]
pattern = zigzag
# Infill density %. Line spacing = line width / (density / 100)
density = 100
# Explicit line spacing (mm). 0 = use density
line_spacing = 0
# Infill angle (degrees); odd layers add 90
angle = -45
# true: join neighbouring scan lines into a serpentine; false: separate lines
connect_lines = true

[extrusion]
filament_diameter = 1.75
nozzle_diameter = 4
# Printed line width (mm). 0 = nozzle_diameter
line_width_mm = 0
extrusion_multiplier = 1.0
# Layer height used for extrusion maths (should match [layers] layer_height)
layer_height = 3

[fill]
# Smallest kept feature, as a multiple of the line width. Smaller colour
# blobs and holes merge into the colour around them.
min_feature_factor = 1.5
# Edge smoothing (majority vote radius), in line widths
smoothing = 0.5

[edge]
# 0 = only bold structure, 1 = fine detail
detail = 0.5
# Local contrast boost (CLAHE clip limit). 0 disables
contrast = 1.5
# Drop strokes shorter than this many line widths
min_length_factor = 2
# Save the binary line map here before tracing (empty = off). Same as --debug-edges
debug_edges_path =

[contour]
# Number of brightness levels traced
levels = 8
# Blur before tracing, in line widths. Higher = smoother, more abstract
smoothing = 1.5

[oneline]
# Number of dots the line visits. 0 = automatic
points = 0
# Darkness curve: > 1 pushes the line into the darkest areas
gamma = 1.0
# Time limit for untangling the line (seconds)
tsp_seconds = 15

[squiggle]
# Distance between rows, in line widths
row_spacing = 3
# Wave frequency limit as a fraction of the printable maximum (0-1)
max_frequency = 1.0

[hatch]
# Number of tone bands; the lightest band is left empty
bands = 4
# Hatch directions in degrees, used from lightest hatched band to darkest
angles = 45,-45,0
# Line spacing, in line widths
spacing = 2.5
```

`img2gcode/styles/__init__.py`:
```python
"""Drawing styles. Each module exposes make(rgb, cfg) -> Artwork."""

from __future__ import annotations

import importlib
from typing import Callable

STYLES = ("fill", "edge", "contour", "oneline", "squiggle", "hatch")
ALIASES = {"lines": "edge"}


def resolve_style(name: str) -> str:
    key = name.strip().lower()
    key = ALIASES.get(key, key)
    if key not in STYLES:
        raise ValueError(f"Unknown style {name!r}; choose from {', '.join(STYLES)}")
    return key


def get_style(name: str) -> Callable:
    return importlib.import_module(f"{__name__}.{resolve_style(name)}").make
```

`pytest.ini`:
```ini
[pytest]
testpaths = tests
pythonpath = .
markers =
    slow: runtime checks on large images (run with -m slow)
addopts = -m "not slow"
```

`tests/conftest.py`:
```python
import os

os.environ.setdefault("MPLBACKEND", "Agg")

import pytest  # noqa: E402

from img2gcode.config import Config  # noqa: E402


@pytest.fixture
def cfg():
    """Fresh default config (identical to configs/default.cfg)."""
    return Config()
```

`tests/synth.py`:
```python
"""Synthetic images with known geometry, plus shared assertions."""

from __future__ import annotations

import cv2
import numpy as np
from scipy.spatial import cKDTree

SHAPES_W, SHAPES_H = 800, 600
CIRCLE = ((250, 300), 150)          # centre (x, y) px, radius px
RECT = (480, 150, 720, 450)         # x0, y0, x1, y1 px (inclusive)
LW = 4.0                            # default line width (mm)


def page_scale(cfg, w: int, h: int) -> float:
    return min(cfg.machine.print_width / w, cfg.machine.print_height / h)


def _noisy(img: np.ndarray, noise: float, seed: int) -> np.ndarray:
    if noise <= 0:
        return img
    rng = np.random.default_rng(seed)
    return np.clip(img + rng.normal(0, noise, img.shape), 0, 255).astype(np.uint8)


def shapes_rgb(noise: float = 0.0, seed: int = 0) -> np.ndarray:
    """Dark circle + red rectangle on light grey, 800x600."""
    img = np.full((SHAPES_H, SHAPES_W, 3), 235, np.uint8)
    cv2.circle(img, CIRCLE[0], CIRCLE[1], (40, 40, 40), -1)
    cv2.rectangle(img, RECT[:2], RECT[2:], (200, 60, 60), -1)
    return _noisy(img, noise, seed)


def shapes_outline_mm(cfg, step: float = 1.0) -> np.ndarray:
    """Points along the true outlines of shapes_rgb(), in page mm."""
    s = page_scale(cfg, SHAPES_W, SHAPES_H)
    (cx, cy), r = CIRCLE
    n = max(16, int(2 * np.pi * r * s / step))
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    circle = np.c_[(cx + 0.5 + r * np.cos(t)) * s, (cy + 0.5 + r * np.sin(t)) * s]
    x0, y0, x1, y1 = RECT
    corners = np.array([[x0, y0], [x1 + 1, y0], [x1 + 1, y1 + 1], [x0, y1 + 1], [x0, y0]], float) * s
    return np.vstack([circle, dense(corners, step)])


def radial_rgb(h: int = 500, w: int = 700, noise: float = 0.0, seed: int = 0) -> np.ndarray:
    """Dark centre fading to near-white at the corners."""
    yy, xx = np.mgrid[0:h, 0:w]
    r = np.hypot(xx - w / 2, yy - h / 2) / np.hypot(w / 2, h / 2)
    v = (30 + 205 * r).astype(np.uint8)
    return _noisy(np.dstack([v, v, v]), noise, seed)


def split_rgb(left=(0, 0, 0), right=(255, 255, 255), h: int = 400, w: int = 800) -> np.ndarray:
    img = np.empty((h, w, 3), np.uint8)
    img[:, : w // 2] = left
    img[:, w // 2:] = right
    return img


def bands_rgb(values=(20, 80, 140, 200), h: int = 400, w: int = 800) -> np.ndarray:
    """Vertical grey bands, darkest on the left."""
    img = np.empty((h, w, 3), np.uint8)
    bw = w // len(values)
    for i, v in enumerate(values):
        img[:, i * bw:(i + 1) * bw] = v
    return img


def dense(points: np.ndarray, step: float) -> np.ndarray:
    """Resample a polyline so consecutive points are about *step* apart."""
    p = np.asarray(points, float)
    if len(p) < 2:
        return p
    seg = np.hypot(*np.diff(p, axis=0).T)
    s = np.r_[0.0, np.cumsum(seg)]
    if s[-1] == 0:
        return p[:1]
    t = np.linspace(0, s[-1], max(2, int(np.ceil(s[-1] / step)) + 1))
    return np.c_[np.interp(t, s, p[:, 0]), np.interp(t, s, p[:, 1])]


def recall_precision(polylines, truth: np.ndarray, tol: float, step: float = 1.0):
    """(recall, precision) of drawn polylines against truth points, within tol mm."""
    if not polylines:
        return 0.0, 0.0
    pts = np.vstack([dense(p, step) for p in polylines])
    recall = float(np.mean(cKDTree(pts).query(truth)[0] <= tol))
    precision = float(np.mean(cKDTree(truth).query(pts)[0] <= tol))
    return recall, precision


def total_length(polylines) -> float:
    return float(sum(np.hypot(*np.diff(np.asarray(p, float), axis=0).T).sum() for p in polylines))


def assert_valid_artwork(art, cfg, allow_empty: bool = False) -> None:
    eps = 1e-6
    assert 0 < art.width_mm <= cfg.machine.print_width + eps
    assert 0 < art.height_mm <= cfg.machine.print_height + eps
    n = len(art.palette)
    assert n >= 1
    for c in art.palette:
        assert len(c) == 3 and all(0 <= int(v) <= 255 for v in c)
    for s in art.strokes:
        assert s.points.ndim == 2 and s.points.shape[1] == 2 and len(s.points) >= 2
        assert 0 <= s.tool < n
        assert s.points[:, 0].min() >= -eps and s.points[:, 1].min() >= -eps
        assert s.points[:, 0].max() <= art.width_mm + eps
        assert s.points[:, 1].max() <= art.height_mm + eps
    for r in art.regions:
        assert 0 <= r.tool < n
        assert not r.shape.is_empty and r.shape.is_valid
        minx, miny, maxx, maxy = r.shape.bounds
        assert minx >= -eps and miny >= -eps
        assert maxx <= art.width_mm + eps and maxy <= art.height_mm + eps
    if not allow_empty:
        assert art.regions or art.strokes
```

- [ ] **Step 5: Install and run tests**

Run: `uv pip install --python .venv/Scripts/python.exe -e . pytest` then `.venv/Scripts/python.exe -m pytest tests/test_config.py -q`
Expected: all pass. (Old modules such as `toolpath.py` still import the old config fields and are now broken; they are deleted in Task 15 and nothing imports them in tests.)

- [ ] **Step 6: Commit**

```bash
git add requirements.txt setup.py pytest.ini img2gcode/__init__.py img2gcode/model.py img2gcode/config.py img2gcode/styles/__init__.py configs/default.cfg tests/__init__.py tests/conftest.py tests/synth.py tests/test_config.py
git commit -m "Add data model, generic config loader and style registry"
```

---

### Task 2: geometry.py

**Files:**
- Create: `img2gcode/geometry.py`, `tests/test_geometry.py`

**Interfaces:**
- Consumes: `Stroke` (Task 1).
- Produces:
  - `polyline_length(points) -> float`
  - `resample(points, step) -> np.ndarray` (endpoints kept; closed stays closed)
  - `smooth(points, window) -> np.ndarray` (moving average over ≈`window` mm of arc length; open: endpoints fixed; closed: wraps and stays closed)
  - `simplify(points, tol) -> np.ndarray` (Douglas-Peucker; endpoints kept)
  - `chaikin(points, iterations=2) -> np.ndarray`
  - `rings_to_geometry(rings, rule="nonzero") -> BaseGeometry` (rule ∈ {"nonzero", "evenodd"}; rings may self-intersect or overlap)
  - `geometry_rings(geom) -> list[np.ndarray]` (every exterior and interior ring as a closed (N,2) array)
  - `hatch_lines(geom, spacing, angle_deg, connect=False, connect_tol=0.0) -> list[np.ndarray]`
  - `order_strokes(strokes, start=None) -> list[Stroke]`
  - `chain_polylines(polys, max_gap) -> list[np.ndarray]`
  - `deoverlap(polys, width_mm, height_mm, min_sep, min_length) -> list[np.ndarray]`

- [ ] **Step 1: Write the failing tests** — `tests/test_geometry.py`

```python
import numpy as np
import pytest
from shapely.geometry import LineString, Point, box

from img2gcode.geometry import (
    chain_polylines, chaikin, deoverlap, geometry_rings, hatch_lines, order_strokes,
    polyline_length, resample, rings_to_geometry, simplify, smooth,
)
from img2gcode.model import Stroke

SQ = np.array([[0, 0], [10, 0], [10, 10], [0, 10], [0, 0]], float)


def test_polyline_length():
    assert polyline_length(np.array([[0, 0], [3, 4], [3, 10]], float)) == pytest.approx(11.0)
    assert polyline_length(np.zeros((1, 2))) == 0.0


def test_resample_spacing_and_ends():
    p = np.array([[0, 0], [10, 0]], float)
    q = resample(p, 1.0)
    assert len(q) == 11
    assert np.allclose(q[0], p[0]) and np.allclose(q[-1], p[-1])
    r = resample(SQ, 2.5)
    assert np.allclose(r[0], r[-1])


def test_smooth_keeps_open_endpoints_and_closure():
    zig = np.array([[0, 0], [1, 1], [2, 0], [3, 1], [4, 0], [5, 1], [6, 0]], float)
    s = smooth(zig, 2.0)
    assert np.allclose(s[0], zig[0]) and np.allclose(s[-1], zig[-1])
    assert np.ptp(s[1:-1, 1]) < np.ptp(zig[:, 1])
    c = smooth(SQ, 3.0)
    assert np.allclose(c[0], c[-1])


def test_simplify_straight_line():
    p = np.c_[np.linspace(0, 10, 50), np.zeros(50)]
    assert len(simplify(p, 0.01)) == 2


def test_chaikin_open_keeps_ends():
    p = np.array([[0, 0], [10, 0], [10, 10]], float)
    q = chaikin(p, 2)
    assert np.allclose(q[0], p[0]) and np.allclose(q[-1], p[-1]) and len(q) > len(p)


def test_rings_nonzero_and_evenodd():
    outer = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], float)
    inner = np.array([[3, 3], [7, 3], [7, 7], [3, 7]], float)          # same direction
    assert rings_to_geometry([outer, inner], "nonzero").area == pytest.approx(100)
    assert rings_to_geometry([outer, inner], "evenodd").area == pytest.approx(84)
    assert rings_to_geometry([outer, inner[::-1]], "nonzero").area == pytest.approx(84)


def test_rings_overlapping_evenodd_is_xor():
    a = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], float)
    b = a + 5
    assert rings_to_geometry([a, b], "evenodd").area == pytest.approx(150)
    assert rings_to_geometry([a, b], "nonzero").area == pytest.approx(175)


def test_geometry_rings_counts_holes():
    g = box(0, 0, 10, 10).difference(box(3, 3, 7, 7))
    rings = geometry_rings(g)
    assert len(rings) == 2 and all(np.allclose(r[0], r[-1]) for r in rings)


def test_hatch_lines_square():
    lines = hatch_lines(box(0, 0, 40, 40), 4.0, 0.0)
    assert len(lines) == 10
    ys = sorted(round(float(l[0, 1]), 6) for l in lines)
    assert ys == [2.0 + 4 * i for i in range(10)]
    sq = box(0, 0, 40, 40).buffer(1e-6)
    assert all(sq.covers(LineString(l)) for l in lines)
    joined = hatch_lines(box(0, 0, 40, 40), 4.0, 0.0, connect=True, connect_tol=0.1)
    assert len(joined) == 1


def test_hatch_lines_global_phase_and_angle():
    a = hatch_lines(box(0, 0, 40, 40), 4.0, 0.0)
    b = hatch_lines(box(40, 0, 80, 40), 4.0, 0.0)
    assert sorted(round(float(l[0, 1]), 6) for l in a) == sorted(round(float(l[0, 1]), 6) for l in b)
    diag = hatch_lines(box(0, 0, 40, 40), 4.0, 45.0)
    for l in diag:
        d = l[-1] - l[0]
        assert abs(abs(np.degrees(np.arctan2(d[1], d[0]))) % 180 - 45) < 1e-6 or abs(abs(np.degrees(np.arctan2(d[1], d[0]))) % 180 - 135) < 1e-6


def test_order_strokes_reverses_and_rotates():
    a = Stroke(np.array([[0, 0], [10, 0]], float))
    b = Stroke(np.array([[30, 0], [11, 0]], float))            # tail is nearer to a's end
    sq = Stroke(SQ + [50, 0])
    out = order_strokes([sq, b, a], start=(0, 0))
    assert np.allclose(out[0].points[0], [0, 0])
    assert np.allclose(out[1].points[0], [11, 0])               # b reversed
    assert out[2].closed and np.allclose(out[2].points[0], [50, 0])
    assert len(out) == 3


def test_order_strokes_keeps_tool_and_kind():
    s = Stroke(np.array([[5, 5], [6, 6]], float), tool=1, kind="infill")
    out = order_strokes([s])
    assert out[0].tool == 1 and out[0].kind == "infill"


def test_chain_polylines():
    a = np.array([[0, 0], [10, 0]], float)
    b = np.array([[10.5, 0], [20, 0]], float)
    c = np.array([[50, 50], [60, 50]], float)
    out = chain_polylines([a, b, c], max_gap=1.0)
    assert len(out) == 2
    assert max(polyline_length(p) for p in out) == pytest.approx(20.5, abs=0.6)


def test_deoverlap():
    a = np.array([[10, 50], [90, 50]], float)
    near = np.array([[10, 51], [90, 51]], float)      # 1 mm away → dropped
    far = np.array([[10, 70], [90, 70]], float)       # 20 mm away → kept
    cross = np.array([[50, 20], [50, 90]], float)     # crosses a and far → split
    out = deoverlap([a, near, far, cross], 100, 100, min_sep=4.0, min_length=2.0)
    lens = sorted(polyline_length(p) for p in out)
    assert len(out) == 5                              # a, far, and three pieces of cross
    assert lens[-1] == pytest.approx(80, abs=1) and lens[-2] == pytest.approx(80, abs=1)
    assert all(LineString(p).distance(LineString(a)) >= 3.5 or polyline_length(p) > 70 for p in out)
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/Scripts/python.exe -m pytest tests/test_geometry.py -q` → FAIL (module missing).

- [ ] **Step 3: Implement `img2gcode/geometry.py`**

Reference implementation (adapt freely, keep the signatures):

```python
"""Polyline and polygon helpers in page millimetres."""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np
import shapely
from scipy.spatial import cKDTree
from shapely import affinity
from shapely.geometry import LineString, Polygon
from shapely.ops import polygonize, unary_union

from .model import Stroke


def _arr(points) -> np.ndarray:
    return np.asarray(points, dtype=np.float64).reshape(-1, 2)


def _is_closed(p: np.ndarray) -> bool:
    return len(p) > 2 and bool(np.allclose(p[0], p[-1]))


def polyline_length(points) -> float:
    p = _arr(points)
    if len(p) < 2:
        return 0.0
    return float(np.hypot(*np.diff(p, axis=0).T).sum())


def resample(points, step: float) -> np.ndarray:
    p = _arr(points)
    if len(p) < 2 or step <= 0:
        return p.copy()
    s = np.r_[0.0, np.cumsum(np.hypot(*np.diff(p, axis=0).T))]
    if s[-1] == 0:
        return p[[0, -1]].copy()
    t = np.linspace(0.0, s[-1], max(1, int(np.ceil(s[-1] / step))) + 1)
    return np.c_[np.interp(t, s, p[:, 0]), np.interp(t, s, p[:, 1])]


def smooth(points, window: float) -> np.ndarray:
    p = _arr(points)
    length = polyline_length(p)
    if window <= 0 or len(p) < 3 or length == 0:
        return p.copy()
    closed = _is_closed(p)
    step = min(window / 4.0, length / 8.0)
    q = resample(p, step)
    k = max(3, int(round(window / step)) | 1)
    kernel = np.ones(k) / k
    if closed:
        ring = q[:-1]
        pad = np.vstack([ring[-(k // 2):], ring, ring[: k // 2]])
        out = np.c_[np.convolve(pad[:, 0], kernel, "valid"), np.convolve(pad[:, 1], kernel, "valid")]
        return np.vstack([out, out[:1]])
    pad = np.vstack([np.repeat(q[:1], k // 2, 0), q, np.repeat(q[-1:], k // 2, 0)])
    out = np.c_[np.convolve(pad[:, 0], kernel, "valid"), np.convolve(pad[:, 1], kernel, "valid")]
    out[0], out[-1] = q[0], q[-1]
    return out


def simplify(points, tol: float) -> np.ndarray:
    p = _arr(points)
    if tol <= 0 or len(p) < 3:
        return p.copy()
    out = np.asarray(LineString(p).simplify(tol, preserve_topology=False).coords)
    return out if len(out) >= 2 else p[[0, -1]].copy()


def chaikin(points, iterations: int = 2) -> np.ndarray:
    p = _arr(points)
    for _ in range(iterations):
        if len(p) < 3:
            break
        if _is_closed(p):
            a = p[:-1]
            b = np.roll(a, -1, axis=0)
            new = np.empty((2 * len(a), 2))
            new[0::2], new[1::2] = 0.75 * a + 0.25 * b, 0.25 * a + 0.75 * b
            p = np.vstack([new, new[:1]])
        else:
            a, b = p[:-1], p[1:]
            new = np.empty((2 * len(a), 2))
            new[0::2], new[1::2] = 0.75 * a + 0.25 * b, 0.25 * a + 0.75 * b
            p = np.vstack([p[:1], new, p[-1:]])
    return p


def _winding(ring: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Winding number of a closed ring around each point (vectorised)."""
    r = np.vstack([ring, ring[:1]])
    x0, y0 = r[:-1, 0][:, None], r[:-1, 1][:, None]
    x1, y1 = r[1:, 0][:, None], r[1:, 1][:, None]
    px, py = pts[:, 0][None, :], pts[:, 1][None, :]
    cross = (x1 - x0) * (py - y0) - (px - x0) * (y1 - y0)
    up = (y0 <= py) & (y1 > py) & (cross > 0)
    down = (y0 > py) & (y1 <= py) & (cross < 0)
    return up.sum(axis=0) - down.sum(axis=0)


def rings_to_geometry(rings: Sequence[np.ndarray], rule: str = "nonzero"):
    """Fill closed rings with an SVG fill rule. Handles crossing and nested rings."""
    rings = [_arr(r) for r in rings]
    rings = [r[:-1] if _is_closed(r) else r for r in rings]
    rings = [r for r in rings if len(r) >= 3]
    if not rings:
        return Polygon()
    lines = unary_union([LineString(np.vstack([r, r[:1]])) for r in rings])
    faces = list(polygonize(lines))
    if not faces:
        return Polygon()
    pts = np.array([f.representative_point().coords[0] for f in faces])
    w = np.zeros(len(faces), dtype=np.int64)
    for r in rings:
        w += _winding(r, pts)
    keep = (w != 0) if rule == "nonzero" else (w % 2 != 0)
    kept = [f for f, k in zip(faces, keep) if k]
    return unary_union(kept) if kept else Polygon()


def geometry_rings(geom) -> List[np.ndarray]:
    out: List[np.ndarray] = []
    for poly in getattr(geom, "geoms", [geom]):
        if poly.is_empty or poly.geom_type != "Polygon":
            continue
        out.append(np.asarray(poly.exterior.coords))
        out.extend(np.asarray(i.coords) for i in poly.interiors)
    return out


def _iter_lines(g):
    if g.is_empty:
        return
    if g.geom_type == "LineString":
        yield g
    elif hasattr(g, "geoms"):
        for sub in g.geoms:
            yield from _iter_lines(sub)


def hatch_lines(geom, spacing: float, angle_deg: float, connect: bool = False,
                connect_tol: float = 0.0) -> List[np.ndarray]:
    """Parallel scan lines clipped to *geom*, in serpentine order.

    Rows sit at (k + 0.5) * spacing in the rotated frame, so neighbouring
    regions hatched at the same angle line up.
    """
    if geom is None or geom.is_empty or spacing <= 0:
        return []
    rot = affinity.rotate(geom, -angle_deg, origin=(0, 0))
    minx, miny, maxx, maxy = rot.bounds
    ks = np.arange(int(np.floor(miny / spacing)) - 1, int(np.ceil(maxy / spacing)) + 1)
    ys = (ks + 0.5) * spacing
    ys = ys[(ys > miny) & (ys < maxy)]
    if not len(ys):
        return []
    scan = shapely.linestrings([[(minx - 1, y), (maxx + 1, y)] for y in ys])
    rows = []
    for g in shapely.intersection(scan, rot):
        segs = []
        for ls in _iter_lines(g):
            c = np.asarray(ls.coords)
            c = c[[0, -1]]
            if c[0, 0] > c[1, 0]:
                c = c[::-1]
            if np.hypot(*(c[1] - c[0])) > 1e-9:
                segs.append(c)
        segs.sort(key=lambda c: c[0, 0])
        rows.append(segs)
    ordered: List[np.ndarray] = []
    flip = False
    for segs in rows:
        if not segs:
            continue
        if flip:
            segs = [c[::-1] for c in reversed(segs)]
        ordered.extend(segs)
        flip = not flip
    if connect and ordered:
        zone = rot.buffer(connect_tol)
        shapely.prepare(zone)
        merged = [ordered[0]]
        for c in ordered[1:]:
            a, b = merged[-1][-1], c[0]
            if np.hypot(*(b - a)) <= 2.5 * spacing and zone.covers(LineString([a, b])):
                merged[-1] = np.vstack([merged[-1], c])
            else:
                merged.append(c)
        ordered = merged
    th = np.radians(angle_deg)
    rmat = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    return [c @ rmat.T for c in ordered]


def order_strokes(strokes: Sequence[Stroke], start: Optional[Tuple[float, float]] = None) -> List[Stroke]:
    """Greedy nearest-neighbour order. Open strokes may reverse; closed strokes
    may start at any of up to 32 evenly spaced vertices."""
    if not strokes:
        return []
    pts, owner, vert = [], [], []
    for i, s in enumerate(strokes):
        p = s.points
        if s.closed:
            n = len(p) - 1
            idx = np.unique(np.linspace(0, n - 1, min(n, 32)).round().astype(int))
        else:
            idx = np.array([0, len(p) - 1])
        pts.append(p[idx]); owner.append(np.full(len(idx), i)); vert.append(idx)
    pts = np.vstack(pts); owner = np.concatenate(owner); vert = np.concatenate(vert)
    tree = cKDTree(pts)
    used = np.zeros(len(strokes), dtype=bool)
    cursor = np.asarray(start if start is not None else strokes[0].points[0], float)
    out: List[Stroke] = []
    for _ in range(len(strokes)):
        k = 8
        while True:
            kk = min(k, len(pts))
            _, j = tree.query(cursor, k=kk)
            j = np.atleast_1d(j)
            free = j[~used[owner[j]]]
            if len(free) or kk == len(pts):
                break
            k *= 4
        c = int(free[0])
        i = int(owner[c]); used[i] = True
        s = strokes[i]; p = s.points
        if s.closed:
            v = int(vert[c]); ring = p[:-1]
            p = np.vstack([np.roll(ring, -v, axis=0), ring[v:v + 1]])
        elif vert[c] != 0:
            p = p[::-1]
        out.append(Stroke(points=p, tool=s.tool, kind=s.kind))
        cursor = p[-1]
    return out


def chain_polylines(polys: Sequence[np.ndarray], max_gap: float) -> List[np.ndarray]:
    """Join polylines whose ends lie within max_gap, longest first."""
    pieces = [_arr(p) for p in polys if len(p) >= 2]
    if max_gap <= 0 or len(pieces) < 2:
        return pieces
    heads = np.array([p[0] for p in pieces]); tails = np.array([p[-1] for p in pieces])
    used = np.zeros(len(pieces), dtype=bool)
    out = []
    for start in sorted(range(len(pieces)), key=lambda i: -polyline_length(pieces[i])):
        if used[start]:
            continue
        used[start] = True
        chain = pieces[start]
        if _is_closed(chain):
            out.append(chain); continue
        for _ in range(2):                       # extend the tail, then the head
            while True:
                end = chain[-1]
                cand = np.flatnonzero(~used)
                if not len(cand):
                    break
                dh = np.hypot(*(heads[cand] - end).T); dt = np.hypot(*(tails[cand] - end).T)
                bh, bt = int(np.argmin(dh)), int(np.argmin(dt))
                if min(dh[bh], dt[bt]) > max_gap:
                    break
                if dh[bh] <= dt[bt]:
                    j = int(cand[bh]); seg = pieces[j]
                else:
                    j = int(cand[bt]); seg = pieces[j][::-1]
                if _is_closed(seg):
                    break
                used[j] = True
                chain = np.vstack([chain, seg[1:] if np.allclose(chain[-1], seg[0]) else seg])
            chain = chain[::-1]
        out.append(chain)
    return out


def deoverlap(polys: Sequence[np.ndarray], width_mm: float, height_mm: float,
              min_sep: float, min_length: float) -> List[np.ndarray]:
    """Keep polylines in the given priority order, cutting away any part that
    comes within min_sep of a polyline already kept."""
    px = min_sep / 4.0
    W = int(np.ceil(width_mm / px)) + 2; H = int(np.ceil(height_mm / px)) + 2
    mask = np.zeros((H, W), np.uint8)
    radius = max(1, int(round(min_sep / px)))
    out: List[np.ndarray] = []
    for p in polys:
        q = resample(p, px / 2.0)
        if len(q) < 2:
            continue
        ij = np.clip(np.floor(q / px).astype(int), 0, [W - 1, H - 1])
        free = mask[ij[:, 1], ij[:, 0]] == 0
        runs, start = [], None
        for i, f in enumerate(free):
            if f and start is None:
                start = i
            if (not f or i == len(free) - 1) and start is not None:
                end = i + 1 if f else i
                runs.append((start, end)); start = None
        closed = _is_closed(p)
        if closed and len(runs) > 1 and runs[0][0] == 0 and runs[-1][1] == len(q):
            first = runs.pop(0); last = runs.pop()
            runs.append(None)
            joined = np.vstack([q[last[0]:last[1]], q[first[0] + 1:first[1]]])
        kept = []
        for r in runs:
            seg = joined if r is None else q[r[0]:r[1]]
            if len(seg) >= 2 and polyline_length(seg) >= min_length:
                kept.append(simplify(seg, px / 4.0))
        for seg in kept:
            pts = np.round(seg / px * 16).astype(np.int32).reshape(-1, 1, 2)
            cv2.polylines(mask, [pts], False, 1, thickness=2 * radius + 1, shift=4)
        out.extend(kept)
    return out
```

- [ ] **Step 4: Run tests** — `.venv/Scripts/python.exe -m pytest tests/test_geometry.py -q` → PASS.

- [ ] **Step 5: Commit** — `git add img2gcode/geometry.py tests/test_geometry.py && git commit -m "Add polyline and polygon geometry helpers"`

---

### Task 3: raster.py

**Files:**
- Create: `img2gcode/raster.py`, `tests/test_raster.py`

**Interfaces:**
- Consumes: `Config` (Task 1), `geometry.simplify` (Task 2).
- Produces:
  - `load_rgb(path) -> np.ndarray` (H, W, 3) uint8; alpha composited onto white; EXIF orientation applied; palette/greyscale/16-bit handled
  - `fit_page(src_w, src_h, cfg) -> (width_mm, height_mm, mm_per_src_px)`
  - `working_image(rgb, mm_per_src_px, mm_per_px, max_side=2500) -> (img, actual_mm_per_px)` (INTER_AREA down, INTER_CUBIC up)
  - `estimate_noise(gray) -> float` (Immerkær, 0–255 units)
  - `denoise(img, sigma=None) -> img` (non-local means, h = 1.1·σ, only when σ > 1.5; gray or RGB)
  - `luminance(rgb) -> np.ndarray` float32 in [0, 1] (Lab L)
  - `foreground_mask(rgb, white_threshold) -> np.ndarray` bool
  - `palette_kmeans(rgb, mask, n, max_samples=200_000) -> (palette (k,3) uint8 sorted dark→light, labels (H,W) int32, -1 outside mask)`; `k = min(n, number of masked pixels)`
  - `mask_to_geometry(mask, mm_per_px, simplify_mm=0.0, min_area_mm2=0.0) -> BaseGeometry` (valid; empty Polygon when nothing)
  - `tone_levels(lum, mask, n) -> np.ndarray` (n−1 ascending thresholds at quantiles of `lum[mask]`)
  - `sample_colors(rgb, points_mm, mm_per_px) -> np.ndarray` (N, 3) uint8

- [ ] **Step 1: Write the failing tests** — `tests/test_raster.py`

```python
import numpy as np
import pytest
from PIL import Image

from img2gcode.raster import (
    denoise, estimate_noise, fit_page, foreground_mask, load_rgb, luminance,
    mask_to_geometry, palette_kmeans, sample_colors, tone_levels, working_image,
)


def test_load_rgb_alpha(tmp_path):
    rgba = np.zeros((10, 10, 4), np.uint8)
    rgba[:5] = [255, 0, 0, 255]
    Image.fromarray(rgba, "RGBA").save(tmp_path / "a.png")
    rgb = load_rgb(tmp_path / "a.png")
    assert rgb.shape == (10, 10, 3) and rgb.dtype == np.uint8
    assert tuple(rgb[0, 0]) == (255, 0, 0) and tuple(rgb[9, 9]) == (255, 255, 255)


def test_load_rgb_modes(tmp_path):
    base = np.zeros((10, 10, 3), np.uint8); base[:, 5:] = 200
    Image.fromarray(base).convert("P").save(tmp_path / "p.png")
    assert load_rgb(tmp_path / "p.png").shape == (10, 10, 3)
    g16 = np.linspace(0, 65535, 100).reshape(10, 10).astype(np.uint16)
    Image.fromarray(g16).save(tmp_path / "g16.png")
    g = load_rgb(tmp_path / "g16.png")
    assert g.shape == (10, 10, 3) and g.dtype == np.uint8
    assert g[0, 0, 0] < 20 and g[-1, -1, 0] > 235


def test_load_rgb_exif_orientation(tmp_path):
    img = Image.new("RGB", (40, 20), "white")
    exif = Image.Exif(); exif[0x0112] = 6
    img.save(tmp_path / "r.jpg", exif=exif)
    assert load_rgb(tmp_path / "r.jpg").shape[:2] == (40, 20)


def test_fit_page(cfg):
    w, h, s = fit_page(800, 600, cfg)
    assert s == pytest.approx(700 / 600)
    assert h == pytest.approx(700) and w == pytest.approx(800 * 700 / 600)


def test_working_image_resamples():
    rgb = np.zeros((600, 800, 3), np.uint8)
    img, mm = working_image(rgb, 1.25, 2.0)
    assert img.shape[:2] == (375, 500) and mm == pytest.approx(2.0, rel=0.01)
    up, mm2 = working_image(rgb, 1.25, 0.625)
    assert up.shape[1] == 1600 and mm2 == pytest.approx(0.625, rel=0.01)


def test_working_image_caps_size():
    rgb = np.zeros((4000, 6000, 3), np.uint8)
    img, mm = working_image(rgb, 1000 / 6000, 0.1, max_side=2500)
    assert max(img.shape[:2]) == 2500 and mm == pytest.approx(1000 / 2500, rel=0.01)


def test_estimate_noise():
    rng = np.random.default_rng(0)
    flat = np.full((200, 200), 128, np.uint8)
    assert estimate_noise(flat) < 0.5
    noisy = np.clip(128 + rng.normal(0, 20, (200, 200)), 0, 255).astype(np.uint8)
    assert 17 < estimate_noise(noisy) < 23


def test_denoise_reduces_noise():
    rng = np.random.default_rng(1)
    img = np.clip(128 + rng.normal(0, 25, (120, 160, 3)), 0, 255).astype(np.uint8)
    out = denoise(img)
    assert out.shape == img.shape and out.dtype == np.uint8
    gray = lambda a: a.mean(axis=2).astype(np.uint8)
    assert estimate_noise(gray(out)) < 0.5 * estimate_noise(gray(img))
    clean = np.full((50, 50), 100, np.uint8)
    assert np.array_equal(denoise(clean), clean)


def test_luminance_and_foreground():
    rgb = np.array([[[0, 0, 0], [255, 255, 255], [230, 230, 230], [200, 255, 255]]], np.uint8)
    lum = luminance(rgb)
    assert lum.dtype == np.float32 and lum[0, 0] < 0.01 and lum[0, 1] > 0.99
    assert foreground_mask(rgb, 220).tolist() == [[True, False, False, True]]


def test_palette_kmeans_sorted_dark_to_light():
    img = np.zeros((30, 30, 3), np.uint8)
    img[:, 10:20] = (220, 30, 30); img[:, 20:] = (250, 240, 60)
    pal, labels = palette_kmeans(img, np.ones((30, 30), bool), 3)
    assert pal.shape == (3, 3) and pal.dtype == np.uint8
    assert tuple(pal[0]) == pytest.approx((0, 0, 0), abs=3)
    assert labels[0, 0] == 0 and labels[0, 15] == 1 and labels[0, 25] == 2


def test_palette_kmeans_more_clusters_than_colours():
    img = np.zeros((20, 20, 3), np.uint8); img[:, 10:] = 200
    mask = np.ones((20, 20), bool); mask[0, 0] = False
    pal, labels = palette_kmeans(img, mask, 5)
    assert 1 <= len(pal) <= 5 and labels.max() < len(pal) and labels[0, 0] == -1


def test_palette_kmeans_empty_mask():
    pal, labels = palette_kmeans(np.zeros((5, 5, 3), np.uint8), np.zeros((5, 5), bool), 2)
    assert len(pal) == 0 and (labels == -1).all()


def test_mask_to_geometry_ring_and_island():
    yy, xx = np.mgrid[0:300, 0:300]
    r = np.hypot(xx - 150, yy - 150)
    ring = (r <= 100) & (r > 40)
    g = mask_to_geometry(ring, 1.0)
    assert g.area == pytest.approx(np.pi * (100 ** 2 - 40 ** 2), rel=0.02)
    assert len(list(g.interiors)) == 1
    island = ring | (r <= 20)
    g2 = mask_to_geometry(island, 0.5)
    assert g2.area == pytest.approx(0.25 * np.pi * (100 ** 2 - 40 ** 2 + 20 ** 2), rel=0.02)
    assert g2.geom_type == "MultiPolygon" and len(g2.geoms) == 2
    s = mask_to_geometry(ring, 1.0, simplify_mm=0.5)
    assert len(s.exterior.coords) < len(g.exterior.coords)
    assert mask_to_geometry(np.zeros((5, 5), bool), 1.0).is_empty


def test_mask_to_geometry_min_area():
    m = np.zeros((100, 100), bool); m[10:60, 10:60] = True; m[80:83, 80:83] = True; m[30:32, 30:32] = False
    g = mask_to_geometry(m, 1.0, min_area_mm2=20)
    assert g.geom_type == "Polygon" and len(list(g.interiors)) == 0


def test_tone_levels():
    lum = np.linspace(0, 1, 10001).reshape(1, -1).astype(np.float32)
    t = tone_levels(lum, np.ones_like(lum, bool), 4)
    assert np.allclose(t, [0.25, 0.5, 0.75], atol=1e-3)


def test_sample_colors():
    img = np.zeros((10, 10, 3), np.uint8); img[:, 5:] = (9, 8, 7)
    c = sample_colors(img, np.array([[1.0, 1.0], [8.0, 3.0]]), 1.0)
    assert c.tolist() == [[0, 0, 0], [9, 8, 7]]
```

- [ ] **Step 2: Run to verify failure** — `.venv/Scripts/python.exe -m pytest tests/test_raster.py -q` → FAIL.

- [ ] **Step 3: Implement `img2gcode/raster.py`**

Key code (complete the rest following the interface list):

```python
def load_rgb(path) -> np.ndarray:
    img = ImageOps.exif_transpose(Image.open(path))
    if img.mode in ("I;16", "I;16B", "I;16L", "I", "F"):
        a = np.asarray(img).astype(np.float64)
        top = 65535.0 if a.max() > 255 else 255.0
        img = Image.fromarray(np.clip(a / top * 255.0, 0, 255).astype(np.uint8), "L")
    rgba = img.convert("RGBA")
    bg = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
    bg.alpha_composite(rgba)
    return np.asarray(bg.convert("RGB"), dtype=np.uint8).copy()


def fit_page(src_w, src_h, cfg):
    s = min(cfg.machine.print_width / src_w, cfg.machine.print_height / src_h)
    return src_w * s, src_h * s, s


def working_image(rgb, mm_per_src_px, mm_per_px, max_side=2500):
    H, W = rgb.shape[:2]
    f = min(mm_per_src_px / mm_per_px, max_side / max(H, W))
    w, h = max(8, int(round(W * f))), max(8, int(round(H * f)))
    if (w, h) == (W, H):
        return rgb.copy(), mm_per_src_px
    interp = cv2.INTER_AREA if f < 1 else cv2.INTER_CUBIC
    return cv2.resize(rgb, (w, h), interpolation=interp), W * mm_per_src_px / w


def estimate_noise(gray) -> float:
    g = np.asarray(gray, np.float32)
    if g.ndim == 3:
        g = g.mean(axis=2)
    H, W = g.shape
    if H < 3 or W < 3:
        return 0.0
    k = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], np.float32)
    conv = cv2.filter2D(g, -1, k, borderType=cv2.BORDER_REFLECT)[1:-1, 1:-1]
    return float(np.sqrt(np.pi / 2) * np.abs(conv).sum() / (6 * (W - 2) * (H - 2)))


def denoise(img, sigma=None):
    if sigma is None:
        sigma = estimate_noise(img)
    if sigma <= 1.5:
        return img
    h = 1.1 * sigma
    if img.ndim == 2:
        return cv2.fastNlMeansDenoising(img, None, h=h, templateWindowSize=7, searchWindowSize=21)
    return cv2.fastNlMeansDenoisingColored(img, None, h, h, 7, 21)
```

`palette_kmeans`: Lab via `cv2.cvtColor(rgb, cv2.COLOR_RGB2Lab)`; sample up to `max_samples` masked pixels with `np.random.default_rng(0)`; `cv2.setRNGSeed(0)`; `cv2.kmeans(sample, k, None, (EPS+MAX_ITER, 50, 0.5), 3, cv2.KMEANS_PP_CENTERS)`; sort centres by L; assign every masked pixel to its nearest centre (chunk the distance computation in blocks of 500k pixels); convert centres back to RGB with `cv2.cvtColor(centres[None].astype(np.uint8), cv2.COLOR_Lab2RGB)[0]`.

`mask_to_geometry`: `skimage.measure.find_contours(np.pad(mask, 1).astype(np.float32), 0.5)`; convert each contour `(row, col)` to page mm with `((col - 1) + 0.5) · mm_per_px, ((row - 1) + 0.5) · mm_per_px`; build `Polygon`s; marching-squares rings never cross, so assemble by containment depth: sort by area descending, use `shapely.STRtree` to count, for each ring's `representative_point()`, how many larger polygons contain it; even depth = shell, odd depth = hole attached to its smallest containing shell; union the shells-with-holes. Then `simplify(simplify_mm, preserve_topology=True)` when > 0, drop polygons and holes smaller than `min_area_mm2`, and `buffer(0)` if invalid.

- [ ] **Step 4: Run tests** → PASS.

- [ ] **Step 5: Commit** — `git add img2gcode/raster.py tests/test_raster.py && git commit -m "Add raster loading, resampling, denoising and vectorizing helpers"`

---

### Task 4: writer.py (format-preserving port)

**Files:**
- Modify: `img2gcode/writer.py` (full rewrite)
- Create: `tests/test_writer.py`
- Read-only fixtures: `tests/data/golden_writer.gcode`, `tests/data/golden_writer_input.json`

**Interfaces:**
- Consumes: `Toolpaths`, `ToolPass`, `Stroke` (Task 1), `Config`.
- Produces: `PrintStats` (same fields and `total_time_min`, `filament_grams()` as today), `compute_stats(tp, cfg) -> PrintStats`, `gcode_lines(tp, cfg) -> tuple[list[str], PrintStats]`, `write_gcode(tp, cfg, path) -> PrintStats`.

- [ ] **Step 1: Write the failing tests** — `tests/test_writer.py`

```python
import json
from pathlib import Path

import numpy as np

from img2gcode.config import Config
from img2gcode.model import Stroke, ToolPass, Toolpaths
from img2gcode.writer import compute_stats, gcode_lines, write_gcode

DATA = Path(__file__).parent / "data"


def golden_case():
    """Rebuild the moves the old writer saw, as page-mm Toolpaths.

    The old writer mapped pixel y to (H - 1 - y) * s - H * s / 2. Using
    y_page = (y_px + 1) * s with the new Y = height / 2 - y_page gives the
    same numbers exactly (s = 1.25 keeps every value a binary fraction).
    """
    spec = json.loads((DATA / "golden_writer_input.json").read_text(encoding="utf-8"))
    cfg = Config()
    for section, values in spec["cfg"].items():
        for key, value in values.items():
            setattr(getattr(cfg, section), key, value)
    H, W = spec["image_hw"]
    s = 1.25
    passes = []
    for tl in spec["tool_layers"]:
        strokes = [
            Stroke(np.array([[x * s, (y + 1) * s] for x, y in poly], float), tl["tool"], "perimeter")
            for poly in tl["polys"]
        ]
        passes.append(ToolPass(tl["tool"], tl["layer"], strokes))
    tp = Toolpaths(W * s, H * s, 4.0, cfg.layers.num_layers, passes, [(0, 0, 0), (255, 0, 0)], (W, H))
    return tp, cfg


def test_matches_golden_file(tmp_path):
    tp, cfg = golden_case()
    out = tmp_path / "g.gcode"
    write_gcode(tp, cfg, out)
    got = out.read_text(encoding="utf-8").splitlines()
    want = (DATA / "golden_writer.gcode").read_text(encoding="utf-8").splitlines()
    assert got[0].startswith("; generated by img2gcode on ")
    assert got[1:] == want[1:]


def test_stats_match_header():
    tp, cfg = golden_case()
    lines, stats = gcode_lines(tp, cfg)
    assert stats == compute_stats(tp, cfg)
    assert f"; print_distance: {stats.print_distance_mm:.1f} mm" in "\n".join(lines)


def test_creates_parent_dirs_and_utf8(tmp_path):
    tp, cfg = golden_case()
    out = tmp_path / "a" / "b" / "x.gcode"
    write_gcode(tp, cfg, out)
    assert "→" in out.read_text(encoding="utf-8")


def test_empty_toolpaths_still_valid(tmp_path):
    cfg = Config()
    tp = Toolpaths(100.0, 50.0, 4.0, 1, [ToolPass(0, 0, [])], [(0, 0, 0)], (200, 100))
    lines, stats = gcode_lines(tp, cfg)
    text = "\n".join(lines)
    assert stats.print_distance_mm == 0 and ";LAYER_CHANGE" in text and "M84 ; disable motors" in text
    assert not any(l.startswith("T0") for l in lines)
```

- [ ] **Step 2: Run to verify failure** → FAIL (`gcode_lines` missing).

- [ ] **Step 3: Implement** — port `GCodeWriter.write` and `compute_stats` from the current file line for line:

```python
"""G-code writer. The output format is unchanged from the original writer;
tests/data/golden_writer.gcode pins it."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Tuple

from .config import Config
from .model import ToolPass, Toolpaths

_PLA_DENSITY_G_CM3 = 1.24


@dataclass
class PrintStats:
    print_distance_mm: float
    travel_distance_mm: float
    filament_mm: float
    print_time_min: float
    travel_time_min: float

    @property
    def total_time_min(self) -> float:
        return self.print_time_min + self.travel_time_min

    def filament_grams(self, filament_diameter_mm: float, density_g_cm3: float = _PLA_DENSITY_G_CM3) -> float:
        radius = filament_diameter_mm / 2.0
        return math.pi * radius * radius * self.filament_mm * density_g_cm3 / 1000.0


def _extrusion_length(dist_mm: float, cfg: Config) -> float:
    volume = cfg.extrusion.effective_line_width * cfg.extrusion.layer_height * dist_mm
    fil_r = cfg.extrusion.filament_diameter / 2
    return volume / (math.pi * fil_r ** 2) * cfg.extrusion.extrusion_multiplier


def _pass_moves(tp: Toolpaths, ps: ToolPass) -> List[Tuple[float, float, bool]]:
    """Machine-space moves: a travel to each stroke start, then extrusions."""
    half_w, half_h = tp.width_mm / 2.0, tp.height_mm / 2.0
    moves = []
    for stroke in ps.strokes:
        if len(stroke.points) < 2:
            continue
        for i, (x, y) in enumerate(stroke.points):
            moves.append((round(float(x) - half_w, 4), round(half_h - float(y), 4), i > 0))
    return moves
```
Then `compute_stats` (prev reset per pass, first move skipped, as today), `gcode_lines` (header with `; image size: {W}x{H}px → {w:.1f}x{h:.1f}mm (scale={w / W:.4f})`, `; tools: {tp.n_tools}  layers: {tp.n_layers}`, then the infill / line_width / estimate lines, startup block, `for layer_idx in range(tp.n_layers)` over that layer's passes in `tp.passes` order, end block — every string copied from the current writer), and `write_gcode` (`mkdir(parents=True, exist_ok=True)`, `write_text("\n".join(lines), encoding="utf-8")`, return stats). Do not print from the writer; the CLI reports.

- [ ] **Step 4: Run tests** → PASS (the golden comparison must be exact).

- [ ] **Step 5: Commit** — `git add img2gcode/writer.py tests/test_writer.py && git commit -m "Port G-code writer to Toolpaths, pinned by golden file"`

---

### Task 5: svg_io.py (toolpaths.svg)

**Files:**
- Create: `img2gcode/svg_io.py`, `tests/test_svg_io.py`

**Interfaces:**
- Consumes: `Toolpaths`, `ToolPass`, `Stroke`.
- Produces: `write_toolpaths_svg(tp, path) -> None`, `read_toolpaths_svg(path) -> tuple[Toolpaths, int]` (int = shapes ignored because they sit outside a tool group), `is_toolpaths_svg(path) -> bool`, constants `SVG_NS`, `INKSCAPE_NS`, `SODIPODI_NS`, `XLINK_NS`.

- [ ] **Step 1: Write the failing tests** — `tests/test_svg_io.py`

```python
import xml.etree.ElementTree as ET

import numpy as np

from img2gcode.model import Stroke, ToolPass, Toolpaths
from img2gcode.svg_io import (
    INKSCAPE_NS, SODIPODI_NS, SVG_NS, XLINK_NS, is_toolpaths_svg, read_toolpaths_svg, write_toolpaths_svg,
)

SQ = np.array([[10, 10], [60, 10], [60, 40], [10, 40], [10, 10]], float)
LINE = np.array([[5.5, 70.25], [90.125, 72.0], [120, 90]], float)


def make_tp():
    passes = [
        ToolPass(0, 0, [Stroke(SQ, 0, "perimeter"), Stroke(LINE, 0, "infill")]),
        ToolPass(1, 0, [Stroke(LINE + 5, 1, "line")]),
        ToolPass(0, 1, [Stroke(SQ, 0, "perimeter")]),
        ToolPass(1, 1, []),
    ]
    under = np.zeros((40, 60, 3), np.uint8); under[:, :30] = 255
    return Toolpaths(200.0, 100.0, 4.0, 2, passes, [(20, 20, 20), (200, 40, 40)], (400, 200), under, "edge")


def strokes_of(tp, layer, tool):
    return next(p for p in tp.passes if p.layer == layer and p.tool == tool).strokes


def test_round_trip(tmp_path):
    tp = make_tp(); f = tmp_path / "t.svg"
    write_toolpaths_svg(tp, f)
    got, ignored = read_toolpaths_svg(f)
    assert ignored == 0
    assert (got.width_mm, got.height_mm, got.line_width_mm, got.n_layers) == (200.0, 100.0, 4.0, 2)
    assert got.palette == tp.palette and got.source_px == (400, 200) and got.style == "edge"
    assert [(p.layer, p.tool) for p in got.passes] == [(0, 0), (0, 1), (1, 0), (1, 1)]
    for a, b in zip(tp.passes, got.passes):
        assert len(a.strokes) == len(b.strokes)
        for sa, sb in zip(a.strokes, b.strokes):
            assert sa.kind == sb.kind and sa.tool == sb.tool
            assert np.allclose(sa.points, sb.points, atol=1e-3)
    assert got.underlay is not None and got.underlay.ndim == 3
    half = got.underlay.shape[1] // 2
    assert got.underlay[:, :half].mean() > got.underlay[:, half:].mean() + 100


def test_is_toolpaths_svg(tmp_path):
    f = tmp_path / "t.svg"; write_toolpaths_svg(make_tp(), f)
    assert is_toolpaths_svg(f)
    plain = tmp_path / "p.svg"
    plain.write_text(f'<svg xmlns="{SVG_NS}" viewBox="0 0 10 10"><rect width="5" height="5"/></svg>', encoding="utf-8")
    assert not is_toolpaths_svg(plain)


def _tool_groups(root):
    groups = {}
    for lg in root.iter(f"{{{SVG_NS}}}g"):
        if lg.get("data-layer") is not None:
            for tg in lg:
                if tg.get("data-tool") is not None:
                    groups[(int(lg.get("data-layer")), int(tg.get("data-tool")))] = tg
    return groups


def test_inkscape_style_edits(tmp_path):
    f = tmp_path / "t.svg"; write_toolpaths_svg(make_tp(), f)
    for prefix, uri in (("", SVG_NS), ("inkscape", INKSCAPE_NS), ("sodipodi", SODIPODI_NS), ("xlink", XLINK_NS)):
        ET.register_namespace(prefix, uri)
    tree = ET.parse(f); root = tree.getroot()
    groups = _tool_groups(root)
    g00 = groups[(0, 0)]
    g00.set("transform", "translate(10,5)")
    first = g00[0]
    d = "M " + " L ".join(first.get("points").split())
    path = ET.Element(f"{{{SVG_NS}}}path", {"d": d, "data-kind": "perimeter"})
    g00.remove(first); g00.insert(0, path)
    groups[(0, 1)].set("style", "display:none")
    root.append(ET.Element(f"{{{SVG_NS}}}path", {"d": "M 0,0 L 5,5"}))
    tree.write(f, encoding="utf-8", xml_declaration=True)

    got, ignored = read_toolpaths_svg(f)
    assert ignored == 1
    s00 = strokes_of(got, 0, 0)
    assert s00[0].kind == "perimeter" and np.allclose(s00[0].points, SQ + [10, 5], atol=1e-3)
    assert np.allclose(s00[1].points, LINE + [10, 5], atol=1e-3)
    assert strokes_of(got, 0, 1) == []
    assert len(strokes_of(got, 1, 0)) == 1


def test_nested_transforms(tmp_path):
    f = tmp_path / "n.svg"
    f.write_text(
        f'<svg xmlns="{SVG_NS}" width="100mm" height="100mm" viewBox="0 0 100 100" data-img2gcode="toolpaths" '
        'data-line-width="4" data-layers="1" data-tools="1" data-source-px="100x100">'
        '<g data-layer="0" transform="translate(10,0)"><g data-tool="0" stroke="#000000" transform="rotate(90)">'
        '<polyline points="1,0 2,0"/></g></g></svg>', encoding="utf-8")
    got, _ = read_toolpaths_svg(f)
    assert np.allclose(strokes_of(got, 0, 0)[0].points, [[10, 1], [10, 2]], atol=1e-6)


def test_curved_path_is_sampled(tmp_path):
    f = tmp_path / "c.svg"
    f.write_text(
        f'<svg xmlns="{SVG_NS}" viewBox="0 0 100 100" data-img2gcode="toolpaths" data-line-width="4" '
        'data-layers="1" data-tools="1" data-source-px="100x100"><g data-layer="0"><g data-tool="0" stroke="#000">'
        '<path d="M 0,0 C 0,50 100,50 100,0"/></g></g></svg>', encoding="utf-8")
    got, _ = read_toolpaths_svg(f)
    p = strokes_of(got, 0, 0)[0].points
    assert len(p) >= 20 and np.allclose(p[0], [0, 0]) and np.allclose(p[-1], [100, 0])
    assert abs(p[:, 1].max() - 37.5) < 0.5
```

- [ ] **Step 2: Run to verify failure** → FAIL.

- [ ] **Step 3: Implement `img2gcode/svg_io.py`**

Writer: build the document as a list of strings exactly in the layout of the spec's "toolpaths.svg" section. Points `f"{x:.3f},{y:.3f}"` joined by spaces; colours `'#%02x%02x%02x'`; tool group label `f"Layer {L} · T{t}"`; write every `(layer, tool)` group even when empty; the underlay is the image downscaled to ≤ 1200 px, `cv2.imencode(".jpg", BGR, [cv2.IMWRITE_JPEG_QUALITY, 80])`, base64 in `xlink:href`; the travel group holds one `<path>` with `M x,y L x,y` segments from each stroke end to the next stroke start across the whole print order. Escape attribute text with `xml.sax.saxutils.quoteattr`/`escape`.

Reader:
```python
def read_toolpaths_svg(path):
    root = ET.parse(path).getroot()
    vb = [float(v) for v in root.get("viewBox").replace(",", " ").split()]
    width, height = vb[2], vb[3]
    lw = float(root.get("data-line-width", "0") or 0)
    n_layers = int(root.get("data-layers", "1")); n_tools = int(root.get("data-tools", "1"))
    src = root.get("data-source-px", "0x0").lower().split("x")
    found = {}            # (layer, tool) -> list[Stroke]
    palette = {}          # tool -> RGB
    ignored = 0
    # walk(elem, matrix, layer, tool): skip hidden (style display:none, display="none",
    # visibility hidden) and data-role elements; compose transforms with
    # svgelements.Matrix (child_matrix * parent_matrix, verify with test_nested_transforms);
    # g with data-layer / data-tool updates the context (tool groups record palette from
    # their stroke attribute via svgelements.Color); shapes (polyline, polygon, line, path,
    # rect, circle, ellipse) become one Stroke per subpath via _shape_points(); a shape with
    # no (layer, tool) context counts as ignored.
    ...
    n_layers = max([n_layers] + [l + 1 for l, _ in found]); n_tools = max([n_tools] + [t + 1 for _, t in found])
    passes = [ToolPass(t, l, found.get((l, t), [])) for l in range(n_layers) for t in range(n_tools)]
    pal = [palette.get(t, (128, 128, 128)) for t in range(n_tools)]
    return Toolpaths(width, height, lw, n_layers, passes, pal, (int(src[0]), int(src[1])), underlay, root.get("data-style", "")), ignored
```
`_shape_points`: `polyline`/`polygon` parse `points` (polygon closes); `line` uses x1..y2; `path` → `svgelements.Path(d)`, transformed by the matrix, then for each subpath walk its segments: `Move` starts a new point list, `Line`/`Close` append the end point exactly, curves (`CubicBezier`, `QuadraticBezier`, `Arc`) append `ceil(length / (lw/8))` sampled points; `rect`/`circle`/`ellipse` go through the matching `svgelements` shape → `Path`. Apply the matrix to polyline/line points too (`matrix.point_in_matrix_space`). Underlay: decode the `data:image/...;base64,` href of the `<image>` inside the `data-role="source"` group with `cv2.imdecode`, convert BGR → RGB.

`is_toolpaths_svg`: `ET.iterparse(path, events=("start",))` on the first element, return `attrib.get("data-img2gcode") == "toolpaths"`; any parse error → False.

- [ ] **Step 4: Run tests** → PASS.

- [ ] **Step 5: Commit** — `git add img2gcode/svg_io.py tests/test_svg_io.py && git commit -m "Add toolpaths.svg writer and reader"`

---

### Task 6: skeleton.py (mask → smooth centreline strokes)

**Files:**
- Create: `img2gcode/skeleton.py`, `tests/test_skeleton.py`

**Interfaces:**
- Consumes: `geometry.smooth`, `geometry.simplify`, `geometry.polyline_length`.
- Produces: `skeleton_strokes(mask, mm_per_px, *, spur_mm, smooth_mm, simplify_mm, min_length_mm, max_turn_deg=60.0) -> list[np.ndarray]` (page-mm polylines; closed loops have first == last).

- [ ] **Step 1: Write the failing tests** — `tests/test_skeleton.py`

```python
import cv2
import numpy as np
import pytest

from img2gcode.geometry import polyline_length
from img2gcode.skeleton import skeleton_strokes

KW = dict(spur_mm=8.0, smooth_mm=2.0, simplify_mm=0.2, min_length_mm=6.0)


def mask(lines=(), circles=(), shape=(200, 200), t=5):
    m = np.zeros(shape, np.uint8)
    for a, b in lines:
        cv2.line(m, a, b, 1, t)
    for c, r in circles:
        cv2.circle(m, c, r, 1, t)
    return m.astype(bool)


def straightness(p):
    return np.hypot(*(p[-1] - p[0])) / polyline_length(p)


def test_single_line():
    s = skeleton_strokes(mask([((20, 100), (180, 100))]), 1.0, **KW)
    assert len(s) == 1
    assert polyline_length(s[0]) == pytest.approx(160, abs=8) and straightness(s[0]) > 0.98


def test_plus_gives_two_straight_strokes():
    s = skeleton_strokes(mask([((20, 100), (180, 100)), ((100, 20), (100, 180))]), 1.0, **KW)
    assert len(s) == 2
    assert all(polyline_length(p) > 140 and straightness(p) > 0.95 for p in s)


def test_t_junction_keeps_the_bar_whole():
    s = skeleton_strokes(mask([((20, 50), (180, 50)), ((100, 50), (100, 180))]), 1.0, **KW)
    assert len(s) == 2
    bar = max(s, key=lambda p: np.ptp(p[:, 0]))
    assert np.ptp(bar[:, 0]) > 150 and straightness(bar) > 0.97


def test_ring_is_closed():
    s = skeleton_strokes(mask(circles=[((100, 100), 60)]), 1.0, **KW)
    assert len(s) == 1 and np.allclose(s[0][0], s[0][-1])
    assert polyline_length(s[0]) == pytest.approx(2 * np.pi * 60, rel=0.05)


def test_short_spur_is_pruned():
    s = skeleton_strokes(mask([((20, 100), (180, 100)), ((100, 100), (100, 106))]), 1.0, **KW)
    assert len(s) == 1


def test_specks_dropped_and_empty():
    m = np.zeros((50, 50), bool); m[20:23, 20:23] = True
    assert skeleton_strokes(m, 1.0, **KW) == []
    assert skeleton_strokes(np.zeros((50, 50), bool), 1.0, **KW) == []


def test_pixel_to_mm_mapping():
    s = skeleton_strokes(mask([((20, 100), (180, 100))]), 0.5, spur_mm=4, smooth_mm=1, simplify_mm=0.1, min_length_mm=3)
    p = s[0]
    assert p[:, 1].mean() == pytest.approx(50.25, abs=0.3)
    assert p[:, 0].min() == pytest.approx(10.25, abs=2) and p[:, 0].max() == pytest.approx(90.25, abs=2)
```

- [ ] **Step 2: Run to verify failure** → FAIL.

- [ ] **Step 3: Implement** — algorithm:

1. `skel = skimage.morphology.skeletonize(mask)`; neighbour count via `cv2.filter2D` with a 3×3 ones kernel (centre 0) on the uint8 skeleton.
2. **Nodes**: endpoint pixels (count 1) are nodes; junction pixels (count ≥ 3) are grouped into 8-connected clusters (`cv2.connectedComponents`), one node per cluster positioned at the cluster centroid.
3. **Edges**: from every node pixel, walk each neighbouring non-node skeleton pixel along count-2 pixels (mark visited) until reaching a node pixel; record `(node_a, node_b, pixel_path)` with the node positions prepended/appended. Two node pixels that touch directly form a zero-interior edge. Skeleton components without nodes are closed loops: walk them once and close the path.
4. **Spur pruning**: repeat until stable: remove edges joining a degree-1 node to a degree ≥ 3 node whose arc length < `spur_mm / mm_per_px`.
5. **Joining at junctions**: for each node, direction of each incident edge end = unit vector from the node to the point `max(3, spur_px)` px along the edge (or its far end). Nodes of degree 2 always join their two ends. For degree ≥ 3, sort all pairs by deviation from straight (`angle between d_i and −d_j`) and pair greedily while deviation < `max_turn_deg`.
6. **Chains**: start at every unpaired edge end and follow pairings, concatenating pixel paths (reversing as needed) and marking edges used; leftover unused edges form cycles → closed chains.
7. Convert to mm `(xy + 0.5) · mm_per_px`, drop chains shorter than `min_length_mm`, then `smooth(…, smooth_mm)` and `simplify(…, simplify_mm)`; closed chains stay closed.

- [ ] **Step 4: Run tests** → PASS.

- [ ] **Step 5: Commit** — `git add img2gcode/skeleton.py tests/test_skeleton.py && git commit -m "Add skeleton tracer with spur pruning and straight-through joins"`

---

### Task 7: planner.py

**Files:**
- Create: `img2gcode/planner.py`, `tests/test_planner.py`

**Interfaces:**
- Consumes: `Artwork`, `Region`, `Stroke`, `ToolPass`, `Toolpaths`; `geometry.geometry_rings`, `hatch_lines`, `order_strokes`, `polyline_length`; `skeleton.skeleton_strokes`; `raster` is not needed (rasterize residue with `cv2.fillPoly` on a local grid).
- Produces: `plan(art, cfg, exclusion_zones=None) -> Toolpaths` (`exclusion_zones`: list of `(x0, y0, x1, y1)` page-mm rectangles).

- [ ] **Step 1: Write the failing tests** — `tests/test_planner.py`

```python
import numpy as np
import pytest
from shapely.geometry import LineString, Point, box

from img2gcode.geometry import polyline_length
from img2gcode.model import Artwork, Region, Stroke
from img2gcode.planner import plan

LW = 4.0


def art(regions=(), strokes=(), palette=((0, 0, 0),), w=300.0, h=200.0):
    return Artwork(w, h, list(regions), list(strokes), [tuple(c) for c in palette], (600, 400), None, "test")


def of_kind(tp, kind, tool=None, layer=0):
    return [s for p in tp.passes if p.layer == layer and (tool is None or p.tool == tool)
            for s in p.strokes if s.kind == kind]


def test_first_perimeter_sits_half_a_line_inside(cfg):
    sq = box(20, 20, 120, 120)
    per = of_kind(plan(art([Region(sq, 0)]), cfg), "perimeter")
    assert len(per) == 1 and per[0].closed
    d = [sq.exterior.distance(Point(p)) for p in per[0].points]
    assert min(d) == pytest.approx(LW / 2, abs=0.05) and max(d) == pytest.approx(LW / 2, abs=0.05)


def test_two_loops_and_a_hole(cfg):
    cfg.layers.perimeter_loops = 2
    donut = box(0, 0, 100, 100).difference(box(40, 40, 60, 60))
    per = of_kind(plan(art([Region(donut, 0)], w=100, h=100), cfg), "perimeter")
    assert len(per) == 4 and all(s.closed for s in per)


def test_neighbouring_tools_do_not_overlap(cfg):
    a, b = box(0, 0, 50, 100), box(50, 0, 100, 100)
    tp = plan(art([Region(a, 0), Region(b, 1)], palette=[(0, 0, 0), (255, 0, 0)], w=100, h=100), cfg)
    pa = [LineString(s.points) for s in of_kind(tp, "perimeter", tool=0)]
    pb = [LineString(s.points) for s in of_kind(tp, "perimeter", tool=1)]
    assert min(x.distance(y) for x in pa for y in pb) >= LW - 1e-6


def test_infill_stays_inside(cfg):
    sq = box(0, 0, 100, 80)
    inner = sq.buffer(-(LW + LW / 2) + 0.15 * LW)
    inf = of_kind(plan(art([Region(sq, 0)], w=100, h=80), cfg), "infill")
    assert inf
    zone = inner.buffer(0.05 * LW + 1e-6)
    assert all(zone.covers(LineString(s.points)) for s in inf)


def test_infill_spacing_follows_density(cfg):
    cfg.infill.density = 50; cfg.infill.connect_lines = False; cfg.infill.angle = 0
    inf = of_kind(plan(art([Region(box(0, 0, 100, 100), 0)], w=100, h=100), cfg), "infill")
    ys = np.unique(np.round([s.points[0, 1] for s in inf], 6))
    assert np.allclose(np.diff(ys), 8.0)


def test_exclusion_zone_blocks_infill_only(cfg):
    zone = (30, 30, 70, 70)
    tp = plan(art([Region(box(0, 0, 100, 100), 0)], w=100, h=100), cfg, exclusion_zones=[zone])
    hole = box(*zone).buffer(-0.5)
    assert not any(hole.intersects(LineString(s.points)) for s in of_kind(tp, "infill"))
    assert len(of_kind(tp, "perimeter")) == 1


def test_thin_feature_becomes_centreline(cfg):
    tp = plan(art([Region(box(20, 50, 120, 52.5), 0)], w=150, h=100), cfg)
    assert of_kind(tp, "perimeter") == []
    lines = of_kind(tp, "line")
    assert len(lines) == 1 and polyline_length(lines[0].points) > 85
    assert np.all(np.abs(lines[0].points[:, 1] - 51.25) < 1.0)


def test_wide_shape_has_no_stray_centrelines(cfg):
    tp = plan(art([Region(box(0, 0, 100, 100), 0)], w=100, h=100), cfg)
    assert of_kind(tp, "line") == []


def test_layers_alternate_infill_angle(cfg):
    cfg.layers.num_layers = 2; cfg.infill.connect_lines = False; cfg.infill.angle = 0
    tp = plan(art([Region(box(0, 0, 100, 100), 0)], w=100, h=100), cfg)
    l0, l1 = of_kind(tp, "infill", layer=0), of_kind(tp, "infill", layer=1)
    assert l0 and l1
    assert all(abs(s.points[0, 1] - s.points[-1, 1]) < 1e-6 for s in l0)
    assert all(abs(s.points[0, 0] - s.points[-1, 0]) < 1e-6 for s in l1)


def test_line_strokes_kept_and_every_pass_present(cfg):
    cfg.layers.num_layers = 2
    st = [Stroke(np.array([[10, 10], [50, 10]], float), 0), Stroke(np.array([[60, 10], [90, 40]], float), 1)]
    tp = plan(art([], st, palette=[(0, 0, 0), (9, 9, 9)]), cfg)
    assert [(p.layer, p.tool) for p in tp.passes] == [(0, 0), (0, 1), (1, 0), (1, 1)]
    assert sum(len(p.strokes) for p in tp.passes) == 4


def test_metadata_copied(cfg):
    a = art([Region(box(0, 0, 50, 50), 0)])
    a.underlay = np.zeros((4, 6, 3), np.uint8)
    tp = plan(a, cfg)
    assert (tp.width_mm, tp.height_mm, tp.source_px, tp.style) == (300.0, 200.0, (600, 400), "test")
    assert tp.line_width_mm == LW and tp.n_layers == cfg.layers.num_layers
    assert tp.palette == [(0, 0, 0)] and tp.underlay is a.underlay
```

- [ ] **Step 2: Run to verify failure** → FAIL.

- [ ] **Step 3: Implement** per the spec's Planner section. Shape of the code:

```python
def plan(art, cfg, exclusion_zones=None):
    lw = cfg.extrusion.effective_line_width
    loops = cfg.layers.perimeter_loops
    spacing = cfg.infill.line_spacing if cfg.infill.line_spacing > 0 else lw / (max(1.0, min(100.0, cfg.infill.density)) / 100.0)
    blocked = unary_union([box(*z) for z in exclusion_zones]) if exclusion_zones else None
    per_tool = {}   # tool -> (perimeter strokes, centreline strokes, infill polygon)
    for t in range(art.n_tools):
        R = unary_union([r.shape for r in art.regions if r.tool == t]).buffer(0)
        perims, lines, inner = [], [], None
        if not R.is_empty:
            for k in range(loops):
                for ring in geometry_rings(R.buffer(-(k + 0.5) * lw)):
                    perims.append(Stroke(ring, t, "perimeter"))
            lines = _thin_features(R, lw, t)        # rasterize residue at lw/4, skeleton_strokes(spur_mm=lw, smooth_mm=lw/2, simplify_mm=0.05*lw, min_length_mm=lw)
            inner = R.buffer(-(loops * lw + lw / 2) + 0.15 * lw) if loops > 0 else R.buffer(-lw / 2)
            if blocked is not None:
                inner = inner.difference(blocked)
        lines += [s for s in art.strokes if s.tool == t]
        per_tool[t] = (perims, lines, inner)
    passes = []
    for layer in range(cfg.layers.num_layers):
        angle = cfg.infill.angle + (90.0 if layer % 2 else 0.0)
        for t in range(art.n_tools):
            perims, lines, inner = per_tool[t]
            infill = [Stroke(p, t, "infill") for p in hatch_lines(inner, spacing, angle, cfg.infill.connect_lines, 0.05 * lw)] if inner is not None else []
            ordered, cursor = [], None
            for group in (perims, lines, infill):
                group = [s for s in group if polyline_length(s.points) >= 0.25 * lw]
                if group:
                    group = order_strokes(group, cursor)
                    ordered += group
                    cursor = tuple(group[-1].points[-1])
            passes.append(ToolPass(t, layer, ordered))
    return Toolpaths(art.width_mm, art.height_mm, lw, cfg.layers.num_layers, passes, list(art.palette), art.source_px, art.underlay, art.style)
```
`_thin_features`: `residue = R.difference(R.buffer(-lw / 2).buffer(lw / 2))`; if empty or area < lw²/4 return `[]`; rasterize residue on a grid of `lw/4` mm covering its bounds (+2 px margin) with `cv2.fillPoly` over `geometry_rings` (exteriors 1, holes 0, via `rings_to_geometry`-consistent evenodd fill: fill exteriors then holes with 0), run `skeleton_strokes`, offset back to page coordinates, wrap as `Stroke(p, t, "line")`.

- [ ] **Step 4: Run tests** → PASS.

- [ ] **Step 5: Commit** — `git add img2gcode/planner.py tests/test_planner.py && git commit -m "Add planner: perimeters, thin features, infill, ordering"`

---

### Task 8: svg_input.py (artwork SVG → Artwork)

**Files:**
- Create: `img2gcode/svg_input.py`, `tests/test_svg_input.py`

**Interfaces:**
- Consumes: `Artwork`, `Region`, `Stroke`; `geometry.rings_to_geometry`; `Config`.
- Produces: `load_svg_artwork(path, cfg, n_tools=None) -> Artwork` (`n_tools=None`: one tool per distinct colour; an int merges colours down to that many). Raises `ValueError` when the SVG has no printable shape.

- [ ] **Step 1: Write the failing tests** — `tests/test_svg_input.py`

```python
from pathlib import Path

import numpy as np
import pytest

from img2gcode.svg_input import load_svg_artwork

S = 7.0   # 100-unit viewBox on a 1000x700 print area -> 7 mm per unit
IMAGES = Path(__file__).resolve().parent.parent / "images"


def svg(tmp_path, body, vb="0 0 100 100", extra=""):
    p = tmp_path / "a.svg"
    p.write_text(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="{vb}" {extra}>{body}</svg>', encoding="utf-8")
    return p


def area(a, tool):
    return sum(r.shape.area for r in a.regions if r.tool == tool)


def test_fit_to_print_area(cfg, tmp_path):
    a = load_svg_artwork(svg(tmp_path, '<rect width="10" height="10"/>'), cfg)
    assert (a.width_mm, a.height_mm) == pytest.approx((700, 700))
    assert a.source_px == (100, 100) and a.underlay is None


def test_physical_size_with_tiny_viewbox(cfg, tmp_path):
    a = load_svg_artwork(svg(tmp_path, '<rect width="24" height="12"/>', "0 0 24 12", 'width="10mm" height="5mm"'), cfg)
    assert (a.width_mm, a.height_mm) == pytest.approx((1000, 500))
    assert area(a, 0) == pytest.approx(1000 * 500, rel=0.01)


def test_two_colours_sorted_dark_first(cfg, tmp_path):
    body = '<rect width="40" height="40" fill="#ffcc00"/><rect x="50" y="50" width="40" height="40" fill="#202080"/>'
    a = load_svg_artwork(svg(tmp_path, body), cfg)
    assert a.palette == [(32, 32, 128), (255, 204, 0)]
    assert area(a, 0) == pytest.approx((40 * S) ** 2, rel=0.01)


def test_nonzero_reversed_subpath_is_a_hole(cfg, tmp_path):
    d = "M10,10 H90 V90 H10 Z M30,30 V70 H70 V30 Z"
    a = load_svg_artwork(svg(tmp_path, f'<path d="{d}"/>'), cfg)
    assert area(a, 0) == pytest.approx((80 ** 2 - 40 ** 2) * S * S, rel=0.01)


def test_nonzero_same_direction_is_filled(cfg, tmp_path):
    d = "M10,10 H90 V90 H10 Z M30,30 H70 V70 H30 Z"
    a = load_svg_artwork(svg(tmp_path, f'<path d="{d}"/>'), cfg)
    assert area(a, 0) == pytest.approx(80 ** 2 * S * S, rel=0.01)


def test_evenodd_makes_a_hole(cfg, tmp_path):
    d = "M10,10 H90 V90 H10 Z M30,30 H70 V70 H30 Z"
    a = load_svg_artwork(svg(tmp_path, f'<path d="{d}" fill-rule="evenodd"/>'), cfg)
    assert area(a, 0) == pytest.approx((80 ** 2 - 40 ** 2) * S * S, rel=0.01)


def test_later_shape_covers_earlier(cfg, tmp_path):
    body = '<rect width="60" height="100" fill="#ff0000"/><rect x="40" width="60" height="100" fill="#0000ff"/>'
    a = load_svg_artwork(svg(tmp_path, body), cfg)
    red = a.palette.index((255, 0, 0)); blue = a.palette.index((0, 0, 255))
    assert area(a, red) == pytest.approx(40 * 100 * S * S, rel=0.01)
    assert area(a, blue) == pytest.approx(60 * 100 * S * S, rel=0.01)


def test_white_is_paper(cfg, tmp_path):
    body = '<rect width="100" height="100" fill="#000"/><rect x="25" y="25" width="50" height="50" fill="#fff"/>'
    a = load_svg_artwork(svg(tmp_path, body), cfg)
    assert len(a.palette) == 1
    assert area(a, 0) == pytest.approx((100 ** 2 - 50 ** 2) * S * S, rel=0.01)


def test_thick_stroke_is_region_thin_stroke_is_line(cfg, tmp_path):
    body = ('<path d="M10,20 H90" stroke="#000" stroke-width="2" fill="none"/>'
            '<path d="M10,60 H90" stroke="#000" stroke-width="0.2" fill="none"/>')
    a = load_svg_artwork(svg(tmp_path, body), cfg)
    assert area(a, 0) == pytest.approx(80 * 2 * S * S, rel=0.05)
    assert len(a.strokes) == 1 and a.strokes[0].kind == "line"
    assert np.allclose(a.strokes[0].points[:, 1], 60 * S)


def test_gradient_uses_mean_stop_colour(cfg, tmp_path):
    body = ('<defs><linearGradient id="g"><stop offset="0" stop-color="#000000"/>'
            '<stop offset="1" stop-color="#ffffff"/></linearGradient></defs>'
            '<rect width="50" height="50" fill="url(#g)"/>')
    a = load_svg_artwork(svg(tmp_path, body), cfg)
    assert len(a.palette) == 1 and all(abs(c - 128) <= 2 for c in a.palette[0])


def test_near_identical_colours_merge(cfg, tmp_path):
    body = '<rect width="40" height="40" fill="#5a9821"/><rect x="50" width="40" height="40" fill="#5b9922"/>'
    assert len(load_svg_artwork(svg(tmp_path, body), cfg).palette) == 1


def test_n_tools_merges_colours(cfg, tmp_path):
    body = ''.join(f'<rect x="{i * 25}" width="20" height="20" fill="{c}"/>'
                   for i, c in enumerate(["#000000", "#303030", "#ff0000", "#ffff00"]))
    a = load_svg_artwork(svg(tmp_path, body), cfg, n_tools=2)
    assert len(a.palette) == 2 and {r.tool for r in a.regions} <= {0, 1}


def test_group_transform_applied(cfg, tmp_path):
    a = load_svg_artwork(svg(tmp_path, '<g transform="translate(50,0)"><rect width="10" height="10"/></g>'), cfg)
    assert a.regions[0].shape.bounds[0] == pytest.approx(350, abs=0.5)


def test_no_shapes_raises(cfg, tmp_path):
    with pytest.raises(ValueError):
        load_svg_artwork(svg(tmp_path, ""), cfg)


@pytest.mark.skipif(not list(IMAGES.glob("*.svg")), reason="run scripts/fetch_test_images.py")
def test_downloaded_svgs_load(cfg):
    for f in sorted(IMAGES.glob("*.svg")):
        a = load_svg_artwork(f, cfg)
        assert a.regions or a.strokes, f.name
```

- [ ] **Step 2: Run to verify failure** → FAIL.

- [ ] **Step 3: Implement** per the spec's "Artwork SVG input" section with `svgelements.SVG.parse(path, reify=True)`. Determine the coordinate frame from the parsed document (with `reify=True`, shapes are already mapped through the viewBox into the document's width/height space — confirm with `test_physical_size_with_tiny_viewbox` and `test_fit_to_print_area`), fit it into the print area, then walk `svg.elements()` in order:
  - Skip non-`Shape` elements and invisible ones (`display:none`, `visibility:hidden`, `opacity`/`fill-opacity` 0).
  - Fill: `elem.fill` (a `Color`; `None`/value `None` means none). For `url(#id)` fills, find the gradient by id and average its `<stop>` colours; fall back to (128,128,128). Rings: each subpath of `Path(elem)` sampled at ≤ 0.25 mm on the page (`Line` endpoints exact, curves sampled); geometry = `rings_to_geometry(rings, elem.values.get("fill-rule", "nonzero"))` scaled to page mm.
  - Stroke: `elem.stroke` colour and `elem.stroke_width` (already scaled by `reify`; scale to page mm). Width ≥ 1.5·lw → region `LineString(sub).buffer(width/2)` per subpath (round caps/joins); otherwise `Stroke(points, kind="line")`.
  - Occlusion: keep a paint list `[(geometry, rgb)]`; subtract every new filled geometry from all earlier entries.
  - Paper: drop entries whose colour has all channels ≥ `cfg.image.white_threshold` (they have already occluded).
  - Colours: merge colours closer than ΔE 5 (Lab, `cv2.cvtColor` on uint8) into the first one seen; if `n_tools` is an int and more colours remain, area-weighted k-means in Lab down to `n_tools` (`cv2.kmeans` on repeated samples or a weighted Lloyd loop, seed 0). Palette sorted dark → light (Lab L); regions are the union per tool, `buffer(0)`, clipped to the page box; strokes take the tool of their colour.
  - Raise `ValueError(f"No printable shapes in {path}")` if nothing remains.

- [ ] **Step 4: Run tests** (run `scripts/fetch_test_images.py` first so the downloaded SVGs are exercised) → PASS.

- [ ] **Step 5: Commit** — `git add img2gcode/svg_input.py tests/test_svg_input.py && git commit -m "Add artwork SVG input with fill rules, occlusion and colour merging"`

---

### Task 9: styles/fill.py

**Files:**
- Create: `img2gcode/styles/fill.py`, `tests/test_style_fill.py`

**Interfaces:**
- Consumes: `raster.*` (Task 3), `Artwork`, `Region`.
- Produces: `make(rgb, cfg) -> Artwork` (`style="fill"`, `underlay=rgb`, `source_px=(W, H)`).

- [ ] **Step 1: Write the failing tests** — `tests/test_style_fill.py`

```python
import time

import numpy as np
import pytest
from shapely.ops import unary_union

from img2gcode.styles.fill import make
from tests.synth import CIRCLE, RECT, SHAPES_H, SHAPES_W, assert_valid_artwork, page_scale, shapes_rgb


def tool_geom(a, t):
    return unary_union([r.shape for r in a.regions if r.tool == t])


def test_contract(cfg):
    a = make(shapes_rgb(), cfg)
    assert_valid_artwork(a, cfg)
    assert a.style == "fill" and a.underlay is not None and a.source_px == (SHAPES_W, SHAPES_H)


def test_noisy_shapes_give_clean_regions(cfg):
    cfg.tools.num_tools = 2
    a = make(shapes_rgb(noise=45), cfg)
    s = page_scale(cfg, SHAPES_W, SHAPES_H)
    circle = np.pi * (CIRCLE[1] * s) ** 2
    rect = (RECT[2] - RECT[0] + 1) * (RECT[3] - RECT[1] + 1) * s * s
    g0, g1 = tool_geom(a, 0), tool_geom(a, 1)          # tool 0 is the darker colour (the circle)
    assert g0.area == pytest.approx(circle, rel=0.10)
    assert g1.area == pytest.approx(rect, rel=0.10)
    for g in (g0, g1):
        parts = list(getattr(g, "geoms", [g]))
        assert len(parts) <= 4
        holes = sum(sum(type(p)(h).area for h in p.interiors) for p in parts)
        assert holes < 0.01 * g.area


def test_background_is_not_printed(cfg):
    a = make(shapes_rgb(noise=30), cfg)
    assert unary_union([r.shape for r in a.regions]).area < 0.6 * a.width_mm * a.height_mm


def test_single_tool(cfg):
    cfg.tools.num_tools = 1
    a = make(shapes_rgb(), cfg)
    assert len(a.palette) == 1 and {r.tool for r in a.regions} == {0}


def test_more_tools_than_colours(cfg):
    cfg.tools.num_tools = 6
    a = make(shapes_rgb(), cfg)
    assert_valid_artwork(a, cfg)


def test_blank_image(cfg):
    a = make(np.full((100, 120, 3), 255, np.uint8), cfg)
    assert_valid_artwork(a, cfg, allow_empty=True)
    assert a.regions == [] and a.strokes == []


@pytest.mark.slow
def test_runtime_large(cfg):
    import cv2
    big = cv2.resize(shapes_rgb(noise=40), (2000, 1500), interpolation=cv2.INTER_NEAREST)
    t = time.perf_counter(); make(big, cfg)
    assert time.perf_counter() - t < 30
```

(The `Polygon(h)` construction inside `holes`: `type(p)(h)` builds a Polygon from an interior ring.)

- [ ] **Step 2: Run to verify failure** → FAIL.

- [ ] **Step 3: Implement** per the spec's fill section:
  1. `W, H = rgb.shape[1], rgb.shape[0]`; `width, height, s = fit_page(W, H, cfg)`; `img, mmpp = working_image(rgb, s, lw / 4)`; `lw_px = lw / mmpp`.
  2. `img = denoise(img)`; `img = cv2.pyrMeanShiftFiltering(img, sp=max(2, int(round(lw_px))), sr=18)`.
  3. `fg = foreground_mask(img, cfg.image.white_threshold)`; if `not fg.any()` return empty Artwork with a one-entry dark palette `[(32, 32, 32)]`.
  4. `palette, labels = palette_kmeans(img, fg, cfg.tools.num_tools)`.
  5. Majority smoothing over classes `{-1, 0..k-1}`: stack one-hot maps, `cv2.GaussianBlur(σ = cfg.fill.smoothing · lw_px)`, argmax.
  6. Minimum feature: for each class (including background), connected components with area < `(cfg.fill.min_feature_factor · lw_px)²` px become "unknown"; fill unknown pixels from the nearest known pixel (`scipy.ndimage.distance_transform_edt(unknown, return_indices=True)`).
  7. For each tool `t`: `geom = mask_to_geometry(labels == t, mmpp, simplify_mm=0.1 * lw, min_area_mm2=(cfg.fill.min_feature_factor * lw) ** 2)`; clip to the page box; add `Region(geom, t)` when non-empty.
  8. Return `Artwork(width, height, regions, [], [tuple(map(int, c)) for c in palette], (W, H), rgb, "fill")`.

- [ ] **Step 4: Run tests** (`-m "slow or not slow"` once to include the runtime test) → PASS.

- [ ] **Step 5: Commit** — `git add img2gcode/styles/fill.py tests/test_style_fill.py && git commit -m "Add noise-robust fill style"`

---

### Task 10: styles/edge.py

**Files:**
- Create: `img2gcode/styles/edge.py`, `tests/test_style_edge.py`

**Interfaces:**
- Consumes: `raster.*`, `skeleton.skeleton_strokes` (Task 6), `geometry.deoverlap`, `Artwork`, `Stroke`.
- Produces: `make(rgb, cfg) -> Artwork` (`style="edge"`, strokes only, kind `"line"`).

- [ ] **Step 1: Write the failing tests** — `tests/test_style_edge.py`

```python
import time

import cv2
import numpy as np
import pytest

from img2gcode.styles.edge import make
from tests.synth import (
    LW, assert_valid_artwork, page_scale, recall_precision, shapes_outline_mm, shapes_rgb, total_length,
)


def test_contract(cfg):
    a = make(shapes_rgb(), cfg)
    assert_valid_artwork(a, cfg)
    assert a.style == "edge" and a.regions == [] and all(s.kind == "line" for s in a.strokes)


def test_noisy_shapes_recall_and_precision(cfg):
    cfg.tools.num_tools = 1
    a = make(shapes_rgb(noise=45), cfg)
    r, p = recall_precision([s.points for s in a.strokes], shapes_outline_mm(cfg), tol=1.5 * LW)
    assert r >= 0.85, f"recall {r:.2f}"
    assert p >= 0.85, f"precision {p:.2f}"


def test_clean_shapes(cfg):
    a = make(shapes_rgb(), cfg)
    r, p = recall_precision([s.points for s in a.strokes], shapes_outline_mm(cfg), tol=1.5 * LW)
    assert r >= 0.9 and p >= 0.9


def test_thin_dark_line_is_drawn_once(cfg):
    img = np.full((300, 600, 3), 235, np.uint8)
    cv2.line(img, (50, 150), (550, 150), (0, 0, 0), 3)
    a = make(img, cfg)
    true_len = 500 * page_scale(cfg, 600, 300)
    assert 0.8 * true_len <= total_length([s.points for s in a.strokes]) <= 1.2 * true_len


def test_multi_tool_palette(cfg):
    cfg.tools.num_tools = 2
    a = make(shapes_rgb(), cfg)
    assert len(a.palette) == 2 and {s.tool for s in a.strokes} <= {0, 1}


def test_detail_adds_ink(cfg):
    img = shapes_rgb()
    for i in range(8):
        cv2.circle(img, (60 + 90 * i, 540), 12, (90, 90, 90), 2)
    cfg.edge.detail = 0.1
    low = total_length([s.points for s in make(img, cfg).strokes])
    cfg.edge.detail = 0.9
    high = total_length([s.points for s in make(img, cfg).strokes])
    assert high >= low


def test_debug_edges_path(cfg, tmp_path):
    cfg.edge.debug_edges_path = str(tmp_path / "edges.png")
    make(shapes_rgb(), cfg)
    m = cv2.imread(str(tmp_path / "edges.png"), cv2.IMREAD_GRAYSCALE)
    assert m is not None and set(np.unique(m)) <= {0, 255}


def test_blank_image(cfg):
    a = make(np.full((100, 120, 3), 240, np.uint8), cfg)
    assert_valid_artwork(a, cfg, allow_empty=True)
    assert a.strokes == []


@pytest.mark.slow
def test_runtime_large(cfg):
    big = cv2.resize(shapes_rgb(noise=40), (2000, 1500), interpolation=cv2.INTER_NEAREST)
    t = time.perf_counter(); make(big, cfg)
    assert time.perf_counter() - t < 30
```

- [ ] **Step 2: Run to verify failure** → FAIL.

- [ ] **Step 3: Implement** per the spec's edge section. Reference for the two core filters (from the validated prototype):

```python
def _tangent_field(g, sigma):
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3); gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    jxx = cv2.GaussianBlur(gx * gx, (0, 0), sigma)
    jxy = cv2.GaussianBlur(gx * gy, (0, 0), sigma)
    jyy = cv2.GaussianBlur(gy * gy, (0, 0), sigma)
    lam = 0.5 * (jxx + jyy - np.sqrt((jxx - jyy) ** 2 + 4 * jxy ** 2))
    tx, ty = jxy, lam - jxx
    n = np.hypot(tx, ty) + 1e-9
    return tx / n, ty / n


def _lic(img, tx, ty, length):
    """Average img along the tangent field over ±length px (Gaussian weights)."""
    H, W = img.shape
    ys, xs = np.mgrid[0:H, 0:W].astype(np.float32)
    acc, wsum = img.copy(), np.ones_like(img)
    for sign in (1.0, -1.0):
        px, py = xs.copy(), ys.copy(); dx, dy = tx * sign, ty * sign
        for i in range(1, int(np.ceil(length)) + 1):
            ltx = cv2.remap(tx, px, py, cv2.INTER_LINEAR); lty = cv2.remap(ty, px, py, cv2.INTER_LINEAR)
            flip = (ltx * dx + lty * dy) < 0
            ltx[flip] *= -1; lty[flip] *= -1
            dx, dy = ltx, lty
            px, py = px + dx, py + dy
            w = np.float32(np.exp(-(i * i) / (2 * (length / 2.0) ** 2)))
            acc += w * cv2.remap(img, px, py, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT); wsum += w
    return acc / wsum
```
Pipeline: `working_image(rgb, s, lw / 3)`; `lum = luminance(denoise(img))` (CLAHE on the uint8 L channel first when `contrast > 0`); `σ_e = max(0.7, lw_px · (0.6 − 0.35 · detail))`; `dog = G(σ_e) − 0.98 · G(1.6 σ_e)`; `dog = _lic(dog, *_tangent_field(lum, 2 σ_e), 3 σ_e)`; `resp = max(0, −dog)`; `noise = 1.4826 · median(|dog − median(dog)|)`; `hi = max(0.03 − 0.02 · detail, 4 · noise)`, `lo = 0.4 · hi`; hysteresis via `cv2.connectedComponents` on `resp > lo` keeping components that contain any `resp > hi` pixel; drop components smaller than `(2·lw_px)²`; morphological close with an `lw_px`-wide ellipse; save the map to `debug_edges_path` when set; `skeleton_strokes(mask, mmpp, spur_mm=1.5·lw, smooth_mm=0.5·lw, simplify_mm=0.05·lw, min_length_mm=cfg.edge.min_length_factor·lw)`; sort longest first; `deoverlap(…, min_sep=0.9·lw, min_length=min_length_mm)`; tools: n == 1 → palette `[(30, 30, 30)]`, else `palette_kmeans(img, foreground_mask(...), n)` and each stroke takes the palette colour nearest the median of `sample_colors` along it. Tune constants on `images/` so the tests and a visual check of portraits/noisy photos both pass.

- [ ] **Step 4: Run tests** → PASS.

- [ ] **Step 5: Commit** — `git add img2gcode/styles/edge.py tests/test_style_edge.py && git commit -m "Add noise-robust edge style (flow-aligned DoG)"`

---

### Task 11: styles/contour.py

**Files:**
- Create: `img2gcode/styles/contour.py`, `tests/test_style_contour.py`

**Interfaces:**
- Consumes: `raster.*`, `geometry.smooth/simplify/deoverlap/polyline_length`, `skimage.measure.find_contours`.
- Produces: `make(rgb, cfg) -> Artwork` (`style="contour"`, kind `"line"`).

- [ ] **Step 1: Write the failing tests** — `tests/test_style_contour.py`

```python
import time

import numpy as np
import pytest
from shapely.geometry import LineString

from img2gcode.geometry import polyline_length
from img2gcode.styles.contour import make
from tests.synth import LW, assert_valid_artwork, radial_rgb


def test_contract(cfg):
    a = make(radial_rgb(), cfg)
    assert_valid_artwork(a, cfg)
    assert a.style == "contour" and a.regions == []


def test_rings_on_radial_gradient(cfg):
    cfg.contour.levels = 6
    a = make(radial_rgb(), cfg)
    closed = [s for s in a.strokes if np.allclose(s.points[0], s.points[-1])]
    assert len(closed) >= 4
    cx, cy = a.width_mm / 2, a.height_mm / 2
    for s in closed:
        c = s.points.mean(axis=0)
        assert np.hypot(c[0] - cx, c[1] - cy) < 0.1 * a.width_mm


def test_lines_keep_their_distance(cfg):
    a = make(radial_rgb(noise=20), cfg)
    lines = [LineString(s.points) for s in a.strokes]
    for i in range(len(lines)):
        for j in range(i + 1, len(lines)):
            assert lines[i].distance(lines[j]) >= 0.9 * LW - 0.1


def test_noise_robust(cfg):
    a = make(radial_rgb(noise=40), cfg)
    assert len(a.strokes) <= 3 * cfg.contour.levels
    assert all(polyline_length(s.points) >= 3 * LW for s in a.strokes)


def test_tools_follow_levels(cfg):
    cfg.tools.num_tools = 2
    a = make(radial_rgb(), cfg)
    assert {s.tool for s in a.strokes} == {0, 1}
    centre = np.array([a.width_mm / 2, a.height_mm / 2])
    inner = min(a.strokes, key=lambda s: np.linalg.norm(s.points.mean(axis=0) - centre))
    assert inner.tool == 0


def test_blank_image(cfg):
    a = make(np.full((100, 120, 3), 128, np.uint8), cfg)
    assert_valid_artwork(a, cfg, allow_empty=True)
    assert a.strokes == []


@pytest.mark.slow
def test_runtime_large(cfg):
    big = radial_rgb(1500, 2000, noise=30)
    t = time.perf_counter(); make(big, cfg)
    assert time.perf_counter() - t < 30
```

- [ ] **Step 2: Run to verify failure** → FAIL.
- [ ] **Step 3: Implement** per the spec's contour section: `working_image(…, lw/3)`, `lum = luminance(denoise(img))`, `cv2.GaussianBlur(σ = smoothing · lw_px)`, foreground = `foreground_mask` of the denoised image dilated by `lw_px`; levels = quantiles of foreground luminance at `(i + 0.5) / levels`, deduplicated (a flat image yields no levels and no strokes); `find_contours(blurred, level)` per level, mapped to page mm (`(col + 0.5)·mmpp`, `(row + 0.5)·mmpp`), split where they leave the foreground, `smooth(…, 0.5·lw)`, `simplify(…, 0.05·lw)`, drop < 3·lw; order darker levels first, longest first inside a level; `deoverlap(min_sep=0.9·lw, min_length=3·lw)`; tool of level `i` = `i · n // levels`; palette from `palette_kmeans` sorted dark → light (fallback greys when fewer colours).
- [ ] **Step 4: Run tests** → PASS.
- [ ] **Step 5: Commit** — `git add img2gcode/styles/contour.py tests/test_style_contour.py && git commit -m "Add contour style"`

---

### Task 12: styles/oneline.py

**Files:**
- Create: `img2gcode/styles/oneline.py`, `tests/test_style_oneline.py`

**Interfaces:**
- Consumes: `raster.*`, `geometry.chaikin/simplify`, `scipy.spatial.cKDTree`.
- Produces: `make(rgb, cfg) -> Artwork` (`style="oneline"`, exactly one stroke per tool that has points); `_tour(points, seconds) -> np.ndarray` (permutation of point indices: nearest-neighbour start + 2-opt, open path).

- [ ] **Step 1: Write the failing tests** — `tests/test_style_oneline.py`

```python
import time

import numpy as np
import pytest
from shapely.geometry import LineString

from img2gcode.styles.oneline import _tour, make
from tests.synth import assert_valid_artwork, radial_rgb, split_rgb, total_length


def test_contract(cfg):
    a = make(radial_rgb(), cfg)
    assert_valid_artwork(a, cfg)
    assert a.style == "oneline" and a.regions == []


def test_one_tool_one_stroke(cfg):
    cfg.tools.num_tools = 1
    a = make(radial_rgb(), cfg)
    assert len(a.strokes) == 1 and len(a.strokes[0].points) >= 200


def test_line_goes_where_it_is_dark(cfg):
    cfg.tools.num_tools = 1
    a = make(split_rgb(), cfg)
    p = a.strokes[0].points
    seg = np.hypot(*np.diff(p, axis=0).T)
    mid = (p[:-1, 0] + p[1:, 0]) / 2
    assert seg[mid < a.width_mm / 2].sum() / seg.sum() >= 0.95


def test_one_stroke_per_tool(cfg):
    cfg.tools.num_tools = 2
    a = make(split_rgb(left=(20, 20, 120), right=(220, 40, 40)), cfg)
    assert len(a.strokes) == 2 and {s.tool for s in a.strokes} == {0, 1}


def test_points_setting(cfg):
    cfg.tools.num_tools = 1; cfg.oneline.points = 300
    t = time.perf_counter(); a = make(radial_rgb(), cfg)
    assert time.perf_counter() - t < 10 and len(a.strokes[0].points) >= 300


def test_tour_has_no_crossings():
    pts = np.random.default_rng(3).random((150, 2)) * 100
    order = _tour(pts, 5.0)
    assert sorted(order.tolist()) == list(range(150))
    assert LineString(pts[order]).is_simple


def test_time_cap(cfg):
    cfg.tools.num_tools = 1; cfg.oneline.points = 5000; cfg.oneline.tsp_seconds = 1
    t = time.perf_counter(); make(radial_rgb(), cfg)
    assert time.perf_counter() - t < 15


def test_blank_image(cfg):
    a = make(np.full((100, 120, 3), 255, np.uint8), cfg)
    assert_valid_artwork(a, cfg, allow_empty=True)
    assert a.strokes == []
```

- [ ] **Step 2: Run to verify failure** → FAIL.
- [ ] **Step 3: Implement** per the spec's oneline section: darkness at `lw/2` mm per px (`(1 − luminance)^gamma`, zero where not foreground); automatic point count `ΣD · mmpp² / (2.5·lw)²` clamped to [200, 20000] (exact count when `points > 0`); seeds sampled ∝ D with `default_rng(0)`; 15 weighted Lloyd iterations (assign pixels with D > 0.01 to the nearest point via `cKDTree.query`, move points to `bincount`-weighted centroids); tools: `palette_kmeans` and each point takes the nearest palette colour of the pixel it sits on (n == 1 → palette `[(30, 30, 30)]`); `_tour` per tool: nearest-neighbour path via `cKDTree` with growing `k`, then 2-opt over 8-nearest-neighbour candidate lists with don't-look bits, reversing `order[i+1:j+1]` when `d(a,b) + d(c,d) > d(a,c) + d(b,d)`, stopping at the time cap or when no move improves; stroke = `simplify(chaikin(points[order], 2), 0.05·lw)` in page mm.
- [ ] **Step 4: Run tests** → PASS.
- [ ] **Step 5: Commit** — `git add img2gcode/styles/oneline.py tests/test_style_oneline.py && git commit -m "Add oneline style (stippling + 2-opt tour)"`

---

### Task 13: styles/squiggle.py

**Files:**
- Create: `img2gcode/styles/squiggle.py`, `tests/test_style_squiggle.py`

**Interfaces:**
- Consumes: `raster.*`, `geometry.simplify`.
- Produces: `make(rgb, cfg) -> Artwork` (`style="squiggle"`); row `k` centreline at `y = (k + 0.5) · row_spacing · lw`.

- [ ] **Step 1: Write the failing tests** — `tests/test_style_squiggle.py`

```python
import numpy as np

from img2gcode.styles.squiggle import make
from tests.synth import LW, assert_valid_artwork, radial_rgb, split_rgb


def test_contract(cfg):
    a = make(radial_rgb(), cfg)
    assert_valid_artwork(a, cfg)
    assert a.style == "squiggle" and a.regions == []


def test_rows_never_overlap(cfg):
    a = make(radial_rgb(), cfg)
    sp = cfg.squiggle.row_spacing * LW
    for s in a.strokes:
        y = s.points[:, 1]
        off = np.abs(y - (np.floor(y / sp) + 0.5) * sp)
        assert off.max() <= (sp - LW) / 2 + 1e-6


def test_dark_side_has_more_ink(cfg):
    a = make(split_rgb(left=(20, 20, 20), right=(150, 150, 150)), cfg)
    left = right = 0.0
    for s in a.strokes:
        p = s.points
        seg = np.hypot(*np.diff(p, axis=0).T)
        mid = (p[:-1, 0] + p[1:, 0]) / 2
        left += seg[mid < a.width_mm / 2].sum(); right += seg[mid >= a.width_mm / 2].sum()
    assert left > 1.5 * right


def test_white_background_breaks_rows(cfg):
    img = np.zeros((400, 900, 3), np.uint8); img[:, 300:600] = 255
    a = make(img, cfg)
    x0, x1 = a.width_mm / 3 + 2 * LW, 2 * a.width_mm / 3 - 2 * LW
    for s in a.strokes:
        assert not np.any((s.points[:, 0] > x0) & (s.points[:, 0] < x1))


def test_multi_tool(cfg):
    cfg.tools.num_tools = 2
    a = make(split_rgb(left=(20, 20, 120), right=(220, 40, 40)), cfg)
    assert {s.tool for s in a.strokes} == {0, 1}


def test_blank_image(cfg):
    a = make(np.full((100, 120, 3), 255, np.uint8), cfg)
    assert_valid_artwork(a, cfg, allow_empty=True)
    assert a.strokes == []
```

- [ ] **Step 2: Run to verify failure** → FAIL.
- [ ] **Step 3: Implement** per the spec's squiggle section: spacing `sp = row_spacing · lw`; darkness `1 − luminance` of the image blurred with σ = sp/2 (at `lw/4` mm per px); samples every `lw/4` along each row; `f_max = max_frequency / (2.5·lw)`, `f(x) = f_max · D(x)` (D ∈ [0,1]), phase `φ += 2π f dx`; `y = y_row + A(x) · sin φ` with `A = D · (sp − lw)/2 · 0.95`; break rows wherever the foreground (`foreground_mask`, white threshold) is absent for more than `2·lw`; tools: n > 1 → nearest palette colour per sample, median-filtered over `4·lw`, rows split where the tool changes; serpentine: alternate row direction and join a piece to the previous piece of the same tool when the end-to-start gap ≤ `1.5·sp`; `simplify(…, 0.02·lw)`.
- [ ] **Step 4: Run tests** → PASS.
- [ ] **Step 5: Commit** — `git add img2gcode/styles/squiggle.py tests/test_style_squiggle.py && git commit -m "Add squiggle style"`

---

### Task 14: styles/hatch.py

**Files:**
- Create: `img2gcode/styles/hatch.py`, `tests/test_style_hatch.py`

**Interfaces:**
- Consumes: `raster.*`, `geometry.hatch_lines`.
- Produces: `make(rgb, cfg) -> Artwork` (`style="hatch"`, every stroke a 2-point line, kind `"line"`).

- [ ] **Step 1: Write the failing tests** — `tests/test_style_hatch.py`

```python
import numpy as np

from img2gcode.styles.hatch import make
from tests.synth import LW, assert_valid_artwork, bands_rgb, radial_rgb


def angle_of(s):
    d = s.points[-1] - s.points[0]
    return round(float(np.degrees(np.arctan2(d[1], d[0]))) % 180.0 / 5.0) * 5 % 180


def test_contract(cfg):
    a = make(radial_rgb(), cfg)
    assert_valid_artwork(a, cfg)
    assert a.style == "hatch" and a.regions == []


def test_darker_bands_get_more_directions(cfg):
    a = make(bands_rgb(), cfg)
    bw = a.width_mm / 4
    counts = []
    for b in range(4):
        lo, hi = b * bw + 2 * LW, (b + 1) * bw - 2 * LW
        mids = [s for s in a.strokes if lo < s.points.mean(axis=0)[0] < hi]
        counts.append(len({angle_of(s) for s in mids}))
    assert counts == [3, 2, 1, 0]


def test_strokes_are_straight_segments(cfg):
    assert all(len(s.points) == 2 for s in make(radial_rgb(), cfg).strokes)


def test_spacing(cfg):
    cfg.hatch.angles = "0"; cfg.hatch.bands = 2
    a = make(bands_rgb((20, 200)), cfg)
    ys = np.unique(np.round([s.points[0, 1] for s in a.strokes], 3))
    assert np.allclose(np.diff(ys), cfg.hatch.spacing * LW, atol=0.05)


def test_tools_by_direction(cfg):
    cfg.tools.num_tools = 2
    assert {s.tool for s in make(bands_rgb(), cfg).strokes} == {0, 1}


def test_noise_robust(cfg):
    clean = len(make(radial_rgb(), cfg).strokes)
    noisy = len(make(radial_rgb(noise=40), cfg).strokes)
    assert noisy <= 2 * clean


def test_blank_image(cfg):
    a = make(np.full((100, 120, 3), 255, np.uint8), cfg)
    assert_valid_artwork(a, cfg, allow_empty=True)
    assert a.strokes == []
```

- [ ] **Step 2: Run to verify failure** → FAIL.
- [ ] **Step 3: Implement** per the spec's hatch section: `working_image(…, lw/3)`, `lum = luminance(denoise(img))` blurred with σ = `lw_px`; `fg = foreground_mask`; `t = tone_levels(lum, fg, bands)` (ascending); `angles = [float(a) for a in cfg.hatch.angles.split(",") if a.strip()]`; for `a` in `range(min(bands − 1, len(angles)))`: mask = `fg & (lum <= t[bands − 2 − a])` (a = 0 is the most inclusive mask), `geom = mask_to_geometry(mask, mmpp, simplify_mm=0.1·lw, min_area_mm2=(1.5·lw)²)`, lines = `hatch_lines(geom, cfg.hatch.spacing · lw, angles[a], connect=False)`, tool `a % n`; palette from `palette_kmeans` (dark → light) padded with greys to `n` entries.
- [ ] **Step 4: Run tests** → PASS.
- [ ] **Step 5: Commit** — `git add img2gcode/styles/hatch.py tests/test_style_hatch.py && git commit -m "Add hatch style"`

---

### Task 15: pipeline, CLI, viewer, ROI selector; remove old modules

**Files:**
- Create: `img2gcode/pipeline.py`, `tests/test_cli.py`
- Modify: `img2gcode/__main__.py` (rewrite), `img2gcode/gcode_parser.py` (add `parse_gcode_lines`), `img2gcode/gui.py` (remove the duplicate `slider.on_changed(on_slider)` registration so each move redraws once), `img2gcode/roi_selector.py` (rewrite for `Artwork`)
- Delete: `img2gcode/toolpath.py`, `img2gcode/edge_pipeline.py`, `img2gcode/segmenter.py`, `img2gcode/svg_loader.py`

**Interfaces:**
- Consumes: everything above.
- Produces:
  - `pipeline.Result(toolpaths, svg_path, gcode_path, stats, ignored_shapes)` dataclass
  - `pipeline.make_artwork(input_path, cfg, n_tools_explicit=False) -> Optional[Artwork]` (None for a toolpaths SVG)
  - `pipeline.convert(input_path, output_path, cfg, *, svg_path=None, svg_only=False, exclusion_zones=None, n_tools_explicit=False, artwork=None) -> Result`
  - `gcode_parser.parse_gcode_lines(lines) -> (segments, meta)`; `parse_gcode(path)` delegates to it
  - `__main__.load_segments(path, cfg) -> (segments, meta)` for `.gcode` or toolpaths `.svg`
  - `roi_selector.select_exclusion_zones(art) -> list[tuple[float, float, float, float]]` (page mm)

- [ ] **Step 1: Write the failing tests** — `tests/test_cli.py`

```python
import importlib.util
from pathlib import Path

import cv2
import numpy as np
import pytest

from img2gcode.__main__ import load_segments, main
from img2gcode.gcode_parser import parse_gcode
from img2gcode.pipeline import convert
from img2gcode.styles import STYLES
from img2gcode.svg_io import is_toolpaths_svg
from tests.synth import shapes_rgb

LOGO = Path(__file__).resolve().parent.parent / "images" / "python_logo.png"


def png(tmp_path, img, name="in.png"):
    p = tmp_path / name
    cv2.imwrite(str(p), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    return p


def body(path):
    return Path(path).read_text(encoding="utf-8").splitlines()[1:]


def test_png_to_svg_and_gcode(cfg, tmp_path):
    res = convert(LOGO, tmp_path / "out.gcode", cfg)
    assert res.gcode_path == tmp_path / "out.gcode" and res.gcode_path.exists()
    assert res.svg_path == tmp_path / "out.svg" and is_toolpaths_svg(res.svg_path)
    segs, meta = parse_gcode(res.gcode_path)
    assert any(s["type"] == "print" for s in segs) and meta["total_time_min"] > 0


def test_svg_output_only(cfg, tmp_path):
    res = convert(LOGO, tmp_path / "x.svg", cfg)
    assert res.gcode_path is None and (tmp_path / "x.svg").exists() and not (tmp_path / "x.gcode").exists()


def test_svg_only_flag(cfg, tmp_path):
    res = convert(LOGO, tmp_path / "x.gcode", cfg, svg_only=True)
    assert (tmp_path / "x.svg").exists() and not (tmp_path / "x.gcode").exists() and res.gcode_path is None


def test_toolpaths_svg_reconverts_identically(cfg, tmp_path):
    a = convert(LOGO, tmp_path / "a.gcode", cfg)
    b = convert(a.svg_path, tmp_path / "b.gcode", cfg)
    assert body(a.gcode_path) == body(b.gcode_path)
    assert not (tmp_path / "b.svg").exists()


def test_artwork_svg_input(cfg, tmp_path):
    p = tmp_path / "art.svg"
    p.write_text('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100">'
                 '<rect width="40" height="40" fill="#ff0000"/><rect x="50" width="40" height="40" fill="#0000ff"/></svg>',
                 encoding="utf-8")
    res = convert(p, tmp_path / "art.gcode", cfg)
    assert "; tools: 2  layers: 1" in res.gcode_path.read_text(encoding="utf-8")


@pytest.mark.parametrize("style", STYLES)
def test_every_style_end_to_end(cfg, tmp_path, style):
    cfg.style = style
    src = png(tmp_path, cv2.resize(shapes_rgb(noise=20), (400, 300)))
    res = convert(src, tmp_path / "o.gcode", cfg)
    segs, _ = parse_gcode(res.gcode_path)
    assert any(s["type"] == "print" for s in segs)


@pytest.mark.parametrize("style", STYLES)
def test_blank_image_gives_valid_empty_gcode(cfg, tmp_path, style):
    cfg.style = style
    res = convert(png(tmp_path, np.full((60, 80, 3), 255, np.uint8)), tmp_path / "o.gcode", cfg)
    segs, _ = parse_gcode(res.gcode_path)
    assert not any(s["type"] == "print" for s in segs)
    assert "; End of print" in res.gcode_path.read_text(encoding="utf-8")


@pytest.mark.parametrize("style", STYLES)
def test_tiny_image(cfg, tmp_path, style):
    cfg.style = style
    convert(png(tmp_path, cv2.resize(shapes_rgb(), (32, 24), interpolation=cv2.INTER_AREA)), tmp_path / "o.gcode", cfg)


def test_output_dirs_created(cfg, tmp_path):
    res = convert(LOGO, tmp_path / "a" / "b" / "o.gcode", cfg)
    assert res.gcode_path.exists() and res.svg_path.exists()


def test_cli_headless(tmp_path):
    out = tmp_path / "c.gcode"
    main(["-i", str(LOGO), "-o", str(out), "--headless", "-q", "--style", "contour"])
    assert out.exists() and 'data-style="contour"' in (tmp_path / "c.svg").read_text(encoding="utf-8")


def test_cli_mode_alias_and_svg_path(tmp_path):
    out = tmp_path / "m.gcode"; svg = tmp_path / "other" / "m.svg"
    main(["-i", str(LOGO), "-o", str(out), "--headless", "-q", "--mode", "lines", "--svg", str(svg)])
    assert out.exists() and 'data-style="edge"' in svg.read_text(encoding="utf-8")


def test_cli_rejects_unknown_style(tmp_path):
    with pytest.raises(SystemExit):
        main(["-i", str(LOGO), "-o", str(tmp_path / "x.gcode"), "--headless", "--style", "nope"])


def test_load_segments_for_svg_and_gcode(cfg, tmp_path):
    res = convert(LOGO, tmp_path / "v.gcode", cfg)
    for path in (res.svg_path, res.gcode_path):
        segs, meta = load_segments(path, cfg)
        assert any(s["type"] == "print" for s in segs)


def test_old_modules_removed():
    for m in ("toolpath", "edge_pipeline", "segmenter", "svg_loader"):
        assert importlib.util.find_spec(f"img2gcode.{m}") is None
```

- [ ] **Step 2: Run to verify failure** → FAIL.

- [ ] **Step 3: Implement**

`pipeline.py` per the spec's CLI section: output ending in `.svg` → SVG only at that path; otherwise the SVG goes to `svg_path` or `output.with_suffix(".svg")`. Toolpaths-SVG input: read it, write no SVG (unless SVG-only output to a different path, then write a copy), write G-code. Everything else: `make_artwork` → `plan` → `write_toolpaths_svg` → `read_toolpaths_svg` of that file → `write_gcode`. Create parent directories for both outputs.

`__main__.py` flags: `-i/--input`, `-o/--output`, `-c/--config`, `--style` and `--mode` (both `choices=list(STYLES) + list(ALIASES)`), `--n-tools`, `--svg`, `--svg-only`, `--select-roi`, `--debug-edges`, `--headless`, `--visualise FILE`, `-v/--verbose` (kept, no-op), `-q/--quiet`. Validate the config's `[mode] type` with `resolve_style` (exit with a clear message if unknown). With `--select-roi`, build the artwork first, open `select_exclusion_zones(art)` when it has regions (otherwise print that ROI needs filled regions), and pass both into `convert`. Unless `-q`, print `"[img2gcode] SVG written → {svg}"`, `"[img2gcode] GCode written → {gcode}  ({n} lines)"` and the old `Estimated: …` line, plus `"[img2gcode] warning: {n} shape(s) outside tool layers were ignored"` and a warning when the SVG's line width differs from the config's. Unless `--headless`, `launch_gui(*load_segments(target, cfg), title=…)`. `load_segments`: `.svg` → `read_toolpaths_svg` → `gcode_lines` → `parse_gcode_lines`; otherwise `parse_gcode`.

`roi_selector.py`: same widgets as today (RectangleSelector, Undo, Done, fixed button axes), but draw `art.underlay` with `extent=(0, art.width_mm, art.height_mm, 0)` (or a white background), overlay every region's rings in its palette colour, and return rectangles in page mm.

Delete the four old modules with `git rm`.

- [ ] **Step 4: Run the whole suite** — `.venv/Scripts/python.exe -m pytest -q` → PASS. Then `.venv/Scripts/python.exe -m img2gcode -i images/python_logo.png -o out/logo.gcode --headless` and open `out/logo.svg` in a browser to eyeball it.

- [ ] **Step 5: Commit** — `git add img2gcode tests/test_cli.py && git commit -m "Wire pipeline and CLI through toolpaths.svg; remove old modules"` (the `git rm` deletions are already staged).

---

### Task 16: README and style gallery script

**Files:**
- Modify: `README.md` (rewrite)
- Create: `scripts/style_gallery.py`, `tests/test_gallery.py`

**Interfaces:**
- Consumes: `pipeline.convert`, `svg_io.read_toolpaths_svg`, `STYLES`.
- Produces: `scripts/style_gallery.py` with `main(argv=None)`: `python scripts/style_gallery.py [IMAGES...] --out DIR [--styles fill,edge,...] [-c CFG]`; for each style writes `DIR/<style>/<image>.svg`, `DIR/<style>/<image>.png` (toolpaths rendered with `cv2.polylines` at the true line width on white, ~4 px per mm capped at 1200 px wide) and `DIR/<style>_sheet.jpg` (labelled grid: source thumbnail next to its rendering), plus `DIR/timings.csv` (`style,image,seconds,strokes,print_mm`).

- [ ] **Step 1: Write the failing test** — `tests/test_gallery.py`

```python
import importlib.util
from pathlib import Path

import cv2

from tests.synth import shapes_rgb

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "style_gallery.py"


def test_gallery_writes_sheet_and_timings(tmp_path):
    src = tmp_path / "s.png"
    cv2.imwrite(str(src), cv2.resize(shapes_rgb(), (200, 150)))
    spec = importlib.util.spec_from_file_location("style_gallery", SCRIPT)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    mod.main([str(src), "--out", str(tmp_path / "g"), "--styles", "fill,contour"])
    assert (tmp_path / "g" / "fill_sheet.jpg").exists() and (tmp_path / "g" / "contour_sheet.jpg").exists()
    assert len((tmp_path / "g" / "timings.csv").read_text(encoding="utf-8").splitlines()) == 3
```

- [ ] **Step 2: Run to verify failure** → FAIL.
- [ ] **Step 3: Implement** the script (headless, `cfg.style` set per style, errors for one image are reported in the CSV with `seconds = -1` and do not stop the run). Rewrite `README.md`: what it does (image → toolpaths.svg → G-code), installation, usage examples for each style, the SVG workflow (inspect in a browser, edit in Inkscape: tool/layer groups, hidden layers do not print, only shapes inside tool layers print, then `img2gcode -i edited.svg -o out.gcode`), CLI flag table, config sections with the style keys, tuning tips per style, test images (`python scripts/fetch_test_images.py`), gallery script, running tests. Keep the MIT licence line.
- [ ] **Step 4: Run tests** → PASS.
- [ ] **Step 5: Commit** — `git add README.md scripts/style_gallery.py tests/test_gallery.py && git commit -m "Rewrite README and add style gallery script"`

---

### Task 17: Visual QA and default tuning

**Files:** may modify style modules, `config.py` + `configs/default.cfg` defaults (keep them equal), and their tests.

- [ ] **Step 1:** `python scripts/fetch_test_images.py` (if `images/` is incomplete), then `python scripts/style_gallery.py images/* --out <scratch>/gallery`.
- [ ] **Step 2:** Review every `<style>_sheet.jpg` against its sources. Record per style: images that look wrong (noise speckle, missing structure, broken strokes, overlapping lines, empty output), and the slowest runs from `timings.csv`.
- [ ] **Step 3:** Fix causes in the style code or tune defaults; re-run the gallery for the affected style; repeat until the sheets look right and every run is < 30 s.
- [ ] **Step 4:** Full suite including slow tests: `.venv/Scripts/python.exe -m pytest -q -m "slow or not slow"` → PASS.
- [ ] **Step 5:** Commit — `git commit -m "Tune style defaults on the test image set"` with only the files changed.
