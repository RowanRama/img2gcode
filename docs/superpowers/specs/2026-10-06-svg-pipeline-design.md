# img2gcode rework: image → toolpaths.svg → G-code

Date: 2026-10-06 · Branch: `rework/svg-pipeline` · Status: approved (option A, all four new styles, build straight through)

Design page with diagrams and before/after images: https://claude.ai/artifact/Q5TcALGK6VizokUHrfncpe

## Goal

1. Every conversion first writes an SVG holding the exact toolpaths, then converts that SVG to G-code. The SVG is
   viewable in a browser and editable in Inkscape; the G-code is a straight serialization of it.
2. Line-drawing output must survive noisy photos (today, edge mode turns noise into thousands of dashes).
3. Add four abstract drawing styles: `contour`, `oneline`, `squiggle`, `hatch`.
4. The G-code format stays the same: the new writer, given the same moves, produces the same lines as the
   writer at commit `d77193c` (pinned by `tests/data/golden_writer.gcode`).

Non-goals: new G-code dialects, a new GUI toolkit, 3-D relief generation, configurable per-region infill.

## Pipeline

```
raster ─► style ─► Artwork ─┐
                            ├─► planner ─► Toolpaths ─► toolpaths.svg ─► read back ─► Toolpaths ─► writer ─► .gcode
artwork SVG ─► svg_input ───┘
toolpaths SVG ─────────────────────────────────────────► read ─────────────► Toolpaths ─► writer ─► .gcode
```

The CLI always writes the SVG and then reads it back to make the G-code, so the file on disk and the G-code can
never disagree.

## Units and coordinates

- All geometry after a style runs is in **millimetres on the page**: origin top-left, x right, y down (SVG
  convention). Page size = the source fitted into `print_width × print_height`, keeping aspect ratio
  (`mm_per_src_px = min(print_width / W, print_height / H)`), exactly as today.
- Raster pixel `(col, row)` centre maps to `((col + 0.5) · mm_per_px, (row + 0.5) · mm_per_px)`.
- The writer maps page → machine as `X = x − width/2`, `Y = height/2 − y` (centred on 0,0, Y up). Today's extra
  one-pixel Y offset (`H − 1 − y`) is dropped. The golden test feeds `y_page = (y_px + 1) · scale` to reproduce the
  old numbers exactly.
- `lw` below means `cfg.extrusion.effective_line_width` in mm.

## Data model (`img2gcode/model.py`)

```python
@dataclass
class Stroke:
    points: np.ndarray            # (N, 2) float64, page mm; closed if first == last
    tool: int = 0
    kind: str = "line"            # "perimeter" | "infill" | "line"

@dataclass
class Region:
    shape: BaseGeometry           # shapely Polygon / MultiPolygon, page mm
    tool: int

@dataclass
class Artwork:
    width_mm: float
    height_mm: float
    regions: list[Region]
    strokes: list[Stroke]
    palette: list[tuple[int, int, int]]   # RGB per tool index; len == number of tools
    source_px: tuple[int, int]            # (W, H) of the source image or SVG viewBox, for the G-code header
    underlay: np.ndarray | None = None    # RGB image shown faintly under the toolpaths in the SVG
    style: str = ""

@dataclass
class ToolPass:                           # one tool on one layer; today's ToolLayer
    tool: int
    layer: int
    strokes: list[Stroke]                 # print order; a travel move precedes every stroke

@dataclass
class Toolpaths:
    width_mm: float
    height_mm: float
    line_width_mm: float
    n_layers: int
    passes: list[ToolPass]                # every (layer, tool) pair, layer-major, tool-minor, empties included
    palette: list[tuple[int, int, int]]
    source_px: tuple[int, int]
    underlay: np.ndarray | None = None
    style: str = ""
```

## Modules

| Module | Job |
|---|---|
| `__init__.py` | `cv2.setNumThreads(1)` at import (OpenCV's thread pool raises "Unknown C++ exception" on Windows once SciPy is loaded). |
| `config.py` | Dataclass per section; one generic loader that reads every field by its annotated type, falls back to the dataclass default, ignores unknown keys. Dataclass defaults are the single source of truth; `configs/default.cfg` documents them with the user's machine values. |
| `model.py` | Types above. |
| `geometry.py` | Pure numpy/shapely: polyline length, resample, smooth (moving average over arc length, endpoints and closure kept), Douglas-Peucker simplify, `rings_to_geometry(rings, rule)` (nonzero/evenodd via face winding), `hatch_lines(geom, spacing, angle, connect)`, `order_strokes(strokes, start)`, `chain_strokes`, `deoverlap(strokes, …)`. |
| `raster.py` | Image I/O and pixel work: `load_rgb`, `fit_page`, `working_image`, `estimate_noise` (Immerkær), `denoise`, `foreground_mask`, `palette_kmeans`, `mask_to_geometry`, `tone_levels`, `sample_colors`. |
| `skeleton.py` | `skeleton_strokes(mask, mm_per_px, …)`: skeletonize a binary mask and trace it into smooth strokes (graph walk, spur pruning, straight-through joining at junctions). |
| `styles/` | `fill`, `edge`, `contour`, `oneline`, `squiggle`, `hatch`. Each exposes `make(rgb, cfg) -> Artwork`. `styles/__init__.py` holds the `STYLES` registry and `ALIASES = {"lines": "edge"}`. |
| `svg_input.py` | Any non-toolpath SVG → `Artwork`. |
| `planner.py` | `Artwork` → `Toolpaths`: perimeters, thin-feature centrelines, infill, ordering, layers. |
| `svg_io.py` | `write_toolpaths_svg(tp, path)`, `read_toolpaths_svg(path) -> Toolpaths`, `is_toolpaths_svg(path)`. |
| `writer.py` | `Toolpaths` → G-code, format unchanged. |
| `gcode_parser.py`, `gui.py` | Kept. GUI: remove the duplicate slider handler. `--visualise` accepts a toolpaths SVG too. |
| `roi_selector.py` | Adapted: shows the underlay and region outlines in page mm, returns rectangles in page mm. |
| `pipeline.py` | `convert(input_path, output_path, cfg, *, svg_path=None, svg_only=False, exclusion_zones=None, n_tools_explicit=False) -> Result` (paths written, toolpaths, warnings). The CLI and the end-to-end tests both call it. |
| `__main__.py` | CLI: parse args, apply overrides to `cfg`, optional ROI selection, call `pipeline`, open the viewer. |

Removed: `toolpath.py`, `edge_pipeline.py`, `segmenter.py`, `svg_loader.py` (their useful parts move
into the modules above). Dependency changes: add `shapely>=2.0`, `scikit-image>=0.22` (brings SciPy); drop
`scikit-learn` (OpenCV k-means replaces it).

## Writer

Port today's `GCodeWriter.write` and `compute_stats` loop line for line. Each `ToolPass` becomes today's move list:
a travel to `stroke.points[0]`, then extrusions to the remaining points. Unchanged:

- Header lines and their wording, including `; image size: {W}x{H}px → {w:.1f}x{h:.1f}mm (scale={s:.4f})` where
  `(W, H) = source_px` and `s = width_mm / W`, and the `estimated_print_time` / `print_distance` / `filament` lines the
  parser reads. `; tools:` uses the number of tools in the toolpaths.
- Startup and end blocks, `;LAYER_CHANGE` / `;Z:` / `;HEIGHT:` / `G92 E0` for every layer (even empty ones), per
  non-empty pass `; --- Tool N ---` / `TN` / `G92 E0` on tool change, the single `;TYPE:` line chosen by
  `perimeter_loops > 0`, Z-lift travel, `G1 F{print}` after lowering, `E` accumulated per pass, `G92 E0` after each
  pass, coordinates rounded to 4 decimals, extrusion `E{:.5f}`.
- Stats skip the first move of each pass, as today.

Fix: write the file as UTF-8 (`write_text(..., encoding="utf-8")`); keep platform newlines.

## toolpaths.svg

```xml
<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink"
     xmlns:inkscape="http://www.inkscape.org/namespaces/inkscape"
     xmlns:sodipodi="http://sodipodi.sourceforge.net/DTD/sodipodi-0.dtd"
     width="{w}mm" height="{h}mm" viewBox="0 0 {w} {h}"
     data-img2gcode="toolpaths" data-line-width="{lw}" data-layers="{n}" data-tools="{t}"
     data-source-px="{W}x{H}" data-style="{style}">
  <rect data-role="background" width="{w}" height="{h}" fill="#ffffff"/>
  <g inkscape:groupmode="layer" inkscape:label="Source image" sodipodi:insensitive="true"
     data-role="source" opacity="0.25">
    <image x="0" y="0" width="{w}" height="{h}" preserveAspectRatio="none" xlink:href="data:image/jpeg;base64,…"/>
  </g>
  <g inkscape:groupmode="layer" inkscape:label="Layer 0" data-layer="0">
    <g inkscape:groupmode="layer" inkscape:label="Layer 0 · T0" data-tool="0"
       fill="none" stroke="#rrggbb" stroke-width="{lw}" stroke-linecap="round" stroke-linejoin="round">
      <polyline data-kind="perimeter" points="x,y x,y …"/>
    </g>
  </g>
  <g inkscape:groupmode="layer" inkscape:label="Travel moves" data-role="travel" style="display:none"
     fill="none" stroke="#888888" stroke-width="{lw/6}">
    <path d="M… L… M… L…"/>
  </g>
</svg>
```

- Coordinates written with 3 decimals. Underlay: the source downscaled to ≤ 1200 px, JPEG quality 80; omitted when
  `underlay is None`.
- Every `(layer, tool)` group is written even when empty, so the reader recovers layer and tool counts.
- Reading (`xml.etree` walk + `svgelements` for `d` attributes and transforms):
  - Only shapes inside a `data-tool` group inside a `data-layer` group print, in document order.
  - `polyline`, `polygon` (closed), `line`, `path` are accepted; path curves are sampled at ≤ lw/8 spacing.
  - `transform` attributes on any ancestor or the element are applied (Inkscape adds them when you move things).
  - Elements or groups with `display:none` (style or attribute) are skipped: hiding a layer in Inkscape removes it
    from the print. `data-role` groups (source, travel, background) are ignored.
  - Shapes outside tool groups are ignored; the reader returns a count and the CLI prints a warning.
  - Palette comes from each tool group's `stroke`; kind from `data-kind` (default `line`).
- `write → read` reproduces every stroke to within 0.001 mm.

## Artwork SVG input (`svg_input.py`)

- Fit the viewBox (or width/height) into the print area like an image.
- Walk shapes in paint order with `svgelements` (`reify=True`). Fill: build the shape's polygon from its subpaths
  sampled at ≤ 0.25 mm with `rings_to_geometry` using the shape's `fill-rule`. Paint order means occlusion: each new
  filled shape is subtracted from every earlier region.
- Stroke: if stroke width on the page ≥ 1.5 · lw it becomes a filled region (shapely `buffer(width/2)` of the
  centreline, round caps/joins); otherwise a centreline `Stroke` of kind `line`. A shape with both gets both.
- Gradient fills use the mean of their stop colours; unresolvable paints fall back to mid-grey.
- Colours with all channels ≥ `image.white_threshold` are paper: they occlude but do not print.
- Tools: one per distinct remaining colour (today's behaviour); colours closer than ΔE 5 in Lab count as one. If `--n-tools` was given explicitly, merge colours
  down to that many with area-weighted k-means in Lab. Palette sorted dark → light.
- `underlay`: none (no raster available).

## Planner (`planner.py`)

`plan(art: Artwork, cfg, exclusion_zones=None) -> Toolpaths`, exclusion zones are page-mm rectangles.

Per tool `t` (geometry computed once, reused for every layer):

- `R` = union of tool `t`'s regions, `buffer(0)`-cleaned.
- **Perimeters**: loop `k` follows the boundary (exterior and holes) of `R.buffer(−(k + 0.5)·lw)`. The bead's
  outer edge sits on the region edge, so neighbouring tools no longer overlap (replaces today's dilation hack).
- **Thin features**: `residue = R − R.buffer(−lw/2).buffer(lw/2)`. Rasterize residue at lw/4 per px, take
  `skeleton_strokes`, keep strokes ≥ lw long as kind `line`. Text strokes and hairlines narrower than the nozzle
  stay visible instead of vanishing.
- **Infill**: `inner = R.buffer(−(loops·lw + lw/2) + 0.15·lw)` when `loops > 0`, else `R.buffer(−lw/2)`, minus
  exclusion rectangles. `spacing = infill.line_spacing` if > 0 else `lw / (density/100)`. Angle = `infill.angle`,
  +90° on odd layers. `hatch_lines(inner, spacing, angle, connect=infill.connect_lines)`; connected mode joins
  consecutive scan segments when the connector stays inside `inner` (0.05·lw tolerance) and is ≤ 2.5·spacing.
- **Line strokes**: the artwork's strokes for tool `t`, kind `line`.
- **Order inside a pass**: perimeters, then line strokes, then infill; each group ordered by `order_strokes` from the
  current cursor (greedy nearest end; open strokes may reverse; closed strokes may start at any of ≤ 32 evenly
  spaced vertices and are rotated to start there).
- Every layer repeats the same geometry (today, edge mode printed only layer 0; now all styles build height the same
  way). Strokes shorter than 0.25·lw are dropped.

## Styles

All raster styles start from `load_rgb` (alpha composited onto white), work on a resampled copy at a fixed
pixels-per-line-width (`working_image`; INTER_AREA down, INTER_CUBIC up; longest side capped at 2500 px), and
return `Artwork` with `underlay = rgb`. Palettes come from `palette_kmeans` (Lab k-means on foreground pixels,
fixed seed, sorted dark → light). `n = cfg.tools.num_tools`. Each style reads its own config section; every
section has at most six keys. Defaults below are starting points; tune them on `images/`.

### fill (reworked)
- Work at lw/4 mm per px. Denoise to match the measured noise (`denoise`: non-local means, h ≈ 1.1·σ when
  σ > 1.5), then `cv2.pyrMeanShiftFiltering(sp ≈ lw_px, sr ≈ 18)` to flatten texture.
- Foreground = not near-white (`image.white_threshold`) after smoothing. k-means (n clusters) in Lab.
- Majority smoothing: blur each label's one-hot map (σ = 0.5·lw_px), take argmax.
- Minimum feature: components smaller than `(fill.min_feature_factor · lw)²` mm² (default factor 1.5), both colour
  blobs and background holes, are reassigned to the nearest surviving label (distance-transform fill). No holes are
  created by cleanup.
- Regions per tool via `mask_to_geometry` (marching squares at 0.5, evenodd, simplify 0.1·lw).
- Keys: `min_feature_factor`, `smoothing`.

### edge (reworked)
- Work at lw/3 mm per px. Luminance; CLAHE (`edge.contrast`, default 1.5; 0 disables); denoise to measured noise.
- Flow-aligned DoG: structure-tensor tangent field (σ_t ≈ 2·σ_e), DoG `G(σ_e) − τ·G(1.6·σ_e)` with τ = 0.98,
  smoothed along the tangent field by line-integral convolution over ±3·σ_e. `σ_e` scales with `detail`
  (0 … 1, default 0.5; higher = finer features).
- Dark-side response `max(0, −DoG)`. Hysteresis with **absolute** thresholds: `hi = max(base(detail), 3·σ_resid)`
  in normalized luminance, `lo = 0.4·hi` (σ_resid = noise left after denoising). No percentile thresholds.
- Drop components smaller than `(2·lw)²`; close with an lw-wide disk so near-parallel lines merge; skeletonize.
- `skeleton_strokes`: graph trace, prune spurs < 1.5·lw, join edges through junctions by straightest continuation
  (turn < 60°), smooth over 0.5·lw, simplify 0.05·lw, drop strokes < `edge.min_length_factor`·lw (default 2).
- `deoverlap` at 0.9·lw, longest strokes first.
- Tools: n == 1 → all T0 with a near-black palette entry. n > 1 → palette from the image; each stroke takes the
  palette colour nearest its median sampled colour.
- `debug_edges_path` (and `--debug-edges`) saves the binary line map before skeletonization.
- Keys: `detail`, `contrast`, `min_length_factor`, `debug_edges_path`.

### contour (new)
- Work at lw/3. Luminance, denoise, Gaussian blur σ = `contour.smoothing` · lw (default 1.5).
- `contour.levels` (default 8) iso-levels at evenly spaced quantiles of foreground luminance.
- `skimage.measure.find_contours` per level → strokes (closed when the ends meet), smooth, simplify, drop
  < 3·lw, clip away parts lying in the white background, `deoverlap` at 0.9·lw.
- Tools: level `i` (dark → light) → tool `i · n // levels`.
- Keys: `levels`, `smoothing`.

### oneline (new)
- Darkness `D = (1 − L)^gamma` at lw/2 mm per px, zero on background; `oneline.gamma` default 1.0.
- Point count: `oneline.points` (0 = auto: `ΣD · px_area / (spacing_factor·lw)²`, spacing factor 2.5, clamped to
  200 … 20000).
- Weighted Voronoi stippling: seed ∝ D (seed 0), 15 Lloyd iterations with `scipy.spatial.cKDTree`.
- Tour: nearest-neighbour start, then 2-opt with 8-nearest neighbour lists and don't-look bits, capped at
  `oneline.tsp_seconds` (default 15). Chaikin-smooth twice, simplify.
- Tools: n == 1 → one stroke. n > 1 → each point takes the nearest palette colour; one tour per tool.
- Keys: `points`, `gamma`, `tsp_seconds`.

### squiggle (new)
- Horizontal rows `squiggle.row_spacing` · lw apart (default 3.0). Darkness sampled every lw/4 from the image
  blurred with σ = spacing/2.
- Phase accumulation: frequency rises with darkness up to a minimum wavelength of 2.5·lw; amplitude
  `D · (spacing − lw)/2 · 0.95`, so neighbouring rows never touch.
- Rows break where the background (white) runs longer than 2·lw. Consecutive row pieces join into one serpentine
  stroke when the end-to-start gap ≤ 1.5·spacing.
- Tools: n > 1 → nearest palette colour per sample, median-filtered over 4·lw, rows split where the tool changes.
- Keys: `row_spacing`, `max_frequency` (as a fraction of the 2.5·lw limit, default 1.0).

### hatch (new)
- Work at lw/3. Luminance, denoise, blur σ = lw. `hatch.bands` (default 4) tone levels by foreground quantile.
- Nested masks `mask_j = L ≤ level_j` (j = 1 … bands−1, lighter → darker). Mask `j` is hatched at `angles[j−1]`
  (`hatch.angles` default `45,-45,0`), spacing `hatch.spacing` · lw (default 2.5), so the darkest areas get every
  direction. Masks → geometry via `mask_to_geometry` (min feature 1.5·lw) → `hatch_lines(connect=False)`.
- Tools: direction `j` → tool `j % n`.
- Keys: `bands`, `angles`, `spacing`.

## CLI

```
img2gcode -i INPUT -o OUTPUT [-c CFG] [--style NAME] [--mode NAME] [--n-tools N] [--svg PATH] [--svg-only]
          [--select-roi] [--debug-edges PATH] [--headless] [-q]
img2gcode --visualise FILE.gcode|FILE.svg
```

- `OUTPUT` ending in `.svg`, or `--svg-only` → write the toolpaths SVG only (with `--svg-only` and a `.gcode`
  OUTPUT, the SVG goes to the sibling path). Otherwise write G-code and a sibling SVG (same stem),
  or the SVG at `--svg PATH`.
- Input `.svg`: toolpaths SVG → read and write G-code directly (style ignored); other SVG → `svg_input` → planner.
- `--style` picks the style (`[mode] type` in the config otherwise); `--mode` is an alias kept for old scripts.
  `edge` is still `edge`; `lines` is accepted as an alias.
- `--select-roi` works for any artwork with regions; rectangles exclude infill only.
- After writing, unless `--headless`, open the existing viewer on the G-code (or on the SVG when only an SVG was
  written). Print the SVG and G-code paths.

## Config

Sections `[machine] [tools] [image] [layers] [infill] [extrusion]` keep every key and meaning. `[mode] type` takes
any style name. New sections `[fill] [edge] [contour] [oneline] [squiggle] [hatch]` with the keys listed above.
Old `[edge]` keys (`clahe_clip`, `detection_scales`, `collapse_factor`, …) are ignored without error.

## Testing

- `tests/test_writer_golden.py`: rebuild the moves in `tests/data/golden_writer_input.json` as `Toolpaths`
  (`x = x_px·s`, `y = (y_px + 1)·s`, `s = 1.25`), write, compare every line except the timestamp line with
  `tests/data/golden_writer.gcode` (generated by the writer at `d77193c`).
- `tests/test_svg_io.py`: round trip; Inkscape-style edits (group `transform`, `path` instead of `polyline`,
  hidden layer, stray shape outside tool groups, travel/source layers ignored).
- `tests/test_svg_input.py`: hole by nonzero winding and by evenodd, occlusion order, white is paper, thick
  stroke → region, thin stroke → line, gradient fallback, `--n-tools` merging.
- `tests/test_geometry.py`, `tests/test_raster.py`, `tests/test_skeleton.py`: units, including `skeleton_strokes` on `+`, `T`, ring and a
  spur, and `rings_to_geometry` winding rules.
- `tests/test_planner.py`: first perimeter is lw/2 inside the region, abutting tools' beads don't overlap, infill
  stays inside `inner`, exclusion zones, thin feature → centreline, layer angle alternation, every stroke emitted
  once.
- `tests/test_style_<name>.py` (one file per style): every style on synthetic images returns non-empty in-bounds artwork with valid tools.
  Noise test for `edge`: circle + rectangle at σ = 45 noise; ≥ 85 % of the true outline within 1.5·lw of a stroke
  (recall) and ≥ 85 % of stroke length within 1.5·lw of the true outline (precision). `fill` on the same image:
  ≤ 4 regions per tool, region area within 10 % of truth. `oneline`: exactly one stroke per used tool. `squiggle`:
  adjacent rows never closer than lw. Runtime per style on a 2000 px image ≤ 30 s.
- `tests/test_cli.py`: end-to-end on `images/python_logo.png` (tracked) and a synthetic photo, `.svg` output,
  toolpaths SVG input, artwork SVG input; all headless.
- Visual QA (not in pytest): run every style over every file in `images/`, build contact sheets, review them, tune
  defaults. Missing `images/` downloads never fail the test suite.
