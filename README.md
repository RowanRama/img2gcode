# img2gcode

Convert raster images, photos, and SVGs into multi-tool FDM GCode.

Two generation modes:

- **fill** — segments the image by colour (K-Means), traces perimeters, and fills each region with a configurable zigzag infill. One tool per colour cluster. Best for logos, flat-colour illustrations, and SVGs.
- **edge** — Canny edge detection with CLAHE enhancement and multi-scale gradient stacking, then traces the resulting line drawing as continuous polylines. Best for portraits and photos where you want internal feature detail (eyes, jawline, hair) rather than solid-colour fills.

SVGs go through a vector-aware path: every distinct `fill` colour becomes its own tool with no K-Means.

---

## Installation

Requires Python ≥ 3.9.

```bash
git clone git@github.com:RowanRama/img2gcode.git
cd img2gcode
pip install -e .
```

---

## Usage

```bash
# Default fill mode — opens the visualiser when done
python -m img2gcode -i logo.png -o out.gcode

# Edge mode for portraits / line drawings
python -m img2gcode -i portrait.jpg -o out.gcode --mode edge --n-tools 1

# SVG input (one tool per fill colour)
python -m img2gcode -i logo.svg -o out.gcode

# Headless (skip the GUI)
python -m img2gcode -i logo.png -o out.gcode --headless

# Custom config
python -m img2gcode -i logo.png -o out.gcode -c my_config.cfg

# Override the tool count
python -m img2gcode -i logo.png -o out.gcode --n-tools 3

# Fill mode — interactive ROI selection (draw rectangles to leave hollow)
python -m img2gcode -i logo.png -o out.gcode --select-roi

# Edge mode — save the pre-trace edge map for tuning
python -m img2gcode -i portrait.jpg -o out.gcode --mode edge --debug-edges edges.png

# Visualise an existing GCode file
python -m img2gcode --visualise out.gcode
```

If installed via `pip install .` the `img2gcode` command is available on `$PATH`.

### CLI flags

| Flag | Description |
|---|---|
| `-i, --input PATH` | Input image (PNG / JPG / BMP / SVG). Required unless `--visualise`. |
| `-o, --output PATH` | Output GCode path. Required unless `--visualise`. |
| `-c, --config PATH` | Override the default `.cfg` file. |
| `--mode {fill,edge}` | Generation mode. Overrides `[mode] type` in the config. |
| `--n-tools N` | Override the number of tools / colour clusters. Ignored for SVG. |
| `--select-roi` | Fill mode only — open a window to draw rectangles excluded from infill. |
| `--debug-edges PATH` | Edge mode only — save the pre-trace edge map as a PNG for tuning. |
| `--headless` | Skip the GUI after generation. |
| `--visualise PATH` | Open the GUI for an already-generated GCode file (no conversion). |
| `-q, --quiet` | Suppress progress output. |

---

## Configuration

All settings live in a single `.cfg` file (INI format). Start by copying `configs/default.cfg`:

```bash
cp configs/default.cfg my_config.cfg
python -m img2gcode -i logo.png -o out.gcode -c my_config.cfg
```

Sections that matter most:

- **`[mode]`** — `type = fill` or `edge`.
- **`[machine]`** — bed dimensions, print area, travel/print speeds, Z lift.
- **`[tools]`** — `num_tools`, custom tool-change GCode.
- **`[image]`** — `white_threshold`, `min_cluster_area` (segmentation cleanup).
- **`[layers]`** — `num_layers`, `layer_height`, `perimeter_loops`.
- **`[infill]`** — fill mode pattern, `density`, `angle`, `connect_lines`.
- **`[extrusion]`** — `filament_diameter`, `nozzle_diameter`, `line_width_mm`, `extrusion_multiplier`.
- **`[edge]`** — edge mode: CLAHE strength, multi-scale Canny sigmas, parallel-edge collapse, polyline chaining, and stroke-separation tolerances. The default config has every field documented inline.

### Tuning edge mode

Edge mode has a few interacting knobs. The intended workflow:

1. Run with `--debug-edges edges.png` and check the edge map. If it's already wrong, tune **detection**: `clahe_clip`, `bilateral_d`, `detection_scales`, or pin `canny_low` / `canny_high`.
2. If the edge map is right but the gcode looks fragmented, tune the **trace stage**: raise `collapse_factor` toward 1.0 to merge close parallel edges, raise `chain_gap_factor` to stitch broken polylines, or lower `min_separation_factor` to let nearby strokes coexist.
3. Want chunkier strokes → raise `collapse_factor` to 1.5. Want maximum texture detail → put `0` in `detection_scales`

---

## License

MIT
