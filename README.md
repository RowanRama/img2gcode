# img2gcode

Convert any raster image (PNG, JPG, BMP, …) into multi-tool FDM GCode.  
The pipeline segments the image by colour, traces perimeter contours, and fills each region with a configurable infill pattern — ready to print on a dual or multi-extruder machine.

---

## How it works

```
Image (PNG/JPG/…)
  └─ Background removal     – pixels above a whiteness threshold are discarded
  └─ K-Means segmentation   – foreground pixels are clustered into n tool colours
  └─ Perimeter tracing      – outer + inner (hole) contours extracted per region
  └─ Infill generation      – parallel or serpentine raster lines clipped to mask
  └─ GCode writer           – pixel → mm transform, extrusion calc, tool changes
  └─ Interactive visualiser – layer slider, travel toggle, per-layer stats
```

---

## Installation

Requires Python ≥ 3.9.

```bash
git clone git@github.com:RowanRama/img2gcode.git
cd img2gcode
pip install -e .
```

Dependencies (`numpy`, `Pillow`, `scikit-learn`, `opencv-python-headless`, `matplotlib`) are installed automatically.

---

## Quick start

```bash
# Convert an image and open the visualiser
python -m img2gcode -i logos/my_logo.png -o output/my_logo.gcode

# Headless — generate GCode without opening the GUI
python -m img2gcode -i logos/my_logo.png -o output/my_logo.gcode --headless

# Use a custom config file
python -m img2gcode -i logos/my_logo.png -o output/my_logo.gcode -c my_config.cfg

# Override number of colours/tools from the command line
python -m img2gcode -i logos/my_logo.png -o output/my_logo.gcode --n-tools 3

# Open the visualiser for an already-generated GCode file
python -m img2gcode --visualise output/my_logo.gcode
```

If installed via `pip install .` the `img2gcode` command is available directly:

```bash
img2gcode -i logos/my_logo.png -o output/my_logo.gcode
```

---

## Configuration

All settings live in a single `.cfg` file (INI format).  Copy `configs/default.cfg` and pass it with `-c`:

```bash
python -m img2gcode -i logo.png -o out.gcode -c my_config.cfg
```

### `[machine]` — printer geometry and speeds

| Key | Default | Description |
|---|---|---|
| `bed_size_x / bed_size_y` | 1200 | Print bed dimensions (mm) |
| `print_width / print_height` | 1000 | Maximum size of the printed image (mm). The image is scaled to fit inside this box while keeping its aspect ratio. |
| `origin_x / origin_y` | 10 | Offset of the print origin from the bed corner (mm) |
| `travel_speed` | 7800 | Non-printing move speed (mm/min) |
| `print_speed` | 1800 | Printing move speed (mm/min) |
| `z_lift` | 1.0 | Z hop height on travel moves (mm) |

### `[tools]` — multi-tool / multi-extruder

| Key | Default | Description |
|---|---|---|
| `num_tools` | 2 | Number of colour clusters. Each cluster maps to one extruder. Can be overridden with `--n-tools`. |
| `tool_change_gcode` | `T{tool}\nG92 E0` | GCode snippet injected on every tool change. Use `{tool}` as a placeholder for the tool index (0-based). |

### `[image]` — segmentation

| Key | Default | Description |
|---|---|---|
| `white_threshold` | 220 | Pixels whose R, G, and B channels are all ≥ this value are treated as background and ignored. Lower this value to keep more near-white detail. |
| `min_cluster_area` | 10 | Minimum pixel count for a cluster to be kept. Removes tiny speckles left after segmentation. |

### `[layers]` — layer stack

| Key | Default | Description |
|---|---|---|
| `num_layers` | 1 | How many identical layers to generate |
| `layer_height` | 3 | Height of each layer (mm) |
| `perimeter_loops` | 1 | Number of outer contour passes drawn around each colour region before infill |
| `horizontal_shell_layers` | 3 | Reserved — solid top/bottom shells (not yet implemented) |

### `[infill]` — fill pattern

| Key | Default | Description |
|---|---|---|
| `pattern` | `zigzag` | Fill pattern. Only `zigzag` (parallel lines) is implemented. |
| `density` | 100 | Infill density as a percentage (0–100). Controls line spacing: `spacing = nozzle_diameter / (density / 100)`. At 100 % lines are packed edge-to-edge; lower values leave larger gaps. |
| `line_spacing` | 0 | Explicit line spacing override (mm). If non-zero this takes precedence over `density`. |
| `angle` | 45 | Infill angle in degrees. Even layers use this angle, odd layers rotate by +90° so successive layers are perpendicular. |
| `connect_lines` | `true` | `true` — adjacent scan lines are joined into a continuous serpentine (PrusaSlicer-style, fewer travel moves). `false` — each line is its own stroke with a travel between them, giving visually even parallel hatching. |

### `[extrusion]` — extrusion maths

| Key | Default | Description |
|---|---|---|
| `filament_diameter` | 1.75 | Filament diameter (mm) |
| `nozzle_diameter` | 4 | Nozzle diameter (mm). Also used as the default line width. |
| `line_width_mm` | 0 | Physical line width (mm). `0` = use `nozzle_diameter`. Typical values are 100–150 % of the nozzle diameter. |
| `extrusion_multiplier` | 1.0 | Scales all extrusion values. Use to fine-tune over/under-extrusion. |
| `layer_height` | 3 | Layer height used in extrusion volume calculations (should match `layers.layer_height`). |

---

## Tips

**Image preparation**  
- Use images with a clean white background. If your logo has a drop shadow or near-white border, lower `white_threshold` (e.g. 200).
- Higher contrast between colours gives better K-Means separation.

**Choosing `num_tools`**  
- Set `num_tools` to the number of distinct colours you want to print. For a two-colour logo use `2`; for a full multi-colour image increase as needed.
- The K-Means algorithm always produces exactly `num_tools` clusters — if some colours are very similar they may merge.

**Infill density vs. line spacing**  
- At `density = 100` and `nozzle_diameter = 4` the line spacing is 4 mm (edge-to-edge packed).
- At `density = 50` the spacing doubles to 8 mm.
- Set `line_spacing` directly (e.g. `line_spacing = 6`) if you want a fixed gap regardless of nozzle size.

**Serpentine vs. parallel hatching**  
- `connect_lines = true` joins adjacent lines into one continuous path, minimising travel moves (PrusaSlicer-style). Short perpendicular connectors appear at the boundary edges.
- `connect_lines = false` emits each line as an independent stroke, producing perfectly even parallel gaps with no edge connectors. Useful when verifying that density is correct.

---

## Project layout

```
img2gcode/
├── configs/
│   └── default.cfg        # Default configuration (copy and customise)
├── img2gcode/
│   ├── config.py          # Config dataclasses + .cfg loader
│   ├── segmenter.py       # Image loading, background removal, K-Means
│   ├── toolpath.py        # Contour extraction + infill generation
│   ├── writer.py          # GCode writer (pixel→mm, extrusion calc)
│   ├── gcode_parser.py    # GCode → segment list for the visualiser
│   ├── gui.py             # Interactive matplotlib visualiser
│   ├── pipeline.py        # High-level pipeline helper
│   └── __main__.py        # CLI entry point
├── logos/                 # Example input images
├── output/                # Generated GCode files (git-ignored)
├── tests/
├── requirements.txt
└── setup.py
```

---

## Requirements

- Python ≥ 3.9
- numpy ≥ 1.24
- Pillow ≥ 10.0
- scikit-learn ≥ 1.3
- opencv-python-headless ≥ 4.8
- matplotlib ≥ 3.7

---

## License

MIT
