# img2gcode

Turn images and SVGs into multi-tool G-code for a large-format FDM printer. The defaults suit a 4 mm nozzle and a
1000 × 700 mm print area; change them in a config file.

Every conversion happens in two steps:

```
image ──► style ──┐
                  ├──► planner ──► toolpaths.svg ──► G-code
artwork SVG ──────┘
toolpaths SVG ────────────────────────────────────► G-code
```

1. A **style** turns the image into shapes and lines on the page, in millimetres. An artwork SVG (a logo, a flag, an
   icon) skips this step: its own shapes are used.
2. The **planner** turns them into toolpaths: perimeters and infill for filled areas, centrelines for lines,
   ordered to keep travel short, repeated for every layer.
3. The toolpaths are written to a **toolpaths SVG**. Open it in a browser to check it, or edit it in Inkscape.
4. The **G-code** is made from that SVG as read back from disk, so the SVG and the G-code always match. Convert an
   edited SVG and the G-code follows your edits.

---

## Installation

Python 3.9 or newer.

```bash
git clone git@github.com:RowanRama/img2gcode.git
cd img2gcode
pip install -e .
```

This installs the `img2gcode` command. `python -m img2gcode` does the same thing.

---

## Usage

```bash
# Image → output/logo.gcode and output/logo.svg, then open the viewer
img2gcode -i logo.png -o output/logo.gcode

# Toolpaths SVG only (no G-code)
img2gcode -i logo.png -o output/logo.svg

# G-code from a toolpaths SVG, for example after editing it in Inkscape
img2gcode -i output/logo.svg -o output/logo.gcode

# Artwork SVG: one tool per fill colour
img2gcode -i flag.svg -o output/flag.gcode

# Skip the viewer, use your own config
img2gcode -i logo.png -o output/logo.gcode --headless -c my_printer.cfg

# Open the viewer on an existing G-code file or toolpaths SVG
img2gcode --visualise output/logo.gcode
```

After writing, the CLI prints the paths, the estimated print time, the filament use and the print and travel
distances.

---

## Styles

Pick a style with `--style NAME` or with `[mode] type` in the config. The style only applies to images: an artwork
SVG is always converted from its own shapes, and a toolpaths SVG is converted as it is.

| Style | What it prints | Good for |
|---|---|---|
| `fill` | Flat colour areas: perimeters plus infill, one tool per colour | Logos, flags, posters, flat-colour art. Photos come out posterized. |
| `edge` | Clean outlines and feature lines | Line drawings, portraits, photos you want as a sketch. Copes with noisy and grainy photos. |
| `contour` | Topographic lines of brightness | Portraits and photos with smooth shading. Abstract look. |
| `oneline` | One unbroken line per tool; density follows darkness | Portraits and bold subjects. Almost no travel moves. |
| `squiggle` | Rows of waves, taller and tighter where the image is dark | Photos with a clear range of tones. Reads well from a distance. |
| `hatch` | Engraving-style hatching, more directions in darker areas | Portraits, engravings, shaded drawings. |

`[tools] num_tools` (or `--n-tools`) sets how many tools a style uses. Tool colours come from the image, darkest
first (T0); they only colour the SVG and the viewer, the printer uses whatever filament each tool holds.

- `fill`: one tool per colour.
- `edge`, `oneline`, `squiggle`: each line takes the tool whose colour is nearest the image under it.
- `contour`: darker levels go to lower tool numbers.
- `hatch`: each hatch direction gets the next tool, cycling through them.

Use `--n-tools 1` for a single-colour print.

```bash
img2gcode -i logo.png      -o output/logo.gcode     --style fill --n-tools 3
img2gcode -i sketch.jpg    -o output/sketch.gcode   --style edge --n-tools 1
img2gcode -i portrait.jpg  -o output/contour.gcode  --style contour
img2gcode -i portrait.jpg  -o output/oneline.gcode  --style oneline --n-tools 1
img2gcode -i photo.jpg     -o output/squiggle.gcode --style squiggle
img2gcode -i engraving.jpg -o output/hatch.gcode    --style hatch --n-tools 1
```

`lines` is accepted as another name for `edge`, and `--mode` as another name for `--style`, so old scripts keep
working.

---

## Editing the toolpaths SVG

The toolpaths SVG is a normal SVG in millimetres. Open it in a browser to look at it. In Inkscape it opens as layers:

```
Source image          the input, faint and locked (never printed)
Layer 0               one group per printed layer
  Layer 0 · T0        everything tool 0 prints on layer 0, in print order
  Layer 0 · T1
Layer 1 ...
Travel moves          hidden by default (never printed)
```

To edit:

1. Open the SVG in Inkscape and use the Layers panel (Layer ▸ Layers and Objects).
2. Delete, move, reshape or draw lines inside the tool layers. Before drawing, select the tool layer you want the new
   line in. Moving a line to another tool layer changes which tool prints it.
3. Save (File ▸ Save, as Inkscape SVG).
4. Convert it: `img2gcode -i edited.svg -o out.gcode`

What prints:

- **Only shapes inside a tool layer** (`Layer N · TK`) print. Shapes anywhere else are ignored, and the CLI prints a
  warning with how many it ignored. A new layer you create in Inkscape has no tool number, so nothing in it prints:
  draw in an existing tool layer, or duplicate one (its `data-tool` attribute, visible in the XML editor, sets the
  tool).
- **Hidden layers and hidden objects do not print.** Hide a tool layer to leave that tool out; hide a layer to skip it.
- Each shape prints as one line along its outline, at the nozzle width. Lines, polylines, polygons, paths, rectangles,
  circles and ellipses work; curves are followed closely. Fills are not printed as infill. Convert text to a path
  first (Path ▸ Object to Path).
- Shapes print in document order within their tool layer. Moving and transforming objects is fine.
- The source image and travel-move layers are ignored. Show the travel-move layer to check the travel between
  lines.

The SVG records the line width it was planned for. If your config uses a different line width, the CLI warns you.

---

## CLI flags

| Flag | Description |
|---|---|
| `-i, --input PATH` | Image (PNG, JPG, …), artwork SVG or toolpaths SVG. Required unless `--visualise`. |
| `-o, --output PATH` | `.gcode`: write the G-code and a toolpaths SVG next to it with the same name. `.svg`: write the toolpaths SVG only. Required unless `--visualise`. |
| `-c, --config PATH` | Config file. Its values override `configs/default.cfg`. |
| `--style NAME` | `fill`, `edge`, `contour`, `oneline`, `squiggle` or `hatch` (`lines` = `edge`). Overrides `[mode] type`. |
| `--mode NAME` | Same as `--style`, kept for old scripts. |
| `--n-tools N` | Number of tools (colours). Overrides `[tools] num_tools`. For an artwork SVG, merges its colours down to N. |
| `--svg PATH` | Where to write the toolpaths SVG (default: next to the G-code). |
| `--svg-only` | Write the toolpaths SVG only, no G-code. |
| `--select-roi` | Draw rectangles to leave without infill before planning (perimeters stay). Needs filled areas: the `fill` style or an artwork SVG. |
| `--debug-edges PATH` | `edge` style only: save the line map it traces as a PNG. |
| `--headless` | Do not open the viewer after writing. |
| `--visualise PATH` | Open the viewer on an existing `.gcode` file or toolpaths `.svg`. No conversion. |
| `-q, --quiet` | No progress messages. |
| `-v, --verbose` | Accepted for old scripts; messages are on unless `-q`. |

If the toolpaths SVG would overwrite an input artwork SVG, it is written as `<name>.toolpaths.svg` instead.

---

## Configuration

Settings live in an INI file. `configs/default.cfg` lists every key with its default and a comment. Copy it, change
what you need, and pass it with `-c`; keys you leave out keep their defaults, and unknown keys are ignored.

```bash
cp configs/default.cfg my_printer.cfg
img2gcode -i logo.png -o output/logo.gcode -c my_printer.cfg
```

| Section | Keys |
|---|---|
| `[mode]` | `type`: the default style. |
| `[machine]` | `bed_size_x`, `bed_size_y`, `origin_x`, `origin_y`, `print_width`, `print_height` (the image is fitted into this area, centred on X0 Y0), `travel_speed`, `print_speed` (mm/min), `z_lift`. |
| `[tools]` | `num_tools`. |
| `[image]` | `white_threshold`: pixels with R, G and B all at or above it are paper and do not print. |
| `[layers]` | `num_layers`, `layer_height`, `perimeter_loops` (loops around every filled area). |
| `[infill]` | `pattern` (`zigzag`), `density` (%), `line_spacing` (mm, 0 = from density), `angle` (odd layers add 90°), `connect_lines`. |
| `[extrusion]` | `filament_diameter`, `nozzle_diameter`, `line_width_mm` (0 = nozzle diameter), `extrusion_multiplier`, `layer_height`. |
| `[fill]` | `min_feature_factor`, `smoothing`. |
| `[edge]` | `detail`, `contrast`, `min_length_factor`, `debug_edges_path`. |
| `[contour]` | `levels`, `smoothing`. |
| `[oneline]` | `points`, `gamma`, `tsp_seconds`. |
| `[squiggle]` | `row_spacing`, `max_frequency`. |
| `[hatch]` | `bands`, `angles`, `spacing`. |

Sizes in the style sections are multiples of the line width (`line_width_mm`, or the nozzle diameter when that
is 0), so the same settings work for any nozzle.

---

## Tuning tips

Write the SVG only (`-o out.svg --headless`) while tuning: it is quick to make and opens in any browser. For every
style, if light paper or a pale background prints, lower `[image] white_threshold`; if pale colours disappear,
raise it.

**fill**
- Speckles and tiny islands: raise `min_feature_factor` (2–3). Fine details lost: lower it toward 1.
- Jagged colour borders: raise `smoothing`.
- Two colours merged into one: raise `num_tools`.
- Thin text and hairlines narrower than the nozzle print as single centrelines rather than disappearing.

**edge**
- Too many lines: lower `detail`. Fine features missing: raise it.
- Short dashes: raise `min_length_factor`.
- Flat, low-contrast photo: raise `contrast` (2–3). Lines all over a textured background: lower it, or set it to
  0 to turn the boost off.
- Run with `--debug-edges edges.png` to see the line map before tracing. If the map is wrong, change `detail` or
  `contrast`; if the map is right but the lines are not, change `min_length_factor`.

**contour**
- More or fewer lines: change `levels`.
- Smoother, more abstract lines: raise `smoothing`. More detail: lower it.

**oneline**
- Denser or sparser line: set `points` (0 picks a count from how dark the image is).
- More contrast between light and dark areas: raise `gamma` (1.5–2).
- Line crosses itself a lot on large point counts: raise `tsp_seconds`.

**squiggle**
- More detail: lower `row_spacing` (2 or more leaves the waves room between rows). Bolder: raise it.
- Calmer waves: lower `max_frequency`.

**hatch**
- More tone steps: raise `bands` and give one angle per band after the first, e.g. `bands = 5`,
  `angles = 45,-45,0,90`. Bands without an angle are not hatched.
- Lighter overall: raise `spacing`.

---

## Test images

`images/` holds four logos in the repository. Download the rest of the test set (flags, artwork SVGs, line art,
noisy photos and portraits; all public domain or openly licensed, listed in the script) with:

```bash
python scripts/fetch_test_images.py
```

Files already present are skipped.

## Style gallery

`scripts/style_gallery.py` runs styles over a set of images and makes contact sheets for comparing them:

```bash
python scripts/style_gallery.py --out gallery                              # every style, every file in images/
python scripts/style_gallery.py images/portrait_*.jpg --out gallery --styles edge,hatch -c my_printer.cfg -j 8
```

For each style it writes:

- `gallery/<style>/<image>.svg`: the toolpaths SVG.
- `gallery/<style>/<image>.png`: the toolpaths drawn at the real line width on white (about 4 px per mm, at most
  1200 px wide).
- `gallery/<style>_sheet.jpg`: every image's source next to its rendering, labelled with the time and stroke count.
- `gallery/timings.csv`: `style,image,seconds,strokes,print_mm` for every run (rewritten on each run, so give
  parallel runs their own `--out`).

An image that fails is listed in the CSV with `seconds` = -1, marked on the sheet, and its traceback is printed; the
run carries on. Arguments can be files, folders or glob patterns; with none, it uses `images/`. `-j N` converts N
images at once. Artwork SVGs convert the same way whatever the style.

## Running tests

```bash
pip install pytest
python -m pytest -q                        # fast tests
python -m pytest -q -m slow                # runtime checks on large images only
python -m pytest -q -m "slow or not slow"  # everything
```

On Windows with the project virtual environment, use `.venv\Scripts\python.exe -m pytest -q`. The tests use synthetic
images and the four logos in the repository, so they pass without downloading the test set.

---

## License

MIT
