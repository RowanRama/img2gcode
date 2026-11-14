# img2gcode

Convert images, photos and SVGs into smooth multi-tool G-code, with a live
preview GUI for Windows and Linux and a small CLI.

- **fill** mode: splits the image by colour (K-means, or an SVG's own fill
  colours), one tool per colour, and draws each region as perimeters plus
  infill. Best for logos, text, flat-colour art and SVGs.
- **edge** mode: detects edges (CLAHE, bilateral filter, multi-scale Canny),
  thins them and traces them as continuous strokes. Best for portraits and
  photos.

Paths are smooth rather than pixel-stepped. Fill mode cuts perimeters and
infill ends from a smoothed signed distance field with sub-pixel iso-lines.
Edge strokes are smoothed along their length. Both are then simplified to a
tolerance in mm, so the G-code has few, evenly spaced points.

## Install

Download from [Releases](https://github.com/RowanRama/img2gcode/releases):

- **Windows installer** (`…-windows-x64-setup.exe`): no admin needed. Tick
  *Add img2gcode to PATH* to use the CLI from any terminal; the GUI goes in
  the Start menu. Uninstall from Settings > Apps.
- **Windows portable** (`…-windows-x64-portable.zip`): unzip anywhere.
  `img2gcode-gui.exe` opens the GUI, `img2gcode.exe` is the CLI.
- **Debian / Ubuntu** (`…_amd64.deb`): `sudo apt install ./img2gcode_*.deb`
  puts `img2gcode` in `/usr/bin` and the GUI in the application menu.
- **Other Linux** (`…-linux-x86_64.tar.gz`): unpack and run `./install.sh`
  (installs to `~/.local`, or `/usr/local` as root; `--uninstall` removes it).

Releases are built by `.github/workflows/release.yml` when a `v*` tag is
pushed.

## Build

Requires Rust 1.88+.

```bash
cargo build --release            # target/release/img2gcode(.exe)
cargo install --path .           # or put it on PATH
```

Linux needs a C linker only; at run time the GUI uses X11 or Wayland with
OpenGL, and file dialogs go through xdg-desktop-portal. For a machine without
a display, build the CLI alone: `cargo build --release --no-default-features`.

## CLI

```bash
img2gcode                                   # open the GUI
img2gcode gui logo.png                      # open the GUI on an image
img2gcode convert logo.png                  # writes logo.gcode next to it
img2gcode convert logo.svg -o out.gcode --size 300x200
img2gcode convert logo.png --preview logo-preview.png
img2gcode convert face.jpg --mode edge --set edge.collapse=1.0
img2gcode convert logo.png -c my.toml --set infill.density=40 --set layers.count=3
img2gcode view out.gcode                    # preview an existing G-code file
img2gcode config > my.toml                  # start a config file
```

| Option | |
|---|---|
| `-o, --output FILE` | Output G-code (default: input name with `.gcode`) |
| `-c, --config FILE` | Config file (TOML) |
| `--mode fill\|edge` | Generation mode |
| `--tools N` | Number of colours / tools |
| `--size WxH` | Print area in mm; the image is scaled to fit, aspect kept. One number means a square |
| `--set key=value` | Any config key, repeatable: `--set infill.angle=0` |
| `--preview PNG` | Also save a picture of the toolpaths at printed line width |
| `--debug-edges PNG` | Edge mode: save the detected edge map, for tuning |
| `-q, --quiet` | Print nothing but errors |

## GUI

Every parameter is on the left; any change regenerates on a background
thread and the canvas updates within milliseconds (loading, colour
segmentation and edge detection are cached, so most changes skip them).

- Open an image, SVG or G-code file, or drop one on the window.
- The canvas shows the print area as white paper, toolpaths at their real
  line width, each tool in its image colour (infill a little lighter).
  Drag to pan, scroll to zoom, double-click to fit.
- Bottom bar: travel moves, source image overlay, line width on/off, layer
  filter, and a progress slider that replays the print order.
- Infill exclusions: turn on *Draw exclusions* and drag rectangles where
  infill should be left out; right-click one to remove it.
- Save and load the configuration as TOML; *Export G-code* writes the file.

## Text

Add text in any installed font, on its own or on top of an image. Each text
item is placed on the canvas in mm (centred on X/Y, independent of the image)
and printed like a fill-mode region: perimeters, plus infill unless you turn
it off for outline-only lettering.

- GUI: *Text* > *Add text*, then pick the font (type to search), size,
  position, bold/italic, infill and tool. Changes preview live.
- CLI, one item: `--text "Rowan Wuz Here" --font "Arial Black" --text-size 40
  --text-pos 0,-250` (`--text-outline` for no infill, `--bold`, `--italic`,
  `--text-tool N`, `\n` for a new line). With no input image the job is just
  the text.
- Several items: `[[text]]` blocks in a config file (see `img2gcode config`).
- `img2gcode fonts [filter]` lists the names you can use: families
  ("Arial") and the full names of their faces ("Arial Black",
  "Segoe UI Semibold").

The size is the font size in mm; capital letters are roughly 0.7 of it.
Outlines follow `layers.perimeter_inset` like everything else: 0.5 keeps the
printed edge on the letter's edge, 0 traces the outline itself.

## Configuration

`img2gcode config` prints the default file with every key documented
(`configs/default.toml`). Every key is optional. The main sections:

- `[canvas]` print area (`width`, `height`) and where its centre lands.
- `[extrusion]` nozzle, line width, filament, multiplier.
- `[layers]` count, height, perimeters, `perimeter_inset`.
- `[image]` `tools`, `white_threshold`, `min_area`, `resolution`, `trim`
  (crop white margins so the drawing itself fills the canvas).
- `[infill]` `density` or `spacing`, `angle`, `connect` (serpentine), `overlap`.
- `[path]` `smoothing` (px) and `tolerance` (mm): smoother curves vs detail.
- `[edge]` edge-mode detection and stroke settings.
- `[gcode]` start, tool-change and end blocks with `{tool}`,
  `{travel_speed}` and `{print_speed}` placeholders.
- `exclusions` infill exclusion rectangles as image fractions.

### Tuning edge mode

1. Run with `--debug-edges edges.png` (or watch the GUI). If the edges are
   wrong, tune detection: `clahe_clip`, `bilateral_d`, `scales`, or pin
   `canny_low` / `canny_high`.
2. If the edges are right but the strokes look fragmented: raise `collapse`
   toward 1.0 to merge close parallel edges, raise `chain_gap` to join broken
   strokes, or lower `min_separation` to let nearby strokes coexist.
3. Chunkier strokes: `collapse` 1.5. Maximum texture: add `0` to `scales`.

## Development

```bash
cargo test
cargo clippy --all-targets -- -D warnings
IMG2GCODE_LAYOUT_PROBE=1 cargo run -- gui some.png   # GUI layout-shift check
```

The layout probe steps the GUI through results arriving, canvas and tool
changes and an error, then reports any control whose position changed.

## License

MIT
