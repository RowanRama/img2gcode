//! The live-preview GUI: every parameter change regenerates on a worker
//! thread and the canvas shows the result.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::mpsc::{Receiver, Sender, channel};
use std::time::Instant;

use crate::config::Mode;
use crate::toolpath::{PathKind, Toolpaths};
use crate::{Config, Pipeline, Stats, gcode};
use eframe::egui::{self, Color32, Pos2, Rect, Sense, Stroke, StrokeKind, Vec2};

const IMAGE_EXT: &[&str] = &["png", "jpg", "jpeg", "bmp", "gif", "webp", "tif", "tiff", "svg"];
const ICON_PNG: &[u8] = include_bytes!("../assets/icon.png");

/// Open the window. `input` is an image to start on; `gcode_file` a G-code
/// file to view instead.
pub fn run(cfg: Config, input: Option<PathBuf>, gcode_file: Option<PathBuf>) -> Result<(), String> {
    let mut viewport = egui::ViewportBuilder::default()
        .with_inner_size([1400.0, 900.0])
        .with_min_inner_size([900.0, 600.0])
        .with_title("img2gcode");
    if let Ok(icon) = image::load_from_memory(ICON_PNG) {
        let icon = icon.to_rgba8();
        let (width, height) = icon.dimensions();
        viewport = viewport.with_icon(egui::IconData { rgba: icon.into_raw(), width, height });
    }
    let options = eframe::NativeOptions { viewport, ..Default::default() };
    eframe::run_native(
        "img2gcode",
        options,
        Box::new(move |cc| {
            let mut app = App::new(&cc.egui_ctx, cfg);
            if let Some(p) = gcode_file {
                app.open_gcode(&p);
            } else if let Some(p) = input {
                app.open_image(p);
            }
            Ok(Box::new(app))
        }),
    )
    .map_err(|e| e.to_string())
}

struct Job {
    generation: u64,
    input: Option<PathBuf>,
    cfg: Config,
}

struct Reply {
    generation: u64,
    result: Result<(crate::Output, Config), String>,
    ms: f64,
}

/// What the canvas shows.
struct Shown {
    toolpaths: Toolpaths,
    stats: Stats,
    /// Config the toolpaths were made with (`None` for a loaded G-code file).
    cfg: Option<Config>,
    segs: Vec<Seg>,
    layers: usize,
    image: Option<Arc<crate::load::Loaded>>,
    /// Bounds of each text item in the result.
    text_bounds: Vec<Option<[f64; 4]>>,
    ms: f64,
}

/// Something on the canvas that can be selected and dragged.
#[derive(Clone, Copy, PartialEq, Debug)]
enum Target {
    Image,
    Text(usize),
}

#[derive(Clone, Copy, PartialEq)]
enum Handle {
    /// Scales the item.
    Corner,
    /// Sets a text item's wrap width.
    Side,
}

/// What a primary-button drag on the canvas is doing.
#[derive(Clone, Copy)]
enum Drag {
    None,
    Pan,
    Exclusion { from: [f64; 2] },
    Move { target: Target, start: [f64; 2], orig: [f64; 2] },
    Scale { target: Target, centre: [f64; 2], start: f64, orig: f64, orig_width: f64 },
    Wrap { index: usize, centre_x: f64 },
}

/// One drawable segment, in print order.
#[derive(Clone, Copy)]
struct Seg {
    a: [f32; 2],
    b: [f32; 2],
    color: Color32,
    travel: bool,
    layer: u32,
}

struct App {
    cfg: Config,
    input: Option<PathBuf>,
    gcode_file: Option<PathBuf>,
    config_path: Option<PathBuf>,
    jobs: Sender<Job>,
    replies: Receiver<Reply>,
    generation: u64,
    busy: bool,
    shown: Option<Shown>,
    error: Option<String>,
    note: String,
    texture: Option<(usize, egui::TextureHandle)>,
    center: [f64; 2],
    zoom: f32,
    fit_pending: bool,
    show_travel: bool,
    show_image: bool,
    real_width: bool,
    layer: usize,
    progress: f32,
    draw_exclusions: bool,
    selected: Option<Target>,
    drag: Drag,
    probe_run: Option<ProbeRun>,
    /// Installed font names, filled in by a background thread.
    fonts: Option<Vec<String>>,
    font_rx: Receiver<Vec<String>>,
    font_filter: String,
}

impl App {
    fn new(ctx: &egui::Context, cfg: Config) -> Self {
        let (jobs, job_rx) = channel::<Job>();
        let (reply_tx, replies) = channel::<Reply>();
        // Scanning every installed font takes a moment; do it off the UI thread.
        let (font_tx, font_rx) = channel();
        let font_ctx = ctx.clone();
        std::thread::spawn(move || {
            let _ = font_tx.send(crate::text::font_names());
            font_ctx.request_repaint();
        });
        let ctx = ctx.clone();
        std::thread::spawn(move || {
            let mut pipeline = Pipeline::default();
            while let Ok(mut job) = job_rx.recv() {
                // Only the newest request matters.
                while let Ok(newer) = job_rx.try_recv() {
                    job = newer;
                }
                let t = Instant::now();
                let result =
                    pipeline.run(job.input.as_deref(), &job.cfg).map(|o| (o, job.cfg)).map_err(|e| e.to_string());
                let ms = t.elapsed().as_secs_f64() * 1000.0;
                if reply_tx.send(Reply { generation: job.generation, result, ms }).is_err() {
                    break;
                }
                ctx.request_repaint();
            }
        });
        Self {
            cfg,
            input: None,
            gcode_file: None,
            config_path: None,
            jobs,
            replies,
            generation: 0,
            busy: false,
            shown: None,
            error: None,
            note: "Open an image or SVG (or drop one on the window).".into(),
            texture: None,
            center: [0.0, 0.0],
            zoom: 1.0,
            fit_pending: true,
            show_travel: false,
            show_image: false,
            real_width: true,
            layer: 0,
            progress: 1.0,
            draw_exclusions: false,
            selected: None,
            drag: Drag::None,
            probe_run: std::env::var_os("IMG2GCODE_LAYOUT_PROBE").map(|_| {
                PROBE.with(|p| *p.borrow_mut() = Some(Default::default()));
                ProbeRun { step: 0, idle_frames: 0 }
            }),
            fonts: None,
            font_rx,
            font_filter: String::new(),
        }
    }

    fn open_image(&mut self, path: PathBuf) {
        self.note = format!("Showing {}", path.display());
        self.input = Some(path);
        self.gcode_file = None;
        self.cfg.exclusions.clear();
        self.fit_pending = true;
        self.regenerate();
    }

    fn open_gcode(&mut self, path: &Path) {
        match std::fs::read_to_string(path) {
            Ok(text) => {
                let tp = gcode::parse(&text);
                let stats = gcode::stats(&tp, &self.cfg);
                self.input = None;
                self.gcode_file = Some(path.to_path_buf());
                self.error = None;
                self.note = format!("Viewing {}", path.display());
                self.set_shown(Shown {
                    segs: Vec::new(),
                    layers: tp.layer_count(),
                    toolpaths: tp,
                    stats,
                    cfg: None,
                    image: None,
                    text_bounds: Vec::new(),
                    ms: 0.0,
                });
                self.fit_pending = true;
            }
            Err(e) => self.error = Some(format!("{}: {e}", path.display())),
        }
    }

    fn regenerate(&mut self) {
        if self.gcode_file.is_some() {
            return;
        }
        if self.input.is_none() && self.cfg.text.iter().all(|t| t.content.trim().is_empty()) {
            // Nothing left to show.
            self.shown = None;
            self.busy = false;
            self.generation += 1;
            return;
        }
        self.generation += 1;
        self.busy = true;
        let _ = self.jobs.send(Job { generation: self.generation, input: self.input.clone(), cfg: self.cfg.clone() });
    }

    fn set_shown(&mut self, mut shown: Shown) {
        shown.segs = flatten(&shown.toolpaths);
        if self.layer > shown.layers {
            self.layer = 0;
        }
        self.shown = Some(shown);
    }

    fn poll(&mut self) {
        if let Ok(fonts) = self.font_rx.try_recv() {
            self.fonts = Some(fonts);
        }
        while let Ok(reply) = self.replies.try_recv() {
            if reply.generation != self.generation {
                continue;
            }
            self.busy = false;
            match reply.result {
                Ok((out, cfg)) => {
                    self.error = None;
                    let stats = gcode::stats(&out.toolpaths, &cfg);
                    let layers = out.toolpaths.layer_count();
                    self.set_shown(Shown {
                        segs: Vec::new(),
                        layers,
                        toolpaths: out.toolpaths,
                        stats,
                        cfg: Some(cfg),
                        image: out.image,
                        text_bounds: out.text_bounds,
                        ms: reply.ms,
                    });
                }
                Err(e) => self.error = Some(e),
            }
        }
    }

    fn texture_for(&mut self, ctx: &egui::Context) -> Option<egui::TextureId> {
        let img = self.shown.as_ref()?.image.as_ref()?;
        let key = Arc::as_ptr(img) as usize;
        if self.texture.as_ref().is_none_or(|(k, _)| *k != key) {
            let rgba = &img.rgba;
            let ci = egui::ColorImage::from_rgba_unmultiplied(
                [rgba.width() as usize, rgba.height() as usize],
                rgba.as_raw(),
            );
            self.texture = Some((key, ctx.load_texture("source", ci, egui::TextureOptions::LINEAR)));
        }
        self.texture.as_ref().map(|(_, t)| t.id())
    }

    // ----- top bar ---------------------------------------------------------

    fn top_bar(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            if probe("open image", ui.button("Open image…")).clicked()
                && let Some(p) = rfd::FileDialog::new().add_filter("Images and SVG", IMAGE_EXT).pick_file()
            {
                self.open_image(p);
            }
            if probe("open gcode", ui.button("Open G-code…")).clicked()
                && let Some(p) = rfd::FileDialog::new().add_filter("G-code", &["gcode", "gco", "nc", "txt"]).pick_file()
            {
                self.open_gcode(&p);
            }
            ui.separator();
            if probe("load config", ui.button("Load config…")).clicked()
                && let Some(p) = rfd::FileDialog::new().add_filter("Config", &["toml"]).pick_file()
            {
                match std::fs::read_to_string(&p)
                    .map_err(|e| e.to_string())
                    .and_then(|t| Config::from_toml(&t).map_err(|e| e.to_string()))
                {
                    Ok(cfg) => {
                        self.cfg = cfg;
                        self.config_path = Some(p);
                        self.regenerate();
                    }
                    Err(e) => self.error = Some(format!("{}: {e}", p.display())),
                }
            }
            if probe("save config", ui.button("Save config…")).clicked() {
                let mut d = rfd::FileDialog::new().add_filter("Config", &["toml"]).set_file_name("img2gcode.toml");
                if let Some(dir) = self.config_path.as_ref().and_then(|p| p.parent()) {
                    d = d.set_directory(dir);
                }
                if let Some(p) = d.save_file() {
                    match std::fs::write(&p, self.cfg.to_toml()) {
                        Ok(()) => {
                            self.note = format!("Saved {}", p.display());
                            self.config_path = Some(p);
                        }
                        Err(e) => self.error = Some(format!("{}: {e}", p.display())),
                    }
                }
            }
            ui.separator();
            let can_export = self.shown.as_ref().is_some_and(|s| s.cfg.is_some());
            if probe("export", ui.add_enabled(can_export, egui::Button::new("Export G-code…"))).clicked() {
                self.export();
            }
            if probe("fit", ui.button("Fit view")).clicked() {
                self.fit_pending = true;
            }
            ui.separator();
            // Status text goes last so its width never moves a button.
            if self.busy {
                ui.spinner();
                ui.label("Generating…");
            } else if let Some(e) = &self.error {
                ui.colored_label(Color32::from_rgb(255, 110, 110), e);
            } else {
                ui.label(&self.note);
            }
        });
    }

    fn export(&mut self) {
        let Some(shown) = &self.shown else { return };
        let Some(cfg) = &shown.cfg else { return };
        let stem = self
            .input
            .as_ref()
            .and_then(|p| p.file_stem())
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_else(|| "text".into());
        let mut d = rfd::FileDialog::new().add_filter("G-code", &["gcode"]).set_file_name(format!("{stem}.gcode"));
        if let Some(dir) = self.input.as_ref().and_then(|p| p.parent()) {
            d = d.set_directory(dir);
        }
        if let Some(p) = d.save_file() {
            let name = self
                .input
                .as_ref()
                .and_then(|p| p.file_name())
                .map(|n| n.to_string_lossy().into_owned())
                .unwrap_or_default();
            match std::fs::write(&p, gcode::write(&shown.toolpaths, cfg, &name)) {
                Ok(()) => self.note = format!("Wrote {}", p.display()),
                Err(e) => self.error = Some(format!("{}: {e}", p.display())),
            }
        }
    }

    // ----- parameters ------------------------------------------------------

    fn params(&mut self, ui: &mut egui::Ui) -> bool {
        let mut ch = false;
        let editable = self.gcode_file.is_none();
        let size =
            self.shown.as_ref().filter(|s| s.cfg.is_some()).map(|s| {
                (s.toolpaths.extent[2] - s.toolpaths.extent[0], s.toolpaths.extent[3] - s.toolpaths.extent[1])
            });
        let px = self.shown.as_ref().and_then(|s| s.image.as_ref()).map(|i| i.rgba.dimensions());
        ui.add_enabled_ui(editable, |ui| {
            let c = &mut self.cfg;
            section(ui, "Source", true, |ui| {
                let name = self
                    .input
                    .as_ref()
                    .and_then(|p| p.file_name())
                    .map(|n| n.to_string_lossy().into_owned())
                    .unwrap_or_else(|| "(none)".into());
                ui.horizontal(|ui| {
                    // Button first: the file name's width must not move it.
                    if probe("remove image", ui.add_enabled(self.input.is_some(), egui::Button::new("Remove")))
                        .clicked()
                    {
                        self.input = None;
                        c.exclusions.clear();
                        ch = true;
                    }
                    ui.label(format!("File: {name}"));
                });
                ui.label(match (size, px) {
                    (Some((w, h)), Some((iw, ih))) => format!("{iw} × {ih} px  ->  {w:.1} × {h:.1} mm"),
                    _ => "Size: -".into(),
                });
                ui.horizontal(|ui| {
                    ch |= probe("mode fill", ui.radio_value(&mut c.mode, Mode::Fill, "Fill (colours)")).changed();
                    ch |= probe("mode edge", ui.radio_value(&mut c.mode, Mode::Edge, "Edge (strokes)")).changed();
                });
                grid(ui, "placement", |ui| {
                    ch |= num(ui, "Image scale", &mut c.image.scale, 0.01, 0.01..=100.0, "×");
                    ch |= num(ui, "Image offset X", &mut c.image.x, 1.0, -100_000.0..=100_000.0, " mm");
                    ch |= num(ui, "Image offset Y", &mut c.image.y, 1.0, -100_000.0..=100_000.0, " mm");
                });
                if probe("reset placement", ui.button("Fit image to canvas")).clicked() {
                    (c.image.x, c.image.y, c.image.scale) = (0.0, 0.0, 1.0);
                    ch = true;
                }
                ui.small("Click the image or text on the canvas to move it; drag a corner to resize.");
            });
            section(ui, "Canvas", true, |ui| {
                grid(ui, "canvas", |ui| {
                    ch |= num(ui, "Width", &mut c.canvas.width, 1.0, 1.0..=100_000.0, " mm");
                    ch |= num(ui, "Height", &mut c.canvas.height, 1.0, 1.0..=100_000.0, " mm");
                    ch |= num(ui, "Centre X", &mut c.canvas.center_x, 1.0, -100_000.0..=100_000.0, " mm");
                    ch |= num(ui, "Centre Y", &mut c.canvas.center_y, 1.0, -100_000.0..=100_000.0, " mm");
                });
                ui.small("The image is scaled to fit inside the canvas, aspect kept.");
            });
            section(ui, "Text", true, |ui| {
                let mut remove = None;
                for (i, t) in c.text.iter_mut().enumerate() {
                    let first = t.content.lines().next().unwrap_or("").chars().take(28).collect::<String>();
                    egui::CollapsingHeader::new(format!("{}. {first}", i + 1))
                        .id_salt(("text", i))
                        .default_open(true)
                        .show(ui, |ui| {
                            ch |= ui
                                .add(
                                    egui::TextEdit::multiline(&mut t.content)
                                        .desired_rows(2)
                                        .desired_width(f32::INFINITY),
                                )
                                .changed();
                            grid(ui, &format!("text{i}"), |ui| {
                                ui.label("Font");
                                ch |= font_picker(ui, i, &mut t.font, self.fonts.as_deref(), &mut self.font_filter);
                                ui.end_row();
                                ch |= num(ui, "Size", &mut t.size, 0.5, 0.5..=100_000.0, " mm");
                                ch |= num(ui, "X (centre)", &mut t.x, 1.0, -100_000.0..=100_000.0, " mm");
                                ch |= num(ui, "Y (centre)", &mut t.y, 1.0, -100_000.0..=100_000.0, " mm");
                                // Single-line letters are strokes: nothing to embolden or fill.
                                let outline = !crate::stroke_font::is_stroke(&t.font);
                                let why = "A single-line font draws each stroke once";
                                ui.label("Style");
                                ui.horizontal(|ui| {
                                    ch |= ui
                                        .add_enabled(outline, egui::Checkbox::new(&mut t.bold, "Bold"))
                                        .on_disabled_hover_text(why)
                                        .changed();
                                    ch |= ui.checkbox(&mut t.italic, "Italic").changed();
                                });
                                ui.end_row();
                                ui.label("Infill");
                                ch |= ui
                                    .add_enabled(outline, egui::Checkbox::new(&mut t.infill, "Fill the letters"))
                                    .on_disabled_hover_text(why)
                                    .changed();
                                ui.end_row();
                                ch |= num(ui, "Wrap width (0 = off)", &mut t.width, 1.0, 0.0..=100_000.0, " mm");
                                ui.label("Align");
                                ui.horizontal(|ui| {
                                    use crate::text::Align;
                                    ch |= ui.selectable_value(&mut t.align, Align::Left, "Left").changed();
                                    ch |= ui.selectable_value(&mut t.align, Align::Center, "Centre").changed();
                                    ch |= ui.selectable_value(&mut t.align, Align::Right, "Right").changed();
                                });
                                ui.end_row();
                                ch |= int(ui, "Tool", &mut t.tool, 0..=15);
                            });
                            if ui.button("Remove text").clicked() {
                                remove = Some(i);
                            }
                        });
                }
                if let Some(i) = remove {
                    c.text.remove(i);
                    ch = true;
                }
                if probe("add text", ui.button("Add text")).clicked() {
                    c.text.push(crate::text::TextItem {
                        x: c.canvas.center_x,
                        y: c.canvas.center_y,
                        size: (c.canvas.height / 8.0).max(1.0),
                        ..Default::default()
                    });
                    ch = true;
                }
                ui.small("Any installed font. Text is placed in mm, centred on X/Y; untick Infill for outlines only.");
            });
            section(ui, "Extrusion", true, |ui| {
                grid(ui, "extrusion", |ui| {
                    ch |= num(ui, "Nozzle", &mut c.extrusion.nozzle_diameter, 0.05, 0.05..=50.0, " mm");
                    ch |= num(ui, "Line width (0 = nozzle)", &mut c.extrusion.line_width, 0.05, 0.0..=50.0, " mm");
                    ch |= num(ui, "Filament", &mut c.extrusion.filament_diameter, 0.05, 0.5..=10.0, " mm");
                    ch |= num(ui, "Multiplier", &mut c.extrusion.multiplier, 0.01, 0.1..=5.0, "");
                });
            });
            section(ui, "Layers", true, |ui| {
                grid(ui, "layers", |ui| {
                    ch |= int(ui, "Layers", &mut c.layers.count, 1..=500);
                    ch |= num(ui, "Layer height", &mut c.layers.height, 0.05, 0.01..=50.0, " mm");
                    if c.mode == Mode::Fill {
                        ch |= int(ui, "Perimeters", &mut c.layers.perimeters, 0..=20);
                        ch |= num(ui, "Perimeter inset", &mut c.layers.perimeter_inset, 0.05, 0.0..=2.0, " lines");
                    }
                });
            });
            if c.mode == Mode::Fill {
                section(ui, "Colours", true, |ui| {
                    grid(ui, "colours", |ui| {
                        ch |= int(ui, "Tools (colours)", &mut c.image.tools, 1..=16);
                        ch |= int_u8(ui, "White threshold", &mut c.image.white_threshold);
                        ch |= int(ui, "Min speck area", &mut c.image.min_area, 0..=100_000);
                    });
                    // One fixed line whatever the tool count, so nothing below moves.
                    let colors = self.shown.as_ref().map(|s| s.toolpaths.tool_colors.clone()).unwrap_or_default();
                    ui.horizontal(|ui| {
                        ui.set_height(18.0);
                        if colors.is_empty() {
                            ui.label("Tool colours: -");
                        }
                        for (i, col) in colors.iter().enumerate() {
                            let (r, _) = ui.allocate_exact_size(Vec2::splat(14.0), Sense::hover());
                            ui.painter().rect_filled(r, 2.0, Color32::from_rgb(col[0], col[1], col[2]));
                            ui.label(format!("T{i}"));
                        }
                    });
                });
                section(ui, "Infill", true, |ui| {
                    grid(ui, "infill", |ui| {
                        ch |= slider(ui, "Density", &mut c.infill.density, 1.0..=100.0, " %");
                        ch |= num(ui, "Spacing (0 = density)", &mut c.infill.spacing, 0.05, 0.0..=1000.0, " mm");
                        ch |= slider(ui, "Angle", &mut c.infill.angle, -180.0..=180.0, "°");
                        ch |= num(ui, "Overlap", &mut c.infill.overlap, 0.01, 0.0..=1.0, " lines");
                        ui.label("Connect lines");
                        ch |= probe("connect", ui.checkbox(&mut c.infill.connect, "")).changed();
                        ui.end_row();
                    });
                    ui.horizontal(|ui| {
                        probe("draw exclusions", ui.toggle_value(&mut self.draw_exclusions, "Draw exclusions"));
                        if probe(
                            "clear exclusions",
                            ui.add_enabled(!c.exclusions.is_empty(), egui::Button::new("Clear")),
                        )
                        .clicked()
                        {
                            c.exclusions.clear();
                            ch = true;
                        }
                        ui.label(format!("{} zone(s)", c.exclusions.len()));
                    });
                    ui.small(
                        "Drag on the canvas to leave a rectangle without infill; right-click a zone to remove it.",
                    );
                });
            } else {
                section(ui, "Edge detection", true, |ui| {
                    let e = &mut c.edge;
                    grid(ui, "edge", |ui| {
                        ch |= num(ui, "CLAHE clip (0 = off)", &mut e.clahe_clip, 0.05, 0.0..=40.0, "");
                        ch |= int(ui, "CLAHE grid", &mut e.clahe_grid, 1..=64);
                        ch |= int(ui, "Bilateral d (0 = off)", &mut e.bilateral_d, 0..=25);
                        ch |= num(ui, "Bilateral colour σ", &mut e.bilateral_sigma_color, 1.0, 1.0..=255.0, "");
                        ch |= num(ui, "Bilateral space σ", &mut e.bilateral_sigma_space, 1.0, 1.0..=255.0, "");
                        ch |= int_u8(ui, "Canny low (0 = auto)", &mut e.canny_low);
                        ch |= int_u8(ui, "Canny high (0 = auto)", &mut e.canny_high);
                        ui.label("Scales (σ px)");
                        let mut text = e.scales.iter().map(|v| v.to_string()).collect::<Vec<_>>().join(", ");
                        if ui.add(egui::TextEdit::singleline(&mut text).desired_width(120.0)).changed() {
                            e.scales = text.split(',').filter_map(|v| v.trim().parse().ok()).collect();
                            ch = true;
                        }
                        ui.end_row();
                        ui.label("Foreground only");
                        ch |= ui.checkbox(&mut e.restrict_to_foreground, "").changed();
                        ui.end_row();
                        ch |= int(ui, "Foreground grow", &mut e.foreground_dilate, 0..=50);
                    });
                });
                section(ui, "Strokes", true, |ui| {
                    let e = &mut c.edge;
                    grid(ui, "strokes", |ui| {
                        ch |= num(ui, "Collapse", &mut e.collapse, 0.05, 0.0..=5.0, " lines");
                        ch |= num(ui, "Min separation", &mut e.min_separation, 0.05, 0.0..=5.0, " lines");
                        ch |= num(ui, "Chain gap", &mut e.chain_gap, 0.05, 0.0..=10.0, " lines");
                        ch |= num(ui, "Min length", &mut e.min_length, 0.5, 0.0..=1000.0, " px");
                        ch |= int(ui, "Tool", &mut e.tool, 0..=15);
                    });
                });
            }
            section(ui, "Smoothing", true, |ui| {
                grid(ui, "path", |ui| {
                    ch |= slider(ui, "Smoothing", &mut c.path.smoothing, 0.0..=8.0, " px");
                    ch |= num(ui, "Tolerance", &mut c.path.tolerance, 0.005, 0.0..=10.0, " mm");
                });
            });
            section(ui, "Image", false, |ui| {
                grid(ui, "image", |ui| {
                    ch |= int_u32(ui, "Resolution", &mut c.image.resolution, 64..=8000);
                    ui.label("Trim margins");
                    ch |= ui.checkbox(&mut c.image.trim, "").changed();
                    ui.end_row();
                });
            });
            section(ui, "Machine", false, |ui| {
                grid(ui, "machine", |ui| {
                    ch |= num(ui, "Print speed", &mut c.machine.print_speed, 10.0, 1.0..=100_000.0, " mm/min");
                    ch |= num(ui, "Travel speed", &mut c.machine.travel_speed, 10.0, 1.0..=100_000.0, " mm/min");
                    ch |= num(ui, "Z lift", &mut c.machine.z_lift, 0.1, 0.0..=100.0, " mm");
                });
            });
            section(ui, "G-code text", false, |ui| {
                for (label, text) in [
                    ("Start", &mut c.gcode.start),
                    ("Tool change", &mut c.gcode.tool_change),
                    ("End", &mut c.gcode.end),
                ] {
                    ui.label(label);
                    ch |= ui
                        .add(egui::TextEdit::multiline(text).code_editor().desired_rows(4).desired_width(f32::INFINITY))
                        .changed();
                }
                ui.small("Placeholders: {travel_speed} {print_speed} {tool}");
            });
        });
        ch
    }

    // ----- canvas ----------------------------------------------------------

    /// Where the image sits now: fitted to the canvas, then scaled and moved.
    /// Follows the settings live, before the toolpaths catch up.
    fn image_box(&self) -> Option<[f64; 4]> {
        let s = self.shown.as_ref()?;
        s.image.as_ref()?;
        self.input.as_ref()?;
        let e = s.toolpaths.extent;
        let aspect = (e[2] - e[0]) / (e[3] - e[1]).max(1e-9);
        let c = &self.cfg.canvas;
        let w = c.width.min(c.height * aspect) * self.cfg.image.scale;
        let h = w / aspect.max(1e-9);
        let (x, y) = (c.center_x + self.cfg.image.x, c.center_y + self.cfg.image.y);
        Some([x - w / 2.0, y - h / 2.0, x + w / 2.0, y + h / 2.0])
    }

    /// Where text item `i` sits now: its last drawn size, moved and scaled
    /// with the settings until the next result arrives.
    fn text_box(&self, i: usize) -> Option<[f64; 4]> {
        let s = self.shown.as_ref()?;
        let then = s.cfg.as_ref()?.text.get(i)?;
        let b = (*s.text_bounds.get(i)?)?;
        let now = self.cfg.text.get(i)?;
        let k = now.size / then.size.max(1e-9);
        let (hw, hh) = ((b[2] - b[0]) / 2.0 * k, (b[3] - b[1]) / 2.0 * k);
        Some([now.x - hw, now.y - hh, now.x + hw, now.y + hh])
    }

    fn box_of(&self, t: Target) -> Option<[f64; 4]> {
        match t {
            Target::Image => self.image_box(),
            Target::Text(i) => self.text_box(i),
        }
    }

    /// The topmost item under a point: text above the image.
    fn hit(&self, p: [f64; 2]) -> Option<Target> {
        let inside = |b: [f64; 4]| p[0] >= b[0] && p[0] <= b[2] && p[1] >= b[1] && p[1] <= b[3];
        (0..self.cfg.text.len())
            .rev()
            .find(|&i| self.text_box(i).is_some_and(inside))
            .map(Target::Text)
            .or_else(|| self.image_box().filter(|b| inside(*b)).map(|_| Target::Image))
    }

    fn position_of(&self, t: Target) -> [f64; 2] {
        match t {
            Target::Image => [self.cfg.image.x, self.cfg.image.y],
            Target::Text(i) => self.cfg.text.get(i).map_or([0.0, 0.0], |t| [t.x, t.y]),
        }
    }

    fn canvas(&mut self, ui: &mut egui::Ui) -> bool {
        let mut changed = false;
        let (resp, painter) = ui.allocate_painter(ui.available_size(), Sense::click_and_drag());
        let rect = resp.rect;
        painter.rect_filled(rect, 0.0, Color32::from_gray(40));
        let c = &self.cfg.canvas;
        let canvas_mm = [
            c.center_x - c.width / 2.0,
            c.center_y - c.height / 2.0,
            c.center_x + c.width / 2.0,
            c.center_y + c.height / 2.0,
        ];
        let (canvas_rect_mm, extent) = match &self.shown {
            Some(s) if s.cfg.is_some() => (canvas_mm, self.image_box().unwrap_or(s.toolpaths.extent)),
            Some(s) => (s.toolpaths.extent, s.toolpaths.extent),
            None => (canvas_mm, canvas_mm),
        };
        if self.fit_pending && rect.width() > 10.0 {
            let r = canvas_rect_mm;
            let (w, h) = ((r[2] - r[0]).max(1e-3), (r[3] - r[1]).max(1e-3));
            self.zoom = ((rect.width() as f64 / (w * 1.08)).min(rect.height() as f64 / (h * 1.08))) as f32;
            self.center = [(r[0] + r[2]) / 2.0, (r[1] + r[3]) / 2.0];
            if self.shown.is_some() {
                self.fit_pending = false;
            }
        }
        let (cx, cy, zoom) = (self.center[0], self.center[1], self.zoom as f64);
        let mid = rect.center();
        let to_screen =
            move |p: [f64; 2]| Pos2::new(mid.x + ((p[0] - cx) * zoom) as f32, mid.y - ((p[1] - cy) * zoom) as f32);
        let to_mm = move |s: Pos2| [cx + (s.x - mid.x) as f64 / zoom, cy - (s.y - mid.y) as f64 / zoom];
        let rect_of = |r: [f64; 4]| Rect::from_two_pos(to_screen([r[0], r[1]]), to_screen([r[2], r[3]]));

        // Paper: the canvas the image is fitted into.
        painter.rect_filled(rect_of(canvas_rect_mm), 0.0, Color32::WHITE);
        painter.rect_stroke(
            rect_of(canvas_rect_mm),
            0.0,
            Stroke::new(1.0, Color32::from_gray(150)),
            StrokeKind::Outside,
        );

        if self.show_image
            && let Some(tex) = self.texture_for(ui.ctx())
        {
            painter.image(
                tex,
                rect_of(extent),
                Rect::from_min_max(Pos2::ZERO, Pos2::new(1.0, 1.0)),
                Color32::from_white_alpha(90),
            );
        }

        if let Some(s) = &self.shown {
            let lw = s.cfg.as_ref().unwrap_or(&self.cfg).line_width();
            let width = if self.real_width { (lw * zoom).max(1.0) as f32 } else { 1.0 };
            let visible = rect.expand(width);
            let n = ((s.segs.len() as f32) * self.progress).round() as usize;
            let mut mesh = egui::Mesh::default();
            let mut travel = egui::Mesh::default();
            let mut last = None;
            for seg in &s.segs[..n.min(s.segs.len())] {
                if self.layer != 0 && seg.layer as usize + 1 != self.layer {
                    continue;
                }
                if seg.travel && !self.show_travel {
                    continue;
                }
                let a = to_screen([seg.a[0] as f64, seg.a[1] as f64]);
                let b = to_screen([seg.b[0] as f64, seg.b[1] as f64]);
                last = Some(b);
                if !visible.intersects(Rect::from_two_pos(a, b)) {
                    continue;
                }
                if seg.travel {
                    quad(&mut travel, a, b, 1.0, Color32::from_rgba_unmultiplied(90, 90, 90, 140), false);
                } else {
                    quad(&mut mesh, a, b, width, seg.color, true);
                }
            }
            painter.add(egui::Shape::mesh(mesh));
            painter.add(egui::Shape::mesh(travel));
            if self.progress < 1.0
                && let Some(p) = last
            {
                painter.circle_filled(p, 4.0, Color32::from_rgb(255, 60, 60));
            }
        }

        for (i, r) in self.cfg.exclusions.iter().enumerate() {
            let sr = rect_of(frac_to_mm(*r, extent));
            painter.rect_filled(sr, 0.0, Color32::from_rgba_unmultiplied(255, 160, 0, 40));
            painter.rect_stroke(sr, 0.0, Stroke::new(1.5, Color32::from_rgb(230, 140, 0)), StrokeKind::Inside);
            if resp.secondary_clicked()
                && let Some(p) = resp.interact_pointer_pos()
                && sr.contains(p)
            {
                self.cfg.exclusions.remove(i);
                changed = true;
                break;
            }
        }

        // ----- selection, moving, resizing --------------------------------
        let editable = self.gcode_file.is_none();
        let drawing = self.draw_exclusions && editable && self.cfg.mode == Mode::Fill && self.input.is_some();
        if matches!(self.selected, Some(Target::Text(i)) if i >= self.cfg.text.len())
            || (self.selected == Some(Target::Image) && self.input.is_none())
            || !editable
        {
            self.selected = None;
        }
        // Handles of the selected item: corners scale, text sides set the wrap width.
        let handles: Vec<(Handle, Pos2)> = match self.selected.and_then(|t| self.box_of(t).map(|b| (t, b))) {
            Some((t, b)) => {
                let r = rect_of(b);
                let mut v = vec![
                    (Handle::Corner, r.left_top()),
                    (Handle::Corner, r.right_top()),
                    (Handle::Corner, r.left_bottom()),
                    (Handle::Corner, r.right_bottom()),
                ];
                if matches!(t, Target::Text(_)) {
                    v.push((Handle::Side, r.left_center()));
                    v.push((Handle::Side, r.right_center()));
                }
                v
            }
            None => Vec::new(),
        };
        let handle_at = |p: Pos2| handles.iter().find(|(_, h)| h.distance(p) <= 9.0).map(|(k, _)| *k);

        // A drag is only reported once the pointer has travelled a little, so
        // decide what was grabbed from where the button went down.
        if resp.drag_started_by(egui::PointerButton::Primary)
            && let Some(p) = ui.input(|i| i.pointer.press_origin()).or(resp.interact_pointer_pos())
        {
            let pm = to_mm(p);
            self.drag = if drawing {
                Drag::Exclusion { from: pm }
            } else if let (Some(handle), Some(t)) = (handle_at(p), self.selected) {
                let b = self.box_of(t).unwrap_or([pm[0], pm[1], pm[0], pm[1]]);
                let centre = [(b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0];
                match (handle, t) {
                    (Handle::Side, Target::Text(index)) => Drag::Wrap { index, centre_x: centre[0] },
                    _ => {
                        let (orig, orig_width) = match t {
                            Target::Image => (self.cfg.image.scale, 0.0),
                            Target::Text(i) => (self.cfg.text[i].size, self.cfg.text[i].width),
                        };
                        let start = (pm[0] - centre[0]).hypot(pm[1] - centre[1]).max(1e-6);
                        Drag::Scale { target: t, centre, start, orig, orig_width }
                    }
                }
            } else if editable && let Some(t) = self.hit(pm) {
                self.selected = Some(t);
                Drag::Move { target: t, start: pm, orig: self.position_of(t) }
            } else {
                self.selected = None;
                Drag::Pan
            };
        }
        if resp.drag_started_by(egui::PointerButton::Middle) || resp.drag_started_by(egui::PointerButton::Secondary) {
            self.drag = Drag::Pan;
        }
        if resp.dragged()
            && let Some(p) = resp.interact_pointer_pos()
        {
            let pm = to_mm(p);
            match self.drag {
                Drag::Pan => {
                    let d = resp.drag_delta();
                    self.center[0] -= d.x as f64 / zoom;
                    self.center[1] += d.y as f64 / zoom;
                }
                Drag::Move { target, start, orig } => {
                    let pos = [orig[0] + pm[0] - start[0], orig[1] + pm[1] - start[1]];
                    match target {
                        Target::Image => (self.cfg.image.x, self.cfg.image.y) = (pos[0], pos[1]),
                        Target::Text(i) => (self.cfg.text[i].x, self.cfg.text[i].y) = (pos[0], pos[1]),
                    }
                    changed = true;
                    ui.ctx().set_cursor_icon(egui::CursorIcon::Grabbing);
                }
                Drag::Scale { target, centre, start, orig, orig_width } => {
                    let k = (pm[0] - centre[0]).hypot(pm[1] - centre[1]) / start;
                    match target {
                        Target::Image => self.cfg.image.scale = (orig * k).max(0.01),
                        Target::Text(i) => {
                            self.cfg.text[i].size = (orig * k).max(0.5);
                            // A wrapped block keeps its shape as it grows.
                            self.cfg.text[i].width = orig_width * k;
                        }
                    }
                    changed = true;
                }
                Drag::Wrap { index, centre_x } => {
                    let t = &mut self.cfg.text[index];
                    t.width = (2.0 * (pm[0] - centre_x).abs()).max(t.size * 0.5);
                    changed = true;
                }
                Drag::Exclusion { from } => {
                    painter.rect_stroke(
                        rect_of([from[0], from[1], pm[0], pm[1]]),
                        0.0,
                        Stroke::new(1.5, Color32::from_rgb(230, 140, 0)),
                        StrokeKind::Inside,
                    );
                }
                Drag::None => {}
            }
        }
        if resp.drag_stopped() {
            if let (Drag::Exclusion { from }, Some(p)) = (self.drag, resp.interact_pointer_pos()) {
                let to = to_mm(p);
                let f = mm_to_frac([from[0], from[1], to[0], to[1]], extent);
                if (f[2] - f[0]).abs() > 0.002 && (f[3] - f[1]).abs() > 0.002 {
                    self.cfg.exclusions.push(f);
                    changed = true;
                }
            }
            self.drag = Drag::None;
        }
        if resp.clicked()
            && !drawing
            && let Some(p) = resp.interact_pointer_pos()
        {
            self.selected = if editable { self.hit(to_mm(p)) } else { None };
        }
        if resp.double_clicked() && self.selected.is_none() {
            self.fit_pending = true;
        }
        // Delete removes the selected text, unless a text field is being edited.
        if let Some(Target::Text(i)) = self.selected
            && ui.input(|inp| inp.key_pressed(egui::Key::Delete))
            && ui.ctx().memory(|m| m.focused().is_none())
        {
            self.cfg.text.remove(i);
            self.selected = None;
            changed = true;
        }

        // Hover feedback and the selection box.
        let accent = Color32::from_rgb(40, 130, 255);
        if let Some(hover) = resp.hover_pos()
            && !drawing
            && matches!(self.drag, Drag::None)
        {
            match handle_at(hover) {
                Some(Handle::Corner) => ui.ctx().set_cursor_icon(egui::CursorIcon::ResizeNwSe),
                Some(Handle::Side) => ui.ctx().set_cursor_icon(egui::CursorIcon::ResizeHorizontal),
                None => {
                    if editable && let Some(t) = self.hit(to_mm(hover)) {
                        ui.ctx().set_cursor_icon(egui::CursorIcon::Grab);
                        if Some(t) != self.selected
                            && let Some(b) = self.box_of(t)
                        {
                            painter.rect_stroke(
                                rect_of(b),
                                0.0,
                                Stroke::new(1.0, accent.gamma_multiply(0.6)),
                                StrokeKind::Outside,
                            );
                        }
                    }
                }
            }
        }
        if let Some(t) = self.selected
            && let Some(b) = self.box_of(t)
        {
            painter.rect_stroke(rect_of(b), 0.0, Stroke::new(1.5, accent), StrokeKind::Outside);
            if let Target::Text(i) = t
                && self.cfg.text[i].width > 0.0
            {
                // The wrap width, centred on the text.
                let hw = self.cfg.text[i].width / 2.0;
                let x = self.cfg.text[i].x;
                let wr = rect_of([x - hw, b[1], x + hw, b[3]]);
                let faint = Stroke::new(1.0, accent.gamma_multiply(0.5));
                painter.line_segment([wr.left_top(), wr.left_bottom()], faint);
                painter.line_segment([wr.right_top(), wr.right_bottom()], faint);
            }
            for (kind, pos) in &handles {
                match kind {
                    Handle::Corner => {
                        let r = Rect::from_center_size(*pos, Vec2::splat(9.0));
                        painter.rect_filled(r, 1.0, Color32::WHITE);
                        painter.rect_stroke(r, 1.0, Stroke::new(1.5, accent), StrokeKind::Inside);
                    }
                    Handle::Side => {
                        painter.circle_filled(*pos, 5.0, Color32::WHITE);
                        painter.circle_stroke(*pos, 5.0, Stroke::new(1.5, accent));
                    }
                }
            }
            let hint = match t {
                Target::Image => "Drag to move · corners resize",
                Target::Text(_) => "Drag to move · corners resize · sides set the wrap width · Delete removes",
            };
            painter.text(
                rect.left_top() + Vec2::new(8.0, 8.0),
                egui::Align2::LEFT_TOP,
                hint,
                egui::FontId::proportional(13.0),
                Color32::from_gray(210),
            );
        }

        if let Some(hover) = resp.hover_pos() {
            let scroll = ui.input(|i| i.smooth_scroll_delta().y);
            let pinch = ui.input(|i| i.zoom_delta());
            let factor = (scroll / 300.0).exp() * pinch;
            if (factor - 1.0).abs() > 1e-4 {
                let before = to_mm(hover);
                self.zoom = (self.zoom * factor).clamp(1e-4, 1e4);
                let z = self.zoom as f64;
                // Keep the point under the cursor fixed.
                self.center = [before[0] - (hover.x - mid.x) as f64 / z, before[1] + (hover.y - mid.y) as f64 / z];
            }
            let mm = to_mm(hover);
            painter.text(
                rect.left_bottom() + Vec2::new(8.0, -8.0),
                egui::Align2::LEFT_BOTTOM,
                format!("{:.1}, {:.1} mm", mm[0], mm[1]),
                egui::FontId::monospace(12.0),
                Color32::from_gray(200),
            );
        }
        changed
    }

    // ----- bottom bar ------------------------------------------------------

    fn bottom_bar(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            probe("travel", ui.checkbox(&mut self.show_travel, "Travel"));
            probe("image", ui.checkbox(&mut self.show_image, "Image"));
            probe("line width", ui.checkbox(&mut self.real_width, "Line width"));
            ui.separator();
            let layers = self.shown.as_ref().map_or(0, |s| s.layers);
            ui.label("Layer");
            let combo = egui::ComboBox::from_id_salt("layer")
                .width(64.0)
                .selected_text(if self.layer == 0 { "All".to_string() } else { self.layer.to_string() })
                .show_ui(ui, |ui| {
                    ui.selectable_value(&mut self.layer, 0, "All");
                    for l in 1..=layers {
                        ui.selectable_value(&mut self.layer, l, l.to_string());
                    }
                });
            probe("layer", combo.response);
            ui.label("Progress");
            probe(
                "progress",
                ui.add_sized([220.0, 18.0], egui::Slider::new(&mut self.progress, 0.0..=1.0).show_value(false)),
            );
            ui.separator();
            // Variable-width text last, so nothing to its right can move.
            if let Some(s) = &self.shown {
                let st = &s.stats;
                ui.label(format!(
                    "{} paths · {} pts · print {:.0} mm · travel {:.0} mm · ~{:.1} min · {:.1} g · {:.0} ms",
                    st.paths, st.points, st.print_mm, st.travel_mm, st.minutes, st.filament_g, s.ms
                ));
            }
        });
    }
}

impl eframe::App for App {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        self.poll();
        let dropped: Vec<PathBuf> = ui.input(|i| {
            i.raw.dropped_files.iter().map(|f| f.path().to_path_buf()).filter(|p| !p.as_os_str().is_empty()).collect()
        });
        if let Some(p) = dropped.into_iter().next() {
            let ext = p.extension().and_then(|e| e.to_str()).map(|e| e.to_ascii_lowercase()).unwrap_or_default();
            if IMAGE_EXT.contains(&ext.as_str()) {
                self.open_image(p);
            } else {
                self.open_gcode(&p);
            }
        }
        egui::Panel::top("top").show(ui, |ui| {
            ui.add_space(4.0);
            self.top_bar(ui);
            ui.add_space(2.0);
        });
        egui::Panel::bottom("bottom").show(ui, |ui| {
            ui.add_space(2.0);
            self.bottom_bar(ui);
            ui.add_space(2.0);
        });
        let mut changed = false;
        egui::Panel::left("params").resizable(true).default_size(330.0).show(ui, |ui| {
            egui::ScrollArea::vertical().show(ui, |ui| {
                changed |= self.params(ui);
            });
        });
        egui::CentralPanel::default().frame(egui::Frame::NONE).show(ui, |ui| {
            changed |= self.canvas(ui);
        });
        if changed {
            self.regenerate();
        }
        let ctx = ui.ctx().clone();
        self.probe_step(&ctx);
    }
}

// ----- layout probe ----------------------------------------------------------
//
// With IMG2GCODE_LAYOUT_PROBE=1 the app steps itself through the states that
// change content (a result arriving, canvas size, tool count, density, an
// error, the progress slider), records every control's rectangle each frame,
// then prints any control that moved and exits.

thread_local! {
    static PROBE: std::cell::RefCell<Option<std::collections::BTreeMap<String, Vec<Rect>>>> = const { std::cell::RefCell::new(None) };
}

fn probe(name: &str, r: egui::Response) -> egui::Response {
    PROBE.with(|p| {
        if let Some(map) = p.borrow_mut().as_mut() {
            let v = map.entry(name.to_string()).or_default();
            if v.last() != Some(&r.rect) {
                v.push(r.rect);
            }
        }
    });
    r
}

struct ProbeRun {
    step: usize,
    idle_frames: usize,
}

impl App {
    fn probe_step(&mut self, ctx: &egui::Context) {
        let Some(run) = self.probe_run.as_mut() else { return };
        ctx.request_repaint();
        if self.busy {
            run.idle_frames = 0;
            return;
        }
        run.idle_frames += 1;
        if run.idle_frames < 3 {
            return;
        }
        run.idle_frames = 0;
        run.step += 1;
        if run.step == 1 {
            // Startup is over (egui's first-frame sizing pass, the first
            // result): measure from here.
            PROBE.with(|p| *p.borrow_mut() = Some(Default::default()));
        }
        match run.step {
            1 => self.cfg.canvas.width *= 3.0,
            2 => self.cfg.image.tools = 5,
            3 => self.cfg.infill.density = 20.0,
            4 => self.cfg.image.white_threshold = 0,
            5 => self.cfg.image.white_threshold = 220,
            6 => self.progress = 0.3,
            7 => self.layer = 1,
            _ => {
                PROBE.with(|p| {
                    let map = p.borrow_mut().take().unwrap_or_default();
                    let moved: Vec<_> = map.iter().filter(|(_, v)| v.iter().any(|r| r.min != v[0].min)).collect();
                    let resized: Vec<_> =
                        map.iter().filter(|(_, v)| v.iter().any(|r| r.size() != v[0].size())).collect();
                    eprintln!(
                        "layout probe: {} controls tracked, {} moved, {} changed size",
                        map.len(),
                        moved.len(),
                        resized.len()
                    );
                    for (name, v) in moved {
                        eprintln!("  moved {name}: {v:?}");
                    }
                    for (name, v) in resized {
                        eprintln!("  resized {name}: {v:?}");
                    }
                });
                ctx.send_viewport_cmd(egui::ViewportCommand::Close);
                return;
            }
        }
        if (1..=5).contains(&run.step) {
            self.regenerate();
        }
    }
}

// ----- helpers ---------------------------------------------------------------

fn flatten(tp: &Toolpaths) -> Vec<Seg> {
    let mut segs = Vec::new();
    let mut cursor: Option<[f32; 2]> = None;
    for pass in &tp.passes {
        let c = tp.tool_colors.get(pass.tool).copied().unwrap_or([0, 0, 0]);
        let base = Color32::from_rgb(c[0], c[1], c[2]);
        // Infill a little lighter than outlines so both read.
        let light = Color32::from_rgb(lerp(c[0], 255, 0.25), lerp(c[1], 255, 0.25), lerp(c[2], 255, 0.25));
        for path in &pass.paths {
            let color = if path.kind == PathKind::Infill { light } else { base };
            let pts: Vec<[f32; 2]> = path.points.iter().map(|p| [p[0] as f32, p[1] as f32]).collect();
            let Some(&first) = pts.first() else { continue };
            if let Some(c) = cursor {
                segs.push(Seg { a: c, b: first, color, travel: true, layer: pass.layer as u32 });
            }
            for w in pts.windows(2) {
                segs.push(Seg { a: w[0], b: w[1], color, travel: false, layer: pass.layer as u32 });
            }
            cursor = pts.last().copied();
        }
    }
    segs
}

fn lerp(a: u8, b: u8, t: f32) -> u8 {
    (a as f32 + (b as f32 - a as f32) * t).round() as u8
}

/// A segment as a quad `width` px wide, extended by half the width at both
/// ends (when `caps`) so consecutive segments overlap at the joints.
fn quad(mesh: &mut egui::Mesh, a: Pos2, b: Pos2, width: f32, color: Color32, caps: bool) {
    let d = b - a;
    let len = d.length();
    let dir = if len > 1e-6 { d / len } else { Vec2::X };
    let n = Vec2::new(-dir.y, dir.x) * (width / 2.0);
    let ext = if caps { dir * (width / 2.0) } else { Vec2::ZERO };
    let (a, b) = (a - ext, b + ext);
    let i = mesh.vertices.len() as u32;
    mesh.colored_vertex(a + n, color);
    mesh.colored_vertex(b + n, color);
    mesh.colored_vertex(b - n, color);
    mesh.colored_vertex(a - n, color);
    mesh.add_triangle(i, i + 1, i + 2);
    mesh.add_triangle(i, i + 2, i + 3);
}

fn frac_to_mm(f: [f64; 4], e: [f64; 4]) -> [f64; 4] {
    let (w, h) = (e[2] - e[0], e[3] - e[1]);
    [e[0] + f[0] * w, e[3] - f[1] * h, e[0] + f[2] * w, e[3] - f[3] * h]
}

fn mm_to_frac(r: [f64; 4], e: [f64; 4]) -> [f64; 4] {
    let (w, h) = ((e[2] - e[0]).max(1e-9), (e[3] - e[1]).max(1e-9));
    let fx = |x: f64| ((x - e[0]) / w).clamp(0.0, 1.0);
    let fy = |y: f64| ((e[3] - y) / h).clamp(0.0, 1.0);
    let (x0, x1) = (fx(r[0]).min(fx(r[2])), fx(r[0]).max(fx(r[2])));
    let (y0, y1) = (fy(r[1]).min(fy(r[3])), fy(r[1]).max(fy(r[3])));
    [x0, y0, x1, y1]
}

/// A font chooser: every installed family and face, filtered as you type.
fn font_picker(ui: &mut egui::Ui, id: usize, font: &mut String, fonts: Option<&[String]>, filter: &mut String) -> bool {
    let mut changed = false;
    egui::ComboBox::from_id_salt(("font", id))
        .width(190.0)
        .selected_text(font.as_str())
        // Clicking into the search box must not close the list.
        .close_behavior(egui::PopupCloseBehavior::CloseOnClickOutside)
        .show_ui(ui, |ui| {
            ui.add(egui::TextEdit::singleline(filter).hint_text("Search fonts").desired_width(200.0));
            let Some(fonts) = fonts else {
                ui.label("Loading fonts...");
                return;
            };
            let f = filter.to_lowercase();
            egui::ScrollArea::vertical().max_height(320.0).show(ui, |ui| {
                let generic = ["sans-serif", "serif", "monospace"].map(String::from);
                for name in generic.iter().chain(fonts).filter(|n| n.to_lowercase().contains(&f)) {
                    let label = if crate::stroke_font::is_stroke(name) {
                        format!("{name}  (single-line)")
                    } else {
                        name.clone()
                    };
                    if ui.selectable_label(font == name, label).clicked() {
                        *font = name.clone();
                        changed = true;
                        ui.close();
                    }
                }
            });
        });
    changed
}

fn section(ui: &mut egui::Ui, title: &str, open: bool, add: impl FnOnce(&mut egui::Ui)) {
    egui::CollapsingHeader::new(title).default_open(open).show(ui, add);
}

fn grid(ui: &mut egui::Ui, id: &str, add: impl FnOnce(&mut egui::Ui)) {
    egui::Grid::new(id).num_columns(2).spacing([12.0, 4.0]).show(ui, add);
}

fn num(
    ui: &mut egui::Ui,
    label: &str,
    v: &mut f64,
    speed: f64,
    range: std::ops::RangeInclusive<f64>,
    suffix: &str,
) -> bool {
    ui.label(label);
    let r = ui.add(egui::DragValue::new(v).speed(speed).range(range).suffix(suffix.to_string()).max_decimals(3));
    ui.end_row();
    probe(label, r).changed()
}

fn slider(ui: &mut egui::Ui, label: &str, v: &mut f64, range: std::ops::RangeInclusive<f64>, suffix: &str) -> bool {
    ui.label(label);
    let r = ui.add(egui::Slider::new(v, range).suffix(suffix.to_string()).max_decimals(1));
    ui.end_row();
    probe(label, r).changed()
}

fn int(ui: &mut egui::Ui, label: &str, v: &mut usize, range: std::ops::RangeInclusive<usize>) -> bool {
    ui.label(label);
    let r = ui.add(egui::DragValue::new(v).speed(0.1).range(range));
    ui.end_row();
    probe(label, r).changed()
}

fn int_u32(ui: &mut egui::Ui, label: &str, v: &mut u32, range: std::ops::RangeInclusive<u32>) -> bool {
    ui.label(label);
    let r = ui.add(egui::DragValue::new(v).speed(5.0).range(range).suffix(" px"));
    ui.end_row();
    probe(label, r).changed()
}

fn int_u8(ui: &mut egui::Ui, label: &str, v: &mut u8) -> bool {
    ui.label(label);
    let r = ui.add(egui::DragValue::new(v).speed(0.5).range(0..=255));
    ui.end_row();
    probe(label, r).changed()
}

#[cfg(test)]
mod tests {
    use super::*;

    const SCREEN: Vec2 = Vec2::new(1000.0, 800.0);

    /// One frame of the canvas alone, with these input events.
    fn frame(ctx: &egui::Context, app: &mut App, events: Vec<egui::Event>) {
        let raw =
            egui::RawInput { screen_rect: Some(Rect::from_min_size(Pos2::ZERO, SCREEN)), events, ..Default::default() };
        let mut out = ctx.run_ui(raw, |ui| {
            app.poll();
            if app.canvas(ui) {
                app.regenerate();
            }
        });
        // No renderer here to upload textures to.
        out.textures_delta.clear();
    }

    /// Run frames until the worker's result is in. The first text job loads
    /// every installed font, which took over 5 s on a busy CI runner.
    fn settle(ctx: &egui::Context, app: &mut App) {
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(60);
        while std::time::Instant::now() < deadline {
            frame(ctx, app, vec![]);
            if app.shown.is_some() && !app.busy {
                return;
            }
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
        panic!("no result after 60 s: {:?}", app.error);
    }

    fn screen(app: &App, p: [f64; 2]) -> Pos2 {
        let mid = (SCREEN / 2.0).to_pos2();
        let z = app.zoom as f64;
        Pos2::new(mid.x + ((p[0] - app.center[0]) * z) as f32, mid.y - ((p[1] - app.center[1]) * z) as f32)
    }

    /// Press at `from`, move to `to` in steps, release.
    fn drag(ctx: &egui::Context, app: &mut App, from: Pos2, to: Pos2) {
        let button = |pos, pressed| egui::Event::PointerButton {
            pos,
            button: egui::PointerButton::Primary,
            pressed,
            modifiers: Default::default(),
        };
        frame(ctx, app, vec![egui::Event::PointerMoved(from)]);
        frame(ctx, app, vec![button(from, true)]);
        for k in 1..=10 {
            let p = from + (to - from) * (k as f32 / 10.0);
            frame(ctx, app, vec![egui::Event::PointerMoved(p)]);
        }
        frame(ctx, app, vec![button(to, false)]);
        frame(ctx, app, vec![]);
    }

    #[test]
    fn text_on_the_canvas_can_be_moved_and_resized() {
        let ctx = egui::Context::default();
        let mut cfg = Config::default();
        cfg.text.push(crate::text::TextItem { content: "HHHH".into(), size: 100.0, ..Default::default() });
        let mut app = App::new(&ctx, cfg);
        app.regenerate();
        settle(&ctx, &mut app);
        let b = app.text_box(0).expect("text has a box");

        // Drag from the middle of the text: it follows the pointer.
        let start = screen(&app, [(b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0]);
        drag(&ctx, &mut app, start, start + Vec2::new(80.0, -40.0));
        assert_eq!(app.selected, Some(Target::Text(0)));
        let t = &app.cfg.text[0];
        let z = app.zoom as f64;
        assert!((t.x * z - 80.0).abs() < 12.0, "moved {} px right", t.x * z);
        assert!((t.y * z - 40.0).abs() < 12.0, "moved {} px up", t.y * z);
        settle(&ctx, &mut app);

        // Drag a corner outwards: the text grows.
        let b = app.text_box(0).unwrap();
        let corner = screen(&app, [b[2], b[3]]);
        drag(&ctx, &mut app, corner, corner + Vec2::new(60.0, -30.0));
        assert!(app.cfg.text[0].size > 110.0, "size {}", app.cfg.text[0].size);
        settle(&ctx, &mut app);

        // A side handle sets the wrap width.
        let b = app.text_box(0).unwrap();
        let side = screen(&app, [b[2], (b[1] + b[3]) / 2.0]);
        let x_before = app.cfg.text[0].x;
        drag(&ctx, &mut app, side, side - Vec2::new(150.0, 0.0));
        let t = &app.cfg.text[0];
        assert_eq!(t.x, x_before, "a side handle must not move the text");
        assert!(t.width > 0.0 && t.width < b[2] - b[0], "wrap width {}", t.width);

        // Pressing on empty canvas lets go of the selection.
        drag(&ctx, &mut app, Pos2::new(5.0, 5.0), Pos2::new(40.0, 40.0));
        assert_eq!(app.selected, None);
    }

    #[test]
    fn the_image_can_be_moved_and_resized_on_the_canvas() {
        let dir = std::env::temp_dir().join(format!("img2gcode-gui-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("square.png");
        let mut img = image::RgbaImage::from_pixel(200, 100, image::Rgba([255, 255, 255, 255]));
        for y in 20..80 {
            for x in 20..180 {
                img.put_pixel(x, y, image::Rgba([0, 0, 0, 255]));
            }
        }
        img.save(&path).unwrap();
        let ctx = egui::Context::default();
        let mut app = App::new(&ctx, Config::default());
        app.open_image(path);
        settle(&ctx, &mut app);
        // The image fills the canvas width; zoom out so its corners are on screen.
        app.zoom *= 0.6;
        let b = app.image_box().expect("image has a box");

        let start = screen(&app, [(b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0]);
        drag(&ctx, &mut app, start, start + Vec2::new(-50.0, 30.0));
        assert_eq!(app.selected, Some(Target::Image));
        let z = app.zoom as f64;
        assert!((app.cfg.image.x * z + 50.0).abs() < 2.0, "moved {} px", app.cfg.image.x * z);
        assert!((app.cfg.image.y * z + 30.0).abs() < 2.0, "moved {} px", app.cfg.image.y * z);
        settle(&ctx, &mut app);

        // Drag a corner inwards: the image shrinks, keeping its aspect.
        let b = app.image_box().unwrap();
        let corner = screen(&app, [b[0], b[3]]);
        let centre = screen(&app, [(b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0]);
        drag(&ctx, &mut app, corner, corner + (centre - corner) * 0.5);
        assert!((app.cfg.image.scale - 0.5).abs() < 0.05, "scale {}", app.cfg.image.scale);
        settle(&ctx, &mut app);
        let e = app.shown.as_ref().unwrap().toolpaths.extent;
        let nb = app.image_box().unwrap();
        assert!((e[2] - e[0] - (nb[2] - nb[0])).abs() < 1.0, "drawn width matches the box");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn exclusion_fractions_roundtrip_through_mm() {
        let e = [-50.0, -20.0, 50.0, 20.0];
        let f = [0.1, 0.25, 0.6, 0.75];
        let back = mm_to_frac(frac_to_mm(f, e), e);
        for k in 0..4 {
            assert!((back[k] - f[k]).abs() < 1e-12);
        }
    }
}
