//! Image in, toolpaths out, with the slow stages cached between runs so the
//! GUI can regenerate on every slider move.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Instant, SystemTime};

use crate::config::{Config, Mode};
use crate::edge::{self, DetectParams, TraceParams};
use crate::fill::{self, FillParams};
use crate::geom::Pt;
use crate::load::{self, Loaded};
use crate::raster::Grid;
use crate::segment::{self, Segmented};
use crate::toolpath::{Pass, Path as TPath, PathKind, Toolpaths};
use crate::{Error, default_tool_colors};

/// The result of one run.
#[derive(Debug, Clone)]
pub struct Output {
    /// The job.
    pub toolpaths: Toolpaths,
    /// The source image at processing resolution.
    pub image: Arc<Loaded>,
    /// The colour segmentation.
    pub segmented: Arc<Segmented>,
    /// The edge map (edge mode only).
    pub edges: Option<Arc<Grid<bool>>>,
    /// Milliseconds per stage, in order.
    pub timings: Vec<(&'static str, f64)>,
}

#[derive(Debug, Clone, PartialEq)]
struct SourceKey {
    path: PathBuf,
    modified: Option<SystemTime>,
    resolution: u32,
    trim: bool,
    white: u8,
}

#[derive(Debug, Clone, PartialEq)]
struct SegKey {
    source: SourceKey,
    tools: usize,
    white: u8,
    min_area: usize,
}

/// A converter that remembers its last image, segmentation and edge map.
#[derive(Default)]
pub struct Pipeline {
    source: Option<(SourceKey, Arc<Loaded>)>,
    seg: Option<(SegKey, Arc<Segmented>)>,
    edges: Option<(SegKey, DetectParams, Arc<Grid<bool>>)>,
}

/// Convert once, with no cache to keep.
pub fn convert(input: &Path, cfg: &Config) -> Result<Output, Error> {
    Pipeline::default().run(input, cfg)
}

impl Pipeline {
    /// Run the whole pipeline, reusing cached stages whose inputs did not change.
    pub fn run(&mut self, input: &Path, cfg: &Config) -> Result<Output, Error> {
        let mut timings = Vec::new();
        let mut t = Instant::now();
        let mut lap = |name: &'static str, timings: &mut Vec<(&'static str, f64)>| {
            timings.push((name, t.elapsed().as_secs_f64() * 1000.0));
            t = Instant::now();
        };

        let source_key = SourceKey {
            path: input.to_path_buf(),
            modified: std::fs::metadata(input).and_then(|m| m.modified()).ok(),
            resolution: cfg.image.resolution,
            trim: cfg.image.trim,
            white: cfg.image.white_threshold,
        };
        let image = match &self.source {
            Some((k, v)) if *k == source_key => v.clone(),
            _ => {
                let v = Arc::new(load::load(input, cfg.image.resolution, cfg.image.trim, cfg.image.white_threshold)?);
                self.source = Some((source_key.clone(), v.clone()));
                self.seg = None;
                self.edges = None;
                v
            }
        };
        lap("load", &mut timings);

        // Edge mode only needs the grey image and the foreground mask.
        let tools = if cfg.mode == Mode::Edge { 1 } else { cfg.image.tools.max(1) };
        let seg_key =
            SegKey { source: source_key, tools, white: cfg.image.white_threshold, min_area: cfg.image.min_area };
        let segmented = match &self.seg {
            Some((k, v)) if *k == seg_key => v.clone(),
            _ => {
                let palette = if cfg.mode == Mode::Edge { None } else { image.palette.as_deref() };
                let v = Arc::new(segment::segment(
                    &image.rgba,
                    palette,
                    tools,
                    cfg.image.white_threshold,
                    cfg.image.min_area,
                )?);
                self.seg = Some((seg_key.clone(), v.clone()));
                v
            }
        };
        lap("segment", &mut timings);

        let (w, h) = (segmented.labels.w as f64, segmented.labels.h as f64);
        let scale = (cfg.canvas.width / w).min(cfg.canvas.height / h);
        let (cx, cy) = (cfg.canvas.center_x, cfg.canvas.center_y);
        let to_mm = |p: Pt| -> Pt { [cx + (p[0] + 0.5 - w / 2.0) * scale, cy + (h / 2.0 - (p[1] + 0.5)) * scale] };
        let lw_px = cfg.line_width() / scale;
        let tol_px = cfg.path.tolerance / scale;
        let mut passes = Vec::new();
        let mut edges_out = None;
        let tool_colors;

        match cfg.mode {
            Mode::Fill => {
                let params = FillParams {
                    line_width: lw_px,
                    spacing: cfg.infill_spacing() / scale,
                    perimeters: cfg.layers.perimeters,
                    inset: cfg.layers.perimeter_inset,
                    overlap: cfg.infill.overlap,
                    smoothing: cfg.path.smoothing,
                    tolerance: tol_px,
                    connect: cfg.infill.connect,
                    exclusions: cfg.exclusions.iter().map(|r| [r[0] * w, r[1] * h, r[2] * w, r[3] * h]).collect(),
                };
                let n_tools = segmented.colors.len();
                let layers =
                    fill::build(&segmented.labels, n_tools, cfg.layers.count.max(1), cfg.infill.angle, &params);
                lap("fill", &mut timings);
                for (l, per_tool) in layers.into_iter().enumerate() {
                    for (tool, tp) in per_tool.into_iter().enumerate() {
                        let mut paths = Vec::with_capacity(tp.perimeters.len() + tp.infill.len());
                        let conv = |kind, poly: Vec<Pt>| TPath { kind, points: poly.into_iter().map(to_mm).collect() };
                        paths.extend(tp.perimeters.into_iter().map(|p| conv(PathKind::Perimeter, p)));
                        paths.extend(tp.infill.into_iter().map(|p| conv(PathKind::Infill, p)));
                        passes.push(Pass { layer: l, tool, z: cfg.layers.height * (l + 1) as f64, paths });
                    }
                }
                tool_colors = segmented.colors.clone();
            }
            Mode::Edge => {
                let e = &cfg.edge;
                let dp = DetectParams {
                    clahe_clip: e.clahe_clip,
                    clahe_grid: e.clahe_grid,
                    bilateral_d: e.bilateral_d,
                    bilateral_sigma_color: e.bilateral_sigma_color,
                    bilateral_sigma_space: e.bilateral_sigma_space,
                    low: e.canny_low,
                    high: e.canny_high,
                    scales: e.scales.clone(),
                    restrict_to_foreground: e.restrict_to_foreground,
                    foreground_dilate: e.foreground_dilate,
                };
                let edges = match &self.edges {
                    Some((k, p, v)) if *k == seg_key && *p == dp => v.clone(),
                    _ => {
                        let v = Arc::new(edge::detect(&segmented.gray, &segmented.foreground, &dp));
                        self.edges = Some((seg_key, dp, v.clone()));
                        v
                    }
                };
                lap("detect", &mut timings);
                let tp = TraceParams {
                    line_width: lw_px,
                    collapse: e.collapse,
                    min_separation: e.min_separation,
                    chain_gap: e.chain_gap,
                    min_length: e.min_length,
                    smoothing: cfg.path.smoothing,
                    tolerance: tol_px,
                };
                let strokes = edge::trace(&edges, &tp);
                lap("trace", &mut timings);
                // Strokes print once, on the first layer: retracing them on
                // every layer would stack lines on top of each other.
                passes.push(Pass {
                    layer: 0,
                    tool: e.tool,
                    z: cfg.layers.height,
                    paths: strokes
                        .into_iter()
                        .map(|s| TPath { kind: PathKind::Stroke, points: s.into_iter().map(to_mm).collect() })
                        .collect(),
                });
                edges_out = Some(edges);
                tool_colors = default_tool_colors(e.tool + 1);
            }
        }
        let extent = [cx - w * scale / 2.0, cy - h * scale / 2.0, cx + w * scale / 2.0, cy + h * scale / 2.0];
        Ok(Output { toolpaths: Toolpaths { passes, tool_colors, extent }, image, segmented, edges: edges_out, timings })
    }
}
