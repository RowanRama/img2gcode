//! The generated job: polylines in machine millimetres, by layer and tool.

use crate::geom::{Pt, dist};

/// What a path is, for the G-code `;TYPE:` comment and the preview.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PathKind {
    /// A region outline.
    Perimeter,
    /// Infill inside a region.
    Infill,
    /// An edge-mode stroke.
    Stroke,
}

/// One extruded polyline. Travel to its first point is implied.
#[derive(Debug, Clone, PartialEq)]
pub struct Path {
    /// Perimeter, infill or stroke.
    pub kind: PathKind,
    /// Points in mm, Y up.
    pub points: Vec<Pt>,
}

/// Everything one tool prints on one layer, in order.
#[derive(Debug, Clone, PartialEq)]
pub struct Pass {
    /// Layer index, 0 = first.
    pub layer: usize,
    /// Tool index.
    pub tool: usize,
    /// Nozzle height for the layer (mm).
    pub z: f64,
    /// Paths in print order.
    pub paths: Vec<Path>,
}

/// A whole job.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Toolpaths {
    /// Passes in print order (layer by layer, tool by tool).
    pub passes: Vec<Pass>,
    /// Display colour per tool.
    pub tool_colors: Vec<[u8; 3]>,
    /// Image extent on the machine `[xmin, ymin, xmax, ymax]` (mm).
    pub extent: [f64; 4],
}

/// Distances, time and filament for a job.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Stats {
    /// Extruding distance (mm).
    pub print_mm: f64,
    /// Travel distance between paths (mm).
    pub travel_mm: f64,
    /// Filament used (mm).
    pub filament_mm: f64,
    /// Filament mass at PLA density (g).
    pub filament_g: f64,
    /// Estimated time (min), excluding heating and homing.
    pub minutes: f64,
    /// Number of paths.
    pub paths: usize,
    /// Number of points.
    pub points: usize,
}

impl Toolpaths {
    /// Number of layers.
    pub fn layer_count(&self) -> usize {
        self.passes.iter().map(|p| p.layer + 1).max().unwrap_or(0)
    }

    /// Distances, time and filament. `e_per_mm` is filament per mm printed.
    pub fn stats(&self, print_speed: f64, travel_speed: f64, e_per_mm: f64, filament_diameter: f64) -> Stats {
        let mut s = Stats::default();
        let mut cursor: Option<Pt> = None;
        for pass in &self.passes {
            for path in &pass.paths {
                if let (Some(c), Some(first)) = (cursor, path.points.first()) {
                    s.travel_mm += dist(c, *first);
                }
                s.print_mm += crate::geom::length(&path.points);
                s.paths += 1;
                s.points += path.points.len();
                cursor = path.points.last().copied().or(cursor);
            }
        }
        s.filament_mm = s.print_mm * e_per_mm;
        let r = filament_diameter / 2.0;
        s.filament_g = std::f64::consts::PI * r * r * s.filament_mm * PLA_G_PER_CM3 / 1000.0;
        s.minutes = s.print_mm / print_speed.max(1.0) + s.travel_mm / travel_speed.max(1.0);
        s
    }
}

/// PLA density used for the mass estimate.
pub const PLA_G_PER_CM3: f64 = 1.24;
