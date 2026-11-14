//! Configuration: one TOML file, every key optional.

use serde::{Deserialize, Serialize};

use crate::Error;

/// The documented default configuration, as shipped.
pub const DEFAULT_TOML: &str = include_str!("../configs/default.toml");

/// How the image becomes toolpaths.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Mode {
    /// One tool per colour: perimeters plus infill.
    Fill,
    /// Edge detection traced as strokes.
    Edge,
}

/// Every setting. Field docs live in `configs/default.toml`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Config {
    /// Generation mode.
    pub mode: Mode,
    /// Infill exclusion rectangles `[x0, y0, x1, y1]` in image fractions.
    pub exclusions: Vec<[f64; 4]>,
    /// Text blocks placed on the canvas.
    pub text: Vec<crate::text::TextItem>,
    /// Print area and placement.
    pub canvas: Canvas,
    /// Speeds and Z lift.
    pub machine: Machine,
    /// Filament and line width.
    pub extrusion: Extrusion,
    /// Layers and perimeters.
    pub layers: Layers,
    /// Image loading and colour segmentation.
    pub image: ImageCfg,
    /// Infill pattern.
    pub infill: Infill,
    /// Path smoothing and simplification.
    pub path: PathCfg,
    /// Edge-mode detection and tracing.
    pub edge: Edge,
    /// Raw G-code snippets.
    pub gcode: GcodeCfg,
}

/// Print area (mm) and where its centre lands.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
#[allow(missing_docs)]
pub struct Canvas {
    pub width: f64,
    pub height: f64,
    pub center_x: f64,
    pub center_y: f64,
}

/// Speeds (mm/min) and travel lift (mm).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
#[allow(missing_docs)]
pub struct Machine {
    pub travel_speed: f64,
    pub print_speed: f64,
    pub z_lift: f64,
}

/// Filament, nozzle and line width (mm).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
#[allow(missing_docs)]
pub struct Extrusion {
    pub filament_diameter: f64,
    pub nozzle_diameter: f64,
    pub line_width: f64,
    pub multiplier: f64,
}

/// Layer count and height, perimeter loops.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
#[allow(missing_docs)]
pub struct Layers {
    pub count: usize,
    pub height: f64,
    pub perimeters: usize,
    pub perimeter_inset: f64,
}

/// Image loading and colour segmentation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
#[allow(missing_docs)]
pub struct ImageCfg {
    pub tools: usize,
    pub white_threshold: u8,
    pub min_area: usize,
    pub resolution: u32,
    pub trim: bool,
}

/// Infill settings.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
#[allow(missing_docs)]
pub struct Infill {
    pub density: f64,
    pub spacing: f64,
    pub angle: f64,
    pub connect: bool,
    pub overlap: f64,
}

/// Path smoothing (px) and simplification tolerance (mm).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
#[allow(missing_docs)]
pub struct PathCfg {
    pub smoothing: f64,
    pub tolerance: f64,
}

/// Edge-mode settings.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
#[allow(missing_docs)]
pub struct Edge {
    pub clahe_clip: f64,
    pub clahe_grid: usize,
    pub bilateral_d: usize,
    pub bilateral_sigma_color: f64,
    pub bilateral_sigma_space: f64,
    pub canny_low: u8,
    pub canny_high: u8,
    pub scales: Vec<f64>,
    pub restrict_to_foreground: bool,
    pub foreground_dilate: usize,
    pub collapse: f64,
    pub min_separation: f64,
    pub chain_gap: f64,
    pub min_length: f64,
    pub tool: usize,
}

/// G-code text emitted verbatim (with placeholders).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
#[allow(missing_docs)]
pub struct GcodeCfg {
    pub start: String,
    pub tool_change: String,
    pub end: String,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            mode: Mode::Fill,
            exclusions: Vec::new(),
            text: Vec::new(),
            canvas: Canvas::default(),
            machine: Machine::default(),
            extrusion: Extrusion::default(),
            layers: Layers::default(),
            image: ImageCfg::default(),
            infill: Infill::default(),
            path: PathCfg::default(),
            edge: Edge::default(),
            gcode: GcodeCfg::default(),
        }
    }
}

impl Default for Canvas {
    fn default() -> Self {
        Self { width: 1000.0, height: 700.0, center_x: 0.0, center_y: 0.0 }
    }
}

impl Default for Machine {
    fn default() -> Self {
        Self { travel_speed: 7800.0, print_speed: 1800.0, z_lift: 1.0 }
    }
}

impl Default for Extrusion {
    fn default() -> Self {
        Self { filament_diameter: 1.75, nozzle_diameter: 4.0, line_width: 0.0, multiplier: 1.0 }
    }
}

impl Default for Layers {
    fn default() -> Self {
        Self { count: 1, height: 3.0, perimeters: 1, perimeter_inset: 0.5 }
    }
}

impl Default for ImageCfg {
    fn default() -> Self {
        Self { tools: 2, white_threshold: 220, min_area: 10, resolution: 1200, trim: true }
    }
}

impl Default for Infill {
    fn default() -> Self {
        Self { density: 100.0, spacing: 0.0, angle: -45.0, connect: true, overlap: 0.25 }
    }
}

impl Default for PathCfg {
    fn default() -> Self {
        Self { smoothing: 1.0, tolerance: 0.05 }
    }
}

impl Default for Edge {
    fn default() -> Self {
        Self {
            clahe_clip: 1.5,
            clahe_grid: 8,
            bilateral_d: 5,
            bilateral_sigma_color: 35.0,
            bilateral_sigma_space: 35.0,
            canny_low: 0,
            canny_high: 0,
            scales: vec![0.2, 1.0, 2.0],
            restrict_to_foreground: false,
            foreground_dilate: 3,
            collapse: 0.0,
            min_separation: 0.8,
            chain_gap: 1.5,
            min_length: 4.0,
            tool: 0,
        }
    }
}

impl Default for GcodeCfg {
    fn default() -> Self {
        Self {
            start: "M104 S200 ; set temperature\nG28 ; home all axes\nG1 Z5 F{travel_speed} ; lift nozzle\n\
                    M109 S200 ; set temperature and wait\nG21 ; millimetres\nG90 ; absolute coordinates\n\
                    M82 ; absolute extrusion\nG92 E0\nM107\n"
                .into(),
            tool_change: "T{tool}\nG92 E0\n".into(),
            end: "G1 Z10 F3000 ; raise Z\nG28 X0 Y0 ; home X Y\nM104 S0 ; heater off\nM84 ; motors off\n".into(),
        }
    }
}

impl Config {
    /// Parse a TOML document; missing keys take their defaults.
    pub fn from_toml(text: &str) -> Result<Self, Error> {
        toml::from_str(text).map_err(|e| Error::Config(e.to_string()))
    }

    /// Serialise to TOML (without the comments of the shipped default).
    pub fn to_toml(&self) -> String {
        // Serialising plain structs of numbers, strings and arrays cannot fail.
        toml::to_string_pretty(self).expect("config serialises")
    }

    /// Apply `key.path=value` overrides, e.g. `infill.density=60`.
    /// The value is read as TOML (`true`, `1.5`, `[0.5, 1]`), or as a plain
    /// string when it is not valid TOML (`mode=edge`).
    pub fn with_overrides(&self, sets: &[String]) -> Result<Self, Error> {
        if sets.is_empty() {
            return Ok(self.clone());
        }
        let mut root = toml::Value::try_from(self).map_err(|e| Error::Config(e.to_string()))?;
        for set in sets {
            let (key, raw) =
                set.split_once('=').ok_or_else(|| Error::Config(format!("--set {set}: expected key=value")))?;
            let value = parse_value(raw.trim());
            let mut node = &mut root;
            let parts: Vec<&str> = key.trim().split('.').collect();
            for (i, part) in parts.iter().enumerate() {
                let table = node
                    .as_table_mut()
                    .ok_or_else(|| Error::Config(format!("--set {set}: {part} is not a section")))?;
                if i + 1 == parts.len() {
                    table.insert((*part).to_string(), value.clone());
                    break;
                }
                node = table
                    .get_mut(*part)
                    .ok_or_else(|| Error::Config(format!("--set {set}: no section named {part}")))?;
            }
        }
        root.try_into().map_err(|e: toml::de::Error| Error::Config(e.to_string()))
    }

    /// Printed line width (mm): `line_width`, or the nozzle when that is 0.
    pub fn line_width(&self) -> f64 {
        if self.extrusion.line_width > 0.0 { self.extrusion.line_width } else { self.extrusion.nozzle_diameter }
    }

    /// Infill line spacing (mm).
    pub fn infill_spacing(&self) -> f64 {
        if self.infill.spacing > 0.0 {
            self.infill.spacing
        } else {
            self.line_width() / (self.infill.density.clamp(1.0, 100.0) / 100.0)
        }
    }
}

fn parse_value(raw: &str) -> toml::Value {
    match toml::from_str::<toml::Table>(&format!("v = {raw}")) {
        Ok(mut t) => t.remove("v").unwrap_or_else(|| toml::Value::String(raw.into())),
        Err(_) => toml::Value::String(raw.into()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shipped_default_file_matches_code_defaults() {
        assert_eq!(Config::from_toml(DEFAULT_TOML).unwrap(), Config::default());
    }

    #[test]
    fn missing_keys_take_defaults() {
        let cfg = Config::from_toml("[canvas]\nwidth = 300.0\n").unwrap();
        assert_eq!(cfg.canvas.width, 300.0);
        assert_eq!(cfg.canvas.height, 700.0);
        assert_eq!(cfg.infill, Infill::default());
    }

    #[test]
    fn unknown_key_is_refused() {
        assert!(Config::from_toml("[canvas]\nwidht = 300.0\n").is_err());
    }

    #[test]
    fn overrides_set_numbers_bools_strings_and_arrays() {
        let cfg = Config::default()
            .with_overrides(&[
                "infill.density=60".into(),
                "infill.connect=false".into(),
                "mode=edge".into(),
                "edge.scales=[1, 2.5]".into(),
                "canvas.width=250".into(),
            ])
            .unwrap();
        assert_eq!(cfg.infill.density, 60.0);
        assert!(!cfg.infill.connect);
        assert_eq!(cfg.mode, Mode::Edge);
        assert_eq!(cfg.edge.scales, vec![1.0, 2.5]);
        assert_eq!(cfg.canvas.width, 250.0);
    }

    #[test]
    fn override_of_unknown_section_is_an_error() {
        assert!(Config::default().with_overrides(&["nope.x=1".into()]).is_err());
        assert!(Config::default().with_overrides(&["infill.nope=1".into()]).is_err());
    }

    #[test]
    fn spacing_follows_density_unless_set() {
        let mut cfg = Config::default();
        cfg.infill.density = 50.0;
        assert!((cfg.infill_spacing() - 8.0).abs() < 1e-9);
        cfg.infill.spacing = 3.0;
        assert_eq!(cfg.infill_spacing(), 3.0);
    }

    #[test]
    fn toml_roundtrip() {
        let cfg = Config::default();
        assert_eq!(Config::from_toml(&cfg.to_toml()).unwrap(), cfg);
    }
}
