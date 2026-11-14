//! img2gcode: images, photos and SVGs to smooth multi-tool G-code.
//!
//! ```no_run
//! use img2gcode::{Config, convert, gcode};
//! let mut cfg = Config::default();
//! cfg.canvas.width = 300.0;
//! cfg.text.push(img2gcode::text::TextItem { content: "Hello".into(), y: -300.0, ..Default::default() });
//! let out = convert(Some(std::path::Path::new("logo.png")), &cfg).unwrap();
//! std::fs::write("logo.gcode", gcode::write(&out.toolpaths, &cfg, "logo.png")).unwrap();
//! ```
#![warn(missing_docs)]

pub mod config;
pub mod edge;
pub mod fill;
pub mod gcode;
pub mod geom;
#[cfg(feature = "gui")]
pub mod gui;
pub mod load;
pub mod pipeline;
pub mod preview;
pub mod raster;
pub mod segment;
pub mod text;
pub mod toolpath;

pub use config::{Config, Mode};
pub use pipeline::{Output, Pipeline, convert};
pub use toolpath::{Stats, Toolpaths};

/// Everything that can go wrong.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// A file could not be read or written.
    #[error("{0}: {1}")]
    Io(String, #[source] std::io::Error),
    /// A raster image could not be decoded.
    #[error("image: {0}")]
    Image(#[from] image::ImageError),
    /// An SVG could not be parsed.
    #[error("svg: {0}")]
    Svg(String),
    /// The configuration is invalid.
    #[error("config: {0}")]
    Config(String),
    /// A text item names a font that is not installed.
    #[error("font not found: {0} (see `img2gcode fonts`)")]
    Font(String),
    /// No image and no text.
    #[error("nothing to convert: give an image or some text")]
    NothingToDo,
    /// Every pixel counted as background.
    #[error("no foreground pixels: everything is white or transparent (try a higher image.white_threshold)")]
    NoForeground,
}

/// Distinct preview colours for tools that have no image colour.
pub fn default_tool_colors(n: usize) -> Vec<[u8; 3]> {
    const P: [[u8; 3]; 6] =
        [[230, 57, 70], [42, 157, 143], [244, 162, 97], [69, 123, 157], [142, 202, 230], [168, 218, 220]];
    (0..n).map(|i| P[i % P.len()]).collect()
}
