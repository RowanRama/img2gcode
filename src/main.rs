use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::Instant;

use clap::{Parser, Subcommand, ValueEnum};
use img2gcode::{Config, Mode, gcode};

const EXAMPLES: &str = "\
Examples:
  img2gcode                                  open the GUI
  img2gcode gui logo.png                     open the GUI on an image
  img2gcode convert logo.png                 write logo.gcode next to it
  img2gcode convert logo.svg -o out.gcode --size 300x200
  img2gcode convert face.jpg --mode edge --set edge.collapse=1.0
  img2gcode convert logo.png -c my.toml --set infill.density=40 --set layers.count=3
  img2gcode view out.gcode                   preview an existing file
  img2gcode config > my.toml                 start a config file";

#[derive(Parser)]
#[command(name = "img2gcode", version, about = "Images, photos and SVGs to smooth multi-tool G-code", after_help = EXAMPLES)]
struct Cli {
    #[command(subcommand)]
    cmd: Option<Cmd>,
}

#[derive(Subcommand)]
enum Cmd {
    /// Convert an image (PNG, JPG, BMP, GIF, WebP, TIFF) or SVG to G-code.
    Convert(ConvertArgs),
    /// Open the GUI (the default with no command).
    Gui {
        /// Image or SVG to open.
        input: Option<PathBuf>,
        #[command(flatten)]
        opts: ConfigOpts,
    },
    /// Preview an existing G-code file.
    View {
        /// G-code file.
        file: PathBuf,
    },
    /// Print the default configuration file.
    Config,
}

#[derive(clap::Args)]
struct ConvertArgs {
    /// Image or SVG to convert.
    input: PathBuf,
    /// Output file [default: the input with a .gcode extension].
    #[arg(short, long)]
    output: Option<PathBuf>,
    #[command(flatten)]
    opts: ConfigOpts,
    /// Also save a picture of the toolpaths, lines at printed width.
    #[arg(long, value_name = "PNG")]
    preview: Option<PathBuf>,
    /// Save the detected edge map as a PNG (edge mode), for tuning.
    #[arg(long, value_name = "PNG")]
    debug_edges: Option<PathBuf>,
    /// Print nothing but errors.
    #[arg(short, long)]
    quiet: bool,
}

#[derive(clap::Args, Default)]
struct ConfigOpts {
    /// Config file (TOML); see `img2gcode config`.
    #[arg(short, long)]
    config: Option<PathBuf>,
    /// Generation mode.
    #[arg(long)]
    mode: Option<ModeArg>,
    /// Number of tools (colours).
    #[arg(long)]
    tools: Option<usize>,
    /// Print area in mm, WIDTHxHEIGHT or one number for a square; the image
    /// is scaled to fit.
    #[arg(long, value_name = "WxH")]
    size: Option<String>,
    /// Set any config key, e.g. --set infill.density=60 (repeatable).
    #[arg(long = "set", value_name = "KEY=VALUE")]
    set: Vec<String>,
}

#[derive(Clone, Copy, ValueEnum)]
enum ModeArg {
    Fill,
    Edge,
}

impl ConfigOpts {
    fn build(&self) -> Result<Config, String> {
        let mut cfg = match &self.config {
            Some(p) => {
                let text = std::fs::read_to_string(p).map_err(|e| format!("{}: {e}", p.display()))?;
                Config::from_toml(&text).map_err(|e| format!("{}: {e}", p.display()))?
            }
            None => Config::default(),
        };
        if let Some(m) = self.mode {
            cfg.mode = match m {
                ModeArg::Fill => Mode::Fill,
                ModeArg::Edge => Mode::Edge,
            };
        }
        if let Some(t) = self.tools {
            cfg.image.tools = t;
        }
        if let Some(s) = &self.size {
            let (w, h) = parse_size(s)?;
            cfg.canvas.width = w;
            cfg.canvas.height = h;
        }
        cfg.with_overrides(&self.set).map_err(|e| e.to_string())
    }
}

fn parse_size(s: &str) -> Result<(f64, f64), String> {
    let bad = || format!("--size {s}: expected WIDTHxHEIGHT in mm, e.g. 300x200");
    let mut parts = s.split(['x', 'X', '*']);
    let w: f64 = parts.next().and_then(|v| v.trim().parse().ok()).ok_or_else(bad)?;
    let h: f64 = match parts.next() {
        Some(v) => v.trim().parse().map_err(|_| bad())?,
        None => w,
    };
    if w <= 0.0 || h <= 0.0 || parts.next().is_some() {
        return Err(bad());
    }
    Ok((w, h))
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    let result = match cli.cmd {
        Some(Cmd::Convert(args)) => convert(&args),
        Some(Cmd::Config) => {
            print!("{}", img2gcode::config::DEFAULT_TOML);
            Ok(())
        }
        Some(Cmd::Gui { input, opts }) => opts.build().and_then(|cfg| run_gui(cfg, input, None)),
        Some(Cmd::View { file }) => run_gui(Config::default(), None, Some(file)),
        None => run_gui(Config::default(), None, None),
    };
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
    }
}

#[cfg(feature = "gui")]
fn run_gui(cfg: Config, input: Option<PathBuf>, gcode_file: Option<PathBuf>) -> Result<(), String> {
    img2gcode::gui::run(cfg, input, gcode_file)
}

#[cfg(not(feature = "gui"))]
fn run_gui(_: Config, _: Option<PathBuf>, _: Option<PathBuf>) -> Result<(), String> {
    Err("this build has no GUI; use `img2gcode convert` (see --help)".into())
}

fn convert(args: &ConvertArgs) -> Result<(), String> {
    let cfg = args.opts.build()?;
    let started = Instant::now();
    let out = img2gcode::convert(&args.input, &cfg).map_err(|e| format!("{}: {e}", args.input.display()))?;
    let output = args.output.clone().unwrap_or_else(|| args.input.with_extension("gcode"));
    let name = args.input.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
    let text = gcode::write(&out.toolpaths, &cfg, &name);
    write_file(&output, text.as_bytes())?;
    if let Some(p) = &args.preview {
        let img = img2gcode::preview::render(&out.toolpaths, cfg.line_width(), 1600, false);
        img.save(p).map_err(|e| format!("{}: {e}", p.display()))?;
    }
    if let (Some(p), Some(edges)) = (&args.debug_edges, &out.edges) {
        let img = image::GrayImage::from_fn(edges.w as u32, edges.h as u32, |x, y| {
            image::Luma([if edges.get(x as usize, y as usize) { 255 } else { 0 }])
        });
        img.save(p).map_err(|e| format!("{}: {e}", p.display()))?;
    }
    if !args.quiet {
        let s = gcode::stats(&out.toolpaths, &cfg);
        let e = out.toolpaths.extent;
        println!("{} -> {}", args.input.display(), output.display());
        println!(
            "  {:.1} x {:.1} mm, {:?} mode, {} layer(s)",
            e[2] - e[0],
            e[3] - e[1],
            cfg.mode,
            cfg.layers.count.max(1)
        );
        if cfg.mode == Mode::Fill {
            for (i, c) in out.toolpaths.tool_colors.iter().enumerate() {
                let paths: usize = out.toolpaths.passes.iter().filter(|p| p.tool == i).map(|p| p.paths.len()).sum();
                println!("  T{i}: #{:02x}{:02x}{:02x}  {paths} paths", c[0], c[1], c[2]);
            }
        }
        println!(
            "  {} paths, {} points; print {:.0} mm, travel {:.0} mm; ~{:.1} min; filament {:.0} mm ({:.1} g PLA)",
            s.paths, s.points, s.print_mm, s.travel_mm, s.minutes, s.filament_mm, s.filament_g
        );
        println!("  done in {:.0} ms", started.elapsed().as_secs_f64() * 1000.0);
    }
    Ok(())
}

fn write_file(path: &Path, bytes: &[u8]) -> Result<(), String> {
    if let Some(dir) = path.parent().filter(|d| !d.as_os_str().is_empty()) {
        std::fs::create_dir_all(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    }
    std::fs::write(path, bytes).map_err(|e| format!("{}: {e}", path.display()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn size_accepts_wxh_and_a_single_number() {
        assert_eq!(parse_size("300x200").unwrap(), (300.0, 200.0));
        assert_eq!(parse_size("150").unwrap(), (150.0, 150.0));
        assert!(parse_size("0x5").is_err());
        assert!(parse_size("axb").is_err());
    }
}
