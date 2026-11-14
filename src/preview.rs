//! Render a job to a PNG-ready image, lines drawn at their printed width.

use image::RgbaImage;
use resvg::tiny_skia::{Color, LineCap, LineJoin, Paint, PathBuilder, Pixmap, Stroke, Transform};

use crate::toolpath::{PathKind, Toolpaths};

/// Draw `tp` with lines `line_width` mm wide; the longer side of the
/// drawing is `size` px. Travel moves are thin grey lines when `travel`.
pub fn render(tp: &Toolpaths, line_width: f64, size: u32, travel: bool) -> RgbaImage {
    let e = tp.extent;
    let margin = line_width;
    let (w_mm, h_mm) = (e[2] - e[0] + 2.0 * margin, e[3] - e[1] + 2.0 * margin);
    let s = size as f64 / w_mm.max(h_mm).max(1e-6);
    let (w, h) = (((w_mm * s).ceil() as u32).max(1), ((h_mm * s).ceil() as u32).max(1));
    // Size is at least 1x1, so the pixmap exists.
    let mut pm = Pixmap::new(w, h).expect("pixmap");
    pm.fill(Color::WHITE);
    let to_px = |p: [f64; 2]| (((p[0] - e[0] + margin) * s) as f32, ((e[3] + margin - p[1]) * s) as f32);
    let stroke = Stroke {
        width: (line_width * s) as f32,
        line_cap: LineCap::Round,
        line_join: LineJoin::Round,
        ..Default::default()
    };
    let thin = Stroke { width: 1.0, ..Default::default() };
    let mut travel_pb = PathBuilder::new();
    let mut cursor: Option<(f32, f32)> = None;
    for pass in &tp.passes {
        let c = tp.tool_colors.get(pass.tool).copied().unwrap_or([0, 0, 0]);
        for path in &pass.paths {
            let mut pb = PathBuilder::new();
            for (i, p) in path.points.iter().enumerate() {
                let (x, y) = to_px(*p);
                if i == 0 {
                    pb.move_to(x, y);
                    if let Some((cx, cy)) = cursor {
                        travel_pb.move_to(cx, cy);
                        travel_pb.line_to(x, y);
                    }
                } else {
                    pb.line_to(x, y);
                }
            }
            if let Some(last) = path.points.last() {
                cursor = Some(to_px(*last));
            }
            let mut paint = Paint::default();
            // Infill a touch lighter than outlines so both read.
            let k = if path.kind == PathKind::Infill { 0.8 } else { 1.0 };
            let f = |v: u8| ((v as f32 * k) + 255.0 * (1.0 - k)) as u8;
            paint.set_color_rgba8(f(c[0]), f(c[1]), f(c[2]), 255);
            paint.anti_alias = true;
            if let Some(p) = pb.finish() {
                pm.stroke_path(&p, &paint, &stroke, Transform::identity(), None);
            }
        }
    }
    if travel && let Some(p) = travel_pb.finish() {
        let mut paint = Paint::default();
        paint.set_color_rgba8(120, 120, 120, 160);
        pm.stroke_path(&p, &paint, &thin, Transform::identity(), None);
    }
    let mut out = RgbaImage::new(w, h);
    for (dst, src) in out.pixels_mut().zip(pm.pixels()) {
        let c = src.demultiply();
        *dst = image::Rgba([c.red(), c.green(), c.blue(), c.alpha()]);
    }
    out
}
