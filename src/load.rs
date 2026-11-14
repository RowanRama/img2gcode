//! Loading raster images and SVGs into one RGBA buffer at processing size.

use std::path::Path;

use image::{RgbaImage, imageops::FilterType};
use resvg::{tiny_skia, usvg};

use crate::Error;

/// An image ready for segmentation.
#[derive(Debug, Clone)]
pub struct Loaded {
    /// Pixels at processing resolution.
    pub rgba: RgbaImage,
    /// For an SVG drawn only in plain fills: its distinct fill colours, one
    /// tool each. `None` for rasters (colours come from K-means).
    pub palette: Option<Vec<[u8; 3]>>,
}

/// Load `path` and resample it so its longest side is `resolution` px.
/// With `trim`, background margins (white above `white` on every channel,
/// or transparent) are cropped first, so the drawing itself fills the
/// resolution and later the canvas.
pub fn load(path: &Path, resolution: u32, trim: bool, white: u8) -> Result<Loaded, Error> {
    let resolution = resolution.max(16);
    let is_svg = path.extension().and_then(|e| e.to_str()).is_some_and(|e| e.eq_ignore_ascii_case("svg"));
    if is_svg {
        let data = std::fs::read(path).map_err(|e| Error::Io(path.display().to_string(), e))?;
        load_svg(&data, path.parent(), resolution, trim.then_some(white))
    } else {
        let mut img = image::open(path)?.to_rgba8();
        if trim && let Some([x0, y0, x1, y1]) = content_box(&img, white, 2) {
            img = image::imageops::crop_imm(&img, x0, y0, x1 - x0, y1 - y0).to_image();
        }
        Ok(Loaded { rgba: resample(img, resolution), palette: None })
    }
}

/// Bounding box `[x0, y0, x1, y1)` of the non-background pixels, grown by
/// `pad` px; `None` when there are none.
pub fn content_box(img: &RgbaImage, white: u8, pad: u32) -> Option<[u32; 4]> {
    let (w, h) = img.dimensions();
    let (mut x0, mut y0, mut x1, mut y1) = (u32::MAX, u32::MAX, 0, 0);
    for (x, y, p) in img.enumerate_pixels() {
        let [r, g, b, a] = p.0;
        let mix = |c: u8| (c as u32 * a as u32 + 255 * (255 - a as u32)) / 255;
        if a >= 8 && [r, g, b].iter().any(|&c| mix(c) < white as u32) {
            (x0, y0, x1, y1) = (x0.min(x), y0.min(y), x1.max(x + 1), y1.max(y + 1));
        }
    }
    (x0 < x1).then(|| [x0.saturating_sub(pad), y0.saturating_sub(pad), (x1 + pad).min(w), (y1 + pad).min(h)])
}

fn resample(img: RgbaImage, resolution: u32) -> RgbaImage {
    let (w, h) = img.dimensions();
    let longest = w.max(h).max(1);
    if longest == resolution {
        return img;
    }
    let s = resolution as f64 / longest as f64;
    let (nw, nh) = (((w as f64 * s).round() as u32).max(1), ((h as f64 * s).round() as u32).max(1));
    let filter = if s < 1.0 { FilterType::Triangle } else { FilterType::CatmullRom };
    image::imageops::resize(&img, nw, nh, filter)
}

/// Render SVG bytes. Text is laid out with the system's fonts; relative
/// image links resolve against `dir`. With `trim = Some(white)` only the
/// drawing's bounding box is rendered, at full resolution, instead of the
/// whole page.
pub fn load_svg(data: &[u8], dir: Option<&Path>, resolution: u32, trim: Option<u8>) -> Result<Loaded, Error> {
    let opt = usvg::Options {
        resources_dir: dir.map(Path::to_path_buf),
        fontdb: crate::text::system_fonts(),
        ..Default::default()
    };
    let tree = usvg::Tree::from_data(data, &opt).map_err(|e| Error::Svg(e.to_string()))?;
    let size = tree.size();
    let s = resolution as f32 / size.width().max(size.height()).max(1e-3);
    let mut rgba = render(&tree, s, [0.0, 0.0], [size.width() * s, size.height() * s])?;
    if let Some(white) = trim
        && let Some([x0, y0, x1, y1]) = content_box(&rgba, white, 2)
    {
        // Re-render just the box, in page units, at full resolution.
        let (bx, by) = (x0 as f32 / s, y0 as f32 / s);
        let (bw, bh) = ((x1 - x0) as f32 / s, (y1 - y0) as f32 / s);
        let s2 = resolution as f32 / bw.max(bh).max(1e-3);
        rgba = render(&tree, s2, [bx, by], [bw * s2, bh * s2])?;
    }
    let mut colours = Vec::new();
    let mut images = false;
    collect_fills(tree.root(), &mut colours, &mut images);
    let palette = (!images && !colours.is_empty()).then_some(colours);
    Ok(Loaded { rgba, palette })
}

/// Render the page at `scale` with page point `origin` at the top-left,
/// into a `size` px image.
fn render(tree: &usvg::Tree, scale: f32, origin: [f32; 2], size: [f32; 2]) -> Result<RgbaImage, Error> {
    let (w, h) = ((size[0].round() as u32).max(1), (size[1].round() as u32).max(1));
    let mut pixmap = tiny_skia::Pixmap::new(w, h).ok_or_else(|| Error::Svg("empty canvas".into()))?;
    let t = tiny_skia::Transform::from_row(scale, 0.0, 0.0, scale, -origin[0] * scale, -origin[1] * scale);
    resvg::render(tree, t, &mut pixmap.as_mut());
    // tiny-skia stores premultiplied RGBA; undo it.
    let mut rgba = RgbaImage::new(w, h);
    for (dst, src) in rgba.pixels_mut().zip(pixmap.pixels()) {
        let c = src.demultiply();
        *dst = image::Rgba([c.red(), c.green(), c.blue(), c.alpha()]);
    }
    Ok(rgba)
}

fn collect_fills(group: &usvg::Group, out: &mut Vec<[u8; 3]>, images: &mut bool) {
    let add = |paint: &usvg::Paint, out: &mut Vec<[u8; 3]>, images: &mut bool| match paint {
        usvg::Paint::Color(c) => {
            let rgb = [c.red, c.green, c.blue];
            if !out.contains(&rgb) {
                out.push(rgb);
            }
        }
        // Gradients and patterns have no single colour: segment by K-means.
        _ => *images = true,
    };
    for node in group.children() {
        match node {
            usvg::Node::Group(g) => collect_fills(g, out, images),
            usvg::Node::Path(p) => {
                if let Some(f) = p.fill() {
                    add(f.paint(), out, images);
                }
                if let Some(s) = p.stroke() {
                    add(s.paint(), out, images);
                }
            }
            usvg::Node::Image(_) => *images = true,
            usvg::Node::Text(t) => collect_fills(t.flattened(), out, images),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plain_svg_gives_its_fill_colours_as_palette() {
        let svg = br##"<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 50">
            <rect x="0" y="0" width="40" height="50" fill="#ff0000"/>
            <rect x="60" y="0" width="40" height="50" fill="#000000"/>
            <circle cx="20" cy="25" r="5" fill="red"/></svg>"##;
        let l = load_svg(svg, None, 200, None).unwrap();
        assert_eq!(l.rgba.dimensions(), (200, 100));
        assert_eq!(l.palette, Some(vec![[255, 0, 0], [0, 0, 0]]));
        assert_eq!(l.rgba.get_pixel(40, 50).0, [255, 0, 0, 255]);
        assert_eq!(l.rgba.get_pixel(100, 50).0[3], 0);
    }

    #[test]
    fn svg_with_gradient_has_no_palette() {
        let svg = br##"<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 10 10">
            <linearGradient id="g"><stop offset="0" stop-color="#000"/><stop offset="1" stop-color="#f00"/></linearGradient>
            <rect width="10" height="10" fill="url(#g)"/></svg>"##;
        assert_eq!(load_svg(svg, None, 64, None).unwrap().palette, None);
    }

    #[test]
    fn trimmed_svg_renders_only_the_drawing_at_full_resolution() {
        let svg = br##"<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100">
            <rect x="40" y="45" width="20" height="10" fill="#000"/></svg>"##;
        let l = load_svg(svg, None, 200, Some(220)).unwrap();
        let (w, h) = l.rgba.dimensions();
        assert_eq!(w, 200);
        assert!((95..=110).contains(&h), "{w}x{h}");
        assert_eq!(l.rgba.get_pixel(100, h / 2).0, [0, 0, 0, 255]);
    }

    #[test]
    fn content_box_finds_dark_pixels_and_pads() {
        let mut img = RgbaImage::from_pixel(50, 40, image::Rgba([255, 255, 255, 255]));
        img.put_pixel(10, 20, image::Rgba([0, 0, 0, 255]));
        img.put_pixel(30, 25, image::Rgba([0, 0, 0, 255]));
        assert_eq!(content_box(&img, 220, 2), Some([8, 18, 33, 28]));
        assert_eq!(content_box(&RgbaImage::new(5, 5), 220, 2), None);
    }

    #[test]
    fn raster_is_resampled_to_resolution() {
        let img = RgbaImage::from_pixel(300, 100, image::Rgba([0, 0, 0, 255]));
        let out = resample(img, 600);
        assert_eq!(out.dimensions(), (600, 200));
    }
}
