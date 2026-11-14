//! Colour segmentation: background removal, K-means (or an SVG's own
//! palette), then speck removal.

use image::RgbaImage;
use rayon::prelude::*;

use crate::Error;
use crate::raster::{Grid, components};

/// Background label in a [`Segmented::labels`] grid.
pub const BACKGROUND: u8 = u8::MAX;

/// An image split into tool regions.
#[derive(Debug, Clone)]
pub struct Segmented {
    /// Tool index per pixel, or [`BACKGROUND`].
    pub labels: Grid<u8>,
    /// Representative colour per tool, darkest first.
    pub colors: Vec<[u8; 3]>,
    /// Image composited onto white, as 8-bit grey (for edge mode).
    pub gray: Grid<u8>,
    /// Non-background pixels.
    pub foreground: Grid<bool>,
}

/// Composite `rgba` onto white.
pub fn flatten(rgba: &RgbaImage) -> Vec<[u8; 3]> {
    rgba.pixels()
        .map(|p| {
            let [r, g, b, a] = p.0;
            let a = a as u32;
            let mix = |c: u8| ((c as u32 * a + 255 * (255 - a) + 127) / 255) as u8;
            [mix(r), mix(g), mix(b)]
        })
        .collect()
}

/// Split the image into tools. With a palette every foreground pixel takes
/// the nearest palette colour; otherwise K-means finds `tools` colours.
pub fn segment(
    rgba: &RgbaImage,
    palette: Option<&[[u8; 3]]>,
    tools: usize,
    white_threshold: u8,
    min_area: usize,
) -> Result<Segmented, Error> {
    let (w, h) = (rgba.width() as usize, rgba.height() as usize);
    let rgb = flatten(rgba);
    let is_bg = |c: &[u8; 3]| c.iter().all(|&v| v >= white_threshold);
    let foreground = Grid { w, h, data: rgb.iter().map(|c| !is_bg(c)).collect::<Vec<_>>() };
    let gray = Grid {
        w,
        h,
        data: rgb
            .iter()
            .map(|c| (0.299 * c[0] as f64 + 0.587 * c[1] as f64 + 0.114 * c[2] as f64).round() as u8)
            .collect(),
    };
    if !foreground.data.iter().any(|&f| f) {
        return Err(Error::NoForeground);
    }
    // Fit colours on interior pixels only: anti-aliased edge pixels are
    // blends that would otherwise grab a cluster of their own.
    let interior = |i: usize| {
        let (x, y) = (i % w, i / w);
        x > 0
            && y > 0
            && x + 1 < w
            && y + 1 < h
            && (-1isize..=1).all(|dy| {
                (-1isize..=1).all(|dx| foreground.data[(y as isize + dy) as usize * w + (x as isize + dx) as usize])
            })
    };
    let mut fg_pixels: Vec<[u8; 3]> =
        (0..w * h).filter(|&i| foreground.data[i] && interior(i)).map(|i| rgb[i]).collect();
    if fg_pixels.len() < 16 {
        fg_pixels = rgb.iter().zip(&foreground.data).filter(|(_, f)| **f).map(|(c, _)| *c).collect();
    }
    let centres: Vec<[f64; 3]> = match palette {
        Some(p) => {
            let kept: Vec<[f64; 3]> = p.iter().filter(|c| !is_bg(c)).map(|c| c.map(f64::from)).collect();
            if kept.is_empty() { kmeans(&fg_pixels, tools.max(1)) } else { kept }
        }
        None => kmeans(&fg_pixels, tools.max(1)),
    };
    // Darkest first, so a black outline is always tool 0.
    let mut order: Vec<usize> = (0..centres.len()).collect();
    order.sort_by(|&a, &b| luma(centres[a]).total_cmp(&luma(centres[b])));
    let centres: Vec<[f64; 3]> = order.iter().map(|&i| centres[i]).collect();
    let mut labels = Grid {
        w,
        h,
        data: rgb
            .par_iter()
            .zip(&foreground.data)
            .map(|(c, &f)| if f { nearest(&centres, c.map(f64::from)) as u8 } else { BACKGROUND })
            .collect(),
    };
    if min_area > 1 {
        for t in 0..centres.len() as u8 {
            let mask = labels.map(|l| l == t);
            let (comp, sizes) = components(&mask);
            for (l, c) in labels.data.iter_mut().zip(&comp.data) {
                if *c != u32::MAX && sizes[*c as usize] < min_area {
                    *l = BACKGROUND;
                }
            }
        }
    }
    let colors = centres.iter().map(|c| c.map(|v| v.round().clamp(0.0, 255.0) as u8)).collect();
    Ok(Segmented { labels, colors, gray, foreground })
}

fn luma(c: [f64; 3]) -> f64 {
    0.299 * c[0] + 0.587 * c[1] + 0.114 * c[2]
}

fn d2(a: [f64; 3], b: [f64; 3]) -> f64 {
    (a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)
}

fn nearest(centres: &[[f64; 3]], c: [f64; 3]) -> usize {
    (0..centres.len()).min_by(|&a, &b| d2(centres[a], c).total_cmp(&d2(centres[b], c))).unwrap_or(0)
}

/// K-means with k-means++ seeding on a deterministic sample of at most
/// 30k pixels. Fewer distinct colours than `k` gives fewer centres.
pub fn kmeans(pixels: &[[u8; 3]], k: usize) -> Vec<[f64; 3]> {
    let step = (pixels.len() / 30_000).max(1);
    let pts: Vec<[f64; 3]> = pixels.iter().step_by(step).map(|c| c.map(f64::from)).collect();
    let mut rng = 0x2545_f491_4f6c_dd1du64;
    let mut rand = move || {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        (rng >> 11) as f64 / (1u64 << 53) as f64
    };
    let mut centres = vec![pts[(rand() * pts.len() as f64) as usize % pts.len()]];
    let mut dmin: Vec<f64> = pts.iter().map(|p| d2(*p, centres[0])).collect();
    while centres.len() < k {
        let total: f64 = dmin.iter().sum();
        if total <= 0.0 {
            break; // every remaining pixel sits on a centre already
        }
        let mut target = rand() * total;
        let mut pick = pts.len() - 1;
        for (i, d) in dmin.iter().enumerate() {
            target -= d;
            if target <= 0.0 {
                pick = i;
                break;
            }
        }
        let c = pts[pick];
        centres.push(c);
        dmin.iter_mut().zip(&pts).for_each(|(d, p)| *d = d.min(d2(*p, c)));
    }
    for _ in 0..60 {
        let assign: Vec<usize> = pts.par_iter().map(|p| nearest(&centres, *p)).collect();
        let mut sums = vec![[0f64; 4]; centres.len()];
        for (p, &a) in pts.iter().zip(&assign) {
            sums[a][0] += p[0];
            sums[a][1] += p[1];
            sums[a][2] += p[2];
            sums[a][3] += 1.0;
        }
        let mut moved = 0.0;
        for (c, s) in centres.iter_mut().zip(&sums) {
            if s[3] > 0.0 {
                let n = [s[0] / s[3], s[1] / s[3], s[2] / s[3]];
                moved += d2(*c, n);
                *c = n;
            }
        }
        if moved < 1e-4 {
            break;
        }
    }
    centres
}

#[cfg(test)]
mod tests {
    use super::*;

    fn two_colour_image() -> RgbaImage {
        let mut img = RgbaImage::from_pixel(60, 30, image::Rgba([255, 255, 255, 255]));
        for y in 5..25 {
            for x in 5..25 {
                img.put_pixel(x, y, image::Rgba([200, 20, 20, 255]));
            }
            for x in 35..55 {
                img.put_pixel(x, y, image::Rgba([10, 10, 10, 255]));
            }
        }
        img
    }

    #[test]
    fn kmeans_finds_two_colours_darkest_first() {
        let s = segment(&two_colour_image(), None, 2, 220, 0).unwrap();
        assert_eq!(s.colors.len(), 2);
        assert!(s.colors[0][0] < 50);
        assert!(s.colors[1][0] > 150);
        assert_eq!(s.labels.get(40, 10), 0);
        assert_eq!(s.labels.get(10, 10), 1);
        assert_eq!(s.labels.get(0, 0), BACKGROUND);
    }

    #[test]
    fn specks_below_min_area_become_background() {
        let mut img = two_colour_image();
        img.put_pixel(30, 2, image::Rgba([10, 10, 10, 255]));
        let s = segment(&img, None, 2, 220, 5).unwrap();
        assert_eq!(s.labels.get(30, 2), BACKGROUND);
        assert_eq!(s.labels.get(40, 10), 0);
    }

    #[test]
    fn palette_labels_by_nearest_colour() {
        let s = segment(&two_colour_image(), Some(&[[255, 0, 0], [0, 0, 0]]), 9, 220, 0).unwrap();
        assert_eq!(s.colors, vec![[0, 0, 0], [255, 0, 0]]);
        assert_eq!(s.labels.get(10, 10), 1);
    }

    #[test]
    fn transparent_pixels_are_background() {
        let mut img = RgbaImage::from_pixel(10, 10, image::Rgba([0, 0, 0, 0]));
        img.put_pixel(5, 5, image::Rgba([0, 0, 0, 255]));
        let s = segment(&img, None, 1, 220, 0).unwrap();
        assert_eq!(s.foreground.data.iter().filter(|&&f| f).count(), 1);
    }

    #[test]
    fn blank_image_is_an_error() {
        let img = RgbaImage::from_pixel(10, 10, image::Rgba([255, 255, 255, 255]));
        assert!(matches!(segment(&img, None, 2, 220, 0), Err(Error::NoForeground)));
    }
}
