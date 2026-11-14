//! Fill mode: per tool, perimeters plus infill, cut from a smoothed signed
//! distance field so every point lands off the pixel grid.
//!
//! The field is positive inside the tool's region and measures pixels to its
//! edge. A perimeter is the iso-line at its inset; an infill line ends where
//! the field crosses the infill level. Blurring the field (path smoothing)
//! is what turns pixel stair-steps into smooth curves.

use rayon::prelude::*;

use crate::geom::{self, Pt, dist};
use crate::raster::{Grid, gaussian_blur, signed_distance};
use crate::segment::BACKGROUND;

/// Fill-mode settings in pixel units.
#[derive(Debug, Clone)]
pub struct FillParams {
    /// Printed line width.
    pub line_width: f64,
    /// Infill line spacing.
    pub spacing: f64,
    /// Perimeter loops.
    pub perimeters: usize,
    /// Outer perimeter centre, in line widths inside the edge.
    pub inset: f64,
    /// Infill overlap with the last perimeter, in line widths.
    pub overlap: f64,
    /// Gaussian sigma applied to the distance field.
    pub smoothing: f64,
    /// Douglas-Peucker tolerance.
    pub tolerance: f64,
    /// Join neighbouring infill lines.
    pub connect: bool,
    /// Infill exclusion rectangles `[x0, y0, x1, y1]` in pixels.
    pub exclusions: Vec<[f64; 4]>,
    /// Fill inside the perimeters; `false` gives outlines only.
    pub infill: bool,
}

/// One tool's paths for one layer, in image pixels.
#[derive(Debug, Clone, Default)]
pub struct ToolPaths {
    /// Closed loops (last point repeats the first).
    pub perimeters: Vec<Vec<Pt>>,
    /// Infill polylines.
    pub infill: Vec<Vec<Pt>>,
}

/// A tool's region as a padded, smoothed signed distance field.
pub struct ToolField {
    field: Grid<f32>,
    pad: usize,
}

impl ToolField {
    /// Build the field for `tool`, or `None` when the tool has no pixels.
    pub fn new(labels: &Grid<u8>, tool: u8, smoothing: f64) -> Option<Self> {
        let mask = labels.map(|l| l == tool && l != BACKGROUND);
        if !mask.data.iter().any(|&b| b) {
            return None;
        }
        // Enough empty border that blurring never reaches the edge and
        // every iso-line closes.
        let pad = (smoothing * 3.0).ceil() as usize + 2;
        let sdf = signed_distance(&mask.padded(pad, false));
        Some(Self { field: gaussian_blur(&sdf, smoothing), pad })
    }

    /// Perimeter loops and infill for one layer. `angle` is the infill
    /// direction in degrees, counter-clockwise from +X with Y up.
    pub fn paths(&self, p: &FillParams, angle: f64) -> ToolPaths {
        let off = self.pad as f64;
        let to_image = |poly: Vec<Pt>| -> Vec<Pt> { poly.into_iter().map(|q| [q[0] - off, q[1] - off]).collect() };
        let mut perimeters = Vec::new();
        for k in 0..p.perimeters {
            let level = ((p.inset + k as f64) * p.line_width) as f32;
            for l in geom::iso_lines(&self.field, level) {
                let s = geom::simplify(&l, p.tolerance);
                if s.len() >= 3 {
                    perimeters.push(to_image(s));
                }
            }
        }
        if !p.infill {
            return ToolPaths { perimeters, infill: Vec::new() };
        }
        let infill_level = if p.perimeters > 0 {
            (p.inset + p.perimeters as f64 - p.overlap) * p.line_width
        } else {
            p.inset * p.line_width
        };
        let field = if p.exclusions.is_empty() { None } else { Some(self.excluded(&p.exclusions)) };
        let field = field.as_ref().unwrap_or(&self.field);
        let lines = scan(field, infill_level as f32, p.spacing, angle);
        let slack = (infill_level - 0.5 * p.line_width) as f32;
        let polys = if p.connect {
            connect(lines, field, slack, 2.0 * p.spacing)
        } else {
            lines.into_iter().flatten().map(|(a, b)| vec![a, b]).collect()
        };
        let infill = polys.into_iter().map(|l| to_image(geom::simplify(&l, p.tolerance))).collect();
        ToolPaths { perimeters, infill }
    }

    fn excluded(&self, rects: &[[f64; 4]]) -> Grid<f32> {
        let mut f = self.field.clone();
        let off = self.pad as f64;
        for r in rects {
            let x0 = ((r[0].min(r[2]) + off).floor().max(0.0) as usize).min(f.w);
            let x1 = ((r[0].max(r[2]) + off).ceil().max(0.0) as usize).min(f.w);
            let y0 = ((r[1].min(r[3]) + off).floor().max(0.0) as usize).min(f.h);
            let y1 = ((r[1].max(r[3]) + off).ceil().max(0.0) as usize).min(f.h);
            for y in y0..y1 {
                for x in x0..x1 {
                    f.set(x, y, -1e6);
                }
            }
        }
        f
    }
}

/// Infill segments, one list per scan line in scan order, each list ordered
/// along the line.
fn scan(field: &Grid<f32>, level: f32, spacing: f64, angle: f64) -> Vec<Vec<(Pt, Pt)>> {
    let th = angle.to_radians();
    // Image Y points down, so a counter-clockwise world angle is -th here.
    let d = [th.cos(), -th.sin()];
    let n = [th.sin(), th.cos()];
    let corners = [
        [0.0, 0.0],
        [field.w as f64 - 1.0, 0.0],
        [0.0, field.h as f64 - 1.0],
        [field.w as f64 - 1.0, field.h as f64 - 1.0],
    ];
    let proj = |v: [f64; 2], p: Pt| v[0] * p[0] + v[1] * p[1];
    let (nmin, nmax) = corners.iter().fold((f64::MAX, f64::MIN), |(a, b), &c| (a.min(proj(n, c)), b.max(proj(n, c))));
    let (tmin, tmax) = corners.iter().fold((f64::MAX, f64::MIN), |(a, b), &c| (a.min(proj(d, c)), b.max(proj(d, c))));
    let spacing = spacing.max(0.25);
    // Lines sit on a fixed lattice so every tool's infill lines up.
    let k0 = (nmin / spacing - 0.5).floor() as i64;
    let k1 = (nmax / spacing - 0.5).ceil() as i64;
    let step = 0.5;
    let steps = ((tmax - tmin) / step).ceil() as usize + 1;
    (k0..=k1)
        .into_par_iter()
        .map(|k| {
            let o = (k as f64 + 0.5) * spacing;
            let at = |t: f64| [o * n[0] + t * d[0], o * n[1] + t * d[1]];
            let val = |t: f64| {
                let p = at(t);
                field.sample(p[0], p[1], -1e6)
            };
            let mut segs = Vec::new();
            let mut prev_t = tmin;
            let mut prev_v = val(tmin);
            let mut start: Option<f64> = (prev_v >= level).then_some(tmin);
            for i in 1..=steps {
                let t = tmin + i as f64 * step;
                let v = val(t);
                if (prev_v >= level) != (v >= level) {
                    let tc = prev_t + (t - prev_t) * ((level - prev_v) / (v - prev_v)) as f64;
                    match start.take() {
                        Some(s) if tc - s > 0.25 => segs.push((at(s), at(tc))),
                        Some(_) => {}
                        None => start = Some(tc),
                    }
                }
                prev_t = t;
                prev_v = v;
            }
            segs
        })
        .filter(|s: &Vec<(Pt, Pt)>| !s.is_empty())
        .collect()
}

/// Join segments of consecutive scan lines into serpentines where the
/// connector is short and stays inside the region.
fn connect(lines: Vec<Vec<(Pt, Pt)>>, field: &Grid<f32>, slack: f32, max_gap: f64) -> Vec<Vec<Pt>> {
    let mut chains: Vec<Vec<Pt>> = Vec::new();
    let mut open: Vec<usize> = Vec::new();
    let inside = |a: Pt, b: Pt| {
        (1..4).all(|i| {
            let t = i as f64 / 4.0;
            field.sample(a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, -1e6) >= slack
        })
    };
    for segs in lines {
        let mut used = vec![false; segs.len()];
        let mut next_open = Vec::with_capacity(segs.len());
        for ci in open {
            // A chain always holds at least one segment.
            let e = *chains[ci].last().expect("non-empty chain");
            let mut best: Option<(f64, usize, bool)> = None;
            for (j, &(a, b)) in segs.iter().enumerate() {
                if used[j] {
                    continue;
                }
                let (da, db) = (dist(a, e), dist(b, e));
                let (dmin, from_a) = if da <= db { (da, true) } else { (db, false) };
                if dmin <= max_gap && best.is_none_or(|b| dmin < b.0) {
                    best = Some((dmin, j, from_a));
                }
            }
            if let Some((_, j, from_a)) = best {
                let (a, b) = segs[j];
                let (near, far) = if from_a { (a, b) } else { (b, a) };
                if inside(e, near) {
                    used[j] = true;
                    chains[ci].push(near);
                    chains[ci].push(far);
                    next_open.push(ci);
                }
            }
        }
        for (j, &(a, b)) in segs.iter().enumerate() {
            if !used[j] {
                chains.push(vec![a, b]);
                next_open.push(chains.len() - 1);
            }
        }
        open = next_open;
    }
    chains
}

/// Fill-mode paths for every tool and layer: `out[layer][tool]`.
pub fn build(labels: &Grid<u8>, tools: usize, layers: usize, base_angle: f64, p: &FillParams) -> Vec<Vec<ToolPaths>> {
    let fields: Vec<Option<ToolField>> =
        (0..tools).into_par_iter().map(|t| ToolField::new(labels, t as u8, p.smoothing)).collect();
    // Odd layers turn the infill 90 degrees; only two distinct layers exist.
    let per_angle: Vec<Vec<ToolPaths>> = (0..layers.min(2))
        .map(|l| {
            let angle = base_angle + if l % 2 == 1 { 90.0 } else { 0.0 };
            fields
                .par_iter()
                .map(|f| match f {
                    None => ToolPaths::default(),
                    Some(f) => {
                        let tp = f.paths(p, angle);
                        let perimeters = geom::order(tp.perimeters, None);
                        let cursor = perimeters.last().map(|l| l[l.len() - 1]);
                        let infill = geom::order(tp.infill, cursor);
                        ToolPaths { perimeters, infill }
                    }
                })
                .collect()
        })
        .collect();
    (0..layers).map(|l| per_angle[l % 2].clone()).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn square_labels(n: usize, lo: usize, hi: usize) -> Grid<u8> {
        let mut g = Grid::new(n, n, BACKGROUND);
        for y in lo..hi {
            for x in lo..hi {
                g.set(x, y, 0);
            }
        }
        g
    }

    fn params() -> FillParams {
        FillParams {
            line_width: 4.0,
            spacing: 4.0,
            perimeters: 1,
            inset: 0.5,
            overlap: 0.0,
            smoothing: 1.0,
            tolerance: 0.05,
            connect: true,
            exclusions: vec![],
            infill: true,
        }
    }

    #[test]
    fn perimeter_of_a_square_sits_half_a_line_inside() {
        // Pixels 20..80 cover [19.5, 79.5] in pixel-centre coordinates.
        let f = ToolField::new(&square_labels(100, 20, 80), 0, 1.0).unwrap();
        let tp = f.paths(&params(), 0.0);
        assert_eq!(tp.perimeters.len(), 1);
        let l = &tp.perimeters[0];
        assert_eq!(l[0], l[l.len() - 1]);
        // Where the loop crosses the square's middle row: 2 px inside the edge.
        let left = l
            .windows(2)
            .filter(|s| (s[0][1] - 49.5) * (s[1][1] - 49.5) <= 0.0 && s[0][1] != s[1][1])
            .map(|s| s[0][0] + (49.5 - s[0][1]) / (s[1][1] - s[0][1]) * (s[1][0] - s[0][0]))
            .fold(f64::MAX, f64::min);
        assert!((left - 21.5).abs() < 0.15, "left edge at {left}");
        // Smooth: a 60 px square needs only a handful of points once simplified.
        assert!(l.len() < 60, "{} points", l.len());
    }

    #[test]
    fn horizontal_infill_spans_the_square_at_the_spacing() {
        let f = ToolField::new(&square_labels(100, 20, 80), 0, 1.0).unwrap();
        let mut p = params();
        p.connect = false;
        let tp = f.paths(&p, 0.0);
        // Perimeter centre 2 px in, its inner edge 4 px in, so infill line
        // centres run from 6 px in: [25.5, 73.5], 48 px / 4 px = 12 lines.
        assert!((11..=13).contains(&tp.infill.len()), "{} lines", tp.infill.len());
        let mid = tp.infill.iter().find(|l| (l[0][1] - 49.5).abs() < 2.1).unwrap();
        let (x0, x1) = (mid[0][0].min(mid[1][0]), mid[0][0].max(mid[1][0]));
        assert!((x0 - 25.5).abs() < 0.2 && (x1 - 73.5).abs() < 0.2, "{x0}..{x1}");
        for l in &tp.infill {
            assert!((l[0][1] - l[1][1]).abs() < 1e-6, "not horizontal");
        }
    }

    #[test]
    fn connected_infill_of_a_square_is_one_serpentine() {
        let f = ToolField::new(&square_labels(100, 20, 80), 0, 1.0).unwrap();
        let tp = f.paths(&params(), 45.0);
        assert_eq!(tp.infill.len(), 1);
    }

    #[test]
    fn exclusion_removes_infill_but_not_perimeters() {
        let f = ToolField::new(&square_labels(100, 20, 80), 0, 1.0).unwrap();
        let mut p = params();
        p.connect = false;
        p.exclusions = vec![[0.0, 0.0, 100.0, 50.0]];
        let tp = f.paths(&p, 0.0);
        assert_eq!(tp.perimeters.len(), 1);
        assert!(!tp.infill.is_empty());
        assert!(tp.infill.iter().all(|l| l[0][1] >= 49.9));
    }

    #[test]
    fn a_ring_gets_inner_and_outer_perimeters() {
        let mut g = square_labels(100, 10, 90);
        for y in 40..60 {
            for x in 40..60 {
                g.set(x, y, BACKGROUND);
            }
        }
        let f = ToolField::new(&g, 0, 1.0).unwrap();
        let mut p = params();
        p.perimeters = 2;
        assert_eq!(f.paths(&p, 0.0).perimeters.len(), 4);
    }

    #[test]
    fn layers_alternate_infill_direction() {
        let out = build(&square_labels(100, 20, 80), 1, 2, 0.0, &FillParams { connect: false, ..params() });
        let l0 = &out[0][0].infill[0];
        let l1 = &out[1][0].infill[0];
        assert!((l0[0][1] - l0[1][1]).abs() < 1e-6);
        assert!((l1[0][0] - l1[1][0]).abs() < 1e-6);
    }
}
