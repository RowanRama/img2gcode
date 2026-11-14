//! Edge mode: detect edges, thin them to a skeleton, trace the skeleton into
//! strokes, keep strokes from retracing each other, then join and smooth.

use std::collections::HashMap;

use crate::geom::{self, Pt, dist, length};
use crate::raster::{Grid, bilateral, canny, clahe, dilate, gaussian_blur_u8, otsu, thin};

/// Edge-detection settings.
#[derive(Debug, Clone, PartialEq)]
pub struct DetectParams {
    /// CLAHE clip limit; 0 disables.
    pub clahe_clip: f64,
    /// CLAHE tiles per side.
    pub clahe_grid: usize,
    /// Bilateral diameter; 0 disables.
    pub bilateral_d: usize,
    /// Bilateral colour sigma.
    pub bilateral_sigma_color: f64,
    /// Bilateral space sigma.
    pub bilateral_sigma_space: f64,
    /// Canny low threshold; 0 with `high` 0 = Otsu.
    pub low: u8,
    /// Canny high threshold.
    pub high: u8,
    /// Blur sigmas Canny runs at.
    pub scales: Vec<f64>,
    /// Keep only edges near the foreground.
    pub restrict_to_foreground: bool,
    /// Foreground dilation (px) before restricting.
    pub foreground_dilate: usize,
}

/// Stroke-tracing settings in pixel units.
#[derive(Debug, Clone)]
pub struct TraceParams {
    /// Printed line width.
    pub line_width: f64,
    /// Merge edges within this many line widths.
    pub collapse: f64,
    /// Minimum stroke separation in line widths.
    pub min_separation: f64,
    /// Join stroke ends within this many line widths.
    pub chain_gap: f64,
    /// Drop strokes shorter than this.
    pub min_length: f64,
    /// Gaussian smoothing sigma along strokes.
    pub smoothing: f64,
    /// Douglas-Peucker tolerance.
    pub tolerance: f64,
}

/// The edge map: multi-scale Canny after CLAHE and bilateral filtering.
pub fn detect(gray: &Grid<u8>, foreground: &Grid<bool>, p: &DetectParams) -> Grid<bool> {
    let mut work = gray.clone();
    if p.clahe_clip > 0.0 {
        work = clahe(&work, p.clahe_clip, p.clahe_grid);
    }
    if p.bilateral_d > 0 {
        work = bilateral(&work, p.bilateral_d, p.bilateral_sigma_color, p.bilateral_sigma_space);
    }
    let (lo, hi) = if p.low > 0 && p.high > 0 {
        (p.low as f32, p.high as f32)
    } else {
        // Otsu on the image; the median heuristic breaks on white backgrounds.
        let t = otsu(&work) as f32;
        (t * 0.5, t.max(t * 0.5 + 1.0))
    };
    let scales = if p.scales.is_empty() { vec![0.0] } else { p.scales.clone() };
    let mut edges = Grid::new(work.w, work.h, false);
    for s in scales {
        let blurred = if s > 0.0 { gaussian_blur_u8(&work, s) } else { work.clone() };
        let e = canny(&blurred, lo, hi);
        edges.data.iter_mut().zip(e.data).for_each(|(a, b)| *a |= b);
    }
    if p.restrict_to_foreground {
        // Edges sit on the subject's boundary, just outside the mask.
        let fg = dilate(foreground, p.foreground_dilate.max(1) as f64);
        edges.data.iter_mut().zip(fg.data).for_each(|(a, b)| *a &= b);
    }
    edges
}

/// Strokes traced from an edge map, in drawing order, in pixels.
pub fn trace(edges: &Grid<bool>, p: &TraceParams) -> Vec<Vec<Pt>> {
    if !edges.data.iter().any(|&b| b) {
        return Vec::new();
    }
    let collapse_r = (p.line_width * p.collapse / 2.0).round();
    let thick = if collapse_r > 0.0 { dilate(edges, collapse_r) } else { edges.clone() };
    let skeleton = thin(&thick);
    // Short pieces are kept until after chaining: a diagonal or noisy edge
    // thins into many short fragments that only add up once joined.
    let mut strokes: Vec<Vec<Pt>> = trace_skeleton(&skeleton)
        .into_iter()
        .filter(|c| c.len() >= 2)
        .map(|c| c.into_iter().map(|(x, y)| [x as f64, y as f64]).collect::<Vec<Pt>>())
        .map(|c| geom::simplify(&geom::smooth(&c, p.smoothing), p.tolerance))
        .collect();
    // Longest first, so the structural strokes win the overlap guard.
    strokes.sort_by(|a, b| length(b).total_cmp(&length(a)));
    let guard_r = (p.line_width / 2.0).round().max(1.0);
    let forbid_r = (p.min_separation * p.line_width / 2.0).round().max(1.0);
    // A stroke is blocked within forbid_r of a drawn stroke's edge, i.e.
    // within guard_r + forbid_r of its centre line.
    let mut forbidden = Grid::new(edges.w, edges.h, false);
    let mut kept = Vec::new();
    for s in strokes {
        for sub in split_clear(&s, &forbidden) {
            if sub.len() < 2 || length(&sub) < 1.0 {
                continue;
            }
            stamp(&mut forbidden, &sub, guard_r + forbid_r);
            kept.push(sub);
        }
    }
    let chained = chain(kept, p.chain_gap * p.line_width)
        .into_iter()
        .filter(|s| p.min_length <= 0.0 || length(s) >= p.min_length)
        .collect();
    geom::order(chained, None)
}

/// Neighbours, 4-connected first so a walk steps along a staircase instead
/// of cutting its corners and orphaning pixels.
const NEIGH: [(isize, isize); 8] = [(0, -1), (-1, 0), (1, 0), (0, 1), (-1, -1), (1, -1), (-1, 1), (1, 1)];
/// Neighbours in order around the pixel, for counting branches.
const RING: [(isize, isize); 8] = [(0, -1), (1, -1), (1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1)];

/// Walk a one-pixel skeleton into pixel chains. Junction pixels are removed
/// first so the rest is simple chains and loops, then re-attached to the
/// chain ends they touch. A pixel's branches are the separate runs of set
/// neighbours around it (its crossing number): 3+ is a junction, 1 an end.
/// Counting neighbours instead would call every step of a diagonal
/// staircase a junction and erase the diagonal.
fn trace_skeleton(skel: &Grid<bool>) -> Vec<Vec<(usize, usize)>> {
    let (w, h) = (skel.w, skel.h);
    let on = |g: &Grid<bool>, x: isize, y: isize| {
        x >= 0 && y >= 0 && x < w as isize && y < h as isize && g.get(x as usize, y as usize)
    };
    let branches = |g: &Grid<bool>, x: usize, y: usize| {
        let set: Vec<bool> = RING.iter().map(|(dx, dy)| on(g, x as isize + dx, y as isize + dy)).collect();
        (0..8).filter(|&k| !set[k] && set[(k + 1) % 8]).count() + usize::from(set.iter().all(|&b| b))
    };
    let mut work = skel.clone();
    let mut junctions = Vec::new();
    for y in 0..h {
        for x in 0..w {
            if skel.get(x, y) && branches(skel, x, y) >= 3 {
                junctions.push((x, y));
            }
        }
    }
    for &(x, y) in &junctions {
        work.set(x, y, false);
    }
    let mut visited = Grid::new(w, h, false);
    let walk = |sx: usize, sy: usize, visited: &mut Grid<bool>| -> Vec<(usize, usize)> {
        let mut path = vec![(sx, sy)];
        visited.set(sx, sy, true);
        let (mut cx, mut cy) = (sx, sy);
        loop {
            let next = NEIGH
                .iter()
                .map(|(dx, dy)| (cx as isize + dx, cy as isize + dy))
                .find(|&(nx, ny)| on(&work, nx, ny) && !visited.get(nx as usize, ny as usize));
            match next {
                Some((nx, ny)) => {
                    (cx, cy) = (nx as usize, ny as usize);
                    visited.set(cx, cy, true);
                    path.push((cx, cy));
                }
                None => return path,
            }
        }
    };
    let mut chains = Vec::new();
    // Chains from their end points first, then the closed loops left over.
    for y in 0..h {
        for x in 0..w {
            if work.get(x, y) && !visited.get(x, y) && branches(&work, x, y) <= 1 {
                chains.push(walk(x, y, &mut visited));
            }
        }
    }
    for y in 0..h {
        for x in 0..w {
            if work.get(x, y) && !visited.get(x, y) {
                let mut path = walk(x, y, &mut visited);
                let (a, b) = (path[0], path[path.len() - 1]);
                if path.len() > 2 && a != b && a.0.abs_diff(b.0) <= 1 && a.1.abs_diff(b.1) <= 1 {
                    path.push(a);
                }
                chains.push(path);
            }
        }
    }
    // Re-attach each junction to the chain ends next to it so the printed
    // strokes actually meet.
    let mut ends: HashMap<(usize, usize), Vec<(usize, bool)>> = HashMap::new();
    for (i, c) in chains.iter().enumerate() {
        ends.entry(c[0]).or_default().push((i, true));
        ends.entry(c[c.len() - 1]).or_default().push((i, false));
    }
    for (jx, jy) in junctions {
        for (dx, dy) in NEIGH {
            let (nx, ny) = (jx as isize + dx, jy as isize + dy);
            if nx < 0 || ny < 0 {
                continue;
            }
            let Some(list) = ends.remove(&(nx as usize, ny as usize)) else { continue };
            for (i, head) in list {
                if head {
                    chains[i].insert(0, (jx, jy));
                } else {
                    chains[i].push((jx, jy));
                }
                let c = &chains[i];
                let key = if head { c[0] } else { c[c.len() - 1] };
                ends.entry(key).or_default().push((i, head));
            }
        }
    }
    chains
}

/// The parts of `poly` that stay clear of `forbidden`, checked at every
/// pixel along each segment so a long segment's clear middle survives.
fn split_clear(poly: &[Pt], forbidden: &Grid<bool>) -> Vec<Vec<Pt>> {
    let blocked = |p: Pt| {
        let (x, y) = (p[0].round(), p[1].round());
        x >= 0.0
            && y >= 0.0
            && (x as usize) < forbidden.w
            && (y as usize) < forbidden.h
            && forbidden.get(x as usize, y as usize)
    };
    let mut runs = Vec::new();
    let mut cur: Vec<Pt> = Vec::new();
    let mut prev_blocked = blocked(poly[0]);
    let mut last_clear = poly[0];
    if !prev_blocked {
        cur.push(poly[0]);
    }
    for w in poly.windows(2) {
        let (a, b) = (w[0], w[1]);
        let n = dist(a, b).ceil().max(1.0) as usize;
        for s in 1..=n {
            let t = s as f64 / n as f64;
            let p = [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t];
            let blk = blocked(p);
            if blk && !prev_blocked {
                // The run ends at the last clear sample before the block.
                if cur.last().is_some_and(|q| dist(*q, last_clear) > 1e-9) {
                    cur.push(last_clear);
                }
                if cur.len() >= 2 {
                    runs.push(std::mem::take(&mut cur));
                } else {
                    cur.clear();
                }
            }
            if !blk {
                // Keep the vertices and the points where a run re-enters
                // clear space; samples in between add nothing.
                if prev_blocked || s == n {
                    cur.push(p);
                }
                last_clear = p;
            }
            prev_blocked = blk;
        }
    }
    if cur.len() >= 2 {
        runs.push(cur);
    }
    runs
}

/// Mark every pixel within `r` of the polyline.
fn stamp(mask: &mut Grid<bool>, poly: &[Pt], r: f64) {
    for w in poly.windows(2) {
        let (a, b) = (w[0], w[1]);
        let x0 = (a[0].min(b[0]) - r).floor().max(0.0) as usize;
        let x1 = ((a[0].max(b[0]) + r).ceil().max(0.0) as usize).min(mask.w.saturating_sub(1));
        let y0 = (a[1].min(b[1]) - r).floor().max(0.0) as usize;
        let y1 = ((a[1].max(b[1]) + r).ceil().max(0.0) as usize).min(mask.h.saturating_sub(1));
        for y in y0..=y1 {
            for x in x0..=x1 {
                if geom::seg_dist([x as f64, y as f64], a, b) <= r {
                    mask.set(x, y, true);
                }
            }
        }
    }
}

/// Join strokes whose ends are within `gap`, longest first, always
/// extending a chain's tail with the nearest free end.
fn chain(strokes: Vec<Vec<Pt>>, gap: f64) -> Vec<Vec<Pt>> {
    let strokes: Vec<Vec<Pt>> = strokes.into_iter().filter(|s| s.len() >= 2).collect();
    if gap <= 0.0 || strokes.len() < 2 {
        return strokes;
    }
    let cell = |p: Pt| ((p[0] / gap).floor() as i64, (p[1] / gap).floor() as i64);
    let mut grid: HashMap<(i64, i64), Vec<(usize, bool)>> = HashMap::new();
    for (i, s) in strokes.iter().enumerate() {
        grid.entry(cell(s[0])).or_default().push((i, true));
        grid.entry(cell(s[s.len() - 1])).or_default().push((i, false));
    }
    let mut order: Vec<usize> = (0..strokes.len()).collect();
    order.sort_by(|&a, &b| length(&strokes[b]).total_cmp(&length(&strokes[a])));
    let mut used = vec![false; strokes.len()];
    let mut out = Vec::new();
    for start in order {
        if used[start] {
            continue;
        }
        used[start] = true;
        let mut c = strokes[start].clone();
        loop {
            let e = c[c.len() - 1];
            let (cx, cy) = cell(e);
            let mut best: Option<(f64, usize, bool)> = None;
            for gy in cy - 1..=cy + 1 {
                for gx in cx - 1..=cx + 1 {
                    for &(i, head) in grid.get(&(gx, gy)).into_iter().flatten() {
                        if used[i] {
                            continue;
                        }
                        let s = &strokes[i];
                        let d = dist(e, if head { s[0] } else { s[s.len() - 1] });
                        if d <= gap && best.is_none_or(|b| d < b.0) {
                            best = Some((d, i, head));
                        }
                    }
                }
            }
            let Some((_, i, head)) = best else { break };
            used[i] = true;
            let mut s = strokes[i].clone();
            if !head {
                s.reverse();
            }
            let skip = usize::from(dist(c[c.len() - 1], s[0]) < 1e-9);
            c.extend_from_slice(&s[skip..]);
        }
        out.push(c);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn trace_params() -> TraceParams {
        TraceParams {
            line_width: 2.0,
            collapse: 0.0,
            min_separation: 0.8,
            chain_gap: 1.5,
            min_length: 4.0,
            smoothing: 1.0,
            tolerance: 0.1,
        }
    }

    #[test]
    fn a_square_outline_traces_to_one_closed_stroke() {
        let mut e = Grid::new(60, 60, false);
        for i in 10..50 {
            e.set(i, 10, true);
            e.set(i, 49, true);
            e.set(10, i, true);
            e.set(49, i, true);
        }
        let s = trace(&e, &trace_params());
        assert_eq!(s.len(), 1, "{s:?}");
        assert!(length(&s[0]) > 140.0);
    }

    #[test]
    fn a_diagonal_staircase_traces_as_one_stroke() {
        // A 4-connected staircase: every pixel has 2-3 neighbours.
        let mut e = Grid::new(60, 60, false);
        for i in 5..50 {
            e.set(i, i, true);
            e.set(i + 1, i, true);
        }
        let s = trace(&e, &trace_params());
        assert_eq!(s.len(), 1, "{} strokes", s.len());
        assert!(length(&s[0]) > 60.0, "length {}", length(&s[0]));
    }

    #[test]
    fn crossing_lines_do_not_retrace() {
        let mut e = Grid::new(60, 60, false);
        for i in 5..55 {
            e.set(i, 30, true);
            e.set(30, i, true);
        }
        let s = trace(&e, &trace_params());
        let total: f64 = s.iter().map(|p| length(p)).sum();
        // Two 50 px lines; nothing drawn twice.
        assert!((90.0..=102.0).contains(&total), "total {total}");
    }

    #[test]
    fn detect_finds_a_dark_disc_on_white() {
        let mut g = Grid::new(80, 80, 255u8);
        for y in 0..80 {
            for x in 0..80 {
                if ((x as f64 - 40.0).hypot(y as f64 - 40.0)) < 20.0 {
                    g.set(x, y, 20);
                }
            }
        }
        let p = DetectParams {
            clahe_clip: 1.5,
            clahe_grid: 8,
            bilateral_d: 5,
            bilateral_sigma_color: 35.0,
            bilateral_sigma_space: 35.0,
            low: 0,
            high: 0,
            scales: vec![1.0],
            restrict_to_foreground: false,
            foreground_dilate: 3,
        };
        let e = detect(&g, &Grid::new(80, 80, true), &p);
        assert!(e.get(20, 40) || e.get(21, 40) || e.get(19, 40));
        assert!(!e.get(40, 40));
        assert!(!e.get(2, 2));
    }

    #[test]
    fn split_keeps_the_clear_middle_between_two_blocked_ends() {
        let mut forbidden = Grid::new(40, 10, false);
        stamp(&mut forbidden, &[[2.0, 5.0], [2.0, 5.0]], 2.0);
        stamp(&mut forbidden, &[[37.0, 5.0], [37.0, 5.0]], 2.0);
        let runs = split_clear(&[[2.0, 5.0], [37.0, 5.0]], &forbidden);
        assert_eq!(runs.len(), 1);
        let r = &runs[0];
        assert!(r[0][0] > 4.0 && r[0][0] < 6.0, "{r:?}");
        assert!(r[r.len() - 1][0] > 33.0 && r[r.len() - 1][0] < 35.0, "{r:?}");
    }

    #[test]
    fn chain_joins_close_ends_and_reverses() {
        let a = vec![[0.0, 0.0], [10.0, 0.0]];
        let b = vec![[20.0, 0.0], [11.0, 0.0]];
        let out = chain(vec![a, b], 2.0);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0], vec![[0.0, 0.0], [10.0, 0.0], [11.0, 0.0], [20.0, 0.0]]);
    }
}
