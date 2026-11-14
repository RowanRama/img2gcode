//! Polyline geometry: iso-contours, smoothing, simplification, ordering.

use crate::raster::Grid;

/// A 2-D point.
pub type Pt = [f64; 2];

/// Euclidean distance.
#[inline]
pub fn dist(a: Pt, b: Pt) -> f64 {
    (a[0] - b[0]).hypot(a[1] - b[1])
}

/// Total length of a polyline.
pub fn length(poly: &[Pt]) -> f64 {
    poly.windows(2).map(|w| dist(w[0], w[1])).sum()
}

/// Closed iso-lines of `field` at `level` (marching squares, linear
/// interpolation, saddles resolved by the cell centre). Cells beyond the grid
/// are not visited, so the field should be below `level` along its border
/// for every line to close. Each loop's last point repeats its first.
pub fn iso_lines(field: &Grid<f32>, level: f32) -> Vec<Vec<Pt>> {
    let (w, h) = (field.w, field.h);
    if w < 2 || h < 2 {
        return Vec::new();
    }
    let inside = |x: usize, y: usize| field.get(x, y) >= level;
    // Edge ids: 2*(y*w+x) is the horizontal edge (x,y)-(x+1,y),
    // 2*(y*w+x)+1 the vertical edge (x,y)-(x,y+1).
    let hid = |x: usize, y: usize| 2 * (y * w + x);
    let vid = |x: usize, y: usize| 2 * (y * w + x) + 1;
    let mut next = vec![u32::MAX; 2 * w * h];
    let mut starts = Vec::new();
    for y in 0..h - 1 {
        for x in 0..w - 1 {
            // Corners clockwise on screen: a(x,y) b(x+1,y) c(x+1,y+1) d(x,y+1).
            let c = [inside(x, y), inside(x + 1, y), inside(x + 1, y + 1), inside(x, y + 1)];
            let code = c.iter().enumerate().fold(0u8, |acc, (i, &b)| acc | ((b as u8) << i));
            if code == 0 || code == 15 {
                continue;
            }
            // Edges in the same clockwise order: top, right, bottom, left.
            let edges = [hid(x, y), vid(x + 1, y), hid(x, y + 1), vid(x, y)];
            // A crossing on edge k is an entry when we go outside -> inside.
            let mut cross: Vec<(usize, bool)> = Vec::with_capacity(4);
            for k in 0..4 {
                let (p, q) = (c[k], c[(k + 1) % 4]);
                if p != q {
                    cross.push((k, q));
                }
            }
            let centre_in =
                (field.get(x, y) + field.get(x + 1, y) + field.get(x + 1, y + 1) + field.get(x, y + 1)) / 4.0 >= level;
            let n = cross.len();
            for i in 0..n {
                let (k, entry) = cross[i];
                if !entry {
                    continue;
                }
                // An entry pairs with the next crossing clockwise when the
                // centre is outside (inside corners cut off), else with the
                // previous one (outside corners cut off).
                let j = if centre_in && n == 4 { (i + n - 1) % n } else { (i + 1) % n };
                let from = edges[k];
                next[from] = edges[cross[j].0] as u32;
                starts.push(from);
            }
        }
    }
    let point = |e: usize| -> Pt {
        let cell = e / 2;
        let (x, y) = (cell % w, cell / w);
        let (x2, y2) = if e.is_multiple_of(2) { (x + 1, y) } else { (x, y + 1) };
        let (va, vb) = (field.get(x, y), field.get(x2, y2));
        let t = if (vb - va).abs() > 1e-12 { ((level - va) / (vb - va)).clamp(0.0, 1.0) } else { 0.5 } as f64;
        [x as f64 + t * (x2 as f64 - x as f64), y as f64 + t * (y2 as f64 - y as f64)]
    };
    let mut loops = Vec::new();
    for s in starts {
        if next[s] == u32::MAX {
            continue;
        }
        let mut poly = vec![point(s)];
        let mut e = s;
        loop {
            let n = next[e];
            next[e] = u32::MAX;
            if n == u32::MAX {
                break; // open end at the grid border
            }
            let n = n as usize;
            poly.push(point(n));
            if n == s {
                break;
            }
            e = n;
        }
        if poly.len() >= 4 {
            loops.push(poly);
        }
    }
    loops
}

/// Gaussian smoothing along the polyline, sigma in the same units as the
/// points. Closed polylines (last == first) wrap around; open ones keep
/// their end points.
pub fn smooth(poly: &[Pt], sigma: f64) -> Vec<Pt> {
    let n = poly.len();
    if sigma <= 0.0 || n < 3 {
        return poly.to_vec();
    }
    let closed = dist(poly[0], poly[n - 1]) < 1e-9;
    let pts = if closed { &poly[..n - 1] } else { poly };
    let m = pts.len();
    let spacing = (length(poly) / (n - 1) as f64).max(1e-9);
    let s = sigma / spacing;
    let r = ((3.0 * s).ceil() as usize).min(if closed { m / 2 } else { m });
    if r == 0 {
        return poly.to_vec();
    }
    let k: Vec<f64> = (0..=r).map(|i| (-((i * i) as f64) / (2.0 * s * s)).exp()).collect();
    let mut out: Vec<Pt> = (0..m)
        .map(|i| {
            if !closed && (i == 0 || i == m - 1) {
                return pts[i];
            }
            // Open lines: shrink the window near the ends so it stays
            // symmetric and the ends do not pull the curve.
            let rr = if closed { r } else { r.min(i).min(m - 1 - i) };
            let (mut sx, mut sy, mut sw) = (pts[i][0] * k[0], pts[i][1] * k[0], k[0]);
            for (d, kd) in k.iter().enumerate().take(rr + 1).skip(1) {
                let (a, b) = if closed { ((i + m - d) % m, (i + d) % m) } else { (i - d, i + d) };
                sx += (pts[a][0] + pts[b][0]) * kd;
                sy += (pts[a][1] + pts[b][1]) * kd;
                sw += 2.0 * kd;
            }
            [sx / sw, sy / sw]
        })
        .collect();
    if closed {
        out.push(out[0]);
    }
    out
}

/// Douglas-Peucker simplification: drop points while the line stays within
/// `eps` of the original. End points are kept.
pub fn simplify(poly: &[Pt], eps: f64) -> Vec<Pt> {
    let n = poly.len();
    if eps <= 0.0 || n < 3 {
        return poly.to_vec();
    }
    let mut keep = vec![false; n];
    keep[0] = true;
    keep[n - 1] = true;
    let mut stack = vec![(0usize, n - 1)];
    // A closed loop's two ends coincide, so split it at its farthest point first.
    if dist(poly[0], poly[n - 1]) < 1e-9 {
        let far = (1..n - 1).max_by(|&a, &b| dist(poly[0], poly[a]).total_cmp(&dist(poly[0], poly[b]))).unwrap_or(1);
        keep[far] = true;
        stack = vec![(0, far), (far, n - 1)];
    }
    while let Some((a, b)) = stack.pop() {
        if b <= a + 1 {
            continue;
        }
        let (mut best, mut idx) = (0.0, a);
        for i in a + 1..b {
            let d = seg_dist(poly[i], poly[a], poly[b]);
            if d > best {
                best = d;
                idx = i;
            }
        }
        if best > eps {
            keep[idx] = true;
            stack.push((a, idx));
            stack.push((idx, b));
        }
    }
    poly.iter().zip(keep).filter_map(|(p, k)| k.then_some(*p)).collect()
}

/// Distance from `p` to segment `a`-`b`.
pub fn seg_dist(p: Pt, a: Pt, b: Pt) -> f64 {
    let (dx, dy) = (b[0] - a[0], b[1] - a[1]);
    let l2 = dx * dx + dy * dy;
    if l2 < 1e-18 {
        return dist(p, a);
    }
    let t = (((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / l2).clamp(0.0, 1.0);
    dist(p, [a[0] + t * dx, a[1] + t * dy])
}

/// Greedy nearest-neighbour ordering to cut travel. An open polyline can be
/// entered from either end (it is reversed when needed); a closed one
/// (last == first) can be entered at any vertex (it is rotated). Starts at
/// `start`, or with the first polyline as given.
pub fn order(polys: Vec<Vec<Pt>>, start: Option<Pt>) -> Vec<Vec<Pt>> {
    let polys: Vec<Vec<Pt>> = polys.into_iter().filter(|p| p.len() >= 2).collect();
    if polys.len() < 2 {
        return polys;
    }
    let closed: Vec<bool> = polys.iter().map(|p| dist(p[0], p[p.len() - 1]) < 1e-9).collect();
    // Anchors: (polyline, vertex). Open lines anchor at both ends, closed
    // ones at every vertex but the repeated last.
    let mut anchors: Vec<(usize, usize, Pt)> = Vec::new();
    for (i, p) in polys.iter().enumerate() {
        if closed[i] {
            anchors.extend(p[..p.len() - 1].iter().enumerate().map(|(v, &pt)| (i, v, pt)));
        } else {
            anchors.push((i, 0, p[0]));
            anchors.push((i, p.len() - 1, p[p.len() - 1]));
        }
    }
    let grid = AnchorGrid::new(&anchors);
    let mut used = vec![false; polys.len()];
    let mut out = Vec::with_capacity(polys.len());
    let mut slots: Vec<Option<Vec<Pt>>> = polys.into_iter().map(Some).collect();
    let mut cursor = match start {
        Some(p) => p,
        None => {
            used[0] = true;
            // Index 0 exists: there are at least two polylines.
            let first = slots[0].take().expect("first polyline");
            let end = first[first.len() - 1];
            out.push(first);
            end
        }
    };
    while out.len() < slots.len() {
        let Some((pi, v)) = grid.nearest(cursor, &anchors, &used) else { break };
        used[pi] = true;
        // Each index is taken once: `used` guards it.
        let mut p = slots[pi].take().expect("unused polyline");
        if closed[pi] {
            if v != 0 {
                p.pop();
                p.rotate_left(v);
                p.push(p[0]);
            }
        } else if v != 0 {
            p.reverse();
        }
        cursor = p[p.len() - 1];
        out.push(p);
    }
    out
}

struct AnchorGrid {
    min: Pt,
    cell: f64,
    nx: usize,
    ny: usize,
    cells: Vec<Vec<u32>>,
}

impl AnchorGrid {
    fn new(anchors: &[(usize, usize, Pt)]) -> Self {
        let mut min = [f64::MAX; 2];
        let mut max = [f64::MIN; 2];
        for a in anchors {
            for k in 0..2 {
                min[k] = min[k].min(a.2[k]);
                max[k] = max[k].max(a.2[k]);
            }
        }
        let span = [(max[0] - min[0]).max(1e-6), (max[1] - min[1]).max(1e-6)];
        let cell = ((span[0] * span[1]) / anchors.len() as f64).sqrt().max(1e-6) * 2.0;
        let nx = ((span[0] / cell) as usize + 1).min(4096);
        let ny = ((span[1] / cell) as usize + 1).min(4096);
        let cell = (span[0] / nx as f64).max(span[1] / ny as f64).max(1e-6) * 1.000_001;
        let mut cells = vec![Vec::new(); nx * ny];
        for (i, a) in anchors.iter().enumerate() {
            let (cx, cy) = Self::cell_of(min, cell, nx, ny, a.2);
            cells[cy * nx + cx].push(i as u32);
        }
        Self { min, cell, nx, ny, cells }
    }

    fn cell_of(min: Pt, cell: f64, nx: usize, ny: usize, p: Pt) -> (usize, usize) {
        let cx = (((p[0] - min[0]) / cell).floor().max(0.0) as usize).min(nx - 1);
        let cy = (((p[1] - min[1]) / cell).floor().max(0.0) as usize).min(ny - 1);
        (cx, cy)
    }

    fn nearest(&self, p: Pt, anchors: &[(usize, usize, Pt)], used: &[bool]) -> Option<(usize, usize)> {
        let (cx, cy) = Self::cell_of(self.min, self.cell, self.nx, self.ny, p);
        let mut best: Option<(f64, usize, usize)> = None;
        let max_r = self.nx.max(self.ny);
        for r in 0..=max_r {
            // Anything in ring r is at least (r - 1) cells away.
            if let Some((d, _, _)) = best
                && d < (r as f64 - 1.0) * self.cell
            {
                break;
            }
            let (x0, x1) = (cx as isize - r as isize, cx as isize + r as isize);
            let (y0, y1) = (cy as isize - r as isize, cy as isize + r as isize);
            for y in y0..=y1 {
                if y < 0 || y >= self.ny as isize {
                    continue;
                }
                let on_edge_row = y == y0 || y == y1;
                let mut x = x0;
                while x <= x1 {
                    if x >= 0 && x < self.nx as isize {
                        for &ai in &self.cells[y as usize * self.nx + x as usize] {
                            let (pi, v, q) = anchors[ai as usize];
                            if used[pi] {
                                continue;
                            }
                            let d = dist(p, q);
                            if best.is_none_or(|b| d < b.0) {
                                best = Some((d, pi, v));
                            }
                        }
                    }
                    // Interior rows only visit the ring's two side cells.
                    x = if on_edge_row || x == x1 { x + 1 } else { x1 };
                }
            }
        }
        best.map(|(_, pi, v)| (pi, v))
    }
}

/// Total travel between consecutive polylines.
pub fn travel(polys: &[Vec<Pt>]) -> f64 {
    polys.windows(2).map(|w| dist(w[0][w[0].len() - 1], w[1][0])).sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn disc_field(n: usize, cx: f64, cy: f64, r: f64) -> Grid<f32> {
        let mut g = Grid::new(n, n, 0.0f32);
        for y in 0..n {
            for x in 0..n {
                g.set(x, y, (r - ((x as f64 - cx).hypot(y as f64 - cy))) as f32);
            }
        }
        g
    }

    #[test]
    fn iso_line_of_a_disc_is_one_closed_circle() {
        let g = disc_field(50, 24.3, 25.1, 15.0);
        let loops = iso_lines(&g, 0.0);
        assert_eq!(loops.len(), 1);
        let l = &loops[0];
        assert_eq!(l[0], l[l.len() - 1]);
        for p in l {
            let r = (p[0] - 24.3).hypot(p[1] - 25.1);
            assert!((r - 15.0).abs() < 0.05, "r = {r}");
        }
    }

    #[test]
    fn iso_lines_of_a_ring_are_two_loops_and_offsets_shrink() {
        let mut g = disc_field(60, 30.0, 30.0, 25.0);
        for (i, v) in g.data.iter_mut().enumerate() {
            let (x, y) = ((i % 60) as f64, (i / 60) as f64);
            let hole = 10.0 - (x - 30.0).hypot(y - 30.0);
            *v = v.min(-hole as f32);
        }
        assert_eq!(iso_lines(&g, 0.0).len(), 2);
        let inner = iso_lines(&g, 3.0);
        assert_eq!(inner.len(), 2);
        let outer_r = inner.iter().map(|l| (l[0][0] - 30.0).hypot(l[0][1] - 30.0)).fold(0.0, f64::max);
        assert!((outer_r - 22.0).abs() < 0.1);
    }

    #[test]
    fn saddle_cells_do_not_cross_lines() {
        // Two diagonal blobs touching at a corner: every loop stays closed.
        let mut g = Grid::new(6, 6, -1.0f32);
        g.set(2, 2, 1.0);
        g.set(3, 3, 1.0);
        for l in iso_lines(&g, 0.0) {
            assert_eq!(l[0], l[l.len() - 1]);
        }
    }

    #[test]
    fn simplify_drops_collinear_points_and_keeps_corners() {
        let poly: Vec<Pt> = (0..=10).map(|i| [i as f64, 0.0]).chain((1..=10).map(|i| [10.0, i as f64])).collect();
        let s = simplify(&poly, 0.01);
        assert_eq!(s, vec![[0.0, 0.0], [10.0, 0.0], [10.0, 10.0]]);
    }

    #[test]
    fn simplify_keeps_a_closed_square_closed() {
        let sq = vec![[0.0, 0.0], [5.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0], [0.0, 0.0]];
        let s = simplify(&sq, 0.1);
        assert_eq!(s.first(), s.last());
        assert_eq!(s.len(), 5);
    }

    #[test]
    fn smoothing_flattens_a_zigzag_and_keeps_open_ends() {
        let zig: Vec<Pt> = (0..40).map(|i| [i as f64, if i % 2 == 0 { 0.0 } else { 1.0 }]).collect();
        let s = smooth(&zig, 2.0);
        assert_eq!(s[0], zig[0]);
        assert_eq!(s[39], zig[39]);
        for p in &s[8..32] {
            assert!((p[1] - 0.5).abs() < 0.05, "{p:?}");
        }
    }

    #[test]
    fn order_reverses_and_rotates_to_cut_travel() {
        let a = vec![[0.0, 0.0], [10.0, 0.0]];
        let b = vec![[30.0, 0.0], [11.0, 0.0]]; // its tail is nearest
        let sq = vec![[50.0, 50.0], [31.0, 1.0], [50.0, 0.0], [50.0, 50.0]];
        let out = order(vec![a, sq, b], None);
        assert_eq!(out[1], vec![[11.0, 0.0], [30.0, 0.0]]);
        assert_eq!(out[2][0], [31.0, 1.0]);
        assert_eq!(out[2][0], out[2][3]);
        assert!(travel(&out) < 3.0);
    }

    #[test]
    fn order_handles_many_lines_quickly() {
        let polys: Vec<Vec<Pt>> = (0..20_000)
            .map(|i| {
                vec![
                    [(i % 200) as f64 * 3.0, (i / 200) as f64 * 3.0],
                    [(i % 200) as f64 * 3.0 + 1.0, (i / 200) as f64 * 3.0],
                ]
            })
            .collect();
        let t = std::time::Instant::now();
        let out = order(polys, Some([0.0, 0.0]));
        assert_eq!(out.len(), 20_000);
        assert!(t.elapsed().as_secs_f64() < 2.0);
    }
}
