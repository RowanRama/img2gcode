//! Small raster toolkit: just the image operations the pipeline needs.
//!
//! Coordinates: pixel (x, y) has its centre at (x, y); row-major storage.

use rayon::prelude::*;

/// A row-major 2-D grid.
#[derive(Debug, Clone, PartialEq)]
pub struct Grid<T> {
    /// Width in cells.
    pub w: usize,
    /// Height in cells.
    pub h: usize,
    /// `w * h` cells, row by row.
    pub data: Vec<T>,
}

impl<T: Copy> Grid<T> {
    /// A grid filled with `v`.
    pub fn new(w: usize, h: usize, v: T) -> Self {
        Self { w, h, data: vec![v; w * h] }
    }

    /// Cell at (x, y).
    #[inline]
    pub fn get(&self, x: usize, y: usize) -> T {
        self.data[y * self.w + x]
    }

    /// Set cell (x, y).
    #[inline]
    pub fn set(&mut self, x: usize, y: usize, v: T) {
        self.data[y * self.w + x] = v;
    }

    /// Apply `f` to every cell.
    pub fn map<U: Copy + Send, F: Fn(T) -> U + Sync>(&self, f: F) -> Grid<U>
    where
        T: Sync,
    {
        Grid { w: self.w, h: self.h, data: self.data.par_iter().map(|&v| f(v)).collect() }
    }

    /// The grid with a `pad`-cell border of `v` on every side.
    pub fn padded(&self, pad: usize, v: T) -> Self {
        let (w, h) = (self.w + 2 * pad, self.h + 2 * pad);
        let mut out = Grid::new(w, h, v);
        for y in 0..self.h {
            let src = &self.data[y * self.w..(y + 1) * self.w];
            out.data[(y + pad) * w + pad..(y + pad) * w + pad + self.w].copy_from_slice(src);
        }
        out
    }
}

impl Grid<f32> {
    /// Bilinear sample at a fractional position; `outside` beyond the grid.
    pub fn sample(&self, x: f64, y: f64, outside: f32) -> f32 {
        if !(x >= 0.0 && y >= 0.0) || x > (self.w - 1) as f64 || y > (self.h - 1) as f64 {
            return outside;
        }
        let x0 = (x.floor() as usize).min(self.w.saturating_sub(2));
        let y0 = (y.floor() as usize).min(self.h.saturating_sub(2));
        let (x1, y1) = ((x0 + 1).min(self.w - 1), (y0 + 1).min(self.h - 1));
        let (fx, fy) = ((x - x0 as f64) as f32, (y - y0 as f64) as f32);
        let top = self.get(x0, y0) * (1.0 - fx) + self.get(x1, y0) * fx;
        let bot = self.get(x0, y1) * (1.0 - fx) + self.get(x1, y1) * fx;
        top * (1.0 - fy) + bot * fy
    }
}

fn gaussian_kernel(sigma: f64) -> Vec<f32> {
    let r = (sigma * 3.0).ceil().max(1.0) as i64;
    let mut k: Vec<f32> = (-r..=r).map(|i| (-(i * i) as f64 / (2.0 * sigma * sigma)).exp() as f32).collect();
    let sum: f32 = k.iter().sum();
    k.iter_mut().for_each(|v| *v /= sum);
    k
}

/// Separable Gaussian blur; edges clamp. `sigma <= 0` returns a copy.
pub fn gaussian_blur(src: &Grid<f32>, sigma: f64) -> Grid<f32> {
    if sigma <= 0.0 || src.w == 0 || src.h == 0 {
        return src.clone();
    }
    let k = gaussian_kernel(sigma);
    let r = (k.len() / 2) as isize;
    let (w, h) = (src.w, src.h);
    let mut tmp = vec![0f32; w * h];
    tmp.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
        let s = &src.data[y * w..(y + 1) * w];
        for (x, out) in row.iter_mut().enumerate() {
            let mut acc = 0.0;
            for (i, kv) in k.iter().enumerate() {
                let xx = (x as isize + i as isize - r).clamp(0, w as isize - 1) as usize;
                acc += s[xx] * kv;
            }
            *out = acc;
        }
    });
    let mut out = vec![0f32; w * h];
    out.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
        for (i, kv) in k.iter().enumerate() {
            let yy = (y as isize + i as isize - r).clamp(0, h as isize - 1) as usize;
            let s = &tmp[yy * w..(yy + 1) * w];
            for (o, v) in row.iter_mut().zip(s) {
                *o += v * kv;
            }
        }
    });
    Grid { w, h, data: out }
}

/// Blur an 8-bit image, rounding back to 8 bits.
pub fn gaussian_blur_u8(src: &Grid<u8>, sigma: f64) -> Grid<u8> {
    if sigma <= 0.0 {
        return src.clone();
    }
    gaussian_blur(&src.map(|v| v as f32), sigma).map(|v| v.round().clamp(0.0, 255.0) as u8)
}

/// Contrast Limited Adaptive Histogram Equalisation, as OpenCV does it:
/// per-tile clipped histograms, LUTs interpolated bilinearly between tile
/// centres.
pub fn clahe(src: &Grid<u8>, clip: f64, tiles: usize) -> Grid<u8> {
    let tiles = tiles.max(1);
    let (w, h) = (src.w, src.h);
    if w < tiles || h < tiles {
        return src.clone();
    }
    let tx0 = |i: usize| i * w / tiles;
    let ty0 = |j: usize| j * h / tiles;
    let mut luts = vec![[0u8; 256]; tiles * tiles];
    luts.par_iter_mut().enumerate().for_each(|(idx, lut)| {
        let (i, j) = (idx % tiles, idx / tiles);
        let mut hist = [0u32; 256];
        for y in ty0(j)..ty0(j + 1) {
            for x in tx0(i)..tx0(i + 1) {
                hist[src.get(x, y) as usize] += 1;
            }
        }
        let area = ((tx0(i + 1) - tx0(i)) * (ty0(j + 1) - ty0(j))).max(1) as u32;
        let limit = ((clip * area as f64 / 256.0) as u32).max(1);
        let mut excess = 0;
        for v in hist.iter_mut() {
            if *v > limit {
                excess += *v - limit;
                *v = limit;
            }
        }
        let (bonus, rest) = (excess / 256, (excess % 256) as usize);
        for (k, v) in hist.iter_mut().enumerate() {
            *v += bonus + u32::from(k < rest);
        }
        let mut cdf = 0u32;
        for (k, v) in hist.iter().enumerate() {
            cdf += v;
            lut[k] = ((cdf as f64 * 255.0 / area as f64).round()).min(255.0) as u8;
        }
    });
    let cx = |i: usize| (tx0(i) + tx0(i + 1)) as f64 / 2.0;
    let cy = |j: usize| (ty0(j) + ty0(j + 1)) as f64 / 2.0;
    let mut out = vec![0u8; w * h];
    out.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
        let fy = (y as f64 - cy(0)) / (cy(tiles - 1) - cy(0)).max(1.0) * (tiles - 1) as f64;
        let j0 = (fy.floor().max(0.0) as usize).min(tiles - 1);
        let j1 = (j0 + 1).min(tiles - 1);
        let ay = (fy - j0 as f64).clamp(0.0, 1.0);
        for (x, o) in row.iter_mut().enumerate() {
            let fx = (x as f64 - cx(0)) / (cx(tiles - 1) - cx(0)).max(1.0) * (tiles - 1) as f64;
            let i0 = (fx.floor().max(0.0) as usize).min(tiles - 1);
            let i1 = (i0 + 1).min(tiles - 1);
            let ax = (fx - i0 as f64).clamp(0.0, 1.0);
            let v = src.get(x, y) as usize;
            let l = |i: usize, j: usize| luts[j * tiles + i][v] as f64;
            let top = l(i0, j0) * (1.0 - ax) + l(i1, j0) * ax;
            let bot = l(i0, j1) * (1.0 - ax) + l(i1, j1) * ax;
            *o = (top * (1.0 - ay) + bot * ay).round() as u8;
        }
    });
    Grid { w, h, data: out }
}

/// Edge-preserving bilateral filter over a disc of diameter `d`.
pub fn bilateral(src: &Grid<u8>, d: usize, sigma_color: f64, sigma_space: f64) -> Grid<u8> {
    let r = (d / 2) as isize;
    if r == 0 {
        return src.clone();
    }
    let mut offsets = Vec::new();
    for dy in -r..=r {
        for dx in -r..=r {
            let d2 = (dx * dx + dy * dy) as f64;
            if d2 <= (r * r) as f64 {
                offsets.push((dx, dy, (-d2 / (2.0 * sigma_space * sigma_space)).exp() as f32));
            }
        }
    }
    let color_w: Vec<f32> =
        (0..256).map(|i| (-((i * i) as f64) / (2.0 * sigma_color * sigma_color)).exp() as f32).collect();
    let (w, h) = (src.w as isize, src.h as isize);
    let mut out = vec![0u8; src.data.len()];
    out.par_chunks_mut(src.w).enumerate().for_each(|(y, row)| {
        for (x, o) in row.iter_mut().enumerate() {
            let c = src.get(x, y) as i32;
            let (mut acc, mut wsum) = (0f32, 0f32);
            for &(dx, dy, sw) in &offsets {
                let xx = (x as isize + dx).clamp(0, w - 1) as usize;
                let yy = (y as isize + dy).clamp(0, h - 1) as usize;
                let v = src.get(xx, yy) as i32;
                let wt = sw * color_w[(v - c).unsigned_abs() as usize];
                acc += wt * v as f32;
                wsum += wt;
            }
            *o = (acc / wsum).round() as u8;
        }
    });
    Grid { w: src.w, h: src.h, data: out }
}

/// Otsu's threshold: the level that best separates the histogram in two.
pub fn otsu(src: &Grid<u8>) -> u8 {
    let mut hist = [0u64; 256];
    src.data.iter().for_each(|&v| hist[v as usize] += 1);
    let total = src.data.len() as f64;
    let sum_all: f64 = hist.iter().enumerate().map(|(i, &c)| i as f64 * c as f64).sum();
    let (mut w_b, mut sum_b, mut best, mut best_t) = (0.0, 0.0, -1.0, 0u8);
    for (t, &c) in hist.iter().enumerate() {
        w_b += c as f64;
        if w_b == 0.0 {
            continue;
        }
        let w_f = total - w_b;
        if w_f == 0.0 {
            break;
        }
        sum_b += t as f64 * c as f64;
        let (m_b, m_f) = (sum_b / w_b, (sum_all - sum_b) / w_f);
        let between = w_b * w_f * (m_b - m_f) * (m_b - m_f);
        if between > best {
            best = between;
            best_t = t as u8;
        }
    }
    best_t
}

/// Canny edge detector (3x3 Sobel, L1 magnitude, hysteresis), like OpenCV's
/// default. Returns `true` on edge pixels.
pub fn canny(src: &Grid<u8>, low: f32, high: f32) -> Grid<bool> {
    let (w, h) = (src.w, src.h);
    let mut out = Grid::new(w, h, false);
    if w < 3 || h < 3 {
        return out;
    }
    let px = |x: usize, y: usize| src.get(x, y) as i32;
    let mut mag = vec![0f32; w * h];
    let mut dir = vec![0u8; w * h];
    mag.par_chunks_mut(w).zip(dir.par_chunks_mut(w)).enumerate().for_each(|(y, (mrow, drow))| {
        if y == 0 || y == h - 1 {
            return;
        }
        for x in 1..w - 1 {
            let gx = px(x + 1, y - 1) + 2 * px(x + 1, y) + px(x + 1, y + 1)
                - px(x - 1, y - 1)
                - 2 * px(x - 1, y)
                - px(x - 1, y + 1);
            let gy = px(x - 1, y + 1) + 2 * px(x, y + 1) + px(x + 1, y + 1)
                - px(x - 1, y - 1)
                - 2 * px(x, y - 1)
                - px(x + 1, y - 1);
            mrow[x] = (gx.abs() + gy.abs()) as f32;
            // Quantise the gradient direction to 0/45/90/135 degrees.
            let (ax, ay) = (gx.abs() as f32, gy.abs() as f32);
            drow[x] = if ay <= ax * 0.414_213_57 {
                0
            } else if ay >= ax * 2.414_213_5 {
                2
            } else if (gx > 0) == (gy > 0) {
                1
            } else {
                3
            };
        }
    });
    // Non-maximum suppression, then strong/weak classification.
    let mut state = vec![0u8; w * h]; // 0 none, 1 weak, 2 strong
    state.par_chunks_mut(w).enumerate().for_each(|(y, srow)| {
        if y == 0 || y == h - 1 {
            return;
        }
        for (x, s) in srow.iter_mut().enumerate().take(w - 1).skip(1) {
            let i = y * w + x;
            let m = mag[i];
            if m <= low {
                continue;
            }
            let (a, b) = match dir[i] {
                0 => (mag[i - 1], mag[i + 1]),
                2 => (mag[i - w], mag[i + w]),
                1 => (mag[i - w - 1], mag[i + w + 1]),
                _ => (mag[i - w + 1], mag[i + w - 1]),
            };
            if m > a && m >= b {
                *s = if m > high { 2 } else { 1 };
            }
        }
    });
    let mut stack: Vec<usize> = (0..w * h).filter(|&i| state[i] == 2).collect();
    for &i in &stack {
        out.data[i] = true;
    }
    while let Some(i) = stack.pop() {
        let (x, y) = (i % w, i / w);
        for dy in -1isize..=1 {
            for dx in -1isize..=1 {
                let (nx, ny) = (x as isize + dx, y as isize + dy);
                if nx < 0 || ny < 0 || nx >= w as isize || ny >= h as isize {
                    continue;
                }
                let j = ny as usize * w + nx as usize;
                if state[j] == 1 && !out.data[j] {
                    out.data[j] = true;
                    stack.push(j);
                }
            }
        }
    }
    out
}

/// Exact Euclidean distance (px) from every cell to the nearest `true` cell
/// (Felzenszwalb-Huttenlocher). Cells with no feature anywhere get a large
/// value.
pub fn distance_to(features: &Grid<bool>) -> Grid<f32> {
    const INF: f64 = 1e20;
    let (w, h) = (features.w, features.h);
    let mut f: Vec<f64> = features.data.iter().map(|&b| if b { 0.0 } else { INF }).collect();
    // Columns.
    let cols: Vec<Vec<f64>> = (0..w)
        .into_par_iter()
        .map(|x| {
            let col: Vec<f64> = (0..h).map(|y| f[y * w + x]).collect();
            dt_1d(&col)
        })
        .collect();
    for (x, col) in cols.iter().enumerate() {
        for (y, v) in col.iter().enumerate() {
            f[y * w + x] = *v;
        }
    }
    // Rows.
    let data = f.par_chunks(w).flat_map_iter(|row| dt_1d(row).into_iter().map(|v| v.sqrt() as f32)).collect();
    Grid { w, h, data }
}

fn dt_1d(f: &[f64]) -> Vec<f64> {
    let n = f.len();
    if n == 0 {
        return Vec::new();
    }
    let mut d = vec![0.0; n];
    let mut v = vec![0usize; n];
    let mut z = vec![0.0f64; n + 1];
    let mut k = 0usize;
    z[0] = f64::NEG_INFINITY;
    z[1] = f64::INFINITY;
    let meet = |q: usize, p: usize| ((f[q] + (q * q) as f64) - (f[p] + (p * p) as f64)) / (2.0 * (q as f64 - p as f64));
    for q in 1..n {
        let mut s = meet(q, v[k]);
        // z[0] is -inf, so this stops at k = 0 at the latest.
        while s <= z[k] {
            k -= 1;
            s = meet(q, v[k]);
        }
        k += 1;
        v[k] = q;
        z[k] = s;
        z[k + 1] = f64::INFINITY;
    }
    k = 0;
    for (q, dq) in d.iter_mut().enumerate() {
        while z[k + 1] < q as f64 {
            k += 1;
        }
        let p = v[k];
        let dx = q as f64 - p as f64;
        *dq = dx * dx + f[p];
    }
    d
}

/// Signed distance (px): positive inside `mask`, negative outside, zero half
/// way between an inside and an outside pixel centre.
pub fn signed_distance(mask: &Grid<bool>) -> Grid<f32> {
    let to_outside = distance_to(&mask.map(|b| !b));
    let to_inside = distance_to(mask);
    let data = mask
        .data
        .par_iter()
        .zip(&to_outside.data)
        .zip(&to_inside.data)
        .map(|((&m, &dout), &din)| if m { dout - 0.5 } else { -(din - 0.5) })
        .collect();
    Grid { w: mask.w, h: mask.h, data }
}

/// Grow `mask` by a disc of radius `r` px.
pub fn dilate(mask: &Grid<bool>, r: f64) -> Grid<bool> {
    if r <= 0.0 {
        return mask.clone();
    }
    let r = r as f32;
    distance_to(mask).map(|d| d <= r)
}

/// Guo-Hall thinning to a one-pixel-wide skeleton. (Zhang-Suen erases
/// two-pixel-thick diagonals, which is exactly what Canny gives on a
/// diagonal edge.)
pub fn thin(mask: &Grid<bool>) -> Grid<bool> {
    let (w, h) = (mask.w, mask.h);
    let mut img: Vec<u8> = mask.data.iter().map(|&b| b as u8).collect();
    // The border never survives (matches the reference implementation).
    for x in 0..w {
        img[x] = 0;
        img[(h - 1) * w + x] = 0;
    }
    for y in 0..h {
        img[y * w] = 0;
        img[y * w + w - 1] = 0;
    }
    if w < 3 || h < 3 {
        return Grid { w, h, data: img.iter().map(|&v| v != 0).collect() };
    }
    loop {
        let mut changed = false;
        for sub in 0..2 {
            let del: Vec<usize> = (1..h - 1)
                .into_par_iter()
                .flat_map_iter(|y| {
                    let img = &img;
                    (1..w - 1).filter_map(move |x| {
                        let i = y * w + x;
                        if img[i] == 0 {
                            return None;
                        }
                        let p = [
                            img[i - w],
                            img[i - w + 1],
                            img[i + 1],
                            img[i + w + 1],
                            img[i + w],
                            img[i + w - 1],
                            img[i - 1],
                            img[i - w - 1],
                        ];
                        let [p2, p3, p4, p5, p6, p7, p8, p9] = p.map(|v| v != 0);
                        let c = u8::from(!p2 && (p3 || p4))
                            + u8::from(!p4 && (p5 || p6))
                            + u8::from(!p6 && (p7 || p8))
                            + u8::from(!p8 && (p9 || p2));
                        let n1 = u8::from(p9 || p2) + u8::from(p3 || p4) + u8::from(p5 || p6) + u8::from(p7 || p8);
                        let n2 = u8::from(p2 || p3) + u8::from(p4 || p5) + u8::from(p6 || p7) + u8::from(p8 || p9);
                        let n = n1.min(n2);
                        let m = if sub == 0 { (p6 || p7 || !p9) && p8 } else { (p2 || p3 || !p5) && p4 };
                        (c == 1 && (2..=3).contains(&n) && !m).then_some(i)
                    })
                })
                .collect();
            if !del.is_empty() {
                changed = true;
                for i in del {
                    img[i] = 0;
                }
            }
        }
        if !changed {
            break;
        }
    }
    Grid { w, h, data: img.iter().map(|&v| v != 0).collect() }
}

/// 8-connected components of `mask`: per-cell component id (`u32::MAX` off
/// the mask) and each component's size.
pub fn components(mask: &Grid<bool>) -> (Grid<u32>, Vec<usize>) {
    let (w, h) = (mask.w, mask.h);
    let mut label = Grid::new(w, h, u32::MAX);
    let mut sizes = Vec::new();
    let mut stack = Vec::new();
    for start in 0..w * h {
        if !mask.data[start] || label.data[start] != u32::MAX {
            continue;
        }
        let id = sizes.len() as u32;
        let mut size = 0;
        label.data[start] = id;
        stack.push(start);
        while let Some(i) = stack.pop() {
            size += 1;
            let (x, y) = ((i % w) as isize, (i / w) as isize);
            for dy in -1..=1 {
                for dx in -1..=1 {
                    let (nx, ny) = (x + dx, y + dy);
                    if nx < 0 || ny < 0 || nx >= w as isize || ny >= h as isize {
                        continue;
                    }
                    let j = ny as usize * w + nx as usize;
                    if mask.data[j] && label.data[j] == u32::MAX {
                        label.data[j] = id;
                        stack.push(j);
                    }
                }
            }
        }
        sizes.push(size);
    }
    (label, sizes)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn square_mask(n: usize, lo: usize, hi: usize) -> Grid<bool> {
        let mut m = Grid::new(n, n, false);
        for y in lo..hi {
            for x in lo..hi {
                m.set(x, y, true);
            }
        }
        m
    }

    #[test]
    fn distance_transform_matches_brute_force() {
        let mut f = Grid::new(17, 11, false);
        f.set(3, 4, true);
        f.set(12, 9, true);
        f.set(16, 0, true);
        let d = distance_to(&f);
        for y in 0..11 {
            for x in 0..17 {
                let want = [(3, 4), (12, 9), (16, 0)]
                    .iter()
                    .map(|&(fx, fy): &(i32, i32)| (((x as i32 - fx).pow(2) + (y as i32 - fy).pow(2)) as f32).sqrt())
                    .fold(f32::MAX, f32::min);
                assert!((d.get(x, y) - want).abs() < 1e-4, "({x},{y}) {} vs {want}", d.get(x, y));
            }
        }
    }

    #[test]
    fn signed_distance_is_zero_on_the_boundary() {
        let m = square_mask(20, 5, 15);
        let sd = signed_distance(&m);
        assert_eq!(sd.get(5, 10), 0.5); // first inside pixel: 1 px to outside, -0.5
        assert_eq!(sd.get(4, 10), -0.5);
        assert_eq!(sd.get(10, 10), 4.5);
        // Half way between pixel 4 and 5 the field crosses zero.
        assert!(sd.sample(4.5, 10.0, -1.0).abs() < 1e-6);
    }

    #[test]
    fn gaussian_blur_preserves_constant_and_mass() {
        let g = Grid::new(30, 20, 7.0f32);
        let b = gaussian_blur(&g, 2.0);
        assert!(b.data.iter().all(|&v| (v - 7.0).abs() < 1e-4));
        let mut spike = Grid::new(41, 41, 0.0f32);
        spike.set(20, 20, 100.0);
        let b = gaussian_blur(&spike, 1.5);
        let total: f32 = b.data.iter().sum();
        assert!((total - 100.0).abs() < 1e-2);
        assert!(b.get(20, 20) > b.get(21, 20) && b.get(21, 20) > b.get(23, 20));
    }

    #[test]
    fn otsu_splits_two_levels() {
        let mut g = Grid::new(10, 10, 40u8);
        for i in 0..50 {
            g.data[i] = 200;
        }
        let t = otsu(&g);
        assert!((40..200).contains(&t));
    }

    #[test]
    fn canny_finds_the_outline_of_a_square_only() {
        let m = square_mask(40, 10, 30);
        let g = m.map(|b| if b { 0u8 } else { 255u8 });
        let e = canny(&g, 100.0, 200.0);
        assert!(e.get(10, 20) || e.get(9, 20));
        assert!(!e.get(20, 20));
        assert!(!e.get(2, 2));
        let n = e.data.iter().filter(|&&b| b).count();
        assert!((60..=100).contains(&n), "edge pixels {n}");
    }

    #[test]
    fn thinning_reduces_a_bar_to_a_line() {
        let mut m = Grid::new(40, 15, false);
        for y in 5..10 {
            for x in 5..35 {
                m.set(x, y, true);
            }
        }
        let t = thin(&m);
        for x in 10..30 {
            let col = (0..15).filter(|&y| t.get(x, y)).count();
            assert_eq!(col, 1, "column {x}");
        }
    }

    #[test]
    fn thinning_keeps_a_two_pixel_diagonal() {
        let mut m = Grid::new(60, 60, false);
        for i in 5..50 {
            m.set(i, i, true);
            m.set(i + 1, i, true);
        }
        let t = thin(&m);
        let n = t.data.iter().filter(|&&b| b).count();
        assert!(n >= 40, "{n} pixels left");
    }

    #[test]
    fn components_counts_blobs_and_sizes() {
        let mut m = square_mask(20, 2, 5);
        m.set(15, 15, true);
        m.set(16, 16, true); // diagonal: same 8-connected blob
        let (_, sizes) = components(&m);
        let mut s = sizes.clone();
        s.sort();
        assert_eq!(s, vec![2, 9]);
    }

    #[test]
    fn clahe_keeps_a_flat_image_flat_and_stretches_contrast() {
        let flat = Grid::new(64, 64, 100u8);
        let out = clahe(&flat, 2.0, 8);
        let v0 = out.data[0];
        assert!(out.data.iter().all(|&v| v == v0));
        let mut ramp = Grid::new(64, 64, 0u8);
        for y in 0..64 {
            for x in 0..64 {
                ramp.set(x, y, 100 + (x / 8) as u8);
            }
        }
        let out = clahe(&ramp, 40.0, 1);
        let (mn, mx) = (out.data.iter().min().unwrap(), out.data.iter().max().unwrap());
        assert!(mx - mn > 100, "{mn}..{mx}");
    }

    #[test]
    fn bilateral_keeps_a_step_edge() {
        let mut g = Grid::new(20, 20, 0u8);
        for y in 0..20 {
            for x in 10..20 {
                g.set(x, y, 200);
            }
        }
        let b = bilateral(&g, 5, 35.0, 35.0);
        assert_eq!(b.get(9, 10), 0);
        assert_eq!(b.get(10, 10), 200);
    }
}
