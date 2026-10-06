"""Image I/O and pixel work shared by the raster styles.

Page coordinates are millimetres, origin top-left, y down; pixel (col, row)
centre sits at ((col + 0.5) * mm_per_px, (row + 0.5) * mm_per_px).
"""

from __future__ import annotations

import cv2
import numpy as np
import shapely
from PIL import Image, ImageOps
from shapely.geometry import MultiPolygon, Polygon
from skimage.measure import find_contours


def load_rgb(path) -> np.ndarray:
    """(H, W, 3) uint8 RGB: EXIF orientation applied, alpha composited onto white."""
    with Image.open(path) as im:
        img = ImageOps.exif_transpose(im)
        if img.mode in ("I;16", "I;16B", "I;16L", "I", "F"):
            a = np.asarray(img).astype(np.float64)
            top = 65535.0 if img.mode.startswith("I;16") or a.max() > 255 else 255.0
            img = Image.fromarray(np.clip(a / top * 255.0, 0, 255).astype(np.uint8), "L")
        rgba = img.convert("RGBA")
    bg = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
    bg.alpha_composite(rgba)
    return np.asarray(bg.convert("RGB"), dtype=np.uint8).copy()


def fit_page(src_w, src_h, cfg):
    """Fit the source into the print area keeping aspect: (width_mm, height_mm, mm_per_src_px)."""
    s = min(cfg.machine.print_width / src_w, cfg.machine.print_height / src_h)
    return src_w * s, src_h * s, s


def working_image(rgb, mm_per_src_px, mm_per_px, max_side=2500):
    """Resample to about *mm_per_px* (longest side capped at *max_side*): (img, actual_mm_per_px)."""
    H, W = rgb.shape[:2]
    f = min(mm_per_src_px / mm_per_px, max_side / max(H, W))
    w, h = max(8, int(round(W * f))), max(8, int(round(H * f)))
    if (w, h) == (W, H):
        return rgb.copy(), mm_per_src_px
    interp = cv2.INTER_AREA if f < 1 else cv2.INTER_CUBIC
    return cv2.resize(rgb, (w, h), interpolation=interp), W * mm_per_src_px / w


def estimate_noise(gray) -> float:
    """Gaussian noise sigma in 0-255 units (Immerkaer 1996). RGB is averaged to gray first."""
    g = np.asarray(gray, np.float32)
    if g.ndim == 3:
        g = g.mean(axis=2)
    H, W = g.shape
    if H < 3 or W < 3:
        return 0.0
    k = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], np.float32)
    conv = cv2.filter2D(g, -1, k, borderType=cv2.BORDER_REFLECT)[1:-1, 1:-1]
    return float(np.sqrt(np.pi / 2) * np.abs(conv).sum() / (6 * (W - 2) * (H - 2)))


def denoise(img, sigma=None):
    """Non-local means with h = 1.1 * sigma; returns *img* itself when sigma <= 1.5."""
    if sigma is None:
        sigma = estimate_noise(img)
    if sigma <= 1.5:
        return img
    h = 1.1 * sigma
    if img.ndim == 2:
        return cv2.fastNlMeansDenoising(img, None, h=h, templateWindowSize=7, searchWindowSize=21)
    # OpenCV expects BGR here (it splits luminance from colour internally).
    bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return cv2.cvtColor(cv2.fastNlMeansDenoisingColored(bgr, None, h, h, 7, 21), cv2.COLOR_BGR2RGB)


def luminance(rgb) -> np.ndarray:
    """Lab lightness L scaled to [0, 1], float32."""
    lab = cv2.cvtColor(np.asarray(rgb, np.float32) / 255.0, cv2.COLOR_RGB2Lab)
    return np.clip(lab[..., 0] / 100.0, 0.0, 1.0)


def foreground_mask(rgb, white_threshold) -> np.ndarray:
    """True where the pixel is not paper (paper = every channel >= white_threshold)."""
    return ~np.all(np.asarray(rgb) >= white_threshold, axis=2)


def _nearest(points, centres, chunk=500_000) -> np.ndarray:
    out = np.empty(len(points), np.int64)
    for s in range(0, len(points), chunk):
        d = ((points[s:s + chunk, None, :] - centres[None]) ** 2).sum(axis=2)
        out[s:s + chunk] = d.argmin(axis=1)
    return out


def palette_kmeans(rgb, mask, n, max_samples=200_000):
    """Lab k-means over the masked pixels.

    Returns (palette (k, 3) uint8 sorted dark -> light, labels (H, W) int32 with -1
    outside the mask). k <= min(n, masked pixels); clusters no pixel ends up in
    (e.g. more clusters than distinct colours) are dropped.
    """
    labels = np.full(mask.shape, -1, np.int32)
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2Lab)[mask].astype(np.float32)
    if len(lab) == 0:
        return np.zeros((0, 3), np.uint8), labels
    sample = lab
    if len(lab) > max_samples:
        sample = lab[np.random.default_rng(0).choice(len(lab), max_samples, replace=False)]
    k = min(n, len(sample))
    if k == len(sample):
        # Every sample is its own cluster. (Also: OpenCV 5 reads a (1, 3) array as three 1-D samples.)
        centres = sample
    else:
        cv2.setRNGSeed(0)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.5)
        _, _, centres = cv2.kmeans(sample, k, None, criteria, 3, cv2.KMEANS_PP_CENTERS)
    centres = centres[np.argsort(centres[:, 0], kind="stable")]
    centres = np.clip(np.round(centres), 0, 255).astype(np.uint8)
    used, idx = np.unique(_nearest(lab, centres.astype(np.float32)), return_inverse=True)
    labels[mask] = idx
    palette = cv2.cvtColor(centres[used][None], cv2.COLOR_Lab2RGB)[0]
    return palette, labels


def mask_to_geometry(mask, mm_per_px, simplify_mm=0.0, min_area_mm2=0.0):
    """Vectorize a binary mask (marching squares at 0.5) into a valid shapely geometry in page mm.

    Holes and islands nest to any depth. Holes and polygons smaller than
    *min_area_mm2* are dropped (a dropped hole is filled). Returns an empty
    Polygon when nothing is left.
    """
    rings = find_contours(np.pad(np.asarray(mask, bool), 1).astype(np.float32), 0.5)
    polys = [Polygon((c[:, ::-1] - 0.5) * mm_per_px) for c in rings]  # (row, col) -> (x, y), undo pad
    # A ring below min area can go now: everything nested inside it is smaller still.
    polys = sorted((p for p in polys if p.area > 0 and p.area >= min_area_mm2), key=lambda p: p.area, reverse=True)
    if not polys:
        return Polygon()
    # Marching-squares rings never cross or touch, so nesting depth = number of
    # larger rings around a ring: even depth is a shell, odd depth a hole.
    tree = shapely.STRtree(polys)
    inner, outer = tree.query(shapely.point_on_surface(polys), predicate="within")
    keep = outer < inner  # larger rings only (sorted by area, descending)
    inner, outer = inner[keep], outer[keep]
    depth = np.bincount(inner, minlength=len(polys))
    parent = np.full(len(polys), -1)
    np.maximum.at(parent, inner, outer)  # innermost container = smallest = highest index
    holes = {i: [] for i in np.flatnonzero(depth % 2 == 0)}
    for i in np.flatnonzero(depth % 2 == 1):
        holes[parent[i]].append(polys[i].exterior)
    parts = [Polygon(polys[i].exterior, hs) for i, hs in holes.items()]
    parts = [p for p in parts if p.area >= min_area_mm2]
    if simplify_mm > 0:
        # Part by part: simplifying the whole MultiPolygon at once is quadratic in the part count.
        parts = list(shapely.simplify(parts, simplify_mm, preserve_topology=True))
    # The parts never touch, so they form a MultiPolygon without a (slow) union.
    geom = parts[0] if len(parts) == 1 else MultiPolygon(parts)
    if not geom.is_valid:
        geom = geom.buffer(0)
    return geom if not geom.is_empty else Polygon()


def tone_levels(lum, mask, n) -> np.ndarray:
    """n - 1 ascending thresholds splitting the masked luminance into n equal-count bands.

    With an empty mask the thresholds are evenly spaced over [0, 1].
    """
    q = np.arange(1, n) / n
    vals = lum[mask]
    return np.quantile(vals, q) if vals.size else q


def sample_colors(rgb, points_mm, mm_per_px) -> np.ndarray:
    """(N, 3) uint8 colour of the pixel under each page-mm point (clamped to the image)."""
    H, W = rgb.shape[:2]
    p = np.asarray(points_mm, np.float64).reshape(-1, 2)
    cols = np.clip(np.floor(p[:, 0] / mm_per_px).astype(int), 0, W - 1)
    rows = np.clip(np.floor(p[:, 1] / mm_per_px).astype(int), 0, H - 1)
    return rgb[rows, cols].astype(np.uint8)
