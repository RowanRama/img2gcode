"""Synthetic images with known geometry, plus shared assertions."""

from __future__ import annotations

import cv2
import numpy as np
from scipy.spatial import cKDTree

SHAPES_W, SHAPES_H = 800, 600
CIRCLE = ((250, 300), 150)          # centre (x, y) px, radius px
RECT = (480, 150, 720, 450)         # x0, y0, x1, y1 px (inclusive)
LW = 4.0                            # default line width (mm)


def page_scale(cfg, w: int, h: int) -> float:
    return min(cfg.machine.print_width / w, cfg.machine.print_height / h)


def _noisy(img: np.ndarray, noise: float, seed: int) -> np.ndarray:
    if noise <= 0:
        return img
    rng = np.random.default_rng(seed)
    return np.clip(img + rng.normal(0, noise, img.shape), 0, 255).astype(np.uint8)


def shapes_rgb(noise: float = 0.0, seed: int = 0) -> np.ndarray:
    """Dark circle + red rectangle on light grey, 800x600."""
    img = np.full((SHAPES_H, SHAPES_W, 3), 235, np.uint8)
    cv2.circle(img, CIRCLE[0], CIRCLE[1], (40, 40, 40), -1)
    cv2.rectangle(img, RECT[:2], RECT[2:], (200, 60, 60), -1)
    return _noisy(img, noise, seed)


def shapes_outline_mm(cfg, step: float = 1.0) -> np.ndarray:
    """Points along the true outlines of shapes_rgb(), in page mm."""
    s = page_scale(cfg, SHAPES_W, SHAPES_H)
    (cx, cy), r = CIRCLE
    n = max(16, int(2 * np.pi * r * s / step))
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    circle = np.c_[(cx + 0.5 + r * np.cos(t)) * s, (cy + 0.5 + r * np.sin(t)) * s]
    x0, y0, x1, y1 = RECT
    corners = np.array([[x0, y0], [x1 + 1, y0], [x1 + 1, y1 + 1], [x0, y1 + 1], [x0, y0]], float) * s
    return np.vstack([circle, dense(corners, step)])


def radial_rgb(h: int = 500, w: int = 700, noise: float = 0.0, seed: int = 0) -> np.ndarray:
    """Dark centre fading to near-white at the corners."""
    yy, xx = np.mgrid[0:h, 0:w]
    r = np.hypot(xx - w / 2, yy - h / 2) / np.hypot(w / 2, h / 2)
    v = (30 + 205 * r).astype(np.uint8)
    return _noisy(np.dstack([v, v, v]), noise, seed)


def split_rgb(left=(0, 0, 0), right=(255, 255, 255), h: int = 400, w: int = 800) -> np.ndarray:
    img = np.empty((h, w, 3), np.uint8)
    img[:, : w // 2] = left
    img[:, w // 2:] = right
    return img


def bands_rgb(values=(20, 80, 140, 200), h: int = 400, w: int = 800) -> np.ndarray:
    """Vertical grey bands, darkest on the left."""
    img = np.empty((h, w, 3), np.uint8)
    bw = w // len(values)
    for i, v in enumerate(values):
        img[:, i * bw:(i + 1) * bw] = v
    return img


def dense(points: np.ndarray, step: float) -> np.ndarray:
    """Resample a polyline so consecutive points are about *step* apart."""
    p = np.asarray(points, float)
    if len(p) < 2:
        return p
    seg = np.hypot(*np.diff(p, axis=0).T)
    s = np.r_[0.0, np.cumsum(seg)]
    if s[-1] == 0:
        return p[:1]
    t = np.linspace(0, s[-1], max(2, int(np.ceil(s[-1] / step)) + 1))
    return np.c_[np.interp(t, s, p[:, 0]), np.interp(t, s, p[:, 1])]


def recall_precision(polylines, truth: np.ndarray, tol: float, step: float = 1.0):
    """(recall, precision) of drawn polylines against truth points, within tol mm."""
    if not polylines:
        return 0.0, 0.0
    pts = np.vstack([dense(p, step) for p in polylines])
    recall = float(np.mean(cKDTree(pts).query(truth)[0] <= tol))
    precision = float(np.mean(cKDTree(truth).query(pts)[0] <= tol))
    return recall, precision


def total_length(polylines) -> float:
    return float(sum(np.hypot(*np.diff(np.asarray(p, float), axis=0).T).sum() for p in polylines))


def assert_valid_artwork(art, cfg, allow_empty: bool = False) -> None:
    eps = 1e-6
    assert 0 < art.width_mm <= cfg.machine.print_width + eps
    assert 0 < art.height_mm <= cfg.machine.print_height + eps
    n = len(art.palette)
    assert n >= 1
    for c in art.palette:
        assert len(c) == 3 and all(0 <= int(v) <= 255 for v in c)
    for s in art.strokes:
        assert s.points.ndim == 2 and s.points.shape[1] == 2 and len(s.points) >= 2
        assert 0 <= s.tool < n
        assert s.points[:, 0].min() >= -eps and s.points[:, 1].min() >= -eps
        assert s.points[:, 0].max() <= art.width_mm + eps
        assert s.points[:, 1].max() <= art.height_mm + eps
    for r in art.regions:
        assert 0 <= r.tool < n
        assert not r.shape.is_empty and r.shape.is_valid
        minx, miny, maxx, maxy = r.shape.bounds
        assert minx >= -eps and miny >= -eps
        assert maxx <= art.width_mm + eps and maxy <= art.height_mm + eps
    if not allow_empty:
        assert art.regions or art.strokes
