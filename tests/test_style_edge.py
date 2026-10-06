import time

import cv2
import numpy as np
import pytest

from img2gcode.raster import estimate_noise
from img2gcode.styles.edge import _noise_map, make
from tests.synth import (
    CIRCLE, LW, SHAPES_H, SHAPES_W, assert_valid_artwork, page_scale, recall_precision, shapes_outline_mm,
    shapes_rgb, total_length,
)


def test_contract(cfg):
    a = make(shapes_rgb(), cfg)
    assert_valid_artwork(a, cfg)
    assert a.style == "edge" and a.regions == [] and all(s.kind == "line" for s in a.strokes)


def test_noisy_shapes_recall_and_precision(cfg):
    cfg.tools.num_tools = 1
    a = make(shapes_rgb(noise=45), cfg)
    r, p = recall_precision([s.points for s in a.strokes], shapes_outline_mm(cfg), tol=1.5 * LW)
    assert r >= 0.85, f"recall {r:.2f}"
    assert p >= 0.85, f"precision {p:.2f}"


def test_clean_shapes(cfg):
    a = make(shapes_rgb(), cfg)
    r, p = recall_precision([s.points for s in a.strokes], shapes_outline_mm(cfg), tol=1.5 * LW)
    assert r >= 0.9 and p >= 0.9


def test_thin_dark_line_is_drawn_once(cfg):
    img = np.full((300, 600, 3), 235, np.uint8)
    cv2.line(img, (50, 150), (550, 150), (0, 0, 0), 3)
    a = make(img, cfg)
    true_len = 500 * page_scale(cfg, 600, 300)
    assert 0.8 * true_len <= total_length([s.points for s in a.strokes]) <= 1.2 * true_len


def test_multi_tool_palette(cfg):
    cfg.tools.num_tools = 2
    a = make(shapes_rgb(), cfg)
    assert len(a.palette) == 2 and {s.tool for s in a.strokes} <= {0, 1}


def test_detail_adds_ink(cfg):
    img = shapes_rgb()
    for i in range(8):
        cv2.circle(img, (60 + 90 * i, 540), 12, (90, 90, 90), 2)
    cfg.edge.detail = 0.1
    low = total_length([s.points for s in make(img, cfg).strokes])
    cfg.edge.detail = 0.9
    high = total_length([s.points for s in make(img, cfg).strokes])
    assert high >= low


def test_debug_edges_path(cfg, tmp_path):
    cfg.edge.debug_edges_path = str(tmp_path / "edges.png")
    make(shapes_rgb(), cfg)
    m = cv2.imread(str(tmp_path / "edges.png"), cv2.IMREAD_GRAYSCALE)
    assert m is not None and set(np.unique(m)) <= {0, 255}


def test_blank_image(cfg):
    a = make(np.full((100, 120, 3), 240, np.uint8), cfg)
    assert_valid_artwork(a, cfg, allow_empty=True)
    assert a.strokes == []


@pytest.mark.slow
def test_runtime_large(cfg):
    big = cv2.resize(shapes_rgb(noise=40), (2000, 1500), interpolation=cv2.INTER_NEAREST)
    t = time.perf_counter(); make(big, cfg)
    assert time.perf_counter() - t < 30


def _region_noise(img, region, sigma_in, sigma_out, seed=0):
    """Gaussian noise of sigma_in inside *region* (bool mask) and sigma_out elsewhere."""
    sigma = np.where(region[..., None], sigma_in, sigma_out)
    noise = np.random.default_rng(seed).normal(0, 1, img.shape) * sigma
    return np.clip(img + noise, 0, 255).astype(np.uint8)


def test_shadow_noise(cfg):
    """Noise that is strong only in the dark circle (like sensor noise) must not turn into squiggles."""
    cfg.tools.num_tools = 1
    circle = np.zeros((SHAPES_H, SHAPES_W), np.uint8)
    cv2.circle(circle, *CIRCLE, 1, -1)
    a = make(_region_noise(shapes_rgb(), circle > 0, 45, 10), cfg)
    r, p = recall_precision([s.points for s in a.strokes], shapes_outline_mm(cfg), tol=1.5 * LW)
    assert r >= 0.85, f"recall {r:.2f}"
    assert p >= 0.85, f"precision {p:.2f}"


def test_noiseless_margin(cfg):
    """A clean border around a noisy picture must not dilute the noise estimate for the picture."""
    cfg.tools.num_tools = 1
    pad = 120
    img = cv2.copyMakeBorder(shapes_rgb(noise=45), pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=(235, 235, 235))
    a = make(img, cfg)
    s0, s1 = page_scale(cfg, SHAPES_W, SHAPES_H), page_scale(cfg, img.shape[1], img.shape[0])
    truth = (shapes_outline_mm(cfg) / s0 + pad) * s1
    r, p = recall_precision([s.points for s in a.strokes], truth, tol=1.5 * LW)
    assert r >= 0.85, f"recall {r:.2f}"
    assert p >= 0.85, f"precision {p:.2f}"


def test_noise_map_ignores_hatching():
    """Isotropic noise raises the local denoising strength; fine oriented hatching does not."""
    hatch = np.full((300, 300), 200, np.uint8)
    for x in range(-300, 300, 3):
        cv2.line(hatch, (x, 0), (x + 300, 300), 140, 1)
    noisy = np.clip(200 + np.random.default_rng(0).normal(0, 30, (300, 300)), 0, 255).astype(np.uint8)
    L = np.hstack([noisy, np.full((300, 600), 200, np.uint8), hatch])
    g = estimate_noise(L)
    assert estimate_noise(hatch) > 2 * g                # the hatching alone reads as strong noise
    m = _noise_map(L, 3.0)
    assert m[150, 100] > 1.25 * g                       # noisy third: raised above the global estimate
    assert m[150, 1100] == pytest.approx(g, rel=1e-3)   # hatched third: left at the global estimate
