import time

import numpy as np
import pytest
from shapely.geometry import LineString

from img2gcode.geometry import polyline_length
from img2gcode.styles.contour import make
from tests.synth import LW, assert_valid_artwork, radial_rgb


def test_contract(cfg):
    a = make(radial_rgb(), cfg)
    assert_valid_artwork(a, cfg)
    assert a.style == "contour" and a.regions == []


def test_rings_on_radial_gradient(cfg):
    cfg.contour.levels = 6
    # Square, so 4 of the 6 equal-area rings fit inside the image (on 700x500 the 4th
    # quantile ring is wider than the image is tall and cannot close).
    a = make(radial_rgb(600, 600), cfg)
    closed = [s for s in a.strokes if np.allclose(s.points[0], s.points[-1])]
    assert len(closed) >= 4
    cx, cy = a.width_mm / 2, a.height_mm / 2
    for s in closed:
        c = s.points.mean(axis=0)
        assert np.hypot(c[0] - cx, c[1] - cy) < 0.1 * a.width_mm


def test_lines_keep_their_distance(cfg):
    a = make(radial_rgb(noise=20), cfg)
    lines = [LineString(s.points) for s in a.strokes]
    for i in range(len(lines)):
        for j in range(i + 1, len(lines)):
            assert lines[i].distance(lines[j]) >= 0.9 * LW - 0.1


def test_noise_robust(cfg):
    a = make(radial_rgb(noise=40), cfg)
    assert len(a.strokes) <= 3 * cfg.contour.levels
    assert all(polyline_length(s.points) >= 3 * LW for s in a.strokes)


def test_tools_follow_levels(cfg):
    cfg.tools.num_tools = 2
    a = make(radial_rgb(), cfg)
    assert {s.tool for s in a.strokes} == {0, 1}
    centre = np.array([a.width_mm / 2, a.height_mm / 2])
    inner = min(a.strokes, key=lambda s: np.linalg.norm(s.points.mean(axis=0) - centre))
    assert inner.tool == 0


def test_blank_image(cfg):
    a = make(np.full((100, 120, 3), 128, np.uint8), cfg)
    assert_valid_artwork(a, cfg, allow_empty=True)
    assert a.strokes == []


@pytest.mark.slow
def test_runtime_large(cfg):
    big = radial_rgb(1500, 2000, noise=30)
    t = time.perf_counter(); make(big, cfg)
    assert time.perf_counter() - t < 30
