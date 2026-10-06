import cv2
import numpy as np
import pytest

from img2gcode.geometry import polyline_length
from img2gcode.skeleton import skeleton_strokes

KW = dict(spur_mm=8.0, smooth_mm=2.0, simplify_mm=0.2, min_length_mm=6.0)


def mask(lines=(), circles=(), shape=(200, 200), t=5):
    m = np.zeros(shape, np.uint8)
    for a, b in lines:
        cv2.line(m, a, b, 1, t)
    for c, r in circles:
        cv2.circle(m, c, r, 1, t)
    return m.astype(bool)


def straightness(p):
    return np.hypot(*(p[-1] - p[0])) / polyline_length(p)


def test_single_line():
    s = skeleton_strokes(mask([((20, 100), (180, 100))]), 1.0, **KW)
    assert len(s) == 1
    assert polyline_length(s[0]) == pytest.approx(160, abs=8) and straightness(s[0]) > 0.98


def test_plus_gives_two_straight_strokes():
    s = skeleton_strokes(mask([((20, 100), (180, 100)), ((100, 20), (100, 180))]), 1.0, **KW)
    assert len(s) == 2
    assert all(polyline_length(p) > 140 and straightness(p) > 0.95 for p in s)


def test_t_junction_keeps_the_bar_whole():
    s = skeleton_strokes(mask([((20, 50), (180, 50)), ((100, 50), (100, 180))]), 1.0, **KW)
    assert len(s) == 2
    bar = max(s, key=lambda p: np.ptp(p[:, 0]))
    assert np.ptp(bar[:, 0]) > 150 and straightness(bar) > 0.97


def test_ring_is_closed():
    s = skeleton_strokes(mask(circles=[((100, 100), 60)]), 1.0, **KW)
    assert len(s) == 1 and np.allclose(s[0][0], s[0][-1])
    assert polyline_length(s[0]) == pytest.approx(2 * np.pi * 60, rel=0.05)


def test_short_spur_is_pruned():
    s = skeleton_strokes(mask([((20, 100), (180, 100)), ((100, 100), (100, 106))]), 1.0, **KW)
    assert len(s) == 1


def test_specks_dropped_and_empty():
    m = np.zeros((50, 50), bool); m[20:23, 20:23] = True
    assert skeleton_strokes(m, 1.0, **KW) == []
    assert skeleton_strokes(np.zeros((50, 50), bool), 1.0, **KW) == []


def test_pixel_to_mm_mapping():
    s = skeleton_strokes(mask([((20, 100), (180, 100))]), 0.5, spur_mm=4, smooth_mm=1, simplify_mm=0.1, min_length_mm=3)
    p = s[0]
    assert p[:, 1].mean() == pytest.approx(50.25, abs=0.3)
    assert p[:, 0].min() == pytest.approx(10.25, abs=2) and p[:, 0].max() == pytest.approx(90.25, abs=2)
