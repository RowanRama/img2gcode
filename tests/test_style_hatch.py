import time

import cv2
import numpy as np
import pytest

from img2gcode.styles.hatch import make
from tests.synth import LW, assert_valid_artwork, bands_rgb, radial_rgb


def angle_of(s):
    d = s.points[-1] - s.points[0]
    return round(float(np.degrees(np.arctan2(d[1], d[0]))) % 180.0 / 5.0) * 5 % 180


def test_contract(cfg):
    a = make(radial_rgb(), cfg)
    assert_valid_artwork(a, cfg)
    assert a.style == "hatch" and a.regions == []


def test_darker_bands_get_more_directions(cfg):
    a = make(bands_rgb(), cfg)
    bw = a.width_mm / 4
    counts = []
    for b in range(4):
        lo, hi = b * bw + 2 * LW, (b + 1) * bw - 2 * LW
        mids = [s for s in a.strokes if lo < s.points.mean(axis=0)[0] < hi]
        counts.append(len({angle_of(s) for s in mids}))
    assert counts == [3, 2, 1, 0]


def test_strokes_are_straight_segments(cfg):
    assert all(len(s.points) == 2 for s in make(radial_rgb(), cfg).strokes)


def test_spacing(cfg):
    cfg.hatch.angles = "0"; cfg.hatch.bands = 2
    a = make(bands_rgb((20, 200)), cfg)
    ys = np.unique(np.round([s.points[0, 1] for s in a.strokes], 3))
    assert np.allclose(np.diff(ys), cfg.hatch.spacing * LW, atol=0.05)


def test_tools_by_direction(cfg):
    cfg.tools.num_tools = 2
    assert {s.tool for s in make(bands_rgb(), cfg).strokes} == {0, 1}


def test_noise_robust(cfg):
    clean = len(make(radial_rgb(), cfg).strokes)
    noisy = len(make(radial_rgb(noise=40), cfg).strokes)
    assert noisy <= 2 * clean


def test_flat_colours_are_hatched_cleanly(cfg):
    # Two flat ellipses, upscaled: a tone quantile lands on the yellow plateau,
    # which must be hatched whole, not as a speckled rim of short dashes.
    img = np.full((256, 256, 3), 255, np.uint8)
    cv2.ellipse(img, (90, 108), (49, 69), 0, 0, 360, (55, 118, 171), -1)
    cv2.ellipse(img, (161, 143), (49, 69), 0, 0, 360, (255, 212, 59), -1)
    a = make(img, cfg)
    short = [s for s in a.strokes if np.hypot(*(s.points[-1] - s.points[0])) < 3 * LW]
    assert len(short) <= 10  # 34 when the plateau is split


def test_blank_image(cfg):
    a = make(np.full((100, 120, 3), 255, np.uint8), cfg)
    assert_valid_artwork(a, cfg, allow_empty=True)
    assert a.strokes == []
    assert len(a.palette) == cfg.tools.num_tools
    assert all(type(v) is int for c in a.palette for v in c)


@pytest.mark.slow
def test_runtime_large(cfg):
    big = radial_rgb(1500, 2000, noise=30)
    t = time.perf_counter(); make(big, cfg)
    assert time.perf_counter() - t < 30
