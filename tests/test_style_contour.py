import time

import cv2
import numpy as np
import pytest
from scipy.ndimage import distance_transform_edt
from shapely.geometry import LineString

from img2gcode.geometry import polyline_length
from img2gcode.styles.contour import make
from tests.synth import LW, assert_valid_artwork, dense, radial_rgb


def _disc_rgb(grey_patch=False):
    """Black disc on white; optionally a light-grey patch, which adds light levels."""
    img = np.full((500, 700, 3), 255, np.uint8)
    cv2.circle(img, (250, 250), 150, (0, 0, 0), -1)
    if grey_patch:
        cv2.rectangle(img, (480, 150), (650, 350), (200, 200, 200), -1)
    return img


def _plateaus_rgb():
    """Flat tones: a dark disc on a large mid-grey plateau, a light-grey strip, white paper."""
    img = np.full((500, 700, 3), 255, np.uint8)
    cv2.rectangle(img, (100, 100), (600, 400), (120, 120, 120), -1)
    cv2.circle(img, (350, 250), 50, (30, 30, 30), -1)
    cv2.rectangle(img, (630, 100), (670, 400), (200, 200, 200), -1)
    return img


def _dist_mm(mask, art):
    """Per stroke, the largest distance (mm) from its points to the nearest True pixel of mask."""
    s = art.width_mm / mask.shape[1]
    d = distance_transform_edt(~mask) * s
    out = []
    for st in art.strokes:
        ij = np.clip((dense(st.points, 1.0) / s).astype(int), 0, [mask.shape[1] - 1, mask.shape[0] - 1])
        out.append(d[ij[:, 1], ij[:, 0]].max())
    return np.array(out)


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


def test_lines_stay_on_the_foreground(cfg):
    # The grey patch puts light levels whose iso-lines run around the disc out on the paper.
    img = _disc_rgb(grey_patch=True)
    a = make(img, cfg)
    assert a.strokes
    fg = np.any(img < cfg.image.white_threshold, axis=2)
    assert _dist_mm(fg, a).max() <= LW


def test_flat_areas_do_not_vote(cfg):
    # Beyond the blur kernel's 4-sigma reach from every edge the image is exactly flat, so a
    # level on a plateau value would trace rounding noise there. Real lines stay well inside.
    img = _plateaus_rgb()
    a = make(img, cfg)
    assert a.strokes
    g = img[:, :, 0]
    edge = np.zeros(g.shape, bool)
    edge[:, 1:] |= g[:, 1:] != g[:, :-1]
    edge[1:, :] |= g[1:, :] != g[:-1, :]
    sigma = cfg.contour.smoothing * LW
    assert _dist_mm(edge, a).max() <= 3.75 * sigma


def test_thick_edge_has_no_dashes(cfg):
    # Many levels stack closer than min_sep along a sharp edge; whatever survives stays whole.
    cfg.contour.levels = 16
    a = make(_disc_rgb(), cfg)
    assert a.strokes
    assert all(np.allclose(s.points[0], s.points[-1]) for s in a.strokes)


def test_tools_spread_over_kept_levels(cfg):
    # A faint gradient keeps only a few of the 8 levels; they still span dark -> light tools.
    yy, xx = np.mgrid[0:500, 0:700]
    v = (150 + 10 * np.hypot(xx - 350, yy - 250) / np.hypot(350, 250)).astype(np.uint8)
    cfg.tools.num_tools = 2
    a = make(np.dstack([v, v, v]), cfg)
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
