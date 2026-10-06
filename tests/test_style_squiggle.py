import numpy as np

from img2gcode.styles.squiggle import make
from tests.synth import LW, assert_valid_artwork, radial_rgb, split_rgb


def test_contract(cfg):
    a = make(radial_rgb(), cfg)
    assert_valid_artwork(a, cfg)
    assert a.style == "squiggle" and a.regions == []


def test_rows_never_overlap(cfg):
    a = make(radial_rgb(), cfg)
    sp = cfg.squiggle.row_spacing * LW
    for s in a.strokes:
        y = s.points[:, 1]
        off = np.abs(y - (np.floor(y / sp) + 0.5) * sp)
        assert off.max() <= (sp - LW) / 2 + 1e-6


def test_dark_side_has_more_ink(cfg):
    a = make(split_rgb(left=(20, 20, 20), right=(150, 150, 150)), cfg)
    left = right = 0.0
    for s in a.strokes:
        p = s.points
        seg = np.hypot(*np.diff(p, axis=0).T)
        mid = (p[:-1, 0] + p[1:, 0]) / 2
        left += seg[mid < a.width_mm / 2].sum(); right += seg[mid >= a.width_mm / 2].sum()
    assert left > 1.5 * right


def test_white_background_breaks_rows(cfg):
    img = np.zeros((400, 900, 3), np.uint8); img[:, 300:600] = 255
    a = make(img, cfg)
    x0, x1 = a.width_mm / 3 + 2 * LW, 2 * a.width_mm / 3 - 2 * LW
    for s in a.strokes:
        assert not np.any((s.points[:, 0] > x0) & (s.points[:, 0] < x1))


def test_multi_tool(cfg):
    cfg.tools.num_tools = 2
    a = make(split_rgb(left=(20, 20, 120), right=(220, 40, 40)), cfg)
    assert {s.tool for s in a.strokes} == {0, 1}


def test_page_shorter_than_one_row(cfg):
    img = np.zeros((10, 1000, 3), np.uint8); img[:, 500:] = (200, 30, 30)
    a = make(img, cfg)
    assert a.strokes == [] and len(a.palette) == 2


def test_blank_image(cfg):
    a = make(np.full((100, 120, 3), 255, np.uint8), cfg)
    assert_valid_artwork(a, cfg, allow_empty=True)
    assert a.strokes == []
