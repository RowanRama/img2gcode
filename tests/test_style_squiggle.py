import cv2
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
    # 1.49 with D = sqrt(1 - L); the brief's 1.5 assumed linear D, which draws 150 grey as near-straight rows.
    assert left > 1.4 * right


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


def _ink(a):
    return sum(np.hypot(*np.diff(s.points, axis=0).T).sum() for s in a.strokes)


def test_tone_is_absolute(cfg):
    dark, light = (make(np.full((300, 400, 3), v, np.uint8), cfg) for v in (30, 200))
    assert _ink(dark) > 1.3 * _ink(light)


def test_three_tools_keep_sharp_edges(cfg):
    # Black | 200 grey | 110 grey at 1 mm per px. The black/200 edge has a 5 px blend (100 grey,
    # as an upscaled antialiased edge) that the median filter cannot remove; it must not turn
    # into 110-grey ink.
    cfg.tools.num_tools = 3
    img = np.empty((400, 1000, 3), np.uint8)
    img[:, :333] = 0; img[:, 333:338] = 100; img[:, 338:667] = 200; img[:, 667:] = 110
    a = make(img, cfg)
    assert np.allclose(a.palette, [(0,) * 3, (110,) * 3, (200,) * 3], atol=5)   # tool 1 is the 110 grey
    w = a.width_mm
    spans = {0: (0, w * 335 / 1000), 2: (w * 335 / 1000, w * 667 / 1000), 1: (w * 667 / 1000, w)}
    assert {s.tool for s in a.strokes} == {0, 1, 2}
    for s in a.strokes:
        lo, hi = spans[s.tool]
        assert s.points[:, 0].min() > lo - LW and s.points[:, 0].max() < hi + LW


def test_antialiased_rim_does_not_become_strokes(cfg):
    # With two tools a black disc's second palette colour is its grey antialiased rim.
    cfg.tools.num_tools = 2
    img = np.full((600, 800, 3), 255, np.uint8)
    cv2.circle(img, (400, 300), 230, (0, 0, 0), -1, cv2.LINE_AA)
    a = make(img, cfg)
    assert len(a.palette) == 2 and {s.tool for s in a.strokes} == {0}


def test_solid_page_is_one_serpentine_stroke(cfg):
    assert len(make(np.zeros((400, 900, 3), np.uint8), cfg).strokes) == 1


def test_separate_blocks_are_separate_strokes(cfg):
    img = np.zeros((400, 900, 3), np.uint8); img[:, 300:600] = 255
    assert len(make(img, cfg).strokes) == 2


def test_page_narrower_than_one_sample(cfg):
    a = make(np.zeros((1000, 1, 3), np.uint8), cfg)
    assert_valid_artwork(a, cfg, allow_empty=True)


def test_page_shorter_than_one_row(cfg):
    img = np.zeros((10, 1000, 3), np.uint8); img[:, 500:] = (200, 30, 30)
    a = make(img, cfg)
    assert a.strokes == [] and len(a.palette) == 2


def test_blank_image(cfg):
    a = make(np.full((100, 120, 3), 255, np.uint8), cfg)
    assert_valid_artwork(a, cfg, allow_empty=True)
    assert a.strokes == []
