import time

import numpy as np
from shapely.geometry import LineString

from img2gcode.styles.oneline import _tour, make
from tests.synth import assert_valid_artwork, radial_rgb, split_rgb


def test_contract(cfg):
    a = make(radial_rgb(), cfg)
    assert_valid_artwork(a, cfg)
    assert a.style == "oneline" and a.regions == []


def test_one_tool_one_stroke(cfg):
    cfg.tools.num_tools = 1
    a = make(radial_rgb(), cfg)
    assert len(a.strokes) == 1 and len(a.strokes[0].points) >= 200


def test_line_goes_where_it_is_dark(cfg):
    cfg.tools.num_tools = 1
    a = make(split_rgb(), cfg)
    p = a.strokes[0].points
    seg = np.hypot(*np.diff(p, axis=0).T)
    mid = (p[:-1, 0] + p[1:, 0]) / 2
    assert seg[mid < a.width_mm / 2].sum() / seg.sum() >= 0.95


def test_one_stroke_per_tool(cfg):
    cfg.tools.num_tools = 2
    a = make(split_rgb(left=(20, 20, 120), right=(220, 40, 40)), cfg)
    assert len(a.strokes) == 2 and {s.tool for s in a.strokes} == {0, 1}


def test_points_setting(cfg):
    cfg.tools.num_tools = 1; cfg.oneline.points = 300
    t = time.perf_counter(); a = make(radial_rgb(), cfg)
    assert time.perf_counter() - t < 10 and len(a.strokes[0].points) >= 300


def test_tour_has_no_crossings():
    pts = np.random.default_rng(3).random((150, 2)) * 100
    order = _tour(pts, 5.0)
    assert sorted(order.tolist()) == list(range(150))
    assert LineString(pts[order]).is_simple


def test_time_cap(cfg):
    cfg.tools.num_tools = 1; cfg.oneline.points = 5000; cfg.oneline.tsp_seconds = 1
    t = time.perf_counter(); make(radial_rgb(), cfg)
    assert time.perf_counter() - t < 15


def test_blank_image(cfg):
    a = make(np.full((100, 120, 3), 255, np.uint8), cfg)
    assert_valid_artwork(a, cfg, allow_empty=True)
    assert a.strokes == []
