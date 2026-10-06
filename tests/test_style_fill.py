import time

import numpy as np
import pytest
from shapely.ops import unary_union

from img2gcode.styles.fill import make
from tests.synth import CIRCLE, RECT, SHAPES_H, SHAPES_W, assert_valid_artwork, page_scale, shapes_rgb


def tool_geom(a, t):
    return unary_union([r.shape for r in a.regions if r.tool == t])


def test_contract(cfg):
    a = make(shapes_rgb(), cfg)
    assert_valid_artwork(a, cfg)
    assert a.style == "fill" and a.underlay is not None and a.source_px == (SHAPES_W, SHAPES_H)


def test_noisy_shapes_give_clean_regions(cfg):
    cfg.tools.num_tools = 2
    a = make(shapes_rgb(noise=45), cfg)
    s = page_scale(cfg, SHAPES_W, SHAPES_H)
    circle = np.pi * (CIRCLE[1] * s) ** 2
    rect = (RECT[2] - RECT[0] + 1) * (RECT[3] - RECT[1] + 1) * s * s
    g0, g1 = tool_geom(a, 0), tool_geom(a, 1)          # tool 0 is the darker colour (the circle)
    assert g0.area == pytest.approx(circle, rel=0.10)
    assert g1.area == pytest.approx(rect, rel=0.10)
    for g in (g0, g1):
        parts = list(getattr(g, "geoms", [g]))
        assert len(parts) <= 4
        holes = sum(sum(type(p)(h).area for h in p.interiors) for p in parts)
        assert holes < 0.01 * g.area


def test_background_is_not_printed(cfg):
    a = make(shapes_rgb(noise=30), cfg)
    assert unary_union([r.shape for r in a.regions]).area < 0.6 * a.width_mm * a.height_mm


def test_single_tool(cfg):
    cfg.tools.num_tools = 1
    a = make(shapes_rgb(), cfg)
    assert len(a.palette) == 1 and {r.tool for r in a.regions} == {0}


def test_more_tools_than_colours(cfg):
    cfg.tools.num_tools = 6
    a = make(shapes_rgb(), cfg)
    assert_valid_artwork(a, cfg)


def test_blank_image(cfg):
    a = make(np.full((100, 120, 3), 255, np.uint8), cfg)
    assert_valid_artwork(a, cfg, allow_empty=True)
    assert a.regions == [] and a.strokes == []


@pytest.mark.slow
def test_runtime_large(cfg):
    import cv2
    big = cv2.resize(shapes_rgb(noise=40), (2000, 1500), interpolation=cv2.INTER_NEAREST)
    t = time.perf_counter(); make(big, cfg)
    assert time.perf_counter() - t < 30
