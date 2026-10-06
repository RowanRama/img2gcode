import time

import cv2
import numpy as np
import pytest

from img2gcode.styles.edge import make
from tests.synth import (
    LW, assert_valid_artwork, page_scale, recall_precision, shapes_outline_mm, shapes_rgb, total_length,
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
