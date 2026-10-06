import numpy as np
import pytest
from shapely.geometry import LineString, box

from img2gcode.geometry import (
    chain_polylines, chaikin, deoverlap, geometry_rings, hatch_lines, order_strokes,
    polyline_length, resample, rings_to_geometry, simplify, smooth,
)
from img2gcode.model import Stroke

SQ = np.array([[0, 0], [10, 0], [10, 10], [0, 10], [0, 0]], float)


def test_polyline_length():
    assert polyline_length(np.array([[0, 0], [3, 4], [3, 10]], float)) == pytest.approx(11.0)
    assert polyline_length(np.zeros((1, 2))) == 0.0


def test_resample_spacing_and_ends():
    p = np.array([[0, 0], [10, 0]], float)
    q = resample(p, 1.0)
    assert len(q) == 11
    assert np.allclose(q[0], p[0]) and np.allclose(q[-1], p[-1])
    r = resample(SQ, 2.5)
    assert np.allclose(r[0], r[-1])


def test_smooth_keeps_open_endpoints_and_closure():
    zig = np.array([[0, 0], [1, 1], [2, 0], [3, 1], [4, 0], [5, 1], [6, 0]], float)
    s = smooth(zig, 2.0)
    assert np.allclose(s[0], zig[0]) and np.allclose(s[-1], zig[-1])
    assert np.ptp(s[1:-1, 1]) < np.ptp(zig[:, 1])
    c = smooth(SQ, 3.0)
    assert np.allclose(c[0], c[-1])


def test_simplify_straight_line():
    p = np.c_[np.linspace(0, 10, 50), np.zeros(50)]
    assert len(simplify(p, 0.01)) == 2


def test_chaikin_open_keeps_ends():
    p = np.array([[0, 0], [10, 0], [10, 10]], float)
    q = chaikin(p, 2)
    assert np.allclose(q[0], p[0]) and np.allclose(q[-1], p[-1]) and len(q) > len(p)


def test_rings_nonzero_and_evenodd():
    outer = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], float)
    inner = np.array([[3, 3], [7, 3], [7, 7], [3, 7]], float)          # same direction
    assert rings_to_geometry([outer, inner], "nonzero").area == pytest.approx(100)
    assert rings_to_geometry([outer, inner], "evenodd").area == pytest.approx(84)
    assert rings_to_geometry([outer, inner[::-1]], "nonzero").area == pytest.approx(84)


def test_rings_overlapping_evenodd_is_xor():
    a = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], float)
    b = a + 5
    assert rings_to_geometry([a, b], "evenodd").area == pytest.approx(150)
    assert rings_to_geometry([a, b], "nonzero").area == pytest.approx(175)


def test_geometry_rings_counts_holes():
    g = box(0, 0, 10, 10).difference(box(3, 3, 7, 7))
    rings = geometry_rings(g)
    assert len(rings) == 2 and all(np.allclose(r[0], r[-1]) for r in rings)


def test_hatch_lines_square():
    lines = hatch_lines(box(0, 0, 40, 40), 4.0, 0.0)
    assert len(lines) == 10
    ys = sorted(round(float(l[0, 1]), 6) for l in lines)
    assert ys == [2.0 + 4 * i for i in range(10)]
    sq = box(0, 0, 40, 40).buffer(1e-6)
    assert all(sq.covers(LineString(l)) for l in lines)
    joined = hatch_lines(box(0, 0, 40, 40), 4.0, 0.0, connect=True, connect_tol=0.1)
    assert len(joined) == 1


def test_hatch_lines_global_phase_and_angle():
    a = hatch_lines(box(0, 0, 40, 40), 4.0, 0.0)
    b = hatch_lines(box(40, 0, 80, 40), 4.0, 0.0)
    assert sorted(round(float(l[0, 1]), 6) for l in a) == sorted(round(float(l[0, 1]), 6) for l in b)
    diag = hatch_lines(box(0, 0, 40, 40), 4.0, 45.0)
    for l in diag:
        d = l[-1] - l[0]
        assert abs(abs(np.degrees(np.arctan2(d[1], d[0]))) % 180 - 45) < 1e-6 or abs(abs(np.degrees(np.arctan2(d[1], d[0]))) % 180 - 135) < 1e-6


def test_order_strokes_reverses_and_rotates():
    a = Stroke(np.array([[0, 0], [10, 0]], float))
    b = Stroke(np.array([[30, 0], [11, 0]], float))            # tail is nearer to a's end
    sq = Stroke(SQ + [50, 0])
    out = order_strokes([sq, b, a], start=(0, 0))
    assert np.allclose(out[0].points[0], [0, 0])
    assert np.allclose(out[1].points[0], [11, 0])               # b reversed
    assert out[2].closed and np.allclose(out[2].points[0], [50, 0])
    assert len(out) == 3


def test_order_strokes_keeps_tool_and_kind():
    s = Stroke(np.array([[5, 5], [6, 6]], float), tool=1, kind="infill")
    out = order_strokes([s])
    assert out[0].tool == 1 and out[0].kind == "infill"


def test_chain_polylines():
    a = np.array([[0, 0], [10, 0]], float)
    b = np.array([[10.5, 0], [20, 0]], float)
    c = np.array([[50, 50], [60, 50]], float)
    out = chain_polylines([a, b, c], max_gap=1.0)
    assert len(out) == 2
    assert max(polyline_length(p) for p in out) == pytest.approx(20.5, abs=0.6)


def test_deoverlap():
    a = np.array([[10, 50], [90, 50]], float)
    near = np.array([[10, 51], [90, 51]], float)      # 1 mm away → dropped
    far = np.array([[10, 70], [90, 70]], float)       # 20 mm away → kept
    cross = np.array([[50, 20], [50, 90]], float)     # crosses a and far → split
    out = deoverlap([a, near, far, cross], 100, 100, min_sep=4.0, min_length=2.0)
    lens = sorted(polyline_length(p) for p in out)
    assert len(out) == 5                              # a, far, and three pieces of cross
    assert lens[-1] == pytest.approx(80, abs=1) and lens[-2] == pytest.approx(80, abs=1)
    assert all(LineString(p).distance(LineString(a)) >= 3.5 or polyline_length(p) > 70 for p in out)


@pytest.mark.parametrize("offset", [0.0, 0.25, 0.5, 0.75])  # sub-cell positions (cell = min_sep / 4 = 1 mm)
def test_deoverlap_cuts_at_min_sep(offset):
    y = 50.0 + offset
    a = np.array([[10, y], [90, y]], float)
    for f, kept in ((0.8, False), (1.3, True)):
        b = a + [0, f * 4.0]
        out = deoverlap([a, b], 100, 100, min_sep=4.0, min_length=2.0)
        assert (sum(polyline_length(p) for p in out[1:]) > 79) == kept, f
