import cv2
import numpy as np
import pytest
from shapely.geometry import LineString, Point, box

from img2gcode.geometry import polyline_length
from img2gcode.model import Artwork, Region, Stroke
from img2gcode.planner import plan
from img2gcode.raster import mask_to_geometry

LW = 4.0


def art(regions=(), strokes=(), palette=((0, 0, 0),), w=300.0, h=200.0):
    return Artwork(w, h, list(regions), list(strokes), [tuple(c) for c in palette], (600, 400), None, "test")


def of_kind(tp, kind, tool=None, layer=0):
    return [s for p in tp.passes if p.layer == layer and (tool is None or p.tool == tool)
            for s in p.strokes if s.kind == kind]


def test_first_perimeter_sits_half_a_line_inside(cfg):
    sq = box(20, 20, 120, 120)
    per = of_kind(plan(art([Region(sq, 0)]), cfg), "perimeter")
    assert len(per) == 1 and per[0].closed
    d = [sq.exterior.distance(Point(p)) for p in per[0].points]
    assert min(d) == pytest.approx(LW / 2, abs=0.05) and max(d) == pytest.approx(LW / 2, abs=0.05)


def test_two_loops_and_a_hole(cfg):
    cfg.layers.perimeter_loops = 2
    donut = box(0, 0, 100, 100).difference(box(40, 40, 60, 60))
    per = of_kind(plan(art([Region(donut, 0)], w=100, h=100), cfg), "perimeter")
    assert len(per) == 4 and all(s.closed for s in per)


def test_neighbouring_tools_do_not_overlap(cfg):
    a, b = box(0, 0, 50, 100), box(50, 0, 100, 100)
    tp = plan(art([Region(a, 0), Region(b, 1)], palette=[(0, 0, 0), (255, 0, 0)], w=100, h=100), cfg)
    pa = [LineString(s.points) for s in of_kind(tp, "perimeter", tool=0)]
    pb = [LineString(s.points) for s in of_kind(tp, "perimeter", tool=1)]
    assert min(x.distance(y) for x in pa for y in pb) >= LW - 1e-6


def test_infill_stays_inside(cfg):
    sq = box(0, 0, 100, 80)
    inner = sq.buffer(-(LW + LW / 2) + 0.15 * LW)
    inf = of_kind(plan(art([Region(sq, 0)], w=100, h=80), cfg), "infill")
    assert inf
    zone = inner.buffer(0.05 * LW + 1e-6)
    assert all(zone.covers(LineString(s.points)) for s in inf)


def test_infill_spacing_follows_density(cfg):
    cfg.infill.density = 50; cfg.infill.connect_lines = False; cfg.infill.angle = 0
    inf = of_kind(plan(art([Region(box(0, 0, 100, 100), 0)], w=100, h=100), cfg), "infill")
    ys = np.unique(np.round([s.points[0, 1] for s in inf], 6))
    assert np.allclose(np.diff(ys), 8.0)


def test_exclusion_zone_blocks_infill_only(cfg):
    zone = (30, 30, 70, 70)
    tp = plan(art([Region(box(0, 0, 100, 100), 0)], w=100, h=100), cfg, exclusion_zones=[zone])
    hole = box(*zone).buffer(-0.5)
    assert not any(hole.intersects(LineString(s.points)) for s in of_kind(tp, "infill"))
    assert len(of_kind(tp, "perimeter")) == 1


def test_thin_feature_becomes_centreline(cfg):
    tp = plan(art([Region(box(20, 50, 120, 52.5), 0)], w=150, h=100), cfg)
    assert of_kind(tp, "perimeter") == []
    lines = of_kind(tp, "line")
    assert len(lines) == 1 and polyline_length(lines[0].points) > 85
    assert np.all(np.abs(lines[0].points[:, 1] - 51.25) < 1.0)


def test_wide_shape_has_no_stray_centrelines(cfg):
    tp = plan(art([Region(box(0, 0, 100, 100), 0)], w=100, h=100), cfg)
    assert of_kind(tp, "line") == []


def test_jagged_outline_has_no_stray_centrelines(cfg):
    m = np.zeros((200, 300), np.uint8)
    cv2.circle(m, (150, 100), 80, 1, -1)       # marching-squares outline: a zigzag at every pixel step
    tp = plan(art([Region(mask_to_geometry(m, 1.0), 0)]), cfg)
    assert of_kind(tp, "line") == []


def test_layers_alternate_infill_angle(cfg):
    cfg.layers.num_layers = 2; cfg.infill.connect_lines = False; cfg.infill.angle = 0
    tp = plan(art([Region(box(0, 0, 100, 100), 0)], w=100, h=100), cfg)
    l0, l1 = of_kind(tp, "infill", layer=0), of_kind(tp, "infill", layer=1)
    assert l0 and l1
    assert all(abs(s.points[0, 1] - s.points[-1, 1]) < 1e-6 for s in l0)
    assert all(abs(s.points[0, 0] - s.points[-1, 0]) < 1e-6 for s in l1)


def test_line_strokes_kept_and_every_pass_present(cfg):
    cfg.layers.num_layers = 2
    st = [Stroke(np.array([[10, 10], [50, 10]], float), 0), Stroke(np.array([[60, 10], [90, 40]], float), 1)]
    tp = plan(art([], st, palette=[(0, 0, 0), (9, 9, 9)]), cfg)
    assert [(p.layer, p.tool) for p in tp.passes] == [(0, 0), (0, 1), (1, 0), (1, 1)]
    assert sum(len(p.strokes) for p in tp.passes) == 4


def test_metadata_copied(cfg):
    a = art([Region(box(0, 0, 50, 50), 0)])
    a.underlay = np.zeros((4, 6, 3), np.uint8)
    tp = plan(a, cfg)
    assert (tp.width_mm, tp.height_mm, tp.source_px, tp.style) == (300.0, 200.0, (600, 400), "test")
    assert tp.line_width_mm == LW and tp.n_layers == cfg.layers.num_layers
    assert tp.palette == [(0, 0, 0)] and tp.underlay is a.underlay
