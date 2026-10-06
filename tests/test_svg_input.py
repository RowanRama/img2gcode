from pathlib import Path

import numpy as np
import pytest
from shapely.geometry import Point

from img2gcode.svg_input import load_svg_artwork

S = 7.0   # 100-unit viewBox on a 1000x700 print area -> 7 mm per unit
IMAGES = Path(__file__).resolve().parent.parent / "images"


def svg(tmp_path, body, vb="0 0 100 100", extra=""):
    p = tmp_path / "a.svg"
    p.write_text(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="{vb}" {extra}>{body}</svg>', encoding="utf-8")
    return p


def area(a, tool):
    return sum(r.shape.area for r in a.regions if r.tool == tool)


def test_fit_to_print_area(cfg, tmp_path):
    a = load_svg_artwork(svg(tmp_path, '<rect width="10" height="10"/>'), cfg)
    assert (a.width_mm, a.height_mm) == pytest.approx((700, 700))
    assert a.source_px == (100, 100) and a.underlay is None


def test_physical_size_with_tiny_viewbox(cfg, tmp_path):
    a = load_svg_artwork(svg(tmp_path, '<rect width="24" height="12"/>', "0 0 24 12", 'width="10mm" height="5mm"'), cfg)
    assert (a.width_mm, a.height_mm) == pytest.approx((1000, 500))
    assert area(a, 0) == pytest.approx(1000 * 500, rel=0.01)


def test_two_colours_sorted_dark_first(cfg, tmp_path):
    body = '<rect width="40" height="40" fill="#ffcc00"/><rect x="50" y="50" width="40" height="40" fill="#202080"/>'
    a = load_svg_artwork(svg(tmp_path, body), cfg)
    assert a.palette == [(32, 32, 128), (255, 204, 0)]
    assert area(a, 0) == pytest.approx((40 * S) ** 2, rel=0.01)


def test_nonzero_reversed_subpath_is_a_hole(cfg, tmp_path):
    d = "M10,10 H90 V90 H10 Z M30,30 V70 H70 V30 Z"
    a = load_svg_artwork(svg(tmp_path, f'<path d="{d}"/>'), cfg)
    assert area(a, 0) == pytest.approx((80 ** 2 - 40 ** 2) * S * S, rel=0.01)


def test_nonzero_same_direction_is_filled(cfg, tmp_path):
    d = "M10,10 H90 V90 H10 Z M30,30 H70 V70 H30 Z"
    a = load_svg_artwork(svg(tmp_path, f'<path d="{d}"/>'), cfg)
    assert area(a, 0) == pytest.approx(80 ** 2 * S * S, rel=0.01)


def test_evenodd_makes_a_hole(cfg, tmp_path):
    d = "M10,10 H90 V90 H10 Z M30,30 H70 V70 H30 Z"
    a = load_svg_artwork(svg(tmp_path, f'<path d="{d}" fill-rule="evenodd"/>'), cfg)
    assert area(a, 0) == pytest.approx((80 ** 2 - 40 ** 2) * S * S, rel=0.01)


def test_later_shape_covers_earlier(cfg, tmp_path):
    body = '<rect width="60" height="100" fill="#ff0000"/><rect x="40" width="60" height="100" fill="#0000ff"/>'
    a = load_svg_artwork(svg(tmp_path, body), cfg)
    red = a.palette.index((255, 0, 0)); blue = a.palette.index((0, 0, 255))
    assert area(a, red) == pytest.approx(40 * 100 * S * S, rel=0.01)
    assert area(a, blue) == pytest.approx(60 * 100 * S * S, rel=0.01)


def test_white_is_paper(cfg, tmp_path):
    body = '<rect width="100" height="100" fill="#000"/><rect x="25" y="25" width="50" height="50" fill="#fff"/>'
    a = load_svg_artwork(svg(tmp_path, body), cfg)
    assert len(a.palette) == 1
    assert area(a, 0) == pytest.approx((100 ** 2 - 50 ** 2) * S * S, rel=0.01)


def test_thick_stroke_is_region_thin_stroke_is_line(cfg, tmp_path):
    body = ('<path d="M10,20 H90" stroke="#000" stroke-width="2" fill="none"/>'
            '<path d="M10,60 H90" stroke="#000" stroke-width="0.2" fill="none"/>')
    a = load_svg_artwork(svg(tmp_path, body), cfg)
    assert area(a, 0) == pytest.approx(80 * 2 * S * S, rel=0.05)
    assert len(a.strokes) == 1 and a.strokes[0].kind == "line"
    assert np.allclose(a.strokes[0].points[:, 1], 60 * S)


def test_thick_stroke_follows_svg_caps_and_joins(cfg, tmp_path):
    # SVG defaults are butt caps and miter joins: the 2-unit gap between dashes stays open
    # (flag trigrams rely on it) and the corner stays square.
    d = "M10,20 H40 M42,20 H90 M10,50 H90 V90"
    gap, corner = Point(41 * S, 20 * S), Point(91.9 * S, 48.1 * S)
    a = load_svg_artwork(svg(tmp_path, f'<path d="{d}" stroke="#000" stroke-width="4" fill="none"/>'), cfg)
    assert not a.regions[0].shape.covers(gap) and a.regions[0].shape.covers(corner)
    body = f'<path d="{d}" stroke="#000" stroke-width="4" fill="none" stroke-linecap="round" stroke-linejoin="round"/>'
    r = load_svg_artwork(svg(tmp_path, body), cfg)
    assert r.regions[0].shape.covers(gap) and not r.regions[0].shape.covers(corner)


def test_gradient_uses_mean_stop_colour(cfg, tmp_path):
    body = ('<defs><linearGradient id="g"><stop offset="0" stop-color="#000000"/>'
            '<stop offset="1" stop-color="#ffffff"/></linearGradient></defs>'
            '<rect width="50" height="50" fill="url(#g)"/>')
    a = load_svg_artwork(svg(tmp_path, body), cfg)
    assert len(a.palette) == 1 and all(abs(c - 128) <= 2 for c in a.palette[0])


def test_near_identical_colours_merge(cfg, tmp_path):
    body = '<rect width="40" height="40" fill="#5a9821"/><rect x="50" width="40" height="40" fill="#5b9922"/>'
    assert len(load_svg_artwork(svg(tmp_path, body), cfg).palette) == 1


def test_n_tools_merges_colours(cfg, tmp_path):
    body = ''.join(f'<rect x="{i * 25}" width="20" height="20" fill="{c}"/>'
                   for i, c in enumerate(["#000000", "#303030", "#ff0000", "#ffff00"]))
    a = load_svg_artwork(svg(tmp_path, body), cfg, n_tools=2)
    assert len(a.palette) == 2 and {r.tool for r in a.regions} <= {0, 1}


def test_group_transform_applied(cfg, tmp_path):
    a = load_svg_artwork(svg(tmp_path, '<g transform="translate(50,0)"><rect width="10" height="10"/></g>'), cfg)
    assert a.regions[0].shape.bounds[0] == pytest.approx(350, abs=0.5)


def test_rotated_primitives_keep_their_transform(cfg, tmp_path):
    body = ('<rect x="40" y="40" width="20" height="10" transform="rotate(45 50 45)"/>'
            '<ellipse cx="50" cy="80" rx="20" ry="5" transform="rotate(90 50 80)" fill="#ff0000"/>')
    a = load_svg_artwork(svg(tmp_path, body), cfg)
    rect = [r.shape for r in a.regions if a.palette[r.tool] == (0, 0, 0)][0]
    ell = [r.shape for r in a.regions if a.palette[r.tool] == (255, 0, 0)][0]
    assert rect.bounds == pytest.approx(tuple(v * S for v in (39.393, 34.393, 60.607, 55.607)), abs=0.1)
    assert rect.area == pytest.approx(200 * S * S, rel=0.01)
    assert ell.bounds == pytest.approx(tuple(v * S for v in (45, 60, 55, 100)), abs=0.5)


def test_mirrored_primitive_keeps_its_transform(cfg, tmp_path):
    body = '<g transform="translate(100,0) scale(-1,1)"><rect x="10" y="10" width="20" height="10"/></g>'
    a = load_svg_artwork(svg(tmp_path, body), cfg)
    assert a.regions[0].shape.bounds == pytest.approx(tuple(v * S for v in (70, 10, 90, 20)), abs=0.1)
    assert a.regions[0].shape.area == pytest.approx(200 * S * S, rel=0.01)


def test_rotated_stroke_with_document_scale(cfg, tmp_path):
    # width 200 on a 100-unit viewBox: 2 document units per user unit, still 7 mm per user unit on the page.
    body = ('<rect x="40" y="40" width="20" height="10" transform="rotate(45 50 45)" '
            'fill="none" stroke="#000" stroke-width="4"/>')
    a = load_svg_artwork(svg(tmp_path, body, extra='width="200" height="200"'), cfg)
    half = (24 + 14) / 2 * np.sqrt(0.5)                  # outer edge 24 x 14, rotated 45 degrees
    assert a.regions[0].shape.bounds == pytest.approx(
        tuple(v * S for v in (50 - half, 45 - half, 50 + half, 45 + half)), abs=0.1)
    assert a.regions[0].shape.area == pytest.approx((24 * 14 - 16 * 6) * S * S, rel=0.01)


@pytest.mark.parametrize("attrs", [
    'fill="#ff0000" display="none"', 'fill="#ff0000" visibility="hidden"', 'fill="#ff0000" opacity="0"',
    'fill="#ff0000" fill-opacity="0"', 'fill="url(#g)" fill-opacity="0"',
    'fill="none" stroke="url(#g)" stroke-width="4" stroke-opacity="0"',
])
def test_invisible_paint_is_skipped(cfg, tmp_path, attrs):
    body = ('<defs><linearGradient id="g"><stop offset="0" stop-color="#ff0000"/></linearGradient></defs>'
            f'<rect width="40" height="40"/><rect x="50" width="40" height="40" {attrs}/>')
    a = load_svg_artwork(svg(tmp_path, body), cfg)
    assert a.palette == [(0, 0, 0)] and area(a, 0) == pytest.approx((40 * S) ** 2, rel=0.01)


def test_no_shapes_raises(cfg, tmp_path):
    with pytest.raises(ValueError):
        load_svg_artwork(svg(tmp_path, ""), cfg)


@pytest.mark.skipif(not list(IMAGES.glob("*.svg")), reason="run scripts/fetch_test_images.py")
def test_downloaded_svgs_load(cfg):
    for f in sorted(IMAGES.glob("*.svg")):
        a = load_svg_artwork(f, cfg)
        assert a.regions or a.strokes, f.name
