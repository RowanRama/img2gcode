import xml.etree.ElementTree as ET

import numpy as np

from img2gcode.model import Stroke, ToolPass, Toolpaths
from img2gcode.svg_io import (
    INKSCAPE_NS, SODIPODI_NS, SVG_NS, XLINK_NS, is_toolpaths_svg, read_toolpaths_svg, write_toolpaths_svg,
)

SQ = np.array([[10, 10], [60, 10], [60, 40], [10, 40], [10, 10]], float)
LINE = np.array([[5.5, 70.25], [90.125, 72.0], [120, 90]], float)


def make_tp():
    passes = [
        ToolPass(0, 0, [Stroke(SQ, 0, "perimeter"), Stroke(LINE, 0, "infill")]),
        ToolPass(1, 0, [Stroke(LINE + 5, 1, "line")]),
        ToolPass(0, 1, [Stroke(SQ, 0, "perimeter")]),
        ToolPass(1, 1, []),
    ]
    under = np.zeros((40, 60, 3), np.uint8); under[:, :30] = 255
    return Toolpaths(200.0, 100.0, 4.0, 2, passes, [(20, 20, 20), (200, 40, 40)], (400, 200), under, "edge")


def strokes_of(tp, layer, tool):
    return next(p for p in tp.passes if p.layer == layer and p.tool == tool).strokes


def test_round_trip(tmp_path):
    tp = make_tp(); f = tmp_path / "t.svg"
    write_toolpaths_svg(tp, f)
    got, ignored = read_toolpaths_svg(f)
    assert ignored == 0
    assert (got.width_mm, got.height_mm, got.line_width_mm, got.n_layers) == (200.0, 100.0, 4.0, 2)
    assert got.palette == tp.palette and got.source_px == (400, 200) and got.style == "edge"
    assert [(p.layer, p.tool) for p in got.passes] == [(0, 0), (0, 1), (1, 0), (1, 1)]
    for a, b in zip(tp.passes, got.passes):
        assert len(a.strokes) == len(b.strokes)
        for sa, sb in zip(a.strokes, b.strokes):
            assert sa.kind == sb.kind and sa.tool == sb.tool
            assert np.allclose(sa.points, sb.points, atol=1e-3)
    assert got.underlay is not None and got.underlay.ndim == 3
    half = got.underlay.shape[1] // 2
    assert got.underlay[:, :half].mean() > got.underlay[:, half:].mean() + 100


def test_is_toolpaths_svg(tmp_path):
    f = tmp_path / "t.svg"; write_toolpaths_svg(make_tp(), f)
    assert is_toolpaths_svg(f)
    plain = tmp_path / "p.svg"
    plain.write_text(f'<svg xmlns="{SVG_NS}" viewBox="0 0 10 10"><rect width="5" height="5"/></svg>', encoding="utf-8")
    assert not is_toolpaths_svg(plain)


def _tool_groups(root):
    groups = {}
    for lg in root.iter(f"{{{SVG_NS}}}g"):
        if lg.get("data-layer") is not None:
            for tg in lg:
                if tg.get("data-tool") is not None:
                    groups[(int(lg.get("data-layer")), int(tg.get("data-tool")))] = tg
    return groups


def test_inkscape_style_edits(tmp_path):
    f = tmp_path / "t.svg"; write_toolpaths_svg(make_tp(), f)
    for prefix, uri in (("", SVG_NS), ("inkscape", INKSCAPE_NS), ("sodipodi", SODIPODI_NS), ("xlink", XLINK_NS)):
        ET.register_namespace(prefix, uri)
    tree = ET.parse(f); root = tree.getroot()
    groups = _tool_groups(root)
    g00 = groups[(0, 0)]
    g00.set("transform", "translate(10,5)")
    first = g00[0]
    d = "M " + " L ".join(first.get("points").split())
    path = ET.Element(f"{{{SVG_NS}}}path", {"d": d, "data-kind": "perimeter"})
    g00.remove(first); g00.insert(0, path)
    groups[(0, 1)].set("style", "display:none")
    root.append(ET.Element(f"{{{SVG_NS}}}path", {"d": "M 0,0 L 5,5"}))
    tree.write(f, encoding="utf-8", xml_declaration=True)

    got, ignored = read_toolpaths_svg(f)
    assert ignored == 1
    s00 = strokes_of(got, 0, 0)
    assert s00[0].kind == "perimeter" and np.allclose(s00[0].points, SQ + [10, 5], atol=1e-3)
    assert np.allclose(s00[1].points, LINE + [10, 5], atol=1e-3)
    assert strokes_of(got, 0, 1) == []
    assert len(strokes_of(got, 1, 0)) == 1


def test_nested_transforms(tmp_path):
    f = tmp_path / "n.svg"
    f.write_text(
        f'<svg xmlns="{SVG_NS}" width="100mm" height="100mm" viewBox="0 0 100 100" data-img2gcode="toolpaths" '
        'data-line-width="4" data-layers="1" data-tools="1" data-source-px="100x100">'
        '<g data-layer="0" transform="translate(10,0)"><g data-tool="0" stroke="#000000" transform="rotate(90)">'
        '<polyline points="1,0 2,0"/></g></g></svg>', encoding="utf-8")
    got, _ = read_toolpaths_svg(f)
    assert np.allclose(strokes_of(got, 0, 0)[0].points, [[10, 1], [10, 2]], atol=1e-6)


def test_curved_path_is_sampled(tmp_path):
    f = tmp_path / "c.svg"
    f.write_text(
        f'<svg xmlns="{SVG_NS}" viewBox="0 0 100 100" data-img2gcode="toolpaths" data-line-width="4" '
        'data-layers="1" data-tools="1" data-source-px="100x100"><g data-layer="0"><g data-tool="0" stroke="#000">'
        '<path d="M 0,0 C 0,50 100,50 100,0"/></g></g></svg>', encoding="utf-8")
    got, _ = read_toolpaths_svg(f)
    p = strokes_of(got, 0, 0)[0].points
    assert len(p) >= 20 and np.allclose(p[0], [0, 0]) and np.allclose(p[-1], [100, 0])
    assert abs(p[:, 1].max() - 37.5) < 0.5


def test_other_shapes_and_hiding(tmp_path):
    f = tmp_path / "s.svg"
    f.write_text(
        f'<svg xmlns="{SVG_NS}" viewBox="0 0 100 100" data-img2gcode="toolpaths" data-line-width="4" '
        'data-layers="1" data-tools="1" data-source-px="100x100"><defs><path d="M 0,0 L 1,1"/></defs>'
        '<g data-layer="0"><g data-tool="0" style="stroke:#ff0000">'
        '<polygon points="10,10 20,10 20,20"/><line x1="5" y1="5" x2="9" y2="5"/>'
        '<rect x="30" y="30" width="10" height="5"/><circle cx="50" cy="50" r="10"/>'
        '<path d="M 1,1 L 2,2" display="none"/><path d="M 1,1 L 2,2" style="visibility:hidden"/>'
        '</g></g></svg>', encoding="utf-8")
    got, ignored = read_toolpaths_svg(f)
    assert ignored == 0 and got.palette == [(255, 0, 0)]
    poly, line, rect, circle = strokes_of(got, 0, 0)
    assert np.allclose(poly.points, [[10, 10], [20, 10], [20, 20], [10, 10]])
    assert np.allclose(line.points, [[5, 5], [9, 5]])
    assert rect.closed and len(rect.points) == 5
    assert circle.closed and np.allclose(np.hypot(*(circle.points - 50).T), 10, atol=1e-6)
    assert np.linalg.norm(np.diff(circle.points, axis=0), axis=1).max() <= 0.5 + 1e-9
