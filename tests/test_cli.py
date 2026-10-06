import importlib.util
from pathlib import Path

import cv2
import numpy as np
import pytest

from img2gcode.__main__ import load_segments, main
from img2gcode.gcode_parser import parse_gcode
from img2gcode.pipeline import convert
from img2gcode.styles import STYLES
from img2gcode.svg_io import is_toolpaths_svg
from tests.synth import shapes_rgb

LOGO = Path(__file__).resolve().parent.parent / "images" / "python_logo.png"


def png(tmp_path, img, name="in.png"):
    p = tmp_path / name
    cv2.imwrite(str(p), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    return p


def body(path):
    return Path(path).read_text(encoding="utf-8").splitlines()[1:]


def test_png_to_svg_and_gcode(cfg, tmp_path):
    res = convert(LOGO, tmp_path / "out.gcode", cfg)
    assert res.gcode_path == tmp_path / "out.gcode" and res.gcode_path.exists()
    assert res.svg_path == tmp_path / "out.svg" and is_toolpaths_svg(res.svg_path)
    segs, meta = parse_gcode(res.gcode_path)
    assert any(s["type"] == "print" for s in segs) and meta["total_time_min"] > 0


def test_svg_output_only(cfg, tmp_path):
    res = convert(LOGO, tmp_path / "x.svg", cfg)
    assert res.gcode_path is None and (tmp_path / "x.svg").exists() and not (tmp_path / "x.gcode").exists()


def test_svg_only_flag(cfg, tmp_path):
    res = convert(LOGO, tmp_path / "x.gcode", cfg, svg_only=True)
    assert (tmp_path / "x.svg").exists() and not (tmp_path / "x.gcode").exists() and res.gcode_path is None


def test_toolpaths_svg_reconverts_identically(cfg, tmp_path):
    a = convert(LOGO, tmp_path / "a.gcode", cfg)
    b = convert(a.svg_path, tmp_path / "b.gcode", cfg)
    assert body(a.gcode_path) == body(b.gcode_path)
    assert not (tmp_path / "b.svg").exists()


def test_artwork_svg_input(cfg, tmp_path):
    p = tmp_path / "art.svg"
    p.write_text('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100">'
                 '<rect width="40" height="40" fill="#ff0000"/><rect x="50" width="40" height="40" fill="#0000ff"/></svg>',
                 encoding="utf-8")
    res = convert(p, tmp_path / "art.gcode", cfg)
    assert "; tools: 2  layers: 1" in res.gcode_path.read_text(encoding="utf-8")


@pytest.mark.parametrize("style", STYLES)
def test_every_style_end_to_end(cfg, tmp_path, style):
    cfg.style = style
    src = png(tmp_path, cv2.resize(shapes_rgb(noise=20), (400, 300)))
    res = convert(src, tmp_path / "o.gcode", cfg)
    segs, _ = parse_gcode(res.gcode_path)
    assert any(s["type"] == "print" for s in segs)


@pytest.mark.parametrize("style", STYLES)
def test_blank_image_gives_valid_empty_gcode(cfg, tmp_path, style):
    cfg.style = style
    res = convert(png(tmp_path, np.full((60, 80, 3), 255, np.uint8)), tmp_path / "o.gcode", cfg)
    segs, _ = parse_gcode(res.gcode_path)
    assert not any(s["type"] == "print" for s in segs)
    assert "; End of print" in res.gcode_path.read_text(encoding="utf-8")


@pytest.mark.parametrize("style", STYLES)
def test_tiny_image(cfg, tmp_path, style):
    cfg.style = style
    convert(png(tmp_path, cv2.resize(shapes_rgb(), (32, 24), interpolation=cv2.INTER_AREA)), tmp_path / "o.gcode", cfg)


def test_output_dirs_created(cfg, tmp_path):
    res = convert(LOGO, tmp_path / "a" / "b" / "o.gcode", cfg)
    assert res.gcode_path.exists() and res.svg_path.exists()


def test_cli_headless(tmp_path):
    out = tmp_path / "c.gcode"
    main(["-i", str(LOGO), "-o", str(out), "--headless", "-q", "--style", "contour"])
    assert out.exists() and 'data-style="contour"' in (tmp_path / "c.svg").read_text(encoding="utf-8")


def test_cli_mode_alias_and_svg_path(tmp_path):
    out = tmp_path / "m.gcode"; svg = tmp_path / "other" / "m.svg"
    main(["-i", str(LOGO), "-o", str(out), "--headless", "-q", "--mode", "lines", "--svg", str(svg)])
    assert out.exists() and 'data-style="edge"' in svg.read_text(encoding="utf-8")


def test_cli_rejects_unknown_style(tmp_path):
    with pytest.raises(SystemExit):
        main(["-i", str(LOGO), "-o", str(tmp_path / "x.gcode"), "--headless", "--style", "nope"])


def test_load_segments_for_svg_and_gcode(cfg, tmp_path):
    res = convert(LOGO, tmp_path / "v.gcode", cfg)
    for path in (res.svg_path, res.gcode_path):
        segs, meta = load_segments(path, cfg)
        assert any(s["type"] == "print" for s in segs)


def test_old_modules_removed():
    for m in ("toolpath", "edge_pipeline", "segmenter", "svg_loader"):
        assert importlib.util.find_spec(f"img2gcode.{m}") is None


def test_roi_selector_returns_page_mm(monkeypatch):
    import matplotlib.pyplot as plt
    from matplotlib.backend_bases import MouseEvent
    from shapely.geometry import box

    from img2gcode import roi_selector
    from img2gcode.model import Artwork, Region

    art = Artwork(300.0, 200.0, [Region(box(10, 10, 100, 100), 0)], [], [(200, 30, 30)], (600, 400),
                  np.full((40, 60, 3), 128, np.uint8), "fill")

    def drag_then_done():
        fig = plt.gcf()
        ax, ax_done = fig.axes[0], fig.axes[-1]
        fig.canvas.draw()

        def event(name, xy, transform):
            MouseEvent(name, fig.canvas, *transform.transform(xy), button=1)._process()

        event("button_press_event", (20, 30), ax.transData)
        event("motion_notify_event", (400, 90), ax.transData)     # past the right edge: clipped
        event("button_release_event", (400, 90), ax.transData)
        event("button_press_event", (0.5, 0.5), ax_done.transAxes)
        event("button_release_event", (0.5, 0.5), ax_done.transAxes)

    monkeypatch.setattr(roi_selector.plt, "show", drag_then_done)
    zones = roi_selector.select_exclusion_zones(art)
    assert np.allclose(zones, [(20, 30, 300, 90)])
    assert not plt.get_fignums()
