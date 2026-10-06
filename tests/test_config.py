import pytest

from img2gcode.config import Config


def test_default_cfg_file_matches_dataclass_defaults():
    assert Config.load() == Config()


def test_key_values():
    cfg = Config.load()
    assert cfg.style == "fill"
    assert (cfg.machine.print_width, cfg.machine.print_height) == (1000.0, 700.0)
    assert cfg.extrusion.effective_line_width == 4.0
    assert cfg.edge.detail == 0.5
    assert cfg.oneline.tsp_seconds == 15.0
    assert cfg.hatch.angles == "45,-45,0"


def test_user_file_overrides_and_ignores_unknown_keys(tmp_path):
    p = tmp_path / "u.cfg"
    p.write_text(
        "[mode]\ntype = Contour\n"
        "[edge]\nclahe_clip = 3\ndetail = 0.8\n"
        "[infill]\nconnect_lines = false\n"
        "[tools]\nnum_tools = 3\n",
        encoding="utf-8",
    )
    cfg = Config.load(p)
    assert cfg.style == "contour"
    assert cfg.edge.detail == 0.8
    assert cfg.infill.connect_lines is False
    assert cfg.tools.num_tools == 3


def test_missing_user_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        Config.load(tmp_path / "nope.cfg")


def test_line_width_override():
    cfg = Config()
    cfg.extrusion.line_width_mm = 4.5
    assert cfg.extrusion.effective_line_width == 4.5


def test_style_registry():
    from img2gcode.styles import ALIASES, STYLES, resolve_style

    assert STYLES == ("fill", "edge", "contour", "oneline", "squiggle", "hatch")
    assert ALIASES == {"lines": "edge"}
    assert resolve_style("Lines") == "edge"
    assert resolve_style("hatch") == "hatch"
    with pytest.raises(ValueError):
        resolve_style("nope")


def test_cv2_threads_pinned():
    import cv2

    import img2gcode  # noqa: F401

    assert cv2.getNumThreads() == 1


def test_user_file_with_bom_loads(tmp_path):
    p = tmp_path / "bom.cfg"
    p.write_text("\ufeff[mode]\ntype = hatch\n", encoding="utf-8")
    assert Config.load(p).style == "hatch"


def test_percent_in_value_is_literal(tmp_path):
    p = tmp_path / "pct.cfg"
    p.write_text("[edge]\ndebug_edges_path = out_50%.png\n", encoding="utf-8")
    assert Config.load(p).edge.debug_edges_path == "out_50%.png"


def test_int_fields_round(tmp_path):
    p = tmp_path / "i.cfg"
    p.write_text("[tools]\nnum_tools = 2.9\n", encoding="utf-8")
    assert Config.load(p).tools.num_tools == 3


def test_stroke_membership_is_identity_based():
    import numpy as np

    from img2gcode.model import Stroke

    s = Stroke(np.zeros((3, 2)))
    other = Stroke(np.ones((3, 2)))
    assert s in [other, s]
    lst = [other, s]
    lst.remove(s)
    assert lst == [other]


def test_bands_rgb_fills_uneven_width():
    from tests.synth import bands_rgb

    img = bands_rgb(values=(20, 80, 140), h=4, w=10)
    assert (img[:, 9] == 140).all() and (img[:, 6] == 140).all()
