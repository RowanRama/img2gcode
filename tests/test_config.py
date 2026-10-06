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
