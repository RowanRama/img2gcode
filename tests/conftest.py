import os

os.environ.setdefault("MPLBACKEND", "Agg")

import pytest  # noqa: E402

from img2gcode.config import Config  # noqa: E402


@pytest.fixture
def cfg():
    """Fresh default config (identical to configs/default.cfg)."""
    return Config()
