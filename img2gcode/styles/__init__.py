"""Drawing styles. Each module exposes make(rgb, cfg) -> Artwork."""

from __future__ import annotations

import importlib
from typing import Callable

STYLES = ("fill", "edge", "contour", "oneline", "squiggle", "hatch")
ALIASES = {"lines": "edge"}


def resolve_style(name: str) -> str:
    key = name.strip().lower()
    key = ALIASES.get(key, key)
    if key not in STYLES:
        raise ValueError(f"Unknown style {name!r}; choose from {', '.join(STYLES)}")
    return key


def get_style(name: str) -> Callable:
    return importlib.import_module(f"{__name__}.{resolve_style(name)}").make
