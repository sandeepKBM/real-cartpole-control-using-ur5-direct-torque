"""Shared pytest setup: put the repo root on sys.path once, apply dir markers."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

_DIR_MARKERS = ("unit", "mujoco", "hardware")


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    for item in items:
        parts = Path(str(item.fspath)).parts
        for marker in _DIR_MARKERS:
            if marker in parts:
                item.add_marker(getattr(pytest.mark, marker))
