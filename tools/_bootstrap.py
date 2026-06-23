"""Helpers for adding the repo root to ``sys.path`` in standalone scripts."""

from __future__ import annotations

import sys
from pathlib import Path


def ensure_repo_root() -> Path:
    repo_root = Path(__file__).resolve().parents[1]
    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)
    simulation_dir = repo_root / "simulation"
    simulation_dir_str = str(simulation_dir)
    if simulation_dir_str not in sys.path:
        sys.path.insert(0, simulation_dir_str)
    return repo_root
