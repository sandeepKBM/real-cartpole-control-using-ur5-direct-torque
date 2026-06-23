"""Load ``config/controller.yaml`` into plain dicts (no ROS dependency)."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def load_yaml_config(path: str | Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("PyYAML required: pip install pyyaml") from exc
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Config file not found: {p}")
    with p.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return data


def torque_dict_to_array(order: tuple[str, ...], d: dict[str, Any]) -> list[float]:
    missing = [k for k in order if k not in d]
    if missing:
        raise KeyError(f"Torque dict missing keys: {missing}")
    return [float(d[k]) for k in order]
