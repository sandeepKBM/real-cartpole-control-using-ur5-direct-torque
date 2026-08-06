"""
Small, permissive YAML-value parsing helpers for CartesianImpedanceConfig.

Split out of the former single-file ``x_axis_cartesian_impedance.py`` module
(pure structural refactor; see the package ``__init__.py`` for the original
module docstring).
"""

from __future__ import annotations

from typing import Any, Literal


def _parse_y_control_mode(raw: Any) -> Literal["tight", "corridor"]:
    """Case-insensitive parse of the ``y_control_mode`` YAML value, matching
    _parse_friction_model's permissive convention: an unrecognized value
    falls back to "tight" (the historical default) rather than raising."""
    value = str(raw).lower()
    if value == "corridor":
        return "corridor"
    return "tight"


def _parse_friction_model(raw: Any) -> Literal["static", "lugre", "karnopp"]:
    """Case-insensitive parse of the ``friction_model`` YAML value.

    Matches the pre-existing (2026-08-01) permissive convention: any value that
    isn't a recognized model name silently falls back to ``"static"`` rather
    than raising, so a typo in a config never breaks loading -- it just leaves
    the historical default behavior in place. Extended 2026-08-02 to recognize
    ``"karnopp"`` alongside the pre-existing ``"lugre"``.
    """
    value = str(raw).lower()
    if value == "lugre":
        return "lugre"
    if value == "karnopp":
        return "karnopp"
    return "static"
