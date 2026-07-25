"""Behavior-preserving evidence for the hardware/transport_common.py extractions.

Each test reconstructs the exact pre-refactor inline code and asserts the new
shared helper produces byte-identical output, so these extractions can be
trusted on physical-robot safety paths that have no live-hardware test access.
"""

from __future__ import annotations

from dataclasses import fields
from pathlib import Path

import pytest
import yaml

from controller_core.safety import ImpedanceSafetyConfig
from hardware.safety import CartesianMoveLimits
from hardware.transport_common import (
    impedance_safety_config_from_section,
    max_abs_qd_from_trace,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
TUNED_CONFIG = REPO_ROOT / "config" / "ur5e_mujoco_torque_osc_tuned.yaml"

# The 5 fields both original loaders (_load_impedance_bundle / _load_safety_cfg)
# ever set. Everything else on ImpedanceSafetyConfig was left at its dataclass
# default in both call sites.
_SAFETY_SCALAR_FIELDS = (
    "max_abs_y_drift_m",
    "max_abs_z_drift_m",
    "max_abs_orthogonal_drift_m",
    "max_orientation_error_rad",
    "max_joint_velocity_radps",
)


def _old_impedance_safety_config(safety_raw: dict) -> ImpedanceSafetyConfig:
    """Verbatim copy of the pre-refactor inline construction shared by
    direct_torque_transport._load_impedance_bundle and
    urscript_transport._load_safety_cfg (note the 3.0 velocity default)."""
    return ImpedanceSafetyConfig(
        max_abs_y_drift_m=float(safety_raw.get("max_abs_y_drift_m", 0.03)),
        max_abs_z_drift_m=float(safety_raw.get("max_abs_z_drift_m", 0.03)),
        max_abs_orthogonal_drift_m=float(safety_raw.get("max_abs_orthogonal_drift_m", 0.03)),
        max_orientation_error_rad=float(safety_raw.get("max_orientation_error_rad", 0.25)),
        max_joint_velocity_radps=float(safety_raw.get("max_joint_velocity_radps", 3.0)),
    )


def _assert_same_impedance_safety(new: ImpedanceSafetyConfig, old: ImpedanceSafetyConfig) -> None:
    # ImpedanceSafetyConfig carries np.ndarray fields, so `==` is ambiguous;
    # compare every field explicitly instead.
    for f in fields(ImpedanceSafetyConfig):
        new_v = getattr(new, f.name)
        old_v = getattr(old, f.name)
        if f.name in ("q_lower", "q_upper"):
            assert (new_v == old_v).all(), f.name
        else:
            assert new_v == old_v, f.name


@pytest.mark.parametrize(
    "safety_raw",
    [
        {},  # all defaults -- exercises the 3.0 velocity default in particular
        {"max_joint_velocity_radps": 1.5},  # single override
        {
            "max_abs_y_drift_m": 0.01,
            "max_abs_z_drift_m": 0.02,
            "max_abs_orthogonal_drift_m": 0.015,
            "max_orientation_error_rad": 0.1,
            "max_joint_velocity_radps": 2.0,
        },
        {"unrelated_key": 999},  # ignored keys must not leak in
    ],
)
def test_impedance_safety_config_matches_old_inline(safety_raw: dict) -> None:
    assert TUNED_CONFIG.exists()
    _assert_same_impedance_safety(
        impedance_safety_config_from_section(safety_raw),
        _old_impedance_safety_config(safety_raw),
    )


def test_impedance_safety_config_none_is_empty_dict() -> None:
    _assert_same_impedance_safety(
        impedance_safety_config_from_section(None),
        _old_impedance_safety_config({}),
    )


def test_impedance_safety_config_velocity_default_is_three_not_dataclass_default() -> None:
    # Guards the subtle bug: ImpedanceSafetyConfig's own default is 1.5, but the
    # hardware loaders deliberately default to 3.0 when the YAML omits it.
    cfg = impedance_safety_config_from_section({})
    assert cfg.max_joint_velocity_radps == 3.0
    assert ImpedanceSafetyConfig().max_joint_velocity_radps == 1.5


def test_impedance_safety_config_from_real_tuned_config() -> None:
    cfg = yaml.safe_load(TUNED_CONFIG.read_text(encoding="utf-8"))
    safety_raw = (cfg.get("controller", {}) or {}).get("safety", {}) or {}
    _assert_same_impedance_safety(
        impedance_safety_config_from_section(safety_raw),
        _old_impedance_safety_config(safety_raw),
    )


def _old_move_limits(safety_cfg: ImpedanceSafetyConfig) -> CartesianMoveLimits:
    """Verbatim copy of the pre-refactor inline construction shared by
    direct_torque_transport and urscript_transport."""
    return CartesianMoveLimits(
        qd_max_radps=safety_cfg.max_joint_velocity_radps,
        max_off_axis_drift_m=min(safety_cfg.max_abs_y_drift_m, safety_cfg.max_abs_z_drift_m),
        max_orientation_error_rad=safety_cfg.max_orientation_error_rad,
    )


@pytest.mark.parametrize(
    "safety_cfg",
    [
        ImpedanceSafetyConfig(),
        ImpedanceSafetyConfig(max_joint_velocity_radps=3.0),
        ImpedanceSafetyConfig(
            max_abs_y_drift_m=0.01,
            max_abs_z_drift_m=0.02,
            max_orientation_error_rad=0.1,
            max_joint_velocity_radps=2.0,
        ),
        ImpedanceSafetyConfig(max_abs_y_drift_m=0.05, max_abs_z_drift_m=0.005),
    ],
)
def test_cartesian_move_limits_from_safety_matches_old_inline(
    safety_cfg: ImpedanceSafetyConfig,
) -> None:
    new = CartesianMoveLimits.from_impedance_safety_config(safety_cfg)
    old = _old_move_limits(safety_cfg)
    # CartesianMoveLimits is all-scalar -> dataclass __eq__ is well-defined.
    assert new == old


def test_cartesian_move_limits_picks_smaller_of_y_z_and_keeps_new_defaults() -> None:
    safety_cfg = ImpedanceSafetyConfig(max_abs_y_drift_m=0.05, max_abs_z_drift_m=0.005)
    limits = CartesianMoveLimits.from_impedance_safety_config(safety_cfg)
    assert limits.max_off_axis_drift_m == 0.005  # min(y, z)
    # The genuinely new guards fall back to the class defaults, unchanged.
    default = CartesianMoveLimits()
    assert limits.max_tcp_speed_mps == default.max_tcp_speed_mps
    assert limits.max_tcp_accel_mps2 == default.max_tcp_accel_mps2
    assert limits.max_waypoint_jump_m == default.max_waypoint_jump_m
    assert limits.max_axis_error_growth_steps == default.max_axis_error_growth_steps


def _old_max_abs_qd(trace_rows: list[dict]) -> float:
    """Verbatim copy of the pre-refactor inline expression shared by
    position_transport and direct_torque_transport summary building."""
    return float(
        max(
            (max(abs(v) for v in row.get("qd", [0.0] * 6)) for row in trace_rows),
            default=0.0,
        )
    )


@pytest.mark.parametrize(
    "trace_rows",
    [
        [],  # empty trace -> 0.0
        [{"qd": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]}],
        [{"qd": [0.1, -0.5, 0.3, 0.0, -0.2, 0.4]}, {"qd": [0.05, 0.6, -0.1, 0.2, 0.0, -0.3]}],
        [{"qd": [-1.2, 0.0, 0.0, 0.0, 0.0, 0.0]}],  # negative magnitude dominates
        [{"other": 1}],  # missing "qd" key -> falls back to six zeros
        [{"qd": [0.1] * 6}, {"other": 1}, {"qd": [0.9] * 6}],
    ],
)
def test_max_abs_qd_from_trace_matches_old_inline(trace_rows: list[dict]) -> None:
    assert max_abs_qd_from_trace(trace_rows) == _old_max_abs_qd(trace_rows)
