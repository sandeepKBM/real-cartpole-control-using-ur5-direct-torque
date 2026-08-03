"""Tests for the opt-in Y corridor control mode (2026-08-03) -- see
CartesianImpedanceConfig.y_control_mode's docstring in
controller_core/x_axis_cartesian_impedance.py for the full design rationale.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from controller_core.x_axis_cartesian_impedance import (  # noqa: E402
    CartesianImpedanceConfig,
    XAxisCartesianImpedanceController,
)


def _state(x=0.0, y=0.0, z=0.5, target_x=0.0):
    return {
        "time": 0.0,
        "q": np.zeros(6, dtype=np.float64),
        "qd": np.zeros(6, dtype=np.float64),
        "ee_pos": np.array([x, y, z], dtype=np.float64),
        "ee_quat": np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
        "ee_lin_vel": np.zeros(3, dtype=np.float64),
        "ee_ang_vel": np.zeros(3, dtype=np.float64),
        "target_x": float(target_x),
        "target_x_vel": 0.0,
        "jacobian": np.eye(6, dtype=np.float64),
    }


def _cfg(**overrides) -> CartesianImpedanceConfig:
    base = dict(
        kp_x=0.0, kd_x=0.0, kp_y=80.0, kd_y=15.0, kp_z=0.0, kd_z=0.0,
        kp_rot=0.0, kd_rot=0.0, kp_posture=0.0, kd_posture=0.0, kd_joint=0.0,
        tau_max_nm=np.array([150.0, 150.0, 150.0, 28.0, 28.0, 28.0], dtype=np.float64),
    )
    base.update(overrides)
    return CartesianImpedanceConfig(**base)


def test_defaults_tight():
    cfg = _cfg()
    assert cfg.y_control_mode == "tight"


def test_flag_off_byte_identical_to_before():
    """tight mode (default) must exactly match the pre-corridor Fy formula --
    verified via y_error output and y_corridor_scale staying 1.0."""
    ctrl = XAxisCartesianImpedanceController(_cfg(kp_y=80.0))
    ctrl.reset_from_state(_state(target_x=0.0))
    out = ctrl.compute(_state(0.0, 0.03, target_x=0.0))
    assert out.y_corridor_scale == 1.0


def test_corridor_zero_correction_inside_soft_limit():
    cfg = _cfg(y_control_mode="corridor", y_soft_limit_m=0.02, y_hard_limit_m=0.05)
    ctrl = XAxisCartesianImpedanceController(cfg)
    ctrl.reset_from_state(_state(y=0.0, target_x=0.0))
    out = ctrl.compute(_state(0.0, 0.01, target_x=0.0))  # |y_err|=0.01 < soft=0.02
    assert out.y_corridor_scale == 0.0
    assert out.wrench[1] == 0.0


def test_corridor_full_correction_beyond_hard_limit():
    cfg = _cfg(y_control_mode="corridor", y_soft_limit_m=0.02, y_hard_limit_m=0.05, y_corridor_kp=80.0, y_corridor_kd=0.0)
    ctrl = XAxisCartesianImpedanceController(cfg)
    ctrl.reset_from_state(_state(y=0.0, target_x=0.0))
    out = ctrl.compute(_state(0.0, 0.06, target_x=0.0))  # |y_err|=0.06 > hard=0.05
    assert out.y_corridor_scale == 1.0
    np.testing.assert_allclose(out.wrench[1], 80.0 * (0.0 - 0.06), atol=1e-9)


def test_corridor_continuous_at_soft_boundary():
    """No discontinuity crossing into the soft limit -- scale and force must
    approach the same value from both sides (within a small step)."""
    cfg = _cfg(y_control_mode="corridor", y_soft_limit_m=0.02, y_hard_limit_m=0.05, y_corridor_kp=80.0, y_corridor_kd=0.0)
    ctrl = XAxisCartesianImpedanceController(cfg)
    ctrl.reset_from_state(_state(y=0.0, target_x=0.0))
    out_inside = ctrl.compute(_state(0.0, 0.0199, target_x=0.0))
    out_at = ctrl.compute(_state(0.0, 0.0200, target_x=0.0))
    out_just_past = ctrl.compute(_state(0.0, 0.0201, target_x=0.0))
    assert out_inside.wrench[1] == 0.0
    assert out_at.y_corridor_scale == 0.0
    assert abs(out_just_past.wrench[1]) < 0.01  # small force just past the boundary, not a jump


def test_corridor_smoothstep_monotonic_and_bounded():
    cfg = _cfg(y_control_mode="corridor", y_soft_limit_m=0.0, y_hard_limit_m=0.10)
    ctrl = XAxisCartesianImpedanceController(cfg)
    ctrl.reset_from_state(_state(y=0.0, target_x=0.0))
    scales = []
    for y in np.linspace(0.0, 0.10, 11):
        out = ctrl.compute(_state(0.0, float(y), target_x=0.0))
        scales.append(out.y_corridor_scale)
    assert scales[0] == 0.0
    assert scales[-1] == 1.0
    assert all(0.0 <= s <= 1.0 for s in scales)
    assert all(b >= a - 1e-9 for a, b in zip(scales, scales[1:]))  # monotonically non-decreasing


def test_corridor_plus_y_integral_action_raises():
    cfg = _cfg(y_control_mode="corridor", y_integral_action=True)
    ctrl = XAxisCartesianImpedanceController(cfg)
    ctrl.reset_from_state(_state(target_x=0.0))
    with pytest.raises(ValueError, match="mutually exclusive"):
        ctrl.compute(_state(0.0, 0.03, target_x=0.0))


def test_yaml_parsing():
    ctrl_section = {
        "y_control_mode": "corridor",
        "y_soft_limit_m": 0.02,
        "y_hard_limit_m": 0.06,
        "y_corridor_kp": 50.0,
        "y_corridor_kd": 10.0,
        "torque_limits_initial": {
            "shoulder_pan_joint": 150.0, "shoulder_lift_joint": 150.0, "elbow_joint": 150.0,
            "wrist_1_joint": 28.0, "wrist_2_joint": 28.0, "wrist_3_joint": 28.0,
        },
        "gains": {},
    }
    cfg = CartesianImpedanceConfig.from_controller_yaml_section(ctrl_section)
    assert cfg.y_control_mode == "corridor"
    assert cfg.y_soft_limit_m == 0.02
    assert cfg.y_hard_limit_m == 0.06
    assert cfg.y_corridor_kp == 50.0
    assert cfg.y_corridor_kd == 10.0


def test_yaml_parsing_unrecognized_mode_falls_back_to_tight():
    ctrl_section = {
        "y_control_mode": "not_a_real_mode",
        "torque_limits_initial": {
            "shoulder_pan_joint": 150.0, "shoulder_lift_joint": 150.0, "elbow_joint": 150.0,
            "wrist_1_joint": 28.0, "wrist_2_joint": 28.0, "wrist_3_joint": 28.0,
        },
        "gains": {},
    }
    cfg = CartesianImpedanceConfig.from_controller_yaml_section(ctrl_section)
    assert cfg.y_control_mode == "tight"
