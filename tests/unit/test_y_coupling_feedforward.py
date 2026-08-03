"""Tests for the opt-in Y-coupling feedforward term (2026-08-02) -- see
``CartesianImpedanceConfig.y_coupling_feedforward``'s docstring in
``controller_core/x_axis_cartesian_impedance.py`` for the full design
rationale. Untested against real drift reduction here -- these tests only
prove the mechanism does exactly what it's specified to do (shift y_des by
``-gain*(x_des - x0)``, byte-identical to off when disabled/gain=0); whether
it actually reduces real Y-drift at the -45deg pose is a separate sim/A-B
question, not a unit-test question.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from controller_core.x_axis_cartesian_impedance import (  # noqa: E402
    CartesianImpedanceConfig,
    XAxisCartesianImpedanceController,
)


def _state(x, y=0.0, z=0.5, target_x=0.0, target_x_vel=0.0):
    return {
        "time": 0.0,
        "q": np.zeros(6, dtype=np.float64),
        "qd": np.zeros(6, dtype=np.float64),
        "ee_pos": np.array([x, y, z], dtype=np.float64),
        "ee_quat": np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
        "ee_lin_vel": np.zeros(3, dtype=np.float64),
        "ee_ang_vel": np.zeros(3, dtype=np.float64),
        "target_x": float(target_x),
        "target_x_vel": float(target_x_vel),
        "jacobian": np.eye(6, dtype=np.float64),
    }


def _zero_gain_config(**overrides) -> CartesianImpedanceConfig:
    base = dict(
        kp_x=0.0, kd_x=0.0, kp_y=0.0, kd_y=0.0, kp_z=0.0, kd_z=0.0,
        kp_rot=0.0, kd_rot=0.0, kp_posture=0.0, kd_posture=0.0, kd_joint=0.0,
        tau_max_nm=np.array([100.0] * 6, dtype=np.float64),
    )
    base.update(overrides)
    return CartesianImpedanceConfig(**base)


def test_defaults_off():
    cfg = _zero_gain_config()
    assert cfg.y_coupling_feedforward is False
    assert cfg.y_coupling_gain == 0.7


def test_flag_off_byte_identical_regardless_of_gain():
    ctrl_off_default_gain = XAxisCartesianImpedanceController(_zero_gain_config(kp_y=10.0))
    ctrl_off_other_gain = XAxisCartesianImpedanceController(_zero_gain_config(kp_y=10.0, y_coupling_gain=5.0))
    ctrl_off_default_gain.reset_from_state(_state(0.0, target_x=0.0))
    ctrl_off_other_gain.reset_from_state(_state(0.0, target_x=0.0))

    out_a = ctrl_off_default_gain.compute(_state(0.1, target_x=0.1))
    out_b = ctrl_off_other_gain.compute(_state(0.1, target_x=0.1))
    assert out_a.y_error == out_b.y_error
    np.testing.assert_allclose(out_a.tau, out_b.tau, atol=1e-12)


def test_zero_gain_matches_flag_off():
    cfg_off = _zero_gain_config(kp_y=10.0)
    cfg_on_zero_gain = _zero_gain_config(kp_y=10.0, y_coupling_feedforward=True, y_coupling_gain=0.0)
    ctrl_off = XAxisCartesianImpedanceController(cfg_off)
    ctrl_on = XAxisCartesianImpedanceController(cfg_on_zero_gain)
    ctrl_off.reset_from_state(_state(0.0, target_x=0.0))
    ctrl_on.reset_from_state(_state(0.0, target_x=0.0))

    out_off = ctrl_off.compute(_state(0.05, target_x=0.15))
    out_on = ctrl_on.compute(_state(0.05, target_x=0.15))
    assert out_off.y_error == out_on.y_error


def test_feedforward_shifts_y_target_by_expected_amount():
    """x0 anchored at 0.0 (reset_from_state), target_x moves to 0.1 ->
    y_des should shift by -gain*0.1 relative to the flag-off case, so
    y_error (= y_des - actual_y, actual_y held fixed) should differ by
    exactly -gain*0.1."""
    gain = 0.7
    dx = 0.1
    cfg_off = _zero_gain_config(kp_y=10.0)
    cfg_on = _zero_gain_config(kp_y=10.0, y_coupling_feedforward=True, y_coupling_gain=gain)
    ctrl_off = XAxisCartesianImpedanceController(cfg_off)
    ctrl_on = XAxisCartesianImpedanceController(cfg_on)
    # Anchor both at the same x0/y0.
    ctrl_off.reset_from_state(_state(0.0, y=0.02, target_x=0.0))
    ctrl_on.reset_from_state(_state(0.0, y=0.02, target_x=0.0))

    actual_y = 0.02  # held fixed -- same actual position for both controllers
    out_off = ctrl_off.compute(_state(dx, y=actual_y, target_x=dx))
    out_on = ctrl_on.compute(_state(dx, y=actual_y, target_x=dx))

    expected_shift = -gain * dx
    np.testing.assert_allclose(out_on.y_error - out_off.y_error, expected_shift, atol=1e-12)
