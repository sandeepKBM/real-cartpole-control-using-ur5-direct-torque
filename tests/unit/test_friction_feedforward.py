"""Tests for the opt-in friction feedforward term (2026-07-31) -- see
CartesianImpedanceConfig.friction_feedforward in
controller_core/x_axis_cartesian_impedance.py for the design rationale
(closing a real sim-to-real steady-state tracking gap that pure proportional
gain cannot fully cancel).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from controller_core.x_axis_cartesian_impedance import (  # noqa: E402
    JOINT_NAME_ORDER,
    CartesianImpedanceConfig,
    XAxisCartesianImpedanceController,
)


def _state(q, qd, target_x=0.0):
    return {
        "time": 0.0,
        "q": np.asarray(q, dtype=np.float64),
        "qd": np.asarray(qd, dtype=np.float64),
        "ee_pos": np.array([0.0, 0.0, 0.5], dtype=np.float64),
        "ee_quat": np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
        "ee_lin_vel": np.zeros(3, dtype=np.float64),
        "ee_ang_vel": np.zeros(3, dtype=np.float64),
        "target_x": float(target_x),
        "jacobian": np.eye(6, dtype=np.float64),
    }


def _zero_gain_config(**overrides) -> CartesianImpedanceConfig:
    """All PD/posture/damping gains zeroed so tau_friction_ff can be read in
    isolation from tau_preclip without needing to invert the backtrack/clip
    pipeline."""
    base = dict(
        kp_x=0.0, kd_x=0.0, kp_y=0.0, kd_y=0.0, kp_z=0.0, kd_z=0.0,
        kp_rot=0.0, kd_rot=0.0, kp_posture=0.0, kd_posture=0.0, kd_joint=0.0,
        tau_max_nm=np.array([100.0] * 6, dtype=np.float64),
    )
    base.update(overrides)
    return CartesianImpedanceConfig(**base)


def test_friction_feedforward_defaults_to_off():
    cfg = _zero_gain_config()
    assert cfg.friction_feedforward is False
    ctrl = XAxisCartesianImpedanceController(cfg)
    q0 = np.zeros(6)
    qd = np.array([0.5, -0.3, 0.2, 0.0, 0.1, -0.1], dtype=np.float64)
    st0 = _state(q0, np.zeros(6))
    ctrl.reset_from_state(st0)
    out = ctrl.compute(_state(q0, qd))
    np.testing.assert_array_equal(out.tau_friction_ff, np.zeros(6))
    assert out.friction_feedforward_active is False
    np.testing.assert_array_equal(out.tau, np.zeros(6))


def test_friction_feedforward_off_is_byte_identical_to_pre_change_behavior():
    """Regression: with the flag at its default (False), output must be
    identical regardless of what the (now-unused) coulomb/viscous arrays are
    set to -- proves the term genuinely gates off, not just defaults to
    small values."""
    q0 = np.zeros(6)
    qd = np.array([1.0, -1.0, 0.5, -0.5, 0.2, -0.2], dtype=np.float64)
    cfg_a = _zero_gain_config(friction_feedforward=False)
    cfg_b = _zero_gain_config(
        friction_feedforward=False,
        friction_ff_coulomb_nm=np.array([99.0] * 6, dtype=np.float64),
        friction_ff_viscous=np.array([99.0] * 6, dtype=np.float64),
    )
    for cfg in (cfg_a, cfg_b):
        ctrl = XAxisCartesianImpedanceController(cfg)
        ctrl.reset_from_state(_state(q0, np.zeros(6)))
        out = ctrl.compute(_state(q0, qd))
        np.testing.assert_array_equal(out.tau, np.zeros(6))


def test_friction_feedforward_opposes_commanded_friction_direction():
    """Positive qd -> positive feedforward torque (pushes WITH the motion,
    to cancel friction that opposes it); negative qd -> negative; qd=0 ->
    exactly zero (both tanh(0) and viscous*0 vanish)."""
    coulomb = np.array([5.0, 5.0, 5.0, 1.0, 1.0, 1.0], dtype=np.float64)
    viscous = np.array([0.4, 0.4, 0.4, 0.15, 0.15, 0.15], dtype=np.float64)
    cfg = _zero_gain_config(
        friction_feedforward=True,
        friction_ff_coulomb_nm=coulomb,
        friction_ff_viscous=viscous,
        friction_ff_qd_deadband=0.01,
    )
    ctrl = XAxisCartesianImpedanceController(cfg)
    q0 = np.zeros(6)
    ctrl.reset_from_state(_state(q0, np.zeros(6)))

    qd_zero = np.zeros(6)
    out_zero = ctrl.compute(_state(q0, qd_zero))
    np.testing.assert_allclose(out_zero.tau_friction_ff, np.zeros(6), atol=1e-12)
    assert out_zero.friction_feedforward_active is True

    qd_pos = np.array([0.5, 0.3, 0.2, 0.1, 0.05, 0.02], dtype=np.float64)
    out_pos = ctrl.compute(_state(q0, qd_pos))
    assert np.all(out_pos.tau_friction_ff > 0.0)

    qd_neg = -qd_pos
    out_neg = ctrl.compute(_state(q0, qd_neg))
    assert np.all(out_neg.tau_friction_ff < 0.0)
    # Odd function: tanh(-x) = -tanh(x), viscous*(-qd) = -(viscous*qd).
    np.testing.assert_allclose(out_neg.tau_friction_ff, -out_pos.tau_friction_ff, atol=1e-10)


def test_friction_feedforward_matches_closed_form_and_is_smooth_at_zero():
    coulomb = np.array([5.0, 5.0, 5.0, 1.0, 1.0, 1.0], dtype=np.float64)
    viscous = np.array([0.4, 0.4, 0.4, 0.15, 0.15, 0.15], dtype=np.float64)
    deadband = 0.02
    cfg = _zero_gain_config(
        friction_feedforward=True,
        friction_ff_coulomb_nm=coulomb,
        friction_ff_viscous=viscous,
        friction_ff_qd_deadband=deadband,
    )
    ctrl = XAxisCartesianImpedanceController(cfg)
    q0 = np.zeros(6)
    ctrl.reset_from_state(_state(q0, np.zeros(6)))

    for qd_scalar in (-0.5, -0.001, 0.0, 0.001, 0.5):
        qd = np.full(6, qd_scalar, dtype=np.float64)
        out = ctrl.compute(_state(q0, qd))
        expected = coulomb * np.tanh(qd / deadband) + viscous * qd
        np.testing.assert_allclose(out.tau_friction_ff, expected, atol=1e-10)

    # No sign()-style discontinuity: feedforward magnitude at a tiny positive
    # qd (deep in tanh's linear region, well under the deadband) must be
    # small and close to the tiny-negative case in magnitude -- not jump by
    # ~2*coulomb the way a hard sign(qd) term would right at qd=0.
    out_tiny_pos = ctrl.compute(_state(q0, np.full(6, 1e-4)))
    out_tiny_neg = ctrl.compute(_state(q0, np.full(6, -1e-4)))
    jump = np.abs(out_tiny_pos.tau_friction_ff - out_tiny_neg.tau_friction_ff)
    assert np.all(jump < 0.1 * coulomb)


def test_friction_feedforward_flows_through_backtrack_and_clip():
    """With real (nonzero) other gains too, the feedforward term must still
    show up in the final clipped tau -- i.e. it isn't bypassing the existing
    torque pipeline."""
    cfg = CartesianImpedanceConfig(
        kp_x=25.0, kd_x=8.0, kp_y=80.0, kd_y=15.0, kp_z=120.0, kd_z=20.0,
        kp_rot=0.0, kd_rot=10.0, kp_posture=2.0, kd_posture=0.5, kd_joint=0.8,
        tau_max_nm=np.array([100.0] * 6, dtype=np.float64),
        friction_feedforward=True,
        friction_ff_coulomb_nm=np.array([5.0] * 6, dtype=np.float64),
        friction_ff_viscous=np.array([0.4] * 6, dtype=np.float64),
    )
    ctrl_on = XAxisCartesianImpedanceController(cfg)
    ctrl_off = XAxisCartesianImpedanceController(
        CartesianImpedanceConfig(
            kp_x=25.0, kd_x=8.0, kp_y=80.0, kd_y=15.0, kp_z=120.0, kd_z=20.0,
            kp_rot=0.0, kd_rot=10.0, kp_posture=2.0, kd_posture=0.5, kd_joint=0.8,
            tau_max_nm=np.array([100.0] * 6, dtype=np.float64),
        )
    )
    q0 = np.zeros(6)
    qd = np.array([0.3, -0.2, 0.1, 0.0, 0.05, -0.05], dtype=np.float64)
    st0 = _state(q0, np.zeros(6))
    ctrl_on.reset_from_state(st0)
    ctrl_off.reset_from_state(st0)
    out_on = ctrl_on.compute(_state(q0, qd, target_x=0.05))
    out_off = ctrl_off.compute(_state(q0, qd, target_x=0.05))
    assert not np.allclose(out_on.tau, out_off.tau)


def test_from_controller_yaml_section_parses_friction_ff_fields():
    ctrl_yaml = {
        "gains": {},
        "friction_feedforward": True,
        "friction_ff_qd_deadband": 0.02,
        "friction_ff_coulomb_nm": {name: float(i + 1) for i, name in enumerate(JOINT_NAME_ORDER)},
        "friction_ff_viscous": {name: float(i + 1) * 0.1 for i, name in enumerate(JOINT_NAME_ORDER)},
        "torque_limits_initial": {name: 50.0 for name in JOINT_NAME_ORDER},
    }
    cfg = CartesianImpedanceConfig.from_controller_yaml_section(ctrl_yaml)
    assert cfg.friction_feedforward is True
    assert cfg.friction_ff_qd_deadband == 0.02
    np.testing.assert_allclose(cfg.friction_ff_coulomb_nm, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    np.testing.assert_allclose(cfg.friction_ff_viscous, [0.1, 0.2, 0.3, 0.4, 0.5, 0.6])


def test_from_controller_yaml_section_defaults_friction_ff_off():
    ctrl_yaml = {"gains": {}, "torque_limits_initial": {name: 50.0 for name in JOINT_NAME_ORDER}}
    cfg = CartesianImpedanceConfig.from_controller_yaml_section(ctrl_yaml)
    assert cfg.friction_feedforward is False
    np.testing.assert_allclose(cfg.friction_ff_coulomb_nm, [5.0, 5.0, 5.0, 1.0, 1.0, 1.0])
    np.testing.assert_allclose(cfg.friction_ff_viscous, [0.4, 0.4, 0.4, 0.15, 0.15, 0.15])
    # 0.05, not a smaller value -- deadband=0.01 was found to produce a real
    # closed-loop limit cycle in a same-night sim smoke test (typical
    # hold-phase |qd| sits right in the tanh term's steep transition region
    # at that setting). Locking in the validated default here so it can't
    # silently regress.
    assert cfg.friction_ff_qd_deadband == 0.05
