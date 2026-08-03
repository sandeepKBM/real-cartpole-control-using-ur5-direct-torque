"""Tests for per-joint posture gain override (2026-08-03) -- see
CartesianImpedanceConfig.posture_kp_by_joint's docstring in
controller_core/x_axis_cartesian_impedance.py.
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


def _state(q, target_x=0.0):
    return {
        "time": 0.0,
        "q": np.asarray(q, dtype=np.float64),
        "qd": np.zeros(6, dtype=np.float64),
        "ee_pos": np.array([0.0, 0.0, 0.5], dtype=np.float64),
        "ee_quat": np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
        "ee_lin_vel": np.zeros(3, dtype=np.float64),
        "ee_ang_vel": np.zeros(3, dtype=np.float64),
        "target_x": float(target_x),
        "target_x_vel": 0.0,
        "jacobian": np.eye(6, dtype=np.float64),
    }


def _cfg(**overrides) -> CartesianImpedanceConfig:
    base = dict(
        kp_x=0.0, kd_x=0.0, kp_y=0.0, kd_y=0.0, kp_z=0.0, kd_z=0.0,
        kp_rot=0.0, kd_rot=0.0, kp_posture=25.0, kd_posture=6.0, kd_joint=0.0,
        tau_max_nm=np.array([150.0, 150.0, 150.0, 150.0, 150.0, 150.0], dtype=np.float64),
    )
    base.update(overrides)
    return CartesianImpedanceConfig(**base)


def test_defaults_none():
    cfg = _cfg()
    assert cfg.posture_kp_by_joint is None
    assert cfg.posture_kd_by_joint is None


def test_none_matches_scalar_exactly():
    q0 = np.array([-0.7, -0.8, -1.2, -1.0, 0.2, 0.0])
    q_drifted = q0 + np.array([0.1, -0.05, 0.02, 0.0, -0.03, 0.01])
    ctrl_scalar = XAxisCartesianImpedanceController(_cfg())
    ctrl_none = XAxisCartesianImpedanceController(_cfg(posture_kp_by_joint=None, posture_kd_by_joint=None))
    ctrl_scalar.reset_from_state(_state(q0))
    ctrl_none.reset_from_state(_state(q0))
    out_a = ctrl_scalar.compute(_state(q_drifted))
    out_b = ctrl_none.compute(_state(q_drifted))
    np.testing.assert_allclose(out_a.tau_posture, out_b.tau_posture, atol=1e-12)


def test_per_joint_override_matches_hand_computed():
    q0 = np.array([-0.7, -0.8, -1.2, -1.0, 0.2, 0.0])
    q_drifted = q0 + np.array([0.1, -0.05, 0.02, 0.0, -0.03, 0.01])
    kp_vec = np.array([150.0, 10.0, 10.0, 25.0, 50.0, 15.0])
    kd_vec = np.array([30.0, 2.0, 2.0, 6.0, 10.0, 4.0])
    cfg = _cfg(posture_kp_by_joint=kp_vec, posture_kd_by_joint=kd_vec)
    ctrl = XAxisCartesianImpedanceController(cfg)
    ctrl.reset_from_state(_state(q0))
    out = ctrl.compute(_state(q_drifted))
    expected = kp_vec * (q0 - q_drifted) - kd_vec * np.zeros(6)
    np.testing.assert_allclose(out.tau_posture, expected, atol=1e-9)


def test_partial_override_kp_only_falls_back_to_scalar_kd():
    q0 = np.array([-0.7, -0.8, -1.2, -1.0, 0.2, 0.0])
    q_drifted = q0 + np.array([0.1, 0.0, 0.0, 0.0, 0.0, 0.0])
    kp_vec = np.array([150.0, 10.0, 10.0, 25.0, 50.0, 15.0])
    cfg = _cfg(posture_kp_by_joint=kp_vec, kd_posture=6.0)
    ctrl = XAxisCartesianImpedanceController(cfg)
    ctrl.reset_from_state(_state(q0))
    out = ctrl.compute(_state(q_drifted))
    expected = kp_vec * (q0 - q_drifted) - 6.0 * np.zeros(6)
    np.testing.assert_allclose(out.tau_posture, expected, atol=1e-9)


def test_yaml_parsing():
    ctrl_section = {
        "posture_kp_by_joint": {
            "shoulder_pan_joint": 150.0, "shoulder_lift_joint": 10.0, "elbow_joint": 10.0,
            "wrist_1_joint": 25.0, "wrist_2_joint": 50.0, "wrist_3_joint": 15.0,
        },
        "posture_kd_by_joint": {
            "shoulder_pan_joint": 30.0, "shoulder_lift_joint": 2.0, "elbow_joint": 2.0,
            "wrist_1_joint": 6.0, "wrist_2_joint": 10.0, "wrist_3_joint": 4.0,
        },
        "torque_limits_initial": {
            "shoulder_pan_joint": 150.0, "shoulder_lift_joint": 150.0, "elbow_joint": 150.0,
            "wrist_1_joint": 28.0, "wrist_2_joint": 28.0, "wrist_3_joint": 28.0,
        },
        "gains": {},
    }
    cfg = CartesianImpedanceConfig.from_controller_yaml_section(ctrl_section)
    np.testing.assert_allclose(cfg.posture_kp_by_joint, [150.0, 10.0, 10.0, 25.0, 50.0, 15.0])
    np.testing.assert_allclose(cfg.posture_kd_by_joint, [30.0, 2.0, 2.0, 6.0, 10.0, 4.0])
