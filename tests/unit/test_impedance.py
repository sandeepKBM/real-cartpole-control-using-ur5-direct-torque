"""Smoke tests for Cartesian impedance core."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from controller_core.x_axis_cartesian_impedance import (  # noqa: E402
    CartesianImpedanceConfig,
    XAxisCartesianImpedanceController,
)


def _state(t, x, vx, y, vy, z, vz, quat, wx, wy, wz, q, qd, target_x, J):
    return {
        "time": t,
        "q": np.asarray(q, dtype=np.float64),
        "qd": np.asarray(qd, dtype=np.float64),
        "ee_pos": np.array([x, y, z], dtype=np.float64),
        "ee_quat": np.asarray(quat, dtype=np.float64),
        "ee_lin_vel": np.array([vx, vy, vz], dtype=np.float64),
        "ee_ang_vel": np.array([wx, wy, wz], dtype=np.float64),
        "target_x": float(target_x),
        "jacobian": np.asarray(J, dtype=np.float64),
    }


def test_hold_at_goal_zero_wrench_components() -> None:
    cfg = CartesianImpedanceConfig(
        kp_x=25.0,
        kd_x=8.0,
        kp_y=80.0,
        kd_y=15.0,
        kp_z=120.0,
        kd_z=20.0,
        kp_rot=20.0,
        kd_rot=5.0,
        kp_posture=2.0,
        kd_posture=0.5,
        kd_joint=0.8,
        tau_max_nm=np.array([50.0] * 6),
    )
    ctrl = XAxisCartesianImpedanceController(cfg)
    J = np.eye(6)
    quat = np.array([1.0, 0.0, 0.0, 0.0])
    q0 = np.zeros(6)
    st0 = _state(0.0, 0.1, 0.0, 0.0, 0.0, 0.5, 0.0, quat, 0, 0, 0, q0, np.zeros(6), 0.1, J)
    ctrl.reset_from_state(st0)
    out = ctrl.compute(st0)
    assert abs(out.x_error) < 1e-9
    assert np.linalg.norm(out.wrench[:3]) < 1e-6


def test_x_error_produces_positive_fx() -> None:
    cfg = CartesianImpedanceConfig(tau_max_nm=np.array([100.0] * 6))
    ctrl = XAxisCartesianImpedanceController(cfg)
    J = np.eye(6)
    quat = np.array([1.0, 0.0, 0.0, 0.0])
    st0 = _state(0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, quat, 0, 0, 0, np.zeros(6), np.zeros(6), 0.0, J)
    ctrl.reset_from_state(st0)
    st1 = _state(0.01, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, quat, 0, 0, 0, np.zeros(6), np.zeros(6), 0.05, J)
    out = ctrl.compute(st1)
    assert out.x_error > 0
    assert out.wrench[0] > 0


def test_torque_backtracking_shrinks_task_scale_under_tight_limits() -> None:
    cfg = CartesianImpedanceConfig(
        kp_x=25.0,
        kd_x=0.0,
        kp_y=0.0,
        kd_y=0.0,
        kp_z=0.0,
        kd_z=0.0,
        kp_rot=0.0,
        kd_rot=0.0,
        kp_posture=0.0,
        kd_posture=0.0,
        kd_joint=0.0,
        tau_max_nm=np.array([0.5] * 6, dtype=np.float64),
        torque_headroom=0.9,
        task_resample_factor=0.5,
        task_resample_min_scale=1.0 / 64.0,
        task_resample_max_iters=8,
    )
    ctrl = XAxisCartesianImpedanceController(cfg)
    J = np.eye(6)
    quat = np.array([1.0, 0.0, 0.0, 0.0])
    q0 = np.zeros(6)
    st0 = _state(0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, quat, 0.0, 0.0, 0.0, q0, np.zeros(6), 0.0, J)
    ctrl.reset_from_state(st0)
    st1 = _state(0.01, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, quat, 0.0, 0.0, 0.0, q0, np.zeros(6), 1.0, J)
    out = ctrl.compute(st1)
    assert out.task_backtrack_iters >= 1
    assert out.task_backtrack_scale < 1.0
    assert out.task_scale < 1.0
    assert out.task_feasible
    assert np.isclose(out.tau_task[0], out.tau_task_nominal[0] * out.task_backtrack_scale)
    assert np.max(np.abs(out.tau_preclip)) <= 0.5 * 0.9 + 1e-9
    assert np.max(np.abs(out.tau)) <= 0.5 + 1e-9


def test_bias_only_saturation_backtracks_full_torque_candidate() -> None:
    cfg = CartesianImpedanceConfig(
        kp_x=0.0,
        kd_x=0.0,
        kp_y=0.0,
        kd_y=0.0,
        kp_z=0.0,
        kd_z=0.0,
        kp_rot=0.0,
        kd_rot=0.0,
        kp_posture=2.0,
        kd_posture=0.5,
        kd_joint=0.8,
        tau_max_nm=np.array([0.5] * 6, dtype=np.float64),
        torque_headroom=0.9,
        task_resample_factor=0.5,
        task_resample_min_scale=1.0 / 64.0,
        task_resample_max_iters=8,
    )
    ctrl = XAxisCartesianImpedanceController(cfg)
    J = np.eye(6)
    quat = np.array([1.0, 0.0, 0.0, 0.0])
    q0 = np.zeros(6)
    st0 = _state(0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, quat, 0.0, 0.0, 0.0, q0, np.zeros(6), 0.0, J)
    ctrl.reset_from_state(st0)
    q1 = np.array([0.0, 1.5, 1.5, 1.5, 1.5, 1.5], dtype=np.float64)
    st1 = _state(0.01, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, quat, 0.0, 0.0, 0.0, q1, np.zeros(6), 0.0, J)
    out = ctrl.compute(st1)
    assert out.task_backtrack_iters >= 1
    assert out.task_backtrack_scale < 1.0
    assert out.task_feasible
    assert np.allclose(
        out.tau_preclip,
        out.tau_task + out.tau_damping + out.tau_posture + out.tau_gravity,
    )
    assert np.max(np.abs(out.tau_preclip)) <= 0.5 * 0.9 + 1e-9
    assert np.max(np.abs(out.tau)) <= 0.5 + 1e-9


def test_hold_current_pose_reanchors_controller_state() -> None:
    cfg = CartesianImpedanceConfig(
        kp_x=0.0,
        kd_x=0.0,
        kp_y=0.0,
        kd_y=0.0,
        kp_z=0.0,
        kd_z=0.0,
        kp_rot=0.0,
        kd_rot=0.0,
        kp_posture=0.0,
        kd_posture=0.0,
        kd_joint=0.0,
        tau_max_nm=np.array([10.0] * 6, dtype=np.float64),
    )
    ctrl = XAxisCartesianImpedanceController(cfg)
    J = np.eye(6)
    quat0 = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    st0 = _state(0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, quat0, 0.0, 0.0, 0.0, np.zeros(6), np.zeros(6), 0.0, J)
    ctrl.reset_from_state(st0)
    q1 = np.array([0.25, -0.35, 0.45, -0.55, 0.65, -0.75], dtype=np.float64)
    quat1 = np.array([0.70710678, 0.0, 0.70710678, 0.0], dtype=np.float64)
    st1 = _state(0.1, 0.2, -0.1, 0.3, 0.0, 0.7, 0.0, quat1, 0.0, 0.0, 0.0, q1, np.zeros(6), 1.0, J)
    st1["hold_current_pose"] = True
    out = ctrl.compute(st1)
    assert np.allclose(out.wrench, np.zeros(6))
    assert np.allclose(out.tau_posture, np.zeros(6))
    assert np.allclose(out.tau_damping, np.zeros(6))
    assert np.allclose(ctrl._q_rest, q1)
    assert np.isclose(ctrl._x0, 0.2)
    assert np.isclose(ctrl._y0, 0.3)
    assert np.isclose(ctrl._z0, 0.7)
    assert np.allclose(ctrl._quat0, quat1)
    st2 = _state(
        0.2,
        0.4,
        0.0,
        0.6,
        0.0,
        0.8,
        0.0,
        quat0,
        0.0,
        0.0,
        0.0,
        np.zeros(6),
        np.zeros(6),
        2.0,
        J,
    )
    st2["hold_current_pose"] = True
    _ = ctrl.compute(st2)
    assert np.isclose(ctrl._x0, 0.2)
    assert np.isclose(ctrl._y0, 0.3)
    assert np.isclose(ctrl._z0, 0.7)
    assert np.allclose(ctrl._q_rest, q1)
    assert np.allclose(ctrl._quat0, quat1)


if __name__ == "__main__":
    test_hold_at_goal_zero_wrench_components()
    test_x_error_produces_positive_fx()
    test_torque_backtracking_shrinks_task_scale_under_tight_limits()
    test_bias_only_saturation_backtracks_full_torque_candidate()
    test_hold_current_pose_reanchors_controller_state()
    print("impedance tests OK")
