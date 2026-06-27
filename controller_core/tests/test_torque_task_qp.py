"""Tests for task-space torque QP inner loop."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from controller_core.box_qp import solve_box_qp  # noqa: E402
from controller_core.torque_task_qp import (  # noqa: E402
    TorqueTaskQPConfig,
    TorqueTaskQPController,
    _velocity_implied_torque_bounds,
)


def test_box_qp_respects_bounds() -> None:
    h = np.eye(3, dtype=np.float64)
    f = np.array([-10.0, 0.0, 10.0], dtype=np.float64)
    lo = np.array([-1.0, -2.0, -3.0], dtype=np.float64)
    hi = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    x = solve_box_qp(h, f, lo, hi)
    assert np.all(x >= lo - 1e-9)
    assert np.all(x <= hi + 1e-9)


def test_velocity_bounds_tighten_torque_box() -> None:
    q = np.zeros(6, dtype=np.float64)
    qd = np.array([2.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
    kp = np.full(6, 10.0, dtype=np.float64)
    kd = np.full(6, 5.0, dtype=np.float64)
    lo, hi = _velocity_implied_torque_bounds(q, qd, q, kp=kp, kd=kd, qd_max=1.5)
    assert float(hi[0]) < float(kp[0] * 1.5)


def test_qp_controller_y_transport_holds_orthogonal_axes() -> None:
    """World-Y transport must not command motion along X/Z from target_ee_pos."""
    ctrl = TorqueTaskQPController(
        TorqueTaskQPConfig(
            kp_x=10.0,
            kp_y=20.0,
            kp_z=30.0,
            tau_max_nm=np.array([8.0, 8.0, 8.0, 2.5, 2.5, 2.5], dtype=np.float64),
            max_joint_velocity_radps=2.0,
        )
    )
    jacobian = np.eye(6, dtype=np.float64)
    ee0 = np.array([-0.1, -0.4, 0.4], dtype=np.float64)
    state = {
        "time": 0.0,
        "q": np.zeros(6, dtype=np.float64),
        "qd": np.zeros(6, dtype=np.float64),
        "ee_pos": ee0.copy(),
        "ee_quat": np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
        "ee_lin_vel": np.zeros(3, dtype=np.float64),
        "ee_ang_vel": np.zeros(3, dtype=np.float64),
        "target_x": float(ee0[0]),
        "target_axis": float(ee0[1] + 0.02),
        "target_axis_vel": 0.01,
        "transport_axis_index": 1,
        "target_ee_pos": np.array([ee0[0], ee0[1] + 0.02, ee0[2]], dtype=np.float64),
        "target_ee_vel": np.array([0.0, 0.01, 0.0], dtype=np.float64),
        "jacobian": jacobian,
    }
    ctrl.reset_from_state(state)
    out = ctrl.compute(state)
    assert out.y_error > 0.0
    assert abs(out.x_error) < 1.0e-9
    assert abs(out.z_error) < 1.0e-9
    assert float(out.wrench[1]) != 0.0
    assert float(out.wrench[0]) == 0.0
    assert float(out.wrench[2]) == 0.0


def test_qp_controller_returns_finite_torque() -> None:
    ctrl = TorqueTaskQPController(
        TorqueTaskQPConfig(
            tau_max_nm=np.array([8.0, 8.0, 8.0, 2.5, 2.5, 2.5], dtype=np.float64),
            max_joint_velocity_radps=2.0,
        )
    )
    jacobian = np.eye(6, dtype=np.float64)
    state = {
        "time": 0.0,
        "q": np.zeros(6, dtype=np.float64),
        "qd": np.zeros(6, dtype=np.float64),
        "ee_pos": np.array([0.0, -0.4, 0.4], dtype=np.float64),
        "ee_quat": np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
        "ee_lin_vel": np.zeros(3, dtype=np.float64),
        "ee_ang_vel": np.zeros(3, dtype=np.float64),
        "target_x": 0.0,
        "target_axis": -0.38,
        "target_axis_vel": 0.02,
        "transport_axis_index": 1,
        "jacobian": jacobian,
    }
    ctrl.reset_from_state(state)
    out = ctrl.compute(state)
    assert np.all(np.isfinite(out.tau))
    assert float(np.max(np.abs(out.tau))) <= 8.0 + 1e-9
