"""Smoke tests for ``controller_core``. Run with any Python + numpy."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from controller_core import (  # noqa: E402
    SafetyConfig,
    SafetyMonitor,
    XAxisController,
    XAxisControllerConfig,
    as_robot_state,
    cartesian_force_to_joint_torque,
)


def _baseline_state(target_x: float = 0.1, x_now: float = 0.0) -> dict:
    # Simple diagonal Jacobian: each joint moves EE along one axis. Makes
    # tau = J^T * F easy to reason about (tau_0 == Fx, others zero).
    j_pos = np.eye(3, 6, dtype=np.float64)
    return {
        "time": 0.0,
        "q": np.zeros(6, dtype=np.float64),
        "qd": np.zeros(6, dtype=np.float64),
        "ee_pos": np.array([x_now, 0.0, 0.54], dtype=np.float64),
        "ee_quat": np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
        "target_x": float(target_x),
        "jacobian_pos": j_pos,
    }


def test_state_validation_happy_path() -> None:
    state = as_robot_state(_baseline_state())
    assert state["q"].shape == (6,)
    assert state["jacobian_pos"].shape == (3, 6)


def test_controller_returns_positive_fx_when_target_is_ahead() -> None:
    ctrl = XAxisController(XAxisControllerConfig(kp_x=200.0, kd_x=40.0, fx_max_n=50.0))
    state = as_robot_state(_baseline_state(target_x=0.1, x_now=0.0))
    out = ctrl.compute(state)
    assert out.mode == "cartesian_x_force"
    assert out.fx is not None and out.fx > 0
    assert out.x_error is not None
    assert abs(out.x_error - 0.1) < 1e-9


def test_controller_saturates_at_fx_max() -> None:
    ctrl = XAxisController(XAxisControllerConfig(kp_x=10000.0, kd_x=0.0, fx_max_n=5.0))
    out = ctrl.compute(as_robot_state(_baseline_state(target_x=1.0, x_now=0.0)))
    assert out.saturated is True
    assert out.fx == 5.0


def test_jt_adapter_projects_fx_to_first_joint() -> None:
    fx = 7.5
    j_pos = np.eye(3, 6, dtype=np.float64)
    out = cartesian_force_to_joint_torque(fx, j_pos, qd=np.zeros(6), kd_joint=0.0)
    assert out.mode == "torque"
    assert out.tau is not None
    expected = np.array([fx, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
    assert np.allclose(out.tau, expected)


def test_jt_adapter_damping_subtracts_kd_joint_times_qd() -> None:
    j_pos = np.zeros((3, 6), dtype=np.float64)
    qd = np.array([1.0, 0.5, -0.25, 0.0, 0.0, 0.0])
    out = cartesian_force_to_joint_torque(0.0, j_pos, qd=qd, kd_joint=2.0)
    expected = -2.0 * qd
    assert out.tau is not None
    assert np.allclose(out.tau, expected)


def test_jt_adapter_clips_to_tau_max() -> None:
    j_pos = np.eye(3, 6, dtype=np.float64)
    tau_max = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    out = cartesian_force_to_joint_torque(10.0, j_pos, tau_max=tau_max)
    assert out.saturated is True
    assert out.tau is not None
    assert out.tau[0] == 1.0


def test_safety_monitor_flags_torque_saturation() -> None:
    mon = SafetyMonitor(SafetyConfig(tau_max=np.array([1.0] * 6)))
    state = as_robot_state(_baseline_state())
    status = mon.check(state, tau=np.array([2.0, 0, 0, 0, 0, 0]))
    assert status.ok is False
    assert "tau saturation" in status.reason


def test_safety_monitor_flags_yz_drift() -> None:
    cfg = SafetyConfig(yz_drift_max_m=0.01)
    mon = SafetyMonitor(cfg)
    ok_state = as_robot_state(_baseline_state())
    assert mon.check(ok_state, tau=np.zeros(6)).ok is True
    drifted = _baseline_state()
    drifted["ee_pos"] = np.array([0.0, 0.05, 0.54])
    drifted_state = as_robot_state(drifted)
    result = mon.check(drifted_state, tau=np.zeros(6))
    assert result.ok is False
    assert "Y/Z drift" in result.reason


if __name__ == "__main__":
    tests = [
        test_state_validation_happy_path,
        test_controller_returns_positive_fx_when_target_is_ahead,
        test_controller_saturates_at_fx_max,
        test_jt_adapter_projects_fx_to_first_joint,
        test_jt_adapter_damping_subtracts_kd_joint_times_qd,
        test_jt_adapter_clips_to_tau_max,
        test_safety_monitor_flags_torque_saturation,
        test_safety_monitor_flags_yz_drift,
    ]
    for fn in tests:
        fn()
        print(f"OK  {fn.__name__}")
    print(f"\n{len(tests)} tests passed.")
