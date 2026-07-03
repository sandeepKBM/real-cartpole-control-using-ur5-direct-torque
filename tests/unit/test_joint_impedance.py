"""Smoke tests for the joint-space impedance torque helper."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from controller_core import JointImpedanceConfig, JointImpedanceController  # noqa: E402


def test_joint_impedance_generates_restoring_torque() -> None:
    ctrl = JointImpedanceController(
        JointImpedanceConfig(
            kp_nm_per_rad=np.array([10.0] * 6, dtype=np.float64),
            kd_nm_per_rad_s=np.array([1.0] * 6, dtype=np.float64),
            tau_max_nm=np.array([100.0] * 6, dtype=np.float64),
        )
    )
    out = ctrl.compute(
        q=np.zeros(6, dtype=np.float64),
        qd=np.zeros(6, dtype=np.float64),
        q_ref=np.array([0.1, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64),
    )
    assert out.tau.shape == (6,)
    assert out.tau[0] > 0.0
    assert out.saturated is False


def test_joint_impedance_clips_to_limits() -> None:
    ctrl = JointImpedanceController(
        JointImpedanceConfig(
            kp_nm_per_rad=np.array([100.0] * 6, dtype=np.float64),
            kd_nm_per_rad_s=np.array([0.0] * 6, dtype=np.float64),
            tau_max_nm=np.array([1.0] * 6, dtype=np.float64),
        )
    )
    out = ctrl.compute(
        q=np.zeros(6, dtype=np.float64),
        qd=np.zeros(6, dtype=np.float64),
        q_ref=np.array([1.0] * 6, dtype=np.float64),
    )
    assert out.saturated is True
    assert np.all(np.abs(out.tau) <= 1.0 + 1e-12)


def test_joint_impedance_uses_velocity_feedback() -> None:
    ctrl = JointImpedanceController(
        JointImpedanceConfig(
            kp_nm_per_rad=np.array([0.0] * 6, dtype=np.float64),
            kd_nm_per_rad_s=np.array([2.0] * 6, dtype=np.float64),
            tau_max_nm=np.array([100.0] * 6, dtype=np.float64),
        )
    )
    out = ctrl.compute(
        q=np.zeros(6, dtype=np.float64),
        qd=np.array([0.5, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64),
        q_ref=np.zeros(6, dtype=np.float64),
        qd_ref=np.zeros(6, dtype=np.float64),
    )
    assert out.tau[0] < 0.0
    assert np.allclose(out.tau_feedforward, np.zeros(6))


if __name__ == "__main__":
    test_joint_impedance_generates_restoring_torque()
    test_joint_impedance_clips_to_limits()
    test_joint_impedance_uses_velocity_feedback()
    print("joint impedance tests OK")
