"""Tests for the simulation-only hardware shadow torque model."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from controller_core import HardwareShadowConfig, HardwareShadowModel  # noqa: E402


def test_shadow_identity_when_disabled_parameters_are_neutral() -> None:
    ctrl = HardwareShadowModel(
        HardwareShadowConfig(
            tau_max_nm=np.array([100.0] * 6, dtype=np.float64),
            command_delay_steps=0,
            torque_scale=1.0,
            torque_rate_limit_nm_per_s=np.full(6, np.inf, dtype=np.float64),
            viscous_damping_nm_per_rad_s=np.zeros(6, dtype=np.float64),
            coulomb_friction_nm=np.zeros(6, dtype=np.float64),
            deadzone_nm=np.zeros(6, dtype=np.float64),
            friction_velocity_eps_rad_s=1e-3,
            dt_s=0.01,
        )
    )
    cmd = np.array([1.0, -2.0, 3.0, -4.0, 5.0, -6.0], dtype=np.float64)
    out = ctrl.apply(cmd, qvel=np.zeros(6, dtype=np.float64))
    assert np.allclose(out.tau_applied_nm, cmd)
    assert out.clipped is False
    assert out.delayed is False
    assert out.rate_limited is False
    assert out.deadzone_applied is False
    assert out.friction_applied is False


def test_shadow_applies_command_delay_and_slew_rate_limit() -> None:
    ctrl = HardwareShadowModel(
        HardwareShadowConfig(
            tau_max_nm=np.array([100.0] * 6, dtype=np.float64),
            command_delay_steps=1,
            torque_scale=1.0,
            torque_rate_limit_nm_per_s=np.array([1.0] * 6, dtype=np.float64),
            viscous_damping_nm_per_rad_s=np.zeros(6, dtype=np.float64),
            coulomb_friction_nm=np.zeros(6, dtype=np.float64),
            deadzone_nm=np.zeros(6, dtype=np.float64),
            friction_velocity_eps_rad_s=1e-3,
            dt_s=1.0,
        )
    )
    cmd = np.array([2.0] + [0.0] * 5, dtype=np.float64)
    first = ctrl.apply(cmd, qvel=np.zeros(6, dtype=np.float64))
    second = ctrl.apply(cmd, qvel=np.zeros(6, dtype=np.float64))

    assert np.allclose(first.tau_applied_nm, np.zeros(6, dtype=np.float64))
    assert second.delayed is True
    assert second.rate_limited is True
    assert np.isclose(second.tau_applied_nm[0], 1.0)
    assert second.queue_depth == 1


def test_shadow_deadzone_and_friction_reduce_torque() -> None:
    ctrl = HardwareShadowModel(
        HardwareShadowConfig(
            tau_max_nm=np.array([100.0] * 6, dtype=np.float64),
            command_delay_steps=0,
            torque_scale=1.0,
            torque_rate_limit_nm_per_s=np.full(6, np.inf, dtype=np.float64),
            viscous_damping_nm_per_rad_s=np.array([0.2] * 6, dtype=np.float64),
            coulomb_friction_nm=np.array([0.3] * 6, dtype=np.float64),
            deadzone_nm=np.array([0.5] * 6, dtype=np.float64),
            friction_velocity_eps_rad_s=1e-3,
            dt_s=0.01,
        )
    )
    out = ctrl.apply(
        np.array([1.0] + [0.0] * 5, dtype=np.float64),
        qvel=np.array([2.0] + [0.0] * 5, dtype=np.float64),
    )
    expected = 1.0 - (0.2 * 2.0 + 0.3)
    assert out.deadzone_applied is False
    assert out.friction_applied is True
    assert np.isclose(out.tau_applied_nm[0], expected)

    deadzone_out = ctrl.apply(
        np.array([0.1] + [0.0] * 5, dtype=np.float64),
        qvel=np.zeros(6, dtype=np.float64),
    )
    assert deadzone_out.deadzone_applied is True
    assert np.isclose(deadzone_out.tau_applied_nm[0], 0.0)


if __name__ == "__main__":
    test_shadow_identity_when_disabled_parameters_are_neutral()
    test_shadow_applies_command_delay_and_slew_rate_limit()
    test_shadow_deadzone_and_friction_reduce_torque()
    print("hardware shadow tests OK")
