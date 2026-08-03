"""Tests for controller_core/cartesian_velocity_controller.py -- pure P-only
resolved-rate Cartesian velocity law, no dynamics."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from controller_core.cartesian_velocity_controller import (  # noqa: E402
    CartesianVelocityConfig,
    CartesianVelocityController,
)


def _state(ee_pos, ee_quat=(1.0, 0.0, 0.0, 0.0), target_ee_pos=None, target_ee_vel=None):
    return {
        "time": 0.0,
        "q": np.zeros(6),
        "qd": np.zeros(6),
        "ee_pos": np.asarray(ee_pos, dtype=np.float64),
        "ee_quat": np.asarray(ee_quat, dtype=np.float64),
        "target_x": float(ee_pos[0]),
        "target_ee_pos": None if target_ee_pos is None else np.asarray(target_ee_pos, dtype=np.float64),
        "target_ee_vel": None if target_ee_vel is None else np.asarray(target_ee_vel, dtype=np.float64),
    }


def test_defaults_are_velocity_gains_not_force_gains():
    cfg = CartesianVelocityConfig()
    assert cfg.kp_x == 2.0
    assert cfg.max_lin_speed_mps == 0.25


def test_requires_reset_before_compute():
    cfg = CartesianVelocityConfig()
    ctrl = CartesianVelocityController(cfg)
    try:
        ctrl.compute(_state([0.0, 0.0, 0.0]))
        assert False, "expected RuntimeError"
    except RuntimeError:
        pass


def test_at_rest_with_no_target_produces_zero_velocity():
    cfg = CartesianVelocityConfig()
    ctrl = CartesianVelocityController(cfg)
    p0 = [0.4, -0.2, 0.3]
    ctrl.reset_from_state(_state(p0))
    xd = ctrl.compute(_state(p0))
    np.testing.assert_allclose(xd, np.zeros(6), atol=1e-10)


def test_position_error_produces_proportional_restoring_velocity():
    cfg = CartesianVelocityConfig(kp_x=2.0, kp_y=2.0, kp_z=2.0, kp_rot=2.0, max_lin_speed_mps=10.0)
    ctrl = CartesianVelocityController(cfg)
    p0 = [0.0, 0.0, 0.0]
    ctrl.reset_from_state(_state(p0))
    # Actual position is behind target -> command should push toward target.
    xd = ctrl.compute(_state([0.0, 0.0, 0.0], target_ee_pos=[0.05, 0.0, 0.0]))
    assert xd[0] == pytest.approx(2.0 * 0.05)
    np.testing.assert_allclose(xd[1:], np.zeros(5), atol=1e-10)


def test_feedforward_velocity_is_added_to_p_correction():
    cfg = CartesianVelocityConfig(kp_x=2.0, max_lin_speed_mps=10.0)
    ctrl = CartesianVelocityController(cfg)
    p0 = [0.0, 0.0, 0.0]
    ctrl.reset_from_state(_state(p0))
    xd = ctrl.compute(_state(p0, target_ee_pos=p0, target_ee_vel=[0.03, 0.0, 0.0]))
    assert abs(xd[0] - 0.03) < 1e-9


def test_holds_y_z_at_reset_value_when_only_x_target_moves():
    cfg = CartesianVelocityConfig(kp_x=2.0, kp_y=2.0, kp_z=2.0, max_lin_speed_mps=10.0)
    ctrl = CartesianVelocityController(cfg)
    p0 = [0.0, -0.2, 0.3]
    ctrl.reset_from_state(_state(p0))
    # Robot has drifted off Y/Z; target_ee_pos still holds Y0/Z0 -> restoring command.
    drifted = [0.02, -0.19, 0.31]
    xd = ctrl.compute(_state(drifted, target_ee_pos=[0.02, p0[1], p0[2]]))
    assert xd[1] == pytest.approx(2.0 * (p0[1] - drifted[1]))
    assert xd[2] == pytest.approx(2.0 * (p0[2] - drifted[2]))


def test_linear_speed_is_clamped_to_configured_ceiling():
    cfg = CartesianVelocityConfig(kp_x=100.0, max_lin_speed_mps=0.1)
    ctrl = CartesianVelocityController(cfg)
    p0 = [0.0, 0.0, 0.0]
    ctrl.reset_from_state(_state(p0))
    xd = ctrl.compute(_state(p0, target_ee_pos=[1.0, 0.0, 0.0]))  # huge error
    lin_norm = float(np.linalg.norm(xd[:3]))
    assert lin_norm <= cfg.max_lin_speed_mps + 1e-9
    assert lin_norm == pytest.approx(cfg.max_lin_speed_mps, abs=1e-6)


def test_orientation_error_produces_proportional_angular_velocity():
    cfg = CartesianVelocityConfig(kp_rot=1.0, max_ang_speed_radps=10.0)
    ctrl = CartesianVelocityController(cfg)
    p0 = [0.0, 0.0, 0.0]
    ctrl.reset_from_state(_state(p0, ee_quat=[1.0, 0.0, 0.0, 0.0]))
    # Small rotation about Z away from the held reference.
    theta = 0.02
    quat_now = [np.cos(theta / 2), 0.0, 0.0, np.sin(theta / 2)]
    xd = ctrl.compute(_state(p0, ee_quat=quat_now, target_ee_pos=p0))
    assert xd[3] == pytest.approx(0.0, abs=1e-6)
    assert xd[4] == pytest.approx(0.0, abs=1e-6)
    assert abs(xd[5]) > 1e-6  # restoring angular velocity about Z


def test_angular_speed_is_clamped_to_configured_ceiling():
    cfg = CartesianVelocityConfig(kp_rot=1000.0, max_ang_speed_radps=0.2)
    ctrl = CartesianVelocityController(cfg)
    p0 = [0.0, 0.0, 0.0]
    ctrl.reset_from_state(_state(p0))
    theta = 2.0  # large rotation error
    quat_now = [np.cos(theta / 2), 0.0, 0.0, np.sin(theta / 2)]
    xd = ctrl.compute(_state(p0, ee_quat=quat_now, target_ee_pos=p0))
    ang_norm = float(np.linalg.norm(xd[3:]))
    assert ang_norm <= cfg.max_ang_speed_radps + 1e-9


def test_yaml_parsing_reads_velocity_control_block():
    ctrl_section = {
        "velocity_control": {
            "kp_x": 3.5,
            "kp_y": 1.5,
            "kp_z": 1.5,
            "kp_rot": 0.8,
            "max_lin_speed_mps": 0.4,
            "max_ang_speed_radps": 0.6,
        }
    }
    cfg = CartesianVelocityConfig.from_controller_yaml_section(ctrl_section)
    assert cfg.kp_x == 3.5
    assert cfg.kp_y == 1.5
    assert cfg.max_lin_speed_mps == 0.4
    assert cfg.max_ang_speed_radps == 0.6


def test_yaml_parsing_defaults_when_block_absent():
    cfg = CartesianVelocityConfig.from_controller_yaml_section({})
    assert cfg.kp_x == 2.0
    assert cfg.max_lin_speed_mps == 0.25
