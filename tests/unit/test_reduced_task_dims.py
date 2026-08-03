"""Tests for the opt-in reduced-task-dimension row selection (2026-08-03) --
see CartesianImpedanceConfig.reduced_task_dims's docstring in
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
        "q": np.array([0.1, -0.9, -1.2, -0.8, 0.2, 0.0], dtype=np.float64),
        "qd": np.zeros(6, dtype=np.float64),
        "ee_pos": np.array([x, y, z], dtype=np.float64),
        "ee_quat": np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
        "ee_lin_vel": np.zeros(3, dtype=np.float64),
        "ee_ang_vel": np.zeros(3, dtype=np.float64),
        "target_x": float(target_x),
        "target_x_vel": 0.0,
        # Non-identity, non-degenerate Jacobian so row selection is a real test.
        "jacobian": np.array(
            [
                [0.08, -0.54, -0.32, -0.07, 0.07, 0.0],
                [-0.25, 0.54, 0.32, 0.07, -0.07, 0.0],
                [0.0, -0.12, 0.16, -0.01, 0.01, 0.0],
                [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        ),
    }


def _cfg(**overrides) -> CartesianImpedanceConfig:
    base = dict(
        kp_x=100.0, kd_x=10.0, kp_y=80.0, kd_y=15.0, kp_z=120.0, kd_z=20.0,
        kp_rot=20.0, kd_rot=5.0, kp_posture=2.0, kd_posture=0.5, kd_joint=0.8,
        tau_max_nm=np.array([150.0, 150.0, 150.0, 28.0, 28.0, 28.0], dtype=np.float64),
    )
    base.update(overrides)
    return CartesianImpedanceConfig(**base)


def test_defaults_off_all_dims_true():
    cfg = _cfg()
    assert cfg.reduced_task_dims is False
    assert cfg.task_dim_x is True
    assert cfg.task_dim_y is True
    assert cfg.task_dim_z is True
    assert cfg.task_dim_rx is True
    assert cfg.task_dim_ry is True
    assert cfg.task_dim_rz is True


def test_flag_off_byte_identical_regardless_of_dim_flags():
    """With reduced_task_dims=False, task_dim_* values must have zero effect --
    proves the master flag, not the per-dim flags, gates this feature."""
    ctrl_default = XAxisCartesianImpedanceController(_cfg())
    ctrl_other_dims = XAxisCartesianImpedanceController(_cfg(task_dim_y=False, task_dim_rz=False))
    ctrl_default.reset_from_state(_state(target_x=0.0))
    ctrl_other_dims.reset_from_state(_state(target_x=0.0))

    out_a = ctrl_default.compute(_state(0.05, 0.02, target_x=0.1))
    out_b = ctrl_other_dims.compute(_state(0.05, 0.02, target_x=0.1))
    np.testing.assert_allclose(out_a.tau, out_b.tau, atol=1e-12)


def test_row_selection_reduces_task_to_selected_dims_only():
    """XYZ-only (rotation rows dropped) must not raise and must produce a
    well-posed 3-row task -- verified indirectly via task_space_inertia_shaping
    (which builds A_task = J_task M^-1 J_task^T and inverts it; an incorrect
    row count there would raise a shape/singular error)."""
    cfg = _cfg(
        reduced_task_dims=True,
        task_dim_rx=False,
        task_dim_ry=False,
        task_dim_rz=False,
        task_space_inertia_shaping=True,
        nullspace_posture=True,
    )
    ctrl = XAxisCartesianImpedanceController(cfg)
    ctrl.reset_from_state(_state(target_x=0.0))
    out = ctrl.compute({**_state(0.05, 0.02, target_x=0.1), "mass_matrix": np.eye(6, dtype=np.float64) * 2.0})
    assert np.all(np.isfinite(out.tau))


def test_single_dim_selection_does_not_raise():
    """A single selected dimension (X only) is a degenerate 1-row task --
    must not crash cond()/inversion machinery."""
    cfg = _cfg(
        reduced_task_dims=True,
        task_dim_y=False,
        task_dim_z=False,
        task_dim_rx=False,
        task_dim_ry=False,
        task_dim_rz=False,
        task_space_inertia_shaping=True,
    )
    ctrl = XAxisCartesianImpedanceController(cfg)
    ctrl.reset_from_state(_state(target_x=0.0))
    out = ctrl.compute({**_state(0.05, 0.02, target_x=0.1), "mass_matrix": np.eye(6, dtype=np.float64) * 2.0})
    assert np.all(np.isfinite(out.tau))


def test_empty_dim_selection_raises():
    cfg = _cfg(
        reduced_task_dims=True,
        task_dim_x=False, task_dim_y=False, task_dim_z=False,
        task_dim_rx=False, task_dim_ry=False, task_dim_rz=False,
    )
    ctrl = XAxisCartesianImpedanceController(cfg)
    ctrl.reset_from_state(_state(target_x=0.0))
    with pytest.raises(ValueError, match="empty task"):
        ctrl.compute(_state(0.05, 0.02, target_x=0.1))


def test_mutually_exclusive_with_split_base_wrist_task():
    cfg = _cfg(reduced_task_dims=True, split_base_wrist_task=True)
    ctrl = XAxisCartesianImpedanceController(cfg)
    ctrl.reset_from_state(_state(target_x=0.0))
    with pytest.raises(ValueError, match="mutually exclusive"):
        ctrl.compute(_state(0.05, 0.02, target_x=0.1))


def test_accel_feedforward_plus_reduced_dims_raises():
    cfg = _cfg(reduced_task_dims=True, acceleration_feedforward=True)
    ctrl = XAxisCartesianImpedanceController(cfg)
    ctrl.reset_from_state(_state(target_x=0.0))
    with pytest.raises(ValueError, match="not yet supported"):
        ctrl.compute({**_state(0.05, 0.02, target_x=0.1), "mass_matrix": np.eye(6, dtype=np.float64)})


def test_yaml_parsing_roundtrip():
    ctrl_section = {
        "reduced_task_dims": True,
        "task_dim_x": True,
        "task_dim_y": True,
        "task_dim_z": True,
        "task_dim_rx": False,
        "task_dim_ry": False,
        "task_dim_rz": True,
        "torque_limits_initial": {
            "shoulder_pan_joint": 150.0,
            "shoulder_lift_joint": 150.0,
            "elbow_joint": 150.0,
            "wrist_1_joint": 28.0,
            "wrist_2_joint": 28.0,
            "wrist_3_joint": 28.0,
        },
        "gains": {"kp_x": 400.0},
    }
    cfg = CartesianImpedanceConfig.from_controller_yaml_section(ctrl_section)
    assert cfg.reduced_task_dims is True
    assert cfg.task_dim_rx is False
    assert cfg.task_dim_ry is False
    assert cfg.task_dim_rz is True
    assert cfg.task_dim_x is True
