"""Tests for the opt-in hard per-joint task exclusion (2026-08-03) -- see
CartesianImpedanceConfig.task_lock_shoulder_pan's docstring in
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


def _state(q, x=0.0, y=0.0, z=0.5, target_x=0.1):
    return {
        "time": 0.0,
        "q": np.asarray(q, dtype=np.float64),
        "qd": np.zeros(6, dtype=np.float64),
        "ee_pos": np.array([x, y, z], dtype=np.float64),
        "ee_quat": np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
        "ee_lin_vel": np.zeros(3, dtype=np.float64),
        "ee_ang_vel": np.zeros(3, dtype=np.float64),
        "target_x": float(target_x),
        "target_x_vel": 0.0,
        # Non-degenerate Jacobian with real coupling into column 0 (shoulder_pan).
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
        kp_rot=20.0, kd_rot=5.0, kp_posture=0.0, kd_posture=0.0, kd_joint=0.0,
        tau_max_nm=np.array([150.0, 150.0, 150.0, 28.0, 28.0, 28.0], dtype=np.float64),
    )
    base.update(overrides)
    return CartesianImpedanceConfig(**base)


def test_defaults_all_unlocked():
    cfg = _cfg()
    assert cfg.task_lock_shoulder_pan is False
    assert cfg.task_lock_wrist_2 is False


def test_flag_off_byte_identical():
    q0 = np.array([-0.7, -0.8, -1.2, -1.0, 0.2, 0.0])
    ctrl_a = XAxisCartesianImpedanceController(_cfg())
    ctrl_b = XAxisCartesianImpedanceController(
        _cfg(task_lock_shoulder_pan=False, task_lock_wrist_2=False)
    )
    ctrl_a.reset_from_state(_state(q0))
    ctrl_b.reset_from_state(_state(q0))
    out_a = ctrl_a.compute(_state(q0, x=0.02))
    out_b = ctrl_b.compute(_state(q0, x=0.02))
    np.testing.assert_allclose(out_a.tau, out_b.tau, atol=1e-12)


def test_locked_joint_gets_exactly_zero_task_torque():
    """With shoulder_pan locked, tau_task_nominal[0] must be exactly 0 --
    not approximately small -- regardless of how large the task wrench is."""
    q0 = np.array([-0.7, -0.8, -1.2, -1.0, 0.2, 0.0])
    cfg = _cfg(task_lock_shoulder_pan=True, kp_x=1000.0)  # large gain to stress-test
    ctrl = XAxisCartesianImpedanceController(cfg)
    ctrl.reset_from_state(_state(q0))
    out = ctrl.compute(_state(q0, x=0.05, target_x=0.15))  # large x_err
    assert out.tau_task_nominal[0] == 0.0


def test_locking_against_full_6d_task_is_infeasible_and_safely_suppressed():
    """Locking a joint against the FULL 6D task (all 6 dims active) leaves a
    6x6 J_task with one zeroed column -- exactly singular by construction
    (a 6D task has no solution with only 5 actuated directions). The
    PRE-EXISTING jacobian_singular_cond_max guard must catch this and
    suppress the wrench (singular_scale -> 0) rather than blow up -- this
    is correct, safe behavior, not a bug. Locking must be paired with
    reduced_task_dims reducing the task to <= the number of free joints;
    see test_composes_with_reduced_task_dims for the well-posed case."""
    q0 = np.array([-0.7, -0.8, -1.2, -1.0, 0.2, 0.0])
    cfg = _cfg(task_lock_shoulder_pan=True)
    ctrl = XAxisCartesianImpedanceController(cfg)
    ctrl.reset_from_state(_state(q0))
    out = ctrl.compute(_state(q0, x=0.05, target_x=0.15))
    assert out.singular_scale < 1e-6
    assert np.allclose(out.tau_task_nominal, 0.0)


def test_unlocked_joints_still_receive_task_torque_when_well_posed():
    """Same lock, but paired with reduced_task_dims (4D task: X,Y,Z,rz) so
    5 free columns cover a 4D task -- well-posed, no singularity guard.
    Uses a deliberately-constructed, verified-well-conditioned Jacobian
    (cond=1.41 after locking+reducing) where X is driven by joints 0 AND 1
    jointly and rz by joints 0, 4, 5 jointly -- so locking joint 0 leaves a
    genuinely solvable 4D task, unlike the coupled Jacobian in _state()
    (kept for other tests), which happened to make rows 0/1 exact negatives
    of each other once column 0 is zeroed -- a coincidence of that specific
    hand-picked data, not a real conditioning concern with the locking
    mechanism itself; see the git history around this test for the
    debugging trail that found it."""
    q0 = np.array([-0.7, -0.8, -1.2, -1.0, 0.2, 0.0])
    well_posed_J = np.array(
        [
            [1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            [0.5, 0.0, 0.0, 0.0, 0.5, 0.5],
        ],
        dtype=np.float64,
    )
    state = _state(q0, x=0.05, target_x=0.15)
    state["jacobian"] = well_posed_J
    reset_state = _state(q0)
    reset_state["jacobian"] = well_posed_J

    cfg = _cfg(
        task_lock_shoulder_pan=True,
        reduced_task_dims=True,
        task_dim_rx=False,
        task_dim_ry=False,
    )
    ctrl = XAxisCartesianImpedanceController(cfg)
    ctrl.reset_from_state(reset_state)
    out = ctrl.compute(state)
    assert out.tau_task_nominal[0] == 0.0
    assert not np.allclose(out.tau_task_nominal[1:], 0.0)  # other joints do the work


def test_multiple_locks_compose():
    q0 = np.array([-0.7, -0.8, -1.2, -1.0, 0.2, 0.0])
    cfg = _cfg(task_lock_shoulder_pan=True, task_lock_wrist_2=True)
    ctrl = XAxisCartesianImpedanceController(cfg)
    ctrl.reset_from_state(_state(q0))
    out = ctrl.compute(_state(q0, x=0.05, target_x=0.15))
    assert out.tau_task_nominal[0] == 0.0
    assert out.tau_task_nominal[4] == 0.0


def test_composes_with_reduced_task_dims():
    q0 = np.array([-0.7, -0.8, -1.2, -1.0, 0.2, 0.0])
    cfg = _cfg(
        task_lock_shoulder_pan=True,
        reduced_task_dims=True,
        task_dim_rx=False,
        task_dim_ry=False,
        task_space_inertia_shaping=True,
    )
    ctrl = XAxisCartesianImpedanceController(cfg)
    ctrl.reset_from_state(_state(q0))
    out = ctrl.compute({**_state(q0, x=0.05, target_x=0.15), "mass_matrix": np.eye(6, dtype=np.float64) * 2.0})
    assert out.tau_task_nominal[0] == 0.0
    assert np.all(np.isfinite(out.tau))


def test_yaml_parsing():
    ctrl_section = {
        "task_lock_shoulder_pan": True,
        "task_lock_wrist_2": True,
        "torque_limits_initial": {
            "shoulder_pan_joint": 150.0, "shoulder_lift_joint": 150.0, "elbow_joint": 150.0,
            "wrist_1_joint": 28.0, "wrist_2_joint": 28.0, "wrist_3_joint": 28.0,
        },
        "gains": {},
    }
    cfg = CartesianImpedanceConfig.from_controller_yaml_section(ctrl_section)
    assert cfg.task_lock_shoulder_pan is True
    assert cfg.task_lock_wrist_2 is True
    assert cfg.task_lock_shoulder_lift is False
