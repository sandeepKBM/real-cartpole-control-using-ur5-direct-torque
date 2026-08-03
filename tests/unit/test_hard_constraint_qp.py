"""Tests for HardYConstraintQPController (2026-08-03) -- see
controller_core/hard_constraint_qp.py's module docstring for the full
design rationale (a genuine hard constraint on Y-axis acceleration, not
just another soft weighted cost).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from controller_core.hard_constraint_qp import (  # noqa: E402
    HardYConstraintQPConfig,
    HardYConstraintQPController,
)
from controller_core.torque_task_qp import TorqueTaskQPConfig, TorqueTaskQPController  # noqa: E402


def _state(q, x=0.0, y=0.0, z=0.5, target_x=0.0, mass_matrix=None, gravity=None):
    s = {
        "time": 0.0,
        "q": np.asarray(q, dtype=np.float64),
        "qd": np.zeros(6, dtype=np.float64),
        "ee_pos": np.array([x, y, z], dtype=np.float64),
        "ee_quat": np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
        "ee_lin_vel": np.zeros(3, dtype=np.float64),
        "ee_ang_vel": np.zeros(3, dtype=np.float64),
        "target_x": float(target_x),
        "target_x_vel": 0.0,
        "jacobian": np.eye(6, dtype=np.float64),
    }
    if mass_matrix is not None:
        s["mass_matrix"] = mass_matrix
    if gravity is not None:
        s["gravity_torque"] = gravity
    return s


def _base_kwargs():
    return dict(
        kp_x=400.0, kd_x=40.0, kp_y=80.0, kd_y=15.0, kp_z=120.0, kd_z=20.0,
        kp_rot=0.0, kd_rot=10.0, kp_posture=25.0, kd_posture=6.0, kd_joint=4.0,
        tau_max_nm=np.array([150.0, 150.0, 150.0, 28.0, 28.0, 28.0], dtype=np.float64),
    )


def test_defaults_off():
    cfg = HardYConstraintQPConfig(**_base_kwargs())
    assert cfg.hard_y_constraint is False


def test_flag_off_matches_plain_torque_qp_byte_identical():
    q0 = np.array([-0.7, -0.9, -1.2, -0.8, 0.2, 0.0])
    M = np.eye(6, dtype=np.float64) * 2.0
    g = np.array([0.0, -19.1, 6.4, -0.1, 0.02, 0.0])

    hard_cfg = HardYConstraintQPConfig(**_base_kwargs(), hard_y_constraint=False)
    plain_cfg = TorqueTaskQPConfig(**_base_kwargs())

    hard_ctrl = HardYConstraintQPController(hard_cfg)
    plain_ctrl = TorqueTaskQPController(plain_cfg)
    hard_ctrl.reset_from_state(_state(q0))
    plain_ctrl.reset_from_state(_state(q0))

    st = _state(q0, x=0.03, y=0.02, target_x=0.1, mass_matrix=M, gravity=g)
    out_hard = hard_ctrl.compute(st)
    out_plain = plain_ctrl.compute(st)
    np.testing.assert_allclose(out_hard.tau, out_plain.tau, atol=1e-10)


def test_hard_constraint_actually_binds_when_task_would_violate_it():
    """Construct a case where the soft Y task alone would push y_err (and
    thus a_y) well outside a tight tolerance -- with the hard constraint on,
    the resulting a_y must land inside the band regardless."""
    q0 = np.array([-0.7, -0.9, -1.2, -0.8, 0.2, 0.0])
    M = np.eye(6, dtype=np.float64) * 2.0
    g = np.zeros(6, dtype=np.float64)

    cfg = HardYConstraintQPConfig(
        **{**_base_kwargs(), "kp_y": 500.0},
        hard_y_constraint=True,
        hard_y_tolerance_mps2=0.01,
        # Isolate the hard-Y mechanism from the separate velocity-implied
        # torque-bound mechanism (pre-existing in TorqueTaskQPConfig) --
        # with qd=0 and q==q_rest in this synthetic test, that mechanism's
        # own bounds are tighter than tau_max_nm and would otherwise be the
        # actually-binding constraint, not what this test is checking.
        enforce_velocity_torque_bounds=False,
    )
    ctrl = HardYConstraintQPController(cfg)
    ctrl.reset_from_state(_state(q0, y=0.0))
    st = _state(q0, x=0.0, y=0.08, target_x=0.0, mass_matrix=M, gravity=g)  # large y_err -> large soft Fy
    out = ctrl.compute(st)

    m_inv = np.linalg.inv(M)
    j_y_minv = np.eye(6)[1, :] @ m_inv
    a_y = float(j_y_minv @ (out.tau_preclip - g))
    a_y_des = cfg.kp_y * (0.0 - 0.08)  # y0=0, y=0.08
    assert abs(a_y - a_y_des) <= cfg.hard_y_tolerance_mps2 + 1e-6


def test_hard_constraint_reports_infeasible_when_torque_limits_prevent_it():
    q0 = np.array([-0.7, -0.9, -1.2, -0.8, 0.2, 0.0])
    M = np.eye(6, dtype=np.float64) * 2.0
    g = np.zeros(6, dtype=np.float64)
    cfg = HardYConstraintQPConfig(
        **{**_base_kwargs(), "tau_max_nm": np.full(6, 0.001), "kp_y": 5000.0},  # near-zero torque budget
        hard_y_constraint=True, hard_y_tolerance_mps2=0.001,
    )
    ctrl = HardYConstraintQPController(cfg)
    ctrl.reset_from_state(_state(q0, y=0.0))
    st = _state(q0, x=0.0, y=0.5, target_x=0.0, mass_matrix=M, gravity=g)
    out = ctrl.compute(st)
    assert out.task_feasible is False


def test_yaml_parsing():
    ctrl_section = {
        "hard_y_constraint": True,
        "hard_y_tolerance_mps2": 0.02,
        "torque_limits_initial": {
            "shoulder_pan_joint": 150.0, "shoulder_lift_joint": 150.0, "elbow_joint": 150.0,
            "wrist_1_joint": 28.0, "wrist_2_joint": 28.0, "wrist_3_joint": 28.0,
        },
        "gains": {},
    }
    cfg = HardYConstraintQPConfig.from_controller_yaml_section(ctrl_section)
    assert cfg.hard_y_constraint is True
    assert cfg.hard_y_tolerance_mps2 == 0.02
