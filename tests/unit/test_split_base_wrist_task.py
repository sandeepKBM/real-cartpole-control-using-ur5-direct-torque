"""Unit tests for `split_base_wrist_task` (controller_core/x_axis_cartesian_impedance.py).

Pure numpy -- no simulator required. See that flag's docstring for the full
step-1 numeric evidence and design rationale, and
docs/status/split_base_wrist_impedance_2026-08-01.md for the sim validation
this flag drove.
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


def _make_state(*, q=None, qd=None, ee_pos=(0.4, 0.1, 0.5), ee_quat=None, ee_ang_vel=None,
                 target_x=0.42, J=None, mass_matrix=None):
    state = {
        "time": 0.0,
        "q": np.zeros(6) if q is None else np.asarray(q, dtype=np.float64),
        "qd": np.zeros(6) if qd is None else np.asarray(qd, dtype=np.float64),
        "ee_pos": np.asarray(ee_pos, dtype=np.float64),
        "ee_quat": np.array([1.0, 0.0, 0.0, 0.0]) if ee_quat is None else np.asarray(ee_quat, dtype=np.float64),
        "ee_lin_vel": np.zeros(3),
        "ee_ang_vel": np.zeros(3) if ee_ang_vel is None else np.asarray(ee_ang_vel, dtype=np.float64),
        "target_x": float(target_x),
        "jacobian": np.eye(6) if J is None else np.asarray(J, dtype=np.float64),
    }
    if mass_matrix is not None:
        state["mass_matrix"] = np.asarray(mass_matrix, dtype=np.float64)
    return state


def _controller(**cfg_overrides) -> XAxisCartesianImpedanceController:
    cfg = CartesianImpedanceConfig(
        tau_max_nm=np.full(6, 1e6),  # keep clipping/backtracking out of the way
        **cfg_overrides,
    )
    ctl = XAxisCartesianImpedanceController(cfg)
    ctl.reset_from_state(_make_state(target_x=0.4))
    return ctl


# A representative near-singular Jacobian at the real failure pose:
# q = HEIGHT_ALPHA_0_5_Q = [0.0, -0.835398, -1.2, -0.985398, 0.0, 0.0]
# (hardware/poses.py), taken verbatim from the step-1 numeric check
# (docs/status/split_base_wrist_impedance_2026-08-01.md).
FAILURE_POSE_J = np.array(
    [
        [2.340000e-01, -7.648839e-01, -4.497193e-01, -9.927130e-02, 9.927130e-02, 0.000000e00],
        [-1.215331e-01, -0.000000e00, 0.000000e00, 0.000000e00, 1.387779e-17, 0.000000e00],
        [0.000000e00, -1.215331e-01, 1.635919e-01, -1.205028e-02, 1.205028e-02, 0.000000e00],
        [0.000000e00, 0.000000e00, 0.000000e00, 0.000000e00, -1.205028e-01, 0.000000e00],
        [0.000000e00, -1.000000e00, -1.000000e00, -1.000000e00, 0.000000e00, -1.000000e00],
        [1.000000e00, 0.000000e00, 0.000000e00, 0.000000e00, 9.927130e-01, 0.000000e00],
    ],
    dtype=np.float64,
)


def test_split_flag_off_by_default():
    cfg = CartesianImpedanceConfig()
    assert cfg.split_base_wrist_task is False


def test_split_flag_off_is_byte_identical_to_legacy():
    # Flag off (explicit False) must match a reference controller that never
    # even mentions the flag -- proves default-off is a true no-op, not just
    # a coincidentally-equal numeric result on this particular input.
    state = _make_state(
        q=np.full(6, 0.1), qd=np.full(6, 0.05), J=FAILURE_POSE_J,
        mass_matrix=np.diag([2.0, 3.0, 4.0, 1.0, 1.0, 1.0]),
    )
    gated_off = _controller(
        split_base_wrist_task=False,
        task_space_inertia_shaping=True,
        nullspace_posture=True,
        wrist_orientation_task=True,
        kp_rot_wrist=5.0,
        kd_rot_wrist=2.0,
    ).compute(state)
    reference = _controller(
        task_space_inertia_shaping=True,
        nullspace_posture=True,
        wrist_orientation_task=True,
        kp_rot_wrist=5.0,
        kd_rot_wrist=2.0,
    ).compute(state)
    np.testing.assert_allclose(gated_off.tau, reference.tau, atol=1e-12)
    np.testing.assert_allclose(gated_off.jacobian_cond, reference.jacobian_cond, atol=1e-6)
    assert gated_off.split_base_wrist_task_active is False
    assert reference.split_base_wrist_task_active is False


def test_split_on_reports_reduced_jacobian_cond():
    # At the real failure pose the full 6x6 J is numerically singular
    # (cond ~7e16) but the 3x3 position-rows x base-joint-cols block is
    # well conditioned (~7.8, computed directly below for the assertion).
    # jacobian_cond in the trace should reflect the latter once the flag is
    # on, not the former.
    state = _make_state(q=np.full(6, 0.1), J=FAILURE_POSE_J)
    out_off = _controller(split_base_wrist_task=False).compute(state)
    out_on = _controller(split_base_wrist_task=True).compute(_make_state(q=np.full(6, 0.1), J=FAILURE_POSE_J))

    expected_cond_base = float(np.linalg.cond(FAILURE_POSE_J[0:3, 0:3]))
    assert out_off.jacobian_cond > 1.0e10  # full J: numerically singular
    assert out_on.jacobian_cond == pytest.approx(expected_cond_base, rel=1e-9)
    assert out_on.jacobian_cond < 100.0  # well conditioned
    assert out_on.split_base_wrist_task_active is True


def test_split_task_never_routes_translation_through_wrist_columns():
    # J_task must be the position rows with ONLY the base-joint columns
    # nonzero -- direct construction check via a case with kp_rot=kd_rot=0
    # (so the wrench is purely translational) and full gains otherwise off,
    # isolating tau_task_nominal to exactly J_task.T @ [Fx, Fy, Fz].
    ctl = _controller(
        split_base_wrist_task=True,
        kp_x=10.0, kd_x=0.0, kp_y=10.0, kd_y=0.0, kp_z=10.0, kd_z=0.0,
        kp_rot=0.0, kd_rot=0.0, kp_posture=0.0, kd_posture=0.0, kd_joint=0.0,
    )
    state = _make_state(ee_pos=(0.3, 0.05, 0.4), target_x=0.4, J=FAILURE_POSE_J)
    out = ctl.compute(state)

    J_task_expected = np.zeros((3, 6))
    J_task_expected[:, 0:3] = FAILURE_POSE_J[0:3, 0:3]
    wrench_expected = out.wrench[0:3]  # Fx, Fy, Fz only; M dropped in split mode
    tau_expected = J_task_expected.T @ wrench_expected

    np.testing.assert_allclose(out.tau_task_nominal, tau_expected, atol=1e-9)
    # Structural guarantee: wrist joints (indices 3,4,5) get exactly zero
    # translation-task torque, regardless of J's wrist columns.
    np.testing.assert_allclose(out.tau_task_nominal[3:6], np.zeros(3), atol=1e-12)


def test_split_drops_rotational_wrench_from_task_pipeline():
    # Zero translation error (Fx=Fy=Fz=0) but a nonzero angular velocity so
    # M = -kd_rot*omega is nonzero. With the flag on, tau_task_nominal must
    # be exactly zero (M is dropped); with the flag off, the identity
    # Jacobian routes M straight through and tau_task_nominal is nonzero.
    omega = np.array([0.1, -0.2, 0.05])
    common = dict(
        kp_x=10.0, kd_x=5.0, kp_y=10.0, kd_y=5.0, kp_z=10.0, kd_z=5.0,
        kp_rot=0.0, kd_rot=8.0, kp_posture=0.0, kd_posture=0.0, kd_joint=0.0,
    )
    state = _make_state(ee_pos=(0.4, 0.1, 0.5), target_x=0.4, J=np.eye(6), ee_ang_vel=omega)

    out_off = _controller(split_base_wrist_task=False, **common).compute(state)
    out_on = _controller(split_base_wrist_task=True, **common).compute(state)

    assert np.linalg.norm(out_off.tau_task_nominal) > 1e-6
    np.testing.assert_allclose(out_on.tau_task_nominal, np.zeros(6), atol=1e-12)
    # The diagnostic wrench field still reports the would-be M (unaffected --
    # only the task pipeline drops it, not the trace).
    np.testing.assert_allclose(out_on.wrench[3:6], out_off.wrench[3:6], atol=1e-12)


def test_split_nullspace_posture_uses_reduced_task_and_stays_finite():
    # nullspace_posture on top of split_base_wrist_task, evaluated at the
    # real near-singular failure-pose Jacobian, must produce a finite,
    # well-behaved result (no NaN/inf) -- the whole point of routing the
    # nullspace math through the well-conditioned 3x3 reduced task instead
    # of the singular full 6x6 one.
    ctl = _controller(
        split_base_wrist_task=True,
        nullspace_posture=True,
        task_space_inertia_shaping=True,
        lambda_regularization=0.1,
        kp_posture=5.0, kd_posture=1.0,
    )
    state = _make_state(
        q=np.full(6, 0.05), qd=np.full(6, 0.01), J=FAILURE_POSE_J,
        mass_matrix=np.eye(6), target_x=0.42,
    )
    out = ctl.compute(state)

    assert np.all(np.isfinite(out.tau))
    assert np.all(np.isfinite(out.tau_posture))
    assert out.nullspace_posture_active is True
    assert out.split_base_wrist_task_active is True


def test_split_base_wrist_task_yaml_parsing():
    torque_limits = {
        name: 100.0
        for name in (
            "shoulder_pan_joint",
            "shoulder_lift_joint",
            "elbow_joint",
            "wrist_1_joint",
            "wrist_2_joint",
            "wrist_3_joint",
        )
    }
    on_section = {
        "torque_limits_mode": "initial",
        "torque_limits_initial": torque_limits,
        "split_base_wrist_task": True,
    }
    off_section = {
        "torque_limits_mode": "initial",
        "torque_limits_initial": torque_limits,
    }
    cfg_on = CartesianImpedanceConfig.from_controller_yaml_section(on_section)
    cfg_off = CartesianImpedanceConfig.from_controller_yaml_section(off_section)
    assert cfg_on.split_base_wrist_task is True
    assert cfg_off.split_base_wrist_task is False
