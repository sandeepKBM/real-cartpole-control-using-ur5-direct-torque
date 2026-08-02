"""Unit tests for `acceleration_feedforward` (controller_core/x_axis_cartesian_impedance.py).

Pure numpy -- no simulator required. See that flag's docstring for the design
rationale and docs/status/acceleration_feedforward_2026-08-01.md for the sim
validation this flag drove.
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
                 target_x=0.42, target_x_accel=None, J=None, mass_matrix=None):
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
    if target_x_accel is not None:
        state["target_x_accel"] = float(target_x_accel)
    return state


def _controller(**cfg_overrides) -> XAxisCartesianImpedanceController:
    cfg = CartesianImpedanceConfig(
        tau_max_nm=np.full(6, 1e6),  # keep clipping/backtracking out of the way
        **cfg_overrides,
    )
    ctl = XAxisCartesianImpedanceController(cfg)
    ctl.reset_from_state(_make_state(target_x=0.4))
    return ctl


def test_acceleration_feedforward_off_by_default():
    cfg = CartesianImpedanceConfig()
    assert cfg.acceleration_feedforward is False


def test_flag_off_is_byte_identical_regardless_of_target_x_accel():
    # A nonzero target_x_accel in the state must have ZERO effect when the
    # flag is off -- proves the flag actually gates the behavior, not just
    # that a zero-by-default target_x_accel happens to produce zero effect.
    state_with_accel = _make_state(
        q=np.full(6, 0.1), qd=np.full(6, 0.05),
        mass_matrix=np.diag([2.0, 3.0, 4.0, 1.0, 1.0, 1.0]),
        target_x_accel=3.7,
    )
    state_without_accel = _make_state(
        q=np.full(6, 0.1), qd=np.full(6, 0.05),
        mass_matrix=np.diag([2.0, 3.0, 4.0, 1.0, 1.0, 1.0]),
    )
    out_with = _controller(acceleration_feedforward=False).compute(state_with_accel)
    out_without = _controller(acceleration_feedforward=False).compute(state_without_accel)
    np.testing.assert_allclose(out_with.tau, out_without.tau, atol=1e-14)
    assert out_with.acceleration_feedforward_active is False
    assert out_without.acceleration_feedforward_active is False


def test_flag_off_is_byte_identical_to_reference_controller_that_never_mentions_it():
    state = _make_state(
        q=np.full(6, 0.1), qd=np.full(6, 0.05),
        mass_matrix=np.diag([2.0, 3.0, 4.0, 1.0, 1.0, 1.0]),
        target_x_accel=1.5,
    )
    gated_off = _controller(
        acceleration_feedforward=False,
        task_space_inertia_shaping=True,
    ).compute(state)
    reference = _controller(
        task_space_inertia_shaping=True,
    ).compute(state)
    np.testing.assert_allclose(gated_off.tau, reference.tau, atol=1e-14)


def test_flag_on_scales_with_and_points_toward_commanded_acceleration():
    # Identity Jacobian + identity mass matrix -> Lambda ~= (1/(1+eps)) * I,
    # a simple, known effective mass -- isolates the feedforward term's sign
    # and monotonicity without any Jacobian/mass-matrix complexity.
    common = dict(
        acceleration_feedforward=True,
        kp_x=0.0, kd_x=0.0, kp_y=0.0, kd_y=0.0, kp_z=0.0, kd_z=0.0,
        kp_rot=0.0, kd_rot=0.0, kp_posture=0.0, kd_posture=0.0, kd_joint=0.0,
        lambda_regularization=1e-6,
    )
    state_zero = _make_state(J=np.eye(6), mass_matrix=np.eye(6), target_x_accel=0.0)
    state_pos = _make_state(J=np.eye(6), mass_matrix=np.eye(6), target_x_accel=2.0)
    state_neg = _make_state(J=np.eye(6), mass_matrix=np.eye(6), target_x_accel=-2.0)
    state_pos_big = _make_state(J=np.eye(6), mass_matrix=np.eye(6), target_x_accel=4.0)

    out_zero = _controller(**common).compute(state_zero)
    out_pos = _controller(**common).compute(state_pos)
    out_neg = _controller(**common).compute(state_neg)
    out_pos_big = _controller(**common).compute(state_pos_big)

    assert out_zero.acceleration_feedforward_active is True
    assert out_zero.tau[0] == pytest.approx(0.0, abs=1e-9)
    # Positive commanded acceleration -> positive X torque; negative -> negative.
    assert out_pos.tau[0] > 1e-6
    assert out_neg.tau[0] < -1e-6
    assert out_neg.tau[0] == pytest.approx(-out_pos.tau[0], rel=1e-9)
    # Monotonic scaling with the magnitude of the commanded acceleration.
    assert out_pos_big.tau[0] > out_pos.tau[0]
    # wrench_accel_ff reports the same feedforward contribution directly.
    assert out_pos.wrench_accel_ff[0] > 0.0
    assert out_pos.wrench_accel_ff[0] == pytest.approx(0.5 * out_pos_big.wrench_accel_ff[0], rel=1e-9)


def test_flag_on_without_mass_matrix_is_a_graceful_noop():
    # acceleration_feedforward=True but the state carries no mass_matrix at
    # all -- must NOT silently fall back to an identity-matrix effective
    # mass; must be a documented no-op instead (see the flag's own docstring).
    state_with_accel = _make_state(target_x_accel=5.0)  # no mass_matrix key
    state_without_accel = _make_state()
    out_with = _controller(acceleration_feedforward=True).compute(state_with_accel)
    out_without = _controller(acceleration_feedforward=True).compute(state_without_accel)
    np.testing.assert_allclose(out_with.tau, out_without.tau, atol=1e-14)
    assert out_with.acceleration_feedforward_active is False
    np.testing.assert_allclose(out_with.wrench_accel_ff, np.zeros(3), atol=1e-14)


def test_acceleration_feedforward_yaml_parsing():
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
        "acceleration_feedforward": True,
    }
    off_section = {
        "torque_limits_mode": "initial",
        "torque_limits_initial": torque_limits,
    }
    cfg_on = CartesianImpedanceConfig.from_controller_yaml_section(on_section)
    cfg_off = CartesianImpedanceConfig.from_controller_yaml_section(off_section)
    assert cfg_on.acceleration_feedforward is True
    assert cfg_off.acceleration_feedforward is False


def test_flag_on_with_task_space_inertia_shaping_does_not_double_scale():
    # Regression test for a real bug found 2026-08-02 (code review of the
    # 2026-08-01 session's diff): with task_space_inertia_shaping=True, the
    # wrench-shaping step at the bottom of compute() multiplies the ENTIRE
    # wrench_task by Lambda once, uniformly. wrench_task at that point in the
    # pipeline represents a desired task ACCELERATION (see the wrench-shaping
    # step's own comment), not a force -- so the feedforward term must add
    # raw target_x_accel there, not an already-Lambda-scaled force, or it
    # gets Lambda-scaled a second time (effective Lambda^2). This was live in
    # config/ur5e_mujoco_torque_osc_tuned_split_base_wrist_accel_ff.yaml,
    # which sets both flags together -- never previously exercised by any
    # test in this file (every prior test here defaults
    # task_space_inertia_shaping=False).
    #
    # J=I with a NON-identity mass matrix (M_x=4.0) so Lambda_diag[0] ~= 4.0,
    # clearly different from 1.0 -- deliberately not M=I, since at Lambda~=1
    # a single-vs-double scaling bug is numerically almost invisible. With
    # every other gain zeroed, a correct implementation must produce the
    # SAME tau[0] whether the feedforward's Lambda-scaling happens inside the
    # feedforward term itself (shaping off) or once via the downstream
    # shaping step (shaping on) -- shaping doesn't change what Lambda IS,
    # only how/when it's applied. The old, buggy code applied Lambda TWICE
    # in the shaping-on case, so old_ratio ~= Lambda_diag[0] ~= 4.0 instead
    # of 1.0 -- a large, easily-caught discrepancy at this Lambda value.
    common = dict(
        acceleration_feedforward=True,
        kp_x=0.0, kd_x=0.0, kp_y=0.0, kd_y=0.0, kp_z=0.0, kd_z=0.0,
        kp_rot=0.0, kd_rot=0.0, kp_posture=0.0, kd_posture=0.0, kd_joint=0.0,
        lambda_regularization=1e-6,
    )
    mass_matrix = np.diag([4.0, 4.0, 4.0, 1.0, 1.0, 1.0])
    state = _make_state(J=np.eye(6), mass_matrix=mass_matrix, target_x_accel=2.0)

    out_shaping_off = _controller(task_space_inertia_shaping=False, **common).compute(state)
    out_shaping_on = _controller(task_space_inertia_shaping=True, **common).compute(state)

    assert out_shaping_off.acceleration_feedforward_active is True
    assert out_shaping_on.acceleration_feedforward_active is True
    assert out_shaping_off.tau[0] > 1e-6
    ratio = out_shaping_on.tau[0] / out_shaping_off.tau[0]
    # Fixed code: ratio ~= 1.0. The old bug would have given ratio ~= 4.0.
    assert ratio == pytest.approx(1.0, rel=1e-3)
