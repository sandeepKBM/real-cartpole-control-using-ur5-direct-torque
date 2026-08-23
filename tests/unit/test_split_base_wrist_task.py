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


# A synthetic, well-conditioned constant Jacobian for the closed-loop test
# below (cond of its position rows is ~2.0 over the base columns and ~6.5 over
# the alternate set used here, so a 3-joint solution to the 3-row position task
# exists for both). FAILURE_POSE_J deliberately cannot be used there: at that
# real pose (wrist_2 == 0) the position-row block is exactly singular for BOTH
# alternate sets, so it can prove column ROUTING but not convergence. These
# tests exercise the column-selection MECHANISM; which joint sets are
# kinematically usable on the real arm is a separate, pose-dependent question
# (see split_base_wrist_active_joints' docstring).
_LINEAR_RNG = np.random.default_rng(20260812)
LINEAR_J = np.asarray(_LINEAR_RNG.normal(size=(6, 6)), dtype=np.float64) * 0.25 + np.eye(6) * 0.6
LINEAR_P0 = np.array([0.35, -0.12, 0.48], dtype=np.float64)

# The historical hardcoded active set, spelled out so the "explicit == default"
# test below is readable.
BASE_JOINTS = (0, 1, 2)
# The alternate active set used throughout the tests below: elbow, wrist_1,
# wrist_2 drive the task; shoulder_pan, shoulder_lift, wrist_3 are held. Chosen
# (2026-08-12) because it is the best-conditioned 3-joint set on the real UR5e
# that excludes shoulder_pan -- BUT only away from wrist_2 == 0, where it is
# exactly singular. See split_base_wrist_active_joints' docstring for the full
# sweep, including why the superficially obvious (1, 2, 3) =
# shoulder_lift/elbow/wrist_1 is singular at EVERY pose (three parallel axes).
ELBOW_WRIST1_WRIST2 = (2, 3, 4)
HELD_BY_ELBOW_WRIST1_WRIST2 = (0, 1, 5)


def _linear_state(q, qd, target_x):
    """State for the constant-Jacobian pseudo-plant ``ee_pos = P0 + J[0:3] @ q``."""
    q = np.asarray(q, dtype=np.float64)
    qd = np.asarray(qd, dtype=np.float64)
    return {
        "time": 0.0,
        "q": q.copy(),
        "qd": qd.copy(),
        "ee_pos": LINEAR_P0 + LINEAR_J[0:3, :] @ q,
        "ee_quat": np.array([1.0, 0.0, 0.0, 0.0]),
        "ee_lin_vel": LINEAR_J[0:3, :] @ qd,
        "ee_ang_vel": np.zeros(3),
        "jacobian": LINEAR_J.copy(),
        "target_x": float(target_x),
    }


def _linear_rollout(active_joints, *, steps=2000, dt=0.002, dx=0.05, **cfg_overrides):
    """Deterministic closed loop: unit-mass joints, qdd = tau, no gravity.

    Returns ``(q0, q_final, x_error_final, max_abs_tau_task_per_joint)``.
    ``active_joints=None`` leaves ``split_base_wrist_active_joints`` unset
    (i.e. exercises the substituted historical default).
    """
    kwargs = {} if active_joints is None else {"split_base_wrist_active_joints": active_joints}
    kwargs.update(cfg_overrides)
    cfg = CartesianImpedanceConfig(
        tau_max_nm=np.full(6, 1e6),
        split_base_wrist_task=True,
        kp_x=400.0, kd_x=40.0, kp_y=80.0, kd_y=15.0, kp_z=120.0, kd_z=20.0,
        kp_rot=0.0, kd_rot=0.0,
        kp_posture=1.0, kd_posture=0.5, kd_joint=1.0,
        **kwargs,
    )
    ctl = XAxisCartesianImpedanceController(cfg)
    q = np.array([0.1, -1.2, 1.4, -1.7, -1.55, 0.3], dtype=np.float64)
    q0 = q.copy()
    qd = np.zeros(6, dtype=np.float64)
    st0 = _linear_state(q, qd, 0.0)
    ctl.reset_from_state(st0)
    target = float(st0["ee_pos"][0]) + dx
    max_abs_tau_task = np.zeros(6, dtype=np.float64)
    taus = []
    for _ in range(steps):
        out = ctl.compute(_linear_state(q, qd, target))
        max_abs_tau_task = np.maximum(max_abs_tau_task, np.abs(out.tau_task_nominal))
        taus.append(out.tau.copy())
        qd = qd + dt * out.tau  # unit mass matrix: qdd == tau
        q = q + dt * qd
    x_final = float((LINEAR_P0 + LINEAR_J[0:3, :] @ q)[0])
    return q0, q, x_final - target, max_abs_tau_task, np.asarray(taus)


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


# ---------------------------------------------------------------------------
# split_base_wrist_active_joints (2026-08-12): the active joint set the split
# task drives is configurable, defaulting to the historical hardcoded
# (0, 1, 2). Everything below is new; everything above must keep passing
# unchanged, which is itself part of the default-preservation proof.
# ---------------------------------------------------------------------------


def test_split_active_joints_defaults_to_none():
    # None (not (0, 1, 2)) is the default, so "unset" is distinguishable from
    # "explicitly asked for the base joints" at the config level.
    assert CartesianImpedanceConfig().split_base_wrist_active_joints is None


def test_split_active_joints_explicit_base_set_is_identical_to_default():
    # The single most important guarantee: the substituted default must be
    # the historical hardcoded slice, exactly. Compared over a full closed-loop
    # rollout (not one cycle), bit-for-bit -- a per-cycle difference of any
    # size would diverge here.
    _, q_default, err_default, tau_task_default, taus_default = _linear_rollout(None)
    _, q_explicit, err_explicit, tau_task_explicit, taus_explicit = _linear_rollout(BASE_JOINTS)
    assert np.array_equal(taus_default, taus_explicit)
    assert np.array_equal(q_default, q_explicit)
    assert np.array_equal(tau_task_default, tau_task_explicit)
    assert err_default == err_explicit


def test_split_active_joints_reports_cond_of_the_selected_columns():
    # jacobian_cond must track the SELECTED 3x3 position-rows block, not the
    # base-joint one -- that value feeds singular_scale and the adaptive-eps
    # schedules, so reporting the wrong block would mis-scale them.
    state = _make_state(q=np.full(6, 0.1), J=LINEAR_J)
    out_base = _controller(split_base_wrist_task=True).compute(state)
    out_alt = _controller(
        split_base_wrist_task=True,
        split_base_wrist_active_joints=ELBOW_WRIST1_WRIST2,
    ).compute(state)

    cond_base = float(np.linalg.cond(LINEAR_J[0:3, list(BASE_JOINTS)]))
    cond_alt = float(np.linalg.cond(LINEAR_J[0:3, list(ELBOW_WRIST1_WRIST2)]))
    assert cond_base != pytest.approx(cond_alt)  # the two blocks really differ
    assert out_base.jacobian_cond == pytest.approx(cond_base, rel=1e-9)
    assert out_alt.jacobian_cond == pytest.approx(cond_alt, rel=1e-9)
    assert out_base.split_base_wrist_active_joints == BASE_JOINTS
    assert out_alt.split_base_wrist_active_joints == ELBOW_WRIST1_WRIST2


def test_split_active_joints_routes_task_only_through_selected_columns():
    # Same direct-construction check as
    # test_split_task_never_routes_translation_through_wrist_columns, but for
    # the (2, 3, 4) set: the task force must land in columns 2/3/4 and be
    # exactly zero on 0/1/5 -- including shoulder_pan, which the hardcoded
    # version always drove. Uses LINEAR_J, not FAILURE_POSE_J: the (2, 3, 4)
    # block of the latter is exactly singular (wrist_2 == 0 there), so the pre-existing
    # singular_scale term would zero the whole task wrench and the routing
    # question would be untestable there (that zeroing is correct behavior,
    # just not what this test is about).
    ctl = _controller(
        split_base_wrist_task=True,
        split_base_wrist_active_joints=ELBOW_WRIST1_WRIST2,
        kp_x=10.0, kd_x=0.0, kp_y=10.0, kd_y=0.0, kp_z=10.0, kd_z=0.0,
        kp_rot=0.0, kd_rot=0.0, kp_posture=0.0, kd_posture=0.0, kd_joint=0.0,
    )
    state = _make_state(ee_pos=(0.3, 0.05, 0.4), target_x=0.4, J=LINEAR_J)
    out = ctl.compute(state)

    J_task_expected = np.zeros((3, 6))
    J_task_expected[:, list(ELBOW_WRIST1_WRIST2)] = LINEAR_J[0:3, list(ELBOW_WRIST1_WRIST2)]
    tau_expected = J_task_expected.T @ out.wrench[0:3]

    np.testing.assert_allclose(out.tau_task_nominal, tau_expected, atol=1e-9)
    for joint in HELD_BY_ELBOW_WRIST1_WRIST2:
        assert out.tau_task_nominal[joint] == 0.0
    # ...and the selected set really is carrying the task (not trivially zero).
    assert np.linalg.norm(out.tau_task_nominal[list(ELBOW_WRIST1_WRIST2)]) > 1e-6
    # Sanity contrast: the default set would have driven shoulder_pan here.
    out_base = _controller(
        split_base_wrist_task=True,
        kp_x=10.0, kd_x=0.0, kp_y=10.0, kd_y=0.0, kp_z=10.0, kd_z=0.0,
        kp_rot=0.0, kd_rot=0.0, kp_posture=0.0, kd_posture=0.0, kd_joint=0.0,
    ).compute(_make_state(ee_pos=(0.3, 0.05, 0.4), target_x=0.4, J=LINEAR_J))
    assert abs(out_base.tau_task_nominal[0]) > 1e-6
    assert out_base.tau_task_nominal[3] == 0.0


def test_split_active_joints_held_joints_are_held_by_posture_not_task():
    # shoulder_pan/shoulder_lift/wrist_3 get zero task torque but a real posture
    # restoring torque when displaced from q_rest -- the "held by the existing
    # nullspace_posture joint-space spring" half of the mechanism.
    ctl = _controller(
        split_base_wrist_task=True,
        split_base_wrist_active_joints=ELBOW_WRIST1_WRIST2,
        kp_posture=25.0, kd_posture=6.0, kd_joint=0.0,
        kp_rot=0.0, kd_rot=0.0,
    )
    displaced = np.array([0.05, 0.0, 0.0, 0.0, -0.04, 0.03])
    out = ctl.compute(_make_state(q=displaced, ee_pos=(0.3, 0.05, 0.4), J=LINEAR_J))

    # The task is genuinely live on the active joints this cycle...
    assert np.linalg.norm(out.tau_task_nominal[list(ELBOW_WRIST1_WRIST2)]) > 1e-6
    # ...while the held joints see posture only.
    for joint in HELD_BY_ELBOW_WRIST1_WRIST2:
        assert out.tau_task_nominal[joint] == 0.0
        # Posture is the ONLY thing acting there, and it pulls back toward
        # q_rest (== zeros, captured by _controller's reset_from_state).
        assert out.tau_posture[joint] == pytest.approx(-25.0 * displaced[joint])
        assert out.tau[joint] == pytest.approx(out.tau_posture[joint])


def test_split_active_joints_nullspace_posture_stays_finite():
    # Same guarantee test_split_nullspace_posture_uses_reduced_task_and_stays_
    # finite makes for the base set, for the (2, 3, 4) set at the same
    # near-singular pose (its selected 3x3 block is itself singular there --
    # see LINEAR_J's comment -- so this is the harshest available input).
    out = _controller(
        split_base_wrist_task=True,
        split_base_wrist_active_joints=ELBOW_WRIST1_WRIST2,
        nullspace_posture=True,
        task_space_inertia_shaping=True,
        lambda_regularization=0.1,
        kp_posture=5.0, kd_posture=1.0,
    ).compute(
        _make_state(
            q=np.full(6, 0.05), qd=np.full(6, 0.01), J=FAILURE_POSE_J,
            mass_matrix=np.eye(6), target_x=0.42,
        )
    )
    assert np.all(np.isfinite(out.tau))
    assert np.all(np.isfinite(out.tau_posture))
    assert out.split_base_wrist_active_joints == ELBOW_WRIST1_WRIST2


def test_split_active_joints_closed_loop_converges_using_only_active_joints():
    # Closed-loop proof for the case this generalization was built for.
    # Preconditions made explicit: the selected 3x3 block is well conditioned
    # (so a 3-joint solution to the 3-row position task exists at all).
    assert np.linalg.cond(LINEAR_J[0:3, list(ELBOW_WRIST1_WRIST2)]) < 10.0

    q0, q_final, x_err, max_abs_tau_task, _ = _linear_rollout(ELBOW_WRIST1_WRIST2)

    # Converged onto the commanded 0.05 m task-axis displacement.
    assert abs(x_err) < 0.05 * 0.1
    # Only joints 1/2/3 ever saw task torque...
    for joint in HELD_BY_ELBOW_WRIST1_WRIST2:
        assert max_abs_tau_task[joint] == 0.0
    assert np.all(max_abs_tau_task[list(ELBOW_WRIST1_WRIST2)] > 1e-6)
    # ...and with no gravity and no posture error on the held joints, they
    # receive exactly zero total torque, so they never move at all: the
    # displacement was produced by elbow/wrist_1/wrist_2 alone.
    for joint in HELD_BY_ELBOW_WRIST1_WRIST2:
        assert q_final[joint] == q0[joint]
    assert np.all(np.abs(q_final[list(ELBOW_WRIST1_WRIST2)] - q0[list(ELBOW_WRIST1_WRIST2)]) > 1e-3)


@pytest.mark.parametrize(
    "bad",
    [
        (0, 1),  # too few for a 3-row position task
        (0, 1, 2, 3),  # too many
        (0, 1, 6),  # out of range high
        (-1, 1, 2),  # out of range low
        (1, 1, 2),  # duplicate joint
        (0, 1, True),  # bool masquerading as joint 1
        (0, 1, 2.5),  # non-integral
        "012",  # string, not a joint sequence
        5,  # scalar, not a sequence
    ],
)
def test_split_active_joints_rejects_malformed_values(bad):
    with pytest.raises(ValueError):
        _controller(
            split_base_wrist_task=True,
            split_base_wrist_active_joints=bad,
        ).compute(_make_state(J=FAILURE_POSE_J))


def test_split_active_joints_without_the_flag_raises():
    # Loud rather than a silent no-op: a config that names an active joint set
    # while the mechanism reading it is off would otherwise quietly run the
    # full-Jacobian task.
    with pytest.raises(ValueError, match="split_base_wrist_task is False"):
        _controller(
            split_base_wrist_task=False,
            split_base_wrist_active_joints=ELBOW_WRIST1_WRIST2,
        ).compute(_make_state(J=FAILURE_POSE_J))


def test_split_active_joints_yaml_parsing():
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
    base_section = {
        "torque_limits_mode": "initial",
        "torque_limits_initial": torque_limits,
        "split_base_wrist_task": True,
    }
    # Absent -> None (historical default).
    assert (
        CartesianImpedanceConfig.from_controller_yaml_section(dict(base_section))
        .split_base_wrist_active_joints
        is None
    )
    # A YAML list becomes a tuple of plain ints.
    cfg = CartesianImpedanceConfig.from_controller_yaml_section(
        dict(base_section, split_base_wrist_active_joints=[2, 3, 4])
    )
    assert cfg.split_base_wrist_active_joints == ELBOW_WRIST1_WRIST2
    assert all(type(i) is int for i in cfg.split_base_wrist_active_joints)
    # Integral floats (how plain YAML scalars often deserialize) are accepted.
    assert CartesianImpedanceConfig.from_controller_yaml_section(
        dict(base_section, split_base_wrist_active_joints=[2.0, 3.0, 4.0])
    ).split_base_wrist_active_joints == ELBOW_WRIST1_WRIST2
    # Malformed values raise at load time, not at the first control cycle.
    with pytest.raises(ValueError):
        CartesianImpedanceConfig.from_controller_yaml_section(
            dict(base_section, split_base_wrist_active_joints=[1, 2])
        )
