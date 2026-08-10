"""Tests for controller_core/cartesian_velocity_controller.py -- resolved-
rate Cartesian velocity law (P + optional reduced-task redundancy
resolution), no torque/gravity/mass-matrix dynamics."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from controller_core.cartesian_velocity_controller import (  # noqa: E402
    CartesianVelocityConfig,
    CartesianVelocityController,
    _damped_pinv,
)
from controller_core.cartesian_velocity_controller.modes import compute_ik_seeded  # noqa: E402
from controller_core.kinematics_utils import (  # noqa: E402
    null_space_basis,
    orientation_error_vec_wxyz,
    quat_multiply_wxyz,
    rotvec_to_quat_wxyz,
    swing_twist_axis_error,
)


def test_swing_twist_matches_small_angle_vector_for_small_pure_axis_rotation():
    """For a small, pure rotation about one axis, the exact swing-twist
    angle and the small-angle 2*vec(q_err) approximation must agree
    closely -- confirms the new function isn't a different convention,
    just a more exact version for large angles."""
    theta = 0.01
    quat_now = rotvec_to_quat_wxyz(np.array([0.0, 0.0, theta]))
    quat0 = np.array([1.0, 0.0, 0.0, 0.0])
    e_rot = orientation_error_vec_wxyz(quat0, quat_now)
    twist_z = swing_twist_axis_error(quat0, quat_now, 2)
    assert twist_z == pytest.approx(e_rot[2], abs=1e-6)
    assert swing_twist_axis_error(quat0, quat_now, 0) == pytest.approx(0.0, abs=1e-9)
    assert swing_twist_axis_error(quat0, quat_now, 1) == pytest.approx(0.0, abs=1e-9)


def test_swing_twist_is_exact_for_large_pure_axis_rotation():
    """For a LARGE pure rotation about one axis, the exact twist angle must
    equal that rotation exactly -- unlike 2*vec(q_err), which is only a
    small-angle approximation and visibly deviates at this magnitude."""
    theta = 2.5
    quat_now = rotvec_to_quat_wxyz(np.array([0.0, 0.0, theta]))
    quat0 = np.array([1.0, 0.0, 0.0, 0.0])
    twist_z = swing_twist_axis_error(quat0, quat_now, 2)
    assert twist_z == pytest.approx(theta, abs=1e-9)
    e_rot = orientation_error_vec_wxyz(quat0, quat_now)
    assert abs(e_rot[2] - theta) > 0.1  # the small-angle approximation is visibly wrong here


def test_swing_twist_z_component_is_exact_regardless_of_other_axis_rotation():
    """The real property this function guarantees (and 2*vec(q_err)'s row 2
    does not, once rotations grow large): composing a pure-Z rotation with
    ANY other-axis rotation, the swing-twist Z-angle recovers the pure-Z
    contribution EXACTLY, for several other-axis magnitudes -- genuinely
    axis-separable, not just "close enough" for one specific case."""
    small_z = 0.02
    quat_z_small = rotvec_to_quat_wxyz(np.array([0.0, 0.0, small_z]))
    quat0 = np.array([1.0, 0.0, 0.0, 0.0])
    for other_axis_theta in [0.1, 0.5, 1.0, 1.5, 2.0]:
        quat_x = rotvec_to_quat_wxyz(np.array([other_axis_theta, 0.0, 0.0]))
        quat_compound = quat_multiply_wxyz(quat_x, quat_z_small)
        twist_z = swing_twist_axis_error(quat0, quat_compound, 2)
        assert twist_z == pytest.approx(small_z, abs=1e-9), (
            f"failed at other_axis_theta={other_axis_theta}"
        )


def test_swing_twist_axis_error_rejects_bad_axis_index():
    quat0 = np.array([1.0, 0.0, 0.0, 0.0])
    with pytest.raises(ValueError):
        swing_twist_axis_error(quat0, quat0, 3)


def test_damped_pinv_matches_plain_pinv_when_well_conditioned():
    rng = np.random.default_rng(3)
    j = np.eye(4, 6) + 0.1 * rng.standard_normal((4, 6))
    np.testing.assert_allclose(_damped_pinv(j, 0.0), np.linalg.pinv(j), atol=1e-10)


def test_damped_pinv_bounds_amplification_near_singularity():
    """A near-rank-deficient matrix (last row nearly a copy of the third,
    scaled tiny) blows up under plain pinv but stays bounded under damping
    -- this is exactly the numerical regime measured at the wrist_2~0
    pose (moderately small, not exactly zero, singular values)."""
    j = np.eye(4, 6)
    j[3, :] = j[2, :] * 1e-3  # near-singular row
    plain_norm = float(np.linalg.norm(np.linalg.pinv(j), ord=2))
    damped_norm = float(np.linalg.norm(_damped_pinv(j, 0.05), ord=2))
    assert damped_norm < plain_norm


def _state(ee_pos, ee_quat=(1.0, 0.0, 0.0, 0.0), target_ee_pos=None, target_ee_vel=None, jacobian=None):
    return {
        "time": 0.0,
        "q": np.zeros(6),
        "qd": np.zeros(6),
        "ee_pos": np.asarray(ee_pos, dtype=np.float64),
        "ee_quat": np.asarray(ee_quat, dtype=np.float64),
        "target_x": float(ee_pos[0]),
        "target_ee_pos": None if target_ee_pos is None else np.asarray(target_ee_pos, dtype=np.float64),
        "target_ee_vel": None if target_ee_vel is None else np.asarray(target_ee_vel, dtype=np.float64),
        "jacobian": None if jacobian is None else np.asarray(jacobian, dtype=np.float64),
    }


# --------------------------------------------------------------------------- #
# Base P-law (reduced_task_dims=False, no jacobian needed) -- the original
# full-3-axis-hold behavior, still available as an explicit opt-out.
# --------------------------------------------------------------------------- #
def test_defaults_are_velocity_gains_not_force_gains():
    cfg = CartesianVelocityConfig()
    assert cfg.kp_x == 2.0
    assert cfg.max_lin_speed_mps == 0.25


def test_reduced_task_dims_defaults_on():
    cfg = CartesianVelocityConfig()
    assert cfg.reduced_task_dims is True
    assert cfg.task_dim_rx is False
    assert cfg.task_dim_ry is False
    assert cfg.task_dim_rz is True


def test_requires_reset_before_compute():
    cfg = CartesianVelocityConfig(reduced_task_dims=False)
    ctrl = CartesianVelocityController(cfg)
    try:
        ctrl.compute(_state([0.0, 0.0, 0.0]))
        assert False, "expected RuntimeError"
    except RuntimeError:
        pass


def test_reduced_task_dims_without_jacobian_raises():
    cfg = CartesianVelocityConfig()  # reduced_task_dims=True by default
    ctrl = CartesianVelocityController(cfg)
    p0 = [0.0, 0.0, 0.0]
    ctrl.reset_from_state(_state(p0))
    with pytest.raises(ValueError, match="jacobian"):
        ctrl.compute(_state(p0))  # no jacobian supplied


def test_at_rest_with_no_target_produces_zero_velocity():
    cfg = CartesianVelocityConfig(reduced_task_dims=False)
    ctrl = CartesianVelocityController(cfg)
    p0 = [0.4, -0.2, 0.3]
    ctrl.reset_from_state(_state(p0))
    xd = ctrl.compute(_state(p0))
    np.testing.assert_allclose(xd, np.zeros(6), atol=1e-10)


def test_position_error_produces_proportional_restoring_velocity():
    cfg = CartesianVelocityConfig(
        kp_x=2.0, kp_y=2.0, kp_z=2.0, kp_rot=2.0, max_lin_speed_mps=10.0, reduced_task_dims=False
    )
    ctrl = CartesianVelocityController(cfg)
    p0 = [0.0, 0.0, 0.0]
    ctrl.reset_from_state(_state(p0))
    # Actual position is behind target -> command should push toward target.
    xd = ctrl.compute(_state([0.0, 0.0, 0.0], target_ee_pos=[0.05, 0.0, 0.0]))
    assert xd[0] == pytest.approx(2.0 * 0.05)
    np.testing.assert_allclose(xd[1:], np.zeros(5), atol=1e-10)


def test_feedforward_velocity_is_added_to_p_correction():
    cfg = CartesianVelocityConfig(kp_x=2.0, max_lin_speed_mps=10.0, reduced_task_dims=False)
    ctrl = CartesianVelocityController(cfg)
    p0 = [0.0, 0.0, 0.0]
    ctrl.reset_from_state(_state(p0))
    xd = ctrl.compute(_state(p0, target_ee_pos=p0, target_ee_vel=[0.03, 0.0, 0.0]))
    assert abs(xd[0] - 0.03) < 1e-9


def test_holds_y_z_at_reset_value_when_only_x_target_moves():
    cfg = CartesianVelocityConfig(kp_x=2.0, kp_y=2.0, kp_z=2.0, max_lin_speed_mps=10.0, reduced_task_dims=False)
    ctrl = CartesianVelocityController(cfg)
    p0 = [0.0, -0.2, 0.3]
    ctrl.reset_from_state(_state(p0))
    # Robot has drifted off Y/Z; target_ee_pos still holds Y0/Z0 -> restoring command.
    drifted = [0.02, -0.19, 0.31]
    xd = ctrl.compute(_state(drifted, target_ee_pos=[0.02, p0[1], p0[2]]))
    assert xd[1] == pytest.approx(2.0 * (p0[1] - drifted[1]))
    assert xd[2] == pytest.approx(2.0 * (p0[2] - drifted[2]))


def test_linear_speed_is_clamped_to_configured_ceiling():
    cfg = CartesianVelocityConfig(kp_x=100.0, max_lin_speed_mps=0.1, reduced_task_dims=False)
    ctrl = CartesianVelocityController(cfg)
    p0 = [0.0, 0.0, 0.0]
    ctrl.reset_from_state(_state(p0))
    xd = ctrl.compute(_state(p0, target_ee_pos=[1.0, 0.0, 0.0]))  # huge error
    lin_norm = float(np.linalg.norm(xd[:3]))
    assert lin_norm <= cfg.max_lin_speed_mps + 1e-9
    assert lin_norm == pytest.approx(cfg.max_lin_speed_mps, abs=1e-6)


def test_orientation_error_produces_proportional_angular_velocity():
    cfg = CartesianVelocityConfig(kp_rot=1.0, max_ang_speed_radps=10.0, reduced_task_dims=False)
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
    cfg = CartesianVelocityConfig(kp_rot=1000.0, max_ang_speed_radps=0.2, reduced_task_dims=False)
    ctrl = CartesianVelocityController(cfg)
    p0 = [0.0, 0.0, 0.0]
    ctrl.reset_from_state(_state(p0))
    theta = 2.0  # large rotation error
    quat_now = [np.cos(theta / 2), 0.0, 0.0, np.sin(theta / 2)]
    xd = ctrl.compute(_state(p0, ee_quat=quat_now, target_ee_pos=p0))
    ang_norm = float(np.linalg.norm(xd[3:]))
    assert ang_norm <= cfg.max_ang_speed_radps + 1e-9


# --------------------------------------------------------------------------- #
# reduced_task_dims=True (the new default) -- redundancy resolution via
# pinv(J_task) then projection through the full J, not zeroing/P-holding
# rx/ry directly.
# --------------------------------------------------------------------------- #
def test_reduced_task_with_identity_jacobian_drops_rx_ry_keeps_rz():
    """With J=I, pinv(J_task) @ xd_task for task rows [x,y,z,rz] reproduces
    exactly [vx,vy,vz,0,0,wz] when projected back through J=I -- rx/ry
    columns are unreachable from that row selection, so minimum-norm sets
    qd there to 0, and identity J makes the resulting xd match qd exactly."""
    cfg = CartesianVelocityConfig(kp_x=2.0, kp_y=2.0, kp_z=2.0, kp_rot=5.0, max_lin_speed_mps=10.0)
    ctrl = CartesianVelocityController(cfg)
    p0 = [0.0, 0.0, 0.0]
    ctrl.reset_from_state(_state(p0, ee_quat=[1.0, 0.0, 0.0, 0.0]))
    # A rotation with nonzero components on ALL three axes.
    quat_now = [0.9, 0.3, 0.2, 0.15]
    quat_now = list(np.asarray(quat_now) / np.linalg.norm(quat_now))
    xd = ctrl.compute(_state(p0, ee_quat=quat_now, target_ee_pos=p0, jacobian=np.eye(6)))
    assert xd[3] == pytest.approx(0.0, abs=1e-9)
    assert xd[4] == pytest.approx(0.0, abs=1e-9)
    assert abs(xd[5]) > 1e-6  # rz still actively held


def test_reduced_task_with_nontrivial_jacobian_lets_redundancy_produce_nonzero_rx_ry():
    """With a Jacobian that actually couples joints across rows (not
    identity), the null-space motion of the reduced (x,y,z,rz) task
    generally produces NONZERO rx/ry velocity when projected through the
    full J -- proving this is genuine redundancy resolution, not a
    disguised zero/lock on rx/ry."""
    rng = np.random.default_rng(0)
    # A well-conditioned, non-identity, non-block-diagonal 6x6 Jacobian.
    jac = np.eye(6) + 0.4 * rng.standard_normal((6, 6))
    cfg = CartesianVelocityConfig(kp_x=2.0, kp_y=2.0, kp_z=2.0, kp_rot=3.0, max_lin_speed_mps=10.0, max_ang_speed_radps=10.0)
    ctrl = CartesianVelocityController(cfg)
    p0 = [0.0, 0.0, 0.0]
    ctrl.reset_from_state(_state(p0, ee_quat=[1.0, 0.0, 0.0, 0.0]))
    quat_now = [0.9, 0.25, 0.2, 0.15]
    quat_now = list(np.asarray(quat_now) / np.linalg.norm(quat_now))
    xd = ctrl.compute(_state(p0, ee_quat=quat_now, target_ee_pos=[0.02, 0.0, 0.0], jacobian=jac))
    # rx/ry are no longer forced to zero -- some nonzero value falls out of
    # the coupled Jacobian's null-space motion.
    assert abs(xd[3]) > 1e-6 or abs(xd[4]) > 1e-6


def test_reduced_task_result_is_consistent_with_its_own_qd():
    """xd_cmd = J @ pinv(J_task) @ xd_task by construction -- verify the
    returned (pre-clamp-scale) command is achievable by SOME qd through the
    full Jacobian, i.e. J @ qd reproduces xd_cmd, not an inconsistent
    fabrication. Uses generous gains/ceilings so clamping never engages,
    keeping this a pure consistency check of the unclamped law."""
    rng = np.random.default_rng(1)
    jac = np.eye(6) + 0.3 * rng.standard_normal((6, 6))
    cfg = CartesianVelocityConfig(
        kp_x=1.0, kp_y=1.0, kp_z=1.0, kp_rot=1.0, max_lin_speed_mps=1000.0, max_ang_speed_radps=1000.0,
        kp_posture=0.0,  # isolate the primary-task projection; q==q_rest below makes this moot either way
    )
    ctrl = CartesianVelocityController(cfg)
    p0 = [0.0, 0.0, 0.0]
    ctrl.reset_from_state(_state(p0))
    xd = ctrl.compute(_state(p0, target_ee_pos=[0.01, 0.0, 0.0], jacobian=jac))

    rot_flags = [cfg.task_dim_rx, cfg.task_dim_ry, cfg.task_dim_rz]
    selected = [0, 1, 2] + [3 + i for i, on in enumerate(rot_flags) if on]
    j_task = jac[selected, :]
    xd_task_expected = xd[selected]  # after projection, task rows must be reproduced exactly
    qd = np.linalg.pinv(j_task) @ xd_task_expected
    np.testing.assert_allclose(jac @ qd, xd, atol=1e-8)


def test_nullspace_posture_term_never_perturbs_primary_task():
    """With q != q_rest (so the posture term is genuinely active), a nonzero
    kp_posture, and pinv_damping=0 (isolating the EXACT, undamped
    null-space projector -- with damping>0 this only holds approximately,
    by design, see _damped_pinv's docstring), the primary X/Y/Z/rz task
    rows of xd_cmd must still be EXACTLY what the primary-only law would
    produce -- the null-space projection guarantees the posture term can't
    leak into the task, same guarantee nullspace_posture provides in the
    torque controller."""
    rng = np.random.default_rng(2)
    jac = np.eye(6) + 0.35 * rng.standard_normal((6, 6))
    cfg_with_posture = CartesianVelocityConfig(
        kp_x=1.5, kp_y=1.5, kp_z=1.5, kp_rot=1.5, max_lin_speed_mps=1000.0,
        max_ang_speed_radps=1000.0, kp_posture=3.0, pinv_damping=0.0,
    )
    cfg_no_posture = CartesianVelocityConfig(
        kp_x=1.5, kp_y=1.5, kp_z=1.5, kp_rot=1.5, max_lin_speed_mps=1000.0,
        max_ang_speed_radps=1000.0, kp_posture=0.0, pinv_damping=0.0,
    )
    q0 = np.array([0.1, -0.2, 0.3, -0.1, 0.05, -0.05])
    q_now = q0 + np.array([0.02, -0.01, 0.03, 0.0, -0.02, 0.01])  # drifted from q_rest

    p0 = [0.0, 0.0, 0.0]
    st_reset = _state(p0)
    st_reset["q"] = q0.copy()

    st_compute_with = _state(p0, target_ee_pos=[0.02, 0.0, 0.0], jacobian=jac)
    st_compute_with["q"] = q_now.copy()
    st_compute_without = _state(p0, target_ee_pos=[0.02, 0.0, 0.0], jacobian=jac)
    st_compute_without["q"] = q_now.copy()

    ctrl_with = CartesianVelocityController(cfg_with_posture)
    ctrl_with.reset_from_state(st_reset)
    xd_with = ctrl_with.compute(st_compute_with)

    ctrl_without = CartesianVelocityController(cfg_no_posture)
    ctrl_without.reset_from_state(st_reset)
    xd_without = ctrl_without.compute(st_compute_without)

    rot_flags = [cfg_with_posture.task_dim_rx, cfg_with_posture.task_dim_ry, cfg_with_posture.task_dim_rz]
    selected = [0, 1, 2] + [3 + i for i, on in enumerate(rot_flags) if on]
    np.testing.assert_allclose(xd_with[selected], xd_without[selected], atol=1e-8)
    # But the posture term DOES change the un-selected (rx/ry) rows -- proof
    # it's doing something, not a silent no-op.
    unselected = [3, 4] if not cfg_with_posture.task_dim_rx and not cfg_with_posture.task_dim_ry else []
    if unselected:
        assert not np.allclose(xd_with[unselected], xd_without[unselected], atol=1e-8)


def test_posture_reanchor_on_settle_stops_unbounded_hold_drift():
    """Real bug found 2026-08-03: without reanchoring, the posture term
    keeps pulling toward the STALE reset-time q_rest even once the arm is
    correctly holding a new position, and because the null-space projector
    itself changes with q, this does not converge -- it drifts unboundedly.
    With reanchoring on, once position error drops under reanchor_pos_tol_m
    (and feedforward velocity is ~0, i.e. a real settled hold, not just a
    transient), q_rest is recaptured at the current q, and the posture term
    (and the total qd it feeds) collapses toward zero -- proving the drift
    stops, not just slows."""
    rng = np.random.default_rng(4)
    jac = np.eye(6) + 0.2 * rng.standard_normal((6, 6))
    cfg = CartesianVelocityConfig(
        kp_x=2.0, kp_y=2.0, kp_z=2.0, kp_rot=2.0, max_lin_speed_mps=10.0, max_ang_speed_radps=10.0,
        kp_posture=1.0, pinv_damping=0.01, posture_reanchor_on_settle=True, reanchor_pos_tol_m=0.002,
        reanchor_settle_cycles=1,  # simplifies this test to a single settled call; the
        # consecutive-cycle requirement itself is covered separately below.
    )
    ctrl = CartesianVelocityController(cfg)
    q0 = np.array([0.1, -0.2, 0.3, -0.1, 0.05, -0.05])
    p0 = [0.0, 0.0, 0.0]
    st_reset = _state(p0)
    st_reset["q"] = q0.copy()
    ctrl.reset_from_state(st_reset)

    # Simulate settling: position has already reached the (small) target,
    # feedforward velocity is zero (a static hold), but q has drifted from
    # q0 (as it would after a real move) -- before reanchoring fires this
    # is exactly the unbounded-drift scenario; after it fires, qd_secondary
    # (and thus the whole command, since xd_task ~ 0 at a true settle)
    # should collapse toward zero.
    q_drifted = q0 + np.array([0.02, -0.01, 0.03, 0.0, -0.02, 0.01])
    target = [0.0005, 0.0, 0.0]  # within reanchor_pos_tol_m of p0

    st1 = _state(target, target_ee_pos=target, target_ee_vel=[0.0, 0.0, 0.0], jacobian=jac)
    st1["q"] = q_drifted.copy()
    xd_first = ctrl.compute(st1)
    assert ctrl._reanchored is True
    np.testing.assert_allclose(ctrl._q_rest, q_drifted, atol=1e-12)

    # A second call at the SAME (already-reanchored) q must now produce a
    # near-zero command -- nothing left for the posture term to correct.
    xd_second = ctrl.compute(st1)
    assert float(np.linalg.norm(xd_second)) < float(np.linalg.norm(xd_first)) + 1e-6
    assert float(np.linalg.norm(xd_second)) < 1e-6


def test_reanchor_requires_consecutive_settled_cycles_not_one_instant():
    """Real bug found 2026-08-03: at the very start of a move, pos_err=0
    and v_ff=0 for exactly one cycle by construction (a min-jerk profile
    starts at rest at its own current position) -- a naive single-cycle
    settle check reanchors immediately to the SAME q_rest already set at
    reset(), a total no-op that never catches the real post-move settle.
    Verifies: one settled cycle alone does NOT reanchor; reanchor_settle_
    cycles consecutive settled cycles DOES."""
    jac = np.eye(6)
    cfg = CartesianVelocityConfig(
        kp_x=2.0, kp_y=2.0, kp_z=2.0, kp_rot=2.0, max_lin_speed_mps=10.0, max_ang_speed_radps=10.0,
        kp_posture=1.0, pinv_damping=0.01, posture_reanchor_on_settle=True,
        reanchor_pos_tol_m=0.002, reanchor_settle_cycles=5,
    )
    ctrl = CartesianVelocityController(cfg)
    p0 = [0.0, 0.0, 0.0]
    ctrl.reset_from_state(_state(p0))

    st_settled = _state(p0, target_ee_pos=p0, target_ee_vel=[0.0, 0.0, 0.0], jacobian=jac)
    ctrl.compute(st_settled)
    assert ctrl._reanchored is False  # one settled cycle: not yet

    for _ in range(3):
        ctrl.compute(st_settled)
    assert ctrl._reanchored is False  # 4 total: still not yet

    ctrl.compute(st_settled)
    assert ctrl._reanchored is True  # 5th consecutive settled cycle: now it fires

    # An unsettled cycle in between must reset the streak.
    ctrl2 = CartesianVelocityController(cfg)
    ctrl2.reset_from_state(_state(p0))
    for _ in range(4):
        ctrl2.compute(st_settled)
    st_moving = _state(p0, target_ee_pos=[0.05, 0.0, 0.0], target_ee_vel=[0.1, 0.0, 0.0], jacobian=jac)
    ctrl2.compute(st_moving)  # breaks the streak
    ctrl2.compute(st_settled)
    assert ctrl2._reanchored is False  # streak had to restart, only 1 settled cycle since the break


def test_linear_speed_clamp_still_applies_with_reduced_task_dims():
    cfg = CartesianVelocityConfig(kp_x=100.0, max_lin_speed_mps=0.1)
    ctrl = CartesianVelocityController(cfg)
    p0 = [0.0, 0.0, 0.0]
    ctrl.reset_from_state(_state(p0))
    xd = ctrl.compute(_state(p0, target_ee_pos=[1.0, 0.0, 0.0], jacobian=np.eye(6)))
    lin_norm = float(np.linalg.norm(xd[:3]))
    assert lin_norm <= cfg.max_lin_speed_mps + 1e-9


def test_yaml_parsing_reads_velocity_control_block():
    ctrl_section = {
        "velocity_control": {
            "kp_x": 3.5,
            "kp_y": 1.5,
            "kp_z": 1.5,
            "kp_rot": 0.8,
            "max_lin_speed_mps": 0.4,
            "max_ang_speed_radps": 0.6,
            "reduced_task_dims": False,
            "task_dim_rx": True,
            "task_dim_ry": True,
            "task_dim_rz": False,
            "kp_posture": 2.5,
        }
    }
    cfg = CartesianVelocityConfig.from_controller_yaml_section(ctrl_section)
    assert cfg.kp_x == 3.5
    assert cfg.kp_y == 1.5
    assert cfg.max_lin_speed_mps == 0.4
    assert cfg.max_ang_speed_radps == 0.6
    assert cfg.reduced_task_dims is False
    assert cfg.task_dim_rx is True
    assert cfg.task_dim_ry is True
    assert cfg.task_dim_rz is False
    assert cfg.kp_posture == 2.5


def test_yaml_parsing_defaults_when_block_absent():
    cfg = CartesianVelocityConfig.from_controller_yaml_section({})
    assert cfg.kp_x == 2.0
    assert cfg.max_lin_speed_mps == 0.25
    assert cfg.reduced_task_dims is True
    assert cfg.task_dim_rz is True
    assert cfg.kp_posture == 1.0
    assert cfg.pinv_damping == 0.005


# --------------------------------------------------------------------------- #
# split_base_wrist_task / ik_seeded_resolution -- mutual exclusivity, and
# ik_seeded_resolution's key property (path-independence, the actual fix for
# reduced_task_dims'/split_base_wrist_task's multistability -- see this
# module's own module docstring, and controller_core/cartesian_velocity_
# controller.py's, for the full investigation this resolves).
# --------------------------------------------------------------------------- #
def test_more_than_one_resolution_mode_raises():
    cfg = CartesianVelocityConfig(reduced_task_dims=True, split_base_wrist_task=True)
    ctrl = CartesianVelocityController(cfg)
    ctrl.reset_from_state(_state([0.0, 0.0, 0.0]))
    with pytest.raises(ValueError, match="mutually exclusive"):
        ctrl.compute(_state([0.0, 0.0, 0.0], target_ee_pos=[0.01, 0.0, 0.0], jacobian=np.eye(6)))


def test_ik_seeded_resolution_without_fk_fn_raises():
    cfg = CartesianVelocityConfig(reduced_task_dims=False, ik_seeded_resolution=True)
    ctrl = CartesianVelocityController(cfg)
    ctrl.reset_from_state(_state([0.0, 0.0, 0.0]))
    with pytest.raises(ValueError, match="fk_jacobian_fn"):
        ctrl.compute(_state([0.0, 0.0, 0.0], target_ee_pos=[0.01, 0.0, 0.0]))


def _toy_fk(q: np.ndarray):
    """A deterministic, nontrivial (non-block-diagonal) toy forward-
    kinematics + Jacobian function for testing ik_seeded_resolution without
    needing MuJoCo -- controller_core stays simulator-independent, so the
    real fk_jacobian_fn always comes from the caller (see hardware/
    velocity_transport.py / tools/diagnostics/ur5e_velocity_control_
    kinematic_sim.py for the real MuJoCo-backed versions)."""
    q = np.asarray(q, dtype=np.float64).reshape(6)
    pos = q[0:3].copy() + 0.05 * q[3:6]
    quat = rotvec_to_quat_wxyz(0.3 * q[3:6])
    jac = np.eye(6) + 0.1 * np.sin(q).reshape(1, 6) * np.ones((6, 1))
    return pos, quat, jac


def test_ik_seeded_resolution_q_target_is_path_independent():
    """The actual property this mode exists to guarantee: the joint-space
    target the controller drives toward is a deterministic function of ONLY
    (q_rest, target) -- recovering it from two completely different current
    joint configurations (simulating two different move histories arriving
    at the same moment) must give the identical result, unlike reduced_
    task_dims'/split_base_wrist_task's rate-integrated null-space walks,
    which are provably path-dependent (see the module docstring's
    multistability findings)."""
    cfg = CartesianVelocityConfig(
        kp_x=2.0, kp_y=2.0, kp_z=2.0, kp_rot=2.0, max_lin_speed_mps=1000.0, max_ang_speed_radps=1000.0,
        reduced_task_dims=False, ik_seeded_resolution=True, ik_iterations=8, ik_joint_gain=4.0, pinv_damping=0.01,
    )
    q_rest = np.zeros(6)
    p0, quat0, _ = _toy_fk(q_rest)
    target = p0.copy()
    target[0] += 0.05

    def q_target_from(q_current: np.ndarray) -> np.ndarray:
        ctrl = CartesianVelocityController(cfg)
        ctrl.reset_from_state(
            {"time": 0.0, "q": q_rest, "qd": np.zeros(6), "ee_pos": p0, "ee_quat": quat0, "target_x": float(p0[0])}
        )
        p_c, quat_c, jac_c = _toy_fk(q_current)
        xd = ctrl.compute(
            {
                "time": 1.0,
                "q": q_current,
                "qd": np.zeros(6),
                "ee_pos": p_c,
                "ee_quat": quat_c,
                "target_x": float(target[0]),
                "target_ee_pos": target,
                "target_ee_vel": np.zeros(3),
                "fk_jacobian_fn": _toy_fk,
            }
        )
        qd_joint = np.linalg.pinv(jac_c) @ xd
        return q_current + qd_joint / cfg.ik_joint_gain

    q_target_a = q_target_from(q_rest + np.array([0.3, -0.2, 0.1, 0.05, -0.05, 0.02]))
    q_target_b = q_target_from(q_rest + np.array([-0.4, 0.35, -0.15, 0.1, 0.08, -0.03]))
    np.testing.assert_allclose(q_target_a, q_target_b, atol=1e-9)


def _toy_fk_with_free_rx(q: np.ndarray):
    """Like _toy_fk, but with a joint (index 3) that is WEAKLY coupled into
    the position task (jac[0,3]=0.15 -- moving it helps reduce X error a
    little, so the QP's least-squares solve will use it, exactly like a
    real coupled Jacobian) while ALSO driving rx (rotvec index 0, jac[3,3]
    =0.6) which is entirely outside the default task selection (task_dim_rx
    =False). This is the actual mechanism traced in the real MuJoCo case
    that motivated this fix: satisfying the position/rz task via a joint
    that ALSO happens to move an unconstrained rotation axis, with nothing
    but the QP's weak reg*||dq||^2 term (deliberately tiny here, matching
    the real gain search's found pinv_damping) discouraging that joint from
    moving. An earlier version of this toy FK made column 3 exactly zero in
    every task row, which meant q[3] (and therefore rx) never moved at all
    regardless of posture_gain -- that tested nothing; this version keeps
    the coupling deliberately real but small so posture_gain's pull is
    measurable against it, not fighting a stronger primary-task signal."""
    q = np.asarray(q, dtype=np.float64).reshape(6)
    pos = q[0:3].copy()
    pos[0] += 0.15 * q[3]
    quat = rotvec_to_quat_wxyz(np.array([0.6 * q[3], 0.0, 0.3 * q[5]]))
    jac = np.eye(6)
    jac[0:3, 0:3] = np.eye(3)
    jac[0, 3] = 0.15  # position-x weakly coupled to joint 3
    jac[5, 5] = 1.0  # rz driven only by q[5]
    jac[3, 3] = 0.6  # rx driven by q[3] -- outside the x/y/z/rz task entirely
    jac[4, 4] = 0.0  # ry unreachable (kept simple/degenerate on purpose)
    return pos, quat, jac


# --------------------------------------------------------------------------- #
# ik_max_joint_deviation_rad -- hard bound on the null-space (redundant) part
# of compute_ik_seeded's solve, via null_space_basis coordinate clipping.
# Added 2026-08-06 after TWO prior mechanisms (a soft posture pull, then a
# uniform per-joint hard bound) were each found wrong -- see modes.py's git
# history. This is the corrected version: exact task preservation, proven
# both mathematically (null_space_basis's own tests) and here end-to-end.
# --------------------------------------------------------------------------- #
def test_ik_max_joint_deviation_default_none_matches_prior_behavior():
    q_rest = np.zeros(6)
    p0, quat0, _ = _toy_fk(q_rest)
    target = p0.copy()
    target[0] += 0.05
    q_current = q_rest + np.array([0.05, -0.03, 0.02, 0.01, -0.02, 0.015])

    def run(cfg):
        ctrl = CartesianVelocityController(cfg)
        ctrl.reset_from_state(
            {"time": 0.0, "q": q_rest, "qd": np.zeros(6), "ee_pos": p0, "ee_quat": quat0, "target_x": float(p0[0])}
        )
        p_c, quat_c, _ = _toy_fk(q_current)
        return ctrl.compute(
            {
                "time": 1.0, "q": q_current, "qd": np.zeros(6), "ee_pos": p_c, "ee_quat": quat_c,
                "target_x": float(target[0]), "target_ee_pos": target, "target_ee_vel": np.zeros(3),
                "fk_jacobian_fn": _toy_fk,
            }
        )

    cfg_explicit_none = CartesianVelocityConfig(
        kp_x=2.0, kp_y=2.0, kp_z=2.0, kp_rot=2.0, max_lin_speed_mps=1000.0, max_ang_speed_radps=1000.0,
        reduced_task_dims=False, ik_seeded_resolution=True, ik_iterations=8, ik_joint_gain=4.0,
        pinv_damping=0.05, qp_task_weight=1.0e4, ik_max_joint_deviation_rad=None,
    )
    cfg_default = CartesianVelocityConfig(
        kp_x=2.0, kp_y=2.0, kp_z=2.0, kp_rot=2.0, max_lin_speed_mps=1000.0, max_ang_speed_radps=1000.0,
        reduced_task_dims=False, ik_seeded_resolution=True, ik_iterations=8, ik_joint_gain=4.0,
        pinv_damping=0.05, qp_task_weight=1.0e4,
    )
    assert cfg_default.ik_max_joint_deviation_rad is None
    np.testing.assert_allclose(run(cfg_explicit_none), run(cfg_default), atol=1e-14)


def test_ik_max_joint_deviation_preserves_task_exactly():
    """The core guarantee: with a genuinely redundant task (4 rows, 6
    joints -- 2D null space), q_target's TASK-SPACE image (j_task @
    q_target's implied step) must be IDENTICAL whether or not
    ik_max_joint_deviation_rad clips the null-space part -- even at a
    very tight bound that clearly must be altering SOMETHING (confirmed
    separately by the next test)."""
    q_rest = np.zeros(6)
    p0, quat0, _ = _toy_fk(q_rest)
    target = p0.copy()
    target[0] += 0.05

    def q_target_for(max_dev):
        cfg = CartesianVelocityConfig(
            kp_x=2.0, kp_y=2.0, kp_z=2.0, kp_rot=2.0, max_lin_speed_mps=1000.0, max_ang_speed_radps=1000.0,
            reduced_task_dims=False, ik_seeded_resolution=True, ik_iterations=10, ik_joint_gain=4.0,
            pinv_damping=0.05, qp_task_weight=1.0e4, ik_max_joint_deviation_rad=max_dev,
        )
        ctrl = CartesianVelocityController(cfg)
        ctrl.reset_from_state(
            {"time": 0.0, "q": q_rest, "qd": np.zeros(6), "ee_pos": p0, "ee_quat": quat0, "target_x": float(p0[0])}
        )
        xd = ctrl.compute(
            {
                "time": 1.0, "q": q_rest, "qd": np.zeros(6), "ee_pos": p0, "ee_quat": quat0,
                "target_x": float(target[0]), "target_ee_pos": target, "target_ee_vel": np.zeros(3),
                "fk_jacobian_fn": _toy_fk,
            }
        )
        return xd

    xd_unconstrained = q_target_for(None)
    xd_tight = q_target_for(0.01)
    # task rows are x,y,z,rz (indices 0,1,2,5 of the 6D twist) -- these must
    # match regardless of how tightly the null-space (rows 3,4 here, since
    # _toy_fk's jac happens to be close to identity-like) is clipped.
    np.testing.assert_allclose(xd_unconstrained[[0, 1, 2, 5]], xd_tight[[0, 1, 2, 5]], atol=1e-6)


def _toy_fk_rotating_null_space(q: np.ndarray):
    """Unlike _toy_fk_with_free_rx (a CONSTANT jacobian), this toy's
    jacobian is deliberately pose-dependent in a way that makes its null
    space itself rotate as q moves -- required to exercise real, organic
    null-space drift at all. (Found the hard way: a single Newton/QP step
    is ALWAYS the minimum-norm solution, which by construction has EXACTLY
    zero null-space component -- pinv(J)'s range is exactly row(J). Drift
    can only accumulate across MULTIPLE iterations whose jacobians
    genuinely differ from each other; _toy_fk's mild 0.1*sin(q) term and
    _toy_fk_with_free_rx's constant jacobian both measured as exactly zero
    organic drift for this reason, however far pinv_damping was pushed.)
    x saturates via tanh (so a large target needs several Newton
    corrections, not one) and jac[0,3] grows with q[0] (so joint 3 --
    outside the x/y/z/rz task selection -- becomes progressively coupled
    into the task as the solve progresses, rotating the null space away
    from its q_rest orientation)."""
    q = np.asarray(q, dtype=np.float64).reshape(6)
    pos = np.array([2.0 * np.tanh(q[0]), q[1], q[2]])
    quat = rotvec_to_quat_wxyz(0.3 * q[3:6])
    jac = np.eye(6)
    jac[0, 0] = 2.0 * (1.0 - np.tanh(q[0]) ** 2)
    jac[0, 3] = 0.8 * q[0]
    return pos, quat, jac


def test_ik_max_joint_deviation_bounds_the_null_space_coordinate():
    """Directly checks the mathematically-guaranteed quantity itself: at
    the iteration where compute_ik_seeded's internal null-space clip
    fires, the resulting q_k's coordinate against THAT iteration's own
    null_space_basis (of its own j_task_k) is confined to
    [-max_dev, max_dev]. Uses ik_iterations=2 specifically: iteration 1 is
    always a no-op for this clip (q_k starts at q_rest exactly, and a
    fresh min-norm step has zero null-space component by construction --
    see _toy_fk_rotating_null_space's docstring), so the first iteration
    where clipping can matter at all is iteration 2, evaluated against the
    jacobian at q_k-after-iteration-1 (itself independent of max_dev)."""
    q_rest = np.zeros(6)
    p0, quat0, jac0 = _toy_fk_rotating_null_space(q_rest)
    target = p0.copy()
    target[0] += 1.5
    selected = [0, 1, 2, 5]  # x, y, z, rz (task_dim_rz=True, default)

    def make_cfg(iterations, max_dev):
        return CartesianVelocityConfig(
            kp_x=2.0, kp_y=2.0, kp_z=2.0, kp_rot=2.0, max_lin_speed_mps=1000.0, max_ang_speed_radps=1000.0,
            reduced_task_dims=False, ik_seeded_resolution=True, ik_iterations=iterations, ik_joint_gain=4.0,
            pinv_damping=0.005, qp_task_weight=1.0e4, ik_max_joint_deviation_rad=max_dev,
        )

    def q_target_for(iterations, max_dev):
        cfg = make_cfg(iterations, max_dev)
        xd = compute_ik_seeded(cfg, _toy_fk_rotating_null_space, q_rest, target, quat0, q_rest)
        # q_current == q_rest here, so q_target = qd_joint/ik_joint_gain directly.
        qd_joint = np.linalg.pinv(jac0) @ xd
        return q_rest + qd_joint / cfg.ik_joint_gain

    # q_k after iteration 1 doesn't depend on max_dev (clipping is a no-op
    # there) -- this is the jacobian/basis iteration 2's clip actually uses.
    q_after_iter1 = q_target_for(1, None)
    _, _, jac_iter2 = _toy_fk_rotating_null_space(q_after_iter1)
    basis = null_space_basis(jac_iter2[selected, :])
    assert basis.shape[1] > 0, "test setup: expected a genuine null space"

    def null_coord(max_dev):
        q_target = q_target_for(2, max_dev)
        return basis.T @ (q_target - q_rest)

    coord_unconstrained = null_coord(None)
    coord_bound_0p05 = null_coord(0.05)
    coord_bound_0p02 = null_coord(0.02)
    assert np.any(np.abs(coord_unconstrained) > 0.05), (
        f"test setup: expected a genuine unconstrained null-space excursion, got {coord_unconstrained}"
    )
    # np.clip bounds EACH basis coordinate independently to +-max_dev, not
    # the coordinate vector's norm -- check per-component, and (since this
    # toy's coordinate 0 stays at exactly 0 regardless, see printed coords
    # in the mechanism's own derivation) that clipping is actually active
    # on the component that does move.
    assert np.all(np.abs(coord_bound_0p05) <= 0.05 + 1e-6)
    assert np.all(np.abs(coord_bound_0p02) <= 0.02 + 1e-6)
    assert np.isclose(np.max(np.abs(coord_bound_0p05)), 0.05, atol=1e-6)
    assert np.isclose(np.max(np.abs(coord_bound_0p02)), 0.02, atol=1e-6)
    assert np.linalg.norm(coord_bound_0p02) < np.linalg.norm(coord_bound_0p05) < np.linalg.norm(coord_unconstrained)


def test_ik_max_joint_deviation_preserves_path_independence():
    cfg = CartesianVelocityConfig(
        kp_x=2.0, kp_y=2.0, kp_z=2.0, kp_rot=2.0, max_lin_speed_mps=1000.0, max_ang_speed_radps=1000.0,
        reduced_task_dims=False, ik_seeded_resolution=True, ik_iterations=8, ik_joint_gain=4.0,
        pinv_damping=0.01, ik_max_joint_deviation_rad=0.2,
    )
    q_rest = np.zeros(6)
    p0, quat0, _ = _toy_fk(q_rest)
    target = p0.copy()
    target[0] += 0.05

    def q_target_from(q_current: np.ndarray) -> np.ndarray:
        ctrl = CartesianVelocityController(cfg)
        ctrl.reset_from_state(
            {"time": 0.0, "q": q_rest, "qd": np.zeros(6), "ee_pos": p0, "ee_quat": quat0, "target_x": float(p0[0])}
        )
        p_c, quat_c, jac_c = _toy_fk(q_current)
        xd = ctrl.compute(
            {
                "time": 1.0, "q": q_current, "qd": np.zeros(6), "ee_pos": p_c, "ee_quat": quat_c,
                "target_x": float(target[0]), "target_ee_pos": target, "target_ee_vel": np.zeros(3),
                "fk_jacobian_fn": _toy_fk,
            }
        )
        qd_joint = np.linalg.pinv(jac_c) @ xd
        return q_current + qd_joint / cfg.ik_joint_gain

    q_target_a = q_target_from(q_rest + np.array([0.3, -0.2, 0.1, 0.05, -0.05, 0.02]))
    q_target_b = q_target_from(q_rest + np.array([-0.4, 0.35, -0.15, 0.1, 0.08, -0.03]))
    np.testing.assert_allclose(q_target_a, q_target_b, atol=1e-9)


def test_yaml_parsing_reads_ik_max_joint_deviation_rad():
    cfg = CartesianVelocityConfig.from_controller_yaml_section(
        {"velocity_control": {"ik_max_joint_deviation_rad": 0.2}}
    )
    assert cfg.ik_max_joint_deviation_rad == 0.2


def test_yaml_parsing_defaults_ik_max_joint_deviation_rad_to_none():
    cfg = CartesianVelocityConfig.from_controller_yaml_section({})
    assert cfg.ik_max_joint_deviation_rad is None


def test_ik_seeded_resolution_zero_error_at_q_rest_when_target_is_p0():
    """Sanity check: with the target equal to p0 (no move commanded) and
    q_current already at q_rest, the command should be ~zero -- q_rest is
    already the exact solution."""
    cfg = CartesianVelocityConfig(
        kp_x=2.0, kp_y=2.0, kp_z=2.0, kp_rot=2.0, max_lin_speed_mps=1000.0, max_ang_speed_radps=1000.0,
        reduced_task_dims=False, ik_seeded_resolution=True, ik_iterations=8, ik_joint_gain=4.0, pinv_damping=0.0,
    )
    q_rest = np.zeros(6)
    p0, quat0, _ = _toy_fk(q_rest)
    ctrl = CartesianVelocityController(cfg)
    ctrl.reset_from_state(
        {"time": 0.0, "q": q_rest, "qd": np.zeros(6), "ee_pos": p0, "ee_quat": quat0, "target_x": float(p0[0])}
    )
    xd = ctrl.compute(
        {
            "time": 1.0,
            "q": q_rest,
            "qd": np.zeros(6),
            "ee_pos": p0,
            "ee_quat": quat0,
            "target_x": float(p0[0]),
            "target_ee_pos": p0,
            "target_ee_vel": np.zeros(3),
            "fk_jacobian_fn": _toy_fk,
        }
    )
    np.testing.assert_allclose(xd, np.zeros(6), atol=1e-8)


# --------------------------------------------------------------------------- #
# orientation_priority -- promote the DISABLED rotation axes to co-primary IK
# task rows in a second solve, then blend by how much position accuracy that
# promotion actually cost. Added 2026-08-06 after a direct linear-algebra
# check proved hanging_alpha_0_5's -X orientation failure lives in the TASK
# (row) space, out of reach of every null-space mechanism this controller had
# (see modes.py / config.py). Default off; when off, bit-for-bit prior
# behaviour, which is the first thing these tests pin down.
# --------------------------------------------------------------------------- #
def _smooth_falloff():
    from controller_core.cartesian_velocity_controller.math_utils import smooth_falloff

    return smooth_falloff


def test_smooth_falloff_is_exactly_one_below_the_tolerance():
    f = _smooth_falloff()
    assert f(0.0, 0.002, 0.010) == 1.0
    assert f(0.0015, 0.002, 0.010) == 1.0
    assert f(0.002, 0.002, 0.010) == 1.0
    # Sign-independent: it is a residual MAGNITUDE.
    assert f(-0.0015, 0.002, 0.010) == 1.0


def test_smooth_falloff_is_exactly_zero_above_the_falloff():
    """The exact-zero endpoint is the mechanism's no-regression guarantee:
    past this point the promoted solve is discarded outright rather than
    blended in at some small weight, so a case orientation_priority cannot
    help is not perturbed by it at all."""
    f = _smooth_falloff()
    assert f(0.010, 0.002, 0.010) == 0.0
    assert f(0.05, 0.002, 0.010) == 0.0
    assert f(1.0e9, 0.002, 0.010) == 0.0


def test_smooth_falloff_is_monotonically_decreasing_in_between():
    f = _smooth_falloff()
    xs = np.linspace(0.002, 0.010, 40)
    ys = [f(x, 0.002, 0.010) for x in xs]
    assert all(0.0 <= y <= 1.0 for y in ys)
    assert all(ys[i + 1] <= ys[i] + 1.0e-12 for i in range(len(ys) - 1))
    assert ys[0] == 1.0
    assert ys[-1] == 0.0


def test_smooth_falloff_power_shapes_the_curve():
    """power > 1 keeps the blend closer to fully-promoted over most of the
    band (the point of defaulting to 2.0) -- at the midpoint, quadratic is
    0.25 where linear is 0.50."""
    f = _smooth_falloff()
    mid = 0.006  # midpoint of [0.002, 0.010]
    assert f(mid, 0.002, 0.010, power=1.0) == pytest.approx(0.5)
    assert f(mid, 0.002, 0.010, power=2.0) == pytest.approx(0.25)
    assert f(mid, 0.002, 0.010, power=3.0) == pytest.approx(0.125)


def test_smooth_falloff_degenerate_band_is_a_step_not_a_division_by_zero():
    f = _smooth_falloff()
    assert f(0.001, 0.002, 0.002) == 1.0
    assert f(0.003, 0.002, 0.002) == 0.0
    assert f(0.003, 0.010, 0.002) == 1.0  # zero_above < full_below: still no blow-up


def _op_cfg(**kwargs):
    base = dict(
        kp_x=2.0, kp_y=2.0, kp_z=2.0, kp_rot=2.0,
        max_lin_speed_mps=1000.0, max_ang_speed_radps=1000.0,
        reduced_task_dims=False, ik_seeded_resolution=True,
        ik_iterations=8, ik_joint_gain=4.0, pinv_damping=0.05, qp_task_weight=1.0e4,
    )
    base.update(kwargs)
    return CartesianVelocityConfig(**base)


def _run_ik_seeded(cfg, fk, q_rest, q_current, target):
    p0, quat0, _ = fk(q_rest)
    ctrl = CartesianVelocityController(cfg)
    ctrl.reset_from_state(
        {"time": 0.0, "q": q_rest, "qd": np.zeros(6), "ee_pos": p0, "ee_quat": quat0, "target_x": float(p0[0])}
    )
    p_c, quat_c, _ = fk(q_current)
    return ctrl.compute(
        {
            "time": 1.0, "q": q_current, "qd": np.zeros(6), "ee_pos": p_c, "ee_quat": quat_c,
            "target_x": float(target[0]), "target_ee_pos": target, "target_ee_vel": np.zeros(3),
            "fk_jacobian_fn": fk,
        }
    )


def test_orientation_priority_defaults_off():
    cfg = CartesianVelocityConfig()
    assert cfg.orientation_priority is False
    assert cfg.orientation_priority_weight == 1.0
    assert cfg.orientation_priority_residual_tol_m == 0.0001
    assert cfg.orientation_priority_residual_falloff_m == 0.0005
    assert cfg.orientation_priority_falloff_power == 2.0


def test_orientation_priority_off_is_bit_for_bit_prior_behavior():
    """The exactness bar every mechanism in this file has been held to:
    with the flag off, the controller output must be IDENTICAL to what it
    was before the mechanism existed -- not merely close. Compared here
    against a config that also has a large weight and thresholds set, to
    prove the flag alone gates everything."""
    q_rest = np.zeros(6)
    q_current = q_rest + np.array([0.05, -0.03, 0.02, 0.01, -0.02, 0.015])
    target = _toy_fk_with_free_rx(q_rest)[0].copy()
    target[0] += 0.05

    off = _run_ik_seeded(_op_cfg(), _toy_fk_with_free_rx, q_rest, q_current, target)
    off_explicit = _run_ik_seeded(
        _op_cfg(
            orientation_priority=False, orientation_priority_weight=25.0,
            orientation_priority_residual_tol_m=10.0, orientation_priority_residual_falloff_m=20.0,
        ),
        _toy_fk_with_free_rx, q_rest, q_current, target,
    )
    np.testing.assert_array_equal(off, off_explicit)


def test_orientation_priority_zero_weight_is_bit_for_bit_prior_behavior():
    """orientation_priority_weight == 0 means "no promoted rows," which must
    short-circuit the whole second solve rather than promote rows at zero
    weight (a zero-weight term still perturbs nothing mathematically, but
    the blend would still fire -- this pins the short-circuit)."""
    q_rest = np.zeros(6)
    q_current = q_rest + np.array([0.05, -0.03, 0.02, 0.01, -0.02, 0.015])
    target = _toy_fk_with_free_rx(q_rest)[0].copy()
    target[0] += 0.05
    off = _run_ik_seeded(_op_cfg(), _toy_fk_with_free_rx, q_rest, q_current, target)
    zero_w = _run_ik_seeded(
        _op_cfg(orientation_priority=True, orientation_priority_weight=0.0),
        _toy_fk_with_free_rx, q_rest, q_current, target,
    )
    np.testing.assert_array_equal(off, zero_w)


def test_orientation_priority_is_a_no_op_when_all_rotation_axes_already_selected():
    """With task_dim_rx/ry/rz all True there are no DISABLED rotation axes
    left to promote, so the mechanism has nothing to add and must not
    perturb the solve."""
    q_rest = np.zeros(6)
    q_current = q_rest + np.array([0.05, -0.03, 0.02, 0.01, -0.02, 0.015])
    target = _toy_fk_with_free_rx(q_rest)[0].copy()
    target[0] += 0.05
    kw = dict(task_dim_rx=True, task_dim_ry=True, task_dim_rz=True)
    off = _run_ik_seeded(_op_cfg(**kw), _toy_fk_with_free_rx, q_rest, q_current, target)
    on = _run_ik_seeded(
        _op_cfg(orientation_priority=True, orientation_priority_weight=1.0, **kw),
        _toy_fk_with_free_rx, q_rest, q_current, target,
    )
    np.testing.assert_array_equal(off, on)


def test_orientation_priority_reduces_unselected_axis_rotation_in_q_target():
    """The mechanism's actual purpose: rx is outside the default task
    (task_dim_rx=False) but IS checked by the safety guard, and satisfying
    the position task through the coupled joint 3 drags it along. Promoting
    it must leave q_target with less rx excursion than the position-only
    solve does."""
    q_rest = np.zeros(6)
    q_current = q_rest.copy()
    target = _toy_fk_with_free_rx(q_rest)[0].copy()
    target[0] += 0.05

    def rx_of_q_target(cfg):
        xd = _run_ik_seeded(cfg, _toy_fk_with_free_rx, q_rest, q_current, target)
        _, _, jac_c = _toy_fk_with_free_rx(q_current)
        qd_joint = np.linalg.pinv(jac_c) @ xd
        q_target = q_current + qd_joint / cfg.ik_joint_gain
        return abs(float(q_target[3]))  # joint 3 is the one driving rx

    rx_off = rx_of_q_target(_op_cfg())
    rx_on = rx_of_q_target(
        _op_cfg(
            orientation_priority=True, orientation_priority_weight=1.0,
            orientation_priority_residual_tol_m=1.0, orientation_priority_residual_falloff_m=2.0,
        )
    )
    assert rx_off > 1.0e-6, "toy FK must actually induce rx, or this tests nothing"
    assert rx_on < rx_off


def test_orientation_priority_q_target_stays_path_independent():
    """ik_seeded_resolution's defining property must survive the new
    mechanism: the blend gate reads the PROMOTED SOLVE's own residual, never
    q_current, so q_target is still a deterministic function of (q_rest,
    p_des, quat0) alone. Scheduling on the MEASURED orientation error
    instead -- the obvious first design -- would break exactly this."""
    cfg = _op_cfg(
        orientation_priority=True, orientation_priority_weight=1.0,
        orientation_priority_residual_tol_m=1.0, orientation_priority_residual_falloff_m=2.0,
    )
    q_rest = np.zeros(6)
    target = _toy_fk(q_rest)[0].copy()
    target[0] += 0.05

    def q_target_from(q_current):
        xd = _run_ik_seeded(cfg, _toy_fk, q_rest, q_current, target)
        _, _, jac_c = _toy_fk(q_current)
        return q_current + (np.linalg.pinv(jac_c) @ xd) / cfg.ik_joint_gain

    a = q_target_from(q_rest + np.array([0.3, -0.2, 0.1, 0.05, -0.05, 0.02]))
    b = q_target_from(q_rest + np.array([-0.4, 0.35, -0.15, 0.1, 0.08, -0.03]))
    np.testing.assert_allclose(a, b, atol=1e-9)


def test_orientation_priority_falls_back_exactly_when_residual_exceeds_falloff():
    """The gate's whole point: where promoting orientation costs real
    position accuracy, the promoted solve is discarded and behaviour returns
    to the position-only solve EXACTLY. Forced here by setting the falloff
    threshold to zero, so any nonzero residual demotes."""
    q_rest = np.zeros(6)
    q_current = q_rest.copy()
    target = _toy_fk_with_free_rx(q_rest)[0].copy()
    target[0] += 0.05
    off = _run_ik_seeded(_op_cfg(), _toy_fk_with_free_rx, q_rest, q_current, target)
    demoted = _run_ik_seeded(
        _op_cfg(
            orientation_priority=True, orientation_priority_weight=1.0,
            orientation_priority_residual_tol_m=-1.0, orientation_priority_residual_falloff_m=0.0,
        ),
        _toy_fk_with_free_rx, q_rest, q_current, target,
    )
    np.testing.assert_array_equal(off, demoted)


def test_orientation_priority_yaml_round_trip():
    cfg = CartesianVelocityConfig.from_controller_yaml_section(
        {
            "velocity_control": {
                "orientation_priority": True,
                "orientation_priority_weight": 0.5,
                "orientation_priority_residual_tol_m": 0.004,
                "orientation_priority_residual_falloff_m": 0.02,
                "orientation_priority_falloff_power": 3.0,
            }
        }
    )
    assert cfg.orientation_priority is True
    assert cfg.orientation_priority_weight == pytest.approx(0.5)
    assert cfg.orientation_priority_residual_tol_m == pytest.approx(0.004)
    assert cfg.orientation_priority_residual_falloff_m == pytest.approx(0.02)
    assert cfg.orientation_priority_falloff_power == pytest.approx(3.0)


def test_orientation_priority_absent_from_yaml_defaults_off():
    cfg = CartesianVelocityConfig.from_controller_yaml_section({"velocity_control": {}})
    assert cfg.orientation_priority is False


# --------------------------------------------------------------------------- #
# singularity_velocity_scaling (added 2026-08-07) -- throttles the commanded
# task velocity/error itself as the FULL 6x6 Jacobian approaches a
# singularity (sigma_min -> 0), rather than only damping some later
# inversion of a fixed-magnitude command. See config.py's docstring for the
# full rationale, including the measured evidence for why this keys off the
# full Jacobian rather than J_task's own (well-conditioned at both
# documented spikes) submatrix.
# --------------------------------------------------------------------------- #


def _singularity_speed_scale():
    from controller_core.cartesian_velocity_controller.math_utils import singularity_speed_scale

    return singularity_speed_scale


def test_singularity_speed_scale_is_one_at_or_above_full_speed_threshold():
    f = _singularity_speed_scale()
    assert f(0.03, 0.003, 0.03) == pytest.approx(1.0)
    assert f(0.05, 0.003, 0.03) == pytest.approx(1.0)
    assert f(1.0e9, 0.003, 0.03) == pytest.approx(1.0)


def test_singularity_speed_scale_is_zero_at_or_below_stop_threshold():
    f = _singularity_speed_scale()
    assert f(0.003, 0.003, 0.03) == pytest.approx(0.0)
    assert f(0.001, 0.003, 0.03) == pytest.approx(0.0)
    assert f(0.0, 0.003, 0.03) == pytest.approx(0.0)


def test_singularity_speed_scale_is_monotonically_increasing_with_sigma_min():
    f = _singularity_speed_scale()
    xs = np.linspace(0.003, 0.03, 40)
    ys = [f(x, 0.003, 0.03) for x in xs]
    assert all(0.0 <= y <= 1.0 for y in ys)
    assert all(ys[i + 1] >= ys[i] - 1.0e-12 for i in range(len(ys) - 1))
    assert ys[0] == pytest.approx(0.0)
    assert ys[-1] == pytest.approx(1.0)


def test_singularity_speed_scale_stays_in_unit_interval_for_out_of_range_inputs():
    f = _singularity_speed_scale()
    assert 0.0 <= f(-1.0, 0.003, 0.03) <= 1.0  # a negative sigma_min should never occur, but must not blow up
    assert 0.0 <= f(100.0, 0.003, 0.03) <= 1.0


def test_singularity_velocity_scaling_defaults_off():
    cfg = CartesianVelocityConfig()
    assert cfg.singularity_velocity_scaling is False
    assert cfg.singularity_sigma_min_stop == pytest.approx(0.003)
    assert cfg.singularity_sigma_min_full_speed == pytest.approx(0.03)
    assert cfg.singularity_scale_power == pytest.approx(2.0)


def test_singularity_velocity_scaling_off_is_bit_for_bit_prior_behavior():
    """The exactness bar every mechanism in this file is held to: with the
    flag off, output must be IDENTICAL to before the mechanism existed --
    not merely close. Uses a jacobian that is deliberately near-singular
    (so the mechanism would visibly do SOMETHING if it were mistakenly
    active) to make sure the flag, not merely a lucky sigma_min, is what
    gates it off."""
    q_rest = np.zeros(6)
    q_current = q_rest + np.array([0.05, -0.03, 0.02, 0.01, -0.02, 0.015])
    target = _toy_fk_with_free_rx(q_rest)[0].copy()
    target[0] += 0.05

    off = _run_ik_seeded(_op_cfg(), _toy_fk_with_free_rx, q_rest, q_current, target)
    off_explicit = _run_ik_seeded(
        _op_cfg(
            singularity_velocity_scaling=False,
            singularity_sigma_min_stop=0.5, singularity_sigma_min_full_speed=0.9,
        ),
        _toy_fk_with_free_rx, q_rest, q_current, target,
    )
    np.testing.assert_array_equal(off, off_explicit)


def test_singularity_velocity_scaling_reduces_command_near_a_singular_jacobian():
    """The mechanism's actual purpose: a toy FK whose Jacobian's smallest
    singular value sits inside the throttle band must produce a strictly
    smaller-magnitude command with the mechanism on than off."""

    def _toy_fk_near_singular(q: np.ndarray):
        q = np.asarray(q, dtype=np.float64).reshape(6)
        pos = q[0:3].copy()
        quat = rotvec_to_quat_wxyz(0.1 * q[3:6])
        jac = np.eye(6)
        jac[2, 2] = 0.01  # one row nearly degenerate -> full-J sigma_min small
        return pos, quat, jac

    q_rest = np.zeros(6)
    q_current = q_rest.copy()
    target = _toy_fk_near_singular(q_rest)[0].copy()
    target[0] += 0.05
    target[2] += 0.05

    off = _run_ik_seeded(_op_cfg(), _toy_fk_near_singular, q_rest, q_current, target)
    on = _run_ik_seeded(
        _op_cfg(
            singularity_velocity_scaling=True,
            singularity_sigma_min_stop=0.001, singularity_sigma_min_full_speed=0.5,
        ),
        _toy_fk_near_singular, q_rest, q_current, target,
    )
    assert float(np.linalg.norm(on)) < float(np.linalg.norm(off))


def test_singularity_velocity_scaling_matches_full_speed_far_from_singularity():
    """Conversely, a comfortably well-conditioned Jacobian (sigma_min above
    the full-speed threshold everywhere the solve visits) must reproduce the
    mechanism-off command exactly -- not merely closely -- since scale is
    EXACTLY 1.0 throughout. Uses _toy_fk (near-identity, well-conditioned),
    NOT _toy_fk_with_free_rx -- that toy is DELIBERATELY singular in the
    full Jacobian (jac[4,4]=0.0, "ry unreachable... on purpose" per its own
    docstring), so it is the wrong fixture for a "far from singularity"
    check (confirmed directly: the mechanism correctly collapses the
    command to zero there, which is the CORRECT behavior for a genuinely
    singular Jacobian, not a bug in this test's premise)."""
    q_rest = np.zeros(6)
    q_current = q_rest + np.array([0.05, -0.03, 0.02, 0.01, -0.02, 0.015])
    target = _toy_fk(q_rest)[0].copy()
    target[0] += 0.05

    off = _run_ik_seeded(_op_cfg(), _toy_fk, q_rest, q_current, target)
    on = _run_ik_seeded(
        _op_cfg(
            singularity_velocity_scaling=True,
            singularity_sigma_min_stop=1.0e-6, singularity_sigma_min_full_speed=1.0e-5,
        ),
        _toy_fk, q_rest, q_current, target,
    )
    np.testing.assert_array_equal(off, on)


def test_singularity_velocity_scaling_yaml_round_trip():
    cfg = CartesianVelocityConfig.from_controller_yaml_section(
        {
            "velocity_control": {
                "singularity_velocity_scaling": True,
                "singularity_sigma_min_stop": 0.001,
                "singularity_sigma_min_full_speed": 0.02,
                "singularity_scale_power": 3.0,
            }
        }
    )
    assert cfg.singularity_velocity_scaling is True
    assert cfg.singularity_sigma_min_stop == pytest.approx(0.001)
    assert cfg.singularity_sigma_min_full_speed == pytest.approx(0.02)
    assert cfg.singularity_scale_power == pytest.approx(3.0)


def test_singularity_velocity_scaling_absent_from_yaml_defaults_off():
    cfg = CartesianVelocityConfig.from_controller_yaml_section({"velocity_control": {}})
    assert cfg.singularity_velocity_scaling is False


# --------------------------------------------------------------------------- #
# singularity_windup_clamp_rad (added 2026-08-07) -- anti-windup fix for
# singularity_velocity_scaling's own throttle-then-release dynamic: q_target
# is recomputed fresh from (q_rest, p_des) every cycle regardless of
# throttling, so the gap (q_target - q_current) can grow unboundedly while
# scale_current < 1.0, then release a large spike the instant scale_current
# recovers toward 1.0. See config.py for the full rationale and the measured
# 128-cell grid evidence (105/128 tie -> 111/128, zero new regressions, at
# clamp=0.03 rad gated on scale_current < 1.0).
# --------------------------------------------------------------------------- #


def _toy_fk_near_singular_const(q: np.ndarray):
    """Jacobian is near-singular (jac[2,2]=0.01) for EVERY q -- unlike a
    real robot, this toy fixture guarantees scale_current < 1.0 regardless
    of q_current, which is exactly what's needed to test the windup
    clamp's "actively throttled" gate in isolation, independent of any
    particular pose."""
    q = np.asarray(q, dtype=np.float64).reshape(6)
    pos = q[0:3].copy()
    quat = rotvec_to_quat_wxyz(0.1 * q[3:6])
    jac = np.eye(6)
    jac[2, 2] = 0.01
    return pos, quat, jac


def test_singularity_windup_clamp_defaults_off():
    cfg = CartesianVelocityConfig()
    assert cfg.singularity_windup_clamp_rad is None


def test_singularity_windup_clamp_off_is_bit_for_bit_prior_behavior():
    """With singularity_velocity_scaling on but singularity_windup_clamp_rad
    left at its default (None), output must be IDENTICAL to a config that
    never mentions the field at all -- the exactness bar every mechanism in
    this file is held to."""
    q_rest = np.zeros(6)
    q_current = q_rest + np.array([0.3, -0.03, 0.02, 0.01, -0.02, 0.015])
    target = _toy_fk_near_singular_const(q_rest)[0].copy()
    target[0] += 0.05
    target[2] += 0.05

    kw = dict(
        singularity_velocity_scaling=True,
        singularity_sigma_min_stop=0.001, singularity_sigma_min_full_speed=0.5,
    )
    default = _run_ik_seeded(_op_cfg(**kw), _toy_fk_near_singular_const, q_rest, q_current, target)
    explicit_off = _run_ik_seeded(
        _op_cfg(singularity_windup_clamp_rad=None, **kw),
        _toy_fk_near_singular_const, q_rest, q_current, target,
    )
    np.testing.assert_array_equal(default, explicit_off)


def test_singularity_windup_clamp_reduces_command_when_actively_throttled():
    """The mechanism's actual purpose: with the arm actively throttled
    (scale_current < 1.0, guaranteed by _toy_fk_near_singular_const) and a
    large gap between q_target and q_current (simulating the windup this
    mechanism targets -- q_current deliberately far from where the Newton
    solve places q_target), clamping the gap before it is multiplied by
    ik_joint_gain must produce a strictly smaller-magnitude command than
    leaving it unclamped."""
    q_rest = np.zeros(6)
    q_current = q_rest + np.array([0.4, 0.0, 0.0, 0.0, 0.0, 0.0])  # large gap, well-conditioned direction
    target = _toy_fk_near_singular_const(q_rest)[0].copy()
    target[0] += 0.05

    kw = dict(
        singularity_velocity_scaling=True,
        singularity_sigma_min_stop=0.001, singularity_sigma_min_full_speed=0.5,
    )
    no_clamp = _run_ik_seeded(_op_cfg(**kw), _toy_fk_near_singular_const, q_rest, q_current, target)
    clamped = _run_ik_seeded(
        _op_cfg(singularity_windup_clamp_rad=0.03, **kw),
        _toy_fk_near_singular_const, q_rest, q_current, target,
    )
    assert float(np.linalg.norm(clamped)) < float(np.linalg.norm(no_clamp))


def test_singularity_windup_clamp_inactive_when_singularity_velocity_scaling_off():
    """The clamp is purpose-built anti-windup for singularity_velocity_
    scaling's own dynamic -- with that flag off, setting singularity_
    windup_clamp_rad must have ZERO effect, even with a huge gap that
    would clearly bind if the clamp were active."""
    q_rest = np.zeros(6)
    q_current = q_rest + np.array([0.4, 0.0, 0.0, 0.0, 0.0, 0.0])
    target = _toy_fk_near_singular_const(q_rest)[0].copy()
    target[0] += 0.05

    off = _run_ik_seeded(_op_cfg(), _toy_fk_near_singular_const, q_rest, q_current, target)
    off_with_clamp_set = _run_ik_seeded(
        _op_cfg(singularity_velocity_scaling=False, singularity_windup_clamp_rad=0.001),
        _toy_fk_near_singular_const, q_rest, q_current, target,
    )
    np.testing.assert_array_equal(off, off_with_clamp_set)


def test_singularity_windup_clamp_inactive_when_not_actively_throttled():
    """Real regression this gating fixes (see config.py's docstring): a
    first version of this mechanism clamped whenever the flag was on,
    regardless of whether this cycle was actually throttled -- which was
    measured to also bind on ordinary, never-near-a-singularity moves with
    a naturally large gap, introducing new guard regressions unrelated to
    the windup this mechanism targets. Using _toy_fk (well-conditioned
    everywhere) with a loose enough threshold that scale_current is
    EXACTLY 1.0, a huge gap, and a tiny clamp (0.001 rad, which would
    obviously bind if active) must still reproduce the mechanism-on-no-
    clamp output exactly -- the clamp must not engage at all when this
    cycle isn't actively throttled."""
    q_rest = np.zeros(6)
    q_current = q_rest + np.array([0.4, -0.3, 0.2, 0.1, -0.2, 0.15])
    target = _toy_fk(q_rest)[0].copy()
    target[0] += 0.05

    kw = dict(
        singularity_velocity_scaling=True,
        singularity_sigma_min_stop=1.0e-6, singularity_sigma_min_full_speed=1.0e-5,
    )
    no_clamp = _run_ik_seeded(_op_cfg(**kw), _toy_fk, q_rest, q_current, target)
    tiny_clamp = _run_ik_seeded(
        _op_cfg(singularity_windup_clamp_rad=0.001, **kw),
        _toy_fk, q_rest, q_current, target,
    )
    np.testing.assert_array_equal(no_clamp, tiny_clamp)


def test_singularity_windup_clamp_rad_yaml_round_trip():
    cfg = CartesianVelocityConfig.from_controller_yaml_section(
        {
            "velocity_control": {
                "singularity_velocity_scaling": True,
                "singularity_windup_clamp_rad": 0.03,
            }
        }
    )
    assert cfg.singularity_velocity_scaling is True
    assert cfg.singularity_windup_clamp_rad == pytest.approx(0.03)


def test_singularity_windup_clamp_rad_absent_from_yaml_defaults_off():
    cfg = CartesianVelocityConfig.from_controller_yaml_section({"velocity_control": {}})
    assert cfg.singularity_windup_clamp_rad is None


# --------------------------------------------------------------------------- #
# ik_joint_gain_step_scaling (added 2026-08-07) -- reduces the EFFECTIVE
# ik_joint_gain as ||q_target - q_current|| grows, a genuinely different
# failure mechanism from singularity_velocity_scaling: measured directly
# (see config.py and this package's status doc), the 11 real cells this
# targets (neg40_wrist2offset/neg45_wrist2offset at the largest tested +X
# displacements) keep sigma_min(J_full) at/above singularity_sigma_min_
# full_speed for the entire episode in 10/11 of them -- singularity_
# velocity_scaling never engages there at all, so no amount of retuning it
# could help; the guard trips are large-step overshoots at a fixed high
# gain, independent of Jacobian conditioning.
# --------------------------------------------------------------------------- #


def _ik_joint_gain_step_scale():
    from controller_core.cartesian_velocity_controller.math_utils import ik_joint_gain_step_scale

    return ik_joint_gain_step_scale


def test_ik_joint_gain_step_scale_is_one_at_or_below_full_below():
    f = _ik_joint_gain_step_scale()
    assert f(0.05, 0.15, 1.0, 0.15) == 1.0
    assert f(0.15, 0.15, 1.0, 0.15) == 1.0
    assert f(0.0, 0.15, 1.0, 0.15) == 1.0


def test_ik_joint_gain_step_scale_is_exactly_min_scale_at_or_above_zero_above():
    f = _ik_joint_gain_step_scale()
    assert f(1.0, 0.15, 1.0, 0.15) == pytest.approx(0.15)
    assert f(5.0, 0.15, 1.0, 0.15) == pytest.approx(0.15)


def test_ik_joint_gain_step_scale_is_monotonically_decreasing_in_between():
    f = _ik_joint_gain_step_scale()
    xs = np.linspace(0.15, 1.0, 25)
    ys = [f(float(x), 0.15, 1.0, 0.15) for x in xs]
    assert all(a >= b - 1e-12 for a, b in zip(ys, ys[1:]))
    assert ys[0] == pytest.approx(1.0)
    assert ys[-1] == pytest.approx(0.15)


def test_ik_joint_gain_step_scale_never_drops_below_min_scale():
    """The property that distinguishes this from smooth_falloff's own hard
    zero floor: even far past zero_above, or with pathological inputs, the
    scale must never go below min_scale -- a full freeze of the joint-space
    P-law is a much stronger intervention than this mechanism is meant to
    apply (see math_utils.py's docstring)."""
    f = _ik_joint_gain_step_scale()
    for gap in (1.0, 10.0, 1000.0, 1.0e6):
        s = f(gap, 0.15, 1.0, 0.15)
        assert 0.15 - 1e-12 <= s <= 1.0


def test_ik_joint_gain_step_scale_zero_min_scale_matches_smooth_falloff():
    """min_scale=0.0 collapses to a bare smooth_falloff weight -- confirms
    the floor term is additive, not a separate code path that could drift
    out of sync with smooth_falloff's own shape."""
    from controller_core.cartesian_velocity_controller.math_utils import smooth_falloff

    f = _ik_joint_gain_step_scale()
    for gap in (0.05, 0.2, 0.5, 0.9, 1.5):
        assert f(gap, 0.15, 1.0, 0.0) == pytest.approx(smooth_falloff(gap, 0.15, 1.0, 2.0))


def test_ik_joint_gain_step_scale_degenerate_band_is_a_step_not_a_division_by_zero():
    f = _ik_joint_gain_step_scale()
    assert f(0.05, 1.0, 0.2, 0.15) == 1.0  # below the (degenerate) full_below -> full scale
    assert f(2.0, 1.0, 0.2, 0.15) == pytest.approx(0.15)  # above it -> floor, no ZeroDivisionError


def test_ik_joint_gain_step_scaling_defaults_off():
    cfg = CartesianVelocityConfig()
    assert cfg.ik_joint_gain_step_scaling is False
    assert cfg.ik_joint_gain_step_full_below_rad == pytest.approx(0.10)
    assert cfg.ik_joint_gain_step_zero_above_rad == pytest.approx(0.30)
    assert cfg.ik_joint_gain_step_min_scale == pytest.approx(0.15)
    assert cfg.ik_joint_gain_step_falloff_power == pytest.approx(2.0)


def test_ik_joint_gain_step_scaling_off_is_bit_for_bit_prior_behavior():
    """The exactness bar every mechanism in this file is held to: with the
    flag off, output must be IDENTICAL to before the mechanism existed --
    not merely close. Uses a deliberately large q_current offset (a large
    gap, so the mechanism would visibly do SOMETHING if mistakenly active)
    to make sure the flag, not merely a lucky gap size, is what gates it
    off."""
    q_rest = np.zeros(6)
    q_current = q_rest + np.array([0.5, -0.3, 0.2, 0.1, -0.2, 0.15])
    target = _toy_fk(q_rest)[0].copy()
    target[0] += 0.05

    off = _run_ik_seeded(_op_cfg(), _toy_fk, q_rest, q_current, target)
    off_explicit = _run_ik_seeded(
        _op_cfg(
            ik_joint_gain_step_scaling=False,
            ik_joint_gain_step_full_below_rad=0.001, ik_joint_gain_step_zero_above_rad=0.002,
            ik_joint_gain_step_min_scale=0.0,
        ),
        _toy_fk, q_rest, q_current, target,
    )
    np.testing.assert_array_equal(off, off_explicit)


def test_ik_joint_gain_step_scaling_matches_full_gain_for_a_small_step():
    """A comfortably small gap (below full_below throughout) must reproduce
    the mechanism-off command exactly -- not merely closely -- since
    step_scale is EXACTLY 1.0 the whole solve."""
    q_rest = np.zeros(6)
    q_current = q_rest + np.array([0.001, -0.0005, 0.0002, 0.0001, -0.0002, 0.00015])
    target = _toy_fk(q_rest)[0].copy()
    target[0] += 0.001

    off = _run_ik_seeded(_op_cfg(), _toy_fk, q_rest, q_current, target)
    on = _run_ik_seeded(
        _op_cfg(
            ik_joint_gain_step_scaling=True,
            ik_joint_gain_step_full_below_rad=1.0, ik_joint_gain_step_zero_above_rad=2.0,
            ik_joint_gain_step_min_scale=0.15,
        ),
        _toy_fk, q_rest, q_current, target,
    )
    np.testing.assert_array_equal(off, on)


def test_ik_joint_gain_step_scaling_reduces_command_for_a_large_step():
    """The mechanism's actual purpose: a large gap (deliberately past
    zero_above) must produce a strictly smaller-magnitude command with the
    mechanism on than off, at a Jacobian that is NOT near-singular (_toy_fk)
    -- confirming this triggers on step SIZE alone, not on conditioning."""
    q_rest = np.zeros(6)
    q_current = q_rest + np.array([0.6, -0.4, 0.3, 0.2, -0.1, 0.25])
    target = _toy_fk(q_rest)[0].copy()
    target[0] += 0.05

    off = _run_ik_seeded(_op_cfg(), _toy_fk, q_rest, q_current, target)
    on = _run_ik_seeded(
        _op_cfg(
            ik_joint_gain_step_scaling=True,
            ik_joint_gain_step_full_below_rad=0.15, ik_joint_gain_step_zero_above_rad=1.0,
            ik_joint_gain_step_min_scale=0.15,
        ),
        _toy_fk, q_rest, q_current, target,
    )
    assert float(np.linalg.norm(on)) < float(np.linalg.norm(off))


def test_ik_joint_gain_step_scaling_composes_exactly_with_windup_clamp():
    """Applied AFTER singularity_windup_clamp_rad's own clip -- verified by
    reconstructing the expected command by hand from the same two-step
    pipeline (clamp the gap, then scale it) and comparing exactly, not just
    checking a direction of change."""
    q_rest = np.zeros(6)
    q_current = q_rest + np.array([0.4, 0.0, 0.0, 0.0, 0.0, 0.0])
    target = _toy_fk_near_singular_const(q_rest)[0].copy()
    target[0] += 0.05

    kw = dict(
        singularity_velocity_scaling=True,
        singularity_sigma_min_stop=0.001, singularity_sigma_min_full_speed=0.5,
        singularity_windup_clamp_rad=0.03,
        ik_joint_gain_step_scaling=True,
        ik_joint_gain_step_full_below_rad=0.01, ik_joint_gain_step_zero_above_rad=0.02,
        ik_joint_gain_step_min_scale=0.2,
    )
    combined = _run_ik_seeded(_op_cfg(**kw), _toy_fk_near_singular_const, q_rest, q_current, target)

    # windup-clamp-only reference, then hand-scale it by the expected
    # step_scale for a fully-clamped (0.03 rad) gap under this band, i.e.
    # min_scale exactly (0.03 > zero_above=0.02).
    clamp_only = _run_ik_seeded(
        _op_cfg(
            singularity_velocity_scaling=True,
            singularity_sigma_min_stop=0.001, singularity_sigma_min_full_speed=0.5,
            singularity_windup_clamp_rad=0.03,
        ),
        _toy_fk_near_singular_const, q_rest, q_current, target,
    )
    np.testing.assert_allclose(combined, clamp_only * 0.2, rtol=1e-9, atol=1e-12)


def test_ik_joint_gain_step_scaling_inactive_without_ik_seeded_resolution_is_not_applicable():
    """ik_joint_gain_step_scaling is only read inside compute_ik_seeded --
    confirms the flag has zero effect on any other mode (reduced_task_dims
    here), the same isolation every mode-specific mechanism in this file
    is held to."""
    cfg_off = CartesianVelocityConfig(
        reduced_task_dims=True, kp_x=2.0, kp_y=2.0, kp_z=2.0, kp_rot=2.0,
        max_lin_speed_mps=1000.0, max_ang_speed_radps=1000.0, kp_posture=1.0, pinv_damping=0.01,
    )
    cfg_on = CartesianVelocityConfig(
        reduced_task_dims=True, kp_x=2.0, kp_y=2.0, kp_z=2.0, kp_rot=2.0,
        max_lin_speed_mps=1000.0, max_ang_speed_radps=1000.0, kp_posture=1.0, pinv_damping=0.01,
        ik_joint_gain_step_scaling=True, ik_joint_gain_step_full_below_rad=0.001,
        ik_joint_gain_step_zero_above_rad=0.002, ik_joint_gain_step_min_scale=0.0,
    )
    q = np.array([0.1, -0.05, 0.02, 0.01, -0.02, 0.015])
    _, _, jac = _toy_fk(q)
    p, quat, _ = _toy_fk(q)
    target = p.copy()
    target[0] += 0.05

    def _compute(cfg):
        ctrl = CartesianVelocityController(cfg)
        ctrl.reset_from_state({"time": 0.0, "q": q, "qd": np.zeros(6), "ee_pos": p, "ee_quat": quat, "target_x": float(p[0])})
        return ctrl.compute(
            {
                "time": 1.0, "q": q, "qd": np.zeros(6), "ee_pos": p, "ee_quat": quat,
                "target_x": float(target[0]), "target_ee_pos": target, "target_ee_vel": np.zeros(3),
                "jacobian": jac,
            }
        )

    np.testing.assert_array_equal(_compute(cfg_off), _compute(cfg_on))


def test_ik_joint_gain_step_scaling_yaml_round_trip():
    cfg = CartesianVelocityConfig.from_controller_yaml_section(
        {
            "velocity_control": {
                "ik_joint_gain_step_scaling": True,
                "ik_joint_gain_step_full_below_rad": 0.2,
                "ik_joint_gain_step_zero_above_rad": 1.5,
                "ik_joint_gain_step_min_scale": 0.1,
                "ik_joint_gain_step_falloff_power": 3.0,
            }
        }
    )
    assert cfg.ik_joint_gain_step_scaling is True
    assert cfg.ik_joint_gain_step_full_below_rad == pytest.approx(0.2)
    assert cfg.ik_joint_gain_step_zero_above_rad == pytest.approx(1.5)
    assert cfg.ik_joint_gain_step_min_scale == pytest.approx(0.1)
    assert cfg.ik_joint_gain_step_falloff_power == pytest.approx(3.0)


def test_ik_joint_gain_step_scaling_absent_from_yaml_defaults_off():
    cfg = CartesianVelocityConfig.from_controller_yaml_section({"velocity_control": {}})
    assert cfg.ik_joint_gain_step_scaling is False
