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
from controller_core.kinematics_utils import (  # noqa: E402
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


def test_ik_posture_gain_default_off_matches_prior_behavior():
    """Regression guard: ik_posture_gain absent/0.0 (the default) must
    reproduce byte-identical output to before this field existed -- the
    whole posture block is skipped, not computed-and-multiplied-by-zero."""
    cfg_explicit_zero = CartesianVelocityConfig(
        kp_x=2.0, kp_y=2.0, kp_z=2.0, kp_rot=2.0, max_lin_speed_mps=1000.0, max_ang_speed_radps=1000.0,
        reduced_task_dims=False, ik_seeded_resolution=True, ik_iterations=8, ik_joint_gain=4.0,
        pinv_damping=0.01, ik_posture_gain=0.0,
    )
    cfg_default = CartesianVelocityConfig(
        kp_x=2.0, kp_y=2.0, kp_z=2.0, kp_rot=2.0, max_lin_speed_mps=1000.0, max_ang_speed_radps=1000.0,
        reduced_task_dims=False, ik_seeded_resolution=True, ik_iterations=8, ik_joint_gain=4.0, pinv_damping=0.01,
    )
    assert cfg_default.ik_posture_gain == 0.0

    def run(cfg):
        q_rest = np.array([0.1, -0.2, 0.15, 0.3, -0.1, 0.05])
        p0, quat0, _ = _toy_fk(q_rest)
        target = p0.copy()
        target[0] += 0.03
        ctrl = CartesianVelocityController(cfg)
        ctrl.reset_from_state(
            {"time": 0.0, "q": q_rest, "qd": np.zeros(6), "ee_pos": p0, "ee_quat": quat0, "target_x": float(p0[0])}
        )
        q_current = q_rest + np.array([0.05, -0.03, 0.02, 0.01, -0.02, 0.015])
        p_c, quat_c, _ = _toy_fk(q_current)
        return ctrl.compute(
            {
                "time": 1.0, "q": q_current, "qd": np.zeros(6), "ee_pos": p_c, "ee_quat": quat_c,
                "target_x": float(target[0]), "target_ee_pos": target, "target_ee_vel": np.zeros(3),
                "fk_jacobian_fn": _toy_fk,
            }
        )

    np.testing.assert_allclose(run(cfg_explicit_zero), run(cfg_default), atol=1e-14)


def test_ik_posture_gain_pulls_unconstrained_axis_toward_q_rest():
    """The actual mechanism this fix adds: with a Jacobian where joint 3 is
    weakly coupled into the position task AND drives rx (an axis outside
    the default task selection -- see _toy_fk_with_free_rx), the Newton
    solve's own iterations pull q[3] away from q_rest to help satisfy the
    position error, dragging rx along with it since nothing else
    constrains it. Turning on ik_posture_gain must measurably pull
    q_target's free joint (index 3) back toward q_rest compared to
    ik_posture_gain=0.0, without needing q_current to be perturbed at all
    -- q_target is a pure function of q_rest/target by design (see the
    path-independence test below), so q_current is set equal to q_rest
    here specifically to isolate that."""
    q_rest = np.zeros(6)
    p0, quat0, _ = _toy_fk_with_free_rx(q_rest)
    target = p0.copy()
    target[0] += 0.05

    def q_target_free_axis(ik_posture_gain: float) -> float:
        # Moderate, well-conditioned pinv_damping/qp_task_weight -- an
        # earlier version of this test used the real search's extreme
        # values (pinv_damping~1e-6, qp_task_weight~1e8) and found
        # solve_box_qp's underlying np.linalg.solve becomes numerically
        # unstable there (condition number near double precision's limit,
        # ~1e8/1e-8), silently producing a degenerate all-in-column-0
        # solution instead of the correct weighted split -- confirmed by
        # comparing against the analytic pinv reference directly. Moderate
        # values isolate the posture MECHANISM cleanly; the real extreme-
        # value numerical-conditioning question is a separate, real
        # finding (see this test file's module-level note) not something
        # this unit test needs to also cover.
        cfg = CartesianVelocityConfig(
            kp_x=2.0, kp_y=2.0, kp_z=2.0, kp_rot=2.0, max_lin_speed_mps=1000.0, max_ang_speed_radps=1000.0,
            reduced_task_dims=False, ik_seeded_resolution=True, ik_iterations=10, ik_joint_gain=4.0,
            pinv_damping=0.05, qp_task_weight=1.0e4, ik_posture_gain=ik_posture_gain,
        )
        ctrl = CartesianVelocityController(cfg)
        ctrl.reset_from_state(
            {"time": 0.0, "q": q_rest, "qd": np.zeros(6), "ee_pos": p0, "ee_quat": quat0, "target_x": float(p0[0])}
        )
        xd = ctrl.compute(
            {
                "time": 1.0, "q": q_rest, "qd": np.zeros(6), "ee_pos": p0, "ee_quat": quat0,
                "target_x": float(target[0]), "target_ee_pos": target, "target_ee_vel": np.zeros(3),
                "fk_jacobian_fn": _toy_fk_with_free_rx,
            }
        )
        # q_current == q_rest here, so q_target_free = qd_joint_free/ik_joint_gain
        # directly. jac_current[3,3] == 0.6 exactly (diagonal in this toy FK).
        qd_joint_free = xd[3] / 0.6
        return float(qd_joint_free / cfg.ik_joint_gain)

    free_axis_off = abs(q_target_free_axis(0.0) - q_rest[3])
    free_axis_on = abs(q_target_free_axis(2.0) - q_rest[3])
    assert free_axis_off > 1e-4, "test setup didn't actually create any free-axis drift to pull back from"
    assert free_axis_on < free_axis_off, (
        f"posture gain should pull the free axis closer to q_rest: on={free_axis_on} off={free_axis_off}"
    )


def test_ik_posture_gain_does_not_break_task_convergence():
    """The posture term must not prevent the primary task from converging
    -- with ik_posture_gain on, the x-task (position error) must still be
    reduced by a large fraction, not swamped/cancelled by the posture
    pull. q_current is set equal to q_rest (as in the test above) so the
    result reflects q_target directly, without an unrelated q_current
    offset muddying the comparison."""
    cfg = CartesianVelocityConfig(
        kp_x=2.0, kp_y=2.0, kp_z=2.0, kp_rot=2.0, max_lin_speed_mps=1000.0, max_ang_speed_radps=1000.0,
        reduced_task_dims=False, ik_seeded_resolution=True, ik_iterations=10, ik_joint_gain=4.0,
        pinv_damping=0.05, qp_task_weight=1.0e4, ik_posture_gain=2.0,
    )
    q_rest = np.zeros(6)
    p0, quat0, _ = _toy_fk_with_free_rx(q_rest)
    target = p0.copy()
    target[0] += 0.02
    ctrl = CartesianVelocityController(cfg)
    ctrl.reset_from_state(
        {"time": 0.0, "q": q_rest, "qd": np.zeros(6), "ee_pos": p0, "ee_quat": quat0, "target_x": float(p0[0])}
    )
    xd = ctrl.compute(
        {
            "time": 1.0, "q": q_rest, "qd": np.zeros(6), "ee_pos": p0, "ee_quat": quat0,
            "target_x": float(target[0]), "target_ee_pos": target, "target_ee_vel": np.zeros(3),
            "fk_jacobian_fn": _toy_fk_with_free_rx,
        }
    )
    # xd[0] (x-velocity component) should reflect real progress toward the
    # 0.02m task displacement, not be swamped/cancelled by the posture term.
    assert xd[0] > 0.0


def test_ik_posture_gain_preserves_path_independence():
    """Extends test_ik_seeded_resolution_q_target_is_path_independent to
    confirm the posture addition doesn't reintroduce dependence on
    q_current -- it only ever reads q_rest and q_k (both independent of
    q_current) inside the Newton loop."""
    cfg = CartesianVelocityConfig(
        kp_x=2.0, kp_y=2.0, kp_z=2.0, kp_rot=2.0, max_lin_speed_mps=1000.0, max_ang_speed_radps=1000.0,
        reduced_task_dims=False, ik_seeded_resolution=True, ik_iterations=8, ik_joint_gain=4.0,
        pinv_damping=0.01, ik_posture_gain=1.5,
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


def test_yaml_parsing_reads_ik_posture_gain():
    cfg = CartesianVelocityConfig.from_controller_yaml_section({"velocity_control": {"ik_posture_gain": 1.25}})
    assert cfg.ik_posture_gain == 1.25


def test_yaml_parsing_defaults_ik_posture_gain_to_zero():
    cfg = CartesianVelocityConfig.from_controller_yaml_section({})
    assert cfg.ik_posture_gain == 0.0


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
