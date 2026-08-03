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
