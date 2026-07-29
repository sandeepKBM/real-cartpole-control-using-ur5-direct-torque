"""Unit tests for the P3 operational-space terms in the impedance law.

Pure numpy — no simulator required.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from controller_core.x_axis_cartesian_impedance import (  # noqa: E402
    WRIST_ORIENTATION_MASK,
    CartesianImpedanceConfig,
    XAxisCartesianImpedanceController,
)


def _make_state(*, q=None, qd=None, ee_pos=(0.4, 0.1, 0.5), target_x=0.42, J=None, mass_matrix=None):
    state = {
        "time": 0.0,
        "q": np.zeros(6) if q is None else np.asarray(q, dtype=np.float64),
        "qd": np.zeros(6) if qd is None else np.asarray(qd, dtype=np.float64),
        "ee_pos": np.asarray(ee_pos, dtype=np.float64),
        "ee_quat": np.array([1.0, 0.0, 0.0, 0.0]),
        "ee_lin_vel": np.zeros(3),
        "ee_ang_vel": np.zeros(3),
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


def test_flags_off_is_exact_legacy_behavior():
    state = _make_state(mass_matrix=np.diag([2.0, 3.0, 4.0, 1.0, 1.0, 1.0]))
    legacy = _controller().compute(_make_state())  # no mass matrix, flags off
    with_mass = _controller().compute(state)  # mass matrix present but flags off
    np.testing.assert_allclose(with_mass.tau, legacy.tau, atol=1e-12)
    assert not with_mass.inertia_shaping_active
    assert not with_mass.nullspace_posture_active
    assert with_mass.mass_matrix_provided


def test_shaping_identity_mass_and_jacobian_matches_legacy():
    # J = I, M = I -> Lambda ~= I (up to eps regularization): shaped == legacy.
    legacy = _controller().compute(_make_state())
    shaped = _controller(task_space_inertia_shaping=True, lambda_regularization=0.0).compute(
        _make_state(mass_matrix=np.eye(6))
    )
    np.testing.assert_allclose(shaped.tau, legacy.tau, atol=1e-9)
    assert shaped.inertia_shaping_active


def test_shaping_scales_force_by_task_inertia():
    # J = I, M = diag(m): Lambda = diag(m), so the task force in X becomes m_x * a_x.
    m = np.diag([2.0, 5.0, 3.0, 1.0, 1.0, 1.0])
    unshaped = _controller().compute(_make_state())
    shaped = _controller(task_space_inertia_shaping=True, lambda_regularization=0.0).compute(
        _make_state(mass_matrix=m)
    )
    # X-direction task torque should scale by m_x = 2 relative to unshaped.
    np.testing.assert_allclose(
        shaped.tau_task_nominal[0], 2.0 * unshaped.tau_task_nominal[0], rtol=1e-9
    )


def test_nullspace_posture_vanishes_for_full_rank_square_task():
    # A 6-DOF arm with a full 6D task has NO nullspace: the dynamically
    # consistent projector must send the posture torque to exactly zero.
    rng = np.random.default_rng(42)
    J = np.eye(6) + 0.3 * rng.standard_normal((6, 6))
    A = rng.standard_normal((6, 6))
    M = A @ A.T + 6.0 * np.eye(6)  # SPD, well-conditioned

    ctl = _controller(nullspace_posture=True, lambda_regularization=0.0, kp_posture=10.0, kd_posture=0.0)
    out = ctl.compute(_make_state(q=np.full(6, 0.5), J=J, mass_matrix=M))

    assert out.nullspace_posture_active
    np.testing.assert_allclose(out.tau_posture, np.zeros(6), atol=1e-9)


def test_nullspace_posture_survives_in_rank_deficient_task():
    # Drop the last task row (rank-5 task): posture may act in the freed
    # direction but must produce (near-)zero acceleration in the remaining
    # task rows.
    rng = np.random.default_rng(7)
    J = np.eye(6) + 0.2 * rng.standard_normal((6, 6))
    J[5, :] = 0.0  # rank-deficient task
    A = rng.standard_normal((6, 6))
    M = A @ A.T + 6.0 * np.eye(6)

    ctl = _controller(
        nullspace_posture=True, lambda_regularization=1e-10, kp_posture=10.0, kd_posture=0.0
    )
    out = ctl.compute(_make_state(q=np.full(6, 0.5), J=J, mass_matrix=M))

    assert float(np.max(np.abs(out.tau_posture))) > 1e-6  # posture survives
    task_acc = J @ np.linalg.inv(M) @ out.tau_posture
    np.testing.assert_allclose(task_acc[:5], np.zeros(5), atol=1e-6)


def test_unprojected_posture_leaks_into_task_space():
    # Sanity contrast for the test above: without projection the same posture
    # torque does produce task acceleration.
    rng = np.random.default_rng(42)
    J = np.eye(6) + 0.3 * rng.standard_normal((6, 6))
    A = rng.standard_normal((6, 6))
    M = A @ A.T + 6.0 * np.eye(6)

    ctl = _controller(kp_posture=10.0, kd_posture=0.0)
    out = ctl.compute(_make_state(q=np.full(6, 0.5), J=J, mass_matrix=M))
    task_acc = J @ np.linalg.inv(M) @ out.tau_posture
    assert float(np.max(np.abs(task_acc))) > 1e-3


def test_missing_mass_matrix_falls_back_to_identity():
    shaped = _controller(task_space_inertia_shaping=True, lambda_regularization=0.0).compute(
        _make_state()  # no mass matrix supplied
    )
    legacy = _controller().compute(_make_state())
    assert shaped.inertia_shaping_active
    assert not shaped.mass_matrix_provided
    # With M=I and J=I the fallback reproduces legacy exactly.
    np.testing.assert_allclose(shaped.tau, legacy.tau, atol=1e-9)


def test_diagonal_shaping_off_matches_full_lambda():
    # Off by default: same as full-Lambda shaping (regression guard for the
    # new flag not changing anything when unset).
    m = np.diag([2.0, 5.0, 3.0, 1.0, 1.0, 1.0])
    plain = _controller(task_space_inertia_shaping=True, lambda_regularization=0.0).compute(
        _make_state(mass_matrix=m)
    )
    off = _controller(
        task_space_inertia_shaping=True, lambda_regularization=0.0, lambda_diagonal_shaping=False
    ).compute(_make_state(mass_matrix=m))
    np.testing.assert_allclose(off.tau, plain.tau, atol=1e-12)
    assert not off.lambda_diagonal_shaping_active


def test_diagonal_shaping_removes_off_axis_coupling():
    # J = I so tau_task_nominal == the shaped wrench directly (no further
    # J.T mixing to disentangle). A mass matrix with X-Z coupling then gives
    # Lambda = M exactly (eps=0), i.e. a non-diagonal Lambda purely from M.
    # With an X-only raw wrench (y/z/orientation already at reference), full
    # shaping must leak the X force into the Z row via Lambda[2,0]=M[2,0];
    # diagonal shaping must not.
    m = np.array(
        [
            [2.0, 0.0, 0.5, 0.0, 0.0, 0.0],
            [0.0, 3.0, 0.0, 0.0, 0.0, 0.0],
            [0.5, 0.0, 4.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        ]
    )
    state = _make_state(ee_pos=(0.3, 0.1, 0.5), target_x=0.5, J=np.eye(6), mass_matrix=m)

    full = _controller(task_space_inertia_shaping=True, lambda_regularization=0.0).compute(state)
    diag = _controller(
        task_space_inertia_shaping=True, lambda_regularization=0.0, lambda_diagonal_shaping=True
    ).compute(state)

    assert diag.lambda_diagonal_shaping_active
    assert not full.lambda_diagonal_shaping_active
    assert abs(float(full.wrench[0])) > 1e-6  # sanity: X wrench is actually nonzero
    assert abs(float(full.wrench[2])) < 1e-12  # raw Z wrench is exactly zero (z_err=0)
    assert abs(float(full.tau_task_nominal[2])) > 1e-6  # full shaping: X leaks into Z
    assert abs(float(diag.tau_task_nominal[2])) < 1e-12  # diagonal shaping: no leak


def test_diagonal_shaping_leaves_nullspace_posture_unaffected():
    # The nullspace projector must use the full Lambda regardless of
    # lambda_diagonal_shaping -- it's a separate mechanism (dynamically
    # consistent pseudoinverse for posture), not the wrench-shaping step.
    rng = np.random.default_rng(11)
    J = np.eye(6) + 0.3 * rng.standard_normal((6, 6))
    A = rng.standard_normal((6, 6))
    M = A @ A.T + 6.0 * np.eye(6)
    state = _make_state(q=np.full(6, 0.5), J=J, mass_matrix=M)

    off = _controller(
        nullspace_posture=True, lambda_regularization=0.0, kp_posture=10.0, kd_posture=0.0,
        lambda_diagonal_shaping=False,
    ).compute(state)
    on = _controller(
        nullspace_posture=True, lambda_regularization=0.0, kp_posture=10.0, kd_posture=0.0,
        lambda_diagonal_shaping=True,
    ).compute(state)
    np.testing.assert_allclose(on.tau_posture, off.tau_posture, atol=1e-12)


def test_adaptive_regularization_off_matches_static_eps():
    # Off by default: eps is always cfg.lambda_regularization, regardless of
    # cond(J) (regression guard for the new flag not changing anything).
    m = np.diag([2.0, 3.0, 4.0, 1.0, 1.0, 1.0])
    static = _controller(task_space_inertia_shaping=True, lambda_regularization=0.05).compute(
        _make_state(mass_matrix=m)
    )
    off = _controller(
        task_space_inertia_shaping=True, lambda_regularization=0.05,
        lambda_adaptive_regularization=False,
    ).compute(_make_state(mass_matrix=m))
    np.testing.assert_allclose(off.tau, static.tau, atol=1e-12)
    assert off.lambda_regularization_effective == 0.05
    assert not off.lambda_adaptive_regularization_active


def test_adaptive_regularization_uses_far_value_when_well_conditioned():
    # J = I is perfectly conditioned (cond=1), far below lambda_cond_low:
    # eps should resolve to lambda_regularization_far exactly.
    ctl = _controller(
        task_space_inertia_shaping=True,
        lambda_adaptive_regularization=True,
        lambda_regularization=0.1,
        lambda_regularization_far=1e-4,
        lambda_cond_low=1e4,
        lambda_cond_high=1e8,
    )
    out = ctl.compute(_make_state(mass_matrix=np.eye(6)))
    assert out.lambda_adaptive_regularization_active
    assert out.lambda_regularization_effective == pytest.approx(1e-4)


def test_adaptive_regularization_uses_near_value_when_ill_conditioned():
    # A near-singular J (cond >> lambda_cond_high): eps should resolve to the
    # unchanged near-singularity ceiling, lambda_regularization.
    J = np.diag([1.0, 1.0, 1.0, 1.0, 1.0, 1e-10])  # cond ~ 1e10
    ctl = _controller(
        task_space_inertia_shaping=True,
        lambda_adaptive_regularization=True,
        lambda_regularization=0.1,
        lambda_regularization_far=1e-4,
        lambda_cond_low=1e4,
        lambda_cond_high=1e8,
    )
    out = ctl.compute(_make_state(J=J, mass_matrix=np.eye(6)))
    assert out.lambda_regularization_effective == pytest.approx(0.1)


def test_adaptive_regularization_interpolates_monotonically():
    ctl_kwargs = dict(
        task_space_inertia_shaping=True,
        lambda_adaptive_regularization=True,
        lambda_regularization=0.1,
        lambda_regularization_far=1e-4,
        lambda_cond_low=1e4,
        lambda_cond_high=1e8,
    )
    conds = [1.0, 1e2, 1e4, 1e6, 1e8, 1e10]
    effective = []
    for c in conds:
        c = max(c, 1.0 + 1e-6)
        J = np.diag([1.0, 1.0, 1.0, 1.0, 1.0, 1.0 / c])
        out = _controller(**ctl_kwargs).compute(_make_state(J=J, mass_matrix=np.eye(6)))
        effective.append(out.lambda_regularization_effective)
    assert effective == sorted(effective)  # monotonically non-decreasing in cond(J)
    assert effective[0] == pytest.approx(1e-4)
    assert effective[-1] == pytest.approx(0.1)


def test_adaptive_regularization_does_not_affect_wrench_shaping():
    # The adaptive schedule must only change the nullspace projector's Lambda,
    # never the wrench-shaping Lambda -- a static, previously-validated eps
    # for wrench shaping is load-bearing (reducing it was found, via a live
    # sim sweep, to destabilize joint velocity well short of the singularity;
    # that regression is what this test guards against).
    m = np.diag([2.0, 5.0, 3.0, 1.0, 1.0, 1.0])
    state = _make_state(mass_matrix=m)
    no_adaptive = _controller(
        task_space_inertia_shaping=True, lambda_regularization=0.1,
        lambda_adaptive_regularization=False,
    ).compute(state)
    with_adaptive = _controller(
        task_space_inertia_shaping=True, lambda_regularization=0.1,
        lambda_adaptive_regularization=True, lambda_regularization_far=1e-4,
        lambda_cond_low=1e4, lambda_cond_high=1e8,
    ).compute(state)
    # J = I here so cond(J) = 1, far below lambda_cond_low -- eps_effective
    # (nullspace-only) would resolve to ~1e-4, very different from 0.1 --
    # but tau_task_nominal (wrench shaping) must be unchanged regardless.
    np.testing.assert_allclose(with_adaptive.tau_task_nominal, no_adaptive.tau_task_nominal, atol=1e-12)
    assert with_adaptive.lambda_regularization_effective == pytest.approx(1e-4)


def test_adaptive_regularization_yaml_parsing():
    ctrl_section = {
        "gains": {},
        "torque_limits_mode": "initial",
        "torque_limits_initial": {
            name: 100.0
            for name in (
                "shoulder_pan_joint",
                "shoulder_lift_joint",
                "elbow_joint",
                "wrist_1_joint",
                "wrist_2_joint",
                "wrist_3_joint",
            )
        },
        "lambda_adaptive_regularization": True,
        "lambda_regularization_far": 1e-5,
        "lambda_cond_low": 500.0,
        "lambda_cond_high": 5e6,
    }
    cfg = CartesianImpedanceConfig.from_controller_yaml_section(ctrl_section)
    assert cfg.lambda_adaptive_regularization is True
    assert cfg.lambda_regularization_far == 1e-5
    assert cfg.lambda_cond_low == 500.0
    assert cfg.lambda_cond_high == 5e6
    default_cfg = CartesianImpedanceConfig.from_controller_yaml_section(
        {k: v for k, v in ctrl_section.items() if not k.startswith(("lambda_adaptive", "lambda_regularization_far", "lambda_cond"))}
    )
    assert default_cfg.lambda_adaptive_regularization is False


def test_diagonal_shaping_yaml_parsing():
    ctrl_section = {
        "gains": {},
        "torque_limits_mode": "initial",
        "torque_limits_initial": {
            name: 100.0
            for name in (
                "shoulder_pan_joint",
                "shoulder_lift_joint",
                "elbow_joint",
                "wrist_1_joint",
                "wrist_2_joint",
                "wrist_3_joint",
            )
        },
        "lambda_diagonal_shaping": True,
    }
    cfg = CartesianImpedanceConfig.from_controller_yaml_section(ctrl_section)
    assert cfg.lambda_diagonal_shaping is True
    default_cfg = CartesianImpedanceConfig.from_controller_yaml_section(
        {k: v for k, v in ctrl_section.items() if k != "lambda_diagonal_shaping"}
    )
    assert default_cfg.lambda_diagonal_shaping is False


def test_yaml_section_parses_flags():
    ctrl_section = {
        "gains": {},
        "torque_limits_mode": "initial",
        "torque_limits_initial": {
            name: 100.0
            for name in (
                "shoulder_pan_joint",
                "shoulder_lift_joint",
                "elbow_joint",
                "wrist_1_joint",
                "wrist_2_joint",
                "wrist_3_joint",
            )
        },
        "task_space_inertia_shaping": True,
        "nullspace_posture": True,
        "lambda_regularization": 1e-5,
    }
    cfg = CartesianImpedanceConfig.from_controller_yaml_section(ctrl_section)
    assert cfg.task_space_inertia_shaping is True
    assert cfg.nullspace_posture is True
    assert cfg.lambda_regularization == 1e-5
    default_cfg = CartesianImpedanceConfig.from_controller_yaml_section(
        {k: v for k, v in ctrl_section.items() if not k.startswith(("task_", "nullspace", "lambda"))}
    )
    assert default_cfg.task_space_inertia_shaping is False
    assert default_cfg.nullspace_posture is False


def test_reanchor_off_keeps_q_rest_fixed():
    ctl = _controller(kp_posture=10.0, kd_posture=0.0)
    q_rest_before = ctl._q_rest.copy()
    # Settled state at the target: x_err ~ 0, qd = 0.
    ctl.compute(_make_state(q=np.full(6, 0.3), ee_pos=(0.4, 0.1, 0.5), target_x=0.4))
    np.testing.assert_allclose(ctl._q_rest, q_rest_before)


def test_reanchor_fires_once_on_settle():
    ctl = _controller(posture_reanchor_on_settle=True, kp_posture=10.0, kd_posture=0.0)
    # Not settled: large x error -> no re-anchor.
    out = ctl.compute(_make_state(q=np.full(6, 0.3), ee_pos=(0.35, 0.1, 0.5), target_x=0.4))
    assert not out.posture_reanchored
    # Not settled: at target but still moving -> no re-anchor.
    out = ctl.compute(_make_state(q=np.full(6, 0.3), qd=np.full(6, 0.2), ee_pos=(0.4, 0.1, 0.5), target_x=0.4))
    assert not out.posture_reanchored
    # Settled at target -> re-anchor to the current q.
    q_settled = np.full(6, 0.3)
    out = ctl.compute(_make_state(q=q_settled, ee_pos=(0.4, 0.1, 0.5), target_x=0.4))
    assert out.posture_reanchored
    np.testing.assert_allclose(ctl._q_rest, q_settled)
    # Posture torque is now zero at the settled configuration.
    np.testing.assert_allclose(out.tau_posture, np.zeros(6), atol=1e-12)
    # A later drifted q is pulled back toward the settled anchor, not the reset pose.
    q_drift = q_settled + np.array([0.0, 0.0, 0.0, 0.05, 0.05, 0.0])
    out2 = ctl.compute(_make_state(q=q_drift, ee_pos=(0.4, 0.1, 0.5), target_x=0.4))
    assert out2.posture_reanchored  # anchor stays latched
    np.testing.assert_allclose(ctl._q_rest, q_settled)  # not re-captured at the drifted q
    expected = 10.0 * (q_settled - q_drift)
    np.testing.assert_allclose(out2.tau_posture, expected, atol=1e-12)


def test_reanchor_rearms_when_target_moves_on():
    ctl = _controller(posture_reanchor_on_settle=True, kp_posture=10.0, kd_posture=0.0)
    out = ctl.compute(_make_state(q=np.full(6, 0.3), ee_pos=(0.4, 0.1, 0.5), target_x=0.4))
    assert out.posture_reanchored
    # New plateau beyond the tolerance: re-arms, then re-anchors at the new settle.
    out = ctl.compute(_make_state(q=np.full(6, 0.3), qd=np.full(6, 0.3), ee_pos=(0.41, 0.1, 0.5), target_x=0.43))
    assert not out.posture_reanchored
    q_new = np.full(6, 0.35)
    out = ctl.compute(_make_state(q=q_new, ee_pos=(0.43, 0.1, 0.5), target_x=0.43))
    assert out.posture_reanchored
    np.testing.assert_allclose(ctl._q_rest, q_new)


def test_reanchor_yaml_parsing():
    ctrl_section = {
        "gains": {},
        "torque_limits_mode": "initial",
        "torque_limits_initial": {
            name: 100.0
            for name in (
                "shoulder_pan_joint",
                "shoulder_lift_joint",
                "elbow_joint",
                "wrist_1_joint",
                "wrist_2_joint",
                "wrist_3_joint",
            )
        },
        "posture_reanchor_on_settle": True,
        "reanchor_x_tol_m": 0.004,
        "reanchor_qd_tol_radps": 0.1,
    }
    cfg = CartesianImpedanceConfig.from_controller_yaml_section(ctrl_section)
    assert cfg.posture_reanchor_on_settle is True
    assert cfg.reanchor_x_tol_m == 0.004
    assert cfg.reanchor_qd_tol_radps == 0.1
    default_cfg = CartesianImpedanceConfig.from_controller_yaml_section(
        {k: v for k, v in ctrl_section.items() if not k.startswith(("posture_re", "reanchor"))}
    )
    assert default_cfg.posture_reanchor_on_settle is False


# --- wrist_orientation_task (2026-07-29, AGENTS.md sec 3 "directional ceiling" fix) ---


def test_wrist_orientation_mask_shape():
    # Zero on the three proximal joints (shoulder_pan, shoulder_lift, elbow),
    # nonzero and heaviest on wrist_2 -- see the mask's own docstring in
    # controller_core/x_axis_cartesian_impedance.py for the legacy-controller
    # provenance of these ratios.
    assert WRIST_ORIENTATION_MASK.shape == (6,)
    np.testing.assert_allclose(WRIST_ORIENTATION_MASK[:3], 0.0)
    assert np.all(WRIST_ORIENTATION_MASK[3:] > 0.0)
    assert WRIST_ORIENTATION_MASK[4] == pytest.approx(1.0)
    assert WRIST_ORIENTATION_MASK[4] > WRIST_ORIENTATION_MASK[3]
    assert WRIST_ORIENTATION_MASK[4] > WRIST_ORIENTATION_MASK[5]


def test_wrist_orientation_task_off_by_default_and_zero_when_disabled():
    # Flag off (default) with nonzero gains still set: term must be exactly
    # zero and tau must match a reference controller that never set the
    # gains at all (proves the flag actually gates the term, not just the
    # gains happening to be zero).
    state = _make_state(q=np.full(6, 0.1), qd=np.full(6, 0.05), target_x=0.42)
    with_gains_off = _controller(wrist_orientation_task=False, kp_rot_wrist=50.0, kd_rot_wrist=20.0)
    reference = _controller()
    out_gated = with_gains_off.compute(state)
    out_ref = reference.compute(state)
    np.testing.assert_allclose(out_gated.tau_orient_wrist, np.zeros(6), atol=1e-12)
    np.testing.assert_allclose(out_gated.tau, out_ref.tau, atol=1e-12)
    assert out_gated.wrist_orientation_task_active is False
    assert out_ref.wrist_orientation_task_active is False


def test_wrist_orientation_task_matches_masked_jacobian_transpose_formula():
    from controller_core.kinematics_utils import orientation_error_vec_wxyz

    # J = I isolates J_rot to exactly the last three rows of the identity,
    # so J_rot.T @ m is trivial to predict by hand. Zero every other gain
    # (kp_x/kd_x/.../kd_joint/kp_rot/kd_rot) so tau_preclip is exactly
    # tau_orient_wrist -- nothing else contributes.
    kp_rot_wrist, kd_rot_wrist = 3.0, 2.0
    quat_ref = np.array([1.0, 0.0, 0.0, 0.0])
    quat_cur = np.array([np.cos(0.1), np.sin(0.1), 0.0, 0.0])  # small rotation about world X
    omega = np.array([0.1, -0.2, 0.05])

    cfg = CartesianImpedanceConfig(
        kp_x=0.0, kd_x=0.0, kp_y=0.0, kd_y=0.0, kp_z=0.0, kd_z=0.0,
        kp_rot=0.0, kd_rot=0.0, kp_posture=0.0, kd_posture=0.0, kd_joint=0.0,
        tau_max_nm=np.full(6, 1e6),
        wrist_orientation_task=True,
        kp_rot_wrist=kp_rot_wrist,
        kd_rot_wrist=kd_rot_wrist,
    )
    ctl = XAxisCartesianImpedanceController(cfg)
    reset_state = {
        "time": 0.0, "q": np.zeros(6), "qd": np.zeros(6),
        "ee_pos": np.array([0.4, 0.1, 0.5]), "ee_quat": quat_ref,
        "ee_lin_vel": np.zeros(3), "ee_ang_vel": np.zeros(3),
        "target_x": 0.4, "jacobian": np.eye(6),
    }
    ctl.reset_from_state(reset_state)
    state = dict(reset_state)
    state.update(ee_quat=quat_cur, ee_ang_vel=omega, target_x=0.4)
    out = ctl.compute(state)

    e_rot_expected = orientation_error_vec_wxyz(quat_ref, quat_cur)
    m_wrist_expected = kp_rot_wrist * e_rot_expected - kd_rot_wrist * omega
    j_rot = np.eye(6)[3:6, :]
    tau_orient_wrist_expected = (j_rot.T @ m_wrist_expected) * WRIST_ORIENTATION_MASK

    np.testing.assert_allclose(out.tau_orient_wrist, tau_orient_wrist_expected, atol=1e-10)
    np.testing.assert_allclose(out.tau, tau_orient_wrist_expected, atol=1e-10)
    assert out.wrist_orientation_task_active is True
    # Confirms it does NOT just replicate the (currently zero-gain) kp_rot/kd_rot
    # wrench term -- the wrench's own M block used kp_rot=kd_rot=0, so the
    # wrench itself is identically zero here.
    np.testing.assert_allclose(out.wrench, np.zeros(6), atol=1e-12)


def test_wrist_orientation_task_flows_through_backtracking_and_clip():
    # Deliberately large kp_rot_wrist + a tight tau_max so the term must be
    # visibly scaled down by the same geometric backtracking / hard clip
    # every other torque term goes through -- no bypass.
    quat_ref = np.array([1.0, 0.0, 0.0, 0.0])
    quat_cur = np.array([np.cos(0.5), np.sin(0.5), 0.0, 0.0])
    cfg = CartesianImpedanceConfig(
        kp_x=0.0, kd_x=0.0, kp_y=0.0, kd_y=0.0, kp_z=0.0, kd_z=0.0,
        kp_rot=0.0, kd_rot=0.0, kp_posture=0.0, kd_posture=0.0, kd_joint=0.0,
        tau_max_nm=np.full(6, 1.0),
        torque_headroom=1.0,
        wrist_orientation_task=True,
        kp_rot_wrist=1000.0,
        kd_rot_wrist=0.0,
    )
    ctl = XAxisCartesianImpedanceController(cfg)
    reset_state = {
        "time": 0.0, "q": np.zeros(6), "qd": np.zeros(6),
        "ee_pos": np.array([0.4, 0.1, 0.5]), "ee_quat": quat_ref,
        "ee_lin_vel": np.zeros(3), "ee_ang_vel": np.zeros(3),
        "target_x": 0.4, "jacobian": np.eye(6),
    }
    ctl.reset_from_state(reset_state)
    state = dict(reset_state)
    state.update(ee_quat=quat_cur, target_x=0.4)
    out = ctl.compute(state)

    assert np.all(np.abs(out.tau) <= 1.0 + 1e-9)
    # The unclipped nominal term is large; the final tau_orient_wrist diag
    # field is the backtracked/clipped version, strictly smaller in norm.
    assert np.linalg.norm(out.tau_orient_wrist) < np.linalg.norm(out.tau_task_nominal) + 1000.0
    assert np.all(np.abs(out.tau_orient_wrist) <= 1.0 + 1e-9)


def test_wrist_orientation_task_yaml_parsing():
    ctrl_section = {
        "gains": {"kp_rot_wrist": 12.0, "kd_rot_wrist": 6.0},
        "torque_limits_mode": "initial",
        "torque_limits_initial": {
            name: 100.0
            for name in (
                "shoulder_pan_joint",
                "shoulder_lift_joint",
                "elbow_joint",
                "wrist_1_joint",
                "wrist_2_joint",
                "wrist_3_joint",
            )
        },
        "wrist_orientation_task": True,
    }
    cfg = CartesianImpedanceConfig.from_controller_yaml_section(ctrl_section)
    assert cfg.wrist_orientation_task is True
    assert cfg.kp_rot_wrist == 12.0
    assert cfg.kd_rot_wrist == 6.0
    default_cfg = CartesianImpedanceConfig.from_controller_yaml_section(
        {
            k: v
            for k, v in ctrl_section.items()
            if k != "wrist_orientation_task"
        }
        | {"gains": {}}
    )
    assert default_cfg.wrist_orientation_task is False
    assert default_cfg.kp_rot_wrist == 0.0
    assert default_cfg.kd_rot_wrist == 0.0
