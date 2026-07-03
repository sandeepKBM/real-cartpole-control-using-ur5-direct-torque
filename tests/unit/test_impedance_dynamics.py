"""Unit tests for the P3 operational-space terms in the impedance law.

Pure numpy — no simulator required.
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
