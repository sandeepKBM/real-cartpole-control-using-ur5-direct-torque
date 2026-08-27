"""Torque-parity gate for the 2026-08-26 non-QP performance pass on
``XTaskYZCorridorQPController.compute()`` (the hardware direct_torque path,
``config/ur5e_direct_torque_x_task_yz_corridor_qp.yaml`` + ``local_pinocchio``
dynamics).

WHAT CHANGED (pure refactors, no new behavior):
  1. ``cond(J)`` and ``mu(J) = prod(sigma)`` used to each run their own
     ``np.linalg.svd`` of the SAME Jacobian; now one SVD is computed and
     threaded to both (``_cond_from_sigma`` in ``controller.py``).
  2. ``manipulability_gradient``/``manipulability_directional_curvature``
     used to each re-evaluate ``jacobian_fn(q)`` (and, for curvature,
     ``manipulability(...)`` too) at the CURRENT ``q`` -- exactly the
     Jacobian/mu ``compute()`` already has in hand. Both now accept optional
     ``jac0``/``mu0`` to skip that redundant evaluation.
  3. ``task_space_inertia_shaping``'s own ``M^-1`` is threaded to the shared
     corridor/CBF/orientation ``m_inv`` instead of inverting ``mass_matrix``
     a second time when both are active.

None of this is supposed to change the torque the controller computes -- it
only removes work that reproduced a value already sitting in a local
variable. This file locks that down: reference ``tau`` vectors below were
captured from ``XTaskYZCorridorQPController.compute()`` at HEAD
(commit dd84f64, before the perf pass) at four states covering the shapes of
cycle this controller actually sees on hardware -- typical (corridor slack),
jammed (corridor tight, rows active), a mid-move joint configuration, and a
near-wall configuration off the exact wrist singularity. A refactor that
changes ANY floating-point value beyond reassociation noise fails this test;
see AGENTS.md sec 7, "verify the effect, not the invocation."
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

pytest.importorskip("pinocchio")

from controller_core.x_task_yz_corridor_qp import (  # noqa: E402
    XTaskYZCorridorQPConfig,
    XTaskYZCorridorQPController,
)
from hardware.local_dynamics import LocalPinocchioFastDynamics  # noqa: E402

CONFIG_PATH = REPO_ROOT / "config" / "ur5e_direct_torque_x_task_yz_corridor_qp.yaml"
ARM_Q0 = np.array([-2.3688, -2.1801, -1.8838, -0.7962, 0.004714693, 0.0206], dtype=np.float64)

# Tolerance for "unchanged": reordering pure floating-point sums can differ in
# the last bit or two; anything past 1e-9 Nm is treated as a real behavior
# change, not noise (matches the task's own stated target).
TAU_ATOL = 1.0e-9

_Q_MID = ARM_Q0.copy()
_Q_MID[0] += 0.01
_Q_MID[1] += -0.008
_Q_MID[4] += 0.02

_Q_WALL = ARM_Q0.copy()
_Q_WALL[4] += 0.15

# name -> (q, qd, target_x_delta_m, config overrides)
STATES: dict[str, tuple[np.ndarray, np.ndarray, float, dict]] = {
    "arm_q0_typical": (ARM_Q0.copy(), np.full(6, 0.05), 0.02, {}),
    "arm_q0_jammed": (
        ARM_Q0.copy(), np.full(6, 0.05), 0.02,
        {"y_corridor_half_width_m": 1.0e-4, "z_corridor_half_width_m": 1.0e-4},
    ),
    "mid_move": (
        _Q_MID, np.array([0.1, -0.05, 0.08, 0.02, -0.15, 0.03]), 0.04, {},
    ),
    "near_wall": (
        _Q_WALL, np.array([0.2, 0.1, -0.1, 0.05, 0.1, -0.05]), 0.06,
        {"y_corridor_half_width_m": 1.0e-3, "z_corridor_half_width_m": 1.0e-3},
    ),
}

# Captured from HEAD (dd84f64, pre-perf-pass) via the exact construction below.
REFERENCE_TAU: dict[str, list[float]] = {
    "arm_q0_typical": [
        -0.5, -18.299999999999997, -18.299999999999997,
        1.7461062416928406, 0.028385853420557383, -6.1,
    ],
    "arm_q0_jammed": [
        -0.5, -18.299999999999997, -4.6653557968540005,
        4.115674236405873, -0.07801748649796157, -6.1,
    ],
    "mid_move": [
        -1.0, -15.95897025712605, -18.48, 5.96, 0.24486040345821655, -6.06,
    ],
    "near_wall": [
        -2.0, -18.6, -7.438630773438506, 5.9, -1.6296529177926518, -5.9,
    ],
}

REFERENCE_MANIPULABILITY: dict[str, float] = {
    "arm_q0_typical": 0.0004326116407777192,
    "arm_q0_jammed": 0.0004326116407777192,
    "mid_move": 0.0022681501287087446,
    "near_wall": 0.014139823078378789,
}

REFERENCE_COND: dict[str, float] = {
    "arm_q0_typical": 1395.7631611199342,
    "arm_q0_jammed": 1395.7631611199342,
    "mid_move": 265.691327667082,
    "near_wall": 42.38086173058329,
}


def _load_controller_yaml() -> dict:
    with open(CONFIG_PATH, "r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    ctrl = dict(cfg["controller"])
    ctrl.pop("controller_kind", None)
    return ctrl


def _make_state(local_dynamics, q: np.ndarray, qd: np.ndarray, target_x_delta_m: float) -> dict:
    jacobian, mass_matrix = local_dynamics.jacobian_and_mass_matrix(q)
    ee_pos = np.array([0.4, -0.2, 0.3])
    ee_quat = np.array([1.0, 0.0, 0.0, 0.0])
    twist = jacobian @ qd
    return {
        "time": 0.0,
        "q": q,
        "qd": qd,
        "ee_pos": ee_pos,
        "ee_quat": ee_quat,
        "ee_lin_vel": twist[:3],
        "ee_ang_vel": twist[3:6],
        "jacobian": jacobian,
        "mass_matrix": mass_matrix,
        "target_x": float(ee_pos[0]) + target_x_delta_m,
        "target_x_vel": 0.0,
        "target_x_accel": None,
        "transport_axis_index": 0,
        "dt_s": 0.002,
    }


@pytest.fixture(scope="module")
def local_dynamics():
    return LocalPinocchioFastDynamics()


@pytest.mark.parametrize("name", sorted(STATES.keys()))
def test_tau_unchanged_by_non_qp_perf_refactor(local_dynamics, name: str) -> None:
    q, qd, target_x_delta_m, overrides = STATES[name]
    ctrl_yaml = _load_controller_yaml()
    ctrl_yaml.update(overrides)
    cfg = XTaskYZCorridorQPConfig.from_controller_yaml_section(ctrl_yaml)
    controller = XTaskYZCorridorQPController(
        cfg,
        jacobian_fn=local_dynamics.jacobian,
        jacobian_derivative_fn=local_dynamics.jacobian_derivative,
    )
    state = _make_state(local_dynamics, q, qd, target_x_delta_m)
    controller.reset_from_state(state)
    out = controller.compute(state)

    tau_ref = np.asarray(REFERENCE_TAU[name], dtype=np.float64)
    max_diff = float(np.max(np.abs(out.tau - tau_ref)))
    assert max_diff <= TAU_ATOL, (
        f"{name}: tau changed by {max_diff:.3e} Nm (max allowed {TAU_ATOL:.1e}) -- "
        f"got {out.tau.tolist()}, expected {tau_ref.tolist()}"
    )
    assert out.manipulability is not None
    assert abs(out.manipulability - REFERENCE_MANIPULABILITY[name]) <= 1.0e-12
    assert abs(out.jacobian_cond - REFERENCE_COND[name]) <= 1.0e-9


def test_svd_dedup_matches_numpy_cond_and_manipulability(local_dynamics) -> None:
    """Directly pins the two dedup targets: the SVD-reuse helper
    (``_cond_from_sigma``) must reproduce ``np.linalg.cond`` bit-for-bit, and
    ``prod(sigma)`` must reproduce ``manipulability(jac)`` bit-for-bit, on a
    real (non-synthetic) Jacobian -- not just at ARM_Q0.
    """
    from controller_core.manipulability_cbf import manipulability
    from controller_core.x_task_yz_corridor_qp.controller import _cond_from_sigma

    rng = np.random.default_rng(7)
    for name, (q, _qd, _dx, _ov) in STATES.items():
        jac = local_dynamics.jacobian(q)
        sigma = np.linalg.svd(jac, compute_uv=False)
        assert _cond_from_sigma(sigma) == pytest.approx(float(np.linalg.cond(jac)), abs=0.0, rel=1e-12)
        assert float(np.prod(sigma)) == pytest.approx(manipulability(jac), abs=0.0, rel=1e-12)

    # A handful of random 6x6 matrices too, not just robot Jacobians.
    for _ in range(20):
        m = rng.normal(size=(6, 6))
        sigma = np.linalg.svd(m, compute_uv=False)
        assert _cond_from_sigma(sigma) == pytest.approx(float(np.linalg.cond(m)), abs=0.0, rel=1e-12)
        assert float(np.prod(sigma)) == pytest.approx(manipulability(m), abs=0.0, rel=1e-12)


def test_jac0_mu0_reuse_matches_recompute(local_dynamics) -> None:
    """``manipulability_gradient(jac0=...)``/``manipulability_directional_
    curvature(mu0=...)`` must equal the default (recomputing) path exactly --
    the new kwargs are a pure "skip a redundant call" optimization, not a
    different algorithm.
    """
    from controller_core.manipulability_cbf import (
        manipulability,
        manipulability_directional_curvature,
        manipulability_gradient,
    )

    q = ARM_Q0.copy()
    qd = np.full(6, 0.05, dtype=np.float64)
    jac = local_dynamics.jacobian(q)
    mu = manipulability(jac)

    grad_default = manipulability_gradient(
        local_dynamics.jacobian, q, jacobian_derivative_fn=local_dynamics.jacobian_derivative
    )
    grad_reused = manipulability_gradient(
        local_dynamics.jacobian, q,
        jacobian_derivative_fn=local_dynamics.jacobian_derivative, jac0=jac,
    )
    assert np.array_equal(grad_default, grad_reused)

    curv_default = manipulability_directional_curvature(local_dynamics.jacobian, q, qd)
    curv_reused = manipulability_directional_curvature(local_dynamics.jacobian, q, qd, mu0=mu)
    assert curv_default == curv_reused
