"""Correctness + benchmark tests for the residual-observer dynamics
optimization (2026-07-30): replacing the two-call
``gravity(q)`` + ``bias(q, qd)`` formula in
``hardware/direct_torque_transport.py``'s residual observer with a single
``coriolis(q, qd)`` call, since the ``g(q)`` terms cancel algebraically
(``tau_true_total - bias == tau - C(q, qd) @ qd``, see
``controller_core/dynamics_residual.py::predict_joint_acceleration``'s
docstring and
``docs/status/residual_observer_dynamics_optimization_2026-07-30.md``).

``PinocchioUR5eDynamics.coriolis`` itself changed too: it now computes
``C(q, qd) @ qd`` via a single ``rnea`` call against a dedicated
zero-gravity model/data pair, instead of ``bias(q, qd) - gravity(q)``
(two calls). This file checks that change is both correct (matches the old
two-call formula to tight tolerance across random poses) and faster
(real timing, this machine, matching the rigor of
``docs/status/local_dynamics_speedup_investigation_2026-07-29.md``).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

pin = pytest.importorskip("pinocchio")

from controller_core.dynamics_residual import predict_joint_acceleration  # noqa: E402
from controller_core.model_dynamics import PinocchioUR5eDynamics  # noqa: E402

N_SAMPLES = 200
CORIOLIS_TOL = 1e-9  # far tighter than test_pinocchio_parity.py's existing 1e-12 already
QDD_TOL = 1e-9


@pytest.fixture(scope="module")
def dyn() -> PinocchioUR5eDynamics:
    return PinocchioUR5eDynamics()


def _random_qqd(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    q = rng.uniform(-np.pi, np.pi, size=6)
    qd = rng.uniform(-1.5, 1.5, size=6)
    return q, qd


def test_coriolis_matches_old_bias_minus_gravity_formula(dyn: PinocchioUR5eDynamics) -> None:
    """New single-call coriolis() must match the old bias(q,qd) - gravity(q)
    two-call formula across many random poses, not just the handful of
    samples in test_pinocchio_parity.py::test_coriolis_is_bias_minus_gravity."""
    rng = np.random.default_rng(7)
    worst = 0.0
    for _ in range(N_SAMPLES):
        q, qd = _random_qqd(rng)
        c_new = dyn.coriolis(q, qd)
        c_old = dyn.bias(q, qd) - dyn.gravity(q)
        worst = max(worst, float(np.max(np.abs(c_new - c_old))))
    assert worst < CORIOLIS_TOL, f"coriolis() vs bias()-gravity() worst |delta| = {worst}"


def test_qdd_pred_matches_old_two_call_residual_formula(dyn: PinocchioUR5eDynamics) -> None:
    """End-to-end: the old residual-observer formula
    (tau_true_total = tau + gravity(q); qdd = M^-1(tau_true_total - bias))
    vs the new one (qdd = M^-1(tau - coriolis(q,qd))) must agree to tight
    tolerance across random (q, qd, tau, M) samples -- this is the exact
    computation hardware/direct_torque_transport.py's residual observer
    performs every cycle."""
    rng = np.random.default_rng(11)
    worst = 0.0
    for _ in range(N_SAMPLES):
        q, qd = _random_qqd(rng)
        tau = rng.uniform(-20.0, 20.0, size=6)
        M = dyn.mass_matrix(q)

        # Old path: two dynamics calls (gravity + bias).
        tau_true_total_old = tau + dyn.gravity(q)
        bias_old = dyn.bias(q, qd)
        qdd_old = predict_joint_acceleration(M, tau_true_total_old, bias_old)

        # New path: one dynamics call (coriolis), no gravity() needed.
        coriolis_new = dyn.coriolis(q, qd)
        qdd_new = predict_joint_acceleration(M, tau, coriolis_new)

        worst = max(worst, float(np.max(np.abs(qdd_new - qdd_old))))
    assert worst < QDD_TOL, f"qdd_pred old-vs-new formula worst |delta| = {worst}"


def test_coriolis_vanishes_at_zero_velocity(dyn: PinocchioUR5eDynamics) -> None:
    rng = np.random.default_rng(13)
    for _ in range(20):
        q = rng.uniform(-np.pi, np.pi, size=6)
        np.testing.assert_allclose(dyn.coriolis(q, np.zeros(6)), np.zeros(6), atol=1e-9)


@pytest.mark.slow
def test_coriolis_single_call_is_not_slower_than_old_two_call_path(
    dyn: PinocchioUR5eDynamics,
) -> None:
    """Real timing on this machine: the new single-call coriolis() should be
    at least as fast as the old bias()+gravity() two-call formula it
    replaces (measured ~1.5-2x faster in
    docs/status/residual_observer_dynamics_optimization_2026-07-30.md's
    dedicated benchmark script). Uses best-of-3 trials of a warmed-up,
    many-call timing loop to reduce shared-machine noise; asserts only a
    generous non-regression margin (not the full measured speedup) to avoid
    flakiness on a loaded cluster host -- see AGENTS.md SS8 on `westeros`
    contention."""
    rng = np.random.default_rng(42)
    n_warmup = 100
    n_timed = 1000
    qs = rng.uniform(-np.pi, np.pi, size=(n_warmup + n_timed, 6))
    qds = rng.uniform(-1.5, 1.5, size=(n_warmup + n_timed, 6))

    def old_formula(i: int) -> np.ndarray:
        q, qd = qs[i], qds[i]
        return dyn.bias(q, qd) - dyn.gravity(q)

    def new_formula(i: int) -> np.ndarray:
        q, qd = qs[i], qds[i]
        return dyn.coriolis(q, qd)

    def best_of_3_mean_ms(fn) -> float:
        best = float("inf")
        for _ in range(3):
            for i in range(n_warmup):
                fn(i)
            t0 = time.perf_counter()
            for i in range(n_warmup, n_warmup + n_timed):
                fn(i)
            t1 = time.perf_counter()
            best = min(best, (t1 - t0) / n_timed * 1e3)
        return best

    old_ms = best_of_3_mean_ms(old_formula)
    new_ms = best_of_3_mean_ms(new_formula)

    # Generous margin: only assert the new path isn't slower, not that it
    # hits the full measured speedup (loaded shared machine can compress
    # or invert small microsecond-scale gaps run-to-run).
    assert new_ms <= old_ms * 1.15, (
        f"expected coriolis() single-call path to not regress vs the old "
        f"bias()+gravity() two-call formula: old={old_ms:.4f} ms, new={new_ms:.4f} ms"
    )
