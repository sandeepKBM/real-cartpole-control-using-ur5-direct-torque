"""Warm-started ``solve_constrained_box_qp`` (2026-08-26). The opt-in warm path
must (1) be numerically identical between the numba and numpy implementations,
(2) reach the SAME fixed point as a cold solve regardless of the warm seed and
NOT drift from a good solution -- it changes only the iteration count, not the
control law -- and (3) leave the cold path (no warm args) byte-for-byte unchanged.

The base solver is a dual coordinate-ascent HEURISTIC that is only guaranteed
globally convergent when the general constraints are weakly coupled and the
Hessian is well conditioned (a property of the SHIPPED solver, independent of
warm-start; the config docstrings already note the projected-gradient inner solve
is slow at high conditioning). These unit tests therefore assert the numerical
INVARIANTS on well-conditioned, single-constraint problems where the solver is
unconditionally convergent, so any deviation is a warm-start bug rather than the
base heuristic's own limitation. The real m=5 corridor-QP at ARM_Q0 IS well-
behaved (a 20000-iter solve is stable to ~1e-11) and the end-to-end tau-parity vs
that ground truth -- the actual hard gate -- is asserted against captured real
instances in tests/mujoco/test_corridor_qp_warm_start_parity.py.
"""

from __future__ import annotations

import numpy as np
import pytest

import controller_core.constrained_box_qp as cbq
from controller_core.constrained_box_qp import solve_constrained_box_qp

numba_required = pytest.mark.skipif(not cbq.USE_NUMBA, reason="numba not installed")


def _rand_problem(seed, m=1):
    """Random PD box QP, cond(H) <= ~3, with ``m`` general rows (default 1).

    Low conditioning + a single (1-D, exactly root-found) constraint is the
    regime where the dual coordinate-ascent solver is unconditionally
    convergent, which is what makes the fixed-point assertions below meaningful.
    """
    rng = np.random.default_rng(seed)
    n = 6
    U, _ = np.linalg.qr(rng.standard_normal((n, n)))
    H = (U * np.geomspace(1.0, 10 ** rng.uniform(0.1, 0.45), n)) @ U.T
    f = rng.standard_normal(n)
    lo = -rng.uniform(0.3, 2.0, n)
    hi = rng.uniform(0.3, 2.0, n)
    A = rng.standard_normal((m, n))
    b = rng.uniform(-0.6, 0.6, m)
    return H, f, lo, hi, A, b


def _converged(H, f, lo, hi, A, b, xw=None, lw=None):
    return solve_constrained_box_qp(
        H, f, lo, hi, A, b, dual_sweeps=20, dual_root_iters=120,
        max_iters=8000, tol=1e-13, x_warm=xw, lam_warm=lw)


def _warm(H, f, lo, hi, A, b, xw, lw, *, max_iters=30, use_numba):
    prev = cbq.USE_NUMBA
    cbq.USE_NUMBA = use_numba
    try:
        return solve_constrained_box_qp(
            H, f, lo, hi, A, b, dual_sweeps=4, dual_root_iters=10,
            max_iters=max_iters, x_warm=xw, lam_warm=lw)
    finally:
        cbq.USE_NUMBA = prev


@numba_required
@pytest.mark.parametrize("seed", range(30))
def test_warm_numba_matches_numpy_bit_for_bit(seed):
    """The two implementations of the new warm path must agree exactly -- the
    same bar tests/unit/test_box_qp_numba.py holds the cold path to."""
    H, f, lo, hi, A, b = _rand_problem(seed)
    xr, lr, _ = _converged(H, f, lo, hi, A, b)
    x_np, lam_np, feas_np = _warm(H, f, lo, hi, A, b, xr, lr, use_numba=False)
    x_nb, lam_nb, feas_nb = _warm(H, f, lo, hi, A, b, xr, lr, use_numba=True)
    assert feas_np == feas_nb
    assert np.max(np.abs(x_np - x_nb)) == 0.0, "warm numba x must be bit-identical to numpy"
    assert np.max(np.abs(lam_np - lam_nb)) == 0.0, "warm numba lambda must be bit-identical"


@pytest.mark.parametrize("seed", range(30))
def test_fixed_point_is_seed_independent(seed):
    """Warm-starting changes iteration count, NOT the fixed point: generous-budget
    solves seeded from very different starts land on the same optimum."""
    H, f, lo, hi, A, b = _rand_problem(seed)
    m = A.shape[0]
    x_cold, _, _ = _converged(H, f, lo, hi, A, b)
    x1, _, _ = _converged(H, f, lo, hi, A, b,
                          xw=np.clip(np.zeros_like(f), lo, hi), lw=np.zeros(m))
    rng = np.random.default_rng(1000 + seed)
    x2, _, _ = _converged(H, f, lo, hi, A, b,
                          xw=np.clip(rng.standard_normal(f.shape[0]), lo, hi),
                          lw=np.abs(rng.standard_normal(m)))
    assert np.max(np.abs(x1 - x_cold)) <= 1e-6
    assert np.max(np.abs(x2 - x_cold)) <= 1e-6


@pytest.mark.parametrize("seed", range(30))
def test_warm_low_budget_stays_near_optimum(seed):
    """Seeded from the converged solution (the slowly-varying cycle-to-cycle
    case) a SMALL-budget warm solve stays essentially at that solution -- the
    stability the controller relies on. Bound is loose (1e-2) only because the
    4-sweep/10-root dual budget re-resolves lambda to ~1e-3; the primal does not
    drift."""
    H, f, lo, hi, A, b = _rand_problem(seed)
    xr, lr, _ = _converged(H, f, lo, hi, A, b)
    x_warm, _, _ = _warm(H, f, lo, hi, A, b, xr, lr, use_numba=cbq.USE_NUMBA)
    assert np.max(np.abs(x_warm - xr)) <= 1e-2


def test_cold_path_unchanged_by_warm_args_absence():
    """Passing no warm args must be byte-for-byte the pre-warm cold behavior."""
    H, f, lo, hi, A, b = _rand_problem(1, m=3)
    x1, l1, fe1 = solve_constrained_box_qp(H, f, lo, hi, A, b, dual_sweeps=4, dual_root_iters=10)
    x2, l2, fe2 = solve_constrained_box_qp(
        H, f, lo, hi, A, b, dual_sweeps=4, dual_root_iters=10,
        x_warm=None, lam_warm=None)
    assert np.array_equal(x1, x2) and np.array_equal(l1, l2) and fe1 == fe2


def test_warm_shape_mismatch_falls_back_to_cold():
    """A stale warm start whose shape does not match the current problem (row
    count changed) is refused and the cold solve runs -- silently correct."""
    H, f, lo, hi, A, b = _rand_problem(2, m=3)
    x_cold, l_cold, _ = solve_constrained_box_qp(
        H, f, lo, hi, A, b, dual_sweeps=4, dual_root_iters=10)
    bad_lam = np.zeros(A.shape[0] + 1)  # pretend a different row count last cycle
    x_fb, l_fb, _ = solve_constrained_box_qp(
        H, f, lo, hi, A, b, dual_sweeps=4, dual_root_iters=10,
        x_warm=np.zeros_like(f), lam_warm=bad_lam)
    assert np.array_equal(x_fb, x_cold) and np.array_equal(l_fb, l_cold)


@numba_required
def test_warmup_compiles_warm_kernels():
    assert cbq.numba_warmup() is True
    H, f, lo, hi, A, b = _rand_problem(0)
    x, lam, feas = solve_constrained_box_qp(
        H, f, lo, hi, A, b, max_iters=20,
        x_warm=np.zeros_like(f), lam_warm=np.zeros(A.shape[0]))
    assert x.shape == f.shape
