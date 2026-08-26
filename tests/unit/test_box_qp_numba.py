"""The optional Numba hot path for solve_constrained_box_qp must be numerically
IDENTICAL to the pure-numpy fallback (2026-08-26) -- it reproduces the same
algorithm operation-for-operation, so the only thing it changes is speed. If
numba is not installed these tests skip (the fallback is what runs).
"""

from __future__ import annotations

import numpy as np
import pytest

import controller_core.constrained_box_qp as cbq
from controller_core.constrained_box_qp import solve_constrained_box_qp

numba_required = pytest.mark.skipif(not cbq.USE_NUMBA, reason="numba not installed")


def _solve(H, f, lo, hi, A, b, *, use_numba):
    prev = cbq.USE_NUMBA
    cbq.USE_NUMBA = use_numba
    try:
        return solve_constrained_box_qp(H, f, lo, hi, A, b, dual_sweeps=4, dual_root_iters=10)
    finally:
        cbq.USE_NUMBA = prev


@numba_required
@pytest.mark.parametrize("seed", range(25))
def test_numba_matches_numpy_bit_for_bit(seed):
    rng = np.random.default_rng(seed)
    n, m = 6, rng.integers(1, 6)
    U, _ = np.linalg.qr(rng.standard_normal((n, n)))
    H = (U * np.geomspace(1.0, 10 ** rng.uniform(1, 4), n)) @ U.T
    f = rng.standard_normal(n)
    lo = -rng.uniform(0.3, 2.0, n)
    hi = rng.uniform(0.3, 2.0, n)
    A = rng.standard_normal((m, n))
    b = rng.uniform(-0.6, 0.6, m)

    x_np, lam_np, feas_np = _solve(H, f, lo, hi, A, b, use_numba=False)
    x_nb, lam_nb, feas_nb = _solve(H, f, lo, hi, A, b, use_numba=True)

    assert feas_np == feas_nb
    assert np.max(np.abs(x_np - x_nb)) == 0.0, "numba x must be bit-identical to numpy"
    assert np.max(np.abs(lam_np - lam_nb)) == 0.0, "numba lambda must be bit-identical"


@numba_required
def test_numba_handles_pins_and_infeasible():
    # a pinned coordinate (lo==hi) plus a row the box cannot satisfy -> infeasible,
    # and the two paths must agree on that verdict and the returned x.
    rng = np.random.default_rng(7)
    n = 6
    H = np.eye(n) + 0.05 * rng.standard_normal((n, n))
    H = 0.5 * (H + H.T) + n * np.eye(n)
    f = rng.standard_normal(n)
    lo = -np.ones(n); hi = np.ones(n)
    lo[0] = hi[0] = 0.25  # a pin
    A = np.zeros((1, n)); A[0, 1] = 1.0
    b = np.array([-5.0])  # x[1] <= -5 is unreachable inside [-1,1] -> infeasible
    x_np, _, feas_np = _solve(H, f, lo, hi, A, b, use_numba=False)
    x_nb, _, feas_nb = _solve(H, f, lo, hi, A, b, use_numba=True)
    assert feas_np is False and feas_nb is False
    assert np.max(np.abs(x_np - x_nb)) == 0.0
    assert x_nb[0] == 0.25  # pin honored


@numba_required
def test_warmup_runs():
    assert cbq.numba_warmup() is True
