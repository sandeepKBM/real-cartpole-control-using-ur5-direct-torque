"""The analytic trace-formula ``manipulability_gradient`` (2026-08-26) must be
numerically identical to differencing ``mu`` directly, and must fall back to
that exact behavior near a true singularity where ``(J J^T)^-1`` is ill-posed.

It replaced a 12-SVD-per-call finite difference to cut the manipulability CBF's
per-cycle cost (measured: manip-on control cycle 2.0 -> 1.79 ms, back under the
500 Hz / 2 ms budget). These tests pin the correctness the speedup must not cost.
"""

from __future__ import annotations

import numpy as np
import pytest

from controller_core.manipulability_cbf import manipulability, manipulability_gradient


def _fd_of_mu_gradient(jacobian_fn, q, step=1e-6):
    """Reference: difference mu itself (the pre-2026-08-26 implementation)."""
    q = np.asarray(q, dtype=np.float64).reshape(-1)
    g = np.zeros_like(q)
    for i in range(len(q)):
        qp = q.copy(); qm = q.copy()
        qp[i] += step; qm[i] -= step
        g[i] = (manipulability(jacobian_fn(qp)) - manipulability(jacobian_fn(qm))) / (2 * step)
    return g


def _smooth_jac_fn(n=6, boost=1.5):
    """A smooth, well-conditioned J(q): diagonal ``boost`` keeps it away from
    singular so ``(J J^T)^-1`` is well-posed (the trace-formula path)."""
    def jfn(q):
        q = np.asarray(q, dtype=np.float64).reshape(-1)
        J = np.zeros((n, n))
        for r in range(n):
            for c in range(n):
                J[r, c] = (np.sin(q[c] + 0.31 * r + 0.7 * c)
                           + 0.4 * np.cos(2.0 * q[(c + 1) % n] - 0.2 * r)
                           + (boost if r == c else 0.0))
        return J
    return jfn


@pytest.mark.parametrize("seed", range(8))
def test_trace_gradient_matches_fd_of_mu(seed):
    jfn = _smooth_jac_fn()
    rng = np.random.default_rng(seed)
    q = rng.uniform(-1.0, 1.0, 6)
    g_trace = manipulability_gradient(jfn, q, step=1e-6)
    g_ref = _fd_of_mu_gradient(jfn, q, step=1e-6)
    rel = np.max(np.abs(g_trace - g_ref)) / max(np.max(np.abs(g_ref)), 1e-12)
    assert rel < 1e-6, f"trace gradient differs from FD-of-mu by rel {rel:.2e}"


def test_trace_gradient_finite_and_correct_near_singularity():
    """With no diagonal boost, some q make J nearly rank-deficient -> the
    (J J^T)^-1 path is ill-posed and the code must fall back to FD-of-mu,
    still returning a finite gradient close to the reference."""
    jfn = _smooth_jac_fn(boost=0.0)
    rng = np.random.default_rng(0)
    worst = 0.0
    for _ in range(40):
        q = rng.uniform(-2.0, 2.0, 6)
        g = manipulability_gradient(jfn, q, step=1e-6)
        assert np.all(np.isfinite(g)), "gradient must be finite even near singular J"
        g_ref = _fd_of_mu_gradient(jfn, q, step=1e-6)
        scale = max(np.max(np.abs(g_ref)), 1e-9)
        worst = max(worst, float(np.max(np.abs(g - g_ref)) / scale))
    assert worst < 1e-3, f"near-singular gradient off by rel {worst:.2e}"


def test_gradient_length_matches_configuration_dim():
    jfn = _smooth_jac_fn()
    for n_probe in (6,):
        q = np.zeros(n_probe)
        g = manipulability_gradient(jfn, q)
        assert g.shape == (n_probe,)


def test_rejects_nonpositive_step():
    jfn = _smooth_jac_fn()
    with pytest.raises(ValueError):
        manipulability_gradient(jfn, np.zeros(6), step=0.0)
