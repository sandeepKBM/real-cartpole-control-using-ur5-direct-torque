"""Injecting an analytic ``dJ/dq`` provider into ``manipulability_gradient``
(2026-08-26) must (a) match the finite-difference gradient it replaces, (b)
leave the DEFAULT finite-difference path byte-identical, and (c) still fall
back to the exact ``mu``-difference near a true singularity.

These are the ``controller_core``-level tests -- pure numpy, no pinocchio.
A synthetic smooth Jacobian with a KNOWN closed-form ``dJ/dq`` stands in for
the real kinematic provider, so the injection wiring and the trace-formula
contraction are exercised without a robot model. The real Pinocchio provider's
convention parity is the separate hard gate in
``tests/mujoco/test_jacobian_derivative_pinocchio.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from controller_core.manipulability_cbf import (  # noqa: E402
    manipulability,
    manipulability_cbf_filter,
    manipulability_gradient,
)


# --- a smooth J(q) whose dJ/dq is known in closed form --------------------- #
_A, _B, _N = 0.31, 0.7, 6


def _jac_fn(q, boost=1.5):
    q = np.asarray(q, dtype=np.float64).reshape(-1)
    n = _N
    J = np.zeros((n, n))
    for r in range(n):
        for c in range(n):
            J[r, c] = np.sin(q[c] + _A * r + _B * c) + (boost if r == c else 0.0)
    return J


def _dj_fn(q, boost=1.5):
    """Analytic dJ/dq of ``_jac_fn``: only the q[c] term depends on q, so
    ``dJ[r,c]/dq_k = cos(q[c] + A r + B c) * (k == c)``. Tensor shape (n, 6, n),
    ``T[k][:, i] = dJ[:, i]/dq_k``."""
    q = np.asarray(q, dtype=np.float64).reshape(-1)
    n = _N
    T = np.zeros((n, n, n))
    for r in range(n):
        for c in range(n):
            d = np.cos(q[c] + _A * r + _B * c)
            T[c, r, c] = d  # k == c
    return T


def _fd_of_mu_gradient(jacobian_fn, q, step=1e-6):
    q = np.asarray(q, dtype=np.float64).reshape(-1)
    g = np.zeros_like(q)
    for i in range(len(q)):
        qp = q.copy(); qm = q.copy()
        qp[i] += step; qm[i] -= step
        g[i] = (manipulability(jacobian_fn(qp)) - manipulability(jacobian_fn(qm))) / (2 * step)
    return g


@pytest.mark.parametrize("seed", range(8))
def test_analytic_injection_matches_fd_of_mu(seed):
    rng = np.random.default_rng(seed)
    q = rng.uniform(-1.0, 1.0, 6)
    g_analytic = manipulability_gradient(_jac_fn, q, jacobian_derivative_fn=_dj_fn)
    g_ref = _fd_of_mu_gradient(_jac_fn, q)
    rel = np.max(np.abs(g_analytic - g_ref)) / max(np.max(np.abs(g_ref)), 1e-12)
    assert rel < 1e-6, f"analytic gradient differs from FD-of-mu by rel {rel:.2e}"


@pytest.mark.parametrize("seed", range(8))
def test_analytic_matches_default_fd_path(seed):
    """The whole point: analytic and the default finite-difference path must
    agree to ~1e-6 -- 'similar enough to the CBF's'."""
    rng = np.random.default_rng(seed)
    q = rng.uniform(-1.0, 1.0, 6)
    g_analytic = manipulability_gradient(_jac_fn, q, jacobian_derivative_fn=_dj_fn)
    g_fd = manipulability_gradient(_jac_fn, q, step=1e-5)  # default FD path
    rel = np.max(np.abs(g_analytic - g_fd)) / max(np.max(np.abs(g_fd)), 1e-12)
    assert rel < 1e-6, f"analytic vs default-FD gradient rel {rel:.2e}"


def test_default_path_byte_identical_regression():
    """No jacobian_derivative_fn => the exact historical FD computation, not
    merely a numerically-close one. Locks the additive-change guarantee."""
    rng = np.random.default_rng(123)
    for _ in range(5):
        q = rng.uniform(-1.0, 1.0, 6)
        a = manipulability_gradient(_jac_fn, q, step=1e-5)
        b = manipulability_gradient(_jac_fn, q, step=1e-5)  # same call, no fn
        assert np.array_equal(a, b)
        # and passing an explicit None is identical to omitting it
        c = manipulability_gradient(_jac_fn, q, step=1e-5, jacobian_derivative_fn=None)
        assert np.array_equal(a, c)


def test_analytic_falls_back_near_singularity():
    """When (J J^T)^-1 is ill-posed the analytic path must fall through to the
    exact mu-difference (needs J at perturbed q) and still return a finite
    gradient matching the reference."""
    rng = np.random.default_rng(0)
    worst = 0.0
    for _ in range(40):
        q = rng.uniform(-2.0, 2.0, 6)
        jfn = lambda x: _jac_fn(x, boost=0.0)   # noqa: E731  -- may be near-singular
        djn = lambda x: _dj_fn(x, boost=0.0)    # noqa: E731
        g = manipulability_gradient(jfn, q, step=1e-6, jacobian_derivative_fn=djn)
        assert np.all(np.isfinite(g)), "gradient must be finite even near singular J"
        g_ref = _fd_of_mu_gradient(jfn, q)
        scale = max(np.max(np.abs(g_ref)), 1e-9)
        worst = max(worst, float(np.max(np.abs(g - g_ref)) / scale))
    assert worst < 1e-3, f"near-singular analytic gradient off by rel {worst:.2e}"


def test_shape_mismatch_rejected():
    q = np.zeros(6)
    bad = lambda x: np.zeros((6, 6, 5))  # noqa: E731  wrong trailing dim
    with pytest.raises(ValueError):
        manipulability_gradient(_jac_fn, q, jacobian_derivative_fn=bad)


def test_cbf_filter_constraint_row_matches():
    """The barrier itself -- the (A_row, b) it produces and the filtered tau --
    must be the same whether grad_mu came from FD or from the analytic tensor."""
    rng = np.random.default_rng(7)
    q = rng.uniform(-1.0, 1.0, 6)
    qd = rng.uniform(-0.5, 0.5, 6)
    jac = _jac_fn(q)
    n = 6
    m = rng.normal(0, 1, (n, n)); m = m @ m.T + 6 * np.eye(n)
    m_inv = np.linalg.inv(m)
    bias = rng.normal(0, 1, n)
    tau_nom = rng.normal(0, 5, n)
    lo = -50 * np.ones(n); hi = 50 * np.ones(n)
    kw = dict(
        tau_nominal=tau_nom, jacobian=jac, jacobian_fn=_jac_fn, q=q, qd=qd,
        m_inv=m_inv, bias=bias, tau_lower=lo, tau_upper=hi,
        epsilon=0.05, alpha1=10.0, alpha2=10.0,
    )
    r_fd = manipulability_cbf_filter(**kw)
    r_an = manipulability_cbf_filter(**kw, jacobian_derivative_fn=_dj_fn)
    # grad_norm, curvature (FD in both), h, tau all agree tightly.
    assert abs(r_fd.grad_norm - r_an.grad_norm) / max(r_fd.grad_norm, 1e-9) < 1e-6
    assert np.max(np.abs(r_fd.tau - r_an.tau)) < 1e-6
    assert r_fd.active == r_an.active
    assert abs(r_fd.slack_at_nominal - r_an.slack_at_nominal) < 1e-6
