"""Analytic ``dJ/dq`` from Pinocchio (2026-08-26) must be the exact derivative
of the SAME Jacobian ``LocalPinocchioFastDynamics.jacobian`` returns, so the
manipulability CBF gradient it feeds is byte-for-byte the barrier the 2*n
finite-difference evaluations computed -- only faster.

This is the hard convention gate: a frame/order mismatch in the analytic tensor
silently produces a wrong gradient that still "runs". The parity checks below
span ARM_Q0 +- 0.5 rad per joint, random configs, and a near-singular config;
the speed check confirms the whole point (one call, not 2*n).

Marked ``mujoco`` by directory, but the real dependency is ``pinocchio``.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

pytest.importorskip("pinocchio")

from controller_core.manipulability_cbf import (  # noqa: E402
    manipulability_cbf_constraint_row,
    manipulability_directional_curvature,
    manipulability_gradient,
)
from hardware.local_dynamics import LocalPinocchioFastDynamics  # noqa: E402

ARM_Q0 = np.array([-2.3688, -2.1801, -1.8838, -0.7962, 0.004714693, 0.0206])


@pytest.fixture(scope="module")
def dyn():
    return LocalPinocchioFastDynamics()


def _fd_dJ(dyn, q, step=1e-6):
    n = 6
    d = np.empty((n, 6, n))
    for k in range(n):
        qp = q.copy(); qm = q.copy()
        qp[k] += step; qm[k] -= step
        d[k] = (dyn.jacobian(qp) - dyn.jacobian(qm)) / (2 * step)
    return d


def _configs():
    cfgs = [ARM_Q0.copy()]
    # ARM_Q0 +- 0.5 on each joint individually
    for j in range(6):
        for s in (+0.5, -0.5):
            q = ARM_Q0.copy(); q[j] += s
            cfgs.append(q)
    rng = np.random.default_rng(0)
    for _ in range(5):
        cfgs.append(ARM_Q0 + rng.uniform(-0.5, 0.5, 6))
    # near the wrist singularity (wrist_2 -> 0)
    q_sing = ARM_Q0.copy(); q_sing[4] = 1e-4
    cfgs.append(q_sing)
    return cfgs


def test_jacobian_derivative_matches_finite_difference(dyn):
    """Analytic dJ/dq vs central difference of the SAME jacobian(), across
    ARM_Q0 +- 0.5, random, and near-singular configs. FD is O(step^2)~1e-10,
    analytic is exact, so they must agree to the FD noise floor."""
    worst = 0.0
    for q in _configs():
        da = dyn.jacobian_derivative(q)
        df = _fd_dJ(dyn, q)
        worst = max(worst, float(np.max(np.abs(da - df))))
    assert worst < 1e-6, f"analytic dJ/dq vs FD max abs diff {worst:.2e}"


def test_manipulability_gradient_analytic_vs_fd(dyn):
    """The gradient the CBF actually consumes: analytic (one dJ/dq call) vs the
    default finite-difference path (2*n jacobian evals)."""
    worst_abs = 0.0
    worst_rel = 0.0
    for q in _configs():
        g_fd = manipulability_gradient(dyn.jacobian, q)
        g_an = manipulability_gradient(
            dyn.jacobian, q, jacobian_derivative_fn=dyn.jacobian_derivative
        )
        worst_abs = max(worst_abs, float(np.max(np.abs(g_an - g_fd))))
        worst_rel = max(
            worst_rel,
            float(np.max(np.abs(g_an - g_fd)) / max(np.max(np.abs(g_fd)), 1e-12)),
        )
    assert worst_abs < 1e-6, f"gradient analytic vs FD abs {worst_abs:.2e}"
    assert worst_rel < 1e-6, f"gradient analytic vs FD rel {worst_rel:.2e}"


def test_cbf_constraint_row_analytic_vs_fd(dyn):
    """The barrier row (A, b) and the full ``A tau <= b`` inequality must be
    unchanged to the gradient tolerance -- the CBF must BEHAVE the same, just
    compute grad_mu faster. Curvature is FD in both, so only grad differs."""
    from controller_core.manipulability_cbf import manipulability

    rng = np.random.default_rng(3)
    worst_A = 0.0
    worst_b = 0.0
    for q in _configs():
        J, M = dyn.jacobian_and_mass_matrix(q)
        m_inv = np.linalg.inv(M)
        qd = rng.uniform(-0.4, 0.4, 6)
        bias = rng.normal(0, 1, 6)
        mu = float(manipulability(J))
        curv = manipulability_directional_curvature(dyn.jacobian, q, qd)
        g_fd = manipulability_gradient(dyn.jacobian, q)
        g_an = manipulability_gradient(
            dyn.jacobian, q, jacobian_derivative_fn=dyn.jacobian_derivative
        )
        a_fd, b_fd = manipulability_cbf_constraint_row(
            grad_mu=g_fd, m_inv=m_inv, bias=bias, qd=qd, mu=mu, curvature=curv,
            epsilon=0.05, alpha1=10.0, alpha2=10.0,
        )
        a_an, b_an = manipulability_cbf_constraint_row(
            grad_mu=g_an, m_inv=m_inv, bias=bias, qd=qd, mu=mu, curvature=curv,
            epsilon=0.05, alpha1=10.0, alpha2=10.0,
        )
        worst_A = max(worst_A, float(np.max(np.abs(a_fd - a_an))))
        worst_b = max(worst_b, abs(float(b_fd) - float(b_an)))
    assert worst_A < 1e-6, f"CBF A-row analytic vs FD {worst_A:.2e}"
    assert worst_b < 1e-6, f"CBF b analytic vs FD {worst_b:.2e}"


def _median_us(fn, iters=400):
    for _ in range(100):  # warm up (pinocchio + numpy first-call costs)
        fn()
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t0) * 1e6)
    return float(np.median(ts))


def test_analytic_tensor_cheaper_than_finite_difference(dyn):
    """The load-bearing speed claim: the analytic dJ/dq tensor (one kinematic
    pass) is cheaper to form than the 2*n Jacobian evaluations the finite
    difference needs -- this is the deterministic pinocchio-call saving and is
    robust to numpy-dispatch noise on shared hosts."""
    q = ARM_Q0.copy()
    an_us = _median_us(lambda: dyn.jacobian_derivative(q))

    def _fd_tensor():
        for k in range(6):
            qp = q.copy(); qm = q.copy()
            qp[k] += 1e-5; qm[k] -= 1e-5
            dyn.jacobian(qp); dyn.jacobian(qm)

    fd_us = _median_us(_fd_tensor)
    assert an_us < fd_us, (
        f"analytic tensor {an_us:.1f}us not cheaper than 2*n jacobian evals {fd_us:.1f}us"
    )


def test_analytic_gradient_not_slower(dyn):
    """End-to-end, the analytic gradient must not be slower than the FD path it
    replaces. The margin is modest when jacobian_fn is already the fast
    Pinocchio provider (the shared trace-formula contraction dominates both), so
    this asserts 'not slower' with a small tolerance rather than a fixed speedup
    -- the real per-cycle saving is the pinocchio-call count, asserted above."""
    q = ARM_Q0.copy()
    fd_us = _median_us(lambda: manipulability_gradient(dyn.jacobian, q))
    an_us = _median_us(
        lambda: manipulability_gradient(
            dyn.jacobian, q, jacobian_derivative_fn=dyn.jacobian_derivative
        )
    )
    assert an_us < 1.15 * fd_us, (
        f"analytic gradient {an_us:.1f}us slower than FD {fd_us:.1f}us"
    )
