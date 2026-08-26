"""Singularity-robust damped least-squares (DLS) Jacobian inverse.

Standalone, numpy-only module. **Deliberately does not touch or import**
``controller_core/x_task_yz_corridor_qp/controller.py`` or
``controller_core/constrained_box_qp.py`` -- another actor is concurrently
editing those files (see hardware/joint_velocity_transport.py's module
docstring for why this exists and how it's used).

Motivation: the existing velocity-control mode (``hardware/velocity_transport.py``)
streams RTDE ``speedL`` -- a CARTESIAN velocity -- so the UR firmware inverts
its own Jacobian internally to get joint velocities. Near the wrist-2
singularity that firmware IK protective-stops instead of degrading (measured:
a +X move completed 100%, the mirrored -X move singularity-stopped). This
module computes the joint-velocity command OURSELVES with a damped
least-squares inverse so a near-singular Jacobian yields a bounded, if
imperfectly-tracked, joint velocity instead of an IK failure -- the resulting
command is streamed via RTDE ``speedJ`` instead (see ``hardware/link.py``'s
``speed_j``/``verify_speedj_signature``), bypassing firmware IK entirely while
still using the firmware's own joint servo loop (the same servo loop that let
speedL's +X move break static friction where torque control froze solid).

Standard Nakamura/Wampler variable-damping form::

    qd = J^T (J J^T + lambda^2 I)^-1 xd

    lambda^2 = lambda_max^2 * (1 - (sigma_min/sigma0)^2)   if sigma_min < sigma0
             = 0                                            otherwise

``sigma_min`` is the smallest singular value of ``J`` (via SVD). Away from a
singularity (``sigma_min >> sigma0``) ``lambda -> 0`` and this reduces
exactly to the Moore-Penrose pseudoinverse -- near-exact Cartesian tracking.
Near a singularity, ``lambda`` ramps up and this deliberately trades away
Cartesian tracking accuracy for a BOUNDED joint-velocity command -- that
trade is the entire point. Do not expect ``qd ~= J^-1 xd`` to hold near a
singularity; test for boundedness there, not tracking accuracy (see
``tests/unit/test_damped_least_squares.py``).

Implemented via a single SVD of ``J`` (numerically equivalent to the
matrix-inverse form above, but avoids forming/inverting ``J J^T`` directly):
for ``J = U diag(sigma) V^T``,

    J^T (J J^T + lambda^2 I)^-1 = V diag(sigma_i / (sigma_i^2 + lambda^2)) U^T

References: Y. Nakamura & H. Hanafusa, "Inverse Kinematic Solutions With
Singularity Robustness for Robot Manipulator Control" (1986); C. W. Wampler,
"Manipulator Inverse Kinematic Solutions Based on Vector Formulations and
Damped Least-Squares Methods" (IEEE Trans. SMC, 1986).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class DampedLeastSquaresConfig:
    """Variable-damping parameters. Both are in the units of a singular value
    of the (mixed linear/angular) Jacobian -- there is no single "correct"
    physical unit here since ``J`` mixes m/rad (linear rows) and rad/rad
    (angular rows); these are tuning constants, not derived physical
    quantities. See this module's docstring and
    ``hardware/joint_velocity_transport.py`` for the chosen defaults and why
    they are flagged as an open question pending real-hardware validation.
    """

    lambda_max: float = 0.05
    sigma0: float = 0.05

    def validate(self) -> None:
        if not np.isfinite(self.lambda_max) or self.lambda_max <= 0.0:
            raise ValueError("lambda_max must be positive and finite")
        if not np.isfinite(self.sigma0) or self.sigma0 <= 0.0:
            raise ValueError("sigma0 must be positive and finite")


@dataclass
class DampedLeastSquaresResult:
    """``qd`` plus the diagnostics needed to log/audit a DLS resolution
    step -- callers should trace ``sigma_min``/``lambda_used`` every cycle so
    a run can be checked after the fact for how close it came to the
    singularity and how much damping (i.e. how much Cartesian tracking
    fidelity) was traded away."""

    qd: np.ndarray
    sigma_min: float
    lambda_used: float
    qd_norm: float


def damped_least_squares_qd(
    jacobian: np.ndarray,
    xd: np.ndarray,
    *,
    lambda_max: float = 0.05,
    sigma0: float = 0.05,
) -> DampedLeastSquaresResult:
    """Resolve a desired Cartesian velocity ``xd`` (shape ``(m,)``) into a
    joint velocity ``qd`` (shape ``(n,)``) via variable-damping DLS given the
    task Jacobian ``jacobian`` (shape ``(m, n)``).

    ``lambda_max``/``sigma0`` must both be positive and finite -- see
    ``DampedLeastSquaresConfig``. Raises ``ValueError`` on non-finite input,
    a non-2D Jacobian, or a shape mismatch between ``jacobian`` and ``xd``.
    """
    if not np.isfinite(lambda_max) or lambda_max <= 0.0:
        raise ValueError("lambda_max must be positive and finite")
    if not np.isfinite(sigma0) or sigma0 <= 0.0:
        raise ValueError("sigma0 must be positive and finite")

    J = np.asarray(jacobian, dtype=np.float64)
    if J.ndim != 2:
        raise ValueError(f"jacobian must be 2D (m, n); got shape {J.shape}")
    m, n = J.shape
    xd_arr = np.asarray(xd, dtype=np.float64).reshape(-1)
    if xd_arr.shape[0] != m:
        raise ValueError(f"xd shape {xd_arr.shape} does not match jacobian rows ({m})")
    if not np.all(np.isfinite(J)):
        raise ValueError("jacobian contains NaN/Inf")
    if not np.all(np.isfinite(xd_arr)):
        raise ValueError("xd contains NaN/Inf")

    # full_matrices=False -> U is (m, k), s is (k,), Vt is (k, n), k=min(m,n).
    # np.linalg.svd returns singular values in DESCENDING order, so s[-1] is
    # sigma_min for this Jacobian.
    U, s, Vt = np.linalg.svd(J, full_matrices=False)
    sigma_min = float(s[-1]) if s.size > 0 else 0.0

    if sigma_min < sigma0:
        lambda_sq = float(lambda_max) ** 2 * (1.0 - (sigma_min / float(sigma0)) ** 2)
        lambda_sq = max(lambda_sq, 0.0)
    else:
        lambda_sq = 0.0
    lambda_used = float(np.sqrt(lambda_sq))

    # Damped pseudoinverse in SVD form: V diag(sigma_i / (sigma_i^2 + lambda^2)) U^T.
    damped_gains = s / (s**2 + lambda_sq)
    qd = Vt.T @ (damped_gains * (U.T @ xd_arr))
    qd = np.asarray(qd, dtype=np.float64).reshape(n)

    return DampedLeastSquaresResult(
        qd=qd,
        sigma_min=sigma_min,
        lambda_used=lambda_used,
        qd_norm=float(np.linalg.norm(qd)),
    )
