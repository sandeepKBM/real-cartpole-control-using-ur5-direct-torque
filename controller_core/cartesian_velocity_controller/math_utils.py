"""Shared numeric helpers for the Cartesian velocity controller package."""

from __future__ import annotations

import numpy as np


def _damped_pinv(j: np.ndarray, damping: float) -> np.ndarray:
    """Tikhonov-damped least-squares pseudoinverse: J^T @ (J J^T + d^2 I)^-1.

    Plain np.linalg.pinv only truncates singular values that are negligible
    relative to the LARGEST one (its rcond default) -- near, but not at, a
    true singularity, moderately small singular values (e.g. ~0.05-0.1, the
    range this repo's wrist_2=0 approach measures) survive that truncation
    and get inverted almost as-is, amplifying any input in that direction by
    a large but finite factor. Measured directly (2026-08-03): with plain
    pinv, a jacobian_singular_cond_max analog wasn't in play here (that flag
    only exists in the torque controller), so this same amplification showed
    up as wildly non-monotonic required joint velocity vs. kp_posture (3.1 ->
    11.7 -> 2.5 -> 21.4 rad/s as kp_posture stepped 0.05 -> 0.1 -> 0.2 ->
    1.0) -- a numerically ill-conditioned regime, not a real physical
    requirement. Damping trades a small amount of task-tracking exactness
    (J @ J^+_damped != I exactly near the singularity) for bounded qd, the
    same tradeoff this repo's torque-control lane already made via
    lambda_regularization for its own Lambda inversion."""
    j = np.asarray(j, dtype=np.float64)
    d2 = float(damping) ** 2
    gram = j @ j.T + d2 * np.eye(j.shape[0], dtype=np.float64)
    return j.T @ np.linalg.inv(gram)
