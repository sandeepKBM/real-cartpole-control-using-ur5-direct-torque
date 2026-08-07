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


def smooth_falloff(value: float, full_below: float, zero_above: float, power: float = 2.0) -> float:
    """Smooth [0, 1] blend weight, EXACTLY 1.0 at or below ``full_below`` and
    EXACTLY 0.0 at or above ``zero_above``, falling off as ``power`` in
    between:

        u = (|value| - full_below) / (zero_above - full_below), clipped to [0, 1]
        weight = (1 - u) ** power

    Pure scalar math with no simulator dependency, deliberately factored out
    here so its SHAPE is unit-testable on its own.

    Used by ``modes.py``'s ``compute_ik_seeded`` for ``orientation_priority``
    (see ``config.py``): ``value`` is the orientation-prioritised IK solve's
    own POSITION residual, so the blend is 1.0 (orientation fully promoted)
    exactly where promoting it costs no position accuracy, and decays to 0.0
    (today's position-only behaviour, bit-for-bit) where it would.

    Both exact endpoints matter. The exact 0.0 above ``zero_above`` is what
    lets the caller skip the blend entirely and return the unmodified
    position-only solution, so a case the mechanism cannot help is not
    perturbed by it at all -- the same exactness-preservation discipline
    ``ik_max_joint_deviation_rad`` (null-space clipping that provably cannot
    change ``J_task @ dq``) was held to. The exact 1.0 below ``full_below``
    is what makes the common, comfortably-reachable case a single clean
    branch rather than a near-1 blend of two nearly-identical solutions.

    ``power`` defaults to 2.0 rather than 1.0 so the blend stays close to
    fully-promoted over most of the band and only drops off sharply as the
    residual approaches the point where orientation genuinely conflicts with
    position.

    Degenerate ``zero_above <= full_below`` collapses to a step at
    ``full_below`` rather than dividing by zero.
    """
    v = abs(float(value))
    lo = float(full_below)
    hi = float(zero_above)
    if v <= lo:
        return 1.0
    if hi <= lo or v >= hi:
        return 0.0
    u = (v - lo) / (hi - lo)
    return float((1.0 - u) ** max(float(power), 0.0))
