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


def singularity_speed_scale(
    sigma_min: float, sigma_min_stop: float, sigma_min_full_speed: float, power: float = 2.0
) -> float:
    """Smooth [0, 1] speed-scaling factor from a Jacobian's smallest singular
    value: EXACTLY 1.0 (full speed) at or above ``sigma_min_full_speed``,
    EXACTLY 0.0 (stopped) at or below ``sigma_min_stop``, ramping as
    ``power`` in between. This is the mirror image of ``smooth_falloff``
    (danger GROWS as ``sigma_min`` shrinks, rather than as a residual
    grows), so it is implemented directly in terms of it:
    ``1.0 - smooth_falloff(sigma_min, sigma_min_stop, sigma_min_full_speed, power)``.

    Used by ``modes.py``'s ``singularity_velocity_scaling`` (see
    ``config.py``) to throttle the commanded task velocity/error as the
    controller approaches a kinematic (Jacobian) singularity, rather than
    only damping the pseudoinverse used to invert it -- the standard
    "adaptive Cartesian velocity scaling near singularities" fix: bound
    joint velocity by reducing the INPUT, not just by regularizing the
    inversion of a fixed-magnitude one.

    Both exact endpoints matter for the same reason as ``smooth_falloff``'s:
    a case comfortably far from any singularity (``sigma_min >=
    sigma_min_full_speed``) is byte-identical to the mechanism being off,
    and a case at/inside the stop threshold gets EXACTLY zero commanded
    velocity rather than some small residual crawl."""
    return 1.0 - smooth_falloff(sigma_min, sigma_min_stop, sigma_min_full_speed, power)
