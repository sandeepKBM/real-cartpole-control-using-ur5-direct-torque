#!/usr/bin/env python3
"""Analytic model of the 15cm pole, free-hinged at the UR5e end-effector,
with the end-effector's horizontal (X) position treated as a PRESCRIBED
(imposed) trajectory rather than a free coupled cart -- justified because
the pole (~250g) is negligible next to the arm's own task-space inertia
(~2.3-4.8kg measured, see session history) and the arm's controller has
~50x acceleration headroom over anything the swing-up needs, so the arm's
own tracking is not meaningfully perturbed by pendulum reaction forces.

Derivation (Euler-Lagrange, theta measured from hanging-down = 0, pi =
inverted), for a general rigid body pivoted at an accelerating base with
total mass M, first moment about pivot S1 = M*d_com, moment of inertia
about pivot I_p:

    I_p * theta_ddot = -M*g*d_com*sin(theta) - M*d_com*a(t)*cos(theta) - b*theta_dot

where a(t) = x_c_ddot is the base's horizontal acceleration (exactly what
this repo's controllers command/measure as TCP acceleration) and b is
hinge viscous damping (N*m*s/rad).

Modeled as a compound body: a uniform 50g rod (length 0.15m) plus a 200g
lumped bracket mass at radius r_b from the pivot (real value not measured
yet -- no physical hinge exists; r_b=0 is the physically likely case for a
compact mounting bracket, sensitivity checked in session history: r_b in
[0, 5cm] moves the natural frequency 1.58-2.1Hz and the idealized minimum
tip speed 2.97-3.94 m/s, same order of magnitude either way).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

G = 9.81


@dataclass(frozen=True)
class PendulumParams:
    L: float = 0.15  # pole length, m
    m_rod: float = 0.050  # kg
    m_bracket: float = 0.200  # kg
    r_b: float = 0.0  # bracket lumped-mass distance from pivot, m
    b_damping: float = 0.01  # N*m*s/rad -- unmeasured guess, only documented reference in this repo

    @property
    def M(self) -> float:
        return self.m_rod + self.m_bracket

    @property
    def d_com(self) -> float:
        return (self.m_rod * (self.L / 2) + self.m_bracket * self.r_b) / self.M

    @property
    def I_p(self) -> float:
        return (1.0 / 3.0) * self.m_rod * self.L**2 + self.m_bracket * self.r_b**2

    @property
    def natural_freq_hz(self) -> float:
        k = self.M * self.d_com / self.I_p
        return float(np.sqrt(k * G) / (2 * np.pi))

    @property
    def min_energy_j(self) -> float:
        """Energy to go from hanging (-d_com) to inverted (+d_com), ignoring damping."""
        return self.M * G * 2 * self.d_com


DEFAULT_PARAMS = PendulumParams()


def theta_ddot(theta: float, theta_dot: float, a: float, p: PendulumParams = DEFAULT_PARAMS) -> float:
    return (
        -(p.M * G * p.d_com / p.I_p) * np.sin(theta)
        - (p.M * p.d_com / p.I_p) * a * np.cos(theta)
        - (p.b_damping / p.I_p) * theta_dot
    )


def theta_ddot_2d(theta: float, theta_dot: float, a_x: float, a_z: float, p: PendulumParams = DEFAULT_PARAMS) -> float:
    """Generalization of theta_ddot allowing the pivot to also accelerate
    vertically (a_z). Derived the same way (Euler-Lagrange with pivot
    position (x_c(t), z_c(t)) prescribed): the a_z term couples through
    sin(theta) instead of cos(theta) -- structurally identical to how
    gravity couples in, i.e. a_z acts like a time-varying modification to
    effective gravity in the pivot's accelerating frame (the classic
    "Kapitza pendulum" mechanism: pump a swing by raising/lowering your
    center of mass, not just pushing horizontally). This is a genuinely
    different energy-injection channel from a_x -- worth combining, not a
    redundant restatement of the same physics."""
    k = p.M * p.d_com / p.I_p
    return -k * G * np.sin(theta) - k * (a_x * np.cos(theta) + a_z * np.sin(theta)) - (p.b_damping / p.I_p) * theta_dot


def energy(theta: float, theta_dot: float, p: PendulumParams = DEFAULT_PARAMS) -> float:
    """Mechanical energy relative to the pivot, hanging (theta=0) at the minimum."""
    return 0.5 * p.I_p * theta_dot**2 - p.M * G * p.d_com * np.cos(theta)


def dist_from_inverted(theta: float) -> float:
    """Angular distance from theta=pi (mod 2pi), always in [0, pi]."""
    return min(abs(theta - np.pi), abs(theta + np.pi), abs(((theta % (2 * np.pi)) - np.pi)))


def simulate(
    accel_fn,
    *,
    t_max: float,
    dt: float = 0.001,
    p: PendulumParams = DEFAULT_PARAMS,
    x_bounds: tuple[float, float] | None = None,
    theta0: float = 0.001,
) -> dict:
    """Integrate the pendulum under a prescribed base-acceleration profile
    ``accel_fn(t, x_c, v_c, theta, theta_dot) -> a``. If ``x_bounds`` is
    given, the base position is hard-clipped there (mirrors the real,
    physical rail limit) -- accel_fn is responsible for not commanding
    something that fights the wall; this function does not auto-correct
    accel_fn's choices, it only prevents position from leaving the box.

    Returns time series + summary (best/final swing angle reached, whether
    it got within 0.15 rad of inverted, peak base velocity/position)."""
    n = int(t_max / dt)
    t = 0.0
    theta, theta_dot = theta0, 0.0
    x_c, v_c = 0.0, 0.0
    ts, thetas, x_cs, v_cs = [], [], [], []
    best_dist = dist_from_inverted(theta)
    flipped_at = None
    for _ in range(n):
        a = accel_fn(t, x_c, v_c, theta, theta_dot)
        if x_bounds is not None:
            lo, hi = x_bounds
            if x_c >= hi and a > 0:
                a = 0.0
            elif x_c <= lo and a < 0:
                a = 0.0
        theta_dot += theta_ddot(theta, theta_dot, a, p) * dt
        theta += theta_dot * dt
        v_c += a * dt
        x_c += v_c * dt
        if x_bounds is not None:
            x_c = float(np.clip(x_c, x_bounds[0], x_bounds[1]))
        t += dt
        ts.append(t)
        thetas.append(theta)
        x_cs.append(x_c)
        v_cs.append(v_c)
        d = dist_from_inverted(theta)
        best_dist = min(best_dist, d)
        if flipped_at is None and d < 0.15:
            flipped_at = t

    return {
        "t": np.array(ts),
        "theta": np.array(thetas),
        "x_c": np.array(x_cs),
        "v_c": np.array(v_cs),
        "best_dist_from_inverted": best_dist,
        "flipped_at": flipped_at,
        "peak_v_c": float(np.max(np.abs(v_cs))) if v_cs else 0.0,
        "peak_x_c": float(np.max(np.abs(x_cs))) if x_cs else 0.0,
    }


def simulate_2d(
    accel_fn,
    *,
    t_max: float,
    dt: float = 0.001,
    p: PendulumParams = DEFAULT_PARAMS,
    x_bounds: tuple[float, float] | None = None,
    z_bounds: tuple[float, float] | None = None,
    theta0: float = 0.001,
) -> dict:
    """Same as simulate() but the pivot can also accelerate vertically.
    ``accel_fn(t, x_c, v_c, z_c, v_z, theta, theta_dot) -> (a_x, a_z)``.
    x_bounds/z_bounds are independent hard clips -- typically X is tightly
    rail-bounded and Z is either unbounded or loosely bounded, reflecting
    that this task is far less constrained vertically than horizontally."""
    n = int(t_max / dt)
    t = 0.0
    theta, theta_dot = theta0, 0.0
    x_c, v_c = 0.0, 0.0
    z_c, v_z = 0.0, 0.0
    ts, thetas, x_cs, z_cs = [], [], [], []
    best_dist = dist_from_inverted(theta)
    flipped_at = None
    for _ in range(n):
        a_x, a_z = accel_fn(t, x_c, v_c, z_c, v_z, theta, theta_dot)
        if x_bounds is not None:
            lo, hi = x_bounds
            if x_c >= hi and a_x > 0:
                a_x = 0.0
            elif x_c <= lo and a_x < 0:
                a_x = 0.0
        if z_bounds is not None:
            lo, hi = z_bounds
            if z_c >= hi and a_z > 0:
                a_z = 0.0
            elif z_c <= lo and a_z < 0:
                a_z = 0.0
        theta_dot += theta_ddot_2d(theta, theta_dot, a_x, a_z, p) * dt
        theta += theta_dot * dt
        v_c += a_x * dt
        x_c += v_c * dt
        v_z += a_z * dt
        z_c += v_z * dt
        if x_bounds is not None:
            x_c = float(np.clip(x_c, x_bounds[0], x_bounds[1]))
        if z_bounds is not None:
            z_c = float(np.clip(z_c, z_bounds[0], z_bounds[1]))
        t += dt
        ts.append(t)
        thetas.append(theta)
        x_cs.append(x_c)
        z_cs.append(z_c)
        d = dist_from_inverted(theta)
        best_dist = min(best_dist, d)
        if flipped_at is None and d < 0.15:
            flipped_at = t

    return {
        "t": np.array(ts),
        "theta": np.array(thetas),
        "x_c": np.array(x_cs),
        "z_c": np.array(z_cs),
        "best_dist_from_inverted": best_dist,
        "flipped_at": flipped_at,
        "peak_x_c": float(np.max(np.abs(x_cs))) if x_cs else 0.0,
        "peak_z_c": float(np.max(np.abs(z_cs))) if z_cs else 0.0,
    }
