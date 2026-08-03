#!/usr/bin/env python3
"""Solve for a bang-bang base-acceleration profile that swings the pendulum
(tools/swingup/pendulum_model.py) from hanging to inverted within a bounded
rail, via direct optimization over switch timing -- NOT the naive
"instantaneously maximize dE/dt" heuristic from earlier tonight, which
saturates around 20-25 degrees because it reacts to the rail wall instead
of anticipating it and desynchronizes from the pendulum's own (amplitude-
dependent) phase every time it bounces off a limit.

Parameterization: N bang-bang segments of fixed amplitude A_max, alternating
sign, with free (optimized) durations -- this is the standard form for
time/effort-constrained pendulum swing-up (a finite sequence of "pumps"),
tractable with a global optimizer (scipy.optimize.differential_evolution)
since the objective (final distance from inverted) is highly non-convex in
switch timing.

Run directly for a quick answer:
    python tools/swingup/solve_swingup_trajectory.py --a-max 3.0 --rail -0.15 0.20
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from tools.swingup.pendulum_model import DEFAULT_PARAMS, PendulumParams, dist_from_inverted, simulate  # noqa: E402


def build_bangbang_accel_fn(switch_times: np.ndarray, initial_sign: int, a_max: float):
    def accel_fn(t, x_c, v_c, theta, theta_dot):
        idx = int(np.searchsorted(switch_times, t))
        sign = initial_sign if idx % 2 == 0 else -initial_sign
        return sign * a_max

    return accel_fn


def greedy_phase_switch_seed(
    *, a_max: float, rail: tuple[float, float], p: PendulumParams, n_segments: int, dt: float = 0.002
) -> tuple[np.ndarray, int]:
    """Generate a physically-sensible starting guess for the optimizer by
    running an adaptive online law that switches acceleration sign at each
    theta_dot zero-crossing (i.e. pump exactly at the extremes of the
    swing, like timing a push at the top of a playground-swing's arc) --
    this correctly tracks the pendulum's true, amplitude-dependent period
    (unlike a fixed-frequency drive) without needing switch times chosen in
    advance. Used only to seed the optimizer's initial population; the
    optimizer is free to move away from it."""
    theta, theta_dot, x_c, v_c, t = 0.001, 0.0, 0.0, 0.0, 0.0
    sign = 1
    switch_times: list[float] = []
    t_max = n_segments * 4.0
    n_steps = int(t_max / dt)
    for _ in range(n_steps):
        a = sign * a_max
        if x_c >= rail[1] and a > 0:
            a = 0.0
        elif x_c <= rail[0] and a < 0:
            a = 0.0
        from tools.swingup.pendulum_model import theta_ddot as _tddot

        prev_theta_dot = theta_dot
        theta_dot += _tddot(theta, theta_dot, a, p) * dt
        theta += theta_dot * dt
        v_c += a * dt
        x_c = float(np.clip(x_c + v_c * dt, rail[0], rail[1]))
        t += dt
        if prev_theta_dot * theta_dot < 0:  # zero crossing -> flip pump direction
            switch_times.append(t)
            sign = -sign
        if len(switch_times) >= n_segments:
            break
    durations = np.diff([0.0] + switch_times)
    if len(durations) < n_segments:
        durations = np.concatenate([durations, np.full(n_segments - len(durations), 1.0)])
    initial_sign = 1
    return durations[:n_segments], initial_sign


def objective(
    durations: np.ndarray,
    initial_sign: int,
    a_max: float,
    rail: tuple[float, float],
    p: PendulumParams,
    dt: float,
) -> float:
    switch_times = np.cumsum(np.abs(durations))
    t_max = float(switch_times[-1])
    accel_fn = build_bangbang_accel_fn(switch_times, initial_sign, a_max)
    result = simulate(accel_fn, t_max=t_max, dt=dt, p=p, x_bounds=rail)
    return result["best_dist_from_inverted"]


def solve(
    *,
    a_max: float,
    rail: tuple[float, float],
    n_segments: int = 8,
    max_segment_s: float = 4.0,
    p: PendulumParams = DEFAULT_PARAMS,
    search_dt: float = 0.005,
    seed: int = 0,
    maxiter: int = 80,
    popsize: int = 20,
) -> dict:
    from scipy.optimize import differential_evolution

    bounds = [(0.05, max_segment_s)] * n_segments
    best = None
    for initial_sign in (1, -1):
        seed_durations, _ = greedy_phase_switch_seed(a_max=a_max, rail=rail, p=p, n_segments=n_segments)
        seed_durations = np.clip(seed_durations, 0.05, max_segment_s)
        res = differential_evolution(
            objective,
            bounds,
            args=(initial_sign, a_max, rail, p, search_dt),
            seed=seed,
            maxiter=maxiter,
            popsize=popsize,
            tol=1e-4,
            polish=True,
            workers=1,
            x0=seed_durations,
        )
        if best is None or res.fun < best["cost"]:
            switch_times = np.cumsum(np.abs(res.x))
            best = {"cost": res.fun, "durations": res.x, "switch_times": switch_times, "initial_sign": initial_sign}

    # Re-simulate the winner at fine dt for an accurate final report.
    accel_fn = build_bangbang_accel_fn(best["switch_times"], best["initial_sign"], a_max)
    fine = simulate(accel_fn, t_max=float(best["switch_times"][-1]), dt=0.0005, p=p, x_bounds=rail)
    best["fine_result"] = fine
    return best


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--a-max", type=float, default=3.0, help="Bang-bang acceleration amplitude, m/s^2.")
    parser.add_argument("--rail", type=float, nargs=2, default=[-0.15, 0.20], metavar=("LO", "HI"))
    parser.add_argument("--n-segments", type=int, default=8)
    parser.add_argument("--r-b", type=float, default=0.0, help="Bracket lumped-mass offset from pivot, m.")
    parser.add_argument("--maxiter", type=int, default=80)
    args = parser.parse_args()

    p = PendulumParams(r_b=args.r_b)
    print(f"Pendulum: natural_freq={p.natural_freq_hz:.2f}Hz  min_energy={p.min_energy_j*1000:.1f}mJ")
    print(f"a_max={args.a_max} m/s^2  rail={tuple(args.rail)}  n_segments={args.n_segments}\n")

    best = solve(a_max=args.a_max, rail=tuple(args.rail), n_segments=args.n_segments, p=p, maxiter=args.maxiter)
    fine = best["fine_result"]
    print(f"initial_sign={best['initial_sign']}")
    print(f"switch_times={np.round(best['switch_times'], 3).tolist()}")
    print(f"best_dist_from_inverted={fine['best_dist_from_inverted']:.4f} rad ({np.degrees(fine['best_dist_from_inverted']):.1f} deg)")
    print(f"flipped_at={fine['flipped_at']}")
    print(f"peak_v_c={fine['peak_v_c']:.3f} m/s  peak_x_c={fine['peak_x_c']:.3f} m")


if __name__ == "__main__":
    main()
