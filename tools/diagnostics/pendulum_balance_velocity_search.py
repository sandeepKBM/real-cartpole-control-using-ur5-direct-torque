#!/usr/bin/env python3
"""Gain search for the velocity-lane balance controller (kp, kd,
ik_joint_gain) via scipy.optimize.differential_evolution -- see
pendulum_balance_torque_lqr_search.py's docstring for why this (not RL)
is this repo's established tool for this kind of problem, and why the
objective is evaluated only at perturbations where the PASSIVE (kp=kd=0)
baseline is independently confirmed to fail (structurally cannot reward a
do-nothing candidate).

A manual sweep the same day found kp=0.1 essentially inert regardless of
ik_joint_gain/control rate -- this search covers a much wider kp/kd range
to check whether that was simply too weak a gain, not a structural limit.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.optimize import differential_evolution

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import tools.diagnostics.pendulum_balance_test as base  # noqa: E402

TEST_PERTURBATIONS = [0.3, 0.5, 0.8, 1.0]
DURATION_S = 5.0
RATE_HZ = 500.0

# (name, lo, hi, log_scale)
PARAM_BOUNDS = [
    ("kp", 0.01, 50.0, True),
    ("kd", 0.001, 20.0, True),
    ("ik_joint_gain", 10.0, 900.0, False),
]


def decode(x: np.ndarray) -> dict:
    params = {}
    for xi, (name, lo, hi, log) in zip(x, PARAM_BOUNDS):
        if log:
            params[name] = float(np.exp(np.log(lo) + xi * (np.log(hi) - np.log(lo))))
        else:
            params[name] = float(lo + xi * (hi - lo))
    return params


def fitness(x: np.ndarray) -> float:
    p = decode(x)
    total_cost = 0.0
    for pert in TEST_PERTURBATIONS:
        r = base.run_balance_trial(p["kp"], p["kd"], perturbation_rad=pert,
                                    ik_joint_gain=p["ik_joint_gain"],
                                    orientation_priority=True, max_lin_speed_mps=0.25)
        if r["survived_full_duration"]:
            total_cost += abs(r["final_theta_err"])
        else:
            total_cost += 5.0 + max(0.0, DURATION_S - r["fell_at_s"]) / DURATION_S
    return total_cost


def main() -> int:
    base.TEST_DURATION_S = DURATION_S
    base.RATE_HZ = RATE_HZ
    base.CONTROL_DT = 1.0 / RATE_HZ
    base.SUBSTEPS_PER_CONTROL = max(1, round(base.CONTROL_DT / base.PHYSICS_DT))

    result = differential_evolution(
        fitness, bounds=[(0.0, 1.0)] * len(PARAM_BOUNDS),
        maxiter=15, popsize=12, tol=1e-3, seed=0, workers=4, polish=False, disp=True,
    )
    best_params = decode(result.x)
    print(f"\nBest params found: {best_params}")
    print(f"Best fitness: {result.fun}")

    print(f"\n{'pert':>6} {'ACTIVE (found)':>20} {'PASSIVE':>18}")
    for pert in [0.1, 0.2, 0.3, 0.5, 0.8, 1.0, 1.2, 1.5]:
        r_a = base.run_balance_trial(best_params["kp"], best_params["kd"], perturbation_rad=pert,
                                      ik_joint_gain=best_params["ik_joint_gain"],
                                      orientation_priority=True, max_lin_speed_mps=0.25)
        r_p = base.run_balance_trial(0.0, 0.0, perturbation_rad=pert,
                                      ik_joint_gain=best_params["ik_joint_gain"],
                                      orientation_priority=True, max_lin_speed_mps=0.25)
        a = f"S({r_a['final_theta_err']:.3f})" if r_a["survived_full_duration"] else f"F@{r_a['fell_at_s']:.2f}"
        p_s = f"S({r_p['final_theta_err']:.3f})" if r_p["survived_full_duration"] else f"F@{r_p['fell_at_s']:.2f}"
        print(f"{pert:6.2f} {a:>20} {p_s:>18}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
