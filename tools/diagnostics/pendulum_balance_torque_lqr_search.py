#!/usr/bin/env python3
"""Gain search for the torque-lane LQR balance controller via
scipy.optimize.differential_evolution -- a gradient-free global optimizer,
matching this repo's own established preference for this class of problem
(see velocity_gain_tuning/optimize.py's docstring: bandit/black-box search
over RL/rl_gain_scheduling/, whose failure modes -- deceptive sit-still
optimum, exploration collapse -- are exactly the trap this session's
manual balance-gain search kept falling into: a candidate that behaves
like doing nothing looks deceptively fine unless the objective is built
to catch it).

Searches (q_arm_pos, q_arm_vel, q_pend_angle, q_pend_vel, r_weight) --
log-scaled, since a manual sweep found the useful range spans several
orders of magnitude and a bad combination is either torque-saturating
(chatters, fails) or produces the same negligible correction as K=0.

Objective is evaluated ONLY at perturbations independently confirmed
(same day, manual sweep) to make the PASSIVE (K=0) baseline fail --
0.1/0.2/0.3/0.4/0.5 rad at the corrected unstable equilibrium (theta=0).
This is deliberate: it structurally cannot reward a "no better than
passive" candidate, since passive itself scores near the floor on every
term in this objective.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.optimize import differential_evolution

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import tools.diagnostics.pendulum_balance_torque_lqr as base  # noqa: E402

TEST_PERTURBATIONS = [0.1, 0.2, 0.3, 0.5, 0.65, 0.8]
DURATION_S = 5.0

# (name, lo, hi, log_scale)
PARAM_BOUNDS = [
    ("q_arm_pos", 1.0, 1.0e5, True),
    ("q_arm_vel", 1.0, 1.0e4, True),
    ("q_pend_angle", 10.0, 1.0e4, True),
    ("q_pend_vel", 0.1, 1.0e3, True),
    ("r_weight", 0.001, 1.0e4, True),
]


def decode(x: np.ndarray) -> dict:
    params = {}
    for xi, (name, lo, hi, log) in zip(x, PARAM_BOUNDS):
        if log:
            params[name] = float(np.exp(np.log(lo) + xi * (np.log(hi) - np.log(lo))))
        else:
            params[name] = float(lo + xi * (hi - lo))
    return params


def fitness(x: np.ndarray, model, inverted_angle) -> float:
    params = decode(x)
    try:
        K, q_eq, diag = base.linearize_and_design_lqr(model, inverted_angle, **params)
    except Exception:
        return 1000.0
    if np.any(np.abs(diag["eigvals_discrete"]) >= 1.0):
        return 1000.0  # linear design itself unstable -- hard reject

    total_cost = 0.0
    for pert in TEST_PERTURBATIONS:
        r = base.run_torque_balance_trial(model, K, q_eq, pert, duration_s=DURATION_S)
        if r["survived_full_duration"]:
            total_cost += abs(r["final_theta_err"])  # reward tight convergence
        else:
            # Penalize failure, scaled by how EARLY it failed (falling
            # immediately is worse than falling near the end) plus a flat
            # penalty so any survival beats any failure.
            total_cost += 5.0 + max(0.0, DURATION_S - r["fell_at_s"]) / DURATION_S
    return total_cost


def main() -> int:
    model = base.compose_ur5e_pendulum_model()
    import mujoco
    data0 = mujoco.MjData(model)
    pend_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "/pendulum_hinge")
    pend_qpos_adr = model.jnt_qposadr[pend_joint_id]
    inverted_angle = base.find_inverted_angle(model, data0, pend_qpos_adr)
    print(f"inverted_angle = {inverted_angle:.6f}")

    # Seed the search with the PREVIOUS search's own best result (found
    # over TEST_PERTURBATIONS up to 0.5 rad only -- survived cleanly
    # 0.05-0.4 rad, failed at 0.5+; this run widens TEST_PERTURBATIONS to
    # explicitly include 0.5-0.8 rad, targeting exactly that known gap).
    seed_raw = {"q_arm_pos": 1008.735572590068, "q_arm_vel": 32.27913994759627,
                "q_pend_angle": 35.85445272321039, "q_pend_vel": 310.35751752823853,
                "r_weight": 4.301938997677817}
    seed_x = []
    for name, lo, hi, log in PARAM_BOUNDS:
        v = seed_raw[name]
        seed_x.append((np.log(v) - np.log(lo)) / (np.log(hi) - np.log(lo)) if log else (v - lo) / (hi - lo))
    init_pop = np.clip(
        np.array(seed_x)[None, :] + np.random.default_rng(0).normal(0, 0.15, size=(30, len(PARAM_BOUNDS))),
        0.0, 1.0,
    )
    init_pop[0] = seed_x  # exact seed included

    result = differential_evolution(
        fitness, bounds=[(0.0, 1.0)] * len(PARAM_BOUNDS), args=(model, inverted_angle),
        init=init_pop, maxiter=25, popsize=12, tol=1e-4, seed=1, workers=4, polish=False,
        disp=True,
    )
    best_params = decode(result.x)
    print(f"\nBest params found: {best_params}")
    print(f"Best fitness: {result.fun}")

    K, q_eq, diag = base.linearize_and_design_lqr(model, inverted_angle, **best_params)
    K_zero = np.zeros_like(K)
    print(f"\n{'pert':>6} {'ACTIVE (found)':>20} {'PASSIVE':>18}")
    for pert in [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0]:
        r_a = base.run_torque_balance_trial(model, K, q_eq, pert, duration_s=8.0)
        r_p = base.run_torque_balance_trial(model, K_zero, q_eq, pert, duration_s=8.0)
        a = f"S({r_a['final_theta_err']:.3f})" if r_a["survived_full_duration"] else f"F@{r_a['fell_at_s']:.2f}"
        p = f"S({r_p['final_theta_err']:.3f})" if r_p["survived_full_duration"] else f"F@{r_p['fell_at_s']:.2f}"
        print(f"{pert:6.2f} {a:>20} {p:>18}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
