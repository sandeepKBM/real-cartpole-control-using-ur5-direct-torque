#!/usr/bin/env python3
"""Deliverable 3: guard-clean flip via differential_evolution over swing-up
PARAMETERS (never RL, per this repo's own documented RL gain-scheduling
failure history).

TWO control laws are implemented and both are searched with DE; which one
can actually reach a guard-clean flip is an empirical question this script
answers, not an assumption:

1. ``run_kick_trial`` -- a single open-loop half-sine acceleration pulse
   along tool Y, ``a_cmd(t) = A*sin(pi*t/T)`` for ``0<=t<T``, then coast.
   MEASURED (before committing to a full search) to NOT work at this pose:
   even a single guard-clean-*enforced* kick trips the tool-X-drift guard
   (``|Y-Y0|>0.03m`` in the safety monitor's own axis-role labels -- see
   pendulum_toolY_common.py's header for why "Y" here means "tool X",
   not world Y) after only ~0.3-0.4s, having delivered at most ~16% of
   E_top in a hand-swept probe (A up to 10 m/s^2) -- nowhere near the
   ~190% figure this task's own brief quotes. That figure was evidently
   measured under a different (pose/axis/guard) condition than this one;
   it is not reproduced here and is not assumed to transfer.
2. ``run_energy_shaping_trial`` -- the standard Astrom-Furuta feedback law,
   a_cmd = clip(-k_e*thetadot*cos(phi)*(E_top-E) - k_pos*(x-x0), -a_max,
   a_max), preceded by a brief open-loop seed kick (thetadot=0 makes the
   energy term identically zero at rest). Unlike a single one-way kick,
   this naturally OSCILLATES the cart back and forth in phase with the
   pendulum, so cart position stays bounded near x0 (the k_pos term pulls
   it back) even while energy accumulates over many swing periods -- a much
   better match to a hard, tight drift-guard budget than one one-directional
   impulse. Searched over (k_e, k_pos, a_max, seed_A, seed_T).

Both objectives minimize the closest approach to the inverted angle
(phi=0) reached at any point in the trial, with a guard-trip penalty and a
mild penalty on angular rate at that closest approach (so what the LQR is
later asked to catch is itself plausible -- an arbitrarily fast flyby at
phi=0 is not usefully "captured").
"""

from __future__ import annotations

import argparse
import functools
import json
import multiprocessing as mp
import os
import sys
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import differential_evolution

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.diagnostics import pendulum_toolY_common as C  # noqa: E402


def _de_workers() -> int:
    try:
        mp.set_start_method("fork")
    except RuntimeError:
        pass
    env = os.environ.get("DE_WORKERS")
    if env:
        return max(1, int(env))
    return max(1, int((os.cpu_count() or 2) * 0.9))


def run_kick_trial(
    model: mujoco.MjModel,
    *,
    A: float,
    T: float,
    duration_s: float,
    v_max: float = 1.0,
    enforce_guard: bool = True,
    track_history: bool = False,
) -> dict:
    data = mujoco.MjData(model)
    data.qpos[:6] = C.ARM_Q0
    data.qvel[:] = 0.0
    pend_jid, hub_bid, site_id = C.hinge_ids(model)
    joint_ids = C.joint_ids_for(model)
    hanging, inverted = C.resolve_equilibria(model)
    data.qpos[model.jnt_qposadr[pend_jid]] = hanging
    pend_qpos_adr = model.jnt_qposadr[pend_jid]
    pend_dof_adr = model.jnt_dofadr[pend_jid]
    mujoco.mj_forward(model, data)

    tool_x, tool_y, tool_z = C.tool_frame_world(data, site_id)
    R = C.pumping_rotation_matrix(tool_x, tool_y, tool_z)

    config = C.load_config()
    state0, adapter = C.build_rotated_initial_state_and_adapter(
        model, data, site_id, joint_ids, R=R,
        controller_cfg=config["controller"],
        gravity_mode=config["mujoco"].get("gravity_mode", "gravity_comp"),
    )
    x_ref = float(state0.ee_pos[0])
    reference_quat = state0.reference_quat if state0.reference_quat is not None else state0.ee_quat

    n_steps = int(duration_s * C.RATE_HZ)
    target_x = x_ref
    target_x_vel = 0.0
    guard_fired = False
    guard_reason = None
    first_guard_t = None
    steps_done = 0
    min_abs_phi = np.pi
    thetadot_at_min_phi = 0.0
    t_at_min_phi = 0.0
    peak_cond_j = 0.0
    history = [] if track_history else None

    for step in range(n_steps):
        t = step * C.CONTROL_DT
        u = float(A * np.sin(np.pi * t / T)) if 0.0 <= t < T else 0.0
        target_x_vel = float(np.clip(target_x_vel + u * C.CONTROL_DT, -v_max, v_max))
        target_x = target_x + target_x_vel * C.CONTROL_DT

        state, ee_world = C.build_step_state(
            model, data, site_id=site_id, joint_ids=joint_ids, time_s=t,
            target_x=target_x, target_x_vel=target_x_vel, target_x_accel=u,
            reference_quat=reference_quat, R=R,
        )
        tau, diag = adapter.step(state=state)
        if not diag["safety_ok"] and not guard_fired:
            guard_fired = True
            guard_reason = diag["safety_reason"]
            first_guard_t = t
            if enforce_guard:
                break
        data.ctrl[:6] = tau
        mujoco.mj_step(model, data)
        steps_done += 1

        theta = float(data.qpos[pend_qpos_adr])
        thetadot = float(data.qvel[pend_dof_adr])
        phi = C.wrap_pi(theta - inverted)
        if abs(phi) < min_abs_phi:
            min_abs_phi = abs(phi)
            thetadot_at_min_phi = thetadot
            t_at_min_phi = t

        if track_history:
            jacp = np.zeros((3, model.nv))
            jacr = np.zeros((3, model.nv))
            mujoco.mj_jacSite(model, data, jacp, jacr, site_id)
            J6 = np.vstack([jacp[:, :6], jacr[:, :6]])
            s = np.linalg.svd(J6, compute_uv=False)
            cond_j = float(s[0] / max(s[-1], 1e-12))
            peak_cond_j = max(peak_cond_j, cond_j)
            history.append({
                "t": t, "phi_deg": float(np.degrees(phi)), "theta_deg": float(np.degrees(theta)),
                "thetadot": thetadot, "u": u, "target_x": target_x, "target_x_vel": target_x_vel,
                "safety_ok": bool(diag["safety_ok"]), "cond_j": cond_j,
                "qd_max_abs": float(np.max(np.abs(data.qvel[:6]))),
                "ee_world": ee_world.tolist(),
                "orientation_error_rad": float(diag["orientation_error_norm"]),
            })

    return {
        "A": A, "T": T, "duration_s": duration_s,
        "guard_fired": guard_fired, "guard_reason": guard_reason, "first_guard_t": first_guard_t,
        "steps_done": steps_done, "n_steps": n_steps,
        "min_abs_phi_rad": min_abs_phi, "min_abs_phi_deg": float(np.degrees(min_abs_phi)),
        "thetadot_at_min_phi": thetadot_at_min_phi, "t_at_min_phi": t_at_min_phi,
        "inverted_angle": inverted, "hanging_angle": hanging,
        "peak_cond_j": peak_cond_j,
        "history": history,
    }


def objective(params, model, duration_s: float) -> float:
    A, T = float(params[0]), float(np.clip(params[1], 0.05, 3.0))
    res = run_kick_trial(model, A=A, T=T, duration_s=duration_s)
    cost = res["min_abs_phi_rad"]
    if res["guard_fired"]:
        cost += 5.0
    # Mild penalty on angular rate at closest approach past ~1.2x the LQR
    # cascade's own tuning envelope (|thetadot|<=4 rad/s) -- discourages a
    # kick so hard the flyby is uncapturable.
    if abs(res["thetadot_at_min_phi"]) > 4.0:
        cost += 0.3 * (abs(res["thetadot_at_min_phi"]) - 4.0)
    return cost


def search(model, *, maxiter: int, popsize: int, seed: int, duration_s: float,
           a_bounds=(-8.0, 8.0), t_bounds=(0.1, 1.5)) -> dict:
    bounds = [a_bounds, t_bounds]
    res = differential_evolution(
        functools.partial(objective, model=model, duration_s=duration_s),
        bounds, maxiter=maxiter, popsize=popsize, tol=1e-4, seed=seed,
        workers=_de_workers(), polish=False,
    )
    A, T = float(res.x[0]), float(np.clip(res.x[1], *t_bounds))
    best = run_kick_trial(model, A=A, T=T, duration_s=duration_s, track_history=True)
    return {"A": A, "T": T, "cost": float(res.fun), "best_trial": best}


def run_energy_shaping_trial(
    model: mujoco.MjModel,
    *,
    k_e: float,
    k_pos: float,
    a_max: float,
    seed_A: float,
    seed_T: float,
    duration_s: float,
    k_vel: float = 0.0,
    v_max: float = 1.0,
    enforce_guard: bool = True,
    track_history: bool = False,
) -> dict:
    """Astrom-Furuta energy-shaping law along tool Y, with a brief open-loop
    seed kick to bootstrap thetadot away from exactly 0 (the energy term is
    identically zero at rest). See module docstring for the derivation and
    for why this is expected to respect a tight drift-guard budget better
    than a single one-way kick: the k_pos term pulls the cart back toward
    x0 every cycle, so net drift stays bounded while energy still
    accumulates swing over swing."""
    data = mujoco.MjData(model)
    data.qpos[:6] = C.ARM_Q0
    data.qvel[:] = 0.0
    pend_jid, hub_bid, site_id = C.hinge_ids(model)
    joint_ids = C.joint_ids_for(model)
    hanging, inverted = C.resolve_equilibria(model)
    data.qpos[model.jnt_qposadr[pend_jid]] = hanging
    pend_qpos_adr = model.jnt_qposadr[pend_jid]
    pend_dof_adr = model.jnt_dofadr[pend_jid]
    mujoco.mj_forward(model, data)

    const = C.default_constants()
    mgr, i_pivot, e_top = const.mgr_nm, const.i_pivot_kgm2, const.e_top_j

    tool_x, tool_y, tool_z = C.tool_frame_world(data, site_id)
    R = C.pumping_rotation_matrix(tool_x, tool_y, tool_z)

    config = C.load_config()
    state0, adapter = C.build_rotated_initial_state_and_adapter(
        model, data, site_id, joint_ids, R=R,
        controller_cfg=config["controller"],
        gravity_mode=config["mujoco"].get("gravity_mode", "gravity_comp"),
    )
    x_ref = float(state0.ee_pos[0])
    reference_quat = state0.reference_quat if state0.reference_quat is not None else state0.ee_quat

    n_steps = int(duration_s * C.RATE_HZ)
    target_x = x_ref
    target_x_vel = 0.0
    guard_fired = False
    guard_reason = None
    first_guard_t = None
    steps_done = 0
    min_abs_phi = np.pi
    thetadot_at_min_phi = 0.0
    t_at_min_phi = 0.0
    peak_cond_j = 0.0
    peak_abs_x_rel = 0.0
    history = [] if track_history else None

    for step in range(n_steps):
        t = step * C.CONTROL_DT
        theta = float(data.qpos[pend_qpos_adr])
        thetadot = float(data.qvel[pend_dof_adr])
        phi_from_hanging = C.wrap_pi(theta - hanging)  # 0 at hanging, +-pi at inverted
        E = 0.5 * i_pivot * thetadot ** 2 + mgr * (1.0 - np.cos(phi_from_hanging))
        x_rel = target_x - x_ref  # commanded reference position, not measured -- matches predecessor convention
        peak_abs_x_rel = max(peak_abs_x_rel, abs(x_rel))

        if t < seed_T:
            u = float(seed_A * np.sin(np.pi * t / seed_T))
        else:
            # k_vel: extra damping on the REFERENCE velocity itself (not
            # measured EE velocity) -- matches the validated recipe in
            # render_annotated_swingup_flip.py's own energy-shaping law
            # (a_recenter = -k_pos*(target_x-x0) - k_vel*target_x_vel), added
            # 2026-08-14 after a parallel finding pointed at that script as a
            # working method (different pose/asset, not a drop-in gain set --
            # see this module's own measured result at THIS pose/asset).
            u = (
                -k_e * thetadot * np.cos(phi_from_hanging) * (e_top - E)
                - k_pos * x_rel
                - k_vel * target_x_vel
            )
        u = float(np.clip(u, -a_max, a_max))

        target_x_vel = float(np.clip(target_x_vel + u * C.CONTROL_DT, -v_max, v_max))
        target_x = target_x + target_x_vel * C.CONTROL_DT

        state, ee_world = C.build_step_state(
            model, data, site_id=site_id, joint_ids=joint_ids, time_s=t,
            target_x=target_x, target_x_vel=target_x_vel, target_x_accel=u,
            reference_quat=reference_quat, R=R,
        )
        tau, diag = adapter.step(state=state)
        if not diag["safety_ok"] and not guard_fired:
            guard_fired = True
            guard_reason = diag["safety_reason"]
            first_guard_t = t
            if enforce_guard:
                break
        data.ctrl[:6] = tau
        mujoco.mj_step(model, data)
        steps_done += 1

        theta_new = float(data.qpos[pend_qpos_adr])
        thetadot_new = float(data.qvel[pend_dof_adr])
        phi_inv = C.wrap_pi(theta_new - inverted)
        if abs(phi_inv) < min_abs_phi:
            min_abs_phi = abs(phi_inv)
            thetadot_at_min_phi = thetadot_new
            t_at_min_phi = t

        if track_history:
            jacp = np.zeros((3, model.nv))
            jacr = np.zeros((3, model.nv))
            mujoco.mj_jacSite(model, data, jacp, jacr, site_id)
            J6 = np.vstack([jacp[:, :6], jacr[:, :6]])
            s = np.linalg.svd(J6, compute_uv=False)
            cond_j = float(s[0] / max(s[-1], 1e-12))
            peak_cond_j = max(peak_cond_j, cond_j)
            history.append({
                "t": t, "phi_deg": float(np.degrees(phi_inv)), "theta_deg": float(np.degrees(theta_new)),
                "thetadot": thetadot_new, "u": u, "target_x": target_x, "target_x_vel": target_x_vel,
                "E": E, "safety_ok": bool(diag["safety_ok"]), "cond_j": cond_j,
                "qd_max_abs": float(np.max(np.abs(data.qvel[:6]))),
                "ee_world": ee_world.tolist(),
                "orientation_error_rad": float(diag["orientation_error_norm"]),
            })

    return {
        "k_e": k_e, "k_pos": k_pos, "a_max": a_max, "seed_A": seed_A, "seed_T": seed_T,
        "duration_s": duration_s,
        "guard_fired": guard_fired, "guard_reason": guard_reason, "first_guard_t": first_guard_t,
        "steps_done": steps_done, "n_steps": n_steps,
        "min_abs_phi_rad": min_abs_phi, "min_abs_phi_deg": float(np.degrees(min_abs_phi)),
        "thetadot_at_min_phi": thetadot_at_min_phi, "t_at_min_phi": t_at_min_phi,
        "inverted_angle": inverted, "hanging_angle": hanging,
        "peak_cond_j": peak_cond_j, "peak_abs_x_rel_m": peak_abs_x_rel,
        "history": history,
    }


def objective_energy_shaping(params, model, duration_s: float) -> float:
    k_e = float(np.clip(params[0], 0.0, 400.0))
    k_pos = float(np.clip(params[1], 0.0, 50.0))
    a_max = float(np.clip(params[2], 0.3, 10.0))
    seed_A = float(params[3])
    seed_T = float(np.clip(params[4], 0.02, 0.5))
    k_vel = float(np.clip(params[5], 0.0, 30.0))
    res = run_energy_shaping_trial(
        model, k_e=k_e, k_pos=k_pos, a_max=a_max, seed_A=seed_A, seed_T=seed_T,
        k_vel=k_vel, duration_s=duration_s,
    )
    cost = res["min_abs_phi_rad"]
    if res["guard_fired"]:
        cost += 5.0
    if abs(res["thetadot_at_min_phi"]) > 4.0:
        cost += 0.3 * (abs(res["thetadot_at_min_phi"]) - 4.0)
    return cost


def search_energy_shaping(model, *, maxiter: int, popsize: int, seed: int, duration_s: float,
                           init=None) -> dict:
    bounds = [
        (0.0, 400.0),   # k_e
        (0.0, 50.0),    # k_pos
        (0.3, 10.0),    # a_max
        (-8.0, 8.0),    # seed_A
        (0.02, 0.5),    # seed_T
        (0.0, 30.0),    # k_vel
    ]
    kwargs = dict(
        bounds=bounds, maxiter=maxiter, popsize=popsize, tol=1e-4, seed=seed,
        workers=_de_workers(), polish=False,
    )
    if init is not None:
        kwargs["init"] = init
    res = differential_evolution(
        functools.partial(objective_energy_shaping, model=model, duration_s=duration_s),
        **kwargs,
    )
    k_e, k_pos, a_max = float(res.x[0]), float(res.x[1]), float(np.clip(res.x[2], 0.3, 10.0))
    seed_A, seed_T = float(res.x[3]), float(np.clip(res.x[4], 0.02, 0.5))
    k_vel = float(np.clip(res.x[5], 0.0, 30.0))
    best = run_energy_shaping_trial(
        model, k_e=k_e, k_pos=k_pos, a_max=a_max, seed_A=seed_A, seed_T=seed_T,
        k_vel=k_vel, duration_s=duration_s, track_history=True,
    )
    return {
        "k_e": k_e, "k_pos": k_pos, "a_max": a_max, "seed_A": seed_A, "seed_T": seed_T,
        "k_vel": k_vel, "cost": float(res.fun), "best_trial": best,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--law", choices=["kick", "energy_shaping"], default="energy_shaping")
    parser.add_argument("--maxiter", type=int, default=25)
    parser.add_argument("--popsize", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--duration-s", type=float, default=3.0)
    parser.add_argument("--output-json", type=str, default=None)
    args = parser.parse_args(argv)

    model = C.build_model()
    if args.law == "kick":
        print("=== searching single-kick params (A, T) via differential_evolution, tool-Y pumping ===")
        result = search(model, maxiter=args.maxiter, popsize=args.popsize, seed=args.seed,
                         duration_s=args.duration_s)
        print(f"A={result['A']:.4f} m/s^2, T={result['T']:.4f} s, cost={result['cost']:.4f}")
    else:
        print("=== searching energy-shaping params (k_e, k_pos, a_max, seed_A, seed_T) "
              "via differential_evolution, tool-Y pumping ===")
        result = search_energy_shaping(model, maxiter=args.maxiter, popsize=args.popsize,
                                        seed=args.seed, duration_s=args.duration_s)
        print(f"k_e={result['k_e']:.4f} k_pos={result['k_pos']:.4f} a_max={result['a_max']:.4f} "
              f"seed_A={result['seed_A']:.4f} seed_T={result['seed_T']:.4f} cost={result['cost']:.4f}")
    best = result["best_trial"]
    print(f"min|phi| = {best['min_abs_phi_deg']:.3f} deg at t={best['t_at_min_phi']:.3f}s, "
          f"thetadot there = {best['thetadot_at_min_phi']:.3f} rad/s")
    print(f"guard_fired={best['guard_fired']} reason={best['guard_reason']} "
          f"peak_cond_j={best['peak_cond_j']:.2f} steps_done={best['steps_done']}/{best['n_steps']}")

    if args.output_json:
        with Path(args.output_json).open("w") as fp:
            json.dump(result, fp, indent=2, default=str)
        print("wrote", args.output_json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
