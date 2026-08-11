#!/usr/bin/env python3
"""Gain-scheduled X-transport search, CLI-configurable (pose/config/
scenarios/schedule-to-validate are all flags now -- previously required
editing this file's module-level constants for every new pose/experiment,
which doesn't scale and makes it hard to reproduce or diff runs).

Reuses the SAME config-loading + adapter-construction path as the rest of
this session's swing-up work (build_initial_state_and_adapter), then adds
ONE new thing: a live schedule that mutates adapter.controller.cfg.kp_x/
kd_x each cycle as a function of the current |x_error|/|x_speed|, instead
of using the config's fixed kp_x/kd_x for the whole move.

Schedule form (8 free parameters, proven simple/searchable):
  kp_x = kp_base + kp_err_gain * min(|x_err|, err_cap) + kp_vel_gain * min(|x_vel|, vel_cap)
  kd_x = kd_base + kd_err_gain * min(|x_err|, err_cap) + kd_vel_gain * min(|x_vel|, vel_cap)

differential_evolution, NOT RL -- this repo's rl_gain_scheduling/ package
has a documented 4-run failure history (docs/CURRENT_STATUS.md), explicitly
why this script exists instead.

Modes (--mode):
  search-schedule    DE search for the 8-param schedule over --scenarios.
  search-single-gain DE search for a single smoothed (kp_x,kd_x) pair over
                     the same scenarios (a simpler, non-scheduled distillation).
  rigor-sweep        Run the standard 4-category rigor sweep
                     (canonical_grid/long_holds/large_displacements/
                     torque_scale_robustness, both +X/-X) against
                     --schedule-json, --fixed-gains, or the plain baseline.
  all                search-schedule, then rigor-sweep both the found
                     schedule and the fixed baseline, then search-single-gain,
                     then rigor-sweep that too. (what this script used to do
                     unconditionally before this CLI refactor.)

Example:
  python x_transport_gain_scheduling_newpose.py --mode search-schedule \\
      --start-q-rad 0.0 -1.091985784398452 2.0935362786892546 \\
      -2.7685637962327356 1.5620693866337145 0.0

  python x_transport_gain_scheduling_newpose.py --mode rigor-sweep \\
      --schedule-json '{"kp_base": 97.1, ...}'
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import mujoco
import numpy as np
import yaml
from scipy.optimize import differential_evolution

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from simulation.ur5e_mujoco_torque import (  # noqa: E402
    build_initial_state_and_adapter,
    build_mujoco_state,
    x_profile_target,
)
from tools.diagnostics.pendulum_swingup_energy_shaping import (  # noqa: E402
    ARM_Q0 as DEFAULT_ARM_Q0, CONTROL_DT, RATE_HZ,
)

SCENE_XML = REPO_ROOT / "assets" / "ur5e_torque" / "scene.xml"
DEFAULT_CONFIG_PATH = REPO_ROOT / "config" / "ur5e_mujoco_torque_osc_hanging_pose_friction_ff.yaml"
JOINT_NAMES = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
               "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]
SITE_NAME = "attachment_site"

# Default scenario set -- matches this pose's validated envelope from this
# session's real move_hold_transport sweeps (clean X range +-0.25m, clean
# speed ceiling ~0.78-1.10 m/s). Override with --scenarios-json for a
# different pose/envelope instead of editing this file.
DEFAULT_TEST_SCENARIOS = [
    (0.20, 0.30), (0.20, 0.50), (0.25, 0.40), (0.25, 0.60),
    (0.30, 0.50), (0.30, 0.80),
    (-0.20, 0.40), (-0.20, 0.60), (-0.25, 0.50), (-0.25, 0.80),
    (-0.30, 0.60), (-0.30, 1.00),
]

RIGOR_CATEGORY_GRIDS = {
    "canonical_grid": {
        "target_x_deltas": [0.01, 0.02, 0.03, 0.04], "move_durations": [1.0],
        "hold_durations": [1.0, 2.0], "torque_limit_scales": [1.0],
    },
    "long_holds": {
        "target_x_deltas": [0.03, 0.06], "move_durations": [1.0],
        "hold_durations": [4.0, 10.0, 20.0, 30.0], "torque_limit_scales": [1.0],
    },
    "large_displacements": {
        "target_x_deltas": [0.05, 0.10, 0.15, 0.20], "move_durations": [1.0],
        "hold_durations": [1.0, 2.0], "torque_limit_scales": [1.0],
    },
    "torque_scale_robustness": {
        "target_x_deltas": [0.03, 0.06], "move_durations": [1.0],
        "hold_durations": [2.0], "torque_limit_scales": [0.10, 0.25, 0.40, 0.55, 0.70, 0.85, 1.00],
    },
}

SCHEDULE_KEYS = ["kp_base", "kp_err_gain", "kp_vel_gain", "kd_base", "kd_err_gain",
                 "kd_vel_gain", "err_cap", "vel_cap"]


def decode_schedule(x: np.ndarray) -> dict:
    return dict(zip(SCHEDULE_KEYS, x))


def load_controller_config(config_path: Path) -> dict:
    with Path(config_path).open() as fp:
        return yaml.safe_load(fp)


def run_transport_trial(target_x_delta_m: float, move_duration_s: float, duration_s: float,
                         arm_q: np.ndarray, config: dict,
                         schedule: dict | None = None,
                         torque_limit_scale: float = 1.0,
                         fixed_gains: tuple[float, float] | None = None) -> dict:
    """fixed_gains: optional (kp_x, kd_x) pair used FIXED for the whole
    trial, overriding the config's own kp_x/kd_x -- mutually exclusive with
    schedule (schedule takes priority if both given). Neither given -> the
    config's own fixed kp_x/kd_x are used for the whole trial (baseline)."""
    model = mujoco.MjModel.from_xml_path(str(SCENE_XML))
    data = mujoco.MjData(model)
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, SITE_NAME)
    joint_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in JOINT_NAMES]
    data.qpos[:6] = arm_q
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)

    state0, adapter = build_initial_state_and_adapter(
        model, data, site_id, joint_ids,
        controller_cfg=config["controller"],
        transport_axis_index=0,
        target_x_delta=0.0,
        controller_kind="impedance",
        force_hold_current_pose=False,
        gravity_mode=config["mujoco"].get("gravity_mode", "gravity_comp"),
        gravity_source=config["mujoco"].get("gravity_source", "pinocchio"),
        coriolis_feedforward=bool(config["mujoco"].get("coriolis_feedforward", True)),
        torque_limit_scale=torque_limit_scale,
    )
    x0 = float(state0.ee_pos[0])
    base_kp_x = float(config["controller"]["gains"]["kp_x"])
    base_kd_x = float(config["controller"]["gains"]["kd_x"])

    n_steps = int(duration_s * RATE_HZ)
    guard_fired = False
    guard_reason = None
    fell_at = None
    max_abs_x_err = 0.0
    x_err_final = None

    for step in range(n_steps):
        t = step * CONTROL_DT
        target_x, target_x_vel = x_profile_target(
            profile="min_jerk_move_hold", x0=x0, target_x_delta=target_x_delta_m,
            t_s=t, duration_s=duration_s, move_duration_s=move_duration_s,
        )

        ee_x = float(data.site_xpos[site_id][0])
        jacp = np.zeros((3, model.nv))
        jacr = np.zeros((3, model.nv))
        mujoco.mj_jacSite(model, data, jacp, jacr, site_id)
        ee_x_vel = float(jacp[0, :6] @ data.qvel[:6])
        x_err_now = target_x - ee_x

        if schedule is not None:
            err_mag = min(abs(x_err_now), schedule["err_cap"])
            vel_mag = min(abs(ee_x_vel), schedule["vel_cap"])
            adapter.controller.cfg.kp_x = max(1.0, schedule["kp_base"] + schedule["kp_err_gain"] * err_mag + schedule["kp_vel_gain"] * vel_mag)
            adapter.controller.cfg.kd_x = max(0.1, schedule["kd_base"] + schedule["kd_err_gain"] * err_mag + schedule["kd_vel_gain"] * vel_mag)
        elif fixed_gains is not None:
            adapter.controller.cfg.kp_x = fixed_gains[0]
            adapter.controller.cfg.kd_x = fixed_gains[1]
        else:
            adapter.controller.cfg.kp_x = base_kp_x
            adapter.controller.cfg.kd_x = base_kd_x

        state = build_mujoco_state(
            model, data, site_id=site_id, joint_ids=joint_ids, time_s=t, dt_s=CONTROL_DT,
            target_x=target_x, target_x_vel=target_x_vel,
            reference_quat=state0.ee_quat, transport_axis_index=0,
            gravity_compensation=True,
        )
        tau, diag = adapter.step(state=state)
        if not bool(diag.get("safety_ok", True)):
            guard_fired = True
            guard_reason = str(diag.get("safety_reason", ""))
            fell_at = t
            break

        data.ctrl[:6] = np.asarray(tau, dtype=np.float64).reshape(6)
        mujoco.mj_step(model, data)
        max_abs_x_err = max(max_abs_x_err, abs(x_err_now))
        x_err_final = x_err_now

    final_x_error = x_err_final if x_err_final is not None else target_x_delta_m
    achieved_x_delta_m = target_x_delta_m - final_x_error
    return {
        "target_x_delta_m": target_x_delta_m, "move_duration_s": move_duration_s,
        "achieved_x_delta_m": achieved_x_delta_m, "final_x_error_m": final_x_error,
        "max_abs_x_error_m": max_abs_x_err,
        "survived": not guard_fired, "fell_at_s": fell_at, "guard_reason": guard_reason,
    }


def schedule_fitness(x: np.ndarray, arm_q: np.ndarray, config: dict, scenarios: list) -> float:
    schedule = decode_schedule(x)
    total = 0.0
    for dx, move_dur in scenarios:
        r = run_transport_trial(dx, move_dur, duration_s=move_dur + 2.0, arm_q=arm_q, config=config, schedule=schedule)
        if r["survived"]:
            frac_err = abs(r["final_x_error_m"]) / max(abs(dx), 1e-6)
            total += 3.0 * frac_err  # max 3.0
        else:
            total += 4.0 + max(0.0, (move_dur + 2.0) - r["fell_at_s"]) / (move_dur + 2.0)  # min 4.0 > 3.0
    return total


def fixed_gain_fitness(x: np.ndarray, arm_q: np.ndarray, config: dict, scenarios: list) -> float:
    kp_x, kd_x = float(x[0]), float(x[1])
    total = 0.0
    for dx, move_dur in scenarios:
        r = run_transport_trial(dx, move_dur, duration_s=move_dur + 2.0, arm_q=arm_q, config=config,
                                 fixed_gains=(kp_x, kd_x))
        if r["survived"]:
            frac_err = abs(r["final_x_error_m"]) / max(abs(dx), 1e-6)
            total += 3.0 * frac_err
        else:
            total += 4.0 + max(0.0, (move_dur + 2.0) - r["fell_at_s"]) / (move_dur + 2.0)
    return total


def print_scenario_table(arm_q, config, scenarios, columns: dict) -> None:
    """columns: {label: schedule_or_None} evaluated with fixed_gains=None
    unless label starts with 'FIXED:' (parsed as 'FIXED:kp,kd')."""
    header = f"{'dx':>6} {'move_dur':>8} " + " ".join(f"{label:>20}" for label in columns)
    print(header)
    for dx, move_dur in scenarios:
        cells = []
        for label, schedule in columns.items():
            fixed_gains = None
            if isinstance(label, str) and label.startswith("FIXED:"):
                kp, kd = (float(v) for v in label.split(":", 1)[1].split(","))
                fixed_gains = (kp, kd)
            r = run_transport_trial(dx, move_dur, duration_s=move_dur + 2.0, arm_q=arm_q, config=config,
                                     schedule=schedule, fixed_gains=fixed_gains)
            cells.append(f"S(err={r['final_x_error_m']:.4f})" if r["survived"] else f"F@{r['fell_at_s']:.2f}")
        print(f"{dx:6.2f} {move_dur:8.2f} " + " ".join(f"{c:>20}" for c in cells))


def run_search_schedule(arm_q, config, scenarios, maxiter, popsize, seed) -> dict:
    print("=== Baseline (config's own fixed kp_x/kd_x) at search scenarios ===")
    print_scenario_table(arm_q, config, scenarios, {"BASELINE": None})

    bounds = [
        (1.0, 1500.0), (0.0, 3000.0), (0.0, 500.0),   # kp_base, kp_err_gain, kp_vel_gain
        (0.1, 150.0), (0.0, 300.0), (0.0, 200.0),     # kd_base, kd_err_gain, kd_vel_gain
        (0.01, 0.35), (0.05, 1.5),                    # err_cap, vel_cap
    ]
    print("\n=== Searching gain schedule via differential_evolution ===")
    result = differential_evolution(
        lambda x: schedule_fitness(x, arm_q, config, scenarios), bounds,
        maxiter=maxiter, popsize=popsize, tol=1e-5, seed=seed, workers=-1, polish=False,
    )
    schedule = decode_schedule(result.x)
    print(f"Best params: {schedule}")
    print(f"Best fitness: {result.fun}")
    print_scenario_table(arm_q, config, scenarios, {"SCHEDULED": schedule, "BASELINE": None})
    return {"schedule": schedule, "fitness": float(result.fun)}


def run_search_single_gain(arm_q, config, scenarios, maxiter, popsize, seed,
                            compare_schedule: dict | None = None) -> dict:
    print("=== Searching a single FIXED (kp_x, kd_x) pair across the scenario set ===")
    bounds = [(1.0, 1500.0), (0.1, 150.0)]
    result = differential_evolution(
        lambda x: fixed_gain_fitness(x, arm_q, config, scenarios), bounds,
        maxiter=maxiter, popsize=popsize, tol=1e-6, seed=seed, workers=-1, polish=True,
    )
    kp_x, kd_x = float(result.x[0]), float(result.x[1])
    print(f"Best single gains: kp_x={kp_x:.4f}, kd_x={kd_x:.4f}")
    print(f"Best fitness: {result.fun}")
    columns = {f"FIXED:{kp_x},{kd_x}": None, "BASELINE": None}
    if compare_schedule is not None:
        columns = {f"FIXED:{kp_x},{kd_x}": None, "SCHEDULED": compare_schedule, "BASELINE": None}
    print_scenario_table(arm_q, config, scenarios, columns)
    return {"kp_x": kp_x, "kd_x": kd_x, "fitness": float(result.fun)}


def run_rigor_sweep(arm_q, config, schedule: dict | None = None,
                     fixed_gains: tuple[float, float] | None = None, label: str = "") -> dict:
    results_by_category = {}
    for category, grid in RIGOR_CATEGORY_GRIDS.items():
        rows = []
        for dx in grid["target_x_deltas"]:
            for move_dur in grid["move_durations"]:
                for hold_dur in grid["hold_durations"]:
                    for tscale in grid["torque_limit_scales"]:
                        for sign in (1.0, -1.0):  # AGENTS.md: always sweep both +X and -X
                            signed_dx = sign * dx
                            r = run_transport_trial(
                                signed_dx, move_dur, duration_s=move_dur + hold_dur,
                                arm_q=arm_q, config=config,
                                schedule=schedule, fixed_gains=fixed_gains, torque_limit_scale=tscale,
                            )
                            rows.append((signed_dx, move_dur, hold_dur, tscale, r))
        n_valid = sum(1 for *_, r in rows if r["survived"])
        results_by_category[category] = {"n_valid": n_valid, "n_total": len(rows), "rows": rows}
        print(f"[{label}] {category}: {n_valid}/{len(rows)}")
        for signed_dx, move_dur, hold_dur, tscale, r in rows:
            if not r["survived"]:
                print(f"    FAIL dx={signed_dx:+.3f} move_dur={move_dur:.2f} hold={hold_dur:.1f} "
                      f"tscale={tscale:.2f} fell_at={r['fell_at_s']:.2f} reason={r['guard_reason']}")
    total_valid = sum(v["n_valid"] for v in results_by_category.values())
    total_all = sum(v["n_total"] for v in results_by_category.values())
    print(f"[{label}] TOTAL: {total_valid}/{total_all}")
    return results_by_category


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=["search-schedule", "search-single-gain", "rigor-sweep", "all"],
                   default="all")
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH,
                   help="Controller YAML config path.")
    p.add_argument("--start-q-rad", type=float, nargs=6, default=None, metavar=("Q1", "Q2", "Q3", "Q4", "Q5", "Q6"),
                   help="6-joint start pose in radians. Default: this session's mega-pose-search winner.")
    p.add_argument("--scenarios-json", type=str, default=None,
                   help='JSON list of [dx, move_duration_s] pairs, e.g. \'[[0.2,0.3],[-0.2,0.4]]\'. '
                        "Default: this pose's own validated envelope scenarios.")
    p.add_argument("--schedule-json", type=str, default=None,
                   help="JSON object with the 8 schedule keys, for --mode rigor-sweep.")
    p.add_argument("--fixed-gains", type=float, nargs=2, default=None, metavar=("KP_X", "KD_X"),
                   help="Single fixed (kp_x, kd_x) pair, for --mode rigor-sweep.")
    p.add_argument("--maxiter", type=int, default=40)
    p.add_argument("--popsize", type=int, default=16)
    p.add_argument("--single-gain-maxiter", type=int, default=15)
    p.add_argument("--single-gain-popsize", type=int, default=10)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--output-json", type=Path, default=None,
                   help="Write the final result dict to this path as JSON.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    arm_q = np.array(args.start_q_rad) if args.start_q_rad is not None else DEFAULT_ARM_Q0
    config = load_controller_config(args.config)
    scenarios = json.loads(args.scenarios_json) if args.scenarios_json else DEFAULT_TEST_SCENARIOS
    scenarios = [tuple(s) for s in scenarios]

    output: dict = {}

    if args.mode == "search-schedule":
        output = run_search_schedule(arm_q, config, scenarios, args.maxiter, args.popsize, args.seed)

    elif args.mode == "search-single-gain":
        compare_schedule = json.loads(args.schedule_json) if args.schedule_json else None
        output = run_search_single_gain(arm_q, config, scenarios, args.single_gain_maxiter,
                                         args.single_gain_popsize, args.seed, compare_schedule=compare_schedule)

    elif args.mode == "rigor-sweep":
        schedule = json.loads(args.schedule_json) if args.schedule_json else None
        fixed_gains = tuple(args.fixed_gains) if args.fixed_gains else None
        label = "SCHEDULE" if schedule else ("FIXED" if fixed_gains else "BASELINE")
        output = run_rigor_sweep(arm_q, config, schedule=schedule, fixed_gains=fixed_gains, label=label)

    elif args.mode == "all":
        print("########## STAGE 1: search for a gain schedule ##########")
        search_result = run_search_schedule(arm_q, config, scenarios, args.maxiter, args.popsize, args.seed)
        schedule = search_result["schedule"]

        print("\n########## STAGE 2: rigor-sweep the found schedule vs. baseline ##########")
        rigor_schedule = run_rigor_sweep(arm_q, config, schedule=schedule, label="SCHEDULE")
        rigor_baseline = run_rigor_sweep(arm_q, config, label="BASELINE")

        print("\n########## STAGE 3: search for a single smoothed fixed gain ##########")
        single = run_search_single_gain(arm_q, config, scenarios, args.single_gain_maxiter,
                                         args.single_gain_popsize, args.seed, compare_schedule=schedule)

        print("\n########## STAGE 4: rigor-sweep the single fixed gain ##########")
        rigor_single = run_rigor_sweep(arm_q, config, fixed_gains=(single["kp_x"], single["kd_x"]), label="SINGLE-GAIN")

        output = {
            "schedule": schedule, "schedule_fitness": search_result["fitness"],
            "single_gain": single,
            "rigor_schedule_total": f"{sum(v['n_valid'] for v in rigor_schedule.values())}/{sum(v['n_total'] for v in rigor_schedule.values())}",
            "rigor_baseline_total": f"{sum(v['n_valid'] for v in rigor_baseline.values())}/{sum(v['n_total'] for v in rigor_baseline.values())}",
            "rigor_single_gain_total": f"{sum(v['n_valid'] for v in rigor_single.values())}/{sum(v['n_total'] for v in rigor_single.values())}",
        }

    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with args.output_json.open("w") as fp:
            json.dump(output, fp, indent=2, default=str)
        print(f"\nWrote result to {args.output_json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
