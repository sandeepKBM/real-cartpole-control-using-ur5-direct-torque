#!/usr/bin/env python3
"""Gain-scheduled X-transport search for the torque-lane impedance
controller: instead of one FIXED (kp_x, kd_x) for an entire move, vary
them live each cycle as a function of the CURRENT |x_error| and
|x_speed| -- covering more of the (displacement, duration) transport
envelope than any single fixed gain pair can, the same logic already
applied to the pendulum-balance LQR work this session (a fixed linear
design is only locally valid; scheduling covers more ground by adapting
to where in the state space the system currently is).

Reuses this repo's own validated simulation machinery directly (NOT
reimplemented): simulation.ur5e_mujoco_torque's MujocoUR5eTorqueAdapter
(torque limits, gravity/Coriolis, safety monitor, all already correct)
wraps controller_core.x_axis_cartesian_impedance's real controller. The
only new piece is a live schedule that mutates controller.cfg.kp_x/kd_x
each cycle BEFORE calling adapter.step() -- the same "mutate cfg live"
pattern velocity_gain_tuning/envs/velocity_transport_env.py already uses
for its own (non-scheduled, fixed-per-episode) gain search.

Schedule form (additive, position+speed jointly, 6 free parameters):
  kp_x = kp_base + kp_err_gain * min(|x_err|, err_cap) + kp_vel_gain * min(|x_vel|, vel_cap)
  kd_x = kd_base + kd_err_gain * min(|x_err|, err_cap) + kd_vel_gain * min(|x_vel|, vel_cap)
Deliberately simple (linear-in-clipped-magnitude, not a bigger neural/
spline schedule) -- few enough parameters for differential_evolution to
search efficiently, matching this repo's own stated preference for
bandit/black-box search over anything RL-shaped for this class of
problem.
"""

from __future__ import annotations

import sys
from pathlib import Path

import mujoco
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from controller_core.x_axis_cartesian_impedance import (  # noqa: E402
    CartesianImpedanceConfig,
    XAxisCartesianImpedanceController,
)
from simulation.ur5e_mujoco_torque import (  # noqa: E402
    MujocoUR5eTorqueAdapter,
    MujocoUR5eTorqueAdapterConfig,
    build_mujoco_state,
    x_profile_target,
)

SCENE_XML = REPO_ROOT / "assets" / "ur5e_torque" / "scene.xml"
JOINT_NAMES = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
               "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]
SITE_NAME = "attachment_site"
ARM_Q0 = np.array([0, -np.pi / 2, np.pi / 2, -np.pi / 2, -np.pi / 2, 0])
RATE_HZ = 500.0
DT = 1.0 / RATE_HZ

MAX_ORTHOGONAL_DRIFT_M = 0.03  # matches ImpedanceSafetyMonitor's own default


def make_adapter(controller_cfg: CartesianImpedanceConfig):
    model = mujoco.MjModel.from_xml_path(str(SCENE_XML))
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, SITE_NAME)
    joint_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in JOINT_NAMES]
    controller = XAxisCartesianImpedanceController(controller_cfg)
    adapter_cfg = MujocoUR5eTorqueAdapterConfig(
        controller_kind="impedance", gravity_mode="gravity_comp",
    )
    adapter = MujocoUR5eTorqueAdapter(model=model, site_id=site_id, joint_ids=joint_ids,
                                        controller=controller, config=adapter_cfg)
    return model, site_id, joint_ids, adapter


def run_transport_trial(controller_cfg_kwargs: dict, target_x_delta_m: float, move_duration_s: float,
                          duration_s: float, schedule=None) -> dict:
    """schedule: optional dict with keys kp_base/kp_err_gain/kp_vel_gain/
    kd_base/kd_err_gain/kd_vel_gain/err_cap/vel_cap. If None, kp_x/kd_x
    from controller_cfg_kwargs are used FIXED (the baseline)."""
    cfg = CartesianImpedanceConfig(**controller_cfg_kwargs)
    model, site_id, joint_ids, adapter = make_adapter(cfg)
    data = mujoco.MjData(model)
    data.qpos[:6] = ARM_Q0
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)

    gravity_scratch = mujoco.MjData(model)
    state0 = build_mujoco_state(model, data, site_id=site_id, joint_ids=joint_ids, time_s=0.0,
                                  dt_s=DT, target_x=0.0, gravity_scratch_data=gravity_scratch)
    x0 = float(state0.ee_pos[0])
    p0 = state0.ee_pos.copy()
    adapter.reset(state0)

    n_steps = int(duration_s * RATE_HZ)
    guard_reason = None
    fell_at = None
    final_axis_error = None

    for step in range(n_steps):
        t = step * DT
        target_x, target_x_vel = x_profile_target("min_jerk_move_hold", x0, target_x_delta_m, t,
                                                     duration_s, move_duration_s=move_duration_s)
        state = build_mujoco_state(model, data, site_id=site_id, joint_ids=joint_ids, time_s=t, dt_s=DT,
                                     target_x=target_x, target_x_vel=target_x_vel,
                                     gravity_scratch_data=gravity_scratch)
        if schedule is not None:
            x_err = abs(target_x - float(state.ee_pos[0]))
            x_vel = abs(float(state.ee_lin_vel[0]))
            x_err_c = min(x_err, schedule["err_cap"])
            x_vel_c = min(x_vel, schedule["vel_cap"])
            adapter.controller.cfg.kp_x = max(0.1, schedule["kp_base"] + schedule["kp_err_gain"] * x_err_c
                                               + schedule["kp_vel_gain"] * x_vel_c)
            adapter.controller.cfg.kd_x = max(0.01, schedule["kd_base"] + schedule["kd_err_gain"] * x_err_c
                                               + schedule["kd_vel_gain"] * x_vel_c)
        tau, diag = adapter.step(state=state)
        final_axis_error = diag["axis_error"]
        if not diag["safety_ok"] and guard_reason is None:
            guard_reason = diag["safety_reason"]
            fell_at = t
            break
        data.ctrl[:6] = tau
        mujoco.mj_step(model, data)

    return {"target_x_delta_m": target_x_delta_m, "move_duration_s": move_duration_s,
            "guard_reason": guard_reason, "fell_at_s": fell_at,
            "final_axis_error": final_axis_error, "survived": guard_reason is None}


TEST_SCENARIOS = [(0.5, 1.0), (0.6, 1.0), (0.7, 1.0), (0.8, 1.0), (0.6, 0.5), (0.7, 0.5)]

SCHED_PARAM_BOUNDS = [
    ("kp_base", 1.0, 100.0, True),
    ("kp_err_gain", 0.0, 500.0, False),
    ("kp_vel_gain", 0.0, 200.0, False),
    ("kd_base", 0.1, 50.0, True),
    ("kd_err_gain", 0.0, 200.0, False),
    ("kd_vel_gain", 0.0, 100.0, False),
    ("err_cap", 0.05, 1.0, False),
    ("vel_cap", 0.05, 2.0, False),
]


def decode_schedule(x: np.ndarray) -> dict:
    out = {}
    for xi, (name, lo, hi, log) in zip(x, SCHED_PARAM_BOUNDS):
        out[name] = float(np.exp(np.log(lo) + xi * (np.log(hi) - np.log(lo)))) if log else float(lo + xi * (hi - lo))
    return out


def schedule_fitness(x: np.ndarray) -> float:
    schedule = decode_schedule(x)
    total = 0.0
    for dx, move_dur in TEST_SCENARIOS:
        r = run_transport_trial({}, dx, move_dur, move_dur + 1.0, schedule=schedule)
        if r["survived"]:
            total += abs(r["final_axis_error"])
        else:
            total += 2.0 + max(0.0, (move_dur + 1.0) - r["fell_at_s"]) / (move_dur + 1.0)
    return total


def run_search():
    from scipy.optimize import differential_evolution
    print("=== Baseline (fixed kp_x=25, kd_x=8) at search test scenarios ===")
    for dx, move_dur in TEST_SCENARIOS:
        r = run_transport_trial({"kp_x": 25.0, "kd_x": 8.0}, dx, move_dur, move_dur + 1.0)
        s = f"S(err={r['final_axis_error']:.4f})" if r["survived"] else f"F({r['guard_reason']})@{r['fell_at_s']:.2f}"
        print(f"  dx={dx:.2f} move_dur={move_dur}: {s}")

    print("\n=== Searching gain schedule ===")
    result = differential_evolution(
        schedule_fitness, bounds=[(0.0, 1.0)] * len(SCHED_PARAM_BOUNDS),
        maxiter=30, popsize=16, tol=1e-4, seed=0, workers=12, polish=False, disp=True,
    )
    best = decode_schedule(result.x)
    print(f"\nBest schedule: {best}")
    print(f"Best fitness: {result.fun}")

    print(f"\n{'dx':>6} {'move_dur':>10} {'SCHEDULED':>20} {'FIXED-BASELINE':>20}")
    for dx in [0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        move_dur = 1.0
        r_s = run_transport_trial({}, dx, move_dur, move_dur + 1.0, schedule=best)
        r_b = run_transport_trial({"kp_x": 25.0, "kd_x": 8.0}, dx, move_dur, move_dur + 1.0)
        s = f"S(err={r_s['final_axis_error']:.4f})" if r_s["survived"] else f"F@{r_s['fell_at_s']:.2f}"
        b = f"S(err={r_b['final_axis_error']:.4f})" if r_b["survived"] else f"F@{r_b['fell_at_s']:.2f}"
        print(f"{dx:6.2f} {move_dur:10.2f} {s:>20} {b:>20}")


def main() -> int:
    run_search()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
