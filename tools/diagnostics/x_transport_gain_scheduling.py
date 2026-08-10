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
    from controller_cfg_kwargs are used FIXED (the baseline).

    x_integral_action/ki_x/x_integral_limit_m_s (added for the overnight
    ki_x search, see module docstring update below) are NOT part of the
    live per-cycle schedule -- they are FIXED controller config, set once
    via controller_cfg_kwargs, exactly like kp_x/kd_x are in the
    no-schedule baseline case. This matches the real mechanism: ki_x is a
    gain on an internally-accumulated state (x_integral), not something
    that makes sense to re-derive from instantaneous (x_err, x_vel) the
    way the live kp_x/kd_x schedule does."""
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


# RESCOPED (2026-08-10, second same-night correction): dx=0.4 and beyond were
# in the original TEST_SCENARIOS, but direct verification found they fail via
# ||orientation error|| > 0.35 rad -- the SAME guard, at nearly the SAME dx
# threshold, as the plain fixed baseline (kp_x=25/kd_x=8, no ki_x at all).
# This is the plain "impedance" controller_kind's known orientation-holding
# weakness at this pose (the wrist-singularity/nullspace-posture issue the
# OSC lane's wrist_orientation_task + jacobian_singular_cond_max fixes
# address, see AGENTS.md sec 3) -- NOT a friction/stiction problem, and NOT
# something a kp_x/kd_x/ki_x schedule has any mechanism to fix. Including
# dx=0.4-0.8 in the objective meant every candidate failed 4 of 6 scored
# scenarios no matter how well-tuned, which is why the first corrected run
# (see docs/status/x_transport_ki_search_2026-08-10.md) converged on "fail
# everywhere, as late as possible" rather than a usable schedule -- DE was
# solving an unwinnable aggregate penalty instead of the real, achievable
# problem. Rescoped to dx in [0.1, 0.4], i.e. from the previously-hardest
# tracked case through the actual guard boundary
# (0.3 survives at the fixed baseline, 0.4 fails) -- what a kp/kd/ki_x
# schedule can plausibly influence. Extending the SURVIVABLE range past 0.3
# would need the OSC-family orientation fixes, a separate, already-solved
# problem in a different controller_kind, deliberately out of scope here.
TEST_SCENARIOS = [(0.1, 1.0), (0.15, 1.0), (0.2, 1.0), (0.25, 1.0), (0.3, 1.0), (0.4, 1.0)]
HOLD_TAIL_S = 6.0

SCHED_PARAM_BOUNDS = [
    ("kp_base", 1.0, 100.0, True),
    ("kp_err_gain", 0.0, 500.0, False),
    ("kp_vel_gain", 0.0, 200.0, False),
    ("kd_base", 0.1, 50.0, True),
    ("kd_err_gain", 0.0, 200.0, False),
    ("kd_vel_gain", 0.0, 100.0, False),
    ("err_cap", 0.05, 1.0, False),
    ("vel_cap", 0.05, 2.0, False),
    # NEW: x_integral_action gains, searched jointly with the kp/kd
    # schedule above (not fixed by hand) -- see module docstring update.
    # ki_x upper bound (1000) chosen from the same-day finding that
    # ki_x=150-300 was the useful range at x_integral_limit_m_s=1.0;
    # giving the search headroom past that rather than clamping to
    # exactly what worked by hand.
    ("ki_x", 0.0, 1000.0, False),
    ("x_integral_limit_m_s", 0.02, 5.0, True),
]


def decode_schedule(x: np.ndarray) -> dict:
    out = {}
    for xi, (name, lo, hi, log) in zip(x, SCHED_PARAM_BOUNDS):
        out[name] = float(np.exp(np.log(lo) + xi * (np.log(hi) - np.log(lo)))) if log else float(lo + xi * (hi - lo))
    return out


def schedule_fitness(x: np.ndarray) -> float:
    """FIXED (2026-08-10, same-night correction): the original version scored a
    "survived" trial by raw abs(final_axis_error) (max ~0.8, bounded by the
    largest tested dx) versus >=2.0 for any guard trip. Verified directly (single
    trajectory trace, see docs/status/x_transport_ki_search_2026-08-10.md) that
    the FIRST overnight run's "winning" schedule collapsed kp_base/kd_base to
    near their floor (kp_x~1.1-1.4 vs the 25.0 baseline, let alone the tuned
    OSC's 400) -- low enough to never approach a guard threshold, but so weak
    against real joint friction that a dx=0.10m target trial was still 90% short
    of its target after 6s of hold (moved ~0.01m of 0.10m). That is a real
    stuck-not-moved result, not a working controller -- it "survived" only
    because the fitness function rewarded avoiding risk over actually
    transporting. Fixed by normalizing tracking error to a FRACTION of the
    commanded displacement (frac_err in ~[0,1] regardless of dx) and weighting it
    3x, putting a fully-stuck trial's cost (~2.7) in the same range as a guard
    trip (~2.0-3.0) instead of an order of magnitude cheaper."""
    params = decode_schedule(x)
    ki_x = params["ki_x"]
    x_integral_limit_m_s = params["x_integral_limit_m_s"]
    schedule = {k: v for k, v in params.items() if k not in ("ki_x", "x_integral_limit_m_s")}
    cfg_kwargs = {"x_integral_action": ki_x > 0.0, "ki_x": ki_x, "x_integral_limit_m_s": x_integral_limit_m_s}
    total = 0.0
    for dx, move_dur in TEST_SCENARIOS:
        duration_s = move_dur + HOLD_TAIL_S
        r = run_transport_trial(cfg_kwargs, dx, move_dur, duration_s, schedule=schedule)
        if r["survived"]:
            frac_err = abs(r["final_axis_error"]) / max(abs(dx), 1e-6)
            total += 3.0 * frac_err
        else:
            total += 2.0 + max(0.0, duration_s - r["fell_at_s"]) / duration_s
    return total


def run_search():
    from scipy.optimize import differential_evolution
    print("=== Baseline (fixed kp_x=25, kd_x=8, no integral action) at search test scenarios ===")
    for dx, move_dur in TEST_SCENARIOS:
        duration_s = move_dur + HOLD_TAIL_S
        r = run_transport_trial({"kp_x": 25.0, "kd_x": 8.0}, dx, move_dur, duration_s)
        s = f"S(err={r['final_axis_error']:.4f})" if r["survived"] else f"F({r['guard_reason']})@{r['fell_at_s']:.2f}"
        print(f"  dx={dx:.2f} move_dur={move_dur}: {s}")

    print("\n=== Searching gain schedule + x_integral_action jointly (overnight budget) ===")
    result = differential_evolution(
        schedule_fitness, bounds=[(0.0, 1.0)] * len(SCHED_PARAM_BOUNDS),
        maxiter=80, popsize=24, tol=1e-5, seed=0, workers=16, polish=False, disp=True,
    )
    best = decode_schedule(result.x)
    print(f"\nBest params: {best}")
    print(f"Best fitness: {result.fun}")

    ki_x = best["ki_x"]
    x_integral_limit_m_s = best["x_integral_limit_m_s"]
    schedule = {k: v for k, v in best.items() if k not in ("ki_x", "x_integral_limit_m_s")}
    cfg_kwargs = {"x_integral_action": ki_x > 0.0, "ki_x": ki_x, "x_integral_limit_m_s": x_integral_limit_m_s}

    print(f"\n{'dx':>6} {'move_dur':>10} {'SCHEDULED+ki_x':>20} {'FIXED-BASELINE':>20}")
    for dx in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        move_dur = 1.0
        duration_s = move_dur + HOLD_TAIL_S
        r_s = run_transport_trial(cfg_kwargs, dx, move_dur, duration_s, schedule=schedule)
        r_b = run_transport_trial({"kp_x": 25.0, "kd_x": 8.0}, dx, move_dur, duration_s)
        s = f"S(err={r_s['final_axis_error']:.4f})" if r_s["survived"] else f"F@{r_s['fell_at_s']:.2f}"
        b = f"S(err={r_b['final_axis_error']:.4f})" if r_b["survived"] else f"F@{r_b['fell_at_s']:.2f}"
        print(f"{dx:6.2f} {move_dur:10.2f} {s:>20} {b:>20}")


def main() -> int:
    run_search()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
