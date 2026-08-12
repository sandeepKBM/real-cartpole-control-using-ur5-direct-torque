#!/usr/bin/env python3
"""Multi-kick (event-triggered pumping) swing-up for the pendulum, replacing
the single-kick + continuous-feedback design in
pendulum_swingup_energy_shaping.py, which was found to produce one real
swing (17deg, 3.6% of the energy needed to invert) but then flatline --
the continuous law wasn't sustaining repeated pumping cycles.

Design: fire a new discrete kick (the same validated raised-cosine, zero-
endpoint-velocity pulse from the single-kick version) every time the
pendulum crosses near the BOTTOM of its swing (|phi| < phi_trigger_rad)
while already moving (avoids re-triggering at true rest), in the SAME
direction as its current motion (push with the swing, adding energy --
the same intuition as the classical energy-shaping law's phase-lock, just
applied as discrete pulses instead of continuous force). Between kicks the
arm holds its position (with a weak pull back toward x0 to bound drift
across many cycles) and the pendulum swings freely under gravity.

Deliberately event-triggered (not fixed-period): a FIXED-frequency drive
was already tried and failed for exactly the reason a pendulum's period
lengthens with amplitude (pendulum_swingup_x_oscillation.py) -- triggering
on the pendulum's own state instead of a clock avoids repeating that
mistake.

3-parameter search (kick_amplitude_m, kick_duration_s, phi_trigger_rad) via
scipy.optimize.differential_evolution, NOT RL (AGENTS.md: 6 documented RL
gain-scheduling failures in this repo).

CLI-configurable (--config/--friction-model/--start-q-rad/--output-json)
since 2026-08-11 -- previously required editing this file's module-level
CONFIG_PATH/ARM_Q0 constants for every new pose/friction-model experiment.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import differential_evolution

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from simulation.ur5e_pendulum_compose import compose_ur5e_pendulum_model  # noqa: E402
from simulation.ur5e_mujoco_torque import (  # noqa: E402
    build_initial_state_and_adapter,
    build_mujoco_state,
)
from tools.diagnostics.pendulum_swingup_energy_shaping import (  # noqa: E402
    load_config, find_hanging_and_inverted_angle, CONFIG_PATH, DEFAULT_ARM_Q0,
    CONTROL_DT, RATE_HZ, M_TOTAL_KG, R_COM_M, I_PIVOT_KGM2, G, E_TOP,
)

MIN_KICK_GAP_S = 0.15   # debounce: minimum time between the end of one kick and the next trigger
THETADOT_DEADBAND = 0.02  # rad/s -- don't trigger at near-exact rest
K_RECENTER = 3.0        # weak pull toward x0 during hold phases, bounds drift across many kicks


def find_nearby_equilibrium(model, arm_qpos6, pend_qpos_adr, near_theta,
                             search_half_range=np.pi / 2, n_points=37):
    """Re-derive the pendulum's CURRENT equilibrium angle, using the arm's
    LIVE (possibly orientation-drifted) posture, via the same qfrc_bias
    zero-crossing technique as pendulum_balance_torque_lqr.py's
    find_inverted_angle -- NOT a "release and watch it settle" simulation
    (that approach was documented there as unreliable for this system's
    slow dynamics). Found necessary after tracing a real trial: sustained
    orientation drift during a long multi-kick run shifts where the
    pendulum's own passive equilibrium actually is (measured: from 0deg to
    ~38deg in one trial), and the ORIGINAL trigger (anchored to the
    pose's starting hanging_angle) simply stopped firing once the system
    settled at the new location -- not because kicks stopped helping, but
    because the trigger's reference point had gone stale.

    Narrower/coarser than the original (a +-90deg window around the
    pendulum's own current angle, not the full circle) since this runs
    many times per trial (once per kick) and only needs the equilibrium
    NEAREST where the pendulum already is, not a global search."""
    scratch = mujoco.MjData(model)

    def qfrc_bias_pend(theta: float) -> float:
        scratch.qpos[:6] = arm_qpos6
        scratch.qpos[pend_qpos_adr] = theta
        scratch.qvel[:] = 0.0
        mujoco.mj_forward(model, scratch)
        return float(scratch.qfrc_bias[pend_qpos_adr])

    thetas = np.linspace(near_theta - search_half_range, near_theta + search_half_range, n_points)
    vals = np.array([qfrc_bias_pend(t) for t in thetas])
    best, best_dist = None, np.inf
    for i in range(len(thetas) - 1):
        if vals[i] == 0.0 or (vals[i] > 0) != (vals[i + 1] > 0):
            lo, hi = thetas[i], thetas[i + 1]
            for _ in range(30):
                mid = (lo + hi) / 2
                if (qfrc_bias_pend(mid) > 0) == (vals[i] > 0):
                    lo = mid
                else:
                    hi = mid
            crossing = (lo + hi) / 2
            d = abs(crossing - near_theta)
            if d < best_dist:
                best_dist, best = d, crossing
    return best if best is not None else near_theta  # no crossing found in window -- keep prior estimate


def run_multi_kick_trial(
    model,
    kick_amplitude_m: float,
    kick_duration_s: float,
    phi_trigger_rad: float,
    duration_s: float,
    hanging_angle: float,
    inverted_angle: float,
    config: dict,
    arm_q: np.ndarray = DEFAULT_ARM_Q0,
    enforce_guard: bool = True,
    track_history: bool = False,
) -> dict:
    """enforce_guard=False (added for sim-only investigation, per-user
    request): don't break the trial on a guard trip -- record the FIRST
    occurrence and reason, but keep simulating and applying torque exactly
    as normal for the rest of the duration. Never do this on real hardware;
    this is only meaningful/safe as a simulation ceiling-finding tool.
    track_history=True additionally records (t, phi, thetadot, E/E_top,
    orientation_error_norm, safety_ok) every step, to check directly
    whether orientation-guard violations correlate with the pendulum's own
    energy stalling/dropping, or are basically decoupled from it.

    config: an already-loaded (and, if desired, already friction-model-
    overridden) controller config dict -- callers own loading it (via
    load_config()) rather than this function reloading from disk every
    call, so a CLI-selected friction_model/config_path actually takes
    effect instead of silently reading the hardcoded default."""
    data = mujoco.MjData(model)
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "attachment_site")
    joint_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in [
        "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
        "wrist_1_joint", "wrist_2_joint", "wrist_3_joint",
    ]]
    pend_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "/pendulum_hinge")
    pend_qpos_adr = model.jnt_qposadr[pend_jid]
    pend_dof_adr = model.jnt_dofadr[pend_jid]

    data.qpos[:6] = arm_q
    data.qpos[pend_qpos_adr] = hanging_angle
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
        torque_limit_scale=1.0,
    )
    x0 = float(state0.ee_pos[0])

    n_steps = int(duration_s * RATE_HZ)
    target_x = x0
    target_x_vel = 0.0
    min_theta_dist = np.pi
    guard_fired = False
    guard_reason = None
    steps_done = 0
    num_kicks = 0

    kick_active = False
    kick_start_t = 0.0
    kick_sign = 1.0
    kick_hold_x = x0  # where target_x sits during hold phases (updated after each kick)
    last_kick_end_t = -1e9
    first_guard_t = None
    first_guard_reason = None
    history = [] if track_history else None
    current_hanging_angle = hanging_angle  # live estimate, refreshed after each kick

    for step in range(n_steps):
        t = step * CONTROL_DT
        theta = float(data.qpos[pend_qpos_adr])
        thetadot = float(data.qvel[pend_dof_adr])
        phi = float(np.mod(theta - current_hanging_angle + np.pi, 2 * np.pi) - np.pi)

        if kick_active and (t - kick_start_t) >= kick_duration_s:
            kick_active = False
            kick_hold_x = target_x
            last_kick_end_t = t
            # Re-derive the local equilibrium using the arm's CURRENT
            # (possibly drifted) posture -- the fix for the trigger going
            # stale once the arm's own orientation has moved enough to
            # shift where the pendulum actually wants to rest.
            current_hanging_angle = find_nearby_equilibrium(
                model, data.qpos[:6].copy(), pend_qpos_adr, theta
            )

        is_bootstrap_kick = (num_kicks == 0 and step == 0)
        if is_bootstrap_kick or (
            not kick_active and abs(phi) < phi_trigger_rad
            and abs(thetadot) > THETADOT_DEADBAND
            and (t - last_kick_end_t) >= MIN_KICK_GAP_S
        ):
            kick_active = True
            kick_start_t = t
            # Bootstrap kick has no real thetadot to key off of yet -- pick
            # a fixed direction (+1); every later, real event-triggered kick
            # follows the pendulum's own current motion instead.
            kick_sign = 1.0 if is_bootstrap_kick else (1.0 if thetadot >= 0.0 else -1.0)
            num_kicks += 1

        if kick_active:
            tau_local = t - kick_start_t
            omega_kick = 2.0 * np.pi / kick_duration_s
            target_x = kick_hold_x + kick_sign * 0.5 * kick_amplitude_m * (1.0 - np.cos(omega_kick * tau_local))
            target_x_vel = kick_sign * 0.5 * kick_amplitude_m * omega_kick * np.sin(omega_kick * tau_local)
            a_cmd = float(kick_sign * 0.5 * kick_amplitude_m * omega_kick * omega_kick * np.cos(omega_kick * tau_local))
        else:
            a_cmd = float(-K_RECENTER * (kick_hold_x - x0))
            target_x_vel = 0.0
            target_x = kick_hold_x

        state = build_mujoco_state(
            model, data, site_id=site_id, joint_ids=joint_ids,
            time_s=t, dt_s=CONTROL_DT,
            target_x=target_x, target_x_vel=target_x_vel, target_x_accel=a_cmd,
            reference_quat=state0.ee_quat, transport_axis_index=0,
            gravity_compensation=True,
        )
        tau, diag = adapter.step(state=state)
        step_safety_ok = bool(diag.get("safety_ok", True))
        if not step_safety_ok and first_guard_t is None:
            first_guard_t = t
            first_guard_reason = str(diag.get("safety_reason", ""))
        if not step_safety_ok:
            guard_fired = True
            guard_reason = first_guard_reason
            if enforce_guard:
                break

        data.ctrl[:6] = np.asarray(tau, dtype=np.float64).reshape(6)
        mujoco.mj_step(model, data)
        steps_done += 1

        dist = abs(float(np.mod(theta - inverted_angle + np.pi, 2 * np.pi) - np.pi))
        min_theta_dist = min(min_theta_dist, dist)

        if track_history:
            E = 0.5 * I_PIVOT_KGM2 * thetadot * thetadot + M_TOTAL_KG * G * R_COM_M * (1.0 - np.cos(phi))
            history.append({
                "t": t, "phi_deg": float(np.degrees(phi)), "thetadot": thetadot,
                "E_over_Etop": float(E / E_TOP),
                "orientation_error_norm": float(diag.get("orientation_error_norm", 0.0)),
                "safety_ok": step_safety_ok,
            })

    return {
        "kick_amplitude_m": kick_amplitude_m, "kick_duration_s": kick_duration_s,
        "phi_trigger_rad": phi_trigger_rad, "duration_s": duration_s,
        "min_theta_dist_from_inverted_rad": min_theta_dist,
        "flipped": min_theta_dist < 0.35,
        "guard_fired": guard_fired, "guard_reason": guard_reason,
        "first_guard_t": first_guard_t, "first_guard_reason": first_guard_reason,
        "steps_completed": steps_done, "num_kicks": num_kicks,
        "history": history,
    }


def objective(x, config: dict, arm_q: np.ndarray):
    kick_amplitude_m, kick_duration_s, phi_trigger_rad = x
    model = compose_ur5e_pendulum_model()
    data = mujoco.MjData(model)
    pend_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "/pendulum_hinge")
    pend_qpos_adr = model.jnt_qposadr[pend_jid]
    hanging_angle, inverted_angle = find_hanging_and_inverted_angle(model, data, pend_qpos_adr, arm_q=arm_q)
    result = run_multi_kick_trial(model, kick_amplitude_m, kick_duration_s, phi_trigger_rad,
                                   duration_s=8.0, hanging_angle=hanging_angle, inverted_angle=inverted_angle,
                                   config=config, arm_q=arm_q)
    cost = result["min_theta_dist_from_inverted_rad"]
    if result["guard_fired"]:
        cost += 5.0  # strictly worse than any survived outcome (max survived cost = pi < 5.0)
    return cost


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path, default=CONFIG_PATH, help="Controller YAML config path.")
    p.add_argument("--friction-model", choices=["static", "lugre", "karnopp"], default=None,
                   help="Override controller.friction_model (and force friction_feedforward=true). "
                        "Default: use whatever the config file says.")
    p.add_argument("--start-q-rad", type=float, nargs=6, default=None,
                   metavar=("Q1", "Q2", "Q3", "Q4", "Q5", "Q6"),
                   help="6-joint start pose in radians. Default: this session's mega-pose-search winner.")
    p.add_argument("--maxiter", type=int, default=30)
    p.add_argument("--popsize", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--duration-s", type=float, default=15.0, help="Final validation trial duration.")
    p.add_argument("--output-json", type=Path, default=None)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config, friction_model=args.friction_model)
    arm_q = np.array(args.start_q_rad) if args.start_q_rad is not None else DEFAULT_ARM_Q0

    model = compose_ur5e_pendulum_model()
    data = mujoco.MjData(model)
    pend_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "/pendulum_hinge")
    pend_qpos_adr = model.jnt_qposadr[pend_jid]
    hanging_angle, inverted_angle = find_hanging_and_inverted_angle(model, data, pend_qpos_adr, arm_q=arm_q)
    print(f"hanging_angle={hanging_angle:.4f} rad  inverted_angle={inverted_angle:.4f} rad")
    print(f"friction_model={config['controller'].get('friction_model', 'static')} "
          f"friction_feedforward={config['controller'].get('friction_feedforward', False)}")

    # kick_amplitude_m upper bound widened 0.15->0.28 (2026-08-11): the
    # prior search converged to 0.143, pinned right at the old 0.15 bound --
    # a real signal it wanted more amplitude, not that more doesn't help.
    # 0.28 is still inside this pose's own validated safe reach (see the
    # mega pose-oscillation-stability search this pose came from).
    bounds = [(0.02, 0.28), (0.1, 0.5), (np.radians(3.0), np.radians(40.0))]
    print("=== searching (kick_amplitude_m, kick_duration_s, phi_trigger_rad) via differential_evolution ===")
    res = differential_evolution(objective, bounds, args=(config, arm_q), maxiter=args.maxiter,
                                  popsize=args.popsize, tol=1e-4, seed=args.seed, workers=-1, polish=False)
    print(f"Best params: kick_amplitude_m={res.x[0]:.4f}, kick_duration_s={res.x[1]:.4f}, "
          f"phi_trigger_rad={res.x[2]:.4f} ({np.degrees(res.x[2]):.2f}deg)")
    print(f"Best cost: {res.fun:.4f}")

    best = run_multi_kick_trial(model, res.x[0], res.x[1], res.x[2], duration_s=args.duration_s,
                                 hanging_angle=hanging_angle, inverted_angle=inverted_angle,
                                 config=config, arm_q=arm_q)
    print("Best candidate, re-validated:", {k: v for k, v in best.items() if k != "history"})

    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with args.output_json.open("w") as fp:
            json.dump({
                "friction_model": config["controller"].get("friction_model", "static"),
                "best_params": {"kick_amplitude_m": res.x[0], "kick_duration_s": res.x[1],
                                 "phi_trigger_rad": res.x[2]},
                "best_cost": float(res.fun),
                "validation": {k: v for k, v in best.items() if k != "history"},
            }, fp, indent=2, default=str)
        print(f"Wrote result to {args.output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
