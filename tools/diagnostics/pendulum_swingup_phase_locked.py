#!/usr/bin/env python3
"""Phase-locked continuous-resonance swing-up -- a 4th, distinct control
structure after the fixed-frequency sinusoid (pendulum_swingup_x_oscillation.py,
failed -- a pendulum's period lengthens with amplitude, so no FIXED
frequency stays resonant through a full swing-up), continuous energy-
shaping (pendulum_swingup_energy_shaping.py, best ~40deg/2.40-2.47rad via
discrete seed-kick + continuous law), and event-triggered discrete kicks
(pendulum_swingup_multi_kick.py, same ~2.39-2.47rad ceiling, root-caused to
accumulated orientation-error guard trips during sustained oscillation --
NOT insufficient energy/torque).

Built per Step 4 of the approved friction/resonance plan
(docs-external: /common/home/ss5772/.claude/plans/elegant-watching-bentley.md)
after Step 2 found neither LuGre nor Karnopp friction models move that
ceiling (best case +0.078rad, below the plan's 0.1rad gate; 4/5 non-
baseline runs still fail via the identical orientation guard) -- real
evidence the bottleneck is orientation-holding during oscillation, not
friction. This script targets that mechanism directly: multi_kick's
discrete pulses leave orientation-recovery gaps between kicks (arm holds,
guard-relevant error can only decay or drift, not correct mid-swing);
a continuous phase-locked drive has no such gap.

Design: drive continuously, phase-locked to a live period estimate:

    tau = t - t_last_crossing
    E = 0.5*I*thetadot^2 + m*g*r*(1 - cos(phi))          # live pendulum energy
    A = min(A_max, k_a * (E_top - E))                     # shrinks as it nears inversion
    phase = 2*pi/T_est * tau + phase_offset_bias
    target_x       = x0 + dir * A * sin(phase)
    target_x_vel   =       dir * A * omega_est * cos(phase)
    target_x_accel = -     dir * A * omega_est**2 * sin(phase)

`dir` = sign(thetadot) at the most recent zero crossing (push with the
current swing direction, the same "push with the swing" intuition as
multi-kick's kick_sign, just applied continuously instead of as discrete
pulses); `dir` and `T_est` are re-latched at EVERY zero crossing (both
directions), not held fixed for the trial -- the specific thing a fixed-
frequency drive (already tried, failed) could not do.

An earlier version of this script tried to MEASURE T_est passively, by
seed-kicking once and waiting for the free pendulum to complete a couple
of its own zero crossings before ever driving it. Direct instrumentation
(a standalone decay probe, not shown here) found this pendulum has real
Coulomb stiction (frictionloss=0.01, assets/ur5e_pendulum/
pendulum_attachment.xml): even a strong 1 rad/s isolated kick decays to
EXACTLY thetadot=0.000 and physically locks within ~1.6s, not a smooth
multi-cycle decay -- so passive coasting never produced enough crossings
to lock onto. Fixed by never coasting at all: T_est starts at the
analytically-derived small-oscillation natural period
(T_NATURAL_S = 2*pi*sqrt(I_PIVOT_KGM2/(M_TOTAL_KG*G*R_COM_M)), ~0.897s)
and the amplitude law A = k_a*(E_top-E) is nonzero from E=0 at rest (no
thetadot factor, unlike the plain energy-shaping law's a_energy ~ thetadot,
which WAS zero at rest -- that script's own documented cold-start problem)
-- so driving begins at step 0 with no seed kick needed. T_est/dir refine
from real crossings as soon as any occur, which now happens reliably
because active driving is exactly what overcomes the stiction that killed
passive coasting.

`current_hanging_angle` (the phi=0 reference) is refreshed at every
crossing via find_nearby_equilibrium (imported from
pendulum_swingup_multi_kick.py) -- reuses that script's fix for sustained
orientation drift shifting the pendulum's true rest point.

4-parameter search (k_a, A_max, phase_offset_bias, crossing_debounce_s)
via scipy.optimize.differential_evolution, NOT RL (AGENTS.md).

CLI matches pendulum_swingup_{energy_shaping,multi_kick}.py's pattern from
day one (--config/--friction-model/--start-q-rad/--output-json etc.).
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
from tools.diagnostics.pendulum_swingup_multi_kick import find_nearby_equilibrium  # noqa: E402

# Small-oscillation natural period (simple-pendulum approximation about the
# hanging equilibrium) -- the initial T_est before any real crossing has
# been observed, and the fallback the live estimate reverts toward if a
# crossing is ever debounced away entirely. ~0.897s for this pendulum.
T_NATURAL_S = 2.0 * np.pi * np.sqrt(I_PIVOT_KGM2 / (M_TOTAL_KG * G * R_COM_M))
MIN_CROSSINGS_FOR_LIVE_T_EST = 2  # two most recent crossings (any direction) = one half-period, doubled


def run_phase_locked_trial(
    model,
    k_a: float,
    a_max: float,
    phase_offset_bias: float,
    crossing_debounce_s: float,
    duration_s: float,
    hanging_angle: float,
    inverted_angle: float,
    config: dict,
    arm_q: np.ndarray = DEFAULT_ARM_Q0,
    enforce_guard: bool = True,
) -> dict:
    """enforce_guard=False: sim-only ceiling-finding mode (never on real
    hardware) -- record the first guard trip but keep simulating."""
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
    min_theta_dist = np.pi
    guard_fired = False
    guard_reason = None
    first_guard_t = None
    first_guard_reason = None
    steps_done = 0

    current_hanging_angle = hanging_angle
    crossing_ts = []     # timestamps of zero crossings
    prev_phi = None

    # Drive from step 0 -- no passive coast phase (see module docstring for
    # why: this pendulum's real stiction kills free-swing coasting within
    # about 1.6s, never producing enough crossings to measure a period
    # from). T_est starts at the analytical estimate and self-corrects.
    T_est = T_NATURAL_S
    t_last_crossing = 0.0
    dir_current = 1.0

    for step in range(n_steps):
        t = step * CONTROL_DT
        theta = float(data.qpos[pend_qpos_adr])
        thetadot = float(data.qvel[pend_dof_adr])
        phi = float(np.mod(theta - current_hanging_angle + np.pi, 2 * np.pi) - np.pi)

        if prev_phi is not None:
            crossed = (prev_phi == 0.0) or ((prev_phi > 0.0) != (phi > 0.0))
            debounced = crossing_ts and (t - crossing_ts[-1]) < crossing_debounce_s
            if crossed and not debounced:
                crossing_ts.append(t)
                t_last_crossing = t
                dir_current = 1.0 if thetadot >= 0.0 else -1.0
                if len(crossing_ts) >= MIN_CROSSINGS_FOR_LIVE_T_EST:
                    live_T = 2.0 * (crossing_ts[-1] - crossing_ts[-2])
                    if live_T > 1e-6:
                        T_est = live_T
                current_hanging_angle = find_nearby_equilibrium(
                    model, data.qpos[:6].copy(), pend_qpos_adr, theta
                )
        prev_phi = phi

        E = 0.5 * I_PIVOT_KGM2 * thetadot * thetadot + M_TOTAL_KG * G * R_COM_M * (1.0 - np.cos(phi))
        A = float(np.clip(k_a * (E_TOP - E), 0.0, a_max))
        omega_est = 2.0 * np.pi / T_est
        phase = omega_est * (t - t_last_crossing) + phase_offset_bias
        target_x = x0 + dir_current * A * np.sin(phase)
        target_x_vel = dir_current * A * omega_est * np.cos(phase)
        a_cmd = float(-dir_current * A * omega_est * omega_est * np.sin(phase))

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

    return {
        "k_a": k_a, "a_max": a_max, "phase_offset_bias": phase_offset_bias,
        "crossing_debounce_s": crossing_debounce_s, "duration_s": duration_s,
        "min_theta_dist_from_inverted_rad": min_theta_dist,
        "flipped": min_theta_dist < 0.35,
        "guard_fired": guard_fired, "guard_reason": guard_reason,
        "first_guard_t": first_guard_t, "first_guard_reason": first_guard_reason,
        "steps_completed": steps_done, "num_crossings": len(crossing_ts),
        "final_T_est_s": T_est,
    }


def objective(x, config: dict, arm_q: np.ndarray):
    k_a, a_max, phase_offset_bias, crossing_debounce_s = x
    model = compose_ur5e_pendulum_model()
    data = mujoco.MjData(model)
    pend_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "/pendulum_hinge")
    pend_qpos_adr = model.jnt_qposadr[pend_jid]
    hanging_angle, inverted_angle = find_hanging_and_inverted_angle(model, data, pend_qpos_adr, arm_q=arm_q)
    result = run_phase_locked_trial(model, k_a, a_max, phase_offset_bias, crossing_debounce_s,
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
    print(f"hanging_angle={hanging_angle:.4f} rad  inverted_angle={inverted_angle:.4f} rad  "
          f"T_natural={T_NATURAL_S:.4f}s")
    print(f"friction_model={config['controller'].get('friction_model', 'static')} "
          f"friction_feedforward={config['controller'].get('friction_feedforward', False)}")

    bounds = [
        (0.0, 5.0),                     # k_a
        (0.02, 0.28),                   # a_max (m) -- same range as multi_kick's kick_amplitude_m
        (-np.pi / 2, np.pi / 2),        # phase_offset_bias (rad)
        (0.1, 0.5),                     # crossing_debounce_s
    ]
    print("=== searching (k_a, a_max, phase_offset_bias, crossing_debounce_s) via differential_evolution ===")
    res = differential_evolution(objective, bounds, args=(config, arm_q), maxiter=args.maxiter,
                                  popsize=args.popsize, tol=1e-4, seed=args.seed, workers=-1, polish=False)
    print(f"Best params: k_a={res.x[0]:.4f}, a_max={res.x[1]:.4f}, "
          f"phase_offset_bias={res.x[2]:.4f} ({np.degrees(res.x[2]):.2f}deg), "
          f"crossing_debounce_s={res.x[3]:.4f}")
    print(f"Best cost: {res.fun:.4f}")

    best = run_phase_locked_trial(model, res.x[0], res.x[1], res.x[2], res.x[3], duration_s=args.duration_s,
                                   hanging_angle=hanging_angle, inverted_angle=inverted_angle,
                                   config=config, arm_q=arm_q)
    print("Best candidate, re-validated:", best)

    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with args.output_json.open("w") as fp:
            json.dump({
                "friction_model": config["controller"].get("friction_model", "static"),
                "best_params": {"k_a": res.x[0], "a_max": res.x[1],
                                 "phase_offset_bias": res.x[2], "crossing_debounce_s": res.x[3]},
                "best_cost": float(res.fun), "validation": best,
            }, fp, indent=2, default=str)
        print(f"Wrote result to {args.output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
