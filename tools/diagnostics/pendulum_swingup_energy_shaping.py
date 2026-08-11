#!/usr/bin/env python3
"""Energy-shaping (Astrom-Furuta style) swing-up for the pendulum at the new
"hanging" transport pose (no base rotation, wrist_2=11deg) -- replaces
pendulum_swingup_x_oscillation.py's fixed-frequency sinusoid, which searched
(amplitude, frequency) and found NOTHING better than ~5.5deg of arc: a fixed
frequency necessarily falls out of resonance as swing amplitude grows (a
pendulum's period is NOT amplitude-independent away from small angles), so
no (amplitude, frequency) pair can complete a full swing-up. This is a
different CONTROL STRUCTURE, not just different numbers: cart acceleration
is a live feedback function of the pendulum's instantaneous (angle,
angular velocity), not a pre-computed trajectory.

Law: a_cmd = k_e * thetadot * cos(phi) * (E_top - E) - k_pos * (x - x0),
clipped to +-a_max. phi = theta - hanging_angle (0 at hanging, pi at
inverted); E is the pendulum's own rotational+gravitational energy
referenced to hanging = 0 (NOT total system energy -- the moving pivot's
own translational KE is deliberately excluded, matching the classical
derivation, which assumes the cart's kinematics -- not energy -- couples
into the pendulum). The k_pos term (not in the textbook-minimal version) is
a practical recentering term: pure energy shaping has no reason to return
the cart to x0 and can walk arbitrarily far from it.

Reuses the same torque-lane Cartesian impedance controller/config as every
other transport check this session -- a_cmd is integrated into
target_x/target_x_vel/target_x_accel each step, fed through the same
controller interface (build_mujoco_state -> adapter.step), same as the
sinusoid version.

3-parameter search (k_e, a_max, k_pos) via scipy.optimize.differential_
evolution, NOT RL, per this repo's own documented history of RL
gain-scheduling failures (AGENTS.md).
"""

from __future__ import annotations

import sys
from pathlib import Path

import mujoco
import numpy as np
import yaml
from scipy.optimize import differential_evolution

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from simulation.ur5e_pendulum_compose import compose_ur5e_pendulum_model  # noqa: E402
from simulation.ur5e_mujoco_torque import (  # noqa: E402
    build_initial_state_and_adapter,
    build_mujoco_state,
)

CONFIG_PATH = REPO_ROOT / "config" / "ur5e_mujoco_torque_osc_hanging_pose_friction_ff.yaml"
# Updated 2026-08-11 to the mega pose-oscillation-stability search's winner
# (tools/diagnostics/pose_oscillation_stability_search.py): 6/6 alternating
# kicks survived cleanly at 0.777 m/s peak, cond(J)=6.93 -- meaningfully
# more stable under repeated direction reversals than the previous
# hand-picked pose, which is exactly the property swing-up needs. Hinge
# axis re-verified horizontal here too (attachment_site local X, |z|=0.009).
ARM_Q0 = np.array([0.0, -1.091985784398452, 2.0935362786892546,
                    -2.7685637962327356, 1.5620693866337145, 0.0])
RATE_HZ = 500.0
CONTROL_DT = 1.0 / RATE_HZ

# Analytically estimated pendulum physical constants (rod 0.1183kg/L=0.30m
# steel + hub 0.06kg near the pivot) -- see this session's earlier
# back-of-envelope estimate. The gain search absorbs any scale error in
# these (they only enter the control law multiplied by the free gain k_e),
# so exact precision is not required for the law to work.
M_TOTAL_KG = 0.1783
R_COM_M = 0.0995
I_PIVOT_KGM2 = 0.003549
G = 9.81
E_TOP = 2.0 * M_TOTAL_KG * G * R_COM_M  # energy at fully inverted, ref=hanging


def load_config() -> dict:
    with CONFIG_PATH.open() as fp:
        return yaml.safe_load(fp)


def find_hanging_and_inverted_angle(model, data, pend_qpos_adr: int) -> tuple[float, float]:
    data.qpos[:6] = ARM_Q0
    data.qpos[pend_qpos_adr] = 0.0
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)
    for _ in range(5000):
        mujoco.mj_step(model, data)
    hanging = float(np.mod(data.qpos[pend_qpos_adr] + np.pi, 2 * np.pi) - np.pi)
    inverted = float(np.mod(hanging + np.pi + np.pi, 2 * np.pi) - np.pi)
    return hanging, inverted


def run_energy_swingup_trial(
    model,
    k_e: float,
    a_max: float,
    k_pos: float,
    k_vel: float,
    duration_s: float,
    hanging_angle: float,
    inverted_angle: float,
    kick_amplitude_m: float = 0.0,
    kick_duration_s: float = 0.0,
) -> dict:
    """kick_amplitude_m/kick_duration_s (added after the plain energy law
    alone was found to stall): the energy term a_energy = k_e*thetadot*...
    is exactly zero when thetadot=0, so starting from PERFECT rest it can
    only ever grow from whatever numerical residual leaks in -- confirmed
    real coupling exists (a deliberate fast single move reached thetadot=
    0.62 rad/s in 0.08s) but the feedback law alone never escapes near-zero
    thetadot on its own. A brief open-loop half-sine "seed kick" for
    t < kick_duration_s gets thetadot away from zero before the energy law
    takes over -- the standard practical fix for this exact bootstrap
    problem in energy-shaping swing-up."""
    config = load_config()
    data = mujoco.MjData(model)
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "attachment_site")
    joint_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in [
        "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
        "wrist_1_joint", "wrist_2_joint", "wrist_3_joint",
    ]]
    pend_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "/pendulum_hinge")
    pend_qpos_adr = model.jnt_qposadr[pend_jid]
    pend_dof_adr = model.jnt_dofadr[pend_jid]

    data.qpos[:6] = ARM_Q0
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

    for step in range(n_steps):
        t = step * CONTROL_DT
        theta = float(data.qpos[pend_qpos_adr])
        thetadot = float(data.qvel[pend_dof_adr])
        phi = float(np.mod(theta - hanging_angle + np.pi, 2 * np.pi) - np.pi)
        E = 0.5 * I_PIVOT_KGM2 * thetadot * thetadot + M_TOTAL_KG * G * R_COM_M * (1.0 - np.cos(phi))

        if t < kick_duration_s and kick_duration_s > 1e-9:
            # Raised-cosine (Hann) position pulse: x0 -> x0+kick_amplitude -> x0.
            # A plain half-sine's VELOCITY is a cosine, which is nonzero (in
            # fact maximal) at both endpoints -- found empirically the hard
            # way (a large, undamped leftover velocity was dominating the
            # entire post-kick trial, exactly explaining the earlier
            # decay-after-kick result, not a genuine energy-pumping
            # failure). This profile's velocity is a sine, genuinely zero
            # at both t=0 and t=kick_duration_s, so the handoff to the
            # feedback law below sees no velocity discontinuity.
            omega_kick = 2.0 * np.pi / kick_duration_s
            target_x = x0 + 0.5 * kick_amplitude_m * (1.0 - np.cos(omega_kick * t))
            target_x_vel = 0.5 * kick_amplitude_m * omega_kick * np.sin(omega_kick * t)
            a_cmd = float(0.5 * kick_amplitude_m * omega_kick * omega_kick * np.cos(omega_kick * t))
        else:
            a_energy = k_e * thetadot * np.cos(phi) * (E_TOP - E)
            a_recenter = -k_pos * (target_x - x0) - k_vel * target_x_vel
            a_cmd = float(np.clip(a_energy + a_recenter, -a_max, a_max))

            target_x_vel += a_cmd * CONTROL_DT
            target_x += target_x_vel * CONTROL_DT

        state = build_mujoco_state(
            model, data, site_id=site_id, joint_ids=joint_ids,
            time_s=t, dt_s=CONTROL_DT,
            target_x=target_x, target_x_vel=target_x_vel, target_x_accel=a_cmd,
            reference_quat=state0.ee_quat, transport_axis_index=0,
            gravity_compensation=True,
        )
        tau, diag = adapter.step(state=state)
        if not bool(diag.get("safety_ok", True)):
            guard_fired = True
            guard_reason = str(diag.get("safety_reason", ""))
            break

        data.ctrl[:6] = np.asarray(tau, dtype=np.float64).reshape(6)
        mujoco.mj_step(model, data)
        steps_done += 1

        dist = abs(float(np.mod(theta - inverted_angle + np.pi, 2 * np.pi) - np.pi))
        min_theta_dist = min(min_theta_dist, dist)

    return {
        "k_e": k_e, "a_max": a_max, "k_pos": k_pos, "k_vel": k_vel,
        "kick_amplitude_m": kick_amplitude_m, "kick_duration_s": kick_duration_s,
        "duration_s": duration_s,
        "min_theta_dist_from_inverted_rad": min_theta_dist,
        "flipped": min_theta_dist < 0.35,
        "guard_fired": guard_fired, "guard_reason": guard_reason,
        "steps_completed": steps_done,
    }


def objective(x):
    k_e, a_max, k_pos, k_vel, kick_amplitude_m, kick_duration_s = x
    model = compose_ur5e_pendulum_model()
    data = mujoco.MjData(model)
    pend_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "/pendulum_hinge")
    pend_qpos_adr = model.jnt_qposadr[pend_jid]
    hanging_angle, inverted_angle = find_hanging_and_inverted_angle(model, data, pend_qpos_adr)
    result = run_energy_swingup_trial(model, k_e, a_max, k_pos, k_vel, duration_s=6.0,
                                       hanging_angle=hanging_angle, inverted_angle=inverted_angle,
                                       kick_amplitude_m=kick_amplitude_m, kick_duration_s=kick_duration_s)
    cost = result["min_theta_dist_from_inverted_rad"]
    if result["guard_fired"]:
        cost += 5.0  # strictly worse than any survived outcome (max survived cost = pi < 5.0)
    return cost


def main() -> int:
    model = compose_ur5e_pendulum_model()
    data = mujoco.MjData(model)
    pend_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "/pendulum_hinge")
    pend_qpos_adr = model.jnt_qposadr[pend_jid]
    hanging_angle, inverted_angle = find_hanging_and_inverted_angle(model, data, pend_qpos_adr)
    print(f"hanging_angle={hanging_angle:.4f} rad  inverted_angle={inverted_angle:.4f} rad  E_top={E_TOP:.4f} J")

    bounds = [
        (1.0, 400.0),   # k_e
        (0.3, 3.0),     # a_max (m/s^2)
        (0.0, 20.0),    # k_pos
        (0.0, 10.0),    # k_vel
        (0.02, 0.15),   # kick_amplitude_m
        (0.1, 0.6),     # kick_duration_s
    ]
    print("=== searching (k_e, a_max, k_pos, k_vel, kick_amplitude_m, kick_duration_s) via differential_evolution ===")
    res = differential_evolution(objective, bounds, maxiter=30, popsize=14, tol=1e-4,
                                  seed=0, workers=-1, polish=False)
    print(f"Best params: k_e={res.x[0]:.4f}, a_max={res.x[1]:.4f}, k_pos={res.x[2]:.4f}, "
          f"k_vel={res.x[3]:.4f}, kick_amplitude_m={res.x[4]:.4f}, kick_duration_s={res.x[5]:.4f}")
    print(f"Best cost: {res.fun:.4f}")

    best = run_energy_swingup_trial(model, res.x[0], res.x[1], res.x[2], res.x[3], duration_s=10.0,
                                     hanging_angle=hanging_angle, inverted_angle=inverted_angle,
                                     kick_amplitude_m=res.x[4], kick_duration_s=res.x[5])
    print("Best candidate, re-validated at 10s:", best)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
