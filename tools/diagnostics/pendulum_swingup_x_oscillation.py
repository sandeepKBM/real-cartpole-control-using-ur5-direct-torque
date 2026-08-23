#!/usr/bin/env python3
"""Can an oscillating (back-and-forth) X-transport trajectory pump enough
energy into the pendulum to flip it from hanging to inverted, at the new
"hanging" transport pose (no base rotation, wrist_2=11deg -- see this
session's pose-search work)?

Reuses the SAME torque-lane Cartesian impedance controller
(controller_core / simulation.ur5e_mujoco_torque) and config
(config/ur5e_mujoco_torque_osc_hanging_pose_friction_ff.yaml) used for every
other transport/speed/range check this session -- the arm's 6 joints are
actively controlled to track an oscillating target_x(t); the pendulum hinge
has no actuator and is driven purely by the resulting base motion (a real
underactuated cart-pole problem, not a scripted pendulum trajectory).

Per this repo's own documented history (6 prior RL gain-scheduling
failures, see AGENTS.md), NOT using RL for the parameter search --
scipy.optimize.differential_evolution, the established gradient-free method
every other gain search in this repo uses.
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
ARM_Q0 = np.array([0.0, -1.647600774429474, 1.5194589589274516, 0.1281414001301714,
                    np.radians(11.0), 0.0])
RATE_HZ = 500.0
CONTROL_DT = 1.0 / RATE_HZ


def load_config() -> dict:
    with CONFIG_PATH.open() as fp:
        return yaml.safe_load(fp)


def find_hanging_and_inverted_angle(model, data, pend_qpos_adr: int) -> tuple[float, float]:
    """Hanging = the passive equilibrium under gravity with the arm held at
    ARM_Q0 (found by long-settle free simulation from qpos=0); inverted =
    hanging + pi, wrapped."""
    data.qpos[:6] = ARM_Q0
    data.qpos[pend_qpos_adr] = 0.0
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)
    for _ in range(5000):
        mujoco.mj_step(model, data)
    hanging = float(np.mod(data.qpos[pend_qpos_adr] + np.pi, 2 * np.pi) - np.pi)
    inverted = float(np.mod(hanging + np.pi + np.pi, 2 * np.pi) - np.pi)
    return hanging, inverted


def run_swingup_trial(
    model,
    amplitude_m: float,
    frequency_hz: float,
    duration_s: float,
    hanging_angle: float,
    inverted_angle: float,
    x0: float | None = None,
) -> dict:
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
    x0 = float(state0.ee_pos[0]) if x0 is None else float(x0)

    n_steps = int(duration_s * RATE_HZ)
    omega = 2.0 * np.pi * frequency_hz
    peak_theta_dist_from_inverted = np.pi  # start maximally far (hanging)
    min_theta_dist = np.pi
    guard_fired = False
    guard_reason = None
    theta_hist = []

    for step in range(n_steps):
        t = step * CONTROL_DT
        target_x = x0 + amplitude_m * np.sin(omega * t)
        target_x_vel = amplitude_m * omega * np.cos(omega * t)
        target_x_accel = -amplitude_m * omega * omega * np.sin(omega * t)

        state = build_mujoco_state(
            model, data, site_id=site_id, joint_ids=joint_ids,
            time_s=t, dt_s=CONTROL_DT,
            target_x=target_x, target_x_vel=target_x_vel, target_x_accel=target_x_accel,
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

        theta = float(data.qpos[pend_qpos_adr])
        dist = abs(float(np.mod(theta - inverted_angle + np.pi, 2 * np.pi) - np.pi))
        min_theta_dist = min(min_theta_dist, dist)
        theta_hist.append(theta)

    return {
        "amplitude_m": amplitude_m,
        "frequency_hz": frequency_hz,
        "duration_s": duration_s,
        "min_theta_dist_from_inverted_rad": min_theta_dist,
        "flipped": min_theta_dist < 0.35,  # same order as this repo's FALL_THRESHOLD_RAD family
        "guard_fired": guard_fired,
        "guard_reason": guard_reason,
        "steps_completed": len(theta_hist),
    }


# Module-level (not a main()-local closure) and self-contained (composes its
# own model each call) so it survives pickling into differential_evolution's
# multiprocessing workers -- an MjModel/closure cannot cross a process
# boundary otherwise.
def objective(x):
    amplitude_m, frequency_hz = x
    model = compose_ur5e_pendulum_model()
    data = mujoco.MjData(model)
    pend_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "/pendulum_hinge")
    pend_qpos_adr = model.jnt_qposadr[pend_jid]
    hanging_angle, inverted_angle = find_hanging_and_inverted_angle(model, data, pend_qpos_adr)
    result = run_swingup_trial(model, amplitude_m, frequency_hz, duration_s=4.0,
                                hanging_angle=hanging_angle, inverted_angle=inverted_angle)
    cost = result["min_theta_dist_from_inverted_rad"]
    if result["guard_fired"]:
        cost += 5.0  # guard trip strictly worse than any survived outcome (max survived cost = pi < 5.0)
    return cost


def main() -> int:
    model = compose_ur5e_pendulum_model()
    data = mujoco.MjData(model)
    pend_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "/pendulum_hinge")
    pend_qpos_adr = model.jnt_qposadr[pend_jid]
    hanging_angle, inverted_angle = find_hanging_and_inverted_angle(model, data, pend_qpos_adr)
    print(f"hanging_angle={hanging_angle:.4f} rad  inverted_angle={inverted_angle:.4f} rad")

    bounds = [(0.02, 0.15), (0.3, 3.0)]  # amplitude_m, frequency_hz
    print("=== searching (amplitude_m, frequency_hz) via differential_evolution ===")
    # workers=_de_workers() (2026-08-12): third and last site in this family
    # still on workers=-1 after the bounded-worker fix landed elsewhere. Same
    # hazard as the others -- claims every core on a shared host and leaves
    # scipy>=1.17's forkserver start-method behaviour unpinned.
    from tools.diagnostics.pendulum_swingup_multi_kick import _de_workers
    res = differential_evolution(objective, bounds, maxiter=20, popsize=10, tol=1e-4,
                                  seed=0, workers=_de_workers(), polish=False)
    print(f"Best params: amplitude_m={res.x[0]:.4f}, frequency_hz={res.x[1]:.4f}")
    print(f"Best cost: {res.fun:.4f}")

    best = run_swingup_trial(model, res.x[0], res.x[1], duration_s=8.0,
                              hanging_angle=hanging_angle, inverted_angle=inverted_angle)
    print("Best candidate, re-validated at 8s:", best)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
