#!/usr/bin/env python3
"""Mega pose search: which arm pose best tolerates SUSTAINED, SYMMETRIC
back-and-forth oscillation -- the property swing-up actually needs (not
one-directional X-range or speed, which were already characterized
separately and don't predict this).

Two stages:
1. Kinematic pre-filter (see /tmp/pose_search_survivors.pkl / _diverse.pkl,
   built inline in the calling session): sample (shoulder_lift, elbow,
   wrist_1, wrist_2) broadly, keep cond(J) < 30 and site height in
   [0.3, 1.2]m, then pick diverse representatives across (height, wrist_2)
   bins rather than just the globally best-conditioned cluster.
2. THIS script: for each candidate pose, run N forced alternating
   raised-cosine kicks (the same validated zero-endpoint-velocity profile
   from pendulum_swingup_multi_kick.py) on the BARE arm (no pendulum --
   this measures the arm+controller's own oscillation stability, which is
   what limits swing-up regardless of which pendulum is attached) and
   score by how many kicks survive before any safety guard trips, plus
   peak achieved speed as a tie-break.
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import mujoco
import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from simulation.ur5e_mujoco_torque import (  # noqa: E402
    build_initial_state_and_adapter,
    build_mujoco_state,
)

CONFIG_PATH = REPO_ROOT / "config" / "ur5e_mujoco_torque_osc_hanging_pose_friction_ff.yaml"
SCENE_XML = REPO_ROOT / "assets" / "ur5e_torque" / "scene.xml"
RATE_HZ = 500.0
CONTROL_DT = 1.0 / RATE_HZ
KICK_AMPLITUDE_M = 0.08
KICK_DURATION_S = 0.22
N_KICKS = 6
HOLD_GAP_S = 0.15


def load_config() -> dict:
    with CONFIG_PATH.open() as fp:
        return yaml.safe_load(fp)


def run_oscillation_stability_trial(q6: np.ndarray) -> dict:
    config = load_config()
    model = mujoco.MjModel.from_xml_path(str(SCENE_XML))
    data = mujoco.MjData(model)
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "attachment_site")
    joint_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in [
        "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
        "wrist_1_joint", "wrist_2_joint", "wrist_3_joint",
    ]]
    data.qpos[:6] = q6
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

    kicks_survived = 0
    peak_speed = 0.0
    guard_reason = None
    t_global = 0.0

    for kick_idx in range(N_KICKS):
        sign = 1.0 if kick_idx % 2 == 0 else -1.0
        kick_start_x = x0  # always kick from x0 -- isolates oscillation tolerance, not drift accumulation
        n_steps = int(KICK_DURATION_S * RATE_HZ)
        kick_ok = True
        for step in range(n_steps):
            tau_local = step * CONTROL_DT
            omega_kick = 2.0 * np.pi / KICK_DURATION_S
            target_x = kick_start_x + sign * 0.5 * KICK_AMPLITUDE_M * (1.0 - np.cos(omega_kick * tau_local))
            target_x_vel = sign * 0.5 * KICK_AMPLITUDE_M * omega_kick * np.sin(omega_kick * tau_local)
            a_cmd = float(sign * 0.5 * KICK_AMPLITUDE_M * omega_kick * omega_kick * np.cos(omega_kick * tau_local))
            state = build_mujoco_state(
                model, data, site_id=site_id, joint_ids=joint_ids,
                time_s=t_global, dt_s=CONTROL_DT,
                target_x=target_x, target_x_vel=target_x_vel, target_x_accel=a_cmd,
                reference_quat=state0.ee_quat, transport_axis_index=0,
                gravity_compensation=True,
            )
            tau, diag = adapter.step(state=state)
            if not bool(diag.get("safety_ok", True)):
                kick_ok = False
                guard_reason = str(diag.get("safety_reason", ""))
                break
            data.ctrl[:6] = np.asarray(tau, dtype=np.float64).reshape(6)
            mujoco.mj_step(model, data)
            t_global += CONTROL_DT
            jacp = np.zeros((3, model.nv))
            jacr = np.zeros((3, model.nv))
            mujoco.mj_jacSite(model, data, jacp, jacr, site_id)
            speed = float(np.linalg.norm(jacp[:, :6] @ data.qvel[:6]))
            peak_speed = max(peak_speed, speed)
        if not kick_ok:
            break
        kicks_survived += 1
        # brief hold between kicks
        hold_steps = int(HOLD_GAP_S * RATE_HZ)
        hold_ok = True
        for _ in range(hold_steps):
            state = build_mujoco_state(
                model, data, site_id=site_id, joint_ids=joint_ids,
                time_s=t_global, dt_s=CONTROL_DT,
                target_x=x0, target_x_vel=0.0, target_x_accel=0.0,
                reference_quat=state0.ee_quat, transport_axis_index=0,
                gravity_compensation=True,
            )
            tau, diag = adapter.step(state=state)
            if not bool(diag.get("safety_ok", True)):
                hold_ok = False
                guard_reason = str(diag.get("safety_reason", ""))
                break
            data.ctrl[:6] = np.asarray(tau, dtype=np.float64).reshape(6)
            mujoco.mj_step(model, data)
            t_global += CONTROL_DT
        if not hold_ok:
            break

    return {
        "kicks_survived": kicks_survived, "n_kicks_target": N_KICKS,
        "peak_speed_mps": peak_speed, "guard_reason": guard_reason,
    }


def main() -> int:
    diverse_path = REPO_ROOT / "outputs" / "hanging_pose_fwd_check" / "pose_search_diverse.pkl"
    with open(diverse_path, "rb") as f:
        diverse = pickle.load(f)

    results = []
    for i, (cond, sl, el, w1, w2, pos) in enumerate(diverse):
        q6 = np.array([0.0, sl, el, w1, w2, 0.0])
        r = run_oscillation_stability_trial(q6)
        results.append({
            "idx": i, "cond": cond, "sl": sl, "el": el, "w1": w1, "w2": w2,
            "height": float(pos[2]), **r,
        })
        print(f"[{i+1}/{len(diverse)}] cond={cond:.2f} h={pos[2]:.2f} w2={w2:.2f} -> "
              f"kicks={r['kicks_survived']}/{N_KICKS} peak_speed={r['peak_speed_mps']:.3f} "
              f"reason={r['guard_reason']}", flush=True)

    results.sort(key=lambda r: (-r["kicks_survived"], -r["peak_speed_mps"]))
    print("\n=== TOP 10 ===")
    for r in results[:10]:
        print(r)

    results_path = REPO_ROOT / "outputs" / "hanging_pose_fwd_check" / "pose_search_results.pkl"
    with open(results_path, "wb") as f:
        pickle.dump(results, f)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
