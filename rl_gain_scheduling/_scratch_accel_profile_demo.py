#!/usr/bin/env python3
"""Ad-hoc diagnostic: run the trained RL gain-scheduling policy at mid-height
under a constant-acceleration-then-reverse X target profile it was never
trained on (training only used min-jerk move-hold targets), and record a
video + full trace.

Profile: target velocity ramps at +accel_mps2 for phase1_s (1s), then at
-accel_mps2 for phase2_s (4s) -- X axis only (the only driven axis in this
whole system; Y/Z are held).

This duplicates gain_scheduling_env.py's per-step logic (obs construction,
action->gains, adapter step) rather than modifying the trained env, since
this is a one-off stress test of an out-of-distribution target shape, not a
permanent feature.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import mujoco
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from stable_baselines3 import PPO  # noqa: E402

from controller_core.kinematics_utils import orientation_error_vec_wxyz  # noqa: E402
from controller_core.logging_utils import JsonlTraceWriter, json_dumps_safe  # noqa: E402
from rl_gain_scheduling.gain_scheduling_env import (  # noqa: E402
    ACTION_DIM,
    ACTIVE_ORIGIN_Q,
    LOWER_B_Q,
    OBS_DIM,
    rescale_action_to_gains,
)
from simulation.ur5e_mujoco_torque import (  # noqa: E402
    apply_start_q,
    build_initial_state_and_adapter,
    build_mujoco_state,
    load_model,
)
from transport_metrics import GAIN_FIELDS  # noqa: E402

MODEL_PATH = REPO_ROOT / "outputs" / "rl_gain_scheduling" / "reward_v3_2M" / "models" / "ppo_gain_scheduler_final.zip"
CONFIG_PATH = REPO_ROOT / "config" / "rl_gain_scheduling_reward_v3.yaml"

ACCEL_MPS2 = 0.03  # chosen amplitude -- see report for reasoning
PHASE1_S = 1.0  # forward acceleration
PHASE2_S = 4.0  # reverse acceleration, same amplitude
TOTAL_S = PHASE1_S + PHASE2_S
HEIGHT_ALPHA = 0.5  # mid height


def accel_profile_target(start_x: float, t: float) -> tuple[float, float]:
    """Piecewise-constant-acceleration target: +ACCEL_MPS2 for PHASE1_S, then
    -ACCEL_MPS2 for PHASE2_S. Returns (target_x, target_x_vel)."""
    if t <= PHASE1_S:
        v = ACCEL_MPS2 * t
        x = 0.5 * ACCEL_MPS2 * t * t
    else:
        t2 = t - PHASE1_S
        v1 = ACCEL_MPS2 * PHASE1_S
        x1 = 0.5 * ACCEL_MPS2 * PHASE1_S * PHASE1_S
        v = v1 - ACCEL_MPS2 * t2
        x = x1 + v1 * t2 - 0.5 * ACCEL_MPS2 * t2 * t2
    return start_x + x, v


def main() -> int:
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--accel-mps2", type=float, default=None, help="Override ACCEL_MPS2 for this run.")
    p.add_argument("--height-alpha", type=float, default=None, help="Override HEIGHT_ALPHA for this run (can be outside [0,1] to extrapolate).")
    args = p.parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    global ACCEL_MPS2, HEIGHT_ALPHA
    if args.accel_mps2 is not None:
        ACCEL_MPS2 = float(args.accel_mps2)
    if args.height_alpha is not None:
        HEIGHT_ALPHA = float(args.height_alpha)

    model = PPO.load(str(MODEL_PATH))

    import yaml

    cfg = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    ctrl_cfg = cfg["controller"]
    gain_bounds = {name: (float(b[0]), float(b[1])) for name, b in cfg["env"]["gain_bounds"].items()}

    mj_model, data, site_id, joint_ids, actuator_ids = load_model(REPO_ROOT / cfg["mujoco"]["scene_xml"])
    q_start = (1.0 - HEIGHT_ALPHA) * ACTIVE_ORIGIN_Q + HEIGHT_ALPHA * LOWER_B_Q
    apply_start_q(mj_model, data, q_start)

    state0, adapter = build_initial_state_and_adapter(
        mj_model, data, site_id, joint_ids,
        controller_cfg=ctrl_cfg,
        transport_axis_index=0,
        target_x_delta=0.0,
        controller_kind="impedance",
        force_hold_current_pose=False,
        gravity_mode="gravity_comp",
        gravity_source=str(cfg["mujoco"].get("gravity_source", "mujoco_qfrc")),
        coriolis_feedforward=bool(cfg["mujoco"].get("coriolis_feedforward", False)),
        torque_limit_scale=1.0,
    )

    start_x = float(state0.ee_pos[0])
    max_steps = max(1, round(TOTAL_S / float(mj_model.opt.timestep)))
    prev_action = np.zeros(ACTION_DIM, dtype=np.float32)
    prev_tau = np.zeros(6, dtype=np.float64)

    def build_obs(ee_pos, ee_quat, q, qd, ee_lin_vel, ee_ang_vel, x_error, y_error, z_error, orientation_error_norm, elapsed, target_now):
        q = np.asarray(q, dtype=np.float64).reshape(6)
        qd = np.asarray(qd, dtype=np.float64).reshape(6)
        ori_vec = orientation_error_vec_wxyz(
            np.asarray(state0.reference_quat, dtype=np.float64), np.asarray(ee_quat, dtype=np.float64)
        )
        move_phase_indicator = min(1.0, elapsed / PHASE1_S)
        obs = np.concatenate([
            np.sin(q), np.cos(q), qd / 1.5,
            np.asarray(ee_pos, dtype=np.float64).reshape(3),
            [x_error, y_error, z_error], ori_vec,
            np.asarray(ee_lin_vel, dtype=np.float64).reshape(3),
            np.asarray(ee_ang_vel, dtype=np.float64).reshape(3),
            [elapsed / TOTAL_S], [move_phase_indicator],
            [target_now - start_x],  # dynamic target delta -- out-of-distribution vs the trained static value
            prev_action,
        ]).astype(np.float32)
        assert obs.shape == (OBS_DIM,)
        return obs

    obs = build_obs(
        state0.ee_pos, state0.ee_quat, state0.q, state0.qd, state0.ee_lin_vel, state0.ee_ang_vel,
        0.0, 0.0, 0.0, 0.0, 0.0, start_x,
    )

    trace_path = output_dir / "trace.jsonl"
    terminated = False
    termination_reason = ""
    step_count = 0
    with JsonlTraceWriter(trace_path) as writer:
        while step_count < max_steps:
            action, _ = model.predict(obs, deterministic=True)
            action = np.clip(np.asarray(action, dtype=np.float64).reshape(ACTION_DIM), -1.0, 1.0)
            gains = rescale_action_to_gains(action, gain_bounds)
            adapter.controller.set_gains(gains)

            t_s = float(data.time)
            target_x_now, target_x_vel_now = accel_profile_target(start_x, t_s)
            target_ee_pos = np.array([target_x_now, state0.ee_pos[1], state0.ee_pos[2]], dtype=np.float64)
            target_ee_vel = np.array([target_x_vel_now, 0.0, 0.0], dtype=np.float64)

            pre_state = build_mujoco_state(
                mj_model, data, site_id=site_id, joint_ids=joint_ids,
                time_s=t_s, dt_s=float(mj_model.opt.timestep),
                target_x=target_x_now, target_x_vel=target_x_vel_now,
                target_axis=target_ee_pos[0], target_axis_vel=target_ee_vel[0],
                target_ee_pos=target_ee_pos, target_ee_vel=target_ee_vel,
                reference_quat=state0.reference_quat,
                hold_current_pose=False, transport_axis_index=0, gravity_compensation=True,
            )
            tau, diag = adapter.step(state=pre_state)
            data.ctrl[:6] = np.asarray(tau, dtype=np.float64).reshape(6)
            mujoco.mj_step(mj_model, data)
            step_count += 1

            post_state = build_mujoco_state(
                mj_model, data, site_id=site_id, joint_ids=joint_ids,
                time_s=float(data.time), dt_s=float(mj_model.opt.timestep),
                target_x=target_x_now, target_x_vel=target_x_vel_now,
                target_axis=target_ee_pos[0], target_axis_vel=target_ee_vel[0],
                target_ee_pos=target_ee_pos, target_ee_vel=target_ee_vel,
                reference_quat=state0.reference_quat,
                hold_current_pose=False, transport_axis_index=0, gravity_compensation=True,
            )

            x_error = float(diag.get("axis_error", target_x_now - float(post_state.ee_pos[0])))
            y_error = float(state0.ee_pos[1] - post_state.ee_pos[1])
            z_error = float(state0.ee_pos[2] - post_state.ee_pos[2])
            orientation_error_norm = float(diag.get("orientation_error_norm", 0.0))
            tau_applied = np.asarray(diag.get("tau_applied", tau), dtype=np.float64).reshape(6)
            tau_controller = np.asarray(diag.get("tau_controller", tau), dtype=np.float64).reshape(6)
            safety_ok = bool(diag.get("safety_ok", True))

            row = {
                "step": int(step_count),
                "time_s": float(data.time),
                "ee_pos": np.asarray(post_state.ee_pos, dtype=np.float64).tolist(),
                "ee_quat": np.asarray(post_state.ee_quat, dtype=np.float64).tolist(),
                "qd": np.asarray(post_state.qd, dtype=np.float64).tolist(),
                "q": np.asarray(post_state.q, dtype=np.float64).tolist(),
                "orientation_error_norm": orientation_error_norm,
                "x_error": x_error,
                "target_x": float(target_x_now),
                "tau_controller": tau_controller.tolist(),
                "tau_applied": tau_applied.tolist(),
                "tau": np.asarray(tau, dtype=np.float64).tolist(),
                "gains": {name: float(gains[name]) for name in GAIN_FIELDS},
            }
            writer.write_row(row)

            if not safety_ok:
                terminated = True
                termination_reason = str(diag.get("safety_reason", ""))
                print(f"[SAFETY GUARD FIRED] step={step_count} t={t_s:.3f}s reason={termination_reason}")
                break

            prev_action = action.astype(np.float32).copy()
            prev_tau = np.asarray(tau, dtype=np.float64).copy()
            obs = build_obs(
                post_state.ee_pos, post_state.ee_quat, post_state.q, post_state.qd,
                post_state.ee_lin_vel, post_state.ee_ang_vel,
                x_error, y_error, z_error, orientation_error_norm, float(data.time), target_x_now,
            )

    summary = {
        "terminated": terminated,
        "termination_reason": termination_reason,
        "steps_completed": step_count,
        "max_steps": max_steps,
        "final_ee_x": float(post_state.ee_pos[0]),
        "start_ee_x": start_x,
        "achieved_x_delta_m": float(post_state.ee_pos[0] - start_x),
        "accel_mps2": ACCEL_MPS2,
        "phase1_s": PHASE1_S,
        "phase2_s": PHASE2_S,
        "height_alpha": HEIGHT_ALPHA,
    }
    (output_dir / "summary.json").write_text(json_dumps_safe(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
