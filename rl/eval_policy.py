#!/usr/bin/env python3
"""
Evaluate a trained PPO policy for UR5 Y-axis transport in CoppeliaSim.

Runs a deterministic rollout, records EE trajectory and torques, and
produces a summary JSON compatible with the existing runner format.

Usage:
    python rl/eval_policy.py --model outputs/rl_models/ppo_y_transport
    python rl/eval_policy.py --model outputs/rl_models/ppo_y_transport --episodes 5
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

DEFAULT_CONFIG = REPO_ROOT / "rl" / "config.yaml"
DEFAULT_SUMMARY_DIR = REPO_ROOT / "outputs" / "rl_eval"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=str, required=True, help="Path to saved PPO model")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--host", type=str, default=None)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--coppelia-root", type=str, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_SUMMARY_DIR)
    parser.add_argument("--deterministic", action="store_true", default=True)
    args = parser.parse_args()

    from stable_baselines3 import PPO
    from rl.coppelia_y_transport_env import CoppeliaYTransportEnv

    args.output_dir.mkdir(parents=True, exist_ok=True)

    env = CoppeliaYTransportEnv(
        config_path=args.config,
        host=args.host,
        port=args.port,
        coppelia_root=args.coppelia_root,
    )

    model = PPO.load(args.model)
    print(f"Loaded model from {args.model}")

    all_summaries = []

    for ep in range(args.episodes):
        obs, info = env.reset()
        initial_ee = info["initial_ee_pos"]
        target_y = info["target_y"]

        trajectory = []
        total_reward = 0.0
        step = 0
        terminated = False
        truncated = False
        max_z_drift = 0.0
        max_x_drift = 0.0
        max_ori_err = 0.0
        max_qd = 0.0

        t_start = time.time()

        while not terminated and not truncated:
            action, _ = model.predict(obs, deterministic=args.deterministic)
            obs, reward, terminated, truncated, info = env.step(action)

            total_reward += reward
            step += 1

            z_drift = abs(info["z_err"])
            x_drift = abs(info["x_err"])
            max_z_drift = max(max_z_drift, z_drift)
            max_x_drift = max(max_x_drift, x_drift)
            max_ori_err = max(max_ori_err, info["ori_err"])
            max_qd = max(max_qd, info["max_qd"])

            trajectory.append({
                "step": step,
                "ee_pos": info["ee_pos"],
                "x_err": info["x_err"],
                "y_err": info["y_err"],
                "z_err": info["z_err"],
                "ori_err": info["ori_err"],
                "reward": float(reward),
            })

        t_elapsed = time.time() - t_start
        final_info = info

        y_displacement = float(initial_ee[1]) - float(final_info["ee_pos"][1])

        summary = {
            "episode": ep,
            "model_path": args.model,
            "total_reward": total_reward,
            "steps": step,
            "terminated": terminated,
            "truncated": truncated,
            "termination_reason": final_info.get("termination_reason", ""),
            "initial_ee_pos": initial_ee,
            "final_ee_pos": final_info["ee_pos"],
            "target_y": target_y,
            "y_displacement_m": y_displacement,
            "max_z_drift_m": max_z_drift,
            "max_x_drift_m": max_x_drift,
            "max_orientation_error_rad": max_ori_err,
            "max_joint_velocity_radps": max_qd,
            "final_x_err_m": final_info["x_err"],
            "final_y_err_m": final_info["y_err"],
            "final_z_err_m": final_info["z_err"],
            "final_ori_err_rad": final_info["ori_err"],
            "wall_time_s": t_elapsed,
            "success": not terminated,
        }

        all_summaries.append(summary)

        status = "TRUNCATED (full episode)" if truncated else f"TERMINATED ({final_info.get('termination_reason', '?')})"
        print(f"\nEpisode {ep}: {status}")
        print(f"  Steps: {step}, Total reward: {total_reward:.2f}")
        print(f"  Y displacement: {y_displacement:.4f} m")
        print(f"  Max Z drift: {max_z_drift:.4f} m")
        print(f"  Max X drift: {max_x_drift:.4f} m")
        print(f"  Max ori error: {max_ori_err:.4f} rad ({np.degrees(max_ori_err):.1f} deg)")
        print(f"  Max joint vel: {max_qd:.3f} rad/s")

    summary_path = args.output_dir / "eval_summary.json"
    with open(summary_path, "w") as f:
        json.dump(all_summaries, f, indent=2)
    print(f"\nSummary saved to {summary_path}")

    if len(all_summaries) > 1:
        rewards = [s["total_reward"] for s in all_summaries]
        y_disps = [s["y_displacement_m"] for s in all_summaries]
        z_drifts = [s["max_z_drift_m"] for s in all_summaries]
        successes = sum(1 for s in all_summaries if s["success"])
        print(f"\n=== Aggregate ({args.episodes} episodes) ===")
        print(f"  Success rate: {successes}/{args.episodes}")
        print(f"  Avg reward: {np.mean(rewards):.2f} +/- {np.std(rewards):.2f}")
        print(f"  Avg Y disp: {np.mean(y_disps):.4f} m")
        print(f"  Avg max Z drift: {np.mean(z_drifts):.4f} m")

    env.close()


if __name__ == "__main__":
    main()
