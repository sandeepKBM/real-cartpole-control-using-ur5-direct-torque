#!/usr/bin/env python3
"""Render an MP4 of the bare UR5e (no pendulum/tool attachment) doing an
X-transport move under the CartesianVelocityController (velocity lane,
ik_seeded_resolution) -- reuses velocity_gain_tuning/envs/VelocityTransportEnv
directly (the same env this repo's gain searches were run against, not a
reimplementation) so the rendered motion is exactly what that env's own
metrics were computed from, not a hand-rolled approximation.

Gains: the best empirically-found result across every search_result_*.json
in outputs/velocity_gain_tuning/ by pass_fraction on the env's own 128-case
safety grid (85.2%, search_result_nullspace_v2_largebudget_20260806_211416.json)
-- this lane has never reached a fully clean 100% pass rate in this repo's
history, unlike the torque lane; reported honestly, not rounded up.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import mujoco
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from velocity_gain_tuning.envs.velocity_transport_env import (  # noqa: E402
    ACTION_FIELDS,
    VelocityTransportEnv,
    VelocityTransportEnvConfig,
)
from velocity_gain_tuning.poses import scenario_by_name  # noqa: E402
from hardware.local_dynamics import DEFAULT_SCENE_XML, DEFAULT_SITE_NAME  # noqa: E402

BEST_GAINS = {
    "kp_x": 17.787446555647378,
    "kp_rot": 29.813962363403302,
    "ik_joint_gain": 19.688775153158705,
    "pinv_damping": 2.98967594085723e-05,
    "qp_task_weight": 298703278.7344936,
    "ik_max_joint_deviation_rad": 0.33823317650230095,
}


def gains_to_action(gains: dict) -> np.ndarray:
    """Inverse of VelocityTransportEnv's action_to_gains -- maps a physical
    gain dict back to the [-1,1]^6 action space (ACTION_FIELDS order)."""
    action = np.zeros(len(ACTION_FIELDS))
    for i, (name, lo, hi, is_log) in enumerate(ACTION_FIELDS):
        v = gains[name]
        if is_log:
            frac = (np.log(v) - np.log(lo)) / (np.log(hi) - np.log(lo))
        else:
            frac = (v - lo) / (hi - lo)
        action[i] = np.clip(2.0 * frac - 1.0, -1.0, 1.0)
    return action


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scenario", default="hanging_alpha_0_5",
                    help="Pose scenario name (see velocity_gain_tuning/poses.py). "
                         "hanging_alpha_0_5 has by far the largest validated safe range (~0.185m).")
    p.add_argument("--target-x-delta-m", type=float, default=0.10)
    p.add_argument("--move-duration-s", type=float, default=1.0)
    p.add_argument("--duration-s", type=float, default=3.0)
    p.add_argument("--output", type=Path, default=REPO_ROOT / "outputs" / "velocity_x_transport_demo.mp4")
    p.add_argument("--fps", type=float, default=30.0)
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=960)
    p.add_argument("--azimuth", type=float, default=90.0)
    p.add_argument("--elevation", type=float, default=-12.0)
    p.add_argument("--distance", type=float, default=2.2)
    p.add_argument("--lookat", type=float, nargs=3, default=[0.0, -0.3, 0.6], metavar=("X", "Y", "Z"))
    return p.parse_args()


def main() -> int:
    args = parse_args()
    scenario = scenario_by_name(args.scenario)

    # settle_cycles_for_early_stop disabled (set to an unreachable count) --
    # the env normally truncates an episode a few cycles after the target is
    # reached (a real, deliberate speedup for the gain-search harness this
    # env was built for), but that cuts a demo video off right as the move
    # finishes with no hold. Forcing the full duration_s instead.
    env_cfg = VelocityTransportEnvConfig(
        duration_s=args.duration_s, move_duration_s=args.move_duration_s,
        settle_cycles_for_early_stop=10**9,
    )
    env = VelocityTransportEnv(env_cfg, seed=0)
    action = gains_to_action(BEST_GAINS)
    obs, info = env.reset(seed=0, options={
        "scenario": scenario,
        "target_x_delta_m": args.target_x_delta_m,
        "move_duration_s": args.move_duration_s,
    })

    q_hist = [env._q.copy()]
    guard_reason = None
    terminated = truncated = False
    while not (terminated or truncated):
        obs, reward, terminated, truncated, last_info = env.step(action)
        q_hist.append(env._q.copy())
        if last_info.get("guard_reason") is not None:
            guard_reason = last_info["guard_reason"]

    print(f"scenario={args.scenario} target_x_delta_m={args.target_x_delta_m} "
          f"steps={len(q_hist)} guard_reason={guard_reason} "
          f"final_x_error={last_info.get('x_error')} achieved_dx={last_info.get('achieved_x_delta_m')}")
    if guard_reason is not None:
        print("WARNING: a safety guard tripped during this trial -- not rendering, pick a smaller "
              "target_x_delta_m or a different scenario.")
        return 1

    model = mujoco.MjModel.from_xml_path(str(DEFAULT_SCENE_XML))
    model.vis.global_.offwidth = max(int(model.vis.global_.offwidth), int(args.width))
    model.vis.global_.offheight = max(int(model.vis.global_.offheight), int(args.height))
    data = mujoco.MjData(model)

    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(model, cam)
    cam.azimuth = float(args.azimuth)
    cam.elevation = float(args.elevation)
    cam.distance = float(args.distance)
    cam.lookat[:] = np.asarray(args.lookat, dtype=np.float64)

    renderer = mujoco.Renderer(model, height=args.height, width=args.width)

    dt_s = 1.0 / env_cfg.rate_hz
    frame_stride = max(1, round(1.0 / (args.fps * dt_s)))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg_cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pixel_format", "rgb24",
        "-video_size", f"{args.width}x{args.height}", "-framerate", str(args.fps),
        "-i", "-", "-pix_fmt", "yuv420p", str(args.output),
    ]
    proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)
    written = 0
    for i, q in enumerate(q_hist):
        if i % frame_stride != 0:
            continue
        data.qpos[:6] = q
        mujoco.mj_forward(model, data)
        renderer.update_scene(data, camera=cam)
        proc.stdin.write(renderer.render().tobytes())
        written += 1
    proc.stdin.close()
    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg exited with code {proc.returncode}")

    print(f"Wrote {written} frames ({written / args.fps:.1f}s at {args.fps} fps) to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
