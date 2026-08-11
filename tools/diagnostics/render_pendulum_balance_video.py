#!/usr/bin/env python3
"""Render an MP4 of the torque-lane LQR pendulum-balance controller running
live on the composed UR5e+pendulum MuJoCo model.

Unlike render_trace_video.py (kinematic replay of a saved trace), this
script IS the sim loop -- it steps physics and applies the real LQR law
each cycle (same control code as pendulum_balance_torque_lqr.py /
pendulum_balance_disturbance_robustness.py's torque-lane run_trial_torque,
reused via import, not reimplemented), recording qpos each frame for
rendering afterward. Uses the DE-search-validated gains (the same "best"
config documented in docs/status/pendulum_balance_gain_search_2026-08-09.md
and used throughout pendulum_balance_disturbance_robustness.py) -- the
torque lane specifically, since the velocity lane was found to be
machine-precision chaotic (docs/status/pendulum_balance_disturbance_robustness_2026-08-10.md)
and is not a controller to put in a demo video.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pendulum_balance_torque_lqr as torque_base  # noqa: E402

TORQUE_GAIN_KWARGS = dict(
    q_arm_pos=432.34106997212757, q_arm_vel=411.8137103747564,
    q_pend_angle=41.77555808266924, q_pend_vel=87.33164380569532,
    r_weight=212.2053758198958,
)
# RETUNED 2026-08-10: the previous gains (q_arm_pos=2655.2, r_weight=0.347)
# saturated shoulder_pan 67.6% of cycles and spiked wrist_2 to 26x its torque
# limit at the current mount geometry (assets/ur5e_pendulum/
# pendulum_attachment.xml pos="0 -0.11 0.08") -- confirmed via
# tools/diagnostics/pendulum_balance_torque_lqr_search.py's saturation-aware
# re-search. These gains: 0% saturation for pert 0.05-0.30 rad, clean
# convergence (final err <=0.04 rad); pert 0.35-0.40 rad survive but only to
# a degraded, non-vertical resting offset (~0.4-0.5 rad residual, not a
# stiction freeze -- verified peak error exceeds the starting perturbation,
# and passive fully diverges at the same perturbation); fails at 0.6+ rad.


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--perturbation-rad", type=float, default=0.15,
                    help="Initial pendulum angle offset from vertical (rad). "
                         "0.15 rad (~8.6 deg) is inside the validated clean-convergence "
                         "range and is where passive-vs-active is most dramatic "
                         "(passive fully diverges at friction_x=1.0, see "
                         "docs/status/pendulum_balance_friction_sweep_2026-08-10.md).")
    p.add_argument("--duration-s", type=float, default=6.0)
    p.add_argument("--output", type=Path, default=REPO_ROOT / "outputs" / "pendulum_balance_demo.mp4")
    p.add_argument("--fps", type=float, default=30.0)
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=960)
    p.add_argument("--azimuth", type=float, default=140.0)
    p.add_argument("--elevation", type=float, default=-15.0)
    p.add_argument("--distance", type=float, default=1.7)
    p.add_argument("--lookat", type=float, nargs=3, default=[-0.35, -0.15, 0.6], metavar=("X", "Y", "Z"))
    p.add_argument("--frame", type=Path, default=None,
                    help="If set, render a single preview PNG at this path instead of a video.")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    model = torque_base.compose_ur5e_pendulum_model()
    data = mujoco.MjData(model)
    pend_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "/pendulum_hinge")
    pend_qpos_adr = model.jnt_qposadr[pend_joint_id]

    inverted_angle = torque_base.find_inverted_angle(model, data, pend_qpos_adr)
    K, q_eq, diag = torque_base.linearize_and_design_lqr(model, inverted_angle, **TORQUE_GAIN_KWARGS)
    n_unstable = int(np.sum(np.abs(diag["eigvals_discrete"]) >= 1.0))
    if n_unstable > 0:
        print("WARNING: linear design is not stable -- aborting.")
        return 1

    data.qpos[:6] = torque_base.ARM_Q0
    data.qpos[pend_qpos_adr] = inverted_angle + args.perturbation_rad
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)

    model.vis.global_.offwidth = max(int(model.vis.global_.offwidth), int(args.width))
    model.vis.global_.offheight = max(int(model.vis.global_.offheight), int(args.height))

    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(model, cam)
    cam.azimuth = float(args.azimuth)
    cam.elevation = float(args.elevation)
    cam.distance = float(args.distance)
    cam.lookat[:] = np.asarray(args.lookat, dtype=np.float64)

    renderer = mujoco.Renderer(model, height=args.height, width=args.width)

    if args.frame is not None:
        renderer.update_scene(data, camera=cam)
        args.frame.parent.mkdir(parents=True, exist_ok=True)
        imageio.imwrite(str(args.frame), renderer.render())
        print(f"Wrote preview frame to {args.frame}")
        return 0

    n_steps = int(args.duration_s * torque_base.RATE_HZ)
    frame_stride = max(1, round(1.0 / (args.fps * torque_base.CONTROL_DT)))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    # ffmpeg via subprocess (piped raw rgb24 frames) instead of imageio.get_writer:
    # imageio has no working video backend in this env (no imageio-ffmpeg/pyav
    # package installed), but the system ffmpeg binary is already present --
    # avoids adding a new pip dependency for what's a one-off diagnostic video.
    ffmpeg_cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pixel_format", "rgb24",
        "-video_size", f"{args.width}x{args.height}", "-framerate", str(args.fps),
        "-i", "-", "-pix_fmt", "yuv420p", str(args.output),
    ]
    proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)
    written = 0
    fell_at = None
    for step in range(n_steps):
        theta = float(data.qpos[pend_qpos_adr])
        theta_err = float(np.mod(theta - inverted_angle + np.pi, 2 * np.pi) - np.pi)
        if abs(theta_err) > torque_base.FALL_THRESHOLD_RAD and fell_at is None:
            fell_at = step * torque_base.CONTROL_DT

        dq = data.qpos.copy()
        dq[6] = theta_err
        dq[:6] = data.qpos[:6] - q_eq[:6]
        dqd = data.qvel.copy()
        state = np.concatenate([dq, dqd])

        tau_extra = -K @ state
        tau_gravity = torque_base.static_gravity_torque(model, data.qpos)[:6]
        tau = np.clip(tau_extra + tau_gravity, -torque_base.TORQUE_LIMIT_NM, torque_base.TORQUE_LIMIT_NM)
        data.ctrl[:6] = tau

        if step % frame_stride == 0:
            renderer.update_scene(data, camera=cam)
            proc.stdin.write(renderer.render().tobytes())
            written += 1

        mujoco.mj_step(model, data)
        if fell_at is not None:
            break
    proc.stdin.close()
    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg exited with code {proc.returncode}")

    final_theta_err = float(np.mod(float(data.qpos[pend_qpos_adr]) - inverted_angle + np.pi, 2 * np.pi) - np.pi)
    print(f"Wrote {written} frames ({written / args.fps:.1f}s at {args.fps} fps) to {args.output}")
    print(f"perturbation_rad={args.perturbation_rad}  fell_at_s={fell_at}  final_theta_err={final_theta_err:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
