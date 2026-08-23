#!/usr/bin/env python3
"""Render an MP4 of the multi-kick pendulum swing-up attempt. Re-tuned
2026-08-12 after the pendulum model correction (real 0.12m rod, real
attachment geometry, real ~0.30kg wrist mass, down from a wrong 0.30m/
~2.17kg placeholder model) -- this now reaches min_theta_dist=0.1785 rad
(~10.2deg from fully inverted) with ZERO guard trips in 8 kicks, a real
qualitative change from the pre-correction ~40deg-and-guard-trip result
the old BEST params/module docstring described. Also reports peak
end-effector Cartesian speed and peak pendulum tip speed alongside the
video, not just the swing angle.
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
sys.path.insert(0, str(REPO_ROOT / "tools" / "diagnostics"))

from simulation.ur5e_pendulum_compose import compose_ur5e_pendulum_model  # noqa: E402
from simulation.ur5e_mujoco_torque import build_initial_state_and_adapter, build_mujoco_state  # noqa: E402
from pendulum_swingup_energy_shaping import load_config, ARM_Q0, CONTROL_DT, RATE_HZ, find_hanging_and_inverted_angle  # noqa: E402
from pendulum_swingup_multi_kick import find_nearby_equilibrium, MIN_KICK_GAP_S, THETADOT_DEADBAND, K_RECENTER  # noqa: E402

BEST = {
    "kick_amplitude_m": 0.0861,
    "kick_duration_s": 0.1226,
    "phi_trigger_rad": 0.4316,
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--duration-s", type=float, default=15.0,
                    help="Full validated trial duration (no guard trip expected with BEST).")
    p.add_argument("--output", type=Path, default=REPO_ROOT / "outputs" / "swingup_multi_kick_demo.mp4")
    p.add_argument("--fps", type=float, default=30.0)
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=960)
    p.add_argument("--azimuth", type=float, default=100.0)
    p.add_argument("--elevation", type=float, default=-15.0)
    p.add_argument("--distance", type=float, default=1.9)
    p.add_argument("--lookat", type=float, nargs=3, default=[-0.3, -0.2, 0.5], metavar=("X", "Y", "Z"))
    p.add_argument("--config", type=Path, default=None,
                    help="Controller config override; default uses this script's own CONFIG_PATH.")
    p.add_argument("--arm-q-rad", type=float, nargs=6, default=None,
                    metavar=("Q1", "Q2", "Q3", "Q4", "Q5", "Q6"),
                    help="Override the module's ARM_Q0 starting pose for both the arm and the "
                         "hanging/inverted-angle computation (which now correctly depends on it "
                         "per the 2026-08-13 find_inverted_angle fix -- default uses ARM_Q0.")
    p.add_argument("--kick-amplitude-m", type=float, default=None, help="Override BEST['kick_amplitude_m'].")
    p.add_argument("--kick-duration-s", type=float, default=None, help="Override BEST['kick_duration_s'].")
    p.add_argument("--phi-trigger-rad", type=float, default=None, help="Override BEST['phi_trigger_rad'].")
    p.add_argument("--continue-through-guard", action="store_true",
                    help="Don't stop 0.4s after the first guard trip -- keep rendering the full "
                         "--duration-s. For sim-only guard-disabled runs (run_multi_kick_trial's "
                         "enforce_guard=False) where a trip is expected near the start and the "
                         "point is to show the full multi-kick sequence anyway.")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    model = compose_ur5e_pendulum_model()
    data = mujoco.MjData(model)
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "attachment_site")
    joint_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in [
        "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
        "wrist_1_joint", "wrist_2_joint", "wrist_3_joint",
    ]]
    pend_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "/pendulum_hinge")
    pend_qpos_adr = model.jnt_qposadr[pend_jid]
    pend_dof_adr = model.jnt_dofadr[pend_jid]
    tip_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "/pendulum_tip_site")
    jacp_ee = np.zeros((3, model.nv))
    jacr_ee = np.zeros((3, model.nv))
    jacp_tip = np.zeros((3, model.nv))
    jacr_tip = np.zeros((3, model.nv))

    arm_q = ARM_Q0 if args.arm_q_rad is None else np.asarray(args.arm_q_rad, dtype=np.float64)
    config = load_config() if args.config is None else load_config(args.config)
    # Set the arm pose BEFORE computing hanging/inverted angles -- that
    # computation now correctly depends on data.qpos[:6] (2026-08-13 fix),
    # so it must see the real arm pose, not whatever MjData(model) defaults
    # to (previously this was called with an unset, effectively-zero pose).
    data.qpos[:6] = arm_q
    hanging_angle, inverted_angle = find_hanging_and_inverted_angle(model, data, pend_qpos_adr)
    data.qpos[:6] = arm_q
    data.qpos[pend_qpos_adr] = hanging_angle
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)

    state0, adapter = build_initial_state_and_adapter(
        model, data, site_id, joint_ids,
        controller_cfg=config["controller"], transport_axis_index=0, target_x_delta=0.0,
        controller_kind="impedance", force_hold_current_pose=False,
        gravity_mode=config["mujoco"].get("gravity_mode", "gravity_comp"),
        gravity_source=config["mujoco"].get("gravity_source", "pinocchio"),
        coriolis_feedforward=bool(config["mujoco"].get("coriolis_feedforward", True)),
        torque_limit_scale=1.0,
    )
    x0 = float(state0.ee_pos[0])

    model.vis.global_.offwidth = max(int(model.vis.global_.offwidth), int(args.width))
    model.vis.global_.offheight = max(int(model.vis.global_.offheight), int(args.height))
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(model, cam)
    cam.azimuth, cam.elevation, cam.distance = args.azimuth, args.elevation, args.distance
    cam.lookat[:] = np.asarray(args.lookat, dtype=np.float64)
    renderer = mujoco.Renderer(model, height=args.height, width=args.width)

    n_steps = int(args.duration_s * RATE_HZ)
    frame_stride = max(1, round(1.0 / (args.fps * CONTROL_DT)))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg_cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pixel_format", "rgb24",
        "-video_size", f"{args.width}x{args.height}", "-framerate", str(args.fps),
        "-i", "-", "-pix_fmt", "yuv420p", str(args.output),
    ]
    proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)

    target_x, target_x_vel = x0, 0.0
    kick_active, kick_start_t, kick_sign = False, 0.0, 1.0
    kick_hold_x, last_kick_end_t, num_kicks = x0, -1e9, 0
    current_hanging_angle = hanging_angle
    written, guard_trip_t = 0, None
    kick_amp = args.kick_amplitude_m if args.kick_amplitude_m is not None else BEST["kick_amplitude_m"]
    kick_dur = args.kick_duration_s if args.kick_duration_s is not None else BEST["kick_duration_s"]
    phi_trig = args.phi_trigger_rad if args.phi_trigger_rad is not None else BEST["phi_trigger_rad"]
    min_theta_dist = np.pi
    peak_swing_t = 0.0
    peak_ee_speed = 0.0
    peak_ee_speed_t = 0.0
    peak_tip_speed = 0.0
    peak_tip_speed_t = 0.0
    peak_thetadot = 0.0

    for step in range(n_steps):
        t = step * CONTROL_DT
        theta = float(data.qpos[pend_qpos_adr])
        thetadot = float(data.qvel[pend_dof_adr])
        phi = float(np.mod(theta - current_hanging_angle + np.pi, 2 * np.pi) - np.pi)

        if kick_active and (t - kick_start_t) >= kick_dur:
            kick_active = False
            kick_hold_x = target_x
            last_kick_end_t = t
            current_hanging_angle = find_nearby_equilibrium(model, data.qpos[:6].copy(), pend_qpos_adr, theta)

        is_bootstrap = (num_kicks == 0 and step == 0)
        if is_bootstrap or (
            not kick_active and abs(phi) < phi_trig
            and abs(thetadot) > THETADOT_DEADBAND
            and (t - last_kick_end_t) >= MIN_KICK_GAP_S
        ):
            kick_active = True
            kick_start_t = t
            kick_sign = 1.0 if is_bootstrap else (1.0 if thetadot >= 0.0 else -1.0)
            num_kicks += 1

        if kick_active:
            tl = t - kick_start_t
            omega_k = 2.0 * np.pi / kick_dur
            target_x = kick_hold_x + kick_sign * 0.5 * kick_amp * (1.0 - np.cos(omega_k * tl))
            target_x_vel = kick_sign * 0.5 * kick_amp * omega_k * np.sin(omega_k * tl)
            a_cmd = float(kick_sign * 0.5 * kick_amp * omega_k * omega_k * np.cos(omega_k * tl))
        else:
            a_cmd = float(-K_RECENTER * (kick_hold_x - x0))
            target_x_vel = 0.0
            target_x = kick_hold_x

        state = build_mujoco_state(
            model, data, site_id=site_id, joint_ids=joint_ids, time_s=t, dt_s=CONTROL_DT,
            target_x=target_x, target_x_vel=target_x_vel, target_x_accel=a_cmd,
            reference_quat=state0.ee_quat, transport_axis_index=0, gravity_compensation=True,
        )
        tau, diag = adapter.step(state=state)
        if not bool(diag.get("safety_ok", True)) and guard_trip_t is None:
            guard_trip_t = t
            print(f"Guard tripped at t={t:.2f}s: {diag.get('safety_reason')}")

        data.ctrl[:6] = np.asarray(tau, dtype=np.float64).reshape(6)
        mujoco.mj_step(model, data)

        dist = abs(float(np.mod(theta - inverted_angle + np.pi, 2 * np.pi) - np.pi))
        if dist < min_theta_dist:
            min_theta_dist = dist
            peak_swing_t = t

        mujoco.mj_jacSite(model, data, jacp_ee, jacr_ee, site_id)
        ee_speed = float(np.linalg.norm(jacp_ee @ data.qvel))
        if ee_speed > peak_ee_speed:
            peak_ee_speed, peak_ee_speed_t = ee_speed, t

        mujoco.mj_jacSite(model, data, jacp_tip, jacr_tip, tip_site_id)
        tip_speed = float(np.linalg.norm(jacp_tip @ data.qvel))
        if tip_speed > peak_tip_speed:
            peak_tip_speed, peak_tip_speed_t = tip_speed, t

        peak_thetadot = max(peak_thetadot, abs(thetadot))

        if step % frame_stride == 0:
            renderer.update_scene(data, camera=cam)
            proc.stdin.write(renderer.render().tobytes())
            written += 1

        # Stop shortly after the guard trip -- a short tail shows the trip
        # happening, but continuing on lets the pendulum swing back down
        # near hanging again, which undersells the actual peak swing
        # reached (tracked separately above, not read off the final frame).
        if not args.continue_through_guard and guard_trip_t is not None and t > guard_trip_t + 0.4:
            break

    proc.stdin.close()
    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg exited with code {proc.returncode}")

    print(f"num_kicks={num_kicks} peak_swing_deg={np.degrees(np.pi - min_theta_dist):.1f} "
          f"(min_theta_dist_from_inverted={min_theta_dist:.4f} rad) at t={peak_swing_t:.2f}s  "
          f"guard_trip_t={guard_trip_t}  flipped={min_theta_dist < 0.35}")
    print(f"peak EE Cartesian speed: {peak_ee_speed:.3f} m/s at t={peak_ee_speed_t:.2f}s")
    print(f"peak pendulum TIP speed: {peak_tip_speed:.3f} m/s at t={peak_tip_speed_t:.2f}s")
    print(f"peak pendulum angular rate |thetadot|: {peak_thetadot:.3f} rad/s "
          f"({np.degrees(peak_thetadot):.1f} deg/s)")
    print(f"Wrote {written} frames ({written / args.fps:.1f}s at {args.fps} fps) to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
