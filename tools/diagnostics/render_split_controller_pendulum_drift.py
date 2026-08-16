#!/usr/bin/env python3
"""Video of the split-controller (shoulder_lift/elbow/wrist_1 active, X-only
task) move+hold, WITH the corrected pendulum attached, at the user's real
robot pose. Shows real X tracking and Y/Z drift on a HUD overlay.

SIM-ONLY VISUALIZATION NOTE: by explicit request, this script does NOT stop
the simulation when the real safety guard (e.g. |Z-Z0| > 0.03 m) trips --
it keeps running for the full commanded duration so you can see the whole
attempted motion. The HUD marks the exact moment/reason the guard WOULD have
fired in a real run (position/direct_torque mode always stops there) -- this
script is for visualization only, no safety logic anywhere else in this repo
is changed, and this is not evidence a real or normal sim run would be
allowed to continue past that point.

CLI-configurable (no hand-edited per-experiment constants).

Usage:
  MUJOCO_GL=egl python tools/diagnostics/render_split_controller_pendulum_drift.py \\
      --dx 0.20 --move-duration 2.0 --hold-duration 1.0
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import mujoco
import numpy as np
import yaml
from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from simulation.ur5e_pendulum_compose import compose_ur5e_pendulum_model  # noqa: E402
from simulation.ur5e_mujoco_torque import (  # noqa: E402
    build_initial_state_and_adapter,
    build_mujoco_state,
    x_profile_target,
)
from tools.diagnostics.pendulum_balance_torque_lqr import find_inverted_angle  # noqa: E402

FPS = 30.0
WIDTH, HEIGHT = 1280, 960

#: The real robot pose the split controller (shoulder_lift/elbow/wrist_1
#: active, wrist_2 held away from its physical singularity limit) was built
#: for -- see config/ur5e_mujoco_torque_osc_tuned_split_base_wrist_lift_elbow_wrist1_x_only.yaml.
USER_REAL_POSE_Q = np.array(
    [-2.3688, -2.1801, -1.8838, -0.7962, 0.004714693, 0.0206], dtype=np.float64
)

DEFAULT_CONFIG = REPO_ROOT / "config" / "ur5e_mujoco_torque_osc_tuned_split_base_wrist_lift_elbow_wrist1_x_only.yaml"

HUD_MARGIN = 20


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dx", type=float, default=0.20, help="Target X displacement, meters.")
    p.add_argument("--move-duration", type=float, default=2.0)
    p.add_argument("--hold-duration", type=float, default=1.0)
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--cam-azimuth", type=float, default=100.0)
    p.add_argument("--cam-elevation", type=float, default=-15.0)
    p.add_argument("--cam-distance", type=float, default=1.9)
    p.add_argument("--cam-lookat", type=float, nargs=3, default=[-0.45, -0.18, 0.26])
    p.add_argument("--output", type=Path,
                    default=REPO_ROOT / "outputs" / "pendulum_renders" / "split_controller_pendulum_drift.mp4")
    return p.parse_args()


def draw_hud(frame: np.ndarray, *, t: float, speed_mps: float, x_target: float, x_error: float,
             y_drift: float, z_drift: float, guard_tripped: bool, guard_reason: str,
             guard_time_s: float | None) -> np.ndarray:
    img = Image.fromarray(frame)
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 26)
        font_small = ImageFont.truetype("DejaVuSans.ttf", 20)
    except OSError:
        font = ImageFont.load_default()
        font_small = font

    lines = [
        f"t = {t:5.2f} s",
        f"EE speed = {speed_mps:6.3f} m/s",
        f"X target = {x_target:+.4f} m   X error = {x_error:+.4f} m",
        f"Y drift = {y_drift:+.4f} m   Z drift = {z_drift:+.4f} m",
    ]
    if guard_tripped:
        lines.append(f"*** GUARD WOULD TRIP: {guard_reason} @ t={guard_time_s:.2f}s ***")
    panel_w, panel_h = 620, 24 + 30 * len(lines)
    draw.rectangle([HUD_MARGIN, HUD_MARGIN, HUD_MARGIN + panel_w, HUD_MARGIN + panel_h], fill=(0, 0, 0, 150))
    for i, line in enumerate(lines):
        color = (255, 60, 60, 255) if "GUARD" in line else (255, 255, 255, 255)
        draw.text((HUD_MARGIN + 12, HUD_MARGIN + 8 + 30 * i), line, font=font_small, fill=color)

    # Z-drift bar: fixed +-0.05m axis, red past the real 0.03m guard line.
    bar_y, bar_h = HEIGHT - 90, 36
    bar_x0, bar_x1 = HUD_MARGIN, WIDTH - HUD_MARGIN
    half_range = 0.05
    draw.rectangle([bar_x0, bar_y, bar_x1, bar_y + bar_h], outline=(255, 255, 255, 200), width=2)

    def to_px(v: float) -> int:
        frac = np.clip((v + half_range) / (2.0 * half_range), 0.0, 1.0)
        return int(bar_x0 + frac * (bar_x1 - bar_x0))

    guard_lo, guard_hi = to_px(-0.03), to_px(0.03)
    draw.line([(guard_lo, bar_y - 8), (guard_lo, bar_y + bar_h + 8)], fill=(255, 60, 60, 220), width=2)
    draw.line([(guard_hi, bar_y - 8), (guard_hi, bar_y + bar_h + 8)], fill=(255, 60, 60, 220), width=2)
    z_px = to_px(z_drift)
    draw.polygon([(z_px - 9, bar_y - 10), (z_px + 9, bar_y - 10), (z_px, bar_y - 1)], fill=(60, 180, 255, 255))
    draw.text((bar_x0, bar_y + bar_h + 8),
              f"Z drift (marker) vs real +/-0.03m guard line (red)  |  axis +/-{half_range:.2f}m",
              font=font_small, fill=(255, 255, 255, 255))

    return np.array(Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB"))


def main() -> int:
    args = parse_args()
    with args.config.open() as fp:
        cfg = yaml.safe_load(fp)
    ctrl_cfg = cfg["controller"]

    model = compose_ur5e_pendulum_model()
    data = mujoco.MjData(model)
    joint_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in [
        "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
        "wrist_1_joint", "wrist_2_joint", "wrist_3_joint",
    ]]
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "attachment_site")
    pend_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "/pendulum_hinge")
    pend_qpos_adr = model.jnt_qposadr[pend_jid]

    data.qpos[:6] = USER_REAL_POSE_Q
    mujoco.mj_forward(model, data)
    inverted_angle = find_inverted_angle(model, data, pend_qpos_adr)
    hanging_angle = inverted_angle
    # find_inverted_angle returns the unstable equilibrium; the stable
    # (hanging) one is pi away -- settle there so the pendulum hangs
    # naturally under gravity, matching every other render in this session.
    data.qpos[pend_qpos_adr] = hanging_angle + np.pi
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)

    state0, adapter = build_initial_state_and_adapter(
        model, data, site_id, joint_ids, controller_cfg=ctrl_cfg,
        transport_axis_index=0, target_x_delta=0.0, controller_kind="impedance",
        force_hold_current_pose=False,
        gravity_mode=cfg["mujoco"].get("gravity_mode", "gravity_comp"),
        gravity_source=cfg["mujoco"].get("gravity_source", "pinocchio"),
        coriolis_feedforward=bool(cfg["mujoco"].get("coriolis_feedforward", False)),
        torque_limit_scale=1.0,
    )
    start_ee = np.asarray(state0.ee_pos, dtype=np.float64).copy()
    x0 = float(start_ee[0])

    model.vis.global_.offwidth = max(int(model.vis.global_.offwidth), WIDTH)
    model.vis.global_.offheight = max(int(model.vis.global_.offheight), HEIGHT)
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(model, cam)
    cam.azimuth, cam.elevation, cam.distance = args.cam_azimuth, args.cam_elevation, args.cam_distance
    cam.lookat[:] = np.array(args.cam_lookat)
    renderer = mujoco.Renderer(model, height=HEIGHT, width=WIDTH)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.Popen([
        "ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pixel_format", "rgb24",
        "-video_size", f"{WIDTH}x{HEIGHT}", "-framerate", str(FPS), "-i", "-",
        "-pix_fmt", "yuv420p", str(args.output),
    ], stdin=subprocess.PIPE)

    dt = float(model.opt.timestep)
    duration_s = float(args.move_duration) + float(args.hold_duration)
    n_steps = int(np.ceil(duration_s / dt))
    frame_stride = max(1, round(1.0 / (FPS * dt)))
    jacp = np.zeros((3, model.nv))
    guard_tripped, guard_reason, guard_time_s = False, "", None
    written = 0

    for step in range(n_steps):
        t = step * dt
        target_now, target_vel_now = x_profile_target(
            "min_jerk_move_hold", x0, float(args.dx), t, duration_s,
            move_duration_s=float(args.move_duration),
        )
        target_ee_pos = start_ee.copy()
        target_ee_pos[0] = target_now
        target_ee_vel = np.zeros(3)
        target_ee_vel[0] = target_vel_now

        state = build_mujoco_state(
            model, data, site_id=site_id, joint_ids=joint_ids, time_s=t, dt_s=dt,
            target_x=float(target_now), target_x_vel=float(target_vel_now), target_x_accel=0.0,
            target_axis=float(target_now), target_axis_vel=float(target_vel_now),
            target_ee_pos=target_ee_pos, target_ee_vel=target_ee_vel,
            reference_quat=state0.reference_quat, transport_axis_index=0,
            gravity_compensation=True,
        )
        tau, diag = adapter.step(state=state)
        if not guard_tripped and not bool(diag.get("safety_ok", True)):
            guard_tripped = True
            guard_reason = str(diag.get("safety_reason", "") or "unknown")
            guard_time_s = t
            print(f"*** guard would trip: {guard_reason} @ t={t:.2f}s (continuing per --sim-only request) ***")

        data.ctrl[:6] = np.asarray(tau, dtype=np.float64).reshape(6)
        mujoco.mj_step(model, data)

        ee = np.asarray(data.site_xpos[site_id], dtype=np.float64)
        x_error = target_now - ee[0]
        y_drift = float(ee[1] - start_ee[1])
        z_drift = float(ee[2] - start_ee[2])
        mujoco.mj_jacSite(model, data, jacp, np.zeros((3, model.nv)), site_id)
        speed = float(np.linalg.norm(jacp[:, :6] @ data.qvel[:6]))

        if step % frame_stride == 0:
            renderer.update_scene(data, camera=cam)
            frame = renderer.render()
            frame = draw_hud(
                frame, t=t, speed_mps=speed, x_target=target_now - x0, x_error=x_error,
                y_drift=y_drift, z_drift=z_drift,
                guard_tripped=guard_tripped, guard_reason=guard_reason, guard_time_s=guard_time_s,
            )
            proc.stdin.write(frame.tobytes())
            written += 1

    proc.stdin.close()
    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg exited with code {proc.returncode}")

    final_ee = np.asarray(data.site_xpos[site_id], dtype=np.float64)
    print(f"achieved dx = {final_ee[0] - x0:+.4f} m (target {args.dx:+.4f} m)")
    print(f"final Y drift = {final_ee[1] - start_ee[1]:+.4f} m, Z drift = {final_ee[2] - start_ee[2]:+.4f} m")
    if guard_tripped:
        print(f"guard would have tripped: {guard_reason} @ t={guard_time_s:.2f}s")
    else:
        print("no guard tripped for the full duration")
    print(f"Wrote {written} frames ({written/FPS:.1f}s) to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
