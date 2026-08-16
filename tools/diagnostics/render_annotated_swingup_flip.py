#!/usr/bin/env python3
"""Same trial as render_energy_swingup_flip.py, but composites a live HUD
onto every frame: real measured end-effector speed (m/s), the commanded
pivot acceleration (m/s^2, a_cmd -- what the control law is actually
applying), and a growing red-shaded band showing the full back-and-forth
excursion range the end-effector has covered so far.

CLI-configurable from the start (this session's standing rule: no hand-edited
per-experiment values in diagnostic scripts) -- the damping/frictionloss
defaults match the validated damping=0.0001 case (see
render_energy_swingup_flip.py's own header for the full
overdamped-committed-model caveat, which applies identically here).

IMPORTANT, 2026-08-12: the gain/kick defaults below reproduce the flip that
was achieved at the PREVIOUS ARM_Q0 with the previous (local X) hinge axis.
ARM_Q0 has since changed to the real-hardware pose and the axis to local Z
(see pendulum_swingup_energy_shaping.py, from which this script imports both),
and NO gain/kick combination found so far produces a flip at that pair -- a
2200-evaluation differential-evolution search over (k_e, a_max, k_pos, k_vel,
kick amplitude, kick duration), with the kick bounds widened well past the
committed search's, got no closer than 2.24 rad from inverted (a ~51 deg peak
swing). Running this script with its bare defaults at the current pose is
therefore not a flip demo; it will also trip the controller's |Y-Y0| guard
during the kick. Pass an explicit, guard-surviving --kick-amplitude-m /
--kick-duration-s pair.

Usage:
  MUJOCO_GL=egl python tools/diagnostics/render_annotated_swingup_flip.py \\
      --damping 0.0001 --frictionloss 0.00005
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from simulation.ur5e_pendulum_compose import compose_ur5e_pendulum_model  # noqa: E402
from simulation.ur5e_mujoco_torque import (  # noqa: E402
    build_initial_state_and_adapter,
    build_mujoco_state,
)
from tools.diagnostics.pendulum_swingup_energy_shaping import (  # noqa: E402
    ARM_Q0, CONTROL_DT, E_TOP, G, I_PIVOT_KGM2, M_TOTAL_KG, R_COM_M, RATE_HZ,
    find_hanging_and_inverted_angle, load_config,
)

FPS = 30.0
WIDTH, HEIGHT = 1280, 960
SINGULARITY_COND_THRESHOLD = 1000.0

HUD_MARGIN = 20
BAR_Y = HEIGHT - 90
BAR_HEIGHT = 36
BAR_X0, BAR_X1 = HUD_MARGIN, WIDTH - HUD_MARGIN


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--damping", type=float, default=0.0001,
                    help="Hinge viscous damping override, N m s/rad (committed value: 0.02).")
    p.add_argument("--frictionloss", type=float, default=0.00005,
                    help="Hinge Coulomb frictionloss override, N m (committed value: 0.01).")
    p.add_argument("--k-e", type=float, default=100.0)
    p.add_argument("--a-max", type=float, default=3.0)
    p.add_argument("--k-pos", type=float, default=15.0)
    p.add_argument("--k-vel", type=float, default=5.0)
    p.add_argument("--duration-s", type=float, default=20.0)
    # Kick shape promoted from module constants to CLI flags 2026-08-12, same
    # reason the gains already were: they are per-experiment values, and the
    # 2026-08-12 hinge-axis/pose change made the previous hard-coded pair
    # (0.05 m / 0.58 s) wrong for this pose -- at the new ARM_Q0 a 0.05 m kick
    # trips the controller's |Y-Y0| > 0.03 m guard within ~0.3 s, so it cannot
    # even be run here. Defaults left at the old validated pair so existing
    # invocations reproduce bit-for-bit; pass explicit values for the new pose.
    p.add_argument("--kick-amplitude-m", type=float, default=0.05)
    p.add_argument("--kick-duration-s", type=float, default=0.58)
    # Camera, likewise: the previous hard-coded view was framed for the old
    # ARM_Q0 (flange at world z=0.328) and shows empty floor at the new one
    # (flange at z=0.184). Defaults updated to the new pose; the azimuth
    # default looks along the hinge axis so the swing plane is face-on.
    p.add_argument("--cam-azimuth", type=float, default=314.24)
    p.add_argument("--cam-elevation", type=float, default=-12.0)
    p.add_argument("--cam-distance", type=float, default=1.3)
    p.add_argument("--cam-lookat", type=float, nargs=3, default=[-0.45, -0.18, 0.26])
    p.add_argument("--x-axis-range-m", type=float, default=0.15,
                    help="Half-width of the HUD position bar's fixed axis, meters. "
                         "Default 0.15 gives generous margin above the 0.05m kick "
                         "amplitude plus swing-up excursion measured for the default case.")
    p.add_argument("--output", type=Path,
                    default=REPO_ROOT / "outputs" / "pendulum_renders" / "annotated_swingup_flip.mp4")
    return p.parse_args()


def x_to_pixel(x_dev: float, half_range: float) -> int:
    frac = np.clip((x_dev + half_range) / (2.0 * half_range), 0.0, 1.0)
    return int(BAR_X0 + frac * (BAR_X1 - BAR_X0))


def draw_hud(frame: np.ndarray, *, t: float, speed_mps: float, accel_mps2: float,
             ee_dev: float, min_dev: float, max_dev: float, half_range: float,
             flipped: bool, min_theta_dist: float) -> np.ndarray:
    img = Image.fromarray(frame)
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 26)
        font_small = ImageFont.truetype("DejaVuSans.ttf", 20)
    except OSError:
        font = ImageFont.load_default()
        font_small = font

    # Text panel, top-left, translucent black background for readability.
    lines = [
        f"t = {t:5.2f} s",
        f"EE speed = {speed_mps:6.3f} m/s",
        f"commanded accel = {accel_mps2:+6.3f} m/s^2",
        f"min |theta-inverted| = {min_theta_dist:.3f} rad" + ("  *** FLIPPED ***" if flipped else ""),
    ]
    panel_w, panel_h = 480, 24 + 30 * len(lines)
    draw.rectangle([HUD_MARGIN, HUD_MARGIN, HUD_MARGIN + panel_w, HUD_MARGIN + panel_h],
                   fill=(0, 0, 0, 150))
    for i, line in enumerate(lines):
        color = (255, 220, 60, 255) if "FLIPPED" in line else (255, 255, 255, 255)
        draw.text((HUD_MARGIN + 12, HUD_MARGIN + 8 + 30 * i), line, font=font_small, fill=color)

    # Position range bar: axis, red-shaded cumulative excursion band, current marker.
    draw.rectangle([BAR_X0, BAR_Y, BAR_X1, BAR_Y + BAR_HEIGHT], outline=(255, 255, 255, 200), width=2)
    center_px = x_to_pixel(0.0, half_range)
    draw.line([(center_px, BAR_Y - 8), (center_px, BAR_Y + BAR_HEIGHT + 8)], fill=(120, 120, 120, 220), width=2)
    if max_dev > min_dev:
        x0_px, x1_px = x_to_pixel(min_dev, half_range), x_to_pixel(max_dev, half_range)
        draw.rectangle([x0_px, BAR_Y, x1_px, BAR_Y + BAR_HEIGHT], fill=(220, 40, 40, 110))
    cur_px = x_to_pixel(ee_dev, half_range)
    draw.polygon([(cur_px - 9, BAR_Y - 10), (cur_px + 9, BAR_Y - 10), (cur_px, BAR_Y - 1)],
                 fill=(255, 220, 60, 255))
    draw.text((BAR_X0, BAR_Y + BAR_HEIGHT + 8),
              f"back-and-forth range so far: [{min_dev:+.3f}, {max_dev:+.3f}] m "
              f"(axis +/-{half_range:.2f}m)",
              font=font_small, fill=(255, 255, 255, 255))

    return np.array(Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB"))


def run_trial(args: argparse.Namespace) -> dict:
    config = load_config()
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

    hanging_angle, inverted_angle = find_hanging_and_inverted_angle(model, data, pend_qpos_adr)
    b_crit = 2.0 * np.sqrt(I_PIVOT_KGM2 * M_TOTAL_KG * G * R_COM_M)
    print(f"hanging={hanging_angle:.4f} rad  inverted={inverted_angle:.4f} rad  E_top={E_TOP:.5f} J")
    model.dof_damping[pend_dof_adr] = args.damping
    model.dof_frictionloss[pend_dof_adr] = args.frictionloss
    print(f"damping={args.damping} (zeta={args.damping / b_crit:.4f}), frictionloss={args.frictionloss}")

    data.qpos[:6] = ARM_Q0
    data.qpos[pend_qpos_adr] = hanging_angle
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)

    state0, adapter = build_initial_state_and_adapter(
        model, data, site_id, joint_ids, controller_cfg=config["controller"],
        transport_axis_index=0, target_x_delta=0.0, controller_kind="impedance",
        force_hold_current_pose=False,
        gravity_mode=config["mujoco"].get("gravity_mode", "gravity_comp"),
        gravity_source=config["mujoco"].get("gravity_source", "pinocchio"),
        coriolis_feedforward=bool(config["mujoco"].get("coriolis_feedforward", True)),
        torque_limit_scale=1.0,
    )
    x0 = float(state0.ee_pos[0])

    model.vis.global_.offwidth = max(int(model.vis.global_.offwidth), WIDTH)
    model.vis.global_.offheight = max(int(model.vis.global_.offheight), HEIGHT)
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(model, cam)
    cam.azimuth, cam.elevation, cam.distance = args.cam_azimuth, args.cam_elevation, args.cam_distance
    cam.lookat[:] = np.asarray(args.cam_lookat, dtype=float)
    renderer = mujoco.Renderer(model, height=HEIGHT, width=WIDTH)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.Popen([
        "ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pixel_format", "rgb24",
        "-video_size", f"{WIDTH}x{HEIGHT}", "-framerate", str(FPS), "-i", "-",
        "-pix_fmt", "yuv420p", str(args.output),
    ], stdin=subprocess.PIPE)

    n_steps = int(args.duration_s * RATE_HZ)
    frame_stride = max(1, round(1.0 / (FPS * CONTROL_DT)))
    target_x, target_x_vel = x0, 0.0
    min_theta_dist, flip_t, written = np.pi, None, 0
    min_dev, max_dev = 0.0, 0.0
    max_cond = 0.0
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))

    for step in range(n_steps):
        t = step * CONTROL_DT
        theta = float(data.qpos[pend_qpos_adr])
        thetadot = float(data.qvel[pend_dof_adr])
        phi = float(np.mod(theta - hanging_angle + np.pi, 2 * np.pi) - np.pi)
        E = 0.5 * I_PIVOT_KGM2 * thetadot * thetadot + M_TOTAL_KG * G * R_COM_M * (1.0 - np.cos(phi))

        if t < args.kick_duration_s:
            omega_kick = 2.0 * np.pi / args.kick_duration_s
            target_x = x0 + 0.5 * args.kick_amplitude_m * (1.0 - np.cos(omega_kick * t))
            target_x_vel = 0.5 * args.kick_amplitude_m * omega_kick * np.sin(omega_kick * t)
            a_cmd = float(0.5 * args.kick_amplitude_m * omega_kick ** 2 * np.cos(omega_kick * t))
        else:
            a_energy = -args.k_e * thetadot * np.cos(phi) * (E_TOP - E)
            a_recenter = -args.k_pos * (target_x - x0) - args.k_vel * target_x_vel
            a_cmd = float(np.clip(a_energy + a_recenter, -args.a_max, args.a_max))
            target_x_vel += a_cmd * CONTROL_DT
            target_x += target_x_vel * CONTROL_DT

        state = build_mujoco_state(
            model, data, site_id=site_id, joint_ids=joint_ids, time_s=t, dt_s=CONTROL_DT,
            target_x=target_x, target_x_vel=target_x_vel, target_x_accel=a_cmd,
            reference_quat=state0.ee_quat, transport_axis_index=0, gravity_compensation=True,
        )
        tau, diag = adapter.step(state=state)
        data.ctrl[:6] = np.asarray(tau, dtype=np.float64).reshape(6)
        mujoco.mj_step(model, data)

        mujoco.mj_jacSite(model, data, jacp, jacr, site_id)
        ee_vel = jacp[:, :6] @ data.qvel[:6]
        speed_mps = float(np.linalg.norm(ee_vel))
        ee_x = float(data.site_xpos[site_id][0])
        ee_dev = ee_x - x0
        min_dev, max_dev = min(min_dev, ee_dev), max(max_dev, ee_dev)
        j6 = np.vstack([jacp[:, :6], jacr[:, :6]])
        max_cond = max(max_cond, float(np.linalg.cond(j6)))

        dist = abs(float(np.mod(theta - inverted_angle + np.pi, 2 * np.pi) - np.pi))
        min_theta_dist = min(min_theta_dist, dist)
        if dist < 0.35 and flip_t is None:
            flip_t = t
            print(f"*** reached inverted at t={t:.2f}s ***")

        if step % frame_stride == 0:
            renderer.update_scene(data, camera=cam)
            frame = renderer.render()
            frame = draw_hud(
                frame, t=t, speed_mps=speed_mps, accel_mps2=a_cmd,
                ee_dev=ee_dev, min_dev=min_dev, max_dev=max_dev,
                half_range=args.x_axis_range_m, flipped=(min_theta_dist < 0.35),
                min_theta_dist=min_theta_dist,
            )
            proc.stdin.write(frame.tobytes())
            written += 1

    proc.stdin.close()
    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg exited with code {proc.returncode}")

    reached_singularity = max_cond > SINGULARITY_COND_THRESHOLD
    result = {
        "min_theta_dist_from_inverted_rad": min_theta_dist,
        "flipped": min_theta_dist < 0.35,
        "flip_t": flip_t,
        "max_cond_j": max_cond,
        "reached_singularity": reached_singularity,
        "trustworthy": (min_theta_dist < 0.35) and not reached_singularity,
        "min_dev_m": min_dev, "max_dev_m": max_dev,
    }
    print(f"min_theta_dist={min_theta_dist:.4f} flipped={result['flipped']} "
          f"max_cond_j={max_cond:.2f} reached_singularity={reached_singularity}")
    print(f"back-and-forth range: [{min_dev:+.4f}, {max_dev:+.4f}] m")
    print(f"Wrote {written} frames ({written/FPS:.1f}s) to {args.output}")
    return result


def main() -> int:
    args = parse_args()
    result = run_trial(args)
    return 0 if result["trustworthy"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
