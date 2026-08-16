#!/usr/bin/env python3
"""Renders the full swing-up -> LQR-balance trial
(pendulum_swingup_lqr_handoff.run_handoff_trial) to an HUD video plus a
companion graphs PNG.

Video and trace come from the SAME deterministic simulation pass (via
run_handoff_trial's on_frame hook) -- there is no separate re-simulation
that could silently drift from what the trace reports.

Camera: azimuth chosen by rendering several candidates and inspecting real
frames (not guessed) -- the hinge axis at OLD_POSE is close to world -Y (see
pendulum_lqr_cascade.py's measured n_axis), so an azimuth that looks along Y
would make a genuinely swinging pendulum appear frozen (this exact failure
happened earlier this session). azimuth=270 was the one, of 12 tested at
30-degree steps, that keeps the rod visibly extended (not foreshortened to a
dot) across the swing range from mid-swing through inverted.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from tools.diagnostics.pendulum_swingup_energy_shaping import (  # noqa: E402
    add_common_pendulum_args, context_from_args, describe_context,
)
from tools.diagnostics.pendulum_swingup_lqr_handoff import run_handoff_trial  # noqa: E402
from tools.diagnostics.pendulum_lqr_cascade import DEFAULT_CONFIG  # noqa: E402

FPS = 30.0
CONTROL_DT = 1.0 / 500.0
WIDTH, HEIGHT = 1280, 960
JOINT_NAMES = ["pan", "lift", "elbow", "wr1", "wr2", "wr3"]


def draw_hud(frame: np.ndarray, row: dict, *, joint_torque_limit: float) -> np.ndarray:
    img = Image.fromarray(frame)
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 22)
        font_small = ImageFont.truetype("DejaVuSans.ttf", 19)
    except OSError:
        font = ImageFont.load_default()
        font_small = font

    mode = row["mode"]
    mode_color = (80, 200, 255, 255) if mode == "swingup" else (255, 210, 60, 255)
    guard_ok = row["safety_ok"]
    qd = row["qd_max_abs"]
    lines = [
        (f"t = {row['t']:6.2f} s   CONTROLLER: {mode.upper()}", mode_color),
        (f"pendulum angle (theta) = {np.degrees(row['theta_rad']):+7.2f} deg", (255, 255, 255, 255)),
        (f"|theta - inverted| = {np.degrees(row['dist_from_inverted_rad']):6.2f} deg", (255, 255, 255, 255)),
        (f"E / E_top = {row['E_over_Etop']:.3f}", (255, 255, 255, 255)),
        (f"cond(J) = {row['cond_j']:8.2f}", (255, 255, 255, 255)),
        (f"EE drift  x={row['x_dev_m']:+.4f}  y={row['y_dev_m']:+.4f}  z={row['z_dev_m']:+.4f}  m",
         (255, 255, 255, 255)),
        (f"|qd|max = {qd:.3f} rad/s  (guard 3.0)", (255, 120, 120, 255) if qd > 2.5 else (255, 255, 255, 255)),
        (f"max|tau| = {row['max_abs_tau']:.2f} Nm (limit ~{joint_torque_limit:.0f})", (255, 255, 255, 255)),
        (f"kicks so far: {row['num_kicks']}", (200, 200, 200, 255)),
    ]
    if not guard_ok:
        lines.insert(1, ("*** SAFETY GUARD TRIPPED: " + str(row.get("safety_reason")) + " ***", (255, 60, 60, 255)))

    panel_w, panel_h = 620, 24 + 27 * len(lines)
    draw.rectangle([16, 16, 16 + panel_w, 16 + panel_h], fill=(0, 0, 0, 150))
    for i, (line, color) in enumerate(lines):
        f = font if i == 0 else font_small
        draw.text((28, 22 + 27 * i), line, font=f, fill=color)

    return np.array(Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB"))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_pendulum_args(p, default_config=DEFAULT_CONFIG)
    p.add_argument("--kick-amplitude-m", type=float, required=True)
    p.add_argument("--kick-duration-s", type=float, required=True)
    p.add_argument("--phi-trigger-rad", type=float, required=True)
    p.add_argument("--k-gains", type=float, nargs=4, required=True, metavar=("Kx", "Kxdot", "Kphi", "Kphidot"))
    p.add_argument("--a-max", type=float, required=True)
    p.add_argument("--handoff-phi-deg", type=float, required=True)
    p.add_argument("--handoff-thetadot-radps", type=float, required=True)
    p.add_argument("--handoff-confirm-steps", type=int, default=10)
    p.add_argument("--duration-s", type=float, default=15.0)
    p.add_argument("--max-kicks", type=int, default=None)
    p.add_argument("--cam-azimuth", type=float, default=270.0)
    p.add_argument("--cam-elevation", type=float, default=-12.0)
    p.add_argument("--cam-distance", type=float, default=1.4)
    p.add_argument("--video-output", type=Path,
                    default=REPO_ROOT / "outputs" / "pendulum_renders" / "pendulum_flip_and_balance.mp4")
    p.add_argument("--graphs-output", type=Path,
                    default=REPO_ROOT / "outputs" / "pendulum_renders" / "pendulum_flip_and_balance_graphs.png")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    ctx = context_from_args(args).resolve()
    print(describe_context(ctx))
    model = ctx.build_model()

    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "attachment_site")
    scratch = mujoco.MjData(model)
    scratch.qpos[:6] = ctx.arm_q_array
    mujoco.mj_forward(model, scratch)
    site_pos = scratch.site_xpos[site_id].copy()

    model.vis.global_.offwidth = max(int(model.vis.global_.offwidth), WIDTH)
    model.vis.global_.offheight = max(int(model.vis.global_.offheight), HEIGHT)
    renderer = mujoco.Renderer(model, height=HEIGHT, width=WIDTH)
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(model, cam)
    cam.azimuth, cam.elevation, cam.distance = args.cam_azimuth, args.cam_elevation, args.cam_distance
    cam.lookat[:] = site_pos + np.array([0.0, 0.0, -0.15])

    args.video_output.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.Popen([
        "ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pixel_format", "rgb24",
        "-video_size", f"{WIDTH}x{HEIGHT}", "-framerate", str(FPS), "-i", "-",
        "-pix_fmt", "yuv420p", str(args.video_output),
    ], stdin=subprocess.PIPE)

    frame_stride = max(1, round(1.0 / (FPS * CONTROL_DT)))
    written = [0]
    torque_limit = float(np.max(np.asarray([150.0, 150.0, 150.0, 28.0, 28.0, 28.0])))

    def on_frame(step, t, data, row):
        renderer.update_scene(data, camera=cam)
        frame = renderer.render()
        frame = draw_hud(frame, row, joint_torque_limit=torque_limit)
        proc.stdin.write(frame.tobytes())
        written[0] += 1

    result = run_handoff_trial(
        model,
        kick_amplitude_m=args.kick_amplitude_m, kick_duration_s=args.kick_duration_s,
        phi_trigger_rad=args.phi_trigger_rad,
        K=np.asarray(args.k_gains, dtype=np.float64), a_max=args.a_max,
        handoff_phi_rad=np.radians(args.handoff_phi_deg),
        handoff_thetadot_radps=args.handoff_thetadot_radps,
        handoff_confirm_steps=args.handoff_confirm_steps,
        duration_s=args.duration_s, hanging_angle=ctx.hanging_angle, inverted_angle=ctx.inverted_angle,
        constants=ctx.constants, config_path=Path(ctx.config_path), controller_kind=ctx.controller_kind,
        arm_q=ctx.arm_q_array, max_kicks=args.max_kicks,
        on_frame=on_frame, frame_stride=frame_stride,
    )
    proc.stdin.close()
    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg exited with code {proc.returncode}")
    print({k: v for k, v in result.items() if k != "trace"})
    print(f"wrote {written[0]} frames ({written[0]/FPS:.1f}s) to {args.video_output}")

    make_graphs(result["trace"], args.graphs_output, ctx, result.get("handoff_t"))
    print("wrote", args.graphs_output)
    return 0 if result["flipped"] else 1


def make_graphs(trace: list[dict], output_path: Path, ctx, handoff_t):
    t = np.array([r["t"] for r in trace])
    theta_deg = np.degrees(np.array([r["theta_rad"] for r in trace]))
    thetadot = np.array([r["thetadot_radps"] for r in trace])
    E = np.array([r["E_over_Etop"] for r in trace])
    condj = np.array([r["cond_j"] for r in trace])
    qd = np.array([r["qd_max_abs"] for r in trace])
    tau = np.array([r["tau_applied_clipped"] for r in trace])
    mode = np.array([r["mode"] for r in trace])
    hanging_deg = np.degrees(ctx.hanging_angle)
    inverted_deg = np.degrees(ctx.inverted_angle)

    fig, axes = plt.subplots(3, 2, figsize=(15, 12))

    ax = axes[0, 0]
    ax.plot(t, theta_deg, color="tab:blue", lw=1.2)
    ax.axhline(hanging_deg, color="gray", ls="--", lw=1, label="hanging")
    ax.axhline(inverted_deg, color="green", ls="--", lw=1, label="inverted")
    if handoff_t is not None:
        ax.axvline(handoff_t, color="red", ls=":", lw=1.5, label="handoff")
    ax.set_xlabel("t (s)"); ax.set_ylabel("theta (deg)"); ax.set_title("Pendulum angle")
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    ax.plot(t, E, color="tab:orange", lw=1.2)
    ax.axhline(1.0, color="green", ls="--", lw=1, label="E_top")
    if handoff_t is not None:
        ax.axvline(handoff_t, color="red", ls=":", lw=1.5)
    ax.set_xlabel("t (s)"); ax.set_ylabel("E / E_top"); ax.set_title("Pendulum energy")
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    ax.plot(t, condj, color="tab:purple", lw=1.0)
    if handoff_t is not None:
        ax.axvline(handoff_t, color="red", ls=":", lw=1.5)
    ax.set_xlabel("t (s)"); ax.set_ylabel("cond(J)"); ax.set_title("Jacobian conditioning")
    ax.set_yscale("log")

    ax = axes[1, 1]
    ax.plot(t, qd, color="tab:red", lw=1.0)
    ax.axhline(3.0, color="black", ls="--", lw=1, label="guard 3.0 rad/s")
    if handoff_t is not None:
        ax.axvline(handoff_t, color="red", ls=":", lw=1.5)
    ax.set_xlabel("t (s)"); ax.set_ylabel("max |qd| (rad/s)"); ax.set_title("Joint velocity vs guard")
    ax.legend(fontsize=8)

    ax = axes[2, 0]
    for j in range(6):
        ax.plot(t, tau[:, j], lw=0.8, label=JOINT_NAMES[j])
    if handoff_t is not None:
        ax.axvline(handoff_t, color="red", ls=":", lw=1.5)
    ax.set_xlabel("t (s)"); ax.set_ylabel("tau (Nm)"); ax.set_title("Per-joint applied torque")
    ax.legend(fontsize=7, ncol=3)

    ax = axes[2, 1]
    phi_from_inv_deg = np.array([r["phi_inv_deg"] for r in trace])
    swing = mode == "swingup"
    bal = mode == "balance"
    ax.plot(phi_from_inv_deg[swing], thetadot[swing], color="tab:blue", lw=0.6, alpha=0.6, label="swingup")
    ax.plot(phi_from_inv_deg[bal], thetadot[bal], color="tab:orange", lw=1.0, label="balance")
    ax.axvline(0, color="green", ls="--", lw=1)
    ax.axhline(0, color="green", ls="--", lw=1)
    ax.set_xlabel("phi from inverted (deg)"); ax.set_ylabel("thetadot (rad/s)")
    ax.set_title("Phase portrait")
    ax.legend(fontsize=8)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
