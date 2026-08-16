"""Render the 2-axis (x_task_yz_corridor_qp) controller actually moving, with
the pendulum attached and a HUD carrying the numbers that matter.

Why this exists: the two existing clips
(outputs/pendulum_renders/x_task_yz_corridor_qp_{operating,limits_demo}_2026-08-13.mp4)
show the BARE ARM, and the limits demo's segments are mostly failure cases. Neither
shows the controller carrying the pendulum through a clean move.

Pipeline is deliberately the same one the sim check uses
(build_initial_state_and_adapter / build_mujoco_state / adapter.step), so what is
rendered is the real closed loop and not a kinematic replay -- the arm here is
driven by torques the controller actually produced.

BOTH DIRECTIONS are rendered (+dx and -dx). X-direction asymmetry is a real,
repeatedly-confirmed phenomenon in this repo, so a one-way clip can misrepresent
the safe range (AGENTS.md sec 7).

Guards stay ON. A trip is rendered and reported, never suppressed.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import mujoco  # noqa: E402
import yaml  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

from simulation.ur5e_pendulum_compose import (  # noqa: E402
    compose_ur5e_pendulum_model,
    DEFAULT_ARM_Q,
)
from simulation.ur5e_mujoco_torque import (  # noqa: E402
    build_initial_state_and_adapter,
    build_mujoco_state,
    x_profile_target,
)
from tools.diagnostics.pendulum_swingup_energy_shaping import (  # noqa: E402
    resolve_equilibria,
)

DEFAULT_CONFIG = "config/ur5e_mujoco_torque_x_task_yz_corridor_qp_enabled.yaml"
DEFAULT_PENDULUM = "assets/ur5e_pendulum/pendulum_attachment.xml"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=DEFAULT_CONFIG)
    p.add_argument("--pendulum-xml", default=DEFAULT_PENDULUM,
                   help="Pendulum attachment. The DEFAULT (local-Z hinge) is the live "
                        "one at ARM_Q0; the realrod/longrod (local-X hinge) assets are "
                        "DEAD there -- see AGENTS.md sec 0.")
    p.add_argument("--start-q-rad", nargs=6, type=float, default=None,
                   help="Arm start pose. Default: DEFAULT_ARM_Q (ARM_Q0).")
    p.add_argument("--controller-kind", default="x_task_yz_corridor_qp")
    p.add_argument("--no-pendulum", action="store_true",
                   help="Load the BARE arm scene instead of composing the pendulum. "
                        "Control condition: the controller's published envelope was "
                        "measured without the pendulum, so this isolates whether an "
                        "observed failure is caused by the added tool mass/reaction "
                        "torque or by the controller itself.")
    p.add_argument("--deltas", nargs="+", type=float, default=[0.12, -0.12],
                   help="Task-axis displacements to render, in order. Both signs by "
                        "default -- see module docstring.")
    p.add_argument("--move-duration-s", type=float, default=2.0)
    p.add_argument("--hold-duration-s", type=float, default=1.5)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--width", type=int, default=960)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--out", default="outputs/pendulum_renders/two_axis_controller_demo.mp4")
    return p.parse_args()


def load_cfg(path: str) -> dict:
    p = Path(path)
    if not p.is_absolute():
        p = REPO_ROOT / p
    with open(p, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def run_segment(model, arm_q, *, ctrl_cfg, mj_cfg, controller_kind,
                target_delta_m, move_duration_s, hold_duration_s):
    """One closed-loop move+hold. Returns per-step records."""
    data = mujoco.MjData(model)
    data.qpos[:] = 0.0
    data.qvel[:] = 0.0
    data.qpos[:6] = arm_q

    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "attachment_site")
    joint_ids = list(range(6))
    hinge_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "/pendulum_hinge")
    hinge_qadr = int(model.jnt_qposadr[hinge_jid]) if hinge_jid >= 0 else None

    # START THE PENDULUM AT ITS HANGING EQUILIBRIUM, not at qpos=0. Leaving the
    # hinge at 0 starts the rod cocked; it then falls and whips the arm, and that
    # disturbance would be misread as controller error.
    # Solved analytically rather than by settling: hinge damping is 1e-4, so the
    # pendulum does not settle in any reasonable number of steps (it keeps
    # swinging/spinning), which makes a settle-based seed unreliable.
    if hinge_qadr is not None:
        hanging, _inverted = resolve_equilibria(model, arm_q)
        data.qpos[hinge_qadr] = float(hanging)
    mujoco.mj_forward(model, data)

    state0, adapter = build_initial_state_and_adapter(
        model, data, site_id, joint_ids,
        controller_cfg=ctrl_cfg,
        transport_axis_index=0,
        target_x_delta=float(target_delta_m),
        controller_kind=str(controller_kind),
        force_hold_current_pose=False,
        gravity_mode=str(mj_cfg.get("gravity_mode", "gravity_comp")),
        gravity_source="mujoco_qfrc",
        coriolis_feedforward=False,
        torque_limit_scale=1.0,
    )

    start_ee = np.asarray(state0.ee_pos, dtype=np.float64).copy()
    x0 = float(start_ee[0])
    dt = float(model.opt.timestep)
    duration_s = float(move_duration_s) + float(hold_duration_s)
    steps = max(1, int(np.ceil(duration_s / max(dt, 1e-9))))
    gravity_scratch = mujoco.MjData(model)

    recs = []
    trip = None
    for _ in range(steps):
        t_s = float(data.time)
        target_now, target_vel_now = x_profile_target(
            "min_jerk_move_hold", x0, float(target_delta_m), t_s, duration_s,
            move_duration_s=float(move_duration_s),
        )
        target_ee_pos = start_ee.copy(); target_ee_pos[0] = target_now
        target_ee_vel = np.zeros(3); target_ee_vel[0] = target_vel_now

        state = build_mujoco_state(
            model, data, site_id=site_id, joint_ids=joint_ids, time_s=t_s, dt_s=dt,
            target_x=float(target_now), target_x_vel=float(target_vel_now),
            target_x_accel=0.0,
            target_axis=float(target_now), target_axis_vel=float(target_vel_now),
            target_ee_pos=target_ee_pos, target_ee_vel=target_ee_vel,
            reference_quat=state0.reference_quat, hold_current_pose=False,
            transport_axis_index=0,
            gravity_compensation=bool(str(mj_cfg.get("gravity_mode", "gravity_comp")) == "gravity_comp"),
            gravity_scratch_data=gravity_scratch,
        )
        tau, diag = adapter.step(state=state)
        if not diag.get("safety_ok", True) and trip is None:
            trip = (t_s, str(diag.get("safety_reason", "?")))

        ee = np.asarray(state.ee_pos, dtype=np.float64)
        jac = np.asarray(state.jacobian, dtype=np.float64)
        out = diag.get("controller_output", {}) or {}
        rows = out.get("yz_corridor_active_rows")
        recs.append(dict(
            t=t_s,
            q=np.asarray(data.qpos[:6]).copy(),
            hinge=float(data.qpos[hinge_qadr]) if hinge_qadr is not None else 0.0,
            dx=float(ee[0] - start_ee[0]),
            dy=float(ee[1] - start_ee[1]),
            dz=float(ee[2] - start_ee[2]),
            tgt=float(target_now - x0),
            cond=float(np.linalg.cond(jac)),
            qd=float(np.max(np.abs(np.asarray(state.qd)))),
            tau=float(np.max(np.abs(np.asarray(tau)))),
            corridor=bool(rows is not None and any(bool(r) for r in rows)),
            cbf=bool(out.get("manipulability_cbf_active", False)),
            ok=bool(diag.get("safety_ok", True)),
        ))
        # Apply the controller's torque BEFORE stepping. Omitting this steps the
        # sim at zero torque, so the arm just falls under gravity -- which looks
        # like a controller failure but is identical with/without the pendulum and
        # identical for +dx and -dx, because nothing is being commanded at all.
        data.ctrl[:6] = np.asarray(tau, dtype=np.float64).reshape(6)
        mujoco.mj_step(model, data)

    return recs, trip, start_ee


def main() -> int:
    a = parse_args()
    cfg = load_cfg(a.config)
    ctrl_cfg = dict(cfg["controller"])
    mj_cfg = cfg.get("mujoco", {}) or {}
    arm_q = (np.asarray(a.start_q_rad, dtype=np.float64)
             if a.start_q_rad is not None else np.asarray(DEFAULT_ARM_Q, dtype=np.float64))

    xml = a.pendulum_xml
    if not Path(xml).is_absolute():
        xml = str(REPO_ROOT / xml)
    if a.no_pendulum:
        model = mujoco.MjModel.from_xml_path(str(REPO_ROOT / "assets/ur5e_torque/scene.xml"))
    else:
        model = compose_ur5e_pendulum_model(pendulum_xml=xml)
    model.vis.global_.offwidth = max(int(model.vis.global_.offwidth), a.width)
    model.vis.global_.offheight = max(int(model.vis.global_.offheight), a.height)

    segments = []
    for d in a.deltas:
        print(f"[run] dx = {d:+.3f} m ...", flush=True)
        recs, trip, start_ee = run_segment(
            model, arm_q, ctrl_cfg=ctrl_cfg, mj_cfg=mj_cfg,
            controller_kind=a.controller_kind, target_delta_m=float(d),
            move_duration_s=a.move_duration_s, hold_duration_s=a.hold_duration_s)
        maxy = max(abs(r["dy"]) for r in recs)
        maxz = max(abs(r["dz"]) for r in recs)
        print(f"      achieved {recs[-1]['dx']:+.4f} m of {d:+.3f}  "
              f"({100*recs[-1]['dx']/d:.1f}%)  maxY {maxy:.4f}  maxZ {maxz:.4f}  "
              f"corridor {sum(r['corridor'] for r in recs)} steps  "
              f"cbf {sum(r['cbf'] for r in recs)} steps  "
              f"trip {trip if trip else 'NONE'}", flush=True)
        segments.append((float(d), recs, trip))

    # ---------------- render ----------------
    renderer = mujoco.Renderer(model, height=a.height, width=a.width)
    cam = mujoco.MjvCamera()
    cam.azimuth, cam.elevation, cam.distance = 135.0, -12.0, 1.9
    rdata = mujoco.MjData(model)
    rdata.qpos[:6] = arm_q
    mujoco.mj_forward(model, rdata)
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "attachment_site")
    cam.lookat[:] = np.asarray(rdata.site_xpos[sid]) + np.array([0.0, 0.0, -0.10])

    try:
        F = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 17)
        Fs = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 15)
    except Exception:
        F = Fs = ImageFont.load_default()

    hinge_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "/pendulum_hinge")
    hinge_qadr = int(model.jnt_qposadr[hinge_jid])

    out_path = Path(a.out)
    if not out_path.is_absolute():
        out_path = REPO_ROOT / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Frames -> PNG -> ffmpeg binary. imageio's ffmpeg/pyav BACKENDS are not
    # installed in this env (imageio.get_writer raises "Could not find a backend"),
    # but the ffmpeg BINARY is on PATH, so shell out rather than add a dependency.
    frame_dir = Path(tempfile.mkdtemp(prefix="two_axis_frames_"))

    n_frames = 0
    for d, recs, trip in segments:
        next_t = 0.0
        for i, r in enumerate(recs):
            if r["t"] < next_t and i != len(recs) - 1:
                continue
            next_t += 1.0 / a.fps
            rdata.qpos[:6] = r["q"]
            if hinge_qadr is not None:
                rdata.qpos[hinge_qadr] = r["hinge"]
            rdata.qvel[:] = 0.0
            mujoco.mj_forward(model, rdata)
            renderer.update_scene(rdata, camera=cam)
            pil = Image.fromarray(renderer.render())
            dr = ImageDraw.Draw(pil)
            status = "GUARD TRIP" if not r["ok"] else "guards OK"
            col = (255, 90, 90) if not r["ok"] else (140, 255, 140)
            lines = [
                (f"2-AXIS CONTROLLER  ({a.controller_kind})  dx = {d:+.3f} m", (120, 255, 120)),
                (f"t = {r['t']:5.2f} s     target {r['tgt']:+.4f}   achieved {r['dx']:+.4f} m", (255, 255, 120)),
                (f"off-axis   Y {r['dy']*1000:+6.1f} mm   Z {r['dz']*1000:+6.1f} mm", (255, 255, 120)),
                (f"cond(J) {r['cond']:8.1f}   max|qd| {r['qd']:.3f} rad/s   max tau {r['tau']:5.1f} Nm", (255, 255, 120)),
                (f"corridor {'ACTIVE' if r['corridor'] else '  --  '}   CBF {'ACTIVE' if r['cbf'] else '  --  '}   pendulum {np.degrees(r['hinge']):+7.1f} deg", (200, 220, 255)),
                (status, col),
            ]
            for j, (s, c) in enumerate(lines):
                dr.text((11, 11 + 20 * j), s, fill=(0, 0, 0), font=F if j == 0 else Fs)
                dr.text((10, 10 + 20 * j), s, fill=c, font=F if j == 0 else Fs)
            pil.save(frame_dir / f"f_{n_frames:05d}.png")
            n_frames += 1

    if n_frames == 0:
        print("[error] no frames rendered")
        return 1
    cmd = ["ffmpeg", "-loglevel", "error", "-y", "-framerate", str(a.fps),
           "-i", str(frame_dir / "f_%05d.png"),
           "-c:v", "libx264", "-pix_fmt", "yuv420p", str(out_path)]
    rc = subprocess.run(cmd).returncode
    shutil.rmtree(frame_dir, ignore_errors=True)
    if rc != 0 or not out_path.exists():
        print(f"[error] ffmpeg failed rc={rc}")
        return 1
    print(f"[done] wrote {out_path}  ({n_frames} frames, {out_path.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
