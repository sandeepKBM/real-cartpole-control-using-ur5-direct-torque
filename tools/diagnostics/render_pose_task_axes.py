"""Render a pose with its tool axes, the pendulum hinge, and the PUMPING
ALIGNMENT (kappa) of every candidate drive direction.

RUN THIS BEFORE ANY SWING-UP OR BALANCE RUN, AND LOOK AT IT.

Why this is mandatory rather than nice-to-have. The drive direction is one
scalar axis, and the whole run is worthless if it points somewhere the
pendulum cannot feel. That has now happened twice in this repo:

  * A tool-frame run put the drive on row 0 = tool X, which at ARM_Q0 is 7.3
    deg from VERTICAL. Vertical pivot acceleration exerts ZERO hinge torque at
    the hanging equilibrium, so the law drove an axis with no authority over
    the pole, dumped it into Z, and tripped the corridor guard in 0.134 s with
    the rod tip 4 mm off the floor.
  * Every world-X run at ARM_Q0 spends 69.8% of its motion ALONG the hinge.

Neither failure is visible in a config, a gain, or a log line. Both are obvious
in one picture.

kappa is the fraction of a unit drive direction that lies PERPENDICULAR to the
hinge -- i.e. the part that actually pumps:

    kappa(u) = sqrt(1 - (u . h)^2),   h = hinge axis in world

kappa = 1 is ideal, kappa = 0 is a direction the pendulum cannot feel at all.
It is computed from the compiled model every time, never hardcoded.

  MUJOCO_GL=egl python tools/diagnostics/render_pose_task_axes.py \\
      --pendulum-xml assets/ur5e_pendulum/pendulum_attachment.xml \\
      --output outputs/pose_renders/ARM_Q0_task_axes.png
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

from simulation.ur5e_pendulum_compose import (  # noqa: E402
    DEFAULT_PENDULUM_XML,
    arm_q_for_pendulum_xml,
    compose_ur5e_pendulum_model,
)
from tools.diagnostics._hud_font import draw_text  # noqa: E402

WIDTH, HEIGHT = 1600, 1200


def axis_report(model, data, site_id) -> dict:
    """Tool axes, hinge axis, and kappa for every candidate drive direction."""
    rot = data.site_xmat[site_id].reshape(3, 3)
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "/pendulum_hinge")
    if jid < 0:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "pendulum_hinge")
    hinge = data.xmat[model.jnt_bodyid[jid]].reshape(3, 3) @ model.jnt_axis[jid]
    hinge = hinge / np.linalg.norm(hinge)

    def kappa(u):
        """Fraction NOT wasted along the hinge. Necessary, NOT sufficient."""
        u = np.asarray(u, dtype=np.float64)
        u = u / np.linalg.norm(u)
        return float(np.sqrt(max(0.0, 1.0 - float(u @ hinge) ** 2)))

    # kappa alone is NOT enough, and mistaking it for enough is what broke a
    # real run: with a horizontal hinge, the ENTIRE vertical plane perpendicular
    # to it scores kappa = 1.0, so tool X (vertical) and tool Y (horizontal)
    # are indistinguishable by kappa -- yet driving the vertical one produces
    # zero hinge torque at the HANGING equilibrium and the swing-up cannot even
    # start. The generalized force a pivot acceleration exerts is
    # Q = -m*r*(a . n_hat) with n_hat perpendicular to the ROD; at hanging the
    # rod points down, so n_hat is HORIZONTAL. kappa_hang is that projection --
    # the authority available at t=0, which is the number that decides whether a
    # swing-up can bootstrap at all.
    n_hang = np.array([-hinge[1], hinge[0], 0.0])
    n_hang = n_hang / (np.linalg.norm(n_hang) + 1e-12)

    def kappa_hang(u):
        u = np.asarray(u, dtype=np.float64)
        u = u / np.linalg.norm(u)
        return float(abs(u @ n_hang))

    # The in-plane horizontal: the hinge projected out of the horizontal plane.
    # With shoulder_pan frozen this is the only horizontal direction the arm can
    # produce, and it is perpendicular to the hinge by construction.
    e_h = np.array([-hinge[1], hinge[0], 0.0])
    e_h = e_h / (np.linalg.norm(e_h) + 1e-12)

    cands = [
        ("tool X", rot[:, 0]), ("tool Y", rot[:, 1]), ("tool Z", rot[:, 2]),
        ("world X", [1.0, 0, 0]), ("world Y", [0, 1.0, 0]), ("world Z", [0, 0, 1.0]),
        ("in-plane horiz", e_h),
    ]
    return {
        "hinge": hinge,
        "tool": rot,
        "e_h": e_h,
        "candidates": [(n, np.asarray(v, dtype=np.float64), kappa(v), kappa_hang(v))
                       for n, v in cands],
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pendulum-xml", type=Path, default=DEFAULT_PENDULUM_XML)
    ap.add_argument("--start-q-rad", type=float, nargs=6, default=None,
                    help="Default: the pose registered for --pendulum-xml.")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--azimuth", type=float, default=125.0)
    ap.add_argument("--elevation", type=float, default=-18.0)
    ap.add_argument("--distance", type=float, default=1.9)
    ap.add_argument("--lookat", type=float, nargs=3, default=[-0.30, -0.20, 0.40])
    args = ap.parse_args(argv)

    model = compose_ur5e_pendulum_model(pendulum_xml=str(args.pendulum_xml))
    q = (np.asarray(args.start_q_rad, dtype=np.float64) if args.start_q_rad is not None
         else arm_q_for_pendulum_xml(str(args.pendulum_xml)))
    data = mujoco.MjData(model)
    data.qpos[:6] = q
    mujoco.mj_forward(model, data)
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "attachment_site")
    rep = axis_report(model, data, site_id)

    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    mujoco.mj_jacSite(model, data, jacp, jacr, site_id)
    cond = float(np.linalg.cond(np.vstack([jacp[:, :6], jacr[:, :6]])))

    print(f"pose        = {[round(float(v), 6) for v in q]}")
    print(f"asset       = {Path(args.pendulum_xml).name}")
    print(f"cond(J6)    = {cond:.2f}")
    print(f"hinge axis  = {np.round(rep['hinge'], 4)}")
    print("\n  kappa      = not wasted along the hinge   (necessary, NOT sufficient)")
    print("  kappa_hang = authority AT THE HANGING START -- this is the one that")
    print("               decides whether a swing-up can bootstrap at all\n")
    print("  candidate drive direction        world vector           kappa  k_hang  verdict")
    # Rank by kappa_hang: a direction that cannot start the swing-up is useless
    # however well it scores on kappa.
    best = max(rep["candidates"], key=lambda c: c[3])
    for name, vec, k, kh in rep["candidates"]:
        v = vec / np.linalg.norm(vec)
        verdict = ("IDEAL" if k > 0.999 and kh > 0.99 else
                   "DEAD -- along the hinge" if k < 0.05 else
                   "NO AUTHORITY AT HANGING (vertical)" if kh < 0.2 else
                   f"{100 * (1 - kh):.0f}% short at hanging")
        star = " <==" if name == best[0] else ""
        print(f"  {name:<24} {np.round(v, 4)!s:<23} {k:6.4f} {kh:6.4f}  {verdict}{star}")

    model.vis.global_.offwidth = max(int(model.vis.global_.offwidth), WIDTH)
    model.vis.global_.offheight = max(int(model.vis.global_.offheight), HEIGHT)
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(model, cam)
    cam.azimuth, cam.elevation, cam.distance = args.azimuth, args.elevation, args.distance
    cam.lookat[:] = np.asarray(args.lookat, dtype=np.float64)
    renderer = mujoco.Renderer(model, height=HEIGHT, width=WIDTH)
    renderer.update_scene(data, camera=cam)
    frame = np.ascontiguousarray(renderer.render())

    lines = [
        f"POSE {[round(float(v), 4) for v in q]}",
        f"ASSET {Path(args.pendulum_xml).name}   COND(J) {cond:.1f}",
        f"HINGE AXIS (WORLD) {np.round(rep['hinge'], 3)}",
        "",
        "KAPPA = NOT WASTED ALONG HINGE.  K_HANG = AUTHORITY AT THE HANGING START.",
        "BOTH MUST BE ~1. KAPPA ALONE IS NOT ENOUGH -- VERTICAL SCORES KAPPA 1 AND CANNOT PUMP.",
    ]
    colors = [(255, 255, 255)] * 3 + [(0, 0, 0)] + [(255, 220, 80), (255, 220, 80)]
    for name, vec, k, kh in rep["candidates"]:
        v = vec / np.linalg.norm(vec)
        lines.append(f"  {name:<16} {v[0]:+.3f} {v[1]:+.3f} {v[2]:+.3f}  KAPPA {k:.3f}  K_HANG {kh:.3f}"
                     + ("  <== BEST" if name == best[0] else ""))
        colors.append((80, 255, 120) if (k > 0.999 and kh > 0.99) else
                      (255, 90, 90) if (k < 0.05 or kh < 0.2) else (255, 255, 255))
    for i, (line, col) in enumerate(zip(lines, colors)):
        if line:
            draw_text(frame, line, x=14, y=14 + i * 30, color=col, scale=3)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pixel_format", "rgb24",
         "-video_size", f"{WIDTH}x{HEIGHT}", "-i", "-", "-frames:v", "1", str(args.output)],
        input=frame.tobytes(), check=False)
    if proc.returncode != 0:
        raise SystemExit("ffmpeg failed writing the PNG")
    print(f"\nwrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
