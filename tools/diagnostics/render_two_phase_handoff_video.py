#!/usr/bin/env python3
"""Render the energy-scheduled swing-up -> LQR handoff as an MP4 with a HUD.

Needs MUJOCO_GL=egl on this headless host.

WHY THE HUD CARRIES THESE FIELDS. AGENTS.md's standing rule is that an MP4 is
not evidence -- a video of a pole standing up looks identical whether the run
was guard-clean or tripped three guards on the way. So every frame stamps the
quantities a reviewer would otherwise have to take on trust:

  phi        angle from INVERTED (the thing being controlled)
  thetadot   angular rate -- distinguishes a hold from a fly-through
  s          thetadot + omega*phi, the unstable mode the switch fires on
  orient     ||orientation error||, WITH its guard threshold, because that is
             what actually limits this run
  phase      SWING-UP vs LQR, so the handoff instant is visible rather than
             inferred
  GUARD      turns red and stays red from the first trip onward

A frame can then be checked against any claim made about the run.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

# The model's offscreen framebuffer is 640x480 by default; rendering larger
# needs <visual><global offwidth=.../></visual> in the MJCF. Not worth
# editing a shared asset for a diagnostic video.
WIDTH, HEIGHT = 640, 480


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pendulum-xml", default="assets/ur5e_pendulum/pendulum_attachment_realrod.xml")
    p.add_argument("--start-q-rad", type=float, nargs=6,
                   default=[-2.3688, -2.1801, -1.8838, -0.7962, -1.5707963, 0.0206])
    p.add_argument("--config", type=Path,
                   default=Path("config/ur5e_mujoco_torque_osc_tuned_friction_ff_balance_drift06.yaml"))
    p.add_argument("--controller-kind", default="impedance")
    p.add_argument("--schedule", type=float, nargs=7, required=True,
                   metavar=("A_SLOW", "A_SHARP", "E_CENTER", "E_WIDTH",
                            "DB_SLOW", "DB_SHARP", "E_TARGET"))
    p.add_argument("--lqr-json", type=Path, required=True)
    p.add_argument("--duration-s", type=float, default=16.0)
    p.add_argument("--hold-s", type=float, default=6.0)
    p.add_argument("--phi-switch-max-rad", type=float, default=0.32)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--azimuth", type=float, default=100.0)
    p.add_argument("--elevation", type=float, default=-15.0)
    p.add_argument("--distance", type=float, default=1.4)
    p.add_argument("--lookat", type=float, nargs=3, default=[-0.30, -0.20, 0.45])
    p.add_argument("--velocity-swingup", action="store_true",
                   help="Must MATCH the run being filmed. The renderer re-runs the\n"
                        "trial to capture history, so a mismatch here films a\n"
                        "different rollout than the numbers came from.")
    p.add_argument("--allow-pose-mismatch", action="store_true",
                   help="Render a config off its declared pose/asset/controller. "
                        "Only for a deliberate cross-pose comparison -- the video "
                        "then stamps OFF-PROVENANCE on every frame.")
    p.add_argument("--out", type=Path, required=True)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    import cv2
    import imageio.v2 as imageio
    import mujoco
    from tools.diagnostics.pendulum_two_phase_swingup import (
        EnergyScheduleParams, run_energy_scheduled_trial, _model_for)
    from tools.diagnostics.pendulum_swingup_energy_shaping import (
        resolve_equilibria, measure_pivot_coupling, RATE_HZ)
    from simulation.ur5e_pendulum_compose import derive_pendulum_constants

    from controller_core.config_provenance import (
        check_config_pose, describe_provenance)
    from tools.diagnostics.pendulum_swingup_energy_shaping import load_config

    arm_q = np.asarray(args.start_q_rad, dtype=np.float64)

    # The renderer was the one dispatch point with no provenance check, and it
    # produces the artifact a reviewer is most likely to trust on sight. A
    # video rendered from a config derived at another pose is exactly the
    # failure this guard exists for, only harder to notice.
    provenance = check_config_pose(
        load_config(Path(args.config)), arm_q, args.pendulum_xml,
        controller_kind=str(args.controller_kind),
        config_name=Path(args.config).name,
        allow_mismatch=bool(args.allow_pose_mismatch),
    )
    print(describe_provenance(provenance))

    model = _model_for(str(args.pendulum_xml))
    hanging, inverted = resolve_equilibria(model, arm_q)
    constants = derive_pendulum_constants(model, arm_q)
    # Same drive-axis resolution as the run tool -- a video rendered with a
    # different c0 than the run it claims to show is not that run.
    _rot_cfg = (load_config(Path(args.config)).get("controller") or {}).get("task_rotation")
    if _rot_cfg is not None:
        axis = np.asarray(_rot_cfg, dtype=np.float64).reshape(3, 3)[:, 0]
    else:
        axis = np.zeros(3)
        axis[0] = 1.0
    c0 = float(measure_pivot_coupling(model, arm_q, hanging, axis))
    lqr = json.loads(Path(args.lqr_json).read_text())["lqr"]

    s = args.schedule
    params = EnergyScheduleParams(a_slow=s[0], a_sharp=s[1], e_center=s[2], e_width=s[3],
                                  db_slow=s[4], db_sharp=s[5], e_target=s[6])

    # Re-run WITH history, then replay it. Replaying the same trajectory the
    # metrics came from is what makes the video and the numbers the same run.
    out = run_energy_scheduled_trial(
        model, params, arm_q=arm_q, hanging_angle=hanging, inverted_angle=inverted,
        constants=constants, coupling_c0=c0, config_path=Path(args.config),
        controller_kind=str(args.controller_kind), duration_s=float(args.duration_s),
        track_history=True, velocity_swingup=bool(args.velocity_swingup),
        lqr_K=np.asarray(lqr["K"], dtype=np.float64),
        lqr_a_max=float(lqr["a_max"]), phi_switch_max_rad=float(args.phi_switch_max_rad),
        hold_s=float(args.hold_s))
    hist = out["history"]
    if not hist:
        print("no history recorded")
        return 2
    guard_t = out["first_guard_t"]
    switch_t = out["lqr_engaged_t"]
    print(f"switch={switch_t}  guard={guard_t}  frames={len(hist)}")

    # Replay: re-simulate and capture frames at the requested fps.
    data = mujoco.MjData(model)
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "attachment_site")
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "/pendulum_hinge")
    pend_qpos = model.jnt_qposadr[jid]

    cam = mujoco.MjvCamera()
    cam.azimuth, cam.elevation, cam.distance = args.azimuth, args.elevation, args.distance
    cam.lookat[:] = np.asarray(args.lookat, dtype=np.float64)
    renderer = mujoco.Renderer(model, height=HEIGHT, width=WIDTH)

    stride = max(1, int(round(RATE_HZ / float(args.fps))))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(str(args.out), fps=int(args.fps), macro_block_size=1)

    # The recorded history holds the pendulum angle per step; replay it onto the
    # model so the rendered pose IS the simulated one.
    ORIENT_LIMIT = 0.25
    guarded = False
    for i, row in enumerate(hist):
        if i % stride:
            continue
        data.qpos[:6] = np.asarray(row["qpos6"], dtype=np.float64)
        data.qpos[pend_qpos] = inverted + np.radians(row["phi_inv_deg"])
        mujoco.mj_forward(model, data)
        renderer.update_scene(data, camera=cam)
        frame = renderer.render().copy()

        if guard_t is not None and row["t"] >= guard_t:
            guarded = True
        lines = [
            f"t = {row['t']:6.3f} s     phase: {row['phase'].upper()}",
            f"phi      = {row['phi_inv_deg']:+8.2f} deg  (from inverted)",
            f"thetadot = {row['thetadot']:+8.3f} rad/s",
            f"s        = {row['s']:+8.3f}   (switch band |s| <= 1.2)",
            f"orient   = {row['orientation_error']:8.4f} / {ORIENT_LIMIT:.2f} rad guard",
            f"E/E_top  = {row['energy_over_e_top']:8.4f}",
            f"drive    = {'VELOCITY' if args.velocity_swingup else 'POSITION'}-tracked row",
        ]
        y = 22
        for ln in lines:
            cv2.putText(frame, ln, (14, y), cv2.FONT_HERSHEY_SIMPLEX, 0.44,
                        (255, 255, 255), 2, cv2.LINE_AA)
            y += 21
        tag, col = ("GUARD TRIPPED", (0, 0, 255)) if guarded else ("guards clean", (0, 200, 0))
        cv2.putText(frame, tag, (14, y + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 2, cv2.LINE_AA)
        if provenance.mismatches:
            cv2.putText(frame, "OFF-PROVENANCE", (14, y + 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2, cv2.LINE_AA)
        if switch_t is not None and abs(row["t"] - switch_t) < (stride / RATE_HZ):
            cv2.putText(frame, "<<< LQR HANDOFF", (WIDTH - 250, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 255), 2, cv2.LINE_AA)
        writer.append_data(frame)
    writer.close()
    renderer.close()
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
