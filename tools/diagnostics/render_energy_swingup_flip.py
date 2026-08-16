#!/usr/bin/env python3
"""Render a pendulum swing-up attempt using the sign-corrected energy-shaping
law (pendulum_swingup_energy_shaping.py, fixed 2026-08-12), at a CLI-supplied
hinge damping/frictionloss -- CLI-configurable 2026-08-12 (was file-edit-
per-experiment before this; see AGENTS.md/session history on why that's the
wrong pattern for this repo's diagnostic scripts).

READ THIS BEFORE SHOWING ANY RENDER FROM THIS SCRIPT TO ANYONE
================================================================
The committed hinge damping (assets/ur5e_pendulum/pendulum_attachment.xml,
0.02 N m s/rad) is 3.89x critical (b_crit = 2*sqrt(I*m*g*r) = 0.005145), i.e.
the pendulum AS COMMITTED is OVERDAMPED and has no resonance for ANY
controller to pump -- released from phi=0.3 rad it never crosses zero, it
creeps to hanging and stiction-locks. Both damping=0.02 and frictionloss=0.01
are labelled UNMEASURED PLACEHOLDERS in that file, plausible for the OLD
0.30m rod (zeta=0.40, underdamped) but never revisited after the 2026-08-11/
12 rod-length and mass corrections cut I ~15x and m*g*r ~6x.

This script's --damping/--frictionloss flags override the hinge's dynamics
ONLY inside this script's own compiled model object -- never the committed
asset. Passing --damping 0.02 --frictionloss 0.01 reproduces the actual
committed (overdamped, non-flipping) behavior; lower values quantify what
the plant would have to be for swing-up to become reachable. A render at a
reduced-damping value is evidence the CONTROL LAW is correct, NOT evidence
that swing-up works on the model as committed or on real hardware -- always
report which damping value a given render used.

Usage (headless host):
  MUJOCO_GL=egl python tools/diagnostics/render_energy_swingup_flip.py \\
      --damping 0.0 --frictionloss 0.0
  MUJOCO_GL=egl python tools/diagnostics/render_energy_swingup_flip.py \\
      --damping 0.02 --frictionloss 0.01 --output outputs/pendulum_renders/committed.mp4
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

from simulation.ur5e_pendulum_compose import compose_ur5e_pendulum_model  # noqa: E402
from simulation.ur5e_mujoco_torque import (  # noqa: E402
    build_initial_state_and_adapter,
    build_mujoco_state,
)
from tools.diagnostics.pendulum_swingup_energy_shaping import (  # noqa: E402
    ARM_Q0, CONTROL_DT, E_TOP, G, I_PIVOT_KGM2, M_TOTAL_KG, R_COM_M, RATE_HZ,
    find_hanging_and_inverted_angle, load_config,
)

# Sign-corrected energy-shaping gains. k_e>0 now means "pump" (it did not
# before the 2026-08-12 sign fix). Not exposed as CLI flags -- these are
# validated control-law gains, not the thing under test here; the physical
# damping/frictionloss values are.
KICK_AMPLITUDE_M, KICK_DURATION_S = 0.05, 0.58   # bootstrap (thetadot=0 makes the law inert)

FPS = 30.0
WIDTH, HEIGHT = 1280, 960

# cond(J) above this is treated as "reached a singularity" for reporting
# purposes -- well-conditioned poses tonight have measured cond(J)~7-10;
# this repo's own jacobian_singular_cond_max default (before it was
# disabled as a separate, unrelated fix) was 1e5.
SINGULARITY_COND_THRESHOLD = 1000.0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--damping", type=float, required=True,
                    help="Hinge viscous damping override, N m s/rad (committed value: 0.02).")
    p.add_argument("--frictionloss", type=float, required=True,
                    help="Hinge Coulomb frictionloss override, N m (committed value: 0.01).")
    p.add_argument("--k-e", type=float, default=100.0, help="Energy-shaping gain.")
    p.add_argument("--a-max", type=float, default=3.0, help="m/s^2 ceiling on commanded pivot acceleration.")
    p.add_argument("--k-pos", type=float, default=15.0,
                    help="Recentering spring gain (pulls target_x back toward x0). "
                         "2026-08-12: the old default of 1.0 was NEVER validated (this repo's own "
                         "DE search treats k_pos as a free parameter, bounds (0,20)) and let the "
                         "arm drift 1.4m from x0 into a near-singular pose (cond(J) up to 204185) "
                         "during a real trial -- raised the default to stay well clear of that.")
    p.add_argument("--k-vel", type=float, default=5.0, help="Recentering damper gain. See --k-pos.")
    p.add_argument("--duration-s", type=float, default=20.0)
    p.add_argument("--output", type=Path, default=None,
                    help="Output MP4 path. Default: auto-named from damping/frictionloss.")
    p.add_argument("--no-video", action="store_true",
                    help="Skip rendering/ffmpeg entirely -- just print the numeric result (fast, for a sweep).")
    return p.parse_args()


def run_trial(damping: float, frictionloss: float, k_e: float, a_max: float,
               k_pos: float, k_vel: float, duration_s: float,
               output: Path | None, write_video: bool) -> dict:
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
    committed_damping = float(model.dof_damping[pend_dof_adr])
    print(f"hanging={hanging_angle:.4f} rad  inverted={inverted_angle:.4f} rad  E_top={E_TOP:.5f} J")
    print(f"committed hinge damping = {committed_damping} (b_crit={b_crit:.6f}, "
          f"zeta_committed={committed_damping / b_crit:.2f})")
    model.dof_damping[pend_dof_adr] = damping
    model.dof_frictionloss[pend_dof_adr] = frictionloss
    print(f"OVERRIDDEN for this run -> damping={damping} (zeta={damping / b_crit:.4f}), "
          f"frictionloss={frictionloss}")

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

    proc = None
    renderer = None
    cam = None
    if write_video:
        model.vis.global_.offwidth = max(int(model.vis.global_.offwidth), WIDTH)
        model.vis.global_.offheight = max(int(model.vis.global_.offheight), HEIGHT)
        cam = mujoco.MjvCamera()
        mujoco.mjv_defaultFreeCamera(model, cam)
        cam.azimuth, cam.elevation, cam.distance = 100.0, -15.0, 1.9
        cam.lookat[:] = np.array([-0.3, -0.2, 0.5])
        renderer = mujoco.Renderer(model, height=HEIGHT, width=WIDTH)
        output.parent.mkdir(parents=True, exist_ok=True)
        proc = subprocess.Popen([
            "ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pixel_format", "rgb24",
            "-video_size", f"{WIDTH}x{HEIGHT}", "-framerate", str(FPS), "-i", "-",
            "-pix_fmt", "yuv420p", str(output),
        ], stdin=subprocess.PIPE)

    n_steps = int(duration_s * RATE_HZ)
    frame_stride = max(1, round(1.0 / (FPS * CONTROL_DT)))
    target_x, target_x_vel = x0, 0.0
    min_theta_dist, flip_t, written = np.pi, None, 0
    log_rows = []

    # Workspace-sanity tracking -- added 2026-08-12 after a real incident: the
    # old hardcoded K_POS=1.0 let target_x drift 1.85m from x0 and dragged the
    # arm to cond(J)=204185 (a real singularity) mid-trial, silently
    # invalidating that "flip." Never trust a flip number again without also
    # checking these.
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    max_abs_target_dev = 0.0
    max_abs_ee_dev = 0.0
    max_cond = 0.0
    max_cond_t = 0.0

    for step in range(n_steps):
        t = step * CONTROL_DT
        theta = float(data.qpos[pend_qpos_adr])
        thetadot = float(data.qvel[pend_dof_adr])
        phi = float(np.mod(theta - hanging_angle + np.pi, 2 * np.pi) - np.pi)
        E = 0.5 * I_PIVOT_KGM2 * thetadot * thetadot + M_TOTAL_KG * G * R_COM_M * (1.0 - np.cos(phi))

        if t < KICK_DURATION_S:
            omega_kick = 2.0 * np.pi / KICK_DURATION_S
            target_x = x0 + 0.5 * KICK_AMPLITUDE_M * (1.0 - np.cos(omega_kick * t))
            target_x_vel = 0.5 * KICK_AMPLITUDE_M * omega_kick * np.sin(omega_kick * t)
            a_cmd = float(0.5 * KICK_AMPLITUDE_M * omega_kick ** 2 * np.cos(omega_kick * t))
        else:
            # Leading MINUS = the 2026-08-12 sign fix. Identical to
            # run_energy_swingup_trial's law; see its comment for the derivation
            # and the work-balance measurement validating it.
            a_energy = -k_e * thetadot * np.cos(phi) * (E_TOP - E)
            a_recenter = -k_pos * (target_x - x0) - k_vel * target_x_vel
            a_cmd = float(np.clip(a_energy + a_recenter, -a_max, a_max))
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

        ee_x = float(data.site_xpos[site_id][0])
        max_abs_target_dev = max(max_abs_target_dev, abs(target_x - x0))
        max_abs_ee_dev = max(max_abs_ee_dev, abs(ee_x - x0))
        mujoco.mj_jacSite(model, data, jacp, jacr, site_id)
        j6 = np.vstack([jacp[:, :6], jacr[:, :6]])
        cond = float(np.linalg.cond(j6))
        if cond > max_cond:
            max_cond, max_cond_t = cond, t

        dist = abs(float(np.mod(theta - inverted_angle + np.pi, 2 * np.pi) - np.pi))
        if dist < min_theta_dist:
            min_theta_dist = dist
        if dist < 0.35 and flip_t is None:
            flip_t = t
            print(f"*** reached inverted (|theta - inverted| < 0.35 rad) at t={t:.2f}s ***")

        if step % 250 == 0:
            log_rows.append((t, np.degrees(phi), E / E_TOP, min_theta_dist))
        if write_video and step % frame_stride == 0:
            renderer.update_scene(data, camera=cam)
            proc.stdin.write(renderer.render().tobytes())
            written += 1

    if write_video:
        proc.stdin.close()
        proc.wait()
        if proc.returncode != 0:
            raise RuntimeError(f"ffmpeg exited with code {proc.returncode}")

    print(f"\n{'t':>6} {'phi_deg':>9} {'E/Etop':>8} {'min_dist':>9}")
    for t, pd, ef, md in log_rows:
        print(f"{t:6.2f} {pd:9.2f} {ef:8.4f} {md:9.4f}")
    reached_singularity = max_cond > SINGULARITY_COND_THRESHOLD
    result = {
        "damping": damping, "frictionloss": frictionloss,
        "zeta": damping / b_crit,
        "min_theta_dist_from_inverted_rad": min_theta_dist,
        "flipped": min_theta_dist < 0.35,
        "flip_t": flip_t,
        "max_abs_target_dev_m": max_abs_target_dev,
        "max_abs_ee_dev_m": max_abs_ee_dev,
        "max_cond_j": max_cond,
        "max_cond_j_t": max_cond_t,
        "reached_singularity": reached_singularity,
        "trustworthy": (min_theta_dist < 0.35) and not reached_singularity,
    }
    print(f"\nmin_theta_dist_from_inverted = {min_theta_dist:.4f} rad "
          f"(flipped={result['flipped']}, first reached t={flip_t})")
    print(f"max |target_x - x0| = {max_abs_target_dev:.4f} m   max |ee_x - x0| = {max_abs_ee_dev:.4f} m")
    print(f"max cond(J) = {max_cond:.2f} at t={max_cond_t:.2f}s"
          + (f"  *** SINGULARITY (> {SINGULARITY_COND_THRESHOLD:g}) -- RESULT NOT TRUSTWORTHY ***"
             if reached_singularity else "  (well-conditioned throughout)"))
    if write_video:
        print(f"Wrote {written} frames ({written/FPS:.1f}s) to {output}")
    return result


def main() -> int:
    args = parse_args()
    output = args.output
    if output is None and not args.no_video:
        output = REPO_ROOT / "outputs" / "pendulum_renders" / f"energy_swingup_damping_{args.damping:g}.mp4"
    run_trial(args.damping, args.frictionloss, args.k_e, args.a_max, args.k_pos, args.k_vel,
              args.duration_s, output, write_video=not args.no_video)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
