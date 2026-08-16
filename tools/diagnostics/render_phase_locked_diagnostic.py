#!/usr/bin/env python3
"""Instrumented render of the phase-locked swing-up (best found params, with
wrist_orientation_task enabled): tracks energy/torque/drift/orientation
over time AND writes an MP4, to actually see what's limiting progress
instead of just reading a pass/fail number.

WARNING -- the loop below is a HAND-COPY of run_phase_locked_trial() in
pendulum_swingup_phase_locked.py, not a call into it. It is currently
verified in sync (audit 2026-08-12: crossing detection, debounce,
stall-recovery reset, crossing_ts clearing, amplitude ramp, and all three
drive equations match line for line; the only intended differences are
that this copy never breaks on a guard trip -- it records the first trip
and runs 0.4 s longer so the failure is visible on video -- and that it
logs/renders). Any change to that function MUST be mirrored here; the
shared constants are imported rather than re-declared specifically so
that at least T_NATURAL_S / STALL_* cannot drift.

BEST below is STALE as of 2026-08-12 and is kept only to reproduce the
historical diagnostic run. Its phase_offset_bias=-0.225 came from a search
whose bounds could not express the correct answer (net energy transfer
goes as sin(phase_offset_bias), so that value pumps at -22% of peak, i.e.
it removes energy); its a_max=0.0913 was tuned against an R_COM_M that
made T_NATURAL_S 35% too short. Re-run the search in
pendulum_swingup_phase_locked.py and update these before drawing any
conclusion from a fresh render."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import mujoco
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from simulation.ur5e_pendulum_compose import compose_ur5e_pendulum_model  # noqa: E402
from simulation.ur5e_mujoco_torque import build_initial_state_and_adapter, build_mujoco_state  # noqa: E402
from tools.diagnostics.pendulum_swingup_energy_shaping import (  # noqa: E402
    load_config, ARM_Q0, CONTROL_DT, RATE_HZ, find_hanging_and_inverted_angle,
    M_TOTAL_KG, R_COM_M, I_PIVOT_KGM2, G, E_TOP,
)
from tools.diagnostics.pendulum_swingup_multi_kick import find_nearby_equilibrium  # noqa: E402
from tools.diagnostics.pendulum_swingup_phase_locked import (  # noqa: E402
    T_NATURAL_S, MIN_CROSSINGS_FOR_LIVE_T_EST, STALL_TIMEOUT_PERIODS, STALL_RESET_RAMP_S,
)

BEST = {
    "k_a": 4.638546425723448,
    "a_max": 0.09127659537023416,
    "phase_offset_bias": -0.22513669277004186,
    "crossing_debounce_s": 0.25565726042467485,
}
DURATION_S = 15.0
OUTPUT = REPO_ROOT / "outputs" / "pendulum_renders" / "phase_locked_diagnostic.mp4"
FPS = 30.0
WIDTH, HEIGHT = 1280, 960


def main() -> int:
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
    tip_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "/pendulum_tip_site")

    hanging_angle, inverted_angle = find_hanging_and_inverted_angle(model, data, pend_qpos_adr)
    print(f"hanging_angle={hanging_angle:.4f} inverted_angle={inverted_angle:.4f} T_natural={T_NATURAL_S:.4f}s")

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
    cam.azimuth, cam.elevation, cam.distance = 100.0, -15.0, 1.9
    cam.lookat[:] = np.array([-0.3, -0.2, 0.5])
    renderer = mujoco.Renderer(model, height=HEIGHT, width=WIDTH)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg_cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pixel_format", "rgb24",
        "-video_size", f"{WIDTH}x{HEIGHT}", "-framerate", str(FPS),
        "-i", "-", "-pix_fmt", "yuv420p", str(OUTPUT),
    ]
    proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)

    n_steps = int(DURATION_S * RATE_HZ)
    frame_stride = max(1, round(1.0 / (FPS * CONTROL_DT)))

    current_hanging_angle = hanging_angle
    crossing_ts = []
    prev_phi = None
    T_est = T_NATURAL_S
    t_last_crossing = 0.0
    dir_current = 1.0
    t_last_stall_reset = -1e9

    k_a, a_max = BEST["k_a"], BEST["a_max"]
    phase_offset_bias, crossing_debounce_s = BEST["phase_offset_bias"], BEST["crossing_debounce_s"]

    min_theta_dist = np.pi
    peak_swing_t = 0.0
    written = 0
    guard_trip_t = None

    log_rows = []
    peak_tau_norm = 0.0
    torque_limit_nm = np.array([150.0, 150.0, 150.0, 28.0, 28.0, 28.0])

    for step in range(n_steps):
        t = step * CONTROL_DT
        theta = float(data.qpos[pend_qpos_adr])
        thetadot = float(data.qvel[pend_dof_adr])
        phi = float(np.mod(theta - current_hanging_angle + np.pi, 2 * np.pi) - np.pi)

        if prev_phi is not None:
            crossed = (prev_phi == 0.0) or ((prev_phi > 0.0) != (phi > 0.0))
            debounced = crossing_ts and (t - crossing_ts[-1]) < crossing_debounce_s
            if crossed and not debounced:
                crossing_ts.append(t)
                t_last_crossing = t
                dir_current = 1.0 if thetadot >= 0.0 else -1.0
                if len(crossing_ts) >= MIN_CROSSINGS_FOR_LIVE_T_EST:
                    live_T = 2.0 * (crossing_ts[-1] - crossing_ts[-2])
                    if live_T > 1e-6:
                        T_est = live_T
                # Anchored on current_hanging_angle, not theta -- kept in sync
                # with pendulum_swingup_phase_locked.py 2026-08-12; see that
                # file's comment (and find_nearby_equilibrium's docstring) for
                # why anchoring on theta can silently return the INVERTED
                # equilibrium.
                current_hanging_angle = find_nearby_equilibrium(
                    model, data.qpos[:6].copy(), pend_qpos_adr, current_hanging_angle
                )
            elif (t - t_last_crossing) > STALL_TIMEOUT_PERIODS * T_est:
                # Same stall-recovery fix as pendulum_swingup_phase_locked.py --
                # see that file for the full explanation (including why
                # crossing_ts must be cleared too, not just T_est reset).
                T_est = T_NATURAL_S
                t_last_crossing = t
                crossing_ts.clear()
                t_last_stall_reset = t
                current_hanging_angle = find_nearby_equilibrium(
                    model, data.qpos[:6].copy(), pend_qpos_adr, current_hanging_angle
                )
        prev_phi = phi

        E = 0.5 * I_PIVOT_KGM2 * thetadot * thetadot + M_TOTAL_KG * G * R_COM_M * (1.0 - np.cos(phi))
        A = float(np.clip(k_a * (E_TOP - E), 0.0, a_max))
        ramp = min(1.0, (t - t_last_stall_reset) / STALL_RESET_RAMP_S)
        A *= ramp
        omega_est = 2.0 * np.pi / T_est
        phase = omega_est * (t - t_last_crossing) + phase_offset_bias
        target_x = x0 + dir_current * A * np.sin(phase)
        target_x_vel = dir_current * A * omega_est * np.cos(phase)
        a_cmd = float(-dir_current * A * omega_est * omega_est * np.sin(phase))

        state = build_mujoco_state(
            model, data, site_id=site_id, joint_ids=joint_ids, time_s=t, dt_s=CONTROL_DT,
            target_x=target_x, target_x_vel=target_x_vel, target_x_accel=a_cmd,
            reference_quat=state0.ee_quat, transport_axis_index=0, gravity_compensation=True,
        )
        tau, diag = adapter.step(state=state)
        if not bool(diag.get("safety_ok", True)) and guard_trip_t is None:
            guard_trip_t = t
            print(f"Guard tripped at t={t:.2f}s: {diag.get('safety_reason')}")

        tau_arr = np.asarray(tau, dtype=np.float64).reshape(6)
        tau_norm_frac = float(np.max(np.abs(tau_arr) / torque_limit_nm))
        peak_tau_norm = max(peak_tau_norm, tau_norm_frac)

        data.ctrl[:6] = tau_arr
        mujoco.mj_step(model, data)

        dist = abs(float(np.mod(theta - inverted_angle + np.pi, 2 * np.pi) - np.pi))
        if dist < min_theta_dist:
            min_theta_dist = dist
            peak_swing_t = t

        if step % 250 == 0:  # every 0.5s
            ee_y_drift = float(data.site_xpos[site_id][1] - state0.ee_pos[1])
            ee_z_drift = float(data.site_xpos[site_id][2] - state0.ee_pos[2])
            log_rows.append({
                "t": t, "phi_deg": np.degrees(phi), "E_over_Etop": E / E_TOP, "A_cmd_m": A,
                "T_est_s": T_est, "peak_tau_frac": tau_norm_frac,
                "orientation_err": float(diag.get("orientation_error_norm", 0.0)),
                "y_drift": ee_y_drift, "z_drift": ee_z_drift,
                "min_theta_dist_so_far": min_theta_dist,
            })

        if step % frame_stride == 0:
            renderer.update_scene(data, camera=cam)
            proc.stdin.write(renderer.render().tobytes())
            written += 1

        if guard_trip_t is not None and t > guard_trip_t + 0.4:
            break

    proc.stdin.close()
    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg exited with code {proc.returncode}")

    print(f"\n{'t':>6} {'phi_deg':>8} {'E/Etop':>7} {'A_cmd_m':>8} {'T_est_s':>7} "
          f"{'peak_tau%':>9} {'orient_err':>10} {'y_drift':>8} {'z_drift':>8} {'min_dist':>8}")
    for r in log_rows:
        print(f"{r['t']:6.2f} {r['phi_deg']:8.2f} {r['E_over_Etop']:7.4f} {r['A_cmd_m']:8.4f} "
              f"{r['T_est_s']:7.3f} {r['peak_tau_frac']*100:8.1f}% {r['orientation_err']:10.4f} "
              f"{r['y_drift']:8.4f} {r['z_drift']:8.4f} {r['min_theta_dist_so_far']:8.4f}")

    print(f"\nmin_theta_dist_from_inverted={min_theta_dist:.4f} rad at t={peak_swing_t:.2f}s "
          f"(flipped={min_theta_dist < 0.35})")
    print(f"peak torque usage anywhere: {peak_tau_norm*100:.1f}% of limit")
    print(f"guard_trip_t={guard_trip_t}")
    print(f"Wrote {written} frames ({written/FPS:.1f}s) to {OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
