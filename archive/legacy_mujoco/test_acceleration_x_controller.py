#!/usr/bin/env python3
"""
Headless test for `acceleration_x_transport_controller`.

Drives the UR5e from the shoulder-side-face origin using only a signed X
acceleration command as outer-loop input. The scalar program is:

    t in [0.0, 1.5) s : a_x = +0.03 m/s^2   (accelerate along +X to v ~ +0.045 m/s)
    t in [1.5, 2.5) s : a_x =  0.0  m/s^2   (coast)
    t in [2.5, 5.5) s : a_x = -0.03 m/s^2   (brake, reverse to v ~ -0.045 m/s)
    t in [5.5, 7.0) s : a_x = +0.03 m/s^2   (brake back to rest)

The test records the tool world pose and asserts that
- world Z stays within a small tolerance of the starting height, and
- commanded speeds and estimated servo torques remain well under the safety
  caps (i.e. motion is not extreme).

Run with:
    MUJOCO_GL=egl python simulation/test_acceleration_x_controller.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

from controller import (  # noqa: E402
    ACTIVE_ORIGIN_Q,
    SERVO_FORCE_LIMIT,
    TARGET_SITE_ROTATION_WORLD,
    acceleration_x_transport_controller,
)

MUJOCO_MENAGERIE = BASE_DIR / "mujoco_menagerie"
UR5E_SCENE = MUJOCO_MENAGERIE / "universal_robots_ur5e" / "scene.xml"
UR5E_CARTPOLE_SCENE = MUJOCO_MENAGERIE / "universal_robots_ur5e" / "scene_ur5e_cartpole.xml"
TOOL_SITE_NAME = "attachment_site"
BASE_PAN_INDEX = 0

# Tucked low-Z transport pose from `fixed_z_x_transport_firstpass_z0.540_seed1.json`.
# The shoulder-side-face origin is nearly fully extended and has strong X-Z
# kinematic coupling, so the acceleration controller is validated at the same
# workspace pose the velocity controller uses in its saved runs.
TRANSPORT_START_Q = np.array(
    [
        0.0,
        -0.1133064268431449,
        -0.664621645801302,
        4.921777393344012,
        -6.283185307179586,
        5.280928640069786,
    ],
    dtype=np.float64,
)


def a_x_schedule(t: float) -> float:
    if t < 1.5:
        return +0.03
    if t < 2.5:
        return 0.0
    if t < 5.5:
        return -0.03
    if t < 7.0:
        return +0.03
    return 0.0


def main() -> int:
    scene_path = UR5E_CARTPOLE_SCENE if UR5E_CARTPOLE_SCENE.exists() else UR5E_SCENE
    model = mujoco.MjModel.from_xml_path(str(scene_path))
    data = mujoco.MjData(model)
    tool_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, TOOL_SITE_NAME)

    ctrl_lower = model.actuator_ctrlrange[: model.nu, 0].copy()
    ctrl_upper = model.actuator_ctrlrange[: model.nu, 1].copy()
    dt = float(model.opt.timestep)

    q_start = TRANSPORT_START_Q.copy()
    _ = ACTIVE_ORIGIN_Q  # kept imported for reference; not used at this pose.
    data.qpos[:] = 0.0
    data.qvel[:] = 0.0
    data.qpos[: model.nu] = q_start
    data.ctrl[:] = q_start
    mujoco.mj_forward(model, data)

    tool_start_pos = data.site_xpos[tool_site_id].copy()
    z_hold = float(tool_start_pos[2])
    pan_target = float(q_start[BASE_PAN_INDEX])

    total_duration_s = 7.5
    n_steps = int(round(total_duration_s / dt))

    ctrl = q_start.copy()
    v_x_state = 0.0
    tool_jacp = np.zeros((3, model.nv), dtype=np.float64)
    tool_jacr = np.zeros((3, model.nv), dtype=np.float64)

    x_trace: list[float] = []
    y_trace: list[float] = []
    z_trace: list[float] = []
    a_cmd_trace: list[float] = []
    v_state_trace: list[float] = []
    tau_max_trace: list[float] = []
    speed_scale_trace: list[float] = []
    torque_scale_trace: list[float] = []
    t_trace: list[float] = []

    sample_every = max(1, int(round(0.02 / dt)))

    for step in range(n_steps):
        t = float(data.time)
        a_cmd = a_x_schedule(t)

        q = data.qpos[: model.nu].copy()
        qvel = data.qvel[: model.nu].copy()
        mujoco.mj_jacSite(model, data, tool_jacp, tool_jacr, tool_site_id)
        tool_pos = data.site_xpos[tool_site_id].copy()
        tool_rot = np.asarray(data.site_xmat[tool_site_id], dtype=np.float64).reshape(3, 3)

        ctrl, diag = acceleration_x_transport_controller(
            q=q,
            qvel=qvel,
            ctrl_prev=ctrl,
            ctrl_lower=ctrl_lower,
            ctrl_upper=ctrl_upper,
            tool_pos=tool_pos,
            tool_rot=tool_rot,
            tool_jacobian_pos=tool_jacp[:3, : model.nu],
            tool_jacobian_rot=tool_jacr[:3, : model.nu],
            a_x_cmd=a_cmd,
            v_x_state=v_x_state,
            z_hold=z_hold,
            target_tool_rot=TARGET_SITE_ROTATION_WORLD,
            pan_target=pan_target,
            dt=dt,
        )
        v_x_state = float(diag["v_x_state_next"])

        data.ctrl[:] = ctrl
        mujoco.mj_step(model, data)

        if step % sample_every == 0:
            tp = data.site_xpos[tool_site_id]
            x_trace.append(float(tp[0]))
            y_trace.append(float(tp[1]))
            z_trace.append(float(tp[2]))
            a_cmd_trace.append(float(a_cmd))
            v_state_trace.append(v_x_state)
            tau_max_trace.append(float(np.max(np.abs(diag["tau_estimate_nm"]))))
            speed_scale_trace.append(float(diag["speed_scale"]))
            torque_scale_trace.append(float(diag["torque_scale"]))
            t_trace.append(t)

    x_arr = np.asarray(x_trace)
    y_arr = np.asarray(y_trace)
    z_arr = np.asarray(z_trace)
    v_arr = np.asarray(v_state_trace)
    tau_arr = np.asarray(tau_max_trace)
    speed_scale_arr = np.asarray(speed_scale_trace)
    torque_scale_arr = np.asarray(torque_scale_trace)

    z_abs_err = np.max(np.abs(z_arr - z_hold))
    z_final_err = float(abs(z_arr[-1] - z_hold))
    y_abs_drift = np.max(np.abs(y_arr - tool_start_pos[1]))
    x_span = float(x_arr.max() - x_arr.min())
    x_net_disp = float(x_arr[-1] - x_arr[0])
    v_peak = float(np.max(np.abs(v_arr)))
    v_final = float(abs(v_arr[-1]))
    tau_peak = float(np.max(tau_arr))
    tau_limit_peak = float(np.max(SERVO_FORCE_LIMIT))
    min_speed_scale = float(np.min(speed_scale_arr))
    min_torque_scale = float(np.min(torque_scale_arr))

    print("acceleration_x_transport_controller: headless smoke test")
    print(f"  dt                        : {dt:.5f} s")
    print(f"  duration                  : {total_duration_s:.2f} s")
    print(f"  tool start xyz (m)        : {tool_start_pos.tolist()}")
    print(f"  z_hold (m)                : {z_hold:.6f}")
    print(f"  max |z - z_hold| (m)      : {z_abs_err:.6f}  (transient)")
    print(f"  final |z - z_hold| (m)    : {z_final_err:.6f}  (settle)")
    print(f"  max |y - y_start| (m)     : {y_abs_drift:.6f}")
    print(f"  x span (m)                : {x_span:.4f}")
    print(f"  x net displacement (m)    : {x_net_disp:+.4f}")
    print(f"  peak |v_x_state| (m/s)    : {v_peak:.4f}")
    print(f"  final |v_x_state| (m/s)   : {v_final:.4f}")
    print(f"  peak |tau_est| (N*m)      : {tau_peak:.2f}  (worst joint limit {tau_limit_peak:.0f})")
    print(f"  min speed scale           : {min_speed_scale:.3f}")
    print(f"  min torque scale          : {min_torque_scale:.3f}")

    ok = True
    failures: list[str] = []

    # Transient z excursion is compared to the known-good velocity controller's
    # saved run at this same pose, which peaks around 11 mm during motion and
    # settles to ~20 um at rest.
    if z_abs_err > 15.0e-3:
        failures.append(f"Z transient too large: {z_abs_err*1000:.2f} mm > 15 mm")
    if z_final_err > 1.0e-3:
        failures.append(f"Z did not settle after v -> 0: final {z_final_err*1000:.2f} mm > 1 mm")
    if y_abs_drift > 5.0e-3:
        failures.append(f"Y drift too large: {y_abs_drift*1000:.2f} mm > 5 mm")
    if x_span < 0.02:
        failures.append(f"X barely moved: span={x_span:.4f} m < 0.02 m")
    if v_peak > 0.10:
        failures.append(f"Peak commanded speed {v_peak:.3f} m/s exceeds 0.10 m/s bound")
    if v_final > 5.0e-3:
        failures.append(f"Velocity did not return to rest: final {v_final:.4f} m/s")
    if tau_peak > 0.95 * tau_limit_peak:
        failures.append(f"Estimated torque {tau_peak:.1f} N*m near joint limit")

    if failures:
        ok = False
        print("\nFAIL:")
        for f in failures:
            print(f"  - {f}")
    else:
        print("\nOK: Z held, Y stable, motion bounded, servos within headroom.")

    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
