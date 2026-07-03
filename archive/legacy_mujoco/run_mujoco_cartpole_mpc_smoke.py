#!/usr/bin/env python3
"""MuJoCo smoke test for cart-pole MPC on the UR5e + pendulum scene."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import mujoco
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from controller_core import (  # noqa: E402
    CartPoleMPCConfig,
    CartPoleMPCController,
    CommandGovernorSafetyFilter,
    ControllerState,
    SafetyLimits,
)
from controller_core.mujoco_cartpole_state import (  # noqa: E402
    build_mujoco_cartpole_observer,
    default_cartpole_scene_candidates,
    read_mujoco_cartpole_state,
)
from simulation.controller import acceleration_transport_controller  # noqa: E402


def main() -> int:
    observer = build_mujoco_cartpole_observer(default_cartpole_scene_candidates(REPO_ROOT))
    if observer is None or not observer.has_pendulum:
        print("MuJoCo cart-pole observer unavailable (missing pendulum_hinge).", file=sys.stderr)
        return 2

    model, data = observer.model, observer.data
    dt = float(model.opt.timestep)
    if dt <= 0.0:
        dt = 0.002

    q0 = np.array(model.key("home").qpos[:6], dtype=np.float64)
    qd0 = np.zeros(6, dtype=np.float64)
    data.qpos[:] = 0.0
    data.qvel[:] = 0.0
    for idx, joint_name in enumerate(
        (
            "shoulder_pan_joint",
            "shoulder_lift_joint",
            "elbow_joint",
            "wrist_1_joint",
            "wrist_2_joint",
            "wrist_3_joint",
        )
    ):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        data.qpos[int(model.jnt_qposadr[jid])] = float(q0[idx])

    mujoco.mj_forward(model, data)
    site_id = observer.attachment_site_id
    target_x = float(data.site_xpos[site_id][0])
    duration_s = 6.0
    steps = int(duration_s / dt)

    mpc = CartPoleMPCController(
        CartPoleMPCConfig(
            horizon=20,
            dt_s=dt,
            pole_length_m=0.4,
            command_limit=1.2,
            q_weights=np.array([30.0, 8.0, 220.0, 24.0], dtype=np.float64),
            r_weight=0.3,
            target_x=target_x,
        )
    )
    governor = CommandGovernorSafetyFilter(
        SafetyLimits(
            x_min_m=target_x - 0.15,
            x_max_m=target_x + 0.15,
            max_x_velocity_mps=0.5,
            max_x_acceleration_mps2=1.2,
            max_command_change_per_cycle=0.25,
            dt_s=dt,
        )
    )

    ctrl = q0.copy()
    axis_state = 0.0
    theta_hist: list[float] = []
    x_hist: list[float] = []
    tau_hist: list[float] = []

    for step in range(steps):
        t = step * dt
        q = np.asarray(data.qpos[: model.nq], dtype=np.float64)
        qd = np.asarray(data.qvel[: model.nv], dtype=np.float64)
        st = read_mujoco_cartpole_state(
            observer,
            q[:6],
            qd[:6],
            time_s=t,
            dt_s=dt,
            target_x=target_x,
        )
        if st is None:
            return 3
        raw = mpc.compute(st)
        safe = governor.filter(st, raw)
        a_cmd = float(safe.command.value)

        site_xmat = np.asarray(data.site_xmat[site_id], dtype=np.float64).reshape(3, 3)
        site_jacp = np.zeros((3, model.nv), dtype=np.float64)
        site_jacr = np.zeros((3, model.nv), dtype=np.float64)
        mujoco.mj_jacSite(model, data, site_jacp, site_jacr, site_id)
        ctrl, _ = acceleration_transport_controller(
            q=q[:6],
            qvel=qd[:6],
            ctrl_prev=ctrl,
            ctrl_lower=model.jnt_range[:6, 0],
            ctrl_upper=model.jnt_range[:6, 1],
            tool_pos=np.asarray(data.site_xpos[site_id], dtype=np.float64),
            tool_rot=site_xmat,
            tool_jacobian_pos=site_jacp[:, :6],
            tool_jacobian_rot=site_jacr[:, :6],
            a_axis_cmd=a_cmd,
            axis_state=axis_state,
            transport_axis="x",
            fixed_position=np.asarray(data.site_xpos[site_id], dtype=np.float64),
            dt=dt,
            a_axis_max_m_s2=1.2,
            v_axis_max_m_s=0.4,
            hold_axis_weight=280.0,
            hold_axis_gain=20.0,
            orientation_weight=140.0,
        )
        axis_state = float(np.clip(axis_state + a_cmd * dt, -0.4, 0.4))
        data.ctrl[:6] = ctrl
        mujoco.mj_step(model, data)

        theta_hist.append(float(st.theta))
        x_hist.append(float(st.x))
        tau_hist.append(abs(a_cmd))

    summary = {
        "duration_s": duration_s,
        "dt_s": dt,
        "max_abs_theta_rad": float(max(abs(v) for v in theta_hist)),
        "final_abs_theta_rad": float(abs(theta_hist[-1])),
        "x_span_m": float(max(x_hist) - min(x_hist)),
        "peak_abs_accel": float(max(tau_hist)),
        "success": bool(max(abs(v) for v in theta_hist) < 0.45 and abs(theta_hist[-1]) < 0.25),
    }
    out = REPO_ROOT / "outputs/control_runs/mujoco_cartpole_mpc_smoke_summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0 if summary["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
