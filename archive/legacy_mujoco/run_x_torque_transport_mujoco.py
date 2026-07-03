#!/usr/bin/env python3
"""
MuJoCo torque-mode runner for the UR5e X-axis PD task.

Uses ``controller_core`` (the simulator-independent controller) to drive the
same UR5e MJCF as the velocity/acceleration runners, but in a **pure torque
mode** so that we can compare the closed-loop behavior against CoppeliaSim
running the same controller.

Key differences vs. the existing velocity/position-servo runners:

- Position-servo gains are zeroed at runtime (``model.actuator_gainprm``,
  ``model.actuator_biasprm``), so ``data.ctrl`` has no effect. The XML is
  untouched.
- Joint torques are injected via ``data.qfrc_applied[: model.nu] = tau`` every
  substep.
- Control law is ``Fx = kp_x*(target_x - x) - kd_x*vx`` followed by
  ``tau = J_pos^T @ [Fx, 0, 0] - kd_joint * qd``, both via ``controller_core``.

Output: a JSON trace compatible with
``simulation/compare_mujoco_vs_coppeliasim.py``.

Run:

    MUJOCO_GL=egl python simulation/run_x_torque_transport_mujoco.py \\
        --duration 6.0 --target-x-delta 0.05
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

if "MUJOCO_GL" not in os.environ:
    os.environ["MUJOCO_GL"] = "egl"

SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(BASE_DIR))
sys.path.insert(0, str(SCRIPT_DIR))

import mujoco
import numpy as np

from controller import ACTIVE_ORIGIN_Q
from controller_core import (
    SafetyConfig,
    SafetyMonitor,
    XAxisController,
    XAxisControllerConfig,
    as_robot_state,
    cartesian_force_to_joint_torque,
)
from controller_core.kinematics_utils import rotmat_to_quat


MUJOCO_MENAGERIE = BASE_DIR / "mujoco_menagerie"
UR5E_SCENE = MUJOCO_MENAGERIE / "universal_robots_ur5e" / "scene.xml"
UR5E_CARTPOLE_SCENE = MUJOCO_MENAGERIE / "universal_robots_ur5e" / "scene_ur5e_cartpole.xml"
SUMMARY_DIR = BASE_DIR / "outputs" / "control_runs"
TOOL_SITE_NAME = "attachment_site"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--duration", type=float, default=6.0)
    p.add_argument(
        "--target-x-delta",
        type=float,
        default=0.05,
        help="World-X offset from the initial EE position (meters).",
    )
    p.add_argument("--kp-x", type=float, default=300.0)
    p.add_argument("--kd-x", type=float, default=60.0)
    p.add_argument("--fx-max", type=float, default=50.0)
    p.add_argument(
        "--kd-joint",
        type=float,
        nargs=6,
        default=[2.0, 2.0, 1.5, 0.5, 0.5, 0.5],
    )
    p.add_argument(
        "--tau-max",
        type=float,
        nargs=6,
        default=[10.0, 10.0, 10.0, 3.0, 3.0, 3.0],
        help="Per-joint torque limit (conservative test values by default).",
    )
    p.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON path (default: outputs/control_runs/x_torque_mujoco_<time>.json).",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--gravity-comp",
        action="store_true",
        help="Add MuJoCo's qfrc_bias (gravity/Coriolis) to the commanded torque.",
    )
    return p.parse_args()


def _neutralize_position_servos(model: mujoco.MjModel) -> None:
    """Zero out position-servo effects so only qfrc_applied drives the joints.

    MuJoCo's ``general`` actuator (used by the UR5e) computes actuation from
    ``gainprm`` and ``biasprm``. Zeroing them turns the actuator output into
    zero for any ``data.ctrl``, leaving gravity, damping, and our injected
    torques as the only forces.
    """
    if model.nu == 0:
        return
    model.actuator_gainprm[: model.nu, :] = 0.0
    model.actuator_biasprm[: model.nu, :] = 0.0


def run() -> None:
    args = parse_args()
    np.random.seed(args.seed)

    scene = UR5E_CARTPOLE_SCENE if UR5E_CARTPOLE_SCENE.exists() else UR5E_SCENE
    model = mujoco.MjModel.from_xml_path(str(scene))
    data = mujoco.MjData(model)
    dt = float(model.opt.timestep)

    q_start = ACTIVE_ORIGIN_Q.copy()
    data.qpos[:] = 0.0
    data.qvel[:] = 0.0
    data.qpos[: model.nu] = q_start
    mujoco.mj_forward(model, data)

    _neutralize_position_servos(model)
    data.ctrl[:] = 0.0

    tool_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, TOOL_SITE_NAME)
    tool_start = data.site_xpos[tool_id].copy()
    target_x = float(tool_start[0] + args.target_x_delta)
    print(
        f"MuJoCo torque runner: ee_start=({tool_start[0]:.4f}, {tool_start[1]:.4f}, "
        f"{tool_start[2]:.4f}) target_x={target_x:.4f}"
    )

    controller = XAxisController(
        XAxisControllerConfig(kp_x=args.kp_x, kd_x=args.kd_x, fx_max_n=args.fx_max)
    )
    safety = SafetyMonitor(
        SafetyConfig(
            tau_max=np.asarray(args.tau_max, dtype=np.float64),
            yz_drift_max_m=0.10,
            ee_jump_max_m=0.05,
            x_error_growth_abort_s=5.0,
            watchdog_timeout_s=1e9,  # disabled in offline runs
        )
    )

    num_joints = int(model.nu)
    jacp = np.zeros((3, model.nv), dtype=np.float64)
    jacr = np.zeros((3, model.nv), dtype=np.float64)
    kd_joint = np.asarray(args.kd_joint, dtype=np.float64)

    trace: list[dict] = []
    steps = int(np.ceil(args.duration / dt))
    safety_trip_announced = False
    for step in range(steps):
        t = float(data.time)
        mujoco.mj_jacSite(model, data, jacp, jacr, tool_id)
        j_pos = jacp[:, :num_joints].copy()
        j_rot = jacr[:, :num_joints].copy()

        q = data.qpos[:num_joints].copy()
        qd = data.qvel[:num_joints].copy()
        ee_pos = data.site_xpos[tool_id].copy()
        ee_rot = data.site_xmat[tool_id].reshape(3, 3).copy()
        ee_quat = rotmat_to_quat(ee_rot)
        ee_lin_vel = (j_pos @ qd).astype(np.float64)
        ee_ang_vel = (j_rot @ qd).astype(np.float64)

        state = as_robot_state(
            {
                "time": t,
                "q": q,
                "qd": qd,
                "ee_pos": ee_pos,
                "ee_quat": ee_quat,
                "ee_lin_vel": ee_lin_vel,
                "ee_ang_vel": ee_ang_vel,
                "target_x": target_x,
                "jacobian_pos": j_pos,
                "jacobian_rot": j_rot,
            },
            num_joints=num_joints,
        )

        ctrl_out = controller.compute(state)
        grav_tau = None
        if args.gravity_comp:
            grav_tau = data.qfrc_bias[:num_joints].copy()
        tau_out = cartesian_force_to_joint_torque(
            fx_newtons=float(ctrl_out.fx or 0.0),
            jacobian_pos=j_pos,
            qd=qd,
            kd_joint=kd_joint,
            gravity_torque=grav_tau,
            tau_max=np.asarray(args.tau_max, dtype=np.float64),
        )
        tau = tau_out.tau if tau_out.tau is not None else np.zeros(num_joints)

        status = safety.check(state, tau)
        if not status.ok:
            if not safety_trip_announced:
                print(f"First safety trip at t={t:.3f}s: {status.reason}. Zeroing torque.")
                safety_trip_announced = True
            tau = np.zeros(num_joints)

        data.qfrc_applied[:num_joints] = tau
        mujoco.mj_step(model, data)

        trace.append(
            {
                "time": t,
                "q": q.tolist(),
                "qd": qd.tolist(),
                "ee_pos": ee_pos.tolist(),
                "ee_quat": ee_quat.tolist(),
                "ee_lin_vel": ee_lin_vel.tolist(),
                "target_x": target_x,
                "Fx": float(ctrl_out.fx or 0.0),
                "x_error": float(ctrl_out.x_error or 0.0),
                "tau": tau.tolist(),
                "safety_ok": bool(status.ok),
                "safety_reason": status.reason,
            }
        )

    out_path = (
        Path(args.output)
        if args.output
        else SUMMARY_DIR / "x_torque_mujoco.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "simulator": "mujoco",
        "scene": str(scene),
        "dt": dt,
        "duration": float(data.time),
        "target_x": target_x,
        "initial_ee_pos": tool_start.tolist(),
        "controller": {
            "kp_x": args.kp_x,
            "kd_x": args.kd_x,
            "fx_max_n": args.fx_max,
            "kd_joint": list(args.kd_joint),
            "tau_max_nm": list(args.tau_max),
        },
        "trace": trace,
    }
    out_path.write_text(json.dumps(summary))
    print(f"Saved MuJoCo torque trace to: {out_path}")


if __name__ == "__main__":
    run()
