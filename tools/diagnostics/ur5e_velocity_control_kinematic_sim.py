#!/usr/bin/env python3
"""Kinematic-only sim for hardware/velocity_transport.py's speedL control law.

Deliberately does NOT use MuJoCo physics/dynamics (no mj_step, no torque, no
mass matrix) -- real speedL is resolved to joint velocities by the robot
FIRMWARE's own Jacobian-based IK, not by rigid-body dynamics. This script
reproduces exactly that: MuJoCo is used only for forward kinematics (J(q)
via mj_forward/mj_jacSite), then qd = pinv(J) @ xd_cmd and q is integrated
by plain Euler steps -- the same category of simulation this repo's
LocalMujocoDynamics already does for J/M lookups, just driven by velocity
instead of torque.

Known simplification, stated up front: uses a plain Moore-Penrose
pseudoinverse (np.linalg.pinv), not the damped/singularity-robust IK a real
UR controller likely uses internally -- near a kinematic singularity this
sim's qd can spike higher than real hardware's actual behavior would. Not
expected to matter away from the wrist_2=0 singularity (this script's
default start pose uses the same wrist_2=0.2 rad offset already validated
elsewhere in this repo for exactly this reason).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from controller_core.cartesian_velocity_controller import (  # noqa: E402
    CartesianVelocityConfig,
    CartesianVelocityController,
)
from controller_core.kinematics_utils import orientation_error_vec_wxyz  # noqa: E402
from hardware.local_dynamics import LocalMujocoDynamics  # noqa: E402
from simulation.ur5e_mujoco_torque import x_profile_target  # noqa: E402

DEFAULT_CONFIG = REPO_ROOT / "config" / "ur5e_velocity_control.yaml"
# -40deg base-rotation, wrist_2=0.2 offset pose -- same pose the torque-
# control locked-pan sweep used earlier this session, for direct comparison.
DEFAULT_START_Q = [
    -0.6981317007977318,
    -0.8353981633974483,
    -1.2,
    -0.9853981633974482,
    0.2,
    0.0,
]


def _site_pose(dyn: LocalMujocoDynamics, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Position + wxyz quaternion of the EE site at joint config q. Also
    refreshes dyn.data via mj_forward as a side effect (matches
    jacobian_and_mass_matrix's own contract)."""
    import mujoco

    q_arr = np.asarray(q, dtype=np.float64).reshape(dyn.n_joints)
    dyn.data.qpos[: dyn.n_joints] = q_arr
    mujoco.mj_forward(dyn.model, dyn.data)
    pos = np.asarray(dyn.data.site_xpos[dyn.site_id], dtype=np.float64).copy()
    quat = np.zeros(4, dtype=np.float64)
    mujoco.mju_mat2Quat(quat, dyn.data.site_xmat[dyn.site_id])
    return pos, quat


def run_one(
    *,
    dyn: LocalMujocoDynamics,
    velocity_cfg: CartesianVelocityConfig,
    safety: dict[str, float],
    q0: np.ndarray,
    target_x_delta_m: float,
    move_duration_s: float,
    duration_s: float,
    rate_hz: float,
) -> dict[str, Any]:
    dt = 1.0 / float(rate_hz)
    controller = CartesianVelocityController(velocity_cfg)

    q = np.asarray(q0, dtype=np.float64).reshape(6).copy()
    p0, quat0 = _site_pose(dyn, q)
    x0 = float(p0[0])

    controller.reset_from_state(
        {"time": 0.0, "q": q, "qd": np.zeros(6), "ee_pos": p0, "ee_quat": quat0, "target_x": x0}
    )

    max_abs_y_drift = 0.0
    max_abs_z_drift = 0.0
    max_orientation_error = 0.0
    max_abs_qd = 0.0
    termination_reason = "duration_complete"
    t_s = 0.0
    steps = 0
    final_x = x0

    qd_max = float(safety.get("max_joint_velocity_radps", 3.0))
    ortho_max = float(safety.get("max_abs_orthogonal_drift_m", 0.05))
    orient_max = float(safety.get("max_orientation_error_rad", 0.25))

    while t_s < duration_s - 1e-12:
        target_x, target_x_vel = x_profile_target(
            "min_jerk_move_hold", x0, float(target_x_delta_m), t_s, duration_s, move_duration_s=move_duration_s
        )
        target_ee_pos = p0.copy()
        target_ee_pos[0] = float(target_x)
        target_ee_vel = np.array([target_x_vel, 0.0, 0.0], dtype=np.float64)

        p, quat = _site_pose(dyn, q)
        import mujoco

        jacp = np.zeros((3, dyn.model.nv), dtype=np.float64)
        jacr = np.zeros((3, dyn.model.nv), dtype=np.float64)
        mujoco.mj_jacSite(dyn.model, dyn.data, jacp, jacr, dyn.site_id)
        jacobian = np.vstack([jacp[:, : dyn.n_joints], jacr[:, : dyn.n_joints]]).astype(np.float64)

        robot_state = {
            "time": t_s,
            "q": q,
            "qd": np.zeros(6),
            "ee_pos": p,
            "ee_quat": quat,
            "target_x": float(target_x),
            "target_ee_pos": target_ee_pos,
            "target_ee_vel": target_ee_vel,
            "jacobian": jacobian,
        }
        xd_cmd = controller.compute(robot_state)

        qd = np.linalg.pinv(jacobian) @ xd_cmd
        max_abs_qd = max(max_abs_qd, float(np.max(np.abs(qd))))

        y_drift = abs(float(p[1] - p0[1]))
        z_drift = abs(float(p[2] - p0[2]))
        e_rot = orientation_error_vec_wxyz(quat0, quat)
        orientation_error = float(np.linalg.norm(e_rot))
        max_abs_y_drift = max(max_abs_y_drift, y_drift)
        max_abs_z_drift = max(max_abs_z_drift, z_drift)
        max_orientation_error = max(max_orientation_error, orientation_error)

        if float(np.max(np.abs(qd))) > qd_max:
            termination_reason = f"joint_velocity_guard: |qd|={float(np.max(np.abs(qd))):.4f} > {qd_max}"
            final_x = float(p[0])
            break
        if max(y_drift, z_drift) > ortho_max:
            termination_reason = f"orthogonal_drift_guard: {max(y_drift, z_drift):.4f} > {ortho_max}"
            final_x = float(p[0])
            break
        if orientation_error > orient_max:
            termination_reason = f"orientation_guard: {orientation_error:.4f} > {orient_max}"
            final_x = float(p[0])
            break

        q = q + qd * dt
        t_s += dt
        steps += 1
        final_x = float(p[0])

    achieved_x_delta_m = final_x - x0
    safety_pass = termination_reason == "duration_complete"
    return {
        "target_x_delta_m": float(target_x_delta_m),
        "move_duration_s": float(move_duration_s),
        "achieved_x_delta_m": achieved_x_delta_m,
        "max_abs_y_drift_m": max_abs_y_drift,
        "max_abs_z_drift_m": max_abs_z_drift,
        "max_abs_orthogonal_drift_m": max(max_abs_y_drift, max_abs_z_drift),
        "max_orientation_error_rad": max_orientation_error,
        "max_abs_qd_radps": max_abs_qd,
        "termination_reason": termination_reason,
        "safety_pass": safety_pass,
        "steps": steps,
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--start-q-rad", type=float, nargs=6, default=DEFAULT_START_Q)
    p.add_argument("--target-x-delta", type=float, required=True)
    p.add_argument("--move-duration", type=float, required=True)
    p.add_argument("--duration", type=float, default=3.0)
    p.add_argument("--rate-hz", type=float, default=125.0)
    args = p.parse_args()

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    velocity_cfg = CartesianVelocityConfig.from_controller_yaml_section(cfg.get("controller", {}) or {})
    safety = (cfg.get("controller", {}) or {}).get("safety", {}) or {}
    scene_xml = (cfg.get("mujoco", {}) or {}).get("scene_xml", "assets/ur5e_torque/scene.xml")
    dyn = LocalMujocoDynamics(REPO_ROOT / scene_xml)

    result = run_one(
        dyn=dyn,
        velocity_cfg=velocity_cfg,
        safety=safety,
        q0=np.asarray(args.start_q_rad, dtype=np.float64),
        target_x_delta_m=args.target_x_delta,
        move_duration_s=args.move_duration,
        duration_s=args.duration,
        rate_hz=args.rate_hz,
    )
    print(json.dumps(result, indent=2))
    return 0 if result["safety_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
