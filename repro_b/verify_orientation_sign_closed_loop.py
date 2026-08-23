"""Closed-loop A/B of the rotational P term's sign/frame, at ARM_Q0.

A = shipped code:  m_rot = +kp_rot * e_body   (e_body = 2*vec(conj(q0) q))
B = corrected:     m_rot = -kp_rot * R0 @ e_body   (textbook restoring world moment)

Implemented by patching the module-level `orientation_error_vec_wxyz` the
corridor controller imports, so ONLY the direction of the rotational P/D
vector changes.  ||e|| is invariant under (-1)*R0, so every reported metric
(orientation_error_norm, the guard, the Hessian weight max(kp_rot,1e-6))
is identical in construction between the two arms -- a clean single-variable
A/B.

NOTE this also flips the direction of the kd_rot*omega term's partner? No --
kd_rot multiplies `omega` directly, which is untouched.  Only e_rot changes.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

REPO = Path("/common/users/ss5772/real_Cartpole")
sys.path.insert(0, str(REPO))

import controller_core.x_task_yz_corridor_qp.controller as ctrl_mod
from controller_core.kinematics_utils import (
    orientation_error_vec_wxyz as _orig,
    quat_to_rotmat,
)

spec = importlib.util.spec_from_file_location(
    "sim_check", REPO / "tools" / "diagnostics" / "x_task_yz_corridor_qp_sim_check.py"
)
CHECK = importlib.util.module_from_spec(spec)
sys.modules["sim_check"] = CHECK
spec.loader.exec_module(CHECK)


def corrected(quat_des, quat_cur):
    return -(quat_to_rotmat(np.asarray(quat_des, float).reshape(4)) @ _orig(quat_des, quat_cur))


CONFIG = "config/ur5e_mujoco_torque_x_task_yz_corridor_qp_enabled.yaml"


def run(delta, move_s):
    return CHECK.run_rollout(
        CHECK.ARM_Q0, config_path=CONFIG, corridor=True, cbf=True,
        target_delta_m=delta, move_duration_s=move_s, hold_duration_s=1.0,
        label="ab",
    )


CELLS = [(-0.06, 1.5), (0.06, 0.1), (0.12, 1.5)]

print(f"{'cell':>18} {'arm':>10} {'track':>7} {'maxOri_rad':>11} {'maxOri_deg':>11} "
      f"{'maxY':>8} {'maxZ':>8} {'|qd|':>7} {'guard':>14}")
for delta, mv in CELLS:
    for arm in ("A_shipped", "B_corrected"):
        ctrl_mod.orientation_error_vec_wxyz = _orig if arm == "A_shipped" else corrected
        r = run(delta, mv)
        print(f"{f'dx={delta:+.2f}/{mv}s':>18} {arm:>10} {r.tracking_fraction:>7.3f} "
              f"{r.max_orientation_error_rad:>11.4f} "
              f"{np.degrees(r.max_orientation_error_rad):>11.3f} "
              f"{r.max_abs_y_drift_m:>8.4f} {r.max_abs_z_drift_m:>8.4f} "
              f"{r.max_abs_qd_radps:>7.3f} {(r.guard_reason or '-'):>14}")
ctrl_mod.orientation_error_vec_wxyz = _orig
