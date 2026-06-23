#!/usr/bin/env python3
"""
Estimate world-X range of attachment_site for the UR5e under:

- joint limits from the loaded MJCF
- shoulder_pan_joint fixed at 0 (matches the active control policy)
- tool orientation fixed to TARGET_SITE_ROTATION_WORLD (shoulder-side upright / face frame)

No cartpole dynamics; kinematics only via MuJoCo forward kinematics + optimization.

Also reports:
- raw translational Jacobian row for world X (m/rad), i.e. ∂x/∂q_i with orientation *not* held
- projected gradient P∇x in the null space of the site rotational Jacobian (feasible instantaneous X motion
  while keeping tool angular velocity zero to first order), plus its norm and a finite-difference check

Run:
  cd real_Cartpole && python simulation/study_constrained_ee_x_workspace.py
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

if "MUJOCO_GL" not in os.environ:
    os.environ["MUJOCO_GL"] = "glfw"

import mujoco
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation

from controller import ACTIVE_ORIGIN_Q, TARGET_SITE_ROTATION_WORLD

SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parent
MUJOCO_MENAGERIE = BASE_DIR / "mujoco_menagerie"
UR5E_SCENE = MUJOCO_MENAGERIE / "universal_robots_ur5e" / "scene.xml"
UR5E_CARTPOLE_SCENE = MUJOCO_MENAGERIE / "universal_robots_ur5e" / "scene_ur5e_cartpole.xml"
TOOL_SITE_NAME = "attachment_site"
JOINT_NAMES = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]


def scene_path() -> Path:
    return UR5E_CARTPOLE_SCENE if UR5E_CARTPOLE_SCENE.exists() else UR5E_SCENE


def set_q(model: mujoco.MjModel, data: mujoco.MjData, q6: np.ndarray) -> None:
    data.qpos[:] = 0.0
    data.qvel[:] = 0.0
    data.qpos[: model.nu] = np.asarray(q6, dtype=np.float64)
    mujoco.mj_forward(model, data)


def site_rotation(data: mujoco.MjData, site_id: int) -> np.ndarray:
    return np.asarray(data.site_xmat[site_id], dtype=np.float64).reshape(3, 3)


def orientation_rotvec(R: np.ndarray, R_target: np.ndarray) -> np.ndarray:
    return Rotation.from_matrix(R_target.T @ R).as_rotvec()


def projector_onto_null(J: np.ndarray, damp: float = 1e-8) -> np.ndarray:
    """Projector onto {v : J v = 0} for J with shape (m, n), m <= n, full row rank."""
    m, n = J.shape
    if m == 0:
        return np.eye(n)
    jjt = J @ J.T + damp * np.eye(m)
    return np.eye(n) - J.T @ np.linalg.solve(jjt, J)


def build_study(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    site_id: int,
    q_pan_fixed: float,
    optimization_seed: int = 0,
) -> dict:
    nu = model.nu
    assert nu == 6

    # Bounds for y = q[1:6] from joint ranges (joint id 1..5)
    lowers = []
    uppers = []
    for jid in range(1, 6):
        lo, hi = model.jnt_range[jid]
        lowers.append(float(lo))
        uppers.append(float(hi))
    bounds = list(zip(lowers, uppers, strict=True))

    jacp = np.zeros((3, model.nv), dtype=np.float64)
    jacr = np.zeros((3, model.nv), dtype=np.float64)

    def q_from_y(y: np.ndarray) -> np.ndarray:
        q = np.zeros(6, dtype=np.float64)
        q[0] = q_pan_fixed
        q[1:6] = y
        return q

    def world_x_and_rotvec(y: np.ndarray) -> tuple[float, np.ndarray]:
        set_q(model, data, q_from_y(y))
        x = float(data.site_xpos[site_id][0])
        rv = orientation_rotvec(site_rotation(data, site_id), TARGET_SITE_ROTATION_WORLD)
        return x, rv

    def objective_min_x(y: np.ndarray) -> float:
        x, _ = world_x_and_rotvec(y)
        return x

    def objective_max_x(y: np.ndarray) -> float:
        return -world_x_and_rotvec(y)[0]

    def constraint_rotvec(y: np.ndarray) -> np.ndarray:
        _, rv = world_x_and_rotvec(y)
        return rv

    # Multi-start SLSQP for min and max world X
    rng = np.random.default_rng(optimization_seed)
    seeds: list[np.ndarray] = []
    seeds.append(ACTIVE_ORIGIN_Q[1:6].copy())
    for _ in range(24):
        y = np.array(
            [rng.uniform(lo, hi) for lo, hi in bounds],
            dtype=np.float64,
        )
        seeds.append(y)

    cons = {"type": "eq", "fun": constraint_rotvec}

    best_min = (np.inf, None)
    best_max = (-np.inf, None)

    for y0 in seeds:
        res_min = minimize(
            objective_min_x,
            y0,
            method="SLSQP",
            bounds=bounds,
            constraints=cons,
            options={"maxiter": 500, "ftol": 1e-12},
        )
        if res_min.success:
            xm, _ = world_x_and_rotvec(res_min.x)
            if xm < best_min[0]:
                best_min = (xm, res_min.x.copy())

        res_max = minimize(
            objective_max_x,
            y0,
            method="SLSQP",
            bounds=bounds,
            constraints=cons,
            options={"maxiter": 500, "ftol": 1e-12},
        )
        if res_max.success:
            xm, _ = world_x_and_rotvec(res_max.x)
            if xm > best_max[0]:
                best_max = (xm, res_max.x.copy())

    # Sensitivities at canonical pose
    q0 = ACTIVE_ORIGIN_Q.copy()
    q0[0] = q_pan_fixed
    set_q(model, data, q0)
    mujoco.mj_jacSite(model, data, jacp, jacr, site_id)
    # First 6 columns correspond to the six hinge joints when nv==6
    jacp6 = jacp[:, :6].copy()
    jacr6 = jacr[:, :6].copy()
    raw_dx_dq = jacp6[0, :].tolist()  # m per rad

    # Reduced: pan velocity = 0
    g = jacp6[0, 1:6]  # (5,)
    Jr = jacr6[:, 1:6]  # (3,5)
    P = projector_onto_null(Jr)
    grad_feas = P @ g
    norm_feas = float(np.linalg.norm(grad_feas))
    if norm_feas > 1e-12:
        ascent_unit = (grad_feas / norm_feas).tolist()
    else:
        ascent_unit = [0.0] * 5

    q_lower = np.array([float(model.jnt_range[j][0]) for j in range(6)], dtype=np.float64)
    q_upper = np.array([float(model.jnt_range[j][1]) for j in range(6)], dtype=np.float64)

    set_q(model, data, q0)
    x_ref = float(data.site_xpos[site_id][0])

    # Finite-difference check: move along feasible ascent direction dq = P g (in q1..q5), scale so ||dq||=eps
    eps = 1e-3
    dq_red = grad_feas.copy()
    nrm = float(np.linalg.norm(dq_red))
    if nrm > 1e-12:
        dq_red *= eps / nrm
    dq6 = np.zeros(6, dtype=np.float64)
    dq6[1:6] = dq_red
    q_feas = np.clip(q0 + dq6, q_lower, q_upper)
    q_feas[0] = q_pan_fixed
    set_q(model, data, q_feas)
    x_feas = float(data.site_xpos[site_id][0])
    feasible_directional_dx_dq = (x_feas - x_ref) / eps  # ≈ ||P g|| m/(rad aggregate step)

    return {
        "scene_xml": str(scene_path()),
        "tool_site": TOOL_SITE_NAME,
        "shoulder_pan_fixed_rad": q_pan_fixed,
        "origin_target_q": q0.tolist(),
        "reference_world_x_m": x_ref,
        "slsqp": {
            "min_world_x_m": None if best_min[1] is None else float(best_min[0]),
            "max_world_x_m": None if best_max[1] is None else float(best_max[0]),
            "argmin_y": None if best_min[1] is None else np.concatenate([[q_pan_fixed], best_min[1]]).tolist(),
            "argmax_y": None if best_max[1] is None else np.concatenate([[q_pan_fixed], best_max[1]]).tolist(),
            "x_span_m": None
            if best_min[1] is None or best_max[1] is None
            else float(best_max[0] - best_min[0]),
        },
        "per_joint": [
            {
                "joint": JOINT_NAMES[i],
                "index": i,
                "range_rad": [float(model.jnt_range[i][0]), float(model.jnt_range[i][1])],
                "range_deg": [float(np.degrees(model.jnt_range[i][0])), float(np.degrees(model.jnt_range[i][1]))],
                "raw_dx_dq_m_per_rad": float(raw_dx_dq[i]),
                "raw_dx_per_deg_mm": float(raw_dx_dq[i]) * np.deg2rad(1.0) * 1000.0,
            }
            for i in range(6)
        ],
        "constrained_null_projector": {
            "gradient_world_x_in_null_space_joint1_to_5": grad_feas.tolist(),
            "norm_m_per_rad_aggregate_step": norm_feas,
            "finite_diff_check_m_per_rad": float(feasible_directional_dx_dq),
            "ascent_direction_unit_joint1_to_5": ascent_unit,
            "note": "At this pose, max ẋ under ω=0 (1st order) with ||qdot_{1:5}||=1 is norm_feas (m/s per rad/s Euclidean on joints 1–5). Components are a joint mix, not independent ∂x/∂q_i.",
        },
    }


def run_full_workspace_study(optimization_seed: int = 0) -> dict:
    """Load MuJoCo scene and return the full JSON-serializable report (CLI + ROS service)."""
    path = scene_path()
    model = mujoco.MjModel.from_xml_path(str(path))
    data = mujoco.MjData(model)
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, TOOL_SITE_NAME)
    return build_study(
        model,
        data,
        site_id,
        q_pan_fixed=float(ACTIVE_ORIGIN_Q[0]),
        optimization_seed=optimization_seed,
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--json-out",
        type=Path,
        default=BASE_DIR / "outputs" / "control_runs" / "constrained_ee_x_workspace.json",
        help="Where to write the numeric report.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=0,
        help="RNG seed for multi-start SLSQP extra samples (default 0).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    path = scene_path()
    report = run_full_workspace_study(optimization_seed=int(args.seed))
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(report, indent=2))

    s = report["slsqp"]
    print(f"Scene: {path.name}  seed={args.seed}")
    print(f"Tool site: {TOOL_SITE_NAME} world X (m), pan fixed at {report['shoulder_pan_fixed_rad']:.6f} rad")
    if s["x_span_m"] is not None:
        print(
            f"Constrained range (orientation + joint limits): "
            f"x in [{s['min_world_x_m']:.6f}, {s['max_world_x_m']:.6f}] m  span = {s['x_span_m']:.6f} m"
        )
    else:
        print("SLSQP did not find feasible min/max from all seeds (check constraints / limits).")

    cn = report["constrained_null_projector"]
    print(
        "\nFeasible instantaneous world-X rate (orientation fixed 1st order, pan locked):\n"
        f"  ||P grad x|| = {cn['norm_m_per_rad_aggregate_step']:.6f} m per rad/s "
        f"(unit Euclidean ||qdot_1:5||)"
    )
    print(
        "  Joint mix (unit direction, indices 1..5 = lift, elbow, wrist_1, wrist_2, wrist_3):\n   "
        + np.array2string(np.array(cn["ascent_direction_unit_joint1_to_5"]), precision=4, separator=", ")
    )

    print("\nPer joint at canonical ACTIVE_ORIGIN_Q (raw ∂x/∂q_i, orientation NOT held):")
    print(f"{'joint':<22} {'mm/deg':>12}")
    for row in report["per_joint"]:
        print(f"{row['joint']:<22} {row['raw_dx_per_deg_mm']:12.4f}")

    print(f"\nWrote JSON: {args.json_out}")


if __name__ == "__main__":
    main()
