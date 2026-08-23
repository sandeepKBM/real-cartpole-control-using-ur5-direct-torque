#!/usr/bin/env python3
"""Search the 5 non-base UR5e joints (shoulder_lift, elbow, wrist_1, wrist_2,
wrist_3) for a pose that tolerates sustained oscillation well, with
shoulder_pan held FIXED at a caller-supplied value (real-lab constraint: the
base joint must not rotate).

CORRECTED 2026-08-12: this used to test both transport_axis_index=0 (world-X)
and =1 (world-Y) and keep whichever won. That was invalid -- traced directly
into controller_core/x_axis_cartesian_impedance/controller.py::compute() and
confirmed it hardcodes `p[0]`/`v[0]` (literal world-X) with no axis parameter
anywhere; `transport_axis_index` only ever affects what target value gets
computed and which axis the safety monitor calls "orthogonal" -- it never
redirects the controller's actual force law. So every earlier "Y-axis" trial
in this repo was a broken configuration (target computed from Y's coordinate,
error/force computed against real X), not a fair test of Y transport. Only
world-X is real. This script now only ever tests axis 0, and instead of
picking whichever axis aligns better after the fact, it REQUIRES the
candidate pose's kinematic alignment between world-X and the true
"away-from-base" radial direction to clear a minimum threshold up front
(--align-x-min), since X is the only axis that can actually be driven.

Same DE-search / multiprocessing-hygiene pattern as this repo's other search
scripts: gradient-free (differential_evolution), never RL; bounded worker
pool via _de_workers() (fork start method, ~90% of cores, DE_WORKERS
override); if __name__ == "__main__" guard is required so scipy's forkserver
fallback can't re-import this module into an exponential fork bomb.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import mujoco
import numpy as np
import yaml
from scipy.optimize import differential_evolution

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from simulation.ur5e_mujoco_torque import (  # noqa: E402
    build_initial_state_and_adapter,
    build_mujoco_state,
)
from tools.diagnostics.pendulum_swingup_multi_kick import _de_workers  # noqa: E402

CONFIG_PATH = REPO_ROOT / "config" / "ur5e_mujoco_torque_osc_hanging_pose_friction_ff.yaml"
SCENE_XML = REPO_ROOT / "assets" / "ur5e_torque" / "scene.xml"
RATE_HZ = 500.0
CONTROL_DT = 1.0 / RATE_HZ
KICK_AMPLITUDE_M = 0.08
KICK_DURATION_S = 0.22
N_KICKS = 6
HOLD_GAP_S = 0.15
JOINT_NAMES = [
    "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
    "wrist_1_joint", "wrist_2_joint", "wrist_3_joint",
]


def cheap_kinematics(q6: np.ndarray) -> tuple[float, float, float, float]:
    """cond(J), EE height, tool-face tilt (site local-Z axis's world-Z
    component -- 0 means the face normal is perfectly horizontal, i.e.
    "perpendicular to the base ground frame" per the real-lab requirement
    confirmed against two live safe-pose snapshots on 2026-08-12), and
    align_x (dot product of the EE's radial away-from-base direction in the
    XY plane with world +X -- magnitude 1.0 is perfect alignment, 0.0 is
    perpendicular). No dynamics -- fast prefilter.
    """
    model = mujoco.MjModel.from_xml_path(str(SCENE_XML))
    data = mujoco.MjData(model)
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "attachment_site")
    data.qpos[:6] = q6
    mujoco.mj_forward(model, data)
    ee = data.site_xpos[site_id].copy()
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    mujoco.mj_jacSite(model, data, jacp, jacr, site_id)
    J = np.vstack([jacp[:, :6], jacr[:, :6]])
    cond = float(np.linalg.cond(J))
    R = data.site_xmat[site_id].reshape(3, 3)
    face_tilt = float(R[2, 2])  # world-Z component of local-Z (face normal) axis
    xy_norm = float(np.linalg.norm(ee[:2]))
    align_x = float(ee[0] / xy_norm) if xy_norm > 1e-6 else 0.0
    return cond, float(ee[2]), face_tilt, align_x


def run_trial(q6: np.ndarray, transport_axis_index: int) -> dict:
    with CONFIG_PATH.open() as fp:
        config = yaml.safe_load(fp)
    model = mujoco.MjModel.from_xml_path(str(SCENE_XML))
    data = mujoco.MjData(model)
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "attachment_site")
    joint_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in JOINT_NAMES]
    data.qpos[:6] = q6
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)

    state0, adapter = build_initial_state_and_adapter(
        model, data, site_id, joint_ids,
        controller_cfg=config["controller"],
        transport_axis_index=transport_axis_index,
        target_x_delta=0.0,
        controller_kind="impedance",
        force_hold_current_pose=False,
        gravity_mode=config["mujoco"].get("gravity_mode", "gravity_comp"),
        gravity_source=config["mujoco"].get("gravity_source", "pinocchio"),
        coriolis_feedforward=bool(config["mujoco"].get("coriolis_feedforward", True)),
        torque_limit_scale=1.0,
    )
    x0 = float(state0.ee_pos[transport_axis_index])

    kicks_survived = 0
    peak_speed = 0.0
    guard_reason = None
    t_global = 0.0

    for kick_idx in range(N_KICKS):
        sign = 1.0 if kick_idx % 2 == 0 else -1.0
        n_steps = int(KICK_DURATION_S * RATE_HZ)
        kick_ok = True
        for step in range(n_steps):
            tau_local = step * CONTROL_DT
            omega_kick = 2.0 * np.pi / KICK_DURATION_S
            target_x = x0 + sign * 0.5 * KICK_AMPLITUDE_M * (1.0 - np.cos(omega_kick * tau_local))
            target_x_vel = sign * 0.5 * KICK_AMPLITUDE_M * omega_kick * np.sin(omega_kick * tau_local)
            a_cmd = float(sign * 0.5 * KICK_AMPLITUDE_M * omega_kick * omega_kick * np.cos(omega_kick * tau_local))
            state = build_mujoco_state(
                model, data, site_id=site_id, joint_ids=joint_ids,
                time_s=t_global, dt_s=CONTROL_DT,
                target_x=target_x, target_x_vel=target_x_vel, target_x_accel=a_cmd,
                reference_quat=state0.ee_quat, transport_axis_index=transport_axis_index,
                gravity_compensation=True,
            )
            tau, diag = adapter.step(state=state)
            if not bool(diag.get("safety_ok", True)):
                kick_ok = False
                guard_reason = str(diag.get("safety_reason", ""))
                break
            data.ctrl[:6] = np.asarray(tau, dtype=np.float64).reshape(6)
            mujoco.mj_step(model, data)
            t_global += CONTROL_DT
            jacp = np.zeros((3, model.nv))
            jacr = np.zeros((3, model.nv))
            mujoco.mj_jacSite(model, data, jacp, jacr, site_id)
            speed = float(np.linalg.norm(jacp[:, :6] @ data.qvel[:6]))
            peak_speed = max(peak_speed, speed)
        if not kick_ok:
            break
        kicks_survived += 1
        hold_steps = int(HOLD_GAP_S * RATE_HZ)
        hold_ok = True
        for _ in range(hold_steps):
            state = build_mujoco_state(
                model, data, site_id=site_id, joint_ids=joint_ids,
                time_s=t_global, dt_s=CONTROL_DT,
                target_x=x0, target_x_vel=0.0, target_x_accel=0.0,
                reference_quat=state0.ee_quat, transport_axis_index=transport_axis_index,
                gravity_compensation=True,
            )
            tau, diag = adapter.step(state=state)
            if not bool(diag.get("safety_ok", True)):
                hold_ok = False
                guard_reason = str(diag.get("safety_reason", ""))
                break
            data.ctrl[:6] = np.asarray(tau, dtype=np.float64).reshape(6)
            mujoco.mj_step(model, data)
            t_global += CONTROL_DT
        if not hold_ok:
            break

    return {
        "kicks_survived": kicks_survived, "n_kicks_target": N_KICKS,
        "peak_speed_mps": peak_speed, "guard_reason": guard_reason,
    }


def wrapped_deg_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Sum of |shortest-path angular distance| in degrees, elementwise."""
    raw = np.asarray(a) - np.asarray(b)
    wrapped = (raw + np.pi) % (2 * np.pi) - np.pi
    return float(np.degrees(np.abs(wrapped)).sum())


def objective(
    x: np.ndarray,
    cond_prefilter_max: float,
    current_q6: np.ndarray | None,
    travel_penalty_weight_per_deg: float,
    face_tilt_max: float,
    align_x_min: float,
) -> float:
    q6 = np.asarray(x, dtype=np.float64)
    cond, height, face_tilt, align_x = cheap_kinematics(q6)
    if not np.isfinite(cond) or cond > cond_prefilter_max or not (0.05 <= height <= 1.0):
        return 500.0 + min(cond, 1e6) * 0.01 if np.isfinite(cond) else 1e6
    if abs(face_tilt) > face_tilt_max:
        return 300.0 + abs(face_tilt) * 100.0
    if abs(align_x) < align_x_min:
        # Only world-X is a real, drivable transport axis (see module
        # docstring) -- a candidate whose radial away-from-base direction
        # isn't reasonably aligned with X can't be fixed by axis selection,
        # so reject it outright rather than scoring it on a dynamic test
        # that would drive the wrong direction relative to the wall.
        return 200.0 + (align_x_min - abs(align_x)) * 100.0
    result = run_trial(q6, transport_axis_index=0)
    cost = (N_KICKS - result["kicks_survived"]) * 100.0 - result["peak_speed_mps"] + cond * 0.01
    if current_q6 is not None:
        cost += travel_penalty_weight_per_deg * wrapped_deg_distance(q6, current_q6)
    return float(cost)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--shoulder-pan-bounds", type=float, nargs=2, required=True,
                    metavar=("LOW", "HIGH"),
                    help="Search bounds for shoulder_pan (rad) -- e.g. current +/- 20deg "
                         "(0.349rad) if a small base rotation is now acceptable. Pass the "
                         "same value twice to fully fix it, matching the old --shoulder-pan-rad.")
    p.add_argument("--maxiter", type=int, default=20)
    p.add_argument("--popsize", type=int, default=12)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--cond-prefilter-max", type=float, default=50.0)
    p.add_argument("--face-tilt-max", type=float, default=0.12,
                    help="Max |world-Z component| of the tool-face local-Z axis -- 0 is "
                         "perfectly horizontal (perpendicular to ground). Default 0.12 "
                         "(~7deg) gives margin above the ~0.07-0.08 (~4-5deg) measured on "
                         "two live real-hardware reference snapshots confirmed safe.")
    p.add_argument("--current-q-rad", type=float, nargs=6, default=None,
                    metavar=("PAN", "SL", "EL", "W1", "W2", "W3"),
                    help="Current real full 6-joint pose in radians -- if given, penalizes "
                         "candidates by total shortest-path angular distance from this pose "
                         "(favors a smaller real move).")
    p.add_argument("--travel-penalty-weight-per-deg", type=float, default=0.02,
                    help="Cost added per total degree of travel from --current-q-rad. "
                         "Default 0.02: 100deg of travel costs ~2, comparable to a fraction "
                         "of one missed kick (100/kick) -- biases toward smaller moves among "
                         "comparably-stable candidates without overriding actual stability.")
    p.add_argument("--output-json", type=Path, default=None)
    p.add_argument("--shoulder-lift-bounds", type=float, nargs=2, default=(-3.14, 3.14),
                    metavar=("LOW", "HIGH"),
                    help="Hard search bounds for shoulder_lift (rad) -- e.g. a real-lab "
                         "wall-clearance limit measured from a live probe.")
    p.add_argument("--elbow-bounds", type=float, nargs=2, default=(-3.14, 3.14),
                    metavar=("LOW", "HIGH"))
    p.add_argument("--align-x-min", type=float, default=0.9,
                    help="Minimum required |alignment| between world-X and the true "
                         "away-from-base radial direction (1.0=perfect, 0.0=perpendicular). "
                         "Only world-X is a real, drivable transport axis (Y is not "
                         "implemented in the live controller -- see module docstring), so "
                         "candidates below this are rejected outright, not just penalized.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    bounds = [
        tuple(args.shoulder_pan_bounds),    # shoulder_pan
        tuple(args.shoulder_lift_bounds),   # shoulder_lift
        tuple(args.elbow_bounds),           # elbow
        (-3.14, 3.14),  # wrist_1
        (-3.14, 3.14),  # wrist_2
        (-3.14, 3.14),  # wrist_3
    ]
    print(f"shoulder_pan bounds: {np.degrees(bounds[0]).round(2).tolist()} deg")
    print(f"shoulder_lift bounds: {np.degrees(bounds[1]).round(2).tolist()} deg")
    print(f"elbow bounds: {np.degrees(bounds[2]).round(2).tolist()} deg")
    current_q6 = None if args.current_q_rad is None else np.asarray(args.current_q_rad, dtype=np.float64)
    if current_q6 is not None:
        print(f"Travel penalty active: {args.travel_penalty_weight_per_deg}/deg from "
              f"current pose {np.degrees(current_q6).round(2).tolist()}")
    print(f"Requiring align_x >= {args.align_x_min} (only world-X is a real transport axis)")
    print("Searching (shoulder_pan, shoulder_lift, elbow, wrist_1, wrist_2, wrist_3) via "
          f"differential_evolution, axis=X only, workers={_de_workers()}...")
    res = differential_evolution(
        objective, bounds,
        args=(args.cond_prefilter_max, current_q6,
              args.travel_penalty_weight_per_deg, args.face_tilt_max, args.align_x_min),
        maxiter=args.maxiter, popsize=args.popsize, tol=1e-4, seed=args.seed,
        workers=_de_workers(), polish=False, updating="deferred",
    )
    q_best = np.asarray(res.x, dtype=np.float64)
    cond, height, face_tilt, align_x = cheap_kinematics(q_best)
    result = run_trial(q_best, transport_axis_index=0)

    print(f"\nBest cost: {res.fun:.4f}")
    print(f"Best q (rad): {q_best.tolist()}")
    print(f"Best q (deg): {np.degrees(q_best).round(2).tolist()}")
    print(f"cond(J) = {cond:.2f}, EE height = {height:.3f} m, face_tilt = {face_tilt:+.4f} "
          f"({np.degrees(np.arcsin(np.clip(abs(face_tilt),0,1))):.2f} deg off horizontal)")
    print(f"align_x = {align_x:+.4f} ({np.degrees(np.arccos(np.clip(abs(align_x),0,1))):.2f} deg off X)")
    if current_q6 is not None:
        print(f"Total travel from current (deg): {wrapped_deg_distance(q_best, current_q6):.2f}")
    print(f"X-axis result: {result}")

    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with args.output_json.open("w") as fp:
            json.dump({
                "shoulder_pan_bounds_rad": list(args.shoulder_pan_bounds),
                "best_q_rad": q_best.tolist(),
                "best_cost": float(res.fun),
                "cond_j": cond,
                "ee_height_m": height,
                "face_tilt": face_tilt,
                "align_x": align_x,
                "x_result": result,
                "travel_from_current_deg": (
                    None if current_q6 is None else wrapped_deg_distance(q_best, current_q6)
                ),
            }, fp, indent=2)
        print(f"Wrote {args.output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
