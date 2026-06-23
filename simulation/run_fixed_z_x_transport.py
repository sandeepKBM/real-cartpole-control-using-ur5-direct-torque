#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

if "MUJOCO_GL" not in os.environ:
    os.environ["MUJOCO_GL"] = "egl" if not os.environ.get("DISPLAY") else "glfw"

import mediapy
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.interpolate import PchipInterpolator
from progress_bars import simulation_progress

from controller import ACTIVE_ORIGIN_Q, TARGET_SITE_ROTATION_WORLD
from demo_ee_x_limits_at_z import solve_q_at_xz, wrap_joint_delta
from probe_workspace_xz_envelope import (
    TOOL_SITE_NAME,
    _optimize_x_at_z_one,
    collect_slice_seeds,
    global_max_site_z_above_floor,
    joint_y_bounds,
    q_from_y,
    scene_path,
)
from workspace_guardrails import (
    DEFAULT_GUARDRAIL_CONFIG,
    boundary_summary,
    check_tcp_pose,
    load_guardrail_config,
    overlay_guardrails_on_frame,
)


SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parent
VIDEO_OUTPUT_DIR = BASE_DIR / "demonstration_videos" / "ur5e_cartpole"
SUMMARY_OUTPUT_DIR = BASE_DIR / "outputs" / "control_runs"
BASE_PAN_INDEX = 0
REDUCED_JOINT_INDICES = np.array([1, 2, 3, 4, 5], dtype=np.int64)
REDUCED_JOINT_NAMES = [
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]

# Conservative simulation-side tracking limits used to slow the reference path
# before it reaches the low-level MuJoCo joint servos.
DEFAULT_JOINT_VEL_LIMIT_RAD_S = np.array([0.35, 0.40, 0.50, 0.65, 0.65], dtype=np.float64)
DEFAULT_JOINT_ACCEL_LIMIT_RAD_S2 = np.array([0.80, 0.90, 1.10, 1.40, 1.40], dtype=np.float64)
DEFAULT_JOINT_JERK_LIMIT_RAD_S3 = np.array([5.0, 5.5, 6.5, 8.0, 8.0], dtype=np.float64)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--z",
        type=float,
        default=0.54,
        help="Fixed attachment_site world Z (m).",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=12.0,
        help=(
            "Requested total simulation duration in seconds. "
            "The planner may extend the motion phase automatically to satisfy the joint limits."
        ),
    )
    parser.add_argument(
        "--hold-duration",
        type=float,
        default=1.0,
        help="Hold duration at the start and end of the transport motion.",
    )
    parser.add_argument("--fps", type=int, default=30, help="Video frame rate.")
    parser.add_argument("--width", type=int, default=960, help="Render width.")
    parser.add_argument("--height", type=int, default=720, help="Render height.")
    parser.add_argument(
        "--x-margin",
        type=float,
        default=0.05,
        help="Stay this far inside the exact X extrema to avoid boundary singularities.",
    )
    parser.add_argument(
        "--start-side",
        choices=["min", "max"],
        default="min",
        help="Which side of the reachable X segment to start from.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Seed for SLSQP restart sampling.")
    parser.add_argument("--slsqp-seeds", type=int, default=24, help="Number of SLSQP restart seeds.")
    parser.add_argument("--slsqp-maxiter", type=int, default=600, help="SLSQP iteration cap.")
    parser.add_argument(
        "--path-poses",
        type=int,
        default=48,
        help="Number of continuity-aware joint waypoints to solve across the usable X interval.",
    )
    parser.add_argument(
        "--smooth-accel-weight",
        type=float,
        default=0.20,
        help="Second-order smoothness weight used while solving the fixed-Z joint path.",
    )
    parser.add_argument(
        "--joint-vel-scale",
        type=float,
        default=1.0,
        help="Scale factor applied to the default reduced-joint velocity limits.",
    )
    parser.add_argument(
        "--joint-accel-scale",
        type=float,
        default=1.0,
        help="Scale factor applied to the default reduced-joint acceleration limits.",
    )
    parser.add_argument(
        "--joint-jerk-scale",
        type=float,
        default=1.0,
        help="Scale factor applied to the default reduced-joint jerk limits.",
    )
    parser.add_argument(
        "--video-name",
        type=str,
        default=None,
        help="Optional custom output video file name.",
    )
    parser.add_argument(
        "--json-name",
        type=str,
        default=None,
        help="Optional custom output JSON file name.",
    )
    parser.add_argument(
        "--no-video",
        action="store_true",
        help="Only write JSON and skip MP4 rendering.",
    )
    parser.add_argument(
        "--draw-guardrails",
        action="store_true",
        help="Draw the extracted lab workspace guardrails as a 2D inset on each video frame.",
    )
    parser.add_argument(
        "--guardrail-config",
        type=Path,
        default=DEFAULT_GUARDRAIL_CONFIG,
        help="YAML config produced from the external Einksul scene.",
    )
    parser.add_argument(
        "--guardrail-margin-m",
        type=float,
        default=0.02,
        help="Additional conservative margin applied when checking the trajectory.",
    )
    parser.add_argument(
        "--show-boundary-labels",
        action="store_true",
        help="Annotate the boundary inset with names.",
    )
    return parser.parse_args()


def make_camera(model: mujoco.MjModel) -> mujoco.MjvCamera:
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    mujoco.mjv_defaultFreeCamera(model, camera)
    camera.azimuth = 12.0
    camera.elevation = -18.0
    camera.distance = 2.05
    return camera


def annotate_frame(frame: np.ndarray, lines: list[str]) -> np.ndarray:
    image = Image.fromarray(frame)
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    font = ImageFont.load_default()

    line_height = 16
    panel_width = min(680, image.size[0] - 20)
    panel_height = 18 + line_height * len(lines)
    draw.rounded_rectangle(
        (10, 10, 10 + panel_width, 10 + panel_height),
        radius=10,
        fill=(0, 0, 0, 165),
        outline=(255, 255, 255, 64),
    )

    y = 18
    for line in lines:
        draw.text((20, y), line, fill=(255, 255, 255, 255), font=font)
        y += line_height

    combined = Image.alpha_composite(image.convert("RGBA"), overlay)
    return np.asarray(combined.convert("RGB"))


def xmat_to_rot(xmat: np.ndarray) -> np.ndarray:
    return np.asarray(xmat, dtype=np.float64).reshape(3, 3)


def rotation_matrix_to_rpy_deg(rot: np.ndarray) -> np.ndarray:
    sy = float(np.sqrt(rot[0, 0] ** 2 + rot[1, 0] ** 2))
    singular = sy < 1e-8

    if not singular:
        roll = np.arctan2(rot[2, 1], rot[2, 2])
        pitch = np.arctan2(-rot[2, 0], sy)
        yaw = np.arctan2(rot[1, 0], rot[0, 0])
    else:
        roll = np.arctan2(-rot[1, 2], rot[1, 1])
        pitch = np.arctan2(-rot[2, 0], sy)
        yaw = 0.0

    return np.rad2deg(np.array([roll, pitch, yaw], dtype=np.float64))


def orientation_error_deg(rot: np.ndarray, rot_target: np.ndarray) -> float:
    rel = rot_target.T @ rot
    cos_theta = np.clip((float(np.trace(rel)) - 1.0) * 0.5, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_theta)))


def minimum_jerk_profile(tau: np.ndarray | float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    tau = np.asarray(tau, dtype=np.float64)
    tau = np.clip(tau, 0.0, 1.0)
    alpha = tau**3 * (10.0 - 15.0 * tau + 6.0 * tau**2)
    d1 = 30.0 * tau**2 - 60.0 * tau**3 + 30.0 * tau**4
    d2 = 60.0 * tau - 180.0 * tau**2 + 120.0 * tau**3
    d3 = 60.0 - 360.0 * tau + 360.0 * tau**2
    return alpha, d1, d2, d3


def continuous_reduced_joint_path(y_path: np.ndarray) -> np.ndarray:
    y_path = np.asarray(y_path, dtype=np.float64)
    if len(y_path) <= 1:
        return y_path.copy()

    out = y_path.copy()
    for idx in range(1, len(out)):
        out[idx] = out[idx - 1] + wrap_joint_delta(out[idx] - out[idx - 1])
    return out


def recenter_continuous_reduced_path_within_limits(
    y_path_continuous: np.ndarray,
    ctrl_lower: np.ndarray,
    ctrl_upper: np.ndarray,
) -> np.ndarray:
    y_path_continuous = np.asarray(y_path_continuous, dtype=np.float64)
    out = y_path_continuous.copy()

    for reduced_idx, joint_idx in enumerate(REDUCED_JOINT_INDICES):
        lo = float(ctrl_lower[joint_idx])
        hi = float(ctrl_upper[joint_idx])
        best_shifted = out[:, reduced_idx].copy()
        best_score: tuple[float, float] | None = None

        for k in range(-4, 5):
            shifted = out[:, reduced_idx] + 2.0 * np.pi * float(k)
            if float(np.min(shifted)) < lo - 1e-6 or float(np.max(shifted)) > hi + 1e-6:
                continue

            margin = min(float(np.min(shifted - lo)), float(np.min(hi - shifted)))
            mean_abs = float(np.mean(np.abs(shifted)))
            score = (margin, -mean_abs)
            if best_score is None or score > best_score:
                best_score = score
                best_shifted = shifted

        out[:, reduced_idx] = best_shifted

    return out


def reduced_path_smoothness_summary(y_path: np.ndarray) -> dict:
    y_path = np.asarray(y_path, dtype=np.float64)
    joint_metric_weights = np.ones(len(REDUCED_JOINT_INDICES), dtype=np.float64)

    if len(y_path) >= 2:
        dq = wrap_joint_delta(np.diff(y_path, axis=0))
        step_costs = np.sum(joint_metric_weights[None, :] * (dq * dq), axis=1)
        step_norms = np.sqrt(step_costs)
        max_step_per_joint = np.max(np.abs(dq), axis=0)
    else:
        dq = np.zeros((0, len(REDUCED_JOINT_INDICES)), dtype=np.float64)
        step_costs = np.zeros(0, dtype=np.float64)
        step_norms = np.zeros(0, dtype=np.float64)
        max_step_per_joint = np.zeros(len(REDUCED_JOINT_INDICES), dtype=np.float64)

    if len(y_path) >= 3:
        ddq = wrap_joint_delta(y_path[2:] - 2.0 * y_path[1:-1] + y_path[:-2])
        accel_costs = np.sum(joint_metric_weights[None, :] * (ddq * ddq), axis=1)
        accel_norms = np.sqrt(accel_costs)
        max_accel_per_joint = np.max(np.abs(ddq), axis=0)
    else:
        accel_costs = np.zeros(0, dtype=np.float64)
        accel_norms = np.zeros(0, dtype=np.float64)
        max_accel_per_joint = np.zeros(len(REDUCED_JOINT_INDICES), dtype=np.float64)

    return {
        "metric": "weighted_joint_step_plus_second_difference_over_solved_waypoints",
        "joint_names_reduced": REDUCED_JOINT_NAMES,
        "joint_weights": joint_metric_weights.tolist(),
        "waypoint_count": int(len(y_path)),
        "mean_step_norm_rad": float(np.mean(step_norms)) if step_norms.size else 0.0,
        "max_step_norm_rad": float(np.max(step_norms)) if step_norms.size else 0.0,
        "mean_step_cost": float(np.mean(step_costs)) if step_costs.size else 0.0,
        "max_step_per_joint_rad": max_step_per_joint.tolist(),
        "mean_second_difference_norm_rad": float(np.mean(accel_norms)) if accel_norms.size else 0.0,
        "max_second_difference_norm_rad": float(np.max(accel_norms)) if accel_norms.size else 0.0,
        "mean_second_difference_cost": float(np.mean(accel_costs)) if accel_costs.size else 0.0,
        "max_second_difference_per_joint_rad": max_accel_per_joint.tolist(),
    }


def solve_path_pose_with_restarts(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    site_id: int,
    x_target: float,
    z_target: float,
    q_pan: float,
    bounds: list[tuple[float, float]],
    y_prev: np.ndarray,
    y_prev_prev: np.ndarray | None,
    y_anchor_a: np.ndarray,
    y_anchor_b: np.ndarray,
    z_floor_world_m: float,
    seed: int,
    slsqp_maxiter: int,
    smooth_accel_weight: float,
    random_restart_count: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    lower = np.array([lo for lo, _ in bounds], dtype=np.float64)
    upper = np.array([hi for _, hi in bounds], dtype=np.float64)
    joint_weights = np.ones(len(bounds), dtype=np.float64)

    if y_prev_prev is not None:
        predicted = np.clip(y_prev + wrap_joint_delta(y_prev - y_prev_prev), lower, upper)
    else:
        predicted = np.asarray(y_prev, dtype=np.float64).copy()

    tries: list[np.ndarray] = [
        predicted,
        np.asarray(y_prev, dtype=np.float64).copy(),
        np.asarray(y_anchor_a, dtype=np.float64).copy(),
        np.asarray(y_anchor_b, dtype=np.float64).copy(),
        0.5 * (np.asarray(y_anchor_a, dtype=np.float64) + np.asarray(y_anchor_b, dtype=np.float64)),
        ACTIVE_ORIGIN_Q[1:6].copy(),
    ]
    for _ in range(max(8, int(random_restart_count))):
        tries.append(np.array([rng.uniform(lo, hi) for lo, hi in bounds], dtype=np.float64))

    candidates: list[tuple[float, np.ndarray]] = []
    for y0 in tries:
        ok, y_sol, cost = solve_q_at_xz(
            model=model,
            data=data,
            site_id=site_id,
            x_target=float(x_target),
            z_target=float(z_target),
            q_pan=float(q_pan),
            bounds=bounds,
            y0=np.asarray(y0, dtype=np.float64),
            y_prev=np.asarray(y_prev, dtype=np.float64),
            z_floor=float(z_floor_world_m),
            maxiter=int(slsqp_maxiter),
            joint_weights=joint_weights,
            y_prev_prev=None if y_prev_prev is None else np.asarray(y_prev_prev, dtype=np.float64),
            accel_weight=float(smooth_accel_weight),
        )
        if ok:
            candidates.append((float(cost), y_sol.copy()))

    if not candidates:
        raise RuntimeError(
            f"Could not solve a continuous waypoint at x={x_target:.6f} m, z={z_target:.6f} m."
        )

    candidates.sort(key=lambda item: item[0])
    return candidates[0][1]


def build_continuous_joint_path(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    site_id: int,
    x_start: float,
    x_stop: float,
    z_target: float,
    q_pan: float,
    bounds: list[tuple[float, float]],
    y_start: np.ndarray,
    y_stop: np.ndarray,
    z_floor_world_m: float,
    seed: int,
    slsqp_maxiter: int,
    path_poses: int,
    smooth_accel_weight: float,
    random_restart_count: int,
    max_refinement_depth: int = 5,
) -> tuple[np.ndarray, np.ndarray, dict]:
    x_targets_nominal = np.linspace(float(x_start), float(x_stop), max(2, int(path_poses)))
    x_waypoints: list[float] = [float(x_targets_nominal[0])]
    y_waypoints: list[np.ndarray] = [np.asarray(y_start, dtype=np.float64).copy()]
    y_prev = np.asarray(y_start, dtype=np.float64).copy()
    y_prev_prev: np.ndarray | None = None

    print(f"Planning {len(x_targets_nominal)} continuity-aware fixed-Z waypoints across X...")
    report_every = max(1, len(x_targets_nominal) // 12)

    def advance_to_target(
        x_current: float,
        y_current: np.ndarray,
        y_current_prev: np.ndarray | None,
        x_target: float,
        seed_offset: int,
        depth: int,
    ) -> list[tuple[float, np.ndarray]]:
        try:
            y_target = solve_path_pose_with_restarts(
                model=model,
                data=data,
                site_id=site_id,
                x_target=float(x_target),
                z_target=float(z_target),
                q_pan=float(q_pan),
                bounds=bounds,
                y_prev=y_current,
                y_prev_prev=y_current_prev,
                y_anchor_a=y_start,
                y_anchor_b=y_stop,
                z_floor_world_m=float(z_floor_world_m),
                seed=int(seed) + int(seed_offset),
                slsqp_maxiter=int(slsqp_maxiter) + 200,
                smooth_accel_weight=float(smooth_accel_weight),
                random_restart_count=int(random_restart_count),
            )
            return [(float(x_target), y_target.copy())]
        except RuntimeError as err:
            if depth >= int(max_refinement_depth) or abs(float(x_target) - float(x_current)) < 1e-3:
                raise err
            x_mid = 0.5 * (float(x_current) + float(x_target))
            print(
                "  refining segment "
                f"[{float(x_current):.5f}, {float(x_target):.5f}] at depth {depth + 1}"
            )
            left_segments = advance_to_target(
                x_current=float(x_current),
                y_current=y_current,
                y_current_prev=y_current_prev,
                x_target=float(x_mid),
                seed_offset=int(seed_offset) + 17,
                depth=depth + 1,
            )
            x_mid_solved, y_mid = left_segments[-1]
            y_mid_prev = y_current if len(left_segments) == 1 else left_segments[-2][1]
            right_segments = advance_to_target(
                x_current=float(x_mid_solved),
                y_current=y_mid,
                y_current_prev=y_mid_prev,
                x_target=float(x_target),
                seed_offset=int(seed_offset) + 43,
                depth=depth + 1,
            )
            return left_segments + right_segments

    for idx, x_target in enumerate(x_targets_nominal[1:], start=1):
        new_segments = advance_to_target(
            x_current=float(x_waypoints[-1]),
            y_current=y_prev,
            y_current_prev=y_prev_prev,
            x_target=float(x_target),
            seed_offset=7919 * idx,
            depth=0,
        )
        for x_sol, y_sol in new_segments:
            x_waypoints.append(float(x_sol))
            y_waypoints.append(y_sol.copy())
            y_prev_prev = y_prev.copy()
            y_prev = y_sol.copy()

        if idx == 1 or (idx + 1) % report_every == 0 or idx + 1 == len(x_targets_nominal):
            print(
                f"  nominal target {idx + 1}/{len(x_targets_nominal)}  "
                f"X={float(x_target):.5f} m  actual waypoints={len(x_waypoints)}"
            )

    x_waypoints_arr = np.asarray(x_waypoints, dtype=np.float64)
    y_waypoints_arr = np.asarray(y_waypoints, dtype=np.float64)
    summary = reduced_path_smoothness_summary(y_waypoints_arr)
    summary["nominal_waypoint_count"] = int(len(x_targets_nominal))
    summary["actual_waypoint_count"] = int(len(x_waypoints_arr))
    return x_waypoints_arr, y_waypoints_arr, summary


def fit_reduced_joint_splines(y_waypoints_continuous: np.ndarray) -> tuple[np.ndarray, list[PchipInterpolator]]:
    y_waypoints_continuous = np.asarray(y_waypoints_continuous, dtype=np.float64)
    path_u = np.linspace(0.0, 1.0, len(y_waypoints_continuous))
    splines = [
        PchipInterpolator(path_u, y_waypoints_continuous[:, joint_idx], extrapolate=True)
        for joint_idx in range(y_waypoints_continuous.shape[1])
    ]
    return path_u, splines


def choose_equivalent_joint_angle(angle: float, reference: float, lower: float, upper: float) -> float:
    angle = float(angle)
    reference = float(reference)
    lower = float(lower)
    upper = float(upper)

    base_k = int(np.round((reference - angle) / (2.0 * np.pi)))
    candidates = []
    tolerance = 5e-2
    for dk in range(-4, 5):
        candidate = angle + 2.0 * np.pi * float(base_k + dk)
        if lower - tolerance <= candidate <= upper + tolerance:
            candidates.append(float(np.clip(candidate, lower, upper)))

    if candidates:
        return float(min(candidates, key=lambda cand: abs(cand - reference)))
    return float(np.clip(angle, lower, upper))


def reduced_reference_to_ctrl(
    y_ref_continuous: np.ndarray,
    ctrl_prev: np.ndarray,
    ctrl_lower: np.ndarray,
    ctrl_upper: np.ndarray,
    q_pan: float,
) -> np.ndarray:
    ctrl_prev = np.asarray(ctrl_prev, dtype=np.float64)
    ctrl = ctrl_prev.copy()
    ctrl[BASE_PAN_INDEX] = float(q_pan)
    for reduced_idx, joint_idx in enumerate(REDUCED_JOINT_INDICES):
        ctrl[joint_idx] = choose_equivalent_joint_angle(
            angle=float(y_ref_continuous[reduced_idx]),
            reference=float(ctrl_prev[joint_idx]),
            lower=float(ctrl_lower[joint_idx]),
            upper=float(ctrl_upper[joint_idx]),
        )
    return np.clip(ctrl, ctrl_lower, ctrl_upper)


def plan_motion_duration(
    splines: list[PchipInterpolator],
    vel_limits: np.ndarray,
    accel_limits: np.ndarray,
    jerk_limits: np.ndarray,
    min_move_duration_s: float,
    timing_samples: int = 801,
) -> dict:
    tau = np.linspace(0.0, 1.0, max(21, int(timing_samples)))
    alpha, d1_tau, d2_tau, d3_tau = minimum_jerk_profile(tau)

    q_u = np.column_stack([np.asarray(spline.derivative(1)(alpha), dtype=np.float64) for spline in splines])
    q_uu = np.column_stack([np.asarray(spline.derivative(2)(alpha), dtype=np.float64) for spline in splines])
    q_uuu = np.column_stack([np.asarray(spline.derivative(3)(alpha), dtype=np.float64) for spline in splines])

    d1_col = d1_tau[:, None]
    d2_col = d2_tau[:, None]
    d3_col = d3_tau[:, None]

    qdot_unit = q_u * d1_col
    qddot_unit = q_uu * (d1_col**2) + q_u * d2_col
    qdddot_unit = q_uuu * (d1_col**3) + 3.0 * q_uu * d1_col * d2_col + q_u * d3_col

    vel_unit_peak = np.max(np.abs(qdot_unit), axis=0)
    accel_unit_peak = np.max(np.abs(qddot_unit), axis=0)
    jerk_unit_peak = np.max(np.abs(qdddot_unit), axis=0)

    vel_ratio = np.divide(
        vel_unit_peak,
        vel_limits,
        out=np.zeros_like(vel_unit_peak),
        where=vel_limits > 1e-9,
    )
    accel_ratio = np.divide(
        accel_unit_peak,
        accel_limits,
        out=np.zeros_like(accel_unit_peak),
        where=accel_limits > 1e-9,
    )
    jerk_ratio = np.divide(
        jerk_unit_peak,
        jerk_limits,
        out=np.zeros_like(jerk_unit_peak),
        where=jerk_limits > 1e-9,
    )

    move_duration_from_velocity = float(np.max(vel_ratio)) if vel_ratio.size else 0.0
    move_duration_from_accel = float(np.sqrt(np.max(accel_ratio))) if accel_ratio.size else 0.0
    move_duration_from_jerk = float(np.cbrt(np.max(jerk_ratio))) if jerk_ratio.size else 0.0

    selected_move_duration_s = float(
        max(
            1e-6,
            float(min_move_duration_s),
            move_duration_from_velocity,
            move_duration_from_accel,
            move_duration_from_jerk,
        )
    )

    return {
        "joint_names_reduced": REDUCED_JOINT_NAMES,
        "path_parameterization": "minimum_jerk_scalar_progress_over_pchip_joint_path",
        "timing_samples": int(len(tau)),
        "velocity_limits_rad_s": np.asarray(vel_limits, dtype=np.float64).tolist(),
        "acceleration_limits_rad_s2": np.asarray(accel_limits, dtype=np.float64).tolist(),
        "jerk_limits_rad_s3": np.asarray(jerk_limits, dtype=np.float64).tolist(),
        "unit_duration_peak_velocity_rad_s": vel_unit_peak.tolist(),
        "unit_duration_peak_acceleration_rad_s2": accel_unit_peak.tolist(),
        "unit_duration_peak_jerk_rad_s3": jerk_unit_peak.tolist(),
        "requested_min_move_duration_s": float(min_move_duration_s),
        "required_move_duration_from_velocity_s": move_duration_from_velocity,
        "required_move_duration_from_acceleration_s": move_duration_from_accel,
        "required_move_duration_from_jerk_s": move_duration_from_jerk,
        "selected_move_duration_s": selected_move_duration_s,
        "planned_peak_velocity_rad_s": (vel_unit_peak / selected_move_duration_s).tolist(),
        "planned_peak_acceleration_rad_s2": (
            accel_unit_peak / (selected_move_duration_s**2)
        ).tolist(),
        "planned_peak_jerk_rad_s3": (jerk_unit_peak / (selected_move_duration_s**3)).tolist(),
    }


def reference_phase(
    time_s: float,
    hold_s: float,
    move_duration_s: float,
) -> tuple[float, float, float, float]:
    if time_s <= hold_s:
        return 0.0, 0.0, 0.0, 0.0
    if time_s >= hold_s + move_duration_s:
        return 1.0, 0.0, 0.0, 0.0

    tau = (time_s - hold_s) / max(1e-6, move_duration_s)
    alpha, d1_tau, d2_tau, d3_tau = minimum_jerk_profile(np.array([tau], dtype=np.float64))
    move_duration_s = float(max(1e-6, move_duration_s))
    return (
        float(alpha[0]),
        float(d1_tau[0] / move_duration_s),
        float(d2_tau[0] / (move_duration_s**2)),
        float(d3_tau[0] / (move_duration_s**3)),
    )


def sample_reduced_joint_reference(
    splines: list[PchipInterpolator],
    phase: float,
    phase_dot: float,
) -> tuple[np.ndarray, np.ndarray]:
    y_ref = np.array([float(spline(phase)) for spline in splines], dtype=np.float64)
    ydot_ref = np.array(
        [float(spline.derivative(1)(phase)) * float(phase_dot) for spline in splines],
        dtype=np.float64,
    )
    return y_ref, ydot_ref


def main() -> None:
    args = parse_args()
    guardrail_config = load_guardrail_config(args.guardrail_config) if args.draw_guardrails else None

    VIDEO_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARY_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    path = scene_path()
    model = mujoco.MjModel.from_xml_path(str(path))
    data = mujoco.MjData(model)
    desired_data = mujoco.MjData(model)
    tool_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, TOOL_SITE_NAME)
    ctrl_lower = model.actuator_ctrlrange[: model.nu, 0].copy()
    ctrl_upper = model.actuator_ctrlrange[: model.nu, 1].copy()
    _, _, bounds = joint_y_bounds(model)
    q_pan = float(ACTIVE_ORIGIN_Q[0])
    z_floor = 0.0

    lim_rng = np.random.default_rng(int(args.seed))
    slice_seeds = collect_slice_seeds(
        lim_rng,
        bounds,
        max(24, int(args.slsqp_seeds)),
        ACTIVE_ORIGIN_Q[1:6].copy(),
    )
    z_max = global_max_site_z_above_floor(
        model,
        data,
        tool_site_id,
        q_pan,
        bounds,
        slice_seeds,
        int(args.slsqp_maxiter),
        z_floor_world_m=z_floor,
    )
    if z_max is None:
        raise RuntimeError("Could not compute kinematic z_max above floor.")

    z_target = float(np.clip(args.z, z_floor + 1e-4, z_max - 1e-4))
    r_min = _optimize_x_at_z_one(
        model,
        data,
        tool_site_id,
        z_target,
        q_pan,
        bounds,
        slice_seeds,
        minimize_x=True,
        maxiter=int(args.slsqp_maxiter),
        z_floor_world_m=z_floor,
    )
    r_max = _optimize_x_at_z_one(
        model,
        data,
        tool_site_id,
        z_target,
        q_pan,
        bounds,
        slice_seeds,
        minimize_x=False,
        maxiter=int(args.slsqp_maxiter),
        z_floor_world_m=z_floor,
    )
    if not r_min["ok"] or not r_max["ok"]:
        raise RuntimeError(
            f"Could not solve min/max X at Z={z_target:.4f} m. "
            f"min_ok={r_min['ok']} max_ok={r_max['ok']}"
        )

    x_min = float(r_min["x"])
    x_max = float(r_max["x"])
    span = float(x_max - x_min)
    usable_margin = min(max(0.0, float(args.x_margin)), max(0.0, 0.45 * span))
    x_lo = float(x_min + usable_margin)
    x_hi = float(x_max - usable_margin)
    if not x_lo < x_hi:
        raise RuntimeError(
            f"Usable X segment collapsed after margin. x_min={x_min:.6f}, x_max={x_max:.6f}, "
            f"x_margin={usable_margin:.6f}"
        )

    y_min = np.asarray(r_min["y_joints"], dtype=np.float64)[1:6].copy()
    y_max = np.asarray(r_max["y_joints"], dtype=np.float64)[1:6].copy()

    # Reuse the continuity-aware fixed-XZ pose solver to pick interior endpoint
    # poses before planning the full path.
    def solve_pose_with_restarts(
        x_target: float,
        y_hint: np.ndarray,
        y_alt: np.ndarray,
        seed: int,
    ) -> np.ndarray:
        rng = np.random.default_rng(seed)
        joint_weights = np.ones(len(bounds), dtype=np.float64)
        tries: list[np.ndarray] = [
            np.asarray(y_hint, dtype=np.float64).copy(),
            np.asarray(y_alt, dtype=np.float64).copy(),
            ACTIVE_ORIGIN_Q[1:6].copy(),
            0.5 * (np.asarray(y_hint, dtype=np.float64) + np.asarray(y_alt, dtype=np.float64)),
        ]
        for _ in range(max(12, int(args.slsqp_seeds) // 2)):
            tries.append(np.array([rng.uniform(lo, hi) for lo, hi in bounds], dtype=np.float64))

        candidates: list[tuple[float, np.ndarray]] = []
        for y0 in tries:
            ok, y_sol, cost = solve_q_at_xz(
                model=model,
                data=data,
                site_id=tool_site_id,
                x_target=float(x_target),
                z_target=float(z_target),
                q_pan=float(q_pan),
                bounds=bounds,
                y0=np.asarray(y0, dtype=np.float64),
                y_prev=np.asarray(y_hint, dtype=np.float64),
                z_floor=float(z_floor),
                maxiter=int(args.slsqp_maxiter),
                joint_weights=joint_weights,
                y_prev_prev=None,
                accel_weight=0.0,
            )
            if ok:
                candidates.append((float(cost), y_sol.copy()))

        if not candidates:
            raise RuntimeError(
                f"Could not solve a feasible pose at x={x_target:.6f} m, z={z_target:.6f} m."
            )

        candidates.sort(key=lambda item: item[0])
        return candidates[0][1]

    y_lo = solve_pose_with_restarts(
        x_target=x_lo,
        y_hint=y_min,
        y_alt=y_max,
        seed=int(args.seed) + 101,
    )
    y_hi = solve_pose_with_restarts(
        x_target=x_hi,
        y_hint=y_max,
        y_alt=y_lo,
        seed=int(args.seed) + 303,
    )

    if args.start_side == "min":
        x_start, x_stop = x_lo, x_hi
        y_start, y_stop = y_lo, y_hi
    else:
        x_start, x_stop = x_hi, x_lo
        y_start, y_stop = y_hi, y_lo

    x_waypoints, y_waypoints, path_smoothness = build_continuous_joint_path(
        model=model,
        data=data,
        site_id=tool_site_id,
        x_start=x_start,
        x_stop=x_stop,
        z_target=z_target,
        q_pan=q_pan,
        bounds=bounds,
        y_start=y_start,
        y_stop=y_stop,
        z_floor_world_m=z_floor,
        seed=int(args.seed) + 707,
        slsqp_maxiter=int(args.slsqp_maxiter),
        path_poses=int(args.path_poses),
        smooth_accel_weight=float(args.smooth_accel_weight),
        random_restart_count=max(16, int(args.slsqp_seeds)),
    )
    y_waypoints_continuous = continuous_reduced_joint_path(y_waypoints)
    y_waypoints_continuous = recenter_continuous_reduced_path_within_limits(
        y_path_continuous=y_waypoints_continuous,
        ctrl_lower=ctrl_lower,
        ctrl_upper=ctrl_upper,
    )
    _, path_splines = fit_reduced_joint_splines(y_waypoints_continuous)

    vel_limits = DEFAULT_JOINT_VEL_LIMIT_RAD_S * float(args.joint_vel_scale)
    accel_limits = DEFAULT_JOINT_ACCEL_LIMIT_RAD_S2 * float(args.joint_accel_scale)
    jerk_limits = DEFAULT_JOINT_JERK_LIMIT_RAD_S3 * float(args.joint_jerk_scale)
    requested_move_duration = max(1e-6, float(args.duration) - 2.0 * float(args.hold_duration))
    timing_plan = plan_motion_duration(
        splines=path_splines,
        vel_limits=vel_limits,
        accel_limits=accel_limits,
        jerk_limits=jerk_limits,
        min_move_duration_s=requested_move_duration,
    )
    move_duration = float(timing_plan["selected_move_duration_s"])
    total_duration = float(move_duration + 2.0 * float(args.hold_duration))

    q_start = q_from_y(y_waypoints_continuous[0], q_pan)
    q_stop = q_from_y(y_waypoints_continuous[-1], q_pan)

    data.qpos[:] = 0.0
    data.qvel[:] = 0.0
    data.qpos[: model.nu] = q_start
    data.ctrl[:] = q_start
    mujoco.mj_forward(model, data)
    y_start_world = float(data.site_xpos[tool_site_id][1])

    render_width = min(args.width, int(model.vis.global_.offwidth))
    render_height = min(args.height, int(model.vis.global_.offheight))
    renderer = None
    camera = None
    if not args.no_video:
        renderer = mujoco.Renderer(model, height=render_height, width=render_width)
        camera = make_camera(model)

    print(f"Running fixed-Z X transport for {total_duration:.2f}s at {args.fps} fps")
    print(f"Scene: {path.name}")
    print(f"Z target (m): {z_target:.6f}")
    print(f"Exact X range (m): [{x_min:.6f}, {x_max:.6f}]")
    print(f"Transport X range after margin (m): [{x_lo:.6f}, {x_hi:.6f}]")
    print(f"Start side: {args.start_side}")
    print(f"Planned waypoint count: {len(x_waypoints)}")
    print(f"Requested motion phase (s): {requested_move_duration:.3f}")
    print(f"Selected motion phase after limits (s): {move_duration:.3f}")
    print(f"Start q: {np.array2string(q_start, precision=6, separator=', ')}")
    print(f"Stop q: {np.array2string(q_stop, precision=6, separator=', ')}")

    n_frames = int(round(total_duration * args.fps))
    sim_steps_per_frame = max(1, int(round((1.0 / args.fps) / model.opt.timestep)))

    frames: list[np.ndarray] = []
    x_tracking_error_trace = []
    z_tracking_error_trace = []
    task_x_error_trace = []
    task_z_error_trace = []
    task_orientation_error_trace_deg = []
    orientation_tracking_error_trace_deg = []
    reference_orientation_error_trace_deg = []
    reference_z_error_trace = []
    reference_x_schedule_error_trace = []
    tool_y_drift_trace = []
    joint_speed_trace = []
    joint_tracking_error_trace = []
    reference_joint_speed_trace = []
    command_step_trace = []
    ctrl = q_start.copy()
    tool_rot_target = TARGET_SITE_ROTATION_WORLD.copy()

    with simulation_progress(n_frames, "Transporting") as pbar:
        for _ in range(n_frames):
            frame_max_command_step = 0.0

            for _ in range(sim_steps_per_frame):
                prev_ctrl = ctrl.copy()
                phase, phase_dot, _, _ = reference_phase(
                    time_s=float(data.time),
                    hold_s=float(args.hold_duration),
                    move_duration_s=move_duration,
                )
                y_ref_continuous, _ = sample_reduced_joint_reference(path_splines, phase, phase_dot)
                ctrl = reduced_reference_to_ctrl(
                    y_ref_continuous=y_ref_continuous,
                    ctrl_prev=prev_ctrl,
                    ctrl_lower=ctrl_lower,
                    ctrl_upper=ctrl_upper,
                    q_pan=q_pan,
                )
                frame_max_command_step = max(frame_max_command_step, float(np.max(np.abs(ctrl - prev_ctrl))))
                data.ctrl[:] = ctrl
                mujoco.mj_step(model, data)

            phase, phase_dot, _, _ = reference_phase(
                time_s=float(data.time),
                hold_s=float(args.hold_duration),
                move_duration_s=move_duration,
            )
            x_ref_linear = float(x_start + (x_stop - x_start) * phase)
            y_ref_continuous, ydot_ref = sample_reduced_joint_reference(path_splines, phase, phase_dot)
            q_ref = reduced_reference_to_ctrl(
                y_ref_continuous=y_ref_continuous,
                ctrl_prev=ctrl,
                ctrl_lower=ctrl_lower,
                ctrl_upper=ctrl_upper,
                q_pan=q_pan,
            )

            desired_data.qpos[:] = 0.0
            desired_data.qvel[:] = 0.0
            desired_data.qpos[: model.nu] = q_ref
            mujoco.mj_forward(model, desired_data)

            tool_pos = data.site_xpos[tool_site_id].copy()
            tool_rot = xmat_to_rot(data.site_xmat[tool_site_id])
            tool_rpy_deg = rotation_matrix_to_rpy_deg(tool_rot)
            ref_tool_pos = desired_data.site_xpos[tool_site_id].copy()
            ref_tool_rot = xmat_to_rot(desired_data.site_xmat[tool_site_id])

            x_tracking_error = float(tool_pos[0] - ref_tool_pos[0])
            z_tracking_error = float(tool_pos[2] - ref_tool_pos[2])
            task_x_error = float(tool_pos[0] - x_ref_linear)
            task_z_error = float(tool_pos[2] - z_target)
            y_drift = float(tool_pos[1] - y_start_world)
            task_orientation_error = orientation_error_deg(tool_rot, tool_rot_target)
            orientation_tracking_error = orientation_error_deg(tool_rot, ref_tool_rot)
            reference_orientation_error = orientation_error_deg(ref_tool_rot, tool_rot_target)
            reference_z_error = float(ref_tool_pos[2] - z_target)
            reference_x_schedule_error = float(ref_tool_pos[0] - x_ref_linear)
            joint_speed = float(np.max(np.abs(data.qvel[: model.nu])))
            joint_tracking_error = float(np.max(np.abs(data.qpos[: model.nu] - q_ref)))
            reference_joint_speed = float(np.max(np.abs(ydot_ref))) if ydot_ref.size else 0.0

            x_tracking_error_trace.append(abs(x_tracking_error))
            z_tracking_error_trace.append(abs(z_tracking_error))
            task_x_error_trace.append(abs(task_x_error))
            task_z_error_trace.append(abs(task_z_error))
            task_orientation_error_trace_deg.append(task_orientation_error)
            orientation_tracking_error_trace_deg.append(orientation_tracking_error)
            reference_orientation_error_trace_deg.append(reference_orientation_error)
            reference_z_error_trace.append(abs(reference_z_error))
            reference_x_schedule_error_trace.append(abs(reference_x_schedule_error))
            tool_y_drift_trace.append(abs(y_drift))
            joint_speed_trace.append(joint_speed)
            joint_tracking_error_trace.append(joint_tracking_error)
            reference_joint_speed_trace.append(reference_joint_speed)
            command_step_trace.append(frame_max_command_step)

            if renderer is not None and camera is not None:
                overlay_lines = [
                    f"t={data.time:5.2f}s  phase={phase: .2f}",
                    f"x ref(path)={ref_tool_pos[0]: .3f} m  x now={tool_pos[0]: .3f} m  track err={x_tracking_error: .4f} m",
                    f"x ideal={x_ref_linear: .3f} m  ref-vs-ideal={reference_x_schedule_error: .4f} m",
                    f"z ref(path)={ref_tool_pos[2]: .3f} m  z now={tool_pos[2]: .3f} m  track err={z_tracking_error: .4f} m",
                    f"z target={z_target: .3f} m  ref drift={reference_z_error: .4f} m  task err={task_z_error: .4f} m",
                    f"tool rpy=({tool_rpy_deg[0]: .1f}, {tool_rpy_deg[1]: .1f}, {tool_rpy_deg[2]: .1f}) deg",
                    f"ori err(target)={task_orientation_error: .2f} deg  ori err(track)={orientation_tracking_error: .2f} deg",
                    f"max|qdot|={joint_speed: .3f} rad/s  ref max|qdot|={reference_joint_speed: .3f} rad/s",
                    f"max|q-qref|={joint_tracking_error: .4f} rad  max|dctrl|={frame_max_command_step: .4f} rad/frame",
                ]
                renderer.update_scene(data, camera=camera)
                frame = renderer.render()
                if guardrail_config is not None:
                    guardrail_decision = check_tcp_pose(
                        tool_pos,
                        guardrail_config,
                        frame=guardrail_config.frame,
                        margin_m=float(args.guardrail_margin_m),
                        timestamp_ns=int(round(data.time * 1e9)),
                    )
                    desired_xyz = np.array([x_ref_linear, y_start, z_target], dtype=np.float64)
                    frame = overlay_guardrails_on_frame(
                        frame,
                        guardrail_config,
                        trajectory_xyz=np.column_stack((tool_x_trace, tool_y_trace, tool_z_trace)),
                        current_xyz=tool_pos,
                        desired_xyz=desired_xyz,
                        decision=guardrail_decision,
                        guardrail_margin_m=float(args.guardrail_margin_m),
                        show_labels=bool(args.show_boundary_labels),
                    )
                    overlay_lines.append(
                        f"guardrail={guardrail_decision.state}  boundary={guardrail_decision.boundary_name or 'none'}"
                    )
                frames.append(annotate_frame(frame, overlay_lines))

            pbar.set_postfix_str(
                f"t={data.time:.1f}s/{total_duration:.0f}s | phase={phase:.2f} | "
                f"x={tool_pos[0]:+.3f} | |x task err|={abs(task_x_error):.4f}m",
                refresh=False,
            )
            pbar.update(1)

    if renderer is not None:
        renderer.close()

    final_tool_pos = data.site_xpos[tool_site_id].copy()
    final_tool_rot = xmat_to_rot(data.site_xmat[tool_site_id])
    final_phase, _, _, _ = reference_phase(
        time_s=float(data.time),
        hold_s=float(args.hold_duration),
        move_duration_s=move_duration,
    )
    final_x_ref_linear = float(x_start + (x_stop - x_start) * final_phase)
    final_x_error = float(final_tool_pos[0] - final_x_ref_linear)
    final_z_error = float(final_tool_pos[2] - z_target)
    final_orientation_error_deg = orientation_error_deg(final_tool_rot, tool_rot_target)
    final_joint_tracking_error = float(joint_tracking_error_trace[-1]) if joint_tracking_error_trace else 0.0

    success = bool(
        abs(final_x_error) <= 0.03
        and abs(final_z_error) <= 0.02
        and final_orientation_error_deg <= 4.0
        and final_joint_tracking_error <= 0.05
    )

    stem = f"ur5e_fixed_z_x_transport_z{z_target:.3f}_{args.start_side}_seed{args.seed}"
    json_name = args.json_name or f"{stem}.json"
    if Path(json_name).suffix == "":
        json_name += ".json"
    video_name = args.video_name or f"{stem}.mp4"
    if Path(video_name).suffix == "":
        video_name += ".mp4"
    json_path = SUMMARY_OUTPUT_DIR / json_name
    video_path = VIDEO_OUTPUT_DIR / video_name

    summary = {
        "controller_name": "planned_joint_trajectory_position_servo",
        "planning_solver_name": "continuity_aware_slsqp_fixed_xz_path",
        "scene_xml": str(path),
        "seed": args.seed,
        "requested_total_duration_s": float(args.duration),
        "total_duration_s": total_duration,
        "hold_duration_s": args.hold_duration,
        "motion_duration_s": move_duration,
        "fps": args.fps,
        "z_target_m": z_target,
        "x_exact_limits_m": {"x_min": x_min, "x_max": x_max, "span": span},
        "x_transport_limits_m": {"x_start": x_start, "x_stop": x_stop, "margin": usable_margin},
        "start_side": args.start_side,
        "path_waypoint_count": int(len(x_waypoints)),
        "x_waypoints_m": x_waypoints.tolist(),
        "start_q": q_start.tolist(),
        "stop_q": q_stop.tolist(),
        "final_q": data.qpos[: model.nu].copy().tolist(),
        "final_tool_site_world": final_tool_pos.tolist(),
        "target_tool_rotation_world": tool_rot_target.tolist(),
        "joint_path_smoothness": path_smoothness,
        "trajectory_timing": timing_plan,
        "reference_path_fidelity": {
            "max_reference_x_schedule_error_m_over_run": float(np.max(reference_x_schedule_error_trace))
            if reference_x_schedule_error_trace
            else 0.0,
            "max_reference_z_error_m_over_run": float(np.max(reference_z_error_trace))
            if reference_z_error_trace
            else 0.0,
            "max_reference_orientation_error_deg_over_run": float(np.max(reference_orientation_error_trace_deg))
            if reference_orientation_error_trace_deg
            else 0.0,
        },
        "tracking_error": {
            "final_x_tracking_error_m": float(x_tracking_error_trace[-1]) if x_tracking_error_trace else 0.0,
            "final_z_tracking_error_m": float(z_tracking_error_trace[-1]) if z_tracking_error_trace else 0.0,
            "final_orientation_tracking_error_deg": float(orientation_tracking_error_trace_deg[-1])
            if orientation_tracking_error_trace_deg
            else 0.0,
            "max_x_tracking_error_m_over_run": float(np.max(x_tracking_error_trace))
            if x_tracking_error_trace
            else 0.0,
            "max_z_tracking_error_m_over_run": float(np.max(z_tracking_error_trace))
            if z_tracking_error_trace
            else 0.0,
            "max_orientation_tracking_error_deg_over_run": float(np.max(orientation_tracking_error_trace_deg))
            if orientation_tracking_error_trace_deg
            else 0.0,
            "max_joint_tracking_error_rad_over_run": float(np.max(joint_tracking_error_trace))
            if joint_tracking_error_trace
            else 0.0,
        },
        "final_x_error_m": final_x_error,
        "final_z_error_m": final_z_error,
        "final_orientation_error_deg": final_orientation_error_deg,
        "final_joint_tracking_error_rad": final_joint_tracking_error,
        "max_x_error_m_over_run": float(np.max(task_x_error_trace)) if task_x_error_trace else 0.0,
        "max_z_error_m_over_run": float(np.max(task_z_error_trace)) if task_z_error_trace else 0.0,
        "max_orientation_error_deg_over_run": float(np.max(task_orientation_error_trace_deg))
        if task_orientation_error_trace_deg
        else 0.0,
        "max_tool_y_drift_m_over_run": float(np.max(tool_y_drift_trace)) if tool_y_drift_trace else 0.0,
        "max_joint_speed_rad_s_over_run": float(np.max(joint_speed_trace)) if joint_speed_trace else 0.0,
        "max_reference_joint_speed_rad_s_over_run": float(np.max(reference_joint_speed_trace))
        if reference_joint_speed_trace
        else 0.0,
        "max_command_step_rad_over_run": float(np.max(command_step_trace)) if command_step_trace else 0.0,
        "success": success,
        "guardrail_config": None if guardrail_config is None else boundary_summary(guardrail_config),
        "guardrail_margin_m": float(args.guardrail_margin_m) if guardrail_config is not None else None,
        "video_path": None if args.no_video else str(video_path),
    }
    json_path.write_text(json.dumps(summary, indent=2))

    if args.no_video:
        print(f"Final x error (m): {final_x_error:.6f}")
        print(f"Final z error (m): {final_z_error:.6f}")
        print(f"Final orientation error (deg): {final_orientation_error_deg:.6f}")
        print(f"Final joint tracking error (rad): {final_joint_tracking_error:.6f}")
        print(f"Success: {success}")
        print(f"Saved summary: {json_path}")
        return

    mediapy.write_video(video_path, frames, fps=args.fps)
    summary["video_path"] = str(video_path)
    json_path.write_text(json.dumps(summary, indent=2))

    print(f"Final x error (m): {final_x_error:.6f}")
    print(f"Final z error (m): {final_z_error:.6f}")
    print(f"Final orientation error (deg): {final_orientation_error_deg:.6f}")
    print(f"Final joint tracking error (rad): {final_joint_tracking_error:.6f}")
    print(f"Success: {success}")
    print(f"Saved video: {video_path}")
    print(f"Saved summary: {json_path}")


if __name__ == "__main__":
    main()
