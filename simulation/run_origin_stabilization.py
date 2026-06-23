#!/usr/bin/env python3
"""
Drive the UR5e from a reproducible random joint state to the measured origin.

This is a control-first experiment:

1. sample a seeded random joint configuration near the origin
2. use an outer-loop setpoint controller to command MuJoCo's built-in servos
3. record convergence metrics and render a video

Run with:
  xvfb-run -a python simulation/run_origin_stabilization.py
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

if "MUJOCO_GL" not in os.environ:
    os.environ["MUJOCO_GL"] = "glfw"

import mediapy
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from progress_bars import simulation_progress

from controller import (
    ACTIVE_ORIGIN_Q,
    differential_ik_split_controller,
    FOREARM_ORIGIN_INDICES,
    sample_random_configuration,
    split_forearm_origin_face_controller,
    TARGET_SITE_ROTATION_WORLD,
    TOOL_FACE_INDICES,
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
MUJOCO_MENAGERIE = BASE_DIR / "mujoco_menagerie"
UR5E_SCENE = MUJOCO_MENAGERIE / "universal_robots_ur5e" / "scene.xml"
UR5E_CARTPOLE_SCENE = MUJOCO_MENAGERIE / "universal_robots_ur5e" / "scene_ur5e_cartpole.xml"
VIDEO_OUTPUT_DIR = BASE_DIR / "demonstration_videos" / "ur5e_cartpole"
SUMMARY_OUTPUT_DIR = BASE_DIR / "outputs" / "control_runs"
ORIGIN_REFERENCE_SITE_NAME = "forearm_tip_site"
TOOL_SITE_NAME = "attachment_site"
BASE_PAN_INDEX = 0
LINK_SEGMENTS = {
    "upper": ("upper_arm_link", "forearm_link"),
    "forearm": ("forearm_link", "wrist_1_link"),
    "wrist": ("wrist_1_link", "wrist_3_link"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--controller",
        choices=["split_forearm_origin_face", "differential_ik_split"],
        default="split_forearm_origin_face",
        help="Controller implementation to run.",
    )
    parser.add_argument("--seed", type=int, default=7, help="Random seed for the start configuration.")
    parser.add_argument("--duration", type=float, default=10.0, help="Simulation duration in seconds.")
    parser.add_argument("--fps", type=int, default=50, help="Video frame rate.")
    parser.add_argument(
        "--target-q-json",
        type=Path,
        default=None,
        help="Optional JSON report containing a target joint vector (`peak_q_rad` or `origin_target_q`).",
    )
    parser.add_argument(
        "--wrist2-offset-deg",
        type=float,
        default=0.0,
        help="Optional non-standard wrist_2_joint offset in degrees, applied as a secondary target.",
    )
    parser.add_argument("--width", type=int, default=960, help="Render width.")
    parser.add_argument("--height", type=int, default=720, help="Render height.")
    parser.add_argument(
        "--span-scale",
        type=float,
        default=0.35,
        help="Fraction of each joint's range used for random-start sampling around the origin.",
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


def load_target_q(target_q_json: Path | None, expected_len: int) -> tuple[np.ndarray, str | None]:
    if target_q_json is None:
        return ACTIVE_ORIGIN_Q.copy(), None

    report = json.loads(target_q_json.read_text())
    if "peak_q_rad" in report:
        q = np.asarray(report["peak_q_rad"], dtype=np.float64)
    elif "origin_target_q" in report:
        q = np.asarray(report["origin_target_q"], dtype=np.float64)
    else:
        raise KeyError(
            f"{target_q_json} must contain `peak_q_rad` or `origin_target_q` for target override"
        )

    if q.shape != (expected_len,):
        raise ValueError(f"{target_q_json} target shape {q.shape} does not match expected ({expected_len},)")

    return q, str(target_q_json)


def make_camera(model: mujoco.MjModel) -> mujoco.MjvCamera:
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    mujoco.mjv_defaultFreeCamera(model, camera)
    camera.azimuth = 12.0
    camera.elevation = -18.0
    camera.distance = 2.05
    return camera


def compute_settle_time(
    primary_trace: np.ndarray,
    secondary_trace: np.ndarray,
    fps: int,
    primary_tol: float = 0.03,
    secondary_tol: float = 0.015,
    hold_time_s: float = 0.5,
) -> float | None:
    hold_frames = max(1, int(round(hold_time_s * fps)))
    good = (primary_trace <= primary_tol) & (secondary_trace <= secondary_tol)
    count = 0
    for idx, ok in enumerate(good):
        count = count + 1 if ok else 0
        if count >= hold_frames:
            return float((idx - hold_frames + 1) / fps)
    return None


def group_max_abs_error(q: np.ndarray, q_target: np.ndarray, indices: np.ndarray) -> float:
    return float(np.max(np.abs(q[indices] - q_target[indices])))


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


def wrap_to_pi(angle_rad: float) -> float:
    return float(np.arctan2(np.sin(angle_rad), np.cos(angle_rad)))


def segment_alignment_deg(
    data: mujoco.MjData,
    target_data: mujoco.MjData,
    start_body_id: int,
    end_body_id: int,
) -> float:
    current = data.xpos[end_body_id] - data.xpos[start_body_id]
    target = target_data.xpos[end_body_id] - target_data.xpos[start_body_id]
    current_norm = float(np.linalg.norm(current))
    target_norm = float(np.linalg.norm(target))
    if current_norm < 1e-8 or target_norm < 1e-8:
        return 0.0
    dot = float(np.dot(current / current_norm, target / target_norm))
    return float(np.degrees(np.arccos(np.clip(dot, -1.0, 1.0))))


def annotate_frame(frame: np.ndarray, lines: list[str]) -> np.ndarray:
    image = Image.fromarray(frame)
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    font = ImageFont.load_default()

    line_height = 16
    panel_width = min(560, image.size[0] - 20)
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


def main() -> None:
    args = parse_args()
    guardrail_config = load_guardrail_config(args.guardrail_config) if args.draw_guardrails else None

    VIDEO_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARY_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    scene_path = UR5E_CARTPOLE_SCENE if UR5E_CARTPOLE_SCENE.exists() else UR5E_SCENE
    model = mujoco.MjModel.from_xml_path(str(scene_path))
    data = mujoco.MjData(model)
    origin_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, ORIGIN_REFERENCE_SITE_NAME)
    tool_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, TOOL_SITE_NAME)
    link_segment_ids = {
        name: (
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, start),
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, end),
        )
        for name, (start, end) in LINK_SEGMENTS.items()
    }

    ctrl_lower = model.actuator_ctrlrange[: model.nu, 0].copy()
    ctrl_upper = model.actuator_ctrlrange[: model.nu, 1].copy()
    target_q, target_source_json = load_target_q(args.target_q_json, model.nu)
    rng = np.random.default_rng(args.seed)
    q_start = sample_random_configuration(
        rng=rng,
        lower=ctrl_lower,
        upper=ctrl_upper,
        center=target_q,
        span_scale=args.span_scale,
        min_distance=0.9,
    )
    q_start[BASE_PAN_INDEX] = target_q[BASE_PAN_INDEX]

    target_data = mujoco.MjData(model)
    target_data.qpos[: model.nu] = target_q
    mujoco.mj_forward(model, target_data)
    target_origin_site_pos = target_data.site_xpos[origin_site_id].copy()
    target_tool_site_pos = target_data.site_xpos[tool_site_id].copy()
    target_tool_rot = TARGET_SITE_ROTATION_WORLD.copy()
    wrist_posture_target = target_q.copy()
    if abs(args.wrist2_offset_deg) > 1e-9:
        wrist_posture_target[4] = wrap_to_pi(
            float(target_q[4] + np.deg2rad(args.wrist2_offset_deg))
        )

    data.qpos[:] = 0.0
    data.qvel[:] = 0.0
    data.qpos[: model.nu] = q_start
    data.ctrl[:] = q_start
    mujoco.mj_forward(model, data)

    render_width = min(args.width, int(model.vis.global_.offwidth))
    render_height = min(args.height, int(model.vis.global_.offheight))
    if render_width != args.width or render_height != args.height:
        print(
            "Requested render size "
            f"{args.width}x{args.height} exceeds MuJoCo framebuffer "
            f"{int(model.vis.global_.offwidth)}x{int(model.vis.global_.offheight)}; "
            f"using {render_width}x{render_height}."
        )

    renderer = mujoco.Renderer(model, height=render_height, width=render_width)
    camera = make_camera(model)

    if args.controller == "split_forearm_origin_face":
        controller_fn = split_forearm_origin_face_controller
        controller_name = "split_forearm_origin_face_controller"
        controller_stem = "split"
    else:
        controller_fn = differential_ik_split_controller
        controller_name = "differential_ik_split_controller"
        controller_stem = "dik"

    n_frames = int(round(args.duration * args.fps))
    sim_steps_per_frame = max(1, int(round((1.0 / args.fps) / model.opt.timestep)))

    frames: list[np.ndarray] = []
    full_joint_error_trace = []
    forearm_origin_joint_error_trace = []
    tool_face_joint_error_trace = []
    tool_orientation_error_trace_deg = []
    base_pan_error_trace = []
    origin_site_error_trace = []
    tool_site_position_error_trace = []
    ctrl = q_start.copy()
    origin_jacp = np.zeros((3, model.nv), dtype=np.float64)
    origin_jacr = np.zeros((3, model.nv), dtype=np.float64)
    tool_jacp = np.zeros((3, model.nv), dtype=np.float64)
    tool_jacr = np.zeros((3, model.nv), dtype=np.float64)

    print(f"Running origin stabilization for {args.duration:.2f}s at {args.fps} fps")
    print(f"Controller: {controller_name}")
    print(f"Scene: {scene_path.name}")
    print(f"Seed: {args.seed}")
    print(f"Random start q: {np.array2string(q_start, precision=6, separator=', ')}")
    print(
        "Forearm-origin shoulder-side-face target q: "
        f"{np.array2string(target_q, precision=6, separator=', ')}"
    )
    if target_source_json is not None:
        print(f"Target override JSON: {target_source_json}")
    if wrist_posture_target is not None:
        print(
            "Non-standard wrist_2 target q: "
            f"{np.array2string(wrist_posture_target, precision=6, separator=', ')}"
        )

    with simulation_progress(n_frames, "Stabilizing") as pbar:
        for _ in range(n_frames):
            for _ in range(sim_steps_per_frame):
                q = data.qpos[: model.nu].copy()
                qvel = data.qvel[: model.nu].copy()
                mujoco.mj_jacSite(model, data, origin_jacp, origin_jacr, origin_site_id)
                mujoco.mj_jacSite(model, data, tool_jacp, tool_jacr, tool_site_id)
                tool_rot = xmat_to_rot(data.site_xmat[tool_site_id])
                ctrl = controller_fn(
                    q=q,
                    q_target=target_q,
                    ctrl_prev=ctrl,
                    ctrl_lower=ctrl_lower,
                    ctrl_upper=ctrl_upper,
                    qvel=qvel,
                    origin_pos=data.site_xpos[origin_site_id],
                    origin_target_pos=target_origin_site_pos,
                    origin_jacobian_pos=origin_jacp[:3, : model.nu],
                    tool_rot=tool_rot,
                    target_tool_rot=target_tool_rot,
                    tool_jacobian_rot=tool_jacr[:3, : model.nu],
                    wrist_posture_target=wrist_posture_target,
                )
                data.ctrl[:] = ctrl
                mujoco.mj_step(model, data)

            q_now = data.qpos[: model.nu].copy()
            origin_site_now = data.site_xpos[origin_site_id].copy()
            tool_site_now = data.site_xpos[tool_site_id].copy()
            full_joint_error_trace.append(float(np.max(np.abs(q_now - target_q))))
            forearm_origin_joint_error_trace.append(
                group_max_abs_error(q_now, target_q, FOREARM_ORIGIN_INDICES)
            )
            tool_face_joint_error_trace.append(
                group_max_abs_error(q_now, target_q, TOOL_FACE_INDICES)
            )
            base_pan_error_trace.append(float(abs(q_now[BASE_PAN_INDEX] - target_q[BASE_PAN_INDEX])))
            origin_site_error_trace.append(float(np.linalg.norm(origin_site_now - target_origin_site_pos)))
            tool_site_position_error_trace.append(float(np.linalg.norm(tool_site_now - target_tool_site_pos)))
            tool_rot = xmat_to_rot(data.site_xmat[tool_site_id])
            tool_orientation_error_trace_deg.append(orientation_error_deg(tool_rot, target_tool_rot))
            tool_rpy_deg = rotation_matrix_to_rpy_deg(tool_rot)
            origin_xyz_error = origin_site_now - target_origin_site_pos
            link_alignment_text = []
            for name, (start_body_id, end_body_id) in link_segment_ids.items():
                align_deg = segment_alignment_deg(data, target_data, start_body_id, end_body_id)
                link_alignment_text.append(f"{name}={align_deg:5.1f}deg")
            overlay_lines = [
                f"t={data.time:5.2f}s  ctrl={args.controller}",
                f"forearm ref xyz=({origin_site_now[0]: .3f}, {origin_site_now[1]: .3f}, {origin_site_now[2]: .3f}) m",
                (
                    f"forearm xyz err=({origin_xyz_error[0]: .3f}, {origin_xyz_error[1]: .3f}, {origin_xyz_error[2]: .3f}) m  "
                    f"|err|={origin_site_error_trace[-1]:.4f} m"
                ),
                f"tool xyz=({tool_site_now[0]: .3f}, {tool_site_now[1]: .3f}, {tool_site_now[2]: .3f}) m",
                f"tool rpy=({tool_rpy_deg[0]: .1f}, {tool_rpy_deg[1]: .1f}, {tool_rpy_deg[2]: .1f}) deg",
                (
                    f"face ori err={tool_orientation_error_trace_deg[-1]:.2f} deg  "
                    f"tool pos drift={tool_site_position_error_trace[-1]:.4f} m"
                ),
                (
                    f"forearm joint err={forearm_origin_joint_error_trace[-1]:.4f} rad  "
                    f"wrist pose dev={tool_face_joint_error_trace[-1]:.4f} rad"
                ),
                f"base pan err={base_pan_error_trace[-1]:.4f} rad  full err={full_joint_error_trace[-1]:.4f} rad",
                "link align to target: " + "  ".join(link_alignment_text),
            ]
            if wrist_posture_target is not None:
                overlay_lines.insert(
                    7,
                    (
                        f"wrist_2 target={wrist_posture_target[4]: .3f} rad  "
                        f"now={q_now[4]: .3f} rad"
                    ),
                )
            renderer.update_scene(data, camera=camera)
            frame = renderer.render()
            if guardrail_config is not None:
                guardrail_decision = check_tcp_pose(
                    tool_site_now,
                    guardrail_config,
                    frame=guardrail_config.frame,
                    margin_m=float(args.guardrail_margin_m),
                    timestamp_ns=int(round(data.time * 1e9)),
                )
                frame = overlay_guardrails_on_frame(
                    frame,
                    guardrail_config,
                    current_xyz=tool_site_now,
                    desired_xyz=target_tool_site_pos,
                    decision=guardrail_decision,
                    guardrail_margin_m=float(args.guardrail_margin_m),
                    show_labels=bool(args.show_boundary_labels),
                )
                overlay_lines.append(
                    f"guardrail={guardrail_decision.state}  boundary={guardrail_decision.boundary_name or 'none'}"
                )
            frames.append(annotate_frame(frame, overlay_lines))

            pbar.set_postfix_str(
                f"t={data.time:.1f}s/{args.duration:.0f}s | "
                f"|origin err|={origin_site_error_trace[-1]:.4f}m | "
                f"ori={tool_orientation_error_trace_deg[-1]:.1f}deg",
                refresh=False,
            )
            pbar.update(1)

    renderer.close()

    full_joint_error_trace = np.asarray(full_joint_error_trace, dtype=np.float64)
    forearm_origin_joint_error_trace = np.asarray(forearm_origin_joint_error_trace, dtype=np.float64)
    tool_face_joint_error_trace = np.asarray(tool_face_joint_error_trace, dtype=np.float64)
    tool_orientation_error_trace_deg = np.asarray(tool_orientation_error_trace_deg, dtype=np.float64)
    base_pan_error_trace = np.asarray(base_pan_error_trace, dtype=np.float64)
    origin_site_error_trace = np.asarray(origin_site_error_trace, dtype=np.float64)
    tool_site_position_error_trace = np.asarray(tool_site_position_error_trace, dtype=np.float64)
    final_q = data.qpos[: model.nu].copy()
    final_origin_site_pos = data.site_xpos[origin_site_id].copy()
    final_tool_site_pos = data.site_xpos[tool_site_id].copy()
    final_full_joint_error = float(full_joint_error_trace[-1])
    final_forearm_origin_joint_error = float(forearm_origin_joint_error_trace[-1])
    final_tool_face_joint_error = float(tool_face_joint_error_trace[-1])
    final_tool_orientation_error_deg = float(tool_orientation_error_trace_deg[-1])
    final_base_pan_error = float(base_pan_error_trace[-1])
    final_origin_site_error = float(origin_site_error_trace[-1])
    final_tool_site_position_error = float(tool_site_position_error_trace[-1])
    origin_settle_time_s = compute_settle_time(
        origin_site_error_trace,
        tool_orientation_error_trace_deg,
        fps=args.fps,
        primary_tol=0.015,
        secondary_tol=3.0,
    )
    full_joint_settle_time_s = compute_settle_time(
        full_joint_error_trace,
        origin_site_error_trace,
        fps=args.fps,
    )
    orientation_settle_time_s = compute_settle_time(
        tool_orientation_error_trace_deg,
        origin_site_error_trace,
        fps=args.fps,
        primary_tol=3.0,
        secondary_tol=0.015,
    )

    success = bool(
        final_forearm_origin_joint_error <= 0.03
        and final_tool_orientation_error_deg <= 3.0
        and final_origin_site_error <= 0.015
        and origin_settle_time_s is not None
        and orientation_settle_time_s is not None
    )
    success_full_joint_match = bool(
        final_full_joint_error <= 0.03
        and final_tool_orientation_error_deg <= 3.0
        and final_origin_site_error <= 0.015
        and full_joint_settle_time_s is not None
    )

    stem = f"ur5e_forearm_origin_shoulder_side_face_seed{args.seed}"
    if controller_stem != "split":
        stem += f"_{controller_stem}"
    if abs(args.wrist2_offset_deg) > 1e-9:
        stem += f"_wrist2_offset{int(round(args.wrist2_offset_deg))}deg"
    if args.fps != 50:
        stem += f"_{args.fps}fps"
    video_path = VIDEO_OUTPUT_DIR / (args.video_name or f"{stem}.mp4")
    json_path = SUMMARY_OUTPUT_DIR / (args.json_name or f"{stem}.json")

    mediapy.write_video(video_path, frames, fps=args.fps)

    summary = {
        "controller_name": controller_name,
        "origin_name": "forearm_tip_site_shoulder_side_face_split_control",
        "scene_xml": str(scene_path),
        "seed": args.seed,
        "duration_s": args.duration,
        "fps": args.fps,
        "target_source_json": target_source_json,
        "wrist2_offset_deg": args.wrist2_offset_deg,
        "random_start_q": q_start.tolist(),
        "origin_target_q": target_q.tolist(),
        "wrist_posture_target_q": None if wrist_posture_target is None else wrist_posture_target.tolist(),
        "final_q": final_q.tolist(),
        "target_origin_site_world": target_origin_site_pos.tolist(),
        "final_origin_site_world": final_origin_site_pos.tolist(),
        "target_tool_site_world": target_tool_site_pos.tolist(),
        "final_tool_site_world": final_tool_site_pos.tolist(),
        "target_tool_rotation_world": target_tool_rot.tolist(),
        "final_max_joint_error_rad": final_full_joint_error,
        "final_forearm_origin_joint_error_rad": final_forearm_origin_joint_error,
        "final_tool_face_joint_error_rad": final_tool_face_joint_error,
        "final_tool_orientation_error_deg": final_tool_orientation_error_deg,
        "final_base_pan_error_rad": final_base_pan_error,
        "final_origin_site_error_m": final_origin_site_error,
        "final_tool_site_position_error_m": final_tool_site_position_error,
        "max_joint_error_rad_over_run": float(np.max(full_joint_error_trace)),
        "max_forearm_origin_joint_error_rad_over_run": float(np.max(forearm_origin_joint_error_trace)),
        "max_tool_face_joint_error_rad_over_run": float(np.max(tool_face_joint_error_trace)),
        "max_tool_orientation_error_deg_over_run": float(np.max(tool_orientation_error_trace_deg)),
        "max_base_pan_error_rad_over_run": float(np.max(base_pan_error_trace)),
        "max_origin_site_error_m_over_run": float(np.max(origin_site_error_trace)),
        "max_tool_site_position_error_m_over_run": float(np.max(tool_site_position_error_trace)),
        "settle_time_s": origin_settle_time_s,
        "origin_settle_time_s": origin_settle_time_s,
        "full_joint_settle_time_s": full_joint_settle_time_s,
        "orientation_settle_time_s": orientation_settle_time_s,
        "success": success,
        "success_full_joint_match": success_full_joint_match,
        "guardrail_config": None if guardrail_config is None else boundary_summary(guardrail_config),
        "guardrail_margin_m": float(args.guardrail_margin_m) if guardrail_config is not None else None,
        "video_path": str(video_path),
    }
    json_path.write_text(json.dumps(summary, indent=2))

    print(f"Final q: {np.array2string(final_q, precision=6, separator=', ')}")
    print(f"Final forearm-origin joint error (rad): {final_forearm_origin_joint_error:.6f}")
    print(f"Final wrist pose deviation from origin q (rad): {final_tool_face_joint_error:.6f}")
    print(f"Final tool orientation error (deg): {final_tool_orientation_error_deg:.6f}")
    print(f"Final base-pan error (rad): {final_base_pan_error:.6f}")
    print(f"Final max joint error (rad): {final_full_joint_error:.6f}")
    print(f"Final forearm-origin site error (m): {final_origin_site_error:.6f}")
    print(f"Final tool-site position drift (m): {final_tool_site_position_error:.6f}")
    print(f"Origin settle time (s): {origin_settle_time_s}")
    print(f"Full-joint settle time (s): {full_joint_settle_time_s}")
    print(f"Orientation settle time (s): {orientation_settle_time_s}")
    print(f"Success (forearm-origin + face-orientation objective): {success}")
    print(f"Success (full joint match): {success_full_joint_match}")
    print(f"Saved video: {video_path}")
    print(f"Saved summary: {json_path}")


if __name__ == "__main__":
    main()
