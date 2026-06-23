#!/usr/bin/env python3
"""
Velocity-commanded UR5e end-effector transport along world X.

The robot starts from the stored shoulder-side-face origin pose (see
`controller.ACTIVE_ORIGIN_Q`) and is commanded to translate `attachment_site`
along world X at a constant speed until a target X is reached. Height (world
Z), tool orientation, and `shoulder_pan_joint` are held fixed during the
motion.

The controller stays on MuJoCo's built-in position servos. The commanded
Cartesian velocity is automatically scaled down whenever either

- a joint would need to move faster than its per-step setpoint cap, or
- the servo torque implied by the proposed setpoint would exceed a safe
  fraction of the joint's `forcerange` (taken from ur5e.xml).

The run saves a rendered video and a JSON summary that includes the estimated
servo torques at every frame.

Run with:
  xvfb-run -a python simulation/run_x_velocity_transport.py --v-x 0.05

Fixed-Z “max X span” slice (same family as `run_fixed_z_x_transport.py` / recordings
at `z_demo_m ≈ 0.54` m): pass a prior workspace JSON that contains `start_q` and
`x_transport_limits`, e.g. `outputs/control_runs/fixed_z_x_transport_firstpass_z0.540_seed1.json`,
plus a long enough `--duration` for the full traverse.

**Multi-segment path along X:** give ordered waypoints (world X in meters) and a cruise
speed (m/s) for each leg via `--x-segments` (file) and/or repeated `--segment X V`.
Each line is one row: target `x`, then speed magnitude; direction is chosen automatically
toward that target from the current pose. After each waypoint is reached, the next
leg starts. Omit `--x-goal` / `--delta-x` when using segments.
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
    SERVO_FORCE_LIMIT,
    TARGET_SITE_ROTATION_WORLD,
    velocity_x_transport_controller,
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
TOOL_SITE_NAME = "attachment_site"
BASE_PAN_INDEX = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--v-x",
        type=float,
        default=0.05,
        help="Commanded tool-site world-X velocity in m/s (signed).",
    )
    parser.add_argument(
        "--delta-x",
        type=float,
        default=0.10,
        help=(
            "Signed world-X displacement to command relative to the start "
            "tool pose. Sign should match --v-x."
        ),
    )
    parser.add_argument(
        "--x-goal",
        type=float,
        default=None,
        help=(
            "Absolute world-X target for `attachment_site` in meters. "
            "Overrides --delta-x when provided."
        ),
    )
    parser.add_argument("--seed", type=int, default=7, help="Unused here; kept for parity with other runners.")
    parser.add_argument(
        "--duration",
        type=float,
        default=6.0,
        help="Maximum simulation duration in seconds (acts as a timeout).",
    )
    parser.add_argument("--fps", type=int, default=50, help="Video frame rate.")
    parser.add_argument("--width", type=int, default=960, help="Render width.")
    parser.add_argument("--height", type=int, default=720, help="Render height.")
    parser.add_argument(
        "--torque-headroom",
        type=float,
        default=0.9,
        help="Fraction of each joint's forcerange the servo may use before the v_x command is throttled.",
    )
    parser.add_argument(
        "--stop-tol-m",
        type=float,
        default=2.0e-3,
        help="Stop when the tool is within this distance of x_goal.",
    )
    parser.add_argument(
        "--a-decel",
        type=float,
        default=0.35,
        help="Defines braking zone length s_brake = v_x^2/(2*a) (m/s^2); larger a = shorter zone.",
    )
    parser.add_argument(
        "--k-x-hold",
        type=float,
        default=8.0,
        help="Inside --stop-tol-m, x velocity = k * (x_goal - x) in m/s per m.",
    )
    parser.add_argument(
        "--settle-time-s",
        type=float,
        default=0.5,
        help="Extra time to keep commanding zero velocity after reaching the goal so the servos settle.",
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
    parser.add_argument(
        "--init-json",
        type=Path,
        default=None,
        help=(
            "Optional JSON from a fixed-Z workspace run (e.g. fixed_z_x_transport_*). "
            "Must contain `start_q`; may contain `x_transport_limits` and `z_target_m`."
        ),
    )
    parser.add_argument(
        "--workspace-x-goal",
        choices=["x_stop", "x_start"],
        default="x_stop",
        help=(
            "When using --init-json with `x_transport_limits`, set x_goal from this key "
            "(default: x_stop, i.e. traverse toward the +X margin endpoint when starting from min-X)."
        ),
    )
    parser.add_argument(
        "--x-segments",
        type=Path,
        default=None,
        help=(
            "Optional path to a multi-segment path. Text: each non-comment line is "
            "`x_m v_mps` (whitespace or comma separated). JSON: "
            '`{"segments": [{"x": 0.0, "v": 0.05}, ...]}`. '
            "Incompatible with --segment (use one or the other source)."
        ),
    )
    parser.add_argument(
        "--segment",
        nargs=2,
        type=float,
        metavar=("X_M", "V_MPS"),
        action="append",
        default=None,
        help=(
            "Append one path leg: world-X target (m) and cruise speed magnitude (m/s). "
            "Repeat for multiple segments. First leg starts from the initial tool pose."
        ),
    )
    return parser.parse_args()


def _load_workspace_init(path: Path) -> tuple[np.ndarray, dict]:
    report = json.loads(path.read_text())
    if "start_q" not in report:
        raise KeyError(f"{path} must contain `start_q` for workspace initialization")
    q = np.asarray(report["start_q"], dtype=np.float64)
    return q, report


def _x_transport_segment(report: dict, margin_default: float = 0.05) -> dict[str, float] | None:
    """
    Normalize workspace JSONs: some runs use `x_transport_limits_m`, others
    `x_transport_limits`, and sweep summaries use `x_limits_m` (exact extrema).
    """
    if "x_transport_limits_m" in report:
        return {k: float(v) for k, v in report["x_transport_limits_m"].items()}
    if "x_transport_limits" in report:
        return {k: float(v) for k, v in report["x_transport_limits"].items()}
    if "x_limits_m" in report:
        lim = report["x_limits_m"]
        m = float(margin_default)
        return {
            "x_start": float(lim["x_min"]) + m,
            "x_stop": float(lim["x_max"]) - m,
            "margin": m,
        }
    ex = report.get("x_exact_limits_m")
    if isinstance(ex, dict) and "x_min" in ex and "x_max" in ex:
        m = float(margin_default)
        return {
            "x_start": float(ex["x_min"]) + m,
            "x_stop": float(ex["x_max"]) - m,
            "margin": m,
        }
    return None


def _signed_v_x(x_now: float, x_target: float, v_mag: float) -> float:
    """Signed cruise velocity toward x_target; zero if already there."""
    v_mag = abs(float(v_mag))
    if v_mag < 1e-12:
        return 0.0
    dx = float(x_target) - float(x_now)
    if abs(dx) < 1e-9:
        return 0.0
    return float(np.sign(dx) * v_mag)


def _parse_segments_text(raw: str) -> list[tuple[float, float]]:
    rows: list[tuple[float, float]] = []
    for line in raw.splitlines():
        line = line.split("#")[0].strip()
        if not line:
            continue
        parts = line.replace(",", " ").split()
        if len(parts) != 2:
            raise ValueError(f"Each path line must have exactly two numbers (x_m v_mps); got: {line!r}")
        rows.append((float(parts[0]), float(parts[1])))
    if not rows:
        raise ValueError("Segment file has no data rows.")
    return rows


def _parse_segments_file(path: Path) -> list[tuple[float, float]]:
    text = path.read_text()
    if path.suffix.lower() == ".json":
        data = json.loads(text)
        if isinstance(data, list):
            out: list[tuple[float, float]] = []
            for item in data:
                if isinstance(item, (list, tuple)) and len(item) == 2:
                    out.append((float(item[0]), float(item[1])))
                elif isinstance(item, dict):
                    out.append((float(item["x"]), float(item["v"])))
                else:
                    raise ValueError(f"Unrecognized segment entry: {item!r}")
            if not out:
                raise ValueError("Empty segment list in JSON.")
            return out
        if isinstance(data, dict) and "segments" in data:
            return [
                (float(s["x"]), float(s["v"]))
                for s in data["segments"]
            ]
        raise ValueError("JSON must be a list of [x,v] or {\"segments\": [{\"x\",\"v\"},...]}")
    return _parse_segments_text(text)


def _load_path_segments(args: argparse.Namespace) -> list[tuple[float, float]] | None:
    from_cli = args.segment
    from_file = args.x_segments
    if from_cli is not None and from_file is not None:
        raise SystemExit("Use only one of --x-segments FILE or repeated --segment X V, not both.")
    if from_file is not None:
        return _parse_segments_file(from_file)
    if from_cli is not None:
        return [(float(x), float(v)) for x, v in from_cli]
    return None


def make_camera(model: mujoco.MjModel) -> mujoco.MjvCamera:
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    mujoco.mjv_defaultFreeCamera(model, camera)
    camera.azimuth = 12.0
    camera.elevation = -18.0
    camera.distance = 2.05
    return camera


def xmat_to_rot(xmat: np.ndarray) -> np.ndarray:
    return np.asarray(xmat, dtype=np.float64).reshape(3, 3)


def orientation_error_deg(rot: np.ndarray, rot_target: np.ndarray) -> float:
    rel = rot_target.T @ rot
    cos_theta = np.clip((float(np.trace(rel)) - 1.0) * 0.5, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_theta)))


def annotate_frame(frame: np.ndarray, lines: list[str]) -> np.ndarray:
    image = Image.fromarray(frame)
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    font = ImageFont.load_default()

    line_height = 16
    panel_width = min(620, image.size[0] - 20)
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
    path_segments = _load_path_segments(args)
    guardrail_config = load_guardrail_config(args.guardrail_config) if args.draw_guardrails else None

    VIDEO_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARY_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    scene_path = UR5E_CARTPOLE_SCENE if UR5E_CARTPOLE_SCENE.exists() else UR5E_SCENE
    model = mujoco.MjModel.from_xml_path(str(scene_path))
    data = mujoco.MjData(model)
    tool_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, TOOL_SITE_NAME)

    ctrl_lower = model.actuator_ctrlrange[: model.nu, 0].copy()
    ctrl_upper = model.actuator_ctrlrange[: model.nu, 1].copy()
    dt = float(model.opt.timestep)

    workspace_report: dict | None = None
    if args.init_json is not None:
        q_start, workspace_report = _load_workspace_init(args.init_json)
        if q_start.shape[0] != model.nu:
            raise ValueError(
                f"--init-json start_q length {q_start.shape[0]} does not match model.nu={model.nu}"
            )
        print(f"Workspace init from: {args.init_json}")
        if "z_target_m" in workspace_report:
            print(f"  (reference z_target_m from JSON: {workspace_report['z_target_m']})")
        seg = _x_transport_segment(workspace_report)
        if seg is not None:
            print(
                f"  usable X segment (m): x_start={seg.get('x_start')}  "
                f"x_stop={seg.get('x_stop')}  margin={seg.get('margin')}"
            )
    else:
        # Canonical shoulder-side-face origin (tall configuration).
        q_start = ACTIVE_ORIGIN_Q.copy()

    data.qpos[:] = 0.0
    data.qvel[:] = 0.0
    data.qpos[: model.nu] = q_start
    data.ctrl[:] = q_start
    mujoco.mj_forward(model, data)

    tool_start_pos = data.site_xpos[tool_site_id].copy()
    z_hold = float(tool_start_pos[2])
    pan_target = float(q_start[BASE_PAN_INDEX])

    transport_seg = _x_transport_segment(workspace_report) if workspace_report is not None else None

    segment_idx = 0
    segment_completion_times: list[dict[str, float | int]] = []

    if path_segments is not None:
        if args.x_goal is not None:
            print("Note: --x-goal ignored because a multi-segment path (--segment / --x-segments) is set.")
        current_x_goal = float(path_segments[0][0])
        current_v_x_cmd = _signed_v_x(
            float(tool_start_pos[0]), current_x_goal, path_segments[0][1]
        )
    elif args.x_goal is not None:
        current_x_goal = float(args.x_goal)
        signed_direction = np.sign(current_x_goal - tool_start_pos[0])
        if signed_direction != 0.0 and np.sign(args.v_x) != 0.0 and np.sign(args.v_x) != signed_direction:
            print(
                f"Warning: --v-x sign {np.sign(args.v_x):+.0f} does not match the "
                f"direction to the goal {signed_direction:+.0f}; using the goal direction."
            )
            current_v_x_cmd = float(signed_direction * abs(args.v_x))
        else:
            current_v_x_cmd = float(args.v_x)
    elif transport_seg is not None and args.workspace_x_goal in transport_seg:
        current_x_goal = float(transport_seg[args.workspace_x_goal])
        signed_direction = np.sign(current_x_goal - tool_start_pos[0])
        if signed_direction != 0.0 and np.sign(args.v_x) != 0.0 and np.sign(args.v_x) != signed_direction:
            print(
                f"Warning: --v-x sign {np.sign(args.v_x):+.0f} does not match the "
                f"direction to the goal {signed_direction:+.0f}; using the goal direction."
            )
            current_v_x_cmd = float(signed_direction * abs(args.v_x))
        else:
            current_v_x_cmd = float(args.v_x)
    else:
        current_x_goal = float(tool_start_pos[0] + args.delta_x)
        signed_direction = np.sign(current_x_goal - tool_start_pos[0])
        if signed_direction != 0.0 and np.sign(args.v_x) != 0.0 and np.sign(args.v_x) != signed_direction:
            print(
                f"Warning: --v-x sign {np.sign(args.v_x):+.0f} does not match the "
                f"direction to the goal {signed_direction:+.0f}; using the goal direction."
            )
            current_v_x_cmd = float(signed_direction * abs(args.v_x))
        else:
            current_v_x_cmd = float(args.v_x)

    x_goal_final = (
        float(path_segments[-1][0]) if path_segments is not None else float(current_x_goal)
    )

    render_width = min(args.width, int(model.vis.global_.offwidth))
    render_height = min(args.height, int(model.vis.global_.offheight))
    if render_width != args.width or render_height != args.height:
        print(
            f"Requested render size {args.width}x{args.height} exceeds MuJoCo framebuffer; "
            f"using {render_width}x{render_height}."
        )

    renderer = mujoco.Renderer(model, height=render_height, width=render_width)
    camera = make_camera(model)

    n_frames = int(round(args.duration * args.fps))
    sim_steps_per_frame = max(1, int(round((1.0 / args.fps) / dt)))

    frames: list[np.ndarray] = []
    time_trace: list[float] = []
    tool_x_trace: list[float] = []
    tool_z_trace: list[float] = []
    tool_y_trace: list[float] = []
    orientation_error_trace_deg: list[float] = []
    v_x_cmd_trace: list[float] = []
    v_x_realized_trace: list[float] = []
    tau_estimate_trace: list[list[float]] = []
    speed_scale_trace: list[float] = []
    torque_scale_trace: list[float] = []

    ctrl = q_start.copy()
    tool_jacp = np.zeros((3, model.nv), dtype=np.float64)
    tool_jacr = np.zeros((3, model.nv), dtype=np.float64)

    settle_steps_remaining = int(round(args.settle_time_s / dt))
    goal_reached_at_s: float | None = None
    # Avoid re-firing segment transitions while goal_reached stays true for many substeps.
    arrived_at_segment: int | None = None

    print(f"Running x-velocity transport for up to {args.duration:.2f}s at {args.fps} fps")
    print(f"Scene: {scene_path.name}  dt={dt:.5f} s  steps/frame={sim_steps_per_frame}")
    print(f"Start tool xyz (m): {np.array2string(tool_start_pos, precision=4)}")
    if path_segments is not None:
        print(f"Multi-segment path ({len(path_segments)} leg(s)):")
        for i, (xt, vt) in enumerate(path_segments):
            print(f"  leg {i + 1}:  x_target={xt:+.4f} m  |v|={abs(vt):.4f} m/s")
    print(f"Current x_goal (m): {current_x_goal:.4f}   z_hold (m): {z_hold:.4f}")
    print(f"v_x_cmd (m/s): {current_v_x_cmd:+.4f}   torque_headroom: {args.torque_headroom:.2f}")

    last_diag: dict = {
        "v_x_cmd": current_v_x_cmd,
        "v_x_effective": current_v_x_cmd,
        "v_x_realized_cmd": current_v_x_cmd,
        "tau_estimate_nm": [0.0] * 6,
        "speed_scale": 1.0,
        "torque_scale": 1.0,
    }

    n_path_legs = len(path_segments) if path_segments is not None else 1
    with simulation_progress(n_frames, "x-transport") as pbar:
        for _ in range(n_frames):
            for _ in range(sim_steps_per_frame):
                q = data.qpos[: model.nu].copy()
                qvel = data.qvel[: model.nu].copy()
                mujoco.mj_jacSite(model, data, tool_jacp, tool_jacr, tool_site_id)
                tool_pos = data.site_xpos[tool_site_id].copy()
                tool_rot = xmat_to_rot(data.site_xmat[tool_site_id])

                ctrl, last_diag = velocity_x_transport_controller(
                    q=q,
                    qvel=qvel,
                    ctrl_prev=ctrl,
                    ctrl_lower=ctrl_lower,
                    ctrl_upper=ctrl_upper,
                    tool_pos=tool_pos,
                    tool_rot=tool_rot,
                    tool_jacobian_pos=tool_jacp[:3, : model.nu],
                    tool_jacobian_rot=tool_jacr[:3, : model.nu],
                    v_x_cmd=current_v_x_cmd,
                    x_goal=current_x_goal,
                    z_hold=z_hold,
                    target_tool_rot=TARGET_SITE_ROTATION_WORLD,
                    pan_target=pan_target,
                    dt=dt,
                    stop_tol_m=args.stop_tol_m,
                    torque_headroom=float(args.torque_headroom),
                    a_decel_m_s2=float(args.a_decel),
                    k_x_hold_s_inv=float(args.k_x_hold),
                )
                data.ctrl[:] = ctrl
                mujoco.mj_step(model, data)

                if last_diag["goal_reached"]:
                    if path_segments is not None:
                        if arrived_at_segment != segment_idx:
                            arrived_at_segment = segment_idx
                            segment_completion_times.append(
                                {
                                    "segment_index": int(segment_idx),
                                    "x_target_m": float(path_segments[segment_idx][0]),
                                    "time_s": float(data.time),
                                }
                            )
                            if segment_idx < len(path_segments) - 1:
                                segment_idx += 1
                                current_x_goal = float(path_segments[segment_idx][0])
                                current_v_x_cmd = _signed_v_x(
                                    float(data.site_xpos[tool_site_id][0]),
                                    current_x_goal,
                                    path_segments[segment_idx][1],
                                )
                            elif goal_reached_at_s is None:
                                goal_reached_at_s = float(data.time)
                    elif goal_reached_at_s is None:
                        goal_reached_at_s = float(data.time)

                if goal_reached_at_s is not None:
                    settle_steps_remaining -= 1

            tool_pos_now = data.site_xpos[tool_site_id].copy()
            tool_rot_now = xmat_to_rot(data.site_xmat[tool_site_id])
            time_trace.append(float(data.time))
            tool_x_trace.append(float(tool_pos_now[0]))
            tool_y_trace.append(float(tool_pos_now[1]))
            tool_z_trace.append(float(tool_pos_now[2]))
            orientation_error_trace_deg.append(orientation_error_deg(tool_rot_now, TARGET_SITE_ROTATION_WORLD))
            v_x_cmd_trace.append(float(last_diag["v_x_effective"]))
            v_x_realized_trace.append(float(last_diag["v_x_realized_cmd"]))
            tau_estimate_trace.append(list(last_diag["tau_estimate_nm"]))
            speed_scale_trace.append(float(last_diag["speed_scale"]))
            torque_scale_trace.append(float(last_diag["torque_scale"]))

            tau_now = np.asarray(last_diag["tau_estimate_nm"], dtype=np.float64)
            tau_fraction = np.max(np.abs(tau_now) / np.maximum(SERVO_FORCE_LIMIT, 1e-9))
            seg_note = ""
            if path_segments is not None:
                seg_note = f"  leg {segment_idx + 1}/{len(path_segments)}"
            overlay_lines = [
                f"t={data.time:5.2f}s  v_x_cmd={current_v_x_cmd:+.3f} m/s  v_x_eff={last_diag['v_x_effective']:+.3f} m/s{seg_note}",
                f"tool xyz=({tool_pos_now[0]: .3f}, {tool_pos_now[1]: .3f}, {tool_pos_now[2]: .3f}) m",
                (
                    f"x_err={(current_x_goal - tool_pos_now[0]): .4f} m  "
                    f"z_drift={(tool_pos_now[2] - z_hold): .4f} m  "
                    f"ori_err={orientation_error_trace_deg[-1]: .2f} deg"
                ),
                (
                    f"speed scale={last_diag['speed_scale']:.2f}  "
                    f"torque scale={last_diag['torque_scale']:.2f}  "
                    f"peak tau frac={tau_fraction:.2f}"
                ),
                (
                    "tau (N*m): "
                    + ", ".join(f"{v:+6.1f}" for v in tau_now)
                ),
                (
                    "final goal reached"
                    if goal_reached_at_s is not None
                    else f"approaching x_goal={current_x_goal:+.3f} m"
                ),
            ]
            renderer.update_scene(data, camera=camera)
            frame = renderer.render()
            if guardrail_config is not None:
                guardrail_decision = check_tcp_pose(
                    tool_pos_now,
                    guardrail_config,
                    frame=guardrail_config.frame,
                    margin_m=float(args.guardrail_margin_m),
                    timestamp_ns=int(round(data.time * 1e9)),
                )
                desired_xyz = np.array([current_x_goal, tool_start_pos[1], z_hold], dtype=np.float64)
                frame = overlay_guardrails_on_frame(
                    frame,
                    guardrail_config,
                    trajectory_xyz=np.column_stack((tool_x_trace, tool_y_trace, tool_z_trace)),
                    current_xyz=tool_pos_now,
                    desired_xyz=desired_xyz,
                    decision=guardrail_decision,
                    guardrail_margin_m=float(args.guardrail_margin_m),
                    show_labels=bool(args.show_boundary_labels),
                )
                overlay_lines.append(
                    f"guardrail={guardrail_decision.state}  boundary={guardrail_decision.boundary_name or 'none'}"
                )
            frames.append(annotate_frame(frame, overlay_lines))

            phase = "settle" if goal_reached_at_s is not None else "move"
            pbar.set_postfix_str(
                f"t={data.time:.1f}s/{args.duration:.0f}s | x={tool_pos_now[0]:+.3f} | "
                f"x*={current_x_goal:+.3f} | leg={segment_idx + 1}/{n_path_legs} | {phase}",
                refresh=False,
            )
            pbar.update(1)

            if goal_reached_at_s is not None and settle_steps_remaining <= 0:
                break

    renderer.close()

    final_tool_pos = data.site_xpos[tool_site_id].copy()
    final_rot = xmat_to_rot(data.site_xmat[tool_site_id])
    final_orientation_error_deg = orientation_error_deg(final_rot, TARGET_SITE_ROTATION_WORLD)
    final_x_error = float(x_goal_final - final_tool_pos[0])
    final_z_drift = float(final_tool_pos[2] - z_hold)

    tau_estimate_array = np.asarray(tau_estimate_trace, dtype=np.float64)
    peak_tau_per_joint = np.max(np.abs(tau_estimate_array), axis=0).tolist() if len(tau_estimate_array) else []
    peak_tau_fraction = (
        float(np.max(np.abs(tau_estimate_array) / SERVO_FORCE_LIMIT[None, :]))
        if len(tau_estimate_array)
        else 0.0
    )

    success = bool(
        goal_reached_at_s is not None
        and abs(final_x_error) <= max(5.0e-3, args.stop_tol_m * 2.5)
        and abs(final_z_drift) <= 8.0e-3
        and final_orientation_error_deg <= 3.0
    )

    dx_str = f"{(x_goal_final - tool_start_pos[0]):+.3f}"
    v_stem = f"{current_v_x_cmd:+.3f}"
    if path_segments is not None:
        v_stem = "path"
    if args.init_json is not None:
        stem = f"ur5e_x_velocity_transport_{args.init_json.stem}_vx{v_stem}_dx{dx_str}"
    else:
        stem = f"ur5e_x_velocity_transport_vx{v_stem}_dx{dx_str}"
    if path_segments is not None:
        stem += f"_{len(path_segments)}legs"
    video_path = VIDEO_OUTPUT_DIR / (args.video_name or f"{stem}.mp4")
    json_path = SUMMARY_OUTPUT_DIR / (args.json_name or f"{stem}.json")

    mediapy.write_video(video_path, frames, fps=args.fps)

    summary = {
        "controller_name": "velocity_x_transport_controller",
        "init_json": str(args.init_json) if args.init_json is not None else None,
        "workspace_x_goal_key": args.workspace_x_goal if args.init_json is not None else None,
        "workspace_transport_segment_m": transport_seg,
        "x_path_segments": None
        if path_segments is None
        else [{"x_m": float(x), "v_mps": float(v)} for x, v in path_segments],
        "segment_completion_times": segment_completion_times,
        "scene_xml": str(scene_path),
        "dt_s": dt,
        "fps": args.fps,
        "duration_cap_s": args.duration,
        "settle_time_s": args.settle_time_s,
        "stop_tol_m": args.stop_tol_m,
        "a_decel_m_s2": args.a_decel,
        "k_x_hold_s_inv": args.k_x_hold,
        "torque_headroom": args.torque_headroom,
        "q_start": q_start.tolist(),
        "tool_start_world": tool_start_pos.tolist(),
        "x_goal_final_m": x_goal_final,
        "z_hold_m": z_hold,
        "v_x_cmd_final_mps": float(current_v_x_cmd),
        "goal_reached_at_s": goal_reached_at_s,
        "final_tool_world": final_tool_pos.tolist(),
        "final_x_error_m": final_x_error,
        "final_z_drift_m": final_z_drift,
        "final_orientation_error_deg": final_orientation_error_deg,
        "peak_tau_per_joint_nm": peak_tau_per_joint,
        "peak_tau_fraction_of_limit": peak_tau_fraction,
        "servo_force_limit_nm": SERVO_FORCE_LIMIT.tolist(),
        "time_s_trace": time_trace,
        "tool_x_trace": tool_x_trace,
        "tool_y_trace": tool_y_trace,
        "tool_z_trace": tool_z_trace,
        "orientation_error_deg_trace": orientation_error_trace_deg,
        "v_x_effective_trace": v_x_cmd_trace,
        "v_x_realized_cmd_trace": v_x_realized_trace,
        "tau_estimate_nm_trace": tau_estimate_trace,
        "speed_scale_trace": speed_scale_trace,
        "torque_scale_trace": torque_scale_trace,
        "success": success,
        "guardrail_config": None if guardrail_config is None else boundary_summary(guardrail_config),
        "guardrail_margin_m": float(args.guardrail_margin_m) if guardrail_config is not None else None,
        "video_path": str(video_path),
    }
    json_path.write_text(json.dumps(summary, indent=2))

    print(f"Final tool xyz (m): {np.array2string(final_tool_pos, precision=6)}")
    print(f"Final x error (m): {final_x_error:+.6f}")
    print(f"Final z drift (m): {final_z_drift:+.6f}")
    print(f"Final orientation error (deg): {final_orientation_error_deg:.4f}")
    print(f"Peak |tau| fraction of forcerange: {peak_tau_fraction:.3f}")
    print(f"Goal reached at (s): {goal_reached_at_s}")
    print(f"Success: {success}")
    print(f"Saved video:   {video_path}")
    print(f"Saved summary: {json_path}")


if __name__ == "__main__":
    main()
