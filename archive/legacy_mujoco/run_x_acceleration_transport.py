#!/usr/bin/env python3
"""
Acceleration-commanded UR5e `attachment_site` transport along world X.

The outer loop only supplies a signed world-X acceleration ``a_x`` (m/s^2); sign
sets direction. An internal velocity state is integrated each step and mapped
through ``acceleration_x_transport_controller`` to a joint reference. The inner
actuation mode is selectable:

- ``position_servo``: the existing MuJoCo position-actuator transport lane.
- ``joint_torque``: a simulation-only joint-space impedance controller that
  tracks the generated joint reference with direct ``qfrc_applied`` torques.

The simulation-only joint-torque lane also supports an optional hardware-shadow
actuation model that injects command delay, slew-rate limits, under-delivery,
deadzone, and simple friction mismatch before torques are applied to MuJoCo.

Optional ``--controller lqr`` mode closes the outer loop on measured X position
and velocity with a fixed-X transport LQR before the same low-level transport
reference generator and safety gates.

Schedule (pick one):

- **Default:** built-in demo piecewise schedule (returns to v_x ≈ 0 by t ≈ 7 s).
- **Constant:** ``--a-x-constant 0.03`` for the whole run.
- **Windows:** repeat ``--accel-window T0 T1 A`` (half-open ``[T0, T1)`` in seconds).
- **File:** ``--a-schedule path`` — text lines ``t0 t1 a_mps2`` (``#`` comments ok),
  or JSON ``{\"windows\": [{\"t0\",\"t1\",\"a_mps2\"}, ...]}``.

Height (world Z), tool orientation, and shoulder pan are regulated inside the
controller (same weighted differential IK family as ``velocity_x_transport_controller``).

Run with:
  MUJOCO_GL=egl xvfb-run -a python simulation/run_x_acceleration_transport.py --no-video

Fixed-Z workspace init (recommended, matches velocity recordings at z ≈ 0.54 m):
  python simulation/run_x_acceleration_transport.py \\
    --init-json outputs/control_runs/fixed_z_x_transport_firstpass_z0.540_seed1.json \\
    --no-video
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

if "MUJOCO_GL" not in os.environ:
    os.environ["MUJOCO_GL"] = "glfw"

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from progress_bars import simulation_progress

try:
    import mediapy
except Exception:  # pragma: no cover - video writing is optional in headless runs
    mediapy = None

SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from controller import (
    ACCEL_X_DEFAULT_A_MAX_M_S2,
    ACCEL_X_DEFAULT_V_MAX_M_S,
    ACTIVE_ORIGIN_Q,
    SERVO_FORCE_LIMIT,
    TARGET_SITE_ROTATION_WORLD,
    acceleration_x_transport_controller,
)
from controller_core import (
    HardwareShadowConfig,
    HardwareShadowModel,
    CommandGovernorSafetyFilter,
    ControllerCommand,
    ControllerState,
    FixedXTransportLQRConfig,
    FixedXTransportLQRController,
    JointImpedanceConfig,
    JointImpedanceController,
    SafetyLimits,
)
from workspace_guardrails import (
    DEFAULT_GUARDRAIL_CONFIG,
    boundary_summary,
    check_tcp_pose,
    load_guardrail_config,
    overlay_guardrails_on_frame,
)
MUJOCO_MENAGERIE = BASE_DIR / "mujoco_menagerie"
UR5E_SCENE = MUJOCO_MENAGERIE / "universal_robots_ur5e" / "scene.xml"
UR5E_CARTPOLE_SCENE = MUJOCO_MENAGERIE / "universal_robots_ur5e" / "scene_ur5e_cartpole.xml"
VIDEO_OUTPUT_DIR = BASE_DIR / "demonstration_videos" / "ur5e_cartpole"
SUMMARY_OUTPUT_DIR = BASE_DIR / "outputs" / "control_runs"
TOOL_SITE_NAME = "attachment_site"
BASE_PAN_INDEX = 0

# Default demo schedule (same shape as ``test_acceleration_x_controller.py``).
DEFAULT_ACCEL_WINDOWS: list[tuple[float, float, float]] = [
    (0.0, 1.5, 0.03),
    (1.5, 2.5, 0.0),
    (2.5, 5.5, -0.03),
    (5.5, 7.0, 0.03),
]

JOINT_TORQUE_DEFAULT_TAU_MAX_NM = np.array([150.0, 150.0, 150.0, 28.0, 28.0, 28.0], dtype=np.float64)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--duration",
        type=float,
        default=7.5,
        help="Simulation duration in seconds (default matches built-in demo schedule).",
    )
    p.add_argument("--fps", type=int, default=50, help="Video frame rate.")
    p.add_argument("--width", type=int, default=960, help="Render width.")
    p.add_argument("--height", type=int, default=720, help="Render height.")
    p.add_argument(
        "--no-video",
        action="store_true",
        help="Skip rendering and video write (JSON summary only).",
    )
    p.add_argument(
        "--torque-headroom",
        type=float,
        default=0.9,
        help="Same meaning as in ``acceleration_x_transport_controller``.",
    )
    p.add_argument(
        "--a-x-max",
        type=float,
        default=ACCEL_X_DEFAULT_A_MAX_M_S2,
        help="Clip magnitude of outer-loop a_x_cmd (m/s^2).",
    )
    p.add_argument(
        "--v-x-max",
        type=float,
        default=ACCEL_X_DEFAULT_V_MAX_M_S,
        help="Clip magnitude of integrated v_x state (m/s).",
    )
    p.add_argument(
        "--a-x-constant",
        type=float,
        default=None,
        help="If set, use this a_x (m/s^2) for the entire run (overrides schedule).",
    )
    p.add_argument(
        "--accel-window",
        nargs=3,
        type=float,
        metavar=("T0_S", "T1_S", "A_MPS2"),
        action="append",
        default=None,
        help="Append one half-open time window [T0, T1) with constant a_x (m/s^2). Repeatable.",
    )
    p.add_argument(
        "--a-schedule",
        type=Path,
        default=None,
        help="Text (t0 t1 a per line) or JSON {\"windows\": [...]} schedule file.",
    )
    p.add_argument(
        "--init-json",
        type=Path,
        default=None,
        help="Optional workspace JSON with `start_q` (recommended: fixed_z_x_transport_*).",
    )
    p.add_argument("--video-name", type=str, default=None, help="Optional output video file name.")
    p.add_argument("--json-name", type=str, default=None, help="Optional output JSON file name.")
    p.add_argument(
        "--draw-guardrails",
        action="store_true",
        help="Draw the extracted lab workspace guardrails as a 2D inset on each video frame.",
    )
    p.add_argument(
        "--actuation-mode",
        choices=["position_servo", "joint_torque"],
        default="position_servo",
        help=(
            "Inner actuation mode. position_servo preserves the existing transport lane; "
            "joint_torque uses direct MuJoCo torques with a joint-space impedance inner loop."
        ),
    )
    p.add_argument(
        "--controller",
        choices=["open-loop", "lqr"],
        default="open-loop",
        help="Controller mode for the outer X acceleration command.",
    )
    p.add_argument(
        "--x-goal",
        type=float,
        default=None,
        help="Absolute world-X target for LQR mode. Defaults to the init JSON transport bound when available.",
    )
    p.add_argument(
        "--workspace-x-goal",
        choices=["x_start", "x_stop"],
        default="x_stop",
        help="When LQR mode uses init-json transport limits, choose which bound to target.",
    )
    p.add_argument(
        "--lqr-q-x",
        type=float,
        default=60.0,
        help="LQR weight on X position error.",
    )
    p.add_argument(
        "--lqr-q-xdot",
        type=float,
        default=8.0,
        help="LQR weight on X velocity.",
    )
    p.add_argument(
        "--lqr-r-weight",
        type=float,
        default=1.0,
        help="LQR control penalty on acceleration command.",
    )
    p.add_argument(
        "--lqr-command-change-per-cycle",
        type=float,
        default=0.05,
        help="Additional command-governor slew limit for LQR mode.",
    )
    p.add_argument(
        "--joint-kp-scale",
        type=float,
        default=1.0,
        help="Scale factor applied to the default joint impedance proportional gains.",
    )
    p.add_argument(
        "--joint-kd-scale",
        type=float,
        default=1.0,
        help="Scale factor applied to the default joint impedance derivative gains.",
    )
    p.add_argument(
        "--joint-tau-max-scale",
        type=float,
        default=1.0,
        help="Scale factor applied to the default joint torque saturation limits.",
    )
    p.add_argument(
        "--joint-gravity-comp",
        action="store_true",
        help="Add MuJoCo qfrc_bias feedforward in joint-torque actuation mode.",
    )
    p.add_argument(
        "--hardware-shadow",
        action="store_true",
        help=(
            "Enable the simulation-only hardware-shadow torque model. "
            "This only applies in joint_torque actuation mode."
        ),
    )
    p.add_argument(
        "--shadow-command-delay-steps",
        type=int,
        default=0,
        help="Torque-command delay in simulation steps used by the hardware-shadow model.",
    )
    p.add_argument(
        "--shadow-torque-scale",
        type=float,
        default=1.0,
        help="Scalar under-delivery factor applied by the hardware-shadow model.",
    )
    p.add_argument(
        "--shadow-torque-rate-limit-nm-per-s",
        type=float,
        default=None,
        help=(
            "Per-joint torque slew-rate limit in Nm/s used by the hardware-shadow model. "
            "Use a finite value to mimic actuator bandwidth; omit for no rate limit."
        ),
    )
    p.add_argument(
        "--shadow-viscous-damping-nm-per-rads",
        type=float,
        default=0.0,
        help="Viscous torque loss term used by the hardware-shadow model.",
    )
    p.add_argument(
        "--shadow-coulomb-friction-nm",
        type=float,
        default=0.0,
        help="Coulomb friction term used by the hardware-shadow model.",
    )
    p.add_argument(
        "--shadow-deadzone-nm",
        type=float,
        default=0.0,
        help="Deadzone magnitude used by the hardware-shadow model.",
    )
    p.add_argument(
        "--shadow-friction-velocity-eps-rads",
        type=float,
        default=1e-3,
        help="Smoothing epsilon for the hardware-shadow friction sign approximation.",
    )
    p.add_argument(
        "--terminal-brake-start",
        type=float,
        default=None,
        help=(
            "Optional time (s) when the simulation-only terminal brake phase begins. "
            "If omitted, the run stays open-loop outside any LQR mode."
        ),
    )
    p.add_argument(
        "--terminal-target-x",
        type=float,
        default=None,
        help=(
            "Terminal brake target world-X position in meters. "
            "Defaults to the seed transport origin (the init JSON start pose)."
        ),
    )
    p.add_argument(
        "--terminal-target-vx",
        type=float,
        default=0.0,
        help="Terminal brake target world-X velocity in m/s.",
    )
    p.add_argument(
        "--terminal-kx",
        type=float,
        default=1.0,
        help="Terminal brake proportional gain on X position error.",
    )
    p.add_argument(
        "--terminal-kv",
        type=float,
        default=1.5,
        help="Terminal brake proportional gain on X velocity error.",
    )
    p.add_argument(
        "--terminal-max-accel",
        type=float,
        default=None,
        help=(
            "Optional hard cap on the terminal brake acceleration command. "
            "Defaults to --a-x-max when the brake phase is enabled."
        ),
    )
    p.add_argument(
        "--terminal-max-jerk",
        type=float,
        default=None,
        help=(
            "Optional terminal brake jerk limit in m/s^3. "
            "When set, the brake command is rate-limited by jerk * dt."
        ),
    )
    p.add_argument(
        "--terminal-guardrail-margin",
        type=float,
        default=None,
        help=(
            "Optional extra guardrail margin used by the terminal brake command governor. "
            "Defaults to --guardrail-margin-m when the brake phase is enabled."
        ),
    )
    p.add_argument(
        "--guardrail-config",
        type=Path,
        default=DEFAULT_GUARDRAIL_CONFIG,
        help="YAML config produced from the external Einksul scene.",
    )
    p.add_argument(
        "--guardrail-margin-m",
        type=float,
        default=0.02,
        help="Additional conservative margin applied when checking the trajectory.",
    )
    p.add_argument(
        "--show-boundary-labels",
        action="store_true",
        help="Annotate the boundary inset with names.",
    )
    p.add_argument(
        "--annotation-corner",
        choices=["top-left", "top-right", "bottom-left", "bottom-right"],
        default="top-left",
        help="Corner used for the main text overlay panel.",
    )
    p.add_argument(
        "--guardrail-corner",
        choices=["top-left", "top-right", "bottom-left", "bottom-right"],
        default="bottom-right",
        help="Corner used for the guardrail inset panel.",
    )
    return p.parse_args()


def write_video_ffmpeg(path: Path, frames: list[np.ndarray], fps: int) -> None:
    if not frames:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    height, width, _channels = frames[0].shape
    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s:v",
        f"{width}x{height}",
        "-r",
        str(fps),
        "-i",
        "-",
        "-an",
        "-vcodec",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        str(path),
    ]
    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    if proc.stdin is None:
        raise RuntimeError("ffmpeg stdin pipe was not available")
    for frame in frames:
        proc.stdin.write(np.asarray(frame, dtype=np.uint8).tobytes())
    proc.stdin.close()
    stderr = proc.stderr.read() if proc.stderr is not None else b""
    return_code = proc.wait()
    if return_code != 0:
        raise RuntimeError(
            f"ffmpeg failed (code {return_code}): "
            f"{stderr.decode('utf-8', errors='replace') if stderr else ''}"
        )


def _neutralize_position_servos(model: mujoco.MjModel) -> None:
    """Disable MuJoCo position-servo effects so qfrc_applied is the only actuation path."""
    if model.nu == 0:
        return
    model.actuator_gainprm[: model.nu, :] = 0.0
    model.actuator_biasprm[: model.nu, :] = 0.0


def _load_workspace_init(path: Path) -> tuple[np.ndarray, dict]:
    report = json.loads(path.read_text())
    if "start_q" not in report:
        raise KeyError(f"{path} must contain `start_q`")
    q = np.asarray(report["start_q"], dtype=np.float64)
    return q, report


def _transport_x_limits(report: dict | None, *, margin_default: float = 0.05) -> dict[str, float] | None:
    """Normalize workspace JSONs that describe the usable X transport interval."""
    if report is None:
        return None
    if "x_transport_limits_m" in report:
        return {k: float(v) for k, v in report["x_transport_limits_m"].items()}
    if "x_transport_limits" in report:
        return {k: float(v) for k, v in report["x_transport_limits"].items()}
    if "x_limits_m" in report:
        lim = report["x_limits_m"]
        margin = float(margin_default)
        return {
            "x_start": float(lim["x_min"]) + margin,
            "x_stop": float(lim["x_max"]) - margin,
            "margin": margin,
        }
    ex = report.get("x_exact_limits_m")
    if isinstance(ex, dict) and "x_min" in ex and "x_max" in ex:
        margin = float(margin_default)
        return {
            "x_start": float(ex["x_min"]) + margin,
            "x_stop": float(ex["x_max"]) - margin,
            "margin": margin,
        }
    return None


def _parse_schedule_text(raw: str) -> list[tuple[float, float, float]]:
    rows: list[tuple[float, float, float]] = []
    for line in raw.splitlines():
        line = line.split("#")[0].strip()
        if not line:
            continue
        parts = line.replace(",", " ").split()
        if len(parts) != 3:
            raise ValueError(f"Each schedule line must be t0 t1 a_mps2; got: {line!r}")
        rows.append((float(parts[0]), float(parts[1]), float(parts[2])))
    if not rows:
        raise ValueError("Schedule file has no rows.")
    return rows


def _load_schedule_file(path: Path) -> list[tuple[float, float, float]]:
    text = path.read_text()
    if path.suffix.lower() == ".json":
        data = json.loads(text)
        if isinstance(data, dict) and "windows" in data:
            out = []
            for w in data["windows"]:
                out.append((float(w["t0"]), float(w["t1"]), float(w["a_mps2"])))
            return out
        raise ValueError("JSON schedule must be {\"windows\": [{t0,t1,a_mps2}, ...]}")
    return _parse_schedule_text(text)


def _resolve_windows(args: argparse.Namespace) -> list[tuple[float, float, float]]:
    if args.a_x_constant is not None:
        return [(0.0, float("inf"), float(args.a_x_constant))]
    if args.a_schedule is not None:
        return _load_schedule_file(args.a_schedule)
    if args.accel_window:
        return [(float(t0), float(t1), float(a)) for t0, t1, a in args.accel_window]
    return list(DEFAULT_ACCEL_WINDOWS)


def _a_x_at_time(t: float, windows: list[tuple[float, float, float]]) -> float:
    for t0, t1, a in sorted(windows, key=lambda w: w[0]):
        if t0 <= t < t1:
            return float(a)
    return 0.0


def _schedule_fingerprint(windows: list[tuple[float, float, float]]) -> str:
    payload = json.dumps([[w[0], w[1], w[2]] for w in windows], separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()[:10]


def _clamp(value: float, lower: float, upper: float) -> float:
    return float(np.clip(float(value), float(lower), float(upper)))


def _phase_label(raw_a_x_cmd: float, *, terminal_brake_active: bool) -> str:
    if terminal_brake_active:
        return "terminal_brake"
    return "transport" if abs(float(raw_a_x_cmd)) > 1e-12 else "coast"


def _terminal_brake_command(
    *,
    x_now: float,
    x_dot_now: float,
    target_x: float,
    target_vx: float,
    kx: float,
    kv: float,
    max_accel: float,
    prev_command: float | None,
    dt: float,
    max_jerk: float | None,
) -> tuple[float, float, bool, list[str]]:
    """Return a clipped terminal brake command and its diagnostics."""
    raw = float(kx * (target_x - x_now) + kv * (target_vx - x_dot_now))
    command = raw
    clipped = False
    reasons: list[str] = []

    accel_limit = abs(float(max_accel))
    limited = _clamp(command, -accel_limit, accel_limit)
    if abs(limited - command) > 1e-12:
        clipped = True
        reasons.append(f"terminal brake accel clipped from {command:.6f} to {limited:.6f}")
    command = limited

    delta_limit = abs(float(max_jerk)) * max(float(dt), 1e-6) if max_jerk is not None else accel_limit
    if prev_command is not None:
        limited = _clamp(command, float(prev_command) - delta_limit, float(prev_command) + delta_limit)
        if abs(limited - command) > 1e-12:
            clipped = True
            reasons.append(
                f"terminal brake rate clipped to [{float(prev_command) - delta_limit:.6f}, "
                f"{float(prev_command) + delta_limit:.6f}]"
            )
        command = limited

    return raw, command, clipped, reasons


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


def _corner_rect(
    image_size: tuple[int, int],
    inset_size: tuple[int, int],
    corner: str,
    margin_px: int,
) -> tuple[int, int, int, int]:
    width, height = image_size
    inset_w, inset_h = inset_size
    if corner == "top-left":
        left = margin_px
        top = margin_px
    elif corner == "top-right":
        left = width - inset_w - margin_px
        top = margin_px
    elif corner == "bottom-left":
        left = margin_px
        top = height - inset_h - margin_px
    elif corner == "bottom-right":
        left = width - inset_w - margin_px
        top = height - inset_h - margin_px
    else:
        raise ValueError(
            f"Unknown corner={corner!r}; expected top-left, top-right, bottom-left, or bottom-right"
        )
    return left, top, left + inset_w, top + inset_h


def annotate_frame(frame: np.ndarray, lines: list[str], *, corner: str = "top-left") -> np.ndarray:
    image = Image.fromarray(frame)
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    font = ImageFont.load_default()
    line_height = 16
    panel_width = min(620, image.size[0] - 20)
    panel_height = 18 + line_height * len(lines)
    left, top, right, bottom = _corner_rect(
        (image.size[0], image.size[1]),
        (panel_width, panel_height),
        corner,
        margin_px=10,
    )
    draw.rounded_rectangle(
        (left, top, right, bottom),
        radius=10,
        fill=(0, 0, 0, 165),
        outline=(255, 255, 255, 64),
    )
    y = top + 8
    for line in lines:
        draw.text((left + 10, y), line, fill=(255, 255, 255, 255), font=font)
        y += line_height
    combined = Image.alpha_composite(image.convert("RGBA"), overlay)
    return np.asarray(combined.convert("RGB"))


def main() -> None:
    args = parse_args()
    guardrail_config = load_guardrail_config(args.guardrail_config) if args.draw_guardrails else None
    VIDEO_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARY_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    windows = _resolve_windows(args)
    schedule_tag = _schedule_fingerprint(windows)
    if args.a_x_constant is not None:
        schedule_tag = f"aconst{float(args.a_x_constant):+.4f}".replace(".", "p")

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
            raise ValueError(f"start_q length {q_start.shape[0]} != model.nu={model.nu}")
        print(f"Workspace init from: {args.init_json}")
    else:
        q_start = ACTIVE_ORIGIN_Q.copy()

    data.qpos[:] = 0.0
    data.qvel[:] = 0.0
    data.qpos[: model.nu] = q_start
    data.ctrl[:] = q_start
    mujoco.mj_forward(model, data)

    if args.actuation_mode == "joint_torque":
        if args.joint_kp_scale < 0.0 or args.joint_kd_scale < 0.0 or args.joint_tau_max_scale <= 0.0:
            raise ValueError("joint impedance scales must be non-negative and tau scale must be positive")
        _neutralize_position_servos(model)
        data.ctrl[:] = 0.0

    joint_impedance_controller: JointImpedanceController | None = None
    joint_tau_max_nm_vec: np.ndarray | None = None
    hardware_shadow_model: HardwareShadowModel | None = None
    if args.actuation_mode == "joint_torque":
        base_joint_cfg = JointImpedanceConfig()
        joint_tau_max_nm_vec = JOINT_TORQUE_DEFAULT_TAU_MAX_NM * float(args.joint_tau_max_scale)
        joint_impedance_cfg = JointImpedanceConfig(
            kp_nm_per_rad=np.asarray(base_joint_cfg.kp_nm_per_rad, dtype=np.float64) * float(args.joint_kp_scale),
            kd_nm_per_rad_s=np.asarray(base_joint_cfg.kd_nm_per_rad_s, dtype=np.float64) * float(args.joint_kd_scale),
            tau_max_nm=joint_tau_max_nm_vec,
        )
        joint_impedance_controller = JointImpedanceController(joint_impedance_cfg)
        print(
            "Joint-torque actuation enabled: "
            f"kp_scale={args.joint_kp_scale:.3f} kd_scale={args.joint_kd_scale:.3f} "
            f"tau_scale={args.joint_tau_max_scale:.3f} gravity_comp={'yes' if args.joint_gravity_comp else 'no'}"
        )
        if args.hardware_shadow:
            shadow_rate_limit_vec = (
                np.full(model.nu, np.inf, dtype=np.float64)
                if args.shadow_torque_rate_limit_nm_per_s is None
                else np.full(model.nu, float(args.shadow_torque_rate_limit_nm_per_s), dtype=np.float64)
            )
            hardware_shadow_cfg = HardwareShadowConfig(
                tau_max_nm=joint_tau_max_nm_vec,
                command_delay_steps=int(args.shadow_command_delay_steps),
                torque_scale=float(args.shadow_torque_scale),
                torque_rate_limit_nm_per_s=shadow_rate_limit_vec,
                viscous_damping_nm_per_rad_s=np.full(
                    model.nu, float(args.shadow_viscous_damping_nm_per_rads), dtype=np.float64
                ),
                coulomb_friction_nm=np.full(model.nu, float(args.shadow_coulomb_friction_nm), dtype=np.float64),
                deadzone_nm=np.full(model.nu, float(args.shadow_deadzone_nm), dtype=np.float64),
                friction_velocity_eps_rad_s=float(args.shadow_friction_velocity_eps_rads),
                dt_s=dt,
            )
            hardware_shadow_model = HardwareShadowModel(hardware_shadow_cfg)
            print(
                "Hardware-shadow actuation enabled: "
                f"delay_steps={args.shadow_command_delay_steps} "
                f"torque_scale={args.shadow_torque_scale:.3f} "
                f"rate_limit_nm_per_s={args.shadow_torque_rate_limit_nm_per_s} "
                f"viscous={args.shadow_viscous_damping_nm_per_rads:.3f} "
                f"coulomb={args.shadow_coulomb_friction_nm:.3f} "
                f"deadzone={args.shadow_deadzone_nm:.3f}"
            )
    elif args.hardware_shadow:
        raise ValueError("--hardware-shadow is only supported with --actuation-mode joint_torque")

    tool_start_pos = data.site_xpos[tool_site_id].copy()
    z_hold = float(tool_start_pos[2])
    pan_target = float(q_start[BASE_PAN_INDEX])
    ref_tool_pos = tool_start_pos.copy()

    transport_limits = _transport_x_limits(workspace_report)
    x_goal_m: float | None = None
    x_goal_source: str | None = None
    lqr_controller: FixedXTransportLQRController | None = None
    lqr_filter: CommandGovernorSafetyFilter | None = None
    lqr_gain_matrix: list[list[float]] | None = None
    lqr_converged: bool | None = None
    lqr_iters: int | None = None
    lqr_clipped_count = 0
    lqr_rejected_count = 0

    if args.controller == "lqr":
        if args.x_goal is not None:
            x_goal_m = float(args.x_goal)
            x_goal_source = "cli"
        elif transport_limits is not None and args.workspace_x_goal in transport_limits:
            x_goal_m = float(transport_limits[args.workspace_x_goal])
            x_goal_source = f"workspace_{args.workspace_x_goal}"
        else:
            x_goal_m = float(ref_tool_pos[0] + 0.10)
            x_goal_source = "relative_default"

        lqr_cfg = FixedXTransportLQRConfig(
            q_weights=np.array([float(args.lqr_q_x), float(args.lqr_q_xdot)], dtype=np.float64),
            r_weight=float(args.lqr_r_weight),
            dt_s=dt,
            target_x=float(x_goal_m),
            command_limit=float(args.a_x_max),
            output_mode="x_acceleration",
        )
        lqr_controller = FixedXTransportLQRController(lqr_cfg)

        move_margin = max(float(args.guardrail_margin_m), 0.01)
        x_min = float(min(ref_tool_pos[0], x_goal_m) - move_margin)
        x_max = float(max(ref_tool_pos[0], x_goal_m) + move_margin)

        lqr_limits = SafetyLimits(
            x_min_m=x_min,
            x_max_m=x_max,
            x_warning_margin_m=move_margin,
            max_x_velocity_mps=float(args.v_x_max),
            max_x_acceleration_mps2=float(args.a_x_max),
            max_command_change_per_cycle=float(args.lqr_command_change_per_cycle),
            dt_s=dt,
            reject_on_violation=True,
            fallback_action="brake",
        )
        lqr_filter = CommandGovernorSafetyFilter(lqr_limits)
        lqr_gain_matrix = lqr_controller.gain_matrix.tolist()
        lqr_converged = bool(lqr_controller.riccati_converged)
        lqr_iters = int(lqr_controller.riccati_iters)
        print(f"LQR x goal (m): {x_goal_m:.6f}  source={x_goal_source}")

    terminal_brake_enabled = args.terminal_brake_start is not None
    terminal_brake_start_s = float(args.terminal_brake_start) if terminal_brake_enabled else None
    terminal_target_x_m = float(ref_tool_pos[0] if args.terminal_target_x is None else args.terminal_target_x)
    terminal_target_vx_mps = float(args.terminal_target_vx)
    terminal_kx = float(args.terminal_kx)
    terminal_kv = float(args.terminal_kv)
    terminal_max_accel_mps2 = (
        float(args.terminal_max_accel) if args.terminal_max_accel is not None else float(args.a_x_max)
    )
    terminal_max_jerk_mps3 = float(args.terminal_max_jerk) if args.terminal_max_jerk is not None else None
    terminal_guardrail_margin_m = (
        float(args.terminal_guardrail_margin)
        if args.terminal_guardrail_margin is not None
        else float(args.guardrail_margin_m)
    )
    terminal_filter: CommandGovernorSafetyFilter | None = None
    terminal_safety_limits: SafetyLimits | None = None
    if terminal_brake_enabled:
        if terminal_max_accel_mps2 <= 0.0:
            raise ValueError("--terminal-max-accel must be positive when the terminal brake is enabled")
        if terminal_max_jerk_mps3 is not None and terminal_max_jerk_mps3 < 0.0:
            raise ValueError("--terminal-max-jerk must be non-negative when provided")
        if transport_limits is not None and "x_start" in transport_limits and "x_stop" in transport_limits:
            x_min = float(min(transport_limits["x_start"], transport_limits["x_stop"]))
            x_max = float(max(transport_limits["x_start"], transport_limits["x_stop"]))
        else:
            span_pad = max(float(terminal_guardrail_margin_m), 0.05)
            x_min = float(min(ref_tool_pos[0], terminal_target_x_m) - span_pad)
            x_max = float(max(ref_tool_pos[0], terminal_target_x_m) + span_pad)
        x_min -= float(terminal_guardrail_margin_m)
        x_max += float(terminal_guardrail_margin_m)
        if x_min >= x_max:
            x_min = float(min(ref_tool_pos[0], terminal_target_x_m) - 0.10)
            x_max = float(max(ref_tool_pos[0], terminal_target_x_m) + 0.10)
        command_change_limit = (
            float(terminal_max_jerk_mps3 * dt)
            if terminal_max_jerk_mps3 is not None
            else float(terminal_max_accel_mps2)
        )
        terminal_safety_limits = SafetyLimits(
            x_min_m=x_min,
            x_max_m=x_max,
            x_warning_margin_m=float(max(terminal_guardrail_margin_m, 0.0)),
            max_x_velocity_mps=float(args.v_x_max),
            max_x_acceleration_mps2=float(terminal_max_accel_mps2),
            max_command_change_per_cycle=float(max(command_change_limit, 0.0)),
            dt_s=dt,
            reject_on_violation=True,
            fallback_action="brake",
            brake_gain=float(max(terminal_kx, terminal_kv, 1.0)),
        )
        terminal_filter = CommandGovernorSafetyFilter(terminal_safety_limits)
        print(
            "Terminal brake enabled: "
            f"start={terminal_brake_start_s:.2f}s target_x={terminal_target_x_m:+.6f} "
            f"target_vx={terminal_target_vx_mps:+.6f} kx={terminal_kx:.3f} kv={terminal_kv:.3f} "
            f"max_accel={terminal_max_accel_mps2:.3f} max_jerk={terminal_max_jerk_mps3}"
        )

    n_frames = int(round(args.duration * args.fps))
    sim_steps_per_frame = max(1, int(round((1.0 / args.fps) / dt)))

    time_trace: list[float] = []
    tool_x_trace: list[float] = []
    tool_y_trace: list[float] = []
    tool_z_trace: list[float] = []
    orientation_error_trace_deg: list[float] = []
    a_x_effective_trace: list[float] = []
    v_x_state_trace: list[float] = []
    v_x_realized_trace: list[float] = []
    tau_estimate_trace: list[list[float]] = []
    speed_scale_trace: list[float] = []
    torque_scale_trace: list[float] = []
    tau_command_trace: list[list[float]] = []
    tau_applied_trace: list[list[float]] = []
    tau_shadow_delay_trace: list[list[float]] = []
    tau_shadow_rate_trace: list[list[float]] = []
    tau_shadow_deadzone_trace: list[list[float]] = []
    tau_shadow_friction_trace: list[list[float]] = []
    q_ref_trace: list[list[float]] = []
    qd_ref_trace: list[list[float]] = []
    phase_trace: list[str] = []
    lqr_raw_cmd_trace: list[float | None] = []
    lqr_safe_cmd_trace: list[float | None] = []
    lqr_clipped_trace: list[bool] = []
    lqr_rejected_trace: list[bool] = []
    lqr_x_error_trace: list[float | None] = []
    lqr_vx_error_trace: list[float | None] = []
    terminal_brake_raw_trace: list[float | None] = []
    terminal_brake_safe_trace: list[float | None] = []
    terminal_brake_clipped_trace: list[bool] = []
    terminal_guardrail_clip_trace: list[bool] = []
    terminal_guardrail_reject_trace: list[bool] = []
    terminal_x_error_trace: list[float | None] = []
    terminal_vx_error_trace: list[float | None] = []
    lqr_clip_count = 0
    lqr_reject_count = 0
    terminal_brake_clip_count = 0
    terminal_guardrail_clip_count = 0
    terminal_guardrail_reject_count = 0
    terminal_max_command_accel = 0.0
    terminal_max_command_jerk = 0.0
    joint_impedance_saturated_count = 0
    hardware_shadow_clip_count = 0
    hardware_shadow_delay_count = 0
    hardware_shadow_rate_limit_count = 0
    hardware_shadow_deadzone_count = 0
    hardware_shadow_friction_count = 0

    ctrl = q_start.copy()
    v_x_state = 0.0
    prev_tool_x: float | None = None
    prev_outer_cmd: float | None = None
    tool_jacp = np.zeros((3, model.nv), dtype=np.float64)
    tool_jacr = np.zeros((3, model.nv), dtype=np.float64)

    last_diag: dict = {
        "a_x_cmd": 0.0,
        "a_x_raw_cmd": 0.0,
        "a_x_safe_cmd": 0.0,
        "v_x_state_next": 0.0,
        "v_x_realized_cmd": 0.0,
        "tau_estimate_nm": [0.0] * 6,
        "speed_scale": 1.0,
        "torque_scale": 1.0,
        "safety_severity": "ok",
        "safety_reasons": [],
        "x_error_m": 0.0,
    }

    render_width = min(args.width, int(model.vis.global_.offwidth))
    render_height = min(args.height, int(model.vis.global_.offheight))
    if render_width != args.width or render_height != args.height:
        print(
            f"Requested render size {args.width}x{args.height} exceeds framebuffer; "
            f"using {render_width}x{render_height}."
        )

    renderer = None
    camera = None
    frames: list[np.ndarray] = []
    if not args.no_video:
        renderer = mujoco.Renderer(model, height=render_height, width=render_width)
        camera = make_camera(model)

    print(
        f"Running x-acceleration transport (actuation={args.actuation_mode}) for {args.duration:.2f}s at {args.fps} fps"
    )
    print(f"Scene: {scene_path.name}  dt={dt:.5f}s  steps/frame={sim_steps_per_frame}")
    print(f"z_hold (m): {z_hold:.6f}  a_x_max={args.a_x_max}  v_x_max={args.v_x_max}")
    print(f"Schedule windows: {windows if args.a_x_constant is None else 'constant a_x'}")

    with simulation_progress(n_frames, "x-accel") as pbar:
        for _ in range(n_frames):
            for _ in range(sim_steps_per_frame):
                t = float(data.time)

                q = data.qpos[: model.nu].copy()
                qvel = data.qvel[: model.nu].copy()
                mujoco.mj_jacSite(model, data, tool_jacp, tool_jacr, tool_site_id)
                tool_pos = data.site_xpos[tool_site_id].copy()
                tool_rot = xmat_to_rot(data.site_xmat[tool_site_id])

                x_now = float(tool_pos[0])
                x_dot_est = 0.0 if prev_tool_x is None else float((x_now - prev_tool_x) / dt)
                prev_tool_x = x_now
                terminal_brake_active = bool(
                    terminal_brake_enabled
                    and terminal_brake_start_s is not None
                    and t >= terminal_brake_start_s
                )

                outer_diag: dict[str, object]
                phase = "transport"
                raw_schedule_cmd = float(_a_x_at_time(t, windows))
                controller_cmd = raw_schedule_cmd
                controller_raw_cmd = raw_schedule_cmd
                controller_reasons: list[str] = []
                controller_clipped = False
                controller_rejected = False
                controller_recoverability_score: float | None = None
                controller_intervention_level: str | None = None
                controller_mode_name = "open-loop"

                if args.controller == "lqr":
                    if lqr_controller is None or lqr_filter is None or x_goal_m is None:
                        raise RuntimeError("LQR mode was requested but not initialized correctly")
                    lqr_state = ControllerState(
                        x=x_now,
                        x_dot=x_dot_est,
                        theta=0.0,
                        theta_dot=0.0,
                        time_s=t,
                        dt_s=dt,
                        target_x=float(x_goal_m),
                        target_theta=0.0,
                    )
                    lqr_raw_command = lqr_controller.compute(lqr_state)
                    lqr_safe_result = lqr_filter.filter(lqr_state, lqr_raw_command)
                    controller_raw_cmd = float(lqr_raw_command.value)
                    controller_cmd = float(lqr_safe_result.command.value)
                    controller_mode_name = "lqr"
                    controller_clipped = bool(lqr_safe_result.clipped)
                    controller_rejected = bool(lqr_safe_result.rejected)
                    controller_recoverability_score = (
                        None
                        if lqr_safe_result.recoverability_score is None
                        else float(lqr_safe_result.recoverability_score)
                    )
                    controller_intervention_level = lqr_safe_result.intervention_level
                    controller_reasons = list(lqr_safe_result.reasons)
                    lqr_clip_count += int(controller_clipped)
                    lqr_reject_count += int(controller_rejected)
                    lqr_raw_cmd_trace.append(controller_raw_cmd)
                    lqr_safe_cmd_trace.append(float(controller_cmd))
                    lqr_clipped_trace.append(controller_clipped)
                    lqr_rejected_trace.append(controller_rejected)
                    lqr_x_error_trace.append(float(x_goal_m - x_now))
                    lqr_vx_error_trace.append(float(-x_dot_est))
                    phase = "transport" if abs(float(controller_cmd)) > 1e-12 else "coast"
                else:
                    lqr_raw_cmd_trace.append(None)
                    lqr_safe_cmd_trace.append(None)
                    lqr_clipped_trace.append(False)
                    lqr_rejected_trace.append(False)
                    lqr_x_error_trace.append(None)
                    lqr_vx_error_trace.append(None)

                if terminal_brake_active:
                    if terminal_filter is None:
                        raise RuntimeError("Terminal brake was enabled but not initialized correctly")
                    raw_brake_cmd, limited_brake_cmd, brake_clipped, brake_reasons = _terminal_brake_command(
                        x_now=x_now,
                        x_dot_now=x_dot_est,
                        target_x=terminal_target_x_m,
                        target_vx=terminal_target_vx_mps,
                        kx=terminal_kx,
                        kv=terminal_kv,
                        max_accel=terminal_max_accel_mps2,
                        prev_command=prev_outer_cmd,
                        dt=dt,
                        max_jerk=terminal_max_jerk_mps3,
                    )
                    brake_state = ControllerState(
                        x=x_now,
                        x_dot=x_dot_est,
                        theta=0.0,
                        theta_dot=0.0,
                        time_s=t,
                        dt_s=dt,
                        target_x=terminal_target_x_m,
                        target_theta=0.0,
                    )
                    brake_command = ControllerCommand(
                        mode="x_acceleration",
                        value=limited_brake_cmd,
                        time_s=t,
                        metadata={
                            "phase": "terminal_brake",
                            "raw_terminal_command": raw_brake_cmd,
                            "terminal_target_x_m": terminal_target_x_m,
                            "terminal_target_vx_mps": terminal_target_vx_mps,
                            "terminal_kx": terminal_kx,
                            "terminal_kv": terminal_kv,
                        },
                    )
                    safe_result = terminal_filter.filter(brake_state, brake_command)
                    a_cmd = float(safe_result.command.value)
                    phase = "terminal_brake"
                    terminal_brake_clip_count += int(bool(brake_clipped))
                    terminal_guardrail_clip_count += int(bool(safe_result.clipped))
                    terminal_guardrail_reject_count += int(bool(safe_result.rejected))
                    terminal_max_command_accel = max(terminal_max_command_accel, abs(a_cmd))
                    if prev_outer_cmd is not None:
                        terminal_max_command_jerk = max(
                            terminal_max_command_jerk,
                            abs(a_cmd - float(prev_outer_cmd)) / max(dt, 1e-6),
                        )
                    outer_diag = {
                        "phase": phase,
                        "controller_mode": controller_mode_name,
                        "a_x_cmd": a_cmd,
                        "a_x_raw_cmd": raw_brake_cmd,
                        "a_x_safe_cmd": a_cmd,
                        "x_error_m": float(terminal_target_x_m - x_now),
                        "v_x_error_mps": float(terminal_target_vx_mps - x_dot_est),
                        "safety_severity": safe_result.severity,
                        "safety_reasons": list(safe_result.reasons),
                        "safety_clipped": bool(safe_result.clipped),
                        "safety_rejected": bool(safe_result.rejected),
                        "recoverability_score": None
                        if safe_result.recoverability_score is None
                        else float(safe_result.recoverability_score),
                        "intervention_level": safe_result.intervention_level,
                        "terminal_brake_active": True,
                        "terminal_brake_clipped": bool(brake_clipped or safe_result.clipped or safe_result.rejected),
                        "terminal_brake_reasons": list(brake_reasons),
                        "terminal_guardrail_clipped": bool(safe_result.clipped),
                        "terminal_guardrail_rejected": bool(safe_result.rejected),
                        "controller_raw_cmd": float(controller_raw_cmd),
                        "controller_safe_cmd": float(controller_cmd),
                        "controller_clipped": bool(controller_clipped),
                        "controller_rejected": bool(controller_rejected),
                        "controller_reasons": list(controller_reasons),
                        "controller_recoverability_score": controller_recoverability_score,
                        "controller_intervention_level": controller_intervention_level,
                    }
                    terminal_brake_raw_trace.append(raw_brake_cmd)
                    terminal_brake_safe_trace.append(a_cmd)
                    terminal_brake_clipped_trace.append(bool(brake_clipped or safe_result.clipped or safe_result.rejected))
                    terminal_guardrail_clip_trace.append(bool(safe_result.clipped))
                    terminal_guardrail_reject_trace.append(bool(safe_result.rejected))
                    terminal_x_error_trace.append(float(terminal_target_x_m - x_now))
                    terminal_vx_error_trace.append(float(terminal_target_vx_mps - x_dot_est))
                    prev_outer_cmd = a_cmd
                else:
                    a_cmd = controller_cmd
                    if args.controller == "lqr":
                        phase = "transport" if abs(float(controller_cmd)) > 1e-12 else "coast"
                    else:
                        phase = _phase_label(raw_schedule_cmd, terminal_brake_active=False)
                    outer_diag = {
                        "phase": phase,
                        "controller_mode": controller_mode_name,
                        "a_x_cmd": float(a_cmd),
                        "a_x_raw_cmd": float(controller_raw_cmd),
                        "a_x_safe_cmd": float(a_cmd),
                        "x_error_m": float(terminal_target_x_m - x_now) if terminal_brake_enabled else 0.0,
                        "v_x_error_mps": float(terminal_target_vx_mps - x_dot_est) if terminal_brake_enabled else 0.0,
                        "safety_severity": "ok",
                        "safety_reasons": [],
                        "safety_clipped": False,
                        "safety_rejected": False,
                        "recoverability_score": None,
                        "intervention_level": None,
                        "controller_raw_cmd": float(controller_raw_cmd),
                        "controller_safe_cmd": float(controller_cmd),
                        "controller_clipped": bool(controller_clipped),
                        "controller_rejected": bool(controller_rejected),
                        "controller_reasons": list(controller_reasons),
                        "controller_recoverability_score": controller_recoverability_score,
                        "controller_intervention_level": controller_intervention_level,
                        "terminal_brake_active": False,
                        "terminal_brake_clipped": False,
                    }
                    terminal_brake_raw_trace.append(None)
                    terminal_brake_safe_trace.append(None)
                    terminal_brake_clipped_trace.append(False)
                    terminal_guardrail_clip_trace.append(False)
                    terminal_guardrail_reject_trace.append(False)
                    terminal_x_error_trace.append(None)
                    terminal_vx_error_trace.append(None)
                    prev_outer_cmd = a_cmd

                ctrl, last_diag = acceleration_x_transport_controller(
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
                    a_x_max_m_s2=float(args.a_x_max),
                    v_x_max_m_s=float(args.v_x_max),
                    torque_headroom=float(args.torque_headroom),
                )
                last_diag = {**outer_diag, **last_diag}
                v_x_state = float(last_diag["v_x_state_next"])
                phase_trace.append(str(last_diag.get("phase", phase)))
                if args.actuation_mode == "joint_torque":
                    if joint_impedance_controller is None:
                        raise RuntimeError("Joint torque actuation was requested but not initialized")
                    qd_ref_full = np.zeros(model.nu, dtype=np.float64)
                    qd_ref_red = np.asarray(last_diag.get("q_dot_des_red", [0.0] * 5), dtype=np.float64).reshape(5)
                    qd_ref_full[1:] = qd_ref_red
                    tau_ff = data.qfrc_bias[: model.nu].copy() if args.joint_gravity_comp else None
                    joint_out = joint_impedance_controller.compute(
                        q=q,
                        qd=qvel,
                        q_ref=ctrl,
                        qd_ref=qd_ref_full,
                        tau_feedforward=tau_ff,
                    )
                    shadow_out = None
                    tau_applied = joint_out.tau.copy()
                    if hardware_shadow_model is not None:
                        shadow_out = hardware_shadow_model.apply(joint_out.tau, qvel=qvel)
                        tau_applied = shadow_out.tau_applied_nm.copy()
                        hardware_shadow_delay_count += int(bool(shadow_out.delayed))
                        hardware_shadow_rate_limit_count += int(bool(shadow_out.rate_limited))
                        hardware_shadow_deadzone_count += int(bool(shadow_out.deadzone_applied))
                        hardware_shadow_friction_count += int(bool(shadow_out.friction_applied))
                        hardware_shadow_clip_count += int(bool(shadow_out.clipped))
                    data.ctrl[:] = 0.0
                    data.qfrc_applied[: model.nu] = tau_applied
                    joint_impedance_saturated_count += int(joint_out.saturated)
                    last_diag = {
                        **last_diag,
                        "actuation_mode": "joint_torque",
                        "joint_q_ref": ctrl.tolist(),
                        "joint_qd_ref": qd_ref_full.tolist(),
                        "joint_tau_command_nm": joint_out.tau.tolist(),
                        "joint_tau_applied_nm": tau_applied.tolist(),
                        "joint_tau_preclip_nm": joint_out.tau_preclip.tolist(),
                        "joint_tau_feedback_nm": joint_out.tau_feedback.tolist(),
                        "joint_tau_feedforward_nm": joint_out.tau_feedforward.tolist(),
                        "joint_impedance_saturated": bool(joint_out.saturated),
                        "hardware_shadow_enabled": bool(hardware_shadow_model is not None),
                        "hardware_shadow_output": None if shadow_out is None else shadow_out.as_dict(),
                    }
                    tau_command_trace.append(joint_out.tau.tolist())
                    tau_applied_trace.append(tau_applied.tolist())
                    if shadow_out is None:
                        tau_shadow_delay_trace.append(joint_out.tau.tolist())
                        tau_shadow_rate_trace.append(joint_out.tau.tolist())
                        tau_shadow_deadzone_trace.append(joint_out.tau.tolist())
                        tau_shadow_friction_trace.append(np.zeros(model.nu, dtype=np.float64).tolist())
                    else:
                        tau_shadow_delay_trace.append(shadow_out.tau_after_delay_nm.tolist())
                        tau_shadow_rate_trace.append(shadow_out.tau_after_rate_limit_nm.tolist())
                        tau_shadow_deadzone_trace.append(shadow_out.tau_after_deadzone_nm.tolist())
                        tau_shadow_friction_trace.append(shadow_out.friction_torque_nm.tolist())
                    q_ref_trace.append(ctrl.tolist())
                    qd_ref_trace.append(qd_ref_full.tolist())
                else:
                    data.qfrc_applied[:] = 0.0
                    data.ctrl[:] = ctrl
                    tau_command_trace.append(None)
                    tau_applied_trace.append(None)
                    tau_shadow_delay_trace.append(None)
                    tau_shadow_rate_trace.append(None)
                    tau_shadow_deadzone_trace.append(None)
                    tau_shadow_friction_trace.append(None)
                    q_ref_trace.append(None)
                    qd_ref_trace.append(None)
                mujoco.mj_step(model, data)

            tool_pos_now = data.site_xpos[tool_site_id].copy()
            tool_rot_now = xmat_to_rot(data.site_xmat[tool_site_id])
            ori_err = orientation_error_deg(tool_rot_now, TARGET_SITE_ROTATION_WORLD)

            time_trace.append(float(data.time))
            tool_x_trace.append(float(tool_pos_now[0]))
            tool_y_trace.append(float(tool_pos_now[1]))
            tool_z_trace.append(float(tool_pos_now[2]))
            orientation_error_trace_deg.append(ori_err)
            a_x_effective_trace.append(float(last_diag["a_x_effective"]))
            v_x_state_trace.append(float(last_diag["v_x_state_next"]))
            v_x_realized_trace.append(float(last_diag["v_x_realized_cmd"]))
            if args.actuation_mode == "joint_torque":
                tau_estimate_trace.append(
                    list(
                        last_diag.get(
                            "joint_tau_applied_nm",
                            last_diag.get("joint_tau_command_nm", last_diag["tau_estimate_nm"]),
                        )
                    )
                )
            else:
                tau_estimate_trace.append(list(last_diag["tau_estimate_nm"]))
            speed_scale_trace.append(float(last_diag["speed_scale"]))
            torque_scale_trace.append(float(last_diag["torque_scale"]))

            if renderer is not None and camera is not None:
                if args.actuation_mode == "joint_torque":
                    tau_now = np.asarray(last_diag.get("joint_tau_applied_nm", [0.0] * 6), dtype=np.float64)
                    tau_limit_now = joint_tau_max_nm_vec if joint_tau_max_nm_vec is not None else np.asarray(
                        JointImpedanceConfig().tau_max_nm, dtype=np.float64
                    )
                else:
                    tau_now = np.asarray(last_diag["tau_estimate_nm"], dtype=np.float64)
                    tau_limit_now = SERVO_FORCE_LIMIT
                tau_fraction = float(np.max(np.abs(tau_now) / np.maximum(tau_limit_now, 1e-9)))
                phase_now = str(last_diag.get("phase", "transport"))
                brake_clipped_now = bool(last_diag.get("terminal_brake_clipped", False))
                shadow_enabled_now = bool(last_diag.get("hardware_shadow_enabled", False))
                shadow_rate_limit_display = (
                    "inf"
                    if args.shadow_torque_rate_limit_nm_per_s is None
                    else f"{float(args.shadow_torque_rate_limit_nm_per_s):.1f}"
                )
                shadow_line = (
                    f"shadow=on delay={args.shadow_command_delay_steps} scale={args.shadow_torque_scale:.2f} "
                    f"rate={shadow_rate_limit_display} visc={args.shadow_viscous_damping_nm_per_rads:.2f} "
                    f"coul={args.shadow_coulomb_friction_nm:.2f} dz={args.shadow_deadzone_nm:.2f}"
                    if shadow_enabled_now
                    else "shadow=off"
                )
                brake_status_line = (
                    f"ctrl={controller_mode_name}  phase={phase_now}  act={args.actuation_mode}  "
                    f"brake clipped={'yes' if brake_clipped_now else 'no'}  "
                    f"{shadow_line}  "
                    f"gclip={terminal_guardrail_clip_count}  grej={terminal_guardrail_reject_count}"
                )
                target_status_line = (
                    f"x_tgt={terminal_target_x_m:+.3f}  vx_tgt={terminal_target_vx_mps:+.3f}  "
                    f"x_err={float(last_diag.get('x_error_m', 0.0)):+.3f}  "
                    f"vx_err={float(last_diag.get('v_x_error_mps', 0.0)):+.3f}"
                    if terminal_brake_enabled
                    else f"phase={phase_now}"
                )
                overlay_lines = [
                    f"t={data.time:5.2f}s  a_x={last_diag['a_x_effective']:+.4f} m/s^2  "
                    f"v_x={last_diag['v_x_state_next']:+.4f} m/s",
                    brake_status_line,
                    target_status_line,
                    f"tool xyz=({tool_pos_now[0]: .3f}, {tool_pos_now[1]: .3f}, {tool_pos_now[2]: .3f}) m",
                    (
                        f"z_drift={(tool_pos_now[2] - z_hold)*1000:+.2f} mm  "
                        f"ori_err={ori_err:.2f} deg"
                    ),
                    (
                        f"speed scale={last_diag['speed_scale']:.2f}  "
                        f"torque scale={last_diag['torque_scale']:.2f}  "
                        f"peak tau frac={tau_fraction:.2f}"
                    ),
                    "tau (N*m): " + ", ".join(f"{v:+6.1f}" for v in tau_now),
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
                    desired_xyz = ref_tool_pos.copy()
                    frame = overlay_guardrails_on_frame(
                        frame,
                        guardrail_config,
                        trajectory_xyz=np.column_stack((tool_x_trace, tool_y_trace, tool_z_trace)),
                        current_xyz=tool_pos_now,
                        desired_xyz=desired_xyz,
                        decision=guardrail_decision,
                        guardrail_margin_m=float(args.guardrail_margin_m),
                        show_labels=bool(args.show_boundary_labels),
                        inset_corner=args.guardrail_corner,
                    )
                    overlay_lines.append(
                        f"guardrail={guardrail_decision.state}  boundary={guardrail_decision.boundary_name or 'none'}"
                    )
                frames.append(annotate_frame(frame, overlay_lines, corner=args.annotation_corner))

            pbar.set_postfix_str(
                f"t={data.time:.1f}s/{args.duration:.1f}s | x={tool_pos_now[0]:+.3f} | "
                f"v={last_diag['v_x_state_next']:+.3f} | a={last_diag['a_x_effective']:+.3f}",
                refresh=False,
            )
            pbar.update(1)

    if renderer is not None:
        renderer.close()

    final_tool_pos = data.site_xpos[tool_site_id].copy()
    final_rot = xmat_to_rot(data.site_xmat[tool_site_id])
    final_orientation_error_deg = orientation_error_deg(final_rot, TARGET_SITE_ROTATION_WORLD)
    final_z_drift = float(final_tool_pos[2] - z_hold)
    final_v_x = float(v_x_state)
    terminal_final_x_error = (
        float(terminal_target_x_m - final_tool_pos[0]) if terminal_brake_enabled else None
    )
    terminal_final_vx_error = float(terminal_target_vx_mps - final_v_x) if terminal_brake_enabled else None

    z_arr = np.asarray(tool_z_trace, dtype=np.float64)
    max_z_transient_m = float(np.max(np.abs(z_arr - z_hold)))

    tau_estimate_array = np.asarray(tau_estimate_trace, dtype=np.float64)
    peak_tau_per_joint = np.max(np.abs(tau_estimate_array), axis=0).tolist() if len(tau_estimate_array) else []
    if args.actuation_mode == "joint_torque":
        tau_limit_array = joint_tau_max_nm_vec if joint_tau_max_nm_vec is not None else np.asarray(
            JointImpedanceConfig().tau_max_nm, dtype=np.float64
        )
    else:
        tau_limit_array = SERVO_FORCE_LIMIT
    peak_tau_fraction = (
        float(np.max(np.abs(tau_estimate_array) / tau_limit_array[None, :]))
        if len(tau_estimate_array)
        else 0.0
    )
    failure_reasons: list[str] = []
    if abs(final_z_drift) > 8.0e-3:
        failure_reasons.append(f"final Z drift too large: {final_z_drift:+.6f} m")
    if final_orientation_error_deg > 3.0:
        failure_reasons.append(f"final orientation error too large: {final_orientation_error_deg:.4f} deg")
    if abs(final_v_x) > 0.02:
        failure_reasons.append(f"final x velocity too large: {final_v_x:+.6f} m/s")
    if max_z_transient_m > 15.0e-3:
        failure_reasons.append(f"max Z transient too large: {max_z_transient_m:.6f} m")
    if terminal_brake_enabled:
        if terminal_final_x_error is None or abs(terminal_final_x_error) > 0.01:
            failure_reasons.append(
                "terminal brake final x error too large"
                if terminal_final_x_error is not None
                else "terminal brake final x error unavailable"
            )
        if terminal_final_vx_error is None or abs(terminal_final_vx_error) > 0.02:
            failure_reasons.append(
                "terminal brake final vx error too large"
                if terminal_final_vx_error is not None
                else "terminal brake final vx error unavailable"
            )

    success = bool(not failure_reasons)

    stem_parts = ["ur5e_x_acceleration_transport", schedule_tag]
    if args.init_json is not None:
        stem_parts.append(args.init_json.stem)
    stem = "_".join(stem_parts)
    video_path = VIDEO_OUTPUT_DIR / (args.video_name or f"{stem}.mp4")
    json_path = SUMMARY_OUTPUT_DIR / (args.json_name or f"{stem}.json")

    if not args.no_video and frames:
        if mediapy is not None:
            mediapy.write_video(video_path, frames, fps=args.fps)
        else:
            write_video_ffmpeg(video_path, frames, fps=args.fps)

    windows_serial = [
        {"t0_s": float(w[0]), "t1_s": float(w[1]) if w[1] < 1e300 else None, "a_mps2": float(w[2])}
        for w in windows
    ]

    summary = {
        "controller_name": f"acceleration_x_transport_controller_{args.actuation_mode}",
        "controller_mode": args.controller,
        "actuation_mode": args.actuation_mode,
        "init_json": str(args.init_json) if args.init_json is not None else None,
        "a_x_constant_mps2": float(args.a_x_constant) if args.a_x_constant is not None else None,
        "accel_windows": windows_serial,
        "scene_xml": str(scene_path),
        "dt_s": dt,
        "fps": args.fps,
        "duration_s": args.duration,
        "a_x_max_m_s2": float(args.a_x_max),
        "v_x_max_m_s": float(args.v_x_max),
        "torque_headroom": float(args.torque_headroom),
        "lqr_enabled": bool(args.controller == "lqr"),
        "lqr_target_x_m": float(x_goal_m) if x_goal_m is not None else None,
        "lqr_target_source": x_goal_source,
        "lqr_q_weights": [float(args.lqr_q_x), float(args.lqr_q_xdot)],
        "lqr_r_weight": float(args.lqr_r_weight),
        "lqr_command_change_per_cycle": float(args.lqr_command_change_per_cycle),
        "lqr_gain_matrix": lqr_gain_matrix,
        "lqr_riccati_converged": lqr_converged,
        "lqr_riccati_iters": lqr_iters,
        "lqr_clip_count": lqr_clip_count if args.controller == "lqr" else None,
        "lqr_reject_count": lqr_reject_count if args.controller == "lqr" else None,
        "terminal_brake_enabled": terminal_brake_enabled,
        "terminal_brake_start_s": terminal_brake_start_s,
        "terminal_target_x_m": terminal_target_x_m if terminal_brake_enabled else None,
        "terminal_target_vx_mps": terminal_target_vx_mps if terminal_brake_enabled else None,
        "terminal_kx": terminal_kx if terminal_brake_enabled else None,
        "terminal_kv": terminal_kv if terminal_brake_enabled else None,
        "terminal_max_accel_mps2": terminal_max_accel_mps2 if terminal_brake_enabled else None,
        "terminal_max_jerk_mps3": terminal_max_jerk_mps3 if terminal_brake_enabled else None,
        "terminal_guardrail_margin_m": terminal_guardrail_margin_m if terminal_brake_enabled else None,
        "terminal_final_x_error_m": terminal_final_x_error,
        "terminal_final_vx_error_mps": terminal_final_vx_error,
        "terminal_brake_clip_count": terminal_brake_clip_count if terminal_brake_enabled else None,
        "terminal_guardrail_clip_count": terminal_guardrail_clip_count if terminal_brake_enabled else None,
        "terminal_guardrail_reject_count": terminal_guardrail_reject_count if terminal_brake_enabled else None,
        "terminal_max_command_accel_mps2": terminal_max_command_accel if terminal_brake_enabled else None,
        "terminal_max_command_jerk_mps3": terminal_max_command_jerk if terminal_brake_enabled else None,
        "q_start": q_start.tolist(),
        "tool_start_world": tool_start_pos.tolist(),
        "z_hold_m": z_hold,
        "final_tool_world": final_tool_pos.tolist(),
        "final_z_drift_m": final_z_drift,
        "max_z_transient_abs_m": max_z_transient_m,
        "final_orientation_error_deg": final_orientation_error_deg,
        "final_v_x_state_mps": final_v_x,
        "peak_tau_per_joint_nm": peak_tau_per_joint,
        "peak_tau_fraction_of_limit": peak_tau_fraction,
        "servo_force_limit_nm": SERVO_FORCE_LIMIT.tolist(),
        "time_s_trace": time_trace,
        "tool_x_trace": tool_x_trace,
        "tool_y_trace": tool_y_trace,
        "tool_z_trace": tool_z_trace,
        "orientation_error_deg_trace": orientation_error_trace_deg,
        "a_x_effective_trace": a_x_effective_trace,
        "v_x_state_trace": v_x_state_trace,
        "v_x_realized_cmd_trace": v_x_realized_trace,
        "tau_estimate_nm_trace": tau_estimate_trace,
        "speed_scale_trace": speed_scale_trace,
        "torque_scale_trace": torque_scale_trace,
        "phase_trace": phase_trace,
        "lqr_raw_cmd_trace": lqr_raw_cmd_trace if args.controller == "lqr" else None,
        "lqr_safe_cmd_trace": lqr_safe_cmd_trace if args.controller == "lqr" else None,
        "lqr_clipped_trace": lqr_clipped_trace if args.controller == "lqr" else None,
        "lqr_rejected_trace": lqr_rejected_trace if args.controller == "lqr" else None,
        "lqr_x_error_trace": lqr_x_error_trace if args.controller == "lqr" else None,
        "lqr_vx_error_trace": lqr_vx_error_trace if args.controller == "lqr" else None,
        "terminal_brake_raw_cmd_trace": terminal_brake_raw_trace if terminal_brake_enabled else None,
        "terminal_brake_safe_cmd_trace": terminal_brake_safe_trace if terminal_brake_enabled else None,
        "terminal_brake_clipped_trace": terminal_brake_clipped_trace if terminal_brake_enabled else None,
        "terminal_guardrail_clip_trace": terminal_guardrail_clip_trace if terminal_brake_enabled else None,
        "terminal_guardrail_reject_trace": terminal_guardrail_reject_trace if terminal_brake_enabled else None,
        "terminal_x_error_trace": terminal_x_error_trace if terminal_brake_enabled else None,
        "terminal_vx_error_trace": terminal_vx_error_trace if terminal_brake_enabled else None,
        "joint_impedance_enabled": bool(args.actuation_mode == "joint_torque"),
        "joint_kp_scale": float(args.joint_kp_scale) if args.actuation_mode == "joint_torque" else None,
        "joint_kd_scale": float(args.joint_kd_scale) if args.actuation_mode == "joint_torque" else None,
        "joint_tau_max_scale": float(args.joint_tau_max_scale) if args.actuation_mode == "joint_torque" else None,
        "joint_gravity_comp": bool(args.joint_gravity_comp) if args.actuation_mode == "joint_torque" else None,
        "joint_tau_max_nm": joint_tau_max_nm_vec.tolist() if joint_tau_max_nm_vec is not None else None,
        "joint_impedance_saturated_count": joint_impedance_saturated_count if args.actuation_mode == "joint_torque" else None,
        "hardware_shadow_enabled": bool(hardware_shadow_model is not None),
        "hardware_shadow_command_delay_steps": int(args.shadow_command_delay_steps)
        if hardware_shadow_model is not None
        else None,
        "hardware_shadow_torque_scale": float(args.shadow_torque_scale) if hardware_shadow_model is not None else None,
        "hardware_shadow_torque_rate_limit_nm_per_s": float(args.shadow_torque_rate_limit_nm_per_s)
        if hardware_shadow_model is not None
        else None,
        "hardware_shadow_viscous_damping_nm_per_rad_s": float(args.shadow_viscous_damping_nm_per_rads)
        if hardware_shadow_model is not None
        else None,
        "hardware_shadow_coulomb_friction_nm": float(args.shadow_coulomb_friction_nm)
        if hardware_shadow_model is not None
        else None,
        "hardware_shadow_deadzone_nm": float(args.shadow_deadzone_nm) if hardware_shadow_model is not None else None,
        "hardware_shadow_friction_velocity_eps_rad_s": float(args.shadow_friction_velocity_eps_rads)
        if hardware_shadow_model is not None
        else None,
        "hardware_shadow_clip_count": hardware_shadow_clip_count if hardware_shadow_model is not None else None,
        "hardware_shadow_delay_count": hardware_shadow_delay_count if hardware_shadow_model is not None else None,
        "hardware_shadow_rate_limit_count": hardware_shadow_rate_limit_count
        if hardware_shadow_model is not None
        else None,
        "hardware_shadow_deadzone_count": hardware_shadow_deadzone_count if hardware_shadow_model is not None else None,
        "hardware_shadow_friction_count": hardware_shadow_friction_count if hardware_shadow_model is not None else None,
        "tau_command_nm_trace": tau_command_trace if args.actuation_mode == "joint_torque" else None,
        "tau_applied_nm_trace": tau_applied_trace if args.actuation_mode == "joint_torque" else None,
        "tau_shadow_delay_nm_trace": tau_shadow_delay_trace if hardware_shadow_model is not None else None,
        "tau_shadow_rate_nm_trace": tau_shadow_rate_trace if hardware_shadow_model is not None else None,
        "tau_shadow_deadzone_nm_trace": tau_shadow_deadzone_trace if hardware_shadow_model is not None else None,
        "tau_shadow_friction_nm_trace": tau_shadow_friction_trace if hardware_shadow_model is not None else None,
        "q_ref_trace": q_ref_trace if args.actuation_mode == "joint_torque" else None,
        "qd_ref_trace": qd_ref_trace if args.actuation_mode == "joint_torque" else None,
        "success": success,
        "failure_reasons": failure_reasons,
        "guardrail_config": None if guardrail_config is None else boundary_summary(guardrail_config),
        "guardrail_margin_m": float(args.guardrail_margin_m) if guardrail_config is not None else None,
        "video_path": str(video_path) if not args.no_video and frames else None,
        "workspace_report_keys": list(workspace_report.keys()) if workspace_report else None,
    }
    json_path.write_text(json.dumps(summary, indent=2))

    print(f"Final tool xyz (m): {np.array2string(final_tool_pos, precision=6)}")
    print(f"Final z drift (m): {final_z_drift:+.6f}")
    print(f"Max |z - z_hold| (m): {max_z_transient_m:.6f}")
    print(f"Final orientation error (deg): {final_orientation_error_deg:.4f}")
    print(f"Final v_x state (m/s): {final_v_x:+.6f}")
    tau_limit_label = "joint torque limit" if args.actuation_mode == "joint_torque" else "forcerange"
    print(f"Peak |tau| fraction of {tau_limit_label}: {peak_tau_fraction:.3f}")
    print(f"Success: {success}")
    if args.actuation_mode == "joint_torque":
        print(f"Joint torque saturations: {joint_impedance_saturated_count}")
    if terminal_brake_enabled:
        print(f"Terminal final x error (m): {terminal_final_x_error:+.6f}")
        print(f"Terminal final v_x error (m/s): {terminal_final_vx_error:+.6f}")
        print(f"Terminal brake clips: {terminal_brake_clip_count}")
        print(
            "Terminal guardrail clips/rejects: "
            f"{terminal_guardrail_clip_count}/{terminal_guardrail_reject_count}"
        )
        print(f"Terminal max command accel (m/s^2): {terminal_max_command_accel:.6f}")
        print(f"Terminal max command jerk (m/s^3): {terminal_max_command_jerk:.6f}")
    if failure_reasons:
        print("Failure reasons:")
        for reason in failure_reasons:
            print(f"  - {reason}")
    if not args.no_video and frames:
        print(f"Saved video:   {video_path}")
    else:
        print("Video skipped (--no-video or no frames).")
    print(f"Saved summary: {json_path}")


if __name__ == "__main__":
    main()
