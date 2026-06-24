#!/usr/bin/env python3
"""
Single-owner UR5 Cartesian-impedance controller for headless CoppeliaSim.

This script is the sole orchestrator.  It:
  1. Connects to a running CoppeliaSim ZMQ server (with retries).
  2. Loads the default scene and UR5 model.
  3. Resolves joints / EE, configures torque mode.
  4. Starts stepped simulation.
  5. Runs the torque-control loop, or (with --torque-pulse) a small open-loop
     pulse on one joint, then encodes a video (unless --no-video).

Launch CoppeliaSim separately (see launch_coppeliasim_x_axis_headless.sh).

When recording video, Coppelia’s API requires ``handleVisionSensor`` before
``getVisionSensorImg``; the ZMQ client may return the image as ``bytes`` or a
``list`` of uint8 — both are decoded. Default ``--video-camera smoke`` matches
``run_coppeliasim_video_smoke.py`` framing. See ``docs/coppeliasim/``.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
import socket
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import mujoco

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "simulation"))
sys.path.insert(0, str(REPO_ROOT / "ros2_ws" / "src" / "ur5_x_axis_controller_ros"))

DEFAULT_COPPELIA_ROOT = (
    REPO_ROOT
    / "third_party"
    / "coppelia_runtime"
    / "CoppeliaSim_Edu_V4_10_0_rev0_Ubuntu24_04"
)
DEFAULT_COPPELIA_PYDEPS = REPO_ROOT / "third_party" / "coppelia_pydeps"
MUJOCO_MENAGERIE = REPO_ROOT / "mujoco_menagerie"
MUJOCO_GRAVITY_SCENE_CANDIDATES = (
    MUJOCO_MENAGERIE / "universal_robots_ur5e" / "scene_ur5e_cartpole.xml",
    MUJOCO_MENAGERIE / "universal_robots_ur5e" / "scene.xml",
    MUJOCO_MENAGERIE / "universal_robots_ur5e" / "ur5e.xml",
)

_bootstrap_root = Path(os.environ.get("COPPELIA_ROOT", str(DEFAULT_COPPELIA_ROOT)))
_bootstrap_pydeps = Path(os.environ.get("COPPELIA_PYDEPS", str(DEFAULT_COPPELIA_PYDEPS)))
for candidate in (
    _bootstrap_root / "programming" / "zmqRemoteApi" / "clients" / "python" / "src",
    _bootstrap_pydeps,
):
    if candidate.exists():
        sys.path.insert(0, str(candidate))

import zmq
from coppeliasim_zmqremoteapi_client import RemoteAPIClient

from controller_core.mujoco_cartpole_state import (
    build_mujoco_cartpole_observer,
    default_cartpole_scene_candidates,
    read_mujoco_cartpole_state,
)
from controller_core.filters import TorqueCommandFilter
from controller_core.logging_utils import JsonlTraceWriter
from controller_core.safety import ImpedanceSafetyConfig, ImpedanceSafetyMonitor
from controller_core.kinematics_utils import orientation_error_vec_wxyz
from controller_core import CommandGovernorSafetyFilter, FixedXTransportLQRController
from controller_core.x_axis_cartesian_impedance import (
    JOINT_NAME_ORDER,
    CartesianImpedanceConfig,
    XAxisCartesianImpedanceController,
)
from ur5_x_axis_controller_ros.config_loader import load_yaml_config
from ur5_x_axis_controller_ros.coppeliasim_adapter import (
    CoppeliaSimConfig,
    CoppeliaSimURAdapter,
)
from controller import (
    TARGET_SITE_ROTATION_WORLD,
    acceleration_transport_controller,
    axis_index_to_name,
    axis_name_to_index,
    velocity_x_transport_controller,
    orthogonal_axis_indices,
)
from coppelia_mpc_transport import (
    build_coppelia_mpc_transport,
    compute_coppelia_mpc_outer_command,
)
from coppelia_lqr_transport import (
    build_coppelia_lqr_transport,
    compute_coppelia_lqr_outer_command,
)
from external_zmq_controller_common import (
    controller_ownership_metadata,
    startup_banner_lines,
)
from coppelia_fast_x_transport import (
    minimum_fast_run_duration_s,
    recommend_fast_point_to_point_limits,
)
from coppelia_pendulum import (
    ensure_coppelia_pendulum,
    read_coppelia_pendulum_state,
)
from coppelia_reciprocating_transport import (
    build_reciprocating_plan,
    minimum_run_duration_s,
    point_to_point_accel_reference,
    reciprocating_axis_reference,
    reciprocating_ik_task_weights,
    slew_axis_reference,
)
from coppelia_torque_diagnostics import (
    CoppeliaTorqueDiagnostics,
    CoppeliaTorqueDiagnosticsConfig,
)


DEFAULT_CONFIG = REPO_ROOT / "ros2_ws" / "src" / "ur5_x_axis_controller_ros" / "config" / "controller.yaml"
DEFAULT_VIDEO_DIR = REPO_ROOT / "demonstration_videos" / "ur5e_coppeliasim"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "control_runs"
CONTROLLER_READY_SIGNAL = "real_cartpole_controller_ready"
CONTROLLER_SHUTDOWN_SIGNAL = "real_cartpole_controller_shutdown"
# Default live transport start pose for the direct impedance branch. This is
# the better-conditioned fixed-Z seed that keeps the Cartesian impedance task
# on the side of the Jacobian where the x axis is still well behaved.
DEFAULT_TRANSPORT_START_Q_FIXED_Z = np.array(
    [
        0.0,
        -1.133064268431449e-01,
        -6.646216458013020e-01,
        4.921777393344012e00,
        -6.283185307179586e00,
        5.280928640069786e00,
    ],
    dtype=np.float64,
)

# Coppelia-derived start pose from the successful origin-acquisition run.
# This is useful for the joint-PD transport surrogate, which prefers the
# transport-plane pose that already proved stable in Coppelia.
DEFAULT_TRANSPORT_START_Q_COPPELIA = np.array(
    [
        -2.11069988e-06,
        1.53821459e-01,
        -1.87279755e00,
        5.86278649e00,
        -6.28331410e00,
        5.28092864e00,
    ],
    dtype=np.float64,
)


def parse_joint_vector_env(name: str, fallback: np.ndarray) -> np.ndarray:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return np.asarray(fallback, dtype=np.float64).reshape(6)
    parts = [part.strip() for part in raw.split(",")]
    if len(parts) != 6:
        raise ValueError(f"{name} must contain 6 comma-separated floats")
    try:
        values = np.array([float(part) for part in parts], dtype=np.float64)
    except ValueError as exc:
        raise ValueError(f"{name} must contain 6 comma-separated floats") from exc
    return values.reshape(6)


def mujoco_accel_window_command(t_move: float, a_abs: float) -> float:
    """Legacy MuJoCo smoke-test signed transport-axis acceleration program."""
    a = max(float(a_abs), 0.0)
    if t_move < 1.5:
        return +a
    if t_move < 2.5:
        return 0.0
    if t_move < 5.5:
        return -a
    if t_move < 7.0:
        return +a
    return 0.0


def build_coppelia_mpc_transport_stack(
    *,
    x_start: float,
    target_dx: float,
    dt_s: float,
    horizon: int,
    q_weights: np.ndarray,
    q_terminal_scale: float,
    r_weight: float,
    pole_length_m: float,
    accel_limit: float,
    velocity_limit: float,
    command_change_limit: float,
    guardrail_margin_m: float,
):
    return build_coppelia_mpc_transport(
        x_start=x_start,
        target_dx=target_dx,
        dt_s=dt_s,
        horizon=horizon,
        q_weights=q_weights,
        q_terminal_scale=q_terminal_scale,
        r_weight=r_weight,
        pole_length_m=pole_length_m,
        accel_limit=accel_limit,
        velocity_limit=velocity_limit,
        command_change_limit=command_change_limit,
        guardrail_margin_m=guardrail_margin_m,
    )


def build_coppelia_lqr_transport_stack(
    *,
    x_start: float,
    target_dx: float,
    dt_s: float,
    q_x: float,
    q_xdot: float,
    r_weight: float,
    accel_limit: float,
    velocity_limit: float,
    command_change_limit: float,
    guardrail_margin_m: float,
) -> tuple[FixedXTransportLQRController, CommandGovernorSafetyFilter, float]:
    """Build the fixed-X LQR outer loop and its command-governing safety filter."""
    return build_coppelia_lqr_transport(
        x_start=x_start,
        target_dx=target_dx,
        dt_s=dt_s,
        q_x=q_x,
        q_xdot=q_xdot,
        r_weight=r_weight,
        accel_limit=accel_limit,
        velocity_limit=velocity_limit,
        command_change_limit=command_change_limit,
        guardrail_margin_m=guardrail_margin_m,
    )


def build_mujoco_gravity_estimator() -> tuple[mujoco.MjModel, mujoco.MjData] | None:
    """Load a compact MuJoCo UR5e model for qfrc_bias gravity compensation."""
    for scene in MUJOCO_GRAVITY_SCENE_CANDIDATES:
        if not scene.exists():
            continue
        try:
            model = mujoco.MjModel.from_xml_path(str(scene))
            if model.nu < 6 or model.nq < 6 or model.nv < 6:
                continue
            return model, mujoco.MjData(model)
        except Exception:
            continue
    return None


def compute_mujoco_gravity_bias(
    estimator: tuple[mujoco.MjModel, mujoco.MjData] | None,
    q: np.ndarray,
    qd: np.ndarray,
) -> np.ndarray | None:
    if estimator is None:
        return None
    model, data = estimator
    q = np.asarray(q, dtype=np.float64).reshape(-1)
    qd = np.asarray(qd, dtype=np.float64).reshape(-1)
    data.qpos[:] = 0.0
    data.qvel[:] = 0.0
    for idx, joint_name in enumerate(JOINT_NAME_ORDER):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if jid < 0:
            return None
        qadr = int(model.jnt_qposadr[jid])
        vadr = int(model.jnt_dofadr[jid])
        if qadr < model.nq:
            data.qpos[qadr] = float(q[idx])
        if vadr < model.nv:
            data.qvel[vadr] = float(qd[idx])
    mujoco.mj_forward(model, data)
    return np.asarray(data.qfrc_bias[:6], dtype=np.float64).copy()


def quat_wxyz_to_rotation_matrix(quat: np.ndarray) -> np.ndarray:
    q = np.asarray(quat, dtype=np.float64).reshape(4)
    n = float(np.linalg.norm(q))
    if n < 1e-12:
        return np.eye(3, dtype=np.float64)
    w, x, y, z = q / n
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def rotation_error_angle_rad(current: np.ndarray, target: np.ndarray) -> float:
    rel = np.asarray(target, dtype=np.float64).reshape(3, 3).T @ np.asarray(
        current, dtype=np.float64
    ).reshape(3, 3)
    cos_angle = float(np.clip((np.trace(rel) - 1.0) * 0.5, -1.0, 1.0))
    return float(math.acos(cos_angle))


def estimate_axis_transport_capability(
    jacobian: np.ndarray,
    transport_axis: str | int,
) -> dict[str, Any]:
    """Estimate local ability to move the task frame along one world axis."""
    J = np.asarray(jacobian, dtype=np.float64).reshape(6, 6)
    axis_idx = axis_name_to_index(transport_axis)
    orth_axes = orthogonal_axis_indices(axis_idx)
    b = np.zeros(6, dtype=np.float64)
    b[axis_idx] = 1.0
    qdot, *_ = np.linalg.lstsq(J, b, rcond=None)
    predicted = np.asarray(J @ qdot, dtype=np.float64).reshape(-1)
    return {
        "transport_axis": axis_index_to_name(axis_idx),
        "transport_axis_index": int(axis_idx),
        "task_matrix_shape": list(J.shape),
        "task_rank": int(np.linalg.matrix_rank(J)),
        "task_condition_number": float(np.linalg.cond(J)),
        "qdot_for_unit_transport_axis": qdot.tolist(),
        "predicted_axis_velocity_mps_for_unit_cmd": float(predicted[axis_idx]),
        "predicted_orthogonal_velocity_mps_for_unit_cmd": [
            float(predicted[orth_axes[0]]),
            float(predicted[orth_axes[1]]),
        ],
        "predicted_orientation_norm_radps_for_unit_cmd": float(np.linalg.norm(predicted[3:])),
        "max_abs_qdot_for_unit_cmd": float(np.max(np.abs(qdot))),
        "finite": bool(np.all(np.isfinite(qdot)) and np.all(np.isfinite(predicted))),
    }


def estimate_fixed_z_x_ik_capability(jacobian: np.ndarray) -> dict[str, Any]:
    """Legacy wrapper: estimate transport capability for world X."""
    return estimate_axis_transport_capability(jacobian, "x")


def ik_joint_pd_torque(
    q: np.ndarray,
    qd: np.ndarray,
    q_ref: np.ndarray,
    qdot_ref: np.ndarray,
    tau_limit: np.ndarray,
    kp: float,
    kd: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    q = np.asarray(q, dtype=np.float64).reshape(6)
    qd = np.asarray(qd, dtype=np.float64).reshape(6)
    q_ref = np.asarray(q_ref, dtype=np.float64).reshape(6)
    qdot_ref = np.asarray(qdot_ref, dtype=np.float64).reshape(6)
    tau_limit = np.asarray(tau_limit, dtype=np.float64).reshape(6)
    tau_p = float(kp) * (q_ref - q)
    tau_d = float(kd) * (qdot_ref - qd)
    raw = tau_p + tau_d
    clipped = np.clip(raw, -tau_limit, tau_limit)
    return clipped, {
        "tau_raw": raw.tolist(),
        "tau_impedance_P": tau_p.tolist(),
        "tau_impedance_D": tau_d.tolist(),
        "tau_clipped": clipped.tolist(),
        "tau_saturated": (np.abs(raw - clipped) > 1e-9).astype(np.float64).tolist(),
        "max_abs_tau_raw_nm": float(np.max(np.abs(raw))),
        "max_abs_tau_cmd_nm": float(np.max(np.abs(clipped))),
    }


def build_diagnostics_config(args: argparse.Namespace, raw_cfg: dict) -> CoppeliaTorqueDiagnosticsConfig:
    yaml_diag = raw_cfg.get("diagnostics", {}) or {}
    cli_overrides = {
        "enable_coppelia_torque_diagnostics": args.enable_coppelia_torque_diagnostics,
        "save_controller_logs": args.save_controller_logs,
        "save_controller_plots": args.save_controller_plots,
        "diagnostics_output_dir": args.diagnostics_output_dir,
        "impedance_gain_scale": args.impedance_gain_scale,
        "reference_smoothing_enabled": args.reference_smoothing_enabled,
        "max_reference_step": args.max_reference_step,
        "max_reference_velocity": args.max_reference_velocity,
        "diagnostics_mode": args.torque_diagnostics_mode,
        "sinusoid_joint_index": args.torque_diagnostics_joint_index,
    }
    cfg = CoppeliaTorqueDiagnosticsConfig.from_sources(yaml_diag, cli_overrides)
    if cfg.enable_coppelia_torque_diagnostics:
        if args.save_controller_logs is None:
            cfg.save_controller_logs = True
        if args.save_controller_plots is None:
            cfg.save_controller_plots = True
    return cfg


def apply_diagnostics_mode_overrides(
    args: argparse.Namespace,
    diag_cfg: CoppeliaTorqueDiagnosticsConfig,
) -> None:
    mode = str(diag_cfg.diagnostics_mode)
    if mode == "passive":
        args.zero_torque_test = False
        args.accel_x_transport = False
        args.duration = min(float(args.duration), 2.0)
    elif mode == "hold_soft":
        args.accel_x_transport = False
        args.settle_duration = 0.0
        if diag_cfg.impedance_gain_scale == 1.0:
            diag_cfg.impedance_gain_scale = 0.05
    elif mode == "sinusoid_joint":
        args.accel_x_transport = False
        args.settle_duration = 0.0
        if diag_cfg.impedance_gain_scale == 1.0:
            diag_cfg.impedance_gain_scale = 0.05
    elif mode == "tiny_x_motion":
        args.accel_x_transport = True
        args.accel_profile = "point_to_point"
        args.accel_torque_policy = "cartesian_impedance"
        args.target_dx = min(abs(float(args.target_dx)), 0.002)
        args.a_x_max = min(float(args.a_x_max), 0.05)
        args.v_x_max = min(float(args.v_x_max), 0.02)
        args.settle_duration = 0.5
        if diag_cfg.impedance_gain_scale == 1.0:
            diag_cfg.impedance_gain_scale = 0.1
    elif mode == "ref_step":
        args.accel_x_transport = False
        args.settle_duration = 0.0
        diag_cfg.reference_smoothing_enabled = False
    elif mode == "ref_smooth":
        args.accel_x_transport = False
        args.settle_duration = 0.0
        diag_cfg.reference_smoothing_enabled = True


def apply_torque_filter_step(
    torque_filter: TorqueCommandFilter,
    tau_pre_filter: np.ndarray,
    dt: float,
    *,
    collect_filter_diag: bool,
) -> tuple[np.ndarray, dict[str, Any] | None]:
    if collect_filter_diag:
        return torque_filter.apply_with_diagnostics(tau_pre_filter, dt)
    return torque_filter.apply(tau_pre_filter, dt), None


# ---------------------------------------------------------------------------
# Connection helper — works around ZMQ REQ socket linger deadlock
# ---------------------------------------------------------------------------

def _safe_destroy_client(client: RemoteAPIClient) -> None:
    """Close a RemoteAPIClient without blocking on unsent messages."""
    try:
        client.socket.setsockopt(zmq.LINGER, 0)
        client.socket.close()
        client.context.term()
    except Exception:
        pass
    # The RemoteAPIClient destructor can otherwise try to close/term again.
    class _Dead:
        def close(self) -> None: pass
        def term(self) -> None: pass
        def setsockopt(self, *a: object, **kw: object) -> None: pass
    try:
        client.socket = _Dead()  # type: ignore[assignment]
        client.context = _Dead()  # type: ignore[assignment]
    except Exception:
        pass


def request_controller_shutdown(sim: object, client: RemoteAPIClient | None = None) -> None:
    """Tell the optional Lua bootstrap add-on that Python is done."""
    try:
        sim.setStringSignal(CONTROLLER_SHUTDOWN_SIGNAL, "1")
    except Exception:
        pass
    try:
        sim.quitSimulator()
    except Exception:
        pass
    if client is not None:
        _safe_destroy_client(client)


def release_lua_bootstrap_for_rpc() -> None:
    """Let the Lua bootstrap return from sysCall_init before the first RPC call."""
    marker = os.environ.get("REAL_CARTPOLE_RPC_CONNECT_RELEASE_FILE", "")
    host = os.environ.get("REAL_CARTPOLE_RPC_HOST", "127.0.0.1")
    port = int(os.environ.get("REAL_CARTPOLE_RPC_PORT", "23000") or 23000)
    port_wait_s = float(os.environ.get("REAL_CARTPOLE_RPC_CONNECT_PORT_WAIT_S", "15") or 15.0)
    if port_wait_s > 0.0:
        deadline = time.monotonic() + port_wait_s
        while time.monotonic() < deadline:
            try:
                with socket.create_connection((host, port), timeout=0.2):
                    print(f"[startup] RPC port is listening: {host}:{port}", flush=True)
                    break
            except OSError:
                time.sleep(0.1)
    if marker:
        try:
            Path(marker).write_text("connect\n", encoding="utf-8")
            print(f"[startup] wrote RPC release marker: {marker}", flush=True)
        except Exception as exc:
            print(f"[startup] warning: could not write RPC release marker {marker}: {exc}", flush=True)
    delay_s = float(os.environ.get("REAL_CARTPOLE_RPC_CONNECT_DELAY_S", "0") or 0.0)
    if delay_s > 0.0:
        print(f"[startup] waiting {delay_s:.2f}s for Lua bootstrap to return", flush=True)
        time.sleep(delay_s)
    ready_marker = os.environ.get("REAL_CARTPOLE_RPC_CONNECT_READY_FILE", "")
    if ready_marker:
        ready_path = Path(ready_marker)
        ready_wait_s = float(os.environ.get("REAL_CARTPOLE_RPC_CONNECT_READY_WAIT_S", "5") or 5.0)
        deadline = time.monotonic() + max(0.0, ready_wait_s)
        while not ready_path.exists() and time.monotonic() < deadline:
            time.sleep(0.05)
        if ready_path.exists():
            print(f"[startup] observed RPC ready marker: {ready_marker}", flush=True)
        else:
            print(
                f"[startup] ready marker not present after {ready_wait_s:.2f}s: {ready_marker}",
                flush=True,
            )


def connect(host: str, port: int, retries: int = 120, delay_s: float = 0.25):
    """Connect to CoppeliaSim ZMQ Remote API and validate with a round-trip call.

    Returns (client, sim).  Each failed attempt explicitly closes the socket
    with LINGER=0 so the next attempt doesn't deadlock in __del__.
    """
    last_exc: Exception | None = None
    for attempt in range(1, retries + 1):
        client: RemoteAPIClient | None = None
        try:
            client = RemoteAPIClient(host=host, port=port)
            first_timeout_ms = int(
                os.environ.get("REAL_CARTPOLE_RPC_FIRST_RCVTIMEO_MS", "5000") or 5000
            )
            client.socket.setsockopt(zmq.RCVTIMEO, first_timeout_ms)
            sim = client.require("sim")
            sim.getSimulationState()
            # The library sets RCVTIMEO=5000 on first call and never resets it.
            # Long operations (loadModel, loadScene) need more time.
            client.socket.setsockopt(zmq.RCVTIMEO, 10 * 60 * 1000)
            return client, sim
        except Exception as exc:
            if client is not None:
                _safe_destroy_client(client)
            last_exc = exc
            print(
                f"[connect] attempt {attempt}/{retries}: {exc}",
                file=sys.stderr, flush=True,
            )
            time.sleep(delay_s)
    raise RuntimeError(
        f"Could not connect to CoppeliaSim ZMQ RPC at {host}:{port}"
    ) from last_exc


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--coppelia-root", type=Path, default=DEFAULT_COPPELIA_ROOT)
    p.add_argument("--host", type=str, default="127.0.0.1")
    p.add_argument("--port", type=int, default=23000)
    p.add_argument("--duration", type=float, default=8.0)
    p.add_argument("--settle-duration", type=float, default=2.0)
    p.add_argument("--target-dx", type=float, default=0.02)
    p.add_argument(
        "--accel-x-transport",
        action="store_true",
        help=(
            "Drive the selected transport axis with an acceleration-limited "
            "profile while the torque controller holds the other world axes "
            "and the tool orientation."
        ),
    )
    p.add_argument(
        "--transport-axis",
        choices=("x", "y", "z"),
        default="x",
        help=(
            "World axis for the transport motion. The same direct-torque "
            "controller can run along X, Y, or Z without a separate codepath."
        ),
    )
    p.add_argument(
        "--accel-profile",
        choices=(
            "point_to_point",
            "mujoco_windows",
            "back_and_forth_15s",
            "reciprocating",
            "fast_x",
            "lqr",
            "mpc",
        ),
        default="point_to_point",
        help=(
            "Acceleration schedule for --accel-x-transport. point_to_point "
            "moves target_x by --target-dx and brakes to rest; mujoco_windows "
            "uses the legacy MuJoCo smoke-test acceleration windows; "
            "back_and_forth_15s uses +a for 15s and -a for 15s, then stops; "
            "reciprocating moves origin -> +stroke -> -stroke -> origin with "
            "acceleration-limited segments; fast_x auto-selects the fastest "
            "feasible point-to-point move under joint and safety limits; lqr "
            "closes the outer X loop on measured x/x_dot and still uses the "
            "constrained inner torque allocator."
        ),
    )
    p.add_argument(
        "--a-x-max",
        type=float,
        default=0.20,
        help="Transport-axis acceleration cap in m/s^2.",
    )
    p.add_argument(
        "--v-x-max",
        type=float,
        default=0.12,
        help="Transport-axis speed cap in m/s.",
    )
    p.add_argument(
        "--accel-square-half-period-s",
        type=float,
        default=15.0,
        help="Half-period for --accel-profile back_and_forth_15s.",
    )
    p.add_argument(
        "--reciprocating-stroke-m",
        type=float,
        default=0.03,
        help=(
            "Half-stroke distance for --accel-profile reciprocating: the EE "
            "moves to +stroke, then -stroke, then back to the start position."
        ),
    )
    p.add_argument(
        "--reciprocating-hold-s",
        type=float,
        default=0.25,
        help="Pause duration at each reciprocating endpoint before reversing.",
    )
    p.add_argument(
        "--fast-x-joint-speed-fraction",
        type=float,
        default=None,
        help=(
            "Fraction of safety.max_joint_velocity_radps used when "
            "--accel-profile fast_x auto-computes v_x_max."
        ),
    )
    p.add_argument(
        "--fast-x-accel-fraction",
        type=float,
        default=None,
        help="Acceleration margin for --accel-profile fast_x limit computation.",
    )
    p.add_argument(
        "--fast-x-max-acceleration-mps2",
        type=float,
        default=None,
        help="Hard ceiling on auto-computed transport acceleration for fast_x.",
    )
    p.add_argument(
        "--fast-x-max-velocity-mps",
        type=float,
        default=None,
        help="Hard ceiling on auto-computed transport velocity for fast_x.",
    )
    p.add_argument(
        "--spawn-coppelia-pendulum",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Spawn a passive CoppeliaSim pendulum on the task frame for real "
            "theta/theta_dot (config coppeliasim.spawn_pendulum when unset)."
        ),
    )
    p.add_argument(
        "--pendulum-pole-length-m",
        type=float,
        default=None,
        help="Coppelia pendulum pole length in meters (config default when unset).",
    )
    p.add_argument(
        "--hold-transport-start-pose",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="For --accel-x-transport, start from the MuJoCo fixed-Z transport q pose.",
    )
    p.add_argument(
        "--accel-torque-policy",
        choices=("ik_joint_pd", "cartesian_impedance"),
        default="ik_joint_pd",
        help=(
            "Direct-torque policy used by --accel-x-transport. ik_joint_pd "
            "uses the MuJoCo differential-IK acceleration policy to produce a "
            "joint target, then applies joint PD torques. cartesian_impedance "
            "keeps the older J.T wrench controller."
        ),
    )
    p.add_argument(
        "--lqr-q-x",
        type=float,
        default=60.0,
        help="LQR weight on X position error when --accel-profile lqr is used.",
    )
    p.add_argument(
        "--lqr-q-xdot",
        type=float,
        default=8.0,
        help="LQR weight on X velocity error when --accel-profile lqr is used.",
    )
    p.add_argument(
        "--lqr-r-weight",
        type=float,
        default=1.0,
        help="LQR control-effort weight when --accel-profile lqr is used.",
    )
    p.add_argument(
        "--lqr-command-change-per-cycle",
        type=float,
        default=0.20,
        help="Max change in the outer LQR acceleration command per simulation step.",
    )
    p.add_argument(
        "--lqr-guardrail-margin-m",
        type=float,
        default=0.05,
        help="Safety margin around the LQR start/goal bounds for command-governor clipping.",
    )
    p.add_argument(
        "--lqr-sim-time-step",
        type=float,
        default=0.01,
        help=(
            "Requested simulation time step in seconds for --accel-profile lqr. "
            "This only applies to the LQR outer-loop mode."
        ),
    )
    p.add_argument(
        "--mpc-horizon",
        type=int,
        default=20,
        help="Receding horizon length for --accel-profile mpc.",
    )
    p.add_argument(
        "--mpc-q-x",
        type=float,
        default=40.0,
        help="MPC stage weight on cart/task X position error.",
    )
    p.add_argument(
        "--mpc-q-xdot",
        type=float,
        default=10.0,
        help="MPC stage weight on cart/task X velocity.",
    )
    p.add_argument(
        "--mpc-q-theta",
        type=float,
        default=180.0,
        help="MPC stage weight on pole angle error.",
    )
    p.add_argument(
        "--mpc-q-theta-dot",
        type=float,
        default=20.0,
        help="MPC stage weight on pole angular velocity.",
    )
    p.add_argument(
        "--mpc-q-terminal-scale",
        type=float,
        default=3.0,
        help="Terminal cost scale for the final MPC stage.",
    )
    p.add_argument(
        "--mpc-r-weight",
        type=float,
        default=0.35,
        help="MPC control-effort weight on cart acceleration.",
    )
    p.add_argument(
        "--mpc-pole-length-m",
        type=float,
        default=0.4,
        help="Effective pendulum length used by the MPC prediction model.",
    )
    p.add_argument(
        "--mpc-command-change-per-cycle",
        type=float,
        default=0.20,
        help="Max change in the outer MPC acceleration command per simulation step.",
    )
    p.add_argument(
        "--mpc-guardrail-margin-m",
        type=float,
        default=0.05,
        help="Safety margin around the MPC start/goal bounds for command-governor clipping.",
    )
    p.add_argument(
        "--task-frame-mode",
        choices=("config", "ee_object", "mujoco_attachment_dummy"),
        default="config",
        help="Override coppeliasim.task_frame.mode for the controlled EE/task frame.",
    )
    p.add_argument(
        "--task-orientation-target",
        choices=("initial", "mujoco"),
        default="initial",
        help=(
            "Orientation held by the IK torque policy. initial is safest for Coppelia bring-up; "
            "mujoco uses TARGET_SITE_ROTATION_WORLD and requires a calibrated task frame."
        ),
    )
    p.add_argument("--ik-joint-kp", type=float, default=45.0)
    p.add_argument("--ik-joint-kd", type=float, default=9.0)
    p.add_argument("--ik-torque-headroom", type=float, default=0.65)
    p.add_argument(
        "--max-joint-excursion-rad",
        type=float,
        default=1.2,
        help="Safety/summary threshold for wild configuration drift from the starting q.",
    )
    p.add_argument("--fps", type=int, default=25)
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=360)
    p.add_argument("--rpc-connect-retries", type=int, default=120)
    p.add_argument("--rpc-connect-retry-delay-s", type=float, default=0.25)
    p.add_argument("--video-name", type=str, default="coppeliasim_ur5_x_impedance_headless.mp4")
    p.add_argument(
        "--preloaded-scene",
        action="store_true",
        help="Compatibility flag for the Lua bootstrap add-on; UR5 may already be present.",
    )
    p.add_argument(
        "--legacy-marker-handoff",
        action="store_true",
        help="Use the older Lua release-marker handshake for compatibility only.",
    )
    p.add_argument(
        "--warmup-steps",
        type=int,
        default=8,
        help="Number of zero-torque warmup steps before entering the live controller loop.",
    )
    p.add_argument(
        "--probe-only",
        action="store_true",
        help="Connect, load scene/model, sample state/Jacobian, exit before torque.",
    )
    p.add_argument(
        "--compare-jacobian",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Compare API and numerical Jacobians at startup and include the delta in the summary.",
    )
    p.add_argument(
        "--zero-torque-test",
        action="store_true",
        help="Hold zero torque for the whole run and report drift/velocity response.",
    )
    p.add_argument("--trace-name", type=str, default="coppeliasim_ur5_x_impedance_headless.jsonl")
    p.add_argument("--summary-name", type=str, default="coppeliasim_ur5_x_impedance_headless_summary.json")
    p.add_argument("--no-video", action="store_true")
    p.add_argument(
        "--torque-pulse",
        action="store_true",
        help="Open-loop: apply a small constant torque on one joint for a few steps, then zeros (for video sanity check).",
    )
    p.add_argument(
        "--torque-pulse-joint", type=int, default=0,
        help="Joint index 0..5 in JOINT_NAME_ORDER (default 0 = shoulder pan).",
    )
    p.add_argument(
        "--torque-pulse-nm", type=float, default=0.25,
        help="Torque magnitude in N·m (default 0.25; increase for a more visible nudge).",
    )
    p.add_argument(
        "--torque-pulse-bidirectional",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="When set, run a positive pulse followed by a matching negative pulse on the same joint.",
    )
    p.add_argument(
        "--torque-pulse-steps", type=int, default=30,
        help="Simulation steps to hold the pulse (default 30).",
    )
    p.add_argument(
        "--video-camera",
        choices=("smoke", "ee"),
        default="smoke",
        help=(
            "How to place the ZMQ-captured vision sensor. "
            "'smoke' = same world pose formula as run_coppeliasim_video_smoke.py (default, arm visible). "
            "'ee' = offset from end-effector, same z, look-at EE (tracks motion)."
        ),
    )
    p.add_argument(
        "--enable-coppelia-torque-diagnostics",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Record per-step CoppeliaSim torque/safety diagnostics (default: off).",
    )
    p.add_argument(
        "--save-controller-logs",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Write diagnostic JSONL + summary when torque diagnostics are enabled.",
    )
    p.add_argument(
        "--save-controller-plots",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Generate diagnostic PNG plots when torque diagnostics are enabled.",
    )
    p.add_argument(
        "--diagnostics-output-dir",
        type=Path,
        default=None,
        help="Directory for diagnostic logs/plots (default: outputs/control_runs/coppelia_torque_diagnostics).",
    )
    p.add_argument(
        "--impedance-gain-scale",
        type=float,
        default=None,
        help="Scale Cartesian/IK impedance gains for diagnostic hold/sweep tests.",
    )
    p.add_argument(
        "--reference-smoothing-enabled",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Smooth q_des / x_des references to limit step discontinuities.",
    )
    p.add_argument(
        "--max-reference-step",
        type=float,
        default=None,
        help="Max per-step reference change when reference smoothing is enabled.",
    )
    p.add_argument(
        "--max-reference-velocity",
        type=float,
        default=None,
        help="Max reference velocity when reference smoothing is enabled.",
    )
    p.add_argument(
        "--torque-diagnostics-mode",
        choices=(
            "live",
            "passive",
            "hold_soft",
            "sinusoid_joint",
            "tiny_x_motion",
            "ref_step",
            "ref_smooth",
        ),
        default="live",
        help="Diagnostic test mode for CoppeliaSim torque bring-up.",
    )
    p.add_argument(
        "--torque-diagnostics-joint-index",
        type=int,
        default=5,
        help="Joint index (0..5) for sinusoid_joint diagnostic mode (default wrist_3).",
    )
    p.add_argument(
        "--torque-diagnostics-run-label",
        type=str,
        default="",
        help="Label prefix for diagnostic output files.",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
# Video camera: fixed offset in x–y, same z as EE, looking at the end effector.


def _rotation_matrix_to_quat_xyzw(rot: np.ndarray) -> np.ndarray:
    trace = float(np.trace(rot))
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * s
        qx = (rot[2, 1] - rot[1, 2]) / s
        qy = (rot[0, 2] - rot[2, 0]) / s
        qz = (rot[1, 0] - rot[0, 1]) / s
    elif rot[0, 0] > rot[1, 1] and rot[0, 0] > rot[2, 2]:
        s = math.sqrt(1.0 + rot[0, 0] - rot[1, 1] - rot[2, 2]) * 2.0
        qw = (rot[2, 1] - rot[1, 2]) / s
        qx = 0.25 * s
        qy = (rot[0, 1] + rot[1, 0]) / s
        qz = (rot[0, 2] + rot[2, 0]) / s
    elif rot[1, 1] > rot[2, 2]:
        s = math.sqrt(1.0 + rot[1, 1] - rot[0, 0] - rot[2, 2]) * 2.0
        qw = (rot[0, 2] - rot[2, 0]) / s
        qx = (rot[0, 1] + rot[1, 0]) / s
        qy = 0.25 * s
        qz = (rot[1, 2] + rot[2, 1]) / s
    else:
        s = math.sqrt(1.0 + rot[2, 2] - rot[0, 0] - rot[1, 1]) * 2.0
        qw = (rot[1, 0] - rot[0, 1]) / s
        qx = (rot[0, 2] + rot[2, 0]) / s
        qy = (rot[1, 2] + rot[2, 1]) / s
        qz = 0.25 * s
    quat = np.array([qx, qy, qz, qw], dtype=np.float64)
    quat /= np.linalg.norm(quat)
    return quat


# World-frame horizontal offset (m) from the end effector: camera is placed at EE + (dx, dy, 0)
# in x–y, with **the same z as the EE**, then oriented to look at the EE (arm stays centered).
EE_CAMERA_OFFSET_XY = np.array([-0.95, 0.65], dtype=np.float64)


def ur5_camera_pose_look_at_ee(look_at: np.ndarray) -> list[float]:
    """Camera at same height as the EE, offset only in x–y; view axis points at the EE."""
    look_at = np.asarray(look_at, dtype=np.float64).reshape(3).copy()
    cam_pos = np.array(
        [
            look_at[0] + EE_CAMERA_OFFSET_XY[0],
            look_at[1] + EE_CAMERA_OFFSET_XY[1],
            look_at[2],
        ],
        dtype=np.float64,
    )
    forward = look_at - cam_pos
    n = float(np.linalg.norm(forward))
    if n < 1e-8:
        forward = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    else:
        forward /= n
    world_up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    right = np.cross(forward, world_up)
    nr = float(np.linalg.norm(right))
    if nr < 1e-8:
        right = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    else:
        right /= nr
    up = np.cross(right, forward)
    rot = np.column_stack((right, up, forward))
    quat_xyzw = _rotation_matrix_to_quat_xyzw(rot)
    return [
        float(cam_pos[0]),
        float(cam_pos[1]),
        float(cam_pos[2]),
        float(quat_xyzw[3]),
        float(quat_xyzw[0]),
        float(quat_xyzw[1]),
        float(quat_xyzw[2]),
    ]


def set_vision_sensor_look_at_ee(sim: object, vision_sensor: int, look_at: np.ndarray) -> None:
    """Place vision sensor to face *look_at* (typically the EE world position)."""
    pose = ur5_camera_pose_look_at_ee(look_at)
    sim.setObjectPose(vision_sensor + sim.handleflag_wxyzquat, pose, sim.handle_world)


def _smoke_rotation_matrix_to_quat_xyzw(rot: np.ndarray) -> np.ndarray:
    """Match run_coppeliasim_video_smoke.rotation_matrix_to_quat_xyzw (proven camera)."""
    trace = float(np.trace(rot))
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * s
        qx = (rot[2, 1] - rot[1, 2]) / s
        qy = (rot[0, 2] - rot[2, 0]) / s
        qz = (rot[1, 0] - rot[0, 1]) / s
    elif rot[0, 0] > rot[1, 1] and rot[0, 0] > rot[2, 2]:
        s = math.sqrt(1.0 + rot[0, 0] - rot[1, 1] - rot[2, 2]) * 2.0
        qw = (rot[2, 1] - rot[1, 2]) / s
        qx = 0.25 * s
        qy = (rot[0, 1] + rot[1, 0]) / s
        qz = (rot[0, 2] + rot[2, 0]) / s
    elif rot[1, 1] > rot[2, 2]:
        s = math.sqrt(1.0 + rot[1, 1] - rot[0, 0] - rot[2, 2]) * 2.0
        qw = (rot[0, 2] - rot[2, 0]) / s
        qx = (rot[0, 1] + rot[1, 0]) / s
        qy = 0.25 * s
        qz = (rot[1, 2] + rot[2, 1]) / s
    else:
        s = math.sqrt(1.0 + rot[2, 2] - rot[0, 0] - rot[1, 1]) * 2.0
        qw = (rot[1, 0] - rot[0, 1]) / s
        qx = (rot[0, 2] + rot[2, 0]) / s
        qy = (rot[1, 2] + rot[2, 1]) / s
        qz = 0.25 * s
    quat = np.array([qx, qy, qz, qw], dtype=np.float64)
    quat /= np.linalg.norm(quat)
    return quat


def video_smoke_style_camera_wxyz(*, step_idx: int, total_steps: int) -> list[float]:
    """Identical to run_coppeliasim_video_smoke.camera_pose (keeps the arm in frame)."""
    progress = 0.0 if total_steps <= 1 else float(step_idx) / float(total_steps - 1)
    yaw = math.radians(-48.0 + 18.0 * progress)
    radius = 1.95
    target = np.array([0.0, 0.0, 0.62], dtype=np.float64)
    cam_pos = np.array(
        [
            radius * math.cos(yaw),
            radius * math.sin(yaw),
            target[2] + 0.34,
        ],
        dtype=np.float64,
    )
    forward = target - cam_pos
    forward /= float(np.linalg.norm(forward))
    world_up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    right = np.cross(forward, world_up)
    right /= float(np.linalg.norm(right))
    up = np.cross(right, forward)
    # Vision sensor local +Z is forward; columns = (right, up, forward)
    rot = np.column_stack((right, up, forward))
    quat_xyzw = _smoke_rotation_matrix_to_quat_xyzw(rot)
    return [
        float(cam_pos[0]),
        float(cam_pos[1]),
        float(cam_pos[2]),
        float(quat_xyzw[3]),
        float(quat_xyzw[0]),
        float(quat_xyzw[1]),
        float(quat_xyzw[2]),
    ]


def set_vision_sensor_smoke_traj(
    sim: object, vision_sensor: int, step_idx: int, total_steps: int,
) -> None:
    """Place vision sensor with the same trajectory as the PNG smoke test."""
    pose = video_smoke_style_camera_wxyz(step_idx=step_idx, total_steps=max(1, total_steps))
    sim.setObjectPose(vision_sensor + sim.handleflag_wxyzquat, pose, sim.handle_world)


def place_vision_sensor_for_frame(
    sim: object,
    vision_sensor: int,
    video_camera: str,
    step_idx: int,
    total_steps: int,
    adapter: CoppeliaSimURAdapter,
) -> None:
    """Set camera before physics step (see Coppelia: handleVisionSensor after step, then getImg)."""
    if video_camera == "smoke":
        set_vision_sensor_smoke_traj(sim, vision_sensor, step_idx, total_steps)
    else:
        ee_pos, _, _, _ = adapter.read_ee_pose_twist()
        set_vision_sensor_look_at_ee(
            sim, vision_sensor,
            np.asarray(ee_pos, dtype=np.float64)[:3],
        )


def make_vision_sensor(sim: object, width: int, height: int) -> int:
    # Bit 0 = explicit (sim.createVisionSensor) so ZMQ may call sim.handleVisionSensor;
    # 2|4 = perspective + no frustum mesh (match smoke).
    options = 1 | 2 | 4
    int_params = [int(width), int(height), 0, 0]
    float_params = [
        0.02,
        6.0,
        np.deg2rad(58.0),
        0.1,
        0.0,
        0.0,
        0.82,
        0.86,
        0.92,
        0.0,
        0.0,
    ]
    sensor = int(sim.createVisionSensor(options, int_params, float_params))
    sim.setObjectAlias(sensor, "HeadlessVideoCapture")
    sim.setExplicitHandling(sensor, 1)
    return sensor


# From legacy simConst: sim_visionintparam_entity_to_render.  Manual: -1 = render all objects.
_VISIONINTPARAM_ENTITY_TO_RENDER = 1008


def configure_vision_sensor_render_all(sim: object, vision_handle: int) -> None:
    """Ensure the vision sensor is not limited to a single-entity frustum (can yield black buffer)."""
    try:
        sim.setObjectInt32Param(vision_handle, _VISIONINTPARAM_ENTITY_TO_RENDER, -1)
    except Exception:
        # Older or stripped API: ignore; not all builds expose the same symbol names.
        pass


def _enable_global_vision_and_display(sim: object) -> None:
    """simConst legacy: bool 10 = vision_sensor_handling, 16 = display_enabled. If off, GL may not fill sensors."""
    for bid in (10, 16, 6):  # vision handling, display, dynamics (some builds link render to dynamics pass)
        try:
            sim.setBoolParam(bid, True)  # type: ignore[misc]
        except Exception:
            pass


def read_vision_rgb24(sim: object, vision_sensor: int) -> tuple[np.ndarray, list[int]]:
    """Default main script calls ``handleVisionSensor(handle_all_except_explicit)`` before
    explicit sensors; we mirror that so the render graph matches an interactive sim step.
    """
    hae = getattr(sim, "handle_all_except_explicit", -3)  # simConst / defaultMainScript.lua
    try:
        sim.handleVisionSensor(hae)
    except Exception:
        pass
    sim.handleVisionSensor(vision_sensor)
    img, res = sim.getVisionSensorImg(vision_sensor, 0, 0.0, [0, 0], [0, 0])
    if os.environ.get("COPPELIASIM_DEBUG_VISION") == "1" and not os.environ.get("_COPPELIASIM_VISION_LOGGED"):
        os.environ["_COPPELIASIM_VISION_LOGGED"] = "1"
        try:
            ln = len(img) if hasattr(img, "__len__") and not isinstance(img, (int, float)) else -1
        except TypeError:
            ln = -1
        bsum: int = -1
        if isinstance(img, (bytes, bytearray, memoryview)) and len(img) > 0:
            bsum = int(sum(bytes(img)[:2000]))
        print(f"[vision-debug] type={type(img).__name__} len={ln} res={res} byte_sum2000={bsum}", flush=True)
    return decode_vision_image_buffer(img, res, sim), res


def set_ur5_joints_for_video_framing(sim: object) -> None:
    """Known-good pose for the official UR5.ttt model (same as video smoke)."""
    joint_paths = (
        "/UR5/joint",
        "/UR5/link/joint",
        "/UR5/link/link/joint",
        "/UR5/link/link/link/joint",
        "/UR5/link/link/link/link/joint",
        "/UR5/link/link/link/link/link/joint",
    )
    joint_targets = (0.2, -1.15, 1.55, -1.8, -1.45, 0.35)
    for path, target in zip(joint_paths, joint_targets):
        h = int(sim.getObject(path))
        sim.setJointPosition(h, float(target))


def decode_vision_image_buffer(
    buffer: object,
    resolution: list[int],
    sim: object | None = None,
) -> np.ndarray:
    """Turn sim.getVisionSensorImg first return into HxWx3 uint8, origin bottom-left -> flipud.

    The ZMQ remote API may CBOR-decode the image as ``bytes`` *or* a ``list`` of uint8s
    (Coppelia's Lua API documents using sim.unpackUInt8Table on the buffer in some clients).
    ``np.frombuffer`` only works on bytes-like objects; a Python ``list`` would decode wrong
    (often yielding all zeros) if coerced incorrectly.
    """
    w, h = int(resolution[0]), int(resolution[1])
    need = w * h * 3
    if isinstance(buffer, (bytes, bytearray, memoryview)):
        b = bytes(buffer)
        if len(b) < need:
            raise ValueError(f"vision image bytes too short: {len(b)} < {need}")
        raw = np.frombuffer(b[:need], dtype=np.uint8)
    elif isinstance(buffer, np.ndarray):
        flat = buffer.astype(np.uint8, copy=False).ravel()
        if flat.size < need:
            raise ValueError(f"vision image ndarray too short: {flat.size} < {need}")
        raw = flat[:need]
    else:
        seq = list(buffer)[:need]  # type: ignore[arg-type]
        if len(seq) < need:
            raise ValueError(f"vision image buffer too short: {len(seq)} < {need}")
        raw = np.asarray(seq, dtype=np.uint8)
    img = raw.reshape(h, w, 3)
    return np.flipud(img)


def write_video_ffmpeg(path: Path, frames: list[np.ndarray], fps: int) -> None:
    if not frames:
        return
    h, w, _ = frames[0].shape
    ffmpeg_bin = os.environ.get("FFMPEG_BIN", "ffmpeg")
    cmd = [
        ffmpeg_bin, "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s:v", f"{w}x{h}", "-r", str(fps), "-i", "-",
        "-an", "-vcodec", "libx264", "-pix_fmt", "yuv420p", str(path),
    ]
    data = b"".join(np.asarray(f, dtype=np.uint8).tobytes() for f in frames)
    # Single communicate(input=...) — streaming write + communicate() can hit
    # "flush of closed file" on some Python 3.12 + subprocess paths.
    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    _out, err = proc.communicate(input=data)
    if proc.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed (code {proc.returncode}): "
            f"{err.decode('utf-8', errors='replace') if err else ''}"
        )


def load_runtime(args: argparse.Namespace) -> tuple[dict, CoppeliaSimConfig]:
    raw = load_yaml_config(args.config)
    ctrl_y = raw.get("controller", {}) or {}
    cop_y = raw.get("coppeliasim", {}) or {}
    ta = cop_y.get("torque_application", {}) or {}
    jac = cop_y.get("jacobian", {}) or {}
    tf = cop_y.get("task_frame", {}) or {}
    joint_map = {str(k): str(v) for k, v in (cop_y.get("joint_name_map", {}) or {}).items()}
    task_frame_mode = str(tf.get("mode", "ee_object"))
    if args.task_frame_mode != "config":
        task_frame_mode = str(args.task_frame_mode)
    cfg = CoppeliaSimConfig(
        zmq_host=args.host,
        zmq_port=args.port,
        joint_name_map=joint_map,
        ee_object_name=str(cop_y.get("ee_object_name", "/UR5/UR5_connection")),
        ee_object_name_alternates=tuple(
            str(v) for v in (cop_y.get("ee_object_name_alternates", []) or [])
        ),
        task_frame_mode=task_frame_mode,
        task_frame_parent_object_name=str(tf.get("parent_object_name", "")),
        task_frame_parent_object_alternates=tuple(
            str(v) for v in (tf.get("parent_object_name_alternates", []) or [])
        ),
        task_frame_attachment_offset_m=tuple(
            float(v) for v in (tf.get("attachment_offset_m", [0.0, 0.0, -0.2]) or [0.0, 0.0, -0.2])
        ),
        task_frame_attachment_quat_wxyz=tuple(
            float(v)
            for v in (
                tf.get(
                    "attachment_quat_wxyz",
                    [-0.7071067811865475, 0.7071067811865475, 0.0, 0.0],
                )
                or [-0.7071067811865475, 0.7071067811865475, 0.0, 0.0]
            )
        ),
        task_frame_dummy_size_m=float(tf.get("dummy_size_m", 0.025)),
        stepping=True,
        prefer_signed_target_force=bool(ta.get("prefer_signed_target_force", True)),
        fallback_large_velocity_rad_s=float(ta.get("fallback_large_velocity_rad_s", 10.0)),
        jacobian_source=str(jac.get("source", "auto")),
        numerical_epsilon=float(jac.get("numerical_epsilon", 1.0e-5)),
        connect_retries=int(args.rpc_connect_retries),
        connect_retry_delay_s=float(args.rpc_connect_retry_delay_s),
    )
    return raw, cfg


def sample_controller_state(
    adapter: CoppeliaSimURAdapter,
    target_axis: float,
) -> dict[str, Any]:
    q, qd = adapter.read_joint_state()
    ee_pos, ee_quat, ee_lin, ee_ang = adapter.read_ee_pose_twist()
    j_pos, j_rot = adapter.read_jacobian()
    return {
        "time": 0.0,
        "q": q, "qd": qd,
        "ee_pos": ee_pos, "ee_quat": ee_quat,
        "ee_lin_vel": ee_lin, "ee_ang_vel": ee_ang,
        "target_axis": float(target_axis),
        "target_axis_vel": 0.0,
        "jacobian": np.vstack([j_pos, j_rot]),
    }


def run_zero_torque_warmup(adapter: CoppeliaSimURAdapter, sim: object, warmup_steps: int) -> None:
    """Advance a live stepped simulation with zero torque before the controller starts."""
    steps = max(0, int(warmup_steps))
    if steps == 0:
        return
    sim_time_start = float(sim.getSimulationTime())
    for _ in range(steps):
        adapter.apply_torque(np.zeros(6, dtype=np.float64))
        sim.step()
    sim_time_end = float(sim.getSimulationTime())
    if sim_time_end <= sim_time_start:
        raise RuntimeError("simulation time did not advance during zero-torque warmup")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    video_path = DEFAULT_VIDEO_DIR / args.video_name
    trace_path = DEFAULT_OUTPUT_DIR / args.trace_name
    summary_path = DEFAULT_OUTPUT_DIR / args.summary_name
    video_path.parent.mkdir(parents=True, exist_ok=True)
    trace_path.parent.mkdir(parents=True, exist_ok=True)

    raw_cfg, cop_cfg = load_runtime(args)
    ctrl_y = raw_cfg.get("controller", {}) or {}
    safe_y = raw_cfg.get("safety", {}) or {}
    cop_y = raw_cfg.get("coppeliasim", {}) or {}
    fast_x_y = raw_cfg.get("fast_x_transport", {}) or {}
    use_gravity_compensation = bool(ctrl_y.get("use_gravity_compensation", False))
    diag_cfg = build_diagnostics_config(args, raw_cfg)
    apply_diagnostics_mode_overrides(args, diag_cfg)
    if diag_cfg.enable_coppelia_torque_diagnostics:
        print(
            "[torque-diagnostics] enabled "
            f"mode={diag_cfg.diagnostics_mode} "
            f"gain_scale={diag_cfg.impedance_gain_scale} "
            f"ref_smooth={diag_cfg.reference_smoothing_enabled}",
            flush=True,
        )

    scene_path = args.coppelia_root / "system" / "dfltscn.ttt"
    ur5_model = args.coppelia_root / "models" / "robots" / "non-mobile" / "UR5.ttm"
    if not scene_path.exists():
        raise FileNotFoundError(f"Missing default scene: {scene_path}")
    if not ur5_model.exists():
        raise FileNotFoundError(f"Missing official UR5 model: {ur5_model}")

    startup_t0 = time.monotonic()
    phases: list[dict[str, float | str]] = []

    def mark(label: str) -> None:
        elapsed = time.monotonic() - startup_t0
        phases.append({"phase": label, "elapsed_s": round(elapsed, 3)})
        print(f"[startup +{elapsed:6.2f}s] {label}", flush=True)

    mark("loaded controller configuration")
    ownership_metadata = controller_ownership_metadata(
        legacy_marker_handoff=bool(args.legacy_marker_handoff)
    )
    for line in startup_banner_lines(legacy_marker_handoff=bool(args.legacy_marker_handoff)):
        print(line, flush=True)

    # ---------- 1. Connect to CoppeliaSim ZMQ RPC ----------
    if args.legacy_marker_handoff:
        release_lua_bootstrap_for_rpc()
    client, sim = connect(
        args.host, args.port,
        retries=args.rpc_connect_retries,
        delay_s=args.rpc_connect_retry_delay_s,
    )
    mark(f"connected to RPC at {args.host}:{args.port}")

    # ---------- 2. Ensure simulation is stopped, then load UR5 model ----------
    sim_state = sim.getSimulationState()
    if sim_state != sim.simulation_stopped and not args.preloaded_scene:
        sim.stopSimulation()
        for _ in range(200):
            if sim.getSimulationState() == sim.simulation_stopped:
                break
            time.sleep(0.05)
        mark("existing simulation stopped")
    elif sim_state != sim.simulation_stopped and args.preloaded_scene:
        mark("preloaded scene simulation already running")

    try:
        ur5_handle = int(sim.getObject("/UR5"))
        mark("UR5 already present in scene")
    except Exception:
        mark("loading UR5 model into scene")
        ur5_handle = int(sim.loadModel(str(ur5_model)))
        mark(f"UR5 model loaded (handle={ur5_handle})")

    # ---------- 3. Resolve handles / configure adapter ----------
    adapter = CoppeliaSimURAdapter(cop_cfg)
    adapter.set_logger(lambda m: print(f"[adapter] {m}"))
    adapter.connect_with_existing_client(client)
    mark("adapter connected — joints and EE resolved")
    task_frame_summary = adapter.read_task_frame_summary()
    print(f"[task-frame] {json.dumps(task_frame_summary, sort_keys=True)}", flush=True)
    adapter.configure_force_torque_mode()
    mark("torque mode configured")
    spawn_coppelia_pendulum = (
        bool(args.spawn_coppelia_pendulum)
        if args.spawn_coppelia_pendulum is not None
        else bool(cop_y.get("spawn_pendulum", False))
    )
    pendulum_pole_length_m = (
        float(args.pendulum_pole_length_m)
        if args.pendulum_pole_length_m is not None
        else float(cop_y.get("pendulum_pole_length_m", 0.4))
    )
    coppelia_pendulum_handles = None
    if spawn_coppelia_pendulum:
        pendulum_parent_handle = int(
            task_frame_summary.get("handle", task_frame_summary.get("parent_handle", -1))
        )
        coppelia_pendulum_handles = ensure_coppelia_pendulum(
            sim,
            parent_handle=pendulum_parent_handle,
            pole_length_m=pendulum_pole_length_m,
            parent_path_hint=str(task_frame_summary.get("resolved_path", "")),
        )
        if coppelia_pendulum_handles is not None and coppelia_pendulum_handles.available:
            mark(
                "Coppelia pendulum ready "
                f"(hinge={coppelia_pendulum_handles.hinge_handle}, "
                f"pole_length_m={pendulum_pole_length_m:.3f})"
            )
        else:
            print("[pendulum] Coppelia pendulum spawn failed; theta will be unavailable", flush=True)
    use_lqr_outer_policy = bool(args.accel_x_transport and args.accel_profile == "lqr")
    if args.accel_x_transport and args.hold_transport_start_pose:
        transport_start_q = parse_joint_vector_env(
            "Q_START_RAD",
            DEFAULT_TRANSPORT_START_Q_COPPELIA,
        )
        adapter.set_joint_positions(transport_start_q)
        mark("UR5 set to Coppelia-derived transport start pose")

    # ---------- 4. Vision sensor (optional) ----------
    vision_sensor: int | None = None
    if not args.no_video:
        vision_sensor = make_vision_sensor(sim, args.width, args.height)
        configure_vision_sensor_render_all(sim, vision_sensor)
        mark("vision sensor created (entity_to_render=-1 where supported)")
        if not args.accel_x_transport:
            set_ur5_joints_for_video_framing(sim)
            mark("UR5 set to video-smoke joint pose (visible in camera)")

    actual_lqr_sim_time_step: float | None = None
    lqr_requested_sim_time_step = float(args.lqr_sim_time_step)
    if use_lqr_outer_policy:
        if not math.isfinite(lqr_requested_sim_time_step) or lqr_requested_sim_time_step <= 0.0:
            raise ValueError("--lqr-sim-time-step must be a positive finite float")
        try:
            requested_lqr_sim_time_step = float(lqr_requested_sim_time_step)
            param_ids: list[int] = []
            for attr_name in ("floatparam_simulation_time_step", "floatparam_physicstimestep"):
                if hasattr(sim, attr_name):
                    param_ids.append(int(getattr(sim, attr_name)))
            if not param_ids:
                param_ids.append(1)
            for param_id in dict.fromkeys(param_ids):
                sim.setFloatParam(int(param_id), requested_lqr_sim_time_step)
            mark(
                "LQR sim time step requested before start "
                f"({requested_lqr_sim_time_step:.4f}s)"
            )
        except Exception as exc:
            print(
                "[lqr] WARNING: unable to request simulation time step "
                f"{lqr_requested_sim_time_step:.4f}s "
                f"({type(exc).__name__}: {exc}); continuing with the scene default",
                flush=True,
            )

    # ---------- 5. Start stepped simulation ----------
    sim.setStepping(True)
    if sim.getSimulationState() == sim.simulation_stopped:
        sim.startSimulation()
        mark("simulation started (stepped)")
    else:
        mark("simulation already running — stepping enabled")
    if use_lqr_outer_policy:
        try:
            actual_lqr_sim_time_step = float(sim.getSimulationTimeStep())
            mark(
                "LQR sim time step active after start "
                f"({actual_lqr_sim_time_step:.4f}s)"
            )
            if (
                math.isfinite(lqr_requested_sim_time_step)
                and lqr_requested_sim_time_step > 0.0
                and abs(actual_lqr_sim_time_step - lqr_requested_sim_time_step)
                > max(1e-9, 0.1 * lqr_requested_sim_time_step)
            ):
                print(
                    "[lqr] WARNING: active simulation time step "
                    f"({actual_lqr_sim_time_step:.4f}s) does not match the requested "
                    f"value ({lqr_requested_sim_time_step:.4f}s)",
                    flush=True,
                )
        except Exception as exc:
            print(
                "[lqr] WARNING: unable to read active simulation time step after start "
                f"({type(exc).__name__}: {exc})",
                flush=True,
            )
    if not args.no_video and vision_sensor is not None:
        _enable_global_vision_and_display(sim)
        mark("vision/display bool params set (where supported)")

    # ---------- 6. Sample initial state, init controller, publish ready ----------
    transport_axis_idx = axis_name_to_index(args.transport_axis)
    transport_axis_name = axis_index_to_name(transport_axis_idx)
    orth_axis_idxs = orthogonal_axis_indices(transport_axis_idx)
    use_lqr_outer_policy = bool(args.accel_x_transport and args.accel_profile == "lqr")
    use_mpc_outer_policy = bool(args.accel_x_transport and args.accel_profile == "mpc")
    use_fast_x_profile = bool(args.accel_x_transport and args.accel_profile == "fast_x")
    fast_x_joint_speed_fraction = (
        float(args.fast_x_joint_speed_fraction)
        if args.fast_x_joint_speed_fraction is not None
        else float(fast_x_y.get("joint_speed_fraction", 0.82))
    )
    fast_x_accel_fraction = (
        float(args.fast_x_accel_fraction)
        if args.fast_x_accel_fraction is not None
        else float(fast_x_y.get("accel_fraction", 0.78))
    )
    fast_x_max_acceleration_mps2 = (
        float(args.fast_x_max_acceleration_mps2)
        if args.fast_x_max_acceleration_mps2 is not None
        else float(fast_x_y.get("max_acceleration_mps2", 0.45))
    )
    fast_x_max_velocity_mps = (
        float(args.fast_x_max_velocity_mps)
        if args.fast_x_max_velocity_mps is not None
        else float(fast_x_y.get("max_velocity_mps", 0.18))
    )
    max_joint_velocity_radps = float(safe_y.get("max_joint_velocity_radps", 4.8))
    if use_mpc_outer_policy and transport_axis_idx != 0:
        raise ValueError("--accel-profile mpc currently supports transport-axis x only")
    if use_mpc_outer_policy and args.accel_torque_policy not in {"ik_joint_pd", "cartesian_impedance"}:
        raise ValueError(
            "--accel-profile mpc currently requires --accel-torque-policy ik_joint_pd or cartesian_impedance"
        )
    if use_lqr_outer_policy and transport_axis_idx != 0:
        raise ValueError("--accel-profile lqr currently supports transport-axis x only")
    if use_lqr_outer_policy and args.accel_torque_policy not in {"ik_joint_pd", "cartesian_impedance"}:
        raise ValueError(
            "--accel-profile lqr currently requires --accel-torque-policy ik_joint_pd or cartesian_impedance"
        )
    if use_fast_x_profile and transport_axis_idx != 0:
        raise ValueError("--accel-profile fast_x currently supports transport-axis x only")

    lqr_move_axis_weight = 60.0
    lqr_hold_axis_weight = 260.0
    lqr_orientation_weight = 180.0
    lqr_hold_axis_gain = 16.0
    lqr_ik_joint_kp = 4.0
    lqr_ik_joint_kd = 1.0
    lqr_ik_torque_headroom = 0.02
    lqr_joint_torque_limit_scale = 0.0001
    # Tighten the joint-speed envelope for the LQR branch so the inner
    # allocator stays below the live qd guard instead of converging right at it.
    lqr_joint_speed_limit_scale = 0.20
    lqr_torque_rate_limit_scale = 0.25
    # The LQR outer loop already saturates the requested axis command. Keep the
    # cartesian-impedance inner loop much softer than the default transport
    # gains so the first few simulation steps do not immediately trip the qd
    # and fixed-axis guards.
    lqr_cartesian_impedance_gain_scale = 0.005
    lqr_cartesian_impedance_torque_headroom = 0.02
    initial_state = sample_controller_state(adapter, target_axis=0.0)
    q0 = np.asarray(initial_state["q"], dtype=np.float64)
    qd0 = np.asarray(initial_state["qd"], dtype=np.float64)
    ee_pos0 = np.asarray(initial_state["ee_pos"], dtype=np.float64)
    ee_quat0 = np.asarray(initial_state["ee_quat"], dtype=np.float64)
    lin0 = np.asarray(initial_state["ee_lin_vel"], dtype=np.float64)
    ang0 = np.asarray(initial_state["ee_ang_vel"], dtype=np.float64)
    target_x0 = float(ee_pos0[0])
    transport_axis0 = float(ee_pos0[transport_axis_idx])
    initial_state["target_x"] = target_x0
    initial_state["target_axis"] = transport_axis0
    task_rot0 = quat_wxyz_to_rotation_matrix(ee_quat0)
    mujoco_orientation_error_rad0 = rotation_error_angle_rad(
        task_rot0, TARGET_SITE_ROTATION_WORLD
    )
    orientation_target_rot = (
        TARGET_SITE_ROTATION_WORLD
        if args.task_orientation_target == "mujoco"
        else task_rot0.copy()
    )
    lqr_controller: FixedXTransportLQRController | None = None
    lqr_filter: CommandGovernorSafetyFilter | None = None
    lqr_goal_x: float | None = None
    lqr_gain_matrix: list[list[float]] | None = None
    lqr_riccati_converged: bool | None = None
    lqr_riccati_iters: int | None = None
    lqr_clipped_count = 0
    lqr_rejected_count = 0
    mpc_controller = None
    mpc_filter: CommandGovernorSafetyFilter | None = None
    mpc_goal_x: float | None = None
    mpc_clipped_count = 0
    mpc_rejected_count = 0
    mpc_pole_theta_hist: list[float] = []
    mpc_pole_theta_dot_hist: list[float] = []
    mujoco_cartpole_observer = (
        build_mujoco_cartpole_observer(default_cartpole_scene_candidates(REPO_ROOT))
        if use_mpc_outer_policy
        else None
    )

    def read_pole_state_for_control(
        *,
        q_arr: np.ndarray,
        qd_arr: np.ndarray,
        time_s: float,
        dt_s: float,
        target_x: float,
    ) -> tuple[float, float, str]:
        if coppelia_pendulum_handles is not None and coppelia_pendulum_handles.available:
            coppelia_state = read_coppelia_pendulum_state(
                sim,
                coppelia_pendulum_handles,
                parent_handle=int(task_frame_summary.get("parent_handle", -1)),
            )
            if coppelia_state is not None:
                return float(coppelia_state[0]), float(coppelia_state[1]), "coppelia"
        if mujoco_cartpole_observer is not None:
            pole_state = read_mujoco_cartpole_state(
                mujoco_cartpole_observer,
                q_arr,
                qd_arr,
                time_s=time_s,
                dt_s=dt_s,
                target_x=target_x,
                transport_axis_index=transport_axis_idx,
            )
            if pole_state is not None:
                return (
                    float(pole_state.theta),
                    float(pole_state.theta_dot),
                    "mujoco_observer",
                )
        return 0.0, 0.0, "unavailable"
    frame_reference_summary = {
        "transport_axis": transport_axis_name,
        "transport_axis_index": transport_axis_idx,
        "transport_direction": f"world_{transport_axis_name}",
        "fixed_axis_indices": list(orth_axis_idxs),
        "fixed_axis_names": [axis_index_to_name(i) for i in orth_axis_idxs],
        "orientation_target": str(args.task_orientation_target),
        "mujoco_target_rotation_world": TARGET_SITE_ROTATION_WORLD.tolist(),
        "initial_task_rotation_world": task_rot0.tolist(),
        "initial_to_mujoco_orientation_error_rad": float(mujoco_orientation_error_rad0),
        "initial_to_mujoco_orientation_error_deg": float(
            math.degrees(mujoco_orientation_error_rad0)
        ),
        "task_frame_coherent_with_mujoco_target": bool(
            mujoco_orientation_error_rad0 <= math.radians(3.0)
        ),
    }
    local_axis_capability = estimate_axis_transport_capability(
        initial_state["jacobian"], transport_axis_idx
    )
    mujoco_gravity_estimator = (
        build_mujoco_gravity_estimator() if use_gravity_compensation else None
    )
    initial_gravity_torque = (
        compute_mujoco_gravity_bias(mujoco_gravity_estimator, q0, qd0)
        if use_gravity_compensation
        else None
    )
    if initial_gravity_torque is not None:
        initial_state["gravity_torque"] = initial_gravity_torque

    imp_cfg = CartesianImpedanceConfig.from_controller_yaml_section(ctrl_y)
    transport_gain_scale = (
        float(args.impedance_gain_scale)
        if args.impedance_gain_scale is not None
        else (
            0.30
            if args.accel_x_transport
            and args.accel_profile == "reciprocating"
            and args.accel_torque_policy == "cartesian_impedance"
            else 1.0
        )
    )
    if (
        diag_cfg.enable_coppelia_torque_diagnostics
        and float(diag_cfg.impedance_gain_scale) != 1.0
    ):
        transport_gain_scale = float(diag_cfg.impedance_gain_scale)
    if transport_gain_scale != 1.0:
        gs = float(transport_gain_scale)
        imp_cfg = replace(
            imp_cfg,
            kp_x=float(imp_cfg.kp_x * gs),
            kd_x=float(imp_cfg.kd_x * gs),
            kp_y=float(imp_cfg.kp_y * gs),
            kd_y=float(imp_cfg.kd_y * gs),
            kp_z=float(imp_cfg.kp_z * gs),
            kd_z=float(imp_cfg.kd_z * gs),
            kp_rot=float(imp_cfg.kp_rot * gs),
            kd_rot=float(imp_cfg.kd_rot * gs),
            kp_posture=float(imp_cfg.kp_posture * gs),
            kd_posture=float(imp_cfg.kd_posture * gs),
            kd_joint=float(imp_cfg.kd_joint * gs),
        )
    if use_lqr_outer_policy and args.accel_torque_policy == "cartesian_impedance":
        imp_cfg = replace(
            imp_cfg,
            kp_x=float(imp_cfg.kp_x * lqr_cartesian_impedance_gain_scale),
            kd_x=float(imp_cfg.kd_x * lqr_cartesian_impedance_gain_scale),
            kp_y=float(imp_cfg.kp_y * lqr_cartesian_impedance_gain_scale),
            kd_y=float(imp_cfg.kd_y * lqr_cartesian_impedance_gain_scale),
            kp_z=float(imp_cfg.kp_z * lqr_cartesian_impedance_gain_scale),
            kd_z=float(imp_cfg.kd_z * lqr_cartesian_impedance_gain_scale),
            kp_rot=float(imp_cfg.kp_rot * lqr_cartesian_impedance_gain_scale),
            kd_rot=float(imp_cfg.kd_rot * lqr_cartesian_impedance_gain_scale),
            kp_posture=float(imp_cfg.kp_posture * lqr_cartesian_impedance_gain_scale),
            kd_posture=float(imp_cfg.kd_posture * lqr_cartesian_impedance_gain_scale),
            kd_joint=float(imp_cfg.kd_joint * lqr_cartesian_impedance_gain_scale),
            torque_headroom=float(lqr_cartesian_impedance_torque_headroom),
        )
    controller = XAxisCartesianImpedanceController(imp_cfg)
    controller.reset_from_state(initial_state)

    safety = ImpedanceSafetyMonitor(
        ImpedanceSafetyConfig(
            max_abs_y_drift_m=float(safe_y.get("max_abs_y_drift_m", 0.03)),
            max_abs_z_drift_m=float(safe_y.get("max_abs_z_drift_m", 0.03)),
            max_abs_orthogonal_drift_m=float(
                safe_y.get(
                    "max_abs_orthogonal_drift_m",
                    max(
                        float(safe_y.get("max_abs_y_drift_m", 0.03)),
                        float(safe_y.get("max_abs_z_drift_m", 0.03)),
                    ),
                )
            ),
            max_orientation_error_rad=float(safe_y.get("max_orientation_error_rad", 0.25)),
            max_joint_velocity_radps=float(safe_y.get("max_joint_velocity_radps", 1.5)),
            max_x_error_growth_steps=int(safe_y.get("max_x_error_growth_steps", 100)),
            max_axis_error_growth_steps=int(
                safe_y.get("max_axis_error_growth_steps", safe_y.get("max_x_error_growth_steps", 100))
            ),
            emergency_stop_on_nan=bool(safe_y.get("emergency_stop_on_nan", True)),
            emergency_stop_on_joint_limit=bool(safe_y.get("emergency_stop_on_joint_limit", True)),
        )
    )
    safety.set_initial_position(ee_pos0, transport_axis_idx)
    mark("initial state sampled — controller initialized")

    joint_cfg = adapter.read_joint_configuration_summary()
    joint_cfg_error = (
        joint_cfg["motor_enabled_verified"] is False
        or joint_cfg["ctrl_disabled_verified"] is False
        or joint_cfg["dynamic_mode_verified"] is False
    )
    joint_cfg_verified = (
        joint_cfg["motor_enabled_verified"] is True
        and joint_cfg["ctrl_disabled_verified"] is True
        and joint_cfg["dynamic_mode_verified"] is True
    )
    joint_cfg_readback_available = any(
        joint_cfg[key] is not None
        for key in ("motor_enabled_verified", "ctrl_disabled_verified", "dynamic_mode_verified")
    )
    for row in joint_cfg["joints"]:
        print(
            "[joint-mode] "
            f"{row['name']} handle={row['handle']} "
            f"motor_enabled={row.get('motor_enabled', 'n/a')} "
            f"ctrl_enabled={row.get('ctrl_enabled', 'n/a')} "
            f"joint_mode={row.get('joint_mode', 'n/a')} "
            f"dynamic={row.get('joint_mode_is_dynamic', 'n/a')}",
            flush=True,
        )
    if joint_cfg_error:
        raise RuntimeError(
            "Joint mode readback disagrees with torque-mode configuration; "
            "refusing to continue."
        )
    mark(
        "torque mode configured and verified"
        if joint_cfg_verified
        else "torque mode configured (readback incomplete)"
    )

    jacobian_comparison: dict[str, Any] | None = None
    if args.compare_jacobian:
        try:
            jacobian_comparison = adapter.compare_jacobians(cop_cfg.numerical_epsilon)
        except Exception as exc:
            jacobian_comparison = None
            print(
                "[jacobian-compare] unavailable "
                f"({type(exc).__name__}: {exc}); continuing with numerical Jacobian only",
                flush=True,
            )
            mark("API Jacobian comparison skipped")
        else:
            diff = jacobian_comparison["difference"]
            print(
                "[jacobian-compare] "
                f"max_abs={diff['max_abs']:.3e} "
                f"pos_max_abs={diff['position_max_abs']:.3e} "
                f"rot_max_abs={diff['rotation_max_abs']:.3e} "
                f"rel_fro={diff['relative_frobenius_norm']:.3e}",
                flush=True,
            )
            mark("API vs numerical Jacobian compared")

    sim.setInt32Signal(CONTROLLER_READY_SIGNAL, 1)
    sim.setStringSignal(CONTROLLER_READY_SIGNAL, "1")
    print(f"{CONTROLLER_READY_SIGNAL}=1", flush=True)
    mark("controller ready signal published")

    # ---------- 7. Probe-only exit ----------
    if args.probe_only:
        jac = np.asarray(initial_state["jacobian"], dtype=np.float64)
        mark("probe complete")
        probe_summary = {
            "mode": "probe_only",
            "scene_path": str(scene_path),
            "ur5_model_path": str(ur5_model),
            "config_path": str(args.config),
            "host": args.host,
            "port": args.port,
            "startup_phases": phases,
            "resolved_q": q0.tolist(),
            "resolved_qd": qd0.tolist(),
            "ee_pos": ee_pos0.tolist(),
            "ee_quat": ee_quat0.tolist(),
            "ee_lin_vel": lin0.tolist(),
            "ee_ang_vel": ang0.tolist(),
            "jacobian_condition_number": float(np.linalg.cond(jac)),
            "joint_configuration": joint_cfg,
            "task_frame": task_frame_summary,
            "frame_reference": frame_reference_summary,
            "local_axis_transport_capability": local_axis_capability,
            "startup_jacobian_comparison": jacobian_comparison,
            "torque_mode_verified": bool(joint_cfg_verified),
            "torque_mode_readback_available": bool(joint_cfg_readback_available),
            "controller_family": ownership_metadata["controller_family"],
            "uses_direct_torque_control": ownership_metadata["uses_direct_torque_control"],
            "stepping_owner": ownership_metadata["stepping_owner"],
            "simulation_started_by": ownership_metadata["simulation_started_by"],
            "lua_motion_enabled": ownership_metadata["lua_motion_enabled"],
            "legacy_marker_handoff": ownership_metadata["legacy_marker_handoff"],
            "use_gravity_compensation": use_gravity_compensation,
            "torque_headroom": float(imp_cfg.torque_headroom),
            "target_x": target_x0,
            "target_axis": transport_axis0,
            "probe_passed": bool(
                joint_cfg_verified
                and (
                    jacobian_comparison is None
                    or bool(jacobian_comparison["difference"]["all_finite"])
                )
            ),
        }
        summary_path.write_text(json.dumps(probe_summary, indent=2), encoding="utf-8")
        print(json.dumps(probe_summary, indent=2))
        sim.stopSimulation()
        request_controller_shutdown(sim, client)
        mark("simulation stopped — probe exit")
        return

    if args.zero_torque_test:
        sim_dt = float(sim.getSimulationTimeStep())
        if sim_dt <= 0.0:
            sim_dt = 0.01
        steps_total = int(max(1, round(args.duration / sim_dt)))
        frame_every = max(1, int(round(1.0 / (args.fps * sim_dt))))
        frames: list[np.ndarray] = []
        q_start = np.asarray(q0, dtype=np.float64).copy()
        q_last = q_start.copy()
        qd_last = np.asarray(qd0, dtype=np.float64).copy()
        x_hist: list[float] = []
        y_hist: list[float] = []
        z_hist: list[float] = []
        ori_err_hist: list[float] = []
        ee_vx_hist: list[float] = []
        ee_speed_hist: list[float] = []
        qd_speed_hist: list[float] = []
        try:
            with JsonlTraceWriter(trace_path) as trace:
                for step in range(steps_total):
                    if not args.no_video and vision_sensor is not None:
                        place_vision_sensor_for_frame(
                            sim, vision_sensor, args.video_camera,
                            step, steps_total, adapter,
                        )
                    tau = np.zeros(6, dtype=np.float64)
                    adapter.apply_torque(tau)
                    if step == 0:
                        mark("zero-torque command applied")
                    sim.step()
                    sim_time = float(sim.getSimulationTime())
                    q, qd = adapter.read_joint_state()
                    ee_pos, ee_quat, ee_lin, ee_ang = adapter.read_ee_pose_twist()
                    q_last = np.asarray(q, dtype=np.float64)
                    qd_last = np.asarray(qd, dtype=np.float64)
                    ee_pos_arr = np.asarray(ee_pos, dtype=np.float64)
                    ee_lin_arr = np.asarray(ee_lin, dtype=np.float64)
                    qd_arr = np.asarray(qd, dtype=np.float64)
                    ori_err = float(np.linalg.norm(orientation_error_vec_wxyz(ee_quat0, ee_quat)))
                    trace.write_row({
                        "time": sim_time,
                        "mode": "zero_torque_test",
                        "q": q,
                        "qd": qd,
                        "ee_pos": ee_pos,
                        "ee_quat": ee_quat,
                        "ee_lin_vel": ee_lin,
                        "ee_ang_vel": ee_ang,
                        "tau_cmd": tau.tolist(),
                    })
                    x_hist.append(float(ee_pos_arr[0]))
                    y_hist.append(float(ee_pos_arr[1]))
                    z_hist.append(float(ee_pos_arr[2]))
                    ori_err_hist.append(ori_err)
                    ee_vx_hist.append(float(ee_lin_arr[0]))
                    ee_speed_hist.append(float(np.linalg.norm(ee_lin_arr)))
                    qd_speed_hist.append(float(np.max(np.abs(qd_arr))))
                    if (
                        not args.no_video
                        and vision_sensor is not None
                        and (step % frame_every == 0 or step == steps_total - 1)
                    ):
                        frame_rgb, _ = read_vision_rgb24(sim, vision_sensor)
                        frames.append(frame_rgb)
        finally:
            adapter.apply_torque(np.zeros(6, dtype=np.float64))
            sim.stopSimulation()
            request_controller_shutdown(sim, client)
        mark("simulation stopped — zero-torque exit")
        delta_q = q_last - q_start
        x_span = float(max(x_hist) - min(x_hist)) if x_hist else None
        x_net = float(x_hist[-1] - x_hist[0]) if len(x_hist) >= 2 else None
        max_abs_x_drift = float(max(abs(x - float(ee_pos0[0])) for x in x_hist)) if x_hist else None
        max_abs_y_drift = float(max(abs(y - float(ee_pos0[1])) for y in y_hist)) if y_hist else None
        max_abs_z_drift = float(max(abs(z - float(ee_pos0[2])) for z in z_hist)) if z_hist else None
        max_orientation_error_rad = float(max(ori_err_hist)) if ori_err_hist else None
        peak_abs_ee_vx = float(max(abs(v) for v in ee_vx_hist)) if ee_vx_hist else None
        peak_ee_speed = float(max(ee_speed_hist)) if ee_speed_hist else None
        peak_joint_speed = float(max(qd_speed_hist)) if qd_speed_hist else None
        summary = {
            "mode": "zero_torque_test",
            "scene_path": str(scene_path),
            "ur5_model_path": str(ur5_model),
            "config_path": str(args.config),
            "host": args.host,
            "port": args.port,
            "startup_phases": phases,
            "duration_s": args.duration,
            "sim_dt_s": sim_dt,
            "total_steps": steps_total,
            "zero_torque_test_completed": True,
            "joint_configuration": joint_cfg,
            "task_frame": task_frame_summary,
            "frame_reference": frame_reference_summary,
            "local_axis_transport_capability": local_axis_capability,
            "startup_jacobian_comparison": jacobian_comparison,
            "torque_mode_verified": bool(joint_cfg_verified),
            "torque_mode_readback_available": bool(joint_cfg_readback_available),
            "q_start_rad": q_start.tolist(),
            "qd_start_rad_s": qd_start.tolist(),
            "q_end_rad": q_last.tolist(),
            "qd_end_rad_s": qd_last.tolist(),
            "delta_q_rad": delta_q.tolist(),
            "delta_qd_rad_s": (qd_last - qd_start).tolist(),
            "max_abs_delta_q_rad": float(np.max(np.abs(delta_q))),
            "x_span_m": x_span,
            "x_net_displacement_m": x_net,
            "max_abs_x_drift_m": max_abs_x_drift,
            "initial_ee_world_m": ee_pos0.tolist(),
            "final_ee_world_m": [x_hist[-1], y_hist[-1], z_hist[-1]] if x_hist else None,
            "max_abs_y_drift_m": max_abs_y_drift,
            "max_abs_z_drift_m": max_abs_z_drift,
            "max_orientation_error_rad": max_orientation_error_rad,
            "peak_abs_ee_vx_mps": peak_abs_ee_vx,
            "peak_ee_speed_mps": peak_ee_speed,
            "peak_joint_speed_rad_s": peak_joint_speed,
            "video_path": None if args.no_video else str(video_path),
            "trace_path": str(trace_path),
            "frames_written": 0 if args.no_video else len(frames),
            "video_camera": str(args.video_camera),
            "controller_family": ownership_metadata["controller_family"],
            "uses_direct_torque_control": ownership_metadata["uses_direct_torque_control"],
            "stepping_owner": ownership_metadata["stepping_owner"],
            "simulation_started_by": ownership_metadata["simulation_started_by"],
            "lua_motion_enabled": ownership_metadata["lua_motion_enabled"],
            "legacy_marker_handoff": ownership_metadata["legacy_marker_handoff"],
            "success": None,
        }
        if not args.no_video and frames:
            arr0 = np.asarray(frames[0], dtype=np.float64)
            summary["first_frame_mean_rgb"] = float(np.mean(arr0))
            summary["first_frame_std_rgb"] = float(np.std(arr0))
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        if not args.no_video and frames:
            write_video_ffmpeg(video_path, frames, fps=args.fps)
        if not args.no_video and frames and float(summary.get("first_frame_std_rgb", 0) or 0) < 1.0:
            print(
                "[vision] warning: first captured frame is nearly constant (low std); "
                "check Xvfb/GPU, camera pose, and vision sensor path.",
                flush=True,
            )
        print(json.dumps(summary, indent=2))
        print(f"Saved trace:   {trace_path}")
        if not args.no_video:
            print(f"Saved video:   {video_path}")
        print(f"Saved summary: {summary_path}")
        return

    if args.torque_pulse:
        # Small open-loop torque on one joint; remainder of --duration is zero torque.
        sim_dt = float(sim.getSimulationTimeStep())
        if sim_dt <= 0.0:
            sim_dt = 0.01
        steps_total = int(max(1, round(args.duration / sim_dt)))
        pulse_n = int(min(max(0, args.torque_pulse_steps), steps_total))
        j = int(args.torque_pulse_joint) % 6
        nm = float(args.torque_pulse_nm)
        bidirectional = bool(args.torque_pulse_bidirectional)
        inter_pulse_gap = max(1, pulse_n // 4) if bidirectional else 0
        second_pulse_start = pulse_n + inter_pulse_gap
        second_pulse_n = max(0, min(pulse_n, steps_total - second_pulse_start)) if bidirectional else 0
        mark(
            f"torque pulse: joint {j} τ={nm} N·m for {pulse_n}/{steps_total} steps"
            + (f" + reverse pulse ({second_pulse_n} steps after gap {inter_pulse_gap})" if bidirectional else "")
            + ", "
            f"sim_dt={sim_dt:.4f}s",
        )
        frame_every = max(1, int(round(1.0 / (args.fps * sim_dt))))
        frames: list[np.ndarray] = []
        q_start = np.asarray(q0, dtype=np.float64).copy()
        qd_start = np.asarray(qd0, dtype=np.float64).copy()
        q_last = q_start.copy()
        qd_last = qd_start.copy()
        phase_counts = {"positive": 0, "negative": 0, "zero": 0}
        try:
            with JsonlTraceWriter(trace_path) as trace:
                for step in range(steps_total):
                    # Per Coppelia: set camera pose, then sim.step, then handleVisionSensor + getVisionSensorImg.
                    if not args.no_video and vision_sensor is not None:
                        place_vision_sensor_for_frame(
                            sim, vision_sensor, args.video_camera,
                            step, steps_total, adapter,
                        )
                    tau = np.zeros(6, dtype=np.float64)
                    phase = "zero"
                    if step < pulse_n:
                        tau[j] = nm
                        phase = "positive"
                    elif bidirectional and step >= second_pulse_start and step < second_pulse_start + second_pulse_n:
                        tau[j] = -nm
                        phase = "negative"
                    phase_counts[phase] += 1
                    adapter.apply_torque(tau)
                    if step == 0:
                        mark("first open-loop pulse torque applied")
                    sim.step()
                    sim_time = float(sim.getSimulationTime())
                    q, qd = adapter.read_joint_state()
                    q_last = np.asarray(q, dtype=np.float64)
                    qd_last = np.asarray(qd, dtype=np.float64)
                    trace.write_row({
                        "time": sim_time,
                        "mode": "torque_pulse",
                        "q": q,
                        "qd": qd,
                        "tau_cmd": tau.tolist(),
                        "torque_phase": phase,
                    })
                    if (
                        not args.no_video
                        and vision_sensor is not None
                        and (step % frame_every == 0 or step == steps_total - 1)
                    ):
                        frame_rgb, _ = read_vision_rgb24(sim, vision_sensor)
                        frames.append(frame_rgb)
        finally:
            adapter.apply_torque(np.zeros(6, dtype=np.float64))
            sim.stopSimulation()
            request_controller_shutdown(sim, client)
        mark("simulation stopped — torque-pulse exit")
        delta = q_last - q_start
        delta_qd = qd_last - qd_start
        summary = {
            "mode": "torque_pulse",
            "scene_path": str(scene_path),
            "ur5_model_path": str(ur5_model),
            "config_path": str(args.config),
            "host": args.host, "port": args.port,
            "startup_phases": phases,
            "joint_configuration": joint_cfg,
            "task_frame": task_frame_summary,
            "frame_reference": frame_reference_summary,
            "local_axis_transport_capability": local_axis_capability,
            "startup_jacobian_comparison": jacobian_comparison,
            "torque_mode_verified": bool(joint_cfg_verified),
            "torque_mode_readback_available": bool(joint_cfg_readback_available),
            "pulse_joint": j,
            "pulse_torque_nm": nm,
            "pulse_steps": pulse_n,
            "bidirectional": bidirectional,
            "inter_pulse_gap_steps": inter_pulse_gap,
            "second_pulse_steps": second_pulse_n,
            "phase_counts": phase_counts,
            "total_steps": steps_total,
            "sim_dt_s": sim_dt,
            "duration_s": args.duration,
            "q_start_rad": q_start.tolist(),
            "q_end_rad": q_last.tolist(),
            "qd_start_rad_s": qd_start.tolist(),
            "qd_end_rad_s": qd_last.tolist(),
            "delta_q_rad": delta.tolist(),
            "delta_qd_rad_s": delta_qd.tolist(),
            "max_abs_delta_q_rad": float(np.max(np.abs(delta))),
            "max_abs_delta_qd_rad_s": float(np.max(np.abs(delta_qd))),
            "pulse_joint_delta_q_rad": float(delta[j]),
            "pulse_joint_delta_qd_rad_s": float(delta_qd[j]),
            "video_path": None if args.no_video else str(video_path),
            "trace_path": str(trace_path),
            "frames_written": 0 if args.no_video else len(frames),
            "video_camera": str(args.video_camera),
            "controller_family": ownership_metadata["controller_family"],
            "uses_direct_torque_control": ownership_metadata["uses_direct_torque_control"],
            "stepping_owner": ownership_metadata["stepping_owner"],
            "simulation_started_by": ownership_metadata["simulation_started_by"],
            "lua_motion_enabled": ownership_metadata["lua_motion_enabled"],
            "legacy_marker_handoff": ownership_metadata["legacy_marker_handoff"],
            "success": True,
        }
        if not args.no_video and frames:
            arr0 = np.asarray(frames[0], dtype=np.float64)
            summary["first_frame_mean_rgb"] = float(np.mean(arr0))
            summary["first_frame_std_rgb"] = float(np.std(arr0))
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        if not args.no_video and frames:
            write_video_ffmpeg(video_path, frames, fps=args.fps)
        if not args.no_video and frames and float(summary.get("first_frame_std_rgb", 0) or 0) < 1.0:
            print(
                "[vision] warning: first captured frame is nearly constant (low std); "
                "check Xvfb/GPU, camera pose, and vision sensor path.",
                flush=True,
            )
        print(json.dumps(summary, indent=2))
        print(f"Saved trace:   {trace_path}")
        if not args.no_video:
            print(f"Saved video:   {video_path}")
        print(f"Saved summary: {summary_path}")
        return

    effective_warmup_steps = 0 if args.accel_x_transport else int(args.warmup_steps)
    if args.accel_x_transport and args.warmup_steps > 0:
        mark("zero-torque warmup skipped for accel-transport mode")
    if effective_warmup_steps > 0:
        run_zero_torque_warmup(adapter, sim, effective_warmup_steps)
        mark(f"zero-torque warmup complete ({effective_warmup_steps} steps)")
        initial_state = sample_controller_state(adapter, target_axis=0.0)
        q0 = np.asarray(initial_state["q"], dtype=np.float64)
        qd0 = np.asarray(initial_state["qd"], dtype=np.float64)
        ee_pos0 = np.asarray(initial_state["ee_pos"], dtype=np.float64)
        ee_quat0 = np.asarray(initial_state["ee_quat"], dtype=np.float64)
        lin0 = np.asarray(initial_state["ee_lin_vel"], dtype=np.float64)
        ang0 = np.asarray(initial_state["ee_ang_vel"], dtype=np.float64)
        target_x0 = float(ee_pos0[0])
        transport_axis0 = float(ee_pos0[transport_axis_idx])
        initial_state["target_x"] = target_x0
        initial_state["target_axis"] = transport_axis0
        task_rot0 = quat_wxyz_to_rotation_matrix(ee_quat0)
        mujoco_orientation_error_rad0 = rotation_error_angle_rad(
            task_rot0, TARGET_SITE_ROTATION_WORLD
        )
        orientation_target_rot = (
            TARGET_SITE_ROTATION_WORLD
            if args.task_orientation_target == "mujoco"
            else task_rot0.copy()
        )
        frame_reference_summary = {
            "transport_axis": transport_axis_name,
            "transport_axis_index": transport_axis_idx,
            "transport_direction": f"world_{transport_axis_name}",
            "fixed_axis_indices": list(orth_axis_idxs),
            "fixed_axis_names": [axis_index_to_name(i) for i in orth_axis_idxs],
            "orientation_target": str(args.task_orientation_target),
            "mujoco_target_rotation_world": TARGET_SITE_ROTATION_WORLD.tolist(),
            "initial_task_rotation_world": task_rot0.tolist(),
            "initial_to_mujoco_orientation_error_rad": float(mujoco_orientation_error_rad0),
            "initial_to_mujoco_orientation_error_deg": float(
                math.degrees(mujoco_orientation_error_rad0)
            ),
            "task_frame_coherent_with_mujoco_target": bool(
                mujoco_orientation_error_rad0 <= math.radians(3.0)
            ),
        }
        local_axis_capability = estimate_axis_transport_capability(
            initial_state["jacobian"], transport_axis_idx
        )
        initial_gravity_torque = (
            compute_mujoco_gravity_bias(mujoco_gravity_estimator, q0, qd0)
            if use_gravity_compensation
            else None
        )
        if initial_gravity_torque is not None:
            initial_state["gravity_torque"] = initial_gravity_torque
        controller.reset_from_state(initial_state)
        safety.set_initial_position(ee_pos0, transport_axis_idx)
        mark("warmup state sampled — controller reset")

    # ---------- 8. Torque-control loop ----------
    rate_dict = ctrl_y.get("torque_rate_limit_nm_per_sec", {}) or {}
    rate_arr = np.asarray([float(rate_dict[n]) for n in JOINT_NAME_ORDER], dtype=np.float64)
    if use_lqr_outer_policy:
        rate_arr = rate_arr * float(lqr_torque_rate_limit_scale)
    torque_filter = TorqueCommandFilter(
        num_joints=6,
        lowpass_alpha=float(ctrl_y.get("torque_lowpass_alpha", 0.15)),
        rate_limit_nm_per_sec=rate_arr,
    )

    transport_target_pos = ee_pos0.copy()
    target_axis = transport_target_pos[transport_axis_idx]
    target_axis_vel = 0.0
    target_axis_accel = 0.0
    accel_v_state = 0.0
    accel_total_time = None
    ik_ctrl_ref = q0.copy()
    ik_axis_state = 0.0
    tau_limit = np.asarray(imp_cfg.tau_max_nm, dtype=np.float64).reshape(6)
    mark("controller ready for closed-loop motion")

    torque_diagnostics: CoppeliaTorqueDiagnostics | None = None
    if diag_cfg.enable_coppelia_torque_diagnostics:
        run_label = (
            str(args.torque_diagnostics_run_label).strip()
            or str(diag_cfg.diagnostics_mode)
        )
        controller_mode_label = (
            f"{args.accel_torque_policy}"
            if args.accel_x_transport
            else "cartesian_impedance_hold"
        )
        torque_diagnostics = CoppeliaTorqueDiagnostics(
            diag_cfg,
            tau_limit_nm=np.asarray(imp_cfg.tau_max_nm, dtype=np.float64),
            rate_limit_nm_per_sec=rate_arr,
            controller_mode=controller_mode_label,
            prefer_signed_target_force=bool(cop_cfg.prefer_signed_target_force),
            joint_configuration=joint_cfg,
            run_label=run_label,
        )
    diag_force_hold_pose = diag_cfg.diagnostics_mode in {
        "hold_soft",
        "ref_step",
        "ref_smooth",
    }
    diag_sinusoid_joint = diag_cfg.diagnostics_mode == "sinusoid_joint"
    diag_passive = diag_cfg.diagnostics_mode == "passive"
    diag_ref_step_mid = diag_cfg.diagnostics_mode == "ref_step"
    diag_sinusoid_joint_idx = int(diag_cfg.sinusoid_joint_index) % 6
    diag_ref_step_applied = False

    sim_dt = float(sim.getSimulationTimeStep())
    if sim_dt <= 0.0:
        sim_dt = 0.01
    frame_every = max(1, int(round(1.0 / (args.fps * sim_dt))))
    frames: list[np.ndarray] = []
    x_err_hist: list[float] = []
    y_err_hist: list[float] = []
    z_err_hist: list[float] = []
    ori_err_hist: list[float] = []
    x_hist: list[float] = []
    y_hist: list[float] = []
    z_hist: list[float] = []
    axis_v_hist: list[float] = []
    ee_vx_hist: list[float] = []
    ee_speed_hist: list[float] = []
    qd_speed_hist: list[float] = []
    target_x_hist: list[float] = []
    target_vx_hist: list[float] = []
    target_ax_hist: list[float] = []
    target_axis_hist: list[float] = []
    target_axis_vel_hist: list[float] = []
    target_axis_acc_hist: list[float] = []
    axis_error_hist: list[float] = []
    fixed_axis_1_error_hist: list[float] = []
    fixed_axis_2_error_hist: list[float] = []
    tau_hist: list[np.ndarray] = []
    tau_raw_hist: list[np.ndarray] = []
    ik_speed_scale_hist: list[float] = []
    ik_torque_scale_hist: list[float] = []
    ik_q_ref_excursion_hist: list[float] = []
    q_excursion_hist: list[float] = []
    y_leak_hist: list[float] = []
    task_backtrack_scale_hist: list[float] = []
    task_scale_hist: list[float] = []
    task_backtrack_iters_hist: list[int] = []
    task_feasible_hist: list[bool] = []
    safety_stop_reason: str | None = None
    transport_reanchored = False
    reciprocating_plan = None
    reciprocating_phase = "idle"
    reciprocating_axis_offset_slew = 0.0
    fast_transport_a_max = float(args.a_x_max)
    fast_transport_v_max = float(args.v_x_max)
    fast_x_limits_diag: dict[str, Any] = {}
    if use_fast_x_profile:
        startup_jacobian = np.asarray(initial_state["jacobian"], dtype=np.float64)
        fast_transport_a_max, fast_transport_v_max, fast_x_limits_diag = (
            recommend_fast_point_to_point_limits(
                float(args.target_dx),
                {},
                max_joint_velocity_radps=max_joint_velocity_radps,
                max_acceleration_mps2=fast_x_max_acceleration_mps2,
                joint_speed_fraction=fast_x_joint_speed_fraction,
                accel_fraction=fast_x_accel_fraction,
                jacobian=startup_jacobian,
                transport_axis_index=transport_axis_idx,
            )
        )
        fast_transport_v_max = min(fast_transport_v_max, fast_x_max_velocity_mps)
        fast_transport_a_max = min(fast_transport_a_max, fast_x_max_acceleration_mps2)
        if float(args.duration) <= 0.0:
            args.duration = minimum_fast_run_duration_s(
                float(args.target_dx),
                fast_transport_a_max,
                fast_transport_v_max,
                settle_duration_s=float(args.settle_duration),
            )
        print(
            "[fast_x] startup limits: "
            f"a_max={fast_transport_a_max:.4f} m/s^2, "
            f"v_max={fast_transport_v_max:.4f} m/s, "
            f"target_dx={float(args.target_dx):.4f} m, "
            f"duration={float(args.duration):.2f} s, "
            f"profile={fast_x_limits_diag.get('profile', 'unknown')}",
            flush=True,
        )
    if args.accel_x_transport and args.accel_profile == "reciprocating":
        reciprocating_plan = build_reciprocating_plan(
            stroke_m=float(args.reciprocating_stroke_m),
            a_abs_m_s2=float(args.a_x_max),
            v_abs_m_s=float(args.v_x_max),
            hold_s=float(args.reciprocating_hold_s),
        )
        min_duration = minimum_run_duration_s(
            reciprocating_plan,
            settle_duration_s=float(args.settle_duration),
        )
        if float(args.duration) < min_duration:
            print(
                f"[reciprocating] extending duration from {args.duration:.2f}s "
                f"to {min_duration:.2f}s for full stroke cycle",
                flush=True,
            )
            args.duration = float(min_duration)
        print(
            "[reciprocating] plan: "
            f"stroke={reciprocating_plan.stroke_m:.4f} m, "
            f"motion={reciprocating_plan.motion_duration_s:.2f} s, "
            f"segments={len(reciprocating_plan.segments)}",
            flush=True,
        )
    steps = int(round(args.duration / sim_dt))
    prev_sim_time: float | None = None

    try:
        with JsonlTraceWriter(trace_path) as trace:
            for step in range(steps):
                if not args.no_video and vision_sensor is not None:
                    place_vision_sensor_for_frame(
                        sim, vision_sensor, args.video_camera,
                        step, steps, adapter,
                    )
                sim_time = float(sim.getSimulationTime())
                step_dt = (
                    sim_dt
                    if prev_sim_time is None
                    else max(sim_time - float(prev_sim_time), 1e-6)
                )
                q, qd = adapter.read_joint_state()
                ee_pos, ee_quat, ee_lin, ee_ang = adapter.read_ee_pose_twist()
                j_pos, j_rot = adapter.read_jacobian()
                ee_pos_arr = np.asarray(ee_pos, dtype=np.float64)
                ee_lin_arr = np.asarray(ee_lin, dtype=np.float64)
                gravity_torque = (
                    compute_mujoco_gravity_bias(mujoco_gravity_estimator, q, qd)
                    if use_gravity_compensation
                    else None
                )

                use_ik_torque_policy = (
                    args.accel_x_transport and args.accel_torque_policy == "ik_joint_pd"
                )
                use_point_to_point_ik_policy = (
                    use_ik_torque_policy and args.accel_profile == "point_to_point"
                )
                use_fast_x_ik_policy = (
                    use_ik_torque_policy and args.accel_profile == "fast_x"
                )
                use_reciprocating_ik_policy = (
                    use_ik_torque_policy and args.accel_profile == "reciprocating"
                )
                reciprocating_ik_weights = (
                    reciprocating_ik_task_weights()
                    if use_reciprocating_ik_policy or use_fast_x_ik_policy
                    else {}
                )
                ik_diag: dict[str, Any] = {}
                pd_diag: dict[str, Any] = {}
                outer_diag: dict[str, Any] = {}

                if sim_time >= args.settle_duration:
                    t_move = sim_time - float(args.settle_duration)
                    if args.accel_x_transport and not transport_reanchored:
                        transport_target_pos = np.asarray(ee_pos, dtype=np.float64).copy()
                        transport_axis0 = float(transport_target_pos[transport_axis_idx])
                        target_axis = float(transport_axis0)
                        target_axis_vel = 0.0
                        target_axis_accel = 0.0
                        accel_v_state = 0.0
                        ik_axis_state = 0.0
                        transport_reanchored = True
                        if use_lqr_outer_policy:
                            (
                                lqr_controller,
                                lqr_filter,
                                lqr_goal_x,
                            ) = build_coppelia_lqr_transport_stack(
                                x_start=float(transport_axis0),
                                target_dx=float(args.target_dx),
                                dt_s=float(sim_dt),
                                q_x=float(args.lqr_q_x),
                                q_xdot=float(args.lqr_q_xdot),
                                r_weight=float(args.lqr_r_weight),
                                accel_limit=float(args.a_x_max),
                                velocity_limit=float(args.v_x_max),
                                command_change_limit=float(args.lqr_command_change_per_cycle),
                                guardrail_margin_m=float(args.lqr_guardrail_margin_m),
                            )
                            lqr_gain_matrix = lqr_controller.gain_matrix.tolist()
                            lqr_riccati_converged = bool(lqr_controller.riccati_converged)
                            lqr_riccati_iters = int(lqr_controller.riccati_iters)
                            transport_target_pos[transport_axis_idx] = float(lqr_goal_x)
                            target_axis = float(lqr_goal_x)
                        elif use_mpc_outer_policy:
                            mpc_q = np.array(
                                [
                                    float(args.mpc_q_x),
                                    float(args.mpc_q_xdot),
                                    float(args.mpc_q_theta),
                                    float(args.mpc_q_theta_dot),
                                ],
                                dtype=np.float64,
                            )
                            (
                                mpc_controller,
                                mpc_filter,
                                mpc_goal_x,
                            ) = build_coppelia_mpc_transport_stack(
                                x_start=float(transport_axis0),
                                target_dx=float(args.target_dx),
                                dt_s=float(sim_dt),
                                horizon=int(args.mpc_horizon),
                                q_weights=mpc_q,
                                q_terminal_scale=float(args.mpc_q_terminal_scale),
                                r_weight=float(args.mpc_r_weight),
                                pole_length_m=float(args.mpc_pole_length_m),
                                accel_limit=float(args.a_x_max),
                                velocity_limit=float(args.v_x_max),
                                command_change_limit=float(args.mpc_command_change_per_cycle),
                                guardrail_margin_m=float(args.mpc_guardrail_margin_m),
                            )
                            transport_target_pos[transport_axis_idx] = float(mpc_goal_x)
                            target_axis = float(mpc_goal_x)
                        elif use_fast_x_profile:
                            live_jacobian = np.vstack([j_pos, j_rot])
                            (
                                fast_transport_a_max,
                                fast_transport_v_max,
                                fast_x_limits_diag,
                            ) = recommend_fast_point_to_point_limits(
                                float(args.target_dx),
                                {},
                                max_joint_velocity_radps=max_joint_velocity_radps,
                                max_acceleration_mps2=fast_x_max_acceleration_mps2,
                                joint_speed_fraction=fast_x_joint_speed_fraction,
                                accel_fraction=fast_x_accel_fraction,
                                jacobian=live_jacobian,
                                transport_axis_index=transport_axis_idx,
                            )
                            fast_transport_v_max = min(
                                fast_transport_v_max, fast_x_max_velocity_mps
                            )
                            fast_transport_a_max = min(
                                fast_transport_a_max, fast_x_max_acceleration_mps2
                            )
                            print(
                                "[fast_x] motion-start limits: "
                                f"a_max={fast_transport_a_max:.4f} m/s^2, "
                                f"v_max={fast_transport_v_max:.4f} m/s, "
                                f"profile={fast_x_limits_diag.get('profile', 'unknown')}",
                                flush=True,
                            )
                    if args.accel_x_transport and args.accel_profile == "point_to_point":
                        axis_ref, axis_v_ref, axis_a_ref, total_time = point_to_point_accel_reference(
                            t_move,
                            target_dx=float(args.target_dx),
                            a_abs=float(args.a_x_max),
                            v_abs=float(args.v_x_max),
                        )
                        if use_ik_torque_policy:
                            target_axis_accel = axis_a_ref
                        else:
                            transport_target_pos[transport_axis_idx] = transport_axis0 + axis_ref
                            target_axis = float(transport_target_pos[transport_axis_idx])
                            target_axis_vel = float(axis_v_ref)
                            target_axis_accel = float(axis_a_ref)
                        accel_total_time = total_time
                    elif args.accel_x_transport and args.accel_profile == "fast_x":
                        axis_ref, axis_v_ref, axis_a_ref, total_time = point_to_point_accel_reference(
                            t_move,
                            target_dx=float(args.target_dx),
                            a_abs=float(fast_transport_a_max),
                            v_abs=float(fast_transport_v_max),
                        )
                        if use_ik_torque_policy:
                            axis_pos = float(ee_pos_arr[transport_axis_idx]) - float(
                                transport_axis0
                            )
                            pos_err = float(axis_ref) - axis_pos
                            vel_err = float(axis_v_ref) - float(
                                ee_lin_arr[transport_axis_idx]
                            )
                            track_accel = float(
                                np.clip(
                                    30.0 * pos_err + 10.0 * vel_err,
                                    -abs(float(fast_transport_a_max)),
                                    abs(float(fast_transport_a_max)),
                                )
                            )
                            ik_axis_state = float(axis_v_ref)
                            target_axis_accel = float(
                                0.55 * float(axis_a_ref) + 0.45 * track_accel
                            )
                            transport_target_pos[transport_axis_idx] = transport_axis0 + float(
                                axis_ref
                            )
                            target_axis = float(transport_target_pos[transport_axis_idx])
                            target_axis_vel = float(axis_v_ref)
                        else:
                            transport_target_pos[transport_axis_idx] = transport_axis0 + axis_ref
                            target_axis = float(transport_target_pos[transport_axis_idx])
                            target_axis_vel = float(axis_v_ref)
                            target_axis_accel = float(axis_a_ref)
                        accel_total_time = total_time
                        outer_diag = {
                            "phase": "fast_x_point_to_point",
                            "controller": "limit_aware_point_to_point",
                            "a_x_max_mps2": float(fast_transport_a_max),
                            "v_x_max_mps": float(fast_transport_v_max),
                            "limits": fast_x_limits_diag,
                        }
                    elif args.accel_x_transport and args.accel_profile == "mujoco_windows":
                        target_axis_accel = mujoco_accel_window_command(t_move, float(args.a_x_max))
                        if not use_ik_torque_policy:
                            accel_v_state = float(
                                np.clip(
                                    accel_v_state + target_axis_accel * sim_dt,
                                    -abs(float(args.v_x_max)),
                                    abs(float(args.v_x_max)),
                                )
                            )
                            target_axis_vel = accel_v_state
                            transport_target_pos[transport_axis_idx] += target_axis_vel * sim_dt
                    elif args.accel_x_transport and args.accel_profile == "lqr":
                        if lqr_controller is None or lqr_filter is None or lqr_goal_x is None:
                            raise RuntimeError("LQR transport was not initialized before motion start")
                        target_axis_accel, lqr_diag = compute_coppelia_lqr_outer_command(
                            lqr_controller,
                            lqr_filter,
                            x_now=float(ee_pos_arr[transport_axis_idx]),
                            x_dot_now=float(ee_lin_arr[transport_axis_idx]),
                            time_s=sim_time,
                            dt_s=sim_dt,
                            target_x=float(lqr_goal_x),
                        )
                        lqr_clipped_count += int(bool(lqr_diag.get("clipped", False)))
                        lqr_rejected_count += int(bool(lqr_diag.get("rejected", False)))
                        outer_diag = {
                            "phase": "lqr_outer_loop",
                            "controller": "fixed_x_transport_lqr",
                            "target_x_m": float(lqr_goal_x),
                            "x_now_m": float(ee_pos_arr[transport_axis_idx]),
                            "x_dot_now_mps": float(ee_lin_arr[transport_axis_idx]),
                            "a_x_cmd_mps2": float(target_axis_accel),
                            "diagnostics": lqr_diag,
                        }
                        transport_target_pos[transport_axis_idx] = float(lqr_goal_x)
                        target_axis = float(lqr_goal_x)
                    elif args.accel_x_transport and args.accel_profile == "mpc":
                        if mpc_controller is None or mpc_filter is None or mpc_goal_x is None:
                            raise RuntimeError("MPC transport was not initialized before motion start")
                        theta_now, theta_dot_now, pole_observer_source = read_pole_state_for_control(
                            q_arr=np.asarray(q, dtype=np.float64),
                            qd_arr=np.asarray(qd, dtype=np.float64),
                            time_s=sim_time,
                            dt_s=sim_dt,
                            target_x=float(mpc_goal_x),
                        )
                        mpc_pole_theta_hist.append(theta_now)
                        mpc_pole_theta_dot_hist.append(theta_dot_now)
                        target_axis_accel, mpc_diag = compute_coppelia_mpc_outer_command(
                            mpc_controller,
                            mpc_filter,
                            x_now=float(ee_pos_arr[transport_axis_idx]),
                            x_dot_now=float(ee_lin_arr[transport_axis_idx]),
                            theta_now=theta_now,
                            theta_dot_now=theta_dot_now,
                            time_s=sim_time,
                            dt_s=sim_dt,
                            target_x=float(mpc_goal_x),
                        )
                        mpc_clipped_count += int(bool(mpc_diag.get("clipped", False)))
                        mpc_rejected_count += int(bool(mpc_diag.get("rejected", False)))
                        outer_diag = {
                            "phase": "mpc_outer_loop",
                            "controller": "cartpole_mpc",
                            "target_x_m": float(mpc_goal_x),
                            "x_now_m": float(ee_pos_arr[transport_axis_idx]),
                            "x_dot_now_mps": float(ee_lin_arr[transport_axis_idx]),
                            "theta_now_rad": theta_now,
                            "theta_dot_now_radps": theta_dot_now,
                            "pole_observer_available": pole_observer_source != "unavailable",
                            "pole_observer_source": pole_observer_source,
                            "a_x_cmd_mps2": float(target_axis_accel),
                            "diagnostics": mpc_diag,
                        }
                        transport_target_pos[transport_axis_idx] = float(mpc_goal_x)
                        target_axis = float(mpc_goal_x)
                    elif args.accel_x_transport and args.accel_profile == "back_and_forth_15s":
                        half_period = max(float(args.accel_square_half_period_s), 1e-6)
                        if t_move < half_period:
                            target_axis_accel = float(args.a_x_max)
                        elif t_move < 2.0 * half_period:
                            target_axis_accel = -float(args.a_x_max)
                        else:
                            target_axis_accel = 0.0
                        if not use_ik_torque_policy:
                            accel_v_state = float(
                                np.clip(
                                    accel_v_state + target_axis_accel * sim_dt,
                                    -abs(float(args.v_x_max)),
                                    abs(float(args.v_x_max)),
                                )
                            )
                            target_axis_vel = accel_v_state
                            transport_target_pos[transport_axis_idx] += target_axis_vel * sim_dt
                    elif (
                        args.accel_x_transport
                        and args.accel_profile == "reciprocating"
                        and reciprocating_plan is not None
                    ):
                        axis_ref, axis_v_ref, axis_a_ref, reciprocating_phase, _ = (
                            reciprocating_axis_reference(t_move, reciprocating_plan)
                        )
                        if use_ik_torque_policy:
                            axis_pos = float(ee_pos_arr[transport_axis_idx]) - float(
                                transport_axis0
                            )
                            pos_err = float(axis_ref) - axis_pos
                            vel_err = float(axis_v_ref) - float(
                                ee_lin_arr[transport_axis_idx]
                            )
                            track_accel = float(
                                np.clip(
                                    30.0 * pos_err + 10.0 * vel_err,
                                    -abs(float(args.a_x_max)),
                                    abs(float(args.a_x_max)),
                                )
                            )
                            ik_axis_state = float(axis_v_ref)
                            target_axis_accel = float(
                                0.55 * float(axis_a_ref) + 0.45 * track_accel
                            )
                            transport_target_pos[transport_axis_idx] = transport_axis0 + float(
                                axis_ref
                            )
                            target_axis = float(transport_target_pos[transport_axis_idx])
                            target_axis_vel = float(axis_v_ref)
                        else:
                            slew_step = min(
                                float(ctrl_y.get("target_x_step_max_m", 0.01)),
                                0.35 * abs(float(args.v_x_max)) * step_dt + 1.0e-6,
                            )
                            reciprocating_axis_offset_slew = slew_axis_reference(
                                reciprocating_axis_offset_slew,
                                float(axis_ref),
                                dt=step_dt,
                                max_step_m=slew_step,
                                max_velocity_mps=0.85 * float(args.v_x_max),
                            )
                            transport_target_pos[transport_axis_idx] = (
                                transport_axis0 + reciprocating_axis_offset_slew
                            )
                            target_axis = float(transport_target_pos[transport_axis_idx])
                            target_axis_vel = 0.35 * float(axis_v_ref)
                            target_axis_accel = 0.0
                        accel_total_time = reciprocating_plan.motion_duration_s
                        outer_diag = {
                            "phase": reciprocating_phase,
                            "axis_offset_m": float(axis_ref),
                            "axis_velocity_mps": float(axis_v_ref),
                            "axis_accel_mps2": float(axis_a_ref),
                        }
                    else:
                        transport_target_pos[transport_axis_idx] = transport_axis0 + args.target_dx
                        target_axis = float(transport_target_pos[transport_axis_idx])
                        target_axis_vel = 0.0
                        target_axis_accel = 0.0

                target_x = float(transport_target_pos[0])
                if (
                    diag_ref_step_mid
                    and sim_time >= (float(args.duration) * 0.5)
                    and not diag_ref_step_applied
                ):
                    transport_target_pos[0] = float(ee_pos0[0]) + 0.01
                    diag_ref_step_applied = True
                    target_x = float(transport_target_pos[0])
                if torque_diagnostics is not None and diag_cfg.reference_smoothing_enabled:
                    transport_target_pos[0] = torque_diagnostics.smooth_x_reference(
                        float(transport_target_pos[0]),
                        step_dt,
                    )
                    target_x = float(transport_target_pos[0])
                target_x_vel = 0.0 if transport_axis_idx != 0 else float(target_axis_vel)
                target_x_accel = 0.0 if transport_axis_idx != 0 else float(target_axis_accel)

                state = {
                    "time": sim_time,
                    "q": q, "qd": qd,
                    "ee_pos": ee_pos, "ee_quat": ee_quat,
                    "ee_lin_vel": ee_lin, "ee_ang_vel": ee_ang,
                    "target_x": target_x,
                    "target_x_vel": target_x_vel,
                    "target_axis": float(target_axis),
                    "target_axis_vel": float(target_axis_vel),
                    "jacobian": np.vstack([j_pos, j_rot]),
                }
                if gravity_torque is not None:
                    state["gravity_torque"] = gravity_torque

                task_rot = quat_wxyz_to_rotation_matrix(np.asarray(ee_quat, dtype=np.float64))
                ee_pos_arr = np.asarray(ee_pos, dtype=np.float64)
                x_error = float(transport_target_pos[0] - ee_pos_arr[0])
                y_error = float(transport_target_pos[1] - ee_pos_arr[1])
                z_error = float(transport_target_pos[2] - ee_pos_arr[2])
                axis_error = float(
                    transport_target_pos[transport_axis_idx] - ee_pos_arr[transport_axis_idx]
                )
                fixed_axis_1_error = float(
                    transport_target_pos[orth_axis_idxs[0]] - ee_pos_arr[orth_axis_idxs[0]]
                )
                fixed_axis_2_error = float(
                    transport_target_pos[orth_axis_idxs[1]] - ee_pos_arr[orth_axis_idxs[1]]
                )
                if args.task_orientation_target == "mujoco":
                    orientation_error_norm = rotation_error_angle_rad(
                        task_rot, TARGET_SITE_ROTATION_WORLD
                    )
                else:
                    orientation_error_norm = float(
                        np.linalg.norm(orientation_error_vec_wxyz(ee_quat0, ee_quat))
                    )
                jacobian_cond = float(np.linalg.cond(np.vstack([j_pos, j_rot])))
                singular_scale = 1.0
                tau_raw_for_log = np.zeros(6, dtype=np.float64)
                tau_preclip_log = np.zeros(6, dtype=np.float64)
                tau_task_nominal_log = np.zeros(6, dtype=np.float64)
                tau_task_log = np.zeros(6, dtype=np.float64)
                tau_posture_log = np.zeros(6, dtype=np.float64)
                tau_damping_log = np.zeros(6, dtype=np.float64)
                task_backtrack_scale = 1.0
                task_scale = 1.0
                task_backtrack_iters = 0
                task_feasible = True
                filter_diag: dict[str, Any] | None = None
                collect_filter_diag = torque_diagnostics is not None
                diag_q_des: np.ndarray | None = None
                diag_qd_des: np.ndarray | None = None
                diag_tau_p = np.zeros(6, dtype=np.float64)
                diag_tau_d = np.zeros(6, dtype=np.float64)
                diag_tau_gravity = (
                    np.asarray(gravity_torque, dtype=np.float64).reshape(6)
                    if gravity_torque is not None
                    else np.zeros(6, dtype=np.float64)
                )
                if diag_passive:
                    tau_cmd = np.zeros(6, dtype=np.float64)
                    tau_raw_for_log = np.zeros(6, dtype=np.float64)
                    tau_preclip_log = np.zeros(6, dtype=np.float64)
                    tau_task_nominal_log = np.zeros(6, dtype=np.float64)
                    tau_task_log = np.zeros(6, dtype=np.float64)
                    tau_posture_log = np.zeros(6, dtype=np.float64)
                    tau_damping_log = np.zeros(6, dtype=np.float64)
                    diag_q_des = np.asarray(q, dtype=np.float64)
                    diag_qd_des = np.zeros(6, dtype=np.float64)
                elif diag_sinusoid_joint:
                    q_ref = np.asarray(q0, dtype=np.float64).copy()
                    q_ref[diag_sinusoid_joint_idx] += float(diag_cfg.sinusoid_amplitude_rad) * math.sin(
                        2.0 * math.pi * float(diag_cfg.sinusoid_frequency_hz) * sim_time
                    )
                    qdot_ref = np.zeros(6, dtype=np.float64)
                    if torque_diagnostics is not None:
                        q_ref = torque_diagnostics.smooth_joint_reference(q_ref, step_dt)
                    diag_q_des = q_ref.copy()
                    diag_qd_des = qdot_ref.copy()
                    ik_kp_scaled = float(args.ik_joint_kp) * float(diag_cfg.impedance_gain_scale)
                    ik_kd_scaled = float(args.ik_joint_kd) * float(diag_cfg.impedance_gain_scale)
                    tau_pre_filter, pd_diag = ik_joint_pd_torque(
                        q=np.asarray(q, dtype=np.float64),
                        qd=np.asarray(qd, dtype=np.float64),
                        q_ref=q_ref,
                        qdot_ref=qdot_ref,
                        tau_limit=tau_limit,
                        kp=ik_kp_scaled,
                        kd=ik_kd_scaled,
                    )
                    if gravity_torque is not None:
                        tau_pre_filter = np.asarray(tau_pre_filter, dtype=np.float64) + gravity_torque
                    tau_cmd, filter_diag = apply_torque_filter_step(
                        torque_filter,
                        tau_pre_filter,
                        step_dt,
                        collect_filter_diag=collect_filter_diag,
                    )
                    tau_raw_for_log = np.asarray(pd_diag.get("tau_raw", tau_pre_filter), dtype=np.float64)
                    tau_preclip_log = np.asarray(tau_pre_filter, dtype=np.float64)
                    tau_task_nominal_log = np.zeros(6, dtype=np.float64)
                    tau_task_log = np.zeros(6, dtype=np.float64)
                    tau_posture_log = tau_raw_for_log
                    tau_damping_log = np.zeros(6, dtype=np.float64)
                    diag_tau_p = np.asarray(pd_diag.get("tau_impedance_P", np.zeros(6)), dtype=np.float64)
                    diag_tau_d = np.asarray(pd_diag.get("tau_impedance_D", np.zeros(6)), dtype=np.float64)
                if use_ik_torque_policy and not diag_passive and not diag_sinusoid_joint:
                    ik_kp = (
                        float(lqr_ik_joint_kp) if use_lqr_outer_policy else float(args.ik_joint_kp)
                    )
                    ik_kd = (
                        float(lqr_ik_joint_kd) if use_lqr_outer_policy else float(args.ik_joint_kd)
                    )
                    ik_tau_limit = (
                        np.asarray(tau_limit, dtype=np.float64) * float(lqr_joint_torque_limit_scale)
                        if use_lqr_outer_policy
                        else np.asarray(tau_limit, dtype=np.float64)
                    )
                    qdot_ref = np.zeros(6, dtype=np.float64)
                    fixed_position_ref = (
                        np.asarray(transport_target_pos, dtype=np.float64).copy()
                        if use_lqr_outer_policy
                        else np.asarray(ee_pos0, dtype=np.float64)
                    )
                    if sim_time >= args.settle_duration:
                        if (
                            use_point_to_point_ik_policy
                            or use_reciprocating_ik_policy
                            or use_fast_x_ik_policy
                        ):
                            ik_ctrl_ref, ik_diag = acceleration_transport_controller(
                                q=np.asarray(q, dtype=np.float64),
                                qvel=np.asarray(qd, dtype=np.float64),
                                ctrl_prev=ik_ctrl_ref,
                                ctrl_lower=safety.cfg.q_lower,
                                ctrl_upper=safety.cfg.q_upper,
                                tool_pos=np.asarray(ee_pos, dtype=np.float64),
                                tool_rot=task_rot,
                                tool_jacobian_pos=np.asarray(j_pos, dtype=np.float64),
                                tool_jacobian_rot=np.asarray(j_rot, dtype=np.float64),
                                a_axis_cmd=float(target_axis_accel),
                                axis_state=float(ik_axis_state),
                                transport_axis=args.transport_axis,
                                fixed_position=fixed_position_ref,
                                target_tool_rot=orientation_target_rot,
                                dt=sim_dt,
                                a_axis_max_m_s2=float(
                                    fast_transport_a_max
                                    if use_fast_x_profile
                                    else args.a_x_max
                                ),
                                v_axis_max_m_s=float(
                                    fast_transport_v_max
                                    if use_fast_x_profile
                                    else args.v_x_max
                                ),
                                torque_headroom=(
                                    float(lqr_ik_torque_headroom)
                                    if use_lqr_outer_policy
                                    else float(args.ik_torque_headroom)
                                ),
                                joint_speed_limit_scale=(
                                    float(lqr_joint_speed_limit_scale)
                                    if use_lqr_outer_policy
                                    else (
                                        float(fast_x_joint_speed_fraction)
                                        if use_fast_x_profile
                                        else 1.0
                                    )
                                ),
                                move_axis_weight=(
                                    float(lqr_move_axis_weight)
                                    if use_lqr_outer_policy
                                    else float(
                                        reciprocating_ik_weights.get("move_axis_weight", 120.0)
                                    )
                                ),
                                hold_axis_weight=(
                                    float(lqr_hold_axis_weight)
                                    if use_lqr_outer_policy
                                    else float(
                                        reciprocating_ik_weights.get("hold_axis_weight", 160.0)
                                    )
                                ),
                                orientation_weight=(
                                    float(lqr_orientation_weight)
                                    if use_lqr_outer_policy
                                    else float(
                                        reciprocating_ik_weights.get("orientation_weight", 96.0)
                                    )
                                ),
                                hold_axis_gain=(
                                    float(lqr_hold_axis_gain)
                                    if use_lqr_outer_policy
                                    else float(
                                        reciprocating_ik_weights.get("hold_axis_gain", 12.0)
                                    )
                                ),
                                posture_target=q0,
                            )
                            target_axis_vel = float(
                                ik_diag.get("v_x_realized_cmd", float(target_axis_vel))
                            )
                            ik_axis_state = float(
                                ik_diag.get("v_x_state_next", float(ik_axis_state))
                            )
                            if "q_dot_des_red" in ik_diag:
                                qdot_ref[1:] = np.asarray(
                                    ik_diag["q_dot_des_red"], dtype=np.float64
                                ).reshape(5)
                        else:
                            ik_ctrl_ref, ik_diag = acceleration_transport_controller(
                                q=np.asarray(q, dtype=np.float64),
                                qvel=np.asarray(qd, dtype=np.float64),
                                ctrl_prev=ik_ctrl_ref,
                                ctrl_lower=safety.cfg.q_lower,
                                ctrl_upper=safety.cfg.q_upper,
                                tool_pos=np.asarray(ee_pos, dtype=np.float64),
                                tool_rot=task_rot,
                                tool_jacobian_pos=np.asarray(j_pos, dtype=np.float64),
                                tool_jacobian_rot=np.asarray(j_rot, dtype=np.float64),
                                a_axis_cmd=float(target_axis_accel),
                                axis_state=float(ik_axis_state),
                                transport_axis=args.transport_axis,
                                fixed_position=fixed_position_ref,
                                target_tool_rot=orientation_target_rot,
                                posture_target=q0,
                                dt=sim_dt,
                                a_axis_max_m_s2=float(args.a_x_max),
                                v_axis_max_m_s=float(args.v_x_max),
                                torque_headroom=(
                                    float(lqr_ik_torque_headroom)
                                    if use_lqr_outer_policy
                                    else float(args.ik_torque_headroom)
                                ),
                                joint_speed_limit_scale=(
                                    float(lqr_joint_speed_limit_scale)
                                    if use_lqr_outer_policy
                                    else 1.0
                                ),
                                move_axis_weight=float(
                                    reciprocating_ik_weights.get("move_axis_weight", 120.0)
                                ),
                                hold_axis_weight=float(
                                    reciprocating_ik_weights.get("hold_axis_weight", 100.0)
                                ),
                                orientation_weight=float(
                                    reciprocating_ik_weights.get("orientation_weight", 64.0)
                                ),
                                hold_axis_gain=float(
                                    reciprocating_ik_weights.get("hold_axis_gain", 8.0)
                                ),
                            )
                            ik_axis_state = float(
                                ik_diag.get("v_x_state_next", float(ik_axis_state))
                            )
                            target_axis_vel = float(
                                ik_diag.get("v_x_realized_cmd", float(target_axis_vel))
                            )
                            if "q_dot_des_red" in ik_diag:
                                qdot_ref[1:] = np.asarray(
                                    ik_diag["q_dot_des_red"], dtype=np.float64
                                ).reshape(5)
                        transport_target_pos[transport_axis_idx] += target_axis_vel * sim_dt
                        target_axis = float(transport_target_pos[transport_axis_idx])
                pre_motion_hold = bool(
                    diag_force_hold_pose
                    or (args.accel_x_transport and sim_time < args.settle_duration)
                )
                if not diag_passive and not diag_sinusoid_joint and pre_motion_hold:
                    if use_ik_torque_policy:
                        qdot_ref = np.zeros(6, dtype=np.float64)
                        tau_pre_filter, pd_diag = ik_joint_pd_torque(
                            q=np.asarray(q, dtype=np.float64),
                            qd=np.asarray(qd, dtype=np.float64),
                            q_ref=np.asarray(q0, dtype=np.float64),
                            qdot_ref=qdot_ref,
                            tau_limit=ik_tau_limit,
                            kp=ik_kp,
                            kd=ik_kd,
                        )
                        if gravity_torque is not None:
                            tau_pre_filter = np.asarray(tau_pre_filter, dtype=np.float64) + gravity_torque
                        tau_cmd, filter_diag = apply_torque_filter_step(
                            torque_filter,
                            tau_pre_filter,
                            step_dt,
                            collect_filter_diag=collect_filter_diag,
                        )
                        tau_raw_for_log = np.asarray(pd_diag.get("tau_raw", tau_pre_filter), dtype=np.float64)
                        tau_preclip_log = np.asarray(tau_pre_filter, dtype=np.float64)
                        tau_task_nominal_log = np.zeros(6, dtype=np.float64)
                        tau_task_log = np.zeros(6, dtype=np.float64)
                        tau_posture_log = tau_raw_for_log
                        tau_damping_log = np.zeros(6, dtype=np.float64)
                        diag_q_des = np.asarray(q0, dtype=np.float64)
                        diag_qd_des = qdot_ref.copy()
                        diag_tau_p = np.asarray(pd_diag.get("tau_impedance_P", np.zeros(6)), dtype=np.float64)
                        diag_tau_d = np.asarray(pd_diag.get("tau_impedance_D", np.zeros(6)), dtype=np.float64)
                        task_backtrack_scale = 1.0
                        task_scale = 1.0
                        task_backtrack_iters = 0
                        task_feasible = True
                        x_error = 0.0
                        y_error = 0.0
                        z_error = 0.0
                        orientation_error_norm = 0.0
                        jacobian_cond = float(np.linalg.cond(np.vstack([j_pos, j_rot])))
                        singular_scale = 1.0
                    else:
                        state["hold_current_pose"] = True
                        out = controller.compute(state)
                        tau_cmd, filter_diag = apply_torque_filter_step(
                            torque_filter,
                            out.tau,
                            step_dt,
                            collect_filter_diag=collect_filter_diag,
                        )
                        x_error = float(out.x_error)
                        y_error = float(out.y_error)
                        z_error = float(out.z_error)
                        orientation_error_norm = float(out.orientation_error_norm)
                        jacobian_cond = float(out.jacobian_cond)
                        singular_scale = float(out.singular_scale)
                        tau_raw_for_log = np.asarray(out.tau, dtype=np.float64)
                        tau_preclip_log = np.asarray(out.tau_preclip, dtype=np.float64)
                        tau_task_nominal_log = np.asarray(out.tau_task_nominal, dtype=np.float64)
                        tau_task_log = out.tau_task
                        tau_posture_log = out.tau_posture
                        tau_damping_log = out.tau_damping
                        diag_tau_p = np.asarray(out.tau_task + out.tau_posture, dtype=np.float64)
                        diag_tau_d = np.asarray(out.tau_damping, dtype=np.float64)
                        diag_q_des = np.asarray(controller._q_rest, dtype=np.float64)
                        diag_qd_des = np.zeros(6, dtype=np.float64)
                        task_backtrack_scale = float(out.task_backtrack_scale)
                        task_scale = float(out.task_scale)
                        task_backtrack_iters = int(out.task_backtrack_iters)
                        task_feasible = bool(out.task_feasible)
                        pd_diag = {}
                elif not diag_passive and not diag_sinusoid_joint and use_ik_torque_policy:
                    tau_pre_filter, pd_diag = ik_joint_pd_torque(
                        q=np.asarray(q, dtype=np.float64),
                        qd=np.asarray(qd, dtype=np.float64),
                        q_ref=ik_ctrl_ref,
                        qdot_ref=qdot_ref,
                        tau_limit=ik_tau_limit,
                        kp=ik_kp,
                        kd=ik_kd,
                    )
                    if gravity_torque is not None:
                        tau_pre_filter = np.asarray(tau_pre_filter, dtype=np.float64) + gravity_torque
                    tau_cmd, filter_diag = apply_torque_filter_step(
                        torque_filter,
                        tau_pre_filter,
                        step_dt,
                        collect_filter_diag=collect_filter_diag,
                    )
                    tau_raw_for_log = np.asarray(pd_diag.get("tau_raw", tau_pre_filter), dtype=np.float64)
                    tau_posture_log = tau_raw_for_log
                    diag_q_des = np.asarray(ik_ctrl_ref, dtype=np.float64)
                    diag_qd_des = np.asarray(qdot_ref, dtype=np.float64)
                    diag_tau_p = np.asarray(pd_diag.get("tau_impedance_P", np.zeros(6)), dtype=np.float64)
                    diag_tau_d = np.asarray(pd_diag.get("tau_impedance_D", np.zeros(6)), dtype=np.float64)
                elif not diag_passive and not diag_sinusoid_joint:
                    out = controller.compute(state)
                    tau_cmd, filter_diag = apply_torque_filter_step(
                        torque_filter,
                        out.tau,
                        step_dt,
                        collect_filter_diag=collect_filter_diag,
                    )
                    x_error = float(out.x_error)
                    y_error = float(out.y_error)
                    z_error = float(out.z_error)
                    orientation_error_norm = float(out.orientation_error_norm)
                    jacobian_cond = float(out.jacobian_cond)
                    singular_scale = float(out.singular_scale)
                    tau_raw_for_log = np.asarray(out.tau, dtype=np.float64)
                    tau_preclip_log = np.asarray(out.tau_preclip, dtype=np.float64)
                    tau_task_nominal_log = np.asarray(out.tau_task_nominal, dtype=np.float64)
                    tau_task_log = out.tau_task
                    tau_posture_log = out.tau_posture
                    tau_damping_log = out.tau_damping
                    diag_tau_p = np.asarray(out.tau_task + out.tau_posture, dtype=np.float64)
                    diag_tau_d = np.asarray(out.tau_damping, dtype=np.float64)
                    diag_q_des = np.asarray(controller._q_rest, dtype=np.float64)
                    diag_qd_des = np.zeros(6, dtype=np.float64)
                    task_backtrack_scale = float(out.task_backtrack_scale)
                    task_scale = float(out.task_scale)
                    task_backtrack_iters = int(out.task_backtrack_iters)
                    task_feasible = bool(out.task_feasible)

                safe_st = None
                tau_cmd_to_apply = tau_cmd
                should_break = False
                if sim_time >= args.settle_duration:
                    safe_st = safety.check(
                        state,
                        axis_error=axis_error,
                        x_error=x_error,
                        orientation_error_norm=orientation_error_norm,
                    )
                    if not safe_st.ok:
                        print(f"SAFETY STOP: {safe_st.reason}")
                        safety_stop_reason = safe_st.reason
                        tau_cmd_to_apply = np.zeros(6, dtype=np.float64)
                        should_break = True

                adapter.apply_torque(tau_cmd_to_apply)
                if step == 0:
                    mark("first torque command applied")

                if torque_diagnostics is not None:
                    pole_theta_log = None
                    pole_theta_dot_log = None
                    if (
                        coppelia_pendulum_handles is not None
                        and coppelia_pendulum_handles.available
                    ):
                        pole_read = read_coppelia_pendulum_state(
                            sim,
                            coppelia_pendulum_handles,
                            parent_handle=int(task_frame_summary.get("parent_handle", -1)),
                        )
                        if pole_read is not None:
                            pole_theta_log, pole_theta_dot_log = pole_read
                    joint_guard = bool(
                        safe_st is not None
                        and not safe_st.ok
                        and "joint limit" in str(safe_st.reason).lower()
                    )
                    workspace_guard = bool(
                        safe_st is not None
                        and not safe_st.ok
                        and (
                            "drift" in str(safe_st.reason).lower()
                            or "orientation" in str(safe_st.reason).lower()
                        )
                    )
                    diag_row = torque_diagnostics.record_step(
                        step_idx=step,
                        timestamp=sim_time,
                        dt=step_dt,
                        q=np.asarray(q, dtype=np.float64),
                        qd=np.asarray(qd, dtype=np.float64),
                        q_des=diag_q_des,
                        qd_des=diag_qd_des,
                        cart_position=None,
                        cart_velocity=None,
                        pole_angle=pole_theta_log,
                        pole_angular_velocity=pole_theta_dot_log,
                        tau_impedance_p=diag_tau_p,
                        tau_impedance_d=diag_tau_d,
                        tau_feedforward=None,
                        tau_gravity=diag_tau_gravity,
                        tau_raw_before_safety=tau_raw_for_log,
                        tau_after_saturation=tau_preclip_log,
                        tau_final_sent=tau_cmd_to_apply,
                        filter_diag=filter_diag,
                        torque_api_modes=adapter.last_torque_api_modes(),
                        safety_flags={
                            "joint_limit_guardrail": joint_guard,
                            "workspace_guardrail": workspace_guard,
                        },
                        joint_mode_snapshot=joint_cfg.get("joints"),
                    )

                trace_row = {
                    "time": sim_time,
                    "q": q, "qd": qd,
                    "ee_pos": ee_pos, "ee_quat": ee_quat,
                    "ee_lin_vel": ee_lin, "ee_ang_vel": ee_ang,
                    "target_x": target_x,
                    "target_x_vel": target_x_vel,
                    "target_x_accel": target_x_accel,
                    "target_axis": target_axis,
                    "target_axis_vel": target_axis_vel,
                    "target_axis_accel": target_axis_accel,
                    "x_error": x_error,
                    "y_error": y_error,
                    "z_error": z_error,
                    "axis_error": axis_error,
                    "fixed_axis_1_error": fixed_axis_1_error,
                    "fixed_axis_2_error": fixed_axis_2_error,
                    "orientation_error_norm": orientation_error_norm,
                    "tau_task": tau_task_log,
                    "tau_posture": tau_posture_log,
                    "tau_damping": tau_damping_log,
                    "tau_preclip": tau_preclip_log,
                    "tau_task_nominal": tau_task_nominal_log,
                    "task_backtrack_scale": task_backtrack_scale,
                    "task_scale": task_scale,
                    "task_backtrack_iters": task_backtrack_iters,
                    "task_feasible": task_feasible,
                    "tau_raw": tau_raw_for_log,
                    "tau_cmd": tau_cmd,
                    "jacobian_condition_number": jacobian_cond,
                    "singular_scale": singular_scale,
                    "outer_transport_diagnostics": outer_diag,
                    "accel_torque_policy": args.accel_torque_policy,
                    "transport_axis": transport_axis_name,
                    "ik_diagnostics": ik_diag,
                    "ik_pd_diagnostics": pd_diag,
                    "safety_ok": None if safe_st is None else safe_st.ok,
                }
                if torque_diagnostics is not None:
                    trace_row.update(diag_row)
                trace.write_row(trace_row)

                if should_break:
                    break

                x_err_hist.append(float(x_error))
                y_err_hist.append(float(y_error))
                z_err_hist.append(float(z_error))
                ori_err_hist.append(float(orientation_error_norm))
                axis_error_hist.append(float(axis_error))
                fixed_axis_1_error_hist.append(float(fixed_axis_1_error))
                fixed_axis_2_error_hist.append(float(fixed_axis_2_error))
                x_hist.append(float(np.asarray(ee_pos, dtype=np.float64)[0]))
                y_hist.append(float(np.asarray(ee_pos, dtype=np.float64)[1]))
                z_hist.append(float(np.asarray(ee_pos, dtype=np.float64)[2]))
                ee_lin_arr = np.asarray(ee_lin, dtype=np.float64).reshape(3)
                qd_arr = np.asarray(qd, dtype=np.float64).reshape(6)
                axis_v_hist.append(float(ee_lin_arr[transport_axis_idx]))
                ee_vx_hist.append(float(ee_lin_arr[0]))
                ee_speed_hist.append(float(np.linalg.norm(ee_lin_arr)))
                qd_speed_hist.append(float(np.max(np.abs(qd_arr))))
                target_x_hist.append(float(target_x))
                target_vx_hist.append(float(target_x_vel))
                target_ax_hist.append(float(target_x_accel))
                target_axis_hist.append(float(target_axis))
                target_axis_vel_hist.append(float(target_axis_vel))
                target_axis_acc_hist.append(float(target_axis_accel))
                tau_hist.append(np.asarray(tau_cmd, dtype=np.float64))
                tau_raw_hist.append(np.asarray(tau_raw_for_log, dtype=np.float64))
                task_backtrack_scale_hist.append(float(task_backtrack_scale))
                task_scale_hist.append(float(task_scale))
                task_backtrack_iters_hist.append(int(task_backtrack_iters))
                task_feasible_hist.append(bool(task_feasible))
                if ik_diag:
                    ik_speed_scale_hist.append(float(ik_diag.get("speed_scale", 1.0)))
                    ik_torque_scale_hist.append(float(ik_diag.get("torque_scale", 1.0)))
                    y_leak_hist.append(
                        float(
                            ik_diag.get(
                                "predicted_orthogonal_velocity_mps_for_unit_cmd",
                                [0.0, 0.0],
                            )[0]
                        )
                    )
                ik_q_ref_excursion_hist.append(float(np.max(np.abs(ik_ctrl_ref - q0))))
                q_excursion_hist.append(float(np.max(np.abs(np.asarray(q, dtype=np.float64) - q0))))

                prev_sim_time = sim_time
                sim.step()

                if (
                    not args.no_video
                    and vision_sensor is not None
                    and (step % frame_every == 0 or step == steps - 1)
                ):
                    frame_rgb, _ = read_vision_rgb24(sim, vision_sensor)
                    frames.append(frame_rgb)
    finally:
        adapter.apply_torque(np.zeros(6, dtype=np.float64))
        sim.stopSimulation()
        request_controller_shutdown(sim, client)

    # ---------- 9. Write outputs ----------
    transport_axis_pos_hist = (
        x_hist if transport_axis_idx == 0 else y_hist if transport_axis_idx == 1 else z_hist
    )
    x_span = float(max(x_hist) - min(x_hist)) if x_hist else None
    x_net = float(x_hist[-1] - x_hist[0]) if len(x_hist) >= 2 else None
    transport_axis_span = (
        float(max(transport_axis_pos_hist) - min(transport_axis_pos_hist))
        if transport_axis_pos_hist
        else None
    )
    transport_axis_net = (
        float(transport_axis_pos_hist[-1] - transport_axis_pos_hist[0])
        if len(transport_axis_pos_hist) >= 2
        else None
    )
    max_abs_y_drift = float(max(abs(y - float(ee_pos0[1])) for y in y_hist)) if y_hist else None
    max_abs_z_drift = float(max(abs(z - float(ee_pos0[2])) for z in z_hist)) if z_hist else None
    max_abs_transport_axis_drift = (
        float(max(abs(p - float(transport_axis0)) for p in transport_axis_pos_hist))
        if transport_axis_pos_hist
        else None
    )
    max_abs_fixed_axis_1_drift = (
        float(max(abs(e) for e in fixed_axis_1_error_hist)) if fixed_axis_1_error_hist else None
    )
    max_abs_fixed_axis_2_drift = (
        float(max(abs(e) for e in fixed_axis_2_error_hist)) if fixed_axis_2_error_hist else None
    )
    max_orientation_error_rad = float(max(ori_err_hist)) if ori_err_hist else None
    final_orientation_error_deg = (
        math.degrees(float(ori_err_hist[-1])) if ori_err_hist else None
    )
    max_orientation_error_deg = (
        math.degrees(float(max_orientation_error_rad))
        if max_orientation_error_rad is not None
        else None
    )
    peak_abs_ee_vx = float(max(abs(v) for v in ee_vx_hist)) if ee_vx_hist else None
    peak_abs_transport_axis_v = (
        float(max(abs(v) for v in axis_v_hist)) if axis_v_hist else None
    )
    peak_ee_speed = float(max(ee_speed_hist)) if ee_speed_hist else None
    peak_joint_speed = float(max(qd_speed_hist)) if qd_speed_hist else None
    peak_target_vx = float(max(abs(v) for v in target_vx_hist)) if target_vx_hist else None
    peak_target_ax = float(max(abs(a) for a in target_ax_hist)) if target_ax_hist else None
    peak_target_axis_v = (
        float(max(abs(v) for v in target_axis_vel_hist)) if target_axis_vel_hist else None
    )
    peak_target_axis_a = (
        float(max(abs(a) for a in target_axis_acc_hist)) if target_axis_acc_hist else None
    )
    max_abs_tau = float(max(np.max(np.abs(t)) for t in tau_hist)) if tau_hist else None
    max_abs_tau_raw = (
        float(max(np.max(np.abs(t)) for t in tau_raw_hist)) if tau_raw_hist else None
    )
    tau_saturation_fraction = None
    if tau_raw_hist:
        raw_stack = np.vstack(tau_raw_hist)
        tau_saturation_fraction = float(
            np.mean(np.abs(raw_stack) > tau_limit.reshape(1, 6) + 1.0e-9)
        )
    max_q_excursion = float(max(q_excursion_hist)) if q_excursion_hist else None
    max_q_ref_excursion = float(max(ik_q_ref_excursion_hist)) if ik_q_ref_excursion_hist else None
    min_ik_speed_scale = float(min(ik_speed_scale_hist)) if ik_speed_scale_hist else None
    min_ik_torque_scale = float(min(ik_torque_scale_hist)) if ik_torque_scale_hist else None
    min_task_backtrack_scale = float(min(task_backtrack_scale_hist)) if task_backtrack_scale_hist else None
    min_task_scale = float(min(task_scale_hist)) if task_scale_hist else None
    max_task_backtrack_iters = int(max(task_backtrack_iters_hist)) if task_backtrack_iters_hist else None
    all_task_feasible = bool(all(task_feasible_hist)) if task_feasible_hist else None
    final_target_axis = (
        float(target_axis_hist[-1] - transport_axis0) if target_axis_hist else None
    )
    final_axis_tracking_error = (
        float(transport_axis_pos_hist[-1] - target_axis_hist[-1])
        if transport_axis_pos_hist and target_axis_hist
        else None
    )
    requested_axis_displacement = float(args.target_dx)
    if args.accel_x_transport and args.accel_profile == "reciprocating":
        expected_axis_displacement = 0.0
        expected_axis_span = 2.0 * abs(float(args.reciprocating_stroke_m))
    else:
        expected_axis_displacement = (
            final_target_axis if final_target_axis is not None else requested_axis_displacement
        )
        expected_axis_span = None
    direction_ok = None
    if transport_axis_net is not None and abs(expected_axis_displacement) > 1.0e-9:
        direction_ok = bool(
            math.copysign(1.0, transport_axis_net)
            == math.copysign(1.0, expected_axis_displacement)
        )
    transport_axis_tracking_ok = None
    fixed_axes_ok = None
    orientation_ok = None
    joint_configuration_ok = None
    torque_saturation_ok = None
    frame_reference_ok = None
    failure_reasons: list[str] = []
    transport_success: bool | None = None
    if args.accel_x_transport:
        if args.accel_profile == "reciprocating":
            origin_tol = max(0.012, 0.35 * abs(float(args.reciprocating_stroke_m)))
            tracking_tol = origin_tol
            transport_axis_tracking_ok = bool(
                transport_axis_span is not None
                and transport_axis_net is not None
                and final_axis_tracking_error is not None
                and transport_axis_span >= max(expected_axis_span * 0.35, 0.008)
                and abs(transport_axis_net) <= max(origin_tol, 0.02)
                and abs(final_axis_tracking_error) <= max(tracking_tol, 0.015)
            )
            direction_ok = None
        else:
            tracking_tol = max(0.01, 0.35 * abs(expected_axis_displacement))
            transport_axis_tracking_ok = bool(
                transport_axis_span is not None
                and transport_axis_net is not None
                and final_axis_tracking_error is not None
                and abs(final_axis_tracking_error) <= tracking_tol
                and abs(transport_axis_net) >= min(abs(expected_axis_displacement) * 0.5, 0.005)
                and (direction_ok is not False)
            )
        fixed_axes_ok = bool(
            max_abs_fixed_axis_1_drift is not None
            and max_abs_fixed_axis_2_drift is not None
            and (
                (
                    max_abs_fixed_axis_1_drift <= 0.08
                    and max_abs_fixed_axis_2_drift <= 0.35
                )
                if args.accel_profile == "reciprocating"
                else (
                    max_abs_fixed_axis_1_drift <= 5.0e-3
                    and max_abs_fixed_axis_2_drift <= 5.0e-3
                )
            )
        )
        orientation_ok = bool(
            max_orientation_error_rad is not None
            and max_orientation_error_rad <= math.radians(3.0)
        )
        joint_configuration_ok = bool(
            max_q_excursion is not None
            and max_q_ref_excursion is not None
            and max_q_excursion <= float(args.max_joint_excursion_rad)
            and max_q_ref_excursion <= float(args.max_joint_excursion_rad)
        )
        torque_saturation_ok = bool(
            tau_saturation_fraction is not None and tau_saturation_fraction <= 0.25
        )
        frame_reference_ok = bool(
            task_frame_summary.get("mode") in ("ee_object", "mujoco_attachment_dummy")
            and local_axis_capability.get("finite", False)
            and local_axis_capability.get("task_rank", 0) >= 5
        )
        checks = {
            "safety": safety_stop_reason is None,
            "frame_reference": frame_reference_ok,
            "transport_axis_tracking": transport_axis_tracking_ok,
            "fixed_axes": fixed_axes_ok,
            "orientation": orientation_ok,
            "joint_configuration": joint_configuration_ok,
            "torque_saturation": torque_saturation_ok,
        }
        failure_reasons = [name for name, ok in checks.items() if not ok]
        transport_success = bool(not failure_reasons)
    transport_axis_ok = bool(transport_axis_tracking_ok and fixed_axes_ok)
    final_target_dx = final_target_axis
    final_x_tracking_error = (
        final_axis_tracking_error if transport_axis_idx == 0 else None
    )
    x_tracking_ok = transport_axis_tracking_ok if transport_axis_idx == 0 else None
    single_axis_y_ok = transport_axis_tracking_ok if transport_axis_idx == 1 else None
    fixed_z_ok = transport_axis_tracking_ok if transport_axis_idx == 2 else None
    summary = {
        "probe_only": False,
        "mode": (
            f"accel_{transport_axis_name}_transport_lqr"
            if args.accel_x_transport and args.accel_profile == "lqr"
            else f"accel_{transport_axis_name}_transport"
            if args.accel_x_transport
            else "x_impedance_step"
        ),
        "controller_name": (
            "coppeliasim_lqr_acceleration_transport_controller"
            if args.accel_x_transport and args.accel_profile == "lqr"
            else "coppeliasim_torque_acceleration_transport_controller"
            if args.accel_x_transport
            else "coppeliasim_x_axis_cartesian_impedance_controller"
        ),
        "controller_family": ownership_metadata["controller_family"],
        "uses_direct_torque_control": ownership_metadata["uses_direct_torque_control"],
        "uses_position_servo_setpoints": False,
        "stepping_owner": ownership_metadata["stepping_owner"],
        "simulation_started_by": ownership_metadata["simulation_started_by"],
        "lua_motion_enabled": ownership_metadata["lua_motion_enabled"],
        "legacy_marker_handoff": ownership_metadata["legacy_marker_handoff"],
        "use_gravity_compensation": use_gravity_compensation,
        "accel_profile": str(args.accel_profile) if args.accel_x_transport else None,
        "reciprocating_stroke_m": (
            float(args.reciprocating_stroke_m)
            if args.accel_x_transport and args.accel_profile == "reciprocating"
            else None
        ),
        "reciprocating_hold_s": (
            float(args.reciprocating_hold_s)
            if args.accel_x_transport and args.accel_profile == "reciprocating"
            else None
        ),
        "reciprocating_motion_duration_s": (
            float(reciprocating_plan.motion_duration_s)
            if reciprocating_plan is not None
            else None
        ),
        "expected_axis_span_m": expected_axis_span,
        "outer_transport_controller": (
            "fixed_x_transport_lqr"
            if args.accel_x_transport and args.accel_profile == "lqr"
            else "cartpole_mpc"
            if args.accel_x_transport and args.accel_profile == "mpc"
            else None
        ),
        "ik_transport_solver": (
            "acceleration_transport_controller"
            if args.accel_x_transport and args.accel_torque_policy == "ik_joint_pd"
            else None
        ),
        "a_x_max_m_s2": (
            float(fast_transport_a_max)
            if args.accel_x_transport and args.accel_profile == "fast_x"
            else float(args.a_x_max)
            if args.accel_x_transport
            else None
        ),
        "v_x_max_m_s": (
            float(fast_transport_v_max)
            if args.accel_x_transport and args.accel_profile == "fast_x"
            else float(args.v_x_max)
            if args.accel_x_transport
            else None
        ),
        "a_axis_max_m_s2": (
            float(fast_transport_a_max)
            if args.accel_x_transport and args.accel_profile == "fast_x"
            else float(args.a_x_max)
            if args.accel_x_transport
            else None
        ),
        "v_axis_max_m_s": (
            float(fast_transport_v_max)
            if args.accel_x_transport and args.accel_profile == "fast_x"
            else float(args.v_x_max)
            if args.accel_x_transport
            else None
        ),
        "fast_x_limits": (
            fast_x_limits_diag
            if args.accel_x_transport and args.accel_profile == "fast_x"
            else None
        ),
        "coppelia_pendulum_spawned": bool(
            coppelia_pendulum_handles is not None and coppelia_pendulum_handles.available
        ),
        "transport_axis": transport_axis_name,
        "transport_axis_index": transport_axis_idx,
        "transport_axis_world": f"world_{transport_axis_name}",
        "fixed_axis_indices": list(orth_axis_idxs),
        "fixed_axis_names": [axis_index_to_name(i) for i in orth_axis_idxs],
        "torque_headroom": float(imp_cfg.torque_headroom),
        "lqr_target_x_m": float(lqr_goal_x) if lqr_goal_x is not None else None,
        "lqr_gain_matrix": lqr_gain_matrix,
        "lqr_riccati_converged": lqr_riccati_converged,
        "lqr_riccati_iters": lqr_riccati_iters,
        "lqr_clipped_count": int(lqr_clipped_count) if args.accel_x_transport and args.accel_profile == "lqr" else None,
        "lqr_rejected_count": int(lqr_rejected_count) if args.accel_x_transport and args.accel_profile == "lqr" else None,
        "mpc_horizon": int(args.mpc_horizon) if args.accel_x_transport and args.accel_profile == "mpc" else None,
        "mpc_target_x_m": float(mpc_goal_x) if mpc_goal_x is not None else None,
        "mpc_pole_length_m": float(args.mpc_pole_length_m) if args.accel_x_transport and args.accel_profile == "mpc" else None,
        "mpc_clipped_count": int(mpc_clipped_count) if args.accel_x_transport and args.accel_profile == "mpc" else None,
        "mpc_rejected_count": int(mpc_rejected_count) if args.accel_x_transport and args.accel_profile == "mpc" else None,
        "mpc_max_abs_theta_rad": (
            float(max(abs(v) for v in mpc_pole_theta_hist)) if mpc_pole_theta_hist else None
        ),
        "mpc_pole_observer_has_pendulum": (
            bool(mujoco_cartpole_observer is not None and mujoco_cartpole_observer.has_pendulum)
            if use_mpc_outer_policy
            else None
        ),
        "lqr_ik_joint_kp": float(lqr_ik_joint_kp) if use_lqr_outer_policy else None,
        "lqr_ik_joint_kd": float(lqr_ik_joint_kd) if use_lqr_outer_policy else None,
        "lqr_ik_torque_headroom": float(lqr_ik_torque_headroom) if use_lqr_outer_policy else None,
        "lqr_joint_torque_limit_scale": (
            float(lqr_joint_torque_limit_scale) if use_lqr_outer_policy else None
        ),
        "lqr_joint_speed_limit_scale": (
            float(lqr_joint_speed_limit_scale) if use_lqr_outer_policy else None
        ),
        "lqr_cartesian_impedance_gain_scale": (
            float(lqr_cartesian_impedance_gain_scale) if use_lqr_outer_policy else None
        ),
        "lqr_cartesian_impedance_torque_headroom": (
            float(lqr_cartesian_impedance_torque_headroom) if use_lqr_outer_policy else None
        ),
        "lqr_settle_zero_torque": bool(use_lqr_outer_policy) if args.accel_x_transport else None,
        "requested_lqr_sim_time_step_s": (
            float(lqr_requested_sim_time_step) if use_lqr_outer_policy else None
        ),
        "actual_lqr_sim_time_step_s": (
            float(actual_lqr_sim_time_step)
            if use_lqr_outer_policy and actual_lqr_sim_time_step is not None
            else None
        ),
        "planned_accel_total_time_s": accel_total_time,
        "scene_path": str(scene_path),
        "ur5_model_path": str(ur5_model),
        "config_path": str(args.config),
        "host": args.host, "port": args.port,
        "startup_phases": phases,
        "joint_configuration": joint_cfg,
        "task_frame": task_frame_summary,
        "frame_reference": frame_reference_summary,
        "local_axis_transport_capability": local_axis_capability,
        "startup_jacobian_comparison": jacobian_comparison,
        "duration_s": args.duration,
        "settle_duration_s": args.settle_duration,
        "warmup_steps_requested": int(args.warmup_steps),
        "warmup_steps_effective": int(effective_warmup_steps),
        "target_dx_m": args.target_dx,
        "final_target_dx_m": final_target_dx,
        "final_x_tracking_error_m": final_x_tracking_error,
        "final_transport_axis_displacement_m": final_target_dx,
        "final_transport_axis_error_m": final_axis_tracking_error,
        "direction_ok": direction_ok,
        "accel_torque_policy": str(args.accel_torque_policy) if args.accel_x_transport else None,
        "task_orientation_target": str(args.task_orientation_target),
        "ik_joint_kp": float(args.ik_joint_kp) if args.accel_x_transport else None,
        "ik_joint_kd": float(args.ik_joint_kd) if args.accel_x_transport else None,
        "ik_torque_headroom": float(args.ik_torque_headroom) if args.accel_x_transport else None,
        "torque_mode_verified": bool(joint_cfg_verified),
        "torque_mode_readback_available": bool(joint_cfg_readback_available),
        "initial_ee_world_m": ee_pos0.tolist(),
        "final_ee_world_m": [x_hist[-1], y_hist[-1], z_hist[-1]] if x_hist else None,
        "x_span_m": x_span,
        "x_net_displacement_m": x_net,
        "transport_axis_span_m": transport_axis_span,
        "transport_axis_net_displacement_m": transport_axis_net,
        "max_abs_transport_axis_drift_m": max_abs_transport_axis_drift,
        "max_abs_fixed_axis_1_drift_m": max_abs_fixed_axis_1_drift,
        "max_abs_fixed_axis_2_drift_m": max_abs_fixed_axis_2_drift,
        "max_abs_y_drift_m": max_abs_y_drift,
        "max_abs_z_drift_m": max_abs_z_drift,
        "max_orientation_error_rad": max_orientation_error_rad,
        "max_orientation_error_deg": max_orientation_error_deg,
        "final_orientation_error_deg": final_orientation_error_deg,
        "peak_abs_ee_vx_mps": peak_abs_ee_vx,
        "peak_abs_transport_axis_v_mps": peak_abs_transport_axis_v,
        "peak_ee_speed_mps": peak_ee_speed,
        "peak_joint_speed_rad_s": peak_joint_speed,
        "peak_target_vx_mps": peak_target_vx,
        "peak_target_ax_mps2": peak_target_ax,
        "peak_target_axis_v_mps": peak_target_axis_v,
        "peak_target_axis_a_mps2": peak_target_axis_a,
        "safety_stop_reason": safety_stop_reason,
        "transport_axis_tracking_ok": transport_axis_tracking_ok,
        "fixed_axes_ok": fixed_axes_ok,
        "transport_axis_ok": transport_axis_ok,
        "x_tracking_ok": x_tracking_ok,
        "single_axis_y_ok": single_axis_y_ok,
        "fixed_z_ok": fixed_z_ok,
        "orientation_ok": orientation_ok,
        "joint_configuration_ok": joint_configuration_ok,
        "torque_saturation_ok": torque_saturation_ok,
        "frame_reference_ok": frame_reference_ok,
        "failure_reasons": failure_reasons,
        "success": transport_success,
        "sim_dt_s": sim_dt,
        "video_path": None if args.no_video else str(video_path),
        "trace_path": str(trace_path),
        "final_x_error_m": x_err_hist[-1] if x_err_hist else None,
        "final_y_error_m": y_err_hist[-1] if y_err_hist else None,
        "final_z_error_m": z_err_hist[-1] if z_err_hist else None,
        "final_axis_error_m": axis_error_hist[-1] if axis_error_hist else None,
        "final_fixed_axis_1_error_m": fixed_axis_1_error_hist[-1] if fixed_axis_1_error_hist else None,
        "final_fixed_axis_2_error_m": fixed_axis_2_error_hist[-1] if fixed_axis_2_error_hist else None,
        "final_orientation_error_rad": ori_err_hist[-1] if ori_err_hist else None,
        "max_abs_tau_nm": max_abs_tau,
        "max_abs_tau_raw_nm": max_abs_tau_raw,
        "tau_saturation_fraction": tau_saturation_fraction,
        "max_q_excursion_from_start_rad": max_q_excursion,
        "max_q_ref_excursion_from_start_rad": max_q_ref_excursion,
        "max_joint_excursion_limit_rad": float(args.max_joint_excursion_rad),
        "min_ik_speed_scale": min_ik_speed_scale,
        "min_ik_torque_scale": min_ik_torque_scale,
        "min_task_backtrack_scale": min_task_backtrack_scale,
        "min_task_scale": min_task_scale,
        "max_task_backtrack_iters": max_task_backtrack_iters,
        "all_task_feasible": all_task_feasible,
        "frames_written": 0 if args.no_video else len(frames),
        "video_camera": str(args.video_camera),
    }
    if not args.no_video and frames:
        write_video_ffmpeg(video_path, frames, fps=args.fps)
        f0 = np.asarray(frames[0], dtype=np.float64)
        summary["first_frame_mean_rgb"] = float(np.mean(f0))
        summary["first_frame_std_rgb"] = float(np.std(f0))
    if torque_diagnostics is not None:
        diag_summary = torque_diagnostics.build_summary(duration_s=float(args.duration))
        diag_summary["coppelia_torque_control_path"] = {
            "primary_api": (
                "setJointTargetForce(signed)"
                if cop_cfg.prefer_signed_target_force
                else "setJointTargetVelocity+setJointMaxForce"
            ),
            "joint_mode": "dynamic",
            "motor_enabled": True,
            "ctrl_enabled": False,
            "internal_pid_disabled": True,
            "note": (
                "CoppeliaSim uses joint dynamic mode with external torque via "
                "setJointTargetForce when available; fallback emulates torque using "
                "large target velocity and max force."
            ),
        }
        if diag_cfg.save_controller_logs:
            diag_trace, diag_summary_path = torque_diagnostics.write_logs(
                trace_path,
                diag_summary,
            )
            print(f"Saved torque diagnostics trace:   {diag_trace}", flush=True)
            print(f"Saved torque diagnostics summary: {diag_summary_path}", flush=True)
        if diag_cfg.save_controller_plots:
            plot_paths = torque_diagnostics.generate_plots(
                diag_cfg.diagnostics_output_dir / torque_diagnostics.run_label
            )
            for plot_path in plot_paths:
                print(f"Saved torque diagnostics plot:    {plot_path}", flush=True)
        summary["torque_diagnostics"] = diag_summary
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if not args.no_video and frames and float(summary.get("first_frame_std_rgb", 0) or 0) < 1.0:
        print(
            "[vision] warning: first captured frame is nearly constant (low std); "
            "check Xvfb/GPU, camera pose, and vision sensor path.",
            flush=True,
        )

    print(json.dumps(summary, indent=2))
    print(f"Saved trace:   {trace_path}")
    if not args.no_video:
        print(f"Saved video:   {video_path}")
    print(f"Saved summary: {summary_path}")


def _write_process_exit_code(code: int) -> None:
    path = os.environ.get("REAL_CARTPOLE_PY_EXIT_FILE")
    if not path:
        return
    try:
        Path(path).write_text(f"{int(code)}\n", encoding="utf-8")
    except Exception:
        pass


if __name__ == "__main__":
    _exit_code = 0
    try:
        main()
    except SystemExit as exc:
        _exit_code = int(exc.code or 0) if isinstance(exc.code, int) else 1
        raise
    except BaseException:
        _exit_code = 1
        raise
    finally:
        _write_process_exit_code(_exit_code)
