"""X move+hold via ``servoL`` (position control) — test lane before direct torque."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from controller_core.kinematics_utils import rotvec_to_quat_wxyz
from controller_core.x_axis_cartesian_impedance import (
    CartesianImpedanceConfig,
    XAxisCartesianImpedanceController,
)
from simulation.ur5e_mujoco_torque import x_profile_target
from transport_metrics import compute_valid_move_hold_metrics, summarize_move_hold_trace

from .direct_torque_link import UR5eDirectTorqueLink
from .link import RTDEStateError, UR5eLink
from .local_dynamics import LocalMujocoDynamics
from .safety import CartesianMoveLimits, CartesianMoveMonitor, EStopLatch, is_robot_safety_normal


@dataclass
class PositionTransportResult:
    ok: bool
    reason: str
    summary: dict[str, Any]
    trace_path: Path | None


def _load_cartesian_limits(config_path: Path, robot_ip: str) -> CartesianMoveLimits:
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    safety_raw = (cfg.get("controller", {}) or {}).get("safety", {}) or {}
    base = CartesianMoveLimits(
        max_off_axis_drift_m=float(safety_raw.get("max_abs_orthogonal_drift_m", 0.03)),
        max_orientation_error_rad=float(safety_raw.get("max_orientation_error_rad", 0.25)),
        qd_max_radps=float(safety_raw.get("max_joint_velocity_radps", 3.0)),
    )
    return CartesianMoveLimits.for_robot(robot_ip, base=base)


def run_x_transport_position(
    link: UR5eLink,
    *,
    config_path: Path,
    target_x_delta_m: float,
    move_duration_s: float,
    duration_s: float,
    output_dir: Path | None = None,
    motion_opt_in: bool,
    rate_hz: float = 125.0,
    shadow_osc: bool = True,
    servo_gain: float = 300.0,
    servo_lookahead_s: float = 0.1,
) -> PositionTransportResult:
    """Stream the same min-jerk move+hold X profile through ``servoL``.

    When ``shadow_osc`` is True (default), also runs the tuned OSC controller
    in software using local MuJoCo J+M and logs ``tau_shadow`` — torques are
  **not** sent to the robot. Use this to validate trajectory + safety + logging
    before ``control_mode=direct_torque``.
    """
    estop = EStopLatch()
    if not motion_opt_in:
        raise ValueError("motion_opt_in must be True for live position transport")

    move_duration_s = float(move_duration_s)
    duration_s = float(duration_s)
    if move_duration_s <= 0.0 or duration_s <= 0.0:
        raise ValueError("move_duration_s and duration_s must be positive")
    if move_duration_s > duration_s:
        raise ValueError("move_duration_s must not exceed duration_s")

    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    impedance_cfg = CartesianImpedanceConfig.from_controller_yaml_section(cfg.get("controller", {}) or {})
    controller: XAxisCartesianImpedanceController | None = None
    local_dyn: LocalMujocoDynamics | None = None
    if shadow_osc:
        controller = XAxisCartesianImpedanceController(impedance_cfg)
        try:
            local_dyn = LocalMujocoDynamics()
        except ImportError:
            shadow_osc = False

    monitor = CartesianMoveMonitor(_load_cartesian_limits(config_path, link.robot_ip))
    dt_s = 1.0 / float(rate_hz)
    servo_time_s = dt_s * 1.5

    link.connect(with_control=True)
    try:
        state0 = link.read_state()
    except RTDEStateError as exc:
        estop.trip(f"initial state read failed: {exc}")
        link.safe_stop(str(exc))
        return PositionTransportResult(
            ok=False,
            reason=str(exc),
            summary={"success": False, "termination_reason": str(exc)},
            trace_path=None,
        )

    start_pose = state0.tcp_pose.copy()
    x0 = float(start_pose[0])
    monitor.set_start(start_pose, move_axis_index=0)
    if controller is not None:
        init_state = _robot_state_from_link(
            state0,
            local_dyn=local_dyn,
            time_s=0.0,
            target_x=x0,
            target_x_vel=0.0,
        )
        controller.reset_from_state(init_state)

    trace_rows: list[dict[str, Any]] = []
    termination_reason = "duration_complete"
    steps = 0
    t_s = 0.0
    last_state = state0

    try:
        while t_s < duration_s - 1e-12:
            estop.raise_if_tripped()
            cycle_start = time.monotonic()

            target_x, target_x_vel = x_profile_target(
                "min_jerk_move_hold",
                x0,
                float(target_x_delta_m),
                t_s,
                duration_s,
                move_duration_s=move_duration_s,
            )
            waypoint = start_pose.copy()
            waypoint[0] = float(target_x)

            try:
                link.servo_l(
                    waypoint,
                    speed=0.25,
                    acceleration=1.2,
                    time_s=servo_time_s,
                    lookahead_time=float(servo_lookahead_s),
                    gain=float(servo_gain),
                )
                link_state = link.read_state()
            except RTDEStateError as exc:
                termination_reason = f"rtde_state_error: {exc}"
                estop.trip(termination_reason)
                break

            last_state = link_state
            orientation_error_rad = float(np.linalg.norm(link_state.tcp_pose[3:] - start_pose[3:]))
            decision = monitor.check(
                q=link_state.q,
                qd=link_state.qd,
                tcp_pose=link_state.tcp_pose,
                target_tcp_pose=waypoint,
                orientation_error_rad=orientation_error_rad,
                axis_target_moving=bool(t_s <= move_duration_s),
                dt_s=dt_s,
            )
            if not decision.ok:
                termination_reason = decision.reason or "safety_stop"
                estop.trip(termination_reason)
                break

            if not is_robot_safety_normal(link_state.safety_status):
                termination_reason = f"robot_safety_status_abnormal: {link_state.safety_status}"
                estop.trip(termination_reason)
                break

            row: dict[str, Any] = {
                "time_s": t_s,
                "q": link_state.q.tolist(),
                "qd": link_state.qd.tolist(),
                "tcp_pose": link_state.tcp_pose.tolist(),
                "target_x": target_x,
                "target_x_vel": target_x_vel,
                "command_mode": "servoL",
            }
            tcp = np.asarray(link_state.tcp_pose, dtype=np.float64).reshape(6)
            row["ee_pos"] = tcp[:3].tolist()
            row["ee_quat"] = rotvec_to_quat_wxyz(tcp[3:6]).tolist()
            row["x_error"] = float(target_x - tcp[0])
            row["orientation_error_norm"] = float(np.linalg.norm(tcp[3:] - start_pose[3:]))
            if controller is not None and local_dyn is not None:
                robot_state = _robot_state_from_link(
                    link_state,
                    local_dyn=local_dyn,
                    time_s=t_s,
                    target_x=target_x,
                    target_x_vel=target_x_vel,
                )
                out = controller.compute(robot_state)
                row["tau_shadow"] = np.asarray(out.tau, dtype=np.float64).tolist()
                row["x_error_shadow"] = float(out.x_error)
                row["orientation_error_norm_shadow"] = float(out.orientation_error_norm)
            trace_rows.append(row)
            steps += 1
            t_s += dt_s

            elapsed_s = time.monotonic() - cycle_start
            sleep_s = dt_s - elapsed_s
            if sleep_s > 0:
                time.sleep(sleep_s)
    finally:
        link.servo_stop()
        link.safe_stop("transport_exit")

    achieved_x_delta_m = float(last_state.tcp_pose[0] - x0)
    max_abs_qd = float(
        max((max(abs(v) for v in row.get("qd", [0.0] * 6)) for row in trace_rows), default=0.0)
    )
    summary = {
        "backend": "servoL_position",
        "control_mode": "position",
        "config_path": str(config_path),
        "target_x_delta": float(target_x_delta_m),
        "target_x_delta_m": float(target_x_delta_m),
        "transport_axis_index": 0,
        "move_duration_s": move_duration_s,
        "hold_duration_s": max(duration_s - move_duration_s, 0.0),
        "duration_s": duration_s,
        "total_duration_s": duration_s,
        "sim_time_s": min(t_s, duration_s),
        "frequency_hz": float(rate_hz),
        "steps": steps,
        "shadow_osc": bool(shadow_osc and controller is not None),
        "termination_reason": termination_reason,
        "achieved_x_delta_m": achieved_x_delta_m,
        "final_tcp_pose": last_state.tcp_pose.tolist(),
        "initial_ee_pos": state0.tcp_pose[:3].tolist(),
        "servo_gain": float(servo_gain),
        "servo_lookahead_s": float(servo_lookahead_s),
        "success": termination_reason == "duration_complete" and not estop.tripped,
        "velocity_guard_ok": max_abs_qd <= monitor.limits.qd_max_radps,
        "max_abs_qd_radps": max_abs_qd,
        "joint_limit_guard_ok": True,
        "torque_saturation_percentage": 0.0,
    }
    summary.update(
        summarize_move_hold_trace(
            trace_rows,
            initial_ee_pos=state0.tcp_pose[:3],
            move_duration_s=move_duration_s,
            total_duration_s=duration_s,
            transport_axis_index=0,
        )
    )
    summary.update(compute_valid_move_hold_metrics(summary))
    summary["success"] = bool(
        not estop.tripped
        and summary.get("valid_move_and_hold", False)
        and termination_reason == "duration_complete"
    )

    trace_path = None
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        trace_path = output_dir / "trace.jsonl"
        with trace_path.open("w", encoding="utf-8") as fh:
            for row in trace_rows:
                fh.write(json.dumps(row) + "\n")
        (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    ok = bool(summary["success"])
    reason = "" if ok else str(summary.get("hold_failure_reason") or summary.get("move_failure_reason") or termination_reason)
    return PositionTransportResult(ok=ok, reason=reason, summary=summary, trace_path=trace_path)


def _robot_state_from_link(
    link_state,
    *,
    local_dyn: LocalMujocoDynamics | None,
    time_s: float,
    target_x: float,
    target_x_vel: float,
) -> dict[str, Any]:
    tcp = np.asarray(link_state.tcp_pose, dtype=np.float64).reshape(6)
    q = np.asarray(link_state.q, dtype=np.float64).reshape(6)
    qd = np.asarray(link_state.qd, dtype=np.float64).reshape(6)
    if local_dyn is not None:
        jacobian, mass_matrix = local_dyn.jacobian_and_mass_matrix(q)
        return UR5eDirectTorqueLink.compose_robot_state(
            link_state,
            jacobian=jacobian,
            mass_matrix=mass_matrix,
            time_s=time_s,
            target_x=target_x,
            target_x_vel=target_x_vel,
        )
    return {
        "time": float(time_s),
        "q": q,
        "qd": qd,
        "ee_pos": tcp[:3].copy(),
        "ee_quat": rotvec_to_quat_wxyz(tcp[3:6]),
        "ee_lin_vel": np.zeros(3),
        "ee_ang_vel": np.zeros(3),
        "jacobian": np.eye(6),
        "mass_matrix": np.eye(6),
        "target_x": float(target_x),
        "target_x_vel": float(target_x_vel),
        "transport_axis_index": 0,
    }
