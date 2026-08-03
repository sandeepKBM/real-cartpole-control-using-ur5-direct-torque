"""X move+hold via ``speedL`` (native RTDE Cartesian velocity control).

Still no gravity compensation / mass matrix / torque -- but DOES compute a
kinematic Jacobian each cycle (via LocalMujocoDynamics, the same MuJoCo-
backed J(q) position_transport.py's shadow OSC already uses), because
CartesianVelocityConfig.reduced_task_dims defaults to True and requires it:
sim characterization found holding full 3D orientation drives wrist_2 back
toward its kinematic singularity, capping the safe transport range at
~0.047m -- see controller_core/cartesian_velocity_controller.py's module
docstring for the full story and controller_core/state_types.py's
`RobotState` contract at the ``as_robot_state`` level (no ``mass_matrix``
or ``ee_lin_vel``/``ee_ang_vel`` required here, unlike the impedance
controller's contract). Shares the SAME safety stack as
position_transport.py (CartesianMoveMonitor, DeadlineMonitor,
StaleStateMonitor, EStopLatch, robot safety-status check every cycle) --
this is a new command path, not a new safety policy.
"""

from __future__ import annotations

import gc
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from controller_core.cartesian_velocity_controller import (
    CartesianVelocityConfig,
    CartesianVelocityController,
)
from controller_core.kinematics_utils import rotvec_to_quat_wxyz
from simulation.ur5e_mujoco_torque import x_profile_target
from transport_metrics import compute_valid_move_hold_metrics, summarize_move_hold_trace

from .link import RTDEStateError, UR5eLink
from .local_dynamics import LocalMujocoDynamics
from .position_transport import _load_cartesian_limits
from .safety import (
    CartesianMoveMonitor,
    DeadlineMonitor,
    EStopLatch,
    StaleStateMonitor,
    UR5eSafetyLimits,
    is_robot_safety_normal,
)
from .transport_common import max_abs_qd_from_trace


@dataclass
class VelocityTransportResult:
    ok: bool
    reason: str
    summary: dict[str, Any]
    trace_path: Path | None


def run_x_transport_velocity(
    link: UR5eLink,
    *,
    config_path: Path,
    target_x_delta_m: float,
    move_duration_s: float,
    duration_s: float,
    output_dir: Path | None = None,
    motion_opt_in: bool,
    rate_hz: float = 125.0,
    speed_l_acceleration: float = 1.2,
    max_tcp_accel_mps2_override: float | None = None,
    max_tcp_speed_mps_override: float | None = None,
    accel_gap_cycles_override: int | None = None,
    speed_lowpass_alpha_override: float | None = None,
    speed_limit_gap_cycles_override: int | None = None,
    speed_limit_lowpass_alpha_override: float | None = None,
    accel_max_consecutive_violations_override: int | None = None,
    accel_hard_multiple_override: float | None = None,
    speed_max_consecutive_violations_override: int | None = None,
    speed_hard_multiple_override: float | None = None,
) -> VelocityTransportResult:
    """Stream the same min-jerk move+hold X profile through ``speedL``.

    Unlike position mode's servoL (a position waypoint the robot tracks with
    its own internal gain), this streams a Cartesian VELOCITY every cycle:
    X gets the min-jerk profile's feedforward velocity plus a light P
    correction back onto the reference position; Y/Z/orientation get a pure
    P correction back to their value at reset (hold). See
    CartesianVelocityController's docstring for the exact law.
    """
    estop = EStopLatch()
    if not motion_opt_in:
        raise ValueError("motion_opt_in must be True for live velocity transport")

    move_duration_s = float(move_duration_s)
    duration_s = float(duration_s)
    if move_duration_s <= 0.0 or duration_s <= 0.0:
        raise ValueError("move_duration_s and duration_s must be positive")
    if move_duration_s > duration_s:
        raise ValueError("move_duration_s must not exceed duration_s")

    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    velocity_cfg = CartesianVelocityConfig.from_controller_yaml_section(cfg.get("controller", {}) or {})
    controller = CartesianVelocityController(velocity_cfg)
    local_dyn: LocalMujocoDynamics | None = None
    if velocity_cfg.reduced_task_dims:
        local_dyn = LocalMujocoDynamics()

    monitor = CartesianMoveMonitor(
        _load_cartesian_limits(
            config_path,
            link.robot_ip,
            max_tcp_accel_mps2_override=max_tcp_accel_mps2_override,
            max_tcp_speed_mps_override=max_tcp_speed_mps_override,
            accel_gap_cycles_override=accel_gap_cycles_override,
            speed_lowpass_alpha_override=speed_lowpass_alpha_override,
            speed_limit_gap_cycles_override=speed_limit_gap_cycles_override,
            speed_limit_lowpass_alpha_override=speed_limit_lowpass_alpha_override,
            accel_max_consecutive_violations_override=accel_max_consecutive_violations_override,
            accel_hard_multiple_override=accel_hard_multiple_override,
            speed_max_consecutive_violations_override=speed_max_consecutive_violations_override,
            speed_hard_multiple_override=speed_hard_multiple_override,
        )
    )
    dt_s = 1.0 / float(rate_hz)

    link.connect(with_control=True)
    link.verify_speedl_signature()
    try:
        state0 = link.read_state()
    except RTDEStateError as exc:
        estop.trip(f"initial state read failed: {exc}")
        link.safe_stop(str(exc))
        return VelocityTransportResult(
            ok=False,
            reason=str(exc),
            summary={"success": False, "termination_reason": str(exc)},
            trace_path=None,
        )

    start_pose = state0.tcp_pose.copy()
    x0 = float(start_pose[0])
    monitor.set_start(start_pose, move_axis_index=0)
    init_state = {
        "time": 0.0,
        "q": state0.q,
        "qd": state0.qd,
        "ee_pos": start_pose[:3].copy(),
        "ee_quat": rotvec_to_quat_wxyz(start_pose[3:6]),
        "target_x": x0,
    }
    controller.reset_from_state(init_state)

    trace_rows: list[dict[str, Any]] = []
    termination_reason = "duration_complete"
    steps = 0
    t_s = 0.0
    last_state = state0

    safety_limits = getattr(link, "limits", None) or UR5eSafetyLimits()
    deadline_monitor = DeadlineMonitor(safety_limits.max_deadline_ms)
    stale_monitor = StaleStateMonitor()

    gc.disable()
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
            target_ee_pos = start_pose[:3].copy()
            target_ee_pos[0] = float(target_x)
            target_ee_vel = np.array([target_x_vel, 0.0, 0.0], dtype=np.float64)
            waypoint = start_pose.copy()
            waypoint[0] = float(target_x)

            try:
                link_state = link.read_state()
            except RTDEStateError as exc:
                termination_reason = f"rtde_state_error: {exc}"
                estop.trip(termination_reason)
                break

            robot_state = {
                "time": t_s,
                "q": link_state.q,
                "qd": link_state.qd,
                "ee_pos": link_state.tcp_pose[:3].copy(),
                "ee_quat": rotvec_to_quat_wxyz(link_state.tcp_pose[3:6]),
                "target_x": float(target_x),
                "target_ee_pos": target_ee_pos,
                "target_ee_vel": target_ee_vel,
            }
            if local_dyn is not None:
                robot_state["jacobian"] = local_dyn.jacobian(link_state.q)
            xd_cmd = controller.compute(robot_state)

            try:
                link.speed_l(xd_cmd, acceleration=float(speed_l_acceleration), time_s=dt_s * 1.5)
            except RTDEStateError as exc:
                termination_reason = f"rtde_state_error: {exc}"
                estop.trip(termination_reason)
                break

            last_state = link_state

            stale_reason = stale_monitor.record(link_state.robot_timestamp_s, link_state.host_stamp_ns)
            if stale_reason:
                termination_reason = stale_reason
                estop.trip(termination_reason)
                break

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
                "xd_cmd": xd_cmd.tolist(),
                "command_mode": "speedL",
            }
            tcp = np.asarray(link_state.tcp_pose, dtype=np.float64).reshape(6)
            row["ee_pos"] = tcp[:3].tolist()
            row["ee_quat"] = rotvec_to_quat_wxyz(tcp[3:6]).tolist()
            row["x_error"] = float(target_x - tcp[0])
            row["orientation_error_norm"] = float(np.linalg.norm(tcp[3:] - start_pose[3:]))
            trace_rows.append(row)
            steps += 1
            t_s += dt_s

            elapsed_s = time.monotonic() - cycle_start
            overrun_ns = int(max(0.0, elapsed_s - dt_s) * 1e9)
            deadline_reason = deadline_monitor.record(overrun_ns)
            if deadline_reason:
                termination_reason = deadline_reason
                estop.trip(termination_reason)
                break

            sleep_s = dt_s - elapsed_s
            if sleep_s > 0:
                time.sleep(sleep_s)
    finally:
        gc.enable()
        link.speed_stop()
        link.safe_stop("transport_exit")

    achieved_x_delta_m = float(last_state.tcp_pose[0] - x0)
    max_abs_qd = max_abs_qd_from_trace(trace_rows)
    summary = {
        "backend": "speedL_velocity",
        "control_mode": "velocity",
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
        "termination_reason": termination_reason,
        "achieved_x_delta_m": achieved_x_delta_m,
        "final_tcp_pose": last_state.tcp_pose.tolist(),
        "initial_ee_pos": state0.tcp_pose[:3].tolist(),
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
    return VelocityTransportResult(ok=ok, reason=reason, summary=summary, trace_path=trace_path)
