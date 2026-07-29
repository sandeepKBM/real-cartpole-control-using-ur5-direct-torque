"""The one bounded Cartesian move -- and nothing else.

``move_cartesian_bounded()`` streams a quintic min-jerk waypoint profile
along one Cartesian axis via ``UR5eLink.servo_l()``, reading state and
checking ``hardware.safety.CartesianMoveMonitor`` every single cycle. Any
safety violation or state-read failure aborts immediately via
``link.safe_stop()`` and ``estop.trip()`` -- there is no retry and no
reconnect attempt mid-motion; a reconnect during motion could paper over a
real fault, so any problem here is treated as final.

This uses position-space ``servoL`` streaming, not torque control -- ``servoL``
lets the robot's own firmware do inverse kinematics, so no Jacobian/FK code is
needed here. (Historical note: when this module was first written the
installed RTDE control library had no working torque API at all; a live
torque path was added later in ``hardware/direct_torque_transport.py`` and
``hardware/urscript_transport.py`` -- see AGENTS.md sec 4 for both lanes and
the reasoning behind keeping this one position-only.)
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from math import sqrt
from pathlib import Path
from typing import Any

import numpy as np

from .link import RTDEStateError, UR5eLink
from .safety import (
    CartesianMoveMonitor,
    DeadlineMonitor,
    EStopLatch,
    StaleStateMonitor,
    UR5eSafetyLimits,
    is_robot_safety_normal,
)


@dataclass
class MoveResult:
    ok: bool
    reason: str
    waypoints_sent: int
    stopped_early: bool
    final_tcp_pose: np.ndarray | None
    trace_path: Path | None = None
    summary_path: Path | None = None


def _min_jerk_s(tau: float) -> float:
    """Quintic min-jerk scalar profile s(tau) for tau in [0, 1]: zero
    velocity and acceleration at both endpoints."""
    tau = min(max(tau, 0.0), 1.0)
    return 10.0 * tau**3 - 15.0 * tau**4 + 6.0 * tau**5


def peak_velocity_mps(distance_m: float, duration_s: float) -> float:
    """Peak velocity of the quintic min-jerk profile above (its derivative's
    maximum, at tau=0.5): v_peak = 1.875 * distance / duration."""
    if duration_s <= 0.0:
        raise ValueError("duration_s must be positive")
    return 1.875 * abs(distance_m) / duration_s


def peak_acceleration_mps2(distance_m: float, duration_s: float) -> float:
    """Peak acceleration of the quintic min-jerk profile.

    For s(tau)=10*tau^3-15*tau^4+6*tau^5, max |s''(tau)| = 10*sqrt(3)/3.
    """
    if duration_s <= 0.0:
        raise ValueError("duration_s must be positive")
    return (10.0 * sqrt(3.0) / 3.0) * abs(distance_m) / (duration_s * duration_s)


def plan_waypoints(
    start_tcp_pose,
    *,
    axis_index: int,
    distance_m: float,
    duration_s: float,
    rate_hz: float,
) -> list[np.ndarray]:
    """Generate the full ordered list of target TCP-pose waypoints for one
    bounded single-axis Cartesian move. Orientation (the rotation-vector
    half of the TCP pose) is held fixed at the start value for every
    waypoint -- this only ever asks the firmware's IK to translate."""
    if axis_index not in (0, 1, 2):
        raise ValueError(f"axis_index must be 0, 1, or 2; got {axis_index!r}")
    if duration_s <= 0.0:
        raise ValueError("duration_s must be positive")
    if rate_hz <= 0.0:
        raise ValueError("rate_hz must be positive")
    start_pose = np.asarray(start_tcp_pose, dtype=np.float64).reshape(6)
    n_steps = max(1, int(round(duration_s * rate_hz)))
    axis_unit = np.zeros(3, dtype=np.float64)
    axis_unit[axis_index] = 1.0
    waypoints = []
    for i in range(1, n_steps + 1):
        s = _min_jerk_s(i / n_steps)
        pose = start_pose.copy()
        pose[:3] = start_pose[:3] + s * distance_m * axis_unit
        waypoints.append(pose)
    return waypoints


def move_cartesian_bounded(
    link: UR5eLink,
    monitor: CartesianMoveMonitor,
    estop: EStopLatch,
    *,
    axis_index: int,
    distance_m: float,
    motion_opt_in: bool,
    duration_s: float = 6.0,
    rate_hz: float = 125.0,
    lookahead_time_s: float = 0.1,
    gain: float = 300.0,
    trace_path: Path | None = None,
    summary_path: Path | None = None,
) -> MoveResult:
    """Stream one bounded, safety-monitored Cartesian move.

    Every cycle: send one servoL waypoint, read state, run the monitor. Any
    problem stops the loop immediately via ``link.safe_stop()`` +
    ``estop.trip()`` -- this function never retries or reconnects mid-move.
    """
    estop.raise_if_tripped()
    if not motion_opt_in:
        return MoveResult(
            ok=False,
            reason="motion is blocked until motion_opt_in is enabled",
            waypoints_sent=0,
            stopped_early=False,
            final_tcp_pose=None,
        )
    if not link.has_control:
        raise RuntimeError("move_cartesian_bounded requires link.connect(with_control=True)")

    trace_rows: list[dict[str, Any]] = []

    def _write_artifacts(result: MoveResult) -> MoveResult:
        if trace_path is not None:
            from controller_core.logging_utils import json_dumps_safe

            trace_path.parent.mkdir(parents=True, exist_ok=True)
            with trace_path.open("w", encoding="utf-8") as f:
                for row in trace_rows:
                    f.write(json_dumps_safe(row) + "\n")
            result.trace_path = trace_path
        if summary_path is not None:
            import json

            summary_path.parent.mkdir(parents=True, exist_ok=True)
            summary = {
                "success": bool(result.ok),
                "termination_reason": result.reason or "duration_complete",
                "waypoints_sent": int(result.waypoints_sent),
                "stopped_early": bool(result.stopped_early),
                "final_tcp_pose": None if result.final_tcp_pose is None else result.final_tcp_pose.tolist(),
                "trace_path": None if result.trace_path is None else str(result.trace_path),
                "control_mode": "position_servoL_bounded",
                "rate_hz": float(rate_hz),
                "duration_s": float(duration_s),
                "distance_m": float(distance_m),
                "axis_index": int(axis_index),
                "lookahead_time_s": float(lookahead_time_s),
                "gain": float(gain),
                "safety_limits": {
                    "max_off_axis_drift_m": float(monitor.limits.max_off_axis_drift_m),
                    "max_orientation_error_rad": float(monitor.limits.max_orientation_error_rad),
                    "max_tcp_speed_mps": float(monitor.limits.max_tcp_speed_mps),
                    "max_tcp_accel_mps2": float(monitor.limits.max_tcp_accel_mps2),
                    "max_waypoint_jump_m": float(monitor.limits.max_waypoint_jump_m),
                    "max_axis_error_growth_steps": int(monitor.limits.max_axis_error_growth_steps),
                    "qd_max_radps": float(monitor.limits.qd_max_radps),
                    "accel_gap_cycles": int(monitor.limits.accel_gap_cycles),
                    "speed_lowpass_alpha": float(monitor.limits.speed_lowpass_alpha),
                },
            }
            summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
            result.summary_path = summary_path
        return result

    try:
        start_state = link.read_state()
    except RTDEStateError as exc:
        reason = f"initial state read failed: {exc}"
        link.safe_stop(reason)
        estop.trip(reason)
        return _write_artifacts(
            MoveResult(ok=False, reason=reason, waypoints_sent=0, stopped_early=True, final_tcp_pose=None)
        )

    start_pose = start_state.tcp_pose
    monitor.set_start(start_pose, axis_index)
    waypoints = plan_waypoints(
        start_pose,
        axis_index=axis_index,
        distance_m=distance_m,
        duration_s=duration_s,
        rate_hz=rate_hz,
    )
    dt_s = 1.0 / rate_hz
    # ur_rtde convention: servoL's `time` argument should exceed the control
    # period for smooth blending between waypoints.
    servo_time_s = dt_s * 1.5

    # Enforce the previously-unchecked max_deadline_ms, and catch a
    # frozen-but-non-raising RTDE stream, on every cycle (see hardware/safety.py
    # DeadlineMonitor/StaleStateMonitor for the trip-condition reasoning).
    safety_limits = getattr(link, "limits", None) or UR5eSafetyLimits()
    deadline_monitor = DeadlineMonitor(safety_limits.max_deadline_ms)
    stale_monitor = StaleStateMonitor()

    move_start = time.monotonic()
    for i, waypoint in enumerate(waypoints):
        cycle_start = time.monotonic()
        cycle_start_ns = time.monotonic_ns()
        try:
            link.servo_l(
                waypoint,
                speed=0.25,
                acceleration=1.2,
                time_s=servo_time_s,
                lookahead_time=lookahead_time_s,
                gain=gain,
            )
            state = link.read_state()
        except RTDEStateError as exc:
            reason = f"state read failed during move: {exc}"
            link.safe_stop(reason)
            estop.trip(reason)
            return _write_artifacts(
                MoveResult(ok=False, reason=reason, waypoints_sent=i, stopped_early=True, final_tcp_pose=None)
            )

        stale_reason = stale_monitor.record(state.robot_timestamp_s, state.host_stamp_ns)
        orientation_error_rad = float(np.linalg.norm(state.tcp_pose[3:] - start_pose[3:]))
        axis_error_m = float(waypoint[axis_index] - state.tcp_pose[axis_index])
        off_axis_drift_m = {}
        for axis_name, idx in zip(("x", "y", "z"), range(3)):
            if idx != axis_index:
                off_axis_drift_m[axis_name] = float(state.tcp_pose[idx] - start_pose[idx])
        trace_rows.append(
            {
                "step": int(i),
                "t_s": float(time.monotonic() - move_start),
                "target_t_s": float((i + 1) / rate_hz),
                "cycle_start_ns": int(cycle_start_ns),
                "robot_timestamp_s": state.robot_timestamp_s,
                "host_stamp_ns": state.host_stamp_ns,
                "q": state.q.tolist(),
                "qd": state.qd.tolist(),
                "tcp_pose": state.tcp_pose.tolist(),
                "start_tcp_pose": start_pose.tolist(),
                "target_tcp_pose": waypoint.tolist(),
                "axis_index": int(axis_index),
                "axis_error_m": axis_error_m,
                "off_axis_drift_m": off_axis_drift_m,
                "orientation_error_rad": orientation_error_rad,
                "axis_target_moving": bool(i < len(waypoints) - 1),
            }
        )
        if stale_reason:
            link.safe_stop(stale_reason)
            estop.trip(stale_reason)
            return _write_artifacts(
                MoveResult(
                    ok=False,
                    reason=stale_reason,
                    waypoints_sent=i + 1,
                    stopped_early=True,
                    final_tcp_pose=state.tcp_pose,
                )
            )

        decision = monitor.check(
            q=state.q,
            qd=state.qd,
            tcp_pose=state.tcp_pose,
            target_tcp_pose=waypoint,
            orientation_error_rad=orientation_error_rad,
            axis_target_moving=(i < len(waypoints) - 1),
            dt_s=dt_s,
        )
        if not decision.ok:
            link.safe_stop(decision.reason)
            estop.trip(decision.reason)
            return _write_artifacts(
                MoveResult(
                    ok=False,
                    reason=decision.reason,
                    waypoints_sent=i + 1,
                    stopped_early=True,
                    final_tcp_pose=state.tcp_pose,
                )
            )

        if not is_robot_safety_normal(state.safety_status):
            reason = f"robot_safety_status_abnormal: {state.safety_status}"
            link.safe_stop(reason)
            estop.trip(reason)
            return _write_artifacts(
                MoveResult(
                    ok=False,
                    reason=reason,
                    waypoints_sent=i + 1,
                    stopped_early=True,
                    final_tcp_pose=state.tcp_pose,
                )
            )

        elapsed_s = time.monotonic() - cycle_start
        # Overrun = how far this cycle's work ran past its period budget. The
        # sleep below already absorbs any overrun silently (max(0, ...)), so
        # this is the only place lateness is acted on.
        overrun_ns = int(max(0.0, elapsed_s - dt_s) * 1e9)
        deadline_reason = deadline_monitor.record(overrun_ns)
        if deadline_reason:
            link.safe_stop(deadline_reason)
            estop.trip(deadline_reason)
            return _write_artifacts(
                MoveResult(
                    ok=False,
                    reason=deadline_reason,
                    waypoints_sent=i + 1,
                    stopped_early=True,
                    final_tcp_pose=state.tcp_pose,
                )
            )

        sleep_s = dt_s - elapsed_s
        if sleep_s > 0:
            time.sleep(sleep_s)

    link.servo_stop()
    final_state = link.read_state()
    return _write_artifacts(
        MoveResult(
            ok=True,
            reason="",
            waypoints_sent=len(waypoints),
            stopped_early=False,
            final_tcp_pose=final_state.tcp_pose,
        )
    )
