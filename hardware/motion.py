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

import numpy as np

from .link import RTDEStateError, UR5eLink
from .safety import CartesianMoveMonitor, EStopLatch, is_robot_safety_normal


@dataclass
class MoveResult:
    ok: bool
    reason: str
    waypoints_sent: int
    stopped_early: bool
    final_tcp_pose: np.ndarray | None


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

    try:
        start_state = link.read_state()
    except RTDEStateError as exc:
        reason = f"initial state read failed: {exc}"
        link.safe_stop(reason)
        estop.trip(reason)
        return MoveResult(ok=False, reason=reason, waypoints_sent=0, stopped_early=True, final_tcp_pose=None)

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

    for i, waypoint in enumerate(waypoints):
        cycle_start = time.monotonic()
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
            return MoveResult(
                ok=False, reason=reason, waypoints_sent=i, stopped_early=True, final_tcp_pose=None
            )

        orientation_error_rad = float(np.linalg.norm(state.tcp_pose[3:] - start_pose[3:]))
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
            return MoveResult(
                ok=False,
                reason=decision.reason,
                waypoints_sent=i + 1,
                stopped_early=True,
                final_tcp_pose=state.tcp_pose,
            )

        if not is_robot_safety_normal(state.safety_status):
            reason = f"robot_safety_status_abnormal: {state.safety_status}"
            link.safe_stop(reason)
            estop.trip(reason)
            return MoveResult(
                ok=False,
                reason=reason,
                waypoints_sent=i + 1,
                stopped_early=True,
                final_tcp_pose=state.tcp_pose,
            )

        elapsed_s = time.monotonic() - cycle_start
        sleep_s = dt_s - elapsed_s
        if sleep_s > 0:
            time.sleep(sleep_s)

    link.servo_stop()
    final_state = link.read_state()
    return MoveResult(
        ok=True,
        reason="",
        waypoints_sent=len(waypoints),
        stopped_early=False,
        final_tcp_pose=final_state.tcp_pose,
    )
