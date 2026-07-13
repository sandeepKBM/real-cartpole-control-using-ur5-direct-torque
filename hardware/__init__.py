"""Real-UR5e hardware lane: connect, read state, position moves, and direct torque.

Modules:
- ``hardware.safety`` -- limits, connection health, e-stop latch, move monitors.
- ``hardware.link`` -- ``UR5eLink`` (RTDE receive + optional ``servoL``).
- ``hardware.motion`` -- bounded Cartesian move via ``servoL``.
- ``hardware.direct_torque_link`` -- ``UR5eDirectTorqueLink`` (``directTorque()``).
- ``hardware.direct_torque_transport`` -- 500 Hz OSC X transport on real hardware.

See ``docs/hardware/HARDWARE_GUIDE.md`` for the full learning guide.
"""

from __future__ import annotations

from .link import RTDELinkError, RTDEStateError, UR5eLink, UR5eState
from .logging import JsonlWriter, json_dumps_safe, write_json
from .motion import MoveResult, move_cartesian_bounded, peak_velocity_mps, plan_waypoints
from .safety import (
    CartesianMoveLimits,
    CartesianMoveMonitor,
    ConnectionHealth,
    EStopLatch,
    EStopTripped,
    SafetyDecision,
    UR5eSafetyLimits,
    check_joint_state,
    check_tcp_pose,
)
from .timing import TimingSample, TimingTracker, compute_stats_ns, period_from_frequency

__all__ = [
    "RTDELinkError",
    "RTDEStateError",
    "UR5eLink",
    "UR5eState",
    "JsonlWriter",
    "json_dumps_safe",
    "write_json",
    "MoveResult",
    "move_cartesian_bounded",
    "peak_velocity_mps",
    "plan_waypoints",
    "CartesianMoveLimits",
    "CartesianMoveMonitor",
    "ConnectionHealth",
    "EStopLatch",
    "EStopTripped",
    "SafetyDecision",
    "UR5eSafetyLimits",
    "check_joint_state",
    "check_tcp_pose",
    "TimingSample",
    "TimingTracker",
    "compute_stats_ns",
    "period_from_frequency",
]
