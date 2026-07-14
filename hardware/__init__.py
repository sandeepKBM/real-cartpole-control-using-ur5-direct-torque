"""Real-UR5e hardware lane: connect, safety, position moves, and torque transport.

Active path: ``docs/hardware/README.md`` (learning map) + ``HARDWARE_GUIDE.md``.
Three modes via ``hardware.x_transport``: position / direct_torque / urscript.
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
