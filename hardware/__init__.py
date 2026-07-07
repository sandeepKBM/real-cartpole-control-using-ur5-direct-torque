"""Real-UR5e hardware lane: connect, read state, and one bounded Cartesian move.

Three modules, one job each:
- ``hardware.safety`` -- every numeric limit and safety decision.
- ``hardware.link`` -- RTDE connection and live state (``UR5eLink``).
- ``hardware.motion`` -- the one bounded Cartesian move
  (``move_cartesian_bounded``).

See ``tools/ur5e_connect.py`` (connect + monitor) and ``tools/ur5e_move.py``
(the move) for the CLI entrypoints. Direct torque control is out of scope in
this lane -- see AGENTS.md section 4 for why.
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
