"""Hardware-facing helpers for staged UR5e RTDE bring-up.

This package is intentionally separate from the simulator controller stack.
The modules provide safe-by-default RTDE wrappers, timing helpers, logging
helpers, and optional ROS visualization publishers for receive-only and
very small motion tests.
"""

from .logging import JsonlWriter, json_dumps_safe, write_json
from .safety_limits import (
    MotionCommandGuard,
    SafetyDecision,
    UR5eSafetyLimits,
    UR5eStateGuard,
    check_finite_array,
    check_joint_state,
    check_tcp_pose,
)
from .ros_topics import AsyncGuardrailPublisher, AsyncRosVisualizer, GuardrailStatusSample, RosTopicSample
from .timing import TimingSample, TimingTracker, compute_stats_ns, period_from_frequency
from .ur5e_control_session import (
    UR5eCommandResult,
    UR5eConnectionSnapshot,
    UR5eHardwareSession,
    UR5eHardwareSessionConfig,
)
from .ur5e_rtde_bridge import UR5eRTDEBridge, UR5eState
from .ur5e_stages import (
    StageResult,
    run_direct_torque_probe,
    run_receive_only,
    run_servoj_tiny_motion,
    run_servoj_zero_hold,
)

__all__ = [
    "JsonlWriter",
    "json_dumps_safe",
    "write_json",
    "MotionCommandGuard",
    "SafetyDecision",
    "UR5eSafetyLimits",
    "UR5eStateGuard",
    "AsyncGuardrailPublisher",
    "AsyncRosVisualizer",
    "GuardrailStatusSample",
    "RosTopicSample",
    "check_finite_array",
    "check_joint_state",
    "check_tcp_pose",
    "TimingSample",
    "TimingTracker",
    "compute_stats_ns",
    "period_from_frequency",
    "UR5eCommandResult",
    "UR5eConnectionSnapshot",
    "UR5eHardwareSession",
    "UR5eHardwareSessionConfig",
    "UR5eRTDEBridge",
    "UR5eState",
    "StageResult",
    "run_direct_torque_probe",
    "run_receive_only",
    "run_servoj_tiny_motion",
    "run_servoj_zero_hold",
]
