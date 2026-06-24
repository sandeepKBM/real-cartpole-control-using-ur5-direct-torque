"""Simulator-independent controller core for UR5 control experiments.

This package contains the UR5 X-axis torque / impedance stack, the cart-pole
constrained-control scaffold used for future LQR tuning, and the
simulation-only safety / shadow helpers used by the retained transport lane.

No MuJoCo, CoppeliaSim, or ROS imports inside this package.
"""

from .state_types import ControlOutput, RobotState, as_impedance_robot_state, as_robot_state
from .controller_interfaces import (
    CommandMode,
    ControllerCommand,
    ControllerState,
    FallbackAction,
    InterventionLevel,
    NominalController,
    SafetyFilter,
    SafetyFilterResult,
    SafetyLimits,
    SafetySeverity,
)
from .cartpole_linear_model import CartPoleLinearModel, solve_discrete_lqr
from .recoverability_monitor import HeuristicRecoverabilityMonitor
from .safety_filter import CommandGovernorSafetyFilter
from .joint_impedance import JointImpedanceConfig, JointImpedanceController, JointImpedanceOutput
from .actuation_shadow import HardwareShadowConfig, HardwareShadowModel, HardwareShadowOutput
from .transport_lqr import FixedXTransportLQRConfig, FixedXTransportLQRController
from .lqr_controller import (
    CartPoleFallbackConfig,
    CartPoleFallbackController,
    CartPoleLQRConfig,
    CartPoleLQRController,
)
from .mpc_controller import CartPoleMPCConfig, CartPoleMPCController
from .x_axis_cartesian_impedance import (
    CartesianImpedanceConfig,
    CartesianImpedanceOutput,
    XAxisCartesianImpedanceController,
)
from .filters import TorqueCommandFilter
from .safety import ImpedanceSafetyConfig, ImpedanceSafetyMonitor, ImpedanceSafetyStatus
from .logging_utils import JsonlTraceWriter, json_dumps_safe
from .kinematics_utils import cartesian_force_to_joint_torque
from .x_axis_controller import XAxisController, XAxisControllerConfig
from .safety_utils import SafetyConfig, SafetyMonitor, SafetyStatus

__all__ = [
    "ControlOutput",
    "RobotState",
    "as_robot_state",
    "as_impedance_robot_state",
    "CommandMode",
    "ControllerCommand",
    "ControllerState",
    "FallbackAction",
    "InterventionLevel",
    "NominalController",
    "SafetyFilter",
    "SafetyFilterResult",
    "SafetyLimits",
    "SafetySeverity",
    "CartPoleLinearModel",
    "solve_discrete_lqr",
    "JointImpedanceConfig",
    "JointImpedanceController",
    "JointImpedanceOutput",
    "HardwareShadowConfig",
    "HardwareShadowModel",
    "HardwareShadowOutput",
    "HeuristicRecoverabilityMonitor",
    "CommandGovernorSafetyFilter",
    "FixedXTransportLQRConfig",
    "FixedXTransportLQRController",
    "CartPoleFallbackConfig",
    "CartPoleFallbackController",
    "CartPoleLQRConfig",
    "CartPoleLQRController",
    "CartPoleMPCConfig",
    "CartPoleMPCController",
    "CartesianImpedanceConfig",
    "CartesianImpedanceOutput",
    "XAxisCartesianImpedanceController",
    "TorqueCommandFilter",
    "ImpedanceSafetyConfig",
    "ImpedanceSafetyMonitor",
    "ImpedanceSafetyStatus",
    "JsonlTraceWriter",
    "json_dumps_safe",
    "cartesian_force_to_joint_torque",
    "XAxisController",
    "XAxisControllerConfig",
    "SafetyConfig",
    "SafetyMonitor",
    "SafetyStatus",
]
