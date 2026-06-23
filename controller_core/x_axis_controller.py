"""
Backend-independent X-axis controller.

Control law (per the Stage 6 spec):

    x_error = target_x - ee_pos[0]
    Fx      = Kp_x * x_error - Kd_x * ee_vx

``Fx`` is a task-space force in Newtons. Conversion to joint torques is done
by the J^T adapter in ``kinematics_utils.cartesian_force_to_joint_torque``.
Joint damping and optional gravity compensation also live in the adapter so
the controller core itself stays purely task-space.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .state_types import ControlOutput, RobotState


@dataclass
class XAxisControllerConfig:
    """Gains and bounds for the Cartesian-X PD."""

    kp_x: float = 300.0  # N/m
    kd_x: float = 60.0   # N/(m/s)
    fx_max_n: float = 50.0
    deadband_m: float = 0.0
    # Optional feed-forward X velocity used only for logging (not added to Fx).
    ff_vx: float = 0.0

    def validate(self) -> None:
        if self.kp_x < 0 or self.kd_x < 0:
            raise ValueError("kp_x and kd_x must be non-negative.")
        if self.fx_max_n <= 0:
            raise ValueError("fx_max_n must be positive.")
        if self.deadband_m < 0:
            raise ValueError("deadband_m must be non-negative.")


@dataclass
class XAxisController:
    """Simple PD on world-X position of the end effector.

    The controller output carries ``mode == "cartesian_x_force"``. It does
    not know about joints. Feed it a ``RobotState`` (as produced by any
    adapter) and it returns the task-space X force together with diagnostics.
    """

    config: XAxisControllerConfig = field(default_factory=XAxisControllerConfig)

    def __post_init__(self) -> None:
        self.config.validate()

    def compute(self, state: RobotState) -> ControlOutput:
        target_x = float(state["target_x"])
        ee_pos = np.asarray(state["ee_pos"], dtype=np.float64)
        x_error = target_x - float(ee_pos[0])

        # Prefer the measured EE linear velocity if the adapter provided it.
        # Otherwise fall back to J_pos @ qd, and finally to zero.
        ee_vx = 0.0
        if "ee_lin_vel" in state:
            ee_vx = float(np.asarray(state["ee_lin_vel"])[0])
        elif "jacobian_pos" in state:
            j_pos = np.asarray(state["jacobian_pos"], dtype=np.float64)
            qd = np.asarray(state["qd"], dtype=np.float64)
            ee_vx = float((j_pos @ qd)[0])

        applied_error = x_error
        if abs(applied_error) < self.config.deadband_m:
            applied_error = 0.0

        fx_unsat = self.config.kp_x * applied_error - self.config.kd_x * ee_vx
        fx = float(np.clip(fx_unsat, -self.config.fx_max_n, +self.config.fx_max_n))
        saturated = bool(abs(fx_unsat) > self.config.fx_max_n)

        return ControlOutput(
            mode="cartesian_x_force",
            fx=fx,
            x_error=float(x_error),
            ee_vx=ee_vx,
            saturated=saturated,
        )
