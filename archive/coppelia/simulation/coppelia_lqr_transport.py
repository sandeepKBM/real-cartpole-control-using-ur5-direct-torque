"""CoppeliaSim-only LQR transport helpers.

This module keeps the outer transport controller and its safety governor in a
small testable unit. The runner still owns the CoppeliaSim loop and the inner
torque allocator.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from controller_core import (
    CommandGovernorSafetyFilter,
    ControllerState,
    FixedXTransportLQRConfig,
    FixedXTransportLQRController,
    SafetyLimits,
)


def build_coppelia_lqr_transport(
    *,
    x_start: float,
    target_dx: float,
    dt_s: float,
    q_x: float,
    q_xdot: float,
    r_weight: float,
    accel_limit: float,
    velocity_limit: float,
    command_change_limit: float,
    guardrail_margin_m: float,
) -> tuple[FixedXTransportLQRController, CommandGovernorSafetyFilter, float]:
    """Build the fixed-X LQR outer loop and its command governor."""
    x_start = float(x_start)
    target_dx = float(target_dx)
    target_x = float(x_start + target_dx)
    dt_s = float(dt_s)
    accel_limit = abs(float(accel_limit))
    velocity_limit = abs(float(velocity_limit))
    command_change_limit = abs(float(command_change_limit))
    guardrail_margin_m = max(float(guardrail_margin_m), 0.0)

    lqr_cfg = FixedXTransportLQRConfig(
        q_weights=np.array([float(q_x), float(q_xdot)], dtype=np.float64),
        r_weight=float(r_weight),
        dt_s=dt_s,
        target_x=target_x,
        command_limit=accel_limit,
        output_mode="x_acceleration",
    )
    lqr_controller = FixedXTransportLQRController(lqr_cfg)

    x_min = float(min(x_start, target_x) - guardrail_margin_m)
    x_max = float(max(x_start, target_x) + guardrail_margin_m)
    lqr_limits = SafetyLimits(
        x_min_m=x_min,
        x_max_m=x_max,
        x_warning_margin_m=guardrail_margin_m,
        max_x_velocity_mps=velocity_limit,
        max_x_acceleration_mps2=accel_limit,
        max_command_change_per_cycle=command_change_limit,
        dt_s=dt_s,
        reject_on_violation=True,
        fallback_action="brake",
    )
    lqr_filter = CommandGovernorSafetyFilter(lqr_limits)
    return lqr_controller, lqr_filter, target_x


def compute_coppelia_lqr_outer_command(
    controller: FixedXTransportLQRController,
    safety_filter: CommandGovernorSafetyFilter | None,
    *,
    x_now: float,
    x_dot_now: float,
    time_s: float,
    dt_s: float,
    target_x: float,
) -> tuple[float, dict[str, Any]]:
    """Compute a safe outer-loop acceleration command from the measured axis state."""
    state = ControllerState(
        x=float(x_now),
        x_dot=float(x_dot_now),
        theta=0.0,
        theta_dot=0.0,
        time_s=float(time_s),
        dt_s=float(dt_s),
        target_x=float(target_x),
        target_theta=0.0,
    )
    raw_command = controller.compute(state)
    if safety_filter is None:
        return float(raw_command.value), {
            "raw_command": raw_command.as_dict(),
            "safe_command": raw_command.as_dict(),
            "clipped": False,
            "rejected": False,
            "reasons": [],
            "severity": "ok",
        }

    safe_result = safety_filter.filter(state, raw_command)
    diag = {
        "raw_command": raw_command.as_dict(),
        "safe_command": safe_result.command.as_dict(),
        "clipped": bool(safe_result.clipped),
        "rejected": bool(safe_result.rejected),
        "reasons": list(safe_result.reasons),
        "severity": safe_result.severity,
        "recoverability_score": safe_result.recoverability_score,
        "intervention_level": safe_result.intervention_level,
        "details": dict(safe_result.details),
    }
    return float(safe_result.command.value), diag
