"""CoppeliaSim cart-pole MPC outer-loop helpers."""

from __future__ import annotations

from typing import Any

import numpy as np

from controller_core import (
    CartPoleMPCConfig,
    CartPoleMPCController,
    CommandGovernorSafetyFilter,
    ControllerState,
    SafetyLimits,
)


def build_coppelia_mpc_transport(
    *,
    x_start: float,
    target_dx: float,
    dt_s: float,
    horizon: int,
    q_weights: np.ndarray,
    q_terminal_scale: float,
    r_weight: float,
    pole_length_m: float,
    accel_limit: float,
    velocity_limit: float,
    command_change_limit: float,
    guardrail_margin_m: float,
) -> tuple[CartPoleMPCController, CommandGovernorSafetyFilter, float]:
    """Build the cart-pole MPC outer loop and command governor."""
    x_start = float(x_start)
    target_dx = float(target_dx)
    target_x = float(x_start + target_dx)
    dt_s = float(dt_s)
    accel_limit = abs(float(accel_limit))
    velocity_limit = abs(float(velocity_limit))
    command_change_limit = abs(float(command_change_limit))
    guardrail_margin_m = max(float(guardrail_margin_m), 0.0)

    mpc_cfg = CartPoleMPCConfig(
        horizon=int(horizon),
        q_weights=np.asarray(q_weights, dtype=np.float64).reshape(4),
        q_terminal_scale=float(q_terminal_scale),
        r_weight=float(r_weight),
        pole_length_m=float(pole_length_m),
        dt_s=dt_s,
        target_x=target_x,
        command_limit=accel_limit,
        output_mode="x_acceleration",
    )
    mpc_controller = CartPoleMPCController(mpc_cfg)

    x_min = float(min(x_start, target_x) - guardrail_margin_m)
    x_max = float(max(x_start, target_x) + guardrail_margin_m)
    mpc_limits = SafetyLimits(
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
    mpc_filter = CommandGovernorSafetyFilter(mpc_limits)
    return mpc_controller, mpc_filter, target_x


def compute_coppelia_mpc_outer_command(
    controller: CartPoleMPCController,
    safety_filter: CommandGovernorSafetyFilter | None,
    *,
    x_now: float,
    x_dot_now: float,
    theta_now: float,
    theta_dot_now: float,
    time_s: float,
    dt_s: float,
    target_x: float,
) -> tuple[float, dict[str, Any]]:
    """Compute a safe MPC outer-loop acceleration command."""
    state = ControllerState(
        x=float(x_now),
        x_dot=float(x_dot_now),
        theta=float(theta_now),
        theta_dot=float(theta_dot_now),
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
            "theta_now_rad": float(theta_now),
            "theta_dot_now_radps": float(theta_dot_now),
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
        "theta_now_rad": float(theta_now),
        "theta_dot_now_radps": float(theta_dot_now),
        "mpc_horizon": int(controller.cfg.horizon),
    }
    return float(safe_result.command.value), diag
