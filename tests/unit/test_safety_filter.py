"""Unit tests for the cart-pole command governor and recoverability monitor."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from controller_core import (  # noqa: E402
    CommandGovernorSafetyFilter,
    ControllerCommand,
    ControllerState,
    HeuristicRecoverabilityMonitor,
    SafetyLimits,
)
from controller_core.cartpole_linear_model import CartPoleLinearModel  # noqa: E402


def _limits() -> SafetyLimits:
    return SafetyLimits(
        x_min_m=-0.10,
        x_max_m=0.10,
        x_warning_margin_m=0.02,
        max_x_velocity_mps=0.30,
        max_x_acceleration_mps2=0.50,
        max_command_change_per_cycle=0.10,
        pole_angle_hard_cutoff_rad=0.50,
        pole_angular_velocity_cutoff_radps=4.0,
        dt_s=0.01,
        fallback_action="brake",
        brake_gain=1.5,
    )


def test_safety_filter_projects_acceleration_to_workspace_bounds() -> None:
    limits = _limits()
    model = CartPoleLinearModel(pole_length_m=0.5, gravity_mps2=9.81)
    filt = CommandGovernorSafetyFilter(limits)
    state = ControllerState(x=0.095, x_dot=0.15, theta=0.01, theta_dot=0.0, dt_s=0.01)
    raw = ControllerCommand(mode="x_acceleration", value=1.0)

    result = filt.filter(state, raw)

    assert result.command.mode == "x_acceleration"
    assert result.clipped is True
    assert result.rejected is False
    assert result.command.value <= result.details["safe_upper"] + 1e-12
    assert result.command.value >= result.details["safe_lower"] - 1e-12

    predicted = model.predict_next_state(state, result.command.value)
    assert predicted.x <= limits.x_max_m + 1e-12
    assert predicted.x >= limits.x_min_m - 1e-12


def test_safety_filter_handles_position_delta_mode() -> None:
    limits = _limits()
    filt = CommandGovernorSafetyFilter(limits)
    state = ControllerState(x=0.095, x_dot=0.0, theta=0.0, theta_dot=0.0, dt_s=0.01)
    raw = ControllerCommand(mode="x_position_delta", value=0.05)

    result = filt.filter(state, raw)

    assert result.command.mode == "x_position_delta"
    assert result.clipped is True
    assert result.command.value <= limits.x_max_m - state.x + 1e-12
    assert result.command.value >= limits.x_min_m - state.x - 1e-12


def test_safety_filter_rejects_nonfinite_command_and_returns_fallback() -> None:
    limits = _limits()
    filt = CommandGovernorSafetyFilter(limits)
    state = ControllerState(x=0.0, x_dot=0.0, theta=0.0, theta_dot=0.0, dt_s=0.01)
    raw = ControllerCommand(mode="x_acceleration", value=np.nan)

    result = filt.filter(state, raw)

    assert result.rejected is True
    assert result.severity == "critical"
    assert np.isfinite(result.command.value)


def test_safety_filter_applies_command_change_limit() -> None:
    limits = _limits()
    filt = CommandGovernorSafetyFilter(limits)
    state = ControllerState(x=0.0, x_dot=0.0, theta=0.0, theta_dot=0.0, dt_s=0.01)

    first = filt.filter(state, ControllerCommand(mode="x_acceleration", value=0.0))
    second = filt.filter(state, ControllerCommand(mode="x_acceleration", value=1.0))

    assert abs(second.command.value - first.command.value) <= limits.max_command_change_per_cycle + 1e-12
    assert second.clipped is True


def test_safety_filter_halts_on_theta_cutoff() -> None:
    limits = _limits()
    filt = CommandGovernorSafetyFilter(limits)
    state = ControllerState(x=0.0, x_dot=0.0, theta=0.60, theta_dot=0.0, dt_s=0.01)
    raw = ControllerCommand(mode="x_acceleration", value=0.0)

    result = filt.filter(state, raw)

    assert result.rejected is True
    assert result.severity == "critical"
    assert result.recoverability_score == 0.0


def test_recoverability_monitor_scores_outward_command_lower() -> None:
    limits = _limits()
    monitor = HeuristicRecoverabilityMonitor(limits)
    state = ControllerState(x=0.085, x_dot=0.02, theta=0.0, theta_dot=0.0, dt_s=0.01)
    inward = ControllerCommand(mode="x_velocity", value=-0.01)
    outward = ControllerCommand(mode="x_velocity", value=0.20)

    assert monitor.would_push_outward(state, outward) is True
    assert monitor.would_push_outward(state, inward) is False
    assert monitor.recoverability_score(state, outward) < monitor.recoverability_score(state, inward)
    assert monitor.intervention_level(state, outward) in ("warning", "intervene")


if __name__ == "__main__":
    test_safety_filter_projects_acceleration_to_workspace_bounds()
    test_safety_filter_handles_position_delta_mode()
    test_safety_filter_rejects_nonfinite_command_and_returns_fallback()
    test_safety_filter_applies_command_change_limit()
    test_safety_filter_halts_on_theta_cutoff()
    test_recoverability_monitor_scores_outward_command_lower()
    print("safety filter tests OK")
