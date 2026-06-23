"""Heuristic recoverability monitor for the constrained cart-pole scaffold.

This is intentionally conservative and explicitly not a learned certificate,
Hamilton-Jacobi barrier, or conformal recoverability proof.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .controller_interfaces import ControllerCommand, ControllerState, InterventionLevel, RecoverabilityMonitor, SafetyLimits


def _clamp01(value: float) -> float:
    return float(np.clip(float(value), 0.0, 1.0))


@dataclass
class HeuristicRecoverabilityMonitor(RecoverabilityMonitor):
    """Conservative state-and-command heuristic."""

    limits: SafetyLimits

    def __post_init__(self) -> None:
        self.limits.validate()

    def _dt(self, state: ControllerState) -> float:
        dt = float(state.dt_s if np.isfinite(float(state.dt_s)) and state.dt_s > 0.0 else self.limits.dt_s)
        return max(dt, 1e-6)

    def _command_to_predicted_x(self, state: ControllerState, command: ControllerCommand) -> float:
        dt = self._dt(state)
        value = float(command.value)
        if command.mode == "x_acceleration":
            return float(state.x + state.x_dot * dt + 0.5 * value * dt * dt)
        if command.mode == "x_velocity":
            return float(state.x + value * dt)
        if command.mode == "x_position_delta":
            return float(state.x + value)
        raise ValueError(f"Unsupported command mode: {command.mode!r}")

    def _outward_push(self, state: ControllerState, command: ControllerCommand) -> bool:
        if not state.is_finite() or not command.is_finite():
            return False
        margin_left = float(state.x - self.limits.x_min_m)
        margin_right = float(self.limits.x_max_m - state.x)
        if margin_left < 0.0 or margin_right < 0.0:
            return True
        near_left = margin_left <= self.limits.x_warning_margin_m
        near_right = margin_right <= self.limits.x_warning_margin_m
        if not (near_left or near_right):
            return False

        x_next = self._command_to_predicted_x(state, command)
        if near_right and x_next > state.x + 1e-12:
            return True
        if near_left and x_next < state.x - 1e-12:
            return True
        return False

    def would_push_outward(self, state: ControllerState, command: ControllerCommand) -> bool:
        """Public helper used by the safety filter."""
        return self._outward_push(state, command)

    def recoverability_score(
        self,
        state: ControllerState,
        command: ControllerCommand | None = None,
    ) -> float:
        self.limits.validate()
        if not state.is_finite():
            return 0.0
        x_margin_left = float(state.x - self.limits.x_min_m)
        x_margin_right = float(self.limits.x_max_m - state.x)
        if x_margin_left < 0.0 or x_margin_right < 0.0:
            return 0.0

        x_margin = min(x_margin_left, x_margin_right)
        x_score = _clamp01(x_margin / max(self.limits.x_warning_margin_m, 1e-6))

        theta_ratio = abs(float(state.theta)) / max(self.limits.pole_angle_hard_cutoff_rad, 1e-6)
        theta_score = _clamp01(1.0 - theta_ratio)

        theta_dot_ratio = abs(float(state.theta_dot)) / max(self.limits.pole_angular_velocity_cutoff_radps, 1e-6)
        theta_dot_score = _clamp01(1.0 - theta_dot_ratio)

        score = min(x_score, theta_score, theta_dot_score)
        if command is not None and command.is_finite() and self._outward_push(state, command):
            score *= 0.5
        return _clamp01(score)

    def is_recoverable(
        self,
        state: ControllerState,
        command: ControllerCommand | None = None,
    ) -> bool:
        score = self.recoverability_score(state, command)
        return bool(score >= 0.05)

    def intervention_level(
        self,
        state: ControllerState,
        command: ControllerCommand | None = None,
    ) -> InterventionLevel:
        if not state.is_finite():
            return "halt"
        if abs(float(state.theta)) >= self.limits.pole_angle_hard_cutoff_rad:
            return "halt"
        if abs(float(state.theta_dot)) >= self.limits.pole_angular_velocity_cutoff_radps:
            return "halt"
        x_margin_left = float(state.x - self.limits.x_min_m)
        x_margin_right = float(self.limits.x_max_m - state.x)
        if x_margin_left < 0.0 or x_margin_right < 0.0:
            return "halt"

        score = self.recoverability_score(state, command)
        if command is not None and self._outward_push(state, command):
            if min(x_margin_left, x_margin_right) <= self.limits.x_warning_margin_m:
                return "intervene"

        if score < 0.20:
            return "intervene"
        if score < 0.60:
            return "warning"
        return "monitor"
