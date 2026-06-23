"""Command governor for the constrained cart-pole scaffold."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .controller_interfaces import (
    CommandMode,
    ControllerCommand,
    ControllerState,
    SafetyFilter,
    SafetyFilterResult,
    SafetyLimits,
)
from .recoverability_monitor import HeuristicRecoverabilityMonitor


def _clamp(value: float, lower: float, upper: float) -> float:
    return float(np.clip(float(value), float(lower), float(upper)))


def _safe_dt(state: ControllerState, limits: SafetyLimits) -> float:
    dt = float(state.dt_s if np.isfinite(float(state.dt_s)) and state.dt_s > 0.0 else limits.dt_s)
    return max(dt, 1e-6)


@dataclass
class CommandGovernorSafetyFilter(SafetyFilter):
    """Clip, project, or reject unsafe commands before they hit a downstream layer."""

    limits: SafetyLimits
    recoverability_monitor: HeuristicRecoverabilityMonitor | None = None

    def __post_init__(self) -> None:
        self.limits.validate()
        if self.recoverability_monitor is None:
            self.recoverability_monitor = HeuristicRecoverabilityMonitor(self.limits)
        self._prev_safe_command: ControllerCommand | None = None

    def _zero_command(self, mode: CommandMode, *, time_s: float | None = None) -> ControllerCommand:
        return ControllerCommand(mode=mode, value=0.0, time_s=time_s, metadata={"fallback_action": "hold"})

    def _brake_command(self, state: ControllerState, mode: CommandMode) -> ControllerCommand:
        dt = _safe_dt(state, self.limits)
        if mode == "x_acceleration":
            value = -self.limits.brake_gain * float(state.x_dot)
            value = _clamp(value, -self.limits.max_x_acceleration_mps2, self.limits.max_x_acceleration_mps2)
        elif mode == "x_velocity":
            value = -self.limits.brake_gain * float(state.x_dot)
            value = _clamp(value, -self.limits.max_x_velocity_mps, self.limits.max_x_velocity_mps)
        elif mode == "x_position_delta":
            value = -self.limits.brake_gain * float(state.x_dot) * dt
            max_delta = self.limits.max_x_velocity_mps * dt
            value = _clamp(value, -max_delta, max_delta)
        else:  # pragma: no cover - validated by command.mode
            raise ValueError(f"Unsupported command mode: {mode!r}")
        return ControllerCommand(mode=mode, value=value, time_s=state.time_s, metadata={"fallback_action": "brake"})

    def _mode_absolute_limit(self, mode: CommandMode, dt: float) -> float:
        if mode == "x_acceleration":
            return float(self.limits.max_x_acceleration_mps2)
        if mode == "x_velocity":
            return float(self.limits.max_x_velocity_mps)
        if mode == "x_position_delta":
            return float(self.limits.max_x_velocity_mps * dt)
        raise ValueError(f"Unsupported command mode: {mode!r}")

    def _mode_change_limit(self) -> float:
        return float(self.limits.max_command_change_per_cycle)

    def _predict_x_next(self, state: ControllerState, command: ControllerCommand, dt: float) -> tuple[float, float]:
        if command.mode == "x_acceleration":
            accel = float(command.value)
            x_next = float(state.x + state.x_dot * dt + 0.5 * accel * dt * dt)
            x_dot_next = float(state.x_dot + accel * dt)
            return x_next, x_dot_next
        if command.mode == "x_velocity":
            v = float(command.value)
            return float(state.x + v * dt), v
        if command.mode == "x_position_delta":
            dx = float(command.value)
            v = dx / dt
            return float(state.x + dx), v
        raise ValueError(f"Unsupported command mode: {command.mode!r}")

    def _safe_interval(self, state: ControllerState, command: ControllerCommand, dt: float) -> tuple[float, float]:
        lower = -self._mode_absolute_limit(command.mode, dt)
        upper = +self._mode_absolute_limit(command.mode, dt)

        # Workspace x bounds become command bounds through a one-step prediction.
        if command.mode == "x_acceleration":
            pos_lower = 2.0 * (self.limits.x_min_m - state.x - state.x_dot * dt) / (dt * dt)
            pos_upper = 2.0 * (self.limits.x_max_m - state.x - state.x_dot * dt) / (dt * dt)
            vel_lower = (-self.limits.max_x_velocity_mps - state.x_dot) / dt
            vel_upper = (+self.limits.max_x_velocity_mps - state.x_dot) / dt
            lower = max(lower, pos_lower, vel_lower)
            upper = min(upper, pos_upper, vel_upper)
        elif command.mode == "x_velocity":
            pos_lower = (self.limits.x_min_m - state.x) / dt
            pos_upper = (self.limits.x_max_m - state.x) / dt
            accel_lower = state.x_dot - self.limits.max_x_acceleration_mps2 * dt
            accel_upper = state.x_dot + self.limits.max_x_acceleration_mps2 * dt
            lower = max(lower, pos_lower, accel_lower)
            upper = min(upper, pos_upper, accel_upper)
        elif command.mode == "x_position_delta":
            pos_lower = self.limits.x_min_m - state.x
            pos_upper = self.limits.x_max_m - state.x
            vel_lower = -self.limits.max_x_velocity_mps * dt
            vel_upper = +self.limits.max_x_velocity_mps * dt
            accel_lower = (state.x_dot - self.limits.max_x_acceleration_mps2 * dt) * dt
            accel_upper = (state.x_dot + self.limits.max_x_acceleration_mps2 * dt) * dt
            lower = max(lower, pos_lower, vel_lower, accel_lower)
            upper = min(upper, pos_upper, vel_upper, accel_upper)
        else:  # pragma: no cover - validated by command.mode
            raise ValueError(f"Unsupported command mode: {command.mode!r}")

        if self._prev_safe_command is not None and self._prev_safe_command.mode == command.mode:
            prev = float(self._prev_safe_command.value)
            delta = self._mode_change_limit()
            lower = max(lower, prev - delta)
            upper = min(upper, prev + delta)
        return lower, upper

    def _severity(self, *, clipped: bool, rejected: bool, recoverability_level: str, reasons: list[str]) -> str:
        if rejected and reasons:
            if any("non-finite" in reason or "halt" in reason for reason in reasons):
                return "critical"
            return "rejected"
        if clipped:
            return "warning" if recoverability_level == "warning" else "clipped"
        return "warning" if recoverability_level == "warning" else "ok"

    def filter(self, state: ControllerState, raw_command: ControllerCommand) -> SafetyFilterResult:
        self.limits.validate()
        dt = _safe_dt(state, self.limits)

        if raw_command.mode not in ("x_acceleration", "x_velocity", "x_position_delta"):
            raise ValueError(f"Unsupported command mode: {raw_command.mode!r}")

        if self.recoverability_monitor is None:  # pragma: no cover - defensive
            self.recoverability_monitor = HeuristicRecoverabilityMonitor(self.limits)

        reasons: list[str] = []
        clipped = False
        rejected = False

        score = self.recoverability_monitor.recoverability_score(state, raw_command)
        level = self.recoverability_monitor.intervention_level(state, raw_command)

        if not raw_command.is_finite() or not state.is_finite():
            reasons.append("non-finite state or command")
            rejected = True
            safe_command = self._brake_command(state, raw_command.mode) if self.limits.fallback_action == "brake" else self._zero_command(raw_command.mode, time_s=state.time_s)
            result = SafetyFilterResult(
                raw_command=raw_command,
                command=safe_command,
                clipped=False,
                rejected=True,
                reasons=reasons,
                severity="critical",
                recoverability_score=score,
                intervention_level=level,
                details={"dt_s": dt},
            )
            self._prev_safe_command = safe_command
            return result

        if abs(float(state.theta)) >= self.limits.pole_angle_hard_cutoff_rad:
            reasons.append(
                f"pole angle hard cutoff exceeded: |theta|={abs(float(state.theta)):.6f} rad"
            )
            rejected = True
        if abs(float(state.theta_dot)) >= self.limits.pole_angular_velocity_cutoff_radps:
            reasons.append(
                f"pole angular velocity cutoff exceeded: |theta_dot|={abs(float(state.theta_dot)):.6f} rad/s"
            )
            rejected = True

        if rejected:
            if self.limits.fallback_action == "brake":
                safe_command = self._brake_command(state, raw_command.mode)
            else:
                safe_command = self._zero_command(raw_command.mode, time_s=state.time_s)
            result = SafetyFilterResult(
                raw_command=raw_command,
                command=safe_command,
                clipped=False,
                rejected=True,
                reasons=reasons,
                severity="critical",
                recoverability_score=score,
                intervention_level=level,
                details={"dt_s": dt, "state_x": state.x, "state_theta": state.theta},
            )
            self._prev_safe_command = safe_command
            return result

        lower, upper = self._safe_interval(state, raw_command, dt)

        if not np.isfinite(lower) or not np.isfinite(upper) or lower > upper:
            reasons.append(
                f"empty safe interval for {raw_command.mode}: lower={lower:.6f}, upper={upper:.6f}"
            )
            rejected = True
            safe_command = self._brake_command(state, raw_command.mode) if self.limits.fallback_action == "brake" else self._zero_command(raw_command.mode, time_s=state.time_s)
        else:
            value = float(raw_command.value)
            clipped_value = _clamp(value, lower, upper)
            clipped = clipped or abs(clipped_value - value) > 1e-12
            if clipped:
                reasons.append(
                    f"{raw_command.mode} clipped from {value:.6f} to {clipped_value:.6f}"
                )
            safe_command = ControllerCommand(
                mode=raw_command.mode,
                value=clipped_value,
                time_s=raw_command.time_s,
                metadata={
                    **dict(raw_command.metadata),
                    "safe_interval_lower": lower,
                    "safe_interval_upper": upper,
                    "recoverability_score": score,
                    "intervention_level": level,
                },
            )

        outward_push = self.recoverability_monitor.would_push_outward(state, raw_command)
        if outward_push:
            clipped = True
            reasons.append("command pushes farther into unsafe region")
            if level in ("intervene", "halt"):
                if level == "halt":
                    rejected = True
                    safe_command = self._brake_command(state, raw_command.mode) if self.limits.fallback_action == "brake" else self._zero_command(raw_command.mode, time_s=state.time_s)

        if self._prev_safe_command is not None and self._prev_safe_command.mode == safe_command.mode:
            delta = abs(float(safe_command.value) - float(self._prev_safe_command.value))
            if delta > self.limits.max_command_change_per_cycle + 1e-12:
                clipped = True
                target = float(self._prev_safe_command.value)
                next_lower = target - self.limits.max_command_change_per_cycle
                next_upper = target + self.limits.max_command_change_per_cycle
                safe_value = _clamp(float(safe_command.value), next_lower, next_upper)
                reasons.append(
                    f"command-change clipped to [{next_lower:.6f}, {next_upper:.6f}]"
                )
                safe_command = ControllerCommand(
                    mode=safe_command.mode,
                    value=safe_value,
                    time_s=safe_command.time_s,
                    metadata=dict(safe_command.metadata),
                )

        severity = self._severity(
            clipped=clipped,
            rejected=rejected,
            recoverability_level=level,
            reasons=reasons,
        )

        result = SafetyFilterResult(
            raw_command=raw_command,
            command=safe_command,
            clipped=bool(clipped),
            rejected=bool(rejected),
            reasons=reasons,
            severity=severity,
            recoverability_score=score,
            intervention_level=level,
            details={
                "dt_s": dt,
                "safe_lower": lower,
                "safe_upper": upper,
                "state_x": state.x,
                "state_x_dot": state.x_dot,
                "state_theta": state.theta,
                "state_theta_dot": state.theta_dot,
            },
        )
        self._prev_safe_command = safe_command
        return result
