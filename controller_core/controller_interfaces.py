"""Core controller and safety interfaces for constrained cart-pole control.

This module is intentionally hardware-agnostic. It defines the common state
and command objects used by the simulation-only constrained-control stack.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

CommandMode = Literal["x_acceleration", "x_velocity", "x_position_delta"]
SafetySeverity = Literal["ok", "warning", "clipped", "rejected", "critical"]
InterventionLevel = Literal["monitor", "warning", "intervene", "halt"]
FallbackAction = Literal["hold", "brake"]


def _as_finite_scalar(name: str, value: Any) -> float:
    scalar = float(value)
    if not np.isfinite(scalar):
        raise ValueError(f"{name} must be finite")
    return scalar


def _as_finite_vector(name: str, value: Any, length: int) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64).reshape(-1)
    if arr.shape[0] != length:
        raise ValueError(f"{name} must have length {length}; got shape {np.asarray(value).shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains NaN/Inf")
    return arr


@dataclass
class ControllerState:
    """Minimal 4D cart-pole state plus optional reference metadata.

    The dynamic state is ``[x, x_dot, theta, theta_dot]``.

    Sign convention:
    - ``theta == 0`` is the upright equilibrium.
    - ``theta > 0`` means the pole leans toward world ``+x``.
    - ``x > 0`` means the cart is to world ``+x``.
    """

    x: float
    x_dot: float
    theta: float
    theta_dot: float
    time_s: float = 0.0
    dt_s: float = 0.002
    target_x: float = 0.0
    target_theta: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def as_vector(self) -> np.ndarray:
        return np.array([self.x, self.x_dot, self.theta, self.theta_dot], dtype=np.float64)

    def error_vector(self) -> np.ndarray:
        return np.array(
            [
                self.x - self.target_x,
                self.x_dot,
                self.theta - self.target_theta,
                self.theta_dot,
            ],
            dtype=np.float64,
        )

    def is_finite(self) -> bool:
        return bool(
            np.all(np.isfinite(self.as_vector()))
            and np.isfinite(float(self.time_s))
            and np.isfinite(float(self.dt_s))
            and np.isfinite(float(self.target_x))
            and np.isfinite(float(self.target_theta))
        )

    def validate(self) -> None:
        _ = _as_finite_scalar("time_s", self.time_s)
        dt = _as_finite_scalar("dt_s", self.dt_s)
        if dt <= 0.0:
            raise ValueError("dt_s must be positive")
        _ = _as_finite_vector("state", self.as_vector(), 4)
        _ = _as_finite_scalar("target_x", self.target_x)
        _ = _as_finite_scalar("target_theta", self.target_theta)


@dataclass
class ControllerCommand:
    """Safe controller command in one of the cart x abstractions."""

    mode: CommandMode
    value: float
    time_s: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if self.mode not in ("x_acceleration", "x_velocity", "x_position_delta"):
            raise ValueError(f"Unsupported command mode: {self.mode!r}")
        _ = _as_finite_scalar("command.value", self.value)
        if self.time_s is not None:
            _ = _as_finite_scalar("command.time_s", self.time_s)

    def is_finite(self) -> bool:
        return bool(np.isfinite(float(self.value)) and (self.time_s is None or np.isfinite(float(self.time_s))))

    def as_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"mode": self.mode, "value": float(self.value)}
        if self.time_s is not None:
            out["time_s"] = float(self.time_s)
        if self.metadata:
            out["metadata"] = self.metadata
        return out


@dataclass
class SafetyLimits:
    """Mode-agnostic limits for the cart-pole command governor."""

    x_min_m: float = -0.35
    x_max_m: float = 0.35
    x_warning_margin_m: float = 0.05
    max_x_velocity_mps: float = 0.60
    max_x_acceleration_mps2: float = 1.50
    max_command_change_per_cycle: float = 0.20
    pole_angle_hard_cutoff_rad: float = 0.65
    pole_angular_velocity_cutoff_radps: float = 5.0
    dt_s: float = 0.002
    reject_on_violation: bool = True
    fallback_action: FallbackAction = "brake"
    brake_gain: float = 1.5

    def validate(self) -> None:
        for name in (
            "x_min_m",
            "x_max_m",
            "x_warning_margin_m",
            "max_x_velocity_mps",
            "max_x_acceleration_mps2",
            "max_command_change_per_cycle",
            "pole_angle_hard_cutoff_rad",
            "pole_angular_velocity_cutoff_radps",
            "dt_s",
            "brake_gain",
        ):
            _ = _as_finite_scalar(name, getattr(self, name))
        if self.x_min_m >= self.x_max_m:
            raise ValueError("x_min_m must be smaller than x_max_m")
        if self.x_warning_margin_m < 0.0:
            raise ValueError("x_warning_margin_m must be non-negative")
        if self.max_x_velocity_mps <= 0.0:
            raise ValueError("max_x_velocity_mps must be positive")
        if self.max_x_acceleration_mps2 <= 0.0:
            raise ValueError("max_x_acceleration_mps2 must be positive")
        if self.max_command_change_per_cycle < 0.0:
            raise ValueError("max_command_change_per_cycle must be non-negative")
        if self.pole_angle_hard_cutoff_rad <= 0.0:
            raise ValueError("pole_angle_hard_cutoff_rad must be positive")
        if self.pole_angular_velocity_cutoff_radps <= 0.0:
            raise ValueError("pole_angular_velocity_cutoff_radps must be positive")
        if self.dt_s <= 0.0:
            raise ValueError("dt_s must be positive")
        if self.fallback_action not in ("hold", "brake"):
            raise ValueError("fallback_action must be 'hold' or 'brake'")

    @property
    def x_span_m(self) -> float:
        return float(self.x_max_m - self.x_min_m)


@dataclass
class SafetyFilterResult:
    """Result returned by a safety filter after clipping or rejection."""

    raw_command: ControllerCommand
    command: ControllerCommand
    clipped: bool
    rejected: bool
    reasons: list[str] = field(default_factory=list)
    severity: SafetySeverity = "ok"
    recoverability_score: float | None = None
    intervention_level: InterventionLevel | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "raw_command": self.raw_command.as_dict(),
            "command": self.command.as_dict(),
            "clipped": bool(self.clipped),
            "rejected": bool(self.rejected),
            "reasons": list(self.reasons),
            "severity": self.severity,
        }
        if self.recoverability_score is not None:
            out["recoverability_score"] = float(self.recoverability_score)
        if self.intervention_level is not None:
            out["intervention_level"] = self.intervention_level
        if self.details:
            out["details"] = self.details
        return out


class NominalController(ABC):
    """Interface for the raw controller before safety shaping."""

    @abstractmethod
    def compute(self, state: ControllerState) -> ControllerCommand:
        raise NotImplementedError


class SafetyFilter(ABC):
    """Interface for command-governor style safety shaping."""

    @abstractmethod
    def filter(self, state: ControllerState, raw_command: ControllerCommand) -> SafetyFilterResult:
        raise NotImplementedError


class RecoverabilityMonitor(ABC):
    """Interface for conservative recoverability heuristics."""

    @abstractmethod
    def recoverability_score(
        self,
        state: ControllerState,
        command: ControllerCommand | None = None,
    ) -> float:
        raise NotImplementedError

    @abstractmethod
    def is_recoverable(
        self,
        state: ControllerState,
        command: ControllerCommand | None = None,
    ) -> bool:
        raise NotImplementedError

    @abstractmethod
    def intervention_level(
        self,
        state: ControllerState,
        command: ControllerCommand | None = None,
    ) -> InterventionLevel:
        raise NotImplementedError
