"""Simulation-only torque shadow model for sim-to-real hardening.

The goal is to make the simulation actuation path less ideal without touching
any hardware-facing code. This model can inject command latency, slew-rate
limits, under-delivery, deadzone, and simple friction mismatch into the
commanded joint torques before they are applied to MuJoCo.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np


def _as_vector(name: str, value: Any, length: int, *, allow_inf: bool = False) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim == 0:
        arr = np.full(length, float(arr), dtype=np.float64)
    else:
        arr = arr.reshape(-1)
        if arr.shape[0] != length:
            raise ValueError(f"{name} must have length {length}; got shape {arr.shape}")
    if allow_inf:
        finite_or_inf = np.isfinite(arr) | np.isinf(arr)
        if not np.all(finite_or_inf):
            raise ValueError(f"{name} contains NaN")
    else:
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"{name} contains NaN/Inf")
    return arr


def _as_scalar(name: str, value: Any) -> float:
    scalar = float(value)
    if not np.isfinite(scalar):
        raise ValueError(f"{name} must be finite")
    return scalar


@dataclass
class HardwareShadowConfig:
    """Parameters for the simulation-only torque shadow model."""

    tau_max_nm: np.ndarray = field(
        default_factory=lambda: np.array([150.0, 150.0, 150.0, 28.0, 28.0, 28.0], dtype=np.float64)
    )
    command_delay_steps: int = 0
    torque_scale: float = 1.0
    torque_rate_limit_nm_per_s: np.ndarray = field(
        default_factory=lambda: np.full(6, np.inf, dtype=np.float64)
    )
    viscous_damping_nm_per_rad_s: np.ndarray = field(
        default_factory=lambda: np.zeros(6, dtype=np.float64)
    )
    coulomb_friction_nm: np.ndarray = field(default_factory=lambda: np.zeros(6, dtype=np.float64))
    deadzone_nm: np.ndarray = field(default_factory=lambda: np.zeros(6, dtype=np.float64))
    friction_velocity_eps_rad_s: float = 1e-3
    dt_s: float = 0.002

    def validate(self) -> None:
        tau_max = _as_vector("tau_max_nm", self.tau_max_nm, 6)
        if np.any(tau_max <= 0.0):
            raise ValueError("tau_max_nm must be strictly positive")
        if int(self.command_delay_steps) < 0:
            raise ValueError("command_delay_steps must be non-negative")
        _ = _as_scalar("torque_scale", self.torque_scale)
        if self.torque_scale <= 0.0:
            raise ValueError("torque_scale must be positive")
        rate = _as_vector("torque_rate_limit_nm_per_s", self.torque_rate_limit_nm_per_s, 6, allow_inf=True)
        if np.any((rate <= 0.0) & ~np.isinf(rate)):
            raise ValueError("torque_rate_limit_nm_per_s must be positive or inf")
        viscous = _as_vector("viscous_damping_nm_per_rad_s", self.viscous_damping_nm_per_rad_s, 6)
        if np.any(viscous < 0.0):
            raise ValueError("viscous_damping_nm_per_rad_s must be non-negative")
        coulomb = _as_vector("coulomb_friction_nm", self.coulomb_friction_nm, 6)
        if np.any(coulomb < 0.0):
            raise ValueError("coulomb_friction_nm must be non-negative")
        deadzone = _as_vector("deadzone_nm", self.deadzone_nm, 6)
        if np.any(deadzone < 0.0):
            raise ValueError("deadzone_nm must be non-negative")
        eps = _as_scalar("friction_velocity_eps_rad_s", self.friction_velocity_eps_rad_s)
        if eps <= 0.0:
            raise ValueError("friction_velocity_eps_rad_s must be positive")
        dt = _as_scalar("dt_s", self.dt_s)
        if dt <= 0.0:
            raise ValueError("dt_s must be positive")


@dataclass
class HardwareShadowOutput:
    """Result of passing commanded torques through the shadow model."""

    tau_command_nm: np.ndarray
    tau_after_delay_nm: np.ndarray
    tau_after_rate_limit_nm: np.ndarray
    tau_after_deadzone_nm: np.ndarray
    friction_torque_nm: np.ndarray
    tau_applied_nm: np.ndarray
    clipped: bool
    delayed: bool
    rate_limited: bool
    deadzone_applied: bool
    friction_applied: bool
    queue_depth: int
    reasons: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "tau_command_nm": self.tau_command_nm.tolist(),
            "tau_after_delay_nm": self.tau_after_delay_nm.tolist(),
            "tau_after_rate_limit_nm": self.tau_after_rate_limit_nm.tolist(),
            "tau_after_deadzone_nm": self.tau_after_deadzone_nm.tolist(),
            "friction_torque_nm": self.friction_torque_nm.tolist(),
            "tau_applied_nm": self.tau_applied_nm.tolist(),
            "clipped": bool(self.clipped),
            "delayed": bool(self.delayed),
            "rate_limited": bool(self.rate_limited),
            "deadzone_applied": bool(self.deadzone_applied),
            "friction_applied": bool(self.friction_applied),
            "queue_depth": int(self.queue_depth),
            "reasons": list(self.reasons),
        }


class HardwareShadowModel:
    """Pure torque-channel perturbation model used only in simulation."""

    def __init__(self, config: HardwareShadowConfig | None = None) -> None:
        self.cfg = config if config is not None else HardwareShadowConfig()
        self.cfg.validate()
        self._tau_max = _as_vector("tau_max_nm", self.cfg.tau_max_nm, 6)
        self._rate_limit = _as_vector(
            "torque_rate_limit_nm_per_s", self.cfg.torque_rate_limit_nm_per_s, 6, allow_inf=True
        )
        self._viscous = _as_vector("viscous_damping_nm_per_rad_s", self.cfg.viscous_damping_nm_per_rad_s, 6)
        self._coulomb = _as_vector("coulomb_friction_nm", self.cfg.coulomb_friction_nm, 6)
        self._deadzone = _as_vector("deadzone_nm", self.cfg.deadzone_nm, 6)
        self._queue: deque[np.ndarray] = deque()
        self._prev_applied = np.zeros(6, dtype=np.float64)

    def reset(self) -> None:
        self._queue.clear()
        self._prev_applied = np.zeros(6, dtype=np.float64)

    def apply(self, tau_command_nm: Any, qvel: Any | None = None) -> HardwareShadowOutput:
        tau_cmd = _as_vector("tau_command_nm", tau_command_nm, 6)
        qvel_arr = np.zeros(6, dtype=np.float64) if qvel is None else _as_vector("qvel", qvel, 6)

        self._queue.append(tau_cmd.copy())
        delayed = False
        if len(self._queue) > int(self.cfg.command_delay_steps):
            tau_after_delay = self._queue.popleft()
            delayed = int(self.cfg.command_delay_steps) > 0
        else:
            tau_after_delay = np.zeros(6, dtype=np.float64)

        tau_after_scale = tau_after_delay * float(self.cfg.torque_scale)

        rate_step = self._rate_limit * float(self.cfg.dt_s)
        tau_after_rate_limit = np.clip(
            tau_after_scale,
            self._prev_applied - rate_step,
            self._prev_applied + rate_step,
        )
        rate_limited = bool(np.any(np.abs(tau_after_rate_limit - tau_after_scale) > 1e-12))

        tau_after_deadzone = tau_after_rate_limit.copy()
        deadzone_mask = (np.abs(tau_after_deadzone) < self._deadzone) & (np.abs(tau_after_deadzone) > 1e-12)
        deadzone_applied = bool(np.any(deadzone_mask))
        tau_after_deadzone[deadzone_mask] = 0.0

        friction_torque = self._viscous * qvel_arr + self._coulomb * np.tanh(qvel_arr / float(self.cfg.friction_velocity_eps_rad_s))
        friction_applied = bool(np.any(np.abs(friction_torque) > 1e-12))
        tau_preclip = tau_after_deadzone - friction_torque

        tau_applied = np.clip(tau_preclip, -self._tau_max, +self._tau_max)
        clipped = bool(
            delayed
            or rate_limited
            or deadzone_applied
            or friction_applied
            or np.any(np.abs(tau_applied - tau_preclip) > 1e-12)
            or abs(float(self.cfg.torque_scale) - 1.0) > 1e-12
        )
        reasons: list[str] = []
        if delayed:
            reasons.append(f"command delayed by {int(self.cfg.command_delay_steps)} step(s)")
        if abs(float(self.cfg.torque_scale) - 1.0) > 1e-12:
            reasons.append(f"torque scale applied: {float(self.cfg.torque_scale):.6f}")
        if rate_limited:
            reasons.append("torque slew-rate limited")
        if deadzone_applied:
            reasons.append("torque deadzone applied")
        if friction_applied:
            reasons.append("torque friction mismatch applied")
        if np.any(np.abs(tau_applied - tau_preclip) > 1e-12):
            reasons.append("torque saturated by shadow limits")

        self._prev_applied = tau_applied.copy()

        if not np.all(np.isfinite(tau_applied)):
            raise RuntimeError("HardwareShadowModel produced a non-finite torque command")

        return HardwareShadowOutput(
            tau_command_nm=tau_cmd,
            tau_after_delay_nm=tau_after_delay,
            tau_after_rate_limit_nm=tau_after_rate_limit,
            tau_after_deadzone_nm=tau_after_deadzone,
            friction_torque_nm=friction_torque,
            tau_applied_nm=tau_applied,
            clipped=clipped,
            delayed=delayed,
            rate_limited=rate_limited,
            deadzone_applied=deadzone_applied,
            friction_applied=friction_applied,
            queue_depth=len(self._queue),
            reasons=reasons,
        )
