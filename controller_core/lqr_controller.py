"""Fallback PID/PD and first-serious LQR controllers for the cart-pole scaffold."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .cartpole_linear_model import CartPoleLinearModel, solve_discrete_lqr
from .controller_interfaces import CommandMode, ControllerCommand, ControllerState, NominalController


def _as_vector_or_diag(name: str, value: np.ndarray, length: int) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64)
    if arr.shape == (length,):
        arr = np.diag(arr)
    if arr.shape != (length, length):
        raise ValueError(f"{name} must have shape ({length},) or ({length}, {length}); got {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains NaN/Inf")
    return arr


def _clamp_command(mode: CommandMode, value: float, limit: float) -> float:
    limit = abs(float(limit))
    return float(np.clip(float(value), -limit, +limit))


@dataclass
class CartPoleFallbackConfig:
    """Conservative baseline controller, defaulting to PD with optional x integral."""

    kp_x: float = 2.0
    ki_x: float = 0.0
    kd_x: float = 1.4
    kp_theta: float = 18.0
    kd_theta: float = 4.0
    command_limit: float = 1.5
    output_mode: CommandMode = "x_acceleration"
    integral_limit: float = 0.25
    target_x: float = 0.0
    target_theta: float = 0.0
    dt_s: float = 0.002

    def validate(self) -> None:
        for name in (
            "kp_x",
            "ki_x",
            "kd_x",
            "kp_theta",
            "kd_theta",
            "command_limit",
            "integral_limit",
            "target_x",
            "target_theta",
            "dt_s",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if self.command_limit <= 0.0:
            raise ValueError("command_limit must be positive")
        if self.integral_limit < 0.0:
            raise ValueError("integral_limit must be non-negative")
        if self.dt_s <= 0.0:
            raise ValueError("dt_s must be positive")
        if self.output_mode not in ("x_acceleration", "x_velocity", "x_position_delta"):
            raise ValueError(f"Unsupported output_mode: {self.output_mode!r}")


class CartPoleFallbackController(NominalController):
    """Conservative fallback controller.

    This is intentionally not the primary stabilization path. It exists so the
    architecture can fall back to a simple PD/PI law while LQR becomes the main
    stabilizer.
    """

    def __init__(self, config: CartPoleFallbackConfig | None = None) -> None:
        self.cfg = config if config is not None else CartPoleFallbackConfig()
        self.cfg.validate()
        self._x_integral = 0.0

    def reset(self) -> None:
        self._x_integral = 0.0

    def _raw_acceleration(self, state: ControllerState) -> float:
        state.validate()
        x_error = float(state.x - self.cfg.target_x)
        theta_error = float(state.theta - self.cfg.target_theta)
        self._x_integral += x_error * float(state.dt_s)
        self._x_integral = float(np.clip(self._x_integral, -self.cfg.integral_limit, self.cfg.integral_limit))

        accel = (
            -self.cfg.kp_x * x_error
            - self.cfg.kd_x * float(state.x_dot)
            - self.cfg.ki_x * self._x_integral
            + self.cfg.kp_theta * theta_error
            + self.cfg.kd_theta * float(state.theta_dot)
        )
        if not np.isfinite(accel):
            raise RuntimeError("Fallback controller produced a non-finite acceleration command")
        return float(accel)

    def compute(self, state: ControllerState) -> ControllerCommand:
        accel = self._raw_acceleration(state)
        value: float
        if self.cfg.output_mode == "x_acceleration":
            value = accel
        elif self.cfg.output_mode == "x_velocity":
            value = float(state.x_dot + accel * float(state.dt_s))
        else:
            dt = float(state.dt_s)
            value = float(state.x_dot * dt + 0.5 * accel * dt * dt)
        value = _clamp_command(self.cfg.output_mode, value, self.cfg.command_limit)
        return ControllerCommand(
            mode=self.cfg.output_mode,
            value=value,
            time_s=float(state.time_s),
            metadata={
                "controller": "fallback_pd",
                "acceleration_mps2": accel,
                "kp_x": self.cfg.kp_x,
                "ki_x": self.cfg.ki_x,
                "kd_x": self.cfg.kd_x,
                "kp_theta": self.cfg.kp_theta,
                "kd_theta": self.cfg.kd_theta,
            },
        )


@dataclass
class CartPoleLQRConfig:
    """Discrete LQR configuration for the normalized acceleration-level model."""

    q_weights: np.ndarray = field(default_factory=lambda: np.array([30.0, 8.0, 120.0, 12.0], dtype=np.float64))
    r_weight: float = 1.0
    pole_length_m: float = 0.5
    gravity_mps2: float = 9.81
    output_mode: CommandMode = "x_acceleration"
    command_limit: float = 1.5
    target_x: float = 0.0
    target_theta: float = 0.0
    dt_s: float = 0.002
    riccati_max_iters: int = 2000
    riccati_tol: float = 1e-10

    def validate(self) -> None:
        _ = _as_vector_or_diag("q_weights", self.q_weights, 4)
        for name in ("r_weight", "pole_length_m", "gravity_mps2", "command_limit", "target_x", "target_theta", "dt_s", "riccati_tol"):
            value = float(getattr(self, name))
            if not np.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if self.r_weight <= 0.0:
            raise ValueError("r_weight must be positive")
        if self.pole_length_m <= 0.0:
            raise ValueError("pole_length_m must be positive")
        if self.gravity_mps2 <= 0.0:
            raise ValueError("gravity_mps2 must be positive")
        if self.command_limit <= 0.0:
            raise ValueError("command_limit must be positive")
        if self.dt_s <= 0.0:
            raise ValueError("dt_s must be positive")
        if self.riccati_max_iters <= 0:
            raise ValueError("riccati_max_iters must be positive")
        if self.output_mode not in ("x_acceleration", "x_velocity", "x_position_delta"):
            raise ValueError(f"Unsupported output_mode: {self.output_mode!r}")


class CartPoleLQRController(NominalController):
    """Discrete-time LQR controller over the normalized acceleration-level model."""

    def __init__(self, config: CartPoleLQRConfig | None = None) -> None:
        self.cfg = config if config is not None else CartPoleLQRConfig()
        self.cfg.validate()
        self.model = CartPoleLinearModel(
            pole_length_m=float(self.cfg.pole_length_m),
            gravity_mps2=float(self.cfg.gravity_mps2),
        )
        a_d, b_d = self.model.discrete_matrices(self.cfg.dt_s)
        q = np.asarray(self.cfg.q_weights, dtype=np.float64)
        r = np.array([[float(self.cfg.r_weight)]], dtype=np.float64)
        self.gain_matrix, self.riccati_solution, self.riccati_converged, self.riccati_iters = solve_discrete_lqr(
            a_d,
            b_d,
            q,
            r,
            max_iters=int(self.cfg.riccati_max_iters),
            tol=float(self.cfg.riccati_tol),
        )

    def _raw_acceleration(self, state: ControllerState) -> float:
        state.validate()
        error = np.array(
            [
                float(state.x - self.cfg.target_x),
                float(state.x_dot),
                float(state.theta - self.cfg.target_theta),
                float(state.theta_dot),
            ],
            dtype=np.float64,
        )
        accel = -float((self.gain_matrix @ error.reshape(-1, 1))[0, 0])
        if not np.isfinite(accel):
            raise RuntimeError("LQR controller produced a non-finite acceleration command")
        return accel

    def compute(self, state: ControllerState) -> ControllerCommand:
        accel = self._raw_acceleration(state)
        if self.cfg.output_mode == "x_acceleration":
            value = accel
        elif self.cfg.output_mode == "x_velocity":
            value = float(state.x_dot + accel * float(state.dt_s))
        else:
            dt = float(state.dt_s)
            value = float(state.x_dot * dt + 0.5 * accel * dt * dt)
        value = _clamp_command(self.cfg.output_mode, value, self.cfg.command_limit)
        return ControllerCommand(
            mode=self.cfg.output_mode,
            value=value,
            time_s=float(state.time_s),
            metadata={
                "controller": "lqr",
                "acceleration_mps2": accel,
                "gain_matrix": self.gain_matrix.tolist(),
                "riccati_converged": self.riccati_converged,
                "riccati_iters": self.riccati_iters,
            },
        )
