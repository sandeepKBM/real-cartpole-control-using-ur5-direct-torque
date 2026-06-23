"""LQR controller for the current fixed-X transport task.

This is a simulation-only nominal controller for smooth world-X motion. It is
separate from the cart-pole scaffold so the current task can use LQR feedback
without conflating it with future cart-pole work.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .cartpole_linear_model import solve_discrete_lqr
from .controller_interfaces import CommandMode, ControllerCommand, ControllerState, NominalController


def _clamp(value: float, limit: float) -> float:
    limit = abs(float(limit))
    return float(np.clip(float(value), -limit, +limit))


def _command_from_acceleration(
    state: ControllerState,
    accel_cmd_mps2: float,
    mode: CommandMode,
    *,
    command_limit: float | None = None,
) -> ControllerCommand:
    dt = float(state.dt_s)
    accel = float(accel_cmd_mps2)
    if not np.isfinite(accel):
        raise ValueError("accel_cmd_mps2 must be finite")
    if mode == "x_acceleration":
        value = accel
    elif mode == "x_velocity":
        value = float(state.x_dot + accel * dt)
    elif mode == "x_position_delta":
        value = float(state.x_dot * dt + 0.5 * accel * dt * dt)
    else:  # pragma: no cover - guarded by validation
        raise ValueError(f"Unsupported command mode: {mode!r}")

    metadata = {
        "source": "fixed_x_transport_lqr",
        "acceleration_mps2": accel,
        "dt_s": dt,
    }
    if command_limit is not None:
        value = _clamp(value, command_limit)
        metadata["command_limit"] = abs(float(command_limit))
    return ControllerCommand(mode=mode, value=float(value), time_s=float(state.time_s), metadata=metadata)


@dataclass
class FixedXTransportLQRConfig:
    """Discrete LQR configuration for the 1D fixed-X transport model."""

    q_weights: np.ndarray = field(default_factory=lambda: np.array([60.0, 8.0], dtype=np.float64))
    r_weight: float = 1.0
    dt_s: float = 0.002
    target_x: float = 0.0
    command_limit: float = 0.15
    output_mode: CommandMode = "x_acceleration"
    riccati_max_iters: int = 2000
    riccati_tol: float = 1e-10

    def validate(self) -> None:
        q = np.asarray(self.q_weights, dtype=np.float64)
        if q.shape not in ((2,), (2, 2)):
            raise ValueError(f"q_weights must have shape (2,) or (2, 2); got {q.shape}")
        if not np.all(np.isfinite(q)):
            raise ValueError("q_weights contains NaN/Inf")
        for name in ("r_weight", "dt_s", "target_x", "command_limit", "riccati_tol"):
            value = float(getattr(self, name))
            if not np.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if self.r_weight <= 0.0:
            raise ValueError("r_weight must be positive")
        if self.dt_s <= 0.0:
            raise ValueError("dt_s must be positive")
        if self.command_limit <= 0.0:
            raise ValueError("command_limit must be positive")
        if self.riccati_max_iters <= 0:
            raise ValueError("riccati_max_iters must be positive")
        if self.output_mode not in ("x_acceleration", "x_velocity", "x_position_delta"):
            raise ValueError(f"Unsupported output_mode: {self.output_mode!r}")


class FixedXTransportLQRController(NominalController):
    """Discrete LQR over a double-integrator X transport model."""

    def __init__(self, config: FixedXTransportLQRConfig | None = None) -> None:
        self.cfg = config if config is not None else FixedXTransportLQRConfig()
        self.cfg.validate()

        dt = float(self.cfg.dt_s)
        self.a_d = np.array([[1.0, dt], [0.0, 1.0]], dtype=np.float64)
        self.b_d = np.array([[0.5 * dt * dt], [dt]], dtype=np.float64)
        q = np.asarray(self.cfg.q_weights, dtype=np.float64)
        r = np.array([[float(self.cfg.r_weight)]], dtype=np.float64)
        self.gain_matrix, self.riccati_solution, self.riccati_converged, self.riccati_iters = solve_discrete_lqr(
            self.a_d,
            self.b_d,
            q,
            r,
            max_iters=int(self.cfg.riccati_max_iters),
            tol=float(self.cfg.riccati_tol),
        )

    def _raw_acceleration(self, state: ControllerState) -> float:
        state.validate()
        target_x = float(state.target_x if np.isfinite(float(state.target_x)) else self.cfg.target_x)
        error = np.array(
            [
                float(state.x - target_x),
                float(state.x_dot),
            ],
            dtype=np.float64,
        )
        accel = -float((self.gain_matrix @ error.reshape(-1, 1))[0, 0])
        if not np.isfinite(accel):
            raise RuntimeError("FixedXTransportLQRController produced a non-finite acceleration command")
        return accel

    def compute(self, state: ControllerState) -> ControllerCommand:
        accel = self._raw_acceleration(state)
        value = _command_from_acceleration(
            state,
            accel,
            self.cfg.output_mode,
            command_limit=self.cfg.command_limit,
        )
        value.metadata.update(
            {
                "controller": "fixed_x_transport_lqr",
                "gain_matrix": self.gain_matrix.tolist(),
                "riccati_converged": bool(self.riccati_converged),
                "riccati_iters": int(self.riccati_iters),
                "target_x_m": float(state.target_x if np.isfinite(float(state.target_x)) else self.cfg.target_x),
            }
        )
        return value
