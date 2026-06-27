"""Condensed linear MPC for the normalized 4D cart-pole acceleration model."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .cartpole_linear_model import CartPoleLinearModel
from .box_qp import solve_box_qp
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


def _build_prediction_matrices(
    a_d: np.ndarray,
    b_d: np.ndarray,
    *,
    horizon: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``Phi`` (N*n x n) and ``Gamma`` (N*n x N) for x_k = Phi x0 + Gamma u."""
    n_state = int(a_d.shape[0])
    n_input = int(b_d.shape[1])
    horizon = int(horizon)
    if horizon < 1:
        raise ValueError("horizon must be >= 1")

    phi = np.zeros((horizon * n_state, n_state), dtype=np.float64)
    gamma = np.zeros((horizon * n_state, horizon * n_input), dtype=np.float64)
    a_power = np.eye(n_state, dtype=np.float64)
    for k in range(horizon):
        row = slice(k * n_state, (k + 1) * n_state)
        a_power = a_d @ a_power if k > 0 else a_d.copy()
        phi[row, :] = a_power
        for j in range(k + 1):
            a_j = np.eye(n_state, dtype=np.float64)
            for _ in range(k - j):
                a_j = a_d @ a_j
            gamma[row, j * n_input : (j + 1) * n_input] = a_j @ b_d
    return phi, gamma


@dataclass
class CartPoleMPCConfig:
    """Finite-horizon linear MPC on the acceleration-level cart-pole model."""

    horizon: int = 20
    q_weights: np.ndarray = field(
        default_factory=lambda: np.array([40.0, 10.0, 180.0, 20.0], dtype=np.float64)
    )
    q_terminal_scale: float = 3.0
    r_weight: float = 0.35
    pole_length_m: float = 0.4
    gravity_mps2: float = 9.81
    output_mode: CommandMode = "x_acceleration"
    command_limit: float = 1.5
    target_x: float = 0.0
    target_theta: float = 0.0
    dt_s: float = 0.05

    def validate(self) -> None:
        if int(self.horizon) < 1:
            raise ValueError("horizon must be >= 1")
        _ = _as_vector_or_diag("q_weights", self.q_weights, 4)
        for name in (
            "q_terminal_scale",
            "r_weight",
            "pole_length_m",
            "gravity_mps2",
            "command_limit",
            "target_x",
            "target_theta",
            "dt_s",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if self.q_terminal_scale <= 0.0:
            raise ValueError("q_terminal_scale must be positive")
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
        if self.output_mode not in ("x_acceleration", "x_velocity", "x_position_delta"):
            raise ValueError(f"Unsupported output_mode: {self.output_mode!r}")


class CartPoleMPCController(NominalController):
    """Receding-horizon linear MPC with box constraints on the input sequence."""

    def __init__(self, config: CartPoleMPCConfig | None = None) -> None:
        self.cfg = config if config is not None else CartPoleMPCConfig()
        self.cfg.validate()
        self.model = CartPoleLinearModel(
            pole_length_m=float(self.cfg.pole_length_m),
            gravity_mps2=float(self.cfg.gravity_mps2),
        )
        self.a_d, self.b_d = self.model.discrete_matrices(self.cfg.dt_s)
        self.q = _as_vector_or_diag("q_weights", self.cfg.q_weights, 4)
        self.q_terminal = float(self.cfg.q_terminal_scale) * self.q
        self.r = np.array([[float(self.cfg.r_weight)]], dtype=np.float64)
        self.phi, self.gamma = _build_prediction_matrices(
            self.a_d,
            self.b_d,
            horizon=int(self.cfg.horizon),
        )
        self._assemble_cost_matrices()

    def _assemble_cost_matrices(self) -> None:
        n_state = 4
        horizon = int(self.cfg.horizon)
        q_blocks = [self.q] * (horizon - 1) + [self.q_terminal]
        q_bar = np.zeros((horizon * n_state, horizon * n_state), dtype=np.float64)
        for k, q_k in enumerate(q_blocks):
            q_bar[k * n_state : (k + 1) * n_state, k * n_state : (k + 1) * n_state] = q_k
        r_bar = np.kron(np.eye(horizon, dtype=np.float64), self.r)
        self.q_bar = q_bar
        self.r_bar = r_bar
        self.hessian = self.gamma.T @ self.q_bar @ self.gamma + self.r_bar
        self.hessian = 0.5 * (self.hessian + self.hessian.T)
        u_lim = float(self.cfg.command_limit)
        self.input_lower = np.full(horizon, -u_lim, dtype=np.float64)
        self.input_upper = np.full(horizon, +u_lim, dtype=np.float64)

    def _error_state(self, state: ControllerState) -> np.ndarray:
        state.validate()
        return np.array(
            [
                float(state.x - self.cfg.target_x),
                float(state.x_dot),
                float(state.theta - self.cfg.target_theta),
                float(state.theta_dot),
            ],
            dtype=np.float64,
        )

    def _reference_stack(self, state: ControllerState) -> np.ndarray:
        horizon = int(self.cfg.horizon)
        ref = np.zeros(horizon * 4, dtype=np.float64)
        for k in range(horizon):
            ref[k * 4 : k * 4 + 4] = np.array(
                [
                    float(state.target_x - self.cfg.target_x),
                    0.0,
                    float(state.target_theta - self.cfg.target_theta),
                    0.0,
                ],
                dtype=np.float64,
            )
        return ref

    def solve_input_sequence(self, state: ControllerState) -> tuple[np.ndarray, dict[str, float]]:
        x_err = self._error_state(state)
        x_free = self.phi @ x_err
        ref_stack = self._reference_stack(state)
        linear = self.gamma.T @ self.q_bar @ (x_free - ref_stack)
        u_seq = solve_box_qp(
            self.hessian,
            linear,
            self.input_lower,
            self.input_upper,
        )
        return u_seq, {
            "x_error_norm": float(np.linalg.norm(x_err)),
            "first_input_raw": float(u_seq[0]),
            "input_sequence_norm": float(np.linalg.norm(u_seq)),
        }

    def _raw_acceleration(self, state: ControllerState) -> tuple[float, dict[str, float]]:
        u_seq, diag = self.solve_input_sequence(state)
        accel = float(u_seq[0])
        if not np.isfinite(accel):
            raise RuntimeError("MPC controller produced a non-finite acceleration command")
        return accel, diag

    def compute(self, state: ControllerState) -> ControllerCommand:
        accel, diag = self._raw_acceleration(state)
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
                "controller": "mpc",
                "acceleration_mps2": accel,
                "horizon": int(self.cfg.horizon),
                "pole_length_m": float(self.cfg.pole_length_m),
                **diag,
            },
        )

    def predict_horizon_states(
        self,
        state: ControllerState,
    ) -> list[ControllerState]:
        """Roll the linear plant forward under the current MPC input sequence."""
        u_seq, _ = self.solve_input_sequence(state)
        predicted: list[ControllerState] = []
        current = state
        for k in range(int(self.cfg.horizon)):
            current = self.model.predict_next_state(current, float(u_seq[k]), dt_s=self.cfg.dt_s)
            predicted.append(current)
        return predicted
