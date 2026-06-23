"""Linearized cart-pole dynamics around upright.

This is the simulation-only model used by the constrained controller scaffold.
The convention is intentionally explicit:

- ``theta == 0``: upright pole
- ``theta > 0``: pole leans toward world ``+x``
- input command: desired cart ``x`` acceleration

The model is a normalized acceleration-level abstraction, not a raw torque
model. That makes it suitable for command-governor and LQR scaffolding before
any hardware-facing layer exists.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .controller_interfaces import ControllerCommand, ControllerState, CommandMode


def _as_finite_matrix(
    name: str,
    value: np.ndarray,
    shape: tuple[int, ...] | None = None,
) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64)
    if shape is not None and arr.shape != shape:
        raise ValueError(f"{name} must have shape {shape}; got {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains NaN/Inf")
    return arr


@dataclass(frozen=True)
class CartPoleLinearModel:
    """Acceleration-level linear cart-pole model."""

    pole_length_m: float = 0.5
    gravity_mps2: float = 9.81

    def validate(self) -> None:
        if not np.isfinite(float(self.pole_length_m)) or self.pole_length_m <= 0.0:
            raise ValueError("pole_length_m must be positive and finite")
        if not np.isfinite(float(self.gravity_mps2)) or self.gravity_mps2 <= 0.0:
            raise ValueError("gravity_mps2 must be positive and finite")

    @property
    def angular_gain(self) -> float:
        return float(self.gravity_mps2 / self.pole_length_m)

    @property
    def input_gain(self) -> float:
        return float(1.0 / self.pole_length_m)

    def continuous_matrices(self) -> tuple[np.ndarray, np.ndarray]:
        """Return the linearized continuous-time ``A`` and ``B`` matrices."""
        self.validate()
        a = np.array(
            [
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, self.angular_gain, 0.0],
            ],
            dtype=np.float64,
        )
        b = np.array([[0.0], [1.0], [0.0], [-self.input_gain]], dtype=np.float64)
        return a, b

    def discrete_matrices(self, dt_s: float) -> tuple[np.ndarray, np.ndarray]:
        """Forward-Euler discretization used by the LQR scaffold."""
        if not np.isfinite(float(dt_s)) or dt_s <= 0.0:
            raise ValueError("dt_s must be positive and finite")
        a, b = self.continuous_matrices()
        eye = np.eye(4, dtype=np.float64)
        a_d = eye + a * float(dt_s)
        b_d = b * float(dt_s)
        return a_d, b_d

    def state_to_vector(self, state: ControllerState) -> np.ndarray:
        state.validate()
        return state.as_vector()

    def error_vector(self, state: ControllerState) -> np.ndarray:
        return state.error_vector()

    def predict_next_state(
        self,
        state: ControllerState,
        accel_cmd_mps2: float,
        dt_s: float | None = None,
    ) -> ControllerState:
        """Predict the next state under a piecewise-constant acceleration command."""
        dt = float(state.dt_s if dt_s is None else dt_s)
        if not np.isfinite(dt) or dt <= 0.0:
            raise ValueError("dt_s must be positive and finite")
        a = float(accel_cmd_mps2)
        if not np.isfinite(a):
            raise ValueError("accel_cmd_mps2 must be finite")

        theta_ddot = self.angular_gain * float(state.theta) - self.input_gain * a
        x_next = float(state.x + state.x_dot * dt + 0.5 * a * dt * dt)
        x_dot_next = float(state.x_dot + a * dt)
        theta_next = float(state.theta + state.theta_dot * dt + 0.5 * theta_ddot * dt * dt)
        theta_dot_next = float(state.theta_dot + theta_ddot * dt)

        return ControllerState(
            x=x_next,
            x_dot=x_dot_next,
            theta=theta_next,
            theta_dot=theta_dot_next,
            time_s=float(state.time_s + dt),
            dt_s=dt,
            target_x=float(state.target_x),
            target_theta=float(state.target_theta),
            metadata=dict(state.metadata),
        )

    def command_from_acceleration(
        self,
        state: ControllerState,
        accel_cmd_mps2: float,
        mode: CommandMode,
        *,
        command_limit: float | None = None,
    ) -> ControllerCommand:
        """Convert an acceleration command into a safe control abstraction."""
        dt = float(state.dt_s)
        a = float(accel_cmd_mps2)
        if not np.isfinite(a):
            raise ValueError("accel_cmd_mps2 must be finite")
        if mode == "x_acceleration":
            value = a
        elif mode == "x_velocity":
            value = float(state.x_dot + a * dt)
        elif mode == "x_position_delta":
            value = float(state.x_dot * dt + 0.5 * a * dt * dt)
        else:  # pragma: no cover - guarded by type system and validation
            raise ValueError(f"Unsupported command mode: {mode!r}")

        metadata = {
            "source": "acceleration_abstraction",
            "acceleration_mps2": a,
            "dt_s": dt,
        }
        if command_limit is not None:
            value = float(np.clip(value, -abs(float(command_limit)), +abs(float(command_limit))))
            metadata["command_limit"] = abs(float(command_limit))
        return ControllerCommand(mode=mode, value=value, time_s=float(state.time_s), metadata=metadata)


def solve_discrete_lqr(
    a_d: np.ndarray,
    b_d: np.ndarray,
    q: np.ndarray,
    r: np.ndarray,
    *,
    max_iters: int = 2000,
    tol: float = 1e-10,
) -> tuple[np.ndarray, np.ndarray, bool, int]:
    """Solve the discrete-time algebraic Riccati equation by fixed-point iteration."""
    a_d = _as_finite_matrix("A", a_d)
    b_d = _as_finite_matrix("B", b_d)
    if a_d.ndim != 2 or a_d.shape[0] != a_d.shape[1]:
        raise ValueError(f"A must be square; got {a_d.shape}")
    if b_d.ndim != 2:
        raise ValueError(f"B must be a 2D matrix; got {b_d.shape}")
    if b_d.shape[0] != a_d.shape[0]:
        raise ValueError(f"B must have the same row count as A; got A={a_d.shape}, B={b_d.shape}")
    if b_d.shape[1] < 1:
        raise ValueError("B must have at least one input column")
    if b_d.shape[1] != 1:
        raise ValueError(
            "This discrete LQR helper currently supports a single-input system; "
            f"got B with shape {b_d.shape}"
        )
    n_state = int(a_d.shape[0])

    q = np.asarray(q, dtype=np.float64)
    if q.shape == (n_state,):
        q = np.diag(q)
    q = _as_finite_matrix("Q", q)
    if q.shape != (n_state, n_state):
        raise ValueError(f"Q must have shape ({n_state}, {n_state}); got {q.shape}")

    r = np.asarray(r, dtype=np.float64)
    if r.shape == ():
        r = np.array([[float(r)]], dtype=np.float64)
    if r.shape == (1,):
        r = np.diag(r)
    r = _as_finite_matrix("R", r)
    if r.shape != (1, 1):
        raise ValueError(f"R must have shape (1, 1); got {r.shape}")
    if float(r[0, 0]) <= 0.0:
        raise ValueError("R must be positive definite")

    try:
        from scipy.linalg import solve_discrete_are

        p = solve_discrete_are(a_d, b_d, q, r)
        gain = np.linalg.solve(r + b_d.T @ p @ b_d, b_d.T @ p @ a_d)
        if not np.all(np.isfinite(gain)) or not np.all(np.isfinite(p)):
            raise RuntimeError("SciPy discrete ARE produced non-finite values")
        return gain, p, True, 1
    except Exception:
        pass

    p = q.copy()
    converged = False
    iters = 0
    for iters in range(1, max(1, int(max_iters)) + 1):
        bt_p = b_d.T @ p
        s = r + bt_p @ b_d
        k = np.linalg.solve(s, bt_p @ a_d)
        p_next = a_d.T @ p @ a_d - a_d.T @ p @ b_d @ k + q
        if not np.all(np.isfinite(p_next)):
            raise RuntimeError("Discrete Riccati iteration produced non-finite values")
        if float(np.max(np.abs(p_next - p))) <= float(tol):
            p = p_next
            converged = True
            break
        p = p_next

    bt_p = b_d.T @ p
    gain = np.linalg.solve(r + bt_p @ b_d, bt_p @ a_d)
    return gain, p, converged, iters
