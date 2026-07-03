"""Unit tests for cart-pole MPC."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from controller_core import CartPoleMPCConfig, CartPoleMPCController, ControllerState  # noqa: E402
from controller_core.cartpole_linear_model import CartPoleLinearModel  # noqa: E402


def test_mpc_returns_finite_command_near_upright() -> None:
    ctrl = CartPoleMPCController(
        CartPoleMPCConfig(
            horizon=15,
            dt_s=0.05,
            command_limit=2.0,
            q_weights=np.array([20.0, 5.0, 120.0, 12.0], dtype=np.float64),
            r_weight=0.5,
        )
    )
    state = ControllerState(x=0.0, x_dot=0.0, theta=0.05, theta_dot=0.0, dt_s=0.05)
    cmd = ctrl.compute(state)
    assert cmd.mode == "x_acceleration"
    assert np.isfinite(cmd.value)


def test_mpc_opposes_pole_lean_direction() -> None:
    ctrl = CartPoleMPCController(
        CartPoleMPCConfig(
            horizon=20,
            dt_s=0.05,
            command_limit=3.0,
            q_weights=np.array([10.0, 2.0, 200.0, 20.0], dtype=np.float64),
            r_weight=0.25,
        )
    )
    right = ControllerState(x=0.0, x_dot=0.0, theta=0.08, theta_dot=0.0, dt_s=0.05)
    left = ControllerState(x=0.0, x_dot=0.0, theta=-0.08, theta_dot=0.0, dt_s=0.05)
    right_cmd = ctrl.compute(right)
    left_cmd = ctrl.compute(left)
    assert right_cmd.value > 0.0
    assert left_cmd.value < 0.0


def test_mpc_linear_rollout_keeps_state_finite() -> None:
    cfg = CartPoleMPCConfig(
        horizon=25,
        dt_s=0.02,
        command_limit=2.5,
        q_weights=np.array([8.0, 2.0, 250.0, 25.0], dtype=np.float64),
        r_weight=0.15,
        pole_length_m=0.4,
    )
    ctrl = CartPoleMPCController(cfg)
    model = CartPoleLinearModel(pole_length_m=cfg.pole_length_m, gravity_mps2=cfg.gravity_mps2)
    state = ControllerState(x=0.0, x_dot=0.0, theta=0.08, theta_dot=0.0, dt_s=cfg.dt_s)
    for _ in range(40):
        cmd = ctrl.compute(state)
        state = model.predict_next_state(state, float(cmd.metadata["acceleration_mps2"]), dt_s=cfg.dt_s)
    assert np.all(np.isfinite(state.as_vector()))
