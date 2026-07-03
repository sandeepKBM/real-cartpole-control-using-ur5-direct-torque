"""Unit tests for the cart-pole LQR and fallback nominal controllers."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from controller_core import (  # noqa: E402
    CartPoleFallbackConfig,
    CartPoleFallbackController,
    CartPoleLQRConfig,
    CartPoleLQRController,
    ControllerState,
)


def test_lqr_gain_shape_and_finite() -> None:
    ctrl = CartPoleLQRController(
        CartPoleLQRConfig(
            dt_s=0.01,
            q_weights=np.array([12.0, 3.0, 60.0, 6.0], dtype=np.float64),
            r_weight=1.0,
            command_limit=2.0,
        )
    )
    assert ctrl.gain_matrix.shape == (1, 4)
    assert np.all(np.isfinite(ctrl.gain_matrix))
    assert ctrl.riccati_converged is True
    assert ctrl.riccati_iters >= 1


def test_lqr_returns_finite_acceleration_command_near_upright() -> None:
    ctrl = CartPoleLQRController(
        CartPoleLQRConfig(
            dt_s=0.01,
            q_weights=np.array([8.0, 2.0, 80.0, 8.0], dtype=np.float64),
            r_weight=1.0,
            command_limit=5.0,
        )
    )
    state = ControllerState(x=0.01, x_dot=0.0, theta=0.04, theta_dot=0.0, dt_s=0.01)
    cmd = ctrl.compute(state)
    assert cmd.mode == "x_acceleration"
    assert np.isfinite(cmd.value)


def test_lqr_unstable_state_behavior_pushes_cart_toward_pole() -> None:
    ctrl = CartPoleLQRController(
        CartPoleLQRConfig(
            dt_s=0.01,
            q_weights=np.array([10.0, 2.0, 100.0, 10.0], dtype=np.float64),
            r_weight=1.0,
            command_limit=5.0,
        )
    )
    right_lean = ControllerState(x=0.0, x_dot=0.0, theta=0.08, theta_dot=0.0, dt_s=0.01)
    left_lean = ControllerState(x=0.0, x_dot=0.0, theta=-0.08, theta_dot=0.0, dt_s=0.01)
    right_cmd = ctrl.compute(right_lean)
    left_cmd = ctrl.compute(left_lean)
    assert right_cmd.value > 0.0
    assert left_cmd.value < 0.0


def test_lqr_saturates_on_large_state() -> None:
    ctrl = CartPoleLQRController(
        CartPoleLQRConfig(
            dt_s=0.01,
            q_weights=np.array([20.0, 4.0, 150.0, 15.0], dtype=np.float64),
            r_weight=1.0,
            command_limit=0.25,
        )
    )
    state = ControllerState(x=0.5, x_dot=0.0, theta=0.35, theta_dot=0.0, dt_s=0.01)
    cmd = ctrl.compute(state)
    assert abs(cmd.value) <= 0.25 + 1e-12


def test_fallback_controller_responds_to_cart_error() -> None:
    ctrl = CartPoleFallbackController(
        CartPoleFallbackConfig(
            kp_x=2.0,
            ki_x=0.0,
            kd_x=1.0,
            kp_theta=18.0,
            kd_theta=4.0,
            command_limit=2.0,
            output_mode="x_acceleration",
        )
    )
    state = ControllerState(x=0.10, x_dot=0.0, theta=0.0, theta_dot=0.0, dt_s=0.01)
    cmd = ctrl.compute(state)
    assert cmd.mode == "x_acceleration"
    assert cmd.value < 0.0
    assert np.isfinite(cmd.value)


if __name__ == "__main__":
    test_lqr_gain_shape_and_finite()
    test_lqr_returns_finite_acceleration_command_near_upright()
    test_lqr_unstable_state_behavior_pushes_cart_toward_pole()
    test_lqr_saturates_on_large_state()
    test_fallback_controller_responds_to_cart_error()
    print("lqr controller tests OK")
