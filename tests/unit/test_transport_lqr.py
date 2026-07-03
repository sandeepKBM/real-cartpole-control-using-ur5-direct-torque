"""Unit tests for the fixed-X transport LQR controller."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from controller_core import ControllerState, FixedXTransportLQRConfig, FixedXTransportLQRController  # noqa: E402


def test_transport_lqr_gain_shape_and_finiteness() -> None:
    cfg = FixedXTransportLQRConfig(dt_s=0.002, target_x=0.0, q_weights=np.array([60.0, 8.0], dtype=np.float64))
    ctrl = FixedXTransportLQRController(cfg)

    assert ctrl.gain_matrix.shape == (1, 2)
    assert np.all(np.isfinite(ctrl.gain_matrix))
    assert np.all(np.isfinite(ctrl.riccati_solution))
    assert ctrl.riccati_converged is True


def test_transport_lqr_moves_toward_positive_target() -> None:
    cfg = FixedXTransportLQRConfig(dt_s=0.002, target_x=0.10, command_limit=1.5)
    ctrl = FixedXTransportLQRController(cfg)
    state = ControllerState(x=0.0, x_dot=0.0, theta=0.0, theta_dot=0.0, dt_s=0.002, target_x=0.10)

    cmd = ctrl.compute(state)

    assert cmd.mode == "x_acceleration"
    assert np.isfinite(cmd.value)
    assert cmd.value > 0.0


def test_transport_lqr_brakes_when_past_goal() -> None:
    cfg = FixedXTransportLQRConfig(dt_s=0.002, target_x=0.0, command_limit=1.5)
    ctrl = FixedXTransportLQRController(cfg)
    state = ControllerState(x=0.05, x_dot=0.10, theta=0.0, theta_dot=0.0, dt_s=0.002, target_x=0.0)

    cmd = ctrl.compute(state)

    assert cmd.mode == "x_acceleration"
    assert np.isfinite(cmd.value)
    assert cmd.value < 0.0


def test_transport_lqr_saturates_to_command_limit() -> None:
    cfg = FixedXTransportLQRConfig(dt_s=0.002, target_x=1.0, command_limit=0.05)
    ctrl = FixedXTransportLQRController(cfg)
    state = ControllerState(x=0.0, x_dot=0.0, theta=0.0, theta_dot=0.0, dt_s=0.002, target_x=1.0)

    cmd = ctrl.compute(state)

    assert abs(cmd.value) <= 0.05 + 1e-12
    assert cmd.metadata["controller"] == "fixed_x_transport_lqr"

