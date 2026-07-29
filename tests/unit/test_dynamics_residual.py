"""Pure-numpy tests for controller_core.dynamics_residual (diagnostic-only
joint-space qdd prediction/residual math -- see
docs/status/direct_torque_residual_observer_2026-07-29.md)."""

from __future__ import annotations

import numpy as np
import pytest

from controller_core.dynamics_residual import joint_acceleration_residual, predict_joint_acceleration


def _spd_matrix(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    a = rng.uniform(0.5, 2.0, size=(6, 6))
    return a @ a.T + 6.0 * np.eye(6)  # guaranteed SPD, well-conditioned


def test_predict_joint_acceleration_matches_manual_solve():
    M = _spd_matrix(0)
    tau = np.array([1.0, -2.0, 3.0, 0.5, -0.5, 2.0])
    bias = np.array([0.1, 0.2, -0.3, 0.0, 0.4, -0.1])
    qdd_pred = predict_joint_acceleration(M, tau, bias)
    expected = np.linalg.solve(M, tau - bias)
    np.testing.assert_allclose(qdd_pred, expected, atol=1e-12)
    # Also verify the manipulator equation is satisfied: M @ qdd = tau - bias.
    np.testing.assert_allclose(M @ qdd_pred, tau - bias, atol=1e-9)


def test_predict_joint_acceleration_zero_when_tau_equals_bias():
    M = _spd_matrix(1)
    bias = np.array([1.0, 2.0, -1.0, 0.5, 0.0, -2.0])
    qdd_pred = predict_joint_acceleration(M, bias, bias)
    np.testing.assert_allclose(qdd_pred, np.zeros(6), atol=1e-9)


def test_predict_joint_acceleration_identity_mass_matrix():
    tau = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    bias = np.array([0.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    qdd_pred = predict_joint_acceleration(np.eye(6), tau, bias)
    np.testing.assert_allclose(qdd_pred, tau - bias, atol=1e-12)


def test_predict_joint_acceleration_accepts_list_inputs():
    qdd_pred = predict_joint_acceleration(np.eye(6).tolist(), [1.0] * 6, [0.0] * 6)
    np.testing.assert_allclose(qdd_pred, np.ones(6), atol=1e-12)


def test_joint_acceleration_residual_is_measured_minus_predicted():
    measured = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    predicted = np.array([0.5, 2.0, 2.5, 4.5, 5.0, 6.5])
    residual = joint_acceleration_residual(measured, predicted)
    np.testing.assert_allclose(residual, measured - predicted, atol=1e-12)


def test_joint_acceleration_residual_sign_convention_positive_when_measured_exceeds_predicted():
    measured = np.full(6, 3.0)
    predicted = np.zeros(6)
    residual = joint_acceleration_residual(measured, predicted)
    assert np.all(residual > 0.0)


def test_joint_acceleration_residual_zero_when_measured_equals_predicted():
    v = np.array([1.0, -1.0, 2.0, -2.0, 0.5, -0.5])
    residual = joint_acceleration_residual(v, v)
    np.testing.assert_allclose(residual, np.zeros(6), atol=1e-12)


def test_predict_joint_acceleration_rejects_singular_mass_matrix():
    with pytest.raises(np.linalg.LinAlgError):
        predict_joint_acceleration(np.zeros((6, 6)), np.ones(6), np.zeros(6))
