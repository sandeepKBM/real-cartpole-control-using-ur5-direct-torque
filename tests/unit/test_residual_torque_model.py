"""Pure-numpy tests for controller_core.residual_torque_model -- the
deterministic-cost inference side of the phase-1 residual-torque-regression
pipeline (2026-08-01). See
docs/status/residual_torque_regression_pipeline_2026-08-01.md.

Nothing here exercises MuJoCo/Pinocchio -- these tests only check the fixed
feature basis and the weights -> torque inference function, which is exactly
what would need to run inside controller_core's real-time budget if this
were ever promoted (it is not wired into any controller path currently)."""

from __future__ import annotations

import numpy as np
import pytest

from controller_core.residual_torque_model import (
    NUM_FEATURES_PER_JOINT,
    NUM_JOINTS,
    all_joint_features,
    compute_residual_torque,
    joint_features,
)


def test_joint_features_shape_and_bias_term():
    feats = joint_features(0.3, 0.0)
    assert feats.shape == (NUM_FEATURES_PER_JOINT,)
    assert feats[0] == 1.0  # bias term always 1


def test_joint_features_tanh_term_saturates_for_large_velocity():
    feats_pos = joint_features(0.0, 10.0, deadband=0.05)
    feats_neg = joint_features(0.0, -10.0, deadband=0.05)
    # tanh(qd/deadband) is feature index 2.
    assert feats_pos[2] == pytest.approx(1.0, abs=1e-6)
    assert feats_neg[2] == pytest.approx(-1.0, abs=1e-6)


def test_joint_features_zero_velocity_zeros_velocity_dependent_terms():
    feats = joint_features(1.234, 0.0)
    # qd, tanh(qd/deadband), qd*|qd| are all velocity-dependent (indices 1..3).
    np.testing.assert_allclose(feats[1:4], np.zeros(3))
    # Position terms (sin/cos) are unaffected.
    assert feats[4] == pytest.approx(np.sin(1.234))
    assert feats[5] == pytest.approx(np.cos(1.234))


def test_joint_features_rejects_nonpositive_deadband():
    with pytest.raises(ValueError):
        joint_features(0.0, 0.0, deadband=0.0)
    with pytest.raises(ValueError):
        joint_features(0.0, 0.0, deadband=-1.0)


def test_all_joint_features_uses_each_joints_own_state_only():
    q = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    qd = np.array([1.0, -1.0, 2.0, -2.0, 0.5, -0.5])
    feats = all_joint_features(q, qd)
    assert feats.shape == (NUM_JOINTS, NUM_FEATURES_PER_JOINT)
    for j in range(NUM_JOINTS):
        np.testing.assert_allclose(feats[j], joint_features(q[j], qd[j]))


def test_all_joint_features_rejects_nan_input():
    q = np.array([0.0, 0.0, 0.0, 0.0, 0.0, np.nan])
    qd = np.zeros(6)
    with pytest.raises(ValueError):
        all_joint_features(q, qd)


def test_compute_residual_torque_zero_weights_gives_zero_output():
    weights = np.zeros((NUM_JOINTS, NUM_FEATURES_PER_JOINT))
    q = np.array([0.1, -0.2, 0.3, -0.4, 0.5, -0.6])
    qd = np.array([0.5, -0.5, 1.0, -1.0, 0.0, 0.2])
    out = compute_residual_torque(weights, q, qd)
    np.testing.assert_allclose(out, np.zeros(NUM_JOINTS))


def test_compute_residual_torque_matches_manual_dot_product():
    rng = np.random.default_rng(0)
    weights = rng.uniform(-1.0, 1.0, size=(NUM_JOINTS, NUM_FEATURES_PER_JOINT))
    q = rng.uniform(-1.0, 1.0, size=NUM_JOINTS)
    qd = rng.uniform(-1.0, 1.0, size=NUM_JOINTS)
    out = compute_residual_torque(weights, q, qd)

    feats = all_joint_features(q, qd)
    expected = np.array([weights[j] @ feats[j] for j in range(NUM_JOINTS)])
    np.testing.assert_allclose(out, expected, atol=1e-12)


def test_compute_residual_torque_only_bias_weight_gives_constant_output():
    weights = np.zeros((NUM_JOINTS, NUM_FEATURES_PER_JOINT))
    weights[:, 0] = np.arange(1, NUM_JOINTS + 1, dtype=np.float64)  # bias-only weights
    q = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    qd = np.zeros(NUM_JOINTS)
    out = compute_residual_torque(weights, q, qd)
    np.testing.assert_allclose(out, weights[:, 0])


def test_compute_residual_torque_rejects_wrong_weight_shape():
    q = np.zeros(NUM_JOINTS)
    qd = np.zeros(NUM_JOINTS)
    with pytest.raises(ValueError):
        compute_residual_torque(np.zeros((5, NUM_FEATURES_PER_JOINT)), q, qd)
    with pytest.raises(ValueError):
        compute_residual_torque(np.zeros((NUM_JOINTS, 5)), q, qd)


def test_compute_residual_torque_output_shape_is_fixed():
    # Deterministic-cost real-time constraint: output is always exactly
    # (NUM_JOINTS,) regardless of input values (no data-dependent branching
    # that could change shape/cost).
    weights = np.ones((NUM_JOINTS, NUM_FEATURES_PER_JOINT))
    for q_val, qd_val in [(0.0, 0.0), (10.0, -10.0), (-3.0, 500.0)]:
        q = np.full(NUM_JOINTS, q_val)
        qd = np.full(NUM_JOINTS, qd_val)
        out = compute_residual_torque(weights, q, qd)
        assert out.shape == (NUM_JOINTS,)
        assert np.all(np.isfinite(out))
