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
    NUM_FEATURES_PER_JOINT_COUPLED,
    NUM_JOINTS,
    all_joint_features,
    all_joint_features_coupled,
    compute_residual_torque,
    compute_residual_torque_coupled,
    joint_features,
    joint_features_coupled,
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


def test_compute_residual_torque_clip_abs_bounds_output():
    # Large weights that would otherwise produce an enormous output.
    weights = np.full((NUM_JOINTS, NUM_FEATURES_PER_JOINT), 0.0)
    weights[:, 0] = 1000.0  # bias-only, so output == 1000.0 for every joint before clipping
    q = np.zeros(NUM_JOINTS)
    qd = np.zeros(NUM_JOINTS)
    clip_abs = np.full(NUM_JOINTS, 5.0)
    out = compute_residual_torque(weights, q, qd, clip_abs=clip_abs)
    np.testing.assert_allclose(out, np.full(NUM_JOINTS, 5.0))


def test_compute_residual_torque_clip_abs_scalar_applies_to_all_joints():
    weights = np.zeros((NUM_JOINTS, NUM_FEATURES_PER_JOINT))
    weights[:, 0] = np.array([1.0, -100.0, 3.0, -0.5, 200.0, 0.0])
    q = np.zeros(NUM_JOINTS)
    qd = np.zeros(NUM_JOINTS)
    out = compute_residual_torque(weights, q, qd, clip_abs=10.0)
    np.testing.assert_allclose(out, np.array([1.0, -10.0, 3.0, -0.5, 10.0, 0.0]))


def test_compute_residual_torque_clip_abs_none_is_unchanged_default():
    rng = np.random.default_rng(1)
    weights = rng.uniform(-1.0, 1.0, size=(NUM_JOINTS, NUM_FEATURES_PER_JOINT))
    q = rng.uniform(-1.0, 1.0, size=NUM_JOINTS)
    qd = rng.uniform(-1.0, 1.0, size=NUM_JOINTS)
    out_default = compute_residual_torque(weights, q, qd)
    out_explicit_none = compute_residual_torque(weights, q, qd, clip_abs=None)
    np.testing.assert_allclose(out_default, out_explicit_none)


def test_compute_residual_torque_rejects_negative_clip_abs():
    weights = np.zeros((NUM_JOINTS, NUM_FEATURES_PER_JOINT))
    q = np.zeros(NUM_JOINTS)
    qd = np.zeros(NUM_JOINTS)
    with pytest.raises(ValueError):
        compute_residual_torque(weights, q, qd, clip_abs=-1.0)


def test_compute_residual_torque_rejects_wrong_clip_abs_shape():
    weights = np.zeros((NUM_JOINTS, NUM_FEATURES_PER_JOINT))
    q = np.zeros(NUM_JOINTS)
    qd = np.zeros(NUM_JOINTS)
    with pytest.raises(ValueError):
        compute_residual_torque(weights, q, qd, clip_abs=np.ones(5))


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


# --- Cross-joint-coupled feature basis (added 2026-08-01) ---
# See docs/status/residual_torque_regression_pipeline_2026-08-01.md's "Cross-joint coupling
# feature set" section: this basis measurably improved held-out R^2 on every joint versus the
# own-joint-only basis above, and is now the fit script's default (--feature-set coupled).


def test_num_features_per_joint_coupled_is_21():
    assert NUM_FEATURES_PER_JOINT_COUPLED == 21


def test_joint_features_coupled_shape():
    q = np.array([0.1, -0.2, 0.3, -0.4, 0.5, -0.6])
    qd = np.array([0.5, -0.5, 1.0, -1.0, 0.0, 0.2])
    feats = joint_features_coupled(q, qd, 0)
    assert feats.shape == (NUM_FEATURES_PER_JOINT_COUPLED,)


def test_joint_features_coupled_first_6_match_own_joint_features():
    q = np.array([0.1, -0.2, 0.3, -0.4, 0.5, -0.6])
    qd = np.array([0.5, -0.5, 1.0, -1.0, 0.0, 0.2])
    for j in range(NUM_JOINTS):
        feats = joint_features_coupled(q, qd, j)
        np.testing.assert_allclose(feats[:NUM_FEATURES_PER_JOINT], joint_features(q[j], qd[j]))


def test_joint_features_coupled_remaining_15_are_other_joints_qd_sin_cos():
    q = np.array([0.1, -0.2, 0.3, -0.4, 0.5, -0.6])
    qd = np.array([0.5, -0.5, 1.0, -1.0, 0.0, 0.2])
    j = 2  # elbow
    feats = joint_features_coupled(q, qd, j)
    other = [k for k in range(NUM_JOINTS) if k != j]
    expected_other_qd = qd[other]
    np.testing.assert_allclose(feats[NUM_FEATURES_PER_JOINT : NUM_FEATURES_PER_JOINT + 5], expected_other_qd)
    pos_block = feats[NUM_FEATURES_PER_JOINT + 5 :]
    for i, k in enumerate(other):
        assert pos_block[2 * i] == pytest.approx(np.sin(q[k]))
        assert pos_block[2 * i + 1] == pytest.approx(np.cos(q[k]))


def test_joint_features_coupled_other_block_reflects_peer_joint_state():
    # A change to joint k's own (q, qd) must show up in another joint j's
    # "other joints" block, at the position corresponding to joint k.
    q = np.zeros(NUM_JOINTS)
    qd = np.zeros(NUM_JOINTS)
    q2, qd2 = q.copy(), qd.copy()
    q2[0], qd2[0] = 1.234, 5.0  # perturb joint 0's own state
    feats = joint_features_coupled(q2, qd2, 1)  # joint 1's view of others (includes joint 0)
    other = [k for k in range(NUM_JOINTS) if k != 1]
    idx0 = other.index(0)
    assert feats[NUM_FEATURES_PER_JOINT + idx0] == pytest.approx(5.0)  # other-qd block
    pos_block = feats[NUM_FEATURES_PER_JOINT + 5 :]
    assert pos_block[2 * idx0] == pytest.approx(np.sin(1.234))
    assert pos_block[2 * idx0 + 1] == pytest.approx(np.cos(1.234))


def test_joint_features_coupled_own_block_unaffected_by_other_joints():
    # Joint j's OWN 6 features must depend only on (q[j], qd[j]) -- changing
    # any OTHER joint's state must not perturb them (own-state isolation,
    # same guarantee the un-coupled basis has for its own_joint_features).
    q = np.array([0.3, 0.0, 0.0, 0.0, 0.0, 0.0])
    qd = np.array([0.7, 0.0, 0.0, 0.0, 0.0, 0.0])
    feats_ref = joint_features_coupled(q, qd, 0)[:NUM_FEATURES_PER_JOINT]
    q2, qd2 = q.copy(), qd.copy()
    q2[3], qd2[5] = 9.0, -9.0  # perturb unrelated joints' state
    feats_perturbed = joint_features_coupled(q2, qd2, 0)[:NUM_FEATURES_PER_JOINT]
    np.testing.assert_allclose(feats_ref, feats_perturbed)


def test_joint_features_coupled_rejects_out_of_range_joint_index():
    q = np.zeros(NUM_JOINTS)
    qd = np.zeros(NUM_JOINTS)
    with pytest.raises(ValueError):
        joint_features_coupled(q, qd, 6)
    with pytest.raises(ValueError):
        joint_features_coupled(q, qd, -1)


def test_joint_features_coupled_rejects_nan_input():
    q = np.array([0.0, 0.0, 0.0, 0.0, 0.0, np.nan])
    qd = np.zeros(NUM_JOINTS)
    with pytest.raises(ValueError):
        joint_features_coupled(q, qd, 0)


def test_all_joint_features_coupled_shape_and_matches_per_joint():
    q = np.array([0.1, -0.2, 0.3, -0.4, 0.5, -0.6])
    qd = np.array([0.5, -0.5, 1.0, -1.0, 0.0, 0.2])
    feats = all_joint_features_coupled(q, qd)
    assert feats.shape == (NUM_JOINTS, NUM_FEATURES_PER_JOINT_COUPLED)
    for j in range(NUM_JOINTS):
        np.testing.assert_allclose(feats[j], joint_features_coupled(q, qd, j))


def test_all_joint_features_coupled_rejects_nan_input():
    q = np.array([0.0, 0.0, 0.0, 0.0, 0.0, np.nan])
    qd = np.zeros(NUM_JOINTS)
    with pytest.raises(ValueError):
        all_joint_features_coupled(q, qd)


def test_compute_residual_torque_coupled_zero_weights_gives_zero_output():
    weights = np.zeros((NUM_JOINTS, NUM_FEATURES_PER_JOINT_COUPLED))
    q = np.array([0.1, -0.2, 0.3, -0.4, 0.5, -0.6])
    qd = np.array([0.5, -0.5, 1.0, -1.0, 0.0, 0.2])
    out = compute_residual_torque_coupled(weights, q, qd)
    np.testing.assert_allclose(out, np.zeros(NUM_JOINTS))


def test_compute_residual_torque_coupled_matches_manual_dot_product():
    rng = np.random.default_rng(0)
    weights = rng.uniform(-1.0, 1.0, size=(NUM_JOINTS, NUM_FEATURES_PER_JOINT_COUPLED))
    q = rng.uniform(-1.0, 1.0, size=NUM_JOINTS)
    qd = rng.uniform(-1.0, 1.0, size=NUM_JOINTS)
    out = compute_residual_torque_coupled(weights, q, qd)

    feats = all_joint_features_coupled(q, qd)
    expected = np.array([weights[j] @ feats[j] for j in range(NUM_JOINTS)])
    np.testing.assert_allclose(out, expected, atol=1e-12)


def test_compute_residual_torque_coupled_rejects_wrong_weight_shape():
    q = np.zeros(NUM_JOINTS)
    qd = np.zeros(NUM_JOINTS)
    with pytest.raises(ValueError):
        compute_residual_torque_coupled(np.zeros((5, NUM_FEATURES_PER_JOINT_COUPLED)), q, qd)
    with pytest.raises(ValueError):
        compute_residual_torque_coupled(np.zeros((NUM_JOINTS, 5)), q, qd)
    # Baseline-shaped weights must be rejected too -- the two bases are not interchangeable.
    with pytest.raises(ValueError):
        compute_residual_torque_coupled(np.zeros((NUM_JOINTS, NUM_FEATURES_PER_JOINT)), q, qd)


def test_compute_residual_torque_coupled_clip_abs_bounds_output():
    weights = np.zeros((NUM_JOINTS, NUM_FEATURES_PER_JOINT_COUPLED))
    weights[:, 0] = 1000.0  # bias-only
    q = np.zeros(NUM_JOINTS)
    qd = np.zeros(NUM_JOINTS)
    clip_abs = np.full(NUM_JOINTS, 5.0)
    out = compute_residual_torque_coupled(weights, q, qd, clip_abs=clip_abs)
    np.testing.assert_allclose(out, np.full(NUM_JOINTS, 5.0))


def test_compute_residual_torque_coupled_clip_abs_none_is_unchanged_default():
    rng = np.random.default_rng(1)
    weights = rng.uniform(-1.0, 1.0, size=(NUM_JOINTS, NUM_FEATURES_PER_JOINT_COUPLED))
    q = rng.uniform(-1.0, 1.0, size=NUM_JOINTS)
    qd = rng.uniform(-1.0, 1.0, size=NUM_JOINTS)
    out_default = compute_residual_torque_coupled(weights, q, qd)
    out_explicit_none = compute_residual_torque_coupled(weights, q, qd, clip_abs=None)
    np.testing.assert_allclose(out_default, out_explicit_none)
