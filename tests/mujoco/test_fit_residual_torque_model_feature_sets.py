"""Tests for the cross-joint-coupled feature-set option added 2026-08-01 to
tools/analysis/fit_residual_torque_model.py (``--feature-set {coupled,baseline}``).

See docs/status/residual_torque_regression_pipeline_2026-08-01.md's "Cross-joint coupling
feature set" section: fitting on real UR5e hardware traces found that a 21-feature-per-joint
basis (own-joint 6 features plus every OTHER joint's qd/sin(q)/cos(q)) improved held-out R^2
on every one of the 6 joints versus the original own-joint-only 6-feature basis, across 7
independent real-hardware run-level train/test splits -- motivated by joints 2/3/4 spending
almost their entire trajectory near-stationary during an X-only transport move, so their OWN
state carries little signal but OTHER joints' motion (which genuinely couples through
Coriolis/inertial terms in a serial-chain arm) can still explain some of their residual.

Marked ``mujoco`` per this directory's convention (auto-applied): transitively imports
``tools.analysis.residual_data`` -> ``controller_core.model_dynamics.PinocchioUR5eDynamics``
at import time, matching ``test_fit_residual_torque_model_ridge.py``'s reason for living here.
"""

from __future__ import annotations

import numpy as np
import pytest

from controller_core.residual_torque_model import (
    NUM_FEATURES_PER_JOINT,
    NUM_FEATURES_PER_JOINT_COUPLED,
    NUM_JOINTS,
    all_joint_features,
    all_joint_features_coupled,
)
from tools.analysis.fit_residual_torque_model import (
    FEATURE_SETS,
    _featurize_runs,
    evaluate,
    fit_ols_weights,
    fit_ridge_weights,
)
from tools.analysis.residual_data import ResidualDatasetRun

pytestmark = pytest.mark.mujoco


def _make_synthetic_run(rng: np.random.Generator, n_rows: int, label: str) -> ResidualDatasetRun:
    q = rng.uniform(-1.0, 1.0, size=(n_rows, NUM_JOINTS))
    qd = rng.uniform(-1.0, 1.0, size=(n_rows, NUM_JOINTS))
    tau_residual = rng.normal(scale=0.1, size=(n_rows, NUM_JOINTS))
    qdd_residual = rng.normal(scale=0.1, size=(n_rows, NUM_JOINTS))
    t_s = np.arange(n_rows, dtype=np.float64) * 0.002
    return ResidualDatasetRun(
        source_path=f"/synthetic/{label}/trace.jsonl",
        label=label,
        q=q,
        qd=qd,
        tau_residual=tau_residual,
        qdd_residual=qdd_residual,
        qdd_residual_norm=np.linalg.norm(qdd_residual, axis=1),
        t_s=t_s,
        n_rows_total=n_rows,
        n_rows_valid=n_rows,
    )


def test_feature_sets_registry_has_expected_widths():
    assert FEATURE_SETS["baseline"] == (all_joint_features, NUM_FEATURES_PER_JOINT)
    assert FEATURE_SETS["coupled"] == (all_joint_features_coupled, NUM_FEATURES_PER_JOINT_COUPLED)


def test_featurize_runs_default_is_coupled_width():
    rng = np.random.default_rng(0)
    runs = [_make_synthetic_run(rng, 50, "run0")]
    x, y = _featurize_runs(runs, deadband=0.05)
    assert x.shape == (50, NUM_JOINTS, NUM_FEATURES_PER_JOINT_COUPLED)
    assert y.shape == (50, NUM_JOINTS)


def test_featurize_runs_baseline_width():
    rng = np.random.default_rng(0)
    runs = [_make_synthetic_run(rng, 50, "run0")]
    x, y = _featurize_runs(runs, deadband=0.05, feature_set="baseline")
    assert x.shape == (50, NUM_JOINTS, NUM_FEATURES_PER_JOINT)


def test_featurize_runs_empty_runs_matches_selected_width():
    x_coupled, _ = _featurize_runs([], deadband=0.05, feature_set="coupled")
    assert x_coupled.shape == (0, NUM_JOINTS, NUM_FEATURES_PER_JOINT_COUPLED)
    x_baseline, _ = _featurize_runs([], deadband=0.05, feature_set="baseline")
    assert x_baseline.shape == (0, NUM_JOINTS, NUM_FEATURES_PER_JOINT)


def test_featurize_runs_rejects_unknown_feature_set():
    with pytest.raises(ValueError):
        _featurize_runs([], deadband=0.05, feature_set="not_a_real_feature_set")


def test_fit_ridge_weights_infers_width_from_x_train_coupled():
    rng = np.random.default_rng(1)
    n = 500
    design = rng.normal(size=(n, NUM_FEATURES_PER_JOINT_COUPLED))
    target = design @ rng.normal(size=NUM_FEATURES_PER_JOINT_COUPLED)
    x = np.repeat(design[:, np.newaxis, :], NUM_JOINTS, axis=1)
    y = np.repeat(target[:, np.newaxis], NUM_JOINTS, axis=1)
    w = fit_ridge_weights(x, y, ridge_lambda=1.0)
    assert w.shape == (NUM_JOINTS, NUM_FEATURES_PER_JOINT_COUPLED)


def test_fit_ols_weights_infers_width_from_x_train_coupled():
    rng = np.random.default_rng(2)
    n = 500
    design = rng.normal(size=(n, NUM_FEATURES_PER_JOINT_COUPLED))
    target = design @ rng.normal(size=NUM_FEATURES_PER_JOINT_COUPLED)
    x = np.repeat(design[:, np.newaxis, :], NUM_JOINTS, axis=1)
    y = np.repeat(target[:, np.newaxis], NUM_JOINTS, axis=1)
    w = fit_ols_weights(x, y)
    assert w.shape == (NUM_JOINTS, NUM_FEATURES_PER_JOINT_COUPLED)


def test_evaluate_uses_matching_feature_set_for_weights():
    rng = np.random.default_rng(3)
    train_runs = [_make_synthetic_run(rng, 200, "train0"), _make_synthetic_run(rng, 200, "train1")]
    test_runs = [_make_synthetic_run(rng, 100, "test0")]

    x_train, y_train = _featurize_runs(train_runs, deadband=0.05, feature_set="coupled")
    weights = fit_ridge_weights(x_train, y_train, ridge_lambda=1.0e3)

    metrics = evaluate(test_runs, weights, deadband=0.05, feature_set="coupled")
    assert len(metrics) == NUM_JOINTS
    for m in metrics:
        assert np.isfinite(m.rmse_model)


def test_end_to_end_coupled_vs_baseline_recovers_true_cross_joint_signal():
    """A residual that genuinely depends on ANOTHER joint's qd (not the joint's own state at
    all) should be fit far better by the coupled basis than the baseline (own-state-only)
    basis -- a direct, synthetic sanity check of the real finding on real hardware data
    (docs/status/residual_torque_regression_pipeline_2026-08-01.md)."""
    rng = np.random.default_rng(4)
    n_rows = 2000

    def make_coupled_run(label: str) -> ResidualDatasetRun:
        q = rng.uniform(-1.0, 1.0, size=(n_rows, NUM_JOINTS))
        qd = rng.uniform(-1.0, 1.0, size=(n_rows, NUM_JOINTS))
        # Joint 2's residual torque genuinely depends on joint 0's qd, not its own state.
        tau_residual = np.zeros((n_rows, NUM_JOINTS))
        tau_residual[:, 2] = 3.0 * qd[:, 0] + 0.01 * rng.normal(size=n_rows)
        qdd_residual = np.zeros((n_rows, NUM_JOINTS))
        t_s = np.arange(n_rows, dtype=np.float64) * 0.002
        return ResidualDatasetRun(
            source_path=f"/synthetic/{label}/trace.jsonl",
            label=label,
            q=q,
            qd=qd,
            tau_residual=tau_residual,
            qdd_residual=qdd_residual,
            qdd_residual_norm=np.linalg.norm(qdd_residual, axis=1),
            t_s=t_s,
            n_rows_total=n_rows,
            n_rows_valid=n_rows,
        )

    train_runs = [make_coupled_run("train0"), make_coupled_run("train1"), make_coupled_run("train2")]
    test_runs = [make_coupled_run("test0")]

    def r2_joint2(feature_set: str) -> float:
        x_train, y_train = _featurize_runs(train_runs, deadband=0.05, feature_set=feature_set)
        weights = fit_ridge_weights(x_train, y_train, ridge_lambda=1.0)
        metrics = evaluate(test_runs, weights, deadband=0.05, feature_set=feature_set)
        return metrics[2].r2_vs_zero_baseline

    r2_baseline = r2_joint2("baseline")
    r2_coupled = r2_joint2("coupled")
    # The baseline basis has no way to see joint 0's qd at all for joint 2's model, so it
    # should recover essentially none of the signal; the coupled basis should recover most of
    # it (R^2 close to 1, since the synthetic signal is 3.0*qd[0] plus small noise).
    assert r2_baseline < 0.1
    assert r2_coupled > 0.9
