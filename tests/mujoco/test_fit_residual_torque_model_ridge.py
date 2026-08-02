"""Tests for the ridge-regularization + output-clipping fix to
tools/analysis/fit_residual_torque_model.py (2026-08-01).

See docs/status/residual_torque_regression_pipeline_2026-08-01.md for the
full root-cause writeup: fitting the phase-1 residual-torque regression on
real UR5e hardware traces found that joints 2/3 (elbow/wrist_1), which spend
almost their entire trajectory near-stationary during an X-only transport
move, have a training design matrix with near-perfectly-collinear velocity
features (qd, tanh(qd/deadband), qd*|qd| are all ~linearly dependent for
small |qd|). Plain OLS (numpy.linalg.lstsq) produces enormous, poorly
determined weights there that fit training data fine but blow up
catastrophically -- held-out test R^2 in the tens of thousands negative --
the moment a held-out run has even a modestly larger |qd| than training ever
saw for that joint.

Marked ``mujoco`` per this directory's convention (auto-applied): the module
under test transitively imports ``tools.analysis.residual_data`` ->
``controller_core.model_dynamics.PinocchioUR5eDynamics`` at import time, even
though none of these specific tests exercise Pinocchio directly -- matching
the existing ``tests/mujoco/test_residual_data_pipeline.py``'s reason for
living here.
"""

from __future__ import annotations

import numpy as np
import pytest

from controller_core.residual_torque_model import NUM_FEATURES_PER_JOINT, NUM_JOINTS
from tools.analysis.fit_residual_torque_model import (
    compute_clip_bounds,
    fit_ols_weights,
    fit_ridge_weights,
    predict,
)

pytestmark = pytest.mark.mujoco


def _make_ill_conditioned_dataset(rng: np.random.Generator):
    """Synthetic dataset with the same structural pathology found in the real UR5e fit
    (docs/status/residual_torque_regression_pipeline_2026-08-01.md): a design matrix with one
    very small singular value (near-collinear features, exactly what happens to a near-
    stationary joint's qd/tanh(qd/deadband)/qd*|qd| columns), fit with real noise, then
    evaluated at a held-out point that extrapolates specifically along that near-null
    direction -- the exact mechanism that made OLS produce a coefficient of ~1e5-1e6 for a
    single feature and a held-out-set R^2 in the tens of thousands negative on the real data.

    Built via a controlled SVD (rather than hoping unstructured random features happen to be
    collinear) so the pathology and its magnitude are deterministic given the seed, not
    something that only sometimes reproduces.
    """
    n, p = 300, NUM_FEATURES_PER_JOINT
    u, _ = np.linalg.qr(rng.normal(size=(n, p)))
    v, _ = np.linalg.qr(rng.normal(size=(p, p)))
    singular_values = np.array([10.0, 8.0, 5.0, 3.0, 1.0e-2, 1.0e-4])  # last direction near-null
    design = u @ np.diag(singular_values) @ v.T
    w_true = rng.normal(size=p)
    noise = rng.normal(scale=0.01, size=n)
    target = design @ w_true + noise

    # A held-out point 5 units out along the near-null right-singular direction -- well
    # outside anything the (bounded, orthonormal-basis) training design ever produced.
    v_near_null = v[:, -1]
    x_test = v_near_null * 5.0
    y_test_true = w_true @ x_test
    return design, target, x_test, y_test_true


def test_ols_blows_up_but_ridge_bounds_prediction_on_collinear_extrapolation():
    rng = np.random.default_rng(0)
    design, target, x_test, y_test_true = _make_ill_conditioned_dataset(rng)
    assert np.linalg.cond(design) > 1.0e4  # confirm the construction is actually ill-conditioned

    w_ols, *_ = np.linalg.lstsq(design, target, rcond=None)
    pred_ols = float(x_test @ w_ols)

    x = np.repeat(design[:, np.newaxis, :], NUM_JOINTS, axis=1)
    y = np.repeat(target[:, np.newaxis], NUM_JOINTS, axis=1)
    w_ridge = fit_ridge_weights(x, y, ridge_lambda=100.0)
    pred_ridge = float(x_test @ w_ridge[0])

    err_ols = abs(pred_ols - y_test_true)
    err_ridge = abs(pred_ridge - y_test_true)
    # The whole point: OLS's prediction error on the out-of-range point is wildly larger than
    # ridge's, even though both fit the same (noisy but well-behaved) training data.
    assert err_ols > 10.0 * err_ridge
    # And ridge's prediction stays within a sane multiple of the true value's own scale,
    # unlike OLS's (whose |pred_ols| is tens of times larger than |y_test_true| here).
    assert abs(pred_ridge) < 5.0 * max(abs(y_test_true), 1.0)


def test_fit_ridge_weights_shrinks_as_lambda_grows():
    rng = np.random.default_rng(1)
    n = 500
    design = rng.normal(size=(n, NUM_FEATURES_PER_JOINT))
    target = design @ rng.normal(size=NUM_FEATURES_PER_JOINT) + 0.1 * rng.normal(size=n)
    x = np.repeat(design[:, np.newaxis, :], NUM_JOINTS, axis=1)
    y = np.repeat(target[:, np.newaxis], NUM_JOINTS, axis=1)

    norms = []
    for lam in (0.0, 1.0, 100.0, 1.0e5):
        w = fit_ridge_weights(x, y, ridge_lambda=lam)
        norms.append(np.linalg.norm(w[0]))
    assert norms == sorted(norms, reverse=True)  # monotonically shrinking


def test_fit_ridge_weights_lambda_zero_close_to_ols_on_well_conditioned_data():
    rng = np.random.default_rng(2)
    n = 1000
    design = rng.normal(size=(n, NUM_FEATURES_PER_JOINT))  # well-conditioned, no collinearity
    target = design @ rng.normal(size=NUM_FEATURES_PER_JOINT)
    x = np.repeat(design[:, np.newaxis, :], NUM_JOINTS, axis=1)
    y = np.repeat(target[:, np.newaxis], NUM_JOINTS, axis=1)

    w_ridge0 = fit_ridge_weights(x, y, ridge_lambda=0.0)
    w_ols = fit_ols_weights(x, y)
    np.testing.assert_allclose(w_ridge0, w_ols, atol=1e-6)


def test_fit_ridge_weights_rejects_negative_lambda():
    x = np.zeros((10, NUM_JOINTS, NUM_FEATURES_PER_JOINT))
    y = np.zeros((10, NUM_JOINTS))
    with pytest.raises(ValueError):
        fit_ridge_weights(x, y, ridge_lambda=-1.0)


def test_fit_ridge_weights_handles_too_few_rows_like_ols():
    # Fewer rows than features for one joint -> weights stay zero (matches fit_ols_weights).
    x = np.zeros((2, NUM_JOINTS, NUM_FEATURES_PER_JOINT))
    y = np.zeros((2, NUM_JOINTS))
    w = fit_ridge_weights(x, y, ridge_lambda=10.0)
    np.testing.assert_allclose(w, 0.0)


def test_compute_clip_bounds_is_multiple_of_max_abs_training_residual():
    y_train = np.array(
        [
            [1.0, -2.0, 0.0, 5.0, -0.1, 3.0],
            [-3.0, 1.0, 0.0, -4.0, 0.2, -1.0],
        ]
    )
    bounds = compute_clip_bounds(y_train, clip_multiple=2.0)
    expected = 2.0 * np.array([3.0, 2.0, 1e-9, 5.0, 0.2, 3.0])
    np.testing.assert_allclose(bounds, expected)


def test_compute_clip_bounds_rejects_nonpositive_multiple():
    y_train = np.ones((5, NUM_JOINTS))
    with pytest.raises(ValueError):
        compute_clip_bounds(y_train, clip_multiple=0.0)
    with pytest.raises(ValueError):
        compute_clip_bounds(y_train, clip_multiple=-1.0)


def test_predict_with_clip_bounds_only_affects_exceeding_rows():
    weights = np.zeros((NUM_JOINTS, NUM_FEATURES_PER_JOINT))
    weights[:, 0] = np.array([1.0, 100.0, -50.0, 2.0, 0.0, -3.0])  # bias-only weights
    x = np.zeros((1, NUM_JOINTS, NUM_FEATURES_PER_JOINT))
    x[0, :, 0] = 1.0  # bias feature on
    clip_bounds = np.full(NUM_JOINTS, 10.0)

    unclipped = predict(weights, x)
    clipped = predict(weights, x, clip_bounds=clip_bounds)

    np.testing.assert_allclose(unclipped[0], np.array([1.0, 100.0, -50.0, 2.0, 0.0, -3.0]))
    np.testing.assert_allclose(clipped[0], np.array([1.0, 10.0, -10.0, 2.0, 0.0, -3.0]))


def test_predict_clip_bounds_none_is_unchanged_default():
    rng = np.random.default_rng(3)
    weights = rng.normal(size=(NUM_JOINTS, NUM_FEATURES_PER_JOINT))
    x = rng.normal(size=(5, NUM_JOINTS, NUM_FEATURES_PER_JOINT))
    np.testing.assert_allclose(predict(weights, x), predict(weights, x, clip_bounds=None))
