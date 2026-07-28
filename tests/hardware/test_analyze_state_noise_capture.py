"""Tests for tools/analyze_state_noise_capture.py."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "tools"))

from analyze_state_noise_capture import (  # noqa: E402
    compute_guard_quantities,
    percentiles,
    recommend_threshold,
)


def _rows_from(t_s, tcp_xyz):
    return [{"t_s": float(t), "tcp_pose": [float(x), float(y), float(z), 0.0, 0.0, 0.0]} for t, (x, y, z) in zip(t_s, tcp_xyz)]


def test_compute_guard_quantities_zero_for_perfectly_stationary_constant_dt():
    n = 50
    dt = 0.002
    t = np.arange(n) * dt
    pos = np.tile([0.1, 0.2, 0.3], (n, 1))
    q = compute_guard_quantities(_rows_from(t, pos))
    assert q["n_dt"] == n - 1
    assert q["n_accel"] == n - 2
    assert np.allclose(q["step_m"], 0.0)
    assert np.allclose(q["speed_mps"], 0.0)
    assert np.allclose(q["accel_mps2"], 0.0)
    assert np.allclose(q["dt_ms"], dt * 1e3)


def test_compute_guard_quantities_matches_hand_computed_single_step():
    # Two samples, 2ms apart, moving 1mm in X: speed = 0.001/0.002 = 0.5 m/s.
    t = [0.0, 0.002, 0.004]
    pos = [[0.0, 0.0, 0.0], [0.001, 0.0, 0.0], [0.001, 0.0, 0.0]]
    q = compute_guard_quantities(_rows_from(t, pos))
    np.testing.assert_allclose(q["speed_mps"], [0.5, 0.0], atol=1e-9)
    # accel: |0.0 - 0.5| / 0.002 = 250.0
    np.testing.assert_allclose(q["accel_mps2"], [250.0], atol=1e-6)


def test_compute_guard_quantities_rejects_nonpositive_dt():
    import pytest

    t = [0.0, 0.002, 0.001]  # goes backwards
    pos = [[0.0, 0.0, 0.0]] * 3
    with pytest.raises(ValueError):
        compute_guard_quantities(_rows_from(t, pos))


def test_percentiles_returns_requested_keys():
    x = np.arange(1, 101, dtype=np.float64)  # 1..100
    p = percentiles(x, ps=(50, 99))
    assert p["p50"] == 50.5
    assert p["p99"] == 99.01


def test_recommend_threshold_is_a_percentile_of_accel():
    accel = np.concatenate([np.full(999, 0.1), np.array([5.0])])  # 1 outlier in 1000
    thresh_99_9 = recommend_threshold(accel, 99.9)
    # The 99.9th percentile of 1000 samples with one outlier should sit
    # right at or just under the outlier, comfortably above the bulk (0.1).
    assert thresh_99_9 > 0.1
    assert thresh_99_9 <= 5.0
