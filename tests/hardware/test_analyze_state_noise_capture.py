"""Tests for tools/analyze_state_noise_capture.py."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "tools"))

from analyze_state_noise_capture import (  # noqa: E402
    compute_guard_quantities,
    percentiles,
    recommend_threshold,
)
from hardware.safety import CartesianMoveLimits, CartesianMoveMonitor  # noqa: E402


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


def test_compute_guard_quantities_gap_matches_hand_computed_values():
    # Same scenario as
    # tests/hardware/test_hardware_safety.py::
    # test_cartesian_move_monitor_accel_gap_matches_hand_computed_values --
    # kept in sync deliberately so both implementations agree on the same
    # real numbers.
    t = [0.0, 0.002, 0.004, 0.006, 0.008, 0.010]
    xs = [0.0, 0.001, 0.002, 0.003, 0.010, 0.011]
    pos = [(x, 0.0, 0.5) for x in xs]
    q = compute_guard_quantities(_rows_from(t, pos), gap=3, alpha=1.0)
    # accel index 0 corresponds to the transition into cycle 4 (the real
    # jump): 500.0 m/s^2 exactly, hand-verified in the sibling test.
    assert q["accel_mps2"][0] == pytest.approx(500.0, abs=1e-6)
    # accel index 1 (cycle 5, settled back to steady state): 0.0.
    assert q["accel_mps2"][1] == pytest.approx(0.0, abs=1e-9)


def test_compute_guard_quantities_cross_checked_against_live_monitor():
    # Direct cross-check against CartesianMoveMonitor itself (not just a
    # hand-derivation) for a representative gap/alpha combination, driving
    # both from the SAME synthetic (t, pos) sequence -- guards against the
    # two implementations silently drifting apart.
    rng = np.random.default_rng(1)
    n = 60
    dt = 0.002
    t = np.arange(n) * dt
    true_pos = np.array([0.0, 0.0, 0.5])
    pos = true_pos + rng.normal(0.0, 1e-5, size=(n, 3))
    # Inject a real, sustained motion partway through so accel isn't
    # trivially near-zero throughout.
    pos[30:] += np.arange(n - 30)[:, None] * np.array([0.0005, 0.0, 0.0])

    gap, alpha = 4, 0.4
    q = compute_guard_quantities(_rows_from(t, pos), gap=gap, alpha=alpha)

    monitor = CartesianMoveMonitor(
        CartesianMoveLimits(
            accel_gap_cycles=gap,
            speed_lowpass_alpha=alpha,
            max_tcp_accel_mps2=1e-12,  # trip on everything so every reading is reported
            max_tcp_speed_mps=1e9,
            max_waypoint_jump_m=1e9,
        )
    )
    monitor.set_start(np.concatenate([pos[0], [0.0, 0.0, 0.0]]), move_axis_index=0)
    live_accels = []
    for i in range(1, n):
        decision = monitor.check(
            q=np.zeros(6),
            qd=np.zeros(6),
            tcp_pose=np.concatenate([pos[i], [0.0, 0.0, 0.0]]),
            target_tcp_pose=np.concatenate([pos[i], [0.0, 0.0, 0.0]]),
            orientation_error_rad=0.0,
            axis_target_moving=True,
            dt_s=dt,
        )
        for reason in decision.reasons:
            if reason.startswith("TCP acceleration"):
                live_accels.append(float(reason.split()[2]))

    assert len(live_accels) == len(q["accel_mps2"])
    # atol matches the 4-decimal-place string formatting in
    # SafetyDecision's reason text (the only way to read the monitor's
    # internal accel value without reaching into private state) -- not a
    # real precision gap in the underlying math.
    np.testing.assert_allclose(live_accels, q["accel_mps2"], atol=5e-5)
