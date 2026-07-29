"""Tests for hardware.joint_accel_estimator.JointAccelEstimator -- the
diagnostic-only qd -> qdd_measured estimator feeding the direct_torque
dynamics residual observer (see
docs/status/direct_torque_residual_observer_2026-07-29.md). This estimator
feeds no safety trip condition anywhere."""

from __future__ import annotations

import numpy as np
import pytest

from hardware.joint_accel_estimator import JointAccelEstimator


def test_requires_reset_before_update():
    estimator = JointAccelEstimator()
    with pytest.raises(RuntimeError):
        estimator.update(np.zeros(6), 0.002)


def test_validate_rejects_bad_gap_cycles():
    with pytest.raises(ValueError):
        JointAccelEstimator(gap_cycles=0)


def test_validate_rejects_bad_lowpass_alpha():
    with pytest.raises(ValueError):
        JointAccelEstimator(lowpass_alpha=0.0)
    with pytest.raises(ValueError):
        JointAccelEstimator(lowpass_alpha=1.5)


def test_returns_none_until_gap_window_fills():
    estimator = JointAccelEstimator(gap_cycles=3)
    estimator.reset(np.zeros(6))
    # gap_cycles=3: the first 2 update() calls have too little history.
    assert estimator.update(np.zeros(6), 0.002) is None
    assert estimator.update(np.zeros(6), 0.002) is None
    assert estimator.update(np.zeros(6), 0.002) is not None


def test_constant_velocity_gives_near_zero_qdd():
    estimator = JointAccelEstimator(gap_cycles=1)
    qd = np.full(6, 0.3)
    estimator.reset(qd)
    for _ in range(5):
        qdd = estimator.update(qd, 0.002)
    np.testing.assert_allclose(qdd, np.zeros(6), atol=1e-9)


def test_linear_ramp_matches_hand_computed_slope_at_gap_one():
    # qd ramps at exactly 2.0 rad/s^2 on every joint; dt=0.002s constant.
    # At gap_cycles=1, alpha=1.0 this is the original single-cycle finite
    # difference, so qdd_measured must recover the true 2.0 rad/s^2 slope
    # (up to floating point) every cycle once history is available.
    dt = 0.002
    slope = 2.0
    estimator = JointAccelEstimator(gap_cycles=1, lowpass_alpha=1.0)
    qd0 = np.zeros(6)
    estimator.reset(qd0)
    qd = qd0.copy()
    for step in range(1, 6):
        qd = qd0 + slope * dt * step
        qdd = estimator.update(qd, dt)
        np.testing.assert_allclose(qdd, np.full(6, slope), atol=1e-9)


def test_accel_gap_matches_hand_computed_values():
    # Deterministic single-joint qd sequence (dt=0.002s constant, gap=3):
    #   cycle: 0(reset) 1     2     3     4     5
    #   qd:    0.0      0.001 0.002 0.003 0.010 0.011
    # cycle 3 (index 2): first ready gap-window sample:
    #   qdd = (0.003 - 0.0) / (3*0.002) = 0.5 rad/s^2
    # cycle 4 (index 3): the real jump:
    #   qdd = (0.010 - 0.001) / (3*0.002) = 1.5 rad/s^2
    # cycle 5 (index 4): steady state after the jump has entered the window:
    #   qdd = (0.011 - 0.002) / (3*0.002) = 1.5 rad/s^2
    estimator = JointAccelEstimator(gap_cycles=3, lowpass_alpha=1.0)
    estimator.reset(np.zeros(6))
    qds = [0.001, 0.002, 0.003, 0.010, 0.011]
    outputs = []
    for v in qds:
        outputs.append(estimator.update(np.full(6, v), 0.002))
    assert outputs[0] is None
    assert outputs[1] is None
    np.testing.assert_allclose(outputs[2], np.full(6, 0.5), atol=1e-9)
    np.testing.assert_allclose(outputs[3], np.full(6, 1.5), atol=1e-9)
    np.testing.assert_allclose(outputs[4], np.full(6, 1.5), atol=1e-9)


def test_wider_gap_reduces_stationary_noise_sensitivity():
    # Same mechanism/rationale as
    # test_hardware_safety.py::test_cartesian_move_monitor_wider_gap_reduces_stationary_noise_sensitivity,
    # applied one differentiation order lower (qd -> qdd instead of pos ->
    # speed -> accel). A stationary, noisy qd sequence fed through gap=1 vs
    # gap=8 must show the wider gap producing substantially smaller peak
    # |qdd|.
    rng = np.random.default_rng(0)
    n = 300
    noisy_qds = [rng.normal(0.0, 1e-4, size=6) for _ in range(n)]

    def peak_abs_qdd(gap: int) -> float:
        estimator = JointAccelEstimator(gap_cycles=gap)
        estimator.reset(np.zeros(6))
        peak = 0.0
        for qd in noisy_qds:
            qdd = estimator.update(qd, 0.002)
            if qdd is not None:
                peak = max(peak, float(np.max(np.abs(qdd))))
        return peak

    peak_gap1 = peak_abs_qdd(1)
    peak_gap8 = peak_abs_qdd(8)
    assert peak_gap1 > 0.0
    assert peak_gap8 > 0.0
    assert peak_gap8 < peak_gap1 / 3.0, f"gap=1 peak {peak_gap1}, gap=8 peak {peak_gap8}"


def test_lowpass_filter_smooths_qdd_impulse_from_a_qd_step():
    # A step change in qd (0 -> 1, then held) is an IMPULSE in true qdd, not
    # a sustained step -- qd is constant before and after, so steady-state
    # qdd is 0 on both sides regardless of filtering. At gap_cycles=1,
    # alpha=1.0 (no filtering), hand-computed with dt=0.002s:
    #   update(0):   raw_qdd = (0-0)/0.002   = 0
    #   update(1):   raw_qdd = (1-0)/0.002   = 500   <- the impulse
    #   update(1):   raw_qdd = (1-1)/0.002   = 0     <- drops back instantly
    #   update(1):   raw_qdd = (1-1)/0.002   = 0
    # With alpha=0.2, the EMA spreads that same impulse energy over several
    # cycles instead of one: filtered = [0, 0.2*500=100, 0.8*100=80, 0.8*80=64].
    estimator_raw = JointAccelEstimator(gap_cycles=1, lowpass_alpha=1.0)
    estimator_filtered = JointAccelEstimator(gap_cycles=1, lowpass_alpha=0.2)
    estimator_raw.reset(np.zeros(6))
    estimator_filtered.reset(np.zeros(6))

    qd_sequence = [np.zeros(6), np.full(6, 1.0), np.full(6, 1.0), np.full(6, 1.0)]
    raw_outputs = [estimator_raw.update(qd, 0.002) for qd in qd_sequence]
    filtered_outputs = [estimator_filtered.update(qd, 0.002) for qd in qd_sequence]

    np.testing.assert_allclose(raw_outputs[1], np.full(6, 500.0), atol=1e-9)
    np.testing.assert_allclose(raw_outputs[2], np.zeros(6), atol=1e-9)  # raw drops instantly
    np.testing.assert_allclose(filtered_outputs[1], np.full(6, 100.0), atol=1e-9)
    np.testing.assert_allclose(filtered_outputs[2], np.full(6, 80.0), atol=1e-9)
    np.testing.assert_allclose(filtered_outputs[3], np.full(6, 64.0), atol=1e-9)
    # The filtered peak is far lower than the raw peak, and decays gradually
    # (still nonzero two cycles after the impulse) instead of instantly.
    assert np.max(np.abs(filtered_outputs[1])) < np.max(np.abs(raw_outputs[1]))
    assert np.max(np.abs(filtered_outputs[2])) > 0.0
