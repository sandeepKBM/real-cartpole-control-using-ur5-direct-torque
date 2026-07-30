"""Correctness tests for the candidate threshold designs in
``experiments/safety_envelope_backtest.py`` (the smooth-envelope backtest harness --
see ``docs/status/safety_envelope_backtest_2026-07-30.md`` for the full writeup).

These only test the pure-math ``scale()``/``growth_rate()``/``thresholds()`` methods,
not the real-trace replay (that needs pinocchio + real hardware capture files and is
exercised directly by running the script). Pure numpy, no simulator/robot dependency --
consistent with this directory's existing unit-test convention.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "experiments"))

from safety_envelope_backtest import (  # noqa: E402
    CondJScaledCandidate,
    MoveTimingCbfCandidate,
    QdGrowthRateCandidate,
)


class TestMoveTimingCbfCandidate:
    def test_full_ceiling_during_move(self) -> None:
        cbf = MoveTimingCbfCandidate(settle_window_s=0.5, floor_fraction=0.2)
        assert cbf.scale(t_s=0.0, move_duration_s=1.0) == pytest.approx(1.0)
        assert cbf.scale(t_s=0.999, move_duration_s=1.0) == pytest.approx(1.0)
        assert cbf.scale(t_s=1.0, move_duration_s=1.0) == pytest.approx(1.0)

    def test_shrinks_smoothly_over_settle_window(self) -> None:
        cbf = MoveTimingCbfCandidate(settle_window_s=0.5, floor_fraction=0.2)
        s0 = cbf.scale(t_s=1.0, move_duration_s=1.0)
        s_mid = cbf.scale(t_s=1.25, move_duration_s=1.0)
        s_end = cbf.scale(t_s=1.5, move_duration_s=1.0)
        assert s0 > s_mid > s_end
        assert s_end == pytest.approx(0.2)

    def test_floor_holds_past_settle_window(self) -> None:
        cbf = MoveTimingCbfCandidate(settle_window_s=0.5, floor_fraction=0.2)
        assert cbf.scale(t_s=5.0, move_duration_s=1.0) == pytest.approx(0.2)
        assert cbf.scale(t_s=30.0, move_duration_s=1.0) == pytest.approx(0.2)

    def test_thresholds_scale_both_accel_and_speed_identically(self) -> None:
        cbf = MoveTimingCbfCandidate(settle_window_s=0.5, floor_fraction=0.2)
        accel_thr, speed_thr = cbf.thresholds(
            t_s=1.5, move_duration_s=1.0, base_accel=0.8, base_speed=0.05,
        )
        assert accel_thr == pytest.approx(0.8 * 0.2)
        assert speed_thr == pytest.approx(0.05 * 0.2)


class TestCondJScaledCandidate:
    """``scale()`` is pure math over a precomputed cond(J) value -- no pinocchio
    needed, so ``dynamics=None`` is safe here (only ``cond_of()`` touches it)."""

    def test_well_conditioned_pose_gets_full_ceiling(self) -> None:
        cand = CondJScaledCandidate(None, cond_low=2.0e2, cond_high=5.0e4, floor_fraction=0.2)
        assert cand.scale(1.0) == pytest.approx(1.0)
        assert cand.scale(cand.cond_low) == pytest.approx(1.0)

    def test_singular_pose_gets_floor(self) -> None:
        cand = CondJScaledCandidate(None, cond_low=2.0e2, cond_high=5.0e4, floor_fraction=0.2)
        assert cand.scale(cand.cond_high) == pytest.approx(0.2)
        assert cand.scale(1.0e8) == pytest.approx(0.2)  # clipped past cond_high

    def test_monotonically_shrinks_in_log_cond_space(self) -> None:
        cand = CondJScaledCandidate(None, cond_low=2.0e2, cond_high=5.0e4, floor_fraction=0.2)
        conds = [2.0e2, 1.0e3, 1.0e4, 1.0e4 * 5, 5.0e4]
        scales = [cand.scale(c) for c in conds]
        assert scales == sorted(scales, reverse=True)
        assert scales[0] == pytest.approx(1.0)
        assert scales[-1] == pytest.approx(0.2)

    def test_thresholds_returns_cond_alongside_scaled_values(self) -> None:
        cand = CondJScaledCandidate(None, cond_low=2.0e2, cond_high=5.0e4, floor_fraction=0.2)
        # Bypass cond_of() (needs a real dynamics provider) by calling scale()
        # directly and checking the same relationship thresholds() documents.
        scale = cand.scale(5.0e4)
        assert scale * 0.5 == pytest.approx(0.5 * 0.2)


class TestQdGrowthRateCandidate:
    """The core hypothesis under test (see
    docs/status/safety_envelope_backtest_2026-07-30.md SS8): a growth-RATE metric
    should give full ceiling to a pose that is high-magnitude but FLAT (the real
    wrist_2=0 transport pose Candidate B nuisance-tripped on), and only tighten for
    a pose whose risk quantity is actually climbing cycle-over-cycle (the real
    disqualifying divergence in position_20260728_150847)."""

    def test_growth_rate_needs_a_full_window_before_reacting(self) -> None:
        cand = QdGrowthRateCandidate(window=3, qd_floor=5.0e-3)
        # Feed fewer than `window` samples -- growth_rate() must not react yet
        # (a fresh run/pose starting out should not be pre-tightened).
        assert cand.growth_rate(0.01) == pytest.approx(0.0)
        assert cand.growth_rate(0.02) == pytest.approx(0.0)
        assert cand.growth_rate(0.03) == pytest.approx(0.0)

    def test_flat_static_signal_yields_zero_growth_rate(self) -> None:
        """The core distinguishing case: a large but CONSTANT |qd| (e.g. steady
        real jitter at a singular-but-fine pose) must read as ~0 growth, not as
        dangerous -- this is exactly what should differ from Candidate B, which
        would stay tight here forever purely because the pose is singular."""
        cand = QdGrowthRateCandidate(window=3, qd_floor=5.0e-3)
        r = 0.0
        for _ in range(8):
            r = cand.growth_rate(0.05)  # same value every cycle
        assert r == pytest.approx(0.0, abs=1e-9)

    def test_sustained_exponential_growth_is_measured_correctly(self) -> None:
        """Reproduces the real disqualifying case's documented signature
        (~1.6-1.8x per 8ms step, i.e. growth_rate ~0.6-0.8) with a synthetic
        exact-geometric sequence, and checks growth_rate() recovers the known
        per-step growth factor to high precision."""
        cand = QdGrowthRateCandidate(window=4, qd_floor=1.0e-6)
        true_factor = 1.7  # matches the documented ~1.6-1.8x/step range
        qd = 0.01
        r = 0.0
        for _ in range(10):
            r = cand.growth_rate(qd)
            qd *= true_factor
        assert r == pytest.approx(true_factor - 1.0, rel=1e-6)

    def test_decaying_signal_clips_to_zero_growth_not_negative(self) -> None:
        """A shrinking trend (settling down, not diverging) must not read as
        "extra safe" and loosen anything beyond the baseline ceiling -- scale()
        should clip negative growth_rate to the same 1.0 as flat."""
        cand = QdGrowthRateCandidate(window=3, qd_floor=1.0e-6)
        r = 0.0
        for qd in (0.5, 0.4, 0.3, 0.2, 0.1, 0.05):
            r = cand.growth_rate(qd)
        assert r < 0.0  # the raw estimate is negative (shrinking)...
        assert cand.scale(r) == pytest.approx(1.0)  # ...but scale() floors it at full ceiling

    def test_noise_floor_prevents_spurious_growth_at_near_zero_qd(self) -> None:
        """Without qd_floor, two near-zero noise samples (e.g. 2e-4 vs 1e-4 rad/s,
        the real ~1e-4 rad/s stationary noise level measured on this robot) would
        read as "100% growth" and defeat the whole point of this candidate."""
        cand_floored = QdGrowthRateCandidate(window=3, qd_floor=5.0e-3)
        r = 0.0
        for qd in (1.0e-4, 2.0e-4, 1.0e-4, 2.0e-4):
            r = cand_floored.growth_rate(qd)
        assert r == pytest.approx(0.0, abs=1e-9)  # both floored to qd_floor -> flat

    def test_scale_monotonic_between_r_low_and_r_high(self) -> None:
        cand = QdGrowthRateCandidate(r_low=0.05, r_high=0.5, floor_fraction=0.2)
        rates = [0.0, 0.05, 0.2, 0.35, 0.5, 1.0]
        scales = [cand.scale(r) for r in rates]
        assert scales == sorted(scales, reverse=True)
        assert scales[0] == pytest.approx(1.0)
        assert scales[-1] == pytest.approx(0.2)

    def test_reset_clears_history(self) -> None:
        cand = QdGrowthRateCandidate(window=3, qd_floor=1.0e-6)
        for qd in (0.01, 0.02, 0.04, 0.08):
            cand.growth_rate(qd)
        cand.reset()
        # Immediately after reset, behaves like a fresh instance -- not enough
        # history yet, so growth_rate() must return 0.0 again.
        assert cand.growth_rate(0.5) == pytest.approx(0.0)

    def test_thresholds_scale_both_accel_and_speed_and_return_growth_rate(self) -> None:
        cand = QdGrowthRateCandidate(window=2, qd_floor=1.0e-6, r_low=0.05, r_high=0.5, floor_fraction=0.2)
        for qd_val in (0.01, 0.017, 0.0289):  # ~1.7x growth per step
            accel_thr, speed_thr, r = cand.thresholds(
                qd=[qd_val, 0.0, 0.0, 0.0, 0.0, 0.0], base_accel=0.8, base_speed=0.05,
            )
        assert r == pytest.approx(0.7, rel=0.05)
        expected_scale = cand.scale(r)
        assert accel_thr == pytest.approx(0.8 * expected_scale)
        assert speed_thr == pytest.approx(0.05 * expected_scale)
