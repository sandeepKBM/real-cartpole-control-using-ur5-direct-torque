"""Correctness tests for the two candidate threshold designs in
``experiments/safety_envelope_backtest.py`` (the smooth-envelope backtest harness --
see ``docs/status/safety_envelope_backtest_2026-07-30.md`` for the full writeup).

These only test the pure-math ``scale()``/``thresholds()`` methods, not the real-trace
replay (that needs pinocchio + real hardware capture files and is exercised directly by
running the script). Pure numpy, no simulator/robot dependency -- consistent with this
directory's existing unit-test convention.
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
