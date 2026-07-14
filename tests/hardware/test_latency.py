"""Tests for hardware.latency."""

from __future__ import annotations

from hardware.latency import PhaseLatencyRecorder


def test_phase_latency_summary_and_dominant_phase() -> None:
    rec = PhaseLatencyRecorder()
    for _ in range(10):
        rec.record("read_state_ns", 500_000)
        rec.record("get_jacobian_ns", 2_000_000)
        rec.record("get_mass_matrix_ns", 1_800_000)
        rec.record("build_state_ns", 50_000)
        rec.record("controller_ns", 200_000)
        rec.record("safety_ns", 30_000)
        rec.record("direct_torque_ns", 40_000)
        rec.record("sleep_ns", 100_000)
        rec.record("total_work_ns", 4_620_000)
        rec.record("lateness_ns", 0)

    summary = rec.summary()
    assert summary["phase_count"] == 10
    assert summary["dominant_phase"] == "get_jacobian"
    assert summary["read_state_mean_ms"] == 0.5
    assert summary["get_jacobian_p95_ms"] == 2.0
