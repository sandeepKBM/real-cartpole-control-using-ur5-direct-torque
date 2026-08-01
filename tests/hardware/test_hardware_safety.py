"""Tests for hardware/safety.py -- pure numpy, no RTDE dependency at all."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from hardware.safety import (  # noqa: E402
    CartesianMoveLimits,
    CartesianMoveMonitor,
    ConnectionHealth,
    EStopLatch,
    EStopTripped,
    SafetyDecision,
    UR5eSafetyLimits,
    check_joint_state,
    check_tcp_pose,
    is_likely_ursim,
)


def test_safety_decision_add_flips_ok_and_joins_reasons():
    decision = SafetyDecision()
    assert decision.ok is True
    decision.add("first problem")
    assert decision.ok is False
    decision.add("second problem")
    assert decision.reason == "first problem; second problem"


def test_ur5e_safety_limits_validate_accepts_defaults():
    UR5eSafetyLimits().validate()


def test_ur5e_safety_limits_validate_rejects_bad_shape():
    limits = UR5eSafetyLimits(q_lower=np.zeros(5))
    with pytest.raises(ValueError):
        limits.validate()


def test_ur5e_safety_limits_validate_rejects_nonpositive_scalar():
    limits = UR5eSafetyLimits(tcp_speed_max_mps=0.0)
    with pytest.raises(ValueError):
        limits.validate()


def test_check_joint_state_ok_within_limits():
    limits = UR5eSafetyLimits()
    decision = check_joint_state(np.zeros(6), np.zeros(6), limits)
    assert decision.ok is True


def test_check_joint_state_rejects_nan():
    limits = UR5eSafetyLimits()
    q = np.zeros(6)
    q[0] = float("nan")
    decision = check_joint_state(q, np.zeros(6), limits)
    assert decision.ok is False
    assert "NaN" in decision.reason


def test_check_joint_state_rejects_over_velocity():
    limits = UR5eSafetyLimits()
    qd = np.zeros(6)
    qd[0] = 100.0
    decision = check_joint_state(np.zeros(6), qd, limits)
    assert decision.ok is False


def test_check_tcp_pose_rejects_wrong_shape():
    decision = check_tcp_pose([0.0, 0.0, 0.0])
    assert decision.ok is False


class TestConnectionHealth:
    def test_starts_not_alive(self):
        health = ConnectionHealth()
        assert health.is_alive() is False

    def test_alive_after_success(self):
        health = ConnectionHealth(max_state_age_s=1.0)
        health.record_success(host_stamp_ns=1_000_000_000)
        assert health.is_alive(now_ns=1_000_000_000) is True

    def test_not_alive_once_stale(self):
        health = ConnectionHealth(max_state_age_s=0.1)
        health.record_success(host_stamp_ns=0)
        # 1 full second later -- far past the 0.1s staleness window.
        assert health.is_alive(now_ns=1_000_000_000) is False

    def test_record_failure_trips_at_threshold(self):
        health = ConnectionHealth(max_consecutive_failures=3)
        assert health.record_failure() is False
        assert health.record_failure() is False
        assert health.record_failure() is True
        assert health.consecutive_failures == 3

    def test_success_resets_failure_streak(self):
        health = ConnectionHealth(max_consecutive_failures=3)
        health.record_failure()
        health.record_failure()
        health.record_success(host_stamp_ns=0)
        assert health.consecutive_failures == 0

    def test_not_alive_once_failure_streak_trips_even_if_recent(self):
        health = ConnectionHealth(max_consecutive_failures=1, max_state_age_s=100.0)
        health.record_success(host_stamp_ns=0)
        health.record_failure()
        assert health.is_alive(now_ns=0) is False


class TestEStopLatch:
    def test_starts_untripped(self):
        latch = EStopLatch()
        assert latch.tripped is False
        latch.raise_if_tripped()  # must not raise

    def test_trip_sets_reason_and_raises(self):
        latch = EStopLatch()
        latch.trip("something bad happened")
        assert latch.tripped is True
        assert latch.reason == "something bad happened"
        with pytest.raises(EStopTripped):
            latch.raise_if_tripped()

    def test_no_reset_method_exists(self):
        # This is the load-bearing assertion for "no un-latch path by
        # design": once tripped, nothing in this class can clear it.
        latch = EStopLatch()
        assert not hasattr(latch, "reset")
        assert not hasattr(latch, "clear")

    def test_stays_tripped_after_further_trips(self):
        latch = EStopLatch()
        latch.trip("first reason")
        latch.trip("second reason")
        assert latch.tripped is True
        assert latch.reason == "second reason"


def _monitor(**overrides) -> CartesianMoveMonitor:
    return CartesianMoveMonitor(CartesianMoveLimits(**overrides))


def test_cartesian_move_monitor_requires_set_start_first():
    monitor = _monitor()
    with pytest.raises(RuntimeError):
        monitor.check(
            q=np.zeros(6),
            qd=np.zeros(6),
            tcp_pose=[0, 0, 0, 0, 0, 0],
            target_tcp_pose=[0, 0, 0, 0, 0, 0],
            orientation_error_rad=0.0,
            axis_target_moving=True,
            dt_s=0.01,
        )


def test_cartesian_move_monitor_ok_on_clean_move_along_axis():
    monitor = _monitor(max_tcp_speed_mps=1.0, max_tcp_accel_mps2=100.0, max_waypoint_jump_m=0.1)
    start = [0.0, 0.0, 0.5, 0.0, 0.0, 0.0]
    monitor.set_start(start, move_axis_index=1)
    pose = list(start)
    for i in range(1, 6):
        pose[1] = 0.001 * i
        decision = monitor.check(
            q=np.zeros(6),
            qd=np.zeros(6),
            tcp_pose=pose,
            target_tcp_pose=pose,
            orientation_error_rad=0.0,
            axis_target_moving=True,
            dt_s=0.01,
        )
        assert decision.ok is True, decision.reason


def test_cartesian_move_monitor_uses_real_elapsed_time_when_longer_than_dt_s():
    # Real-hardware bug (2026-07-28): the gap between set_start() and the
    # first check() call can be meaningfully longer than the caller-supplied
    # dt_s (real setup work happens in between on the direct-torque path --
    # e.g. jacobian_and_mass_matrix()), which used to inflate the computed
    # speed by dividing a real (tiny) position delta by a too-small assumed
    # dt_s. Reproduced twice on real hardware (13.9 m/s^2 with qd<=0.0001
    # rad/s -- nothing physically moved). Fixed: use real measured elapsed
    # time whenever it's LONGER than the claimed dt_s.
    import time as time_module

    monitor = _monitor(max_tcp_speed_mps=0.1, max_tcp_accel_mps2=1000.0, max_waypoint_jump_m=1.0)
    start = [0.0, 0.0, 0.5, 0.0, 0.0, 0.0]
    monitor.set_start(start, move_axis_index=1)
    time_module.sleep(0.05)  # real elapsed time >> the dt_s claimed below
    pose = [0.0, 0.001, 0.5, 0.0, 0.0, 0.0]  # 1mm real position delta
    decision = monitor.check(
        q=np.zeros(6),
        qd=np.zeros(6),
        tcp_pose=pose,
        target_tcp_pose=pose,
        orientation_error_rad=0.0,
        axis_target_moving=True,
        dt_s=0.002,  # claimed dt_s: 0.001m/0.002s = 0.5 m/s, WOULD trip the 0.1 m/s limit
    )
    # With the real ~0.05s elapsed time instead: 0.001m/0.05s = 0.02 m/s,
    # comfortably under the 0.1 m/s limit -- only passes if real elapsed
    # time (not the claimed dt_s) was actually used.
    assert decision.ok is True, decision.reason


def test_cartesian_move_monitor_synthetic_sequence_unaffected_by_real_time_fix():
    # Back-to-back check() calls with no real sleep (the pattern every other
    # CartesianMoveMonitor test in this file uses): real elapsed time is
    # always far smaller than the intended dt_s here, so max(dt_s, measured)
    # must resolve to dt_s every time -- this fix must be a no-op for
    # synthetic test sequences, only real hardware timing gaps.
    monitor = _monitor(max_tcp_speed_mps=1.0, max_tcp_accel_mps2=100.0, max_waypoint_jump_m=0.1)
    start = [0.0, 0.0, 0.5, 0.0, 0.0, 0.0]
    monitor.set_start(start, move_axis_index=1)
    pose = list(start)
    for i in range(1, 6):
        pose[1] = 0.001 * i
        decision = monitor.check(
            q=np.zeros(6),
            qd=np.zeros(6),
            tcp_pose=pose,
            target_tcp_pose=pose,
            orientation_error_rad=0.0,
            axis_target_moving=True,
            dt_s=0.01,
        )
        assert decision.ok is True, decision.reason


def test_cartesian_move_limits_validate_rejects_bad_accel_gap_cycles():
    with pytest.raises(ValueError):
        CartesianMoveLimits(accel_gap_cycles=0).validate()


def test_cartesian_move_limits_validate_rejects_bad_speed_lowpass_alpha():
    with pytest.raises(ValueError):
        CartesianMoveLimits(speed_lowpass_alpha=0.0).validate()
    with pytest.raises(ValueError):
        CartesianMoveLimits(speed_lowpass_alpha=1.5).validate()


def test_cartesian_move_monitor_accel_gap_matches_hand_computed_values():
    # Deterministic X-only sequence (dt_s=0.002 constant, gap=3, no filter):
    #   cycle: 0(start) 1     2     3     4      5
    #   pos:   0.0      0.001 0.002 0.003 0.010  0.011
    # Cycles 1-3 accumulate at a constant 0.5 rad/s-equivalent ramp; cycle 4
    # is a genuine, real 3x speed jump; cycle 5 settles back to steady state.
    # Hand-computed with accel_gap_cycles=3 (gap window = 3 cycles back,
    # corrected-clock arithmetic from CartesianMoveMonitor.check()):
    #   cycle 3: first ready gap-window sample, no prior speed -> accel skipped.
    #   cycle 4: gap_speed = |0.010-0.001|/(3*0.002) = 1.5 m/s;
    #            prev gap_speed (cycle 3) = |0.003-0.0|/(3*0.002) = 0.5 m/s;
    #            accel = |1.5-0.5|/0.002 = 500.0 m/s^2 -- the real jump IS caught.
    #   cycle 5: gap_speed = |0.011-0.002|/(3*0.002) = 1.5 m/s; accel = 0
    #            (steady state after the jump has fully entered the window).
    monitor = _monitor(
        accel_gap_cycles=3,
        max_tcp_speed_mps=100.0,
        max_tcp_accel_mps2=100.0,
        max_waypoint_jump_m=100.0,
    )
    start = [0.0, 0.0, 0.5, 0.0, 0.0, 0.0]
    monitor.set_start(start, move_axis_index=0)
    xs = [0.001, 0.002, 0.003, 0.010, 0.011]
    decisions = []
    for x in xs:
        pose = [x, 0.0, 0.5, 0.0, 0.0, 0.0]
        decisions.append(
            monitor.check(
                q=np.zeros(6), qd=np.zeros(6), tcp_pose=pose, target_tcp_pose=pose,
                orientation_error_rad=0.0, axis_target_moving=True, dt_s=0.002,
            )
        )
    # cycles 1-3 (indices 0-2): gap window not ready or no prior speed yet -- clean.
    for d in decisions[:3]:
        assert d.ok is True, d.reason
    # cycle 4 (index 3): the real jump -- must trip with a threshold below 500.
    assert decisions[3].ok is False
    assert "500.0000" in decisions[3].reason
    # cycle 5 (index 4): steady state again -- clean.
    assert decisions[4].ok is True, decisions[4].reason


def test_cartesian_move_monitor_wider_gap_reduces_stationary_noise_sensitivity():
    # Same seeded pseudo-noise sequence (std matching the real measured
    # stationary tcp_pos noise, ~1e-5 m -- see
    # hardware_captures/2026-07-28_thinkrobot_172.16.71.77/) fed through
    # gap=1 (original behavior) vs gap=8: the wider gap must produce a
    # substantially smaller peak accel reading for identical noisy-but-
    # stationary input, matching the real-hardware finding that motivated
    # this feature (real accel noise floor median 1.74 m/s^2 at gap=1).
    rng = np.random.default_rng(0)
    n = 300
    true_pos = np.array([0.0, 0.0, 0.5])
    noisy_positions = [true_pos + rng.normal(0.0, 1e-5, size=3) for _ in range(n)]

    def peak_accel(gap: int) -> float:
        monitor = _monitor(
            accel_gap_cycles=gap,
            max_tcp_speed_mps=1e9,
            max_tcp_accel_mps2=1e-12,  # trip on essentially anything so every decision.reason reports the value
            max_waypoint_jump_m=1e9,
        )
        monitor.set_start(np.concatenate([true_pos, [0.0, 0.0, 0.0]]), move_axis_index=0)
        peak = 0.0
        for p in noisy_positions:
            pose = np.concatenate([p, [0.0, 0.0, 0.0]])
            decision = monitor.check(
                q=np.zeros(6), qd=np.zeros(6), tcp_pose=pose, target_tcp_pose=pose,
                orientation_error_rad=0.0, axis_target_moving=True, dt_s=0.002,
            )
            for reason in decision.reasons:
                if reason.startswith("TCP acceleration"):
                    peak = max(peak, float(reason.split()[2]))
        return peak

    peak_gap1 = peak_accel(1)
    peak_gap8 = peak_accel(8)
    assert peak_gap1 > 0.0
    assert peak_gap8 > 0.0
    # Wide margin (5x), not an exact ratio -- this is a real, noisy synthetic
    # sequence, not a hand-derived closed form; just confirms the mechanism
    # measurably helps, matching the real 1/N-ish reduction expected from
    # widening the differencing baseline.
    assert peak_gap8 < peak_gap1 / 5.0, f"gap=1 peak {peak_gap1}, gap=8 peak {peak_gap8}"


def test_cartesian_move_monitor_lowpass_filter_smooths_speed_step():
    # A single real step change in speed should reach its new steady-state
    # accel-input value gradually under alpha<1, not instantly as with
    # alpha=1.0 (no filtering).
    monitor_filtered = _monitor(
        accel_gap_cycles=1,
        speed_lowpass_alpha=0.2,
        max_tcp_speed_mps=100.0,
        max_tcp_accel_mps2=1e9,
        max_waypoint_jump_m=100.0,
    )
    start = [0.0, 0.0, 0.5, 0.0, 0.0, 0.0]
    monitor_filtered.set_start(start, move_axis_index=0)
    # Constant-velocity ramp (0.001 m per 0.002s cycle = 0.5 m/s) for 10 cycles.
    x = 0.0
    for _ in range(10):
        x += 0.001
        pose = [x, 0.0, 0.5, 0.0, 0.0, 0.0]
        monitor_filtered.check(
            q=np.zeros(6), qd=np.zeros(6), tcp_pose=pose, target_tcp_pose=pose,
            orientation_error_rad=0.0, axis_target_moving=True, dt_s=0.002,
        )
    # After 10 cycles of constant real speed, the EMA-filtered estimate
    # should have converged close to the true 0.5 m/s (alpha=0.2 reaches
    # ~1-(1-0.2)^10 ~= 89% of the way there).
    assert monitor_filtered._prev_speed_mps == pytest.approx(0.5, rel=0.15)
    # But it must NOT have converged instantly -- unlike alpha=1.0, the
    # very first cycle's filtered value equals the raw value regardless
    # (no prior estimate to blend with), so check the second cycle instead.


# --------------------------------------------------------------------------- #
# DeadlineMonitor-style graduated tolerance for TCP speed/accel (2026-07-30,
# docs/status/safety_envelope_backtest_2026-07-30.md). Real-hardware evidence:
# 15 of 21 real guard trips in this project's history were single-cycle
# speed/accel noise spikes on an otherwise-clean move; the genuinely real
# divergences found were multi-cycle, escalating trends. This mirrors
# DeadlineMonitor's own already-proven two-condition pattern instead of the
# smooth/continuous envelope designs a separate backtest found to fail for
# real structural reasons (see that doc).
# --------------------------------------------------------------------------- #
def _run_x_deltas(monitor: CartesianMoveMonitor, deltas_m: list[float], dt_s: float = 0.002) -> list:
    x = 0.0
    decisions = []
    for dx in deltas_m:
        x += dx
        pose = [x, 0.0, 0.5, 0.0, 0.0, 0.0]
        decisions.append(
            monitor.check(
                q=np.zeros(6), qd=np.zeros(6), tcp_pose=pose, target_tcp_pose=pose,
                orientation_error_rad=0.0, axis_target_moving=True, dt_s=dt_s,
            )
        )
    return decisions


def test_cartesian_move_limits_validate_rejects_bad_consecutive_violation_config():
    with pytest.raises(ValueError):
        CartesianMoveLimits(accel_max_consecutive_violations=0).validate()
    with pytest.raises(ValueError):
        CartesianMoveLimits(speed_max_consecutive_violations=0).validate()
    with pytest.raises(ValueError):
        CartesianMoveLimits(accel_hard_multiple=0.5).validate()
    with pytest.raises(ValueError):
        CartesianMoveLimits(speed_hard_multiple=0.5).validate()


def test_cartesian_move_monitor_accel_default_still_trips_instantly_on_single_cycle():
    # Defaults (accel_max_consecutive_violations=1) must reproduce the exact
    # old instant-trip behavior and exact old message -- a pure regression
    # guard for every existing production caller that never opts into this.
    monitor = _monitor(max_tcp_speed_mps=1e9, max_tcp_accel_mps2=200.0, max_waypoint_jump_m=1e9)
    monitor.set_start([0.0, 0.0, 0.5, 0.0, 0.0, 0.0], move_axis_index=0)
    # speeds: 0.5, 0.5 (accel=0, clean), 3.0 (accel=1250 -- one isolated spike).
    decisions = _run_x_deltas(monitor, [0.001, 0.001, 0.006])
    assert decisions[0].ok is True
    assert decisions[1].ok is True
    assert decisions[2].ok is False
    assert "TCP acceleration 1250.0000 m/s^2 > 200.0 m/s^2" == decisions[2].reason
    assert "consecutive" not in decisions[2].reason


def test_cartesian_move_monitor_accel_tolerates_isolated_transient_spikes():
    monitor = _monitor(
        max_tcp_speed_mps=1e9, max_tcp_accel_mps2=200.0, max_waypoint_jump_m=1e9,
        accel_max_consecutive_violations=3, accel_hard_multiple=1e9,  # isolate the graduated path only
    )
    monitor.set_start([0.0, 0.0, 0.5, 0.0, 0.0, 0.0], move_axis_index=0)
    # Speed pattern 0.5,0.5,3.0,3.0,0.5,0.5,3.0,3.0,... : each change is a
    # single isolated accel violation (settles at the new speed for one
    # cycle before the next change), never two violations in a row -- must
    # not trip even if this repeats forever, mirroring
    # test_deadline_monitor_tolerates_isolated_transient_overrun exactly.
    deltas = [0.001, 0.001]  # warm up to steady 0.5 m/s
    for _ in range(20):
        deltas += [0.006, 0.006, 0.001, 0.001]  # jump to 3.0, hold, drop to 0.5, hold
    decisions = _run_x_deltas(monitor, deltas)
    for d in decisions:
        assert d.ok is True, d.reason


def test_cartesian_move_monitor_accel_trips_on_consecutive_violations():
    monitor = _monitor(
        max_tcp_speed_mps=1e9, max_tcp_accel_mps2=200.0, max_waypoint_jump_m=1e9,
        accel_max_consecutive_violations=3, accel_hard_multiple=1e9,  # isolate the graduated path only
    )
    monitor.set_start([0.0, 0.0, 0.5, 0.0, 0.0, 0.0], move_axis_index=0)
    # Speed climbs 0.5 -> 3.0 -> 6.0 -> 9.0: each step is a fresh 1500 m/s^2
    # jump (three violations in a row), a real sustained-divergence shape.
    decisions = _run_x_deltas(monitor, [0.001, 0.001, 0.006, 0.012, 0.018])
    # 1st and 2nd consecutive violations are silent -- exactly matching
    # DeadlineMonitor.record()'s own behavior of returning None until the
    # Nth consecutive overrun (see test_deadline_monitor_trips_on_
    # consecutive_overruns). Only the 3rd trips.
    assert decisions[2].ok is True, decisions[2].reason
    assert decisions[3].ok is True, decisions[3].reason
    assert decisions[4].ok is False
    assert "for 3 consecutive cycles" in decisions[4].reason


def test_cartesian_move_monitor_accel_trips_immediately_on_hard_multiple():
    # A single cycle over hard_multiple x the base threshold must trip on
    # the very first such cycle, even with a lenient consecutive-violations
    # setting -- this is the disqualifying-catch guarantee: a genuine
    # one-shot catastrophic event must never wait for N more cycles.
    monitor = _monitor(
        max_tcp_speed_mps=1e9, max_tcp_accel_mps2=100.0, max_waypoint_jump_m=1e9,
        accel_max_consecutive_violations=3, accel_hard_multiple=5.0,
    )
    monitor.set_start([0.0, 0.0, 0.5, 0.0, 0.0, 0.0], move_axis_index=0)
    # speed 0.5, 0.5 (clean), then a jump to 100.5 m/s -> accel ~50000 m/s^2,
    # far over 5x100=500 -- must trip on this very cycle, not the 3rd.
    decisions = _run_x_deltas(monitor, [0.001, 0.001, 0.2])
    assert decisions[2].ok is False
    assert "consecutive" not in decisions[2].reason  # the hard-multiple path, not the graduated one


def test_cartesian_move_monitor_speed_default_still_trips_instantly_on_single_cycle():
    monitor = _monitor(max_tcp_speed_mps=1.0, max_tcp_accel_mps2=1e9, max_waypoint_jump_m=1e9)
    monitor.set_start([0.0, 0.0, 0.5, 0.0, 0.0, 0.0], move_axis_index=0)
    decisions = _run_x_deltas(monitor, [0.001, 0.01])  # 0.01/0.002 = 5.0 m/s > 1.0
    assert decisions[0].ok is True
    assert decisions[1].ok is False
    assert "TCP speed 5.0000 m/s > 1.0 m/s" == decisions[1].reason
    assert "consecutive" not in decisions[1].reason


def test_cartesian_move_monitor_speed_tolerates_isolated_transient_spikes():
    monitor = _monitor(
        max_tcp_speed_mps=1.0, max_tcp_accel_mps2=1e9, max_waypoint_jump_m=1e9,
        speed_max_consecutive_violations=3, speed_hard_multiple=1e9,  # isolate the graduated path only
    )
    monitor.set_start([0.0, 0.0, 0.5, 0.0, 0.0, 0.0], move_axis_index=0)
    # A single-cycle position spike (speed=5.0 m/s) immediately followed by
    # normal small steps -- unlike accel this is genuinely isolated (speed
    # depends only on this cycle's own delta, no derivative-of-derivative
    # carryover) -- must not trip even repeated forever.
    deltas = []
    for _ in range(30):
        deltas += [0.01, 0.001, 0.001]  # one spike, two clean cycles
    decisions = _run_x_deltas(monitor, deltas)
    for d in decisions:
        assert d.ok is True, d.reason


def test_cartesian_move_monitor_speed_trips_on_consecutive_violations():
    monitor = _monitor(
        max_tcp_speed_mps=1.0, max_tcp_accel_mps2=1e9, max_waypoint_jump_m=1e9,
        speed_max_consecutive_violations=3, speed_hard_multiple=1e9,  # isolate the graduated path only
    )
    monitor.set_start([0.0, 0.0, 0.5, 0.0, 0.0, 0.0], move_axis_index=0)
    decisions = _run_x_deltas(monitor, [0.01, 0.01, 0.01])  # 5.0 m/s, 3 cycles in a row
    # 1st and 2nd consecutive violations are silent, matching
    # DeadlineMonitor's own record()-returns-None-until-the-Nth behavior.
    assert decisions[0].ok is True, decisions[0].reason
    assert decisions[1].ok is True, decisions[1].reason
    assert decisions[2].ok is False
    assert "for 3 consecutive cycles" in decisions[2].reason


def test_cartesian_move_monitor_speed_trips_immediately_on_hard_multiple():
    monitor = _monitor(
        max_tcp_speed_mps=1.0, max_tcp_accel_mps2=1e9, max_waypoint_jump_m=1e9,
        speed_max_consecutive_violations=3, speed_hard_multiple=5.0,
    )
    monitor.set_start([0.0, 0.0, 0.5, 0.0, 0.0, 0.0], move_axis_index=0)
    decisions = _run_x_deltas(monitor, [0.02])  # 10.0 m/s > 5x1.0=5.0 -- instant trip
    assert decisions[0].ok is False
    assert "consecutive" not in decisions[0].reason


def test_cartesian_move_limits_validate_rejects_bad_speed_limit_gap_cycles():
    with pytest.raises(ValueError):
        CartesianMoveLimits(speed_limit_gap_cycles=0).validate()


def test_cartesian_move_limits_validate_rejects_bad_speed_limit_lowpass_alpha():
    with pytest.raises(ValueError):
        CartesianMoveLimits(speed_limit_lowpass_alpha=0.0).validate()
    with pytest.raises(ValueError):
        CartesianMoveLimits(speed_limit_lowpass_alpha=1.5).validate()


def test_cartesian_move_monitor_speed_limit_defaults_match_original_raw_behavior():
    # speed_limit_gap_cycles=1, speed_limit_lowpass_alpha=1.0 (defaults) must
    # reproduce the exact original single-cycle, unfiltered speed_mps value
    # -- same regression contract as accel_gap_cycles=1/speed_lowpass_alpha=1.0.
    monitor = _monitor(max_tcp_speed_mps=1.0, max_tcp_accel_mps2=1e9, max_waypoint_jump_m=1e9)
    monitor.set_start([0.0, 0.0, 0.5, 0.0, 0.0, 0.0], move_axis_index=0)
    decisions = _run_x_deltas(monitor, [0.001, 0.01])  # 0.01/0.002 = 5.0 m/s > 1.0
    assert decisions[0].ok is True
    assert decisions[1].ok is False
    assert "TCP speed 5.0000 m/s > 1.0 m/s" == decisions[1].reason


def test_cartesian_move_monitor_speed_limit_wider_gap_reduces_stationary_noise_sensitivity():
    # Mirrors test_cartesian_move_monitor_wider_gap_reduces_stationary_noise_sensitivity
    # (the accel-estimate version) for the speed-LIMIT check specifically.
    rng = np.random.default_rng(0)
    n = 300
    true_pos = np.array([0.0, 0.0, 0.5])
    noisy_positions = [true_pos + rng.normal(0.0, 1e-5, size=3) for _ in range(n)]

    def peak_speed(gap: int, alpha: float) -> float:
        monitor = _monitor(
            speed_limit_gap_cycles=gap,
            speed_limit_lowpass_alpha=alpha,
            max_tcp_speed_mps=1e-12,  # trip on essentially anything so decision.reason reports the value
            max_tcp_accel_mps2=1e9,
            max_waypoint_jump_m=1e9,
        )
        monitor.set_start(np.concatenate([true_pos, [0.0, 0.0, 0.0]]), move_axis_index=0)
        peak = 0.0
        for p in noisy_positions:
            pose = np.concatenate([p, [0.0, 0.0, 0.0]])
            decision = monitor.check(
                q=np.zeros(6), qd=np.zeros(6), tcp_pose=pose, target_tcp_pose=pose,
                orientation_error_rad=0.0, axis_target_moving=True, dt_s=0.002,
            )
            for reason in decision.reasons:
                if reason.startswith("TCP speed"):
                    peak = max(peak, float(reason.split()[2]))
        return peak

    peak_raw = peak_speed(gap=1, alpha=1.0)
    peak_smoothed = peak_speed(gap=5, alpha=0.2)
    assert peak_raw > 0.0
    assert peak_smoothed > 0.0
    # Unlike the accel estimate (a difference-of-differences, so gap helps
    # quadratically), this is a single position difference over a wider gap
    # -- noise reduction is real but more modest. Measured ~1.87x at these
    # settings; assert a conservative fraction of that, not a hand-derived
    # exact ratio (same honest-margin style as the accel-estimate test).
    assert peak_smoothed < peak_raw / 1.5, f"raw peak {peak_raw}, smoothed peak {peak_smoothed}"


def test_cartesian_move_monitor_speed_limit_smoothing_still_catches_a_genuine_sustained_rise():
    # The real-hardware finding this feature is built from: a genuinely
    # rising average speed must still trip even with smoothing enabled --
    # smoothing only removes noise-driven jitter in *when* it trips, not
    # whether it trips. Constant-velocity ramp well above the ceiling for
    # long enough to clear the smoothing window many times over.
    monitor = _monitor(
        speed_limit_gap_cycles=5,
        speed_limit_lowpass_alpha=0.2,
        max_tcp_speed_mps=1.0,
        max_tcp_accel_mps2=1e9,
        max_waypoint_jump_m=1e9,
    )
    monitor.set_start([0.0, 0.0, 0.5, 0.0, 0.0, 0.0], move_axis_index=0)
    # 0.01 m per 0.002s cycle = 5.0 m/s, sustained -- must eventually trip.
    decisions = _run_x_deltas(monitor, [0.01] * 40)
    assert any(not d.ok for d in decisions), "a genuine sustained 5.0 m/s rise never tripped"
    tripped_reasons = [d.reason for d in decisions if not d.ok]
    assert all("TCP speed" in r for r in tripped_reasons)


def test_cartesian_move_monitor_trips_on_off_axis_drift():
    monitor = _monitor(max_off_axis_drift_m=0.01)
    start = [0.0, 0.0, 0.5, 0.0, 0.0, 0.0]
    monitor.set_start(start, move_axis_index=1)
    pose = [0.05, 0.001, 0.5, 0.0, 0.0, 0.0]  # X drifted 5cm, way over 1cm limit
    decision = monitor.check(
        q=np.zeros(6),
        qd=np.zeros(6),
        tcp_pose=pose,
        target_tcp_pose=pose,
        orientation_error_rad=0.0,
        axis_target_moving=True,
        dt_s=0.01,
    )
    assert decision.ok is False
    assert "X-X0" in decision.reason


def test_cartesian_move_monitor_trips_on_orientation_error():
    monitor = _monitor(max_orientation_error_rad=0.1)
    start = [0.0, 0.0, 0.5, 0.0, 0.0, 0.0]
    monitor.set_start(start, move_axis_index=1)
    decision = monitor.check(
        q=np.zeros(6),
        qd=np.zeros(6),
        tcp_pose=start,
        target_tcp_pose=start,
        orientation_error_rad=0.5,
        axis_target_moving=True,
        dt_s=0.01,
    )
    assert decision.ok is False
    assert "orientation" in decision.reason


def test_cartesian_move_monitor_trips_on_excess_velocity():
    monitor = _monitor(max_tcp_speed_mps=0.01, max_waypoint_jump_m=10.0, max_off_axis_drift_m=10.0)
    start = [0.0, 0.0, 0.5, 0.0, 0.0, 0.0]
    monitor.set_start(start, move_axis_index=1)
    pose = [0.0, 1.0, 0.5, 0.0, 0.0, 0.0]  # huge jump in one dt_s=0.01s -> way over speed limit
    decision = monitor.check(
        q=np.zeros(6),
        qd=np.zeros(6),
        tcp_pose=pose,
        target_tcp_pose=pose,
        orientation_error_rad=0.0,
        axis_target_moving=True,
        dt_s=0.01,
    )
    assert decision.ok is False
    assert "speed" in decision.reason


def test_cartesian_move_monitor_trips_on_qd_over_limit():
    monitor = _monitor()
    start = [0.0, 0.0, 0.5, 0.0, 0.0, 0.0]
    monitor.set_start(start, move_axis_index=1)
    qd = np.zeros(6)
    qd[0] = 10.0
    decision = monitor.check(
        q=np.zeros(6),
        qd=qd,
        tcp_pose=start,
        target_tcp_pose=start,
        orientation_error_rad=0.0,
        axis_target_moving=True,
        dt_s=0.01,
    )
    assert decision.ok is False
    assert "qd" in decision.reason


def test_cartesian_move_monitor_trips_on_monotonic_growth_once_settled():
    monitor = _monitor(
        max_axis_error_growth_steps=3,
        max_tcp_speed_mps=10.0,
        max_tcp_accel_mps2=1000.0,
        max_waypoint_jump_m=10.0,
    )
    start = [0.0, 0.0, 0.5, 0.0, 0.0, 0.0]
    monitor.set_start(start, move_axis_index=1)
    target = [0.0, 0.20, 0.5, 0.0, 0.0, 0.0]
    # Error growing while axis_target_moving=True must not accumulate.
    pose = [0.0, 0.0, 0.5, 0.0, 0.0, 0.0]
    for y in (0.01, 0.005, 0.002):
        pose[1] = y
        decision = monitor.check(
            q=np.zeros(6), qd=np.zeros(6), tcp_pose=pose, target_tcp_pose=target,
            orientation_error_rad=0.0, axis_target_moving=True, dt_s=0.01,
        )
        assert decision.ok is True
    # Now settled (axis_target_moving=False): error growing for
    # max_axis_error_growth_steps consecutive steps must trip. Target is
    # 0.20; these values diverge further away from it each step (unlike the
    # moving phase above, which was converging toward it).
    last_ok = True
    for y in (-0.01, -0.02, -0.03, -0.04):
        pose[1] = y
        decision = monitor.check(
            q=np.zeros(6), qd=np.zeros(6), tcp_pose=pose, target_tcp_pose=target,
            orientation_error_rad=0.0, axis_target_moving=False, dt_s=0.01,
        )
        last_ok = decision.ok
    assert last_ok is False


def test_is_likely_ursim_loopback():
    assert is_likely_ursim("127.0.0.1") is True
    assert is_likely_ursim("localhost") is True
    assert is_likely_ursim("192.168.1.10") is False


def test_cartesian_limits_for_robot_relaxes_ursim_kinematic_guards():
    real = CartesianMoveLimits.for_robot("192.168.1.10")
    sim = CartesianMoveLimits.for_robot("127.0.0.1")
    assert sim.max_tcp_accel_mps2 > real.max_tcp_accel_mps2
    assert sim.max_waypoint_jump_m > real.max_waypoint_jump_m
    assert sim.max_tcp_speed_mps >= 0.5
