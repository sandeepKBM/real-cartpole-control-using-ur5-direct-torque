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
