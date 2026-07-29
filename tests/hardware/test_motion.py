"""Tests for hardware/motion.py -- waypoint math (pure) plus the full servoL
streaming loop against fake RTDE objects (never opens a real socket)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from hardware.link import UR5eLink  # noqa: E402
from hardware.motion import move_cartesian_bounded, peak_acceleration_mps2, peak_velocity_mps, plan_waypoints  # noqa: E402
from hardware.safety import CartesianMoveLimits, CartesianMoveMonitor, EStopLatch  # noqa: E402


def test_peak_velocity_matches_quintic_formula():
    # v_peak = 1.875 * distance / duration
    assert peak_velocity_mps(0.15, 6.0) == pytest.approx(1.875 * 0.15 / 6.0)


def test_peak_acceleration_matches_quintic_formula():
    # max |s''(tau)| = 10*sqrt(3)/3 for the quintic min-jerk profile.
    assert peak_acceleration_mps2(0.15, 6.0) == pytest.approx((10.0 * np.sqrt(3.0) / 3.0) * 0.15 / 36.0)


def test_plan_waypoints_rejects_bad_axis():
    with pytest.raises(ValueError):
        plan_waypoints([0, 0, 0, 0, 0, 0], axis_index=3, distance_m=0.1, duration_s=1.0, rate_hz=10.0)


def test_plan_waypoints_count_matches_duration_and_rate():
    waypoints = plan_waypoints([0, 0, 0, 0, 0, 0], axis_index=1, distance_m=0.1, duration_s=2.0, rate_hz=10.0)
    assert len(waypoints) == 20


def test_plan_waypoints_final_waypoint_reaches_target_distance():
    start = [1.0, 2.0, 3.0, 0.0, 0.0, 0.0]
    waypoints = plan_waypoints(start, axis_index=0, distance_m=0.15, duration_s=1.0, rate_hz=100.0)
    final = waypoints[-1]
    assert final[0] == pytest.approx(1.15, abs=1e-6)
    # Off-axis and orientation components never change.
    assert final[1] == pytest.approx(2.0)
    assert final[2] == pytest.approx(3.0)
    assert np.allclose(final[3:], [0.0, 0.0, 0.0])


def test_plan_waypoints_monotonic_along_move_axis_for_positive_distance():
    waypoints = plan_waypoints([0, 0, 0, 0, 0, 0], axis_index=1, distance_m=0.1, duration_s=1.0, rate_hz=50.0)
    ys = [w[1] for w in waypoints]
    assert all(b >= a - 1e-12 for a, b in zip(ys, ys[1:]))


def test_plan_waypoints_negative_distance_moves_the_other_way():
    waypoints = plan_waypoints([0, 0, 0, 0, 0, 0], axis_index=1, distance_m=-0.1, duration_s=1.0, rate_hz=50.0)
    assert waypoints[-1][1] == pytest.approx(-0.1, abs=1e-6)


class _FakeReceive:
    def __init__(self, tcp_pose_sequence) -> None:
        self._sequence = list(tcp_pose_sequence)
        self._i = -1
        self.q = [0.0] * 6
        self.qd = [0.0] * 6
        self._ts = 0.0

    def getActualQ(self):
        return list(self.q)

    def getActualQd(self):
        return list(self.qd)

    def getActualTCPPose(self):
        self._i = min(self._i + 1, len(self._sequence) - 1)
        return list(self._sequence[self._i])

    def getTimestamp(self):
        # A healthy stream advances its clock every read (500 Hz robot).
        self._ts += 0.002
        return self._ts

    def disconnect(self):
        pass


class _FakeControl:
    def __init__(self) -> None:
        self.servo_l_calls = 0
        self.servo_stop_calls = 0
        self.stop_script_calls = 0

    def servoL(self, pose, speed, acceleration, time, lookahead_time, gain):
        self.servo_l_calls += 1

    def servoStop(self):
        self.servo_stop_calls += 1

    def stopScript(self):
        self.stop_script_calls += 1

    def disconnect(self):
        pass


def _link_with_sequence(tcp_pose_sequence) -> tuple[UR5eLink, _FakeControl]:
    receive = _FakeReceive(tcp_pose_sequence)
    control = _FakeControl()
    link = UR5eLink(
        "127.0.0.1", 500.0,
        receive_factory=lambda ip, freq: receive,
        control_factory=lambda ip, freq: control,
    )
    return link, control


def test_move_cartesian_bounded_blocked_without_motion_opt_in():
    link, control = _link_with_sequence([[0, 0, 0.5, 0, 0, 0]] * 10)
    link.connect(with_control=True)
    monitor = CartesianMoveMonitor(CartesianMoveLimits())
    estop = EStopLatch()
    result = move_cartesian_bounded(
        link, monitor, estop, axis_index=1, distance_m=0.02, motion_opt_in=False,
        duration_s=0.02, rate_hz=100.0,
    )
    assert result.ok is False
    assert control.servo_l_calls == 0
    assert estop.tripped is False  # blocked-by-opt-in is not a fault, just declined


def test_move_cartesian_bounded_completes_a_clean_small_move():
    n_steps = int(round(0.02 * 100.0))
    start = [0.0, 0.0, 0.5, 0.0, 0.0, 0.0]
    # A clean sequence: first read is the "start" read, then each waypoint
    # read matches the commanded target closely enough to pass all checks.
    sequence = [start]
    for i in range(1, n_steps + 1):
        s = min(1.0, i / n_steps)
        pose = list(start)
        pose[1] = 0.02 * (10 * s**3 - 15 * s**4 + 6 * s**5)
        sequence.append(pose)
    sequence.append(sequence[-1])  # final read after servo_stop

    link, control = _link_with_sequence(sequence)
    link.connect(with_control=True)
    monitor = CartesianMoveMonitor(CartesianMoveLimits(max_tcp_speed_mps=1.0, max_tcp_accel_mps2=1000.0, max_waypoint_jump_m=1.0))
    estop = EStopLatch()
    result = move_cartesian_bounded(
        link, monitor, estop, axis_index=1, distance_m=0.02, motion_opt_in=True,
        duration_s=0.02, rate_hz=100.0,
    )
    assert result.ok is True, result.reason
    assert result.stopped_early is False
    assert result.waypoints_sent == n_steps
    assert control.servo_stop_calls == 1
    assert estop.tripped is False


def test_move_cartesian_bounded_writes_trace_and_summary(tmp_path):
    n_steps = int(round(0.02 * 100.0))
    start = [0.0, 0.0, 0.5, 0.0, 0.0, 0.0]
    sequence = [start]
    for i in range(1, n_steps + 1):
        s = min(1.0, i / n_steps)
        pose = list(start)
        pose[1] = 0.02 * (10 * s**3 - 15 * s**4 + 6 * s**5)
        sequence.append(pose)
    sequence.append(sequence[-1])

    trace_path = tmp_path / "trace.jsonl"
    summary_path = tmp_path / "summary.json"
    link, _ = _link_with_sequence(sequence)
    link.connect(with_control=True)
    monitor = CartesianMoveMonitor(
        CartesianMoveLimits(max_tcp_speed_mps=1.0, max_tcp_accel_mps2=1000.0, max_waypoint_jump_m=1.0)
    )
    estop = EStopLatch()
    result = move_cartesian_bounded(
        link,
        monitor,
        estop,
        axis_index=1,
        distance_m=0.02,
        motion_opt_in=True,
        duration_s=0.02,
        rate_hz=100.0,
        trace_path=trace_path,
        summary_path=summary_path,
    )

    assert result.ok is True
    assert result.trace_path == trace_path
    assert result.summary_path == summary_path
    rows = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == n_steps
    assert {"t_s", "q", "qd", "tcp_pose", "target_tcp_pose", "axis_error_m"} <= set(rows[0])
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["success"] is True
    assert summary["trace_path"] == str(trace_path)


def test_move_cartesian_bounded_stops_early_on_off_axis_drift_and_trips_estop():
    n_steps = 5
    start = [0.0, 0.0, 0.5, 0.0, 0.0, 0.0]
    sequence = [start]
    for i in range(1, n_steps + 1):
        pose = list(start)
        pose[1] = 0.001 * i
        if i == 3:
            pose[0] = 0.5  # sudden huge off-axis (X) drift -- must trip the monitor
        sequence.append(pose)

    link, control = _link_with_sequence(sequence)
    link.connect(with_control=True)
    monitor = CartesianMoveMonitor(CartesianMoveLimits(max_off_axis_drift_m=0.01, max_tcp_speed_mps=100.0, max_waypoint_jump_m=100.0))
    estop = EStopLatch()
    result = move_cartesian_bounded(
        link, monitor, estop, axis_index=1, distance_m=0.02, motion_opt_in=True,
        duration_s=0.04, rate_hz=100.0,
    )
    assert result.ok is False
    assert result.stopped_early is True
    assert result.waypoints_sent < n_steps
    assert estop.tripped is True
    assert control.servo_stop_calls == 1  # safe_stop's servoStop call
    assert control.stop_script_calls == 1


def test_move_cartesian_bounded_respects_already_tripped_estop():
    link, control = _link_with_sequence([[0, 0, 0.5, 0, 0, 0]] * 10)
    link.connect(with_control=True)
    monitor = CartesianMoveMonitor(CartesianMoveLimits())
    estop = EStopLatch()
    estop.trip("earlier fault")
    with pytest.raises(Exception):
        move_cartesian_bounded(
            link, monitor, estop, axis_index=1, distance_m=0.02, motion_opt_in=True,
            duration_s=0.02, rate_hz=100.0,
        )
    assert control.servo_l_calls == 0
