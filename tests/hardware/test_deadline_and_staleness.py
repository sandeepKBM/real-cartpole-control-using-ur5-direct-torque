"""Tests for the two safety fixes added to the hardware lane:

* ``DeadlineMonitor`` -- enforces ``UR5eSafetyLimits.max_deadline_ms`` (was
  validated but never checked); a sustained/severe control-loop overrun now
  aborts instead of silently running late.
* ``StaleStateMonitor`` -- detects a frozen-but-non-raising RTDE stream by
  comparing ``robot_timestamp_s`` across cycles.

Unit tests exercise the monitor decision logic deterministically (no timing);
integration tests drive each of the four control loops against fake RTDE
objects and assert the loop aborts with the right reason, following the
fake-object patterns already used in test_motion.py / test_position_transport.py
/ test_direct_torque_transport_timing.py.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from hardware.direct_torque_link import UR5eDirectTorqueLink  # noqa: E402
from hardware.direct_torque_transport import run_x_transport_direct_torque  # noqa: E402
from hardware.link import UR5eLink, UR5eState  # noqa: E402
from hardware.motion import move_cartesian_bounded  # noqa: E402
from hardware.poses import HEIGHT_ALPHA_0_5_Q  # noqa: E402
from hardware.position_transport import run_x_transport_position  # noqa: E402
from hardware.urscript_gen import DEFAULT_CONFIG  # noqa: E402
from hardware.urscript_transport import (  # noqa: E402
    _read_state_from_receive,
    run_urscript_x_transport,
)
from hardware.safety import (  # noqa: E402
    CartesianMoveLimits,
    CartesianMoveMonitor,
    DeadlineMonitor,
    EStopLatch,
    StaleStateMonitor,
    UR5eSafetyLimits,
)

CONFIG = REPO_ROOT / "config" / "ur5e_mujoco_torque_osc_tuned.yaml"


# --------------------------------------------------------------------------- #
# Unit tests: DeadlineMonitor
# --------------------------------------------------------------------------- #
def _ns(ms: float) -> int:
    return int(ms * 1e6)


def test_deadline_monitor_ignores_clean_cycles():
    mon = DeadlineMonitor(max_deadline_ms=3.0)
    for _ in range(1000):
        assert mon.record(_ns(0.0)) is None
        assert mon.record(_ns(2.9)) is None  # under the 3.0 ms ceiling


def test_deadline_monitor_tolerates_isolated_transient_overrun():
    mon = DeadlineMonitor(max_deadline_ms=3.0, max_consecutive_overruns=3)
    # One overrun, then a clean cycle resets the streak -- must not trip even
    # if this repeats forever.
    for _ in range(50):
        assert mon.record(_ns(4.0)) is None  # over 3 ms, but isolated
        assert mon.record(_ns(0.0)) is None  # clean -> resets consecutive count


def test_deadline_monitor_trips_on_consecutive_overruns():
    mon = DeadlineMonitor(max_deadline_ms=3.0, max_consecutive_overruns=3)
    assert mon.record(_ns(4.0)) is None
    assert mon.record(_ns(4.0)) is None
    reason = mon.record(_ns(4.0))  # third in a row
    assert reason is not None
    assert "deadline_overrun" in reason
    assert "consecutive" in reason


def test_deadline_monitor_trips_immediately_on_single_hard_overrun():
    mon = DeadlineMonitor(max_deadline_ms=3.0, hard_overrun_multiple=5.0)
    # 20 ms > 5 x 3 ms = 15 ms -> a stall, trips on the very first cycle.
    reason = mon.record(_ns(20.0))
    assert reason is not None
    assert "single cycle" in reason


def test_deadline_monitor_rejects_bad_config():
    with pytest.raises(ValueError):
        DeadlineMonitor(max_deadline_ms=0.0)
    with pytest.raises(ValueError):
        DeadlineMonitor(max_deadline_ms=3.0, max_consecutive_overruns=0)
    with pytest.raises(ValueError):
        DeadlineMonitor(max_deadline_ms=3.0, hard_overrun_multiple=0.5)


# --------------------------------------------------------------------------- #
# Fix 2a: period-relative deadline cap for the 500 Hz direct_torque loop
# (docs/status/deadline_monitor_period_relative_fix_2026-07-29.md). The flat
# 3.0 ms max_deadline_ms default tolerates up to ~250% of a 2 ms period
# before an overrun is even counted (see test_deadline_monitor_ignores_
# clean_cycles above) -- too loose for that loop's own budget, and a real
# reported incident (4/5 cycles late, overruns up to ~2 ms, total cycle time
# up to ~2x the nominal 2 ms period) slipped under it undetected. These tests
# cover the fix: a tightened DeadlineMonitor(1.0) now catches that exact
# shape, and the UR5eSafetyLimits.max_deadline_fraction_of_period field feeds
# the min() formula used at the direct_torque_transport.py call site.
# --------------------------------------------------------------------------- #
def test_deadline_monitor_trips_on_reported_incident_shape():
    # Post-fix 500 Hz/2 ms-loop effective cap (min(3.0, 0.5 * 2.0) = 1.0 ms).
    mon = DeadlineMonitor(max_deadline_ms=1.0, max_consecutive_overruns=3)
    # The reported incident: 4-of-5 cycles late, overruns up to ~2 ms (total
    # cycle time up to ~2x the nominal 2 ms period), one clean cycle out of
    # every five.
    pattern_ms = [2.0, 2.0, 2.0, 2.0, 0.0]
    reason = None
    cycles_run = 0
    for _ in range(20):  # far more than needed if the fix works
        for overrun_ms in pattern_ms:
            cycles_run += 1
            reason = mon.record(_ns(overrun_ms))
            if reason is not None:
                break
        if reason is not None:
            break
    assert reason is not None, "reported incident shape must now trip the tightened monitor"
    assert "deadline_overrun" in reason
    # 4 late cycles in a row appear before the single clean one in each
    # 5-cycle block, so with max_consecutive_overruns=3 this must trip within
    # the very first block ("a few cycles"), not require many repeats.
    assert cycles_run <= 5


def test_max_deadline_fraction_of_period_config_wiring():
    # Mirrors the exact min() formula used at the direct_torque_transport.py
    # call site: min(max_deadline_ms, max_deadline_fraction_of_period * dt_s * 1000.0).
    limits = UR5eSafetyLimits(max_deadline_ms=3.0, max_deadline_fraction_of_period=0.5)
    limits.validate()

    # 500 Hz / 2 ms period -> 0.5 * 2.0 = 1.0 ms < 3.0 ms default -> tightened.
    dt_s_500hz = 0.002
    effective_500hz = min(limits.max_deadline_ms, limits.max_deadline_fraction_of_period * dt_s_500hz * 1000.0)
    assert effective_500hz == pytest.approx(1.0)

    # 125 Hz / 8 ms period -> 0.5 * 8.0 = 4.0 ms >= 3.0 ms default -> unchanged.
    dt_s_125hz = 0.008
    effective_125hz = min(limits.max_deadline_ms, limits.max_deadline_fraction_of_period * dt_s_125hz * 1000.0)
    assert effective_125hz == pytest.approx(3.0)


# --------------------------------------------------------------------------- #
# Unit tests: StaleStateMonitor
# --------------------------------------------------------------------------- #
def test_stale_monitor_ok_when_timestamp_advances():
    mon = StaleStateMonitor()
    host = 0
    ts = 100.0
    for _ in range(1000):
        host += 1_000_000  # host clock always advances
        ts += 0.002  # robot clock advances every read
        assert mon.record(ts, host) is None


def test_stale_monitor_tolerates_a_few_repeated_timestamps():
    # Poll-faster-than-publish can repeat a timestamp a handful of times; below
    # max_frozen_cycles this must not trip.
    mon = StaleStateMonitor(max_frozen_cycles=5)
    host = 0
    for i in range(4):  # 4 frozen reads < 5
        host += 1_000_000
        assert mon.record(50.0, host) is None
    # A fresh advance clears the frozen streak.
    host += 1_000_000
    assert mon.record(50.5, host) is None
    for i in range(4):
        host += 1_000_000
        assert mon.record(50.5, host) is None


def test_stale_monitor_trips_when_timestamp_frozen():
    mon = StaleStateMonitor(max_frozen_cycles=5)
    host = 0
    reason = None
    for _ in range(10):
        host += 1_000_000
        reason = mon.record(50.0, host)
        if reason is not None:
            break
    assert reason is not None
    assert "stale_state" in reason
    assert "frozen" in reason


def test_stale_monitor_none_timestamp_never_trips():
    # getTimestamp unavailable -> can't verify, never stale.
    mon = StaleStateMonitor(max_frozen_cycles=2)
    host = 0
    for _ in range(100):
        host += 1_000_000
        assert mon.record(None, host) is None


def test_stale_monitor_does_not_count_when_host_frozen_too():
    # Degenerate: neither clock advances -> not the frozen-stream signature.
    mon = StaleStateMonitor(max_frozen_cycles=2)
    assert mon.record(50.0, 1000) is None
    assert mon.record(50.0, 1000) is None
    assert mon.record(50.0, 1000) is None


# --------------------------------------------------------------------------- #
# Integration: motion.move_cartesian_bounded
# --------------------------------------------------------------------------- #
class _MotionReceive:
    """Fake receive for the servoL motion loop. ``frozen_ts`` models a stalled
    stream (constant robot clock); ``read_sleep_s`` forces a deadline overrun."""

    def __init__(self, tcp_pose, *, frozen_ts: bool = False, read_sleep_s: float = 0.0) -> None:
        self._pose = list(tcp_pose)
        self._frozen_ts = frozen_ts
        self._read_sleep_s = read_sleep_s
        self.q = [0.0] * 6
        self.qd = [0.0] * 6
        self._ts = 0.0

    def getActualQ(self):
        if self._read_sleep_s:
            time.sleep(self._read_sleep_s)
        return list(self.q)

    def getActualQd(self):
        return list(self.qd)

    def getActualTCPPose(self):
        return list(self._pose)

    def getTimestamp(self):
        if not self._frozen_ts:
            self._ts += 0.002
        return self._ts

    def disconnect(self):
        pass


class _MotionControl:
    def __init__(self) -> None:
        self.servo_stop_calls = 0
        self.stop_script_calls = 0

    def servoL(self, pose, speed, acceleration, time, lookahead_time, gain):
        pass

    def servoStop(self):
        self.servo_stop_calls += 1

    def stopScript(self):
        self.stop_script_calls += 1

    def disconnect(self):
        pass


def _motion_link(receive) -> UR5eLink:
    return UR5eLink(
        "127.0.0.1", 1000.0,
        receive_factory=lambda ip, freq: receive,
        control_factory=lambda ip, freq: _MotionControl(),
    )


def test_motion_aborts_on_frozen_timestamp():
    receive = _MotionReceive([0.0, 0.0, 0.5, 0.0, 0.0, 0.0], frozen_ts=True)
    link = _motion_link(receive)
    link.connect(with_control=True)
    monitor = CartesianMoveMonitor(
        CartesianMoveLimits(max_tcp_speed_mps=1e3, max_tcp_accel_mps2=1e6, max_waypoint_jump_m=1e3)
    )
    estop = EStopLatch()
    result = move_cartesian_bounded(
        link, monitor, estop, axis_index=1, distance_m=0.0, motion_opt_in=True,
        duration_s=0.05, rate_hz=1000.0,
    )
    assert result.ok is False
    assert result.stopped_early is True
    assert "stale_state" in result.reason
    assert estop.tripped is True


def test_motion_aborts_on_deadline_overrun():
    # A single ~25 ms read stall at 1 kHz (1 ms period) overruns the 3 ms
    # deadline by > 5x -> hard single-cycle trip.
    receive = _MotionReceive([0.0, 0.0, 0.5, 0.0, 0.0, 0.0], read_sleep_s=0.025)
    link = _motion_link(receive)
    link.connect(with_control=True)
    monitor = CartesianMoveMonitor(
        CartesianMoveLimits(max_tcp_speed_mps=1e3, max_tcp_accel_mps2=1e6, max_waypoint_jump_m=1e3)
    )
    estop = EStopLatch()
    result = move_cartesian_bounded(
        link, monitor, estop, axis_index=1, distance_m=0.0, motion_opt_in=True,
        duration_s=0.02, rate_hz=1000.0,
    )
    assert result.ok is False
    assert "deadline_overrun" in result.reason
    assert estop.tripped is True


# --------------------------------------------------------------------------- #
# Integration: position_transport.run_x_transport_position
# --------------------------------------------------------------------------- #
class _PosReceive:
    def __init__(self, *, frozen_ts: bool = False, read_sleep_s: float = 0.0) -> None:
        self._tcp_x = 0.4
        self._frozen_ts = frozen_ts
        self._read_sleep_s = read_sleep_s
        self.q = [0.0, -0.835, -1.2, -0.985, 0.0, 0.0]
        self.qd = [0.0] * 6
        self._ts = 0.0

    def getActualQ(self):
        if self._read_sleep_s:
            time.sleep(self._read_sleep_s)
        return list(self.q)

    def getActualQd(self):
        return list(self.qd)

    def getActualTCPPose(self):
        return [self._tcp_x, -0.2, 0.3, 0.0, 3.14, 0.0]

    def getTimestamp(self):
        if not self._frozen_ts:
            self._ts += 0.002
        return self._ts

    def disconnect(self):
        pass


class _PosControl:
    def __init__(self, receive: _PosReceive) -> None:
        self._receive = receive

    def servoL(self, pose, speed, acceleration, time, lookahead_time, gain):
        # Track commanded X slowly so the kinematic guards stay quiet.
        target_x = float(pose[0])
        delta = target_x - self._receive._tcp_x
        step = min(abs(delta), 0.002)
        if step > 0.0:
            self._receive._tcp_x += step if delta > 0.0 else -step

    def servoStop(self):
        pass

    def stopScript(self):
        pass

    def disconnect(self):
        pass


def _pos_link(receive) -> UR5eLink:
    return UR5eLink(
        "127.0.0.1", 500.0,
        receive_factory=lambda ip, freq: receive,
        control_factory=lambda ip, freq: _PosControl(receive),
    )


@pytest.mark.hardware
def test_position_transport_aborts_on_frozen_timestamp(tmp_path: Path):
    receive = _PosReceive(frozen_ts=True)
    link = _pos_link(receive)
    result = run_x_transport_position(
        link, config_path=CONFIG, target_x_delta_m=0.01,
        move_duration_s=0.1, duration_s=0.2, output_dir=tmp_path,
        motion_opt_in=True, rate_hz=500.0, shadow_osc=False,
    )
    assert result.ok is False
    assert "stale_state" in result.summary["termination_reason"]


@pytest.mark.hardware
def test_position_transport_aborts_on_deadline_overrun(tmp_path: Path):
    receive = _PosReceive(read_sleep_s=0.025)
    link = _pos_link(receive)
    result = run_x_transport_position(
        link, config_path=CONFIG, target_x_delta_m=0.01,
        move_duration_s=0.1, duration_s=0.2, output_dir=tmp_path,
        motion_opt_in=True, rate_hz=500.0, shadow_osc=False,
    )
    assert result.ok is False
    assert "deadline_overrun" in result.summary["termination_reason"]


# --------------------------------------------------------------------------- #
# Integration: urscript_transport (supervisor loop + timestamp population)
# --------------------------------------------------------------------------- #
class _UrReceive:
    def __init__(self, *, frozen_ts: bool = False, read_sleep_s: float = 0.0,
                 has_timestamp: bool = True) -> None:
        self._frozen_ts = frozen_ts
        self._read_sleep_s = read_sleep_s
        self._has_timestamp = has_timestamp
        self._pose = [0.4, -0.2, 0.3, 0.0, 3.14, 0.0]
        self._ts = 0.0

    def getActualQ(self):
        if self._read_sleep_s:
            time.sleep(self._read_sleep_s)
        return [0.0, -0.835, -1.2, -0.985, 0.0, 0.0]

    def getActualQd(self):
        return [0.0] * 6

    def getActualTCPPose(self):
        return list(self._pose)

    def getTimestamp(self):
        if not self._has_timestamp:
            raise AttributeError("no timestamp")
        if not self._frozen_ts:
            self._ts += 0.002
        return self._ts

    def getSafetyStatusBits(self):
        return 1  # IS_NORMAL_MODE

    def disconnect(self):
        pass


class _UrControl:
    def __init__(self) -> None:
        self._regs: dict[int, int] = {}
        self._stop_requested = False
        self.script_sent = False

    def setInputIntRegister(self, reg, value):
        self._regs[int(reg)] = int(value)
        if int(value) == 1:
            self._stop_requested = True

    def sendCustomScript(self, text):
        # Block (as the real call does) until the Python supervisor requests a
        # stop, so a fault the supervisor detects actually ends the run.
        self.script_sent = True
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline:
            if self._stop_requested:
                return True
            time.sleep(0.005)
        return True

    def disconnect(self):
        pass


def _patch_rtde(monkeypatch, control, receive):
    monkeypatch.setattr(
        "hardware.urscript_transport._load_rtde_classes",
        lambda: (lambda ip, freq: control, lambda ip, freq: receive),
    )


def test_read_state_from_receive_now_populates_robot_timestamp():
    # Regression for the fix: _read_state_from_receive used to hardcode
    # robot_timestamp_s=None, defeating StaleStateMonitor.
    receive = _UrReceive()
    state = _read_state_from_receive(receive)
    assert state.robot_timestamp_s is not None
    # None-safe when the getter is absent/raising.
    state2 = _read_state_from_receive(_UrReceive(has_timestamp=False))
    assert state2.robot_timestamp_s is None


@pytest.mark.hardware
def test_urscript_supervisor_aborts_on_frozen_timestamp(tmp_path: Path, monkeypatch):
    control = _UrControl()
    receive = _UrReceive(frozen_ts=True)
    _patch_rtde(monkeypatch, control, receive)
    result = run_urscript_x_transport(
        robot_ip="127.0.0.1", config_path=DEFAULT_CONFIG,
        target_x_delta_m=0.01, move_duration_s=0.1, duration_s=0.4,
        output_dir=tmp_path, motion_opt_in=True, skip_joint_move=True,
        monitor_hz=200.0,
    )
    assert result.ok is False
    assert "stale_state" in result.summary["termination_reason"]
    assert control._stop_requested is True


@pytest.mark.hardware
def test_urscript_supervisor_aborts_on_deadline_overrun(tmp_path: Path, monkeypatch):
    control = _UrControl()
    receive = _UrReceive(read_sleep_s=0.03)  # 30 ms read stall each cycle
    _patch_rtde(monkeypatch, control, receive)
    result = run_urscript_x_transport(
        robot_ip="127.0.0.1", config_path=DEFAULT_CONFIG,
        target_x_delta_m=0.01, move_duration_s=0.1, duration_s=0.6,
        output_dir=tmp_path, motion_opt_in=True, skip_joint_move=True,
        monitor_hz=200.0,
    )
    assert result.ok is False
    assert "deadline_overrun" in result.summary["termination_reason"]
    assert control._stop_requested is True


# --------------------------------------------------------------------------- #
# Integration: direct_torque_transport.run_x_transport_direct_torque
# --------------------------------------------------------------------------- #
class _MockDTLink:
    """Mock UR5eDirectTorqueLink (dynamics_source='rtde', so no pinocchio)."""

    def __init__(self, *, frozen_ts: bool = False, ts_start: float | None = None,
                 slow_first_torque_s: float = 0.0) -> None:
        self._tcp_x = 0.4
        self._frozen_ts = frozen_ts
        self._ts = ts_start
        self._slow = slow_first_torque_s
        self._torque_calls = 0
        # A real link carries a UR5eSafetyLimits under .limits; the loop reads
        # max_deadline_ms from it -- provide it so we don't just hit the
        # fallback default (proves the wiring uses the link's own limit).
        from hardware.safety import UR5eSafetyLimits

        self.limits = UR5eSafetyLimits()

    def connect(self) -> None:
        pass

    def read_state(self) -> UR5eState:
        ts = None
        if self._ts is not None:
            ts = self._ts
            if not self._frozen_ts:
                self._ts += 0.002
        return UR5eState(
            q=HEIGHT_ALPHA_0_5_Q.copy(),
            qd=np.zeros(6),
            tcp_pose=np.array([self._tcp_x, -0.2, 0.3, 0.0, 3.14, 0.0]),
            host_stamp_ns=time.monotonic_ns(),
            robot_timestamp_s=ts,
            safety_status=None,
        )

    def get_jacobian(self) -> np.ndarray:
        return np.eye(6)

    def get_mass_matrix(self) -> np.ndarray:
        return np.eye(6)

    def direct_torque(self, tau_nm: np.ndarray, *, friction_comp: bool = True) -> None:
        self._torque_calls += 1
        if self._slow and self._torque_calls == 1:
            time.sleep(self._slow)
        self._tcp_x += float(tau_nm[0]) * 1e-6

    @staticmethod
    def compose_robot_state(link_state, *, jacobian, mass_matrix, time_s, target_x, target_x_vel):
        return UR5eDirectTorqueLink.compose_robot_state(
            link_state, jacobian=jacobian, mass_matrix=mass_matrix,
            time_s=time_s, target_x=target_x, target_x_vel=target_x_vel,
        )

    def build_robot_state(self, link_state, *, time_s, target_x, target_x_vel):
        return self.compose_robot_state(
            link_state, jacobian=self.get_jacobian(), mass_matrix=self.get_mass_matrix(),
            time_s=time_s, target_x=target_x, target_x_vel=target_x_vel,
        )

    def safe_stop(self, reason: str) -> None:
        pass


@pytest.mark.hardware
def test_direct_torque_aborts_on_frozen_timestamp(tmp_path: Path):
    link = _MockDTLink(frozen_ts=True, ts_start=5.0)
    result = run_x_transport_direct_torque(
        link,  # type: ignore[arg-type]
        config_path=CONFIG, target_x_delta_m=0.01,
        move_duration_s=0.1, duration_s=0.2, output_dir=tmp_path,
        motion_opt_in=True, record_latency=False, dynamics_source="rtde",
    )
    assert result.ok is False
    assert "stale_state" in result.summary["termination_reason"]


@pytest.mark.hardware
def test_direct_torque_aborts_on_deadline_overrun(tmp_path: Path):
    # robot_timestamp_s=None -> staleness can't trip; a 30 ms stall at 500 Hz
    # (2 ms period) makes the next cycle start > 15 ms late -> hard deadline trip.
    link = _MockDTLink(ts_start=None, slow_first_torque_s=0.03)
    result = run_x_transport_direct_torque(
        link,  # type: ignore[arg-type]
        config_path=CONFIG, target_x_delta_m=0.01,
        move_duration_s=0.1, duration_s=0.2, output_dir=tmp_path,
        motion_opt_in=True, record_latency=False, dynamics_source="rtde",
    )
    assert result.ok is False
    assert "deadline_overrun" in result.summary["termination_reason"]
