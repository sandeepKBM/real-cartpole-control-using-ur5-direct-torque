"""Tests for hardware/link.py using fake RTDE objects -- never opens a real
socket. Confirms the core liveness fix: read_state() raises on any problem
instead of returning stale/default data."""

from __future__ import annotations

import inspect
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from hardware.link import RTDELinkError, RTDEStateError, UR5eLink  # noqa: E402


class _FakeReceive:
    def __init__(self) -> None:
        self.q = [0.0, -1.57, 0.0, -1.57, 0.0, 0.0]
        self.qd = [0.0] * 6
        self.tcp_pose = [0.0, -0.234, 1.08, 0.0, 0.0, 0.0]
        self.fail_next = False

    def getActualQ(self):
        if self.fail_next:
            raise RuntimeError("simulated RTDE failure")
        return list(self.q)

    def getActualQd(self):
        return list(self.qd)

    def getActualTCPPose(self):
        return list(self.tcp_pose)

    def getTimestamp(self):
        return 42.0

    def getSafetyStatusBits(self):
        return 1

    def disconnect(self):
        pass


class _FakeControl:
    def __init__(self) -> None:
        self.servo_l_calls = []
        self.servo_stop_calls = 0
        self.stop_script_calls = 0

    def servoL(self, pose, speed, acceleration, time, lookahead_time, gain):
        self.servo_l_calls.append((list(pose), speed, acceleration, time, lookahead_time, gain))

    def servoStop(self):
        self.servo_stop_calls += 1

    def stopScript(self):
        self.stop_script_calls += 1

    def disconnect(self):
        pass


class _BadServoLControl(_FakeControl):
    """servoL with the wrong argument order -- link.py must reject this."""

    def servoL(self, pose, acceleration, speed, time, gain, lookahead_time):
        raise AssertionError("should never be called: signature should have been rejected at connect()")


def _make_link(receive=None, control=None) -> UR5eLink:
    receive = receive or _FakeReceive()
    control = control if control is not None else _FakeControl()
    return UR5eLink(
        "127.0.0.1",
        125.0,
        receive_factory=lambda ip, freq: receive,
        control_factory=lambda ip, freq: control,
    )


def test_connect_receive_only_does_not_touch_control():
    link = _make_link()
    link.connect(with_control=False)
    assert link.has_control is False


def test_connect_with_control_verifies_servol_signature():
    link = _make_link()
    link.connect(with_control=True)
    assert link.has_control is True


def test_connect_rejects_wrong_servol_signature():
    link = _make_link(control=_BadServoLControl())
    with pytest.raises(RTDELinkError):
        link.connect(with_control=True)


def test_read_state_returns_finite_state():
    link = _make_link()
    link.connect(with_control=False)
    state = link.read_state()
    assert state.q.shape == (6,)
    assert state.qd.shape == (6,)
    assert state.tcp_pose.shape == (6,)
    assert np.all(np.isfinite(state.q))


def test_read_state_raises_never_returns_stale_data():
    receive = _FakeReceive()
    link = _make_link(receive=receive)
    link.connect(with_control=False)
    link.read_state()  # first read succeeds, primes health
    receive.fail_next = True
    with pytest.raises(RTDEStateError):
        link.read_state()


def test_read_state_updates_health_on_success():
    link = _make_link()
    link.connect(with_control=False)
    assert link.is_alive() is False
    link.read_state()
    assert link.is_alive() is True


def test_read_state_rejects_nan():
    receive = _FakeReceive()
    receive.q[0] = float("nan")
    link = _make_link(receive=receive)
    link.connect(with_control=False)
    with pytest.raises(RTDEStateError):
        link.read_state()


def test_servo_l_forwards_to_control():
    control = _FakeControl()
    link = _make_link(control=control)
    link.connect(with_control=True)
    link.servo_l([0.1, 0.2, 0.3, 0.0, 0.0, 0.0], speed=0.25, acceleration=1.2, time_s=0.1, lookahead_time=0.1, gain=300.0)
    assert len(control.servo_l_calls) == 1
    pose, speed, accel, time_s, lookahead, gain = control.servo_l_calls[0]
    assert pose == pytest.approx([0.1, 0.2, 0.3, 0.0, 0.0, 0.0])
    assert speed == 0.25 and accel == 1.2 and gain == 300.0


def test_servo_l_before_control_connect_raises():
    link = _make_link()
    link.connect(with_control=False)
    with pytest.raises(RTDEStateError):
        link.servo_l([0, 0, 0, 0, 0, 0], speed=0.25, acceleration=1.2, time_s=0.1, lookahead_time=0.1, gain=300.0)


def test_safe_stop_calls_servo_stop_and_stop_script_then_disconnects():
    control = _FakeControl()
    link = _make_link(control=control)
    link.connect(with_control=True)
    link.safe_stop("test reason")
    assert control.servo_stop_calls == 1
    assert control.stop_script_calls == 1
    assert link.has_control is False


def test_safe_stop_is_best_effort_if_one_method_raises():
    class _PartlyBrokenControl(_FakeControl):
        def servoStop(self):
            raise RuntimeError("servoStop unavailable")

    control = _PartlyBrokenControl()
    link = _make_link(control=control)
    link.connect(with_control=True)
    link.safe_stop("test reason")  # must not raise
    assert control.stop_script_calls == 1
