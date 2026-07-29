"""Regression coverage for a real bug found live on hardware 2026-07-30:
tools/ur5e_connect.py --watch only reacted to read_state() *raising* --
a frozen-but-successfully-returning RTDE stream (real, documented ur_rtde
behavior) was never detected, so the loop silently printed the same stale
state forever instead of reconnecting/failing loudly as documented.

Fake link only -- never opens a real socket. Every scenario is built to
terminate deterministically (bounded scripted data + a reconnect fake that
eventually stops succeeding) so a real bug in `run_watch` shows up as an
assertion failure, not a hung test process.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from hardware.link import RTDEStateError  # noqa: E402
from hardware.safety import ConnectionHealth, EStopLatch  # noqa: E402
import tools.ur5e_connect as ur5e_connect  # noqa: E402


@dataclass
class _FakeState:
    q: np.ndarray
    qd: np.ndarray
    tcp_pose: np.ndarray
    robot_timestamp_s: float | None
    safety_status: int | None = 2049


def _state(ts: float) -> _FakeState:
    return _FakeState(q=np.zeros(6), qd=np.zeros(6), tcp_pose=np.zeros(6), robot_timestamp_s=ts)


class _FakeLink:
    """Plays back a fixed, finite list of robot_timestamp_s values, then
    always raises RTDEStateError -- guarantees the test terminates."""

    def __init__(self, timestamps: list[float]) -> None:
        self.robot_ip = "127.0.0.1"
        self._timestamps = list(timestamps)
        self._idx = 0
        self.health = ConnectionHealth()
        self.connect_calls = 0
        self.disconnect_calls = 0

    def connect(self, *, with_control: bool = False) -> None:
        assert with_control is False, "watch mode must never open the control interface"
        self.connect_calls += 1

    def disconnect(self) -> None:
        self.disconnect_calls += 1

    def read_state(self):
        if self._idx >= len(self._timestamps):
            raise RTDEStateError("scripted reads exhausted")
        ts = self._timestamps[self._idx]
        self._idx += 1
        return _state(ts)


class _SucceedOnceReconnect:
    """First call succeeds (simulating a real recovered connection), every
    call after that fails -- guarantees the test can't loop forever even if
    `run_watch` keeps retrying."""

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, _link) -> bool:
        self.calls += 1
        return self.calls == 1


class _AlwaysFailReconnect:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, _link) -> bool:
        self.calls += 1
        return False


def _make_fake_clock():
    state = {"t": 0}

    def fake_monotonic_ns() -> int:
        state["t"] += 2_000_000  # 2ms/call, matches real 500Hz cadence
        return state["t"]

    return fake_monotonic_ns


@pytest.fixture(autouse=True)
def _no_real_sleep(monkeypatch):
    monkeypatch.setattr(ur5e_connect.time, "sleep", lambda _s: None)


def test_watch_detects_frozen_timestamp_without_any_exception(monkeypatch):
    """The core regression: read_state() never raises, but robot_timestamp_s
    stops advancing -- must be detected and trigger reconnect, not print
    stale data forever."""
    monkeypatch.setattr(ur5e_connect, "monotonic_ns", _make_fake_clock())
    # 3 real advancing reads, then exactly 5 repeats of the same timestamp
    # (StaleStateMonitor's default max_frozen_cycles) -- never raises.
    timestamps = [100.0, 100.5, 101.0, 101.0, 101.0, 101.0, 101.0]
    link = _FakeLink(timestamps)
    reconnect = _SucceedOnceReconnect()
    monkeypatch.setattr(ur5e_connect, "_attempt_reconnect", reconnect)

    result = ur5e_connect.run_watch(link, EStopLatch(), frequency_hz=500.0)

    assert reconnect.calls >= 1, "stale stream must trigger the reconnect path"
    assert result == 1  # scripted data is fully consumed after the detected
    # stall + one successful reconnect; the loop then hits real exhaustion
    # (RTDEStateError) and the second reconnect attempt fails by design.


def test_watch_stays_clean_when_timestamp_keeps_advancing(monkeypatch):
    """No false positives: a real advancing stream must consume ALL scripted
    reads cleanly -- the eventual reconnect here is the pre-existing,
    unrelated RTDEStateError path firing only once scripted data legitimately
    runs out, not the new stale-timestamp detection firing early/wrongly."""
    monkeypatch.setattr(ur5e_connect, "monotonic_ns", _make_fake_clock())
    timestamps = [100.0 + 0.002 * i for i in range(10)]
    link = _FakeLink(timestamps)
    reconnect = _AlwaysFailReconnect()
    monkeypatch.setattr(ur5e_connect, "_attempt_reconnect", reconnect)

    result = ur5e_connect.run_watch(link, EStopLatch(), frequency_hz=500.0)

    assert link._idx == len(timestamps), "all 10 advancing reads must succeed before any reconnect"
    assert result == 1  # exhausting scripted data raises RTDEStateError, unrelated to staleness


def test_watch_reports_failure_and_trips_estop_when_stale_and_reconnect_fails(monkeypatch):
    monkeypatch.setattr(ur5e_connect, "monotonic_ns", _make_fake_clock())
    timestamps = [100.0, 100.5, 101.0, 101.0, 101.0, 101.0, 101.0]
    link = _FakeLink(timestamps)
    reconnect = _AlwaysFailReconnect()
    monkeypatch.setattr(ur5e_connect, "_attempt_reconnect", reconnect)
    estop = EStopLatch()

    result = ur5e_connect.run_watch(link, estop, frequency_hz=500.0)

    assert result == 1
    assert estop.tripped
    assert reconnect.calls == 1  # fails immediately, no further scripted reads to retry against
