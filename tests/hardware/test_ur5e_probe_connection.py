"""Fake-link tests for the pre-flight connection probe -- never opens a real
socket. Mirrors test_ur5e_connect_watch.py's fake/pattern."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from hardware.link import RTDEStateError  # noqa: E402
from hardware.safety import ConnectionHealth  # noqa: E402
import tools.ur5e_probe_connection as probe_mod  # noqa: E402


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
    def __init__(self, timestamps: list[float | None], *, raise_after: int | None = None) -> None:
        self.robot_ip = "127.0.0.1"
        self._timestamps = list(timestamps)
        self._idx = 0
        self._raise_after = raise_after
        self.health = ConnectionHealth()
        self.connect_calls = 0
        self.disconnect_calls = 0

    def connect(self, *, with_control: bool = False) -> None:
        assert with_control is False
        self.connect_calls += 1

    def disconnect(self) -> None:
        self.disconnect_calls += 1

    def read_state(self):
        if self._raise_after is not None and self._idx >= self._raise_after:
            raise RTDEStateError("simulated read failure")
        if self._idx >= len(self._timestamps):
            # Repeat the last value forever (simulating a real, indefinite
            # stall) -- the probe must be the one to stop calling, via its
            # own duration deadline or the stale detection, not run forever.
            ts = self._timestamps[-1] if self._timestamps else None
        else:
            ts = self._timestamps[self._idx]
        self._idx += 1
        return _state(ts)


def _make_fake_clock(step_ns: int = 2_000_000):
    state = {"t": 0}

    def fake_monotonic_ns() -> int:
        state["t"] += step_ns
        return state["t"]

    return fake_monotonic_ns


@pytest.fixture(autouse=True)
def _no_real_sleep_and_fast_deadline(monkeypatch):
    # probe() uses time.monotonic() for its wall-clock deadline and
    # time.sleep() between cycles -- fake both so the test runs instantly
    # regardless of --duration-s, driven entirely by a fake advancing clock.
    monkeypatch.setattr(probe_mod.time, "sleep", lambda _s: None)
    state = {"t": 0.0}

    def fake_monotonic() -> float:
        state["t"] += 0.002
        return state["t"]

    monkeypatch.setattr(probe_mod.time, "monotonic", fake_monotonic)


def test_probe_passes_on_clean_advancing_stream(monkeypatch):
    monkeypatch.setattr(probe_mod, "monotonic_ns", _make_fake_clock())
    timestamps = [100.0 + 0.002 * i for i in range(50)]
    link = _FakeLink(timestamps)

    result = probe_mod.probe(link, duration_s=0.05, frequency_hz=500.0)

    assert result == 0
    assert link.disconnect_calls == 1


def test_probe_fails_fast_on_frozen_timestamp(monkeypatch):
    monkeypatch.setattr(probe_mod, "monotonic_ns", _make_fake_clock())
    # 3 advancing, then frozen forever (the real observed failure mode).
    timestamps = [100.0, 100.5, 101.0]
    link = _FakeLink(timestamps)

    result = probe_mod.probe(link, duration_s=1.0, frequency_hz=500.0)

    assert result == 1
    assert link.disconnect_calls == 1
    # Must detect within a bounded number of cycles, not run the full
    # 1-second/500 = 500-cycle window before giving up.
    assert link._idx < 20


def test_probe_fails_fast_on_read_exception(monkeypatch):
    monkeypatch.setattr(probe_mod, "monotonic_ns", _make_fake_clock())
    timestamps = [100.0, 100.5, 101.0, 101.5, 102.0]
    link = _FakeLink(timestamps, raise_after=3)

    result = probe_mod.probe(link, duration_s=1.0, frequency_hz=500.0)

    assert result == 1
    assert link.disconnect_calls == 1
