from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from hardware import UR5eHardwareSession, UR5eHardwareSessionConfig  # noqa: E402


class _FakeReceive:
    def __init__(self) -> None:
        self._q = np.array([0.2, -1.15, 1.55, -1.8, -1.45, 0.35], dtype=np.float64)
        self._qd = np.zeros(6, dtype=np.float64)

    def getActualQ(self) -> list[float]:
        return self._q.tolist()

    def getActualQd(self) -> list[float]:
        return self._qd.tolist()

    def getActualTCPPose(self) -> list[float]:
        return [0.0, 0.0, 0.4, 0.0, 0.0, 0.0]

    def getTimestamp(self) -> float:
        return 12.34

    def getRobotMode(self) -> int:
        return 7

    def getSafetyStatus(self) -> int:
        return 3


class _FakeControl:
    def __init__(self) -> None:
        self.servoj_calls: list[tuple[list[float], float, float, float, float, float]] = []
        self.speedj_calls: list[tuple[list[float], float, float]] = []
        self.torque_calls: list[list[float]] = []

    def initPeriod(self) -> object:
        return object()

    def waitPeriod(self, _token: object) -> None:
        return None

    def servoJ(self, q: list[float], acceleration: float, velocity: float, period_s: float, lookahead_time: float, gain: float) -> None:
        self.servoj_calls.append((list(q), float(acceleration), float(velocity), float(period_s), float(lookahead_time), float(gain)))

    def speedJ(self, qd: list[float], acceleration: float, duration_s: float) -> None:
        self.speedj_calls.append((list(qd), float(acceleration), float(duration_s)))

    def stopJ(self, _accel: float = 2.0) -> None:
        return None

    def servoStop(self) -> None:
        return None

    def stopScript(self) -> None:
        return None

    def setJointTorque(self, tau: list[float]) -> None:
        self.torque_calls.append(list(tau))


def _session(*, motion_opt_in: bool = False, allow_nonzero_direct_torque: bool = False, direct_torque_zero_only: bool = True) -> UR5eHardwareSession:
    return UR5eHardwareSession(
        UR5eHardwareSessionConfig(
            robot_ip="127.0.0.1",
            frequency_hz=500.0,
            motion_opt_in=motion_opt_in,
            allow_nonzero_direct_torque=allow_nonzero_direct_torque,
            direct_torque_zero_only=direct_torque_zero_only,
        ),
        control_factory=lambda *_args, **_kwargs: _FakeControl(),
        receive_factory=lambda *_args, **_kwargs: _FakeReceive(),
    )


def test_connection_snapshot_reports_capabilities_without_motion() -> None:
    session = _session()

    snapshot = session.capture_snapshot(connect_control=True, include_state=True)

    assert snapshot.bridge["control_connected"] is True
    assert snapshot.bridge["receive_connected"] is True
    assert snapshot.bridge["supports_servoj"] is True
    assert snapshot.bridge["supports_speedj"] is True
    assert snapshot.bridge["supports_direct_torque"] is True
    assert snapshot.state is not None
    assert snapshot.state["state_ok"] is True
    assert np.isfinite(snapshot.state["q"][0])


def test_servoj_hold_requires_explicit_motion_opt_in() -> None:
    session = _session(motion_opt_in=False)
    result = session.request_servoj_hold(np.zeros(6, dtype=np.float64))

    assert result.accepted is False
    assert result.blocked is True
    assert "motion_opt_in" in result.reason


def test_servoj_hold_issues_command_when_opt_in_is_enabled() -> None:
    session = _session(motion_opt_in=True)
    result = session.request_servoj_hold(np.zeros(6, dtype=np.float64))

    assert result.accepted is True
    assert result.blocked is False
    assert result.reason.startswith("servoJ hold")
    control = session.bridge._control
    assert control is not None
    assert len(control.servoj_calls) == 1


def test_direct_torque_is_blocked_by_default() -> None:
    session = _session()
    result = session.request_direct_torque(np.ones(6, dtype=np.float64))

    assert result.accepted is False
    assert result.blocked is True
    assert "blocked by default" in result.reason.lower()


def test_direct_torque_reaches_fake_backend_when_explicitly_enabled() -> None:
    session = _session(allow_nonzero_direct_torque=True, direct_torque_zero_only=False)
    result = session.request_direct_torque(np.ones(6, dtype=np.float64))

    assert result.accepted is True
    assert result.blocked is False
    assert "direct torque" in result.reason.lower()
    control = session.bridge._control
    assert control is not None
    assert control.torque_calls == [np.ones(6, dtype=np.float64).tolist()]

