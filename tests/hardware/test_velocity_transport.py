"""Velocity-mode X transport (speedL) -- mocked RTDE, mirrors
test_position_transport.py's structure."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from hardware.link import RTDELinkError, UR5eLink  # noqa: E402
from hardware.velocity_transport import run_x_transport_velocity  # noqa: E402

CONFIG = REPO_ROOT / "config" / "ur5e_velocity_control.yaml"


class _FakeReceive:
    def __init__(self, tcp_x: float = 0.4) -> None:
        self._tcp_x = float(tcp_x)
        self.q = [0.0, -0.835, -1.2, -0.985, 0.0, 0.0]
        self.qd = [0.0] * 6
        self._ts = 0.0

    def getActualQ(self):
        return list(self.q)

    def getActualQd(self):
        return list(self.qd)

    def getActualTCPPose(self):
        return [self._tcp_x, -0.2, 0.3, 0.0, 3.14, 0.0]

    def getTimestamp(self):
        self._ts += 0.008
        return self._ts

    def getSafetyStatusBits(self):
        return 1

    def disconnect(self):
        pass


class _FakeControl:
    # Cap per-cycle TCP motion so CartesianMoveMonitor kinematic guards stay realistic.
    _MAX_TCP_STEP_M = 0.002

    def __init__(self, receive: _FakeReceive) -> None:
        self._receive = receive
        self.speed_l_calls = 0
        self.speed_stop_calls = 0

    def speedL(self, xd, acceleration, time):
        self.speed_l_calls += 1
        vx = float(xd[0])
        # Simple kinematic integration proxy: apply the commanded X velocity
        # for a nominal single-cycle dt, capped like _FakeControl's servoL
        # counterpart in test_position_transport.py.
        step = min(abs(vx) * 0.008, self._MAX_TCP_STEP_M)
        if step > 0.0:
            self._receive._tcp_x += step if vx > 0.0 else -step

    def speedStop(self, acceleration=10.0):
        self.speed_stop_calls += 1

    def servoL(self, pose, speed, acceleration, time, lookahead_time, gain):
        # Never called by velocity mode -- stubbed only because
        # UR5eLink.connect(with_control=True) unconditionally verifies
        # servoL's presence/signature (a real RTDEControlInterface always
        # has both servoL and speedL; this fake must too).
        raise AssertionError("servoL should never be called by velocity mode")

    def servoStop(self):
        pass

    def stopScript(self):
        pass

    def disconnect(self):
        pass


def _link(tcp_x: float = 0.4) -> tuple[UR5eLink, _FakeControl]:
    receive = _FakeReceive(tcp_x=tcp_x)
    control = _FakeControl(receive)
    link = UR5eLink(
        "127.0.0.1",
        125.0,
        receive_factory=lambda ip, freq: receive,
        control_factory=lambda ip, freq: control,
    )
    return link, control


@pytest.mark.hardware
def test_velocity_transport_streams_speed_l(tmp_path: Path) -> None:
    link, control = _link()
    result = run_x_transport_velocity(
        link,
        config_path=CONFIG,
        target_x_delta_m=0.01,
        move_duration_s=0.04,
        duration_s=0.08,
        output_dir=tmp_path,
        motion_opt_in=True,
        rate_hz=50.0,
    )
    assert control.speed_l_calls >= 2
    assert control.speed_stop_calls >= 1
    assert result.summary["control_mode"] == "velocity"
    assert result.summary["backend"] == "speedL_velocity"
    assert result.trace_path is not None
    assert result.trace_path.exists()


@pytest.mark.hardware
def test_velocity_transport_trace_has_xd_cmd_and_no_dynamics_fields(tmp_path: Path) -> None:
    link, _control = _link()
    result = run_x_transport_velocity(
        link,
        config_path=CONFIG,
        target_x_delta_m=0.01,
        move_duration_s=0.04,
        duration_s=0.08,
        output_dir=tmp_path,
        motion_opt_in=True,
        rate_hz=50.0,
    )
    rows = result.trace_path.read_text(encoding="utf-8").strip().splitlines()
    assert rows
    import json

    first = json.loads(rows[0])
    assert "xd_cmd" in first
    assert len(first["xd_cmd"]) == 6
    assert first["command_mode"] == "speedL"
    assert "tau_shadow" not in first  # no dynamics/torque path in this mode


@pytest.mark.hardware
def test_velocity_transport_trace_scores_move_hold_metrics(tmp_path: Path) -> None:
    link, control = _link(tcp_x=0.4)
    result = run_x_transport_velocity(
        link,
        config_path=CONFIG,
        target_x_delta_m=0.01,
        move_duration_s=0.20,
        duration_s=0.40,
        output_dir=tmp_path,
        motion_opt_in=True,
        rate_hz=50.0,
    )
    assert control.speed_l_calls >= 5
    assert result.summary.get("move_phase_achieved_x_delta_m", 0.0) > 0.0
    assert result.summary.get("target_x_delta") == pytest.approx(0.01)


@pytest.mark.hardware
def test_velocity_transport_requires_motion_opt_in():
    link, _control = _link()
    with pytest.raises(ValueError, match="motion_opt_in"):
        run_x_transport_velocity(
            link,
            config_path=CONFIG,
            target_x_delta_m=0.01,
            move_duration_s=0.04,
            duration_s=0.08,
            output_dir=None,
            motion_opt_in=False,
        )


class _NoSpeedLControl:
    """A control interface without speedL -- verify_speedl_signature() must
    reject this before any streaming starts."""

    def __init__(self, receive: _FakeReceive) -> None:
        self._receive = receive

    def servoL(self, pose, speed, acceleration, time, lookahead_time, gain):
        # Present so connect()'s own servoL check passes -- this test is
        # specifically about the speedL-signature check, not servoL's.
        pass

    def servoStop(self):
        pass

    def stopScript(self):
        pass

    def disconnect(self):
        pass


@pytest.mark.hardware
def test_velocity_transport_rejects_control_interface_without_speedl(tmp_path: Path):
    receive = _FakeReceive()
    control = _NoSpeedLControl(receive)
    link = UR5eLink(
        "127.0.0.1",
        125.0,
        receive_factory=lambda ip, freq: receive,
        control_factory=lambda ip, freq: control,
    )
    with pytest.raises(RTDELinkError, match="speedL"):
        run_x_transport_velocity(
            link,
            config_path=CONFIG,
            target_x_delta_m=0.01,
            move_duration_s=0.04,
            duration_s=0.08,
            output_dir=tmp_path,
            motion_opt_in=True,
            rate_hz=50.0,
        )
