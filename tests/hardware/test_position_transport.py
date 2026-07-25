"""Position-mode X transport (servoL) with optional OSC shadow — mocked RTDE."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from hardware.link import UR5eLink  # noqa: E402
from hardware.position_transport import run_x_transport_position  # noqa: E402

CONFIG = REPO_ROOT / "config" / "ur5e_mujoco_torque_osc_tuned.yaml"


class _FakeReceive:
    def __init__(self, tcp_x: float = 0.4) -> None:
        self._tcp_x = float(tcp_x)
        self.q = [0.0, -0.835, -1.2, -0.985, 0.0, 0.0]
        self.qd = [0.0] * 6

    def getActualQ(self):
        return list(self.q)

    def getActualQd(self):
        return list(self.qd)

    def getActualTCPPose(self):
        return [self._tcp_x, -0.2, 0.3, 0.0, 3.14, 0.0]

    def getTimestamp(self):
        return 0.0

    def disconnect(self):
        pass


class _FakeControl:
    # Cap per-cycle TCP motion so CartesianMoveMonitor kinematic guards stay realistic.
    _MAX_TCP_STEP_M = 0.002

    def __init__(self, receive: _FakeReceive) -> None:
        self._receive = receive
        self.servo_l_calls = 0
        self.servo_stop_calls = 0

    def servoL(self, pose, speed, acceleration, time, lookahead_time, gain):
        self.servo_l_calls += 1
        target_x = float(pose[0])
        delta = target_x - self._receive._tcp_x
        step = min(abs(delta), self._MAX_TCP_STEP_M)
        if step > 0.0:
            self._receive._tcp_x += step if delta > 0.0 else -step

    def servoStop(self):
        self.servo_stop_calls += 1

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
def test_position_transport_streams_servo_l(tmp_path: Path) -> None:
    link, control = _link()
    result = run_x_transport_position(
        link,
        config_path=CONFIG,
        target_x_delta_m=0.01,
        move_duration_s=0.04,
        duration_s=0.08,
        output_dir=tmp_path,
        motion_opt_in=True,
        rate_hz=50.0,
        shadow_osc=False,
    )
    assert control.servo_l_calls >= 2
    assert control.servo_stop_calls >= 1
    assert result.summary["control_mode"] == "position"
    assert result.summary["backend"] == "servoL_position"
    assert result.trace_path is not None
    assert result.trace_path.exists()


@pytest.mark.hardware
def test_position_transport_shadow_osc_logs_tau(tmp_path: Path) -> None:
    pytest.importorskip("mujoco")
    link, _control = _link()
    result = run_x_transport_position(
        link,
        config_path=CONFIG,
        target_x_delta_m=0.01,
        move_duration_s=0.04,
        duration_s=0.08,
        output_dir=tmp_path,
        motion_opt_in=True,
        rate_hz=50.0,
        shadow_osc=True,
    )
    assert result.summary.get("shadow_osc") is True
    rows = result.trace_path.read_text(encoding="utf-8").strip().splitlines()
    assert rows
    import json

    first = json.loads(rows[0])
    assert "tau_shadow" in first
    assert len(first["tau_shadow"]) == 6
    assert "ee_pos" in first
    assert "x_error" in first


@pytest.mark.hardware
def test_position_transport_trace_scores_move_hold_metrics(tmp_path: Path) -> None:
    link, control = _link(tcp_x=0.4)
    result = run_x_transport_position(
        link,
        config_path=CONFIG,
        target_x_delta_m=0.01,
        move_duration_s=0.20,
        duration_s=0.40,
        output_dir=tmp_path,
        motion_opt_in=True,
        rate_hz=50.0,
        shadow_osc=False,
    )
    assert control.servo_l_calls >= 5
    assert result.summary.get("move_phase_achieved_x_delta_m", 0.0) > 0.0
    assert result.summary.get("valid_move_and_hold") is True
    assert result.ok is True
    assert result.summary.get("target_x_delta") == pytest.approx(0.01)
