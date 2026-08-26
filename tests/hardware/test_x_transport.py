"""Tests for hardware/x_transport.py's start_q_rad plumbing -- fake RTDE
objects only, never opens a real socket.

Covers the new --start-q-rad support: pre-move validation (_validate_start_q_rad)
and that a caller-supplied pose actually reaches move_j instead of the
hardcoded default (HEIGHT_ALPHA_0_5_CLEARANCE_Q, since 2026-07-31 -- see
hardware/poses.py for why the default now includes the real-lab base
rotation).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from hardware.poses import HEIGHT_ALPHA_0_5_CLEARANCE_Q  # noqa: E402
from hardware.x_transport import _joint_move_ur5e_link, _validate_start_q_rad, run_x_transport  # noqa: E402
from hardware.link import UR5eLink  # noqa: E402


ALPHA_0_1_Q = np.array([0.0, -1.423717, -0.240000, -1.453717, 0.0, 0.0], dtype=np.float64)


# --------------------------------------------------------------------------- #
# _validate_start_q_rad
# --------------------------------------------------------------------------- #
def test_validate_start_q_rad_accepts_good_pose():
    out = _validate_start_q_rad(ALPHA_0_1_Q)
    np.testing.assert_allclose(out, ALPHA_0_1_Q)


def test_validate_start_q_rad_rejects_wrong_shape():
    with pytest.raises(ValueError, match="6 elements"):
        _validate_start_q_rad(np.array([0.0, 1.0, 2.0]))


def test_validate_start_q_rad_rejects_non_finite():
    bad = ALPHA_0_1_Q.copy()
    bad[2] = float("nan")
    with pytest.raises(ValueError, match="NaN/Inf"):
        _validate_start_q_rad(bad)


def test_validate_start_q_rad_rejects_out_of_bounds():
    # A plausible degrees-instead-of-radians typo: 90 instead of ~1.57.
    bad = ALPHA_0_1_Q.copy()
    bad[1] = 90.0
    with pytest.raises(ValueError, match="exceeds absolute joint limits"):
        _validate_start_q_rad(bad)


# --------------------------------------------------------------------------- #
# _joint_move_ur5e_link -- caller-supplied target actually reaches move_j
# --------------------------------------------------------------------------- #
class _FakeReceiveSettling:
    """Reports getActualQ() == the last-commanded moveJ target, simulating
    an instantly-settling robot so the polling loop in _joint_move_ur5e_link
    returns immediately."""

    def __init__(self) -> None:
        self.q = [0.0, -1.57, 0.0, -1.57, 0.0, 0.0]
        self.qd = [0.0] * 6
        self.tcp_pose = [0.0, -0.234, 1.08, 0.0, 0.0, 0.0]

    def getActualQ(self):
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


class _FakeControlRecordingMoveJ:
    def __init__(self, receive: _FakeReceiveSettling) -> None:
        self._receive = receive
        self.move_j_calls: list[list[float]] = []

    def servoL(self, pose, speed, acceleration, time, lookahead_time, gain):
        raise AssertionError("servoL should not be called by _joint_move_ur5e_link")

    def moveJ(self, q, speed, acceleration):
        self.move_j_calls.append(list(q))
        self._receive.q = list(q)  # simulate instant settle
        return True

    def servoStop(self):
        pass

    def stopScript(self):
        pass

    def disconnect(self):
        pass


def _make_link_with_movej():
    receive = _FakeReceiveSettling()
    control = _FakeControlRecordingMoveJ(receive)
    link = UR5eLink(
        "127.0.0.1",
        125.0,
        receive_factory=lambda ip, freq: receive,
        control_factory=lambda ip, freq: control,
    )
    return link, control


def test_joint_move_defaults_to_height_alpha_0_5():
    link, control = _make_link_with_movej()
    _joint_move_ur5e_link(link, motion_opt_in=True)
    assert len(control.move_j_calls) == 1
    np.testing.assert_allclose(control.move_j_calls[0], HEIGHT_ALPHA_0_5_CLEARANCE_Q, atol=1e-9)


def test_joint_move_uses_caller_supplied_target():
    link, control = _make_link_with_movej()
    _joint_move_ur5e_link(link, motion_opt_in=True, target_q_rad=ALPHA_0_1_Q)
    assert len(control.move_j_calls) == 1
    np.testing.assert_allclose(control.move_j_calls[0], ALPHA_0_1_Q, atol=1e-9)
    # sanity: not the default pose
    assert not np.allclose(control.move_j_calls[0], HEIGHT_ALPHA_0_5_CLEARANCE_Q, atol=1e-3)


def test_joint_move_requires_motion_opt_in():
    link, control = _make_link_with_movej()
    with pytest.raises(ValueError, match="motion_opt_in"):
        _joint_move_ur5e_link(link, motion_opt_in=False, target_q_rad=ALPHA_0_1_Q)
    assert control.move_j_calls == []


# --------------------------------------------------------------------------- #
# run_x_transport(control_mode="direct_torque", motion_opt_in=False) --
# guardrail-ordering regression test.
#
# Real bug found 2026-07-29: the direct_torque branch of run_x_transport()
# called UR5eDirectTorqueLink.connect() -- which unconditionally opens BOTH
# the RTDE receive AND control interfaces -- before motion_opt_in was ever
# checked, unlike the sibling position branch a few lines above (which never
# opens the control socket unless motion_opt_in is True). Not exploitable in
# practice because both real CLI callers already gate motion_opt_in at their
# own outer layer first, but this was a real, untested footgun at the
# library boundary. Fixed by checking motion_opt_in before connect() and
# calling connect(with_control=motion_opt_in).
# --------------------------------------------------------------------------- #
class _FakeDirectTorqueLink:
    """Records every call so the test can assert none of them fired."""

    def __init__(self, robot_ip: str, frequency_hz: float = 500.0) -> None:
        self.robot_ip = robot_ip
        self.frequency_hz = frequency_hz
        self.connect_calls: list[bool] = []
        self.direct_torque_calls = 0
        self.move_j_calls = 0
        self.safe_stop_calls = 0
        self.read_state_calls = 0

    def connect(self, *, with_control: bool = True) -> None:
        self.connect_calls.append(with_control)

    def read_state(self):
        self.read_state_calls += 1
        raise AssertionError("read_state() should never be reached when motion_opt_in is False")

    def direct_torque(self, tau_nm, *, friction_comp: bool = True) -> None:
        self.direct_torque_calls += 1

    def move_j(self, q_rad, *, speed_rad_s: float = 0.5, acceleration_rad_s2: float = 0.5) -> None:
        self.move_j_calls += 1

    def safe_stop(self, reason: str) -> None:
        self.safe_stop_calls += 1


def test_direct_torque_transport_blocks_before_connect_without_motion_opt_in(monkeypatch):
    fakes: list[_FakeDirectTorqueLink] = []

    def _factory(robot_ip: str, frequency_hz: float = 500.0):
        link = _FakeDirectTorqueLink(robot_ip, frequency_hz)
        fakes.append(link)
        return link

    monkeypatch.setattr("hardware.x_transport.UR5eDirectTorqueLink", _factory)

    with pytest.raises(ValueError, match="motion_opt_in"):
        run_x_transport(
            control_mode="direct_torque",
            robot_ip="127.0.0.1",
            config_path=Path("unused.yaml"),
            target_x_delta_m=0.02,
            move_duration_s=1.0,
            duration_s=2.0,
            output_dir=None,
            motion_opt_in=False,
        )

    assert len(fakes) == 1, "run_x_transport should construct exactly one UR5eDirectTorqueLink"
    fake = fakes[0]
    assert fake.connect_calls == [], "connect() must never be called when motion_opt_in is False"
    assert fake.read_state_calls == 0
    assert fake.direct_torque_calls == 0
    assert fake.move_j_calls == 0
    assert fake.safe_stop_calls == 0


def test_direct_torque_transport_connects_with_control_when_opted_in(monkeypatch):
    """Sanity check the positive path: when motion_opt_in is True, connect()
    is still called with with_control=True (i.e. this fix doesn't silently
    downgrade the real, opted-in case to a receive-only connection)."""
    fakes: list[_FakeDirectTorqueLink] = []

    def _factory(robot_ip: str, frequency_hz: float = 500.0):
        link = _FakeDirectTorqueLink(robot_ip, frequency_hz)
        fakes.append(link)
        return link

    monkeypatch.setattr("hardware.x_transport.UR5eDirectTorqueLink", _factory)

    # motion_opt_in=True proceeds past the guard into move_joints_to_pose /
    # run_x_transport_direct_torque, which need real RTDE machinery this fake
    # doesn't provide -- read_state() deliberately raises AssertionError
    # (never RTDEStateError) so any failure past connect() is unambiguous.
    with pytest.raises(AssertionError, match="should never be reached"):
        run_x_transport(
            control_mode="direct_torque",
            robot_ip="127.0.0.1",
            config_path=Path("unused.yaml"),
            target_x_delta_m=0.02,
            move_duration_s=1.0,
            duration_s=2.0,
            output_dir=None,
            motion_opt_in=True,
        )

    assert len(fakes) == 1
    assert fakes[0].connect_calls == [True], "connect(with_control=True) must still fire when opted in"


# --------------------------------------------------------------------------- #
# run_x_transport(control_mode="urscript") -- CartesianMoveLimits override
# dispatch. Real gap found 2026-07-30: run_urscript_x_transport's signature
# had no *_override kwargs at all, so the position/direct_torque branches'
# already-wired overrides were silently dropped for urscript mode. This
# checks the dispatch layer only (the override actually reaching
# CartesianMoveMonitor inside run_urscript_x_transport is covered by
# tests/hardware/test_deadline_and_staleness.py::
# test_urscript_move_limit_overrides_reach_cartesian_move_monitor).
# --------------------------------------------------------------------------- #
def test_run_x_transport_urscript_forwards_move_limit_overrides(monkeypatch):
    captured: dict[str, object] = {}

    def _fake_run_urscript_x_transport(**kwargs):
        captured.update(kwargs)
        from hardware.urscript_transport import UrscriptTransportResult

        return UrscriptTransportResult(
            ok=True, reason="", summary={"success": True}, script_path=None
        )

    monkeypatch.setattr(
        "hardware.x_transport.run_urscript_x_transport", _fake_run_urscript_x_transport
    )

    run_x_transport(
        control_mode="urscript",
        robot_ip="127.0.0.1",
        config_path=Path("unused.yaml"),
        target_x_delta_m=0.02,
        move_duration_s=1.0,
        duration_s=2.0,
        output_dir=None,
        motion_opt_in=True,
        max_tcp_accel_mps2_override=1.23,
        accel_gap_cycles_override=7,
        speed_lowpass_alpha_override=0.33,
        accel_max_consecutive_violations_override=4,
        accel_hard_multiple_override=6.5,
        speed_max_consecutive_violations_override=5,
        speed_hard_multiple_override=8.5,
    )

    assert captured["max_tcp_accel_mps2_override"] == 1.23
    assert captured["accel_gap_cycles_override"] == 7
    assert captured["speed_lowpass_alpha_override"] == 0.33
    assert captured["accel_max_consecutive_violations_override"] == 4
    assert captured["accel_hard_multiple_override"] == 6.5
    assert captured["speed_max_consecutive_violations_override"] == 5
    assert captured["speed_hard_multiple_override"] == 8.5


def test_run_x_transport_velocity_link_frequency_matches_rate_hz(monkeypatch):
    """Regression test for a real desync bug: the velocity dispatch branch
    used to hardcode UR5eLink(robot_ip, frequency_hz=125.0) regardless of the
    rate_hz the streaming loop itself was told to use -- with rate_hz=250 the
    control loop would target 250 Hz while the RTDE link's own
    ConnectionHealth staleness budget and interfaces were built for 125 Hz.
    Fixed by constructing UR5eLink(robot_ip, frequency_hz=rate_hz)."""
    monkeypatch.setattr("hardware.x_transport._joint_move_ur5e_link", lambda *a, **k: None)

    captured_link: dict[str, object] = {}

    def _fake_run_x_transport_velocity(link, **kwargs):
        captured_link["frequency_hz"] = link.frequency_hz
        from hardware.velocity_transport import VelocityTransportResult

        return VelocityTransportResult(ok=True, reason="", summary={"success": True}, trace_path=None)

    monkeypatch.setattr("hardware.x_transport.run_x_transport_velocity", _fake_run_x_transport_velocity)

    run_x_transport(
        control_mode="velocity",
        robot_ip="127.0.0.1",
        config_path=Path("unused.yaml"),
        target_x_delta_m=0.04,
        move_duration_s=1.0,
        duration_s=2.0,
        output_dir=None,
        motion_opt_in=True,
        rate_hz=250.0,
    )

    assert captured_link["frequency_hz"] == pytest.approx(250.0)


def test_run_x_transport_velocity_link_frequency_defaults_to_125(monkeypatch):
    """A caller that doesn't pass rate_hz at all must still get the
    historical 125.0 link frequency (default preserved)."""
    monkeypatch.setattr("hardware.x_transport._joint_move_ur5e_link", lambda *a, **k: None)

    captured_link: dict[str, object] = {}

    def _fake_run_x_transport_velocity(link, **kwargs):
        captured_link["frequency_hz"] = link.frequency_hz
        from hardware.velocity_transport import VelocityTransportResult

        return VelocityTransportResult(ok=True, reason="", summary={"success": True}, trace_path=None)

    monkeypatch.setattr("hardware.x_transport.run_x_transport_velocity", _fake_run_x_transport_velocity)

    run_x_transport(
        control_mode="velocity",
        robot_ip="127.0.0.1",
        config_path=Path("unused.yaml"),
        target_x_delta_m=0.04,
        move_duration_s=1.0,
        duration_s=2.0,
        output_dir=None,
        motion_opt_in=True,
    )

    assert captured_link["frequency_hz"] == pytest.approx(125.0)


# --------------------------------------------------------------------------- #
# run_x_transport(control_mode="velocity") -- dispatch-level test only (the
# real speedL streaming loop is covered by tests/hardware/test_velocity_
# transport.py). Mirrors the urscript dispatch test above: patches both
# _joint_move_ur5e_link (so no real moveJ/RTDE machinery is needed) and
# run_x_transport_velocity, and checks the override kwargs reach it.
# --------------------------------------------------------------------------- #
def test_run_x_transport_velocity_forwards_move_limit_overrides(monkeypatch):
    captured: dict[str, object] = {}

    monkeypatch.setattr("hardware.x_transport._joint_move_ur5e_link", lambda *a, **k: None)

    def _fake_run_x_transport_velocity(link, **kwargs):
        captured.update(kwargs)
        from hardware.velocity_transport import VelocityTransportResult

        return VelocityTransportResult(ok=True, reason="", summary={"success": True}, trace_path=None)

    monkeypatch.setattr("hardware.x_transport.run_x_transport_velocity", _fake_run_x_transport_velocity)

    result = run_x_transport(
        control_mode="velocity",
        robot_ip="127.0.0.1",
        config_path=Path("unused.yaml"),
        target_x_delta_m=0.02,
        move_duration_s=1.0,
        duration_s=2.0,
        output_dir=None,
        motion_opt_in=True,
        max_tcp_accel_mps2_override=1.23,
        accel_gap_cycles_override=7,
        speed_lowpass_alpha_override=0.33,
        accel_max_consecutive_violations_override=4,
        accel_hard_multiple_override=6.5,
        speed_max_consecutive_violations_override=5,
        speed_hard_multiple_override=8.5,
    )

    assert captured["max_tcp_accel_mps2_override"] == 1.23
    assert captured["accel_gap_cycles_override"] == 7
    assert captured["speed_lowpass_alpha_override"] == 0.33
    assert captured["accel_max_consecutive_violations_override"] == 4
    assert captured["accel_hard_multiple_override"] == 6.5
    assert captured["speed_max_consecutive_violations_override"] == 5
    assert captured["speed_hard_multiple_override"] == 8.5
    assert result.ok is True
    assert result.control_mode == "velocity"
