"""Tests for hardware/x_transport.py's start_q_rad plumbing -- fake RTDE
objects only, never opens a real socket.

Covers the new --start-q-rad support: pre-move validation (_validate_start_q_rad)
and that a caller-supplied pose actually reaches move_j instead of the
hardcoded HEIGHT_ALPHA_0_5_Q default.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from hardware.poses import HEIGHT_ALPHA_0_5_Q  # noqa: E402
from hardware.x_transport import _joint_move_ur5e_link, _validate_start_q_rad  # noqa: E402
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
    np.testing.assert_allclose(control.move_j_calls[0], HEIGHT_ALPHA_0_5_Q, atol=1e-9)


def test_joint_move_uses_caller_supplied_target():
    link, control = _make_link_with_movej()
    _joint_move_ur5e_link(link, motion_opt_in=True, target_q_rad=ALPHA_0_1_Q)
    assert len(control.move_j_calls) == 1
    np.testing.assert_allclose(control.move_j_calls[0], ALPHA_0_1_Q, atol=1e-9)
    # sanity: not the default pose
    assert not np.allclose(control.move_j_calls[0], HEIGHT_ALPHA_0_5_Q, atol=1e-3)


def test_joint_move_requires_motion_opt_in():
    link, control = _make_link_with_movej()
    with pytest.raises(ValueError, match="motion_opt_in"):
        _joint_move_ur5e_link(link, motion_opt_in=False, target_q_rad=ALPHA_0_1_Q)
    assert control.move_j_calls == []
