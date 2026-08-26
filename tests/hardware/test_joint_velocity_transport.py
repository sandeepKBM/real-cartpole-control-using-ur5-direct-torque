"""Joint-velocity-mode X transport (speedJ) -- mocked RTDE, mirrors
test_velocity_transport.py's structure.

Unlike velocity_transport.py's speedL fake (which integrates the commanded
CARTESIAN velocity directly into tcp_x), this fake integrates a small,
fixed per-cycle joint delta from the commanded JOINT velocity so tcp_x moves
a little every cycle without needing a real inverse-kinematics model in the
test double -- the point of these tests is to exercise the REAL DLS/clamp
path (hardware/joint_velocity_transport.py calling
controller_core/damped_least_squares.py and then link.speed_j()), not to
reproduce exact forward kinematics.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from hardware.joint_velocity_transport import run_x_transport_joint_velocity  # noqa: E402
from hardware.link import RTDELinkError, UR5eLink  # noqa: E402

# NOT config/ur5e_velocity_control.yaml -- that config's reduced_task_dims:
# true makes CartesianVelocityController resolve the task internally, which
# double-resolves through DLS (see
# tests/hardware/test_joint_velocity_resolution_fix.py and
# config/ur5e_speedj_joint_velocity.yaml's own header for the full story).
# run_x_transport_joint_velocity() itself now refuses that config.
CONFIG = REPO_ROOT / "config" / "ur5e_speedj_joint_velocity.yaml"

# A well-conditioned start pose (same family as velocity_transport.py's own
# fake) -- NOT the singular ARM_Q0 pose, so a real Cartesian move should
# resolve through DLS with lambda_used == 0 (undamped) for these tests.
_WELL_CONDITIONED_Q = [0.0, -0.835, -1.2, -0.985, 0.2, 0.0]


class _FakeReceive:
    def __init__(self, tcp_x: float = 0.4, q: list[float] | None = None) -> None:
        self._tcp_x = float(tcp_x)
        self.q = list(q) if q is not None else list(_WELL_CONDITIONED_Q)
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
    # Cap per-cycle TCP motion (via a fake joint-delta-driven X nudge), same
    # convention as the speedL fake's _MAX_TCP_STEP_M.
    _MAX_TCP_STEP_M = 0.002

    def __init__(self, receive: _FakeReceive) -> None:
        self._receive = receive
        self.speed_j_calls = 0
        self.speed_stop_calls = 0
        self.last_qd = None

    def speedJ(self, qd, acceleration, time):
        self.speed_j_calls += 1
        self.last_qd = list(qd)
        # Simple kinematic integration proxy: nudge tcp_x by a small amount
        # in the sign of the shoulder_pan joint's commanded velocity, capped
        # like the speedL fake.
        vx_proxy = float(qd[0])
        step = min(abs(vx_proxy) * 0.008, self._MAX_TCP_STEP_M)
        if step > 0.0:
            self._receive._tcp_x += step if vx_proxy > 0.0 else -step

    def speedStop(self, acceleration=10.0):
        self.speed_stop_calls += 1

    def servoL(self, pose, speed, acceleration, time, lookahead_time, gain):
        raise AssertionError("servoL should never be called by joint_velocity mode")

    def servoStop(self):
        pass

    def stopScript(self):
        pass

    def disconnect(self):
        pass


def _link(tcp_x: float = 0.4, q: list[float] | None = None) -> tuple[UR5eLink, _FakeControl]:
    receive = _FakeReceive(tcp_x=tcp_x, q=q)
    control = _FakeControl(receive)
    link = UR5eLink(
        "127.0.0.1",
        125.0,
        receive_factory=lambda ip, freq: receive,
        control_factory=lambda ip, freq: control,
    )
    return link, control


@pytest.mark.hardware
def test_joint_velocity_transport_streams_speed_j(tmp_path: Path) -> None:
    link, control = _link()
    result = run_x_transport_joint_velocity(
        link,
        config_path=CONFIG,
        target_x_delta_m=0.01,
        move_duration_s=0.04,
        duration_s=0.08,
        output_dir=tmp_path,
        motion_opt_in=True,
        rate_hz=50.0,
    )
    assert control.speed_j_calls >= 2
    assert control.speed_stop_calls >= 1
    assert result.summary["control_mode"] == "joint_velocity"
    assert result.summary["backend"] == "speedJ_joint_velocity"
    assert result.trace_path is not None
    assert result.trace_path.exists()


@pytest.mark.hardware
def test_joint_velocity_transport_trace_has_dls_diagnostics(tmp_path: Path) -> None:
    link, _control = _link()
    result = run_x_transport_joint_velocity(
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
    assert "qd_cmd" in first and len(first["qd_cmd"]) == 6
    assert "qd_cmd_unclamped" in first and len(first["qd_cmd_unclamped"]) == 6
    assert "sigma_min" in first
    assert "dls_lambda_used" in first
    assert "qd_clamp_hit" in first
    assert first["command_mode"] == "speedJ"
    assert "tau_shadow" not in first  # no dynamics/torque path in this mode


@pytest.mark.hardware
def test_joint_velocity_transport_well_conditioned_pose_has_zero_damping(tmp_path: Path) -> None:
    """Away from a singularity (this fake's well-conditioned start pose),
    DLS should apply essentially no damping -- confirms the REAL DLS path is
    exercised (not stubbed), not just imported. Uses a deliberately large
    joint_velocity_clamp_radps to isolate this from clamp saturation (the
    fast 0.04s move's feedforward velocity, resolved through CONFIG's
    reduced_task_dims=False full-Cartesian-task DLS resolution, genuinely
    needs more than the module's conservative 0.3 rad/s default here --
    clamp behavior itself is covered separately by
    test_joint_velocity_transport_clamps_commanded_qd below)."""
    link, _control = _link(q=_WELL_CONDITIONED_Q)
    result = run_x_transport_joint_velocity(
        link,
        config_path=CONFIG,
        target_x_delta_m=0.01,
        move_duration_s=0.04,
        duration_s=0.08,
        output_dir=tmp_path,
        motion_opt_in=True,
        rate_hz=50.0,
        joint_velocity_clamp_radps=5.0,
    )
    assert result.summary["min_sigma_min"] is not None
    assert result.summary["min_sigma_min"] > 0.05  # well clear of the default sigma0
    assert result.summary["max_dls_lambda_used"] == pytest.approx(0.0, abs=1e-9)
    assert result.summary["any_qd_clamp_hit"] is False


@pytest.mark.hardware
def test_joint_velocity_transport_clamps_commanded_qd(tmp_path: Path) -> None:
    """A deliberately tiny joint_velocity_clamp_radps must actually reduce
    the |qd| reaching speedJ() below what DLS alone would have produced --
    exercises the mandatory hard clamp on the REAL command path, not just
    that the parameter is accepted."""
    link, control = _link(q=_WELL_CONDITIONED_Q)
    tiny_clamp = 1.0e-4
    result = run_x_transport_joint_velocity(
        link,
        config_path=CONFIG,
        target_x_delta_m=0.05,
        move_duration_s=0.04,
        duration_s=0.08,
        output_dir=tmp_path,
        motion_opt_in=True,
        rate_hz=50.0,
        joint_velocity_clamp_radps=tiny_clamp,
    )
    assert control.speed_j_calls >= 2
    assert control.last_qd is not None
    assert max(abs(v) for v in control.last_qd) <= tiny_clamp + 1e-12
    assert result.summary["any_qd_clamp_hit"] is True
    assert result.summary["joint_velocity_clamp_radps"] == pytest.approx(tiny_clamp)


@pytest.mark.hardware
def test_joint_velocity_transport_rejects_non_positive_clamp() -> None:
    link, _control = _link()
    with pytest.raises(ValueError, match="joint_velocity_clamp_radps"):
        run_x_transport_joint_velocity(
            link,
            config_path=CONFIG,
            target_x_delta_m=0.01,
            move_duration_s=0.04,
            duration_s=0.08,
            output_dir=None,
            motion_opt_in=True,
            rate_hz=50.0,
            joint_velocity_clamp_radps=0.0,
        )


@pytest.mark.hardware
def test_joint_velocity_transport_requires_motion_opt_in():
    link, _control = _link()
    with pytest.raises(ValueError, match="motion_opt_in"):
        run_x_transport_joint_velocity(
            link,
            config_path=CONFIG,
            target_x_delta_m=0.01,
            move_duration_s=0.04,
            duration_s=0.08,
            output_dir=None,
            motion_opt_in=False,
        )


class _NoSpeedJControl:
    """A control interface without speedJ -- verify_speedj_signature() must
    reject this before any streaming starts."""

    def __init__(self, receive: _FakeReceive) -> None:
        self._receive = receive

    def servoL(self, pose, speed, acceleration, time, lookahead_time, gain):
        pass

    def servoStop(self):
        pass

    def stopScript(self):
        pass

    def disconnect(self):
        pass


@pytest.mark.hardware
def test_joint_velocity_transport_rejects_control_interface_without_speedj(tmp_path: Path):
    receive = _FakeReceive()
    control = _NoSpeedJControl(receive)
    link = UR5eLink(
        "127.0.0.1",
        125.0,
        receive_factory=lambda ip, freq: receive,
        control_factory=lambda ip, freq: control,
    )
    with pytest.raises(RTDELinkError, match="speedJ"):
        run_x_transport_joint_velocity(
            link,
            config_path=CONFIG,
            target_x_delta_m=0.01,
            move_duration_s=0.04,
            duration_s=0.08,
            output_dir=tmp_path,
            motion_opt_in=True,
            rate_hz=50.0,
        )


@pytest.mark.hardware
def test_joint_velocity_transport_safety_stack_present(tmp_path: Path) -> None:
    """CartesianMoveMonitor's qd guard must still fire on the MEASURED qd
    (from getActualQd, independent of DLS's own clamp) -- confirms the same
    shared safety stack from velocity_transport.py is wired in, not skipped.
    """
    receive = _FakeReceive(q=_WELL_CONDITIONED_Q)
    receive.qd = [5.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # far above qd_max_radps
    control = _FakeControl(receive)
    link = UR5eLink(
        "127.0.0.1",
        125.0,
        receive_factory=lambda ip, freq: receive,
        control_factory=lambda ip, freq: control,
    )
    result = run_x_transport_joint_velocity(
        link,
        config_path=CONFIG,
        target_x_delta_m=0.01,
        move_duration_s=0.04,
        duration_s=0.5,
        output_dir=tmp_path,
        motion_opt_in=True,
        rate_hz=50.0,
    )
    assert result.ok is False
    assert "qd_max_radps" in result.summary["termination_reason"] or "qd" in result.summary["termination_reason"]
