"""Real-hardware Cartesian transport-axis plumbing (``transport_axis_index``).

The three real-hardware transport loops (``position``/``direct_torque``/
``urscript``) and their ``hardware/x_transport.py`` dispatcher used to hardcode
the transport axis to world-X (index 0) at every site: the
``CartesianMoveMonitor.set_start(move_axis_index=...)`` /
``ImpedanceSafetyMonitor.set_initial_position(move_axis=...)`` calls, the
commanded/reference TCP pose component, and the reported metrics. This file
pins both halves of the new plumbing:

* the **default** (omitted / ``transport_axis_index=0``) still drives, guards
  and reports along world-X exactly as before -- this is the only property
  that matters for every existing real-hardware run;
* ``transport_axis_index=1`` really reaches ``move_axis_index=1`` and really
  writes the target into pose component 1, not component 0.

Fake RTDE objects only, following the patterns already used in
test_position_transport.py / test_deadline_and_staleness.py -- no socket is
ever opened and no robot-facing script is invoked.
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
from hardware.poses import HEIGHT_ALPHA_0_5_Q  # noqa: E402
from hardware.position_transport import run_x_transport_position  # noqa: E402
from hardware.safety import CartesianMoveMonitor  # noqa: E402
from hardware.transport_common import validate_transport_axis_index  # noqa: E402
from hardware.urscript_gen import DEFAULT_CONFIG as URSCRIPT_CONFIG  # noqa: E402
from hardware.urscript_transport import run_urscript_x_transport  # noqa: E402
from hardware.x_transport import run_x_transport  # noqa: E402

CONFIG = REPO_ROOT / "config" / "ur5e_mujoco_torque_osc_tuned.yaml"

START_TCP_POSE = [0.4, -0.2, 0.3, 0.0, 3.14, 0.0]


# --------------------------------------------------------------------------- #
# _validate_transport_axis_index
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("idx", [0, 1, 2])
def test_validate_transport_axis_index_accepts_valid_axes(idx: int) -> None:
    assert validate_transport_axis_index(idx) == idx
    assert isinstance(validate_transport_axis_index(idx), int)


def test_validate_transport_axis_index_accepts_numpy_int() -> None:
    assert validate_transport_axis_index(np.int64(2)) == 2


@pytest.mark.parametrize("bad", [3, -1, 6])
def test_validate_transport_axis_index_rejects_out_of_range(bad: int) -> None:
    with pytest.raises(ValueError, match="must be 0, 1, or 2"):
        validate_transport_axis_index(bad)


@pytest.mark.parametrize("bad", [1.0, "1", None, True])
def test_validate_transport_axis_index_rejects_non_int(bad) -> None:
    # bool is deliberately rejected too: True would otherwise index the Y
    # component of a pose through int(True) == 1.
    with pytest.raises(ValueError, match="must be an int 0, 1, or 2"):
        validate_transport_axis_index(bad)


# --------------------------------------------------------------------------- #
# run_x_transport dispatch: the X-only control modes must refuse a non-zero
# axis rather than command X while guarding Y/Z.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("mode", ["direct_torque", "urscript", "velocity"])
def test_run_x_transport_rejects_non_zero_axis_for_x_only_control_modes(mode: str) -> None:
    with pytest.raises(ValueError, match="only supported for control_mode='position'"):
        run_x_transport(
            control_mode=mode,
            robot_ip="127.0.0.1",
            config_path=Path("unused.yaml"),
            target_x_delta_m=0.02,
            move_duration_s=1.0,
            duration_s=2.0,
            output_dir=None,
            motion_opt_in=True,
            transport_axis_index=1,
        )


def test_run_x_transport_rejects_invalid_axis_before_anything_else() -> None:
    with pytest.raises(ValueError, match="must be 0, 1, or 2"):
        run_x_transport(
            control_mode="position",
            robot_ip="127.0.0.1",
            config_path=Path("unused.yaml"),
            target_x_delta_m=0.02,
            move_duration_s=1.0,
            duration_s=2.0,
            output_dir=None,
            motion_opt_in=True,
            transport_axis_index=3,
        )


def test_run_x_transport_position_forwards_transport_axis_index(monkeypatch) -> None:
    captured: dict[str, object] = {}

    monkeypatch.setattr("hardware.x_transport._joint_move_ur5e_link", lambda *a, **k: None)

    def _fake_run_x_transport_position(link, **kwargs):
        captured.update(kwargs)
        from hardware.position_transport import PositionTransportResult

        return PositionTransportResult(ok=True, reason="", summary={"success": True}, trace_path=None)

    monkeypatch.setattr("hardware.x_transport.run_x_transport_position", _fake_run_x_transport_position)

    run_x_transport(
        control_mode="position",
        robot_ip="127.0.0.1",
        config_path=Path("unused.yaml"),
        target_x_delta_m=0.02,
        move_duration_s=1.0,
        duration_s=2.0,
        output_dir=None,
        motion_opt_in=True,
        transport_axis_index=1,
    )
    assert captured["transport_axis_index"] == 1


def test_run_x_transport_defaults_to_axis_zero(monkeypatch) -> None:
    """Omitting the flag entirely must still hand every mode axis 0."""
    captured: dict[str, object] = {}

    monkeypatch.setattr("hardware.x_transport._joint_move_ur5e_link", lambda *a, **k: None)

    def _fake_run_x_transport_position(link, **kwargs):
        captured.update(kwargs)
        from hardware.position_transport import PositionTransportResult

        return PositionTransportResult(ok=True, reason="", summary={"success": True}, trace_path=None)

    monkeypatch.setattr("hardware.x_transport.run_x_transport_position", _fake_run_x_transport_position)

    run_x_transport(
        control_mode="position",
        robot_ip="127.0.0.1",
        config_path=Path("unused.yaml"),
        target_x_delta_m=0.02,
        move_duration_s=1.0,
        duration_s=2.0,
        output_dir=None,
        motion_opt_in=True,
    )
    assert captured["transport_axis_index"] == 0


# --------------------------------------------------------------------------- #
# position mode (servoL) -- the commanded waypoint component itself.
# --------------------------------------------------------------------------- #
class _FakeReceive:
    """Same shape as test_position_transport.py's fake, but axis-generic:
    the TCP position it reports is a mutable 3-vector so _FakeControl can
    advance whichever component servoL was told to move."""

    def __init__(self) -> None:
        self.q = [0.0, -0.835, -1.2, -0.985, 0.0, 0.0]
        self.qd = [0.0] * 6
        self.pos = list(START_TCP_POSE[:3])
        self._ts = 0.0

    def getActualQ(self):
        return list(self.q)

    def getActualQd(self):
        return list(self.qd)

    def getActualTCPPose(self):
        return list(self.pos) + list(START_TCP_POSE[3:])

    def getTimestamp(self):
        self._ts += 0.002
        return self._ts

    def getSafetyStatusBits(self):
        return 1  # IS_NORMAL_MODE

    def disconnect(self):
        pass


class _FakeControl:
    _MAX_TCP_STEP_M = 0.002

    def __init__(self, receive: _FakeReceive, axis: int) -> None:
        self._receive = receive
        self._axis = int(axis)
        self.waypoints: list[list[float]] = []

    def servoL(self, pose, speed, acceleration, time, lookahead_time, gain):
        self.waypoints.append(list(pose))
        delta = float(pose[self._axis]) - self._receive.pos[self._axis]
        step = min(abs(delta), self._MAX_TCP_STEP_M)
        if step > 0.0:
            self._receive.pos[self._axis] += step if delta > 0.0 else -step

    def servoStop(self):
        pass

    def stopScript(self):
        pass

    def disconnect(self):
        pass


def _position_link(axis: int) -> tuple[UR5eLink, _FakeControl]:
    receive = _FakeReceive()
    control = _FakeControl(receive, axis)
    link = UR5eLink(
        "127.0.0.1",
        125.0,
        receive_factory=lambda ip, freq: receive,
        control_factory=lambda ip, freq: control,
    )
    return link, control


def _run_position(link: UR5eLink, tmp_path: Path, **kwargs):
    return run_x_transport_position(
        link,
        config_path=CONFIG,
        target_x_delta_m=0.01,
        move_duration_s=0.20,
        duration_s=0.40,
        output_dir=tmp_path,
        motion_opt_in=True,
        rate_hz=50.0,
        shadow_osc=False,
        **kwargs,
    )


@pytest.mark.hardware
def test_position_default_axis_commands_x_only(tmp_path: Path) -> None:
    link, control = _position_link(axis=0)
    result = _run_position(link, tmp_path)

    assert control.waypoints, "servoL was never called"
    assert result.summary["transport_axis_index"] == 0
    xs = [wp[0] for wp in control.waypoints]
    assert max(xs) - min(xs) > 1e-4, "X component of the commanded waypoint never moved"
    for wp in control.waypoints:
        # every non-transport component stays pinned to the start pose
        assert wp[1] == pytest.approx(START_TCP_POSE[1])
        assert wp[2] == pytest.approx(START_TCP_POSE[2])
    assert result.summary["achieved_x_delta_m"] > 0.0


@pytest.mark.hardware
def test_position_axis_one_commands_y_and_leaves_x_pinned(tmp_path: Path) -> None:
    link, control = _position_link(axis=1)
    result = _run_position(link, tmp_path, transport_axis_index=1)

    assert control.waypoints, "servoL was never called"
    assert result.summary["transport_axis_index"] == 1
    ys = [wp[1] for wp in control.waypoints]
    assert max(ys) - min(ys) > 1e-4, "Y component of the commanded waypoint never moved"
    for wp in control.waypoints:
        # the previously-hardcoded axis must now be the one held constant
        assert wp[0] == pytest.approx(START_TCP_POSE[0])
        assert wp[2] == pytest.approx(START_TCP_POSE[2])
    # achieved delta is measured along Y now, and Y really moved
    assert result.summary["achieved_x_delta_m"] > 0.0


@pytest.mark.hardware
@pytest.mark.parametrize("axis", [0, 1, 2])
def test_position_move_monitor_gets_the_selected_axis(tmp_path: Path, monkeypatch, axis: int) -> None:
    captured: dict[str, object] = {}
    real_cls = CartesianMoveMonitor

    class _SpyMoveMonitor(real_cls):
        def set_start(self, tcp_pose, move_axis_index: int) -> None:
            captured["move_axis_index"] = move_axis_index
            super().set_start(tcp_pose, move_axis_index)

    monkeypatch.setattr("hardware.position_transport.CartesianMoveMonitor", _SpyMoveMonitor)

    link, _control = _position_link(axis=axis)
    _run_position(link, tmp_path, transport_axis_index=axis)
    assert captured["move_axis_index"] == axis


@pytest.mark.hardware
def test_position_rejects_bad_axis(tmp_path: Path) -> None:
    link, _control = _position_link(axis=0)
    with pytest.raises(ValueError, match="must be 0, 1, or 2"):
        _run_position(link, tmp_path, transport_axis_index=3)


# --------------------------------------------------------------------------- #
# direct_torque mode -- the guard's reference pose (the np.concatenate site).
# --------------------------------------------------------------------------- #
class _MockDTLink:
    """Mock UR5eDirectTorqueLink (dynamics_source='rtde', so no pinocchio),
    copied from test_deadline_and_staleness.py's fake."""

    def __init__(self) -> None:
        from hardware.safety import UR5eSafetyLimits

        self.limits = UR5eSafetyLimits()
        self._ts = 0.0

    def connect(self) -> None:
        pass

    def read_state(self) -> UR5eState:
        self._ts += 0.002
        return UR5eState(
            q=HEIGHT_ALPHA_0_5_Q.copy(),
            qd=np.zeros(6),
            tcp_pose=np.array(START_TCP_POSE, dtype=np.float64),
            host_stamp_ns=time.monotonic_ns(),
            robot_timestamp_s=self._ts,
            safety_status=None,
        )

    def get_jacobian(self) -> np.ndarray:
        return np.eye(6)

    def get_mass_matrix(self) -> np.ndarray:
        return np.eye(6)

    def direct_torque(self, tau_nm: np.ndarray, *, friction_comp: bool = True) -> None:
        pass

    @staticmethod
    def compose_robot_state(
        link_state, *, jacobian, mass_matrix, time_s, target_x, target_x_vel,
        dt_s=None, target_x_accel=None, transport_axis_index=0,
    ):
        return UR5eDirectTorqueLink.compose_robot_state(
            link_state, jacobian=jacobian, mass_matrix=mass_matrix,
            time_s=time_s, target_x=target_x, target_x_vel=target_x_vel, dt_s=dt_s,
            target_x_accel=target_x_accel, transport_axis_index=transport_axis_index,
        )

    def build_robot_state(self, link_state, *, time_s, target_x, target_x_vel, dt_s=None, transport_axis_index=0):
        return self.compose_robot_state(
            link_state, jacobian=self.get_jacobian(), mass_matrix=self.get_mass_matrix(),
            time_s=time_s, target_x=target_x, target_x_vel=target_x_vel, dt_s=dt_s,
            transport_axis_index=transport_axis_index,
        )

    def safe_stop(self, reason: str) -> None:
        pass


def _spy_move_monitor(monkeypatch, module_path: str) -> dict[str, object]:
    """Patch ``CartesianMoveMonitor`` in ``module_path`` with a subclass that
    records the axis it was started with and every target pose it was
    checked against, then delegates to the real implementation (so the real
    guards still run)."""
    captured: dict[str, object] = {"targets": []}
    real_cls = CartesianMoveMonitor

    class _SpyMoveMonitor(real_cls):
        def set_start(self, tcp_pose, move_axis_index: int) -> None:
            captured["move_axis_index"] = move_axis_index
            captured["start_pose"] = np.asarray(tcp_pose, dtype=np.float64).copy()
            super().set_start(tcp_pose, move_axis_index)

        def check(self, *args, **kwargs):
            target = kwargs.get("target_tcp_pose")
            if target is not None:
                captured["targets"].append(np.asarray(target, dtype=np.float64).copy())
            return super().check(*args, **kwargs)

    monkeypatch.setattr(f"{module_path}.CartesianMoveMonitor", _SpyMoveMonitor)
    return captured


def _run_direct_torque(link, tmp_path: Path, **kwargs):
    return run_x_transport_direct_torque(
        link,  # type: ignore[arg-type]
        config_path=CONFIG,
        target_x_delta_m=0.01,
        move_duration_s=0.02,
        duration_s=0.03,
        output_dir=tmp_path,
        motion_opt_in=True,
        record_latency=False,
        dynamics_source="rtde",
        enable_residual_observer=False,
        **kwargs,
    )


@pytest.mark.hardware
def test_direct_torque_default_axis_targets_pose_component_zero(tmp_path: Path, monkeypatch) -> None:
    captured = _spy_move_monitor(monkeypatch, "hardware.direct_torque_transport")
    result = _run_direct_torque(_MockDTLink(), tmp_path)

    assert captured["move_axis_index"] == 0
    assert result.summary["transport_axis_index"] == 0
    targets = captured["targets"]
    assert targets, "CartesianMoveMonitor.check() was never called"
    start = np.asarray(START_TCP_POSE, dtype=np.float64)
    for target in targets:
        assert target.shape == (6,)
        # components 1..5 are the untouched start pose (this is exactly what
        # the replaced np.concatenate(([target_x], state0.tcp_pose[1:6]))
        # produced)
        np.testing.assert_allclose(target[1:6], start[1:6], rtol=0, atol=0)
    # the transport component really tracks the profile, i.e. it left x0
    assert max(float(t[0]) for t in targets) > float(start[0])


@pytest.mark.hardware
def test_direct_torque_axis_one_targets_pose_component_one(tmp_path: Path, monkeypatch) -> None:
    captured = _spy_move_monitor(monkeypatch, "hardware.direct_torque_transport")
    result = _run_direct_torque(_MockDTLink(), tmp_path, transport_axis_index=1)

    assert captured["move_axis_index"] == 1
    assert result.summary["transport_axis_index"] == 1
    targets = captured["targets"]
    assert targets, "CartesianMoveMonitor.check() was never called"
    start = np.asarray(START_TCP_POSE, dtype=np.float64)
    for target in targets:
        # regression against the old axis-0 concatenation: X must stay at its
        # start value and the target must land in component 1 instead
        assert float(target[0]) == pytest.approx(float(start[0]))
        np.testing.assert_allclose(target[2:6], start[2:6], rtol=0, atol=0)
    assert max(float(t[1]) for t in targets) > float(start[1])


@pytest.mark.hardware
def test_direct_torque_rejects_bad_axis(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="must be an int 0, 1, or 2"):
        # a float would previously have been silently truncated by int(), i.e.
        # 2.5 would have quietly meant "Z"
        _run_direct_torque(_MockDTLink(), tmp_path, transport_axis_index=2.5)
    with pytest.raises(ValueError, match="must be 0, 1, or 2"):
        _run_direct_torque(_MockDTLink(), tmp_path, transport_axis_index=3)


@pytest.mark.hardware
def test_direct_torque_impedance_safety_monitor_gets_the_selected_axis(
    tmp_path: Path, monkeypatch
) -> None:
    captured: dict[str, object] = {}
    from controller_core.safety import ImpedanceSafetyMonitor

    class _SpySafety(ImpedanceSafetyMonitor):
        def set_initial_position(self, position, move_axis: int) -> None:
            captured["move_axis"] = move_axis
            super().set_initial_position(position, move_axis)

    monkeypatch.setattr("hardware.direct_torque_transport.ImpedanceSafetyMonitor", _SpySafety)
    _run_direct_torque(_MockDTLink(), tmp_path, transport_axis_index=1)
    assert captured["move_axis"] == 1


# --------------------------------------------------------------------------- #
# urscript mode -- Python-side supervisor only (the on-robot script itself is
# world-X, which is why hardware/x_transport.py refuses a non-zero axis for
# this mode; the plumbing below is still pinned so the library boundary is
# consistent and testable).
# --------------------------------------------------------------------------- #
class _UrReceive:
    def __init__(self) -> None:
        self._ts = 0.0

    def getActualQ(self):
        return [0.0, -0.835, -1.2, -0.985, 0.0, 0.0]

    def getActualQd(self):
        return [0.0] * 6

    def getActualTCPPose(self):
        return list(START_TCP_POSE)

    def getTimestamp(self):
        self._ts += 0.002
        return self._ts

    def getSafetyStatusBits(self):
        return 1

    def disconnect(self):
        pass


class _UrControl:
    def __init__(self) -> None:
        self._stop_requested = False

    def setInputIntRegister(self, reg, value):
        if int(value) == 1:
            self._stop_requested = True

    def sendCustomScript(self, text):
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline:
            if self._stop_requested:
                return True
            time.sleep(0.005)
        return True

    def disconnect(self):
        pass


def _run_urscript(monkeypatch, tmp_path: Path, **kwargs):
    control = _UrControl()
    receive = _UrReceive()
    monkeypatch.setattr(
        "hardware.urscript_transport._load_rtde_classes",
        lambda: (lambda ip, freq: control, lambda ip, freq: receive),
    )
    return run_urscript_x_transport(
        robot_ip="127.0.0.1",
        config_path=URSCRIPT_CONFIG,
        target_x_delta_m=0.01,
        move_duration_s=0.05,
        duration_s=0.1,
        output_dir=tmp_path,
        motion_opt_in=True,
        skip_joint_move=True,
        monitor_hz=200.0,
        **kwargs,
    )


@pytest.mark.hardware
def test_urscript_default_axis_is_x(tmp_path: Path, monkeypatch) -> None:
    captured = _spy_move_monitor(monkeypatch, "hardware.urscript_transport")
    result = _run_urscript(monkeypatch, tmp_path)

    assert captured["move_axis_index"] == 0
    assert result.summary["transport_axis_index"] == 0
    targets = captured["targets"]
    assert targets
    start = np.asarray(START_TCP_POSE, dtype=np.float64)
    assert float(targets[0][0]) == pytest.approx(float(start[0]) + 0.01)
    np.testing.assert_allclose(targets[0][1:6], start[1:6], rtol=0, atol=0)


@pytest.mark.hardware
def test_urscript_axis_one_targets_pose_component_one(tmp_path: Path, monkeypatch) -> None:
    captured = _spy_move_monitor(monkeypatch, "hardware.urscript_transport")
    result = _run_urscript(monkeypatch, tmp_path, transport_axis_index=1)

    assert captured["move_axis_index"] == 1
    assert result.summary["transport_axis_index"] == 1
    targets = captured["targets"]
    assert targets
    start = np.asarray(START_TCP_POSE, dtype=np.float64)
    assert float(targets[0][0]) == pytest.approx(float(start[0]))
    assert float(targets[0][1]) == pytest.approx(float(start[1]) + 0.01)


@pytest.mark.hardware
def test_urscript_rejects_bad_axis(tmp_path: Path, monkeypatch) -> None:
    with pytest.raises(ValueError, match="must be 0, 1, or 2"):
        _run_urscript(monkeypatch, tmp_path, transport_axis_index=7)


# --------------------------------------------------------------------------- #
# UR5eDirectTorqueLink.compose_robot_state -- the RobotState contract field.
# --------------------------------------------------------------------------- #
def _link_state() -> UR5eState:
    return UR5eState(
        q=HEIGHT_ALPHA_0_5_Q.copy(),
        qd=np.zeros(6),
        tcp_pose=np.array(START_TCP_POSE, dtype=np.float64),
        host_stamp_ns=0,
        robot_timestamp_s=0.0,
        safety_status=None,
    )


def test_compose_robot_state_defaults_to_axis_zero() -> None:
    st = UR5eDirectTorqueLink.compose_robot_state(
        _link_state(), jacobian=np.eye(6), mass_matrix=np.eye(6),
        time_s=0.0, target_x=0.4, target_x_vel=0.0,
    )
    assert st["transport_axis_index"] == 0


def test_compose_robot_state_carries_selected_axis() -> None:
    st = UR5eDirectTorqueLink.compose_robot_state(
        _link_state(), jacobian=np.eye(6), mass_matrix=np.eye(6),
        time_s=0.0, target_x=-0.2, target_x_vel=0.0, transport_axis_index=1,
    )
    assert st["transport_axis_index"] == 1
