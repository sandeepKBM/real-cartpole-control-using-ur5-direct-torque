"""CLI-parsing and dispatch tests for tools/ur5e_velocity_x_transport.py.

Mirrors tests/hardware/test_direct_torque_x_transport_noise_robust_guards.py's
structure and mocking style (load the tool as a module via importlib, drive
its parse_args()/resolve_move_limit_overrides()/main() directly) -- no robot
connection or motion is ever exercised: main() is driven end-to-end with
hardware.x_transport.run_x_transport, hardware.dashboard.power_on_and_release
and query_remote_control monkeypatched to fakes, and the --probe-only path is
exercised with hardware.link.UR5eLink replaced by a fake receive-only stub.

This is the test that actually exercises the CLI's argument wiring end to
end, per AGENTS.md's "verify the effect, not the invocation" rule: it does
not just import the module, it calls parse_args()/main() and asserts the
kwargs that reach run_x_transport, catching exactly the class of bug that
rule warns about (a flag that parses but is silently dropped before the call
site).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from hardware.safety import CartesianMoveLimits, NOISE_ROBUST_GUARD_OVERRIDES  # noqa: E402


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "ur5e_velocity_x_transport", REPO_ROOT / "tools" / "ur5e_velocity_x_transport.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(REPO_ROOT / "tools"))
    try:
        spec.loader.exec_module(module)  # type: ignore[union-attr]
    finally:
        sys.path.pop(0)
    return module


MODULE = _load_module()

_BASE_ARGV = ["--robot-ip", "10.0.0.5"]


def _parse(monkeypatch: pytest.MonkeyPatch, extra_argv: list[str]):
    monkeypatch.setattr(sys, "argv", ["ur5e_velocity_x_transport.py", *_BASE_ARGV, *extra_argv])
    return MODULE.parse_args()


# ---------------------------------------------------------------------------
# Argument parsing / defaults
# ---------------------------------------------------------------------------


def test_default_config_is_velocity_control_yaml() -> None:
    assert MODULE.DEFAULT_CONFIG == REPO_ROOT / "config" / "ur5e_velocity_control.yaml"
    assert MODULE.DEFAULT_CONFIG.exists()


def test_no_flags_produces_no_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    args = _parse(monkeypatch, [])
    assert MODULE.resolve_move_limit_overrides(args) == {}
    assert args.rate_hz == pytest.approx(125.0)
    assert args.speed_l_acceleration == pytest.approx(1.2)
    assert args.motion_opt_in is False
    assert args.probe_only is False


def test_default_target_x_delta_is_the_checked_stable_value(monkeypatch: pytest.MonkeyPatch) -> None:
    """0.02 m is documented (sim characterization) to diverge; 0.04 m is the
    one point checked stable to a 10s hold -- the default must be 0.04, not
    the diverging 0.02 value."""
    args = _parse(monkeypatch, [])
    assert args.target_x_delta == pytest.approx(0.04)


def test_help_text_shows_the_safety_caveat() -> None:
    """A user reading --help must see the dx-dependent instability caveat --
    not just have it live in a code comment nobody reads. Runs the CLI as a
    real subprocess (still no robot/network I/O -- --help exits before any
    of that) so this checks exactly what a user actually sees."""
    import subprocess

    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "tools" / "ur5e_velocity_x_transport.py"), "--help"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0
    help_text = result.stdout
    assert "NOT" in help_text and "hardware" in help_text.lower()
    assert "0.02" in help_text
    assert "diverge" in help_text.lower()
    assert "0.04" in help_text


def test_rate_and_acceleration_flags_parse(monkeypatch: pytest.MonkeyPatch) -> None:
    args = _parse(monkeypatch, ["--rate-hz", "250.0", "--speed-l-acceleration", "0.6"])
    assert args.rate_hz == pytest.approx(250.0)
    assert args.speed_l_acceleration == pytest.approx(0.6)


def test_individual_guard_override_flags(monkeypatch: pytest.MonkeyPatch) -> None:
    args = _parse(
        monkeypatch,
        [
            "--max-tcp-accel-mps2", "1.5",
            "--max-tcp-speed-mps", "0.2",
            "--accel-gap-cycles", "5",
            "--speed-lowpass-alpha", "0.3",
            "--speed-limit-gap-cycles", "4",
            "--speed-limit-lowpass-alpha", "0.4",
            "--accel-max-consecutive-violations", "3",
            "--accel-hard-multiple", "6.0",
            "--speed-max-consecutive-violations", "2",
            "--speed-hard-multiple", "7.0",
        ],
    )
    overrides = MODULE.resolve_move_limit_overrides(args)
    assert overrides == {
        "max_tcp_accel_mps2": 1.5,
        "max_tcp_speed_mps": 0.2,
        "accel_gap_cycles": 5,
        "speed_lowpass_alpha": 0.3,
        "speed_limit_gap_cycles": 4,
        "speed_limit_lowpass_alpha": 0.4,
        "accel_max_consecutive_violations": 3,
        "accel_hard_multiple": 6.0,
        "speed_max_consecutive_violations": 2,
        "speed_hard_multiple": 7.0,
    }
    limits = CartesianMoveLimits.for_robot("10.0.0.5", **overrides)
    assert limits.max_tcp_accel_mps2 == pytest.approx(1.5)
    assert limits.speed_hard_multiple == pytest.approx(7.0)


def test_noise_robust_guards_preset_and_explicit_override_wins(monkeypatch: pytest.MonkeyPatch) -> None:
    args = _parse(monkeypatch, ["--noise-robust-guards", "--accel-gap-cycles", "12"])
    overrides = MODULE.resolve_move_limit_overrides(args)
    assert overrides["accel_gap_cycles"] == 12
    assert overrides["speed_lowpass_alpha"] == pytest.approx(NOISE_ROBUST_GUARD_OVERRIDES["speed_lowpass_alpha"])
    assert overrides["accel_hard_multiple"] == pytest.approx(NOISE_ROBUST_GUARD_OVERRIDES["accel_hard_multiple"])


def test_no_variable_tolerance_flags_exposed(monkeypatch: pytest.MonkeyPatch) -> None:
    """run_x_transport_velocity has no accel/speed_variable_tolerance kwargs
    (unlike direct_torque) -- confirm this CLI does not offer flags that
    would parse and then silently be dropped before the call site."""
    args = _parse(monkeypatch, [])
    assert not hasattr(args, "accel_variable_tolerance")
    assert not hasattr(args, "speed_variable_tolerance")


# ---------------------------------------------------------------------------
# main() dispatch -- no robot/network I/O
# ---------------------------------------------------------------------------


def test_main_refuses_motion_without_opt_in(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(sys, "argv", [
        "ur5e_velocity_x_transport.py",
        "--robot-ip", "10.0.0.5",
        "--output-dir", str(tmp_path),
    ])
    rc = MODULE.main()
    assert rc == 2


def test_main_dispatches_velocity_mode_with_resolved_kwargs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """End-to-end (still no robot/network I/O): run main() with
    --i-understand-this-moves-the-robot --yes and a fake run_x_transport /
    dashboard pipeline, and confirm control_mode='velocity' plus every
    resolved override/rate/accel kwarg reaches the call -- this is the check
    that would have caught a flag parsed-but-discarded bug."""
    captured: dict[str, object] = {}

    def _fake_run_x_transport(**kwargs):
        captured.update(kwargs)

        class _Result:
            ok = True
            summary = {"control_mode": "velocity"}
            trace_path = None

        return _Result()

    monkeypatch.setattr(MODULE, "run_x_transport", _fake_run_x_transport)
    monkeypatch.setattr(MODULE, "power_on_and_release", lambda ip: {})
    monkeypatch.setattr(MODULE, "query_remote_control", lambda ip: True)
    monkeypatch.setattr(sys, "argv", [
        "ur5e_velocity_x_transport.py",
        "--robot-ip", "10.0.0.5",
        "--i-understand-this-moves-the-robot",
        "--yes",
        "--target-x-delta", "0.03",
        "--move-duration", "1.5",
        "--duration", "4.0",
        "--rate-hz", "250.0",
        "--speed-l-acceleration", "0.8",
        "--noise-robust-guards",
        "--speed-max-consecutive-violations", "9",
        "--output-dir", str(tmp_path),
    ])
    rc = MODULE.main()
    assert rc == 0
    assert captured["control_mode"] == "velocity"
    assert captured["robot_ip"] == "10.0.0.5"
    assert captured["target_x_delta_m"] == pytest.approx(0.03)
    assert captured["move_duration_s"] == pytest.approx(1.5)
    assert captured["duration_s"] == pytest.approx(4.0)
    assert captured["rate_hz"] == pytest.approx(250.0)
    assert captured["speed_l_acceleration"] == pytest.approx(0.8)
    assert captured["motion_opt_in"] is True
    assert captured["speed_max_consecutive_violations_override"] == 9
    assert captured["accel_gap_cycles_override"] == NOISE_ROBUST_GUARD_OVERRIDES["accel_gap_cycles"]
    assert captured["speed_lowpass_alpha_override"] == pytest.approx(
        NOISE_ROBUST_GUARD_OVERRIDES["speed_lowpass_alpha"]
    )


def test_main_probe_only_never_requires_motion_opt_in(monkeypatch: pytest.MonkeyPatch) -> None:
    """--probe-only must connect(with_control=False), read state, and return
    0 without ever touching run_x_transport or requiring the motion opt-in."""

    class _FakeReceive:
        def getActualQ(self):
            return [0.1, -0.8, -1.2, -0.9, 0.0, 0.0]

        def getActualQd(self):
            return [0.0] * 6

        def getActualTCPPose(self):
            return [0.4, -0.2, 0.3, 0.0, 3.14, 0.0]

        def getTimestamp(self):
            return 1.0

        def getSafetyStatusBits(self):
            return 1

        def disconnect(self):
            pass

    class _FakeLink:
        def __init__(self, robot_ip, frequency_hz):
            self.robot_ip = robot_ip
            self.frequency_hz = frequency_hz
            self._receive = _FakeReceive()

        def connect(self, *, with_control: bool):
            assert with_control is False, "probe-only must never open the control interface"

        def read_state(self):
            from hardware.link import UR5eState

            return UR5eState(
                q=np.array(self._receive.getActualQ()),
                qd=np.array(self._receive.getActualQd()),
                tcp_pose=np.array(self._receive.getActualTCPPose()),
                robot_timestamp_s=1.0,
                host_stamp_ns=0,
                safety_status=1,
            )

        def disconnect(self):
            pass

    def _run_x_transport_should_not_be_called(**kwargs):
        raise AssertionError("run_x_transport must not be called in --probe-only mode")

    monkeypatch.setattr(MODULE, "UR5eLink", _FakeLink)
    monkeypatch.setattr(MODULE, "run_x_transport", _run_x_transport_should_not_be_called)
    monkeypatch.setattr(MODULE, "power_on_and_release", lambda ip: {})
    monkeypatch.setattr(MODULE, "query_remote_control", lambda ip: True)
    monkeypatch.setattr(sys, "argv", [
        "ur5e_velocity_x_transport.py",
        "--robot-ip", "10.0.0.5",
        "--probe-only",
    ])
    rc = MODULE.main()
    assert rc == 0
