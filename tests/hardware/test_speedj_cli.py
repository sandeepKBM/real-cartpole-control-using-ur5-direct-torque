"""CLI-parsing and dispatch tests for tools/ur5e_speedj_x_transport.py.

Mirrors tests/hardware/test_velocity_cli.py's structure and mocking style --
no robot connection or motion is ever exercised. Per AGENTS.md's "verify the
effect, not the invocation" rule: this drives parse_args()/main() directly
and asserts the kwargs that reach run_x_transport, catching a flag that
parses but is silently dropped before the call site.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from hardware.joint_velocity_transport import (  # noqa: E402
    DEFAULT_DAMPING_LAMBDA_MAX,
    DEFAULT_DAMPING_SIGMA0,
    DEFAULT_JOINT_VELOCITY_CLAMP_RADPS,
)
from hardware.safety import CartesianMoveLimits, NOISE_ROBUST_GUARD_OVERRIDES  # noqa: E402


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "ur5e_speedj_x_transport", REPO_ROOT / "tools" / "ur5e_speedj_x_transport.py"
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
    monkeypatch.setattr(sys, "argv", ["ur5e_speedj_x_transport.py", *_BASE_ARGV, *extra_argv])
    return MODULE.parse_args()


# ---------------------------------------------------------------------------
# Argument parsing / defaults
# ---------------------------------------------------------------------------


def test_default_config_is_the_dedicated_speedj_config() -> None:
    """Must NOT be config/ur5e_velocity_control.yaml -- that config's
    reduced_task_dims: true causes DLS to double-resolve (see
    tests/hardware/test_joint_velocity_resolution_fix.py and
    config/ur5e_speedj_joint_velocity.yaml's own header)."""
    assert MODULE.DEFAULT_CONFIG == REPO_ROOT / "config" / "ur5e_speedj_joint_velocity.yaml"
    assert MODULE.DEFAULT_CONFIG.exists()
    assert MODULE.DEFAULT_CONFIG != REPO_ROOT / "config" / "ur5e_velocity_control.yaml"


def test_no_flags_produces_no_overrides_and_module_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    args = _parse(monkeypatch, [])
    assert MODULE.resolve_move_limit_overrides(args) == {}
    assert args.rate_hz == pytest.approx(125.0)
    assert args.speed_j_acceleration == pytest.approx(1.2)
    assert args.joint_velocity_clamp == pytest.approx(DEFAULT_JOINT_VELOCITY_CLAMP_RADPS)
    assert args.damping_lambda_max == pytest.approx(DEFAULT_DAMPING_LAMBDA_MAX)
    assert args.damping_sigma0 == pytest.approx(DEFAULT_DAMPING_SIGMA0)
    assert args.motion_opt_in is False
    assert args.probe_only is False


def test_help_text_shows_the_safety_caveat() -> None:
    """A user reading --help must see the "brand new, zero validation"
    caveat -- runs the CLI as a real subprocess (still no robot/network I/O
    -- --help exits before any of that)."""
    import subprocess

    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "tools" / "ur5e_speedj_x_transport.py"), "--help"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0
    help_text = result.stdout
    assert "BRAND NEW" in help_text
    assert "zero" in help_text.lower() or "ZERO" in help_text


def test_damping_and_clamp_flags_parse(monkeypatch: pytest.MonkeyPatch) -> None:
    args = _parse(
        monkeypatch,
        [
            "--damping-lambda-max", "0.1",
            "--damping-sigma0", "0.08",
            "--joint-velocity-clamp", "0.5",
        ],
    )
    assert args.damping_lambda_max == pytest.approx(0.1)
    assert args.damping_sigma0 == pytest.approx(0.08)
    assert args.joint_velocity_clamp == pytest.approx(0.5)


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


# ---------------------------------------------------------------------------
# main() dispatch -- no robot/network I/O
# ---------------------------------------------------------------------------


def test_main_refuses_motion_without_opt_in(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(sys, "argv", [
        "ur5e_speedj_x_transport.py",
        "--robot-ip", "10.0.0.5",
        "--output-dir", str(tmp_path),
    ])
    rc = MODULE.main()
    assert rc == 2


def test_main_dispatches_joint_velocity_mode_with_resolved_kwargs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """End-to-end (still no robot/network I/O): confirm control_mode=
    'joint_velocity' plus every resolved override/rate/damping/clamp kwarg
    reaches the call -- catches a flag parsed-but-discarded bug."""
    captured: dict[str, object] = {}

    def _fake_run_x_transport(**kwargs):
        captured.update(kwargs)

        class _Result:
            ok = True
            summary = {"control_mode": "joint_velocity"}
            trace_path = None

        return _Result()

    monkeypatch.setattr(MODULE, "run_x_transport", _fake_run_x_transport)
    monkeypatch.setattr(MODULE, "power_on_and_release", lambda ip: {})
    monkeypatch.setattr(MODULE, "query_remote_control", lambda ip: True)
    monkeypatch.setattr(sys, "argv", [
        "ur5e_speedj_x_transport.py",
        "--robot-ip", "10.0.0.5",
        "--i-understand-this-moves-the-robot",
        "--yes",
        "--target-x-delta", "0.03",
        "--move-duration", "1.5",
        "--duration", "4.0",
        "--rate-hz", "250.0",
        "--speed-j-acceleration", "0.8",
        "--damping-lambda-max", "0.12",
        "--damping-sigma0", "0.09",
        "--joint-velocity-clamp", "0.4",
        "--noise-robust-guards",
        "--speed-max-consecutive-violations", "9",
        "--output-dir", str(tmp_path),
    ])
    rc = MODULE.main()
    assert rc == 0
    assert captured["control_mode"] == "joint_velocity"
    assert captured["robot_ip"] == "10.0.0.5"
    assert captured["target_x_delta_m"] == pytest.approx(0.03)
    assert captured["move_duration_s"] == pytest.approx(1.5)
    assert captured["duration_s"] == pytest.approx(4.0)
    assert captured["rate_hz"] == pytest.approx(250.0)
    assert captured["speed_j_acceleration"] == pytest.approx(0.8)
    assert captured["damping_lambda_max"] == pytest.approx(0.12)
    assert captured["damping_sigma0"] == pytest.approx(0.09)
    assert captured["joint_velocity_clamp_radps"] == pytest.approx(0.4)
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
        "ur5e_speedj_x_transport.py",
        "--robot-ip", "10.0.0.5",
        "--probe-only",
    ])
    rc = MODULE.main()
    assert rc == 0
