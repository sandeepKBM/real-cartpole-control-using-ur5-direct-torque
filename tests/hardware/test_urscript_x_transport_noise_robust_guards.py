"""CLI-parsing tests for tools/ur5e_urscript_x_transport.py's
CartesianMoveMonitor guard-override flags (--max-tcp-accel-mps2/
--accel-gap-cycles/--speed-lowpass-alpha/--accel-max-consecutive-violations/
--accel-hard-multiple/--speed-max-consecutive-violations/--speed-hard-multiple)
and the --noise-robust-guards convenience preset.

Mirrors tests/hardware/test_direct_torque_x_transport_noise_robust_guards.py --
same preset, same "explicit override wins" contract, same flag names, ported
to the URScript CLI (tools/ur5e_urscript_x_transport.py::
resolve_move_limit_overrides, a deliberate standalone duplicate of the
direct_torque tool's function of the same name -- see that function's
docstring for why it isn't imported across tools).

Evidence for the preset values: docs/status/safety_envelope_backtest_2026-07-30.md
section 9 (experiments/safety-envelope-study branch) -- graduated tolerance
alone still spuriously tripped 30/30 replayed real-RTDE-noise seeds; only
pairing it with accel_gap_cycles/speed_lowpass_alpha closed the gap (0/30
spurious) while still catching the real genuine-catch case.

No robot connection or motion is exercised -- only argparse parsing and the
pure dict-building helper.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from hardware.safety import CartesianMoveLimits, NOISE_ROBUST_GUARD_OVERRIDES  # noqa: E402


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "ur5e_urscript_x_transport", REPO_ROOT / "tools" / "ur5e_urscript_x_transport.py"
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
    monkeypatch.setattr(sys, "argv", ["ur5e_urscript_x_transport.py", *_BASE_ARGV, *extra_argv])
    return MODULE.parse_args()


def test_no_flags_produces_no_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    args = _parse(monkeypatch, [])
    assert MODULE.resolve_move_limit_overrides(args) == {}


def test_individual_graduated_tolerance_flags(monkeypatch: pytest.MonkeyPatch) -> None:
    args = _parse(
        monkeypatch,
        [
            "--accel-max-consecutive-violations", "4",
            "--accel-hard-multiple", "6.5",
            "--speed-max-consecutive-violations", "2",
            "--speed-hard-multiple", "7.5",
        ],
    )
    overrides = MODULE.resolve_move_limit_overrides(args)
    assert overrides == {
        "accel_max_consecutive_violations": 4,
        "accel_hard_multiple": 6.5,
        "speed_max_consecutive_violations": 2,
        "speed_hard_multiple": 7.5,
    }
    limits = CartesianMoveLimits.for_robot("10.0.0.5", **overrides)
    assert limits.accel_max_consecutive_violations == 4
    assert limits.accel_hard_multiple == pytest.approx(6.5)
    assert limits.speed_max_consecutive_violations == 2
    assert limits.speed_hard_multiple == pytest.approx(7.5)


def test_noise_robust_guards_applies_the_validated_preset(monkeypatch: pytest.MonkeyPatch) -> None:
    args = _parse(monkeypatch, ["--noise-robust-guards"])
    overrides = MODULE.resolve_move_limit_overrides(args)
    assert overrides == NOISE_ROBUST_GUARD_OVERRIDES
    assert overrides == {
        "accel_max_consecutive_violations": 3,
        "accel_hard_multiple": 5.0,
        "speed_max_consecutive_violations": 3,
        "speed_hard_multiple": 5.0,
        "accel_gap_cycles": 5,
        "speed_lowpass_alpha": 0.2,
    }
    limits = CartesianMoveLimits.for_robot("10.0.0.5", **overrides)
    assert limits.accel_gap_cycles == 5
    assert limits.speed_lowpass_alpha == pytest.approx(0.2)


def test_explicit_override_wins_over_noise_robust_guards_preset(monkeypatch: pytest.MonkeyPatch) -> None:
    """Individual override flags must win over the --noise-robust-guards
    preset for their own field, per-field, while every other preset field
    stays intact -- the preset must not be silently clobbered wholesale nor
    silently clobber a deliberate individual choice."""
    args = _parse(
        monkeypatch,
        ["--noise-robust-guards", "--accel-gap-cycles", "12", "--speed-hard-multiple", "9.0"],
    )
    overrides = MODULE.resolve_move_limit_overrides(args)
    assert overrides["accel_gap_cycles"] == 12
    assert overrides["speed_hard_multiple"] == pytest.approx(9.0)
    assert overrides["accel_max_consecutive_violations"] == NOISE_ROBUST_GUARD_OVERRIDES[
        "accel_max_consecutive_violations"
    ]
    assert overrides["accel_hard_multiple"] == pytest.approx(NOISE_ROBUST_GUARD_OVERRIDES["accel_hard_multiple"])
    assert overrides["speed_max_consecutive_violations"] == NOISE_ROBUST_GUARD_OVERRIDES[
        "speed_max_consecutive_violations"
    ]
    assert overrides["speed_lowpass_alpha"] == pytest.approx(NOISE_ROBUST_GUARD_OVERRIDES["speed_lowpass_alpha"])
    limits = CartesianMoveLimits.for_robot("10.0.0.5", **overrides)
    assert limits.accel_gap_cycles == 12
    assert limits.speed_hard_multiple == pytest.approx(9.0)
    assert limits.accel_max_consecutive_violations == 3


def test_all_new_flags_default_to_none(monkeypatch: pytest.MonkeyPatch) -> None:
    args = _parse(monkeypatch, [])
    assert args.max_tcp_accel_mps2 is None
    assert args.accel_gap_cycles is None
    assert args.speed_lowpass_alpha is None
    assert args.accel_max_consecutive_violations is None
    assert args.accel_hard_multiple is None
    assert args.speed_max_consecutive_violations is None
    assert args.speed_hard_multiple is None
    assert args.noise_robust_guards is False


def test_run_urscript_x_transport_call_receives_resolved_overrides(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """End-to-end check (still no robot/network I/O): monkeypatch
    hardware.urscript_transport.run_urscript_x_transport as imported into the
    tool module, run main() with a fake robot pipeline, and confirm the
    *_override kwargs it receives match resolve_move_limit_overrides()."""
    args = _parse(
        monkeypatch,
        ["--noise-robust-guards", "--speed-max-consecutive-violations", "9"],
    )
    overrides = MODULE.resolve_move_limit_overrides(args)

    captured: dict[str, object] = {}

    def _fake_run_urscript_x_transport(**kwargs):
        captured.update(kwargs)

        class _Result:
            ok = True
            reason = ""
            summary = {}
            script_path = None

        return _Result()

    monkeypatch.setattr(MODULE, "run_urscript_x_transport", _fake_run_urscript_x_transport)
    monkeypatch.setattr(MODULE, "power_on_and_release", lambda ip: {})
    monkeypatch.setattr(MODULE, "query_remote_control", lambda ip: True)
    monkeypatch.setattr(sys, "argv", [
        "ur5e_urscript_x_transport.py",
        "--robot-ip", "10.0.0.5",
        "--i-understand-this-moves-the-robot",
        "--yes",
        "--noise-robust-guards",
        "--speed-max-consecutive-violations", "9",
        "--output-dir", str(tmp_path),
    ])
    rc = MODULE.main()
    assert rc == 0
    assert captured["speed_max_consecutive_violations_override"] == 9
    assert captured["accel_max_consecutive_violations_override"] == overrides["accel_max_consecutive_violations"]
    assert captured["accel_hard_multiple_override"] == overrides["accel_hard_multiple"]
    assert captured["accel_gap_cycles_override"] == overrides["accel_gap_cycles"]
    assert captured["speed_lowpass_alpha_override"] == overrides["speed_lowpass_alpha"]
