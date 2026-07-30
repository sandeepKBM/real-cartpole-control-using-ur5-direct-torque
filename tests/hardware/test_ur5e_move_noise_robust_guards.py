"""CLI-parsing tests for tools/ur5e_move.py's graduated-tolerance override
flags (--accel-max-consecutive-violations/--accel-hard-multiple/
--speed-max-consecutive-violations/--speed-hard-multiple) and the
--noise-robust-guards convenience preset.

Real-hardware evidence behind the preset values (commit message + AGENTS.md
§4 pointer): docs/status/safety_envelope_backtest_2026-07-30.md section 9,
experiments/safety-envelope-study branch -- the graduated-tolerance fields
ALONE, at their own validated defaults, still spuriously tripped 30/30
replayed real-RTDE-noise seeds; only combining them with the older, separate
accel_gap_cycles/speed_lowpass_alpha filtering closed the gap (0/30 spurious)
while still catching the real genuine-catch case (-0.20m/1.0s move,
theoretical peak accel 1.1547 m/s^2).

No robot connection or motion is exercised here -- only argparse parsing and
the pure dict-building helper that feeds CartesianMoveLimits.for_robot().
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
    spec = importlib.util.spec_from_file_location("ur5e_move", REPO_ROOT / "tools" / "ur5e_move.py")
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(REPO_ROOT / "tools"))
    try:
        spec.loader.exec_module(module)  # type: ignore[union-attr]
    finally:
        sys.path.pop(0)
    return module


MODULE = _load_module()

_BASE_ARGV = ["--robot-ip", "10.0.0.5", "--axis", "x", "--direction", "left", "--distance-m", "0.1"]


def _parse(monkeypatch: pytest.MonkeyPatch, extra_argv: list[str]):
    monkeypatch.setattr(sys, "argv", ["ur5e_move.py", *_BASE_ARGV, *extra_argv])
    return MODULE.parse_args()


def test_no_flags_produces_no_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    args = _parse(monkeypatch, [])
    assert MODULE.build_move_limit_overrides(args) == {}


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
    overrides = MODULE.build_move_limit_overrides(args)
    assert overrides == {
        "accel_max_consecutive_violations": 4,
        "accel_hard_multiple": 6.5,
        "speed_max_consecutive_violations": 2,
        "speed_hard_multiple": 7.5,
    }
    # Feed straight into the real class to confirm the values actually land
    # on CartesianMoveLimits, not just in an intermediate dict.
    limits = CartesianMoveLimits.for_robot("10.0.0.5", **overrides)
    assert limits.accel_max_consecutive_violations == 4
    assert limits.accel_hard_multiple == pytest.approx(6.5)
    assert limits.speed_max_consecutive_violations == 2
    assert limits.speed_hard_multiple == pytest.approx(7.5)


def test_noise_robust_guards_applies_the_validated_preset(monkeypatch: pytest.MonkeyPatch) -> None:
    args = _parse(monkeypatch, ["--noise-robust-guards"])
    overrides = MODULE.build_move_limit_overrides(args)
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
    assert limits.accel_max_consecutive_violations == 3
    assert limits.speed_max_consecutive_violations == 3


def test_explicit_override_wins_over_noise_robust_guards_preset(monkeypatch: pytest.MonkeyPatch) -> None:
    """The convenience preset must not silently clobber a deliberate
    individual choice -- apply the preset first, then let an explicit flag
    win for that one field, leaving every other preset field untouched."""
    args = _parse(
        monkeypatch,
        ["--noise-robust-guards", "--accel-gap-cycles", "12", "--speed-hard-multiple", "9.0"],
    )
    overrides = MODULE.build_move_limit_overrides(args)
    # The two explicitly-passed fields must reflect the CLI value, not the preset's.
    assert overrides["accel_gap_cycles"] == 12
    assert overrides["speed_hard_multiple"] == pytest.approx(9.0)
    # Every other preset field must still be exactly the preset's value.
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
    assert args.accel_max_consecutive_violations is None
    assert args.accel_hard_multiple is None
    assert args.speed_max_consecutive_violations is None
    assert args.speed_hard_multiple is None
    assert args.noise_robust_guards is False
