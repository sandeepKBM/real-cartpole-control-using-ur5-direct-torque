"""Generate URScript OSC inner-loop transport."""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

from _bootstrap import ensure_repo_root

ensure_repo_root()

from hardware.dashboard import power_on_and_release, query_remote_control  # noqa: E402
from hardware.link import RTDELinkError  # noqa: E402
from hardware.poses import HEIGHT_ALPHA_0_5_Q  # noqa: E402
from hardware.safety import NOISE_ROBUST_GUARD_OVERRIDES  # noqa: E402
from hardware.urscript_gen import DEFAULT_CONFIG  # noqa: E402
from hardware.urscript_transport import run_urscript_x_transport  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]


def resolve_move_limit_overrides(args: argparse.Namespace) -> dict[str, float | int]:
    """Combine --noise-robust-guards' preset (if set) with any explicit
    individual override flags, explicit always winning for a given field.

    This is a deliberate duplicate of
    tools/ur5e_direct_torque_x_transport.py::resolve_move_limit_overrides --
    kept as a small standalone copy rather than importing across CLI tools
    (each tool script is meant to run standalone; see that module's docstring
    for the combination-rule rationale and NOISE_ROBUST_GUARD_OVERRIDES in
    hardware/safety.py for the preset evidence). Keep the two in sync if the
    override field set ever changes.
    """
    overrides: dict[str, float | int] = {}
    if args.noise_robust_guards:
        overrides.update(NOISE_ROBUST_GUARD_OVERRIDES)
    if args.max_tcp_accel_mps2 is not None:
        overrides["max_tcp_accel_mps2"] = float(args.max_tcp_accel_mps2)
    if args.accel_gap_cycles is not None:
        overrides["accel_gap_cycles"] = int(args.accel_gap_cycles)
    if args.speed_lowpass_alpha is not None:
        overrides["speed_lowpass_alpha"] = float(args.speed_lowpass_alpha)
    if args.speed_limit_gap_cycles is not None:
        overrides["speed_limit_gap_cycles"] = int(args.speed_limit_gap_cycles)
    if args.speed_limit_lowpass_alpha is not None:
        overrides["speed_limit_lowpass_alpha"] = float(args.speed_limit_lowpass_alpha)
    if args.accel_max_consecutive_violations is not None:
        overrides["accel_max_consecutive_violations"] = int(args.accel_max_consecutive_violations)
    if args.accel_hard_multiple is not None:
        overrides["accel_hard_multiple"] = float(args.accel_hard_multiple)
    if args.speed_max_consecutive_violations is not None:
        overrides["speed_max_consecutive_violations"] = int(args.speed_max_consecutive_violations)
    if args.speed_hard_multiple is not None:
        overrides["speed_hard_multiple"] = float(args.speed_hard_multiple)
    return overrides


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Run tuned OSC X transport with a 500 Hz URScript inner loop on PolyScope "
            "(direct_torque V2, on-robot Jacobian/M). Python only supervises safety."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--robot-ip", required=True)
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--target-x-delta", type=float, default=0.02)
    p.add_argument("--move-duration", type=float, default=1.0)
    p.add_argument("--duration", type=float, default=3.0)
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--skip-joint-move", action="store_true")
    p.add_argument("--skip-dashboard-power-on", action="store_true")
    p.add_argument("--no-lambda", action="store_true", help="Kinematic J.T wrench only (faster, less matched to sim).")
    p.add_argument("--generate-only", action="store_true", help="Write generated .script and exit (no robot).")
    p.add_argument("--i-understand-this-moves-the-robot", dest="motion_opt_in", action="store_true")
    p.add_argument("--yes", action="store_true")
    p.add_argument(
        "--max-tcp-accel-mps2",
        type=float,
        default=None,
        help=(
            "All three control modes. Explicit, opt-in override of "
            "CartesianMoveLimits.max_tcp_accel_mps2 (class default 0.5 m/s^2). Added "
            "2026-07-28: on real hardware the naive one-step-finite-difference accel "
            "estimate amplifies raw RTDE position noise (~1/dt^2 -- ~15,600x at "
            "position mode's 125Hz, ~250,000x at direct_torque's 500Hz) during a "
            "min-jerk move's near-zero-velocity onset, tripping spuriously (observed "
            "0.72 and 0.90 m/s^2 in position mode across two trials, trip point "
            "varying step 1 vs step 6, every other metric -- drift/orientation/qd -- "
            "negligible). Does not fix the underlying numerical issue; a deliberate, "
            "visible override for continuing real-hardware testing, not a silent "
            "threshold change."
        ),
    )
    p.add_argument(
        "--accel-gap-cycles",
        type=int,
        default=None,
        help=(
            "All three control modes. Explicit override of "
            "CartesianMoveLimits.accel_gap_cycles (class default 1 = original "
            "single-cycle behavior). Added 2026-07-28 after "
            "tools/analyze_state_noise_capture.py measured the accel estimate's own "
            "noise floor from a real stationary RTDE capture: median 1.74 m/s^2 at "
            "gap=1, already ~3.5x the 0.5 default. Using position from N cycles back "
            "(instead of 1) to form each speed sample fed into the accel estimate "
            "cuts noise sensitivity substantially without losing detection of a real, "
            "sustained fast motion -- see CartesianMoveLimits' docstring for the "
            "mechanism. Combine with --speed-lowpass-alpha and re-run "
            "analyze_state_noise_capture.py (it accepts the same two flags) against a "
            "real stationary capture before picking --max-tcp-accel-mps2."
        ),
    )
    p.add_argument(
        "--speed-lowpass-alpha",
        type=float,
        default=None,
        help=(
            "All three control modes. Explicit override of "
            "CartesianMoveLimits.speed_lowpass_alpha (class default 1.0 = no "
            "filtering). EMA smoothing factor in (0, 1] applied to the gap-windowed "
            "speed sample before differencing for the accel estimate -- smaller = "
            "more smoothing. See --accel-gap-cycles."
        ),
    )
    p.add_argument(
        "--speed-limit-gap-cycles",
        type=int,
        default=None,
        help=(
            "All three control modes. Explicit override of "
            "CartesianMoveLimits.speed_limit_gap_cycles (class default 1 = original "
            "single-cycle behavior). See --speed-limit-lowpass-alpha and "
            "tools/ur5e_direct_torque_x_transport.py's identical flag for the "
            "real-hardware finding motivating this (2026-08-01)."
        ),
    )
    p.add_argument(
        "--speed-limit-lowpass-alpha",
        type=float,
        default=None,
        help=(
            "All three control modes. Explicit override of "
            "CartesianMoveLimits.speed_limit_lowpass_alpha (class default 1.0 = no "
            "filtering). EMA smoothing factor in (0, 1] applied to the gap-windowed "
            "speed sample used for the speed-LIMIT decision itself -- smaller = more "
            "smoothing. See --speed-limit-gap-cycles."
        ),
    )
    p.add_argument(
        "--accel-max-consecutive-violations",
        type=int,
        default=None,
        help=(
            "All three control modes. Explicit override of "
            "CartesianMoveLimits.accel_max_consecutive_violations (class default 1 "
            "= original instant-trip behavior). Added 2026-07-30 (commit 8eccd1d) -- "
            "DeadlineMonitor-style graduated tolerance: this many consecutive "
            "over-threshold cycles before tripping, tolerating an isolated noise "
            "spike while still catching a sustained real trend. See "
            "--noise-robust-guards below: real-hardware-noise-replay backtest "
            "evidence (docs/status/safety_envelope_backtest_2026-07-30.md) found "
            "this field ALONE, at its own validated value, does not eliminate "
            "spurious trips -- it must be combined with --accel-gap-cycles/"
            "--speed-lowpass-alpha to actually work."
        ),
    )
    p.add_argument(
        "--accel-hard-multiple",
        type=float,
        default=None,
        help=(
            "All three control modes. Explicit override of "
            "CartesianMoveLimits.accel_hard_multiple (class default 5.0). A single "
            "cycle at or above max_tcp_accel_mps2 * this multiple trips immediately "
            "regardless of --accel-max-consecutive-violations, catching a genuine "
            "one-shot catastrophic event. See --accel-max-consecutive-violations."
        ),
    )
    p.add_argument(
        "--speed-max-consecutive-violations",
        type=int,
        default=None,
        help=(
            "All three control modes. Explicit override of "
            "CartesianMoveLimits.speed_max_consecutive_violations (class default 1). "
            "Same graduated-tolerance mechanism as --accel-max-consecutive-violations, "
            "applied to the TCP speed check instead of acceleration."
        ),
    )
    p.add_argument(
        "--speed-hard-multiple",
        type=float,
        default=None,
        help=(
            "All three control modes. Explicit override of "
            "CartesianMoveLimits.speed_hard_multiple (class default 5.0). Same "
            "immediate-trip safety valve as --accel-hard-multiple, applied to the "
            "TCP speed check."
        ),
    )
    p.add_argument(
        "--noise-robust-guards",
        action="store_true",
        help=(
            "All three control modes. Convenience flag applying the full "
            "validated 6-parameter combination found to actually close the "
            "real-hardware noise-driven-spurious-trip gap (2026-07-30 backtest, "
            "docs/status/safety_envelope_backtest_2026-07-30.md section 9, "
            "experiments/safety-envelope-study branch): the graduated-tolerance "
            "fields ALONE still spuriously tripped 30/30 replayed seeds at real "
            "measured RTDE noise magnitudes; only pairing them with "
            "accel_gap_cycles/speed_lowpass_alpha filtering closed it (0/30 "
            "spurious), while still catching the real genuine-catch case "
            "(-0.20m/1.0s move, theoretical peak accel 1.1547 m/s^2). Preset "
            "values: accel_max_consecutive_violations=3, accel_hard_multiple=5.0, "
            "speed_max_consecutive_violations=3, speed_hard_multiple=5.0, "
            "accel_gap_cycles=5, speed_lowpass_alpha=0.2 (see "
            "hardware.safety.NOISE_ROBUST_GUARD_OVERRIDES). The preset is applied "
            "first; any individual override flag above still wins for that "
            "specific field."
        ),
    )
    return p.parse_args()


def _default_output_dir() -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return REPO_ROOT / "outputs" / "hardware_urscript" / f"x_transport_{stamp}"


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir or _default_output_dir()

    if args.generate_only:
        from hardware.urscript_gen import load_params_from_yaml, write_generated_script

        params = load_params_from_yaml(
            args.config,
            target_x_delta_m=float(args.target_x_delta),
            move_duration_s=float(args.move_duration),
            duration_s=float(args.duration),
            use_lambda=not args.no_lambda,
        )
        path = write_generated_script(params, output_dir / "x_axis_osc_inner.script")
        print(f"Wrote {path}")
        return 0

    if not args.motion_opt_in:
        print("Refusing motion without --i-understand-this-moves-the-robot", file=sys.stderr)
        return 2
    if not args.yes:
        typed = input("Type MOVE to run URScript inner-loop transport: ").strip()
        if typed != "MOVE":
            print("Aborted.", file=sys.stderr)
            return 2

    if not args.skip_dashboard_power_on:
        for cmd, resp in power_on_and_release(args.robot_ip).items():
            print(f"dashboard {cmd}: {resp}")
        time.sleep(2.0)

    if not query_remote_control(args.robot_ip):
        print("Remote control is OFF — see docs/hardware/URSIM_REMOTE_CONTROL.md", file=sys.stderr)
        return 2

    move_limit_overrides = resolve_move_limit_overrides(args)
    try:
        result = run_urscript_x_transport(
            robot_ip=args.robot_ip,
            config_path=args.config,
            target_x_delta_m=float(args.target_x_delta),
            move_duration_s=float(args.move_duration),
            duration_s=float(args.duration),
            output_dir=output_dir,
            motion_opt_in=True,
            skip_joint_move=bool(args.skip_joint_move),
            use_lambda=not args.no_lambda,
            max_tcp_accel_mps2_override=move_limit_overrides.get("max_tcp_accel_mps2"),
            accel_gap_cycles_override=move_limit_overrides.get("accel_gap_cycles"),
            speed_lowpass_alpha_override=move_limit_overrides.get("speed_lowpass_alpha"),
            speed_limit_gap_cycles_override=move_limit_overrides.get("speed_limit_gap_cycles"),
            speed_limit_lowpass_alpha_override=move_limit_overrides.get("speed_limit_lowpass_alpha"),
            accel_max_consecutive_violations_override=move_limit_overrides.get(
                "accel_max_consecutive_violations"
            ),
            accel_hard_multiple_override=move_limit_overrides.get("accel_hard_multiple"),
            speed_max_consecutive_violations_override=move_limit_overrides.get(
                "speed_max_consecutive_violations"
            ),
            speed_hard_multiple_override=move_limit_overrides.get("speed_hard_multiple"),
        )
    except RTDELinkError as exc:
        print(f"RTDE failed: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(result.summary, indent=2))
    if result.script_path is not None:
        print(f"script: {result.script_path}")
    return 0 if result.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
