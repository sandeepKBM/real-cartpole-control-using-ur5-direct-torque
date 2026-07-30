#!/usr/bin/env python3
"""Move a real UR5e a bounded distance along one Cartesian axis.

Uses UR's native position-space servoL streaming (the robot firmware does
its own inverse kinematics) with a live safety monitor checking drift,
orientation, velocity, and tracking error every single control cycle --
see hardware/motion.py and hardware/safety.py::CartesianMoveMonitor.

This does NOT run the simulation's torque-based Cartesian impedance
controller -- it's a deliberately simple, position-only first-motion script.
For live torque control see tools/ur5e_direct_torque_x_transport.py
(--control-mode direct_torque or urscript); see AGENTS.md sec 4 for both.

--axis has no default -- you must know (or determine on-site with a small
test move) which axis and sign corresponds to physical left/right for this
robot's particular mounting. This script's own convention: --direction left
is +distance along --axis, --direction right is -distance; if that's
backwards for your setup, the small test move in step 2 below will show it
immediately.

Recommended sequence, smallest first:
  1. python tools/ur5e_move.py --robot-ip <IP> --axis y --direction left \\
       --distance-m 0.15 --dry-run
  2. python tools/ur5e_move.py --robot-ip <IP> --axis y --direction left \\
       --distance-m 0.02 --i-understand-this-moves-the-robot
  3. python tools/ur5e_move.py --robot-ip <IP> --axis y --direction left \\
       --distance-m 0.15 --i-understand-this-moves-the-robot
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

from _bootstrap import ensure_repo_root

ensure_repo_root()

from hardware.link import UR5eLink  # noqa: E402
from hardware.motion import (  # noqa: E402
    move_cartesian_bounded,
    peak_acceleration_mps2,
    peak_velocity_mps,
    plan_waypoints,
)
from hardware.safety import (  # noqa: E402
    NOISE_ROBUST_GUARD_OVERRIDES,
    CartesianMoveLimits,
    CartesianMoveMonitor,
    EStopLatch,
    UR5eSafetyLimits,
)

_AXIS_INDEX = {"x": 0, "y": 1, "z": 2}
REPO_ROOT = Path(__file__).resolve().parents[1]


def signed_distance(distance_m: float, direction: str) -> float:
    distance_m = abs(float(distance_m))
    if direction == "left":
        return distance_m
    if direction == "right":
        return -distance_m
    raise ValueError(f"direction must be 'left' or 'right'; got {direction!r}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--robot-ip", required=True, help="UR5e robot IP address. No default -- always explicit.")
    p.add_argument("--axis", required=True, choices=("x", "y", "z"), help="No default -- you must know your mounting.")
    p.add_argument("--direction", required=True, choices=("left", "right"))
    p.add_argument("--distance-m", required=True, type=float, help="Positive distance in meters, e.g. 0.15.")
    p.add_argument("--duration-s", type=float, default=6.0)
    p.add_argument("--rate-hz", type=float, default=125.0)
    p.add_argument(
        "--max-tcp-speed-mps",
        type=float,
        default=None,
        help="Override CartesianMoveMonitor max_tcp_speed_mps for this run.",
    )
    p.add_argument(
        "--max-tcp-accel-mps2",
        type=float,
        default=None,
        help="Override CartesianMoveMonitor max_tcp_accel_mps2 for this run.",
    )
    p.add_argument(
        "--accel-gap-cycles",
        type=int,
        default=None,
        help="Override acceleration estimator gap window. Larger values reduce TCP accel noise.",
    )
    p.add_argument(
        "--speed-lowpass-alpha",
        type=float,
        default=None,
        help="Override acceleration estimator speed EMA alpha in (0,1]. Smaller is smoother.",
    )
    p.add_argument(
        "--accel-max-consecutive-violations",
        type=int,
        default=None,
        help=(
            "Override CartesianMoveLimits.accel_max_consecutive_violations (class "
            "default 1 = original instant-trip behavior). Added 2026-07-30 (commit "
            "8eccd1d) -- DeadlineMonitor-style graduated tolerance: this many "
            "consecutive over-threshold cycles before tripping. See "
            "--noise-robust-guards below: real-hardware-noise-replay backtest "
            "evidence found this field ALONE does not eliminate spurious trips."
        ),
    )
    p.add_argument(
        "--accel-hard-multiple",
        type=float,
        default=None,
        help=(
            "Override CartesianMoveLimits.accel_hard_multiple (class default 5.0). "
            "A single cycle >= max_tcp_accel_mps2 * this multiple trips immediately "
            "regardless of --accel-max-consecutive-violations."
        ),
    )
    p.add_argument(
        "--speed-max-consecutive-violations",
        type=int,
        default=None,
        help=(
            "Override CartesianMoveLimits.speed_max_consecutive_violations (class "
            "default 1). Same graduated-tolerance mechanism as "
            "--accel-max-consecutive-violations, applied to the TCP speed check."
        ),
    )
    p.add_argument(
        "--speed-hard-multiple",
        type=float,
        default=None,
        help=(
            "Override CartesianMoveLimits.speed_hard_multiple (class default 5.0). "
            "Same immediate-trip safety valve as --accel-hard-multiple, applied to "
            "the TCP speed check."
        ),
    )
    p.add_argument(
        "--noise-robust-guards",
        action="store_true",
        help=(
            "Convenience flag applying the full validated 6-parameter combination "
            "found to actually close the real-hardware noise-driven-spurious-trip "
            "gap (2026-07-30 backtest, "
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
    p.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for servoL motion trace.jsonl and summary.json. Default: timestamped under outputs/hardware_transport/.",
    )
    p.add_argument(
        "--max-shoulder-pan-delta-rad",
        type=float,
        default=None,
        help="Abort if shoulder_pan changes by more than this amount during the servoL move.",
    )
    p.add_argument(
        "--max-stale-state-cycles",
        type=int,
        default=5,
        help=(
            "Abort after this many consecutive repeated robot timestamps while the host loop advances. "
            "Default preserves the normal hardware guard."
        ),
    )
    p.add_argument("--dry-run", action="store_true", help="Plan the move and print it. No connection at all.")
    p.add_argument(
        "--i-understand-this-moves-the-robot",
        dest="motion_opt_in",
        action="store_true",
        help="Required for any real (non-dry-run) motion.",
    )
    p.add_argument(
        "--yes",
        action="store_true",
        help="Skip the interactive typed-MOVE confirmation (for scripted use; still requires the opt-in flag above).",
    )
    return p.parse_args()


def build_move_limit_overrides(args: argparse.Namespace) -> dict[str, float | int]:
    """Combine --noise-robust-guards' preset (if set) with any explicit
    individual override flags, explicit always winning for a given field.

    The preset is applied first via dict.update(), then each explicit flag
    (if not None) overwrites its own key -- so a user who passes both
    --noise-robust-guards and e.g. --accel-gap-cycles 8 gets gap=8, not the
    preset's gap=5. See NOISE_ROBUST_GUARD_OVERRIDES in hardware/safety.py
    and docs/status/safety_envelope_backtest_2026-07-30.md for the evidence
    behind the preset values.
    """
    overrides: dict[str, float | int] = {}
    if args.noise_robust_guards:
        overrides.update(NOISE_ROBUST_GUARD_OVERRIDES)
    if args.max_tcp_speed_mps is not None:
        overrides["max_tcp_speed_mps"] = float(args.max_tcp_speed_mps)
    if args.max_tcp_accel_mps2 is not None:
        overrides["max_tcp_accel_mps2"] = float(args.max_tcp_accel_mps2)
    if args.accel_gap_cycles is not None:
        overrides["accel_gap_cycles"] = int(args.accel_gap_cycles)
    if args.speed_lowpass_alpha is not None:
        overrides["speed_lowpass_alpha"] = float(args.speed_lowpass_alpha)
    if args.accel_max_consecutive_violations is not None:
        overrides["accel_max_consecutive_violations"] = int(args.accel_max_consecutive_violations)
    if args.accel_hard_multiple is not None:
        overrides["accel_hard_multiple"] = float(args.accel_hard_multiple)
    if args.speed_max_consecutive_violations is not None:
        overrides["speed_max_consecutive_violations"] = int(args.speed_max_consecutive_violations)
    if args.speed_hard_multiple is not None:
        overrides["speed_hard_multiple"] = float(args.speed_hard_multiple)
    return overrides


def main() -> int:
    args = parse_args()
    axis_index = _AXIS_INDEX[args.axis]
    distance_m = signed_distance(args.distance_m, args.direction)
    peak_v = peak_velocity_mps(distance_m, args.duration_s)
    peak_a = peak_acceleration_mps2(distance_m, args.duration_s)

    print(f"Planned move: axis={args.axis} direction={args.direction} distance_m={distance_m:+.3f}")
    print(
        f"  duration_s={args.duration_s}  rate_hz={args.rate_hz}  "
        f"peak_velocity_mps={peak_v:.4f}  peak_accel_mps2={peak_a:.4f}"
    )

    if args.dry_run:
        # Waypoints are relative to whatever the start pose happens to be --
        # plan against a zero start pose purely to report waypoint count.
        waypoints = plan_waypoints(
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            axis_index=axis_index,
            distance_m=distance_m,
            duration_s=args.duration_s,
            rate_hz=args.rate_hz,
        )
        print(f"  waypoint_count={len(waypoints)}")
        print("[dry-run] no connection made, no motion commanded.")
        return 0

    if not args.motion_opt_in:
        print("[BLOCKED] pass --i-understand-this-moves-the-robot to actually move the robot.")
        return 1

    # Fixed, evidence-based ceiling -- not derived from this move's own peak_v.
    # The previous max(0.05, peak_v * 1.2) made both this pre-check and the live
    # CartesianMoveMonitor guard during the move tautological: peak_v can never
    # exceed 1.2x itself, so neither could ever trip on an aggressive-but-nominal
    # move, only on >20% overshoot from what was already planned.
    move_limit_overrides = build_move_limit_overrides(args)
    move_limits = CartesianMoveLimits.for_robot(args.robot_ip, **move_limit_overrides)
    if peak_v > move_limits.max_tcp_speed_mps:
        print(
            f"[BLOCKED] planned peak velocity {peak_v:.4f} m/s exceeds the monitor's "
            f"max_tcp_speed_mps={move_limits.max_tcp_speed_mps}; increase --duration-s."
        )
        return 1
    if peak_a > move_limits.max_tcp_accel_mps2:
        print(
            f"[BLOCKED] planned peak acceleration {peak_a:.4f} m/s^2 exceeds the monitor's "
            f"max_tcp_accel_mps2={move_limits.max_tcp_accel_mps2}; increase --duration-s."
        )
        return 1

    if not args.yes:
        confirm = input(f"Type MOVE to confirm this {distance_m:+.3f}m move on {args.robot_ip}: ")
        if confirm.strip() != "MOVE":
            print("[cancelled] confirmation not received.")
            return 1

    output_dir = args.output_dir
    if output_dir is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = REPO_ROOT / "outputs" / "hardware_transport" / f"servo_move_{stamp}"
    trace_path = output_dir / "trace.jsonl"
    summary_path = output_dir / "summary.json"

    limits = UR5eSafetyLimits()
    link = UR5eLink(args.robot_ip, args.rate_hz, limits=limits)
    monitor = CartesianMoveMonitor(move_limits)
    estop = EStopLatch()
    try:
        link.connect(with_control=True)
        print(f"[connected] {args.robot_ip} (receive+control)")
        result = move_cartesian_bounded(
            link,
            monitor,
            estop,
            axis_index=axis_index,
            distance_m=distance_m,
            motion_opt_in=args.motion_opt_in,
            duration_s=args.duration_s,
            rate_hz=args.rate_hz,
            trace_path=trace_path,
            summary_path=summary_path,
            max_shoulder_pan_delta_rad=args.max_shoulder_pan_delta_rad,
            max_stale_state_cycles=args.max_stale_state_cycles,
        )
    finally:
        link.disconnect()

    if result.ok:
        print(f"[PASS] move complete. waypoints_sent={result.waypoints_sent} final_tcp_pose={result.final_tcp_pose}")
        print(f"[logged] trace={result.trace_path} summary={result.summary_path}")
        return 0
    print(
        f"[FAIL] {result.reason} (waypoints_sent={result.waypoints_sent}, "
        f"stopped_early={result.stopped_early})"
    )
    print(f"[logged] trace={result.trace_path} summary={result.summary_path}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
