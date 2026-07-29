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
from hardware.safety import CartesianMoveLimits, CartesianMoveMonitor, EStopLatch, UR5eSafetyLimits  # noqa: E402

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
    move_limit_overrides = {}
    if args.max_tcp_speed_mps is not None:
        move_limit_overrides["max_tcp_speed_mps"] = float(args.max_tcp_speed_mps)
    if args.max_tcp_accel_mps2 is not None:
        move_limit_overrides["max_tcp_accel_mps2"] = float(args.max_tcp_accel_mps2)
    if args.accel_gap_cycles is not None:
        move_limit_overrides["accel_gap_cycles"] = int(args.accel_gap_cycles)
    if args.speed_lowpass_alpha is not None:
        move_limit_overrides["speed_lowpass_alpha"] = float(args.speed_lowpass_alpha)
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
