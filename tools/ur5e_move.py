#!/usr/bin/env python3
"""Move a real UR5e a bounded distance along one Cartesian axis.

Uses UR's native position-space servoL streaming (the robot firmware does
its own inverse kinematics) with a live safety monitor checking drift,
orientation, velocity, and tracking error every single control cycle --
see hardware/motion.py and hardware/safety.py::CartesianMoveMonitor.

This does NOT run the simulation's torque-based Cartesian impedance
controller: the real robot's RTDE control library has no working torque API
in this environment, and there is no Jacobian/forward-kinematics code
anywhere in this repo that works from real robot state without MuJoCo. See
AGENTS.md / the plan file for the full reasoning.

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

from _bootstrap import ensure_repo_root

ensure_repo_root()

from hardware.link import UR5eLink  # noqa: E402
from hardware.motion import move_cartesian_bounded, peak_velocity_mps, plan_waypoints  # noqa: E402
from hardware.safety import CartesianMoveLimits, CartesianMoveMonitor, EStopLatch, UR5eSafetyLimits  # noqa: E402

_AXIS_INDEX = {"x": 0, "y": 1, "z": 2}


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

    print(f"Planned move: axis={args.axis} direction={args.direction} distance_m={distance_m:+.3f}")
    print(f"  duration_s={args.duration_s}  rate_hz={args.rate_hz}  peak_velocity_mps={peak_v:.4f}")

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

    move_limits = CartesianMoveLimits(max_tcp_speed_mps=max(0.05, peak_v * 1.2))
    if peak_v > move_limits.max_tcp_speed_mps:
        print(
            f"[BLOCKED] planned peak velocity {peak_v:.4f} m/s exceeds the monitor's "
            f"max_tcp_speed_mps={move_limits.max_tcp_speed_mps}; increase --duration-s."
        )
        return 1

    if not args.yes:
        confirm = input(f"Type MOVE to confirm this {distance_m:+.3f}m move on {args.robot_ip}: ")
        if confirm.strip() != "MOVE":
            print("[cancelled] confirmation not received.")
            return 1

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
        )
    finally:
        link.disconnect()

    if result.ok:
        print(f"[PASS] move complete. waypoints_sent={result.waypoints_sent} final_tcp_pose={result.final_tcp_pose}")
        return 0
    print(
        f"[FAIL] {result.reason} (waypoints_sent={result.waypoints_sent}, "
        f"stopped_early={result.stopped_early})"
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
