#!/usr/bin/env python3
"""Wrapper for the staged UR5e hardware smoke tests.

Modes:
  - receive-only
  - zero-hold
  - tiny-motion
"""

from __future__ import annotations

import argparse

from _bootstrap import ensure_repo_root

ensure_repo_root()

from hardware.ur5e_stages import run_receive_only, run_servoj_tiny_motion, run_servoj_zero_hold


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="UR5e hardware smoke-test wrapper.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--mode",
        choices=("receive-only", "zero-hold", "tiny-motion"),
        default="receive-only",
        help="Which staged test to run.",
    )
    p.add_argument("--robot-ip", default="172.16.71.77", help="UR5e robot IP address.")
    p.add_argument("--frequency", type=float, default=500.0, help="Target RTDE loop frequency in Hz.")
    p.add_argument("--duration", type=float, default=30.0, help="Stage duration in seconds.")
    p.add_argument("--max-deadline-ms", type=float, default=3.0, help="Deadline threshold in ms.")
    p.add_argument("--gain", type=float, default=100.0, help="servoJ gain.")
    p.add_argument("--lookahead-time", type=float, default=0.1, help="servoJ lookahead time in seconds.")
    p.add_argument("--velocity", type=float, default=0.05, help="servoJ velocity cap.")
    p.add_argument("--acceleration", type=float, default=0.05, help="servoJ acceleration cap.")
    p.add_argument("--joint-index", type=int, default=0, help="Joint index for tiny-motion mode.")
    p.add_argument("--amplitude-rad", type=float, default=0.005, help="Tiny-motion amplitude in radians.")
    p.add_argument("--max-amplitude-rad", type=float, default=0.01, help="Refuse larger amplitudes.")
    p.add_argument("--output", default="logs/hardware_smoke.json", help="JSON output path.")
    p.add_argument("--publish-ros-topics", action="store_true", help="Enable optional ROS visualization.")
    p.add_argument("--ros-prefix", default="/ur5e", help="ROS topic prefix.")
    p.add_argument("--dry-run", action="store_true", help="Run the receive-only stage without robot I/O.")
    p.add_argument(
        "--i-understand-this-moves-the-robot",
        action="store_true",
        help="Required for motion stages.",
    )
    return p


def main() -> int:
    args = build_arg_parser().parse_args()
    if args.mode == "receive-only":
        result = run_receive_only(
            robot_ip=str(args.robot_ip or "127.0.0.1"),
            frequency=float(args.frequency),
            duration=float(args.duration),
            output=str(args.output),
            max_deadline_ms=float(args.max_deadline_ms),
            publish_ros_topics=bool(args.publish_ros_topics),
            ros_prefix=str(args.ros_prefix),
            dry_run=bool(args.dry_run),
        )
    elif args.mode == "zero-hold":
        if not args.robot_ip:
            raise SystemExit("--robot-ip is required for zero-hold mode")
        result = run_servoj_zero_hold(
            robot_ip=str(args.robot_ip),
            frequency=float(args.frequency),
            duration=float(args.duration),
            gain=float(args.gain),
            lookahead_time=float(args.lookahead_time),
            velocity=float(args.velocity),
            acceleration=float(args.acceleration),
            max_deadline_ms=float(args.max_deadline_ms),
            motion_opt_in=bool(args.i_understand_this_moves_the_robot),
            output=str(args.output),
            publish_ros_topics=bool(args.publish_ros_topics),
            ros_prefix=str(args.ros_prefix),
        )
    else:
        if not args.robot_ip:
            raise SystemExit("--robot-ip is required for tiny-motion mode")
        result = run_servoj_tiny_motion(
            robot_ip=str(args.robot_ip),
            frequency=float(args.frequency),
            duration=float(args.duration),
            joint_index=int(args.joint_index),
            amplitude_rad=float(args.amplitude_rad),
            max_amplitude_rad=float(args.max_amplitude_rad),
            gain=float(args.gain),
            lookahead_time=float(args.lookahead_time),
            velocity=float(args.velocity),
            acceleration=float(args.acceleration),
            max_deadline_ms=float(args.max_deadline_ms),
            motion_opt_in=bool(args.i_understand_this_moves_the_robot),
            output=str(args.output),
            publish_ros_topics=bool(args.publish_ros_topics),
            ros_prefix=str(args.ros_prefix),
        )
    return 0 if result.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
