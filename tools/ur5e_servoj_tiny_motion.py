#!/usr/bin/env python3
"""Tiny bounded joint-space motion stage using `servoJ`.

This script refuses to run unless the motion opt-in flag is present and the
requested amplitude is within a tiny bound.
"""

from __future__ import annotations

import argparse

from _bootstrap import ensure_repo_root

ensure_repo_root()

from hardware.ur5e_stages import run_servoj_tiny_motion


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="UR5e tiny servoJ motion stage (explicit motion opt-in required).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--robot-ip", required=True, help="UR5e robot IP address.")
    p.add_argument("--frequency", type=float, default=500.0, help="Target RTDE loop frequency in Hz.")
    p.add_argument("--duration", type=float, default=3.0, help="Motion duration in seconds.")
    p.add_argument("--joint-index", type=int, default=0, help="Joint to perturb (0-5).")
    p.add_argument("--amplitude-rad", type=float, default=0.005, help="Joint amplitude in radians.")
    p.add_argument("--max-amplitude-rad", type=float, default=0.01, help="Refuse amplitudes above this bound.")
    p.add_argument("--gain", type=float, default=100.0, help="servoJ gain.")
    p.add_argument("--lookahead-time", type=float, default=0.1, help="servoJ lookahead time in seconds.")
    p.add_argument("--velocity", type=float, default=0.05, help="servoJ velocity cap.")
    p.add_argument("--acceleration", type=float, default=0.05, help="servoJ acceleration cap.")
    p.add_argument(
        "--max-deadline-ms",
        type=float,
        default=3.0,
        help="Fail the stage if a control cycle exceeds this deadline.",
    )
    p.add_argument(
        "--output",
        default="logs/servoj_tiny_motion.json",
        help="JSON output path for the timing/state report.",
    )
    p.add_argument(
        "--publish-ros-topics",
        action="store_true",
        help="Publish non-blocking visualization topics on /ur5e/*.",
    )
    p.add_argument("--ros-prefix", default="/ur5e", help="ROS topic prefix.")
    p.add_argument(
        "--i-understand-this-moves-the-robot",
        action="store_true",
        help="Required explicit opt-in for any real robot motion.",
    )
    return p


def main() -> int:
    args = build_arg_parser().parse_args()
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
