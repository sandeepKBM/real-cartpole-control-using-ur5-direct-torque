#!/usr/bin/env python3
"""Safe receive-only RTDE probe for a UR5e.

Default behavior is no motion. The script connects to the robot only to read
joint state and TCP pose, measure host-side timing, and optionally publish
visualization topics from a background thread.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from _bootstrap import ensure_repo_root

ensure_repo_root()

from hardware.ur5e_stages import run_receive_only


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="UR5e receive-only RTDE probe (safe; no motion).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--robot-ip", required=False, default="", help="UR5e robot IP address.")
    p.add_argument("--frequency", type=float, default=500.0, help="Target RTDE loop frequency in Hz.")
    p.add_argument("--duration", type=float, default=30.0, help="Probe duration in seconds.")
    p.add_argument(
        "--max-deadline-ms",
        type=float,
        default=3.0,
        help="Fail the probe if a cycle exceeds this deadline.",
    )
    p.add_argument(
        "--output",
        default="logs/receive_only.json",
        help="JSON output path for the timing/state report.",
    )
    p.add_argument(
        "--publish-ros-topics",
        action="store_true",
        help="Publish non-blocking visualization topics on /ur5e/*.",
    )
    p.add_argument(
        "--ros-prefix",
        default="/ur5e",
        help="ROS topic prefix used when --publish-ros-topics is enabled.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Run the timing/report path without connecting to the robot.",
    )
    return p


def main() -> int:
    args = build_arg_parser().parse_args()
    if not args.dry_run and not args.robot_ip:
        raise SystemExit("--robot-ip is required unless --dry-run is used")
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
    return 0 if result.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
