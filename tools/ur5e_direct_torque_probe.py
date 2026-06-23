#!/usr/bin/env python3
"""Heavily gated direct-torque capability probe for UR5e.

Default behavior is zero-only and no motion. Nonzero torque is refused unless
all safety flags are explicitly supplied, and even then this repo currently
does not expose a real direct-torque RTDE implementation for a UR5e.
"""

from __future__ import annotations

import argparse

from _bootstrap import ensure_repo_root

ensure_repo_root()

from hardware.ur5e_stages import run_direct_torque_probe


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="UR5e direct-torque probe (zero-only by default; nonzero torque is heavily gated).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--robot-ip", required=True, help="UR5e robot IP address.")
    p.add_argument("--frequency", type=float, default=500.0, help="Target RTDE loop frequency in Hz.")
    p.add_argument("--duration", type=float, default=0.5, help="Probe duration in seconds.")
    p.add_argument("--max-torque-nm", type=float, default=0.05, help="Very low torque cap for nonzero probes.")
    p.add_argument(
        "--zero-only",
        dest="zero_only",
        action="store_true",
        default=True,
        help="Default zero-torque / no-motion probe.",
    )
    p.add_argument(
        "--enable-nonzero-torque",
        dest="zero_only",
        action="store_false",
        help="Allow nonzero torque probe mode (still refused unless multiple safety flags are present).",
    )
    p.add_argument(
        "--i-understand-direct-torque-is-dangerous",
        action="store_true",
        help="Explicit risk acknowledgement required for any nonzero torque attempt.",
    )
    p.add_argument(
        "--i-am-with-trained-supervisor",
        action="store_true",
        help="Required supervisor confirmation for any nonzero torque attempt.",
    )
    p.add_argument(
        "--output",
        default="logs/direct_torque_probe.json",
        help="JSON output path for the probe report.",
    )
    return p


def main() -> int:
    args = build_arg_parser().parse_args()
    result = run_direct_torque_probe(
        robot_ip=str(args.robot_ip),
        frequency=float(args.frequency),
        duration=float(args.duration),
        max_torque_nm=float(args.max_torque_nm),
        zero_only=bool(args.zero_only),
        understand_danger=bool(args.i_understand_direct_torque_is_dangerous),
        supervisor_present=bool(args.i_am_with_trained_supervisor),
        enable_nonzero_torque=not bool(args.zero_only),
        output=str(args.output),
    )
    return 0 if result.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
