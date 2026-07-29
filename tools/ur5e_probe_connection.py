#!/usr/bin/env python3
"""Pre-flight RTDE connection-stability probe. Receive-only -- cannot move
the robot regardless of flags, same guarantee as ur5e_connect.py.

Built 2026-07-30 after repeated real-hardware direct_torque attempts aborted
mid-move on a genuine RTDE stream stall (StaleStateMonitor correctly
detecting a frozen-but-non-erroring robot_timestamp_s -- see
docs/status/ and AGENTS.md SS4 for the full incident). Run this BEFORE a real
direct_torque attempt: a few seconds of receive-only checking is much
cheaper than discovering the instability mid-move, and gives a clear
go/no-go signal instead of guessing.

Exit code 0 = clean for the whole probe window. Exit code 1 = a stall or
read failure was detected (prints exactly when/what).

Example:
  python tools/ur5e_probe_connection.py --robot-ip 172.16.71.77 --duration-s 3.0
"""

from __future__ import annotations

import argparse
import time

from _bootstrap import ensure_repo_root

ensure_repo_root()

from hardware.link import RTDEStateError, UR5eLink  # noqa: E402
from hardware.safety import StaleStateMonitor, UR5eSafetyLimits  # noqa: E402
from hardware.timing import monotonic_ns  # noqa: E402


def probe(link: UR5eLink, duration_s: float, frequency_hz: float) -> int:
    link.connect(with_control=False)
    print(f"[connected] {link.robot_ip} (receive-only) -- probing for {duration_s}s at {frequency_hz} Hz")
    period_s = 1.0 / frequency_hz
    stale_monitor = StaleStateMonitor()
    deadline = time.monotonic() + duration_s
    cycle = 0
    read_failures = 0
    while time.monotonic() < deadline:
        cycle += 1
        try:
            state = link.read_state()
        except RTDEStateError as exc:
            read_failures += 1
            print(f"[FAIL] read_state() raised at cycle {cycle}: {exc}")
            link.disconnect()
            return 1
        stale_reason = stale_monitor.record(state.robot_timestamp_s, monotonic_ns())
        if stale_reason is not None:
            print(f"[FAIL] {stale_reason} (detected at cycle {cycle}, {cycle * period_s:.2f}s in)")
            link.disconnect()
            return 1
        time.sleep(period_s)
    link.disconnect()
    print(f"[PASS] {cycle} cycles clean over {duration_s}s, 0 stalls, {read_failures} read failures")
    return 0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--robot-ip", required=True)
    p.add_argument(
        "--duration-s",
        type=float,
        default=3.0,
        help="How long to watch before declaring PASS. Longer = more confidence, costs more time.",
    )
    p.add_argument(
        "--frequency-hz",
        type=float,
        default=500.0,
        help="Read rate. Default 500 matches direct_torque's actual rate -- the mode this probe exists for.",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    limits = UR5eSafetyLimits()
    link = UR5eLink(args.robot_ip, args.frequency_hz, limits=limits)
    return probe(link, args.duration_s, args.frequency_hz)


if __name__ == "__main__":
    raise SystemExit(main())
