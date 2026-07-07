#!/usr/bin/env python3
"""Connect to a real UR5e and read its state. Cannot move the robot.

This script never imports ``hardware.motion`` -- it is physically incapable
of commanding any robot motion regardless of what flags you pass it. Use
``tools/ur5e_move.py`` for that.

Two modes:
  --once   connect, read state one time, print it, exit.
  --watch  connect and read state continuously (Ctrl-C to stop), with real
           liveness detection: if reads start failing, this reconnects with
           backoff a bounded number of times, and if that doesn't recover,
           it stops and reports failure rather than looping forever on stale
           data (see hardware/safety.py::ConnectionHealth).

Examples:
  python tools/ur5e_connect.py --robot-ip 192.168.1.10 --once
  python tools/ur5e_connect.py --robot-ip 192.168.1.10 --watch
"""

from __future__ import annotations

import argparse
import sys
import time

from _bootstrap import ensure_repo_root

ensure_repo_root()

from hardware.link import RTDEStateError, UR5eLink  # noqa: E402
from hardware.safety import EStopLatch, UR5eSafetyLimits  # noqa: E402

RECONNECT_ATTEMPTS = 2
RECONNECT_BACKOFF_S = (1.0, 3.0)


def _print_state(state) -> None:
    print(f"  q  = {state.q.round(4).tolist()} rad")
    print(f"  qd = {state.qd.round(4).tolist()} rad/s")
    print(f"  tcp_pose = {state.tcp_pose.round(4).tolist()}  (x,y,z,rx,ry,rz)")
    if state.robot_timestamp_s is not None:
        print(f"  robot_timestamp_s = {state.robot_timestamp_s:.3f}")
    if state.safety_status is not None:
        print(f"  safety_status = {state.safety_status}")


def run_once(link: UR5eLink) -> int:
    link.connect(with_control=False)
    print(f"[connected] {link.robot_ip} (receive-only)")
    state = link.read_state()
    print("[state]")
    _print_state(state)
    return 0


def run_watch(link: UR5eLink, estop: EStopLatch, frequency_hz: float) -> int:
    link.connect(with_control=False)
    print(f"[connected] {link.robot_ip} (receive-only) -- watching at {frequency_hz} Hz, Ctrl-C to stop")
    period_s = 1.0 / frequency_hz
    cycle = 0
    try:
        while True:
            estop.raise_if_tripped()
            cycle += 1
            try:
                state = link.read_state()
            except RTDEStateError as exc:
                tripped = link.health.record_failure()
                print(f"[read failed] ({link.health.consecutive_failures}) {exc}")
                if tripped:
                    print("[reconnecting]")
                    reconnected = False
                    for attempt, backoff_s in enumerate(RECONNECT_BACKOFF_S[:RECONNECT_ATTEMPTS], start=1):
                        time.sleep(backoff_s)
                        try:
                            link.disconnect()
                            link.connect(with_control=False)
                            link.read_state()
                            reconnected = True
                            print(f"[reconnected] on attempt {attempt}")
                            break
                        except (RTDEStateError, Exception) as reconnect_exc:  # noqa: BLE001
                            print(f"[reconnect attempt {attempt} failed] {reconnect_exc}")
                    if not reconnected:
                        reason = "connection lost and reconnect attempts exhausted"
                        estop.trip(reason)
                        link.disconnect()
                        print(f"[FAIL] {reason}")
                        return 1
                continue

            if cycle % max(1, int(frequency_hz)) == 0:
                print(f"[cycle {cycle}]")
                _print_state(state)
            time.sleep(period_s)
    except KeyboardInterrupt:
        print("\n[stopped by user]")
        link.disconnect()
        return 0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--robot-ip", required=True, help="UR5e robot IP address. No default -- always explicit.")
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--once", action="store_true", help="Connect, read state once, print, exit.")
    mode.add_argument("--watch", action="store_true", help="Continuous state monitor (Ctrl-C to stop).")
    p.add_argument("--frequency-hz", type=float, default=125.0)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    limits = UR5eSafetyLimits()
    link = UR5eLink(args.robot_ip, args.frequency_hz, limits=limits)
    estop = EStopLatch()
    try:
        if args.once:
            return run_once(link)
        return run_watch(link, estop, args.frequency_hz)
    finally:
        link.disconnect()


if __name__ == "__main__":
    raise SystemExit(main())
