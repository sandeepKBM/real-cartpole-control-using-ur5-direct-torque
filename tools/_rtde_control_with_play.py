#!/usr/bin/env python3
"""Try RTDE control after dashboard prep steps."""
from __future__ import annotations

import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from hardware.dashboard import dashboard_command  # noqa: E402
from rtde_control import RTDEControlInterface  # noqa: E402

ROBOT_IP = "127.0.0.1"
FREQ = 500.0


def prep(label: str, commands: list[str]) -> None:
    print(f"\n--- prep: {label} ---")
    for cmd in commands:
        print(f"  {cmd}: {dashboard_command(ROBOT_IP, cmd)}")


def try_control() -> bool:
    print("\n--- RTDEControlInterface ---")
    try:
        ctrl = RTDEControlInterface(ROBOT_IP, FREQ, RTDEControlInterface.FLAG_VERBOSE)
        print("  control OK")
        ctrl.disconnect()
        return True
    except Exception as exc:
        print("  FAILED:", exc)
        return False


def main() -> int:
    for cmd in ("robotmode", "is in remote control", "get operational mode", "programstate", "version"):
        print(f"{cmd}: {dashboard_command(ROBOT_IP, cmd)}")

    prep("play", ["play"])
    time.sleep(1.0)
    if try_control():
        return 0

    prep("stop_then_play", ["stop", "play"])
    time.sleep(1.0)
    if try_control():
        return 0

    prep("unlock_protective_stop", ["unlock protective stop", "play"])
    time.sleep(1.0)
    if try_control():
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
