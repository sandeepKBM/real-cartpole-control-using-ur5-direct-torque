#!/usr/bin/env python3
"""Sweep RTDE control connect attempts for URSim bring-up."""
from __future__ import annotations

import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from hardware.dashboard import dashboard_command, query_remote_control  # noqa: E402
from rtde_control import RTDEControlInterface  # noqa: E402

ROBOT_IP = "127.0.0.1"


def try_freq(freq: float, flags: int = 0) -> bool:
    label = f"freq={freq} flags={flags}"
    print(f"\n--- {label} ---")
    try:
        if flags:
            ctrl = RTDEControlInterface(ROBOT_IP, freq, flags)
        else:
            ctrl = RTDEControlInterface(ROBOT_IP, freq)
        print("  OK")
        ctrl.disconnect()
        return True
    except Exception as exc:
        print("  FAIL:", exc)
        return False


def main() -> int:
    for cmd in ("robotmode", "is in remote control", "get operational mode", "programstate", "version"):
        print(f"{cmd}: {dashboard_command(ROBOT_IP, cmd)}")
    print("query_remote_control:", query_remote_control(ROBOT_IP))

    ok = False
    for freq in (500.0, 250.0, 125.0):
        ok = try_freq(freq) or ok
    ok = try_freq(500.0, RTDEControlInterface.FLAG_VERBOSE) or ok
    ok = try_freq(
        500.0,
        RTDEControlInterface.FLAG_VERBOSE | RTDEControlInterface.FLAG_NO_WAIT,
    ) or ok

    print("\n--- dashboard play then retry ---")
    print("play:", dashboard_command(ROBOT_IP, "play"))
    time.sleep(1.0)
    ok = try_freq(500.0, RTDEControlInterface.FLAG_VERBOSE) or ok
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
