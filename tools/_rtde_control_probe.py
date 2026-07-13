#!/usr/bin/env python3
"""Minimal RTDE control socket diagnostic."""
from __future__ import annotations

import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from hardware.dashboard import dashboard_command, query_remote_control  # noqa: E402

ROBOT_IP = "127.0.0.1"
FREQ = 500.0


def main() -> int:
    print("dashboard:")
    for cmd in ("robotmode", "is in remote control", "get operational mode", "safetymode", "programstate"):
        print(f"  {cmd}: {dashboard_command(ROBOT_IP, cmd)}")
    print("  query_remote_control:", query_remote_control(ROBOT_IP))

    from rtde_control import RTDEControlInterface
    from rtde_receive import RTDEReceiveInterface

    print("\nTrying RTDEControlInterface (control only, verbose)...")
    try:
        ctrl = RTDEControlInterface(ROBOT_IP, FREQ, RTDEControlInterface.FLAG_VERBOSE)
        print("  control OK")
        print("  has directTorque:", hasattr(ctrl, "directTorque"))
        if hasattr(ctrl, "directTorque"):
            ok = ctrl.directTorque([0.0] * 6, True)
            print("  directTorque(zeros) ->", ok)
        ctrl.disconnect()
    except Exception as exc:
        print("  control FAILED:", exc)
        return 1

    print("\nTrying RTDEReceiveInterface...")
    try:
        recv = RTDEReceiveInterface(ROBOT_IP, FREQ)
        print("  receive OK, q=", recv.getActualQ()[:2], "...")
        recv.disconnect()
    except Exception as exc:
        print("  receive FAILED:", exc)
        return 1

    print("\nRTDE CONTROL PROBE OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
