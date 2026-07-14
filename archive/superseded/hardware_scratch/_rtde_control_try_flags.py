#!/usr/bin/env python3
"""Try RTDE control with alternate flags / order."""
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


def try_connect(label: str, flags: int = 0) -> bool:
    print(f"\n=== {label} flags={flags} ===")
    try:
        if flags:
            ctrl = RTDEControlInterface(ROBOT_IP, FREQ, flags)
        else:
            ctrl = RTDEControlInterface(ROBOT_IP, FREQ)
        print("  control OK")
        if hasattr(ctrl, "directTorque"):
            print("  directTorque ->", ctrl.directTorque([0.0] * 6, True))
        ctrl.disconnect()
        return True
    except Exception as exc:
        print("  FAILED:", exc)
        return False


def main() -> int:
    print("dashboard:", dashboard_command(ROBOT_IP, "is in remote control"))
    print("operational:", dashboard_command(ROBOT_IP, "get operational mode"))

    cases = [
        ("default", 0),
        ("verbose", RTDEControlInterface.FLAG_VERBOSE),
        ("no_wait", RTDEControlInterface.FLAG_NO_WAIT),
        ("disable_remote_check", RTDEControlInterface.FLAG_DISABLE_REMOTE_CONTROL_CHECK),
        (
            "verbose_no_wait",
            RTDEControlInterface.FLAG_VERBOSE | RTDEControlInterface.FLAG_NO_WAIT,
        ),
    ]
    ok_any = any(try_connect(name, flags) for name, flags in cases)
    return 0 if ok_any else 1


if __name__ == "__main__":
    raise SystemExit(main())
