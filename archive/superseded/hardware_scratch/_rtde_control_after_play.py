#!/usr/bin/env python3
"""Try RTDE control after dashboard play / load empty program."""
from __future__ import annotations

import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from hardware.dashboard import dashboard_command, query_remote_control  # noqa: E402
from rtde_control import RTDEControlInterface  # noqa: E402

ROBOT_IP = "127.0.0.1"
FREQ = 500.0


def try_control(label: str, flags: int = 0) -> bool:
    print(f"\n--- {label} flags={flags} ---")
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
        print("  FAIL:", exc)
        return False


def main() -> int:
    print("remote:", query_remote_control(ROBOT_IP))
    for cmd in ("programstate", "robotmode"):
        print(f"{cmd}: {dashboard_command(ROBOT_IP, cmd)}")

    print("\nload blank program attempt:")
    for cmd in ("load installation default.installation", "load program empty.program"):
        try:
            print(f"  {cmd}: {dashboard_command(ROBOT_IP, cmd)}")
        except Exception as exc:
            print(f"  {cmd}: ERROR {exc}")

    print("\nplay:", dashboard_command(ROBOT_IP, "play"))
    time.sleep(2.0)
    print("programstate:", dashboard_command(ROBOT_IP, "programstate"))

    ok = try_control("default", 0)
    ok = try_control("no_wait", RTDEControlInterface.FLAG_NO_WAIT) or ok
    ok = try_control(
        "verbose",
        RTDEControlInterface.FLAG_VERBOSE | RTDEControlInterface.FLAG_NO_WAIT,
    ) or ok
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
