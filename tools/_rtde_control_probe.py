#!/usr/bin/env python3
"""Minimal RTDE control socket diagnostic (uses the same link layer as production CLIs)."""
from __future__ import annotations

import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from hardware.dashboard import dashboard_command, query_remote_control  # noqa: E402
from hardware.direct_torque_link import UR5eDirectTorqueLink  # noqa: E402
from hardware.link import UR5eLink  # noqa: E402

ROBOT_IP = "127.0.0.1"
FREQ = 125.0


def main() -> int:
    print("dashboard:")
    for cmd in ("robotmode", "is in remote control", "get operational mode", "safetymode", "programstate"):
        print(f"  {cmd}: {dashboard_command(ROBOT_IP, cmd)}")
    print("  query_remote_control:", query_remote_control(ROBOT_IP))

    print("\nTrying UR5eLink receive-only...")
    recv_link = UR5eLink(ROBOT_IP, FREQ)
    try:
        recv_link.connect(with_control=False)
        st = recv_link.read_state()
        print(f"  receive OK q0={st.q[0]:.4f} tcp_x={st.tcp_pose[0]:.4f}")
    except Exception as exc:
        print("  receive FAILED:", exc)
        return 1
    finally:
        recv_link.disconnect()

    print("\nTrying UR5eDirectTorqueLink (control + directTorque)...")
    link = UR5eDirectTorqueLink(ROBOT_IP, frequency_hz=FREQ)
    try:
        link.connect()
        st = link.read_state()
        print(f"  control OK q0={st.q[0]:.4f}")
        deadline = time.monotonic() + 0.5
        while time.monotonic() < deadline:
            link.direct_torque([0.0] * 6, friction_comp=True)
            time.sleep(0.008)
        print("  directTorque(zeros) held for 0.5 s")
    except Exception as exc:
        print("  control FAILED:", exc)
        return 1
    finally:
        link.safe_stop("probe_complete")

    print("\nRTDE CONTROL PROBE OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
