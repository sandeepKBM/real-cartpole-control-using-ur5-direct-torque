#!/usr/bin/env python3
"""Wait for URSim dashboard after docker restart, then power on."""
from __future__ import annotations

import socket
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from hardware.dashboard import dashboard_command, power_on_and_release, query_remote_control

ROBOT_IP = "127.0.0.1"
PORT = 29999
TIMEOUT_S = 180.0
POLL_S = 3.0


def port_open() -> bool:
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(2.0)
        sock.connect((ROBOT_IP, PORT))
        sock.close()
        return True
    except OSError:
        return False


def main() -> int:
    deadline = time.monotonic() + TIMEOUT_S
    while time.monotonic() < deadline:
        if port_open():
            try:
                resp = dashboard_command(ROBOT_IP, "version", timeout_s=5.0)
                if resp.strip():
                    print("dashboard ready:", resp)
                    break
            except Exception as exc:
                print("dashboard port open but not ready:", exc)
        else:
            print("waiting for dashboard port...")
        time.sleep(POLL_S)
    else:
        print("Timed out waiting for dashboard")
        return 1

    print("\nPower on sequence:")
    for cmd, resp in power_on_and_release(ROBOT_IP).items():
        print(f"  {cmd}: {resp}")
    time.sleep(3.0)
    print("remote_control:", query_remote_control(ROBOT_IP))
    print("robotmode:", dashboard_command(ROBOT_IP, "robotmode"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
