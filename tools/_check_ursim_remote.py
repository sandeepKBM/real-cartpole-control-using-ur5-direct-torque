#!/usr/bin/env python3
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from hardware.dashboard import dashboard_command, query_remote_control

ROBOT_IP = "127.0.0.1"

cmds = [
    "robotmode",
    "is in remote control",
    "get operational mode",
    "safetymode",
    "programstate",
    "version",
]
for c in cmds:
    print(f"{c}: {dashboard_command(ROBOT_IP, c)}")
print("query_remote_control:", query_remote_control(ROBOT_IP))
