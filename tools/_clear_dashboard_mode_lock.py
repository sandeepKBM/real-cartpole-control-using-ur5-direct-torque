#!/usr/bin/env python3
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from hardware.dashboard import clear_operational_mode, dashboard_command

print("clear:", clear_operational_mode("127.0.0.1"))
print("mode:", dashboard_command("127.0.0.1", "get operational mode"))
