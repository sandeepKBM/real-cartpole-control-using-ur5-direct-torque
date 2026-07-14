#!/usr/bin/env python3
"""Apply a visible joint torque pulse to verify URSim physics responds."""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from hardware.direct_torque_link import UR5eDirectTorqueLink  # noqa: E402

ROBOT_IP = "127.0.0.1"
FREQ = 500.0
DT = 1.0 / FREQ


def main() -> int:
    link = UR5eDirectTorqueLink(ROBOT_IP, frequency_hz=FREQ)
    link.connect()
    s0 = link.read_state()
    q0 = np.asarray(s0.q, dtype=float).copy()
    print("q0:", np.round(q0, 4).tolist())

    tau = np.zeros(6)
    tau[1] = 15.0  # shoulder pitch, Nm
    for friction_comp in (True, False):
        print(f"\nfriction_comp={friction_comp}: tau[1]=15 Nm for 1.0 s")
        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline:
            link.direct_torque(tau, friction_comp=friction_comp)
            time.sleep(DT)
        s1 = link.read_state()
        dq = np.asarray(s1.q, dtype=float) - q0
        print("  max_abs_dq:", float(np.max(np.abs(dq))))
        if float(np.max(np.abs(dq))) > 1e-4:
            link.safe_stop("pulse_done")
            return 0
    link.safe_stop("pulse_done")
    print("\nNo joint motion detected. On URSim this is expected: direct_torque has no effect.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
