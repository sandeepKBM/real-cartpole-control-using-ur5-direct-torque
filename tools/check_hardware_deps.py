#!/usr/bin/env python3
"""Verify the hardware-lane Python environment actually has what it needs --
no robot connection required. Run this BEFORE a lab session, not during one.

Written after a real 2026-07-29/30 lab session where two dependencies
individually looked "installed" (pip show reported a version satisfying this
project's documented floor) but were missing the specific attribute this
codebase actually calls, discovered only by crashing mid-session on real
hardware:
  - ur_rtde 1.6.1 (satisfies the previously-documented ">=1.6") has no
    RTDEControlInterface.directTorque -- direct_torque mode cannot work at all.
  - pip-installed pinocchio (version floor >=3.1 satisfied) had no
    pin.buildModelFromMJCF -- crashed the diagnostic-only residual observer
    (since made to degrade gracefully instead, see
    docs/status/direct_torque_residual_observer_2026-07-29.md) and would also
    break dynamics_source=local_pinocchio / gravity_source=pinocchio.

Lesson: a version floor being satisfied does not mean the attribute this
codebase calls actually exists in that build. Check the attribute directly.
"""

from __future__ import annotations

import sys


def _check(label: str, fn) -> bool:
    try:
        ok, detail = fn()
    except Exception as exc:  # noqa: BLE001 - this is a diagnostic script, report anything
        print(f"[FAIL] {label}: {type(exc).__name__}: {exc}")
        return False
    print(f"[{'OK' if ok else 'FAIL'}] {label}: {detail}")
    return ok


def check_ur_rtde() -> tuple[bool, str]:
    import rtde_control

    version = getattr(rtde_control, "__version__", "unknown")
    has_direct_torque = hasattr(rtde_control.RTDEControlInterface, "directTorque")
    return has_direct_torque, f"ur_rtde version={version}, RTDEControlInterface.directTorque present={has_direct_torque}"


def check_pinocchio() -> tuple[bool, str]:
    import pinocchio as pin

    version = getattr(pin, "__version__", "unknown")
    has_mjcf = hasattr(pin, "buildModelFromMJCF")
    return has_mjcf, f"pinocchio version={version}, buildModelFromMJCF present={has_mjcf}"


def check_mujoco() -> tuple[bool, str]:
    import mujoco

    version = getattr(mujoco, "__version__", "unknown")
    return True, f"mujoco version={version}"


def check_numpy_scipy_yaml() -> tuple[bool, str]:
    import numpy
    import scipy
    import yaml

    return True, (
        f"numpy={numpy.__version__}, scipy={scipy.__version__}, "
        f"pyyaml={getattr(yaml, '__version__', 'unknown')}"
    )


def main() -> int:
    print("Hardware-lane dependency check (no robot connection required)\n")
    results = [
        _check("ur_rtde directTorque()", check_ur_rtde),
        _check("pinocchio buildModelFromMJCF", check_pinocchio),
        _check("mujoco (local dynamics_source paths)", check_mujoco),
        _check("numpy/scipy/pyyaml", check_numpy_scipy_yaml),
    ]
    print()
    if all(results):
        print("All checks passed.")
        return 0
    print(
        "One or more checks failed. direct_torque mode needs ur_rtde's "
        "directTorque(); the diagnostic residual observer and "
        "dynamics_source=local_pinocchio/gravity_source=pinocchio need "
        "pinocchio's buildModelFromMJCF. Fix before going to the robot, not "
        "during a live session -- see requirements-hardware.txt."
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
