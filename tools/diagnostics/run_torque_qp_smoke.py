#!/usr/bin/env python3
import sys
from pathlib import Path

# tests/ lives at the repo root, one level above tools/diagnostics/; the
# import below was stale (controller_core.tests.test_torque_task_qp, a
# module that no longer exists -- controller_core/tests/ is an empty
# __pycache__-only leftover, the real tests live under tests/unit/). Fixed
# 2026-07-29 bug audit: this made every invocation of this diagnostic fail
# with ModuleNotFoundError before running anything.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tests.unit.test_torque_task_qp import (
    test_box_qp_respects_bounds,
    test_qp_controller_returns_finite_torque,
    test_velocity_bounds_tighten_torque_box,
)

if __name__ == "__main__":
    test_box_qp_respects_bounds()
    test_velocity_bounds_tighten_torque_box()
    test_qp_controller_returns_finite_torque()
    print("torque_task_qp smoke tests passed")
