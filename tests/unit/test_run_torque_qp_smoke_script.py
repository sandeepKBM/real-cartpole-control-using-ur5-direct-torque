"""Regression test (2026-07-29 bug audit) for tools/diagnostics/run_torque_qp_smoke.py.

The script imported from ``controller_core.tests.test_torque_task_qp``, a
module that no longer exists (the real tests live under
``tests/unit/test_torque_task_qp.py``; ``controller_core/tests/`` is an
empty, ``__pycache__``-only leftover). Every invocation of this diagnostic
failed with ``ModuleNotFoundError`` before running anything. Runs the script
as a real subprocess (matching how a human/CI would invoke it) rather than
importing it, since the bug was specifically about the script's own
module-level import failing at the top of the file.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "tools" / "diagnostics" / "run_torque_qp_smoke.py"


def test_run_torque_qp_smoke_script_runs_and_passes() -> None:
    result = subprocess.run(
        [sys.executable, str(SCRIPT)],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert "torque_task_qp smoke tests passed" in result.stdout
