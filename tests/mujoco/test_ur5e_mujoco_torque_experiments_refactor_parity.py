"""Golden-value regression guard for the tools/ur5e_mujoco_torque_experiments.py
plumbing move (x_profile_target, build_initial_state_and_adapter, apply_start_q/
resolve_start_q/coerce_start_q moved to simulation/ur5e_mujoco_torque.py).

Values below were captured by running the identical command against the code
immediately before and after that move (a git-worktree diff of the full
trace.jsonl showed zero mismatches across all 250 steps; summary.json matched
except for absolute path fields). This test pins a handful of representative
summary fields so any future accidental behavior change in that call path is
caught, without checking in a large trace fixture.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

EXPECTED = {
    "termination_reason": "duration_complete",
    "steps": 250,
    "achieved_x_delta_m": pytest.approx(2.1359980823239867e-05, abs=1e-12),
    "final_x_error_m": pytest.approx(0.009978640019176761, abs=1e-12),
    "max_abs_orientation_error_rad": pytest.approx(1.125336179592295e-05, abs=1e-12),
    "max_abs_qd_radps": pytest.approx(0.0011764913904821592, abs=1e-12),
    "max_abs_tau_applied_nm": pytest.approx(0.18833473468883397, abs=1e-9),
    "move_hold_quality_score": pytest.approx(0.19962752575752596, abs=1e-9),
}
EXPECTED_FINAL_Q = [
    2.223046923836849e-06,
    -1.570807223866776,
    -2.256813267603262e-05,
    -1.5707952721838259,
    -1.5088723726911234e-06,
    1.9864844580384265e-05,
]
EXPECTED_FINAL_EE_POS = [2.1359980823014048e-05, -0.23399999995298015, 1.0799999997076304]


def test_controller_rollout_matches_pre_refactor_golden_values(tmp_path: Path) -> None:
    out_root = tmp_path / "run"
    completed = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "tools" / "ur5e_mujoco_torque_experiments.py"),
            "--mode", "controller-rollout",
            "--controller-kind", "impedance",
            "--config", str(REPO_ROOT / "config" / "ur5e_mujoco_torque_osc_tuned.yaml"),
            "--trajectory-profile", "min_jerk_move_hold",
            "--move-duration", "0.3",
            "--duration", "0.5",
            "--target-x-delta", "0.01",
            "--seed", "0",
            "--no-plot",
            "--output-dir", str(out_root),
        ],
        cwd=str(REPO_ROOT),
        check=False,
        text=True,
        capture_output=True,
    )
    assert completed.returncode == 0, completed.stderr
    run_dir = max((p.parent for p in out_root.rglob("summary.json")), key=lambda p: p.stat().st_mtime)
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))

    for key, expected in EXPECTED.items():
        assert summary[key] == expected, f"{key}: {summary[key]!r} != {expected!r}"

    import numpy as np
    np.testing.assert_allclose(summary["final_q"], EXPECTED_FINAL_Q, atol=1e-12)
    np.testing.assert_allclose(summary["final_ee_pos"], EXPECTED_FINAL_EE_POS, atol=1e-12)
