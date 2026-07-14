"""Golden-value regression guard for the tools/ur5e_mujoco_torque_experiments.py
plumbing move (x_profile_target, build_initial_state_and_adapter, apply_start_q/
resolve_start_q/coerce_start_q moved to simulation/ur5e_mujoco_torque.py).

Values below were refreshed after fixing ``expand_mass_matrix`` (correct
``mj_fullM`` usage across MuJoCo binding versions).
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
    "achieved_x_delta_m": pytest.approx(2.4147800612543547e-05, abs=1e-12),
    "final_x_error_m": pytest.approx(0.009975852199387457, abs=1e-12),
    "max_abs_orientation_error_rad": pytest.approx(1.2724684135723752e-05, abs=1e-12),
    "max_abs_qd_radps": pytest.approx(0.001328891168439839, abs=1e-12),
    "max_abs_tau_applied_nm": pytest.approx(0.21255900378184536, abs=1e-9),
    "move_hold_quality_score": pytest.approx(0.1995790397692334, abs=1e-9),
}
EXPECTED_FINAL_Q = [
    2.5126774343635992e-06,
    -1.5708086481437646,
    -2.5510174184623293e-05,
    -1.5707951336467978,
    -1.7053370374942624e-06,
    2.245310874135949e-05,
]
EXPECTED_FINAL_EE_POS = [2.414780061231773e-05, -0.23399999993991763, 1.0799999996263487]


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
