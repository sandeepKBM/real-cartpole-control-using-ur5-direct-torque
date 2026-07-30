"""Golden-value regression guard for the tools/ur5e_mujoco_torque_experiments.py
plumbing move (x_profile_target, build_initial_state_and_adapter, apply_start_q/
resolve_start_q/coerce_start_q moved to simulation/ur5e_mujoco_torque.py).

Values refreshed 2026-07-30: config/ur5e_mujoco_torque_osc_tuned.yaml (used by
this test) was promoted from the singular_scale-enabled default to the
singular_scale-disabled one (jacobian_singular_cond_max: 1.0e18 -- see that
file's header and docs/status/disable_global_singular_scale_validation_2026-07-30.md).
The OLD golden values below captured a real bug, not correct behavior: with
singular_scale enabled, the controller was frozen (tau~1e-13-1e-4 Nm) for
essentially this entire short 0.3s move, so achieved_x_delta_m was ~2.4e-5 m
against a 0.01 m target -- off by ~400x. The new values reflect the
move actually completing close to its target.
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
    "achieved_x_delta_m": pytest.approx(0.010118103975688159, abs=1e-12),
    "final_x_error_m": pytest.approx(-0.00011810397568815835, abs=1e-12),
    "max_abs_orientation_error_rad": pytest.approx(0.010748992082004229, abs=1e-12),
    "max_abs_qd_radps": pytest.approx(0.056309901527987885, abs=1e-12),
    "max_abs_tau_applied_nm": pytest.approx(2.9572777532474124, abs=1e-9),
    "move_hold_quality_score": pytest.approx(0.6120449191460254, abs=1e-9),
}
EXPECTED_FINAL_Q = [
    0.0012684573093999323,
    -1.579297649925123,
    -0.004283204566751049,
    -1.5702074940577369,
    -0.00022665150852423045,
    0.0021185709980759745,
]
EXPECTED_FINAL_EE_POS = [0.010118103975687933, -0.23398735129287185, 1.079945447324358]


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
