"""Golden-value regression guard for the tools/ur5e_mujoco_torque_experiments.py
plumbing move (x_profile_target, build_initial_state_and_adapter, apply_start_q/
resolve_start_q/coerce_start_q moved to simulation/ur5e_mujoco_torque.py).

Values refreshed 2026-07-31: assets/ur5e_torque/ur5e_torque.xml gained real joint
friction (frictionloss/damping on the size3/size1 default classes -- see
docs/status/ur5e_sim_friction_modeling_2026-07-31.md) that did not exist when the
2026-07-30 values below were captured. This is a real, intentional physics change
(the model was previously, unrealistically, frictionless), not a bug -- friction
now measurably slows this short 0.3s move's tracking (achieved_x_delta_m drops
from ~0.0101 m to ~0.0067 m against the same 0.01 m target; move_hold_quality_score
drops from ~0.61 to ~0.30), consistent with the qualitative sim-to-real gap this
friction addition was meant to close. Re-derived by re-running the exact command
below against the friction-enabled model (--seed 0, deterministic); the run still
completes cleanly (termination_reason=duration_complete, valid_move_and_hold=True).

Prior note (2026-07-30, still accurate for that transition): config/
ur5e_mujoco_torque_osc_tuned.yaml was promoted from the singular_scale-enabled
default to the singular_scale-disabled one (jacobian_singular_cond_max: 1.0e18 --
see that file's header and
docs/status/disable_global_singular_scale_validation_2026-07-30.md). The values
captured before that fix reflected a real bug (controller frozen for most of the
move), not correct behavior.
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
    "achieved_x_delta_m": pytest.approx(0.006677818892950525, abs=1e-12),
    "final_x_error_m": pytest.approx(0.0033221811070494756, abs=1e-12),
    "max_abs_orientation_error_rad": pytest.approx(0.008667602130849907, abs=1e-12),
    "max_abs_qd_radps": pytest.approx(0.03594456180001037, abs=1e-12),
    "max_abs_tau_applied_nm": pytest.approx(6.643184104836119, abs=1e-9),
    "move_hold_quality_score": pytest.approx(0.2966448644866346, abs=1e-9),
}
EXPECTED_FINAL_Q = [
    8.401498293147879e-05,
    -1.5758197122204896,
    -0.0040320048044671375,
    -1.5715194862259532,
    -4.27446080881236e-05,
    0.0010956729274157373,
]
EXPECTED_FINAL_EE_POS = [0.006677818892950299, -0.2339994396976498, 1.0799738266265693]


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
