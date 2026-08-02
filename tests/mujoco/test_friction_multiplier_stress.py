"""Tests for --friction-multiplier (added 2026-08-02 to
tools/ur5e_mujoco_torque_experiments.py) -- a deliberate stress-test proxy,
not another calibration attempt. See docs/status/friction_calibration_from_
qdd_residual_2026-08-02.md for the real calibration this multiplies from.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG = REPO_ROOT / "config" / "ur5e_mujoco_torque_osc_tuned.yaml"


def _run(tmp_path: Path, label: str, extra_args: list[str]) -> dict:
    out_dir = tmp_path / label
    cmd = [
        sys.executable,
        str(REPO_ROOT / "tools" / "ur5e_mujoco_torque_experiments.py"),
        "--mode", "controller-rollout",
        "--controller-kind", "impedance",
        "--config", str(CONFIG),
        "--trajectory-profile", "min_jerk_move_hold",
        "--move-duration", "0.3",
        "--duration", "0.5",
        "--target-x-delta", "0.01",
        "--seed", "0",
        "--no-plot",
        "--output-dir", str(out_dir),
        *extra_args,
    ]
    completed = subprocess.run(cmd, cwd=str(REPO_ROOT), capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr
    summary_path = next(out_dir.rglob("summary.json"))
    return json.loads(summary_path.read_text())


def test_default_multiplier_matches_golden_value_exactly(tmp_path: Path) -> None:
    s = _run(tmp_path, "default", [])
    assert s["achieved_x_delta_m"] == 0.005970857523244166


def test_explicit_multiplier_one_matches_default(tmp_path: Path) -> None:
    default = _run(tmp_path, "default2", [])
    explicit = _run(tmp_path, "explicit_one", ["--friction-multiplier", "1.0"])
    assert default["achieved_x_delta_m"] == explicit["achieved_x_delta_m"]


def test_higher_multiplier_reduces_achieved_displacement(tmp_path: Path) -> None:
    """More friction should make an already friction-limited short move
    achieve LESS, not more or the same -- the direction sanity check."""
    baseline = _run(tmp_path, "baseline", [])
    stressed = _run(tmp_path, "stressed", ["--friction-multiplier", "2.0"])
    assert stressed["achieved_x_delta_m"] < baseline["achieved_x_delta_m"]


def test_zero_multiplier_is_clamped_not_negative_or_crashing(tmp_path: Path) -> None:
    s = _run(tmp_path, "zero", ["--friction-multiplier", "0.0"])
    assert s["termination_reason"] == "duration_complete"


def test_run_still_completes_cleanly_at_high_stress_multiplier(tmp_path: Path) -> None:
    """A 5x stress multiplier must not crash the rollout, even though it
    will (correctly) achieve much less of the target."""
    s = _run(tmp_path, "high_stress", ["--friction-multiplier", "5.0"])
    assert s["termination_reason"] == "duration_complete"
