"""Controller-rollout-level test for the wrist_orientation_task fix.

Reproduces a scaled-down version of the 2026-07-27/28 "directional ceiling"
finding (AGENTS.md sec 3): at height_alpha=0.5 (hardware/poses.py::
HEIGHT_ALPHA_0_5_Q), the tuned OSC config's orientation error grows large on
a large -X transport move because orientation is held only as a side effect
of the nullspace-posture projector, whose restoring authority is asymmetric
with wrist_2 sign at that pose. config/ur5e_mujoco_torque_osc_tuned_wrist_
orient.yaml adds a separate, wrist-masked joint-space PD term
(controller.wrist_orientation_task) that gives orientation its own
authority. This test uses a smaller/faster move (dx=-0.10m, 2s total) than
the full validation sweep (see docs/status/wrist_orientation_task_2026-07-29.md
for the full dx=+/-0.20m numbers) purely to keep the test fast; the same
qualitative improvement (orientation error at flag-on well below flag-off)
is what's asserted here.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from hardware.poses import HEIGHT_ALPHA_0_5_Q  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
TUNED_CONFIG = REPO_ROOT / "config" / "ur5e_mujoco_torque_osc_tuned.yaml"
WRIST_ORIENT_CONFIG = REPO_ROOT / "config" / "ur5e_mujoco_torque_osc_tuned_wrist_orient.yaml"


def _latest_run_dir(out_root: Path) -> Path:
    candidates = [p for p in out_root.rglob("summary.json")]
    assert candidates, f"no summary.json found under {out_root}"
    return max(candidates, key=lambda p: p.stat().st_mtime).parent


def _run(tmp_path: Path, config_path: Path, *, label: str) -> dict:
    out_root = tmp_path / label
    completed = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "tools" / "ur5e_mujoco_torque_experiments.py"),
            "--mode", "controller-rollout",
            "--controller-kind", "impedance",
            "--config", str(config_path),
            "--trajectory-profile", "min_jerk_move_hold",
            "--move-duration", "1.0",
            "--duration", "2.0",
            "--target-x-delta", "-0.10",
            "--start-q-rad",
            *[repr(float(v)) for v in HEIGHT_ALPHA_0_5_Q],
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
    run_dir = _latest_run_dir(out_root)
    return json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))


def test_wrist_orientation_task_reduces_orientation_error_at_height_alpha_0_5(tmp_path: Path) -> None:
    baseline = _run(tmp_path, TUNED_CONFIG, label="baseline")
    fixed = _run(tmp_path, WRIST_ORIENT_CONFIG, label="wrist_orient")

    baseline_ori = float(baseline["move_phase_max_abs_orientation_error_rad"])
    fixed_ori = float(fixed["move_phase_max_abs_orientation_error_rad"])

    assert baseline["valid_move_and_hold"] is True
    assert fixed["valid_move_and_hold"] is True
    # Real, substantial improvement -- not a rounding-level difference.
    # Measured at authoring time: baseline ~0.122 rad, fixed ~0.032 rad.
    assert fixed_ori < 0.6 * baseline_ori
    assert fixed["safety_pass"] is True
