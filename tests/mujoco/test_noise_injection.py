"""Tests for the --q-noise-std-rad / --qd-noise-std-radps / --torque-noise-std-nm
flags added to tools/ur5e_mujoco_torque_experiments.py.

Regression net (default 0.0 = byte-identical to before these flags existed)
plus sanity checks that nonzero noise actually perturbs the right things and
nothing else.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG = REPO_ROOT / "config" / "ur5e_mujoco_torque_osc_tuned_adaptive_lambda.yaml"
ALPHA_0_1_Q = ["0.0", "-1.423717", "-0.24", "-1.453717", "0.0", "0.0"]


def _run(tmp_path: Path, label: str, extra_args: list[str]) -> dict:
    out_dir = tmp_path / label
    cmd = [
        sys.executable,
        str(REPO_ROOT / "tools" / "ur5e_mujoco_torque_experiments.py"),
        "--mode", "controller-rollout",
        "--controller-kind", "impedance",
        "--config", str(CONFIG),
        "--trajectory-profile", "min_jerk_move_hold",
        "--gravity-mode", "gravity_comp",
        "--target-x-delta", "0.05",
        "--move-duration", "0.5",
        "--duration", "0.7",
        "--start-q-rad", *ALPHA_0_1_Q,
        "--output-dir", str(out_dir),
        "--seed", "0",
        "--no-plot",
        *extra_args,
    ]
    completed = subprocess.run(cmd, cwd=str(REPO_ROOT), capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr
    trace_path = next(out_dir.rglob("trace.jsonl"))
    return [json.loads(line) for line in trace_path.read_text().splitlines()]


def test_zero_noise_flags_match_no_flags_at_all(tmp_path: Path) -> None:
    baseline = _run(tmp_path, "baseline", [])
    explicit_zero = _run(
        tmp_path, "explicit_zero",
        ["--q-noise-std-rad", "0.0", "--qd-noise-std-radps", "0.0", "--torque-noise-std-nm", "0.0"],
    )
    assert len(baseline) == len(explicit_zero)
    for a, b in zip(baseline, explicit_zero):
        assert a["q"] == b["q"]
        assert a["tau"] == b["tau"]
        assert a["ee_pos"] == b["ee_pos"]


def test_q_noise_perturbs_trajectory(tmp_path: Path) -> None:
    baseline = _run(tmp_path, "baseline2", [])
    noisy = _run(tmp_path, "q_noisy", ["--q-noise-std-rad", "0.05", "--noise-seed", "1"])
    assert baseline[-1]["q"] != noisy[-1]["q"], "q noise should change the resulting trajectory"


def test_torque_noise_changes_tau_but_not_tau_controller(tmp_path: Path) -> None:
    baseline = _run(tmp_path, "baseline3", [])
    noisy = _run(tmp_path, "torque_noisy", ["--torque-noise-std-nm", "2.0", "--noise-seed", "2"])
    # tau (the actually-applied torque, including noise) must differ...
    assert any(b["tau"] != n["tau"] for b, n in zip(baseline, noisy))
    # ...but tau_controller (the controller's own clean diagnostic output,
    # unaffected by the noise this flag injects downstream of it) should not
    # be perturbed by the SAME mechanism -- it can still differ slightly
    # because the physics trajectory itself diverges once torque noise feeds
    # back through mj_step, but at step 0 (before any noise has had a chance
    # to alter the state yet) tau_controller must still match exactly.
    assert baseline[0]["tau_controller"] == noisy[0]["tau_controller"]


def test_noise_seed_is_reproducible(tmp_path: Path) -> None:
    run_a = _run(tmp_path, "seeded_a", ["--q-noise-std-rad", "0.02", "--noise-seed", "42"])
    run_b = _run(tmp_path, "seeded_b", ["--q-noise-std-rad", "0.02", "--noise-seed", "42"])
    assert run_a[-1]["q"] == run_b[-1]["q"]
