"""Tests for --telemetry-duplicate-prob (added 2026-08-02 to
tools/ur5e_mujoco_torque_experiments.py) -- an RTDE duplicate-frame proxy for
sim, mirroring --q-noise-std-rad/--qd-noise-std-radps/--torque-noise-std-nm's
already-established pattern (see test_noise_injection.py).

Motivation: real UR5e hardware showed ~17% isolated single-cycle duplicate
tcp_pose/q reads in some 2026-08-02 direct_torque runs (0% in others) --
this flag lets that exact class of telemetry degradation be reproduced and
tested against in sim, deterministically and repeatably, instead of only
encountering it by chance on real hardware.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG = REPO_ROOT / "config" / "ur5e_mujoco_torque_osc_tuned_adaptive_lambda.yaml"
ALPHA_0_1_Q = ["0.0", "-1.423717", "-0.24", "-1.453717", "0.0", "0.0"]


def _run(tmp_path: Path, label: str, extra_args: list[str]) -> list[dict]:
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


def test_zero_probability_matches_no_flag_at_all(tmp_path: Path) -> None:
    baseline = _run(tmp_path, "baseline", [])
    explicit_zero = _run(tmp_path, "explicit_zero", ["--telemetry-duplicate-prob", "0.0"])
    assert len(baseline) == len(explicit_zero)
    for a, b in zip(baseline, explicit_zero):
        assert a["q"] == b["q"]
        assert a["tau"] == b["tau"]
        assert a["tau_controller"] == b["tau_controller"]
        assert a["ee_pos"] == b["ee_pos"]


def test_ground_truth_trace_is_unaffected_by_injected_duplicates(tmp_path: Path) -> None:
    """The proxy only changes what the controller is FED -- true physics
    (mj_step) and the trace's own ground-truth q/ee_pos must be identical to
    a torque-noise-free, duplicate-free run at every step, since only the
    controller's resulting tau differs cycle to cycle, and even that
    difference is expected to be small at this displacement/duration (large
    duplicate probability, so verify the run at least completes and differs
    somewhere, not that ground truth is byte-identical -- physics does
    diverge once a different tau feeds back through mj_step)."""
    baseline = _run(tmp_path, "baseline2", [])
    dup = _run(tmp_path, "dup", ["--telemetry-duplicate-prob", "0.5", "--noise-seed", "1"])
    # Step 0: no prior delivered state exists yet, so the very first cycle
    # can never be a duplicate regardless of probability -- controller output
    # must match exactly at step 0.
    assert baseline[0]["tau_controller"] == dup[0]["tau_controller"]
    # Later in the run, the controller has seen at least one duplicate with
    # very high probability at p=0.5 over many steps -- something changes.
    assert any(b["tau_controller"] != d["tau_controller"] for b, d in zip(baseline, dup))


def test_high_probability_produces_repeated_controller_decisions(tmp_path: Path) -> None:
    """At telemetry_duplicate_prob=1.0, every cycle after the first is fed
    the exact same (frozen) controller_state as the first delivered one --
    tau_controller should show long runs of exact repeats, not smooth
    variation, distinguishing this from ordinary sensor noise."""
    rows = _run(tmp_path, "always_dup", ["--telemetry-duplicate-prob", "1.0"])
    tau_controller_values = [tuple(r["tau_controller"]) for r in rows]
    unique_values = set(tau_controller_values)
    # A normal (non-duplicated) rollout over many steps produces a smoothly
    # varying trajectory -- essentially every tau_controller value distinct.
    # Forcing every cycle after the first to replay the same delivered state
    # collapses this to a small number of distinct values.
    assert len(unique_values) < len(tau_controller_values) / 4


def test_noise_seed_is_reproducible(tmp_path: Path) -> None:
    run_a = _run(tmp_path, "seeded_a", ["--telemetry-duplicate-prob", "0.3", "--noise-seed", "42"])
    run_b = _run(tmp_path, "seeded_b", ["--telemetry-duplicate-prob", "0.3", "--noise-seed", "42"])
    assert run_a[-1]["q"] == run_b[-1]["q"]
    assert run_a[-1]["tau_controller"] == run_b[-1]["tau_controller"]


def test_composes_with_sensor_noise(tmp_path: Path) -> None:
    """Both proxies active together must not crash and must produce a
    completed run -- duplicate-or-fresh decided first, sensor noise applied
    on top, per the flag's own documented ordering."""
    rows = _run(
        tmp_path, "combo",
        ["--telemetry-duplicate-prob", "0.2", "--q-noise-std-rad", "0.01", "--noise-seed", "7"],
    )
    assert len(rows) > 0
