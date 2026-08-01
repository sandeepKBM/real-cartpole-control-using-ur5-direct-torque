"""Tests for the opt-in sim-side TCP-accel/speed guard.

Background: the real UR5e's ``hardware/safety.py::CartesianMoveMonitor`` (TCP
speed/acceleration finite-difference guard) has no sim-side equivalent, so
this repo's sim experiment harness (``tools/ur5e_mujoco_torque_experiments.py``)
was structurally unable to reproduce the real-hardware TCP-accel guard trips
found on 2026-08-01 -- see docs/status/sim_tcp_accel_guard_2026-08-01.md.
``--enable-tcp-accel-guard`` ports that same, already-thoroughly-tested class
(see tests/hardware/test_hardware_safety.py for its own spike/clean/gap/
lowpass/consecutive-violation coverage -- not duplicated here) into the sim
per-step loop via a local import, opt-in and default off.

This file covers what's new/integration-specific:
  - the CLI override-merge helper (``_resolve_tcp_accel_guard_limits``);
  - the (ee_pos, zeros(3)) tcp_pose padding convention used to feed
    CartesianMoveMonitor.check() from sim state;
  - end-to-end: a real sim rollout with the guard enabled trips on a genuine
    synthetic spike and stays clean on a gentle move;
  - zero regression: an existing run without the flag is unaffected.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from hardware.safety import CartesianMoveLimits, CartesianMoveMonitor, NOISE_ROBUST_GUARD_OVERRIDES  # noqa: E402
from tools.ur5e_mujoco_torque_experiments import _resolve_tcp_accel_guard_limits  # noqa: E402

CONFIG_PATH = REPO_ROOT / "config" / "ur5e_mujoco_torque_osc_tuned_wrist_orient_friction_ff.yaml"


def _base_guard_args(**overrides) -> argparse.Namespace:
    defaults = dict(
        tcp_accel_guard_noise_robust=False,
        max_tcp_accel_mps2=None,
        max_tcp_speed_mps=None,
        accel_gap_cycles=None,
        speed_lowpass_alpha=None,
        accel_max_consecutive_violations=None,
        accel_hard_multiple=None,
        speed_max_consecutive_violations=None,
        speed_hard_multiple=None,
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def test_resolve_tcp_accel_guard_limits_default_matches_class_default():
    limits = _resolve_tcp_accel_guard_limits(_base_guard_args())
    assert limits == CartesianMoveLimits()


def test_resolve_tcp_accel_guard_limits_noise_robust_preset_applies():
    limits = _resolve_tcp_accel_guard_limits(_base_guard_args(tcp_accel_guard_noise_robust=True))
    for key, value in NOISE_ROBUST_GUARD_OVERRIDES.items():
        assert getattr(limits, key) == value


def test_resolve_tcp_accel_guard_limits_explicit_override_wins_over_preset():
    limits = _resolve_tcp_accel_guard_limits(
        _base_guard_args(tcp_accel_guard_noise_robust=True, accel_gap_cycles=8)
    )
    # accel_gap_cycles explicitly set to 8, overriding the preset's 5.
    assert limits.accel_gap_cycles == 8
    # every other preset field still applied.
    assert limits.speed_lowpass_alpha == NOISE_ROBUST_GUARD_OVERRIDES["speed_lowpass_alpha"]


def test_resolve_tcp_accel_guard_limits_individual_override_without_preset():
    limits = _resolve_tcp_accel_guard_limits(_base_guard_args(max_tcp_accel_mps2=0.05))
    assert limits.max_tcp_accel_mps2 == pytest.approx(0.05)
    assert limits.accel_gap_cycles == 1  # untouched class default


def test_ee_pos_zero_padding_convention_check_only_reads_first_three():
    """Sim state has no orientation-axis-angle representation matching real
    UR5e tcp_pose[3:6] -- the integration pads ee_pos with zeros(3) since
    CartesianMoveMonitor.check() only ever reads pose[:3] (confirmed by
    reading the class directly). This locks that assumption down: two calls
    that differ only in the padding bits must produce identical decisions."""
    limits = CartesianMoveLimits(max_tcp_accel_mps2=0.5)
    mon_a = CartesianMoveMonitor(limits)
    mon_b = CartesianMoveMonitor(limits)
    start = np.array([0.0, 0.0, 0.9], dtype=np.float64)
    mon_a.set_start(np.concatenate([start, np.zeros(3)]), move_axis_index=0)
    mon_b.set_start(np.concatenate([start, np.array([1.0, 2.0, 3.0])]), move_axis_index=0)

    dt_s = 0.002
    pos = start.copy()
    for step in range(5):
        pos = pos + np.array([0.0002, 0.0, 0.0])
        target = np.array([pos[0] + 0.01, start[1], start[2]], dtype=np.float64)
        decision_a = mon_a.check(
            q=np.zeros(6), qd=np.zeros(6),
            tcp_pose=np.concatenate([pos, np.zeros(3)]),
            target_tcp_pose=np.concatenate([target, np.zeros(3)]),
            orientation_error_rad=0.0, axis_target_moving=True, dt_s=dt_s,
        )
        decision_b = mon_b.check(
            q=np.zeros(6), qd=np.zeros(6),
            tcp_pose=np.concatenate([pos, np.array([9.0, -9.0, 4.0])]),
            target_tcp_pose=np.concatenate([target, np.array([1.0, 1.0, 1.0])]),
            orientation_error_rad=0.0, axis_target_moving=True, dt_s=dt_s,
        )
        assert decision_a.ok == decision_b.ok
        assert decision_a.reason == decision_b.reason


def test_synthetic_position_spike_trips_the_ported_monitor():
    """Same shape of check the task asked for at the class level (already
    covered by tests/hardware/test_hardware_safety.py) reproduced through
    THIS integration's exact padding/call convention, so a regression in the
    wiring (not just the class) would be caught here too."""
    limits = CartesianMoveLimits(max_tcp_accel_mps2=0.5, max_tcp_speed_mps=1.0)
    mon = CartesianMoveMonitor(limits)
    start = np.zeros(3)
    mon.set_start(np.concatenate([start, np.zeros(3)]), move_axis_index=0)
    dt_s = 0.002

    # A handful of tiny, smooth steps (no spike) should never trip.
    pos = start.copy()
    for _ in range(3):
        pos = pos + np.array([0.0001, 0.0, 0.0])
        decision = mon.check(
            q=np.zeros(6), qd=np.zeros(6),
            tcp_pose=np.concatenate([pos, np.zeros(3)]),
            target_tcp_pose=np.concatenate([pos + 0.01, np.zeros(3)]),
            orientation_error_rad=0.0, axis_target_moving=True, dt_s=dt_s,
        )
        assert decision.ok, decision.reason

    # A single, large instantaneous jump (a genuine acceleration spike) must trip.
    pos = pos + np.array([0.05, 0.0, 0.0])
    decision = mon.check(
        q=np.zeros(6), qd=np.zeros(6),
        tcp_pose=np.concatenate([pos, np.zeros(3)]),
        target_tcp_pose=np.concatenate([pos + 0.01, np.zeros(3)]),
        orientation_error_rad=0.0, axis_target_moving=True, dt_s=dt_s,
    )
    assert not decision.ok
    assert "TCP" in decision.reason


def _run_cli(tmp_path: Path, *args: str) -> tuple[dict, subprocess.CompletedProcess[str]]:
    out_root = tmp_path / "run"
    completed = subprocess.run(
        [sys.executable, str(REPO_ROOT / "tools" / "ur5e_mujoco_torque_experiments.py"), *args, "--output-dir", str(out_root)],
        cwd=str(REPO_ROOT),
        check=False,
        text=True,
        capture_output=True,
    )
    run_dirs = sorted(out_root.glob("*"))
    assert run_dirs, completed.stderr
    summary = json.loads((run_dirs[-1] / "summary.json").read_text(encoding="utf-8"))
    return summary, completed


@pytest.mark.slow
def test_guard_disabled_by_default_produces_no_new_summary_keys(tmp_path: Path):
    summary, completed = _run_cli(
        tmp_path,
        "--mode", "controller-rollout",
        "--config", str(CONFIG_PATH),
        "--controller-kind", "impedance",
        "--target-x-delta", "0.01",
        "--duration", "0.05",
        "--no-plot",
    )
    assert completed.returncode == 0, completed.stderr
    assert "tcp_accel_guard_enabled" not in summary
    assert "tcp_accel_guard_tripped" not in summary
    assert "tcp_accel_guard_limits" not in summary


@pytest.mark.slow
def test_guard_enabled_gentle_move_does_not_trip(tmp_path: Path):
    summary, completed = _run_cli(
        tmp_path,
        "--mode", "controller-rollout",
        "--config", str(CONFIG_PATH),
        "--controller-kind", "impedance",
        "--target-x-delta", "0.005",
        "--duration", "0.1",
        "--enable-tcp-accel-guard",
        "--tcp-accel-guard-noise-robust",
        "--no-plot",
    )
    assert completed.returncode == 0, completed.stderr
    assert summary["tcp_accel_guard_enabled"] is True
    assert summary.get("tcp_accel_guard_tripped") is False
    assert not str(summary["termination_reason"]).startswith("tcp_accel_guard:")


@pytest.mark.slow
def test_guard_enabled_with_tight_ceiling_trips_deterministically(tmp_path: Path):
    """A near-zero max_tcp_accel_mps2 must trip almost immediately on any real
    move, giving a deterministic, fast reproduction of a guard trip end-to-end
    through the actual sim per-step loop (not just the class in isolation)."""
    summary, completed = _run_cli(
        tmp_path,
        "--mode", "controller-rollout",
        "--config", str(CONFIG_PATH),
        "--controller-kind", "impedance",
        "--target-x-delta", "0.02",
        "--duration", "0.2",
        "--enable-tcp-accel-guard",
        "--max-tcp-accel-mps2", "1e-6",
        "--max-tcp-speed-mps", "1e-6",
        "--no-plot",
    )
    assert completed.returncode == 1
    assert summary["success"] is False
    assert summary["tcp_accel_guard_enabled"] is True
    assert summary["tcp_accel_guard_tripped"] is True
    assert str(summary["termination_reason"]).startswith("tcp_accel_guard:")
