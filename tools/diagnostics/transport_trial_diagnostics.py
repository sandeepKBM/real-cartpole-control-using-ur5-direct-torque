#!/usr/bin/env python3
"""Reusable post-trial diagnostics for a single real (or sim) X-transport run.

Exists to replace the pattern this session fell into: after every real-
hardware trial, hand-writing a one-off Python snippet to check guard
category / resonance content / hold-phase error, then deciding the next
command by eye. That doesn't scale past a handful of trials and doesn't
give ``tools/autonomous_transport_explorer.py`` anything to call
automatically. This module is the single place that logic lives now.

Every function here operates on already-written ``summary.json`` /
``trace.jsonl`` files -- it never touches the robot.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

# Guard-category classification is substring matching against the exact
# reason strings emitted by controller_core/safety.py's ImpedanceSafetyMonitor
# and hardware/safety.py's CartesianMoveMonitor/DeadlineMonitor/
# StaleStateMonitor -- confirmed against those files directly (2026-08-02),
# not guessed. Order matters: check more specific substrings before generic
# ones where they could both match.
GUARD_CATEGORY_PATTERNS: tuple[tuple[str, str], ...] = (
    ("duration_complete", "success"),
    ("TCP speed", "tcp_speed_guard"),
    ("TCP acceleration", "tcp_accel_guard"),
    ("||orientation error||", "orientation_guard"),
    ("|Y-Y0|", "y_drift_guard"),
    ("|Z-Z0|", "z_drift_guard"),
    ("orthogonal drift", "orthogonal_drift_guard"),
    ("axis_error| grew", "axis_error_growth_guard"),
    ("|qd| >", "joint_velocity_guard"),
    ("joint limit", "joint_limit_guard"),
    ("NaN/Inf", "nan_inf_guard"),
    ("stale_state", "telemetry_staleness_guard"),
    ("rtde_state_error", "rtde_read_error"),
    ("robot_safety_status_abnormal", "robot_safety_status_guard"),
    ("deadline", "deadline_guard"),
    ("estop", "estop"),
)

# Predicted closed-loop resonance for the tuned/calibrated OSC configs sits
# at ~2.0-2.1Hz (kp_x=400, kd_x=40, real Lambda_xx~2.3kg at the wrist2-offset
# pose -- see docs/status or the 2026-08-02 session history for the
# derivation). Band chosen to comfortably bracket the measured 1.95-2.04Hz
# oscillation with margin on both sides.
RESONANCE_BAND_HZ = (1.4, 2.2)
LIFT_JOINT_INDEX = 1  # shoulder_lift -- does most of the X-axis work at this pose


def categorize_termination(reason: str | None) -> str:
    if not reason:
        return "unknown"
    for pattern, category in GUARD_CATEGORY_PATTERNS:
        if pattern in reason:
            return category
    return "other"


def load_trial(run_dir: Path, *, load_trace: bool = True) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    summary_path = run_dir / "summary.json"
    with open(summary_path) as f:
        summary = json.load(f)
    trace: list[dict[str, Any]] = []
    if load_trace:
        trace_path = run_dir / "trace.jsonl"
        if trace_path.is_file():
            with open(trace_path) as f:
                trace = [json.loads(line) for line in f if line.strip()]
    return summary, trace


def resonance_band_power_fraction(
    trace: list[dict[str, Any]],
    move_duration_s: float,
    *,
    joint_index: int = LIFT_JOINT_INDEX,
    band_hz: tuple[float, float] = RESONANCE_BAND_HZ,
    min_move_cycles: int = 200,
) -> float | None:
    """Fraction of move-phase commanded-torque spectral power sitting in
    ``band_hz``, for ``joint_index``. None if the trace is too short (early
    guard trip) to say anything meaningful. Validated against real data
    2026-08-02: friction_feedforward on/deadband=0.05 -> 0.806; off -> 0.068;
    deadband=0.10 -> 0.152 (see session history for the three trace IDs)."""
    if not trace:
        return None
    t = np.array([row["time_s"] for row in trace])
    dt = float(np.median(np.diff(t))) if len(t) > 1 else None
    if not dt or dt <= 0:
        return None
    move_mask = t <= move_duration_s
    if int(move_mask.sum()) < min_move_cycles:
        return None
    # position-mode traces log tau_shadow (OSC computed but not applied),
    # not tau_controller -- fall back rather than KeyError so this stays
    # callable from position-mode tools (e.g. the rail-bound finder).
    tau_key = "tau_controller" if "tau_controller" in trace[0] else "tau_shadow"
    if tau_key not in trace[0]:
        return None
    tau = np.array([row[tau_key][joint_index] for row in trace])
    seg = tau[move_mask]
    seg = seg - seg.mean()
    win = np.hanning(len(seg))
    segw = seg * win
    freqs = np.fft.rfftfreq(len(segw), d=dt)
    fft_mag = np.abs(np.fft.rfft(segw))
    full_mask = (freqs > 0.3) & (freqs < 10.0)
    band_mask = (freqs > band_hz[0]) & (freqs < band_hz[1])
    total_power = float(np.sum(fft_mag[full_mask] ** 2))
    if total_power <= 0.0:
        return None
    band_power = float(np.sum(fft_mag[band_mask] ** 2))
    return band_power / total_power


def hold_phase_error_mm(
    trace: list[dict[str, Any]], move_duration_s: float, *, min_hold_cycles: int = 5
) -> dict[str, float] | None:
    """Mean/final |x_error| in mm during the hold phase (t > move_duration).
    None if the run never reached a hold phase (tripped during the move)."""
    if not trace:
        return None
    t = np.array([row["time_s"] for row in trace])
    xerr = np.array([row["x_error"] for row in trace])
    hold_mask = t > move_duration_s
    if int(hold_mask.sum()) < min_hold_cycles:
        return None
    seg = xerr[hold_mask]
    return {
        "mean_abs_mm": float(np.abs(seg).mean() * 1000.0),
        "final_abs_mm": float(abs(seg[-1]) * 1000.0),
        "n_hold_cycles": int(hold_mask.sum()),
    }


def diagnose(run_dir: Path) -> dict[str, Any]:
    """One-call full diagnostic for a completed trial directory. This is
    what tools/autonomous_transport_explorer.py calls after every real
    trial -- the automated replacement for this session's manual probing."""
    summary, trace = load_trial(run_dir, load_trace=True)
    move_duration_s = float(summary.get("move_duration_s") or 0.0)
    termination_reason = summary.get("termination_reason")
    result: dict[str, Any] = {
        "run_dir": str(run_dir),
        "success": bool(summary.get("success")),
        "termination_reason": termination_reason,
        "guard_category": categorize_termination(termination_reason),
        "target_x_delta_m": summary.get("target_x_delta_m"),
        "achieved_x_delta_m": summary.get("achieved_x_delta_m"),
        "target_accel_mps2": summary.get("target_accel_mps2"),
        "trajectory_profile": summary.get("trajectory_profile"),
        "max_abs_qd_radps": summary.get("max_abs_qd_radps"),
        "pre_trip_trend": summary.get("pre_trip_trend"),
    }
    target = summary.get("target_x_delta_m")
    achieved = summary.get("achieved_x_delta_m")
    if target and achieved is not None and abs(target) > 1e-9:
        result["achieved_fraction"] = achieved / target

    if move_duration_s > 0:
        result["resonance_band_power_fraction"] = resonance_band_power_fraction(trace, move_duration_s)
        result["hold_phase_error"] = hold_phase_error_mm(trace, move_duration_s)
    return result


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, help="Path to a single direct_torque_* run directory.")
    args = parser.parse_args()
    diag = diagnose(args.run_dir)
    print(json.dumps(diag, indent=2, default=str))


if __name__ == "__main__":
    main()
