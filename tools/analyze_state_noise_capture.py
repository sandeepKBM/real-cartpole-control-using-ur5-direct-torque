#!/usr/bin/env python3
"""Derive an empirical noise model for CartesianMoveMonitor's TCP speed/accel
estimate from a stationary state capture (tools/ur5e_capture_state_noise.py).

The robot was stationary and commanding nothing for the whole capture, so
every consecutive-sample step_m/speed_mps/accel_mps2 computed here is, by
construction, exactly what the guard sees with zero real motion -- the
guard's own false-positive noise floor, from thousands of real samples
instead of the ~1-5 real samples available per live hardware trial.

Reuses hardware/safety.py's exact formulas (not a reimplementation) so this
is guaranteed consistent with what CartesianMoveMonitor.check() actually
computes: step_m = ||pos_i - pos_{i-1}||, speed_mps = step_m / dt_s,
accel_mps2 = |speed_i - speed_{i-1}| / dt_s. dt_s here is the REAL measured
elapsed time between consecutive samples (capture rows carry their own
t_s timestamps) -- the ideal case the max(dt_s, measured) fix in
CartesianMoveMonitor approximates when real timing isn't directly known.

Example:
  python tools/analyze_state_noise_capture.py \
    outputs/hardware_state_noise/capture_20260728_154018.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from _bootstrap import ensure_repo_root

ensure_repo_root()


def load_rows(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def compute_guard_quantities(rows: list[dict]) -> dict:
    """step_m/speed_mps/accel_mps2 for every consecutive pair, using each
    pair's own real measured dt (t_s[i] - t_s[i-1]) -- ground truth, no
    caller-supplied nominal dt_s involved at all.
    """
    t = np.array([r["t_s"] for r in rows], dtype=np.float64)
    pos = np.array([r["tcp_pose"][:3] for r in rows], dtype=np.float64)

    dt = np.diff(t)
    if np.any(dt <= 0.0):
        raise ValueError("non-positive dt between consecutive samples -- capture rows out of order or duplicated")

    step_m = np.linalg.norm(np.diff(pos, axis=0), axis=1)
    speed_mps = step_m / dt

    accel_mps2 = np.abs(np.diff(speed_mps)) / dt[1:]

    return {
        "n_samples": len(rows),
        "n_dt": len(dt),
        "n_accel": len(accel_mps2),
        "dt_ms": dt * 1e3,
        "step_m": step_m,
        "speed_mps": speed_mps,
        "accel_mps2": accel_mps2,
    }


def percentiles(x: np.ndarray, ps=(50, 90, 95, 99, 99.9, 99.99, 100)) -> dict:
    return {f"p{p}": float(np.percentile(x, p)) for p in ps}


def recommend_threshold(accel_mps2: np.ndarray, target_percentile: float) -> float:
    """The accel value the false-positive noise floor exceeds only
    (100 - target_percentile)% of the time, in this real capture."""
    return float(np.percentile(accel_mps2, target_percentile))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("capture_path", type=Path)
    p.add_argument(
        "--target-percentile",
        type=float,
        default=99.9,
        help="Recommend a threshold that the real noise floor exceeds only (100-this)%% of the time.",
    )
    args = p.parse_args()

    rows = load_rows(args.capture_path)
    if len(rows) < 3:
        print(f"[error] only {len(rows)} rows -- need at least 3 for a speed+accel estimate")
        return 1

    q = compute_guard_quantities(rows)
    print(f"[loaded] {q['n_samples']} samples, {q['n_dt']} dt intervals, {q['n_accel']} accel estimates")

    dt_ms = q["dt_ms"]
    print("\n[real inter-sample dt, ms] (should cluster near the nominal control period)")
    for k, v in percentiles(dt_ms).items():
        print(f"  {k}: {v:.4f} ms")
    print(f"  mean: {float(np.mean(dt_ms)):.4f} ms, std: {float(np.std(dt_ms)):.4f} ms")

    speed = q["speed_mps"]
    print("\n[TCP speed_mps -- CartesianMoveMonitor's own estimate, robot stationary]")
    for k, v in percentiles(speed).items():
        print(f"  {k}: {v:.6f} m/s")

    accel = q["accel_mps2"]
    print("\n[TCP accel_mps2 -- CartesianMoveMonitor's own estimate, robot stationary]")
    for k, v in percentiles(accel).items():
        print(f"  {k}: {v:.6f} m/s^2")
    print(f"  mean: {float(np.mean(accel)):.6f} m/s^2, std: {float(np.std(accel)):.6f} m/s^2")

    rec = recommend_threshold(accel, args.target_percentile)
    print(
        f"\n[recommendation] max_tcp_accel_mps2 >= {rec:.4f} would keep this real, "
        f"purely-noise-driven false-positive rate at or below "
        f"{100.0 - args.target_percentile:.3f}% per control cycle (empirical, "
        f"{q['n_accel']} real samples, current pose/dynamics-source only -- "
        f"re-run this capture at other poses/control rates before trusting it elsewhere)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
