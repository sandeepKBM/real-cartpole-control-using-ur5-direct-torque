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


def compute_guard_quantities(rows: list[dict], *, gap: int = 1, alpha: float = 1.0) -> dict:
    """step_m/speed_mps/accel_mps2 for every consecutive pair, using each
    pair's own real measured dt (t_s[i] - t_s[i-1]) -- ground truth, no
    caller-supplied nominal dt_s involved at all.

    ``gap``/``alpha`` mirror ``hardware/safety.py``'s
    ``CartesianMoveLimits.accel_gap_cycles``/``speed_lowpass_alpha`` exactly
    (corrected-clock gap-window + EMA smoothing feeding the accel estimate;
    defaults gap=1/alpha=1.0 reproduce the original single-cycle,
    unfiltered accel_mps2 computation). This is a parallel, vectorized
    reimplementation of that same math (not a direct call into
    CartesianMoveMonitor, which drives its timing off wall-clock time, not
    a capture file's recorded timestamps) -- cross-checked against the live
    class in tests/hardware/test_analyze_state_noise_capture.py.
    """
    if gap < 1:
        raise ValueError("gap must be >= 1")
    if not (0.0 < alpha <= 1.0):
        raise ValueError("alpha must be in (0.0, 1.0]")

    t = np.array([r["t_s"] for r in rows], dtype=np.float64)
    pos = np.array([r["tcp_pose"][:3] for r in rows], dtype=np.float64)

    dt = np.diff(t)
    if np.any(dt <= 0.0):
        raise ValueError("non-positive dt between consecutive samples -- capture rows out of order or duplicated")

    step_m = np.linalg.norm(np.diff(pos, axis=0), axis=1)
    speed_mps = step_m / dt

    if gap == 1:
        gap_step_m = step_m
        gap_dt = dt
    else:
        gap_step_m = np.linalg.norm(pos[gap:] - pos[:-gap], axis=1)
        # corrected-clock span across `gap` real cycles, from cumulative dt.
        cum_t = np.concatenate([[0.0], np.cumsum(dt)])
        gap_dt = cum_t[gap:] - cum_t[:-gap]
    raw_gap_speed_mps = gap_step_m / gap_dt

    if alpha >= 1.0:
        gap_speed_mps = raw_gap_speed_mps
    else:
        gap_speed_mps = np.empty_like(raw_gap_speed_mps)
        gap_speed_mps[0] = raw_gap_speed_mps[0]
        for i in range(1, len(raw_gap_speed_mps)):
            gap_speed_mps[i] = alpha * raw_gap_speed_mps[i] + (1.0 - alpha) * gap_speed_mps[i - 1]

    # accel_i = |gap_speed_i - gap_speed_{i-1}| / (single-cycle real_dt at
    # the point gap_speed_i was formed) -- dt aligned to gap_speed's own
    # index: gap_speed[k] corresponds to sample index (gap + k) in the
    # original series, so its single-cycle dt is dt[gap + k - 1].
    accel_dt = dt[gap:]
    accel_mps2 = np.abs(np.diff(gap_speed_mps)) / accel_dt[: len(gap_speed_mps) - 1]

    return {
        "n_samples": len(rows),
        "n_dt": len(dt),
        "n_accel": len(accel_mps2),
        "dt_ms": dt * 1e3,
        "step_m": step_m,
        "speed_mps": speed_mps,
        "gap_speed_mps": gap_speed_mps,
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
    p.add_argument(
        "--accel-gap-cycles",
        type=int,
        default=1,
        help="Mirrors CartesianMoveLimits.accel_gap_cycles (default 1 = original behavior).",
    )
    p.add_argument(
        "--speed-lowpass-alpha",
        type=float,
        default=1.0,
        help="Mirrors CartesianMoveLimits.speed_lowpass_alpha (default 1.0 = no filtering).",
    )
    args = p.parse_args()

    rows = load_rows(args.capture_path)
    if len(rows) < 3:
        print(f"[error] only {len(rows)} rows -- need at least 3 for a speed+accel estimate")
        return 1

    q = compute_guard_quantities(rows, gap=int(args.accel_gap_cycles), alpha=float(args.speed_lowpass_alpha))
    print(
        f"[loaded] {q['n_samples']} samples, {q['n_dt']} dt intervals, {q['n_accel']} accel estimates "
        f"(accel_gap_cycles={args.accel_gap_cycles}, speed_lowpass_alpha={args.speed_lowpass_alpha})"
    )

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
