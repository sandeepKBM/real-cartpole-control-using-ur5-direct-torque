#!/usr/bin/env python3
"""Sweep height x target-x-delta with the fixed tuned baseline to find, for
each height, the largest X-axis move-and-hold displacement that still
validates. Ad-hoc workspace characterization (not part of the training
pipeline) -- reuses the same interpolated start poses as GainSchedulingEnv
and the same baseline-CLI-subprocess pattern as eval_gain_scheduler.py.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from rl_gain_scheduling.gain_scheduling_env import ACTIVE_ORIGIN_Q, LOWER_B_Q  # noqa: E402
from transport_metrics import compute_valid_move_hold_metrics  # noqa: E402

CONFIG_PATH = REPO_ROOT / "config" / "ur5e_mujoco_torque_osc_tuned.yaml"
ALPHAS = [0.0, 0.25, 0.5, 0.75, 1.0]
DELTAS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35]
MOVE_DURATION_S = 1.0
HOLD_DURATION_S = 2.0


def interpolated_q(alpha: float):
    return (1.0 - alpha) * ACTIVE_ORIGIN_Q + alpha * LOWER_B_Q


def run_one(alpha: float, dx: float, output_dir: Path) -> dict:
    q_start = interpolated_q(alpha)
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(REPO_ROOT / "tools" / "ur5e_mujoco_torque_experiments.py"),
        "--mode", "controller-rollout",
        "--controller-kind", "impedance",
        "--config", str(CONFIG_PATH),
        "--trajectory-profile", "min_jerk_move_hold",
        "--move-duration", str(MOVE_DURATION_S),
        "--duration", str(MOVE_DURATION_S + HOLD_DURATION_S),
        "--target-x-delta", str(dx),
        "--start-q-rad", *[str(float(v)) for v in q_start],
        "--seed", "0",
        "--no-plot",
        "--output-dir", str(output_dir),
    ]
    completed = subprocess.run(cmd, cwd=str(REPO_ROOT), check=False, text=True, capture_output=True)
    if completed.returncode != 0:
        return {"valid_move_and_hold": False, "error": completed.stderr[-2000:]}
    run_dir = max((p.parent for p in output_dir.rglob("summary.json")), key=lambda p: p.stat().st_mtime)
    summary = json.loads((run_dir / "summary.json").read_text())
    summary.update(compute_valid_move_hold_metrics(summary, strict=False))
    return summary


def main() -> int:
    output_root = REPO_ROOT / "outputs" / "rl_gain_scheduling" / "x_envelope_by_height"
    results: dict[float, dict[float, bool]] = {a: {} for a in ALPHAS}
    for alpha in ALPHAS:
        for dx in DELTAS:
            cell = f"alpha{alpha:g}_dx{dx:g}"
            out_dir = output_root / cell
            summary = run_one(alpha, dx, out_dir)
            valid = bool(summary.get("valid_move_and_hold", False))
            results[alpha][dx] = valid
            print(f"{cell}: valid={valid} achieved_x_delta_m={summary.get('achieved_x_delta_m', 'n/a')}", flush=True)

    print("\n=== Max valid X displacement per height ===")
    print(f"{'height_alpha':>12} | {'max_valid_dx_m':>15}")
    for alpha in ALPHAS:
        valid_deltas = [dx for dx, ok in results[alpha].items() if ok]
        max_valid = max(valid_deltas) if valid_deltas else None
        print(f"{alpha:>12.2f} | {max_valid if max_valid is not None else 'NONE VALID':>15}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
