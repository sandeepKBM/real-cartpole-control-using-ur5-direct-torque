#!/usr/bin/env python3
"""Pose sweep wrapper for the residual-torque move-and-hold transport driver.

Runs the same four "rigor categories" this project already holds the canonical
alpha=0 pose and height_alpha=0.5 to (AGENTS.md sec 3: canonical grid, long
holds, large displacements, torque-scale robustness) at one or more
``height_alpha`` values in between (or outside) those two previously-validated
poses. Each category is a single ``tools/ur5e_move_hold_transport.py``
invocation -- this script only chooses the parameter grid, the start pose (via
``hardware.poses.q_for_height_alpha``), and the gain overrides, then
subprocesses the existing driver exactly like other sweep drivers in this repo
subprocess ``tools/ur5e_mujoco_torque_experiments.py``.

Simulation-only. No RL, hardware, RTDE, URScript, or CoppeliaSim code is
imported.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from hardware.poses import q_for_height_alpha  # noqa: E402
from transport_metrics import GAIN_FIELDS  # noqa: E402

MOVE_HOLD_SCRIPT = REPO_ROOT / "tools" / "ur5e_move_hold_transport.py"

# The four rigor categories from AGENTS.md sec 3 / docs/status/wrist_orientation_task_2026-07-29.md.
# Canonical grid and long holds reproduce the exact literal grids used to
# characterize the alpha=0 and height_alpha=0.5 envelopes. "large_displacements"
# and "torque_scale_robustness" are reconstructed from the documented parameter
# *shapes* (AGENTS.md's own text: "dx up to 0.20 m", "dx 0.03/0.06m x scale
# 0.10-1.00, 7 steps x 2 = 14") since their exact literal commands were never
# preserved verbatim anywhere in this repo (confirmed absent 2026-07-29).
CATEGORY_GRIDS: dict[str, dict[str, list[float]]] = {
    "canonical_grid": {
        "target_x_deltas": [0.01, 0.02, 0.03, 0.04],
        "move_durations": [1.0],
        "hold_durations": [1.0, 2.0],
        "torque_limit_scales": [1.0],
    },
    "long_holds": {
        "target_x_deltas": [0.03, 0.06],
        "move_durations": [1.0],
        "hold_durations": [4.0, 10.0, 20.0, 30.0],
        "torque_limit_scales": [1.0],
    },
    "large_displacements": {
        "target_x_deltas": [0.05, 0.10, 0.15, 0.20],
        "move_durations": [1.0],
        "hold_durations": [1.0, 2.0],
        "torque_limit_scales": [1.0],
    },
    "torque_scale_robustness": {
        "target_x_deltas": [0.03, 0.06],
        "move_durations": [1.0],
        "hold_durations": [2.0],
        "torque_limit_scales": [0.10, 0.25, 0.40, 0.55, 0.70, 0.85, 1.00],
    },
}
CATEGORY_ORDER: tuple[str, ...] = (
    "canonical_grid",
    "long_holds",
    "large_displacements",
    "torque_scale_robustness",
)


def _default_output_root() -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return REPO_ROOT / "outputs" / "ur5e_mujoco_torque_transport" / f"pose_sweep_transport_{stamp}"


def _fmt_alpha_token(alpha: float) -> str:
    return f"{float(alpha):g}".replace(".", "p").replace("-", "m")


def pose_for_alpha(alpha: float) -> np.ndarray:
    """Thin wrapper around hardware.poses.q_for_height_alpha for local testability."""
    return q_for_height_alpha(alpha)


def gains_from_config(config_path: Path) -> dict[str, float]:
    """Load controller.gains from a base config YAML and filter to GAIN_FIELDS.

    tools/ur5e_move_hold_transport.py always overwrites its base-config gains
    with its own BASELINE_GAINS (kp_x=80, ...) unless ``--gain-overrides-json``
    is passed explicitly (see docs/status/bug_audit_2026-07-29.md and
    docs/status/wrist_orientation_task_2026-07-29.md sec 4). To actually
    exercise the gains a named config (e.g. the tuned wrist-orient config)
    ships with, those gains must be re-supplied on the command line. This
    helper extracts them from the config so callers don't have to hardcode
    them.
    """
    with config_path.open("r", encoding="utf-8") as fp:
        cfg = yaml.safe_load(fp)
    controller_cfg = cfg.get("controller", {}) if isinstance(cfg, dict) else {}
    gains_cfg = controller_cfg.get("gains", {}) if isinstance(controller_cfg, dict) else {}
    return {name: float(gains_cfg[name]) for name in GAIN_FIELDS if name in gains_cfg}


def build_move_hold_command(
    *,
    config: Path,
    output_root: Path,
    seed: int,
    gain_overrides: dict[str, float] | None,
    start_q_rad: np.ndarray,
    category_params: dict[str, list[float]],
    no_plot: bool = True,
) -> list[str]:
    """Build the exact argv used to subprocess tools/ur5e_move_hold_transport.py.

    Pure function (no subprocess call) so command construction can be unit
    tested without running the simulator.
    """
    cmd = [
        sys.executable,
        str(MOVE_HOLD_SCRIPT),
        "--config",
        str(config),
        "--output-root",
        str(output_root),
        "--seed",
        str(int(seed)),
        "--target-x-deltas",
        *[str(float(v)) for v in category_params["target_x_deltas"]],
        "--move-durations",
        *[str(float(v)) for v in category_params["move_durations"]],
        "--hold-durations",
        *[str(float(v)) for v in category_params["hold_durations"]],
        "--torque-limit-scales",
        *[str(float(v)) for v in category_params["torque_limit_scales"]],
        "--start-q-rad",
        *[str(float(v)) for v in np.asarray(start_q_rad, dtype=np.float64).tolist()],
    ]
    if gain_overrides:
        cmd.extend(["--gain-overrides-json", json.dumps(gain_overrides)])
    if no_plot:
        cmd.append("--no-plot")
    return cmd


def parse_pass_count(summary: dict[str, Any]) -> tuple[int, int]:
    """Extract (num_valid_move_and_hold, num_runs) from a move-hold summary.json dict."""
    return int(summary.get("num_valid_move_and_hold", 0)), int(summary.get("num_runs", 0))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--height-alphas",
        nargs="+",
        type=float,
        required=True,
        help="height_alpha values in [0,1] to sweep (hardware.poses.q_for_height_alpha).",
    )
    p.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Base MuJoCo transport config YAML (e.g. config/ur5e_mujoco_torque_osc_tuned_wrist_orient.yaml).",
    )
    p.add_argument(
        "--categories",
        nargs="+",
        choices=CATEGORY_ORDER,
        default=list(CATEGORY_ORDER),
        help="Subset of rigor categories to run (default: all four).",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Sweep output directory. Defaults under outputs/ur5e_mujoco_torque_transport/.",
    )
    p.add_argument(
        "--no-gain-overrides",
        action="store_true",
        help="Do not auto-extract --config's controller.gains as --gain-overrides-json "
        "(NOT recommended: without this, the child driver silently substitutes its own "
        "BASELINE_GAINS instead of the gains the named config actually ships with).",
    )
    p.add_argument("--plot", action="store_true", help="Allow child driver plot generation (default: --no-plot).")
    return p.parse_args()


def run() -> int:
    args = parse_args()
    output_root = args.output_root if args.output_root is not None else _default_output_root()
    output_root = output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    gain_overrides = None if args.no_gain_overrides else gains_from_config(args.config)

    results: dict[str, dict[str, Any]] = {}
    table_rows: list[dict[str, Any]] = []
    for alpha in args.height_alphas:
        alpha = float(alpha)
        start_q = pose_for_alpha(alpha)
        alpha_token = _fmt_alpha_token(alpha)
        results[alpha_token] = {}
        for category in args.categories:
            category_params = CATEGORY_GRIDS[category]
            category_output_root = output_root / f"alpha_{alpha_token}" / category
            cmd = build_move_hold_command(
                config=args.config,
                output_root=category_output_root,
                seed=args.seed,
                gain_overrides=gain_overrides,
                start_q_rad=start_q,
                category_params=category_params,
                no_plot=not args.plot,
            )
            print(f"[pose_sweep] alpha={alpha} category={category}: {' '.join(cmd)}", file=sys.stderr)
            completed = subprocess.run(cmd, cwd=str(REPO_ROOT), check=False, text=True, capture_output=True)
            summary_path = category_output_root / "summary.json"
            if completed.returncode != 0 or not summary_path.exists():
                raise RuntimeError(
                    f"Category run failed for alpha={alpha}, category={category!r}. "
                    f"Return code={completed.returncode}. stdout={completed.stdout!r}. stderr={completed.stderr!r}"
                )
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            num_valid, num_runs = parse_pass_count(summary)
            results[alpha_token][category] = {
                "num_valid_move_and_hold": num_valid,
                "num_runs": num_runs,
                "summary_path": str(summary_path),
            }
            table_rows.append(
                {
                    "height_alpha": alpha,
                    "category": category,
                    "num_valid_move_and_hold": num_valid,
                    "num_runs": num_runs,
                    "pass_fraction": f"{num_valid}/{num_runs}",
                }
            )
            print(f"[pose_sweep] alpha={alpha} category={category}: {num_valid}/{num_runs}", file=sys.stderr)

    summary_out = {
        "output_root": str(output_root),
        "config_path": str(args.config),
        "height_alphas": [float(a) for a in args.height_alphas],
        "categories": list(args.categories),
        "gain_overrides": gain_overrides,
        "results": results,
        "table_rows": table_rows,
    }
    (output_root / "pose_sweep_summary.json").write_text(json.dumps(summary_out, indent=2), encoding="utf-8")

    print("\nalpha x category pass table:")
    header = f"{'alpha':>8} | " + " | ".join(f"{c:>24}" for c in args.categories)
    print(header)
    for alpha_token, cat_results in results.items():
        cells = [
            f"{cat_results[c]['num_valid_move_and_hold']}/{cat_results[c]['num_runs']}" for c in args.categories
        ]
        row = f"{alpha_token:>8} | " + " | ".join(f"{cell:>24}" for cell in cells)
        print(row)

    return 0


if __name__ == "__main__":
    raise SystemExit(run())
