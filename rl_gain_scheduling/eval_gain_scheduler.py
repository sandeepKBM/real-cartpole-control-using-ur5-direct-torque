#!/usr/bin/env python3
"""Compare a trained gain-scheduling policy against the fixed tuned config.

Sweeps a height x target-x-delta grid. For each cell, runs (a) the learned
policy through GainSchedulingEnv with full_trace_logging=True, and (b) the
fixed --baseline-config gains (default config/ur5e_mujoco_torque_osc_tuned.yaml)
via the unmodified CLI script (tools/ur5e_mujoco_torque_experiments.py),
starting from the same interpolated pose. Both log through
observability.run_logger.RunLogger -- never a bespoke summary schema (the
anti-pattern the archived CoppeliaSim eval script committed).

--baseline-config MUST match the controller flags of whatever config the
policy was actually trained/evaluated against (--config) -- the default only
reproduces the plain-tuned-config problem instance, not any variant that
sets lambda_diagonal_shaping/lambda_adaptive_regularization etc. A mismatch
here silently compares the learned policy against a different physical
problem than the one in --config (found and fixed 2026-07-28: the original
alpha=0.5 directional-asymmetry training run's own comparison used this
default against an env config that had already moved to the adaptive-lambda
controller, so the "baseline" side never reproduced the documented failure).

Prints a per-height valid_move_and_hold-rate comparison table: this is the
artifact that demonstrates the feature's goal -- the fixed config should
degrade at the low pose (this session's original finding), the learned
scheduler should hold up there without regressing at the tall pose.
"""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from stable_baselines3 import PPO, SAC  # noqa: E402
from stable_baselines3.common.base_class import BaseAlgorithm  # noqa: E402

from observability.run_logger import RunLogger  # noqa: E402
from simulation.ur5e_mujoco_torque import build_safety_config  # noqa: E402
from rl_gain_scheduling.gain_scheduling_env import (  # noqa: E402
    ACTIVE_ORIGIN_Q,
    LOWER_B_Q,
    GainSchedulingEnv,
)
from transport_metrics import compute_valid_move_hold_metrics  # noqa: E402

DEFAULT_CONFIG_PATH = REPO_ROOT / "config" / "rl_gain_scheduling.yaml"
DEFAULT_BASELINE_CONFIG_PATH = REPO_ROOT / "config" / "ur5e_mujoco_torque_osc_tuned.yaml"
DEFAULT_ALPHAS = (0.0, 0.25, 0.5, 0.75, 1.0)
DEFAULT_DELTAS = (0.05, 0.10, 0.15, 0.20)


def _interpolated_start_q(alpha: float) -> np.ndarray:
    return (1.0 - alpha) * ACTIVE_ORIGIN_Q + alpha * LOWER_B_Q


def _run_learned(
    *,
    model: BaseAlgorithm,
    env: GainSchedulingEnv,
    alpha: float,
    dx: float,
    output_dir: Path,
) -> dict[str, Any]:
    obs, info = env.reset(seed=0, options={"height_alpha": alpha, "target_x_delta": dx, "output_dir": str(output_dir)})
    terminated = truncated = False
    while not (terminated or truncated):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
    env.close()
    summary_path = output_dir / "summary.json"
    return json.loads(summary_path.read_text(encoding="utf-8"))


def _run_baseline(
    *,
    alpha: float,
    dx: float,
    move_duration_s: float,
    max_episode_seconds: float,
    output_dir: Path,
    baseline_config_path: Path,
) -> dict[str, Any]:
    q_start = _interpolated_start_q(alpha)
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(REPO_ROOT / "tools" / "ur5e_mujoco_torque_experiments.py"),
        "--mode", "controller-rollout",
        "--controller-kind", "impedance",
        "--config", str(baseline_config_path),
        "--trajectory-profile", "min_jerk_move_hold",
        "--move-duration", str(move_duration_s),
        "--duration", str(max_episode_seconds),
        "--target-x-delta", str(dx),
        "--start-q-rad", *[str(float(v)) for v in q_start],
        "--seed", "0",
        "--no-plot",
        "--output-dir", str(output_dir),
    ]
    # NOTE: the CLI script's own exit code is 1 whenever safety_pass=false --
    # that's a normal "ran to completion, but didn't validate" outcome (e.g.
    # this exact dx/height combo genuinely fails the transport envelope), not
    # a crash. Only treat it as a real crash if summary.json wasn't written.
    completed = subprocess.run(cmd, cwd=str(REPO_ROOT), check=False, text=True, capture_output=True)
    if not list(output_dir.rglob("summary.json")):
        raise RuntimeError(f"baseline CLI run produced no summary.json (exit {completed.returncode}): {completed.stderr}")
    run_dir = max((p.parent for p in output_dir.rglob("summary.json")), key=lambda p: p.stat().st_mtime)
    return json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))


def main() -> int:
    import argparse

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-path", type=Path, required=True, help="Path to a trained model .zip.")
    p.add_argument(
        "--algo",
        type=str,
        choices=["ppo", "sac"],
        default="ppo",
        help="Algorithm the model was trained with (must match, since PPO/SAC use different "
        "policy classes and PPO.load() on a SAC checkpoint raises). Default 'ppo' reproduces "
        "this script's original behavior exactly for existing PPO models.",
    )
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    p.add_argument(
        "--baseline-config",
        type=Path,
        default=DEFAULT_BASELINE_CONFIG_PATH,
        help=(
            "Fixed-gain config the baseline CLI run uses (default: "
            "ur5e_mujoco_torque_osc_tuned.yaml). Must match --config's own "
            "controller flags (task_space_inertia_shaping, lambda_diagonal_shaping, "
            "lambda_adaptive_regularization, etc.) for the comparison to mean "
            "anything -- a mismatch silently compares the learned policy against "
            "a different physical problem than the one it trained on."
        ),
    )
    p.add_argument("--alphas", type=float, nargs="+", default=list(DEFAULT_ALPHAS))
    p.add_argument("--deltas", type=float, nargs="+", default=list(DEFAULT_DELTAS))
    p.add_argument("--output-root", type=Path, default=None)
    p.add_argument("--run-name", type=str, default=None)
    args = p.parse_args()

    run_name = args.run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = (args.output_root or (REPO_ROOT / "outputs" / "rl_gain_scheduling" / "eval")) / run_name
    output_root.mkdir(parents=True, exist_ok=True)

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    move_duration_s = float(cfg["env"]["move_duration_s"])
    max_episode_seconds = float(cfg["env"]["max_episode_seconds"])

    model_cls = {"ppo": PPO, "sac": SAC}[args.algo]
    model = model_cls.load(str(args.model_path))
    env = GainSchedulingEnv(config_path=args.config)
    env._full_trace_logging = True

    # Real safety_cfg for each side's own config, not RunLogger's default --
    # see observability/run_logger.py's RunLogger docstring / 2026-07-30 fix
    # for why the default silently produced wrong time-to-limit values.
    # learned/baseline use DIFFERENT config files (--config vs
    # --baseline-config, see that flag's docstring above), so each needs its
    # own safety_cfg, not one shared between them.
    baseline_cfg = yaml.safe_load(args.baseline_config.read_text(encoding="utf-8"))
    learned_logger = RunLogger(
        output_root=output_root / "learned", source_script=__file__, backend="mujoco",
        safety_cfg=build_safety_config(cfg["controller"]),
    )
    baseline_logger = RunLogger(
        output_root=output_root / "baseline", source_script=__file__, backend="mujoco",
        safety_cfg=build_safety_config(baseline_cfg["controller"]),
    )

    results: dict[tuple[float, float], dict[str, bool]] = {}
    for alpha in args.alphas:
        for dx in args.deltas:
            cell_label = f"alpha{alpha:g}_dx{dx:g}"
            print(f"=== {cell_label} ===")

            learned_dir = output_root / "learned" / "runs" / cell_label
            learned_summary = _run_learned(model=model, env=env, alpha=alpha, dx=dx, output_dir=learned_dir)
            learned_logger.log_run(learned_summary, run_dir=learned_dir, seed=0, config_path=args.config, run_label=cell_label)
            learned_valid = bool(learned_summary.get("valid_move_and_hold", False))

            baseline_dir = output_root / "baseline" / "runs" / cell_label
            baseline_summary = _run_baseline(
                alpha=alpha, dx=dx, move_duration_s=move_duration_s,
                max_episode_seconds=max_episode_seconds, output_dir=baseline_dir,
                baseline_config_path=args.baseline_config,
            )
            baseline_summary.update(compute_valid_move_hold_metrics(baseline_summary, strict=False))
            baseline_logger.log_run(baseline_summary, run_dir=baseline_dir, seed=0, config_path=args.baseline_config, run_label=cell_label)
            baseline_valid = bool(baseline_summary.get("valid_move_and_hold", False))

            print(f"    learned:  valid={learned_valid}  quality={learned_summary.get('move_hold_quality_score', 0.0):.3f}")
            print(f"    baseline: valid={baseline_valid}  quality={baseline_summary.get('move_hold_quality_score', 0.0):.3f}")
            results[(alpha, dx)] = {"learned": learned_valid, "baseline": baseline_valid}

    learned_logger.write_sweep_csv_snapshot()
    baseline_logger.write_sweep_csv_snapshot()

    print("\n=== Comparison: valid_move_and_hold rate per height (across all dx) ===")
    print(f"{'height_alpha':>12} | {'learned':>10} | {'baseline':>10}")
    for alpha in args.alphas:
        cells = [results[(alpha, dx)] for dx in args.deltas]
        learned_rate = sum(c["learned"] for c in cells) / len(cells)
        baseline_rate = sum(c["baseline"] for c in cells) / len(cells)
        print(f"{alpha:>12.2f} | {learned_rate:>9.0%} | {baseline_rate:>9.0%}")

    print(f"\nFull records under {output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
