#!/usr/bin/env python3
"""Suggest the next gain candidate to try on the real UR5e, from past trials.

Live-retuning helper for the direct-torque hardware lane: reads every past
trial's summary.json under --trials-dir (each written automatically by
tools/ur5e_direct_torque_x_transport.py --gain-overrides-json ...), replays
them into an Optuna study, and asks for one new candidate within
--search-space-json's bounds. Prints the exact --gain-overrides-json value
and a ready-to-copy CLI command for the next trial -- it does NOT run
anything on the robot itself.

This tool is stateless between invocations: no separate study database to
keep in sync. Every run rebuilds history from the summary.json files already
on disk, so there is nothing to corrupt or lose track of mid-session.

Objective: maximize transport_metrics.py's own move_hold_quality_score
(already computed into every trial's summary.json by
compute_valid_move_hold_metrics -- the exact metric used throughout this
project's sim-side tuning). A trial where valid_move_and_hold is False (a
safety guard tripped or the run didn't complete) is scored -1.0 regardless
of its raw quality_score, so a completed trial always outranks an aborted
one.

Example:
  # after running two-three trials with tools/ur5e_direct_torque_x_transport.py
  # --gain-overrides-json '...' --output-dir outputs/retune/trial_1 (etc.)
  python tools/ur5e_suggest_gains.py \\
    --trials-dir outputs/retune \\
    --search-space-json '{"kp_x": [200, 800], "kd_x": [20, 80], "kd_joint": [2, 10]}' \\
    --seed 0
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from _bootstrap import ensure_repo_root

ensure_repo_root()

from transport_metrics import GAIN_FIELDS  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_trial_summaries(trials_dir: Path) -> list[dict[str, Any]]:
    summaries = []
    for path in sorted(trials_dir.rglob("summary.json")):
        try:
            summaries.append(json.loads(path.read_text(encoding="utf-8")))
        except (json.JSONDecodeError, OSError) as exc:
            print(f"warning: skipping unreadable {path}: {exc}", file=sys.stderr)
    return summaries


def _parse_search_space(raw: str) -> dict[str, tuple[float, float]]:
    path = Path(raw)
    text = path.read_text(encoding="utf-8") if path.exists() else raw
    payload = json.loads(text)
    if not isinstance(payload, dict) or not payload:
        raise ValueError("--search-space-json must decode to a non-empty JSON object")
    space: dict[str, tuple[float, float]] = {}
    for key, bounds in payload.items():
        if key not in GAIN_FIELDS:
            raise ValueError(f"{key!r} is not a schedulable gain field; valid fields: {sorted(GAIN_FIELDS)}")
        if not isinstance(bounds, (list, tuple)) or len(bounds) != 2:
            raise ValueError(f"bounds for {key!r} must be a [low, high] pair; got {bounds!r}")
        low, high = float(bounds[0]), float(bounds[1])
        if not (low < high):
            raise ValueError(f"bounds for {key!r} must have low < high; got [{low}, {high}]")
        space[key] = (low, high)
    return space


def score_trial(summary: dict[str, Any]) -> float:
    """Higher is better. A trial that never validly completed the move+hold
    (guard tripped, or the run just failed outright) always scores below any
    trial that did -- see module docstring."""
    if not bool(summary.get("valid_move_and_hold", False)):
        return -1.0
    return float(summary.get("move_hold_quality_score", 0.0))


def build_study(search_space: dict[str, tuple[float, float]], summaries: list[dict[str, Any]], *, seed: int):
    import optuna
    from optuna.distributions import FloatDistribution
    from optuna.samplers import TPESampler

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=seed))
    distributions = {name: FloatDistribution(low, high) for name, (low, high) in search_space.items()}

    skipped = 0
    for summary in summaries:
        overrides = summary.get("gain_overrides") or {}
        if not overrides or not set(overrides).issuperset(search_space.keys()):
            # A trial that didn't override every gain in the current search
            # space isn't directly comparable (missing dimensions) -- skip it
            # rather than guessing a value for the missing ones.
            skipped += 1
            continue
        params = {name: float(overrides[name]) for name in search_space}
        if any(not (distributions[name].low <= v <= distributions[name].high) for name, v in params.items()):
            skipped += 1
            continue
        trial = optuna.trial.create_trial(
            params=params, distributions=distributions, value=score_trial(summary),
        )
        study.add_trial(trial)

    return study, skipped


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--trials-dir", type=Path, required=True, help="Directory to search recursively for summary.json files.")
    p.add_argument("--search-space-json", required=True, help="Path to a JSON file or inline JSON object: {gain_name: [low, high]}.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--robot-ip", default=None,
        help="If given, also print a ready-to-copy tools/ur5e_direct_torque_x_transport.py command.",
    )
    args = p.parse_args()

    search_space = _parse_search_space(args.search_space_json)
    summaries = _load_trial_summaries(args.trials_dir)
    study, skipped = build_study(search_space, summaries, seed=args.seed)

    n_used = len(study.trials)
    print(f"Loaded {len(summaries)} trial summar{'y' if len(summaries) == 1 else 'ies'} "
          f"from {args.trials_dir} ({n_used} usable, {skipped} skipped -- missing/out-of-range gains).")
    if study.trials:
        best = study.best_trial
        print(f"Best so far: score={best.value:.4f} params={best.params}")

    # ask() registers a new trial in-memory; we only need its suggested
    # params. The study is never persisted (stateless by design -- see module
    # docstring), so there is nothing to reconcile: this process exits right
    # after printing, and next invocation rebuilds history from disk.
    trial = study.ask()
    suggested = {name: trial.suggest_float(name, low, high) for name, (low, high) in search_space.items()}

    print("\nNext candidate to try:")
    print(f"  --gain-overrides-json '{json.dumps(suggested)}'")

    if args.robot_ip:
        print("\nReady-to-copy command (edit --target-x-delta/--move-duration/--start-q-rad as needed):")
        print(
            "  python tools/ur5e_direct_torque_x_transport.py "
            f"--robot-ip {args.robot_ip} --control-mode direct_torque --dynamics-source local \\\n"
            f"    --gain-overrides-json '{json.dumps(suggested)}' \\\n"
            "    --i-understand-this-moves-the-robot"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
