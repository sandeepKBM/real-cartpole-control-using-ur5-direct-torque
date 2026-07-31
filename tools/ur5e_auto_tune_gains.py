#!/usr/bin/env python3
"""Batch, sim-gated candidate proposer for the real-UR5e auto-tuning loop.

Closes the gap between the two pieces of the auto-tuning loop that already
existed (see docs/hardware/AUTO_TUNING_PLAN.md M1/M2/M3):
  - tools/ur5e_suggest_gains.py already builds a stateless Optuna TPE study
    from past trials' summary.json files and proposes ONE candidate.
  - tools/ur5e_move_hold_transport.py already IS the sim gate (M2): run a
    candidate gain vector through MuJoCo first, check valid_move_and_hold.

What was missing: a batch of several candidates proposed together, each one
automatically sim-gated before a human ever sees it, so the human only has to
approve/reject an already-sim-filtered batch instead of manually running each
candidate through the sim gate by hand.

Flow: propose --batch-size candidates via study.ask() (no study.tell() in
between -- see "Batch diversity" below) -> sim-gate every candidate via a
subprocess call to tools/ur5e_move_hold_transport.py at the already-proven-safe
dx=0.10-0.20m range -> print only the sim-passing candidates as a numbered,
human-approvable batch -> only after an explicit typed confirmation, print the
ready-to-copy tools/ur5e_direct_torque_x_transport.py command for each -- for a
human to run themselves. This script never executes anything on real hardware;
it does not import hardware.direct_torque_transport, hardware.link, or any
other real-motion-capable module (see tests/hardware/test_auto_tune_gains.py's
static import check).

Batch diversity (Optuna ask()/tell() semantics): calling study.ask() N times
without an intervening study.tell() does NOT produce duplicate or degenerate
candidates. Verified directly against this repo's optuna (4.1.0) install:
cold-start (no prior trials) draws are independent-sampler (default random
until TPESampler's n_startup_trials, 10, prior COMPLETE trials exist) so they
differ trivially; warm-start (>=10 prior COMPLETE trials already loaded from
--trials-dir, so TPE's model is already fit) draws still differ from each
other on every run -- each ask() re-samples from the fitted Parzen-estimator
posterior using the sampler's own advancing RNG state, it is not memoized per
call. Six sequential ask()-without-tell() calls in both regimes produced six
numerically distinct candidate dicts (see the module test suite for the
reproducible assertion). Only a batch size that exceeds the space's meaningful
diversity, or a degenerate (near-zero-width) search space, would collapse
this -- not the ask/tell pattern itself.

No RL, hardware, RTDE, URScript, or CoppeliaSim code is imported.
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

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "tools"))

from ur5e_suggest_gains import (  # noqa: E402
    _load_trial_summaries,
    _parse_search_space,
    build_study,
    score_trial,
)

MOVE_HOLD_SCRIPT = REPO_ROOT / "tools" / "ur5e_move_hold_transport.py"
DEFAULT_CONFIG = REPO_ROOT / "config" / "ur5e_mujoco_torque_osc_tuned.yaml"

# The plan's "reliability within the already-proven range" decision
# (docs/hardware/AUTO_TUNING_PLAN.md, "Decisions already made"): auto-tuning
# trials are only ever evaluated at displacements already validated in sim,
# not pushed toward the ~0.25-0.3m structural ceiling.
PROVEN_SAFE_DX_RANGE_M = (0.10, 0.20)
DEFAULT_SIM_TARGET_X_DELTAS = [0.10, 0.15, 0.20]


def _default_output_root() -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return REPO_ROOT / "outputs" / "ur5e_mujoco_torque_transport" / f"auto_tune_gains_{stamp}"


def validate_sim_target_x_deltas(deltas: list[float]) -> None:
    """Raise if any requested sim-gate displacement falls outside the
    already-proven-safe dx=0.10-0.20m range (AUTO_TUNING_PLAN.md decision --
    not something this tool should silently widen)."""
    low, high = PROVEN_SAFE_DX_RANGE_M
    eps = 1.0e-9
    bad = [d for d in deltas if not (low - eps <= float(d) <= high + eps)]
    if bad:
        raise ValueError(
            f"--sim-target-x-deltas {bad} fall outside the proven-safe range "
            f"[{low}, {high}] m (docs/hardware/AUTO_TUNING_PLAN.md). This tool "
            "only proposes real-hardware candidates within that range."
        )


def propose_batch(study: Any, search_space: dict[str, tuple[float, float]], batch_size: int) -> list[dict[str, float]]:
    """Draw batch_size candidates via study.ask() with no intervening
    study.tell() -- see module docstring for why this still yields distinct
    candidates. Pure w.r.t. everything except the study's own RNG state."""
    candidates: list[dict[str, float]] = []
    for _ in range(batch_size):
        trial = study.ask()
        candidates.append({name: trial.suggest_float(name, low, high) for name, (low, high) in search_space.items()})
    return candidates


def build_sim_gate_command(
    *,
    config: Path,
    output_root: Path,
    seed: int,
    gain_overrides: dict[str, float],
    target_x_deltas: list[float],
    move_duration: float,
    hold_duration: float,
    torque_limit_scale: float,
    start_q_rad: list[float] | None = None,
) -> list[str]:
    """Build the exact argv used to subprocess tools/ur5e_move_hold_transport.py
    for one candidate's sim gate. Pure function (no subprocess call), matching
    tools/ur5e_pose_sweep_transport.py's build_move_hold_command precedent, so
    the invocation shape can be unit tested without running the simulator."""
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
        *[str(float(v)) for v in target_x_deltas],
        "--move-durations",
        str(float(move_duration)),
        "--hold-durations",
        str(float(hold_duration)),
        "--torque-limit-scales",
        str(float(torque_limit_scale)),
        "--gain-overrides-json",
        json.dumps(gain_overrides),
        "--no-plot",
    ]
    if start_q_rad is not None:
        cmd.extend(["--start-q-rad", *[str(float(v)) for v in start_q_rad]])
    return cmd


def run_sim_gate(
    *,
    candidate_gains: dict[str, float],
    candidate_output_root: Path,
    config: Path,
    seed: int,
    target_x_deltas: list[float],
    move_duration: float,
    hold_duration: float,
    torque_limit_scale: float,
    start_q_rad: list[float] | None = None,
) -> dict[str, Any]:
    """Run one candidate through the sim gate (tools/ur5e_move_hold_transport.py)
    and score it. Reuses ur5e_suggest_gains.score_trial per evaluated dx point
    so "did this candidate sim-gate cleanly" uses the exact same
    valid-beats-any-quality-score rule as real-trial scoring: any evaluated dx
    point that didn't validly complete (guard tripped, etc.) is a hard reject
    for the whole candidate, matching the plan's "any real safety-guard abort
    is a hard reject" rule applied one step earlier, in sim."""
    cmd = build_sim_gate_command(
        config=config,
        output_root=candidate_output_root,
        seed=seed,
        gain_overrides=candidate_gains,
        target_x_deltas=target_x_deltas,
        move_duration=move_duration,
        hold_duration=hold_duration,
        torque_limit_scale=torque_limit_scale,
        start_q_rad=start_q_rad,
    )
    completed = subprocess.run(cmd, cwd=str(REPO_ROOT), check=False, text=True, capture_output=True)
    summary_path = candidate_output_root / "summary.json"
    if completed.returncode != 0 or not summary_path.exists():
        return {
            "sim_pass": False,
            "gate_score": -1.0,
            "num_valid_move_and_hold": 0,
            "num_runs": 0,
            "error": (
                f"sim gate subprocess failed (returncode={completed.returncode}). "
                f"stdout={completed.stdout!r} stderr={completed.stderr!r}"
            ),
            "cmd": cmd,
        }
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    rows = summary.get("rows", [])
    row_scores = [score_trial(row) for row in rows]
    sim_pass = bool(rows) and all(s > -1.0 for s in row_scores)
    gate_score = float(np.mean(row_scores)) if row_scores else -1.0
    return {
        "sim_pass": bool(sim_pass),
        "gate_score": gate_score,
        "num_valid_move_and_hold": int(summary.get("num_valid_move_and_hold", 0)),
        "num_runs": int(summary.get("num_runs", len(rows))),
        "summary_path": str(summary_path),
        "output_root": str(candidate_output_root),
        "cmd": cmd,
    }


def _fmt_gains(gains: dict[str, float]) -> str:
    return ", ".join(f"{name}={value:.4g}" for name, value in gains.items())


def _real_command(*, robot_ip: str, gains: dict[str, float]) -> str:
    """Mirror tools/ur5e_suggest_gains.py's own ready-to-copy command block.
    Deliberately omits --yes: the printed command still requires its own
    typed-MOVE confirmation when a human actually runs it, on top of this
    tool's own batch-approval confirmation -- two independent human gates,
    not one gate presented twice."""
    return (
        "  python tools/ur5e_direct_torque_x_transport.py "
        f"--robot-ip {robot_ip} --control-mode direct_torque --dynamics-source local \\\n"
        f"    --gain-overrides-json '{json.dumps(gains)}' \\\n"
        "    --i-understand-this-moves-the-robot"
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--trials-dir", type=Path, required=True, help="Directory to search recursively for past real-trial summary.json files (same semantics as ur5e_suggest_gains.py).")
    p.add_argument("--search-space-json", required=True, help="Path to a JSON file or inline JSON object: {gain_name: [low, high]}.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=4, help="Number of candidates to propose and sim-gate in one round.")
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="Base MuJoCo config to layer gain overrides onto for the sim gate (default matches ur5e_direct_torque_x_transport.py's own default).")
    p.add_argument("--sim-target-x-deltas", nargs="+", type=float, default=list(DEFAULT_SIM_TARGET_X_DELTAS), help="Sim-gate X displacements in meters; must fall within the proven-safe [0.10, 0.20] range.")
    p.add_argument("--sim-move-duration", type=float, default=1.0)
    p.add_argument("--sim-hold-duration", type=float, default=2.0)
    p.add_argument("--sim-torque-limit-scale", type=float, default=1.0)
    p.add_argument("--sim-start-q-rad", nargs=6, type=float, default=None, metavar=("Q1", "Q2", "Q3", "Q4", "Q5", "Q6"), help="Optional start pose passthrough to the sim gate.")
    p.add_argument("--sim-output-root", type=Path, default=None, help="Where per-candidate sim-gate runs are written. Defaults under outputs/ur5e_mujoco_torque_transport/.")
    p.add_argument("--robot-ip", default=None, help="If given, also print ready-to-copy tools/ur5e_direct_torque_x_transport.py commands for the approved batch.")
    p.add_argument("--yes", action="store_true", help="Skip the typed CONFIRM prompt before printing real-hardware commands (for scripted/test use only).")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    validate_sim_target_x_deltas(args.sim_target_x_deltas)

    search_space = _parse_search_space(args.search_space_json)
    summaries = _load_trial_summaries(args.trials_dir)
    study, skipped = build_study(search_space, summaries, seed=args.seed)

    print(
        f"Loaded {len(summaries)} trial summar{'y' if len(summaries) == 1 else 'ies'} "
        f"from {args.trials_dir} ({len(study.trials)} usable, {skipped} skipped -- missing/out-of-range gains)."
    )
    if study.trials:
        best = study.best_trial
        print(f"Best real-trial score so far: score={best.value:.4f} params={best.params}")

    candidates = propose_batch(study, search_space, args.batch_size)
    print(f"\nProposed batch of {len(candidates)} candidates (Optuna TPE, seed={args.seed}), sim-gating each now...")

    sim_output_root = args.sim_output_root if args.sim_output_root is not None else _default_output_root()
    sim_output_root = sim_output_root.expanduser().resolve()

    results = []
    for i, gains in enumerate(candidates, start=1):
        candidate_output_root = sim_output_root / f"candidate_{i:02d}"
        result = run_sim_gate(
            candidate_gains=gains,
            candidate_output_root=candidate_output_root,
            config=args.config,
            seed=args.seed,
            target_x_deltas=args.sim_target_x_deltas,
            move_duration=args.sim_move_duration,
            hold_duration=args.sim_hold_duration,
            torque_limit_scale=args.sim_torque_limit_scale,
            start_q_rad=args.sim_start_q_rad,
        )
        results.append({"gains": gains, **result})
        status = "SIM PASS" if result["sim_pass"] else "SIM FAIL"
        print(
            f"  [{i}/{len(candidates)}] {_fmt_gains(gains)} -> {status} "
            f"({result['num_valid_move_and_hold']}/{result['num_runs']} dx points valid, "
            f"gate_score={result['gate_score']:.4f})"
        )
        if not result["sim_pass"] and result.get("error"):
            print(f"      {result['error']}", file=sys.stderr)

    passing = [r for r in results if r["sim_pass"]]
    print(f"\nSim gate results: {len(passing)}/{len(results)} candidates passed.")

    if not passing:
        print(
            "\nNo candidates in this batch passed the sim gate -- nothing is safe to propose for "
            "real hardware from this round. Not falling back to an unvalidated candidate.\n"
            "Try again with a different --seed for a fresh batch, or check whether failures cluster "
            "near the edges of --search-space-json (a sign the space itself needs narrowing)."
        )
        return 0

    print("\nApproved-for-real-hardware batch (sim-passing candidates only):")
    for i, r in enumerate(passing, start=1):
        print(f"  {i}. {_fmt_gains(r['gains'])}  (gate_score={r['gate_score']:.4f})")

    if not args.yes:
        typed = input(
            f"\nType CONFIRM to print ready-to-copy real-hardware commands for these "
            f"{len(passing)} candidate(s): "
        ).strip()
        if typed != "CONFIRM":
            print("Aborted -- no real-hardware commands printed.", file=sys.stderr)
            return 2

    print("\nReady-to-copy real-hardware commands (edit --target-x-delta/--move-duration as needed;")
    print("each still requires its own typed MOVE confirmation when run):")
    for i, r in enumerate(passing, start=1):
        print(f"\n  Candidate {i}: {_fmt_gains(r['gains'])}")
        if args.robot_ip:
            print(_real_command(robot_ip=args.robot_ip, gains=r["gains"]))
        else:
            print(f"  --gain-overrides-json '{json.dumps(r['gains'])}'")
            print("  (pass --robot-ip to also print the full ready-to-copy command)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
