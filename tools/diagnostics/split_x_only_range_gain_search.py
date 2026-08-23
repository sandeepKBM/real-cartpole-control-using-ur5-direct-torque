#!/usr/bin/env python3
"""Gradient-free RANGE-extension gain search for the 1-row split transport task.

Context. ``config/ur5e_mujoco_torque_osc_tuned_split_base_wrist_lift_elbow_wrist1_x_only.yaml``
drives a 1D world-X transport with ``{shoulder_lift, elbow, wrist_1}`` while
``shoulder_pan``/``wrist_2``/``wrist_3`` stay posture-held (the real robot's
``wrist_2`` is parked at a physical limit near its singularity, and
``shoulder_pan`` must stay fixed for base clearance). Its measured safe range at
the real start pose is only about +-0.02 m before a drift guard trips
(docs/status/transport_axis_generalization_and_pendulum_axis_2026-08-12.md sec 3).

This script asks, with real closed-loop rollouts on ``assets/ur5e_torque/scene.xml``
only (never a toy plant), how much of that ceiling is GAIN-tunable. It searches
controller gains -- and, when built, the 1-row-aware nullspace-projector eps
schedule (``nullspace_inertia_adaptive_regularization``) -- with
``scipy.optimize.differential_evolution``. Gradient-free by design; this repo
has a documented history of RL gain-scheduling failures (AGENTS.md sec 4) and
deliberately does not use RL here.

SAFETY THRESHOLDS ARE NEVER TOUCHED. The search varies controller GAINS only.
Every candidate runs against the config file's own ``controller.safety`` block
unchanged, and a candidate that trips any guard is scored as having failed that
rung -- guard trips are the objective's signal, not something to tune away.

Both +X and -X are evaluated at every rung (AGENTS.md sec 7: directional
asymmetry is real and recurring in this repo).

Examples
--------
    # the search itself (long)
    DE_WORKERS=32 python tools/diagnostics/split_x_only_range_gain_search.py \
        --mode search --maxiter 30 --seed 0 --json outputs/split_x_only_search.json

    # rigor pass over one gain set (the winner, or the baseline for comparison)
    python tools/diagnostics/split_x_only_range_gain_search.py --mode validate \
        --gains-json outputs/split_x_only_search.json
    python tools/diagnostics/split_x_only_range_gain_search.py --mode validate --baseline
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.diagnostics.split_base_wrist_task_dims_sim_check import (  # noqa: E402
    DEFAULT_CONFIG,
    USER_REAL_POSE_Q,
    load_controller_config,
    run_rollout,
)

#: Displacement ladder, ascending. The objective rewards how far UP this ladder
#: a gain set gets cleanly in BOTH directions -- this is a range-extension
#: search, not a point-tuning search at one displacement.
RUNGS_M: tuple[float, ...] = (0.02, 0.03, 0.04, 0.05)
#: Every rung is run both ways. Never one-directional -- AGENTS.md sec 7.
DIRECTIONS: tuple[float, ...] = (+1.0, -1.0)

MOVE_DURATION_S = 1.5
HOLD_DURATION_S = 1.0
#: A rung counts as "clean" only if the guards stayed quiet AND the move
#: actually arrived. Tracking below this is a real failure to transport, not a
#: cosmetic miss, so it must not be scored as usable range.
MIN_TRACKING_FRACTION = 0.80

#: (name, low, high). GAINS ONLY -- no safety threshold appears here, by design.
SEARCH_SPACE: tuple[tuple[str, float, float], ...] = (
    ("kp_posture", 10.0, 600.0),
    ("kd_posture", 2.0, 60.0),
    ("kd_joint", 0.5, 15.0),
    ("kp_x", 100.0, 900.0),
    ("kd_x", 10.0, 90.0),
    ("nullspace_inertia_eps_ratio", 0.005, 1.0),
)


def _de_workers() -> int:
    """Bounded worker count for ``differential_evolution(workers=...)``.

    Copied deliberately from tools/diagnostics/pendulum_swingup_multi_kick.py --
    never ``workers=-1`` on this shared host (AGENTS.md sec 8), and the
    multiprocessing start method is pinned to ``fork`` so scipy >=1.17 does not
    fall back to ``forkserver`` (which re-imports ``__main__`` in every worker
    and pickles the MuJoCo model instead of inheriting it copy-on-write).
    """
    import multiprocessing as mp

    try:
        mp.set_start_method("fork")
    except RuntimeError:
        pass  # already set
    env = os.environ.get("DE_WORKERS")
    if env:
        return max(1, int(env))
    return max(1, int((os.cpu_count() or 2) * 0.9))


def base_gains(config_path: str | Path = DEFAULT_CONFIG) -> dict[str, float]:
    path = Path(config_path)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return dict(load_controller_config(path)["controller"]["gains"])


def overrides_from_vector(
    x: Sequence[float], *, config_path: str | Path = DEFAULT_CONFIG, use_inertia_eps: bool = True
) -> dict[str, Any]:
    """Map a DE parameter vector to controller-config overrides.

    Only ``controller.gains`` entries and the projector's eps-schedule knobs are
    written; the config's safety block, task-row/column selection and every
    other flag come from the file unchanged.
    """
    named = {name: float(val) for (name, _lo, _hi), val in zip(SEARCH_SPACE, x)}
    gains = base_gains(config_path)
    for key in ("kp_posture", "kd_posture", "kd_joint", "kp_x", "kd_x"):
        if key in named:
            gains[key] = named[key]
    ov: dict[str, Any] = {"gains": gains}
    if use_inertia_eps and "nullspace_inertia_eps_ratio" in named:
        ov["nullspace_inertia_adaptive_regularization"] = True
        ov["nullspace_inertia_eps_ratio"] = named["nullspace_inertia_eps_ratio"]
    return ov


def _run_quality(res) -> tuple[bool, float]:
    """(clean, quality in [0, 1]) for one rollout.

    ``quality`` is deliberately continuous so differential evolution sees a
    gradient rather than a step: a run that trips late scores above one that
    trips early, and a clean run scores above every tripping run.
    """
    total_s = MOVE_DURATION_S + HOLD_DURATION_S
    if res.guard_tripped:
        frac = 0.0 if res.guard_time_s is None else min(max(res.guard_time_s / total_s, 0.0), 1.0)
        return False, 0.4 * frac
    track = float(res.tracking_fraction)
    clean = track >= MIN_TRACKING_FRACTION
    return clean, 0.4 + 0.6 * min(max(track / MIN_TRACKING_FRACTION, 0.0), 1.0)


def evaluate(
    x: Sequence[float],
    *,
    config_path: str | Path = DEFAULT_CONFIG,
    use_inertia_eps: bool = True,
    rungs: Sequence[float] = RUNGS_M,
    detail: bool = False,
) -> float | tuple[float, list[dict[str, Any]]]:
    """Negated range score (differential_evolution minimizes).

    Score = ``100 * reached`` + a continuous tie-breaker, where ``reached`` is
    the largest displacement whose rung -- and every smaller rung -- came back
    clean in BOTH directions. The 100x weight makes actually reaching a rung
    dominate; the tie-breaker (sum of per-rung mean quality weighted by that
    rung's displacement) is what gives DE something to climb between rungs.
    """
    ov = overrides_from_vector(x, config_path=config_path, use_inertia_eps=use_inertia_eps)
    rows: list[dict[str, Any]] = []
    reached = 0.0
    prefix_clean = True
    tiebreak = 0.0
    for rung in rungs:
        qualities: list[float] = []
        rung_clean = True
        for sign in DIRECTIONS:
            dx = float(sign) * float(rung)
            try:
                res = run_rollout(
                    USER_REAL_POSE_Q,
                    config_path=config_path,
                    ctrl_overrides=ov,
                    target_delta_m=dx,
                    move_duration_s=MOVE_DURATION_S,
                    hold_duration_s=HOLD_DURATION_S,
                    pose_label="user_real_pose",
                )
            except Exception:  # noqa: BLE001 -- an unusable gain set is a bad score, not a crash
                return (1.0e6, rows) if detail else 1.0e6
            clean, q = _run_quality(res)
            qualities.append(q)
            rung_clean = rung_clean and clean
            if detail:
                rows.append(
                    dict(
                        dx=dx,
                        tracking=float(res.tracking_fraction),
                        max_y=float(res.max_abs_y_drift_m),
                        max_z=float(res.max_abs_z_drift_m),
                        ori=float(res.max_orientation_error_rad),
                        max_qd=float(res.max_abs_qd_radps),
                        max_tau=float(res.max_abs_tau_nm),
                        guard=res.guard_reason if res.guard_tripped else "",
                        guard_time=res.guard_time_s,
                    )
                )
        tiebreak += float(rung) * float(np.mean(qualities))
        if prefix_clean and rung_clean:
            reached = float(rung)
        else:
            prefix_clean = False
    score = 100.0 * reached + tiebreak
    return (-score, rows) if detail else -score


# --------------------------------------------------------------------------- #
def run_search(*, maxiter: int, popsize: int, seed: int, config_path: str) -> dict[str, Any]:
    from scipy.optimize import differential_evolution

    bounds = [(lo, hi) for _name, lo, hi in SEARCH_SPACE]
    res = differential_evolution(
        evaluate,
        bounds,
        maxiter=int(maxiter),
        popsize=int(popsize),
        seed=int(seed),
        workers=_de_workers(),
        polish=False,
        updating="deferred",
        tol=0.0,
    )
    best = {name: float(v) for (name, _lo, _hi), v in zip(SEARCH_SPACE, res.x)}
    return {"best": best, "cost": float(res.fun), "nit": int(res.nit), "nfev": int(res.nfev)}


def validate(
    params: dict[str, float] | None,
    *,
    config_path: str = DEFAULT_CONFIG,
    use_inertia_eps: bool,
    deltas: Sequence[float],
    hold_s: float,
    move_s: float,
) -> list[dict[str, Any]]:
    """Rigor pass: every displacement, both directions, long hold, full metrics."""
    if params is None:
        ov: dict[str, Any] = {}
    else:
        vec = [params[name] for name, _lo, _hi in SEARCH_SPACE]
        ov = overrides_from_vector(vec, config_path=config_path, use_inertia_eps=use_inertia_eps)
    rows: list[dict[str, Any]] = []
    for rung in deltas:
        for sign in DIRECTIONS:
            dx = float(sign) * float(rung)
            res = run_rollout(
                USER_REAL_POSE_Q,
                config_path=config_path,
                ctrl_overrides=ov,
                target_delta_m=dx,
                move_duration_s=float(move_s),
                hold_duration_s=float(hold_s),
                pose_label="user_real_pose",
            )
            rows.append(
                dict(
                    dx=dx,
                    hold_s=float(hold_s),
                    tracking_pct=100.0 * float(res.tracking_fraction),
                    max_y=float(res.max_abs_y_drift_m),
                    max_z=float(res.max_abs_z_drift_m),
                    ori=float(res.max_orientation_error_rad),
                    max_qd=float(res.max_abs_qd_radps),
                    max_tau=float(res.max_abs_tau_nm),
                    guard=(
                        f"{res.guard_reason}@{res.guard_time_s:.2f}s" if res.guard_tripped else "-"
                    ),
                )
            )
    return rows


def print_table(rows: list[dict[str, Any]], title: str) -> None:
    print(f"\n=== {title} ===")
    print(
        f"{'dx[m]':>8} {'hold':>6} {'track%':>8} {'max|Y|':>9} {'max|Z|':>9} "
        f"{'ori[rad]':>9} {'max|qd|':>9} {'max|tau|':>9} {'guard':>28}"
    )
    for r in rows:
        print(
            f"{r['dx']:>+8.3f} {r['hold_s']:>6.1f} {r['tracking_pct']:>8.2f} "
            f"{r['max_y']:>9.5f} {r['max_z']:>9.5f} {r['ori']:>9.4f} "
            f"{r['max_qd']:>9.4f} {r['max_tau']:>9.3f} {r['guard']:>28}"
        )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--mode", choices=["search", "validate"], default="search")
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    ap.add_argument("--maxiter", type=int, default=30)
    ap.add_argument("--popsize", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json", default=None)
    ap.add_argument("--gains-json", default=None, help="validate the 'best' block of this file")
    ap.add_argument("--baseline", action="store_true",
                    help="validate the config file's own gains, unmodified (the comparison point)")
    ap.add_argument("--deltas", type=float, nargs="+",
                    default=[0.01, 0.02, 0.025, 0.03, 0.04, 0.05])
    ap.add_argument("--hold-duration", type=float, default=5.0)
    ap.add_argument("--move-duration", type=float, default=1.5)
    args = ap.parse_args(argv)

    if args.mode == "search":
        out = run_search(
            maxiter=args.maxiter, popsize=args.popsize, seed=args.seed, config_path=args.config
        )
        print(json.dumps(out, indent=2))
        _cost, rows = evaluate(
            [out["best"][name] for name, _lo, _hi in SEARCH_SPACE],
            config_path=args.config, detail=True,
        )
        out["search_rungs"] = rows
        if args.json:
            Path(args.json).parent.mkdir(parents=True, exist_ok=True)
            Path(args.json).write_text(json.dumps(out, indent=2), encoding="utf-8")
            print(f"wrote {args.json}")
        return 0

    params = None
    use_eps = False
    if args.gains_json:
        params = json.loads(Path(args.gains_json).read_text())["best"]
        use_eps = True
    elif not args.baseline:
        ap.error("--mode validate needs --gains-json or --baseline")
    rows = validate(
        params, config_path=args.config, use_inertia_eps=use_eps,
        deltas=args.deltas, hold_s=args.hold_duration, move_s=args.move_duration,
    )
    print_table(rows, "baseline (config gains, unmodified)" if params is None else "search winner")
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(rows, indent=2), encoding="utf-8")
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
