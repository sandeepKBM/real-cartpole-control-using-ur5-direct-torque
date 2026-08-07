#!/usr/bin/env python3
"""Score CartesianVelocityConfig.orientation_priority (see controller_core/
cartesian_velocity_controller/config.py) on the SAME 128-cell evaluation grid
every historical result in outputs/velocity_gain_tuning/ was scored on, at the
SAME fixed gain vector, mechanism off vs. on.

Why this exists as its own CLI rather than a flag on velocity_gain_tuning/
evaluate.py's ``__main__``: orientation_priority is deliberately NOT an
ACTION_FIELDS dimension (see velocity_transport_env.py's env-config comment --
widening the action space would force every historical action vector to be
padded and silently reinterpreted), so "same gains, mechanism toggled" is an
ENV-config comparison, which that CLI has no way to express. Keeping it here
also keeps the before/after pair a single reproducible command rather than two
hand-edited runs whose env configs have to be trusted to match.

Uses the unmodified evaluate_gains/summarize_safety and the unmodified guards
(|qd| <= 3.0 rad/s, orthogonal drift <= 0.05 m, orientation error <= 0.25 rad).

Example (default gain vector = search_result_nullspace_v2_20260806_194402.json,
this lane's reproducible fixed-gain best at 104/128):

    python tools/evaluate_orientation_priority.py --output-json out.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from velocity_gain_tuning.envs.velocity_transport_env import VelocityTransportEnvConfig
from velocity_gain_tuning.evaluate import evaluate_gains, summarize_safety
from velocity_gain_tuning.optimize import run_search
from velocity_gain_tuning.poses import POSE_SCENARIOS, scenario_by_name

# The exact grid optimize.run_search's own auto_evaluate uses, so the numbers
# printed here are directly comparable to every safety_summary already stored
# in outputs/velocity_gain_tuning/search_result_*.json.
EVAL_DX_FRACTIONS: tuple[float, ...] = (
    0.3, 0.6, 0.9, 1.0, 1.1, 1.3, 1.6, 2.0, -0.3, -0.6, -0.9, -1.0, -1.1, -1.3, -1.6, -2.0,
)

# search_result_nullspace_v2_20260806_194402.json's action -- the reproducible
# fixed-gain best (104/128) this lane's scheduling work is benchmarked against.
DEFAULT_ACTION: tuple[float, ...] = (
    -0.5452930656195676,
    -0.31201103390079576,
    0.19603435480606923,
    -0.40319481871903273,
    0.6634521666673519,
    -0.29877165734428546,
)


def _cell_key(result) -> str:
    return f"{result.scenario}@{result.target_x_delta_m:+.4f}@{result.move_duration_s:g}"


def evaluate_pair(
    action: np.ndarray,
    *,
    scenarios=POSE_SCENARIOS,
    weight: float = 1.0,
    residual_tol_m: float = 0.0001,
    residual_falloff_m: float = 0.0005,
    falloff_power: float = 2.0,
    seed: int = 0,
) -> dict:
    """Runs the full grid twice -- mechanism off, mechanism on -- with every
    other setting identical, and returns both summaries plus a per-cell diff."""
    off_cfg = VelocityTransportEnvConfig(orientation_priority=False)
    on_cfg = replace(
        off_cfg,
        orientation_priority=True,
        orientation_priority_weight=weight,
        orientation_priority_residual_tol_m=residual_tol_m,
        orientation_priority_residual_falloff_m=residual_falloff_m,
        orientation_priority_falloff_power=falloff_power,
    )

    # Third arm, deliberately included in every run: the ZERO-CODE control
    # experiment (just flipping task_dim_rx/task_dim_ry on, so rx/ry join the
    # task at the same weight as position, unconditionally). orientation_
    # priority is only worth its complexity if it beats this -- reporting the
    # two side by side is the only way that claim can be checked rather than
    # assumed.
    rxry_cfg = replace(off_cfg, task_dim_rx=True, task_dim_ry=True)

    out: dict = {}
    per_cell: dict[str, dict] = {}
    for label, env_cfg in (("off", off_cfg), ("on", on_cfg), ("rxry_always", rxry_cfg)):
        t0 = time.time()
        results = evaluate_gains(
            action,
            scenarios=scenarios,
            dx_fractions=EVAL_DX_FRACTIONS,
            fast_move_dx_fractions=EVAL_DX_FRACTIONS,
            env_config=env_cfg,
            seed=seed,
        )
        out[label] = summarize_safety(results)
        out[label]["elapsed_s"] = time.time() - t0
        for r in results:
            per_cell.setdefault(_cell_key(r), {})[label] = {
                "pass": r.guard_reason is None,
                "guard_reason": r.guard_reason,
                "achieved_x_delta_m": r.achieved_x_delta_m,
                "orientation_error": r.orientation_error,
                "max_abs_qd_radps": r.max_abs_qd_radps,
            }

    out["fixed"] = sorted(k for k, v in per_cell.items() if not v["off"]["pass"] and v["on"]["pass"])
    out["broken"] = sorted(k for k, v in per_cell.items() if v["off"]["pass"] and not v["on"]["pass"])
    out["rxry_fixed"] = sorted(k for k, v in per_cell.items() if not v["off"]["pass"] and v["rxry_always"]["pass"])
    out["rxry_broken"] = sorted(k for k, v in per_cell.items() if v["off"]["pass"] and not v["rxry_always"]["pass"])
    out["on_vs_rxry_differs"] = sorted(
        k for k, v in per_cell.items() if v["on"]["pass"] != v["rxry_always"]["pass"]
    )
    out["per_cell"] = per_cell
    return out


def _print_summary(out: dict) -> None:
    for label in ("off", "on", "rxry_always"):
        s = out[label]
        print(
            f"{label:<12} {s['n_pass']:>3}/{s['n_total']} pass  "
            f"slow {s['slow_move']['n_pass']}/{s['slow_move']['n_total']}  "
            f"fast {s['fast_move']['n_pass']}/{s['fast_move']['n_total']}  "
            f"worst |qd| {s['worst_abs_qd_radps']:.3f}  "
            f"worst ori {s['worst_orientation_error_rad']:.4f}  "
            f"({s['elapsed_s']:.0f}s)"
        )
    print()
    for name in sorted(set(out["off"]["per_scenario"]) | set(out["on"]["per_scenario"])):
        a = out["off"]["per_scenario"].get(name, {"pass": 0, "fail": 0})
        b = out["on"]["per_scenario"].get(name, {"pass": 0, "fail": 0})
        print(f"  {name:<24} off {a['pass']:>2}/{a['pass'] + a['fail']:<3} -> on {b['pass']:>2}/{b['pass'] + b['fail']}")
    print(f"\n  on           FIXED  ({len(out['fixed'])}): " + (", ".join(out["fixed"]) or "-"))
    print(f"  on           BROKEN ({len(out['broken'])}): " + (", ".join(out["broken"]) or "-"))
    print(f"  rxry_always  FIXED  ({len(out['rxry_fixed'])}): " + (", ".join(out["rxry_fixed"]) or "-"))
    print(f"  rxry_always  BROKEN ({len(out['rxry_broken'])}): " + (", ".join(out["rxry_broken"]) or "-"))
    print(f"  on != rxry_always ({len(out['on_vs_rxry_differs'])}): "
          + (", ".join(out["on_vs_rxry_differs"]) or "- (identical outcomes)"))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--action", type=float, nargs="+", default=None, help="Gain action vector (default: nullspace_v2).")
    p.add_argument("--action-from-json", type=Path, default=None, help="Read the action from a search_result JSON.")
    p.add_argument("--scenario", action="append", default=[], help="Restrict to named pose scenario(s). Repeatable.")
    p.add_argument("--weight", type=float, default=1.0)
    p.add_argument("--residual-tol-m", type=float, default=0.0001)
    p.add_argument("--residual-falloff-m", type=float, default=0.0005)
    p.add_argument("--falloff-power", type=float, default=2.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output-json", type=Path, default=None)
    args = p.parse_args()

    if args.action_from_json is not None:
        action = np.array(json.loads(args.action_from_json.read_text(encoding="utf-8"))["action"], dtype=np.float64)
    elif args.action is not None:
        action = np.array(args.action, dtype=np.float64)
    else:
        action = np.array(DEFAULT_ACTION, dtype=np.float64)

    scenarios = tuple(scenario_by_name(n) for n in args.scenario) if args.scenario else POSE_SCENARIOS

    out = evaluate_pair(
        action,
        scenarios=scenarios,
        weight=args.weight,
        residual_tol_m=args.residual_tol_m,
        residual_falloff_m=args.residual_falloff_m,
        falloff_power=args.falloff_power,
        seed=args.seed,
    )
    _print_summary(out)

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps(
                {
                    "action": action.tolist(),
                    "scenarios": [s.name for s in scenarios],
                    "mechanism": {
                        "orientation_priority_weight": args.weight,
                        "orientation_priority_residual_tol_m": args.residual_tol_m,
                        "orientation_priority_residual_falloff_m": args.residual_falloff_m,
                        "orientation_priority_falloff_power": args.falloff_power,
                    },
                    **out,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"\nwrote {args.output_json}")


if __name__ == "__main__":
    # run_search is imported only so this module fails loudly at import time if
    # the optimize/evaluate pair ever stops being importable together.
    assert run_search is not None
    main()
