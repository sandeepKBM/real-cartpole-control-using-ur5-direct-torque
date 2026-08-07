"""Stage 3: score a fitted GainSchedule on the SAME safety grid every
global gain search in this repo is scored on.

Deliberately reuses ``velocity_gain_tuning.optimize.run_episode`` and
``velocity_gain_tuning.evaluate.summarize_safety`` unchanged -- same
environment, same guard thresholds (|qd|<=3.0 rad/s, orthogonal drift
<=0.05 m, orientation error <=0.25 rad), same episode bookkeeping. The only
difference from ``evaluate.evaluate_gains`` is WHERE the action comes from:
``evaluate_gains`` applies one fixed vector to every cell, this applies
``schedule.action_for(pose, dx)``. That keeps the resulting "N/128 pass"
number directly comparable to every historical
``outputs/velocity_gain_tuning/search_result_*.json`` figure, which is the
entire point -- a scheduling result that had to be scored on a new,
friendlier grid would prove nothing.

The default grid is optimize.run_search's own ``eval_dx_fractions``
(+/-{0.3,0.6,0.9,1.0,1.1,1.3,1.6,2.0}) applied at BOTH the nominal and the
fast move duration: 8 fractions x 2 signs x 2 durations x 4 poses = 128
cells. Note that +/-1.0 is deliberately NOT a schedule knot (see
cells.DEFAULT_KNOT_FRACTIONS), so 16 of those 128 cells are genuine
held-out interpolation tests.
"""

from __future__ import annotations

import numpy as np

from ..envs.velocity_transport_env import VelocityTransportEnv, VelocityTransportEnvConfig
from ..evaluate import summarize_safety
from ..optimize import FAST_MOVE_DURATION_S, EpisodeResult, run_episode
from ..poses import POSE_SCENARIOS, PoseScenario
from .schedule import GainSchedule

STANDARD_EVAL_DX_FRACTIONS: tuple[float, ...] = (
    0.3, 0.6, 0.9, 1.0, 1.1, 1.3, 1.6, 2.0,
    -0.3, -0.6, -0.9, -1.0, -1.1, -1.3, -1.6, -2.0,
)


def evaluate_schedule(
    schedule: GainSchedule,
    *,
    scenarios: tuple[PoseScenario, ...] = POSE_SCENARIOS,
    dx_fractions: tuple[float, ...] = STANDARD_EVAL_DX_FRACTIONS,
    fast_move_dx_fractions: tuple[float, ...] = STANDARD_EVAL_DX_FRACTIONS,
    env_config: VelocityTransportEnvConfig | None = None,
    seed: int = 0,
    env: VelocityTransportEnv | None = None,
) -> list[EpisodeResult]:
    """Run the full slow+fast safety grid with SCHEDULED (per-cell) gains."""
    env = env or VelocityTransportEnv(env_config, seed=seed)
    results: list[EpisodeResult] = []
    for scenario in scenarios:
        for frac in dx_fractions:
            dx = frac * scenario.max_dx_hint_m
            action = schedule.action_for(scenario.name, dx)
            results.append(run_episode(env, action, scenario=scenario, target_x_delta_m=dx))
        for frac in fast_move_dx_fractions:
            dx = frac * scenario.max_dx_hint_m
            action = schedule.action_for(scenario.name, dx)
            results.append(
                run_episode(
                    env,
                    action,
                    scenario=scenario,
                    target_x_delta_m=dx,
                    move_duration_s=FAST_MOVE_DURATION_S,
                )
            )
    return results


def evaluate_fixed_action(
    action: np.ndarray,
    *,
    scenarios: tuple[PoseScenario, ...] = POSE_SCENARIOS,
    dx_fractions: tuple[float, ...] = STANDARD_EVAL_DX_FRACTIONS,
    fast_move_dx_fractions: tuple[float, ...] = STANDARD_EVAL_DX_FRACTIONS,
    env_config: VelocityTransportEnvConfig | None = None,
    seed: int = 0,
    env: VelocityTransportEnv | None = None,
) -> list[EpisodeResult]:
    """Same grid, one fixed global gain vector -- the baseline to beat.

    Exists here (rather than calling ``evaluate.evaluate_gains``) purely so
    the baseline and the schedule can share ONE env instance and therefore
    provably identical environment state/config; ``evaluate_gains``
    constructs its own. Behaviour is otherwise identical.
    """
    env = env or VelocityTransportEnv(env_config, seed=seed)
    results: list[EpisodeResult] = []
    for scenario in scenarios:
        for frac in dx_fractions:
            dx = frac * scenario.max_dx_hint_m
            results.append(run_episode(env, action, scenario=scenario, target_x_delta_m=dx))
        for frac in fast_move_dx_fractions:
            dx = frac * scenario.max_dx_hint_m
            results.append(
                run_episode(
                    env,
                    action,
                    scenario=scenario,
                    target_x_delta_m=dx,
                    move_duration_s=FAST_MOVE_DURATION_S,
                )
            )
    return results


def knot_coverage_split(
    results: list[EpisodeResult],
    knot_fractions_by_scenario: dict[str, set[float]],
    *,
    scenarios: tuple[PoseScenario, ...] = POSE_SCENARIOS,
    tol: float = 1e-6,
) -> dict:
    """Split a result set into cells AT a searched knot vs. HELD-OUT ones.

    This is not a nicety, it is the difference between a real result and a
    misreported one. Measured 2026-08-06: a PCHIP schedule fitted to
    independently-searched knots scored 117/128 overall against a
    fixed-gain baseline's 104/128 -- but 108/112 of that came from the
    cells whose displacement it had been searched at, while at the single
    held-out fraction it scored 9/16 against the SAME baseline's 16/16.
    The aggregate says "+13 cells"; the split says the interpolation is a
    regression and the headline is knot memorisation. An interpolant
    passes exactly through its own knots by construction, so its aggregate
    score is always part memorisation -- reporting it alone is never
    honest for this class of model.

    A cell counts as "at knot" when its displacement, expressed as a
    fraction of its scenario's ``max_dx_hint_m``, matches one of that
    scenario's knot fractions.
    """
    hint = {s.name: s.max_dx_hint_m for s in scenarios}
    at_knot: list[EpisodeResult] = []
    held_out: list[EpisodeResult] = []
    for r in results:
        frac = r.target_x_delta_m / hint[r.scenario]
        knots = knot_fractions_by_scenario.get(r.scenario, set())
        (at_knot if any(abs(frac - k) < tol for k in knots) else held_out).append(r)

    def _bucket(bucket: list[EpisodeResult]) -> dict:
        n = len(bucket)
        passed = sum(1 for r in bucket if r.guard_reason is None)
        return {
            "n_total": n,
            "n_pass": passed,
            "n_fail": n - passed,
            "pass_fraction": passed / n if n else 0.0,
        }

    return {"at_knot": _bucket(at_knot), "held_out": _bucket(held_out)}


def knot_fractions_from_schedule(schedule: GainSchedule) -> dict[str, set[float]]:
    return {
        scenario: {k.dx_fraction for k in schedule.knots if k.scenario == scenario}
        for scenario in schedule.scenarios
    }


def compare_summaries(
    label_to_results: dict[str, list[EpisodeResult]],
    knot_fractions_by_scenario: dict[str, set[float]] | None = None,
) -> dict[str, dict]:
    """summarize_safety for each labelled result set, keyed by label.

    When ``knot_fractions_by_scenario`` is supplied, each summary also gains
    a ``knot_coverage`` entry -- see ``knot_coverage_split`` for why that
    split is required reading before believing any aggregate here.
    """
    out: dict[str, dict] = {}
    for label, results in label_to_results.items():
        summary = summarize_safety(results)
        if knot_fractions_by_scenario is not None:
            summary["knot_coverage"] = knot_coverage_split(results, knot_fractions_by_scenario)
        out[label] = summary
    return out


def print_comparison(summaries: dict[str, dict]) -> None:
    print(
        f"{'variant':<34} {'pass':>10} {'slow':>10} {'fast':>10} "
        f"{'at_knot':>10} {'held_out':>10} {'worst|qd|':>10} {'worstOri':>9}"
    )
    for label, s in summaries.items():
        cov = s.get("knot_coverage")
        at_knot = f"{cov['at_knot']['n_pass']}/{cov['at_knot']['n_total']}" if cov else "-"
        held = f"{cov['held_out']['n_pass']}/{cov['held_out']['n_total']}" if cov else "-"
        print(
            f"{label:<34} "
            f"{s['n_pass']:>4}/{s['n_total']:<5} "
            f"{s['slow_move']['n_pass']:>4}/{s['slow_move']['n_total']:<5} "
            f"{s['fast_move']['n_pass']:>4}/{s['fast_move']['n_total']:<5} "
            f"{at_knot:>10} {held:>10} "
            f"{s['worst_abs_qd_radps']:>10.3f} {s['worst_orientation_error_rad']:>9.4f}"
        )
    print(
        "\n  at_knot   = evaluation cells at a displacement the schedule was SEARCHED at\n"
        "  held_out  = cells at a displacement it was not (the real generalisation test;\n"
        "              an interpolant passes through its own knots by construction, so the\n"
        "              aggregate 'pass' column is always part memorisation)"
    )


if __name__ == "__main__":
    import argparse
    import json
    from pathlib import Path

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--schedule-json", type=Path, required=True, help="Knot table from search.py.")
    p.add_argument(
        "--baseline-json", type=Path, default=None,
        help="Prior optimize.py result JSON to score as the fixed-gain baseline on the "
             "identical grid. Strongly recommended -- a schedule number without the "
             "baseline it is meant to beat, measured in the same process, is not a result.",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output-json", type=Path, default=None)
    p.add_argument(
        "--variants", nargs="*",
        default=["raw", "dropfail", "smooth3", "dropfail_smooth3"],
        help="Schedule fit variants to score. raw = every knot, no smoothing; "
             "dropfail = exclude knots whose own best gains still trip a guard; "
             "smooth3 = 3-wide moving average across knots (see schedule.smooth_actions).",
    )
    args = p.parse_args()

    raw = json.loads(args.schedule_json.read_text(encoding="utf-8"))
    env = VelocityTransportEnv(None, seed=args.seed)

    variant_kwargs = {
        "raw": {"drop_failed_knots": False, "smoothing_window": 1},
        "dropfail": {"drop_failed_knots": True, "smoothing_window": 1},
        "smooth3": {"drop_failed_knots": False, "smoothing_window": 3},
        "dropfail_smooth3": {"drop_failed_knots": True, "smoothing_window": 3},
        "smooth5": {"drop_failed_knots": False, "smoothing_window": 5},
        "dropfail_smooth5": {"drop_failed_knots": True, "smoothing_window": 5},
    }

    label_to_results: dict[str, list[EpisodeResult]] = {}

    if args.baseline_json is not None:
        prior = json.loads(args.baseline_json.read_text(encoding="utf-8"))
        baseline_action = np.array(prior["action"], dtype=np.float64)
        from ..envs.velocity_transport_env import ACTION_DIM

        if baseline_action.shape[0] < ACTION_DIM:
            baseline_action = np.concatenate(
                [baseline_action, np.full(ACTION_DIM - baseline_action.shape[0], -1.0)]
            )
        label_to_results["baseline_global_fixed"] = evaluate_fixed_action(
            baseline_action[:ACTION_DIM], seed=args.seed, env=env
        )

    for variant in args.variants:
        if variant not in variant_kwargs:
            raise SystemExit(f"unknown variant {variant!r}; known: {sorted(variant_kwargs)}")
        schedule = GainSchedule.from_dict(raw, **variant_kwargs[variant])
        label_to_results[f"schedule_{variant}"] = evaluate_schedule(schedule, seed=args.seed, env=env)

    # Knot fractions come from the schedule table itself, so the split is
    # correct even for a non-default knot grid (--knot-fractions).
    reference = GainSchedule.from_dict(raw)
    summaries = compare_summaries(label_to_results, knot_fractions_from_schedule(reference))
    print_comparison(summaries)

    # The per-cell oracle: how many knot cells had ANY guard-clean gain
    # vector at all. Printed alongside because it bounds every row above
    # that uses scheduled gains.
    knots = raw.get("knots", [])
    n_oracle_pass = sum(1 for k in knots if k.get("passed"))
    print(
        f"\nper-cell oracle at the knots themselves: {n_oracle_pass}/{len(knots)} "
        f"knot cells have a guard-clean best gain vector"
    )

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps(
                {
                    "schedule_json": str(args.schedule_json),
                    "baseline_json": str(args.baseline_json) if args.baseline_json else None,
                    "seed": args.seed,
                    "summaries": {
                        # sets are not JSON-serializable; knot_coverage is
                        # already reduced to counts by compare_summaries.
                        k: v for k, v in summaries.items()
                    },
                    "oracle_knot_pass": n_oracle_pass,
                    "oracle_knot_total": len(knots),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"wrote {args.output_json}")
