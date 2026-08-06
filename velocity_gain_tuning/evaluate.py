"""Multi-pose, multi-displacement evaluation grid for a fixed gain vector --
mirrors the manual sweeps run by hand earlier this session
(tools/diagnostics/ur5e_velocity_control_kinematic_sim.py) but reusable for
any gain set (the current hand-tuned defaults, or an optimize.py result).
"""

from __future__ import annotations

import numpy as np

from .envs.velocity_transport_env import VelocityTransportEnv, VelocityTransportEnvConfig
from .optimize import FAST_MOVE_DURATION_S, EpisodeResult, run_episode
from .poses import POSE_SCENARIOS, PoseScenario


def evaluate_gains(
    action: np.ndarray,
    *,
    scenarios: tuple[PoseScenario, ...] = POSE_SCENARIOS,
    dx_fractions: tuple[float, ...] = (0.3, 0.6, 0.9, 1.0, 1.1, 1.3, -0.3, -0.6, -0.9, -1.0, -1.1, -1.3),
    fast_move_dx_fractions: tuple[float, ...] = (0.3, 0.6, 0.9, 1.0, 1.3, 1.6, -0.3, -0.6, -0.9, -1.0, -1.3, -1.6),
    env_config: VelocityTransportEnvConfig | None = None,
    seed: int = 0,
) -> list[EpisodeResult]:
    """Runs both the default (slow, move_duration_s from env_config -- normally
    1.0s) grid AND a fast_move_dx_fractions grid at FAST_MOVE_DURATION_S, so a
    reported safety summary can never describe a gain vector as "safe" while
    having only tested slow moves -- see optimize.py's FAST_MOVE_DURATION_S
    docstring for why this repo needed both dimensions covered (ik_seeded_
    resolution's real speed governor is ik_joint_gain, not move_duration_s;
    a gain set found safe at the default 1.0s move produced qd~4.7 rad/s,
    over the 3.0 guard, at a fast move -- undetectable by the slow grid
    alone). Both fraction defaults include NEGATIVE values too (added
    2026-08-06, AGENTS.md sec 7's "always sweep both +X and -X" rule) --
    found this same day: a gain vector passing cleanly at +0.37m failed via
    joint_velocity_guard at -0.37m, an asymmetry a positive-only grid
    structurally cannot see."""
    env = VelocityTransportEnv(env_config, seed=seed)
    results: list[EpisodeResult] = []
    for scenario in scenarios:
        for frac in dx_fractions:
            dx = frac * scenario.max_dx_hint_m
            results.append(run_episode(env, action, scenario=scenario, target_x_delta_m=dx))
        for frac in fast_move_dx_fractions:
            dx = frac * scenario.max_dx_hint_m
            results.append(
                run_episode(
                    env, action, scenario=scenario, target_x_delta_m=dx, move_duration_s=FAST_MOVE_DURATION_S
                )
            )
    return results


def print_report(results: list[EpisodeResult]) -> None:
    print(f"{'scenario':<24} {'dx_m':>8} {'move_s':>7} {'achieved':>10} {'qd_max':>7} {'ori_err':>8} {'guard':<30}")
    for r in results:
        guard = r.guard_reason or "duration_complete"
        print(f"{r.scenario:<24} {r.target_x_delta_m:>8.4f} {r.move_duration_s:>7.3f} "
              f"{r.achieved_x_delta_m:>10.5f} {r.max_abs_qd_radps:>7.3f} {r.orientation_error:>8.4f} {guard:<30}")


def _is_fast_move(r: EpisodeResult) -> bool:
    return r.move_duration_s <= FAST_MOVE_DURATION_S + 1e-9


def summarize_safety(results: list[EpisodeResult]) -> dict:
    """Machine-readable pass/fail summary -- built so callers (optimize.py's
    automatic post-search evaluation, or any future automation) can act on
    a search result's safety profile programmatically instead of requiring
    someone to read a printed table by hand each time.

    Splits into "slow_move" (the default/nominal move_duration_s, i.e. the
    range/tracking-focused grid) and "fast_move" (FAST_MOVE_DURATION_S, the
    speed-safety-focused grid, see optimize.py's docstring) so a caller can
    tell the two failure classes apart -- a gain vector can be fully safe
    on the slow grid while tripping the |qd| guard specifically at fast
    moves, and reporting only a combined pass_fraction would hide that."""
    n_total = len(results)
    n_pass = sum(1 for r in results if r.guard_reason is None)
    guard_trips = [
        {
            "scenario": r.scenario,
            "target_x_delta_m": r.target_x_delta_m,
            "move_duration_s": r.move_duration_s,
            "max_abs_qd_radps": r.max_abs_qd_radps,
            "guard_reason": r.guard_reason,
        }
        for r in results
        if r.guard_reason is not None
    ]
    worst_orientation_error = max((r.orientation_error for r in results), default=0.0)
    worst_qd = max((r.max_abs_qd_radps for r in results), default=0.0)
    per_scenario: dict[str, dict] = {}
    for r in results:
        entry = per_scenario.setdefault(r.scenario, {"pass": 0, "fail": 0, "max_passing_dx_m": 0.0})
        if r.guard_reason is None:
            entry["pass"] += 1
            entry["max_passing_dx_m"] = max(entry["max_passing_dx_m"], r.target_x_delta_m)
        else:
            entry["fail"] += 1

    def _bucket_summary(bucket: list[EpisodeResult]) -> dict:
        n = len(bucket)
        passed = sum(1 for r in bucket if r.guard_reason is None)
        return {
            "n_total": n,
            "n_pass": passed,
            "n_fail": n - passed,
            "pass_fraction": passed / n if n else 0.0,
            "worst_abs_qd_radps": max((r.max_abs_qd_radps for r in bucket), default=0.0),
        }

    fast_results = [r for r in results if _is_fast_move(r)]
    slow_results = [r for r in results if not _is_fast_move(r)]

    return {
        "n_total": n_total,
        "n_pass": n_pass,
        "n_fail": n_total - n_pass,
        "pass_fraction": n_pass / n_total if n_total else 0.0,
        "worst_orientation_error_rad": worst_orientation_error,
        "worst_abs_qd_radps": worst_qd,
        "guard_trips": guard_trips,
        "per_scenario": per_scenario,
        "slow_move": _bucket_summary(slow_results),
        "fast_move": _bucket_summary(fast_results),
    }


if __name__ == "__main__":
    import sys

    from .envs.velocity_transport_env import ACTION_DIM

    # Default: evaluate the all-zero action (midpoint of every range, i.e.
    # kp_x~3.25, kp_rot~3.1, ik_joint_gain~6.25, pinv_damping~5e-3,
    # qp_task_weight~1e4) unless a 5-value action is passed on argv.
    if len(sys.argv) == ACTION_DIM + 1:
        action = np.array([float(v) for v in sys.argv[1:]])
    else:
        action = np.zeros(ACTION_DIM)
    results = evaluate_gains(action)
    print_report(results)
