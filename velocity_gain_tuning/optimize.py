"""Gain search for VelocityTransportEnv via scipy.optimize.differential_evolution
-- a gradient-free global optimizer, the standard tool for a small (~5
parameter), noisy, non-convex, derivative-free tuning problem like this one
(see this package's __init__.py docstring for why this is preferred over
RL/rl_gain_scheduling/ here). Already a dependency of this repo (scipy);
no new package installed.

Each fitness evaluation runs a FULL episode per pose scenario with the SAME
action (gain vector) applied every step -- a single-shot "try these gains
for this whole scripted move" evaluation, not sequential per-step
adaptation. This is deliberately a bandit/black-box-optimization framing,
not an RL problem: differential_evolution needs exactly this (a scalar
fitness per parameter vector), and it structurally cannot reproduce
rl_gain_scheduling/'s documented failure modes since there is no temporal
credit assignment or exploration-collapse risk when the action never
changes within an episode.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import differential_evolution

from .envs.velocity_transport_env import (
    ACTION_DIM,
    VelocityTransportEnv,
    VelocityTransportEnvConfig,
    action_to_gains,
)
from .poses import POSE_SCENARIOS, PoseScenario


@dataclass
class EpisodeResult:
    scenario: str
    target_x_delta_m: float
    move_duration_s: float
    total_reward: float
    final_x_error: float
    achieved_x_delta_m: float
    orientation_error: float
    max_abs_qd_radps: float
    guard_reason: str | None


# A short, near-step move_duration used specifically to stress-test speed
# safety -- see this module's docstring update (2026-08-06) for why this
# was added: a real gain search found (and the fix here is a direct
# response to) that ik_seeded_resolution's actual achieved speed is
# governed almost entirely by ik_joint_gain, NOT by move_duration_s or
# max_lin_speed_mps (this mode's IK loop uses only the target POSITION,
# not velocity feedforward, so a "slow" nominal trajectory profile does
# not mean the robot moves slowly -- the joint-space P-controller chases
# whatever q_target the IK solve converges to at ITS OWN rate, set by
# ik_joint_gain, regardless of how gradually the nominal target itself
# moves). Both gain searches run before this fix used ONLY the default
# (slow, 1.0s) move_duration -- confirmed directly afterward that the
# best found gains (ik_joint_gain~62) produce joint velocity ~4.7 rad/s
# (guard is 3.0) at this FAST_MOVE_DURATION_S, a real, dangerous gap the
# slow-move-only search could not have detected.
FAST_MOVE_DURATION_S = 0.02


def run_episode(
    env: VelocityTransportEnv,
    action: np.ndarray,
    *,
    scenario: PoseScenario,
    target_x_delta_m: float | None = None,
    move_duration_s: float | None = None,
    seed: int | None = None,
) -> EpisodeResult:
    options: dict = {"scenario": scenario}
    if target_x_delta_m is not None:
        options["target_x_delta_m"] = target_x_delta_m
    if move_duration_s is not None:
        options["move_duration_s"] = move_duration_s
    obs, info = env.reset(seed=seed, options=options)
    total_reward = 0.0
    max_abs_qd = 0.0
    last_info: dict = info
    terminated = truncated = False
    while not (terminated or truncated):
        obs, reward, terminated, truncated, last_info = env.step(action)
        total_reward += reward
        max_abs_qd = max(max_abs_qd, float(last_info["max_abs_qd_radps"]))
    return EpisodeResult(
        scenario=scenario.name,
        target_x_delta_m=float(info.get("target_x_delta_m", target_x_delta_m or 0.0)),
        move_duration_s=float(move_duration_s if move_duration_s is not None else env.cfg.move_duration_s),
        total_reward=total_reward,
        final_x_error=float(last_info["x_error"]),
        achieved_x_delta_m=float(last_info["achieved_x_delta_m"]),
        orientation_error=float(last_info["orientation_error"]),
        max_abs_qd_radps=max_abs_qd,
        guard_reason=last_info["guard_reason"],
    )


# Per-process lazy-cached env -- deliberately NOT passed as a differential_
# evolution arg. VelocityTransportEnv wraps a mujoco.MjModel/MjData (via
# LocalMujocoDynamics), which is NOT picklable (confirmed directly:
# pickle.dumps(env) raises "cannot pickle 'module' object") -- workers>1
# uses multiprocessing.Pool, which pickles func+args to ship to worker
# processes, so passing a live env instance in args would break immediately
# under any parallelism. Each worker process instead lazily builds its OWN
# env on first use and reuses it for every subsequent fitness() call in
# that process (differential_evolution calls fitness() many times per
# worker across generations) -- avoids paying MuJoCo's ~1.1s model-load
# cost on every single call, which would otherwise dominate the ~0.3s/
# episode simulation cost.
_worker_env: VelocityTransportEnv | None = None
_worker_env_config: VelocityTransportEnvConfig | None = None


def _get_worker_env(env_config: VelocityTransportEnvConfig | None, seed: int) -> VelocityTransportEnv:
    global _worker_env, _worker_env_config
    if _worker_env is None or _worker_env_config != env_config:
        _worker_env = VelocityTransportEnv(env_config, seed=seed)
        _worker_env_config = env_config
    return _worker_env


# Per-(scenario, direction) reward-weight multipliers applied inside
# fitness() before averaging (added 2026-08-06). Rationale: a flat,
# unweighted mean lets a scenario that passes everywhere easily
# (unrotated_wrist2offset) dilute a genuinely-failing scenario/direction's
# contribution to the aggregate objective, simply because there are as
# many easy-pass cells as hard-fail cells. Direct evidence this was
# happening: three independent searches (unseeded, seeded, and one with a
# 6th ik_posture_gain dimension freely available) all converged to nearly
# the SAME extreme (pinv_damping, qp_task_weight) corner regardless of
# what else was available -- the newly-added ik_posture_gain dimension
# went entirely unused (set to exactly 0.0) even though it demonstrably
# helps in isolated, well-conditioned tests, consistent with the mean
# being dominated by unrotated_wrist2offset's strong reward rather than
# genuinely trading off against hanging_alpha_0_5's -X failures (whose
# pass rate actually dropped, 57%->50%, once -X was added to the grid).
# 5.0x on hanging_alpha_0_5's negative-x cells specifically is a first,
# reasoned-but-unvalidated value -- meant to meaningfully shift the
# incentive without one failure class completely dominating the
# objective; whether it's enough is exactly what the next search result
# will show. Scoped to hanging_alpha_0_5/-X specifically (not a blanket
# reweight) because that's the concrete gap this was built to close --
# extend this dict, don't hardcode a new mechanism, if another scenario/
# direction needs the same treatment later.
DEFAULT_CELL_WEIGHT_OVERRIDES: dict[tuple[str, str], float] = {
    ("hanging_alpha_0_5", "negative_x"): 5.0,
}


def _cell_direction(result: EpisodeResult) -> str:
    return "negative_x" if result.target_x_delta_m < 0.0 else "positive_x"


def _cell_weight(result: EpisodeResult, overrides: dict[tuple[str, str], float]) -> float:
    return overrides.get((result.scenario, _cell_direction(result)), 1.0)


def fitness(
    action: np.ndarray,
    scenarios: tuple[PoseScenario, ...],
    env_config: VelocityTransportEnvConfig | None,
    seed: int,
    *,
    dx_fractions: tuple[float, ...] = (0.5, 0.9, 1.15, -0.5, -0.9, -1.15),
    fast_move_dx_fractions: tuple[float, ...] = (0.5, 1.0, -0.5, -1.0),
    cell_weight_overrides: dict[tuple[str, str], float] | None = None,
) -> float:
    """Negative WEIGHTED mean total_reward across (scenario x dx_fraction)
    evaluation cells -- differential_evolution minimizes, so this is what
    it calls. cell_weight_overrides defaults to DEFAULT_CELL_WEIGHT_
    OVERRIDES (see its module-level docstring for the full rationale):
    without it this would be a flat, unweighted mean, which let scenarios
    that pass everywhere easily dilute a genuinely-failing scenario/
    direction's contribution simply by having as many easy cells as hard
    ones.

    dx_fractions default to a spread around each scenario's known boundary
    (max_dx_hint_m) so the fitness rewards gains that hold up both well
    inside the safe range (0.5x) and right at/past the edge (1.15x), not
    just an easy interior case. Includes BOTH signs by default (added
    2026-08-06, AGENTS.md sec 7's "always sweep both +X and -X" rule) --
    a real gain-search result that passed cleanly at +0.37m failed via
    joint_velocity_guard at -0.37m, a genuine asymmetry no prior search
    (which only ever swept positive fractions) could have seen, let alone
    optimized against. Bidirectional sweeping is now structural to the
    objective itself, not a manual follow-up someone has to remember to
    run -- the same pattern already used for FAST_MOVE_DURATION_S.

    fast_move_dx_fractions (added 2026-08-06): every fitness evaluation
    ALSO includes episodes at FAST_MOVE_DURATION_S (a near-step move),
    not just the default slow (1.0s) move_duration -- a real gain search
    without this found gains (ik_joint_gain~62) that scored well on the
    slow-move-only objective but produce joint velocity ~4.7 rad/s (guard
    3.0) on a fast move, a genuine safety gap the optimizer could not see
    and therefore could not avoid. This makes speed safety part of what
    the OPTIMIZER itself is scored on (a fast-move guard trip's reward
    penalty pulls the fitness down directly, the same mechanism that
    already discourages range-guard trips), not just something checked
    after the fact by run_search's auto_evaluate -- catching this class
    of gap during the search is strictly better than catching it in a
    post-hoc report, since the search can then actually steer away from
    it instead of just having it documented."""
    env = _get_worker_env(env_config, seed)
    results = []
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
    overrides = DEFAULT_CELL_WEIGHT_OVERRIDES if cell_weight_overrides is None else cell_weight_overrides
    rewards = np.array([r.total_reward for r in results], dtype=np.float64)
    weights = np.array([_cell_weight(r, overrides) for r in results], dtype=np.float64)
    weighted_mean_reward = float(np.average(rewards, weights=weights))
    return -weighted_mean_reward


def _build_seeded_population(popsize: int, seed_actions: list[np.ndarray], seed: int) -> np.ndarray:
    """Builds a (popsize*ACTION_DIM, ACTION_DIM) initial population for
    differential_evolution's ``init`` argument: a full Latin-hypercube
    sample of [-1,1]^ACTION_DIM (matching the default 'latinhypercube'
    init scipy would otherwise use), with the first len(seed_actions) rows
    overwritten by the supplied vectors -- see run_search's seed_actions
    docstring for why this doesn't narrow the search space. Pure numpy,
    deliberately factored out of run_search so it's unit-testable without
    running an actual (slow, mujoco-backed) search."""
    from scipy.stats.qmc import LatinHypercube

    n_members = popsize * ACTION_DIM
    population = LatinHypercube(d=ACTION_DIM, seed=seed).random(n=n_members) * 2.0 - 1.0
    for i, seed_action in enumerate(seed_actions[:n_members]):
        population[i] = np.clip(np.asarray(seed_action, dtype=np.float64).reshape(ACTION_DIM), -1.0, 1.0)
    return population


def run_search(
    *,
    scenarios: tuple[PoseScenario, ...] = POSE_SCENARIOS,
    env_config: VelocityTransportEnvConfig | None = None,
    maxiter: int = 40,
    popsize: int = 12,
    seed: int = 0,
    workers: int = 1,
    polish: bool = False,
    auto_evaluate: bool = True,
    eval_dx_fractions: tuple[float, ...] = (
        0.3, 0.6, 0.9, 1.0, 1.1, 1.3, 1.6, 2.0, -0.3, -0.6, -0.9, -1.0, -1.1, -1.3, -1.6, -2.0,
    ),
    seed_actions: list[np.ndarray] | None = None,
) -> dict:
    """Returns {"gains": {...}, "action": ndarray, "result": OptimizeResult,
    "safety_summary": {...} if auto_evaluate}.

    auto_evaluate (default True): runs the found gains through the full
    multi-pose evaluation grid (evaluate.py) automatically as the final
    step of every search, rather than requiring a manual follow-up call --
    a search result is not trustworthy until it's been checked against the
    safety grid, so this makes that check a standard, automatic part of
    the pipeline instead of something that can be silently skipped. Widens
    eval_dx_fractions past optimize's own search range (up to 2.0x each
    scenario's max_dx_hint_m, vs. the fitness function's own narrower
    0.5/0.9/1.15x during the search itself) specifically so a search result
    that improves the safe range gets its NEW boundary discovered, not just
    re-confirmation of the OLD hint -- this is what caught, in this exact
    session, that the widened-bounds search's found gains extended the
    hanging-pose boundary well past what the original max_dx_hint_m
    (itself calibrated for the OLD hand-tuned gains) anticipated.

    polish defaults to False -- found and fixed during this module's own
    end-to-end smoke test (2026-08-03): differential_evolution's polish
    step runs a GRADIENT-based local refinement (L-BFGS-B) on the best
    candidate after the population search finishes. This objective is
    deliberately noisy/discontinuous by design (a safety-guard trip is a
    sharp reward cliff, and episodes can early-terminate at different
    step counts for infinitesimally different inputs) -- exactly the kind
    of landscape gradient-based methods are NOT suited for, and L-BFGS-B's
    finite-difference gradient estimation on it was measured to make the
    whole search take vastly longer (a single polish step turned a ~30s
    run into something that did not finish within 180s in direct testing).
    This isn't a workaround for a slow host -- it's the correct choice
    for this specific objective's shape, matching the very reason
    derivative-free global optimization (CMA-ES/differential_evolution)
    was chosen over gradient methods in the first place. Pass polish=True
    explicitly only if you have a specific reason to try it anyway.

    workers>1 uses scipy's multiprocessing-based parallel evaluation --
    each worker process builds its own MuJoCo-backed env lazily (see
    _get_worker_env), so this is safe to run with workers set to the real
    core count of wherever this is executed. Set OPENBLAS_NUM_THREADS=1 /
    OMP_NUM_THREADS=1 / MKL_NUM_THREADS=1 / NUMEXPR_NUM_THREADS=1 in the
    environment before running with workers>1 on a shared host -- without
    that, each worker process's own numpy/scipy calls spawn a full-core-
    count BLAS thread pool, and workers*cores threads can blow through a
    per-user process limit.

    seed_actions (added 2026-08-06): known-good action vectors (e.g. a
    prior search's own result) to seed the initial population WITH,
    without narrowing the search space around them -- this is
    deliberately NOT a warm-start-and-shrink-bounds approach (the
    2026-08-06 session's explicit instruction was "search space should
    not be constrained... let us search and then see if they hit
    guardrails"). Instead, the rest of the population is still a full
    Latin-hypercube sample of the entire [-1,1]^ACTION_DIM box; only
    len(seed_actions) of the popsize*ACTION_DIM members are overwritten
    with the supplied vectors, giving differential_evolution's mutation/
    crossover operators a productive starting point to build from (e.g.
    exploring nearby pinv_damping/qp_task_weight combinations that might
    close a known gap) while every other member explores the space
    exactly as broadly as an unseeded search would."""
    bounds = [(-1.0, 1.0)] * ACTION_DIM
    init: str | np.ndarray = "latinhypercube"
    if seed_actions:
        init = _build_seeded_population(popsize, seed_actions, seed)
    result = differential_evolution(
        fitness,
        bounds,
        args=(scenarios, env_config, seed),
        maxiter=maxiter,
        popsize=popsize,
        seed=seed,
        workers=workers,
        polish=polish,
        updating="deferred" if workers != 1 else "immediate",
        init=init,
    )
    gains = action_to_gains(result.x)
    outcome = {"gains": gains, "action": result.x, "result": result}

    if auto_evaluate:
        # Local import to avoid a circular import (evaluate.py imports
        # run_episode/EpisodeResult from this module at module-load time;
        # this module must not import evaluate.py at module-load time
        # back, but a function-local import here is fine since both
        # modules are already fully loaded by the time run_search() is
        # actually called).
        from .evaluate import evaluate_gains, summarize_safety

        eval_results = evaluate_gains(
            result.x,
            scenarios=scenarios,
            dx_fractions=eval_dx_fractions,
            fast_move_dx_fractions=eval_dx_fractions,
            env_config=env_config,
            seed=seed,
        )
        outcome["safety_summary"] = summarize_safety(eval_results)
        outcome["eval_results"] = eval_results

    return outcome


if __name__ == "__main__":
    import argparse
    import json
    import time

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--maxiter", type=int, default=30)
    p.add_argument("--popsize", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--workers", type=int, default=1, help="Parallel worker processes (multiprocessing).")
    p.add_argument(
        "--polish", action="store_true",
        help="Enable differential_evolution's gradient-based local refinement. Off by default -- "
        "see run_search's docstring for why this objective's noisy/discontinuous shape makes "
        "polish behave very slowly.",
    )
    p.add_argument("--output-json", type=Path, default=None, help="Optional path to write the full result to.")
    p.add_argument(
        "--no-auto-evaluate", action="store_true",
        help="Skip the automatic post-search multi-pose safety evaluation. Default is to always "
        "run it -- a search result should never be reported without knowing whether it actually "
        "holds up across poses/displacements, not just the narrower dx range the search itself used.",
    )
    p.add_argument(
        "--seed-from-json", action="append", default=[], metavar="PATH",
        help="Path to a prior --output-json result (reads its 'action' field) to seed the initial "
        "population with -- see run_search's seed_actions docstring. Repeatable to seed with "
        "several prior results at once. Does NOT narrow the search bounds.",
    )
    args = p.parse_args()

    seed_actions = None
    if args.seed_from_json:
        seed_actions = []
        for path_str in args.seed_from_json:
            prior = json.loads(Path(path_str).read_text(encoding="utf-8"))
            action = np.array(prior["action"], dtype=np.float64)
            if action.shape[0] < ACTION_DIM:
                # A prior result from before a new ACTION_FIELDS entry was
                # added (e.g. ik_posture_gain, 2026-08-06) -- pad with -1.0
                # (each field's own lower-bound action value) for the
                # missing trailing dimensions, the only faithful
                # reinterpretation of a vector from before that dimension
                # existed (for ik_posture_gain specifically, -1.0 maps to
                # 0.0, i.e. "off", reproducing the historical vector's
                # original behavior byte-for-byte).
                pad = np.full(ACTION_DIM - action.shape[0], -1.0)
                action = np.concatenate([action, pad])
            seed_actions.append(action)

    t0 = time.time()
    outcome = run_search(
        maxiter=args.maxiter,
        popsize=args.popsize,
        seed=args.seed,
        workers=args.workers,
        polish=args.polish,
        auto_evaluate=not args.no_auto_evaluate,
        seed_actions=seed_actions,
    )
    elapsed_s = time.time() - t0

    print(json.dumps(outcome["gains"], indent=2))
    print(f"best fitness (negative mean reward): {outcome['result'].fun:.4f}")
    print(f"nfev: {outcome['result'].nfev}  elapsed: {elapsed_s:.1f}s")

    safety_summary = outcome.get("safety_summary")
    if safety_summary is not None:
        print()
        print(
            f"SAFETY: {safety_summary['n_pass']}/{safety_summary['n_total']} cells pass "
            f"({100 * safety_summary['pass_fraction']:.0f}%), "
            f"worst orientation error {safety_summary['worst_orientation_error_rad']:.4f} rad, "
            f"worst |qd| {safety_summary['worst_abs_qd_radps']:.3f} rad/s"
        )
        slow = safety_summary["slow_move"]
        fast = safety_summary["fast_move"]
        print(
            f"  slow_move (range/tracking grid): {slow['n_pass']}/{slow['n_total']} pass, "
            f"worst |qd|={slow['worst_abs_qd_radps']:.3f} rad/s"
        )
        print(
            f"  fast_move (speed-safety grid, {FAST_MOVE_DURATION_S}s): {fast['n_pass']}/{fast['n_total']} pass, "
            f"worst |qd|={fast['worst_abs_qd_radps']:.3f} rad/s"
        )
        for name, cell in safety_summary["per_scenario"].items():
            print(f"  {name:<24} pass={cell['pass']} fail={cell['fail']} "
                  f"max_passing_dx_m={cell['max_passing_dx_m']:.4f}")
        if safety_summary["guard_trips"]:
            print(f"  {len(safety_summary['guard_trips'])} guard trip(s) -- see safety_summary.guard_trips "
                  f"in the output JSON for details.")

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps(
                {
                    "gains": outcome["gains"],
                    "action": outcome["action"].tolist(),
                    "fitness": float(outcome["result"].fun),
                    "nfev": int(outcome["result"].nfev),
                    "elapsed_s": elapsed_s,
                    "maxiter": args.maxiter,
                    "popsize": args.popsize,
                    "seed": args.seed,
                    "workers": args.workers,
                    "safety_summary": safety_summary,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"wrote {args.output_json}")
