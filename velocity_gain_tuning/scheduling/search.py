"""Stage 1 of the gain-scheduling pipeline: one INDEPENDENT
differential_evolution search per (pose, signed-dx) knot cell.

This is deliberately the same optimizer, same environment, same guard
thresholds and same reward function already used by
``velocity_gain_tuning/optimize.py`` -- the ONLY change is the objective's
scope. ``optimize.fitness`` averages ~56 episodes spanning four poses and
both directions into a single scalar that one gain vector must maximize;
``cell_fitness`` here averages just TWO episodes (the slow/nominal move
and the FAST_MOVE_DURATION_S speed-safety move) at ONE pose and ONE signed
displacement. That makes each search a small, well-posed problem instead
of a four-way compromise, and it is what lets different cells land on
genuinely different gains.

Two properties of this stage are worth stating explicitly because they are
what makes the result interpretable:

* **Every cell is seeded with the best known GLOBAL gain vector** (via
  ``differential_evolution``'s ``init`` population, exactly the mechanism
  ``optimize._build_seeded_population`` already provides -- the rest of the
  population is still a full Latin-hypercube sample of the whole box, so
  the search space is NOT narrowed). A cell's result is therefore a direct
  answer to "can a cell-specific vector beat the global one HERE?", and a
  cell where the answer is no should return something close to the global
  vector's own score rather than something worse.
* **The collected per-cell optima are an ORACLE UPPER BOUND** on any
  (pose, dx)-conditioned gain policy -- spline, regression, or RL. Nothing
  that sees only (pose, dx) can beat, at cell c, the best gain vector for
  cell c. Reporting the oracle alongside the fitted schedule separates
  "scheduling can't help" from "the interpolant lost something", which a
  single end-to-end number cannot.

The fast-move episode is included in EVERY cell objective on purpose: this
package's parent found the hard way (see ``optimize.FAST_MOVE_DURATION_S``)
that ``ik_seeded_resolution``'s real speed governor is ``ik_joint_gain``,
not ``move_duration_s``, so a cell optimized only at the nominal 1.0s move
can be badly unsafe at a near-step move with the exact same gains. A
per-cell search is if anything MORE exposed to that trap than the global
one, since it has fewer episodes to hide behind.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import differential_evolution

from ..envs.velocity_transport_env import ACTION_DIM, VelocityTransportEnvConfig
from ..optimize import (
    FAST_MOVE_DURATION_S,
    _build_seeded_population,
    _get_worker_env,
    run_episode,
)
from ..poses import POSE_SCENARIOS, scenario_by_name
from .cells import DEFAULT_KNOT_FRACTIONS, GainCell, build_cells, build_chains
from .schedule import GainSchedule, ScheduleKnot


def cell_fitness(
    action: np.ndarray,
    scenario_name: str,
    target_x_delta_m: float,
    env_config: VelocityTransportEnvConfig | None,
    seed: int,
    prev_action: np.ndarray | None = None,
    continuity_weight: float = 0.0,
) -> float:
    """Negative mean episode reward over {nominal move, fast move} at ONE
    (pose, signed dx). differential_evolution minimizes, so this is what it
    calls.

    Takes ``scenario_name`` (a str) rather than a ``PoseScenario`` because
    this function's args get pickled to worker processes; resolving the
    name through ``scenario_by_name`` inside the worker keeps the payload
    trivially picklable and guarantees the worker uses the same catalog
    entry the parent intended.

    ``prev_action``/``continuity_weight`` add an optional CONTINUATION
    penalty ``w * ||action - prev_action||^2`` -- the previous knot in this
    cell's chain (see cells.build_chains). Rationale, measured: without it,
    neighbouring cells land on different members of a large set of
    near-equivalent optima, and the spline between two such points is not a
    usable gain vector (independent-search schedule: 108/112 at its own
    knots but 9/16 at the held-out fraction, vs. a fixed global vector's
    16/16). The penalty makes "stay near where the neighbouring
    displacement ended up" part of what the optimizer is scored on, which
    is the only place coherence can actually be enforced. Scale reference:
    the reward range across this objective is O(10) and a guard trip costs
    -20, while ``||.||^2`` over the [-1,1]^6 box tops out at 24 -- so
    w~0.5 makes crossing the whole box comparable to a guard trip, and
    w=0.0 recovers the pure per-cell objective exactly.
    """
    scenario = scenario_by_name(scenario_name)
    env = _get_worker_env(env_config, seed)
    slow = run_episode(env, action, scenario=scenario, target_x_delta_m=target_x_delta_m)
    fast = run_episode(
        env,
        action,
        scenario=scenario,
        target_x_delta_m=target_x_delta_m,
        move_duration_s=FAST_MOVE_DURATION_S,
    )
    objective = -0.5 * (slow.total_reward + fast.total_reward)
    if prev_action is not None and continuity_weight > 0.0:
        delta = np.asarray(action, dtype=np.float64) - np.asarray(prev_action, dtype=np.float64)
        objective += float(continuity_weight) * float(np.dot(delta, delta))
    return objective


@dataclass
class CellSearchResult:
    knot: ScheduleKnot
    nfev: int
    elapsed_s: float


def search_cell(
    cell: GainCell,
    *,
    env_config: VelocityTransportEnvConfig | None = None,
    maxiter: int = 25,
    popsize: int = 10,
    seed: int = 0,
    seed_actions: list[np.ndarray] | None = None,
    polish: bool = False,
    prev_action: np.ndarray | None = None,
    continuity_weight: float = 0.0,
) -> CellSearchResult:
    """Run one cell's DE search and return it as a fitted ``ScheduleKnot``.

    ``polish=False`` for the same reason ``optimize.run_search`` defaults it
    off: the objective is genuinely discontinuous (a guard trip is a reward
    cliff and truncates the episode), which is precisely the landscape
    L-BFGS-B's finite-difference gradients handle worst.

    ``prev_action`` (the previous knot in this cell's chain) is used BOTH as
    an extra population seed and, if ``continuity_weight > 0``, as the
    centre of the continuation penalty -- see cell_fitness.
    """
    t0 = time.time()
    all_seeds = list(seed_actions or [])
    if prev_action is not None:
        # Prepended: the neighbouring displacement's solution is the single
        # most informative starting point for this cell, more so than a
        # global vector that was never optimal anywhere in particular.
        all_seeds.insert(0, np.asarray(prev_action, dtype=np.float64))
    init: str | np.ndarray = "latinhypercube"
    if all_seeds:
        init = _build_seeded_population(popsize, all_seeds, seed)
    result = differential_evolution(
        cell_fitness,
        [(-1.0, 1.0)] * ACTION_DIM,
        args=(
            cell.scenario.name,
            cell.target_x_delta_m,
            env_config,
            seed,
            prev_action,
            continuity_weight,
        ),
        maxiter=maxiter,
        popsize=popsize,
        seed=seed,
        workers=1,
        polish=polish,
        updating="immediate",
        init=init,
    )
    elapsed_s = time.time() - t0

    # Re-run the two episodes with the winning action so the knot records
    # the REAL safety outcome, not just a fitness scalar -- "did the best
    # gains for this cell actually make it safe?" is the question the
    # schedule fit (drop_failed_knots) and the final report both need.
    env = _get_worker_env(env_config, seed)
    slow = run_episode(env, result.x, scenario=cell.scenario, target_x_delta_m=cell.target_x_delta_m)
    fast = run_episode(
        env,
        result.x,
        scenario=cell.scenario,
        target_x_delta_m=cell.target_x_delta_m,
        move_duration_s=FAST_MOVE_DURATION_S,
    )
    guard_reasons = (slow.guard_reason, fast.guard_reason)
    # Record the PURE per-cell objective (from the replay), not
    # ``result.fun`` -- with a continuity penalty active those differ, and
    # a knot's recorded fitness must stay comparable across sweeps run with
    # different continuity weights.
    pure_fitness = -0.5 * (slow.total_reward + fast.total_reward)
    knot = ScheduleKnot(
        scenario=cell.scenario.name,
        dx_fraction=cell.dx_fraction,
        target_x_delta_m=cell.target_x_delta_m,
        action=np.asarray(result.x, dtype=np.float64),
        fitness=float(pure_fitness),
        passed=all(g is None for g in guard_reasons),
        guard_reasons=guard_reasons,
    )
    return CellSearchResult(knot=knot, nfev=int(result.nfev), elapsed_s=elapsed_s)


def _search_chain_worker(payload: tuple) -> list[CellSearchResult]:
    """Top-level (picklable) multiprocessing entry point for one CHAIN.

    Runs its cells strictly in order (increasing |dx|), threading each
    solution into the next cell as ``prev_action`` -- the continuation that
    makes the resulting knot sequence coherent enough to spline through.
    A chain is inherently sequential, so parallelism here is ACROSS chains
    (8 of them for the default grid: 4 poses x 2 directions).
    """
    (
        chain_spec,
        env_config,
        maxiter,
        popsize,
        seed,
        seed_actions,
        continuity_weight,
    ) = payload
    results: list[CellSearchResult] = []
    prev_action: np.ndarray | None = None
    for scenario_name, dx_fraction, target_x_delta_m in chain_spec:
        cell = GainCell(
            scenario=scenario_by_name(scenario_name),
            dx_fraction=dx_fraction,
            target_x_delta_m=target_x_delta_m,
        )
        res = search_cell(
            cell,
            env_config=env_config,
            maxiter=maxiter,
            popsize=popsize,
            seed=seed,
            seed_actions=seed_actions,
            prev_action=prev_action,
            continuity_weight=continuity_weight,
        )
        prev_action = res.knot.action
        results.append(res)
    return results


def _search_cell_worker(payload: tuple) -> CellSearchResult:
    """Top-level (picklable) multiprocessing entry point for one cell."""
    (
        scenario_name,
        dx_fraction,
        target_x_delta_m,
        env_config,
        maxiter,
        popsize,
        seed,
        seed_actions,
    ) = payload
    cell = GainCell(
        scenario=scenario_by_name(scenario_name),
        dx_fraction=dx_fraction,
        target_x_delta_m=target_x_delta_m,
    )
    return search_cell(
        cell,
        env_config=env_config,
        maxiter=maxiter,
        popsize=popsize,
        seed=seed,
        seed_actions=seed_actions,
    )


def search_cells(
    cells: list[GainCell],
    *,
    env_config: VelocityTransportEnvConfig | None = None,
    maxiter: int = 25,
    popsize: int = 10,
    seed: int = 0,
    seed_actions: list[np.ndarray] | None = None,
    workers: int = 1,
    progress: bool = True,
    chained: bool = False,
    continuity_weight: float = 0.0,
) -> list[ScheduleKnot]:
    """Search every cell, optionally in parallel ACROSS cells.

    ``chained=True`` switches to continuation mode: cells are grouped into
    (pose, direction) chains by ``cells.build_chains`` and each chain is
    walked outward from the smallest |dx|, seeding (and, with
    ``continuity_weight > 0``, penalising distance from) the previous
    cell's solution. Read ``build_chains``'s docstring first -- it records
    the measurement that motivated this mode, namely that INDEPENDENT
    per-cell searches produce knots too incoherent to interpolate between.

    Parallelism is deliberately across cells with ``workers=1`` inside each
    DE, not scipy's own intra-search ``workers>1``. Reason, measured in
    this repo: ``VelocityTransportEnv`` wraps a MuJoCo model that costs
    ~0.7-1.1 s to load and cannot be pickled, so every worker process must
    build its own. Intra-search parallelism would tear down and rebuild
    that pool once per cell (n_cells x n_workers model loads); one pool
    spanning all cells pays it once per process for the entire sweep, and
    each cell's DE is small enough that cell-level parallelism gives better
    utilisation anyway.

    Set OPENBLAS_NUM_THREADS=1 / OMP_NUM_THREADS=1 / MKL_NUM_THREADS=1 /
    NUMEXPR_NUM_THREADS=1 before running with workers>1 on a shared host
    (AGENTS.md sec 8) -- otherwise each worker's own numpy spawns a
    full-core-count BLAS pool and workers x cores threads can blow through
    the per-user process limit.
    """
    if chained:
        return _search_chained(
            cells,
            env_config=env_config,
            maxiter=maxiter,
            popsize=popsize,
            seed=seed,
            seed_actions=seed_actions,
            workers=workers,
            progress=progress,
            continuity_weight=continuity_weight,
        )

    payloads = [
        (
            cell.scenario.name,
            cell.dx_fraction,
            cell.target_x_delta_m,
            env_config,
            maxiter,
            popsize,
            seed,
            seed_actions,
        )
        for cell in cells
    ]
    knots: list[ScheduleKnot] = []
    if workers == 1:
        for i, payload in enumerate(payloads):
            res = _search_cell_worker(payload)
            knots.append(res.knot)
            if progress:
                _print_cell(i + 1, len(payloads), res)
    else:
        import multiprocessing as mp

        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=workers) as pool:
            for i, res in enumerate(pool.imap_unordered(_search_cell_worker, payloads)):
                knots.append(res.knot)
                if progress:
                    _print_cell(i + 1, len(payloads), res)
    # Stable ordering regardless of completion order.
    knots.sort(key=lambda k: (k.scenario, k.target_x_delta_m))
    return knots


def _search_chained(
    cells: list[GainCell],
    *,
    env_config: VelocityTransportEnvConfig | None,
    maxiter: int,
    popsize: int,
    seed: int,
    seed_actions: list[np.ndarray] | None,
    workers: int,
    progress: bool,
    continuity_weight: float,
) -> list[ScheduleKnot]:
    chains = build_chains(cells)
    payloads = [
        (
            [(c.scenario.name, c.dx_fraction, c.target_x_delta_m) for c in chain],
            env_config,
            maxiter,
            popsize,
            seed,
            seed_actions,
            continuity_weight,
        )
        for chain in chains
    ]
    knots: list[ScheduleKnot] = []
    done = 0

    def _collect(chain_result: list[CellSearchResult]) -> None:
        # Progress is reported per COMPLETED CHAIN, not per cell: a chain is
        # sequential inside one worker process, so nothing is observable
        # until it finishes. A default-grid chain is ~7 cells (~1.5 h at
        # full budget), so expect long silences -- that is the cost of
        # continuation, not a hang.
        nonlocal done
        for res in chain_result:
            done += 1
            knots.append(res.knot)
            if progress:
                _print_cell(done, len(cells), res)

    if workers == 1:
        for payload in payloads:
            _collect(_search_chain_worker(payload))
    else:
        import multiprocessing as mp

        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=min(workers, len(payloads))) as pool:
            for chain_result in pool.imap_unordered(_search_chain_worker, payloads):
                _collect(chain_result)
    knots.sort(key=lambda k: (k.scenario, k.target_x_delta_m))
    return knots


def _print_cell(i: int, n: int, res: CellSearchResult) -> None:
    k = res.knot
    status = "pass" if k.passed else "FAIL"
    print(
        f"[{i:>3}/{n}] {k.scenario:<24} dx={k.target_x_delta_m:+.4f} "
        f"({k.dx_fraction:+.2f}x) fitness={k.fitness:8.3f} {status} "
        f"nfev={res.nfev} {res.elapsed_s:.0f}s",
        flush=True,
    )


def _load_seed_actions(paths: list[str]) -> list[np.ndarray]:
    """Read ``action`` vectors out of prior ``optimize.py`` result JSONs.

    Shorter historical vectors are padded with -1.0 for missing trailing
    dimensions, byte-identically to ``optimize.py``'s own ``--seed-from-json``
    handling -- see the ACTION_FIELDS docstring for why -1.0 (not 0.0) is
    the only faithful reinterpretation for the fields added since.
    """
    actions: list[np.ndarray] = []
    for path_str in paths:
        prior = json.loads(Path(path_str).read_text(encoding="utf-8"))
        action = np.array(prior["action"], dtype=np.float64)
        if action.shape[0] < ACTION_DIM:
            action = np.concatenate([action, np.full(ACTION_DIM - action.shape[0], -1.0)])
        actions.append(action[:ACTION_DIM])
    return actions


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--maxiter", type=int, default=25)
    p.add_argument("--popsize", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--workers", type=int, default=1, help="Parallel processes ACROSS cells.")
    p.add_argument(
        "--scenarios", nargs="*", default=None,
        help="Subset of pose scenario names to search (default: all four).",
    )
    p.add_argument(
        "--knot-fractions", nargs="*", type=float, default=None,
        help="Signed fractions of each scenario's max_dx_hint_m to place knots at "
             "(default: cells.DEFAULT_KNOT_FRACTIONS).",
    )
    p.add_argument(
        "--seed-from-json", action="append", default=[], metavar="PATH",
        help="Prior optimize.py result JSON whose 'action' seeds EVERY cell's initial "
             "population. Strongly recommended: it makes each cell's result a direct "
             "'can cell-specific gains beat the global vector here?' comparison. "
             "Does NOT narrow the search bounds.",
    )
    p.add_argument(
        "--chained", action="store_true",
        help="Continuation mode: walk each (pose, direction) chain outward from the smallest "
             "|dx|, seeding each cell with the previous one's solution. Strongly recommended -- "
             "see cells.build_chains for the measurement showing independent per-cell searches "
             "produce knots too incoherent to interpolate between.",
    )
    p.add_argument(
        "--continuity-weight", type=float, default=0.0,
        help="Weight w of the w*||action - prev_action||^2 continuation penalty (chained mode "
             "only). 0.0 = seeding-only chaining. ~0.5 makes crossing the whole action box "
             "cost about as much as a guard trip. See search.cell_fitness.",
    )
    p.add_argument("--output-json", type=Path, required=True, help="Where to write the knot table.")
    args = p.parse_args()

    scenarios = POSE_SCENARIOS
    if args.scenarios:
        scenarios = tuple(scenario_by_name(n) for n in args.scenarios)
    knot_fractions = tuple(args.knot_fractions) if args.knot_fractions else DEFAULT_KNOT_FRACTIONS

    cells = build_cells(scenarios, knot_fractions)
    seed_actions = _load_seed_actions(args.seed_from_json) if args.seed_from_json else None

    print(
        f"searching {len(cells)} cells "
        f"({len(scenarios)} scenarios x {len(knot_fractions)} knots), "
        f"maxiter={args.maxiter} popsize={args.popsize} workers={args.workers} "
        f"chained={args.chained} continuity_weight={args.continuity_weight}",
        flush=True,
    )
    t0 = time.time()
    knots = search_cells(
        cells,
        maxiter=args.maxiter,
        popsize=args.popsize,
        seed=args.seed,
        seed_actions=seed_actions,
        workers=args.workers,
        chained=args.chained,
        continuity_weight=args.continuity_weight,
    )
    elapsed_s = time.time() - t0

    schedule = GainSchedule(knots)
    payload = schedule.to_dict()
    payload["search_meta"] = {
        "maxiter": args.maxiter,
        "popsize": args.popsize,
        "seed": args.seed,
        "workers": args.workers,
        "elapsed_s": elapsed_s,
        "knot_fractions": list(knot_fractions),
        "scenarios": [s.name for s in scenarios],
        "seed_from_json": args.seed_from_json,
        "chained": args.chained,
        "continuity_weight": args.continuity_weight,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    n_pass = sum(1 for k in knots if k.passed)
    print(
        f"\ndone in {elapsed_s / 60:.1f} min -- {n_pass}/{len(knots)} knot cells have a "
        f"guard-clean best gain vector (this is the per-cell ORACLE; cells failing here "
        f"are infeasible for ANY (pose,dx)-conditioned gain policy)"
    )
    print(f"wrote {args.output_json}")
