"""MuJoCo-backed tests for velocity_gain_tuning/scheduling/ -- the parts that
actually roll out episodes: the per-cell objective (search.cell_fitness),
the per-cell DE search (search.search_cell), and the scheduled evaluation
grid (evaluate.evaluate_schedule / evaluate_fixed_action).

Kept separate from tests/unit/test_gain_schedule_interpolation.py, which
covers the simulator-free interpolation layer. Added 2026-08-06 per
AGENTS.md sec 5.

Everything here uses deliberately tiny DE budgets and 1-2 cell grids: the
purpose is to prove the wiring is correct and the contracts hold, not to
find good gains (that is what the real sweep on ilab is for). A full-budget
search is ~15 min/cell and belongs nowhere near a test suite.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from velocity_gain_tuning.envs.velocity_transport_env import (  # noqa: E402
    ACTION_DIM,
    VelocityTransportEnv,
    VelocityTransportEnvConfig,
)
from velocity_gain_tuning.evaluate import evaluate_gains, summarize_safety  # noqa: E402
from velocity_gain_tuning.optimize import FAST_MOVE_DURATION_S, run_episode  # noqa: E402
from velocity_gain_tuning.poses import POSE_SCENARIOS, scenario_by_name  # noqa: E402
from velocity_gain_tuning.scheduling.cells import GainCell, build_cells  # noqa: E402
from velocity_gain_tuning.scheduling.evaluate import (  # noqa: E402
    STANDARD_EVAL_DX_FRACTIONS,
    compare_summaries,
    evaluate_fixed_action,
    evaluate_schedule,
    knot_coverage_split,
    knot_fractions_from_schedule,
    print_comparison,
)
from velocity_gain_tuning.scheduling.schedule import GainSchedule, ScheduleKnot  # noqa: E402
from velocity_gain_tuning.scheduling.search import (  # noqa: E402
    _load_seed_actions,
    cell_fitness,
    search_cell,
    search_cells,
)

SCENARIO = POSE_SCENARIOS[0]  # neg40_wrist2offset -- smallest safe range, fastest episodes
MID_ACTION = np.zeros(ACTION_DIM)


@pytest.fixture(scope="module")
def env() -> VelocityTransportEnv:
    return VelocityTransportEnv(None, seed=0)


def _cell(frac: float) -> GainCell:
    return GainCell(
        scenario=SCENARIO,
        dx_fraction=frac,
        target_x_delta_m=frac * SCENARIO.max_dx_hint_m,
    )


# --------------------------------------------------------------------------
# cell_fitness
# --------------------------------------------------------------------------


def test_cell_fitness_is_deterministic():
    dx = 0.5 * SCENARIO.max_dx_hint_m
    a = cell_fitness(MID_ACTION, SCENARIO.name, dx, None, 0)
    b = cell_fitness(MID_ACTION, SCENARIO.name, dx, None, 0)
    assert a == pytest.approx(b)


def test_cell_fitness_equals_negative_mean_of_slow_and_fast_rewards(env):
    """The per-cell objective must be exactly the mean of the two episodes
    it claims to average -- if this drifts, every knot in a searched table
    is optimizing something other than what the docstring says."""
    dx = 0.5 * SCENARIO.max_dx_hint_m
    slow = run_episode(env, MID_ACTION, scenario=SCENARIO, target_x_delta_m=dx)
    fast = run_episode(
        env, MID_ACTION, scenario=SCENARIO, target_x_delta_m=dx, move_duration_s=FAST_MOVE_DURATION_S
    )
    expected = -0.5 * (slow.total_reward + fast.total_reward)
    assert cell_fitness(MID_ACTION, SCENARIO.name, dx, None, 0) == pytest.approx(expected, rel=1e-9)


def test_cell_fitness_includes_the_fast_move_dimension():
    """Regression guard for this package's parent's hard-won lesson (see
    optimize.FAST_MOVE_DURATION_S): an objective that only scores the
    nominal 1.0s move cannot see gains that are wildly unsafe at a
    near-step move. A per-cell objective averages only two episodes, so
    silently losing the fast one would halve the safety signal.

    Constructed by finding gains whose slow and fast episodes score
    differently, then asserting the objective moves with the fast one.
    """
    dx = 0.5 * SCENARIO.max_dx_hint_m
    # ik_joint_gain at its maximum: the field that actually governs speed.
    fast_action = np.array([0.0, 0.0, 1.0, 0.0, 0.0, -1.0])
    env_local = VelocityTransportEnv(None, seed=0)
    slow = run_episode(env_local, fast_action, scenario=SCENARIO, target_x_delta_m=dx)
    fast = run_episode(
        env_local, fast_action, scenario=SCENARIO, target_x_delta_m=dx,
        move_duration_s=FAST_MOVE_DURATION_S,
    )
    assert slow.total_reward != pytest.approx(fast.total_reward), (
        "test premise broken: need an action whose slow/fast rewards differ"
    )
    objective = cell_fitness(fast_action, SCENARIO.name, dx, None, 0)
    assert objective == pytest.approx(-0.5 * (slow.total_reward + fast.total_reward), rel=1e-9)
    # And it is genuinely not just the slow episode negated.
    assert objective != pytest.approx(-slow.total_reward)


def test_cell_fitness_signs_are_negated_rewards():
    """differential_evolution MINIMIZES, so a better (higher-reward) action
    must produce a LOWER objective value."""
    dx = 0.5 * SCENARIO.max_dx_hint_m
    # An action known to trip the joint-velocity guard on this pose (max
    # ik_joint_gain, minimal damping) should score worse than the midpoint.
    reckless = np.array([1.0, 1.0, 1.0, -1.0, 1.0, -1.0])
    assert cell_fitness(reckless, SCENARIO.name, dx, None, 0) > cell_fitness(
        MID_ACTION, SCENARIO.name, dx, None, 0
    )


# --------------------------------------------------------------------------
# search_cell / search_cells
# --------------------------------------------------------------------------


def test_cell_fitness_continuity_penalty_is_exact_and_opt_in():
    """The continuation penalty must be exactly w*||a - prev||^2 on top of
    the unmodified per-cell objective, and must vanish at w=0 -- otherwise
    knot fitness values are not comparable between a chained sweep and an
    independent one."""
    dx = 0.5 * SCENARIO.max_dx_hint_m
    prev = np.full(ACTION_DIM, 0.5)
    base = cell_fitness(MID_ACTION, SCENARIO.name, dx, None, 0)
    assert cell_fitness(MID_ACTION, SCENARIO.name, dx, None, 0, prev, 0.0) == pytest.approx(base)
    penalised = cell_fitness(MID_ACTION, SCENARIO.name, dx, None, 0, prev, 2.0)
    expected = base + 2.0 * float(np.dot(prev, prev))  # MID_ACTION is all zeros
    assert penalised == pytest.approx(expected, rel=1e-9)


def test_cell_fitness_continuity_penalty_is_zero_at_the_previous_action():
    dx = 0.5 * SCENARIO.max_dx_hint_m
    prev = np.full(ACTION_DIM, 0.25)
    assert cell_fitness(prev, SCENARIO.name, dx, None, 0, prev, 10.0) == pytest.approx(
        cell_fitness(prev, SCENARIO.name, dx, None, 0)
    )


def test_search_cell_returns_a_consistent_knot():
    res = search_cell(_cell(0.5), maxiter=1, popsize=2, seed=0)
    knot = res.knot
    assert knot.scenario == SCENARIO.name
    assert knot.dx_fraction == pytest.approx(0.5)
    assert knot.target_x_delta_m == pytest.approx(0.5 * SCENARIO.max_dx_hint_m)
    assert knot.action.shape == (ACTION_DIM,)
    assert np.all(knot.action >= -1.0) and np.all(knot.action <= 1.0)
    assert len(knot.guard_reasons) == 2
    assert knot.passed == all(g is None for g in knot.guard_reasons)
    assert res.nfev > 0


def test_search_cell_pass_flag_matches_a_direct_replay(env):
    """``passed`` must reflect a real re-run of the winning action, not the
    fitness scalar -- the schedule's drop_failed_knots option and the
    reported per-cell oracle both depend on that flag being literally
    true."""
    res = search_cell(_cell(0.5), maxiter=1, popsize=2, seed=0)
    slow = run_episode(
        env, res.knot.action, scenario=SCENARIO, target_x_delta_m=res.knot.target_x_delta_m
    )
    fast = run_episode(
        env, res.knot.action, scenario=SCENARIO,
        target_x_delta_m=res.knot.target_x_delta_m, move_duration_s=FAST_MOVE_DURATION_S,
    )
    assert res.knot.guard_reasons == (slow.guard_reason, fast.guard_reason)


def test_search_cell_seeding_places_the_seed_in_the_population():
    """A seeded cell search must be able to return the seed itself when the
    seed is the best thing in the population -- this is what makes a cell's
    result a valid 'can cell-specific gains beat the global vector here?'
    comparison rather than an unrelated random restart."""
    seed_action = np.array([-0.5452930656, -0.3120110339, 0.1960343548, -0.4031948187,
                            0.6634521667, -0.2987716573])
    res = search_cell(_cell(0.5), maxiter=0, popsize=2, seed=0, seed_actions=[seed_action])
    seeded_fitness = cell_fitness(seed_action, SCENARIO.name, _cell(0.5).target_x_delta_m, None, 0)
    assert res.knot.fitness <= seeded_fitness + 1e-6


def test_search_cells_serial_returns_sorted_knots():
    cells = [_cell(0.5), _cell(-0.5)]
    knots = search_cells(cells, maxiter=1, popsize=2, seed=0, workers=1, progress=False)
    assert len(knots) == 2
    dxs = [k.target_x_delta_m for k in knots]
    assert dxs == sorted(dxs), "knots must come back in stable (scenario, dx) order"


@pytest.mark.slow
def test_search_cells_parallel_matches_serial():
    """Cross-process parallelism must not change results -- each worker
    lazily builds its own MuJoCo env, and a stale/shared env would show up
    here as a mismatch."""
    cells = [_cell(0.5), _cell(-0.5)]
    serial = search_cells(cells, maxiter=1, popsize=2, seed=0, workers=1, progress=False)
    parallel = search_cells(cells, maxiter=1, popsize=2, seed=0, workers=2, progress=False)
    for a, b in zip(serial, parallel):
        assert a.scenario == b.scenario
        assert a.target_x_delta_m == pytest.approx(b.target_x_delta_m)
        assert np.allclose(a.action, b.action)


def test_search_cell_records_pure_fitness_not_the_penalised_objective():
    """A knot's recorded fitness must be the pure per-cell objective, so
    tables from sweeps run with different continuity weights stay directly
    comparable. Checked by replaying the winning action through the
    unpenalised objective."""
    prev = np.full(ACTION_DIM, 0.6)
    res = search_cell(
        _cell(0.5), maxiter=1, popsize=2, seed=0, prev_action=prev, continuity_weight=5.0
    )
    pure = cell_fitness(res.knot.action, SCENARIO.name, _cell(0.5).target_x_delta_m, None, 0)
    assert res.knot.fitness == pytest.approx(pure, rel=1e-9)


def test_search_cell_continuity_penalty_pulls_the_solution_toward_prev_action():
    """With a large enough weight the penalty must dominate -- if it does
    not, chained mode is not actually enforcing continuity and the whole
    mechanism is inert."""
    prev = np.array([0.2, -0.2, 0.1, -0.1, 0.3, -0.3])
    free = search_cell(_cell(0.9), maxiter=3, popsize=3, seed=1)
    pinned = search_cell(
        _cell(0.9), maxiter=3, popsize=3, seed=1, prev_action=prev, continuity_weight=50.0
    )
    assert np.linalg.norm(pinned.knot.action - prev) < np.linalg.norm(free.knot.action - prev)


def test_search_cells_chained_returns_every_cell_once():
    cells = [_cell(f) for f in (0.3, 0.6, -0.3, -0.6)]
    knots = search_cells(
        cells, maxiter=1, popsize=2, seed=0, workers=1, progress=False,
        chained=True, continuity_weight=0.5,
    )
    assert len(knots) == len(cells)
    dxs = [round(k.target_x_delta_m, 6) for k in knots]
    assert sorted(dxs) == dxs
    assert len(set(dxs)) == len(cells)


@pytest.mark.slow
def test_search_cells_chained_parallel_matches_serial():
    cells = [_cell(f) for f in (0.3, 0.6, -0.3, -0.6)]
    kw = dict(maxiter=1, popsize=2, seed=0, progress=False, chained=True, continuity_weight=0.5)
    serial = search_cells(cells, workers=1, **kw)
    parallel = search_cells(cells, workers=4, **kw)
    for a, b in zip(serial, parallel):
        assert a.target_x_delta_m == pytest.approx(b.target_x_delta_m)
        assert np.allclose(a.action, b.action)


def test_load_seed_actions_pads_short_historical_vectors(tmp_path):
    """Historical result JSONs predate later ACTION_FIELDS entries. They
    must be padded with -1.0 (each new field's 'behaves like it did before
    this field existed' end), byte-identically to optimize.py's own
    --seed-from-json handling."""
    import json

    path = tmp_path / "old.json"
    path.write_text(json.dumps({"action": [0.1, 0.2, 0.3, 0.4, 0.5]}), encoding="utf-8")
    (action,) = _load_seed_actions([str(path)])
    assert action.shape == (ACTION_DIM,)
    assert action[-1] == pytest.approx(-1.0)


# --------------------------------------------------------------------------
# evaluate_schedule / evaluate_fixed_action
# --------------------------------------------------------------------------


def _tiny_schedule(action: np.ndarray) -> GainSchedule:
    """A constant schedule: same action at every displacement. Lets the
    scheduled path be compared against the fixed-action path exactly."""
    dxs = np.array([-2.0, -0.5, 0.5, 2.0]) * SCENARIO.max_dx_hint_m
    return GainSchedule(
        [
            ScheduleKnot(SCENARIO.name, float(dx / SCENARIO.max_dx_hint_m), float(dx), action.copy())
            for dx in dxs
        ]
    )


def test_evaluate_schedule_matches_fixed_action_for_a_constant_schedule(env):
    """The strongest available correctness check on the scheduled
    evaluation path: with a schedule that returns the SAME action
    everywhere, it must produce results identical to the existing
    fixed-gain evaluator. Any divergence means the schedule lookup, not the
    gains, changed the outcome."""
    fracs = (0.5, -0.5)
    sched = _tiny_schedule(MID_ACTION)
    scheduled = evaluate_schedule(
        sched, scenarios=(SCENARIO,), dx_fractions=fracs,
        fast_move_dx_fractions=fracs, env=env,
    )
    fixed = evaluate_fixed_action(
        MID_ACTION, scenarios=(SCENARIO,), dx_fractions=fracs,
        fast_move_dx_fractions=fracs, env=env,
    )
    assert len(scheduled) == len(fixed) == 4
    for s, f in zip(scheduled, fixed):
        assert s.scenario == f.scenario
        assert s.target_x_delta_m == pytest.approx(f.target_x_delta_m)
        assert s.move_duration_s == pytest.approx(f.move_duration_s)
        assert s.achieved_x_delta_m == pytest.approx(f.achieved_x_delta_m)
        assert s.guard_reason == f.guard_reason


def test_evaluate_fixed_action_matches_the_existing_evaluate_gains(env):
    """evaluate_fixed_action exists only so the baseline can share one env
    instance with the schedule; it must otherwise be the same measurement
    as velocity_gain_tuning.evaluate.evaluate_gains."""
    fracs = (0.5, -0.5)
    mine = evaluate_fixed_action(
        MID_ACTION, scenarios=(SCENARIO,), dx_fractions=fracs,
        fast_move_dx_fractions=fracs, env=env,
    )
    theirs = evaluate_gains(
        MID_ACTION, scenarios=(SCENARIO,), dx_fractions=fracs, fast_move_dx_fractions=fracs
    )
    for a, b in zip(mine, theirs):
        assert a.guard_reason == b.guard_reason
        assert a.achieved_x_delta_m == pytest.approx(b.achieved_x_delta_m)
        assert a.max_abs_qd_radps == pytest.approx(b.max_abs_qd_radps)


def test_evaluate_schedule_runs_both_move_durations(env):
    fracs = (0.5,)
    results = evaluate_schedule(
        _tiny_schedule(MID_ACTION), scenarios=(SCENARIO,),
        dx_fractions=fracs, fast_move_dx_fractions=fracs, env=env,
    )
    durations = sorted({round(r.move_duration_s, 6) for r in results})
    assert durations == [FAST_MOVE_DURATION_S, 1.0]


def test_evaluate_schedule_actually_uses_different_gains_per_cell(env):
    """The whole premise: two cells with different knot actions must
    produce different behaviour. A schedule that silently collapsed to one
    action would still 'pass' every other test here."""
    slow_action = np.array([-1.0, 0.0, -1.0, 0.0, 0.0, -1.0])  # tiny ik_joint_gain
    fast_action = np.array([0.0, 0.0, 1.0, 0.0, 0.0, -1.0])  # max ik_joint_gain
    dx = 0.5 * SCENARIO.max_dx_hint_m
    sched = GainSchedule(
        [
            ScheduleKnot(SCENARIO.name, -0.5, -dx, slow_action),
            ScheduleKnot(SCENARIO.name, 0.5, dx, fast_action),
        ]
    )
    assert not np.allclose(sched.action_for(SCENARIO.name, -dx), sched.action_for(SCENARIO.name, dx))
    results = evaluate_schedule(
        sched, scenarios=(SCENARIO,), dx_fractions=(0.5, -0.5),
        fast_move_dx_fractions=(), env=env,
    )
    achieved = {round(r.target_x_delta_m, 6): abs(r.achieved_x_delta_m) for r in results}
    assert achieved[round(dx, 6)] != pytest.approx(achieved[round(-dx, 6)], rel=1e-3)


def test_standard_eval_grid_is_128_cells_over_four_poses():
    """Guards the comparability of every number this package reports
    against the historical outputs/velocity_gain_tuning/search_result_*.json
    figures, which are all out of 128."""
    assert len(POSE_SCENARIOS) == 4
    assert len(STANDARD_EVAL_DX_FRACTIONS) == 16
    n_cells = len(POSE_SCENARIOS) * len(STANDARD_EVAL_DX_FRACTIONS) * 2  # slow + fast
    assert n_cells == 128


def test_standard_eval_grid_covers_the_held_out_knot_fraction():
    assert 1.0 in STANDARD_EVAL_DX_FRACTIONS and -1.0 in STANDARD_EVAL_DX_FRACTIONS


def test_knot_coverage_split_separates_searched_from_held_out(env):
    """The split that keeps an interpolant's score honest: cells at a
    searched displacement are memorisation, cells elsewhere are the real
    generalisation test. Getting this wrong turns a documented regression
    (9/16 held-out vs a constant's 16/16) into a reported '+13 cells' win."""
    sched = _tiny_schedule(MID_ACTION)
    knot_fracs = knot_fractions_from_schedule(sched)
    assert knot_fracs[SCENARIO.name] == {-2.0, -0.5, 0.5, 2.0}
    results = evaluate_schedule(
        sched, scenarios=(SCENARIO,), dx_fractions=(0.5, 0.9),
        fast_move_dx_fractions=(), env=env,
    )
    split = knot_coverage_split(results, knot_fracs)
    assert split["at_knot"]["n_total"] == 1  # 0.5 is a knot
    assert split["held_out"]["n_total"] == 1  # 0.9 is not
    assert split["at_knot"]["n_total"] + split["held_out"]["n_total"] == len(results)


def test_knot_coverage_split_counts_passes_per_bucket(env):
    sched = _tiny_schedule(MID_ACTION)
    results = evaluate_schedule(
        sched, scenarios=(SCENARIO,), dx_fractions=(0.5, 0.9),
        fast_move_dx_fractions=(), env=env,
    )
    split = knot_coverage_split(results, knot_fractions_from_schedule(sched))
    total_pass = split["at_knot"]["n_pass"] + split["held_out"]["n_pass"]
    assert total_pass == sum(1 for r in results if r.guard_reason is None)
    for bucket in split.values():
        assert bucket["n_pass"] + bucket["n_fail"] == bucket["n_total"]


def test_compare_summaries_attaches_knot_coverage_only_when_asked(env):
    sched = _tiny_schedule(MID_ACTION)
    results = evaluate_schedule(
        sched, scenarios=(SCENARIO,), dx_fractions=(0.5,),
        fast_move_dx_fractions=(), env=env,
    )
    assert "knot_coverage" not in compare_summaries({"s": results})["s"]
    with_split = compare_summaries({"s": results}, knot_fractions_from_schedule(sched))
    assert "knot_coverage" in with_split["s"]


def test_print_comparison_labels_the_generalisation_columns(env, capsys):
    sched = _tiny_schedule(MID_ACTION)
    results = evaluate_schedule(
        sched, scenarios=(SCENARIO,), dx_fractions=(0.5, 0.9),
        fast_move_dx_fractions=(), env=env,
    )
    print_comparison(compare_summaries({"s": results}, knot_fractions_from_schedule(sched)))
    out = capsys.readouterr().out
    assert "at_knot" in out and "held_out" in out


def test_compare_summaries_and_print_do_not_raise(env, capsys):
    fracs = (0.5,)
    results = evaluate_schedule(
        _tiny_schedule(MID_ACTION), scenarios=(SCENARIO,),
        dx_fractions=fracs, fast_move_dx_fractions=fracs, env=env,
    )
    summaries = compare_summaries({"sched": results})
    assert summaries["sched"] == summarize_safety(results)
    print_comparison(summaries)
    assert "sched" in capsys.readouterr().out


def test_build_cells_default_grid_is_searchable_shape():
    cells = build_cells()
    assert len(cells) == 56
    assert {c.scenario.name for c in cells} == {s.name for s in POSE_SCENARIOS}
    for c in cells:
        assert scenario_by_name(c.scenario.name) is c.scenario


def test_env_config_is_threaded_through_cell_fitness():
    """A non-default env config must actually reach the rollout -- otherwise
    a sweep run with tightened guards would silently score against the
    defaults."""
    dx = 0.5 * SCENARIO.max_dx_hint_m
    strict = VelocityTransportEnvConfig(max_joint_velocity_radps=1e-6)
    # Fresh worker env caching is keyed on the config object, so a stricter
    # guard must change the objective.
    default_score = cell_fitness(MID_ACTION, SCENARIO.name, dx, None, 0)
    strict_score = cell_fitness(MID_ACTION, SCENARIO.name, dx, strict, 0)
    assert strict_score > default_score
