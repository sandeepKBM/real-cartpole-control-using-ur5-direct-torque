"""Tests for velocity_gain_tuning/ -- the independent differential_evolution
gain search + multi-pose evaluation gym for
controller_core.cartesian_velocity_controller's ik_seeded_resolution mode.

Added 2026-08-06 per AGENTS.md sec 5's "new modules/packages ship with
pytest coverage" rule -- this package previously had only manual/inline
smoke checks. Particular emphasis on the fast-move (FAST_MOVE_DURATION_S)
stress-testing dimension added the same day: two real gain searches run
before that fix used ONLY the default (slow, 1.0s) move_duration and
missed a real, dangerous gap (a found gain vector produced |qd|~4.7 rad/s,
over the 3.0 rad/s guard, at a fast move) -- these tests exist specifically
so that gap-closing mechanism can't silently regress.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from velocity_gain_tuning.envs.velocity_transport_env import (  # noqa: E402
    ACTION_DIM,
    ACTION_FIELDS,
    OBS_DIM,
    VelocityTransportEnv,
    VelocityTransportEnvConfig,
    action_to_gains,
)
from velocity_gain_tuning.evaluate import evaluate_gains, print_report, summarize_safety  # noqa: E402
from velocity_gain_tuning.optimize import (  # noqa: E402
    DEFAULT_CELL_WEIGHT_OVERRIDES,
    FAST_MOVE_DURATION_S,
    EpisodeResult,
    _build_seeded_population,
    _cell_weight,
    fitness,
    run_episode,
    run_search,
)
from velocity_gain_tuning.poses import POSE_SCENARIOS  # noqa: E402

# The exact gain vector found by this session's second (widened-bounds) real
# gain search -- outputs/velocity_gain_tuning/search_result_widened_20260806_023900.json.
# Historically significant: this is the vector whose fast-move unsafety
# (|qd|=4.74 rad/s at move_duration_s=0.01, guard=3.0) motivated adding
# FAST_MOVE_DURATION_S stress-testing at all. Re-verified directly against
# THIS test's own scenario/dx below (deterministic, checked 3x across fresh
# processes): at unrotated_wrist2offset, dx=0.0245 (0.5x its max_dx_hint_m),
# move_duration_s=FAST_MOVE_DURATION_S, it trips joint_velocity_guard at
# |qd|=3.0805 rad/s.
# Padded with a trailing -1.0 (originally two, for the now-removed
# ik_posture_gain/ik_posture_activation_joint_dev_rad; now one, for
# ik_max_joint_deviation_rad -- see velocity_transport_env.py's
# ACTION_FIELDS for the full replacement history). -1.0 maps to
# ik_max_joint_deviation_rad=2.0 rad (its loose/near-off end -- the field's
# bounds are DELIBERATELY ordered so -1.0 means "unconstrained," see
# ACTION_FIELDS' own comment), reproducing this vector's ORIGINAL 5-dim
# meaning byte-for-byte -- this dimension didn't exist when the historical
# search that produced this vector ran.
_SEARCH2_ACTION = np.array(
    [-0.6386717612031976, 0.5023040867170563, 0.5575301222016413, -0.959391338891284, 0.9906856414248031, -1.0]
)
_UNROTATED_SCENARIO = POSE_SCENARIOS[1]
assert _UNROTATED_SCENARIO.name == "unrotated_wrist2offset"


# ---------------------------------------------------------------------------
# action_to_gains
# ---------------------------------------------------------------------------


def test_action_to_gains_bounds_at_extremes():
    lo_gains = action_to_gains(np.full(ACTION_DIM, -1.0))
    hi_gains = action_to_gains(np.full(ACTION_DIM, 1.0))
    for name, lo, hi, _is_log in ACTION_FIELDS:
        assert lo_gains[name] == pytest.approx(lo, rel=1e-9)
        assert hi_gains[name] == pytest.approx(hi, rel=1e-9)


def test_action_to_gains_midpoint_linear_vs_log():
    mid_gains = action_to_gains(np.zeros(ACTION_DIM))
    for name, lo, hi, is_log in ACTION_FIELDS:
        if is_log:
            # Geometric mean for a log-remapped field.
            assert mid_gains[name] == pytest.approx(np.sqrt(lo * hi), rel=1e-6)
        else:
            assert mid_gains[name] == pytest.approx((lo + hi) / 2.0, rel=1e-9)


def test_action_to_gains_clips_out_of_range():
    over = action_to_gains(np.full(ACTION_DIM, 5.0))
    under = action_to_gains(np.full(ACTION_DIM, -5.0))
    hi_gains = action_to_gains(np.full(ACTION_DIM, 1.0))
    lo_gains = action_to_gains(np.full(ACTION_DIM, -1.0))
    for name, _lo, _hi, _is_log in ACTION_FIELDS:
        assert over[name] == pytest.approx(hi_gains[name], rel=1e-9)
        assert under[name] == pytest.approx(lo_gains[name], rel=1e-9)


# ---------------------------------------------------------------------------
# VelocityTransportEnv
# ---------------------------------------------------------------------------


def test_env_reset_and_step_shapes():
    env = VelocityTransportEnv(VelocityTransportEnvConfig())
    obs, info = env.reset(seed=0, options={"scenario": _UNROTATED_SCENARIO, "target_x_delta_m": 0.01})
    assert obs.shape == (OBS_DIM,)
    assert obs.dtype == np.float32
    assert np.all(np.isfinite(obs))
    assert info["scenario"] == "unrotated_wrist2offset"
    assert info["target_x_delta_m"] == pytest.approx(0.01)

    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    assert obs.shape == (OBS_DIM,)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    for key in (
        "x_error", "orientation_error", "max_abs_qd_radps", "orthogonal_drift_m", "guard_reason",
        "achieved_x_delta_m", "scenario",
    ):
        assert key in info


def test_env_reset_respects_explicit_options():
    env = VelocityTransportEnv(VelocityTransportEnvConfig())
    env.reset(
        seed=0,
        options={"scenario": _UNROTATED_SCENARIO, "target_x_delta_m": 0.017, "move_duration_s": 0.5},
    )
    assert env._scenario_name == "unrotated_wrist2offset"
    assert env._target_x_delta_m == pytest.approx(0.017)
    assert env._move_duration_s == pytest.approx(0.5)
    np.testing.assert_allclose(env._q, _UNROTATED_SCENARIO.q0)


def test_env_step_extreme_gains_can_trip_a_guard():
    # Not asserting WHICH guard (that's covered precisely by the real-vector
    # regression test below) -- just that the guard machinery is reachable
    # via step() at all, with a deliberately hostile action/scenario/dx.
    env = VelocityTransportEnv(VelocityTransportEnvConfig())
    env.reset(
        seed=0,
        options={"scenario": _UNROTATED_SCENARIO, "target_x_delta_m": 0.0049, "move_duration_s": FAST_MOVE_DURATION_S},
    )
    terminated = truncated = False
    info = {}
    for _ in range(400):
        obs, reward, terminated, truncated, info = env.step(_SEARCH2_ACTION)
        if terminated or truncated:
            break
    assert terminated, "expected the hostile action/pose/dx combination to trip a safety guard"
    assert info["guard_reason"] is not None
    assert reward < 0.0  # guard_trip_penalty dominates


# ---------------------------------------------------------------------------
# run_episode / FAST_MOVE_DURATION_S mechanism
# ---------------------------------------------------------------------------


def test_run_episode_move_duration_field_defaults_to_env_cfg():
    env = VelocityTransportEnv(VelocityTransportEnvConfig())
    action = np.zeros(ACTION_DIM)
    result = run_episode(env, action, scenario=_UNROTATED_SCENARIO, target_x_delta_m=0.01)
    assert result.move_duration_s == pytest.approx(env.cfg.move_duration_s)


def test_run_episode_move_duration_field_reflects_override():
    env = VelocityTransportEnv(VelocityTransportEnvConfig())
    action = np.zeros(ACTION_DIM)
    result = run_episode(
        env, action, scenario=_UNROTATED_SCENARIO, target_x_delta_m=0.01, move_duration_s=FAST_MOVE_DURATION_S
    )
    assert result.move_duration_s == pytest.approx(FAST_MOVE_DURATION_S)
    assert result.move_duration_s != pytest.approx(env.cfg.move_duration_s)


def test_fast_move_produces_higher_peak_joint_velocity_than_slow_move():
    """The core causal mechanism this whole fix relies on: for IDENTICAL
    gains and target displacement, a near-step (FAST_MOVE_DURATION_S) move
    produces a measurably higher peak joint velocity than the default slow
    move -- confirming move_duration_s is a genuine additional stress axis
    that a slow-only evaluation grid cannot see, independent of any
    particular guard threshold (this case is picked well below the 3.0
    rad/s guard on both sides, so the assertion isn't coupled to a
    boundary-crossing that could flip with unrelated numerical noise)."""
    env = VelocityTransportEnv(VelocityTransportEnvConfig())
    # kp_x/kp_rot (action indices 0,1) are no-ops for ik_seeded_resolution --
    # compute_ik_seeded never reads them, see controller_core/
    # cartesian_velocity_controller/controller.py's compute() dispatch --
    # so only ik_joint_gain (index 2, set to its max here) drives this.
    action = np.array([0.0, 0.0, 1.0, 0.0, 0.0, -1.0])  # -1.0 -> ik_max_joint_deviation_rad=2.0 (loose/off)
    dx = 0.10
    r_slow = run_episode(env, action, scenario=_UNROTATED_SCENARIO, target_x_delta_m=dx)
    r_fast = run_episode(
        env, action, scenario=_UNROTATED_SCENARIO, target_x_delta_m=dx, move_duration_s=FAST_MOVE_DURATION_S
    )
    assert r_fast.max_abs_qd_radps > 2.0 * r_slow.max_abs_qd_radps
    assert r_slow.max_abs_qd_radps < 3.0  # guard is 3.0 -- confirm slow genuinely doesn't approach it here
    assert r_fast.max_abs_qd_radps < 3.0  # and neither does fast, in this deliberately-clear-of-guard case


def test_fast_move_regression_search2_gains_trip_joint_velocity_guard():
    """Exact reproduction of the real finding that motivated adding
    FAST_MOVE_DURATION_S: this session's second real gain search
    (search_result_widened_20260806_023900.json) found gains that, evaluated
    at a fast move, trip the joint-velocity guard -- a failure mode that
    literally cannot appear in a slow-move-only evaluation grid, since such
    a grid never runs an episode at this move_duration_s at all. Locks in
    the exact deterministic numbers (verified reproducible across 3
    independent fresh processes) so this specific historical gap can't
    silently stop being caught."""
    env = VelocityTransportEnv(VelocityTransportEnvConfig())
    dx = 0.5 * _UNROTATED_SCENARIO.max_dx_hint_m  # 0.0245
    result = run_episode(
        env, _SEARCH2_ACTION, scenario=_UNROTATED_SCENARIO, target_x_delta_m=dx, move_duration_s=FAST_MOVE_DURATION_S
    )
    assert result.guard_reason is not None
    assert "joint_velocity_guard" in result.guard_reason
    assert result.max_abs_qd_radps > 3.0


# ---------------------------------------------------------------------------
# fitness() -- fast-move dimension actually affects the search objective
# ---------------------------------------------------------------------------


def test_fitness_fast_move_dimension_is_actually_exercised():
    """fast_move_dx_fractions must actually change what fitness() computes
    -- if a future edit silently dropped the fast-move loop (making the
    parameter a dead no-op), this test would see identical fitness values
    with vs. without it and fail. Doesn't assert a direction (which way the
    score moves depends on how many steps a given failing episode runs
    before its guard trips, not just pass/fail), only that the dimension is
    genuinely wired into the objective the optimizer sees."""
    fitness_without_fast = fitness(
        _SEARCH2_ACTION,
        scenarios=(_UNROTATED_SCENARIO,),
        env_config=None,
        seed=0,
        dx_fractions=(1.0,),
        fast_move_dx_fractions=(),
    )
    fitness_with_fast = fitness(
        _SEARCH2_ACTION,
        scenarios=(_UNROTATED_SCENARIO,),
        env_config=None,
        seed=0,
        dx_fractions=(1.0,),
        fast_move_dx_fractions=(0.5,),
    )
    assert fitness_with_fast != pytest.approx(fitness_without_fast)


def test_fitness_direction_matches_reward_mean_for_fixed_episodes():
    """A direction-asserting complement to the test above, without
    depending on any particular gain/dx producing a specific pass/fail
    pattern (found empirically fragile -- this reward's guard-trip penalty
    interacts with how many steps ran before the trip, so "more failures"
    does not always mean "more negative reward"): directly confirms
    fitness()'s formula is exactly -mean(total_reward across all episodes
    it ran), i.e. that adding the fast-move episodes to the average is a
    real, correctly-weighted contribution, not silently dropped or
    double-counted. Uses _UNROTATED_SCENARIO specifically, which has no
    entry in DEFAULT_CELL_WEIGHT_OVERRIDES (only hanging_alpha_0_5's
    negative-x cells do), so the WEIGHTED mean equals the plain mean here
    -- the reweighting mechanism itself is covered separately below."""
    result_slow_only = run_episode(
        VelocityTransportEnv(VelocityTransportEnvConfig()),
        _SEARCH2_ACTION,
        scenario=_UNROTATED_SCENARIO,
        target_x_delta_m=1.0 * _UNROTATED_SCENARIO.max_dx_hint_m,
    )
    result_fast = run_episode(
        VelocityTransportEnv(VelocityTransportEnvConfig()),
        _SEARCH2_ACTION,
        scenario=_UNROTATED_SCENARIO,
        target_x_delta_m=0.5 * _UNROTATED_SCENARIO.max_dx_hint_m,
        move_duration_s=FAST_MOVE_DURATION_S,
    )
    expected_fitness = -float(np.mean([result_slow_only.total_reward, result_fast.total_reward]))
    actual_fitness = fitness(
        _SEARCH2_ACTION,
        scenarios=(_UNROTATED_SCENARIO,),
        env_config=None,
        seed=0,
        dx_fractions=(1.0,),
        fast_move_dx_fractions=(0.5,),
    )
    assert actual_fitness == pytest.approx(expected_fitness, rel=1e-6)


# ---------------------------------------------------------------------------
# cell_weight / DEFAULT_CELL_WEIGHT_OVERRIDES (2026-08-06) -- hanging_alpha_
# 0_5's -X cells count more so the aggregate mean can't average them away
# against unrotated_wrist2offset's easy passes.
# ---------------------------------------------------------------------------


def _episode(scenario: str, target_x_delta_m: float) -> EpisodeResult:
    return EpisodeResult(scenario, target_x_delta_m, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, None)


def test_cell_weight_hanging_negative_x_gets_default_override():
    result = _episode("hanging_alpha_0_5", -0.2)
    assert _cell_weight(result, DEFAULT_CELL_WEIGHT_OVERRIDES) == pytest.approx(5.0)


def test_cell_weight_hanging_positive_x_is_unweighted():
    result = _episode("hanging_alpha_0_5", 0.2)
    assert _cell_weight(result, DEFAULT_CELL_WEIGHT_OVERRIDES) == pytest.approx(1.0)


def test_cell_weight_other_scenarios_negative_x_are_unweighted():
    # Only hanging_alpha_0_5 has an override -- confirms this isn't a
    # blanket "all negative-x cells" rule for every scenario.
    for scenario_name in ("neg40_wrist2offset", "unrotated_wrist2offset", "neg45_wrist2offset"):
        result = _episode(scenario_name, -0.02)
        assert _cell_weight(result, DEFAULT_CELL_WEIGHT_OVERRIDES) == pytest.approx(1.0), scenario_name


def test_cell_weight_zero_dx_counts_as_positive_x():
    # target_x_delta_m == 0.0 is an edge case (no move at all) -- must not
    # be misclassified as "negative" and accidentally inherit hanging's
    # override.
    result = _episode("hanging_alpha_0_5", 0.0)
    assert _cell_weight(result, DEFAULT_CELL_WEIGHT_OVERRIDES) == pytest.approx(1.0)


def test_cell_weight_custom_overrides_replace_not_merge_with_default():
    # Passing an explicit overrides dict to _cell_weight (as fitness()
    # does when cell_weight_overrides is not None) uses exactly that dict
    # -- no implicit merging with DEFAULT_CELL_WEIGHT_OVERRIDES.
    result = _episode("hanging_alpha_0_5", -0.2)
    assert _cell_weight(result, {}) == pytest.approx(1.0)
    assert _cell_weight(result, {("hanging_alpha_0_5", "negative_x"): 9.0}) == pytest.approx(9.0)


_HANGING_SCENARIO = POSE_SCENARIOS[3]


def test_fitness_weighting_matches_manual_weighted_mean():
    """End-to-end confirmation (real episodes, real env) that fitness()'s
    formula is exactly -weighted_mean(reward, weight=_cell_weight(...)),
    using a scenario mix that actually exercises the override (unlike
    test_fitness_direction_matches_reward_mean_for_fixed_episodes, which
    deliberately used an unweighted scenario to isolate the fast-move
    dimension)."""
    assert _HANGING_SCENARIO.name == "hanging_alpha_0_5"
    action = np.array([0.0, 0.0, 1.0, 0.0, 0.0, -1.0])
    result_pos = run_episode(
        VelocityTransportEnv(VelocityTransportEnvConfig()),
        action, scenario=_HANGING_SCENARIO, target_x_delta_m=0.3 * _HANGING_SCENARIO.max_dx_hint_m,
    )
    result_neg = run_episode(
        VelocityTransportEnv(VelocityTransportEnvConfig()),
        action, scenario=_HANGING_SCENARIO, target_x_delta_m=-0.3 * _HANGING_SCENARIO.max_dx_hint_m,
    )
    weights = np.array([1.0, 5.0])  # positive_x=1.0 (unweighted), negative_x=5.0 (default override)
    rewards = np.array([result_pos.total_reward, result_neg.total_reward])
    expected_fitness = -float(np.average(rewards, weights=weights))
    actual_fitness = fitness(
        action, scenarios=(_HANGING_SCENARIO,), env_config=None, seed=0,
        dx_fractions=(0.3, -0.3), fast_move_dx_fractions=(),
    )
    assert actual_fitness == pytest.approx(expected_fitness, rel=1e-6)
    # And the unweighted (flat mean) value must be DIFFERENT -- proof the
    # override actually changes the result, not a silent no-op.
    flat_mean_fitness = -float(np.mean(rewards))
    assert actual_fitness != pytest.approx(flat_mean_fitness)


def test_fitness_explicit_overrides_override_the_default():
    action = np.array([0.0, 0.0, 1.0, 0.0, 0.0, -1.0])
    dx_fractions = (0.3, -0.3)
    default_weighted = fitness(
        action, scenarios=(_HANGING_SCENARIO,), env_config=None, seed=0,
        dx_fractions=dx_fractions, fast_move_dx_fractions=(),
    )
    explicitly_unweighted = fitness(
        action, scenarios=(_HANGING_SCENARIO,), env_config=None, seed=0,
        dx_fractions=dx_fractions, fast_move_dx_fractions=(), cell_weight_overrides={},
    )
    assert default_weighted != pytest.approx(explicitly_unweighted)


# ---------------------------------------------------------------------------
# evaluate_gains -- both grids actually run
# ---------------------------------------------------------------------------


def test_evaluate_gains_runs_both_slow_and_fast_grids():
    env_config = VelocityTransportEnvConfig()
    results = evaluate_gains(
        np.zeros(ACTION_DIM),
        scenarios=(_UNROTATED_SCENARIO,),
        dx_fractions=(1.0,),
        fast_move_dx_fractions=(1.0,),
        env_config=env_config,
    )
    assert len(results) == 2
    durations = sorted(r.move_duration_s for r in results)
    assert durations[0] == pytest.approx(FAST_MOVE_DURATION_S)
    assert durations[1] == pytest.approx(env_config.move_duration_s)


def test_print_report_does_not_raise(capsys):
    results = evaluate_gains(
        np.zeros(ACTION_DIM), scenarios=(_UNROTATED_SCENARIO,), dx_fractions=(1.0,), fast_move_dx_fractions=(1.0,)
    )
    print_report(results)
    out = capsys.readouterr().out
    assert "unrotated_wrist2offset" in out


# ---------------------------------------------------------------------------
# summarize_safety -- pure bucketing logic, no simulation needed
# ---------------------------------------------------------------------------


def _synthetic_results() -> list[EpisodeResult]:
    return [
        EpisodeResult("posA", 0.02, 1.0, 5.0, 0.001, 0.0198, 0.05, 0.3, None),  # slow pass
        EpisodeResult(
            "posA", 0.03, 1.0, -10.0, 0.01, 0.025, 0.3, 0.5, "orientation_guard: 0.30 > 0.25"
        ),  # slow fail
        EpisodeResult("posA", 0.01, FAST_MOVE_DURATION_S, 3.0, 0.001, 0.0099, 0.05, 1.0, None),  # fast pass
        EpisodeResult(
            "posA", 0.02, FAST_MOVE_DURATION_S, -20.0, 0.02, 0.01, 0.05, 3.5, "joint_velocity_guard: 3.5 > 3.0"
        ),  # fast fail
        EpisodeResult(
            "posB", 0.01, FAST_MOVE_DURATION_S, -20.0, 0.02, 0.005, 0.05, 4.0, "joint_velocity_guard: 4.0 > 3.0"
        ),  # fast fail
    ]


def test_summarize_safety_overall_counts():
    summary = summarize_safety(_synthetic_results())
    assert summary["n_total"] == 5
    assert summary["n_pass"] == 2
    assert summary["n_fail"] == 3
    assert summary["pass_fraction"] == pytest.approx(2 / 5)
    assert summary["worst_orientation_error_rad"] == pytest.approx(0.3)
    assert summary["worst_abs_qd_radps"] == pytest.approx(4.0)
    assert len(summary["guard_trips"]) == 3


def test_summarize_safety_slow_fast_buckets_are_disjoint_and_correct():
    summary = summarize_safety(_synthetic_results())
    slow = summary["slow_move"]
    fast = summary["fast_move"]
    assert slow["n_total"] + fast["n_total"] == 5
    assert slow["n_total"] == 2
    assert slow["n_pass"] == 1
    assert slow["pass_fraction"] == pytest.approx(0.5)
    assert slow["worst_abs_qd_radps"] == pytest.approx(0.5)
    assert fast["n_total"] == 3
    assert fast["n_pass"] == 1
    assert fast["n_fail"] == 2
    assert fast["pass_fraction"] == pytest.approx(1 / 3)
    assert fast["worst_abs_qd_radps"] == pytest.approx(4.0)


def test_summarize_safety_per_scenario_and_guard_trip_fields():
    summary = summarize_safety(_synthetic_results())
    posA = summary["per_scenario"]["posA"]
    assert posA["pass"] == 2
    assert posA["fail"] == 2
    assert posA["max_passing_dx_m"] == pytest.approx(0.02)  # max of the two passing dx (0.02, 0.01)
    posB = summary["per_scenario"]["posB"]
    assert posB["pass"] == 0
    assert posB["fail"] == 1
    assert posB["max_passing_dx_m"] == pytest.approx(0.0)
    for trip in summary["guard_trips"]:
        assert "move_duration_s" in trip
        assert "max_abs_qd_radps" in trip
        assert "guard_reason" in trip


def test_summarize_safety_empty_input():
    summary = summarize_safety([])
    assert summary["n_total"] == 0
    assert summary["pass_fraction"] == 0.0
    assert summary["slow_move"]["n_total"] == 0
    assert summary["fast_move"]["n_total"] == 0


# ---------------------------------------------------------------------------
# run_search -- seed_actions population seeding (2026-08-06)
# ---------------------------------------------------------------------------


def test_build_seeded_population_includes_seed_rows_exactly():
    """Pure-logic unit test for the population-construction helper (kept
    fast/deterministic on purpose -- earlier tried asserting this via a
    full run_search() call and comparing the winning action to the seed,
    but that's fragile: DE's mutation/crossover can and did beat an
    arbitrary hand-picked seed within a single generation, which says
    nothing about whether the seed was actually injected. This test
    isolates just the injection logic instead.)"""
    seed_actions = [
        np.array([0.0, 0.0, 1.0, 0.0, 0.0, -1.0]), np.array([-1.0, 1.0, -1.0, 1.0, -1.0, 0.5]),
    ]
    popsize = 3
    population = _build_seeded_population(popsize, seed_actions, seed=0)
    assert population.shape == (popsize * ACTION_DIM, ACTION_DIM)
    np.testing.assert_allclose(population[0], seed_actions[0])
    np.testing.assert_allclose(population[1], seed_actions[1])
    assert np.all(population >= -1.0) and np.all(population <= 1.0)


def test_build_seeded_population_clips_out_of_range_seeds():
    population = _build_seeded_population(2, [np.array([5.0, -5.0, 0.0, 0.0, 0.0, 0.0])], seed=0)
    np.testing.assert_allclose(population[0], [1.0, -1.0, 0.0, 0.0, 0.0, 0.0])


def test_build_seeded_population_truncates_more_seeds_than_population():
    # popsize=1 -> only ACTION_DIM=6 members total; 6 seed_actions supplied
    # must not raise or overflow -- extras are silently ignored (documented
    # in run_search's docstring via seed_actions[:n_members]).
    seed_actions = [np.zeros(ACTION_DIM) + i for i in range(6)]
    population = _build_seeded_population(1, seed_actions, seed=0)
    assert population.shape == (ACTION_DIM, ACTION_DIM)


def test_run_search_seed_actions_end_to_end_does_not_raise():
    # Full-pipeline smoke test (small/fast on purpose): confirms run_search
    # actually wires seed_actions through to differential_evolution's init=
    # without error, complementing the pure-logic tests above which cover
    # the injection details.
    outcome = run_search(
        scenarios=(_UNROTATED_SCENARIO,),
        maxiter=1,
        popsize=2,
        seed=0,
        workers=1,
        auto_evaluate=False,
        seed_actions=[np.array([0.0, 0.0, 1.0, 0.0, 0.0, -1.0])],
    )
    assert outcome["action"].shape == (ACTION_DIM,)


def test_run_search_without_seed_actions_still_works():
    # Regression guard: seed_actions=None (the default) must not change
    # run_search's un-seeded behavior or raise.
    outcome = run_search(
        scenarios=(_UNROTATED_SCENARIO,),
        maxiter=1,
        popsize=2,
        seed=0,
        workers=1,
        auto_evaluate=False,
    )
    assert outcome["action"].shape == (ACTION_DIM,)


# ---------------------------------------------------------------------------
# qd_estimate_damping (2026-08-07) -- fixes a bare-pinv(FULL jacobian)
# numerical blowup in step()'s joint-velocity-guard ESTIMATE at the
# wrist_2=0 kinematic singularity, confirmed independent of the
# controller's own (separately, already-damped) reduced task-space
# Jacobian. See velocity_transport_env.py's qd_estimate_damping docstring
# and docs/status/nullspace_v2_search_results_2026-08-06.md for the two
# real spikes (18.14/161.57 rad/s) this closes. Pure-math property tests
# for the underlying _damped_pinv behavior live in tests/unit/
# test_velocity_gain_tuning_qd_estimate_damping.py (no mujoco needed
# there); the tests below cover the actual env field/default and a real,
# exact reproduction of one of the two documented spike cases.
# ---------------------------------------------------------------------------


def test_qd_estimate_damping_field_default_matches_unit_test_literal():
    """VelocityTransportEnvConfig.qd_estimate_damping's real default must
    match the literal QD_ESTIMATE_DAMPING hardcoded in the mujoco-import-free
    unit test (tests/unit/test_velocity_gain_tuning_qd_estimate_damping.py)
    -- that file deliberately avoids importing this module (to stay
    mujoco-free), so this is the one place the two are cross-checked
    against silent drift."""
    assert VelocityTransportEnvConfig().qd_estimate_damping == pytest.approx(1.0e-3)


def test_qd_estimate_damping_positive_default_is_actually_used_in_step():
    """qd_estimate_damping is genuinely wired into step()'s qd
    reconstruction (not a dead config knob): a much heavier damping value
    must measurably change the reported max_abs_qd_radps relative to the
    class default, at a dx large enough to be meaningfully non-trivial.
    (Does NOT use qd_estimate_damping=0.0 as a stand-in for "the old bare
    np.linalg.pinv behavior" -- found directly while writing this test that
    the two are NOT equivalent near a real singularity: _damped_pinv(jac,
    0.0) inverts J@J.T directly, which squares cond(J) and can produce
    numerically UNSTABLE, not just unregularized, results very different
    from SVD-based np.linalg.pinv. See the spike-reproduction test below
    for the real bare-pinv comparison, done via direct reimplementation
    instead.)"""
    action = np.zeros(ACTION_DIM)
    env_light = VelocityTransportEnv(VelocityTransportEnvConfig(qd_estimate_damping=1.0e-3))
    env_heavy = VelocityTransportEnv(VelocityTransportEnvConfig(qd_estimate_damping=0.5))
    for env in (env_light, env_heavy):
        env.reset(seed=0, options={"scenario": _UNROTATED_SCENARIO, "target_x_delta_m": 0.03})
    # t=0 of a min-jerk profile starts at zero velocity/error (qd == 0 for
    # both), so take a few cycles to reach a nontrivial command before
    # comparing.
    info_light = info_heavy = None
    for _ in range(5):
        _, _, _, _, info_light = env_light.step(action)
        _, _, _, _, info_heavy = env_heavy.step(action)
    assert info_light["max_abs_qd_radps"] > 0.0
    assert info_light["max_abs_qd_radps"] != pytest.approx(info_heavy["max_abs_qd_radps"], rel=1e-6)


def test_qd_estimate_damping_fixes_documented_unrotated_wrist2offset_spike():
    """Reproduction of the real, documented 18.14 rad/s false-positive
    guard trip (docs/status/nullspace_v2_search_results_2026-08-06.md,
    outputs/velocity_gain_tuning/search_result_nullspace_v2_seeded108_20260806_201419.json's
    exact saved gains/action) at unrotated_wrist2offset,
    dx=-1.6*max_dx_hint_m, FAST_MOVE_DURATION_S -- a wrist_2=0 crossing
    where the bare-pinv reconstruction (NOT the controller's own
    internals) blows up.

    dx is computed as ``-1.6 * scenario.max_dx_hint_m`` (matching
    evaluate_gains()'s own fast_move_dx_fractions computation) rather than
    the rounded literal -0.0784 -- found directly while writing this test
    that this specific case is CHAOTICALLY sensitive to that last bit of
    floating-point precision: ``-1.6 * 0.049 == -0.07840000000000001``,
    differing from the literal ``-0.0784`` by ~1.4e-17, yet that is enough
    to flip which side of the wrist_2=0 crossing the trajectory falls on
    (18.14 rad/s vs. a clean 1.58 rad/s pass, confirmed reproducibly both
    ways). This is real evidence for why qd_estimate_damping's docstring
    frames the guard's PRE-fix numbers as "numerical artifacts, not a
    faithful estimate of physical risk" -- a result this sensitive to the
    17th decimal digit of a target displacement was never a meaningful
    safety signal in the first place.

    The "before" (bare pinv) ground truth is obtained by monkeypatching the
    ``_damped_pinv`` symbol this module imported, to fall back to literal
    ``np.linalg.pinv`` regardless of the damping argument -- exercises the
    REAL step() code path exactly (all guards, termination/truncation
    logic unchanged) rather than a hand-reimplementation, which was tried
    first and found unreliable for exactly the chaotic-sensitivity reason
    above (a reimplementation float-for-float identical to step() should
    work, but a shorter/refactored one is a real risk of silently taking a
    different bifurcation branch)."""
    import velocity_gain_tuning.envs.velocity_transport_env as vte_mod

    # search_result_nullspace_v2_seeded108_20260806_201419.json's own saved
    # "action" array, used verbatim (not re-derived by hand) to avoid any
    # transcription/rounding mismatch.
    gains_action = np.array(
        [0.4960278430173135, 0.12578844407520084, -0.6130064343833146, -0.20556635071129348, 0.6595114242714273, -0.8063915773455297]
    )
    gains = action_to_gains(gains_action)
    assert gains["kp_x"] == pytest.approx(22.465616253108838, rel=1e-4)
    assert gains["ik_max_joint_deviation_rad"] == pytest.approx(1.197514006313666, rel=1e-4)

    unrotated_scenario = _UNROTATED_SCENARIO
    dx = -1.6 * unrotated_scenario.max_dx_hint_m

    # Ground truth: the old, pre-fix bare-pinv reconstruction really does
    # blow up here (matches the documented 18.14 rad/s finding exactly).
    original_damped_pinv = vte_mod._damped_pinv
    vte_mod._damped_pinv = lambda jac, damping: np.linalg.pinv(jac)
    try:
        result_bare = run_episode(
            VelocityTransportEnv(VelocityTransportEnvConfig()),
            gains_action, scenario=unrotated_scenario, target_x_delta_m=dx, move_duration_s=FAST_MOVE_DURATION_S,
        )
    finally:
        vte_mod._damped_pinv = original_damped_pinv
    assert result_bare.guard_reason is not None
    assert "joint_velocity_guard" in result_bare.guard_reason
    assert result_bare.max_abs_qd_radps == pytest.approx(18.14201498863372, rel=1e-6)

    # The FIXED env (qd_estimate_damping default): no longer trips on this
    # numerical artifact, and achieved tracking is still reasonable -- not
    # just "guard didn't trip".
    result_damped = run_episode(
        VelocityTransportEnv(VelocityTransportEnvConfig()),
        gains_action, scenario=unrotated_scenario, target_x_delta_m=dx, move_duration_s=FAST_MOVE_DURATION_S,
    )
    assert result_damped.guard_reason is None
    assert result_damped.max_abs_qd_radps < 3.0
    assert abs(result_damped.achieved_x_delta_m - dx) < 0.01  # reasonable tracking of the target


# ---------------------------------------------------------------------------
# AGENTS.md sec 7 "always sweep both +X and -X" -- locked in as defaults
# ---------------------------------------------------------------------------


def test_fitness_default_dx_fractions_include_both_signs():
    import inspect

    defaults = inspect.signature(fitness).parameters
    for name in ("dx_fractions", "fast_move_dx_fractions"):
        fracs = defaults[name].default
        assert any(f > 0 for f in fracs), f"{name} default has no positive fraction"
        assert any(f < 0 for f in fracs), f"{name} default has no negative fraction -- X-direction asymmetry is real (AGENTS.md sec 7)"


def test_evaluate_gains_default_dx_fractions_include_both_signs():
    import inspect

    defaults = inspect.signature(evaluate_gains).parameters
    for name in ("dx_fractions", "fast_move_dx_fractions"):
        fracs = defaults[name].default
        assert any(f > 0 for f in fracs), f"{name} default has no positive fraction"
        assert any(f < 0 for f in fracs), f"{name} default has no negative fraction -- X-direction asymmetry is real (AGENTS.md sec 7)"


def test_run_search_default_eval_dx_fractions_include_both_signs():
    import inspect

    fracs = inspect.signature(run_search).parameters["eval_dx_fractions"].default
    assert any(f > 0 for f in fracs)
    assert any(f < 0 for f in fracs), "eval_dx_fractions default has no negative fraction -- X-direction asymmetry is real (AGENTS.md sec 7)"


if __name__ == "__main__":
    test_action_to_gains_bounds_at_extremes()
    test_env_reset_and_step_shapes()
    test_fast_move_produces_higher_peak_joint_velocity_than_slow_move()
    test_fast_move_regression_search2_gains_trip_joint_velocity_guard()
    test_fitness_fast_move_dimension_is_actually_exercised()
    test_summarize_safety_slow_fast_buckets_are_disjoint_and_correct()
    print("velocity_gain_tuning tests OK")
