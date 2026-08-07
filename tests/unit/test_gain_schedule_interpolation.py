"""Unit tests for velocity_gain_tuning/scheduling/ -- the pure-numpy/scipy
interpolation layer of the per-(pose, displacement) gain schedule.

Lives in tests/unit/ (not tests/mujoco/) deliberately: ``schedule.py`` and
``cells.py`` are simulator-free by construction -- the only import that
would pull gymnasium/MuJoCo in (``action_to_gains``) is lazy and guarded.
``test_schedule_module_import_is_simulator_free`` asserts that property so
it cannot silently regress, since the whole point of keeping it is that a
real-hardware lane should be able to read a schedule without loading a
simulator.

Added 2026-08-06 per AGENTS.md sec 5 ("new modules/packages ship with
pytest coverage, no exceptions").
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from velocity_gain_tuning.scheduling.cells import (  # noqa: E402
    DEFAULT_KNOT_FRACTIONS,
    build_cells,
    build_chains,
)
from velocity_gain_tuning.scheduling.schedule import (  # noqa: E402
    GainSchedule,
    ScheduleKnot,
    _infer_action_dim,
    smooth_actions,
)

DIM = 6


def _knot(scenario: str, frac: float, dx: float, action: np.ndarray, passed: bool = True) -> ScheduleKnot:
    return ScheduleKnot(
        scenario=scenario,
        dx_fraction=frac,
        target_x_delta_m=dx,
        action=np.asarray(action, dtype=np.float64),
        fitness=-1.0,
        passed=passed,
        guard_reasons=(None, None) if passed else ("joint_velocity_guard: 9.9 > 3.0", None),
    )


def _linear_knots(scenario: str = "s", dxs=(-0.2, -0.1, 0.1, 0.2)) -> list[ScheduleKnot]:
    """Knots whose first action component equals dx*2 (so it spans [-0.4,0.4])
    and whose remaining components are constant -- a case where the exact
    interpolated value is analytically known."""
    knots = []
    for dx in dxs:
        action = np.array([2.0 * dx, 0.0, 0.25, -0.25, 0.5, -0.5], dtype=np.float64)
        knots.append(_knot(scenario, dx / 0.1, dx, action))
    return knots


# --------------------------------------------------------------------------
# cells.py
# --------------------------------------------------------------------------


def test_default_knot_fractions_are_bidirectional():
    """AGENTS.md sec 7: +X and -X must both be swept structurally, never as a
    manual follow-up. A schedule fitted on one sign only is exactly the
    failure that rule exists to prevent."""
    assert any(f < 0 for f in DEFAULT_KNOT_FRACTIONS)
    assert any(f > 0 for f in DEFAULT_KNOT_FRACTIONS)
    positive = sorted(f for f in DEFAULT_KNOT_FRACTIONS if f > 0)
    negative = sorted(-f for f in DEFAULT_KNOT_FRACTIONS if f < 0)
    assert positive == negative, "knot grid must be symmetric in sign"


def test_default_knot_fractions_hold_out_unit_fraction():
    """+/-1.0 is intentionally NOT a knot so 16 of the 128 evaluation cells
    are genuine held-out interpolation tests. Losing that would make the
    reported schedule score partly a memorisation score."""
    assert 1.0 not in DEFAULT_KNOT_FRACTIONS
    assert -1.0 not in DEFAULT_KNOT_FRACTIONS


def test_build_cells_scales_fraction_by_scenario_hint():
    cells = build_cells(knot_fractions=(0.5, -0.5))
    assert len(cells) == 8  # 4 scenarios x 2 fractions
    for cell in cells:
        assert cell.target_x_delta_m == pytest.approx(cell.dx_fraction * cell.scenario.max_dx_hint_m)


def test_build_cells_key_is_unique_per_cell():
    cells = build_cells()
    assert len({c.key for c in cells}) == len(cells) == 56


# --------------------------------------------------------------------------
# build_chains -- continuation ordering
# --------------------------------------------------------------------------


def test_build_chains_splits_by_pose_and_direction():
    chains = build_chains(build_cells())
    assert len(chains) == 8  # 4 poses x 2 directions
    for chain in chains:
        assert len({c.scenario.name for c in chain}) == 1
        signs = {c.target_x_delta_m >= 0.0 for c in chain}
        assert len(signs) == 1, "a chain must not straddle dx=0"
        assert len(chain) == 7


def test_build_chains_orders_each_chain_outward_from_zero():
    """Continuation only works if the chain starts at the EASY (small |dx|)
    end and walks toward the hard boundary -- starting at the boundary
    would seed every subsequent cell from a solution found where the
    controller is already failing."""
    for chain in build_chains(build_cells()):
        magnitudes = [abs(c.target_x_delta_m) for c in chain]
        assert magnitudes == sorted(magnitudes)


def test_build_chains_covers_every_cell_exactly_once():
    cells = build_cells()
    chained = [c for chain in build_chains(cells) for c in chain]
    assert sorted(c.key for c in chained) == sorted(c.key for c in cells)


def test_build_chains_is_deterministic_in_order():
    a = [[c.key for c in chain] for chain in build_chains(build_cells())]
    b = [[c.key for c in chain] for chain in build_chains(build_cells())]
    assert a == b


def test_build_chains_puts_zero_displacement_in_the_positive_chain():
    cells = build_cells(knot_fractions=(0.0, 0.5))
    chains = build_chains(cells)
    assert all(len({c.target_x_delta_m >= 0.0 for c in chain}) == 1 for chain in chains)
    assert sum(len(chain) for chain in chains) == len(cells)


# --------------------------------------------------------------------------
# smooth_actions
# --------------------------------------------------------------------------


def test_smooth_actions_window_one_is_identity():
    a = np.random.default_rng(0).normal(size=(7, DIM))
    assert np.allclose(smooth_actions(a, 1), a)


def test_smooth_actions_averages_neighbours_with_edge_padding():
    a = np.array([[0.0], [3.0], [0.0]])
    out = smooth_actions(a, 3)
    # Edge padding repeats endpoints: [0,0,3] -> 1, [0,3,0] -> 1, [3,0,0] -> 1
    assert np.allclose(out.ravel(), [1.0, 1.0, 1.0])


def test_smooth_actions_preserves_a_constant_signal():
    a = np.full((5, DIM), 0.37)
    assert np.allclose(smooth_actions(a, 3), 0.37)


def test_smooth_actions_rejects_even_window():
    with pytest.raises(ValueError):
        smooth_actions(np.zeros((3, DIM)), 2)


def test_smooth_actions_shape_preserved():
    a = np.random.default_rng(1).normal(size=(9, DIM))
    assert smooth_actions(a, 5).shape == a.shape


# --------------------------------------------------------------------------
# GainSchedule interpolation behaviour
# --------------------------------------------------------------------------


def test_schedule_reproduces_knot_values_at_the_knots():
    knots = _linear_knots()
    sched = GainSchedule(knots)
    for k in knots:
        assert np.allclose(sched.action_for("s", k.target_x_delta_m), k.action, atol=1e-9)


def test_schedule_interpolates_between_knots_on_a_linear_ramp():
    """PCHIP is exact on linear data, so the held-out midpoint is analytic."""
    sched = GainSchedule(_linear_knots())
    got = sched.action_for("s", 0.15)
    assert got[0] == pytest.approx(0.30, abs=1e-9)
    assert np.allclose(got[1:], [0.0, 0.25, -0.25, 0.5, -0.5])


def test_schedule_never_overshoots_the_bracketing_knots():
    """The reason PCHIP was chosen over a natural cubic spline: an
    interpolated gain must stay inside the hull of the two validated knots
    that bracket it, so the schedule can never emit a setting more extreme
    than anything that was actually evaluated."""
    rng = np.random.default_rng(7)
    dxs = np.array([-0.20, -0.13, -0.06, 0.06, 0.13, 0.20])
    actions = rng.uniform(-1.0, 1.0, size=(len(dxs), DIM))
    knots = [_knot("s", dx / 0.1, dx, a) for dx, a in zip(dxs, actions)]
    sched = GainSchedule(knots)
    for lo, hi, a_lo, a_hi in zip(dxs[:-1], dxs[1:], actions[:-1], actions[1:]):
        for t in np.linspace(0.0, 1.0, 11):
            got = sched.action_for("s", lo + t * (hi - lo))
            lower = np.minimum(a_lo, a_hi) - 1e-9
            upper = np.maximum(a_lo, a_hi) + 1e-9
            assert np.all(got >= lower) and np.all(got <= upper)


def test_schedule_clamps_instead_of_extrapolating():
    knots = _linear_knots()
    sched = GainSchedule(knots)
    lo_action = sched.action_for("s", -999.0)
    hi_action = sched.action_for("s", 999.0)
    assert np.allclose(lo_action, knots[0].action)
    assert np.allclose(hi_action, knots[-1].action)


def test_schedule_output_always_inside_the_legal_action_box():
    rng = np.random.default_rng(3)
    dxs = np.linspace(-0.2, 0.2, 8)
    knots = [_knot("s", dx / 0.1, dx, rng.uniform(-1.0, 1.0, size=DIM)) for dx in dxs]
    sched = GainSchedule(knots)
    for dx in np.linspace(-0.3, 0.3, 61):
        a = sched.action_for("s", float(dx))
        assert np.all(a >= -1.0) and np.all(a <= 1.0)


def test_schedule_is_continuous_through_zero_displacement():
    """Knots span both signs, so gains must not jump as the commanded
    target crosses zero -- the reason a single signed-dx spline was chosen
    over two mirrored per-direction fits."""
    sched = GainSchedule(_linear_knots())
    left = sched.action_for("s", -1e-6)
    right = sched.action_for("s", 1e-6)
    assert np.allclose(left, right, atol=1e-4)


def test_schedule_keeps_scenarios_independent():
    a_knots = _linear_knots("a")
    b_knots = [
        _knot("b", dx / 0.1, dx, np.full(DIM, -0.9)) for dx in (-0.2, -0.1, 0.1, 0.2)
    ]
    sched = GainSchedule(a_knots + b_knots)
    assert sched.scenarios == ("a", "b")
    assert np.allclose(sched.action_for("b", 0.15), -0.9)
    assert sched.action_for("a", 0.15)[0] == pytest.approx(0.30)


def test_schedule_unknown_scenario_raises():
    sched = GainSchedule(_linear_knots())
    with pytest.raises(KeyError):
        sched.action_for("not_a_pose", 0.0)


def test_schedule_single_knot_scenario_is_constant():
    sched = GainSchedule([_knot("s", 1.0, 0.1, np.full(DIM, 0.4))])
    for dx in (-1.0, 0.0, 0.1, 5.0):
        assert np.allclose(sched.action_for("s", dx), 0.4)


def test_schedule_requires_at_least_one_knot():
    with pytest.raises(ValueError):
        GainSchedule([])


def test_schedule_rejects_mixed_action_dimensions():
    knots = [
        _knot("s", -1.0, -0.1, np.zeros(DIM)),
        _knot("s", 1.0, 0.1, np.zeros(DIM + 1)),
    ]
    with pytest.raises(ValueError):
        GainSchedule(knots)


def test_infer_action_dim_on_empty_is_zero():
    assert _infer_action_dim([]) == 0


def test_schedule_dedupes_duplicate_displacements():
    """PchipInterpolator requires strictly increasing x; two knots at the
    same dx (e.g. a re-run appended to an existing table) must not crash."""
    knots = _linear_knots() + [_knot("s", 1.0, 0.1, np.full(DIM, 0.99))]
    sched = GainSchedule(knots)
    assert len(sched.fitted_knots("s")) == 4
    assert sched.action_for("s", 0.1)[0] == pytest.approx(0.2)


# --------------------------------------------------------------------------
# drop_failed_knots
# --------------------------------------------------------------------------


def test_drop_failed_knots_excludes_failing_knot_from_the_fit():
    knots = [
        _knot("s", -1.0, -0.1, np.full(DIM, 0.1), passed=True),
        _knot("s", 0.0, 0.0, np.full(DIM, -1.0), passed=False),
        _knot("s", 1.0, 0.1, np.full(DIM, 0.1), passed=True),
    ]
    kept = GainSchedule(knots, drop_failed_knots=True)
    all_knots = GainSchedule(knots, drop_failed_knots=False)
    assert len(kept.fitted_knots("s")) == 2
    assert len(all_knots.fitted_knots("s")) == 3
    # With the failing knot dropped, the two surviving (identical) knots
    # carry their value straight across the gap.
    assert np.allclose(kept.action_for("s", 0.0), 0.1)
    assert all_knots.action_for("s", 0.0)[0] == pytest.approx(-1.0)


def test_drop_failed_knots_keeps_everything_when_all_knots_failed():
    """A scenario where no cell has a safe gain vector still needs a
    queryable schedule -- dropping every knot would leave nothing to fit."""
    knots = [_knot("s", f, 0.1 * f, np.full(DIM, 0.2), passed=False) for f in (-1.0, 1.0)]
    sched = GainSchedule(knots, drop_failed_knots=True)
    assert len(sched.fitted_knots("s")) == 2


def test_drop_failed_knots_is_recorded_in_serialization():
    sched = GainSchedule(_linear_knots(), drop_failed_knots=True, smoothing_window=3)
    d = sched.to_dict()
    assert d["drop_failed_knots"] is True
    assert d["smoothing_window"] == 3
    assert GainSchedule.from_dict(d)._drop_failed_knots is True


# --------------------------------------------------------------------------
# serialization
# --------------------------------------------------------------------------


def test_to_dict_from_dict_round_trip_preserves_actions():
    sched = GainSchedule(_linear_knots())
    restored = GainSchedule.from_dict(json.loads(json.dumps(sched.to_dict())))
    for dx in np.linspace(-0.2, 0.2, 9):
        assert np.allclose(restored.action_for("s", float(dx)), sched.action_for("s", float(dx)))


def test_from_dict_overrides_beat_stored_fit_options():
    """Lets one searched knot table be re-fitted several different ways
    (raw / dropfail / smoothed) and scored against each other, which is how
    the fit variant was chosen empirically instead of by assumption."""
    sched = GainSchedule(_linear_knots(), smoothing_window=1)
    restored = GainSchedule.from_dict(sched.to_dict(), smoothing_window=3)
    assert restored._smoothing_window == 3


def test_save_load_round_trip(tmp_path):
    sched = GainSchedule(_linear_knots())
    path = tmp_path / "nested" / "sched.json"
    sched.save(path)
    restored = GainSchedule.load(path)
    assert np.allclose(restored.action_for("s", 0.05), sched.action_for("s", 0.05))


def test_knot_to_dict_records_guard_reasons_and_pass_flag():
    knot = _knot("s", 1.0, 0.1, np.zeros(DIM), passed=False)
    d = knot.to_dict()
    assert d["passed"] is False
    assert d["guard_reasons"][0].startswith("joint_velocity_guard")
    assert ScheduleKnot.from_dict(d).passed is False


# --------------------------------------------------------------------------
# import purity
# --------------------------------------------------------------------------


def test_schedule_module_import_is_simulator_free():
    """Importing the schedule must not drag in gymnasium/MuJoCo. Run in a
    subprocess because this test session has almost certainly already
    imported them for other tests."""
    repo = str(Path(__file__).resolve().parents[2])
    code = (
        "import sys;"
        f"sys.path.insert(0, {repo!r});"
        "import velocity_gain_tuning.scheduling.schedule as s;"
        "import velocity_gain_tuning.scheduling.cells as c;"
        "print(int('mujoco' in sys.modules), int('gymnasium' in sys.modules))"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    ).stdout.strip()
    assert out == "0 0", f"schedule/cells import pulled in a simulator: {out}"
