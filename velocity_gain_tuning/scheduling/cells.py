"""Knot-cell definitions for the per-(pose, displacement) gain schedule.

A "cell" here is one (pose scenario, signed target X displacement) pair --
the unit that gets its OWN differential_evolution search in search.py and
becomes one interpolation knot in schedule.py.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..poses import POSE_SCENARIOS, PoseScenario

# Signed fractions of each scenario's ``max_dx_hint_m`` at which a knot is
# searched.
#
# Chosen to line up with the standard post-search evaluation grid used by
# every historical result in outputs/velocity_gain_tuning/ (optimize.py's
# ``run_search(eval_dx_fractions=...)`` default:
# +/-{0.3, 0.6, 0.9, 1.0, 1.1, 1.3, 1.6, 2.0}) with ONE deliberate
# omission: +/-1.0 is NOT a knot. That fraction is therefore a genuine
# held-out interpolation test -- 16 of the 128 evaluation cells are
# scored at a displacement the search never optimized for, which is the
# only honest way to tell "the schedule generalizes between knots" apart
# from "the schedule memorized the grid." Do not add 1.0 back without
# also picking a different held-out fraction.
#
# BOTH SIGNS are included, and that is structural, not optional -- see
# AGENTS.md sec 7 and evaluate.py's docstring: this repo has repeatedly
# measured real +X/-X asymmetry (a gain vector passing cleanly at +0.37m
# tripping the joint-velocity guard at -0.37m). A schedule fitted on
# positive displacements only and mirrored would be exactly the failure
# that rule exists to prevent. Because the knots span negative through
# positive, the fitted interpolant is a function of SIGNED dx and is
# continuous through dx=0 -- gains cannot jump discontinuously as the
# commanded target crosses zero, which is both physically sensible and
# what makes a single per-pose spline (rather than two mirrored halves)
# the right structure.
DEFAULT_KNOT_FRACTIONS: tuple[float, ...] = (
    -2.0, -1.6, -1.3, -1.1, -0.9, -0.6, -0.3,
    0.3, 0.6, 0.9, 1.1, 1.3, 1.6, 2.0,
)


@dataclass(frozen=True)
class GainCell:
    """One (pose, signed displacement) search cell."""

    scenario: PoseScenario
    dx_fraction: float
    target_x_delta_m: float

    @property
    def key(self) -> str:
        return f"{self.scenario.name}@{self.dx_fraction:+.2f}"


def build_cells(
    scenarios: tuple[PoseScenario, ...] = POSE_SCENARIOS,
    knot_fractions: tuple[float, ...] = DEFAULT_KNOT_FRACTIONS,
) -> list[GainCell]:
    """Cartesian product of scenarios x knot_fractions, in a stable order.

    ``target_x_delta_m`` is the fraction scaled by that scenario's own
    ``max_dx_hint_m``, matching how every other module in this package
    (optimize.fitness, evaluate.evaluate_gains) turns a fraction into a
    real displacement -- so a knot at fraction f is the SAME physical
    displacement as the evaluation grid cell at fraction f.
    """
    cells: list[GainCell] = []
    for scenario in scenarios:
        for frac in knot_fractions:
            cells.append(
                GainCell(
                    scenario=scenario,
                    dx_fraction=float(frac),
                    target_x_delta_m=float(frac) * scenario.max_dx_hint_m,
                )
            )
    return cells


def build_chains(cells: list[GainCell]) -> list[list[GainCell]]:
    """Group cells into (pose, direction) CHAINS ordered by increasing |dx|.

    Why chains exist -- measured, not assumed (2026-08-06). Searching every
    cell independently and splining through the results produced a schedule
    that beat the fixed-gain baseline at the searched knots (108/112) but
    was WORSE than it at the one held-out displacement fraction (9/16 vs
    16/16). Diagnosis: the single-cell objective has many near-equivalent
    optima scattered across the action box, so independent searches at
    neighbouring displacements land on entirely different ones. A point
    interpolated between two such optima is not itself a good gain vector
    -- the same effect that made a 3-wide moving average over the knots
    collapse the score from 117/128 to 81/128 with peak |qd| jumping from
    10.6 to 27.1 rad/s. Splining is only valid if the knot SEQUENCE is
    coherent, and coherence has to be produced during the search, not
    recovered afterwards.

    A chain fixes that by continuation: search the smallest |dx| first,
    then seed each subsequent cell's population with (and optionally
    penalise its distance from) the previous cell's solution, so the whole
    chain tracks ONE continuous branch of optima outward from the easy
    interior toward the hard boundary.

    Split by DIRECTION as well as pose, and grown outward from |dx| ~ 0 in
    each direction, because this repo's +X/-X asymmetry (AGENTS.md sec 7)
    means the two directions genuinely are different branches -- chaining
    straight through dx=0 from -2.0x to +2.0x would force one direction's
    branch onto the other. The two chains still meet near dx=0, which is
    where the branches should agree anyway.
    """
    chains: dict[tuple[str, str], list[GainCell]] = {}
    for cell in cells:
        direction = "neg" if cell.target_x_delta_m < 0.0 else "pos"
        chains.setdefault((cell.scenario.name, direction), []).append(cell)
    return [
        sorted(group, key=lambda c: abs(c.target_x_delta_m))
        for _, group in sorted(chains.items())
    ]
