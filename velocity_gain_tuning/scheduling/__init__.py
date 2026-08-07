"""Per-(pose, displacement) gain SCHEDULING for
``controller_core.cartesian_velocity_controller``'s ``ik_seeded_resolution``
mode -- i.e. gains that vary smoothly as a function of the commanded
transport displacement, instead of one fixed global gain vector.

WHY THIS EXISTS
---------------
Every search in ``velocity_gain_tuning/optimize.py`` looks for ONE gain
vector that has to serve all four pose scenarios and the full signed
displacement range simultaneously. Ten-plus real searches on 2026-08-06
(``outputs/velocity_gain_tuning/search_result_*.json``) plateaued in the
90-108 / 128 band regardless of search budget, bounds width, population
seeding, cell reweighting, or which redundancy-resolution mechanism was
available -- the signature of a genuine multi-objective Pareto conflict,
not a search-budget shortfall. If different (pose, dx) cells genuinely
want different gains, no single vector can be optimal for all of them and
the only way forward is to let the gains depend on the cell.

WHY *NOT* RL (the explicitly-considered alternative)
----------------------------------------------------
The obvious "reuse what's already here" answer is to adapt
``rl_gain_scheduling/`` -- this repo's PPO gain scheduler for the separate
TORQUE-control lane. That was evaluated and REJECTED, on this repo's own
recorded evidence rather than on taste:

  * ``docs/CURRENT_STATUS.md`` documents SIX real training attempts
    (run1_200k, run2_continued_2.2M, reward_v2_2M, reward_v3_2M,
    reward_v4/v5_height0.5, and the 2026-07-29 3M-step config-mismatch-fix
    run). NONE beat the fixed-gain baseline on its own eval grid; the best
    was 5/8 vs. the baseline's 7/8, and several were 0/20 or 0/8.
  * The 2026-07-25 read-only audit root-caused it as an EXPLORATION /
    deceptive-local-optimum failure (every dense reward term except
    ``x_error`` is minimized by not moving; the path from "sit still" to
    "clean move" runs through "move clumsily," which scores worse than
    either endpoint) plus ``ent_coef=0`` policy-variance collapse. Those
    are properties of the sequential-RL framing itself, not of the torque
    lane -- porting the framing ports the failure mode.
  * ``AGENTS.md`` sec 3 explicitly recommends against another RL attempt
    for that lane in favour of simpler supervised/regression approaches.
  * This package's parent ``velocity_gain_tuning/__init__.py`` already
    made the same call for the global search, and that call was correct:
    differential_evolution found usable gains on the FIRST run, where six
    RL attempts found none.

WHAT THIS PACKAGE DOES INSTEAD
------------------------------
A two-stage, entirely RL-free pipeline built out of machinery that is
already validated in this repo:

  1. ``search.py`` -- run the SAME ``differential_evolution`` search that
     already works, but INDEPENDENTLY per (pose, signed-dx) knot cell,
     with a single-cell objective. Each cell is a small, easy, unimodal-ish
     problem (~1.5k episode evaluations, ~10 min single-core) instead of
     one hard 56-episode multi-pose compromise. Every cell's population is
     seeded with the best known GLOBAL vector, so a cell's result is a
     strict "can a cell-specific vector beat the global one here?" test.
  2. ``schedule.py`` -- fit a shape-preserving PCHIP interpolant through
     those per-cell optima, per pose, per action dimension, so gains
     become a smooth function of the commanded ``target_x_delta_m``.

The stage-1 output is also, by construction, an ORACLE UPPER BOUND on what
ANY (pose, dx)-conditioned gain function -- spline, regression, or a
hypothetical RL policy observing only (pose, dx) -- could ever achieve on
this evaluation grid. That number is the single most decision-relevant
measurement in this package: if the oracle is barely above the fixed-gain
baseline, then gain scheduling on (pose, dx) is the wrong lever entirely
and no amount of policy-class sophistication rescues it. Measuring it
costs one DE sweep; measuring it via RL would cost six.

Validation reuses ``velocity_gain_tuning/evaluate.py``'s existing
``summarize_safety`` and the same guard thresholds as every other result
this session, so the numbers are directly comparable to the historical
``search_result_*.json`` pass counts.
"""

from .cells import DEFAULT_KNOT_FRACTIONS, GainCell, build_cells
from .schedule import GainSchedule, ScheduleKnot

__all__ = [
    "DEFAULT_KNOT_FRACTIONS",
    "GainCell",
    "GainSchedule",
    "ScheduleKnot",
    "build_cells",
]
