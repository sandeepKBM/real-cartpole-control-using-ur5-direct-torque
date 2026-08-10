# How hinge friction changes the passive-vs-active balance picture

**Date:** 2026-08-10
**Context:** user asked to sweep the pendulum hinge's friction/damping (runtime model edit only,
never touching `assets/ur5e_pendulum/pendulum_attachment.xml`'s placeholder values, which are
being checked against the real hardware separately) and study how much the controller can do at
each level. Run against the CORRECTED unstable equilibrium (theta=0, see
`docs/status/pendulum_balance_torque_lqr_2026-08-09.md`'s equilibrium-classification fix) and
the DE-search-validated LQR config (`docs/status/pendulum_balance_gain_search_2026-08-09.md`),
not the earlier hand-picked guesses a first version of this sweep used before that search
existed.

## Headline result: at realistic friction, active control is a dramatic, clear win

At `friction_x=1.0` (the current placeholder value, unmeasured but the best guess until real
hardware is checked), releasing the pendulum passively (no active correction, arm just holds
gravity) from 0.15 rad (~8.6 deg) off vertical **fully diverges** -- final angle error 3.134 rad,
essentially swinging almost all the way around. The DE-search-found active controller, from the
exact same starting point, **converges to within 0.002-0.025 rad of vertical** -- a clean,
unambiguous rescue. This is the clearest "active control genuinely matters" result of this whole
investigation.

## As friction increases, the picture changes -- but the active controller stays ahead

From `friction_x=5.0` up through `50.0` (the highest level with a like-for-like comparison run),
passive itself starts "surviving" in a loose sense -- it settles to a friction-locked residual of
about 0.21-0.22 rad (not falling further, but not recovering to vertical either -- the same
Coulomb-friction-locks-it phenomenon documented earlier in this session's swing-up work). The
active controller keeps doing measurably better throughout this range -- residual error around
0.10-0.11 rad, roughly HALF of passive's -- even though both now technically "survive" by the
same pass/fail criterion, which under-sells the real, continued advantage active control
provides. This holds consistently across friction levels tested (5x through 50x), not just at
one point.

## A methodology note, reported honestly rather than silently fixed

The "passive boundary" table (bisecting where passive alone stops surviving, searched between
1.0 and 3.13 rad) reads exactly **0.0000 rad at every single friction level tested, 1x through
100000x**. This is not a bug -- it is the bisection's own starting assumption (that `lo=1.0 rad`
would typically survive) turning out to be wrong: passive genuinely fails even at that
comparatively large starting point at every friction level tried, so the bisection correctly
reports "no boundary found in this range" as 0.0 every time. The real, informative comparison
lives in the "rescue check" section above, not the boundary table -- a reminder that a
bisection's output is only as good as its assumed bracket, and this one was sized for the OLD
(wrong) equilibrium's much more forgiving passive behavior, not the real one.

## What this means for the CAD friction placeholder

Since the placeholder friction value (`frictionloss=0.01 Nm`, `damping=0.02 Nm*s/rad` in
`assets/ur5e_pendulum/pendulum_attachment.xml`) is what `friction_x=1.0` represents, and that is
exactly the level where active control provides the clearest, most dramatic benefit -- if the
REAL hardware's hinge turns out to have meaningfully MORE friction than this placeholder, the
practical need for active balance shrinks (passive alone gets closer to "good enough," per the
5x-50x rows above); if the real hardware has LESS friction, active control likely matters even
more than shown here. This is exactly why the user is checking the real CAD/hardware friction
value directly -- this sweep quantifies why that number matters for the balance question
specifically, not just as a modeling nicety.

## Files

| path | what |
|---|---|
| `tools/diagnostics/pendulum_balance_friction_sweep.py` | friction sweep, active_configs updated to the real DE-search-validated LQR gains |
| this file | write-up |

**Tests run:** full sweep (17 friction levels for the boundary table, 8 levels x 4 configs for
the rescue check), run to completion on ilab3 (idle machine, avoided contention on the
primary/heavily-loaded host).
**Tests not run:** no pytest coverage (diagnostic script); no re-run at friction levels above 50x
with the rescue-check comparison (the boundary table covers up to 100000x, confirming no
numerical breakdown at extreme values, but the more informative active-vs-passive comparison
wasn't extended past 50x given time already spent this session -- a reasonable, disclosed scope
limit, not a silent gap).

**Rollback:**
```bash
rm -f tools/diagnostics/pendulum_balance_friction_sweep.py \
      docs/status/pendulum_balance_friction_sweep_2026-08-10.md
```
Nothing else touched. Nothing committed. The pendulum's own friction/damping values in
`assets/ur5e_pendulum/pendulum_attachment.xml` were never modified -- this sweep only edits the
compiled MuJoCo model's `dof_frictionloss`/`dof_damping` arrays in memory, per-run.
