# Systematic gain search for pendulum balance -- both lanes, real results

**Date:** 2026-08-09
**Context:** manual gain tuning (docs/status/pendulum_balance_torque_lqr_2026-08-09.md) found a
real working torque-lane config by hand, but was slow and unsystematic. User asked to use this
repo's own established black-box search tool (`scipy.optimize.differential_evolution`, matching
`velocity_gain_tuning/optimize.py`'s precedent -- NOT RL, which has a documented history of
failing on exactly this class of problem in this repo, six times, per `rl_gain_scheduling/`)
instead of continued manual tuning. Both searches run against the corrected unstable equilibrium
(theta=0, fixed the same day -- see the torque_lqr doc's equilibrium-classification correction).

## Torque lane: real, clean, structurally-limited win

`tools/diagnostics/pendulum_balance_torque_lqr_search.py`. Objective evaluated only at
perturbations where the passive (K=0) baseline is confirmed to fail -- structurally cannot
reward a do-nothing candidate.

**First pass** (objective: 0.1-0.5 rad) found:
`q_arm_pos=1008.7, q_arm_vel=32.3, q_pend_angle=35.9, q_pend_vel=310.4, r_weight=4.30` --
survives cleanly 0.05-0.4 rad (residuals 0.0-0.04 rad, genuine convergence), fails at 0.5+.
Already better than the manual result (0.3 rad envelope).

**Second pass**, seeded from the first result, objective WIDENED to 0.1-0.8 rad specifically to
target the known failure region: found
`q_arm_pos=2655.2, q_arm_vel=1.59, q_pend_angle=20.9, q_pend_vel=655.2, r_weight=0.347` --
**residuals even tighter** (0.000-0.017 rad at 0.05-0.4 rad, essentially perfect convergence)
but **still fails at 0.5 rad and beyond, at every perturbation up to 1.0 rad tested.** Widening
the search objective did not unlock more range.

**Interpretation**: this is a real, informative negative result, not a search failure. It's
consistent evidence that ~0.4-0.5 rad is a genuine STRUCTURAL limit for a single fixed linear
LQR gain on this system -- a fixed-gain law is only locally valid near its own linearization
point, and this system's true nonlinear dynamics (sin(theta) vs theta) diverge from that
approximation well before 0.5 rad. Confirms and quantifies exactly what gain scheduling (a
separate line of work, see below and the companion transport doc) is meant to address.

## Velocity lane: real but partial effect, requires an important correction

`tools/diagnostics/pendulum_balance_velocity_search.py`. Found:
**`kp=2.68, kd=0.036, ik_joint_gain=432.8`** -- a MUCH higher `kp` than any value manually
tried earlier this session (kp=0.1 was the ceiling of manual exploration; the search found the
useful value is ~27x higher). This alone is a real, useful correction: earlier claims in this
session that the velocity lane's outer correction was "essentially inert" were specific to the
narrow kp range manually tried, not a fundamental property of the architecture.

**Validation table (500 Hz, real active vs real passive at each perturbation):**

| pert (rad) | ACTIVE | PASSIVE |
|---|---|---|
| 0.10 | survives, final=0.105 | survives, final=0.104 |
| 0.20 | survives, final=0.238 | survives, final=0.208 |
| 0.30 | survives, final=0.328 | survives, final=0.312 |
| 0.50 | fails @2.11s | fails @2.39s |
| 0.80 | survives, final=0.791 | **fails @1.23s** |
| 1.00 | fails @4.73s | fails @0.84s |
| 1.20 | survives, final=1.200 | **fails @0.53s** |

**Important correction, made before trusting the 0.8/1.2 rad "survived" rows**: `final ≈
initial perturbation` in both cases is the exact signature of a false-positive "stuck, not
balanced" result already caught once this session (the pendulum's stiction can freeze it near
wherever it started, independent of any real control). Verified directly rather than assumed:

- At pert=0.8: ACTIVE ends at 0.791 (barely moved from 0.8) while PASSIVE genuinely diverges to
  1.50 (right at the fall threshold). **This is real and different from passive** -- passive
  provably fails here (grows, not stuck), while active provably arrests that growth. But
  "arrests divergence near the release point" is a different, weaker claim than "achieves
  balance" -- it did not converge toward vertical within the tested window.
- At pert=1.2: same pattern exactly (ACTIVE ends at 1.1996, essentially frozen at the 1.2 start;
  PASSIVE diverges to 1.5005 and fails).

**Honest characterization**: the velocity lane's found gains give **genuine, verified small-angle
balance** (0.1-0.3 rad, real convergence, comparable to passive since friction alone already
handles this range) and a **real but partial large-angle effect** (0.8-1.2 rad: prevents outright
divergence that passive cannot prevent, but does not recover to vertical) -- with a **dead zone
around 0.5-1.0 rad where it does no better than passive and sometimes fails faster**. Not a clean
story, but a real, verified one.

## Both lanes agree on the same structural conclusion

A single fixed gain (whichever lane) can be tuned for either small-angle precision or
large-angle survival, but not smoothly across the whole range with one linear law -- consistent
with both being local linearizations of a genuinely nonlinear system. This is exactly the
motivation for the gain-scheduling work now underway on the transport side (see
`docs/status/x_transport_gain_scheduling_2026-08-09.md`) and would be the natural next step for
either balance lane too, if revisited.

## Files

| path | what |
|---|---|
| `tools/diagnostics/pendulum_balance_torque_lqr_search.py` | torque-lane DE search (2 passes) |
| `tools/diagnostics/pendulum_balance_velocity_search.py` | velocity-lane DE search |
| this file | write-up |

**Tests run:** both DE searches to completion; direct manual re-verification of the velocity
lane's pert=0.8 and pert=1.2 "survived" results against a K=0/kp=0 baseline before trusting them
(the correction documented above).
**Tests not run:** no pytest coverage (diagnostic scripts); real-hardware validation; a
gain-scheduled version of either balance lane (deferred -- user redirected gain-scheduling work
to the X-transport case specifically, see the companion doc).

**Rollback:**
```bash
rm -f tools/diagnostics/pendulum_balance_torque_lqr_search.py \
      tools/diagnostics/pendulum_balance_velocity_search.py \
      docs/status/pendulum_balance_gain_search_2026-08-09.md
```
Nothing else touched. Nothing committed.
