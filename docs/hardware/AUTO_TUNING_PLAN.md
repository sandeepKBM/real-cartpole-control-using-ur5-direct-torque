# Real-hardware auto-tuning plan (not yet implemented)

**Status:** design only. No code in this repo implements any of this yet.
**Scope:** longer-term project, not tied to a specific lab date.
**Last updated:** 2026-07-26

This is the plan for automatically calibrating the OSC controller's gains against the
real UR5e once it's available, instead of hand-tuning by eye the way the MuJoCo-side
gains were tuned (see `AGENTS.md` §3). It grew out of three findings from the
sim-side investigation:

1. Two real controller defects (`lambda_diagonal_shaping`, `lambda_adaptive_regularization`
   — see `controller_core/x_axis_cartesian_impedance.py`) were found and fixed without
   moving the achievable X-displacement ceiling, which points at the ~0.25-0.3m limit
   being a structural property of the `kp_rot=0` design, not a tuning gap.
2. The URScript control law (Mode 3) now has numerical parity with the validated Python
   controller (`tests/hardware/test_urscript_parity.py`), but **nothing on this hardware
   lane has ever executed on real hardware or URSim** — every mode is code-complete and
   only exercised through mocked unit tests.
3. Gains tuned entirely in MuJoCo will have some sim-to-real gap (friction, backlash,
   RTDE/network latency, actuator dynamics the model doesn't capture) that only shows up
   once the real arm moves.

## Decisions already made (don't re-litigate these without a reason)

- **Timeline:** longer-term. No pressure to rush this for an imminent lab visit — get the
  safety architecture right first.
- **Autonomy:** human confirms each batch of candidate gains before it runs on the real
  robot. Not fully unattended, even though the existing safety monitors (`EStopLatch`,
  `CartesianMoveMonitor`, `ImpedanceSafetyMonitor`, `DeadlineMonitor`, `StaleStateMonitor`)
  would in principle catch a bad trial on their own — the first time this whole hardware
  lane runs at all is not the moment to also test them unattended.
- **Goal:** tune for **reliability within the already-proven range** (dx=0.10-0.20m at the
  heights validated in sim), not to push past the ~0.25-0.3m ceiling. That ceiling looks
  structural (§3 above), so spending real-robot trials chasing it is a bad bet until the
  `kp_rot=0` design itself changes.

## Why not step-wise RL on the real robot

`rl_gain_scheduling/gain_scheduling_env.py` already exists and works this way (a policy
picks new gains every control step, trained via PPO). That architecture is right for
MuJoCo, where a training rollout costs milliseconds and a bad episode costs nothing. It
is wrong for real hardware: every trial is slow (real wall-clock seconds per move+hold),
costly (someone has to be present), and a bad candidate is a bad candidate on a real
robot. Real-hardware auto-tuning needs a paradigm that converges in **tens of trials**,
not the thousands-to-millions of environment steps PPO needs. That means black-box
per-episode optimization (Bayesian optimization / Optuna-style TPE) over a **small**
number of scalar gains, not step-wise learning.

## Architecture: sim-gated Bayesian optimization

The real robot is the last, low-frequency step in the loop, not the search engine:

```
propose candidate gains (BO)
        │
        ▼
run in MuJoCo first (tools/ur5e_move_hold_transport.py -- already exists)
        │
   fail? ──► reject, BO proposes a new candidate, no robot involved
        │
   pass
        │
        ▼
queue into a batch (a handful of candidates)
        │
        ▼
show batch to a human for approval  ◄── you, before anything moves
        │
        ▼
run each approved candidate once on the real robot
(hardware/direct_torque_transport.py, existing safety monitors unchanged)
        │
        ▼
score (tracking error + safety pass/fail) via RunLogger, feed back to BO
```

## Milestones (strictly ordered — do not start N+1 before N works)

### M0 — Baseline real-hardware validation (hard prerequisite)

Nothing below this point makes sense until the *unmodified, sim-validated* gains have
actually moved the real arm once, in each mode, with a human present:

1. `tools/ur5e_connect.py --once` — confirm the link works at all.
2. `position` mode, `--distance-m 0.02`, `--i-understand-this-moves-the-robot`, typed
   `MOVE` confirmation. First-ever real motion in this repo.
3. `direct_torque` mode, same tiny displacement, fixed validated gains, no exploration.
   This is Mode 2's first-ever execution anywhere (never run on URSim or real hardware
   before — URSim can only validate the `directTorque()` API call succeeds, not that the
   resulting motion is sane, since URSim has no torque physics).
4. `urscript` mode, same tiny displacement. Mode 3's first-ever execution anywhere too.
5. Only after all three pass at `dx=0.02m`: escalate to the sim-validated envelope
   (dx=0.10-0.20m) with the *unmodified* gains, still no auto-tuning. This establishes the
   real-hardware baseline that M3's search space gets built around.

### M1 — Single-trial hardware harness

A CLI/function that runs exactly one fixed-gain, fixed-`dx` episode on the real robot
through the existing `hardware/direct_torque_transport.py` loop, overriding gains via the
same `controller.set_gains()` mechanism the sim RL env already uses (see
`XAxisCartesianImpedanceController.set_gains()` — no new controller code needed), logs a
run record via `observability/run_logger.py::RunLogger` (already the standard for every
sweep entrypoint), and returns one scalar score. No search yet: the only goal is proving
one trial round-trips correctly using gains already known good from sim, matching M0's
numbers.

### M2 — Sim gate

Reuses `tools/ur5e_move_hold_transport.py` as-is: before any candidate gain vector is
queued for a real trial, it runs there first. Only sim-passing candidates ever reach the
real robot. This is the single biggest risk-reducer in the plan and needs no new code,
just a wrapper that calls the existing tool and checks its `valid_move_and_hold` result.

### M3 — Bayesian optimization driver (new code)

- Search space: small perturbations (~±20-30%) around the M0-validated gains, likely
  restricted to the subset most exposed to unmodeled real-world effects --
  `kp_x`/`kd_x` (tracking stiffness, most sensitive to latency) and `kd_joint` (damping,
  most sensitive to friction/backlash) -- rather than all 11 schedulable gains
  (`CartesianImpedanceConfig._SCHEDULABLE_GAIN_FIELDS`). Widening the space is a later
  decision, not a starting point.
- Evaluated only at displacements already proven safe in sim (dx=0.10-0.20m per the
  "reliability" goal above), not at or near the fragile ~0.25-0.3m boundary.
- Sample-efficient method (Optuna TPE or `scikit-optimize` `gp_minimize`), batched --
  propose a handful of candidates, sim-gate all of them, present the survivors to you as
  one batch, run only what you approve.
- Objective: tracking error (RMS or final `x_error`) plus orientation/drift margin from
  the run record, with any safety-monitor abort treated as a hard reject for that
  candidate, not folded into the score.

### M4 — Land the result

Best real-validated gain set becomes a new named config, e.g.
`config/ur5e_mujoco_torque_osc_tuned_real_calibrated.yaml` -- never overwriting
`config/ur5e_mujoco_torque_osc_tuned_adaptive_lambda.yaml` or any other existing config,
per the repo's "preserve old configs" convention (`AGENTS.md` §7).

## Explicitly out of scope for this plan

- Step-wise/PPO-style RL directly on real hardware (see above).
- Attempting to extend past the ~0.25-0.3m ceiling (per the "reliability" decision --
  revisit only if the `kp_rot=0` structural limit itself gets addressed first).
- Any change to the safety monitors themselves. This plan adds a search loop *around*
  the existing hardware safety stack; it does not touch `hardware/safety.py`,
  `controller_core/safety.py`, or any guard threshold.
- Fully unattended trial execution (per the "human confirms each batch" decision).

## What's buildable before ever touching the real robot

M1 (harness structure, testable against `tests/hardware/`'s existing mocked-RTDE
pattern), M2 (already exists), and M3 (BO driver, testable end-to-end against MuJoCo
alone, no real robot involved) can all be built and validated in sim/mocks now. Only M0
and the "run approved candidates" step of M3 need the physical arm.
