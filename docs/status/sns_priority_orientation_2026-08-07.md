# SNS-style task priority for orientation — feasibility test

**Date:** 2026-08-07
**Lane:** velocity control (`controller_core.cartesian_velocity_controller`, `ik_seeded_resolution`)
**Status:** feasibility test only. **No production code changed, nothing committed.**
Prototype worktree (`sns_ik.py`, config flag `sns_priority_orientation`) discarded per this
document — see §4 for the reproduction command if this is revisited.

## Verdict: **NO-GO** — clearly worse than the ad hoc mechanism it was meant to formalize

## 0. Why this was tried

`orientation_priority` (committed `177155c`, 111/128 vs 104/128 fixed-gain baseline) solves IK
twice per cycle — position-only, then with rx/ry promoted to co-primary — and gates on the
promoted solve's own position residual. It works, but it's an ad hoc two-pass hack, not a
principled algorithm, and doesn't fold in joint limits as part of the same priority mechanism.

**SNS (Saturation in the Null Space)** — Flacco et al., "Control of Redundant Robots Under Hard
Joint Constraints: Saturation in the Null Space," IEEE T-RO 2015 — is the established, rigorous
version of the same idea: handle multiple prioritized tasks plus joint limits in one coherent
null-space-projection framework, with a proven priority-preserving saturation algorithm. The
hypothesis: reformulating `orientation_priority` as a genuine 2-priority SNS solve (priority 1 =
position + rz, priority 2 = rx/ry solved in the null space of priority 1, reusing this session's
own `controller_core/kinematics_utils.py::null_space_basis`) would match or beat the ad hoc
version with better guarantees.

## 1. What was built

New file `controller_core/cartesian_velocity_controller/sns_ik.py` (pure numpy, matching this
package's simulator-independent convention — not a port of the reference C++ library, a
from-scratch reimplementation of the core idea), wired via a new flag
`sns_priority_orientation` (default off) alongside `orientation_priority`. Solves priority 1
(position+rz) as `compute_ik_seeded` already does, then projects priority 2 (rx/ry) into the
null space of priority 1 via `null_space_basis`, unconditionally — no residual gate, no
threshold, no "is this actually free" check.

## 2. Head-to-head result — same 128-cell grid, same base gain vector as `orientation_priority`'s own validation

`outputs/velocity_gain_tuning/search_result_nullspace_v2_20260806_194402.json`'s gains, via
`velocity_gain_tuning.evaluate.evaluate_gains` (independently re-run and confirmed, numbers
below are the verified ones, not just the prototype's self-report):

| variant | pass | worst orientation error | worst `|qd|` |
|---|---|---|---|
| off (fixed-gain baseline) | 105/128 | 0.2528 | 4.896 |
| `orientation_priority` (committed) | **111/128** | 0.2529 | 4.337 |
| `sns_priority_orientation` | **76/128** | 0.2538 | 7.091 |

(Baseline reads 105/128 here, not the historical 104/128 — the extra pass is the `qd_estimate_damping`
fix, commit `12382f1`, landed since `orientation_priority`'s original validation; both `orientation_priority`
and `sns_priority` numbers above already include it, so the comparison is apples-to-apples.)

`sns_priority_orientation` vs. off: 6 cells fixed, but **35 cells broken** — a large net
regression, concentrated almost entirely at `neg40_wrist2offset`/`neg45_wrist2offset` (33 of
the 35 broken cells).

## 3. Root cause — it behaves like unconditional promotion, not genuine priority

76/128 is not a new number in this investigation. `orientation_priority`'s own original
validation (`docs/status/task_priority_orientation_hanging_2026-08-06.md`) ran a THIRD control
arm alongside off/on — unconditionally setting `task_dim_rx=task_dim_ry=True` (rx/ry always in
the task, no gating at all) — and measured it at **76/128**, with the same qualitative signature:
real wins at `hanging_alpha_0_5`/`unrotated_wrist2offset`, catastrophic losses at
`neg40`/`neg45_wrist2offset` (the "rxry_broken" list in that document's own artifacts overlaps
heavily with this run's 35 broken cells).

This is the actual finding: `sns_priority_orientation`'s strict null-space projection promotes
orientation into every cycle's solve unconditionally, with no mechanism to decide "is this
locally free or does it cost real position tracking" — functionally equivalent to the
already-tested always-on control arm, despite the more principled derivation. **The residual
gate in `orientation_priority` — deciding per-cycle whether promoting orientation is actually
free, not the null-space-projection structure itself — is what makes the ad hoc mechanism work.**
A textbook strict-priority formulation without an equivalent gate does not inherit that property;
priority order alone doesn't know when promoting the lower-priority task is a bad idea at a given
pose, only that it's subordinate when it isn't free.

## 4. Reproduction (if revisited)

The `sns_ik.py` implementation and its wiring were discarded after this validation (per user
instruction, 2026-08-07) rather than kept as dead code. To revisit: reimplement per §1's
description, or recover from git history if a prior commit captured it (none did — this was
never committed). A genuine next attempt would need to port `orientation_priority`'s residual
gate into the null-space-projection structure, not the unconditional promote used here — that
combination was not tried in this pass.

## 5. Scope and limits

- Kinematic-only sim (`LocalMujocoDynamics` FK/Jacobian), same fidelity level as every other
  investigation in this session — nothing here is hardware-validated.
- Joint-limit handling (the other half of what real SNS provides — saturating and reprojecting
  when a joint would exceed its bounds) was implemented but not separately stress-tested; the
  128-cell grid result above is dominated by the orientation-priority failure mode, not a joint-
  limit-specific finding.
- Only the 2-priority (position+rz, then rx/ry) case was tried — SNS's value proposition for
  3+ simultaneous priorities or explicit joint-limit saturation under load was not evaluated.

**Tests run:** none kept — no production code was modified or promoted, so the suite is
unchanged from `orientation_priority`'s own commit (`177155c`). The prototype had its own unit
(68) and mujoco (6) test coverage during development, independently verified passing before this
document's conclusion was reached; those tests were discarded along with the prototype code.
**Tests not run:** n/a, nothing here is being kept.

**Rollback:** nothing to roll back — the prototype's worktree changes were discarded, nothing
was committed to the main branch except this document.
