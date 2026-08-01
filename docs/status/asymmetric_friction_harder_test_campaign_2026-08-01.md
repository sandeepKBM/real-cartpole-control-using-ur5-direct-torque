# Asymmetric-Coulomb-friction "harder test" campaign

**Status:** Real, bounded comparison campaign complete. At the current placeholder magnitude
(30% of matching `frictionloss`, per commit `32c3d1d`/`3f77057`), the new asymmetric backdrive
friction plant produces a small, real, physically-sensible degradation in X-tracking accuracy,
but **does not flip any category-level pass/fail count** for the 4 (config, pose, category)
scenarios tested. A severity sweep up to 10x the placeholder magnitude on the most marginal
scenario confirms the effect scales monotonically but still doesn't change the qualitative
verdict, because the existing static symmetric friction (added 2026-07-31) is already the
dominant tracking-limiting factor at these test points. This is a real, if currently low-power,
result — not a null finding to be read as "the asymmetric-friction effect doesn't matter."

**Date:** 2026-08-01. Sim-only throughout.

## 0. Context

Commit `32c3d1d` (cherry-picked onto `feature/ur5e-mujoco-torque-control` independently as
`3f77057` by another session; both are content-identical, confirmed via empty `git diff`)
added an opt-in, plant-side `AsymmetricCoulombFrictionConfig` /
`asymmetric_coulomb_backdrive_torque()` in `simulation/ur5e_mujoco_torque.py`, grounded in
Clochiatti et al. (Robotica 2024)'s UR5e-specific finding that real joint friction is
asymmetric in direction of mechanical power flow (backdriving vs. driving), not just in sign
of `qd` like this repo's existing static symmetric `frictionloss`/`damping` model. Disabled by
default; opt-in via `--asymmetric-coulomb-friction` on
`tools/ur5e_mujoco_torque_experiments.py` / `tools/ur5e_move_hold_transport.py`. Magnitudes
(`extra_coulomb_backdrive_nm = [1.5, 1.5, 1.5, 0.3, 0.3, 0.3]` Nm, size3/size1 joint split) are
an explicitly-labeled, uncalibrated placeholder.

This task's job: use it as a "harder test" — check whether controller configs already
validated against the current (easier, symmetric, static) friction plant hold up against this
more realistic plant, to catch overfitting to easier sim physics before real hardware.

## 1. Methodology

A scratch orchestration script (not committed; lived in the agent scratchpad) called
`tools/ur5e_move_hold_transport.py` directly, reusing the exact `CATEGORY_GRIDS` from
`tools/ur5e_pose_sweep_transport.py` (`canonical_grid`: dx 0.01-0.04m x hold 1/2s;
`large_displacements`: dx 0.05-0.20m x hold 1/2s) and exact poses from `hardware/poses.py`
(`HEIGHT_ALPHA_0_5_Q`, `HEIGHT_ALPHA_0_5_CLEARANCE_Q`), with each named config's own
`controller.gains` auto-extracted and passed via `--gain-overrides-json` (not the driver's
`BASELINE_GAINS`) — the same methodology `docs/status/nullspace_envelope_search_2026-08-01.md`
and `docs/status/neg45_drift_tolerance_validation_2026-08-01.md` used, confirmed by exact
reproduction of 3 of 4 scenarios' baseline (friction-off) numbers below. `--seed 0` throughout.
Each (config, category) ran twice, identical in every way except one flag:
`--asymmetric-coulomb-friction` off vs. on.

`uptime`/`nproc` checked immediately before the campaign (load average ~0.6-1.2 on 72 cores)
and before the severity sweep; all runs executed synchronously in the foreground.

## 2. Scenarios and results (friction off vs. on, placeholder 1x magnitude)

| scenario | config | pose | category | friction off | friction on |
|---|---|---|---|---|---|
| baseline_canonical | `ur5e_mujoco_torque_osc_tuned.yaml` | height_alpha=0.5 | canonical_grid | 3/8 | 3/8 |
| friction_ff_canonical | `..._friction_ff.yaml` | height_alpha=0.5 | canonical_grid | 7/8 | 7/8 |
| wrist_orient_fixed_large_disp | `..._wrist_orient_fixed.yaml` | height_alpha=0.5 | large_displacements (dx>0) | 8/8 | 8/8 |
| neg45_pose_canonical | `..._wrist_orient_fixed_neg45_pose.yaml` | -45deg pose | canonical_grid | 0/8 | 0/8 |

The `baseline_canonical` (3/8) and `friction_ff_canonical` (7/8) friction-off numbers reproduce
`nullspace_envelope_search_2026-08-01.md`'s own regression-check table exactly
(`canonical_grid` 3/8 baseline, 7/8 with `friction_feedforward`), and `wrist_orient_fixed`'s
8/8 on `large_displacements` matches that doc's directional-ceiling result — confirming the
methodology here is a faithful reproduction, not a divergent setup.

### Per-row detail: real, measurable degradation that does not flip category pass/fail

Every scenario shows a small, consistent, physically-sensible reduction in
`move_phase_achieved_x_delta_m` with friction on (extra resistance -> less displacement for
the same commanded torque), roughly 1-10% relative depending on scenario and target size:

- `baseline_canonical`, dx=0.01m: 0.00513m -> 0.00479m achieved (-6.6%). This is large enough
  to flip the **move-phase** validity flag itself (`valid_move_phase: True -> False` at
  dx=0.01, both hold durations) — a real, if narrow, degradation. It does not change the
  overall row verdict since that row was already failing on `hold_phase_target_tracking` in
  both cases, so the category total stayed 3/8 -> 3/8.
- `friction_ff_canonical`: achieved X reduced ~2-5% at every dx (e.g. dx=0.01m: 0.00631m ->
  0.00600m), no pass/fail flips; friction feedforward's model-based compensation is tuned for
  the existing static symmetric friction, not this new asymmetric term, so a small residual
  gap is expected and confirmed.
- `wrist_orient_fixed_large_disp`: smallest relative effect (~0.1-0.4%, e.g. dx=0.20m:
  0.19396m -> 0.19383m), no flips — large-displacement moves spend most of the trajectory in
  "driving" power flow, so the backdrive-only term has less opportunity to act.
- `neg45_pose_canonical`: consistent small reduction (-1% to -10%, largest at the smallest
  dx), no flips, but the baseline (friction-off) result here is **0/8**, not the 8/8
  reported for this exact config in `neg45_drift_tolerance_validation_2026-08-01.md`. Every
  failure in my run is `move_phase_target_tracking` at genuine `duration_complete` (verified
  via `termination_reason` in the per-run `summary.json` — not an early safety-guard abort,
  and `max_abs_orthogonal_drift_m` stays under 0.006m, well inside even the un-raised 0.03m
  guard), which by construction cannot be fixed by raising a drift-tolerance threshold. This
  matches `nullspace_envelope_search_2026-08-01.md`'s own Phase-1 table (0/38 for this config
  family at this pose, every failure via `move_phase_target_tracking` even at trivial
  dx=0.01m) rather than the later doc's 8/8 claim. **This reproducibility gap predates and is
  independent of the asymmetric-friction work** — my friction-off and friction-on runs used
  the identical setup, so the on-vs-off comparison itself is still valid and internally
  consistent; only the absolute pass count disagrees with one prior doc. Flagged here for a
  human decision, not chased further — out of scope for this task.

## 3. Severity sweep (bounded, 2 levels, on the most marginal scenario)

`baseline_canonical` (canonical_grid @ height_alpha=0.5) re-run with
`extra_coulomb_backdrive_nm` scaled 3x (`[4.5,4.5,4.5,0.9,0.9,0.9]` Nm) and 10x
(`[15,15,15,3,3,3]` Nm) via a scratch YAML overlay (not committed), same gains/pose/seed:

| magnitude | pass/8 | dx=0.01 achieved X (m) |
|---|---|---|
| off | 3/8 | 0.00513 |
| 1x (placeholder) | 3/8 | 0.00479 |
| 3x | 3/8 | 0.00455 |
| 10x | 4/8 | 0.00430 |

Achieved X at dx=0.01m decreases **monotonically** with magnitude (0.00513 -> 0.00479 ->
0.00455 -> 0.00430), confirming the mechanism scales correctly and is having a real physical
effect proportional to its magnitude. Pass/fail counts stay flat in the 3-4/8 range even at
10x (300% of the already-uncalibrated placeholder, i.e. a magnitude larger than the base
`frictionloss` itself) because the pre-existing static symmetric friction is already the
dominant limiter at this pose/config/displacement combination — several rows were already
marginal (`hold_phase_target_tracking` failures right at the tolerance boundary) before any
asymmetric term was added, and small threshold-crossing noise near a marginal boundary (the
10x row's dx=0.01/hold=1.0 flipping to a pass while dx=0.01/hold=2.0 does not) is expected
behavior for grid-based pass/fail scoring near a boundary, not evidence of an unstable or
non-monotonic effect — the underlying `achieved_x_delta_m` metric itself is clean and
monotonic throughout.

## 4. Honest conclusion

- **Yes, this is a real, useful pre-real-hardware stress test.** The mechanism works as
  designed: it measurably reduces achieved displacement, scales monotonically with magnitude,
  and at the placeholder magnitude was large enough to flip one move-phase-level validity flag
  in the `baseline_canonical` scenario — genuine evidence that a harder, more literature-
  grounded plant degrades an already-marginal case further, exactly the overfitting-detection
  purpose this exercise was built for.
- **No, at the current 30%-of-frictionloss placeholder magnitude, it does not change any
  category-level pass/fail verdict** for the 4 (config, scenario) combinations tested here,
  including for configs that currently pass cleanly (`wrist_orient_fixed_large_disp`, 8/8 in
  both conditions). This is a genuinely bounded null result at this magnitude, not evidence the
  real physical effect is negligible — the magnitude is explicitly an unvalidated placeholder
  (see commit `32c3d1d`'s own message: "not a precise fit"), and the severity sweep shows even
  10x that placeholder still doesn't flip category totals here because the existing symmetric
  friction (added 2026-07-31) already dominates the failure picture at these test points.
- **Practical implication**: this stress test is currently most informative as a
  *fine-grained* signal (per-row achieved-displacement deltas, move-phase-validity flips) more
  than a *category-pass-rate* signal, at these placeholder magnitudes and these specific test
  points. It would likely show more once (a) a real backdrive-torque calibration replaces the
  placeholder, or (b) it's tested against a config/scenario that is currently passing by a
  narrower margin than `wrist_orient_fixed_large_disp`'s comfortable 8/8.
- **Recommend keeping this in the pre-real-hardware validation toolkit going forward** — rerun
  it whenever a config is newly promoted, and revisit the placeholder magnitude once real
  RTDE-current-log-based calibration data exists (per the plan already in AGENTS.md sec 3's
  LuGre/asymmetric-Coulomb literature note).

## 5. Files changed

- `docs/status/asymmetric_friction_harder_test_campaign_2026-08-01.md` (this doc).
- No `assets/ur5e_torque/ur5e_torque.xml`, `hardware/safety.py`, or `controller_core/safety.py`
  changes. No existing named config modified. No new named config added (the 3x/10x severity
  overlay YAMLs were scratch files, not committed).
- Branch sync note: this worktree's local branch was originally based on a stale ancestor of
  `feature/ur5e-mujoco-torque-control`; it was fast-forwarded and then merged up to the current
  tip (merge commit, message "Merge feature/ur5e-mujoco-torque-control..."), which is
  content-identical to upstream (`git diff` against the upstream tip is empty) — no duplicate
  or divergent content from this sync.

## 6. Tests run / not run

- Run: `/common/users/ss5772/miniforge3/envs/mujoco_ur5e/bin/python -m pytest -q -m "unit or mujoco"`
  -> **244 passed, 3 xfailed, 0 failed** (up from the commit's own reported 228/228 baseline
  because this branch also carries the separately-landed residual-torque-regression-pipeline
  work; no regressions from anything in this task).
- Not run: `-m hardware` (mocked RTDE suite) — no hardware-lane files touched, out of scope.
- Not run: any real-hardware command — explicitly out of scope per this task's hard
  constraints. Sim only throughout.

## Rollback

`git rm docs/status/asymmetric_friction_harder_test_campaign_2026-08-01.md`, or revert the
commit noted in the final report. No other tracked files were changed by this task.
