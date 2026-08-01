# Nullspace/orientation-holding envelope search: wrist_orientation_task vs the two known OSC failures

**Status:** Real, partial win. `wrist_orientation_task` (+ the already-promoted `jacobian_singular_cond_max`
fix) **fully resolves the directional-ceiling failure** (AGENTS.md sec 3, "The ceiling is
directional") with zero regressions anywhere tested. It has **zero measurable effect** on the
separate -45 degree base-rotation Y-drift failure — confirmed not just for the plain flag, but
for three additional combinations (wrench-shaping diagonalization, nullspace-projector adaptive
regularization, and both together) tried in a bounded Phase-2 beam search, all of which fail
identically. No `controller_core/` changes were made or needed. Sim-only throughout; nothing here
has been run on real hardware.

**Date:** 2026-08-01.

## 0. Context

Two known, real, unfixed failures share a root cause per AGENTS.md sec 3: the default controller
(`config/ur5e_mujoco_torque_osc_tuned.yaml`) holds orientation only via a nullspace-projected
posture PD term (`kp_rot=0`, damping-only), and that projector's restoring authority is
asymmetric with pose/direction. `config/ur5e_mujoco_torque_osc_tuned_wrist_orient.yaml` carries a
different, already-implemented orientation mechanism (`wrist_orientation_task`: a dedicated
joint-space PD term routed through the wrist joints only, structurally isolated from the shared
Lambda-weighted wrench pipeline) that had never been validated against either failure, and still
carried the class-default `jacobian_singular_cond_max` (1e5) rather than the promoted fix (1e18)
that eliminates a documented freeze at the transport singularity. This task's job: test the fixed
variant against both failures before inventing anything new.

**A confound found immediately and controlled for throughout:** real joint friction was added to
`assets/ur5e_torque/ur5e_torque.xml` on 2026-07-31 (AGENTS.md sec 3), which roughly halves pass
rates at height_alpha=0.5 for every config that doesn't compensate for it. All comparisons below
run baseline and candidate against the *same* current (friction-including) model, so the
comparison itself is fair, but several categories below are dominated by friction-caused
`*_target_tracking` failures rather than the orientation/Y-drift mechanisms this task targets.
Where that matters, results are also shown with `friction_feedforward: true` added (the already
existing, already-validated fix for that confound) to isolate the real signal.

## 1. What was added

- `config/ur5e_mujoco_torque_osc_tuned_wrist_orient_fixed.yaml` — Phase 1 deliverable:
  `config/ur5e_mujoco_torque_osc_tuned_wrist_orient.yaml` + `jacobian_singular_cond_max: 1.0e18`,
  nothing else changed. Functionally identical to the already-existing
  `..._wrist_orient_no_singular_scale.yaml` (added 2026-07-30 for a different validation pass);
  this file exists under this name because it's what this task was asked to produce, and its own
  header cross-references that earlier file rather than duplicating the freeze-mechanism writeup.
- `config/ur5e_mujoco_torque_osc_tuned_wrist_orient_friction_ff.yaml` — diagnostic-only combo
  (wrist_orient_fixed + the already-validated `friction_feedforward`) used to separate the
  friction-tracking confound from the orientation/Y-drift mechanisms under test.
- Three Phase-2 beam-search candidates, each `wrist_orient_fixed` + one/both of the two other
  existing, orthogonal OSC leak-fixes (never combined with `wrist_orientation_task` before):
  `config/ur5e_mujoco_torque_osc_tuned_wrist_orient_fixed_diag_lambda.yaml` (+
  `lambda_diagonal_shaping`), `..._adaptive_lambda.yaml` (+ `lambda_adaptive_regularization`),
  `..._diag_adaptive_lambda.yaml` (both).
- No `controller_core/` changes. No existing config modified. All new configs preserve the
  project's "add, never mutate" rule.

Every sweep below used
`/common/users/ss5772/miniforge3/envs/mujoco_ur5e/bin/python tools/ur5e_move_hold_transport.py`
directly (the child driver `tools/ur5e_pose_sweep_transport.py` subprocesses), driven by a scratch
orchestration script (not committed — lives in the agent scratchpad) that reuses
`tools/ur5e_pose_sweep_transport.py`'s `CATEGORY_GRIDS`/`build_move_hold_command` so grids are
byte-identical to the project's existing rigor-sweep methodology; it adds support for an arbitrary
`--start-q-rad` (needed for the -45 pose, which `ur5e_pose_sweep_transport.py` itself does not
expose) and for negating `target_x_deltas` (needed for the directional-ceiling case). `--seed 0`
throughout; each config's own gains auto-extracted and passed via `--gain-overrides-json` (not the
child driver's `BASELINE_GAINS`).

## 2. Phase 1: -45 degree pose (`hardware.poses.HEIGHT_ALPHA_0_5_CLEARANCE_Q`)

Full 4-category rigor sweep (canonical_grid, long_holds, large_displacements,
torque_scale_robustness — 38 runs), current model (friction included):

| config | canonical_grid | long_holds | large_displacements | torque_scale_robustness | total |
|---|---|---|---|---|---|
| `ur5e_mujoco_torque_osc_tuned` (baseline) | 0/8 | 0/8 | 0/8 | 0/14 | **0/38** |
| `..._wrist_orient_fixed` (candidate) | 0/8 | 0/8 | 0/8 | 0/14 | **0/38** |
| baseline + `friction_feedforward` | 0/8 | 0/8 | 0/8 | 0/14 | **0/38** |
| candidate + `friction_feedforward` | 0/8 | 0/8 | 0/8 | 0/14 | **0/38** |

**No improvement from the candidate in any configuration, with or without the friction
confound removed.** Every failure at dx >= 0.04-0.06m is a genuine guard trip
(`|Y-Y0| > 0.03 m`), saturating at 0.0300-0.0303m regardless of which of the 4 configs above is
used — matching `docs/status/base_rotation_neg45_retune_2026-07-31.md`'s prior documented
signature almost exactly, just worse overall (that doc's frozen pre-friction baseline scored
18/38; the current friction-including model scores 0/38 for every config tried here, including
at trivial displacements like dx=0.01m, via `move_phase_target_tracking` — friction alone does
not explain this: the same friction-including model recovers 7/8 on `canonical_grid` at the
un-rotated height_alpha=0.5 pose with `friction_feedforward` on (see sec 3), so the -45 pose
degrades far more than friction alone predicts, additional evidence for AGENTS.md's structural-
kinematic-effect hypothesis).

`wrist_orientation_task`'s only measurable effect at this pose: it lowers *orientation* error at
the failing canonical-grid cases (e.g. dx=0.03m: baseline+ff 0.0470/0.0480 rad vs candidate+ff
0.0310/0.0320 rad, a ~34% reduction) — but this never flips pass/fail, because **Y-drift, not
orientation, is what gates failure here**, and the candidate's mechanism doesn't touch Y at all.

## 3. Phase 1: directional-ceiling case (height_alpha=0.5, `hardware.poses.HEIGHT_ALPHA_0_5_Q`)

### Large displacements, negative X (dx -0.05 to -0.20m, hold 1/2s) — the documented failure

| config | pass/8 | detail |
|---|---|---|
| baseline | 6/8 | fails dx=-0.20m both hold durations, `\|\|orientation error\|\| > 0.25 rad` (0.2497 rad — at the ceiling) |
| **candidate** | **8/8** | **fixes both** — orientation error at dx=-0.20m drops to **0.125-0.127 rad, essentially half** |
| baseline + friction_ff | 6/8 | identical failure (0.2499 rad) |
| candidate + friction_ff | 8/8 | fixes both |

### Large displacements, positive X (dx 0.05-0.20m, hold 1/2s) — symmetric check

| config | pass/8 | detail |
|---|---|---|
| baseline | 6/8 | fails dx=+0.20m both hold durations, orientation 0.2499 rad |
| **candidate** | **8/8** | fixes both |

**This is a real, clean structural fix for the documented directional-ceiling failure, in both
directions, with and without the friction confound.**

### Regression checks (same pose, other categories) — byte-identical, zero regressions

| category | baseline | candidate |
|---|---|---|
| canonical_grid (dx 0.01-0.04m) | 3/8 | 3/8 |
| canonical_grid + friction_ff | 7/8 | 7/8 |
| long_holds (dx 0.03/0.06m, hold 4-30s) | 4/8 | 4/8 |
| torque_scale_robustness | 6/14 | 6/14 |

All remaining failures in these three rows are `*_target_tracking` (friction undershoot) or
torque-budget saturation at `scale=0.1` — unrelated to orientation, confirmed identical between
baseline and candidate (same failure reason, same magnitude to 3-4 significant figures).

### Generalization check at two more heights (no gain retuning attempted or needed)

`height_alpha` in {0.2, 0.35}, canonical_grid + large_displacements (positive dx):

| height_alpha | category | baseline | candidate |
|---|---|---|---|
| 0.2 | canonical_grid | 3/8 | 3/8 |
| 0.2 | large_displacements | 8/8 | 8/8 |
| 0.35 | canonical_grid | 3/8 | 3/8 |
| 0.35 | large_displacements | 8/8 | 8/8 |

Byte-identical at every point — the fix generalizes cleanly using the config's existing gains
(same `kp_x`/`kd_x`/... as the already-validated tuned baseline); **no per-height gain retuning
was needed**, so Phase 3's "light per-height tuning" step is not applicable here beyond this
confirmation.

## 4. Phase 2: bounded beam search for the -45 degree failure

Since Phase 1 only partially resolved the two known failures (directional ceiling: fixed;
-45 degree Y-drift: untouched), Phase 2 ran per the task's instructions — but scoped to the
**existing** orthogonal OSC-leak fixes the task itself suggested trying first
(`lambda_diagonal_shaping`, `lambda_adaptive_regularization`), rather than inventing new
`controller_core/` mechanisms, since neither had been tested combined with
`wrist_orientation_task` before.

**Round 1** (3 candidates + the `wrist_orient_fixed` control, trimmed grid dx =
{0.02, 0.06, 0.10, 0.20}m, hold=2.0s, -45 pose):

| config | dx=0.02 | dx=0.06 | dx=0.10 | dx=0.20 | max\|orientation\| at failures |
|---|---|---|---|---|---|
| wrist_orient_fixed (control) | fail (tracking) | fail, Y=0.0301 | fail, Y=0.0303 | fail, Y=0.0302 | 0.026-0.037 rad |
| + lambda_diagonal_shaping | fail (tracking) | fail, Y=0.0300 | fail, Y=0.0301 | fail, Y=0.0302 | 0.013-0.015 rad |
| + lambda_adaptive_regularization | fail (tracking) | fail, Y=0.0301 | fail, Y=0.0303 | fail, Y=0.0300 | 0.025-0.036 rad |
| + both | fail (tracking) | fail, Y=0.0300 | fail, Y=0.0302 | fail, Y=0.0304 | 0.012-0.014 rad |

**0/4 for every candidate — zero cleared the safety hard filter (every one still trips
`|Y-Y0|>0.03m` at every dx tried, at statistically identical magnitude to the control).**
`lambda_diagonal_shaping` again lowers orientation error substantially (as it did at the
un-rotated pose's original motivating case) but the Y-drift magnitude itself is unmoved to 3
significant figures across all 4 configs — direct, quantitative confirmation that the -45 degree
failure is a **task-space Y-axis phenomenon** (`Fy = kp_y*y_err - kd_y*v_y`, a full closed-loop
term, not nullspace-projected), not an orientation/nullspace/Lambda-coupling issue at all — the
entire family of mechanisms this controller currently has for orientation-holding targets the
wrong axis for this specific failure.

**Beam search stopped after round 1** per the task's own stopping rule ("no improvement"): since
zero candidates passed the hard safety filter, there is nothing to rank or expand, and the
uniform, mechanism-consistent non-response across three structurally distinct fixes
(wrist-joint PD, wrench-shaping diagonalization, nullspace-projector regularization) is strong
evidence against the entire orientation/nullspace/Lambda family, not just against these specific
gain values — expanding further within the same family was judged very unlikely to find a real
fix. This adds real new evidence beyond `docs/status/base_rotation_neg45_retune_2026-07-31.md`'s
prior negative gain-search result: not just x/y/z/rot/posture/kd_joint gains fail to fix this, but
now three additional, orthogonal, already-implemented orientation/nullspace mechanisms fail
identically too. A real fix, if one exists within this controller architecture, most plausibly
needs a genuinely different mechanism (e.g. explicit Jacobian-based X-Y decoupling feedforward on
the Y task-space term itself) — out of scope for this task's constraint against speculative new
`controller_core/` mechanisms without stronger evidence they'd help.

## 5. Phase 3

Not applicable to the -45 degree failure (nothing in Phase 1 or 2 worked, so there is no winning
candidate to tune for that failure). For the directional-ceiling winner
(`wrist_orient_fixed`), Phase 3's own instruction ("don't gain-tune something that doesn't
structurally fix the problem... light tuning for the winning config only") is satisfied by the
generalization check in sec 3 above: default gains (unchanged from the already-validated tuned
baseline) reproduce the fix cleanly at height_alpha in {0.2, 0.35} in addition to 0.5, with zero
regressions — no retuning was needed or attempted.

## 6. Test suite

`/common/users/ss5772/miniforge3/envs/mujoco_ur5e/bin/python -m pytest -q -m "unit or mujoco"`:
**228 passed, 243 deselected, 3 xfailed** — no regressions. Expected: this task made no
`controller_core/` changes and did not modify any existing config; all changes are new, additive
YAML files.

## 7. Recommendation

- **Promote-worthy on its own merits**: `config/ur5e_mujoco_torque_osc_tuned_wrist_orient_fixed.yaml`
  is a genuine improvement over the plain tuned baseline for the directional-ceiling failure
  (8/8 vs 6/8 at height_alpha=0.5, both directions, orientation error roughly halved at the worst
  case) with zero regressions found anywhere in this investigation (5 categories x up to 3 heights
  x with/without friction feedforward). It is **not** a fix for the -45 degree pose — do not treat
  it as a general replacement for the real-hardware default
  (`config/ur5e_mujoco_torque_osc_tuned.yaml`, which is unmodified) without a separate decision
  about the -45 degree gap, since that pose is the actual current real-hardware default start pose
  per `hardware/poses.py`.
- **-45 degree Y-drift failure remains open.** This investigation narrows the search space
  meaningfully (rules out the entire orientation/nullspace/Lambda-coupling mechanism family, not
  just specific gains) but does not fix it. Recommend a human decision before any further
  controller-design work on this specific failure, consistent with the prior doc's own
  recommendation.
- **Not validated on real hardware.** Everything in this doc is simulation-only. The friction
  model landing mid-investigation (2026-07-31) is a reminder that this repo's sim and real
  behavior can diverge in magnitude (though not in mechanism, per the neg45 doc's own honest
  caveat about differing dx onset) — real validation of `wrist_orient_fixed` at the
  directional-ceiling case is a reasonable next real-lab step, but only after an explicit decision
  to do so, and never inferred as "safe" from this doc alone.

## 8. Files changed

- `config/ur5e_mujoco_torque_osc_tuned_wrist_orient_fixed.yaml` (new)
- `config/ur5e_mujoco_torque_osc_tuned_wrist_orient_friction_ff.yaml` (new, diagnostic)
- `config/ur5e_mujoco_torque_osc_tuned_wrist_orient_fixed_diag_lambda.yaml` (new, Phase-2 candidate)
- `config/ur5e_mujoco_torque_osc_tuned_wrist_orient_fixed_adaptive_lambda.yaml` (new, Phase-2 candidate)
- `config/ur5e_mujoco_torque_osc_tuned_wrist_orient_fixed_diag_adaptive_lambda.yaml` (new, Phase-2 candidate)
- `docs/status/nullspace_envelope_search_2026-08-01.md` (this doc)
- No `controller_core/` changes. No existing config modified.

Raw sweep outputs (gitignored, not git-recoverable):
`outputs/ur5e_mujoco_torque_transport/nullspace_envelope_search_2026-08-01/` (per-category
`summary.json`/`run_log.csv`/`per_run_traces/`, plus `aggregate_report.json` with every run's
pass/fail + failure detail collected by the orchestration script).

## Tests run / not run

- Run: `pytest -q -m "unit or mujoco"` — 228 passed, 3 xfailed, 0 failed.
- Not run: `-m hardware` (mocked RTDE suite) — no hardware-lane files touched, out of scope.
- Not run: any real-hardware command — explicitly out of scope per this task's hard constraints.

## Rollback

`git rm config/ur5e_mujoco_torque_osc_tuned_wrist_orient_fixed*.yaml
config/ur5e_mujoco_torque_osc_tuned_wrist_orient_friction_ff.yaml
docs/status/nullspace_envelope_search_2026-08-01.md` (or revert the commit noted in the final
report). No other files were modified.
