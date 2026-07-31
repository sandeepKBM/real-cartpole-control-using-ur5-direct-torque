# Adding real joint friction to the UR5e torque MJCF

Context: real UR5e hardware testing on 2026-07-31 showed commanded torque of 5-8 Nm producing
far less real displacement than sim predicted, with steady-state hold-phase torque **not**
decaying toward zero -- a force-balance-against-friction signature. `assets/ur5e_torque/
ur5e_torque.xml`'s joint `<default>` class had `armature="0.1"` but no `frictionloss` or
`damping` anywhere -- real joint friction was completely unmodeled. Full real-session
writeup: `docs/status/real_lab_session_2026-07-31.md` sec 2 (context only).

This is a physics-model asset change only. No friction-compensation logic was added to
`controller_core/`, no gain config was modified, no safety threshold was touched, and no
hardware code was run. Confirmed before making any change that nothing in the existing torque
path (`controller_core/x_axis_cartesian_impedance.py`,
`simulation/ur5e_mujoco_torque.py::MujocoUR5eTorqueAdapter.apply_torque_components`) already
compensates for friction, so adding it to the physics model does not double-count anything.

## 1. Friction values chosen and sourcing

Added to `assets/ur5e_torque/ur5e_torque.xml`'s `size3`/`size1` joint default classes (not
per-joint instances -- see "Two existing validators" below for why):

| joint class | joints | rated torque | `frictionloss` (Coulomb, Nm) | `damping` (viscous, Nm*s/rad) |
|---|---|---|---|---|
| `size3` | shoulder_pan, shoulder_lift, elbow | 150 Nm | 5.0 | 0.4 |
| `size1` | wrist_1, wrist_2, wrist_3 | 28 Nm | 1.0 | 0.15 |

**Sourcing, reported honestly:**
- The task brief suggested the upstream `vendor/mujoco_menagerie/universal_robots_ur5e/ur5e.xml`
  "likely already has real manufacturer-informed friction values." Checked directly: it does
  **not**. It uses `<general>` position-PD actuators (`gaintype=fixed, biastype=affine`, i.e. a
  built-in stiff PD position controller), not direct torque motors, so it has no need for
  explicit friction modeling. This repo's custom torque model has no equivalent reference to
  diff against.
- Searched for published UR5/UR5e joint friction identification values (WebSearch). Found
  several academic dynamic-parameter-identification papers (Coulomb+viscous friction models)
  but no single table I could cite with confidence as an authoritative per-joint Nm value --
  reported as a real limitation rather than fabricating a precise citation.
- Confirmed via search that this repo's existing actuator torque limits (150 Nm size3 / 28 Nm
  size1, both already present in `ur5e_torque.xml` before this change) match real UR5e
  per-joint rated torque. Used that as the grounding basis: `frictionloss` set to ~3.3-3.6% of
  each class's rated torque, an order-of-magnitude consistent with commonly-cited harmonic-drive
  robot joint friction (typically a few percent of rated torque in the literature found).
  `damping` set conservatively below that.
- **This is a reasonable engineering estimate, not a calibrated fit to the real robot.** The
  task's own instructions explicitly warned against overfitting to the one real data point from
  tonight (dx=0.04m: 55%/72% displacement achievement at kp_x=400/600) -- that instruction was
  followed; no attempt was made to hit those numbers.

**Two existing validators constrained *how* friction could be added, and were treated as
binding, not incidental:**
- `mujoco_ur5e_tools.py::validate_ur5e_torque_xml_source_tree` explicitly rejects nonzero
  `damping`/`stiffness` set directly on a **named joint instance** ("this would add hidden
  passive hold behavior") -- checked via literal `<joint name=.../>` attributes, not MuJoCo
  `<default class>` inheritance.
- `mujoco_ur5e_tools.py::validate_compiled_ur5e_torque_model` (mirrored in
  `tests/mujoco/test_ur5e_mujoco_torque.py`) explicitly **allows** compiled `dof_damping` up to
  `0.5` Nm*s/rad ("suspiciously high" ceiling) -- strong evidence a small amount of
  default-class-level damping was anticipated as legitimate.

Resolution: friction set via `<default class="size3">` / `<default class="size1">` rather than
per-instance. `frictionloss` is unconstrained by either validator. `damping` (0.4 / 0.15) stays
comfortably under the 0.5 compiled ceiling. Model compiles cleanly;
`tests/mujoco/test_ur5e_mujoco_torque.py` (27 tests) passes unchanged.

## 2. Smoke test: does friction qualitatively reproduce the real signature?

A/B comparison, dx=0.04m move-hold, `config/ur5e_mujoco_torque_osc_tuned.yaml` (unchanged,
current default gains: kp_x=400 etc.), move=1.0s, hold=2.0s, seed=0. Frictionless side obtained
by temporarily reverting `ur5e_torque.xml` to `git show HEAD:...` for one run, then restoring
the friction-enabled version (verified restored via `git diff --stat`).

| metric | frictionless | with friction | change |
|---|---|---|---|
| `achieved_x_delta_m` (target 0.04) | 0.03998 (99.95%) | 0.03974 (99.35%) | worse |
| `final_x_error_m` | 1.93e-5 m | 2.57e-4 m | **13x larger** |
| `move_tracking_score` | 0.848 | 0.362 | much worse |
| hold-phase controller torque (L1 over 6 joints) at hold-start (t=1.0s) | 0.34 Nm | 6.10 Nm | 18x larger |
| ...same, 0.5s into hold (t=1.5s) | ~0.001 Nm (already ~settled) | 2.86 Nm | still far from settled |
| ...same, end of 2s hold (t=3.0s) | ~0.0000 Nm | 0.36 Nm | **not yet near-zero** |

**Result: yes, qualitatively reproduced.** Frictionless hold-phase torque decays to the
double-precision noise floor within ~0.5s (classic clean exponential settle). With friction,
torque decays far more slowly and is still substantial (0.36 Nm) at the end of a 2s hold --
directly analogous to the real robot's "steady-state torque does not decay toward zero."
Displacement achievement is measurably worse with friction but not dramatically so (99.35% vs
55-72% on the real robot) -- expected and not a target: the closed-loop OSC controller's
stiffness (`kp_x=400`) crushes most of a several-Nm friction budget against a 150 Nm torque
ceiling on the big joints. Matching the real percentage would require either much larger
friction values (risking overfitting to one under-determined trial, explicitly against this
task's instructions) or a different disturbance-rejection mechanism (feedforward/integral
action) that only a controller-side change could add -- out of scope here by design (see "Hard
constraints" in the task and the "concurrent activity" note below, which found exactly this
already being explored elsewhere).

**Long-hold check (does it truly plateau, or just decay slowly?)**: examined a 30s hold
(`dx=0.06m`, from the sweep below) at 2s intervals. Controller torque (L1) goes 1.86 Nm (t=2s)
-> 0.30 Nm (t=6s) -> 0.13 Nm (t=16s) -> 0.038 Nm (t=30s) while `x_error` is already ~1e-5 m by
t=6s. **Correction to a earlier working hypothesis**: this is not a hard, unbounded plateau --
the stiff `kp_x=400` closed loop is strong enough to keep making very small corrections even
against Coulomb friction here (a "creeping" convergence, consistent with dry-friction PD control
theory), so it does continue to decay. But it decays roughly an order of magnitude slower than
the frictionless case and has **not** reached the frictionless near-zero floor even after 30s of
holding -- a real, large, qualitatively distinct signature, just not literally infinite.

## 3. Four-category rigor sweep: does friction break the existing validated envelope?

Ran `tools/ur5e_pose_sweep_transport.py --height-alphas 0.5 --config
config/ur5e_mujoco_torque_osc_tuned.yaml --seed 0` (all four categories, 38 runs) -- the exact
methodology `docs/status/disable_global_singular_scale_validation_2026-07-30.md` used to
establish the documented frictionless baseline at `height_alpha=0.5`. Config unchanged
throughout (current promoted default, `jacobian_singular_cond_max=1.0e18`, i.e. the
singular_scale freeze bug already fixed at this pose -- see sec 4 for why that matters).

| category | frictionless (documented baseline) | **with friction** | regression? |
|---|---|---|---|
| canonical_grid | 8/8 | **3/8** | yes, large |
| long_holds | 8/8 | **4/8** | yes, large |
| large_displacements | 8/8 | **6/8** | yes |
| torque_scale_robustness | 12/14 | **6/14** | yes, large |
| **total** | **36/38 (94.7%)** | **19/38 (50.0%)** | **yes -- pass rate roughly halved** |

**This is a real, substantial regression, not noise.** Per the task's hard constraints, this is
reported here as a finding requiring a deliberate gain-retuning decision -- **no gain retuning
was performed as part of this task.** The default config
(`config/ur5e_mujoco_torque_osc_tuned.yaml`) was validated and tuned entirely against a
frictionless model; that tuning no longer holds once real joint friction is present. This is the
single most important number in this document: adding physically-realistic friction, with the
gains left exactly as validated, roughly halves the pass rate of the previously-established
"safe" envelope at `height_alpha=0.5`. Most failures are `move_phase_target_tracking` /
`hold_phase_target_tracking` (the controller doesn't reach/hold the target within the existing
tolerance windows in the existing move/hold durations) -- consistent with sec 2's smoke-test
finding that friction slows convergence without necessarily preventing it given more time.

Scope note: only `height_alpha=0.5` was run (not the full `{0.1, 0.2, 0.3, 0.5}` used in the
2026-07-30 singular_scale validation) -- chosen to keep this synchronous run bounded; the
qualitative conclusion (friction meaningfully degrades the validated envelope under
unchanged gains) is unlikely to reverse at other poses, but that is not separately confirmed
here.

### 3.1 Addendum (same day, after the above landed): the regression is largely fixable, and not by gain retuning

A separate, concurrent effort (not part of this task; see sec 6) independently found and fixed
this exact regression via an opt-in controller-side `friction_feedforward` term (commit
`5eb9778`, `config/ur5e_mujoco_torque_osc_tuned_friction_ff.yaml` -- gains identical to
`config/ur5e_mujoco_torque_osc_tuned.yaml`, only the feedforward term added) rather than
retuning gains. Their own validation, run at `height_alpha in {0.2, 0.3}`
(`docs/status/friction_ff_alpha_0.2_0.3_sweep_2026-07-31.md`), found baseline (no feedforward)
passing only 22/38 and 21/38 there -- closely matching this document's 19/38 at
`height_alpha=0.5`, i.e. an independent confirmation of the same regression at different poses
-- and feedforward closing that to 38/38 and 35/38 (96% combined, never worse than baseline in
any single cell).

That validation did not cover `height_alpha=0.5` (the pose this document's sec 3 regression was
measured at), so it was run here to close the loop, using the same `tools/
ur5e_pose_sweep_transport.py` methodology, `config/ur5e_mujoco_torque_osc_tuned_friction_ff.yaml`
(read-only input, not modified) against the friction-enabled model, `seed=0`:

| category | no feedforward (sec 3) | **with friction_ff** | frictionless (original baseline) |
|---|---|---|---|
| canonical_grid | 3/8 | **7/8** | 8/8 |
| long_holds | 4/8 | **8/8** | 8/8 |
| large_displacements | 6/8 | **6/8** (unchanged) | 8/8 |
| torque_scale_robustness | 6/14 | **12/14** | 12/14 |
| **total** | **19/38 (50.0%)** | **33/38 (86.8%)** | 36/38 (94.7%) |

**Confirms the finding generalizes**: friction_feedforward recovers most of the regression at
this pose too (50.0% -> 86.8%, most of the way back to the original 94.7% frictionless
baseline), consistent with the 0.2/0.3 result. `torque_scale_robustness` and `long_holds` are
fully recovered to their frictionless pass counts. `large_displacements` is the one category
that did **not** improve here (still 6/8, both failures at `dx=0.2m`, either hold duration) --
diagnosed: both fail via an actual safety-guard trip (`termination_reason: "||orientation
error|| > 0.25 rad"`, `max_abs_orientation_error_rad=0.250`, i.e. the 0.25 rad safety ceiling in
this config), not a tracking-tolerance miss like the other categories' failures. `dx=0.2m` at
this pose was already the documented edge of the frictionless envelope (AGENTS.md sec 3: "large
displacements (dx up to 0.20 m) 16/16... dx=0.25 m breaks via Z-drift -- a genuine
workspace/reach limit"), so this may be friction_ff's own added torque interacting with an
already-marginal case rather than a new, independent problem -- not further diagnosed here
(out of scope for this document; flagged for whoever continues the friction_ff validation work,
alongside their own sec 6-referenced Part D real-hardware step).

No file was modified to produce this addendum beyond this document -- `config/
ur5e_mujoco_torque_osc_tuned_friction_ff.yaml` and `controller_core/x_axis_cartesian_
impedance.py` are read-only inputs here, owned by the concurrent effort in sec 6.

## 4. A second, distinct effect: friction makes a pre-existing bug permanent, not just slower

Independent of sec 3's tracking-accuracy regression (which happens even with the fully-fixed
default config), diagnosing 6 pre-existing test failures (sec 5) surfaced a second, mechanistically
different interaction: `height_alpha=0.5`'s pose family (`hardware/poses.py::
q_for_height_alpha`) sits at `wrist_2=0.0` **exactly**, for *every* `height_alpha` in `[0,1]`
(both interpolation endpoints, `ACTIVE_ORIGIN_Q` and `LOWER_B_Q`, share `wrist_2=0.0`) -- a real
kinematic wrist singularity independent of the other joint angles. AGENTS.md sec 4 and
`docs/status/disable_global_singular_scale_validation_2026-07-30.md` already document that any
config still using the class-default `jacobian_singular_cond_max=1.0e5` (singular_scale enabled)
freezes the controller at this exact pose for roughly the first 0.7-0.8s of a move, only
escaping via floating-point integration noise nudging `wrist_2` off exactly zero.

**With frictionloss now in the model, that escape mechanism can stop working entirely.** The
stray torque that floating-point noise used to produce is far smaller than the new static-
friction dead zone (5.0/1.0 Nm by joint class), so it gets fully absorbed: the pose becomes an
exact, deterministic fixed point instead of a slow-but-eventual escape. Measured directly
(`config/rl_gain_scheduling.yaml`, which -- like `config/ur5e_mujoco_torque_osc_tuned_wrist_
orient.yaml` -- has never had the `jacobian_singular_cond_max` fix applied): commanded torque
stays at the ~1e-12 Nm level for **1500 steps (3s)**, vs. the ~300-400 steps it used to take to
escape. This is a real, repo-wide-relevant amplification of an already-known, already-flagged
bug -- **any config that hasn't had the `jacobian_singular_cond_max` fix applied is now at risk
of a permanent freeze, not just a slow one, at this pose family.** This affects at least
`config/ur5e_mujoco_torque_osc_tuned_wrist_orient.yaml` and `config/rl_gain_scheduling.yaml`
(confirmed); other un-migrated configs were not individually audited here. No config was
modified to fix this -- it is squarely the kind of gain/config decision this task was scoped to
flag, not silently patch.

## 5. Six pre-existing tests broken by friction: diagnosed and fixed individually

Running `python -m pytest -q -m "mujoco and not slow"` after adding friction (before any test
changes) showed 6 failures out of 93 collected (87 passed). Each was individually diagnosed
(root cause identified via direct re-measurement, not guessed) and fixed on its own merits --
no blind tolerance bumps. Final state: **441 passed, 2 deselected, 3 xfailed, 0 failed**
(`python -m pytest -q -m "not slow"`, full repo).

| test | root cause | fix |
|---|---|---|
| `test_ur5e_mujoco_torque_experiments_refactor_parity.py::test_controller_rollout_matches_pre_refactor_golden_values` | Hardcoded exact frictionless golden values (`pytest.approx(..., abs=1e-12)`) for a real, deterministic rollout -- legitimately stale now that the underlying physics changed. | Re-ran the exact same command against the friction-enabled model, replaced `EXPECTED`/`EXPECTED_FINAL_Q`/`EXPECTED_FINAL_EE_POS` with the new real measured values (e.g. `achieved_x_delta_m` 0.01012 -> 0.00668, `move_hold_quality_score` 0.612 -> 0.297), documented why in the module docstring following this file's own existing 2026-07-30-refresh convention. |
| `test_direct_torque_residual_observer.py::test_residual_stays_small_on_clean_move_hold_rollout` | The residual observer predicts `qdd` from `PinocchioUR5eDynamics.bias()` -- pure rigid-body dynamics, no friction term by design (mirrors what the real PolyScope path can predict). Friction is now a second, permanent, legitimate residual source: measured settled-tail median/max went from 0.00204/0.00300 to **1.227/1.318 rad/s^2** (~600x), and the overall guard's measured peak went from 2.816 to 13.577 rad/s^2. | Re-measured on the exact rollout with friction; rescaled both bounds using the *same margin ratios* the 2026-07-30 pass used over its own numbers (~3.5x on the loose NaN/blowup guard, ~4.9x on settled median, ~6.7x on settled max) rather than inventing new ratios -- guard now `<50.0` (was `<10.0`), settled median `<6.0` (was `<0.01`), settled max `<9.0` (was `<0.02`). |
| `test_direct_torque_residual_observer.py::test_residual_detects_and_recovers_from_injected_disturbance` | Same friction-raised baseline floor (~1.3 rad/s^2) as above swamps the original 30N disturbance signal: measured `disturbed_peak/baseline_peak` collapsed from the documented ~6498x to **1.22x** -- no longer a meaningful detection margin. | Raised the injected disturbance force from 30N to 60N (still a plausible bump/snag load, ~1.2x the UR5e's rated payload weight, not an implausible collision) to restore a real, clearly-detectable signal: measured `disturbed_peak/baseline_peak=12.6x`, `disturbed_peak/recovered_peak=11.5x`. Thresholds set at 10x/8x, comfortably under the measured values. This is a scenario change to preserve the test's actual intent against the new noise floor, not a threshold bump against the same underpowered signal. |
| `test_wrist_orientation_task.py::test_wrist_orientation_task_reduces_orientation_error_at_height_alpha_0_5` | `config/ur5e_mujoco_torque_osc_tuned_wrist_orient.yaml` has never had the `jacobian_singular_cond_max` fix (AGENTS.md already flags this). Friction converts its pre-existing freeze into target-tracking failure. Confirmed independent of this test's own scenario choice by sweeping `target_x_delta` in `{-0.01...-0.10}` and `move_duration` in `{1.0, 1.5}`: fails `move_phase_target_tracking` at **every** combination (even -0.01m, where orientation error ~1e-14 shows it barely moves at all) -- not a duration/displacement tuning problem. | `pytest.mark.xfail(strict=True, reason=...)` with the full diagnostic trail (see sec 4). No config or gain touched. `strict=True` means this flips to a loud failure (unexpected xpass) the moment the config gets its fix, prompting removal. |
| `test_gain_scheduling_env.py::test_env_set_gains_actually_changes_torque` | Same root cause as sec 4 (`config/rl_gain_scheduling.yaml` un-migrated, `height_alpha=0.0` pose). Measured directly: even at 1500 steps (5x this test's 300-step window), commanded torque for both gain extremes stays at the ~1e-12 to 1e-6 Nm level -- functionally still frozen (a real controller torque here is Newton-meters). Extending the step count would only catch eventual floating-point-noise divergence, which would be a disguised blind bump -- deliberately not done. | `pytest.mark.xfail(strict=True, reason=...)`. No config touched. |
| `test_gain_scheduling_env.py::test_env_full_trace_logging_produces_valid_move_hold_metrics` | Same root cause. Measured directly: `achieved_x_delta_m=2.0e-14` over the full 1500-step/3s episode against a 0.02m target -- genuinely zero motion, not a scoring-threshold quibble. | `pytest.mark.xfail(strict=True, reason=...)`. No config touched. |

## 6. Concurrent activity noted (informational, not acted on)

While this task was in progress, a separate concurrent process committed
`5eb9778 Add opt-in friction feedforward term to the OSC controller` to
`controller_core/x_axis_cartesian_impedance.py` / `controller_core/torque_task_qp.py` (a
`friction_feedforward` config flag, default `False`). This is exactly the kind of
friction-*compensation* controller change this task's own hard constraints explicitly kept out
of scope. It is default-off and does not affect any config used in this document's measurements
(none of them set `friction_feedforward: true`), so it did not contaminate any result here --
confirmed by the fact that all measurements before and after that commit landed were consistent,
and no controller_core file was modified as part of this task. Flagged here only as context: the
sim-to-real friction gap this document addresses on the asset side is apparently also being
addressed on the controller side elsewhere, and sec 3's "roughly halves the validated envelope's
pass rate" finding may already be relevant input to that separate effort.

## 7. Summary

- Friction is real, physically grounded (though not calibrated to the one real hardware trial by
  design), and now present in the sim: `frictionloss`/`damping` on `size3`/`size1` joint default
  classes in `assets/ur5e_torque/ur5e_torque.xml`.
- The smoke test qualitatively reproduces the real sim-to-real signature (elevated, slowly-
  decaying hold-phase torque; measurably worse tracking) without being tuned to match the exact
  55-72% figure, as instructed.
- **Friction alone, with gains unchanged, roughly halves the previously-validated
  `height_alpha=0.5` envelope's pass rate (36/38 -> 19/38).** This needed a deliberate human
  decision (retune vs. accept vs. add controller-side compensation) -- not made in this task.
  **Update (sec 3.1): a concurrent effort resolved it via controller-side friction feedforward,
  not gain retuning** -- 19/38 -> 33/38 (86.8%) at this same pose, closely matching their
  independently-run 96% at `height_alpha` 0.2/0.3. One residual gap found: `large_displacements`
  at `dx=0.2m` still fails via an actual orientation-safety-guard trip, unchanged from sec 3 --
  flagged for that effort's continued validation, not fixed here.
- A second, distinct effect was found and is not a decision this task makes either: friction
  converts an already-documented, already-flagged transient freeze bug (un-migrated
  `jacobian_singular_cond_max`) into a permanent one, for any config that hasn't had that fix.
- All 6 pre-existing tests broken by this change were individually diagnosed and fixed on their
  own merits (2 golden-value refreshes, 1 scenario-preserving threshold recalibration, 3
  diagnosed `xfail`s tied to the out-of-scope singular_scale gap) -- no blind tolerance bumps.
  Full suite: 441 passed, 2 deselected, 3 xfailed, 0 failed.

## Files changed

- `assets/ur5e_torque/ur5e_torque.xml` -- added `frictionloss`/`damping` to `size3`/`size1`
  joint default classes.
- `tests/mujoco/test_ur5e_mujoco_torque_experiments_refactor_parity.py` -- golden values
  refreshed.
- `tests/mujoco/test_direct_torque_residual_observer.py` -- thresholds recalibrated (guard/
  settled-tail bounds via the same margin methodology; disturbance force raised 30N->60N with
  matching threshold recalibration).
- `tests/mujoco/test_wrist_orientation_task.py` -- diagnosed `xfail(strict=True)` added.
- `tests/mujoco/test_gain_scheduling_env.py` -- two diagnosed `xfail(strict=True)` added.
- This document.

## Rollback

`git checkout -- assets/ur5e_torque/ur5e_torque.xml tests/mujoco/test_ur5e_mujoco_torque_experiments_refactor_parity.py tests/mujoco/test_direct_torque_residual_observer.py tests/mujoco/test_wrist_orientation_task.py tests/mujoco/test_gain_scheduling_env.py && rm docs/status/ur5e_sim_friction_modeling_2026-07-31.md`
(reverts to the frictionless model and the pre-existing, un-diagnosed test states).

## Tests run

- `python -m pytest -q tests/mujoco/test_ur5e_mujoco_torque.py -m "not slow"`: 27 passed.
- `python -m pytest -q -m "mujoco and not slow"`: 90 passed, 3 xfailed, 0 failed.
- `python -m pytest -q -m "not slow"` (full repo): 441 passed, 2 deselected, 3 xfailed, 0 failed.

## Tests not run

- `-m slow` tests anywhere in the repo (not run in this task; no reason to expect friction
  interacts differently there, but not confirmed).
- The four-category sweep at `height_alpha` in `{0.1, 0.2, 0.3}` (only `0.5` was run -- see sec
  3 scope note).
- Real hardware / URSim (explicitly out of scope for this task).
