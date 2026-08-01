# -45 deg base-rotation Y-drift: root-cause diagnosis + fix attempt (sim-only)

**Status:** Diagnosis complete, high confidence. **No fix found that clears the hard safety
filter at the documented failure displacement (dx=0.06m).** Three hypotheses tested:
(1) geometric backtracking neutralizing gain increases -- **ruled out**, backtracking is not
active during the failure; (2) a genuine kinematic/dynamic X-Y coupling disturbance needing
more Y-axis authority (P, D, or I) -- **confirmed as real, but "fixing" it via any PID-family
mechanism forces a hard trade-off against X-tracking that itself fails the move-hold tolerance**,
so nothing in this family is a real fix; (3) (raised mid-investigation) the fixed 0.03m guard
threshold may itself be miscalibrated for this pose's genuine kinematics, independent of any
controller defect -- **real supporting evidence found**, reported here as a candidate
explanation for a human decision, not acted on. An opt-in `y_integral_action` mechanism was
implemented, unit-tested, and validated to cause zero regression anywhere else, but it does not
fix this failure either -- kept as a real, honest, negative but non-destructive result.
**Date:** 2026-08-01. Simulation-only throughout; no real-hardware access exists for this task.

## 0. Context

Real hardware and sim both reproducibly trip `ImpedanceSafetyMonitor`'s `|Y-Y0| > 0.03 m` guard
during a world-X transport move from the -45 deg base-rotation pose
(`hardware.poses.HEIGHT_ALPHA_0_5_CLEARANCE_Q`). Two prior investigations
(`docs/status/base_rotation_neg45_retune_2026-07-31.md`,
`docs/status/nullspace_envelope_search_2026-08-01.md`) exhaustively searched gains (x/y/z/rot/
posture/kd_joint, up to ~1.67x on kp_y/kd_y) and every existing orientation/nullspace mechanism
(`wrist_orientation_task`, `lambda_diagonal_shaping`, `lambda_adaptive_regularization`, combos)
with **zero effect on the trip** in every case -- matching two independent live real-hardware
gain-increase attempts that also had zero effect. Neither investigation instrumented *why*.
This task's job was to actually answer that, using the current (post-2026-07-31 joint-friction)
model, before attempting any new fix.

## 1. Instrumentation added

`tools/ur5e_mujoco_torque_experiments.py`'s per-cycle trace row (both the normal-cycle and
safety-violation-terminal code paths) previously discarded most of `CartesianImpedanceOutput`
every step, keeping only `jacobian_cond`/`singular_scale`/`task_scale`/`task_backtrack_iters`.
Added: `task_backtrack_scale`, `task_feasible`, `y_error`, `z_error`, `wrench` (the raw 6D
task-space force/moment vector before Lambda-shaping), `tau_preclip` (the pre-clip torque
candidate backtracking operates on), `tau_task`, `tau_posture`, `tau_damping`, and
`jacobian_pre_step` (the Jacobian used to compute that step's controller output, for any
task-space/joint-space cross-checks). Purely additive -- new dict keys, no control-path change.
Confirmed inert: the un-rotated canonical grid at height_alpha=0.5 with the plain
`config/ur5e_mujoco_torque_osc_tuned.yaml` still scores exactly 3/8 (byte-identical to
`docs/status/nullspace_envelope_search_2026-08-01.md`'s own baseline number) after this change.

## 2. Reproduction

`tools/ur5e_move_hold_transport.py --config config/ur5e_mujoco_torque_osc_tuned.yaml
--start-q-rad -0.7853981633974483 -0.8353981633974483 -1.2 -0.9853981633974482 0.0 0.0
--target-x-deltas 0.06 --move-durations 1.0 --hold-durations 2.0 --seed 0`, current
(friction-including) model: fails every time, `termination_reason: "|Y-Y0| > 0.03 m"`,
`failure_time_s ~0.68` (during the 1.0s move phase), `max_abs_y_drift_m ~0.0300`. Matches the
prior investigations' signature exactly.

## 3. Hypothesis 1 -- geometric backtracking neutralizing gain increases: RULED OUT

Per-cycle trace of the reproduction above, every ~40ms:

| t (s) | y_error | wrench Fy (N) | task_backtrack_scale | jacobian_cond | max\|tau_preclip\|/tau_limit |
|---|---|---|---|---|---|
| 0.002 | 0.00000 | 0.00 | 1.0 | 1.0e17 | 0.0% |
| 0.242 | 0.00052 | 0.15 | 1.0 | 1.7e4 | 3.8% |
| 0.442 | 0.00876 | 1.84 | 1.0 | 3.3e3 | 4.2% |
| 0.682 | 0.02970 | 3.58 | 1.0 | 1.3e3 | 2.7% |

`task_backtrack_scale == 1.0` for all 342 steps of the run (never drops below 1.0 -- backtracking
never activates), and per-joint torque never exceeds ~4.7% of its headroom limit at any point.
`singular_scale` is also `1.0` throughout (confirms `jacobian_singular_cond_max: 1.0e18` is a
true no-op here, as its own promotion doc claims). **Neither mechanism the task's prompt
hypothesized as "neutralizing the gain increase" is active at all during this failure.** This
also rules out torque saturation as an explanation for why the real +50% kp_y/kd_y live
hardware attempt had zero effect.

`kd_joint` (the one un-nullspace-projected joint-space term, `tau_damping = -kd_joint*qd`) was
also tested directly as a residual candidate (not covered by the prior gain search's tested
range in one direction): `kd_joint=0.0` (vs. baseline 4.0) trips at `y_drift=0.03004m,
t=0.674s` -- statistically identical to baseline (`0.03002m, t=0.684s`). Combined with the
prior investigation's `kd_joint=6/8` (also no effect), `kd_joint` is ruled out in both
directions.

## 4. Hypothesis 2 -- genuine kinematic/dynamic coupling needing more Y authority

A much larger kp_y/kd_y multiplier than anything previously tried (prior attempts topped out at
~1.67x) was tested directly at the failure case (dx=0.06m, move=1.0s, hold=2.0s):

| kp_y / kd_y (x baseline 80/15) | outcome | max\|Y-drift\| | final X error (target 0.06m) |
|---|---|---|---|
| 1.0x (baseline) | **guard trip** `\|Y-Y0\|>0.03m` @ t=0.684s | 0.0300m | (run terminated early) |
| 3x (240/45) | **guard trip** @ t~similar | 0.0300m | 0.026m |
| 5x (400/75) | no trip, `duration_complete` | 0.0232m | 0.028m |
| 10x (800/150) | no trip, `duration_complete` | 0.0125m | 0.033m |
| 10x + `lambda_diagonal_shaping` | no trip | -- | 0.033m (unchanged) |
| 10x, move duration 3.0s (vs 1.0s) | no trip | 0.0126m | 0.033m (**unchanged, non-transient**) |

**Raising kp_y/kd_y far enough (~5x+) does stop the guard trip** -- direct evidence the Y PD
term's math is correct and functioning, and that the earlier ~1.67x live/sim attempts simply
weren't large enough to matter, not that gain increases are categorically ineffective. But every
gain large enough to avoid the trip leaves X tracking at a **steady-state equilibrium** roughly
45-55% short of the 0.06m target (0.026-0.033m final error, vs. the ~0.015m tolerance this
displacement requires) -- confirmed non-transient by tripling the move duration and adding a 2s
hold with zero change to the final X error. `lambda_diagonal_shaping` (the already-known fix for
wrench-shaping Lambda's X-Z off-diagonal leak) has **zero effect** on this X-stall, ruling out
that specific leak as the X-tracking failure's cause too -- it also independently reconfirmed
zero effect on the Y-drift magnitude itself when tested standalone on the plain baseline
(0.03004m @ t=0.628s vs. 0.03002m @ t=0.684s baseline), closing the one gap in the prior
investigation's Phase-2 result (which only tested `lambda_diagonal_shaping` combined with
`wrist_orientation_task`, never alone).

**A new `y_integral_action` mechanism was implemented** (`controller_core/
x_axis_cartesian_impedance.py`: `Fy += ki_y * y_integral`, clamped accumulator, resets in
`reset_from_state()`, mirrors `controller_core/lqr_controller.py`'s existing `_x_integral`
anti-windup pattern) to test whether a *slow-building* bias could hold Y without instantaneous
P/D forces big enough to fight X. Result: **the same trade-off, not a way around it.** A fast,
aggressive dose (`ki_y=3000`, `y_integral_limit_m_s=0.02`) avoids the trip (`max_y_drift=0.0183m`)
but leaves X error at 0.036m (worse than the P/D-only 10x case) and orientation error climbing to
0.168 rad (vs. the 0.25 rad hard ceiling) -- and a moderate, non-destructive dose (`ki_y=150`,
this investigation's committed config) has **zero measurable effect** on the failure at all (see
sec 6). Integral action's usual advantage -- correcting a sustained disturbance without a large
instantaneous force -- doesn't help here because the failure happens **during a ~0.7s move**,
too short a window for a bounded integral term to build up meaningful authority before the guard
already trips.

**Verdict: this is a real, structural steady-state trade-off between Y-holding authority and
X-tracking authority at this pose and this displacement, not a free gain-tuning fix in any
PID-family mechanism (P, D, or I).**

## 5. Hypothesis 3 -- is the 0.03m guard itself miscalibrated for this pose? (raised by user mid-task)

Diagnostic-only run (never committed, never proposed as a real fix): the same failing case
(dx=0.06m) re-run with `max_abs_y_drift_m`/`max_abs_z_drift_m`/`max_abs_orthogonal_drift_m`
raised to 0.5m purely to observe the *natural*, unclipped trajectory shape.

Result: the run completes the full 1.0s move + 2.0s hold with **no instability**.
`move_phase_max_abs_y_drift_m = 0.0423m` (peak, during the move), but
`hold_phase_max_abs_y_drift_m = 0.0058m` -- **the Y excursion is a transient that mostly
self-corrects once the move ends**, not a runaway divergence. `final_x_error_m = 0.0117m`,
notably *better* X-tracking than any of the Y-authority-boosted candidates above (since kp_y/
kd_y are untouched baseline values here). `max_abs_qd_radps = 0.158` throughout -- no sign of
instability or oscillation.

Cross-checked against the canonical-grid dose-response (current, friction-including model, same
sweep as sec 6 below):

| dx (m) | move-phase max\|Y-drift\| (m) |
|---|---|
| 0.01 | 0.0059 |
| 0.02 | 0.0135 |
| 0.03 | 0.0217-0.0234 |
| 0.04 | 0.0299-0.0300 (right at the guard) |
| 0.06 (natural, guard disabled) | 0.0423 (peak) |

Growth is roughly linear in dx (not exploding), consistent between the guarded small-dx cases
and the unguarded dx=0.06m point. **This looks like a bounded, roughly-linear, self-correcting
geometric/dynamic cross-coupling transient that happens to be larger than 0.03m at this specific
pose and displacement, not a hazard signature (spike, oscillation, non-recovering divergence).**

`max_abs_y_drift_m`/`max_abs_z_drift_m`/`max_abs_orthogonal_drift_m` all default to a flat
`0.03` (`controller_core/safety.py`), unconditional on pose, present unchanged since this
repo's very first commit (`git log --follow` on that file: `5ba9605 initial real cartpole
workspace` -> `6d39ae7 Fix axis-error growth check tripping on normal min-jerk tracking lag` ->
current -- no commit ever revisited the 0.03 value itself). No doc or commit message found
anywhere in this repo justifying 0.03m specifically for this task-space transport controller or
this pose; it predates the OSC controller, the -45 deg pose, and the friction model entirely.

**This is real, relevant evidence but a decision this task is explicitly not authorized to make**
(hard constraint: never modify `controller_core/safety.py` guard thresholds). Reporting it here
for a human decision: is ~0.04-0.05m of transient, self-correcting Y motion at this specific pose
and displacement genuinely unsafe in the real lab (e.g. proximity to the wall/obstacle this pose
exists to clear), or is the uniform 0.03m threshold an inherited default that was never
recalibrated per-pose? This task did not and should not answer that -- it requires real-world
judgment about physical clearances this sim cannot see.

## 6. Validation: 4-category rigor sweep, -45 deg pose, current (friction-including) model

Same methodology as `docs/status/nullspace_envelope_search_2026-08-01.md` (`canonical_grid`,
`long_holds`, `large_displacements`, `torque_scale_robustness`, 38 runs, `--seed 0`).

| config | canonical_grid | long_holds | large_displacements | torque_scale_robustness | total |
|---|---|---|---|---|---|
| `ur5e_mujoco_torque_osc_tuned` (baseline) | 0/8 | 0/8 | 0/8 | 0/14 | **0/38** |
| `..._y_integral` (`ki_y=150`, `y_integral_limit_m_s=0.02`) | 0/8 | 0/8 | 0/8 | 0/14 | **0/38** |

**Byte-identical to baseline in every category -- zero improvement, zero regression.** The
gentle, non-destructive `ki_y=150` dose used in the committed config is simply too small to move
the failure (consistent with sec 4's `ki_y=3000` test needing to be ~20x larger to have any
effect at all, at the cost of breaking X-tracking). Most `canonical_grid`/`torque_scale_robustness`
failures here are `move_phase_target_tracking` (the known joint-friction undershoot, unrelated to
Y-drift, see AGENTS.md sec 3) -- `y_integral_action` doesn't touch X tracking or friction, so
these were never expected to change.

### Regression checks (both pass, unchanged from documented baselines)

- Un-rotated canonical grid, height_alpha=0.5, plain baseline config: **3/8**, matching
  `docs/status/nullspace_envelope_search_2026-08-01.md`'s documented baseline exactly.
- Directional-ceiling fix, `config/ur5e_mujoco_torque_osc_tuned_wrist_orient_fixed.yaml`,
  dx=-0.20m x 2 hold durations: **2/2** pass, matching that doc's documented 8/8 (this is the
  subset that specifically motivated the fix).

## 7. Recommendation

No candidate in this investigation (or the two prior ones) clears the hard safety filter at
dx=0.06m from the -45 deg pose. This is now a well-evidenced structural finding, not a search
gap: raising Y-axis authority via any PID-family mechanism (P, D, now also I) demonstrably stops
the guard trip but forces X-tracking into a genuine steady-state failure instead -- there is no
gain, damping, or integral value in this controller architecture that satisfies both constraints
simultaneously at this displacement. A real fix, if one is wanted, needs either (a) a
fundamentally different mechanism this task's constraints put out of scope (e.g. a
priority/hierarchical task-space law that doesn't force X and Y to share the same flat wrench
budget), or (b) a human decision on hypothesis 3 above (is 0.03m real hazard margin at this pose,
or an uncalibrated inherited default) -- if the latter, the practical fix might not be a
controller change at all.

**Practical floor, updated**: unchanged from the prior investigations --
the -45 deg clearance pose is validated safe only at small displacement (dx<=~0.04m in sim);
dx>=0.05-0.06m is a known, reproducible, now root-caused failure with no controller-side fix
found across three independent investigations spanning gain search, orientation/nullspace
mechanisms, and (this task) large-multiplier P/D/I authority and direct backtracking/saturation
instrumentation.

## 8. Test suite

`/common/users/ss5772/miniforge3/envs/mujoco_ur5e/bin/python -m pytest -q -m "unit or mujoco"`:
see final report for pass/fail counts. `controller_core/x_axis_cartesian_impedance.py`'s targeted
unit suites (`tests/unit/test_impedance.py`, `test_impedance_dynamics.py`,
`test_friction_feedforward.py`) pass unchanged (47/47) -- `y_integral_action` defaults to
`False`, `ki_y` defaults to `0.0`, so every existing test's behavior is bit-for-bit unchanged.

## 9. Files changed

- `controller_core/x_axis_cartesian_impedance.py` -- new opt-in `y_integral_action`/`ki_y`/
  `y_integral_limit_m_s` fields + clamped integral term (default off, zero behavior change).
- `tools/ur5e_mujoco_torque_experiments.py` -- additive trace-row diagnostics (sec 1).
- `config/ur5e_mujoco_torque_osc_tuned_y_integral.yaml` (new) -- smoke config for the new
  mechanism, honestly documented as not fixing this failure.
- `docs/status/neg45_y_axis_diagnosis_and_fix_2026-08-01.md` (this doc).
- No `controller_core/safety.py` or `hardware/safety.py` changes. No existing config modified.

Raw sweep outputs (gitignored, not git-recoverable, regenerate via the commands in sec 3-6 above
or the scratch orchestration script referenced in the final report):
`outputs/ur5e_mujoco_torque_transport/` subdirectories are NOT used by this investigation --
all sweeps were written to the agent scratchpad (outside the repo) to avoid clutter; not
git-recoverable either way.

## Rollback

`git rm config/ur5e_mujoco_torque_osc_tuned_y_integral.yaml
docs/status/neg45_y_axis_diagnosis_and_fix_2026-08-01.md`, then revert the diffs to
`controller_core/x_axis_cartesian_impedance.py` and `tools/ur5e_mujoco_torque_experiments.py`
(or revert the commit noted in the final report).
