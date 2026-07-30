# Disabling the global cond(J) singular_scale gate for the wrist_orient config: freeze confirmation + full validation sweep

Context: `config/ur5e_mujoco_torque_osc_tuned.yaml` was promoted earlier tonight (commit
`508d66e`) to set `jacobian_singular_cond_max: 1.0e18`, disabling the global `cond(J)`-based
`singular_scale` term in `controller_core/x_axis_cartesian_impedance.py` (see that config's
own header comment and `docs/status/disable_global_singular_scale_validation_2026-07-30.md`
for the full mechanism and a 304-run rigor sweep: 152/152 pass with the fix vs 140/152
without, zero regressions).

That promotion explicitly did **not** touch `config/ur5e_mujoco_torque_osc_tuned_wrist_orient.yaml`
-- the separately tuned/validated config carrying the `wrist_orientation_task` fix (a
dedicated wrist-only orientation PD term, see that file's own header comment and
`docs/status/wrist_orientation_task_2026-07-29.md`) -- because it was independently
tuned/validated with `singular_scale` on and needed its own re-validation before changing,
per this project's "preserve old configs, don't combine unrelated changes without
validation" discipline. Confirmed via
`grep -n jacobian_singular_cond_max config/ur5e_mujoco_torque_osc_tuned_wrist_orient.yaml`
returning nothing (falls back to the class default `1.0e5`, `singular_scale` enabled).

This doc is that re-validation.

## Verdict

**Freeze confirmed real in this config too, and confirmed fixed by the same
`jacobian_singular_cond_max: 1.0e18` override, with a real measured drop in raw peak TCP
accel. Sweep is clean: zero regressions, byte-identical pass/fail counts and byte-identical
failure signature at every alpha/category tested.** Unlike the base config's validation, the
sweep does not recover any additional passing runs here -- see "Why the sweep doesn't move
the needle" below for the likely reason. Recommendation at the end.

## 1. Freeze confirmation: before/after, real numbers

Reproduced with the exact gentle canonical move specified for this task:
`tools/ur5e_mujoco_torque_experiments.py --mode controller-rollout --controller-kind impedance
--gravity-mode gravity_comp --trajectory-profile min_jerk_move_hold --target-x-delta 0.02
--move-duration 1.5 --duration 2.5 --seed 0 --no-plot`,
`config/ur5e_mujoco_torque_osc_tuned_wrist_orient.yaml` (baseline, singular_scale active) vs
`config/ur5e_mujoco_torque_osc_tuned_wrist_orient_no_singular_scale.yaml` (candidate fix),
default `height_alpha=0.5` transport start pose (`[0,-pi/2,0,-pi/2,0,0]`, the exact wrist
singularity).

### jacobian_cond / singular_scale / |tau_applied| over time

| t (s) | baseline cond(J) | baseline singular_scale | baseline \|tau_applied\| (Nm) | fixed cond(J) | fixed singular_scale | fixed \|tau_applied\| (Nm) |
|---|---|---|---|---|---|---|
| 0.002 | 2.49e17 | 4.02e-13 | 0.000000 | 2.49e17 | 1.0 | 0.000000 |
| 0.126 | 1.11e17 | -- | 0.000000 | 1.30e6  | 1.0 | 0.114173 |
| 0.252 | 3.53e16 | -- | 0.000000 | 1.12e5  | 1.0 | 0.178099 |
| 0.502 | 7.24e12 | -- | 0.000000 | 1.29e4  | 1.0 | 0.054890 |
| 0.752 | 1.17e6  | 8.57e-2 | 1.045084 | 4.78e3 | 1.0 | 0.561185 |
| 1.002 | 3.04e3  | 1.0     | 1.969269 | 2.83e3 | 1.0 | 1.079651 |
| 1.502 | 2.36e3  | 1.0     | 1.207681 | 2.14e3 | 1.0 | 1.199908 |
| 2.500 | 2.30e3  | 1.0     | 1.172739 | 2.18e3 | 1.0 | 1.165432 |

- **Baseline: `|tau_applied|` stays exactly 0.0 through t=0.502s; first `> 0.01 Nm` at
  t=0.690s; first `> 0.5 Nm` at t=0.744s.** Same qualitative freeze as the base config
  (there: `>0.01` at t=0.784s, `>0.5` at t=0.838s) -- slightly earlier here, consistent with
  the `wrist_orientation_task`'s separate wrist-only PD term contributing some torque
  independent of the frozen main wrench, but the main-pipeline freeze itself is the same
  mechanism: `singular_scale` stays near-zero (`4e-13` to `8.6e-2`) until `cond(J)` decays
  numerically below `jacobian_singular_cond_max=1e5`.
- **Fixed: first `|tau_applied| > 0.01 Nm` at t=0.028s; no freeze.** `singular_scale` pinned
  at `1.0` throughout -- `cond(J)` still spans the same ~17 orders of magnitude (same pose,
  same physical singularity), it just no longer gates the wrench.

### CartesianMoveMonitor replay: raw peak TCP accel estimate

Replayed `ee_pos` from each trace through `hardware.safety.CartesianMoveMonitor`
(`accel_gap_cycles=1`, `speed_lowpass_alpha=1.0` = no filtering, all limits set to `1e6` so
nothing trips), cross-checked against the vectorized formula in
`tools/analyze_state_noise_capture.py::compute_guard_quantities` (same cross-check method as
the base config's validation) -- both agree exactly.

| | baseline (singular_scale active) | fixed (no_singular_scale) | change |
|---|---|---|---|
| peak TCP accel estimate | **2.3800 m/s^2** (at t=0.786s) | **0.0601 m/s^2** (at t=0.372s) | **39.6x lower** |
| peak TCP speed estimate | 0.0904 m/s | 0.0255 m/s | 3.5x lower |
| theoretical kinematic peak accel (min-jerk, dx=0.02m/1.5s) | 0.0513 m/s^2 | 0.0513 m/s^2 | -- |
| ratio to theoretical peak | 46.4x | 1.17x | -- |

Same qualitative signature as the base config's fix (there: 2.72 -> 0.058 m/s^2, 46.5x): the
peak accel lands right at the freeze/catchup boundary (t=0.786s here, vs the freeze ending
~0.74-0.75s per the tau table above), and the fixed config's peak tracks the intended
min-jerk profile (1.17x theoretical peak) instead of a step-function catchup (46.4x). At the
real default guard threshold (`max_tcp_accel_mps2=0.5`), the baseline's 2.38 m/s^2 would trip
that guard by 4.8x; the fixed config's 0.060 m/s^2 sits comfortably under it.

## 2. Full four-category rigor sweep

Ran `tools/ur5e_pose_sweep_transport.py` (all four categories: canonical_grid, long_holds,
large_displacements, torque_scale_robustness), baseline
(`config/ur5e_mujoco_torque_osc_tuned_wrist_orient.yaml`) vs candidate fix
(`config/ur5e_mujoco_torque_osc_tuned_wrist_orient_no_singular_scale.yaml`), `--seed 0`, gains
auto-extracted from each config (tool default, not `--no-gain-overrides`).

**Scope actually covered: `height_alpha in {0.1, 0.2, 0.3, 0.5}`** -- matches the base
config's validation scope exactly (all four named in this task: the required 0.5, plus the
three optional 0.1/0.2/0.3). 38 runs/alpha/config, 152 runs per config, 304 runs total.

### Pass/fail counts (num_valid_move_and_hold / num_runs)

| height_alpha | category | baseline (wrist_orient) | fixed (wrist_orient_no_singular_scale) | regression? |
|---|---|---|---|---|
| 0.1 | canonical_grid | 8/8 | 8/8 | no |
| 0.1 | long_holds | 8/8 | 8/8 | no |
| 0.1 | large_displacements | 8/8 | 8/8 | no |
| 0.1 | torque_scale_robustness | 14/14 | 14/14 | no |
| 0.2 | canonical_grid | 8/8 | 8/8 | no |
| 0.2 | long_holds | 8/8 | 8/8 | no |
| 0.2 | large_displacements | 8/8 | 8/8 | no |
| 0.2 | torque_scale_robustness | 14/14 | 14/14 | no |
| 0.3 | canonical_grid | 8/8 | 8/8 | no |
| 0.3 | long_holds | 8/8 | 8/8 | no |
| 0.3 | large_displacements | 8/8 | 8/8 | no |
| 0.3 | torque_scale_robustness | 14/14 | 14/14 | no |
| 0.5 | canonical_grid | 8/8 | 8/8 | no |
| 0.5 | long_holds | 8/8 | 8/8 | no |
| 0.5 | large_displacements | 8/8 | 8/8 | no |
| 0.5 | torque_scale_robustness | **12/14** | **12/14** | no (identical failure) |
| **Total** | | **150/152** | **150/152** | **zero regressions, zero net change** |

(4 alphas x 38 runs/alpha = 152 runs/config; 2 failures/config, both at alpha=0.5
torque_scale_robustness -> 150/152 valid for both configs.)

### The two identical failures (not a regression)

Confirmed by reading each config's `alpha_0p5/torque_scale_robustness/run_log.csv`
`failure_category`/`phase_at_failure` columns directly (not inferred from the pass count
alone): both configs fail the exact same two cases --
`target_x_delta_m in {0.03, 0.06}`, `torque_limit_scale=0.1`, both via `z_drift` during the
`move` phase. Byte-identical failure signature in both configs -- at 10% of the torque
limit, this is a genuine torque-budget saturation case unrelated to `singular_scale` (already
`1.0` at this well-conditioned displacement-move dynamic regardless of the gate), same
conclusion as the base config's validation found for its one shared failure.

### Why the sweep doesn't move the needle here (unlike the base config)

The base config's baseline failed several `canonical_grid` cases at alpha in {0.2, 0.3, 0.5}
(6/8, 4/8, 4/8) because the ~0.7-0.8s freeze ate too much of the 1.0s canonical-grid move
window before `move_phase_target_tracking` could complete. Here, the wrist_orient config's
baseline (singular_scale still active) passes canonical_grid 8/8 at every alpha, including
0.5 -- the freeze is confirmed present (part 1 above) but does not translate into pass/fail
differences at these grid points. Not independently isolated in this pass, but the most
likely explanation given the two configs' only structural difference: `wrist_orientation_task`
gives the wrist joints a dedicated PD torque path (`kd_rot_wrist=10.0`) that is NOT gated by
the frozen main-wrench `singular_scale` term, so orientation/tracking degradation during the
freeze window is damped enough that the move still completes within tolerance by t=1.0s in
these specific canonical-grid cases. This is a plausible mechanism, not confirmed by a
targeted ablation in this pass -- flagging as not fully explained rather than asserting it.

### Not covered

- `height_alpha` values outside `{0.1, 0.2, 0.3, 0.5}` were not swept here.
- Only `--seed 0` was used (matches project convention; sim has no noise source).
- `torque_task_qp.py` has the identical `singular_scale` pattern per AGENTS.md and was not
  touched or evaluated -- out of scope, not the default controller.
- No real-hardware or URScript-path validation -- sim-only.
- The "why the sweep doesn't move the needle" mechanism above (wrist_orientation_task
  absorbing the freeze) is a plausible explanation, not independently confirmed by a
  targeted ablation.

## 3. Test suite

`python -m pytest -q tests/ -m "not slow"`: **413 passed, 1 deselected** -- matches the
stated baseline exactly. No library code was touched by this investigation; only sweeps were
run, one new named config was added, and a scratch replay script was used (not committed
under `tools/`).

## 4. Recommendation

Evidence-backed but more modest than the base config's case: **the freeze is confirmed real
in this config and confirmed fixed (39.6x lower raw peak TCP accel, 1.17x theoretical peak
instead of 46.4x), and the sweep found zero regressions** -- every alpha/category pass/fail
count is byte-identical between baseline and fixed, including an identical two-case failure
signature at alpha=0.5/torque_scale=0.1 that is unrelated to `singular_scale`. Unlike the
base config, this sweep does not show additional *passing* runs recovered, so the case for
promotion here rests on the freeze/accel evidence in part 1 (a real, measured hazard reduction
for real-hardware TCP-accel guard margins) rather than on improved task-completion rate. Given
zero regressions and a real, measured safety-relevant improvement with no downside found in
304 runs, disabling `singular_scale` for this config looks safe to promote -- but this is a
decision for a human to make, not something this investigation acts on unilaterally. Per task
instructions, `config/ur5e_mujoco_torque_osc_tuned_wrist_orient.yaml` was left unchanged.
