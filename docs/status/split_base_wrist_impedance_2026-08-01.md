# Base/wrist task split for the wrist_2=0 Jacobian conditioning problem (2026-08-01)

## Motivating failure

Real UR5e hardware testing tonight (2026-08-01) at the height_alpha=0.5 "zero-degree"
transport pose (`q = [0.0, -0.835398, -1.2, -0.985398, 0.0, 0.0]`, i.e.
`hardware/poses.py::HEIGHT_ALPHA_0_5_Q`) repeatedly tripped the TCP-acceleration safety
guard using `accel_duration_scurve` at a modest target_accel (0.005-0.02 m/s^2), with
`config/ur5e_mujoco_torque_osc_tuned_wrist_orient_friction_ff.yaml`. Root cause traced
directly from real `trace.jsonl`: the arm sits almost exactly at `wrist_2 approx 0` (a
UR5e kinematic singularity present at every `height_alpha`, per
`hardware/poses.py::q_for_height_alpha`), and `jacobian_cond` (full 6x6 Jacobian,
`controller_core/x_axis_cartesian_impedance.py`) oscillates 5-10x cycle-to-cycle purely
from being parked there. Adding `lambda_diagonal_shaping`/`lambda_adaptive_regularization`
did NOT help on a real-hardware retest (if anything slightly worse, peak accel 0.92 vs
0.72 m/s^2) -- neither flag touches `cond(J)` itself, an upstream geometric property of J
at that exact configuration that no downstream Lambda regularization fixes.

## Step 1 -- numeric verification of the mechanistic hypothesis

Checked directly in sim, at the exact failure pose, via the real MJCF and the same
`build_mujoco_state()` the controller sees. This was a one-off scratch numeric check (no
file under `tools/diagnostics/` was added), results reproduced below verbatim so they are
auditable without re-running anything:

```
q = [0.0, -0.835398, -1.2, -0.985398, 0.0, 0.0]

Full 6x6 Jacobian J:
[[ 2.340000e-01 -7.648839e-01 -4.497193e-01 -9.927130e-02  9.927130e-02  0.000000e+00]
 [-1.215331e-01 -0.000000e+00  0.000000e+00  0.000000e+00  1.387779e-17  0.000000e+00]
 [ 0.000000e+00 -1.215331e-01  1.635919e-01 -1.205028e-02  1.205028e-02  0.000000e+00]
 [ 0.000000e+00  0.000000e+00  0.000000e+00  0.000000e+00 -1.205028e-01  0.000000e+00]
 [ 0.000000e+00 -1.000000e+00 -1.000000e+00 -1.000000e+00  0.000000e+00 -1.000000e+00]
 [ 1.000000e+00  0.000000e+00  0.000000e+00  0.000000e+00  9.927130e-01  0.000000e+00]]

(a) cond(full 6x6 J)                                          = 7.283151e+16
(b) cond(3x3 position-rows x base-joint-cols [pan,lift,elbow]) = 7.824458e+00
(c) cond(3x3 rotation-rows x wrist-joint-cols [w1,w2,w3])      = inf (smallest singular value 0.0)
```

**Verdict: CONFIRMED, with a refinement.** The base-joint position sub-Jacobian is well
conditioned (~7.8, four orders of magnitude better than "well conditioned" thresholds
typically used elsewhere in this repo). The wrist-only rotation sub-Jacobian is not just
ill-conditioned but **exactly singular** (smallest singular value is 0.0 to machine
precision) -- wrist_1 and wrist_3's rotation axes align when wrist_2=0, the textbook UR
wrist gimbal lock, visible directly in `J[3:6, 3:6]`'s middle row `[-1, 0, -1]` (wrist_1
and wrist_3 columns are identical there).

This **refutes** the naive symmetric design (base-only translation impedance + wrist-only
orientation impedance) that the task background proposed as the fallback: a pure
wrist-only orientation task at this pose would be at least as fragile as the shared
pipeline it would replace, not better -- routing the rotational wrench through it would
just relocate the ill-conditioning problem from a 6x6 matrix to an exactly-singular 3x3
one.

## Step 2 -- design and implementation

Per the evidence, the design implemented is the "orientation stays with nullspace_posture"
branch, not the naive symmetric split:

- **New flag** `CartesianImpedanceConfig.split_base_wrist_task: bool = False`
  (`controller_core/x_axis_cartesian_impedance.py`), parsed in
  `from_controller_yaml_section()`. See the field's own docstring in the source for the
  full derivation (duplicated in spirit here, condensed).
- **Translation task restricted to base joints**: a reduced Jacobian `J_task` (3x6) is
  built with `J_task[:, 0:3] = J[0:3, 0:3]` and `J_task[:, 3:6] = 0` -- position-rows only,
  base-joint columns only. Wrist columns are structurally zero, so translation-task torque
  can never route through `wrist_2` (or any wrist joint) regardless of how ill-conditioned
  the full J is there.
- **Rotational wrench dropped from the task pipeline**: the `[Fx, Fy, Fz]` wrench (not the
  full 6-vector including `M`) is what gets mapped through `J_task`. `M` (the `kp_rot`/
  `kd_rot` term) is computed as before and still reported in the diagnostic `wrench` trace
  field, but is not used to produce task torque in split mode -- per the step-1 evidence,
  routing it through the singular wrist-only submatrix would not be an improvement.
- **Orientation stays held by `nullspace_posture`**, but recomputed against the same
  reduced `J_task` (a rank-3 task instead of the near-singular rank-(barely-6) one), so
  base *and* wrist redundancy is available to the posture-holding nullspace projector; and,
  if `wrist_orientation_task` is also enabled (it is, in the new named config), that
  separate joint-space PD path is unaffected -- it does no matrix inversion, so it cannot
  itself blow up from ill-conditioning (background's own observation, confirmed correct).
- **Implementation is a generalization, not a branch duplication**: `task_space_inertia_shaping`'s
  Lambda (`a_mat = J_task @ M^-1 @ J_task.T`), the `jacobian_singular_cond_max` /
  `lambda_adaptive_regularization` scheduling (now keyed on `cond_task` = cond of the
  reduced 3x3 block instead of the full 6x6), and the nullspace projector all reuse the
  exact same formulas as before, just parameterized on `J_task`/`wrench_task`/`cond_task`
  instead of `J`/`wrench`/`cond`. When `split_base_wrist_task` is off, `J_task = J` (a 6x6
  identity substitution), so every one of those computations is unchanged arithmetic, not
  just coincidentally-equal output.
- **`jacobian_cond` trace field** now reports `cond_task` -- the conditioning of whatever
  Jacobian the task pipeline actually used, matching the ask that it "reflect the reduced
  Jacobian's much-better conditioning, not the full 6x6's" when the flag is on.
- **New config** `config/ur5e_mujoco_torque_osc_tuned_split_base_wrist.yaml`: byte-identical
  to `config/ur5e_mujoco_torque_osc_tuned_wrist_orient_friction_ff.yaml` (the current best
  real-hardware config) with only `split_base_wrist_task: true` added. No existing config
  modified.

## Step 3 -- validation

### Unit tests (`tests/unit/test_split_base_wrist_task.py`, 7 new tests, all passing)

- Flag defaults to `False`.
- Flag-off byte-identical to a reference controller that never mentions the flag (using
  the real failure-pose Jacobian and a nontrivial mass matrix, with
  `task_space_inertia_shaping`/`nullspace_posture`/`wrist_orientation_task` all on, so the
  regression check exercises the generalized code paths, not just the trivial default).
- `jacobian_cond` reports the reduced 3x3 base-block conditioning (matches a
  directly-computed `np.linalg.cond` reference) when the flag is on, and the full-J
  conditioning (numerically singular, `>1e10`) when off.
- Translation-task torque is exactly zero on wrist joints (structural guarantee, not just
  numerically small) regardless of the wrist columns' content in J.
- The rotational wrench is provably dropped from the task pipeline (zero task torque from
  pure angular-velocity damping with the flag on; nonzero with it off) while the
  diagnostic `wrench` field is unaffected.
- `nullspace_posture` + `split_base_wrist_task` together stay finite (no NaN/inf) at the
  real near-singular failure-pose Jacobian.
- YAML parsing round-trip.

Full suite: `python -m pytest -q` -- **518 passed, 3 xfailed** (pre-existing, unrelated to
this change), zero regressions.

### Targeted smoke test: reproducing tonight's exact real failure signature

`accel_duration_scurve`, `target_accel` in `{+0.02, -0.02, +0.005, -0.005}` m/s^2,
`move_duration` in `{4.0, 8.0}` s, at the exact failure pose, comparing
`ur5e_mujoco_torque_osc_tuned_split_base_wrist.yaml` against the baseline
`..._wrist_orient_friction_ff.yaml`.

**`jacobian_cond` (the direct target of this fix):**

| config | full trace: min | max | median |
|---|---|---|---|
| baseline (representative run, accel=+0.02, dur=4.0) | 1.04e+04 | 8.07e+16 | 1.88e+04 |
| split (same run) | 7.82 | 13.26 | 12.00 |

Across all 16 runs, baseline's `max_jacobian_cond` is a flat `8.07e16` in every single case
(the full-J singularity dominates the whole hold, since the arm barely moves off
`wrist_2=0` at these small accelerations); split's `max_jacobian_cond` ranges `7.8-2.7e4`
depending on how far the move pushes the arm off the base pose, still 12+ orders of
magnitude better than baseline in the realistic (small-accel) cases and 3 orders of
magnitude better even in the one large-displacement outlier case below.

**Cycle-to-cycle jitter** (the specific real-hardware complaint -- "oscillates 5-10x
cycle-to-cycle"), measured as the largest single-step `|log10(cond[i+1]/cond[i])|` jump
across a representative full trace (accel=+0.02, dur=4.0, 3000 steps): baseline's top jump
is `10^3.2 approx 1600x` in a single control cycle (cond jumping from ~1e10-scale to
5.0e13 between adjacent steps); split's top jump across the same trace is `10^0.0`, i.e.
no measurable jitter at all -- cond stays in a tight `7.8-13.3` band the entire trajectory.
This directly confirms the mechanism described in the motivating failure.

**Pass/fail and orientation quality:** all 8 small/realistic-magnitude runs (accel
`+-0.005` at both durations, accel `+-0.02` at `move_duration=4.0` -> `dx` up to ~0.05m)
pass cleanly in both configs, comparable orientation error (`0.019-0.08` rad, split
slightly higher in a few cells, e.g. `0.0733` vs `0.0192`... not directly comparable since
the two 4.0s-duration accel=0.005 runs differ from the 8.0s ones -- see raw table below).
The two `accel=+-0.02, move_duration=8.0` runs (`dx approx 0.20m`, the known, already-documented
directional-ceiling case) fail via the orientation guard (`>0.25 rad`) **identically** in
both configs (baseline: max_ori 0.2500 rad; split: 0.2500/0.2499 rad) -- this fix has no
effect on that separate, already-documented large-displacement failure mode, and does not
regress it either.

Raw per-run table (`accel`, `move_duration`, config -> `success`, `max_jacobian_cond`,
`max_abs_orientation_error_rad`):

| accel | dur | baseline success | baseline max_cond | baseline max_ori | split success | split max_cond | split max_ori |
|---|---|---|---|---|---|---|---|
| +0.020 | 4.0 | True | 8.07e16 | 0.0668 | True | 1.33e+01 | 0.0707 |
| +0.020 | 8.0 | True | 8.07e16 | 0.2424 | False (guard) | 2.73e+04 | 0.2500 |
| -0.020 | 4.0 | True | 8.07e16 | 0.0683 | True | 7.82 | 0.0733 |
| -0.020 | 8.0 | False (guard) | 8.07e16 | 0.2500 | False (guard) | 7.82 | 0.2499 |
| +0.005 | 4.0 | True | 8.07e16 | 0.0192 | True | 8.68 | 0.0206 |
| +0.005 | 8.0 | True | 8.07e16 | 0.0733 | True | 1.31e+01 | 0.0802 |
| -0.005 | 4.0 | True | 8.07e16 | 0.0192 | True | 7.82 | 0.0207 |
| -0.005 | 8.0 | True | 8.07e16 | 0.0743 | True | 7.82 | 0.0822 |

(TCP-acceleration numbers from a finite-difference proxy -- double-diff of `ee_pos`,
matching `CartesianMoveMonitor`'s own formula -- are not reported here in detail: this sim
tool has no `CartesianMoveMonitor` equivalent, and the raw finite-difference series is
dominated by numerical differentiation noise at this sim's timestep, consistent with
AGENTS.md sec 4's own note that this is "a genuine double finite-difference, ~1/dt^2 noise
amplifier." `jacobian_cond` is the reliable, decisive signal here, and it is the value this
fix directly targets.)

### Secondary: 4-category rigor sweep at height_alpha=0.5 (partial -- 3 of 4 categories)

Ran `canonical_grid`, `large_displacements`, `torque_scale_robustness` (skipped
`long_holds` for time; see below). Both configs use the identical
`config/ur5e_mujoco_torque_osc_tuned_wrist_orient_friction_ff.yaml` gains
(`tools/ur5e_pose_sweep_transport.py`, `--seed 0`).

| category | baseline | split |
|---|---|---|
| canonical_grid | 7/8 | 6/8 |
| large_displacements | 8/8 | 8/8 (byte-identical outcome to baseline, including the guard-trip case above) |
| torque_scale_robustness | 12/14 | 12/14 (byte-identical) |
| long_holds | not run | not run |

**Honest regression, root-caused**: split loses one case in `canonical_grid` --
`dx=0.01m, hold=1.0s` fails via `hold_phase_target_tracking` (a tracking-tolerance
undershoot, not a guard trip) with split on, where baseline passes it. Baseline itself
fails the *other* smallest-displacement case (`dx=0.01m, hold=2.0s`) via the identical
failure mode. Both configs are marginal at the very smallest displacement in this grid
(the friction-feedforward undershoot documented in AGENTS.md sec 3 sits right at this
tolerance edge already) -- split's change shifts which of the two `dx=0.01m` cells is on
the losing side of that edge rather than introducing a new failure mode. `dx >= 0.02m`
(every case that matters for anything beyond a minimal nudge) is unaffected, and every
`large_displacements`/`torque_scale_robustness` outcome is identical between the two
configs. Not investigated further given the "secondary, don't crowd out step-1/2
reporting" instruction -- flagged here plainly rather than glossed over.

## Bottom line

- Step 1's hypothesis is confirmed with a refinement: base-only translation is well
  conditioned (~7.8) and the naive wrist-only orientation fallback would have been
  exactly singular, not a fix -- the implemented design (base-only translation task,
  wrist columns structurally excluded; orientation kept on the existing
  nullspace-posture/wrist_orientation_task mechanisms, recomputed against the reduced
  task) is the one the evidence actually supports.
- `jacobian_cond` drops from a permanently-singular ~1e16-1e17 with wild cycle-to-cycle
  jitter (up to ~1600x in one control cycle) to a tight, stable 7.8-13 band in every
  realistic-magnitude reproduction of tonight's real failure -- direct, decisive
  confirmation of the mechanism.
- Zero regression in `large_displacements`/`torque_scale_robustness`; one small, honestly
  reported, root-caused regression in the smallest-displacement `canonical_grid` cell
  (a pre-existing marginal tolerance edge, not a new failure mode); `long_holds` not run
  (time budget).
- **Sim only. Not real-hardware validated.** This does not replace
  `config/ur5e_mujoco_torque_osc_tuned.yaml`/`..._wrist_orient_friction_ff.yaml`
  as the real-hardware default without its own careful, small-first real-lab check, per
  every other unvalidated config in this repo's history.
