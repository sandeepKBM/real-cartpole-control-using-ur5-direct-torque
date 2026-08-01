# `compute()` sub-operation profiling (direct_torque controller phase), 2026-07-31

## Context

Real hardware tonight (thinkrobot, `hardware/direct_torque_transport.py`, 500 Hz / 2 ms
period) showed `dominant_phase: "controller"` (`controller_mean_ms: 0.615`, `p95: 0.677`,
`max: 0.801`) -- by far the largest phase, with only real-but-not-huge total-cycle margin
(mean 1.28 ms, p99 1.47 ms of the 2 ms budget). A deadline-overrun trip happened once
tonight (3 consecutive cycles >1 ms late) before the diagnostic residual observer was
disabled, which fixed that specific trip. This task asks whether there are OTHER real
inefficiencies inside the controller computation itself worth flagging, now that it is the
dominant phase.

**No access to thinkrobot or its real traces.** All numbers below are from this machine
(westeros), profiling `XAxisCartesianImpedanceController.compute()` directly against real
state sequences extracted from a MuJoCo sim rollout at the same pose family/config used
tonight. **Absolute wall-clock numbers are not comparable to thinkrobot's** -- different
CPU, different load, different numpy/BLAS build. What should transfer is the *relative*
cost breakdown (which sub-operations dominate), since that's controlled by the math itself,
not the machine.

## Method

1. Ran the representative rollout:
   `python tools/ur5e_mujoco_torque_experiments.py --mode controller-rollout
   --controller-kind impedance --config config/ur5e_mujoco_torque_osc_tuned.yaml
   --trajectory-profile min_jerk_move_hold --target-x-delta 0.20 --move-duration 20.0
   --duration 24.0 --seed 0 --no-plot` (12000 steps, matches tonight's real 0.20 m / 20 s
   move; run passed: `valid_move_and_hold: true`, 0 clips, 0 backtrack iterations anywhere
   in the trajectory).
2. Confirmed the config flags actually active tonight, read directly from
   `config/ur5e_mujoco_torque_osc_tuned.yaml`: `task_space_inertia_shaping: true`,
   `nullspace_posture: true`, `jacobian_singular_cond_max: 1.0e18`,
   `lambda_regularization: 0.1`, `posture_reanchor_on_settle: true`, `kp_rot: 0.0`,
   `lambda_diagonal_shaping`/`lambda_adaptive_regularization`/`wrist_orientation_task` all
   unset (default off).
3. Replayed the trace's real `q`/`qd` through the real MuJoCo model
   (`simulation/ur5e_mujoco_torque.py::build_mujoco_state`) to reconstruct the *real*
   Jacobian, mass matrix, and gravity torque at each real pose along the trajectory (these
   aren't in the trace file itself, only derived scalars are).
4. Built `XAxisCartesianImpedanceController` from the same config, called
   `reset_from_state()` once, then timed `compute()` two ways:
   - **Whole-function timing**: 3000 sampled real states (every 4th of 12000) x 20 loops =
     60000 calls, `time.perf_counter()` around each call, 2 full warmup loops discarded
     first.
   - **Sub-operation timing**: real `J`/`M`/`quat` extracted at 3 representative points
     (near the wrist_2=0 singularity at move start, mid-move, settled hold), each
     sub-operation timed in isolation with 30000 reps / 500 warmup discarded.
   - **Cross-check**: `cProfile` over 15000 calls of the same real state sequence, sorted
     by `tottime`, to validate *relative* ranking (cProfile's own per-call overhead
     inflates absolute numbers, especially for Python-level function calls vs. raw C
     calls, so it's used only for ranking/call-count confirmation, not absolute cost).
   - Scripts are scratch, not committed:
     `/common/home/ss5772/.tmp/.../scratchpad/profile_compute.py` and
     `profile_cprofile.py`.

## Results

### Whole `compute()` cost (this machine, real states)

| | mean | median | p95 | p99 | max |
|---|---|---|---|---|---|
| ms/call | 0.280 | 0.280 | 0.293 | 0.302 | 1.99 |

The single 1.99 ms max is a one-off outlier out of 60000 calls (westeros is a shared
cluster host -- see AGENTS.md SS8 on background load); not treated as representative.

### Sub-operation cost (representative real J/M, stable across all 3 sampled poses to
within ~1 us -- shown for `mid_move`, cond(J)=2975)

| operation | time | share of 0.280 ms mean |
|---|---|---|
| `np.linalg.cond(J)` (SVD-based 2-norm) | 23.6 us | 8.4% |
| `orientation_error_vec_wxyz` (3x normalize + multiply + norm) | 20.3 us | 7.2% |
| `np.linalg.inv(M)` (`m_inv`) | 8.2 us | 2.9% |
| `J @ m_inv @ J.T` (`a_mat`) | 4.0 us | 1.4% |
| `np.linalg.inv(a_mat + eps*I)` (`lambda_mat`) | 13.1 us | 4.7% |
| -- shaping block subtotal (previous 3 rows) | 30.6 us | 10.9% |
| nullspace projector (`j_bar` + `I - J.T@j_bar.T`) | 9.7 us | 3.5% |
| wrench mapping (`J.T @ Lambda @ wrench`) | 2.9 us | 1.0% |
| **sum of measured "big matrix math" sub-ops** | **87.1 us** | **31%** |
| (everything else: state validation, array plumbing, backtrack check, output construction) | ~193 us | ~69% |

**No single linear-algebra operation dominates.** The 6x6 matrices involved are small
enough that even SVD (`cond`) and two dense inversions together cost under 90 us combined.
The majority of `compute()`'s real cost (~69%) is not attributable to any one "big" matrix
operation -- it's the aggregate of ~30-50 small numpy calls per cycle (`reshape`,
`asarray`, `dot`, `clip`, `eye`, state-dict validation via `as_impedance_robot_state`,
`CartesianImpedanceOutput` construction, the backtrack feasibility check). `cProfile`
(15000 calls, relative ranking only) corroborates this shape: `np.linalg.cond`'s internal
SVD call is the single largest individually-attributed operation (15.9% of profiled
cumulative time), `orientation_error_vec_wxyz`'s quaternion pipeline is close behind
(14.6%), the two `np.linalg.inv` calls together are 12.4%, and `as_impedance_robot_state`
plus backtrack/clip plumbing account for another ~25%+ -- no clean majority contributor.

### Is `np.linalg.cond(J)` dead work now that `jacobian_singular_cond_max: 1.0e18`?

**No -- checked, not assumed, and it is real work, not free.** Two independent things are
true simultaneously:

1. **The `singular_scale` branch it feeds is inert tonight.** `cond > jacobian_singular_cond_max > 0.0`
   requires `cond(J) > 1e18`. Measured max `cond(J)` across the whole real trajectory,
   including the exact wrist_2=0 singularity at move start, was `1.26e12` -- six orders of
   magnitude below the threshold. `singular_scale` is provably `1.0` on every real cycle in
   this trajectory (and any physically realizable one).
2. **`cond(J)` is still consumed unconditionally as a required telemetry field**, independent
   of `jacobian_singular_cond_max`: `CartesianImpedanceOutput.jacobian_cond` is read every
   cycle by `hardware/direct_torque_transport.py:461` (`"jacobian_cond": float(output.jacobian_cond)`)
   and by `tools/ur5e_mujoco_torque_experiments.py`'s trace writer -- both part of the
   project's required observability contract (AGENTS.md SS2: "Do not trust an MP4 or a bare
   exit code as success evidence -- read the run record"). Skipping the SVD call would mean
   either fabricating/omitting that trace field (a real, silent behavior/schema change) or
   computing it anyway just for logging (no timing win).
3. **`jacobian_singular_cond_max` is genuinely still runtime-configurable and used at a low
   value elsewhere**: confirmed via `grep` across `config/*.yaml` -- most current configs
   (including tonight's) explicitly set `1.0e18`, but `config/ur5e_mujoco_torque_osc_tuned_wrist_orient.yaml`
   (AGENTS.md SS3/SS4: "most recently developed, actively-used config for the
   `wrist_orientation_task` fix... still has the class default and likely has the same
   freeze bug, unvalidated") has no override, i.e. uses the class default `1.0e5` --
   `singular_scale` is a live, active branch for that config. An unconditional skip would be
   a real behavior change there, exactly the failure mode the task asked me to rule out.

**Conclusion: not provably dead work. No skip implemented.** If the team decides
`jacobian_cond` telemetry isn't worth ~24 us/cycle in the `direct_torque` hot loop
specifically, that's a real product/observability tradeoff (dropping a required trace
field) for a human to decide, not an optimization to silently apply.

### Top cost contributors -- assessed, none implemented

1. **`cond(J)` (SVD, ~24 us/cycle, largest single attributed op).** See above: not dead,
   not safely skippable without either changing telemetry schema or accepting no timing
   win. No cheaper *bit-compatible* substitute found -- an eigenvalue-of-`JᵀJ` shortcut
   (avoiding SVD) would square the conditioning and lose precision at high `cond(J)`,
   exactly the failure mode AGENTS.md documents was deliberately avoided when building the
   URScript on-robot cond estimate. Not implemented.
2. **Orientation-error quaternion pipeline (~20 us/cycle, `controller_core/kinematics_utils.py::orientation_error_vec_wxyz`).**
   3x `quat_normalize_wxyz` (2 inputs + 1 output) + multiply + conj + norm. Plausible in
   principle that normalizing MuJoCo/RTDE-sourced quaternions (already near-unit) is mostly
   redundant defensive work, but proving that safely (numeric-equivalence across many real
   trajectories, matching the residual-observer doc's rigor) requires touching a file that
   is part of the real-time control path, shared by other callers/tests. Per this task's
   explicit instruction not to modify the real-time control path without extremely explicit
   justification -- and with real hardware testing live tonight on a separate machine --
   this is **flagged for human review, not implemented**.
3. **Lambda-shaping block (`m_inv`, `a_mat`, `lambda_mat`; ~31 us/cycle combined).** Checked
   for a specific suspected inefficiency (duplicate inversion when `lambda_mat_nullspace`
   should equal `lambda_mat`): confirmed **not duplicated** -- the code already does
   `lambda_mat_nullspace = lambda_mat` (aliased, no second inversion) whenever
   `lambda_adaptive_regularization` is off, which it is tonight. Only one 6x6 inversion for
   Lambda actually runs. No inefficiency found here; not flagged.
4. **Everything else (~69% of the 0.280 ms mean).** Not one operation -- the aggregate of
   many small numpy/Python calls throughout `compute()`. The real lever here, if ever
   pursued, would be reducing *call count* (e.g. `as_impedance_robot_state`'s validation
   overhead, redundant `reshape`/`asarray` calls already on well-shaped arrays), not
   optimizing any single linear-algebra step -- a broader refactor, explicitly out of scope
   for a "don't touch the real-time path during a live hardware session" mandate. Reported,
   not attempted.

## What was NOT done, and why

No change was made to `controller_core/x_axis_cartesian_impedance.py`,
`controller_core/kinematics_utils.py`, or any other real-time control-path file. This
task was explicitly scoped as profiling/diagnosis only, real hardware testing is active
tonight on a separate machine by someone else, and every candidate optimization found
required touching exactly the files the task said not to modify without extremely explicit
justification -- none of the findings here rise to that bar. This matches the project's
own established discipline (AGENTS.md: "real controller-behavior changes get proposed, not
silently patched").

## Tests / commands run

- `python -m pytest -q tests/ -m "not slow"` -- confirmed baseline before profiling:
  `430 passed, 1 deselected` (unchanged from AGENTS.md's stated baseline).
- No source files changed; nothing to re-run after.

## Rollback

N/A -- no code changes made. This document and the scratch profiling scripts
(`/common/home/ss5772/.tmp/.../scratchpad/profile_compute.py`, `profile_cprofile.py`,
not under version control) are the only artifacts.
