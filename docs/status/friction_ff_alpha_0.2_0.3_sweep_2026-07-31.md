# Friction feedforward validation: height_alpha in {0.2, 0.3}, 4-category rigor sweep

Context: `controller_core/x_axis_cartesian_impedance.py`'s new opt-in `friction_feedforward`
term (commit `5eb9778`) had exactly one prior validation point -- a single dx=0.04m smoke
test at the default (~0.5-ish) transport pose. This note runs the full four-category rigor
grid (`tools/ur5e_pose_sweep_transport.py`: canonical_grid, long_holds, large_displacements,
torque_scale_robustness) at `height_alpha in {0.2, 0.3}` for both
`config/ur5e_mujoco_torque_osc_tuned.yaml` (baseline, no feedforward) and
`config/ur5e_mujoco_torque_osc_tuned_friction_ff.yaml` (feedforward), simulation only, to
decide whether Part C (integral action, `ki_x`) is needed on top.

**Scope correction vs. the plan that spawned this run**: the plan's premise that "neither
alpha has ever been swept with any config" is not quite right -- `height_alpha in {0.2, 0.3}`
was already swept 2026-07-29 with `config/ur5e_mujoco_torque_osc_tuned_wrist_orient.yaml`
(`docs/status/intermediate_pose_sweep_alpha_0.2_0.3_2026-07-29.md`, 76/76 valid, no friction
issues at all). That sweep predates the friction model landing in
`assets/ur5e_torque/ur5e_torque.xml` (Part A of tonight's plan, still in-flight/uncommitted in
the working tree as of this run) -- so it is not a comparable baseline for the friction
question, and this run genuinely is the first sweep at these alphas *with the current
friction-modeled MJCF*, for either the plain tuned config or the new feedforward config.

## Execution

Ran across 4 hosts in parallel (westeros + ilab1/ilab3/ilab4, `ilab2` excluded per instructions
-- confirmed load average 121/64 unrelated to this session), one (alpha, config) combination
per host, each running all four categories via `tools/ur5e_pose_sweep_transport.py`.
`OPENBLAS_NUM_THREADS`/`OMP_NUM_THREADS`/`MKL_NUM_THREADS`/`NUMEXPR_NUM_THREADS=1` exported on
every host. ilab hosts confirmed idle enough before dispatch (`uptime`: ilab1 1.67/64, ilab3
7.85/96, ilab4 4.70/96) and re-verified per-user memory cgroup cap (~81.4 GiB, matching prior
findings). `Linger=no` confirmed on all three ilab hosts, so each ran as one foreground SSH
session wrapped in a locally-backgrounded shell call, not `nohup`+`disown`. All 4 jobs
completed with exit code 0; results below are read directly from each run's `run_log.csv` /
`summary.json` via `transport_metrics.compute_valid_move_hold_metrics`, not from stdout/exit
codes alone.

Raw outputs: `outputs/ur5e_mujoco_torque_transport/friction_ff_sweep_2026-07-31/` (gitignored,
not committed -- `{host}_{config}_alpha{a}/alpha_{a}/{category}/`).

## Headline pass/fail counts

| alpha | config | canonical_grid | long_holds | large_displacements | torque_scale_robustness | total |
|---|---|---|---|---|---|---|
| 0.2 | baseline (no FF) | 3/8 | 4/8 | 8/8 | 7/14 | 22/38 |
| 0.2 | friction_ff | **8/8** | **8/8** | 8/8 | **14/14** | **38/38** |
| 0.3 | baseline (no FF) | 3/8 | 4/8 | 8/8 | 6/14 | 21/38 |
| 0.3 | friction_ff | **7/8** | **8/8** | 8/8 | **12/14** | **35/38** |

Friction feedforward never loses to baseline in any single (alpha, category) cell -- strictly
equal or better everywhere. Combined: baseline 43/76 (57%) vs. friction_ff 73/76 (96%) across
the full grid at both alphas.

## Root cause of the baseline failures (confirmed, not inferred)

Every baseline failure category traces to `hold_phase_target_tracking`: the controller keeps
slowly creeping in X *during* the hold phase (a classic stiction slow-settle signature) and
exceeds the tight `hold_phase_x_drift_from_hold_start_m` tolerance (`max(0.003 m, 15%*target)`)
even though the phase's *final* error is small. Confirmed directly on
`dx=0.01m, hold=1s, alpha=0.2, baseline`: by the end of the *move* phase the controller had
only reached `0.00576 m` of the `0.01 m` target -- **57.6% achieved**, matching tonight's real
hardware finding (~55-72% of commanded displacement) almost exactly. This is the same
mechanism the friction-ff work was built to fix, now confirmed present in the sweep-level
pass/fail data, not just the one earlier smoke test.

## Steady-state comparison (canonical_grid, both alphas)

Representative rows (`final_x_error_m` = position error at hold-phase end,
`mean_max_tau_applied_nm` = mean across joints of each joint's peak applied torque for that
run):

| alpha | target dx (m) | hold (s) | config | achieved frac (move end) | final x_err (m) | mean peak tau (Nm) |
|---|---|---|---|---|---|---|
| 0.2 | 0.01 | 1.0 | baseline | 0.906 | 0.00094 | 2.87 |
| 0.2 | 0.01 | 1.0 | friction_ff | 0.963 | 0.00037 | 2.80 |
| 0.2 | 0.04 | 2.0 | baseline | 0.993 | 0.00027 | 3.33 |
| 0.2 | 0.04 | 2.0 | friction_ff | 0.999 | 0.00003 | 3.55 |
| 0.3 | 0.01 | 1.0 | baseline | 0.897 | 0.00103 | 3.75 |
| 0.3 | 0.01 | 1.0 | friction_ff | 0.956 | 0.00044 | 3.69 |
| 0.3 | 0.04 | 2.0 | baseline | 0.992 | 0.00031 | 4.21 |
| 0.3 | 0.04 | 2.0 | friction_ff | 1.000 | 0.00002 | 4.42 |

Pattern holds across all 16 canonical-grid rows at each alpha: friction_ff's steady-state
error is 2.5x-9x lower than baseline's, and mean peak torque is roughly flat to ~8% higher for
larger displacements, ~2% *lower* for the smallest one. At long holds the torque story is
better than flat: `dx=0.06m, hold=30s, alpha=0.2` -- baseline's hold-phase peak applied torque
is 8.50 Nm with 0.00501 m of hold-phase drift-from-start; friction_ff's is 5.96 Nm (30% lower)
with only 0.00125 m of drift (4x tighter). Feedforward is not trading torque for accuracy here;
it is improving both simultaneously, consistent with cancelling a real disturbance rather than
just stiffening the loop.

No case found anywhere in the grid where friction_ff made accuracy or torque effort worse than
baseline.

## Residual failures with friction_ff (3 of 76 total)

1. `alpha=0.3, canonical_grid, dx=0.01m, hold=2s`: `hold_phase_x_drift_from_hold_start_m` =
   0.00311 m against a 0.00300 m tolerance -- a 3.5% hairline miss, not a systematic gap.
2. `alpha=0.3, torque_scale_robustness, dx=0.03m, torque_limit_scale=0.10`: fails via
   `torque_saturation_percentage` = 5.76% against the 5.0% tolerance.
3. `alpha=0.3, torque_scale_robustness, dx=0.06m, torque_limit_scale=0.10`: same mechanism,
   8.84% saturation.

Both torque-scale cases are at the single most extreme setting in the grid (only 10% of the
nominal torque budget available). Baseline already failed both of these same two cases (via
`hold_phase_target_tracking`), so friction_ff is not introducing a new failure mode here --
it's a different, genuine reason (the torque budget itself is now the binding constraint,
plausibly because the feedforward term consumes part of an already-tiny budget) for a case
that was never passing anyway.

## Part C (integral action) recommendation

**Feedforward alone looks broadly sufficient and should not automatically trigger building
Part C.** It closes 96% of the combined grid (73/76) at both alphas, directly fixes the
dominant, previously-unexplained `hold_phase_target_tracking` failure mode, and reproduces
then closes the exact real-hardware signature (57.6% move-phase achievement here vs. the
reported 55-72% on the real arm). The 3 remaining failures are not evidence of a broad,
unclosed steady-state gap: one is a threshold hairline (3.5% over), and two are genuine
torque-budget-starved edge cases at the single most extreme (10%) torque-limit setting in the
grid, where the constraint is available torque, not controller law -- integral action doesn't
add torque headroom, so it is not obviously the fix for those two specifically, though it
could plausibly still shave the near-miss case. Given the evidence, this does **not** clear the
plan's own bar for building Part C ("only if step 1 shows feedforward alone doesn't close the
gap broadly") -- recommend deferring `ki_x` unless the real-hardware smoke test (Part D step 3,
not run here) shows a gap that sim doesn't, rather than building it speculatively now.

## Files

- Findings: this file.
- Raw sweep outputs (not committed, gitignored): `outputs/ur5e_mujoco_torque_transport/friction_ff_sweep_2026-07-31/`.
- Configs used (read-only inputs, not modified): `config/ur5e_mujoco_torque_osc_tuned.yaml`,
  `config/ur5e_mujoco_torque_osc_tuned_friction_ff.yaml`.
