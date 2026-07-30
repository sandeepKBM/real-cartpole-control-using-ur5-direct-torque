# Disabling the global cond(J) singular_scale gate: freeze confirmation + full validation sweep

Context: `controller_core/x_axis_cartesian_impedance.py`'s tuned OSC controller has a global
`cond(J)`-based `singular_scale` term (`jacobian_singular_cond_max`, class default `1.0e5`):
when `cond(J)` exceeds this, the *entire* 6-DOF task wrench is scaled down by
`jacobian_singular_cond_max / cond(J)` before `J.T`. AGENTS.md sec 4 already documents this
nulls task authority at the transport start pose (`wrist_2=0`, an exact wrist singularity,
`cond(J)~2.5e17`) and is redundant with `lambda_regularization` (already `0.1` in the tuned
config). A candidate fix config already existed
(`config/ur5e_mujoco_torque_osc_tuned_no_singular_scale.yaml`,
`jacobian_singular_cond_max: 1.0e18`, effectively disabling the gate) but had not been through
a full rigor sweep with real, reproducible numbers in a durable artifact.

Note on a prior claim: commit `7f6d2ed` (2026-07-25, the commit that added this config)'s own
message and the config's in-file comment both claim "full validation sweep run today ...
zero regressions." That claim was never backed by a committed artifact (`outputs/` is
gitignored, not git-recoverable per AGENTS.md sec 1), was never referenced from
`docs/CURRENT_STATUS.md`, and AGENTS.md sec 4 itself — edited 23 minutes *before* that commit,
same night, and never updated after — still lists this as "found, not yet fixed ... needs a
full validation sweep before trusting" as of today. This doc supersedes both: it re-derives
the numbers from scratch rather than relying on the earlier unlogged claim.

## Verdict

**Freeze confirmed real and confirmed fixed. Sweep is clean: zero regressions across every
alpha/category tested, plus a real, measured accuracy improvement in one category.**
Recommendation at the end.

## 1. Freeze confirmation: before/after, real numbers

Reproduced with the exact gentle canonical move from the background investigation:
`tools/ur5e_mujoco_torque_experiments.py --mode controller-rollout --controller-kind impedance
--gravity-mode gravity_comp --trajectory-profile min_jerk_move_hold --target-x-delta 0.02
--move-duration 1.5 --duration 2.5 --seed 0 --no-plot`, `config/ur5e_mujoco_torque_osc_tuned.yaml`
(baseline) vs `config/ur5e_mujoco_torque_osc_tuned_no_singular_scale.yaml` (fixed), default
`height_alpha=0.5` start pose.

### jacobian_cond / singular_scale / |tau_applied| over time

| t (s) | baseline cond(J) | baseline singular_scale | baseline \|tau_applied\| (Nm) | fixed cond(J) | fixed singular_scale | fixed \|tau_applied\| (Nm) |
|---|---|---|---|---|---|---|
| 0.002 | 2.49e17 | 4.02e-13 | 0.000000 | 2.49e17 | 1.0 | 0.000000 |
| 0.126 | 1.13e17 | 8.83e-13 | 0.000000 | 2.02e6 | 1.0 | 0.112576 |
| 0.252 | 5.32e16 | 1.88e-12 | 0.000000 | 2.74e5 | 1.0 | 0.172536 |
| 0.502 | 4.14e14 | 2.41e-10 | 0.000000 | 4.78e4 | 1.0 | 0.055519 |
| 0.752 | 9.79e8  | 1.02e-4  | 0.001287 | 2.10e4 | 1.0 | 0.542870 |
| 1.002 | 1.57e4  | 1.0      | 2.890544 | 1.38e4 | 1.0 | 1.039647 |
| 1.502 | 1.19e4  | 1.0      | 1.149210 | 1.16e4 | 1.0 | 1.146664 |
| 2.500 | 1.22e4  | 1.0      | 1.121216 | 1.19e4 | 1.0 | 1.116169 |

- **Baseline: first `|tau_applied| > 0.01 Nm` at t=0.784s; first `> 0.5 Nm` at t=0.838s.** The
  controller is genuinely frozen (`tau_applied` in the 1e-13-1e-4 Nm range, physically zero)
  for roughly **half the 1.5s move**, only escaping once `cond(J)` numerically decays below
  `jacobian_singular_cond_max=1e5` around t~0.85s.
- **Fixed: first `|tau_applied| > 0.01 Nm` at t=0.028s (13 control steps in); no freeze.**
  `singular_scale` is pinned at `1.0` throughout — `cond(J)` still spans the same 17 orders of
  magnitude (confirming this is the same physical singularity, not a different pose/run), but
  it no longer gates the wrench at all.

### CartesianMoveMonitor replay: raw peak TCP accel estimate

Replayed `ee_pos` from each trace through `hardware.safety.CartesianMoveMonitor`
(`accel_gap_cycles=1`, `speed_lowpass_alpha=1.0` = no filtering, all limits set to `1e6` so
nothing trips) cross-checked against the exact vectorized formula from
`tools/analyze_state_noise_capture.py` (already unit-tested against the live class in
`tests/hardware/test_analyze_state_noise_capture.py`) — both agree exactly since `dt_s` here
is the trace's own recorded sim time, not wall clock (see script header for why that equivalence
holds for a synthetic replay).

| | baseline (singular_scale active) | fixed (no_singular_scale) | change |
|---|---|---|---|
| peak TCP accel estimate | **2.7154 m/s^2** | **0.0584 m/s^2** | **46.5x lower** |
| peak TCP speed estimate | 0.1052 m/s | 0.0255 m/s | 4.1x lower |
| theoretical kinematic peak accel (min-jerk, dx=0.02m/1.5s) | 0.0513 m/s^2 | 0.0513 m/s^2 | — |
| ratio to theoretical peak | 53x | 1.14x | — |

The 2.7154 m/s^2 baseline number independently reproduces the background investigation's 2.72
m/s^2 (same run, same methodology, cross-validated). The fixed config's peak accel (0.0584
m/s^2) is within 14% of the theoretical kinematic peak for this move — i.e. the freeze-then-
catchup dynamic is gone, and the real acceleration profile now tracks the intended min-jerk
trajectory instead of a step-function catchup. At the real default guard threshold
(`max_tcp_accel_mps2=0.5`), the baseline's 2.72 m/s^2 would trip that guard by 5.4x; the fixed
config's 0.058 m/s^2 sits comfortably under it.

Analysis script (not committed as a `tools/` entrypoint per this task's scratch-script
convention): scratchpad copy used interactively, logic summarized above; re-derivable from
`tools/analyze_state_noise_capture.py`'s formula plus the two named configs and the command
above.

## 2. Full four-category rigor sweep

Ran `tools/ur5e_pose_sweep_transport.py` (all four categories: canonical_grid, long_holds,
large_displacements, torque_scale_robustness — 38 runs/alpha/config), baseline
(`config/ur5e_mujoco_torque_osc_tuned.yaml`) vs fixed
(`config/ur5e_mujoco_torque_osc_tuned_no_singular_scale.yaml`), `--seed 0`, gains
auto-extracted from each config (`--gain-overrides-json`, the tool's default, not
`--no-gain-overrides`).

**Scope actually covered: `height_alpha in {0.1, 0.2, 0.3, 0.5}`** — all four values named in
the task (the required 0.5 minimum, plus all three optional ones: 0.1/0.2/0.3). 152 runs per
config, 304 runs total.

### Pass/fail counts (num_valid_move_and_hold / num_runs)

| height_alpha | category | baseline | fixed (no_singular_scale) | regression? |
|---|---|---|---|---|
| 0.1 | canonical_grid | 8/8 | 8/8 | no |
| 0.1 | long_holds | 8/8 | 8/8 | no |
| 0.1 | large_displacements | 8/8 | 8/8 | no |
| 0.1 | torque_scale_robustness | 14/14 | 14/14 | no |
| 0.2 | canonical_grid | **6/8** | **8/8** | no (improvement) |
| 0.2 | long_holds | 8/8 | 8/8 | no |
| 0.2 | large_displacements | 8/8 | 8/8 | no |
| 0.2 | torque_scale_robustness | 14/14 | 14/14 | no |
| 0.3 | canonical_grid | **4/8** | **8/8** | no (improvement) |
| 0.3 | long_holds | 8/8 | 8/8 | no |
| 0.3 | large_displacements | 8/8 | 8/8 | no |
| 0.3 | torque_scale_robustness | 14/14 | 14/14 | no |
| 0.5 | canonical_grid | **4/8** | **8/8** | no (improvement) |
| 0.5 | long_holds | 8/8 | 8/8 | no |
| 0.5 | large_displacements | 8/8 | 8/8 | no |
| 0.5 | torque_scale_robustness | 12/14 | 12/14 | no (identical failure) |
| **Total** | | **140/152** | **152/152** | **zero regressions, +12** |

### What the baseline canonical_grid failures actually are

At `alpha in {0.2, 0.3, 0.5}`, baseline fails the *smallest*-displacement cases
(`target_x_delta=0.01`, and at `alpha=0.3` also `0.02`), both hold durations, always via
`move_phase_target_tracking` — i.e. with only a 1.0s move duration (the canonical-grid
duration), the freeze eats enough of the window that the smallest, most freeze-time-sensitive
moves don't finish tracking in time. `alpha=0.1` is unaffected (baseline 8/8) — at that pose
`cond(J)` apparently doesn't cross `jacobian_singular_cond_max=1e5` enough to matter for a
1.0s move. The fixed config passes all of these (8/8 everywhere) since there's no freeze to eat
the window. This is exactly the mechanism predicted in part 1: a shorter move duration is more
exposed to the fixed ~0.7-0.8s freeze cost than the 1.5s move used there.

### The one identical failure (not a regression)

`height_alpha=0.5`, `torque_scale_robustness`, `torque_limit_scale=0.1`, both `target_x_delta`
values (0.03/0.06): both configs fail identically (`move_phase_incomplete`/
`hold_phase_incomplete`). At 10% of the torque limit, this is a genuine torque-budget
saturation case unrelated to `singular_scale` (already `1.0` at this well-conditioned pose
regardless of the gate) — confirms the sweep is measuring real physics, not just re-running
the same non-differentiating scenario.

### Not covered

- `height_alpha` values outside `{0.1, 0.2, 0.3, 0.5}` (e.g. `0.0`, `0.4`, `0.6-1.0`) were not
  swept here.
- Only `--seed 0` was used (matches this project's existing convention — the sim has no noise
  source in this pipeline, so seed has no effect; not independently re-verified in this pass).
- `torque_task_qp.py` has the identical `singular_scale` pattern per AGENTS.md and was not
  touched or evaluated — out of scope, not the default controller.
- No real-hardware or URScript-path validation — sim-only, per the existing convention for this
  entire controller-math lane.

## 3. Test suite

`python -m pytest -q tests/ -m "not slow"`: **411 passed, 1 deselected** — matches the stated
baseline exactly, no regressions from this investigation (no library code was touched; only
sweeps were run and a scratch analysis script was used, not committed under `tools/`).

## 4. Recommendation

Evidence-backed recommendation: **yes, promote `config/ur5e_mujoco_torque_osc_tuned_no_singular_scale.yaml`
to the default.** Across 4 poses x 4 categories (304 runs total) it strictly dominates the
current default — zero regressions, +12 passing runs, all in cases explained by a confirmed,
measured mechanism (a ~0.7-0.8s freeze at the wrist-adjacent poses this project's entire pose
family sits near). The freeze itself is independently reproduced and quantified (46.5x lower
raw peak TCP accel, tracking the intended min-jerk profile to within 14% instead of 53x
overshooting it) and directly explains why real-hardware TCP-accel guard thresholds have been
hard to set correctly. This is a decision for a human to make, not something this investigation
acts on unilaterally — per task instructions, the default config
(`config/ur5e_mujoco_torque_osc_tuned.yaml`) was left unchanged, AGENTS.md was not edited, and
no tool defaults were repointed.
