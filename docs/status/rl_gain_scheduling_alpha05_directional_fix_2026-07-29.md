# RL gain-scheduling at height_alpha=0.5: training-env mismatch fix, and a fourth real training attempt

Context: AGENTS.md sec 3's 2026-07-27/28 finding documents a real, confirmed, still-open
controller bug -- at `height_alpha=0.5`, the tuned OSC config passes `+0.20m` cleanly but
fails `-0.20m` via orientation error at half the peak wrist_2 excursion, root-caused to the
nullspace-posture projector's restoring authority being asymmetric with wrist_2 sign (not a
gain-tuning problem). Three prior real RL training attempts (gains-mode, residual-torque
unpenalized, residual-torque penalized) scored 0/8, 2/8, 5/8 valid respectively against this
exact grid, all underperforming the fixed-gain baseline's 7/8 (88%). This note covers (1) an
audit of whether the RL training environment actually reproduces the real bug, (2) a found and
fixed mismatch, and (3) one real training attempt on the corrected config plus its evaluation.

## Verdict

**A real training/eval config mismatch was found and fixed: all four `rl_gain_scheduling_
alpha05_bidirectional*.yaml` configs set `controller.safety.max_orientation_error_rad: 0.35`,
but the config that actually produces the documented finding
(`config/ur5e_mujoco_torque_osc_tuned_adaptive_lambda.yaml`, also matched by `tuned.yaml` and
`tuned_diagonal_lambda.yaml`) uses `0.25`.** This is confirmed real, not cosmetic: this field
drives `ImpedanceSafetyMonitor`'s live guard, so during training, episodes with orientation
error between 0.25 and 0.35 rad -- exactly the band the documented bug lives in (baseline
trips at 0.2498 rad on `-0.20m`) -- ran to `duration_complete` instead of guard-terminating
with the -25.0 terminal safety penalty, weakening the training signal for the exact failure
mode being targeted. (`valid_move_and_hold`'s own pass/fail check uses a hardcoded 0.25 rad
tolerance in `transport_metrics.py`, independent of this config field, so past *eval* numbers
were not affected by this bug -- only training-time guard/reward behavior was.) Every other
controller field (gains, `lambda_diagonal_shaping`, `lambda_adaptive_regularization`,
`height_alpha_range: [0.5,0.5]`, `target_x_delta_range_m: [-0.20,0.20]`) already matched.

**Fixing this mismatch and running a fourth real training attempt did NOT fix the underlying
problem -- it made the outcome worse.** The RL policy trained on the corrected config scores
**0/8 valid_move_and_hold**, down from this exact architecture's own prior (mismatched-config)
result of 5/8, and worse than the fixed-gain baseline's 7/8 (88%) on the same grid. On the
real `-0.20m` case specifically, the corrected-config policy is still invalid -- it does not
even fail via the documented orientation mechanism (its max orientation error there is only
0.13 rad, well under the 0.25 rad limit that trips the baseline), it fails a different,
indiscriminate guard (`|axis_error| grew for 100 consecutive steps`) that now trips on
literally all 8 cells, including previously-trivial cases like `dx=+0.05m`. **Honest answer:
no, the RL policy does not beat the fixed-gain baseline on the real `-0.20m` case, or anywhere
else in this grid, after this fix.** The training-env correction was real and worth making,
but it is not sufficient -- three (now four) real attempts at this problem, at this budget,
with this reward shape, have not produced a policy that outperforms the fixed-gain controller.

## Evidence

### 1. Mismatch audit (`rl_gain_scheduling/gain_scheduling_env.py` + all four `alpha05_bidirectional*` configs)

Field-by-field diff of `controller:` blocks, `config/ur5e_mujoco_torque_osc_tuned_adaptive_
lambda.yaml` (source of the documented finding) vs `config/rl_gain_scheduling_alpha05_
bidirectional_adaptive_lambda.yaml` (training env):

```
DIFF safety: baseline={..., 'max_orientation_error_rad': 0.25}
             rl_env ={..., 'max_orientation_error_rad': 0.35}
```

(`qp_posture_regularization`/`qp_enforce_velocity_bounds` also differ but are QP-controller-
only fields never read by `GainSchedulingEnv`, which always uses `controller_kind="impedance"`
-- irrelevant.) The same `0.35` appears in `rl_gain_scheduling_alpha05_bidirectional.yaml`,
`..._adaptive_lambda_residual.yaml`, `..._adaptive_lambda_residual_penalized.yaml`, and the
un-scoped default `config/rl_gain_scheduling.yaml` -- a stray value that predates the
alpha05-scoped family and was never revisited when they branched off. `hardware/poses.py`'s
`ACTIVE_ORIGIN_Q`/`LOWER_B_Q` do match the constants `gain_scheduling_env.py` defines inline,
and `height_alpha_range: [0.5, 0.5]` / `target_x_delta_range_m: [-0.20, 0.20]` are correct in
all four configs -- so this orientation-tolerance field was the one real remaining gap.

Confirmed which three prior attempts already used the correct lambda-flag config (i.e. the
mismatch that mattered for those specific outcomes is orientation-tolerance only, not the
lambda flags) via `outputs/rl_gain_scheduling/eval/2026072{89}_*` run records:

| run (config) | `config_path` used by learned side | `config_path` used by baseline side | learned valid |
|---|---|---|---|
| `alpha05_bidirectional_fix` (plain, mismatched from the start) | `..._bidirectional.yaml` | `ur5e_mujoco_torque_osc_tuned.yaml` | 2/2 (bug not even reproduced -- not one of the "three" counted attempts) |
| `alpha05_bidirectional_adaptive_lambda_fix` | `..._adaptive_lambda.yaml` | `..._osc_tuned_adaptive_lambda.yaml` | 0/8 |
| `alpha05_bidirectional_adaptive_lambda_residual_fix` | `..._adaptive_lambda_residual.yaml` | `..._osc_tuned_adaptive_lambda.yaml` | 2/8 |
| `alpha05_bidirectional_adaptive_lambda_residual_penalized_fix` | `..._residual_penalized.yaml` | `..._osc_tuned_adaptive_lambda.yaml` | 5/8 |

Baseline in all three real attempts: 7/8 (88%), matching the documented finding (fails
exactly `dx=-0.20m` via `||orientation error|| > 0.25 rad`).

### 2. Sanity check (fixed-gain baseline through the corrected pathway, before spending training compute)

Ran `tools/ur5e_mujoco_torque_experiments.py --config config/ur5e_mujoco_torque_osc_tuned_
adaptive_lambda.yaml` directly at `height_alpha=0.5` (`q_start` interpolated the same way
`gain_scheduling_env.py`/`eval_gain_scheduler.py` do), `dx=-0.20m`, move-duration 1.0s,
duration 3.0s, seed 0:

```
move_phase_max_abs_orientation_error_rad: 0.24975217591871768
termination_reason: "||orientation error|| > 0.25 rad"
valid_move_and_hold: false
```

Matches the documented finding exactly (guard trips essentially at the 0.25 rad threshold).
This confirmed the fix target (`config/ur5e_mujoco_torque_osc_tuned_adaptive_lambda.yaml`,
0.25 rad) before any training was launched.

### 3. The fix: new config

`config/rl_gain_scheduling_alpha05_bidirectional_adaptive_lambda_residual_penalized_safety_
fix.yaml` -- exact copy of the best-performing prior config (residual-torque,
magnitude-penalized, 5/8) with only `controller.safety.max_orientation_error_rad` corrected
from `0.35` to `0.25`. No other field touched; the source config is preserved unmodified per
this project's config-preservation rule.

### 4. Training run

PPO, `--n-envs 8` (deliberately small: at launch time on `westeros`, `uptime`/`top`/`mpstat`
showed load average ~76-79 on 72 cores and ~93-95% CPU already in use by another user's real
job (`st1122`'s `run_adaptive.py`, ~66 cores across two processes) -- not idle, so the
standing "use up to 80% of cores" instruction was scaled down per its own "don't disturb an
already-running job" caveat). That westeros run degraded under contention (fps dropped from
448 to 17) and was killed before completing (last checkpoint: 40,960 of 3,000,000 steps).
The coordinator re-ran the identical config on `ilab4.cs.rutgers.edu` (same NFS home/conda
env, no path differences), which completed cleanly: **3,022,848 / 3,000,000 timesteps**,
verified independently by loading the saved model
(`outputs/rl_gain_scheduling/alpha05_bidirectional_adaptive_lambda_residual_penalized_
safety_fix_ilab4_v4/models/ppo_gain_scheduler_final.zip`) and reading `model.num_timesteps`,
and by the TensorBoard event file's own embedded hostname
(`events.out.tfevents.1785312295.ilab4.cs.rutgers.edu.1126393.0`). Same `total_timesteps`
(3,000,000) as all three prior real attempts -- same scale, not silently bigger or smaller.

### 5. Evaluation

`rl_gain_scheduling/eval_gain_scheduler.py --model-path <ilab4 final model> --config config/
rl_gain_scheduling_alpha05_bidirectional_adaptive_lambda_residual_penalized_safety_fix.yaml
--baseline-config config/ur5e_mujoco_torque_osc_tuned_adaptive_lambda.yaml --alphas 0.5
--deltas -0.20 -0.15 -0.10 -0.05 0.05 0.10 0.15 0.20`:

| dx (m) | learned valid | learned quality | baseline valid | baseline quality |
|---|---|---|---|---|
| -0.20 | **False** | 0.312 | False (documented bug) | 0.270 |
| -0.15 | False | 0.413 | True | 0.362 |
| -0.10 | False | 0.535 | True | 0.461 |
| -0.05 | False | 0.667 | True | 0.611 |
| +0.05 | False | 0.663 | True | 0.631 |
| +0.10 | False | 0.542 | True | 0.519 |
| +0.15 | False | 0.433 | True | 0.438 |
| +0.20 | False | 0.339 | True | 0.342 |

**learned: 0/8 (0%), baseline: 7/8 (88%).**

Per-cell inspection of the learned side's `summary.json` shows the failure is *not* a
tracking problem: `achieved_x_delta_m` is within ~0.001 m of `target_x_delta` in all 8 cells,
and `final_x_error_m` is ~0 in all 8. Every cell instead fails with `termination_reason:
"|axis_error| grew for 100 consecutive steps"` (`controller_core/safety.py`'s
`max_axis_error_growth_steps=100` guard -- a monotonic-growth-streak check independent of the
orientation guard). Orientation error stays low throughout (max 0.13 rad on the worst,
`dx=+0.20m`, cell) -- the corrected orientation threshold is never what trips this policy. So
the fix changed *which* guard the policy fails on `-0.20m` (orientation to axis-error-growth)
without producing net progress, and simultaneously broke seven cells the same architecture's
prior (mismatched-config) run had passed cleanly.

## Recommendation

- **Do not adopt this policy.** It underperforms the fixed-gain baseline on every case in the
  grid, including the ones the baseline already handles cleanly.
- **The mismatch fix itself is still correct and worth keeping** -- training against a
  config that doesn't match the documented failure's own safety envelope was a real bug, and
  `config/rl_gain_scheduling_alpha05_bidirectional_adaptive_lambda_residual_penalized_safety_
  fix.yaml` is the first config in this family that reproduces the real problem faithfully.
  But it demonstrates the mismatch was not the reason the prior attempts underperformed --
  fixing it made the outcome worse, not better, which is real information: the residual-torque
  + magnitude-penalty architecture and reward shape used across all four real attempts has not
  found a policy that beats the fixed-gain controller here, independent of this particular
  config bug.
- **Before another training attempt**, the axis-error-growth guard trip deserves its own
  trace-level investigation (e.g. `trace.jsonl` for the `dx=+0.05m` cell, previously a trivial
  pass) -- it looks consistent with a high-frequency residual-torque oscillation on the X axis
  that the existing `torque_smooth_weight` (0.001) isn't damping enough, but that is a
  hypothesis, not verified here.
- **Given four real attempts have now underperformed**, AGENTS.md's own assessment stands:
  fixing the underlying directional asymmetry may need a different orientation-holding
  mechanism (controller-design work), not more RL training at this scale/architecture. This
  note does not change that recommendation -- if anything it adds evidence for it.
