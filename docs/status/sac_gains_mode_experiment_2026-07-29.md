# SAC in gains mode at height_alpha=0.5: a fifth real RL attempt, still behind both fixed-gain controllers

Context: AGENTS.md sec 3 documents a real, structural controller weakness -- at
`height_alpha=0.5`, the fixed-gain OSC controller passes `+0.20m` transport moves but fails
`-0.20m` via orientation error. Four prior real PPO training attempts (`rl_gain_scheduling/`)
tried to fix this via a state-conditioned gain/residual-torque policy: 0/8 (gains mode), 2/8
(residual, unpenalized), 5/8 (residual, penalized, buggy orientation threshold), 0/8 (residual,
penalized, threshold bug fixed -- `docs/status/rl_gain_scheduling_alpha05_directional_fix_2026-07-29.md`).
Separately, a structural (non-RL) fix landed the same day: `controller.wrist_orientation_task`
(commit `bd5bba3`, `config/ur5e_mujoco_torque_osc_tuned_wrist_orient.yaml`,
`docs/status/wrist_orientation_task_2026-07-29.md`), which fixes the exact orientation bug the
RL work was chasing and is now the best available fixed-gain option.

This note covers a fifth real attempt: **SAC (not PPO), in `env.action_mode: "gains"`** (the
policy outputs all 11 schedulable Cartesian-impedance gains every control step -- the original,
most-flexible architecture, matching the very first PPO attempt's 0/8 result). SAC had never
been run for a real training attempt in this project before this task -- only a 2-step smoke
test existed (`tests/mujoco/test_train_gain_scheduler.py::test_sac_smoke_train_two_steps`).

## Verdict

**SAC in gains mode underperforms both fixed-gain baselines, including the old, documented-broken
one.** Real evaluation on the canonical `height_alpha=0.5`, `dx in {-0.20..-0.05, 0.05..0.20}`
(8-cell) grid:

| Comparison | learned (SAC gains) | baseline | learned rate | baseline rate |
|---|---|---|---|---|
| vs `ur5e_mujoco_torque_osc_tuned_adaptive_lambda.yaml` (old, documented-broken baseline -- the one all four PPO attempts compared against) | 2/8 valid | 7/8 valid | **25%** | 88% |
| vs `ur5e_mujoco_torque_osc_tuned_wrist_orient.yaml` (today's new structural fix -- the real bar) | 2/8 valid | 8/8 valid | **25%** | **100%** |

SAC does modestly better than the first (gains-mode) PPO attempt's 0/8, but that is the only
positive comparison available -- it is far below both fixed-gain controllers, and the
wrist-orientation-task controller (a controller-design fix, not RL) now clears this exact grid
perfectly (8/8) while SAC does not.

**The failure mode is informative and was checked directly, not just inferred from the pass/fail
count**: contrary to the pattern in prior RL attempts (orientation-error and axis-error-growth
guard trips), SAC's own failures are *not* safety-guard trips at all in the negative-dx cells --
orientation error stays low throughout (0.008-0.031 rad, well under the 0.25 rad guard) and every
one of those episodes runs to `duration_complete`. They fail `valid_move_and_hold` purely on
**position undershoot**: `dx=-0.20m` only achieves `-0.169m` (15.5% short), `dx=-0.15m` achieves
`-0.120m`, `dx=-0.10m` achieves `-0.079m`, `dx=-0.05m` achieves `-0.042m` -- a systematic,
magnitude-scaled undershoot in the `-X` direction, not a safety failure. The two large positive-dx
failures (`dx=0.15m`, `dx=0.20m`) do trip the same `|axis_error| grew for 100 consecutive steps`
guard the fourth PPO (residual) attempt hit, consistent with that prior note's hypothesis of a
high-frequency oscillation the guard is sensitive to. Only `dx=0.05m` and `dx=0.10m` (the two
smallest positive moves) pass cleanly.

## Evidence

### 1. Config and setup

New config `config/rl_gain_scheduling_sac_gains_alpha05.yaml`, built from
`config/rl_gain_scheduling_alpha05_bidirectional_adaptive_lambda_residual_penalized_safety_fix.yaml`
(the CORRECT, bug-fixed env: `max_orientation_error_rad=0.25`, `height_alpha_range=[0.5,0.5]`,
`target_x_delta_range_m=[-0.20,0.20]`, `lambda_diagonal_shaping`/`lambda_adaptive_regularization`
active) -- NOT `config/rl_gain_scheduling_sac.yaml` (predates both the orientation-threshold fix
and the adaptive-lambda work, intentionally preserved as historical reference only). Changes from
the source config: `env.action_mode: "gains"` (residual_torque block and
`reward.residual_magnitude_weight` removed, since neither applies in gains mode), and the
`training` block replaced with a `training.sac.*` block (field names/shape copied from
`config/rl_gain_scheduling_sac.yaml`'s own template; values chosen independently:
`learning_rate=3e-4`, `buffer_size=300000`, `learning_starts=10000`, `batch_size=256`,
`polyak_tau=0.005`, `gamma=0.99`, `train_freq=1`, `gradient_steps=1`, `ent_coef="auto"`,
`target_update_interval=1`, `target_entropy="auto"`, `net_arch=[128,128]`).

`rl_gain_scheduling/eval_gain_scheduler.py` hardcoded `PPO.load(...)`, which raises on a SAC
checkpoint (different policy class). Added `--algo {ppo,sac}` (default `"ppo"`, byte-identical
existing behavior for PPO models) and a `model_cls` lookup so the same eval driver works for both.

### 2. Compute

Run entirely on `westeros.cs.rutgers.edu`, found idle at launch time (`uptime` load average
2.2-3.8 on 72 cores, `mpstat` 96.9% idle CPU, no per-user memory cgroup cap on this host) --
per AGENTS.md sec 8's "check real capacity first" guidance, this made the ilab4 SSH lane
(with its documented nohup/Linger fragility) unnecessary; training ran as a direct local process
under the harness's own background-job management, not over SSH, so that specific failure mode
does not apply here. `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
NUMEXPR_NUM_THREADS=1` exported before every launch per the same section's BLAS-thread-explosion
warning. `n_envs=24` throughout (the memory-safe value validated on ilab4 earlier the same day),
kept conservative to leave headroom for the parallel residual_torque SAC experiment sharing the
same host.

### 3. Pilot (80,000 timesteps, `--run-name sac_gains_alpha05_pilot`)

Inspected before committing to the full budget, per the config's own pilot-first comment:
80,016 timesteps in ~70s (fps ~1020-1125). No NaNs. `ent_coef` decayed gradually and smoothly
(0.775 -> 0.461 over the run, not an instant collapse to 0). `critic_loss` stable (0.13-0.20).
`actor_loss` trended more negative over training (-29 -> -58), the expected shape for SAC as Q-value
estimates grow. Judged sane -- proceeded to the full run without changes.

### 4. Full run (3,000,000 timesteps, `--run-name sac_gains_alpha05`)

Ran to completion: 3,000,000 / 3,000,000 timesteps in 3185s (~53 min), fps 920-950 throughout,
verified by loading the saved model and reading `model.num_timesteps == 3000000`. Final model:
`outputs/rl_gain_scheduling/sac_gains_alpha05/models/sac_gain_scheduler_final.zip`. `critic_loss`
fluctuated more than in the pilot (spiked to ~4.0 mid-run, back to ~0.08 by the end) but never
diverged/NaN'd; `ent_coef` continued decaying smoothly to ~0.003-0.005 by the end (healthy
auto-entropy-tuning behavior, not premature collapse to exactly 0). TensorBoard logs under
`outputs/rl_gain_scheduling/sac_gains_alpha05/tb/`.

(A cosmetic multiprocessing-tempdir-cleanup `OSError` appears in the log immediately after
"Saved final model" in both the pilot and full run -- unrelated to training; the model save and
`SubprocVecEnv` teardown both completed successfully before it, same as observed independently in
other recent work on this repo.)

### 5. Evaluation

`rl_gain_scheduling/eval_gain_scheduler.py --model-path outputs/rl_gain_scheduling/sac_gains_alpha05/models/sac_gain_scheduler_final.zip --algo sac --config config/rl_gain_scheduling_sac_gains_alpha05.yaml --alphas 0.5 --deltas -0.20 -0.15 -0.10 -0.05 0.05 0.10 0.15 0.20`, run twice (once per `--baseline-config`), matching the exact grid used in
`docs/status/rl_gain_scheduling_alpha05_directional_fix_2026-07-29.md`.

**vs `ur5e_mujoco_torque_osc_tuned_adaptive_lambda.yaml`** (old baseline, direct comparability
with all four prior PPO attempts):

| dx (m) | learned valid | learned quality | baseline valid | baseline quality |
|---|---|---|---|---|
| -0.20 | False | 0.207 | False (documented bug) | 0.270 |
| -0.15 | False | 0.180 | True | 0.362 |
| -0.10 | False | 0.195 | True | 0.461 |
| -0.05 | False | 0.248 | True | 0.611 |
| +0.05 | True | 0.534 | True | 0.631 |
| +0.10 | True | 0.362 | True | 0.519 |
| +0.15 | False | 0.252 | True | 0.438 |
| +0.20 | False | 0.194 | True | 0.342 |

learned: 2/8 (25%), baseline: 7/8 (88%).

**vs `ur5e_mujoco_torque_osc_tuned_wrist_orient.yaml`** (today's structural fix -- the real bar):

| dx (m) | learned valid | learned quality | baseline valid | baseline quality |
|---|---|---|---|---|
| -0.20 | False | 0.207 | True | 0.326 |
| -0.15 | False | 0.180 | True | 0.431 |
| -0.10 | False | 0.195 | True | 0.536 |
| -0.05 | False | 0.248 | True | 0.676 |
| +0.05 | True | 0.534 | True | 0.729 |
| +0.10 | True | 0.362 | True | 0.669 |
| +0.15 | False | 0.252 | True | 0.600 |
| +0.20 | False | 0.194 | True | 0.489 |

learned: 2/8 (25%), baseline: **8/8 (100%)**.

The learned side's per-cell numbers are identical across both eval runs (same model, same env
config -- only the baseline side differs, as expected).

### 6. Failure-mode diagnosis (per-cell `summary.json`, not just pass/fail counts)

| dx (m) | termination_reason | move-phase max orientation err (rad) | achieved_x_delta_m | target x tol failure? |
|---|---|---|---|---|
| -0.20 | duration_complete | 0.031 | -0.169 (target -0.20) | undershoot |
| -0.15 | duration_complete | 0.023 | -0.120 (target -0.15) | undershoot |
| -0.10 | duration_complete | 0.016 | -0.079 (target -0.10) | undershoot |
| -0.05 | duration_complete | 0.008 | -0.042 (target -0.05) | undershoot |
| +0.05 | duration_complete | 0.009 | 0.048 | pass |
| +0.10 | duration_complete | 0.017 | 0.095 | pass |
| +0.15 | `\|axis_error\| grew for 100 consecutive steps` | 0.026 | 0.146 | guard trip |
| +0.20 | `\|axis_error\| grew for 100 consecutive steps` | 0.038 | 0.197 | guard trip |

Two distinct, real failure modes, neither of which is the orientation-error mechanism the RL work
was originally chasing: (1) all four negative-dx cells run safely to completion but the policy
never actually reaches the target X displacement (a systematic, magnitude-scaled undershoot,
worst at 15.5% short on the largest move) -- `valid_move_and_hold`'s position tolerance is the
metric that fails, not any safety guard; (2) the two largest positive-dx cells trip the same
`|axis_error|`-growth guard the fourth PPO (residual) attempt hit, consistent with that note's
unverified hypothesis about high-frequency oscillation not being damped enough by the existing
smoothness reward terms.

## Recommendation

- **Do not adopt this policy.** 25% valid vs the old baseline's 88% and the new structural fix's
  100% on the same grid -- underperforms both, and by a wide margin against the current best
  option (`wrist_orientation_task`).
- **This is a real, useful negative result, not a training bug.** The pilot was sane, the full
  3M-step run completed cleanly with no NaNs/divergence, and `num_timesteps` was verified against
  the saved model. The failure is behavioral, not an artifact of a broken run.
- **SAC in gains mode is a genuinely different failure signature than the prior PPO attempts**:
  it never trips the orientation guard that motivated this whole investigation, but instead
  undershoots the target position outright on the harder (negative) direction -- worth keeping in
  mind if a future attempt tries reward shaping specifically for position-tracking fidelity in
  gains mode, separate from the orientation-safety shaping used so far.
- **Combined with the four prior PPO attempts and (per the parallel residual-torque SAC
  experiment, tracked separately) presumably a sixth data point, five-plus real RL attempts at
  this problem, across two algorithms and two action-space designs, have not beaten the
  fixed-gain controller.** The wrist-orientation-task structural fix (non-RL,
  `docs/status/wrist_orientation_task_2026-07-29.md`) remains the best available solution to the
  original height_alpha=0.5 directional-asymmetry problem; nothing here changes that conclusion,
  and this result adds further evidence for it.

## Files changed

- `config/rl_gain_scheduling_sac_gains_alpha05.yaml` -- new config (see sec 1 above).
- `rl_gain_scheduling/eval_gain_scheduler.py` -- added `--algo {ppo,sac}` support (default
  `"ppo"`, byte-identical existing behavior); `PPO` import extended to `PPO, SAC` and
  `BaseAlgorithm` used for the `_run_learned` type hint (no behavior change, just a type that
  covers both algorithms).
- This file.

Not touched: `AGENTS.md`, any `README.md`, `hardware/` (simulation-only task), any file belonging
to the parallel residual_torque SAC experiment (`config/rl_gain_scheduling_sac_residual_alpha05.yaml`,
`docs/status/sac_residual_torque_experiment_2026-07-29.md` -- both observed to be actively
modified by that other process during this session and left alone).

## Tests run

- `pytest tests/mujoco/test_train_gain_scheduler.py -q` -- 6 passed (includes the pre-existing
  `test_sac_smoke_train_two_steps`, unaffected by the eval-script change).
- `pytest tests/unit -m unit -q` -- 94 passed.
- `pytest tests/mujoco -m mujoco -q -k "not slow"` -- 83 passed.

## Tests not run

- No hardware-in-the-loop or real-RTDE tests (out of scope -- simulation only).
- Full repo-wide `pytest -q` was not re-run in this task (the unit + mujoco subsets above cover
  everything this change could plausibly affect: `eval_gain_scheduler.py` and the new config).
  The one known pre-existing failure unrelated to this work
  (`tests/hardware/test_direct_torque_transport_timing.py::test_transport_records_timing_and_deadline_loop`,
  traced in `docs/status/wrist_orientation_task_2026-07-29.md` to the unrelated
  `hardware/local_dynamics.py` Pinocchio fast-path commit) was not re-verified here since this
  task never touches that path.

## Rollback

```
git revert <this-commit-sha>
```
or, to remove without a revert commit:
```
git checkout <previous-sha> -- rl_gain_scheduling/eval_gain_scheduler.py
rm config/rl_gain_scheduling_sac_gains_alpha05.yaml docs/status/sac_gains_mode_experiment_2026-07-29.md
```
`--algo` defaults to `"ppo"` and reproduces the eval script's original behavior exactly, so
simply not passing `--algo sac` (and not referencing the new config) is itself a full functional
rollback without touching any code. The trained model/logs under
`outputs/rl_gain_scheduling/sac_gains_alpha05*/` are gitignored artifacts, not tracked -- no
rollback needed for those.
