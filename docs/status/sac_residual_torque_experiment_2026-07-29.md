# SAC in residual_torque action mode at height_alpha=0.5: does the algorithm change the outcome?

Context: four real PPO training attempts at the `height_alpha=0.5` directional-asymmetry bug
(AGENTS.md sec 3, "the ceiling is directional") all underperformed the fixed-gain baseline --
0/8 (gains), 2/8 (residual, unpenalized), 5/8 (residual, penalized, training-config
orientation-threshold bug), 0/8 (residual, penalized, bug fixed -- see
`docs/status/rl_gain_scheduling_alpha05_directional_fix_2026-07-29.md`). `--algo sac` support
was built into `rl_gain_scheduling/train_ppo_gain_scheduler.py` earlier but never exercised
beyond a 2-step smoke test (`tests/mujoco/test_train_gain_scheduler.py::
test_sac_smoke_train_two_steps`). This is the first real SAC training attempt in this project,
run in parallel with a second, independent experiment (`--action-mode gains`,
`config/rl_gain_scheduling_sac_gains_alpha05.yaml`, `--run-name sac_gains_alpha05`, launched
by a separate agent on `westeros`) as an autonomous overnight run. This note covers the
`residual_torque` half: same task, same bounded-residual architecture as the 4th (best-designed)
PPO attempt, only the algorithm (SAC vs PPO) differs -- the most directly comparable
one-variable-changed experiment available.

Same day, a different, structural fix landed separately:
`controller.wrist_orientation_task` (`config/ur5e_mujoco_torque_osc_tuned_wrist_orient.yaml`,
`docs/status/wrist_orientation_task_2026-07-29.md`), which already fixes the exact directional
asymmetry the RL work has been chasing (peak orientation error at this pose drops from
0.22-0.25 rad to ~0.06-0.07 rad, zero regressions). That config is evaluated here as the real
bar to clear, alongside the original broken baseline for direct comparability with the 4th PPO
attempt's exact 0/8 result.

## Verdict

TBD -- filled in after the full 3,000,000-step run and grid evaluation complete.

## Setup

- New config: `config/rl_gain_scheduling_sac_residual_alpha05.yaml` -- exact copy of
  `config/rl_gain_scheduling_alpha05_bidirectional_adaptive_lambda_residual_penalized_
  safety_fix.yaml` (the corrected-orientation-threshold config the 4th PPO attempt used) for
  every `mujoco:`/`controller:`/`env:`/`reward:` field: `env.action_mode: "residual_torque"`,
  `env.residual_torque.fixed_gains`/`max_nm` (`[3.0, 3.0, 3.0, 1.5, 1.5, 1.5]` Nm),
  `reward.residual_magnitude_weight: 0.05`, `controller.safety.max_orientation_error_rad: 0.25`,
  `height_alpha_range: [0.5, 0.5]`, `target_x_delta_range_m: [-0.20, 0.20]`. Only the
  `training:` block differs -- PPO's on-policy hyperparameters replaced with `training.sac.*`
  (field names/shapes from `config/rl_gain_scheduling_sac.yaml`'s own template): learning_rate
  3e-4, buffer_size 200000, learning_starts 5000, batch_size 256, polyak_tau 0.005, gamma 0.99,
  train_freq 1, gradient_steps 1, ent_coef "auto" (automatic entropy tuning), target_entropy
  "auto", target_update_interval 1. `n_envs: 24` per AGENTS.md sec 8's memory-safe guidance for
  `ilab4.cs.rutgers.edu` (validated ~19GB steady RSS for a comparable PPO probe earlier the same
  day; this run's own measured cgroup memory usage stayed near 13GB throughout -- see below).
- `rl_gain_scheduling/eval_gain_scheduler.py` already had `--algo {ppo,sac}` support added by
  the parallel gains-mode agent (shared infrastructure, not this experiment's own file) before
  this experiment needed it -- verified working via the pilot sanity eval below, no further
  edit made here.
- System state checked immediately before launch (per AGENTS.md sec 8, don't assume headroom):
  `ilab4` load average ~7.5-8.0/96 cores, cgroup memory cap `87409295360` bytes (~81.4GiB),
  ~466GiB of 1TiB system RAM already used by other users' jobs (irrelevant -- the cgroup cap is
  the real ceiling), no other `ss5772` job running on `ilab4` at launch time (the parallel
  gains-mode SAC experiment was running on `westeros` instead, confirmed via its own config's
  header comment and `ps` checks across `ilab1`-`ilab4`).

## Pilot (80,000 steps, `--run-name sac_residual_alpha05_pilot`)

Ran to completion: 72,000/80,000 timesteps actually executed before the final checkpoint save
(SB3 rounds down to the last full `n_envs`-sized batch under `total_timesteps`), model saved
successfully. Sustained throughput **~1200-1360 fps** (env-steps/sec across all 24 parallel
workers) -- CPU-bound MuJoCo stepping dominates, SAC's gradient-update overhead (small MLP,
batch_size 256) is negligible against it. At this rate, 3,000,000 steps projects to roughly
40-45 minutes wall clock -- comfortably inside the available compute window, so no reason to
deviate from the 3,000,000-step full-run budget used by all four prior PPO attempts.

One benign issue: a `shutil.rmtree` `OSError: [Errno 39] Directory not empty` traceback prints
after "Saved final model" / "TensorBoard logs under ..." -- this is Python's multiprocessing
`util._run_finalizers()` failing to clean up a `SubprocVecEnv` worker's `/tmp/pymp-*` scratch
directory at interpreter shutdown, after the model was already saved and the process's own exit
code was 0. Not a training failure; not investigated further (interpreter-shutdown cleanup
noise, unrelated to `controller_core`/`gain_scheduling_env.py`, out of scope for this task).

**Residual-magnitude sanity check** (the specific failure mode flagged in the task -- did the
policy learn to output near-zero residual and just defer to the baseline controller?): computed
mean/max `|residual_tau|` per joint from the pilot policy's own trace at the real `dx=-0.20m`
and `dx=+0.20m`, `height_alpha=0.5` cells (`residual_tau` logged every step in
`gain_scheduling_env.py`'s lightweight trace):

| cell | steps | mean \|residual\| (Nm) | max \|residual\| (Nm) |
|---|---|---|---|
| dx=-0.20m | 352 | 0.048 | 0.091 |
| dx=+0.20m | 1121 | 0.068 | 0.168 |

Nonzero but small relative to the `[3.0, 3.0, 3.0, 1.5, 1.5, 1.5]` Nm bounds (using roughly
2-11% of available authority at 72k/3M steps) -- consistent with an early-stage, high-entropy
SAC policy still exploring (`ent_coef` was 0.33-0.58 at this point in training, well above its
eventual annealed value), not yet either a confident correction or a confirmed collapse to
zero. This is checked again on the fully-trained policy below, which is the number that
actually matters for the "did it learn to do nothing" question.

Pilot sanity eval (`--alphas 0.5 --deltas -0.20 0.20`, `--baseline-config ..._adaptive_lambda.
yaml`): learned 0/2, baseline 1/2 (fails `-0.20m` via the documented orientation guard,
passes `+0.20m`) -- expected at only 72k/3M steps, not treated as a stop signal; recorded here
only to confirm the eval pipeline itself works end-to-end (`--algo sac` load, trace logging,
`valid_move_and_hold` computation) before spending the full training budget.

## Full run (3,000,000 steps, `--run-name sac_residual_alpha05`)

TBD.

## Evaluation

TBD -- grid: `height_alpha=0.5`, `dx` in `{-0.20, -0.15, -0.10, -0.05, 0.05, 0.10, 0.15, 0.20}`
(matching `docs/status/rl_gain_scheduling_alpha05_directional_fix_2026-07-29.md`'s exact grid),
against both:
- `config/ur5e_mujoco_torque_osc_tuned_adaptive_lambda.yaml` (original broken baseline, direct
  comparability with the 4th PPO attempt's exact 0/8 result), and
- `config/ur5e_mujoco_torque_osc_tuned_wrist_orient.yaml` (today's best fixed-gain controller
  -- the real bar to clear).

## Files changed

- `config/rl_gain_scheduling_sac_residual_alpha05.yaml` (new).
- `docs/status/sac_residual_torque_experiment_2026-07-29.md` (this file, new).
- `rl_gain_scheduling/eval_gain_scheduler.py` -- NOT edited by this task (already had `--algo`
  support from the parallel agent's work by the time this experiment needed it).

## Tests run

TBD (final list at completion) -- `pytest tests/mujoco/test_gain_scheduling_env.py
tests/mujoco/test_train_gain_scheduler.py -q` passed (28 passed) before launching training, to
confirm the shared training/eval code path was unaffected by the parallel agent's
`eval_gain_scheduler.py` edit.

## Rollback

```
rm config/rl_gain_scheduling_sac_residual_alpha05.yaml docs/status/sac_residual_torque_experiment_2026-07-29.md
```
No shared code was modified by this task (the `--algo` support in `eval_gain_scheduler.py` was
added by the parallel agent, not here) -- rollback is config/doc-file removal only.
