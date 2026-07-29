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

**SAC does not beat the fixed-gain baseline in `residual_torque` mode, on either baseline
config -- but it is a genuinely different failure than PPO's.** Full grid results (`height_
alpha=0.5`, `dx` in `{-0.20,-0.15,-0.10,-0.05,0.05,0.10,0.15,0.20}`):

| comparison | SAC-residual valid | baseline valid |
|---|---|---|
| vs `ur5e_mujoco_torque_osc_tuned_adaptive_lambda.yaml` (original broken baseline, PPO-4-comparable) | **1/8 (12%)** | 7/8 (88%) |
| vs `ur5e_mujoco_torque_osc_tuned_wrist_orient.yaml` (today's structural fix, real bar to clear) | **1/8 (12%)** | 8/8 (100%) |

Marginally less bad than the 4th PPO attempt's 0/8 on the same grid/architecture, but not a
meaningful improvement -- both are far below either fixed-gain baseline, and neither today's
structural controller fix nor the algorithm swap (PPO to SAC) at this architecture/reward
budget produces a policy that helps.

**The residual did NOT collapse to near-zero** -- this is the headline honest finding the task
asked to check for specifically. Mean `|residual_tau|` across the grid is **0.25-0.58 Nm**
(using a real, substantial fraction of the `[3.0, 3.0, 3.0, 1.5, 1.5, 1.5]` Nm per-joint
budget, peaking at 1.66 Nm on individual joints/steps), not the near-zero "learned to defer to
baseline" pattern the task flagged as the less-bad alternative failure mode. Instead, SAC
learned an **active, substantive, and actively harmful** correction: 7 of 8 grid cells fail via
`|axis_error| grew for 100 consecutive steps` (`controller_core/safety.py`'s monotonic-growth
guard) -- the exact same failure signature the fixed-config-corrected 4th PPO attempt hit on
all 8 of its cells (see `docs/status/rl_gain_scheduling_alpha05_directional_fix_2026-07-29.md`).
This is real evidence the bottleneck is the `residual_torque` architecture and/or reward shape
(specifically, whatever is inducing this indiscriminate axis-error-growth guard trip) rather
than being specific to PPO's on-policy optimization -- an off-policy algorithm with a replay
buffer, automatic entropy tuning, and 3x this task's typical training-signal reuse reaches a
qualitatively similar dead end.

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

Launched on `ilab4` in the foreground of one continuously-open SSH connection (per AGENTS.md
sec 8), with `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1`
exported, `n_envs=24`. Completed cleanly: verified by loading the saved model and reading
`model.num_timesteps` directly (`3000000`, exact -- no partial/truncated run). Wall clock ~44
minutes (`time_elapsed` field in the training log reached ~2650s at the final logged interval),
sustained throughput settling to **~1050-1080 fps** as the replay buffer filled and
`gradient_steps=1`/`train_freq=1` updates continued each step (slightly below the pilot's
1200-1360 fps early on, expected as buffer-read/update overhead grows, still CPU-step-bound
overall). Memory stayed low and flat throughout -- cgroup `memory.current` measured at ~12.8-
12.9GiB partway through the run, far under the ~81.4GiB per-user cap; no memory-safety concern
at any point. `ent_coef` (automatic entropy tuning) annealed steadily from ~0.58 at the start
down to ~0.003-0.004 by the end (124,428 gradient updates total) -- the policy became
increasingly deterministic/confident over training, not stuck in a permanently-exploratory
regime. One benign `shutil.rmtree` cleanup traceback again printed after "Saved final model"
(same interpreter-shutdown noise as the pilot, not a training failure).

No TensorBoard `rollout/ep_rew_mean` scalar was available for this run (pre-existing property
of this training script's env construction -- `_make_env_factory` in `train_ppo_gain_
scheduler.py` does not wrap `GainSchedulingEnv` in SB3's `Monitor`, so SB3 never populates
`ep_info_buffer` and never logs `rollout/*` scalars; confirmed via `event_accumulator.Reload()`
on the pilot's own event file, which lists only `time/fps` and `train/*` tags). This predates
and is unrelated to this task's own changes -- not fixed here (a training-script instrumentation
gap, out of scope for a single training-run task per this project's rule against mixing training
logic changes with unrelated fixes).

## Evaluation

Grid: `height_alpha=0.5`, `dx` in `{-0.20, -0.15, -0.10, -0.05, 0.05, 0.10, 0.15, 0.20}`
(matching `docs/status/rl_gain_scheduling_alpha05_directional_fix_2026-07-29.md`'s exact grid),
via `rl_gain_scheduling/eval_gain_scheduler.py --algo sac`, against both baselines.

### vs `config/ur5e_mujoco_torque_osc_tuned_adaptive_lambda.yaml` (original broken baseline)

| dx (m) | learned valid | learned quality | baseline valid | baseline quality | learned termination |
|---|---|---|---|---|---|
| -0.20 | False | 0.408 | False (documented bug) | 0.270 | `\|axis_error\| grew for 100 consecutive steps` |
| -0.15 | **True** | 0.441 | True | 0.362 | duration_complete |
| -0.10 | False | 0.606 | True | 0.461 | `\|axis_error\| grew for 100 consecutive steps` |
| -0.05 | False | 0.480 | True | 0.611 | `\|axis_error\| grew for 100 consecutive steps` |
| +0.05 | False | 0.554 | True | 0.631 | `\|axis_error\| grew for 100 consecutive steps` |
| +0.10 | False | 0.556 | True | 0.519 | `\|axis_error\| grew for 100 consecutive steps` |
| +0.15 | False | 0.454 | True | 0.438 | `\|axis_error\| grew for 100 consecutive steps` |
| +0.20 | False | 0.379 | True | 0.342 | `\|axis_error\| grew for 100 consecutive steps` |

**learned: 1/8 (12%), baseline: 7/8 (88%)** (baseline fails exactly `dx=-0.20m` via the
documented orientation guard, matching the original finding).

### vs `config/ur5e_mujoco_torque_osc_tuned_wrist_orient.yaml` (today's structural fix)

| dx (m) | learned valid | learned quality | baseline valid | baseline quality |
|---|---|---|---|---|
| -0.20 | False | 0.408 | **True** | 0.326 |
| -0.15 | **True** | 0.441 | True | 0.431 |
| -0.10 | False | 0.606 | True | 0.536 |
| -0.05 | False | 0.480 | True | 0.676 |
| +0.05 | False | 0.554 | True | 0.729 |
| +0.10 | False | 0.556 | True | 0.669 |
| +0.15 | False | 0.454 | True | 0.600 |
| +0.20 | False | 0.379 | True | 0.489 |

**learned: 1/8 (12%), baseline: 8/8 (100%)** -- the wrist-orientation-task fix cleanly resolves
the exact `dx=-0.20m` case that both the original baseline and every RL attempt (PPO and SAC)
have failed to fix, with no regression anywhere else in the grid. This is the real bar, and
SAC-residual is far from clearing it.

`learned` quality/termination values are identical across both tables (same policy, same
`--config`, only `--baseline-config` differs) -- confirms determinism (`deterministic=True` in
`model.predict`) and that the comparison is apples-to-apples.

**Residual-magnitude check on the fully-trained (3M-step) policy**, computed from the eval
trace (`residual_tau` logged every control step):

| dx (m) | termination | mean \|residual\| (Nm) | max \|residual\| (Nm) |
|---|---|---|---|
| -0.05 | axis-error-growth | 0.269 | 1.175 |
| -0.10 | axis-error-growth | 0.337 | 1.114 |
| -0.15 | duration_complete (valid) | 0.250 | 1.368 |
| -0.20 | axis-error-growth | 0.371 | 1.466 |
| +0.05 | axis-error-growth | 0.254 | 1.354 |
| +0.10 | axis-error-growth | 0.310 | 1.452 |
| +0.15 | axis-error-growth | 0.451 | 1.483 |
| +0.20 | axis-error-growth | 0.578 | 1.639 |

Grid-wide mean `|residual|` = **0.353 Nm** -- a real, substantial, non-collapsed correction
(not the "learned to do nothing" pattern), and one that grows with `|dx|` rather than shrinking
toward the trivial cases. The one cell that stays valid (`dx=-0.15m`) has the second-smallest
mean residual (0.250 Nm) in the whole grid, consistent with "small residual near the edge of
the easy region, large and destabilizing residual further out" rather than random noise.

## Recommendation

- **Do not adopt this policy** -- it underperforms both fixed-gain baselines on nearly the
  entire grid, and where it fails it fails by actively destabilizing (axis-error-growth), not
  by passively doing nothing.
- **Use `config/ur5e_mujoco_torque_osc_tuned_wrist_orient.yaml` for this problem** -- it is a
  strictly better fixed-gain option than both the original baseline and every RL attempt (PPO
  x4, SAC x1) so far, at zero training cost and zero regression risk.
- **The algorithm was not the bottleneck.** Swapping PPO for SAC -- a genuinely different
  optimization paradigm (off-policy, replay buffer, automatic entropy tuning, ~124k gradient
  updates vs PPO's on-policy epochs) -- reached a qualitatively similar dead end (same dominant
  failure signature: `|axis_error| grew for 100 consecutive steps`) rather than a different one.
  Combined with the 4th PPO attempt hitting the identical failure signature on all 8 cells, this
  is real evidence that the `residual_torque` action space and/or this reward shape (specifically
  whatever induces the axis-error-growth guard trip -- hypothesized in the PPO doc as
  high-frequency residual-torque oscillation on the X axis, not yet verified at trace level for
  either algorithm) is the actual bottleneck, not which RL algorithm is doing the optimizing.
- **If this architecture is revisited**, the axis-error-growth guard trip deserves the
  trace-level investigation the PPO doc already flagged as the next step (e.g. plotting
  `residual_tau` frequency content on a previously-easy cell) before spending more training
  compute on either algorithm.
- Per `docs/hardware/AUTO_TUNING_PLAN.md`'s "Revisited 2026-07-29" section, the concrete
  prerequisite for ever considering this bounded-residual architecture for real-hardware
  step-wise fine-tuning was "must first beat the fixed-gain baseline in simulation, at the exact
  case it's meant to fix." That prerequisite is still not met -- now confirmed under both PPO
  and SAC -- so no real-hardware extension of this architecture is warranted at this time.

## Files changed

- `config/rl_gain_scheduling_sac_residual_alpha05.yaml` (new, this task).
- `docs/status/sac_residual_torque_experiment_2026-07-29.md` (this file, new, this task).
- `rl_gain_scheduling/eval_gain_scheduler.py` -- NOT edited by this task; `--algo {ppo,sac}`
  support (commit `60e7c76`) was added by the parallel gains-mode-SAC agent before this
  experiment needed it, and reused as-is here (verified working via the pilot sanity eval).
- No files under `hardware/`, `controller_core/`, or `simulation/` touched -- pure config +
  training-run + doc, matching this task's simulation-only scope.

## Tests run

- `pytest tests/mujoco/test_gain_scheduling_env.py tests/mujoco/test_train_gain_scheduler.py -q`
  (28 passed) -- run before launching training, to confirm the shared training/eval code path
  (including the parallel agent's `--algo` addition) was sound.
- `pytest -q` (full suite) -- run after training/eval completed: **348 passed, 1 failed**. The
  one failure (`tests/hardware/test_direct_torque_transport_timing.py::
  test_transport_records_timing_and_deadline_loop`) is the same pre-existing failure already
  documented in `docs/status/wrist_orientation_task_2026-07-29.md` (a `dominant_phase`
  whitelist assertion that doesn't yet include `"local_dynamics"`, traced to the unrelated,
  earlier-landed commit `dee0190`) -- not a regression from this task, which touched no files
  under `hardware/` or `controller_core/`.

## Tests not run

- No hardware-in-the-loop or real-RTDE tests (out of scope -- simulation only, per task).
- No retraining/ablation of `training.sac.*` hyperparameters -- this was one real training
  attempt at a placeholder-but-reasonable hyperparameter set (per this config's own header),
  not a hyperparameter sweep; a sweep was out of scope for the available compute window.

## Rollback

```
rm config/rl_gain_scheduling_sac_residual_alpha05.yaml docs/status/sac_residual_torque_experiment_2026-07-29.md
```
No shared code was modified by this task (the `--algo` support in `eval_gain_scheduler.py` was
added by the parallel agent, not here) -- rollback is config/doc-file removal only. Trained
model artifacts live under `outputs/rl_gain_scheduling/sac_residual_alpha05*/` (gitignored, not
git-recoverable, not touched by this rollback).
