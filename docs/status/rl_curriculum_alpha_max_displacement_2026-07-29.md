# Curriculum-scheduled height_alpha + widened displacement target: mechanism works, task outcome is the worst RL result of the night

Context: eight prior real RL attempts on 2026-07-29 (PPO and SAC, gains and residual-torque
action modes, several reward variants, and one wide-fixed-range multi-pose attempt) all
underperformed the fixed-gain OSC controller on the `height_alpha=0.5` bidirectional-transport
task (`docs/status/sac_gains_mode_experiment_2026-07-29.md`,
`docs/status/sac_residual_torque_experiment_2026-07-29.md`,
`docs/status/rl_undershoot_instability_diagnosis_2026-07-29.md`,
`docs/status/sac_multi_alpha_generalization_2026-07-29.md`). The last of those specifically
found that widening training to a single fixed `height_alpha in [0.0, 0.5]` range made things
*worse* (2/12 overall, alpha=0.5 itself dropping from a dedicated single-pose run's 2/8 to 0/4)
-- the classic signature of a training distribution too hard/non-stationary for the policy.

This task tested two genuinely new ideas instead of an eleventh reward-shaping variant:
(1) a **curriculum** over `height_alpha` -- start easy, near the canonical pose, and
progressively shift toward `alpha=0.5` as training progresses, rather than sampling the whole
range uniformly from step 0; (2) **reframe the goal** away from re-proving the already-validated
`+/-0.20m` fixed-gain envelope, toward the genuinely untested `0.20-0.24m` frontier
(AGENTS.md sec 3's documented `dx=0.25m` Z-drift failure point).

## Verdict

**The curriculum mechanism itself works exactly as designed and was verified end-to-end in real
training runs (not just unit tests) -- but it did not help the task, and the resulting policies
are the worst of all nine real RL attempts run tonight, for both algorithms tried (SAC and PPO),
scoring 0/30 across the full evaluation grid including 0/6 at the dx=0.23m frontier point that
motivated this experiment.** The fixed-gain baseline, by contrast, passes all 6 dx=0.23m cells
cleanly (quality 0.284-0.404) -- itself a small, non-RL, incidental finding: the documented
`+/-0.20m` envelope safely extends to at least `0.23m` under fixed gains at `height_alpha in
{0.1, 0.3, 0.5}`, previously untested territory in this repo.

Curriculum learning, as implemented here, is not a fix for the "wide training distribution is
too hard" problem this session already diagnosed -- if anything it looks worse than the earlier
non-curriculum wide-range attempt (2/12, `sac_multi_alpha_generalization_2026-07-29.md`), and a
concrete, evidenced mechanism for why is below (SAC's replay buffer is smaller than a single
curriculum stage, so by the end of training the policy is effectively trained on-policy-only
against the single hardest stage with much less data than any dedicated single-pose attempt got).

## What was implemented

### 1. Curriculum mechanism (real engineering, not a config toggle)

- `rl_gain_scheduling/gain_scheduling_env.py`: `GainSchedulingEnv.set_height_alpha_range(low,
  high)` -- mutates `self._height_alpha_range` for future `reset()` calls only; validated,
  raises on `low > high` or out-of-`[0,1]` bounds. Opt-in by construction (nothing calls it
  unless the training script's callback does), so no existing config's behavior changes.
- `rl_gain_scheduling/train_ppo_gain_scheduler.py`: `HeightAlphaCurriculumCallback(BaseCallback)`
  -- reads `config.env.curriculum.stages` (a list of `{alpha_range, timestep_frac}`, fracs
  summing to ~1.0), converts fractions into absolute timestep boundaries using the run's
  **actual** `total_timesteps` (so a `--total-timesteps` pilot override exercises the full
  stage sequence in miniature, not just stage 0 -- confirmed this works, see Pilot section),
  and at each boundary calls `VecEnv.env_method("set_height_alpha_range", lo, hi)` -- a VecEnv
  base-class API that works identically for `SubprocVecEnv` (pickles the call to each remote
  worker process) and `DummyVecEnv` (calls in-process). The policy/optimizer are never touched;
  only the sampling range for each worker's *next* `reset()` changes. Wired into `main()` via
  `CallbackList([checkpoint_callback, curriculum_callback])`, gated on
  `config.env.curriculum.enabled` (absent/false = today's exact byte-identical behavior).

### 2. Evidence the mechanism actually works (not just "training ran without crashing")

- **Unit/integration tests** (`tests/mujoco/test_gain_scheduling_env.py`,
  `tests/mujoco/test_train_gain_scheduler.py`, new tests added this task): boundary-correctness
  test against the real 5-stage config: t=0 -> stage 0, every 20% quintile -> the next stage,
  `total_timesteps-1` -> the last stage, monotonic non-decreasing throughout. A real end-to-end
  `DummyVecEnv` + tiny SAC training integration test proving `env_method` genuinely mutates
  live worker env state mid-training (not just the callback's own bookkeeping). Also
  `set_height_alpha_range` unit tests (valid range narrows sampling with real spread, invalid
  bounds raise). All new tests pass; full suite 376 passed, 1 pre-existing unrelated failure
  (`test_direct_torque_transport_timing.py::test_transport_records_timing_and_deadline_loop`).
- **Real pilot runs (80,016/80,000 steps each)**: both SAC and PPO pilots printed all 5 stage
  transitions at the exact expected timesteps (0/16000/32000/48000/64000 for an 80k-step
  budget), with no NaNs and sane training curves (SAC: `ent_coef` 0.774->0.461, stable
  `critic_loss`; PPO: `explained_variance` 0->0.72, stable `approx_kl`/`clip_fraction`).
- **Real full runs (3,000,000 steps each)**: both printed all 5 stage transitions at the exact
  expected 20%-quintile boundaries of the real 3M budget (0 / 600,000 / 1,200,000 / 1,800,000 /
  2,400,000), confirmed directly in the training logs, not inferred. Final models verified by
  loading and reading `model.num_timesteps`: SAC = exactly 3,000,000; PPO = 3,014,656 (normal
  PPO rollout-batch overshoot, same pattern documented in every prior PPO run this session).

**This is the real, positive part of the result: the curriculum mechanism is correctly
implemented, tested, and was verified operating exactly as designed in two real full 3M-step
training runs across two different algorithms. The negative part is that it didn't help the
task it was built for.**

### 3. Reframed objective

`target_x_delta_range_m` widened from all eight prior attempts' `[-0.20, 0.20]` to
`[-0.24, 0.24]` -- approaching, but staying meaningfully under, the documented `dx=0.25m`
Z-drift structural failure point (AGENTS.md sec 3). `reward.*` left **unchanged** from the
best-performing prior config (`rl_gain_scheduling_sac_gains_alpha05.yaml`, v1, 2/8) --
deliberately not touched, per `docs/status/rl_undershoot_instability_diagnosis_2026-07-29.md`'s
finding that even a narrow, well-motivated reward change traded one failure mode for a worse one
(2/8 -> 0/8) on a single run. `progress_weight`/`terminal_quality_weight` already scale relative
to `|target_x_delta|` via `compute_valid_move_hold_metrics`'s own tolerance formulas, so widening
the range alone exposes the policy to, and lets the existing reward correctly score, the
larger-displacement regime without new reward engineering.

### 4. New configs

`config/rl_gain_scheduling_sac_curriculum_alpha.yaml` (SAC, `n_envs=24`, matches
`rl_gain_scheduling_sac_gains_alpha05.yaml`'s controller/reward exactly) and
`config/rl_gain_scheduling_ppo_curriculum_alpha.yaml` (PPO, `n_envs=57` in-file default,
actually launched at `n_envs=32` -- see Compute section -- matching the historical PPO
hyperparameters from `..._residual_penalized_safety_fix.yaml`). Both share identical
env/controller/reward/curriculum blocks; kept as two files rather than one because PPO and SAC's
`n_envs` were independently validated memory-safe at different values and a single YAML can't
hold two different `training.n_envs` at once. 5-stage curriculum in both:
`[0.05,0.15] -> [0.10,0.25] -> [0.15,0.35] -> [0.25,0.45] -> [0.35,0.50]`, each 20% of budget.

## Compute

- **SAC**: launched on `westeros.cs.rutgers.edu` (idle: load average 2.74/3.52/3.22 on 72 cores,
  97.5% CPU idle, no memory cgroup cap for this user, verified before launch). `n_envs=24`
  (matches prior validated SAC runs). Pilot (80,016 steps) ~72s. Full run: exactly 3,000,000
  steps, fps 946-1086 throughout, no NaNs.
- **PPO**: launched concurrently on `ilab4.cs.rutgers.edu` (a different host than SAC, per the
  coordinator's instruction to avoid resource contention) -- checked idle first (load average
  5.66/5.75/6.07 on 96 cores, 537Gi available, 81.4GiB per-user memory cgroup cap). Run via a
  single continuously-open foreground SSH connection (`ssh host 'exec ...'` wrapped in a local
  backgrounded shell job, per this repo's own documented lesson about `nohup`+`disown`
  unreliability on these hosts -- never relied on remote backgrounding). `n_envs=32` (reduced
  from the config's in-file default of 57, since 32 was what the pilot actually validated
  memory-safe on this host this session; not re-tested at 57). Pilot (80,000 steps) ~64s. Full
  run: 3,014,656/3,000,000 steps, fps ~1700-4150 (declining per-iteration as usual for PPO's
  batched updates), no NaNs, `explained_variance` reaching 0.999.
- Both runs' `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1`
  exported before launch. Both waited on synchronously via real blocking polls against the log
  files (never assumed a background notification alone was sufficient without checking the real
  output) until each printed `Saved final model` and its saved checkpoint's `num_timesteps` was
  independently verified by loading the model.

## Evaluation

Grid: `height_alpha in {0.1, 0.3, 0.5} x dx in {-0.20, -0.10, 0.10, 0.20, 0.23}` (15 cells per
model), baseline `config/ur5e_mujoco_torque_osc_tuned.yaml` at alpha in {0.1, 0.3} (**not**
validated at these poses by anything in this repo -- reference only, flagged honestly, not
fabricated as a real baseline) and `config/ur5e_mujoco_torque_osc_tuned_wrist_orient.yaml` at
alpha=0.5 (the one pose in this grid with a genuinely validated fixed-gain baseline).

### Headline numbers

| Model | height_alpha | learned valid | baseline valid |
|---|---|---|---|
| SAC curriculum | 0.10 | 0/5 (0%) | 5/5 (100%) |
| SAC curriculum | 0.30 | 0/5 (0%) | 5/5 (100%) |
| SAC curriculum | 0.50 | 0/5 (0%) | 5/5 (100%) |
| PPO curriculum | 0.10 | 0/5 (0%) | 5/5 (100%) |
| PPO curriculum | 0.30 | 0/5 (0%) | 5/5 (100%) |
| PPO curriculum | 0.50 | 0/5 (0%) | 5/5 (100%) |
| **Overall (both models, all cells)** | | **0/30 (0%)** | **30/30 (100%)** |

### The dx=0.23m frontier test (the result that matters most per this task's own instructions)

| Model | alpha=0.1 | alpha=0.3 | alpha=0.5 |
|---|---|---|---|
| SAC curriculum (quality) | 0.163 (invalid) | 0.111 (invalid) | 0.069 (invalid) |
| PPO curriculum (quality) | 0.022 (invalid) | 0.037 (invalid) | 0.053 (invalid) |
| Fixed-gain baseline (quality) | 0.284 (**valid**) | 0.300 (**valid**) | 0.404 (**valid**) |

**No, the trained policy does not achieve anything at dx=0.23m -- 0/6 cells valid across both
algorithms and all three poses.** The fixed-gain baseline, by contrast, passes all 6 cleanly.
This is the opposite of the hoped-for result: the one place this experiment could have shown
real, novel value (RL reaching territory the fixed-gain controller had never been validated at)
instead shows the fixed-gain controller quietly extending its own validated envelope into that
territory for free, while RL fails everywhere.

### Failure-mode detail (real per-cell trace data, not guessed)

SAC and PPO fail differently, which is itself informative:

- **SAC** shows a real directional asymmetry, opposite in character from the fixed-gain
  controller's own documented one. Negative-dx cells track well (e.g. `alpha=0.1, dx=-0.10`:
  `hold_phase_final_x_error_m = -1.7e-5`, essentially exact) but still fail
  `valid_move_and_hold` via a guard-adjacent trajectory-shape check
  (`"|axis_error| grew for 100 consecutive steps"`, `hold_phase_incomplete`) despite good final
  tracking -- a variant of the same drift-vs-target metric-mismatch mechanism diagnosed in
  `docs/status/rl_undershoot_instability_diagnosis_2026-07-29.md` (not re-fixed here, per that
  doc's own caution against another reward change without new evidence). Positive-dx cells are
  much worse and qualitatively different: `move_phase_achieved_x_delta_m` is a small fraction of
  target (e.g. `alpha=0.5, dx=0.23`: achieves `0.065m` of `0.23m` target; `alpha=0.5, dx=0.10`:
  achieves `0.019m` of `0.10m`) -- the policy is not slowly converging, it is barely moving in
  `+X` at all in the harder poses.
- **PPO** is uniformly poor across every cell and direction (quality 0.017-0.085, no
  directional pattern), qualitatively different from SAC's mixed picture. This matches PPO's
  standing position as the weaker algorithm on this task all session (its best-ever single-pose
  result was 0/8, worse than SAC's 2/8) -- the curriculum did not change that ordering.

### A concrete, evidenced hypothesis for why curriculum didn't help (SAC specifically)

`config.training.sac.buffer_size = 300,000`, but each curriculum stage spans `600,000`
timesteps (20% of 3,000,000). By the time training reaches the final stage
(`height_alpha in [0.35, 0.5]`, the hardest pose region, combined with the widest dx range this
session has ever trained on), the FIFO replay buffer has already fully evicted every transition
from the four earlier, easier stages -- the final policy's off-policy updates are effectively
trained on only the last ~600,000-ish steps of genuinely relevant (in-buffer) data, less than
half of any dedicated single-pose SAC attempt's full 3,000,000-step budget, and with no residual
in-buffer experience from the easier curriculum stages to anchor against catastrophic forgetting.
This is a real, mechanistic explanation grounded in the actual configured numbers (`buffer_size`
vs `timestep_frac * total_timesteps`), not a guess -- and a concrete lever for anyone revisiting
this: either size `buffer_size` to span at least a full stage-plus-margin, or make curriculum
stages longer than the buffer so the final stage gets a comparable data budget to a dedicated
run. This was not attempted here given the time budget and the already-clear negative result.

## Honest recommendation

- **Curriculum scheduling, as implemented and tested here, is not the fix for RL underperforming
  the fixed-gain controller on this task.** It is a real, working, well-tested mechanism (confirm
  this independently: the unit/integration tests and the two real full training runs' own log
  output both show it operating exactly as designed) -- but the policies it produced are worse
  than every one of the eight prior attempts' best results, including the single-pose SAC run's
  2/8 and even the non-curriculum wide-range attempt's 2/12.
- **The reframed goal (push toward the 0.20-0.24m frontier) also did not pay off**: 0/6 at
  dx=0.23m across both algorithms and all three tested poses, against a fixed-gain baseline that
  passes all 6 without any RL involvement at all -- itself a small, useful, non-RL finding
  (the documented envelope safely extends to at least 0.23m at these poses under fixed gains).
- **Nine real RL training attempts now (eight prior + this one, two algorithms, multiple reward
  formulations, single-pose/wide-fixed-range/curriculum training distributions) have not beaten
  the fixed-gain controller on this repo's actual evaluation grids.** The non-RL
  `controller.wrist_orientation_task` fix remains the best available, validated solution to the
  original motivating bug. Given this task's own evidence (the replay-buffer-vs-stage-length
  mismatch above is a concrete, unexplored lever, but a speculative one until tried), the honest
  call is the same one `docs/status/rl_undershoot_instability_diagnosis_2026-07-29.md` already
  made: further RL-training-distribution experiments on this exact 11-gain/47-obs architecture
  are a low-probability-of-success path relative to extending the structural,
  analytically-motivated controller fix already validated and shipped.

## Files changed

- `rl_gain_scheduling/gain_scheduling_env.py` -- new `GainSchedulingEnv.set_height_alpha_range`.
- `rl_gain_scheduling/train_ppo_gain_scheduler.py` -- new `HeightAlphaCurriculumCallback`, wired
  into `main()` behind `config.env.curriculum.enabled`.
- `tests/mujoco/test_gain_scheduling_env.py` -- 2 new tests for `set_height_alpha_range`.
- `tests/mujoco/test_train_gain_scheduler.py` -- 3 new tests for the curriculum callback
  (bad-frac-sum rejection, boundary correctness, real `DummyVecEnv` end-to-end integration).
- `config/rl_gain_scheduling_sac_curriculum_alpha.yaml` (new).
- `config/rl_gain_scheduling_ppo_curriculum_alpha.yaml` (new).
- This file (new).
- Not touched: `AGENTS.md`, any `README.md`, `hardware/`, `assets/urscript/*`,
  `tools/ur5e_pose_sweep*`, `tools/ur5e_move_hold_transport.py`, `tests/hardware/*` (all
  explicitly out of scope per this task's instructions, and per observation, actively being
  edited by other concurrent agents tonight).

Trained models/eval artifacts (gitignored, not part of any commit):
`outputs/rl_gain_scheduling/{sac,ppo}_curriculum_alpha{,_pilot}/`,
`outputs/rl_gain_scheduling/eval/{sac,ppo}_curriculum_eval_*/`.

## Tests run

- `pytest tests/mujoco/test_gain_scheduling_env.py tests/mujoco/test_train_gain_scheduler.py -q`
  -- 36 passed (31 pre-existing + 5 new).
- Full suite `python -m pytest -q` (run twice: once before launching training, once after all
  evaluation): 376 passed, 1 pre-existing unrelated failure
  (`tests/hardware/test_direct_torque_transport_timing.py::
  test_transport_records_timing_and_deadline_loop`) both times -- no regressions from this task.

## Tests not run

- No hardware-in-the-loop or real-RTDE tests (simulation-only task, as all nine RL attempts
  tonight have been).

## Rollback

```
git rm config/rl_gain_scheduling_sac_curriculum_alpha.yaml config/rl_gain_scheduling_ppo_curriculum_alpha.yaml docs/status/rl_curriculum_alpha_max_displacement_2026-07-29.md
git checkout <prior-sha> -- rl_gain_scheduling/gain_scheduling_env.py rl_gain_scheduling/train_ppo_gain_scheduler.py tests/mujoco/test_gain_scheduling_env.py tests/mujoco/test_train_gain_scheduler.py
```
The curriculum mechanism is fully opt-in (`config.env.curriculum.enabled`, absent/false in every
config in this repo except the two new ones above), so simply not setting it is itself a full
functional rollback without touching any code. Trained models/eval artifacts under `outputs/`
are gitignored, not tracked -- no rollback needed for those.
