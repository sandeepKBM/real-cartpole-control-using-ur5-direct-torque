# SAC multi-pose generalization test (height_alpha 0.0-0.5): no advantage found, and the wide policy is worse everywhere

Context: seven prior real RL attempts (4 PPO + 3 SAC, see `docs/status/sac_gains_mode_experiment_
2026-07-29.md`, `docs/status/sac_residual_torque_experiment_2026-07-29.md`,
`docs/status/rl_undershoot_instability_diagnosis_2026-07-29.md`) all targeted the specific
`height_alpha=0.5`, `-0.20m` orientation-error bug documented in AGENTS.md sec 3. All seven
underperformed a much simpler, non-RL structural fix (`controller.wrist_orientation_task`,
commit `bd5bba3`, `docs/status/wrist_orientation_task_2026-07-29.md`), which already solves that
exact bug (0.237-0.250 rad -> 0.064-0.073 rad, zero regressions). Given that, this experiment
asked a genuinely different question: every fixed-gain config in this repo
(`config/ur5e_mujoco_torque_osc_tuned*.yaml`) is tuned and validated at one specific pose --
can a state-conditioned RL policy do something a *single* fixed gain vector structurally
cannot, i.e. handle a *range* of poses?

## Verdict

**No. The multi-pose SAC policy shows no generalization advantage and is worse than both the
appropriate fixed-gain baselines and the prior single-pose SAC attempt, at every pose tested,
including the pose (alpha=0.5) it should have specialized on most.**

Evaluation grid: `height_alpha in {0.0, 0.25, 0.5}` x `dx in {-0.20, -0.10, 0.10, 0.20}`
(12 cells), baseline per alpha as specified (`config/ur5e_mujoco_torque_osc_tuned.yaml` at
alpha=0.0, `config/ur5e_mujoco_torque_osc_tuned_wrist_orient.yaml` at alpha=0.5; alpha=0.25
has no validated fixed-gain config in this repo -- `ur5e_mujoco_torque_osc_tuned.yaml` used for
reference only, clearly flagged as extrapolation):

| height_alpha | learned valid | baseline valid | baseline config |
|---|---|---|---|
| 0.00 | 2/4 (50%) | 4/4 (100%) | `ur5e_mujoco_torque_osc_tuned.yaml` (validated here) |
| 0.25 | 0/4 (0%) | 4/4 (100%) | `ur5e_mujoco_torque_osc_tuned.yaml` (**not** validated at this pose -- reference only) |
| 0.50 | 0/4 (0%) | 4/4 (100%) | `ur5e_mujoco_torque_osc_tuned_wrist_orient.yaml` (validated here) |
| **overall** | **2/12 (17%)** | **12/12 (100%)** | |

The hoped-for interesting result -- RL doing relatively better at the unvalidated intermediate
pose than at the well-covered endpoints -- did not happen. It is 0/4 at alpha=0.25, exactly the
same as its 0/4 at alpha=0.5, and worse than its own 50% at alpha=0.0. There is no sign of a
generalization benefit, only a roughly monotonic decline in quality as the pose moves away from
alpha=0.0 (the low end of the trained range, which also happens to be the least-difficult pose
kinematically -- see below).

**Widening the training distribution did not trade breadth for a reasonable single-pose
result either -- it made the alpha=0.5 result worse than the previous, dedicated single-pose
SAC attempt.** The prior alpha=0.5-only SAC run (`config/rl_gain_scheduling_sac_gains_alpha05.yaml`,
`docs/status/sac_gains_mode_experiment_2026-07-29.md`) scored 2/8 valid on an 8-cell dx grid at
that exact pose, with real (if narrow) passes at `dx=+0.05m` and `dx=+0.10m`. This run's policy,
trained on the *same* alpha=0.5 pose but as part of a `[0.0, 0.5]` range instead of a fixed
point, scores 0/4 on the 4-cell subset of that same grid it was evaluated on (`dx in {-0.20,
-0.10, +0.10, +0.20}`) -- including failing `dx=+0.10m`, which the dedicated single-pose policy
passed. This is consistent with capacity/attention being diluted across the wider training
distribution rather than the policy learning a genuinely pose-adaptive strategy.

## Precondition check (done first, per the task): is height_alpha actually observable?

**Yes, both directly and redundantly.** `rl_gain_scheduling/gain_scheduling_env.py`'s
`_build_obs_and_info` (lines ~500-525) builds `OBS_DIM=47` from, among other terms,
`np.sin(q), np.cos(q)` (12 dims) and `ee_pos` (3 dims, the raw Cartesian end-effector position).
Since `height_alpha` linearly interpolates `q_start` between `ACTIVE_ORIGIN_Q` and `LOWER_B_Q`
(`reset()`, lines 219-221) and both share the wrist_2=0 singularity by construction, `q` (and
therefore `sin(q)/cos(q)`) directly encodes the sampled pose every step, and `ee_pos[2]` (world
Z height) is a direct, unambiguous readout of it too. This precondition is satisfied without any
env changes -- confirmed before writing any config or launching training, per the task's
instruction to check this first.

## Evidence

### 1. Config

New config `config/rl_gain_scheduling_sac_gains_multi_alpha.yaml`, built from `config/
rl_gain_scheduling_sac_gains_alpha05.yaml` (preserved unmodified, the best-performing of the
seven prior real attempts). Exactly one field changed: `env.height_alpha_range` widened from
`[0.5, 0.5]` (fixed) to `[0.0, 0.5]`. `target_x_delta_range_m` stays `[-0.20, 0.20]`. All
controller flags (`lambda_diagonal_shaping`, `lambda_adaptive_regularization`, etc.), reward
weights, and SAC hyperparameters (`n_envs=24`, `net_arch=[128,128]`, etc.) are unchanged, to
isolate the one variable this experiment tests.

### 2. Compute

Ran directly on `westeros.cs.rutgers.edu` (this session's actual host -- confirmed via
`hostname`), found idle before launch (`uptime` load average 1.86/3.40/5.82 on 72 cores,
`mpstat` 97.9% idle, `free -h` 76Gi free RAM) and had it entirely to itself (no other
experiment running concurrently this time). `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1` exported before every launch, `n_envs=24` (unchanged
from the alpha05 run). Training ran as a genuinely synchronous foreground process this session
blocked on directly (via a wait-loop, never fire-and-forget) until it exited, per this
session's explicit process requirement.

### 3. Pilot (80,016 timesteps, `--run-name sac_gains_multi_alpha_pilot`)

Completed in ~70s (fps ~1018-1119). No NaNs. `ent_coef` decayed smoothly (0.774 -> 0.461),
`critic_loss` stable (0.097-0.26), `actor_loss` trending more negative (-29 -> -58) -- the same
healthy SAC training shape seen in the alpha05 pilot. Judged sane; proceeded to the full run
unchanged.

### 4. Full run (3,000,000 timesteps, `--run-name sac_gains_multi_alpha`)

Ran to completion: 3,000,000 / 3,000,000 timesteps in 3137s (~52 min), fps 918-956 throughout,
verified by loading the saved model and reading `model.num_timesteps == 3000000`. Final model:
`outputs/rl_gain_scheduling/sac_gains_multi_alpha/models/sac_gain_scheduler_final.zip`.
`ent_coef` decayed to ~0.007 by the end (healthy auto-entropy-tuning, not premature collapse to
0); `critic_loss` fluctuated (spiked to ~1.1 mid/late-run) but never diverged/NaN'd. Same
cosmetic multiprocessing-tempdir-cleanup `OSError` seen after "Saved final model" in every prior
run in this project appeared here too -- unrelated to training, model save completed first.

### 5. Evaluation

`rl_gain_scheduling/eval_gain_scheduler.py` does not support a per-cell/per-alpha baseline
config (one `--baseline-config` argument applies to the whole `--alphas`/`--deltas` sweep) --
confirmed by reading its `main()` before assuming otherwise. Ran it three separate times (one
per alpha, matching baseline config passed explicitly each time) and combined the printed
per-cell tables by hand above, exactly as the task anticipated might be necessary:

- `--alphas 0.0 --deltas -0.20 -0.10 0.10 0.20 --baseline-config config/ur5e_mujoco_torque_osc_tuned.yaml --run-name multi_alpha_eval_alpha0_baseline_tuned`
- `--alphas 0.25 --deltas -0.20 -0.10 0.10 0.20 --baseline-config config/ur5e_mujoco_torque_osc_tuned.yaml --run-name multi_alpha_eval_alpha025_baseline_tuned` (baseline shown for reference only -- not validated at this pose by anything in this repo)
- `--alphas 0.5 --deltas -0.20 -0.10 0.10 0.20 --baseline-config config/ur5e_mujoco_torque_osc_tuned_wrist_orient.yaml --run-name multi_alpha_eval_alpha05_baseline_wristorient`

All three used `--config config/rl_gain_scheduling_sac_gains_multi_alpha.yaml --model-path
outputs/rl_gain_scheduling/sac_gains_multi_alpha/models/sac_gain_scheduler_final.zip --algo sac`.

Per-cell detail (from `outputs/rl_gain_scheduling/eval/*/learned/runs/*/summary.json`):

| cell | termination_reason | target dx | achieved dx | final x_err | max orientation err (rad) | max \|qd\| (rad/s) | quality |
|---|---|---|---|---|---|---|---|
| alpha0.00, dx-0.20 | axis_error grew 100 steps | -0.20 | -0.125 | -0.004 | 0.155 | 0.36 | 0.044 |
| alpha0.00, dx-0.10 | axis_error grew 100 steps | -0.10 | -0.101 | -0.001 | 0.126 | 0.30 | 0.335 |
| alpha0.00, dx+0.10 | duration_complete | +0.10 | +0.076 | +0.015 | 0.025 | 0.19 | 0.203 (**valid**) |
| alpha0.00, dx+0.20 | duration_complete | +0.20 | +0.170 | +0.021 | 0.107 | 0.48 | 0.147 (**valid**) |
| alpha0.25, dx-0.20 | \|Y-Y0\|>0.03m | -0.20 | -0.171 | -0.003 | 0.180 | 0.75 | 0.070 |
| alpha0.25, dx-0.10 | axis_error grew 100 steps | -0.10 | -0.101 | -0.000 | 0.097 | 0.49 | 0.336 |
| alpha0.25, dx+0.10 | duration_complete | +0.10 | +0.072 | +0.022 | 0.108 | 0.19 | 0.151 |
| alpha0.25, dx+0.20 | axis_error grew 100 steps | +0.20 | +0.156 | +0.041 | 0.154 | 0.58 | 0.120 |
| alpha0.50, dx-0.20 | duration_complete | -0.20 | -0.201 | +0.000 | 0.302 | 0.81 | 0.178 |
| alpha0.50, dx-0.10 | axis_error grew 100 steps | -0.10 | -0.101 | -0.000 | 0.090 | 0.30 | 0.404 |
| alpha0.50, dx+0.10 | orientation err > 0.25 rad | +0.10 | +0.061 | +0.038 | 0.201 | 0.27 | 0.098 |
| alpha0.50, dx+0.20 | orientation err > 0.25 rad | +0.20 | +0.145 | +0.053 | 0.170 | 0.66 | 0.114 |

Fixed-gain baselines were 4/4 valid at all three alphas tested (alpha=0.0 and 0.5 genuinely
validated poses; alpha=0.25 unvalidated but happened to pass all four cells too -- worth noting
as mild evidence the controller's structural weakness is genuinely localized to alpha=0.5, not
a smooth function of alpha).

The learned policy's failure modes are mixed and inconsistent across the grid (safety-guard
axis-error growth, Y-drift, and orientation-error trips, plus undershoot even in
`duration_complete` cells) -- unlike the cleaner single-mechanism failure signatures found in
several prior single-pose attempts. This is itself informative: it does not look like the
policy learned one coherent strategy that degrades gracefully with pose, but rather learned
something more diffuse across the wider distribution.

## Honest recommendation

Do not pursue RL gain-scheduling further for this problem. Across eight real training attempts
now (seven single/narrow-pose + this multi-pose one), RL has never once beaten the appropriate
fixed-gain baseline on this repo's actual evaluation grids, and widening the training
distribution to test its one plausible structural advantage (pose generalization) made results
worse, not better, including at the pose it should have specialized on most. The
`controller.wrist_orientation_task` fix already closes the original motivating bug with zero
regressions and no RL involved. If pose-range robustness is wanted in the future, the more
promising path based on this evidence is extending that same additive, structural approach
(explicit task decomposition, flag-gated, analytically motivated) to cover intermediate poses,
not further RL training-distribution experiments on the current 11-gain/12-obs-input setup.

## Files changed

- `config/rl_gain_scheduling_sac_gains_multi_alpha.yaml` (new)
- `docs/status/sac_multi_alpha_generalization_2026-07-29.md` (new, this file)

No `controller_core/`, `simulation/`, `rl_gain_scheduling/*.py`, or `AGENTS.md`/`README.md`
files were touched.

## Tests

`python -m pytest -q`: 350 passed, 1 failed (`tests/hardware/test_direct_torque_transport_
timing.py::test_transport_records_timing_and_deadline_loop` -- the pre-existing, documented,
unrelated failure noted in this session's own instructions; unaffected by this change). Also
ran `tests/mujoco/test_gain_scheduling_env.py tests/mujoco/test_train_gain_scheduler.py`
directly (30 passed) before launching training, since those are the tests most directly
relevant to the env/config touched here.

## Rollback

```
git rm config/rl_gain_scheduling_sac_gains_multi_alpha.yaml docs/status/sac_multi_alpha_generalization_2026-07-29.md
```
(Trained model/eval artifacts live under `outputs/` and are gitignored -- not part of any
commit, nothing to roll back there.)
