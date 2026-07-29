# Diagnosing the SAC gains-mode negative-dx "undershoot": a hold-phase drift metric penalizing safe convergence, not a training-budget or genuine-undershoot problem

Context: `docs/status/sac_gains_mode_experiment_2026-07-29.md` (a fifth real RL attempt at the
`height_alpha=0.5` directional-asymmetry problem, AGENTS.md sec 3) reported SAC in `gains` mode
at 2/8 valid vs both fixed-gain baselines, characterizing the four negative-dx failures as
**"undershoot"**: `dx=-0.20m` only achieves `-0.169m`, etc. Separately, the structural
`wrist_orientation_task` fix (`docs/status/wrist_orientation_task_2026-07-29.md`) already solves
the *original* controller bug non-RL, so this task's job was to actually pull per-step trace data
for the worst case and determine whether that framing was right, or whether something more subtle
was going on -- before proposing any fix.

## Verdict

**The "undershoot" framing was incomplete and led toward the wrong diagnosis.** Pulling the raw
`trace.jsonl` for the worst case (`dx=-0.20m`, `height_alpha=0.5`) shows the policy actually
reaches the target almost exactly by episode end: `hold_phase_final_x_error_m = 0.00028m`
(0.28mm). It is not stuck, not satisfied-too-early, and not still shrinking-but-far-off when the
episode ends -- it converges essentially completely. The real, concrete, well-evidenced mechanism,
confirmed identically across **all four** failing negative-dx cells:

- The policy's move-phase convergence (by `t = move_duration_s = 1.0s`) is genuinely slower than
  the fixed-gain baseline's on this specific (harder, `-X`) direction: ~80-85% of the way to target
  at `t=1.0s`, vs the baseline's ~99%+.
- Because of that residual gap, the EE position keeps moving (safely, monotonically, no
  oscillation) for roughly the first 1.5-1.8s of the nominal 2.0s "hold" window before settling.
- `transport_metrics.py`'s `hold_phase_x_drift_from_hold_start_m` -- correctly designed to catch
  wander/instability *after* a move is supposed to already be settled -- measures EE motion
  relative to the position at the **start of the hold window**, not relative to the target. It
  cannot distinguish "still safely converging toward the target" from "drifting away from an
  already-reached one." For all four negative-dx cells this drift exceeds its tolerance
  (`max(0.003, 0.15*|target_x_delta|)`) even though the *final* tracking error is ~0.
- This is not just an eval-reporting quirk: `move_hold_quality_score` (which folds in this exact
  term) is multiplied by `terminal_quality_weight=50.0` and added to the reward at the end of
  **every** truncated training episode -- the single largest lump-sum reward event per episode.
  So the training signal itself penalizes this genuinely safe convergence behavior, on every
  episode in the harder direction where the move isn't already complete by `t=1.0s`.

This is a real, concrete reward/metric-shaping issue (not a guess, not "just needs more training
steps") -- Step 2's trigger condition. A minimal, targeted, flag-gated fix was implemented and a
new full training run launched; see the Fix-Attempt section below for the real (not speculative)
result.

## Evidence (Step 1)

### Per-cell metrics, all 8 cells, `sac_gains_alpha05` (the prior experiment's model), vs
`ur5e_mujoco_torque_osc_tuned_wrist_orient.yaml`

Source: `outputs/rl_gain_scheduling/eval/sac_gains_alpha05_vs_wrist_orient/{learned,baseline}/runs/*/summary.json`
(full per-step `trace.jsonl` also present -- `eval_gain_scheduler.py` runs with
`full_trace_logging=True`).

| dx (m) | valid_move | valid_hold | hold_fail_reason | hold_phase_x_drift_from_hold_start_m | hold_phase_final_x_error_m | move_phase_achieved_x_delta_m | quality |
|---|---|---|---|---|---|---|---|
| -0.20 | True | **False** | `hold_phase_target_tracking` | 0.0324 (tol 0.030) | 0.0003 | -0.1693 | 0.207 |
| -0.15 | True | **False** | `hold_phase_target_tracking` | 0.0328 (tol 0.0225) | 0.0009 | -0.1199 | 0.180 |
| -0.10 | True | **False** | `hold_phase_target_tracking` | 0.0208 (tol 0.015) | 0.0000 | -0.0793 | 0.195 |
| -0.05 | True | **False** | `hold_phase_target_tracking` | 0.0084 (tol 0.0075) | -0.0000 | -0.0416 | 0.248 |
| +0.05 | True | True | none | 0.0016 | -0.0000 | 0.0484 | 0.534 |
| +0.10 | True | True | none | 0.0054 | 0.0000 | 0.0949 | 0.362 |
| +0.15 | True | False | `hold_phase_incomplete` (guard trip: `\|axis_error\|` growth) | 0.0064 | -0.0000 | 0.1463 | 0.252 |
| +0.20 | True | False | `hold_phase_incomplete` (guard trip: `\|axis_error\|` growth) | 0.0085 | -0.0059 | 0.1974 | 0.194 |

Note `valid_move_phase = True` in **every** negative-dx cell -- `move_x_pass` uses a looser
tolerance (`max(0.005, 0.25*|target_x_delta|)` = 0.05m for dx=-0.20) than the hold-phase drift
check (`max(0.003, 0.15*|target_x_delta|)` = 0.03m), and the move-phase residual (~3cm) clears the
looser move tolerance but the *identical* ~3cm of subsequent hold-phase catch-up motion fails the
tighter hold-drift tolerance. All four negative-dx failures are `hold_phase_target_tracking`
specifically via the drift term, not the (also-checked) `hold_phase_final_x_error_m` term, which
passes cleanly in every case (0.0000-0.0009m, all far under the 0.03-0.05m tolerance).

### Same grid, fixed-gain baseline (`ur5e_mujoco_torque_osc_tuned_wrist_orient.yaml`)

| dx (m) | move_phase_achieved_x_delta_m | hold_phase_x_drift_from_hold_start_m | valid_all |
|---|---|---|---|
| -0.20 | -0.1973 (99% of target) | 0.0019 | True |
| -0.15 | -0.1495 | 0.0003 | True |
| -0.10 | -0.1004 | 0.0007 | True |
| -0.05 | -0.0503 | 0.0005 | True |
| +0.05..+0.20 | within ~0.6% of target | 0.0005-0.0007 | True |

The baseline's move-phase convergence is ~1-2mm short of target by `t=1.0s` in every cell --
essentially instant -- so there is no catch-up motion left to trip the hold-drift check. This is
the direct point of comparison that isolates the mechanism: same episode structure, same
tolerances, same metric definitions; the only difference is how fast the move-phase converges.

### Raw trace, worst case (`dx=-0.20m`, `height_alpha=0.5`, learned/SAC)

```
t=0.002  x_error= 0.000   target_x=-0.1215  ee_x=-0.1215
t=0.752  x_error=-0.0522  target_x=-0.3008  ee_x=-0.2491   (target still ramping, min-jerk)
t=0.998  x_error=-0.0307  target_x=-0.3215  ee_x=-0.2906   (move_duration_s cutoff)
t=1.502  x_error=-0.0126  target_x=-0.3215  ee_x=-0.3090   (still closing, well into "hold")
t=2.252  x_error=-0.0092  target_x=-0.3215  ee_x=-0.3123
t=2.802  x_error= 0.0009  target_x=-0.3215  ee_x=-0.3224   (essentially converged)
t=3.000  x_error= 0.0003  target_x=-0.3215  ee_x=-0.3218   (episode end)
```

`x_error` shrinks monotonically and smoothly from move-phase end through the entire hold window --
this directly answers Step 1's diagnostic question: it is **still shrinking, not plateaued/stuck**,
and it does eventually reach essentially zero. It is not a case of the policy "giving up" or
"locking in a safe-but-short position" (the pivot prompt's hypothesized mechanism) -- it keeps
working the whole time and gets there. The failure is entirely in how the *evaluation/reward*
metric characterizes that last ~1.5s of closing motion.

### Why this isn't the "accepted once inside a looser tolerance" mechanism speculated in the task brief

The task brief speculated the fix might be "reshaping progress/terminal-quality terms so
undershooting is penalized proportionally to final distance from target, not just accepted once
inside some looser tolerance." That does not match what was found: final distance from target
*is* already ~0 in every failing case, and is already the (passing) `hold_phase_final_x_error_m`
check. There is no undershoot being "accepted" -- there is a different, stricter metric
(drift-from-hold-start, meant to catch instability, not slow-but-safe convergence) that fires
first and independently of the final-distance check.

## Fix attempt (Step 2)

**Implemented** (not merely designed): `rl_gain_scheduling/gain_scheduling_env.py`'s
`_episode_end_quality_score` gained an opt-in `reward.hold_drift_relative_to_target` flag
(default `false`, byte-identical to prior behavior). When `true`, **only the reward-facing
quality score** substitutes `hold_phase_x_drift_from_hold_start_m` with
`abs(hold_phase_final_x_error_m)` (drift-from-*target* instead of drift-from-hold-start-snapshot)
before scoring. Deliberately scoped narrow:

- `transport_metrics.py` (shared with hardware-adjacent tooling, and where
  `hold_phase_x_drift_from_hold_start_m` is a genuinely correct, safety-relevant signal for its
  real purpose) is **not modified**.
- The persisted `run_summary.json` written by `_finalize_trace_logging` is a separate code path
  and is **not modified** -- `eval_gain_scheduler.py`'s pass/fail reporting keeps using the
  original, unmodified, hardware-honest metric. This fix changes only what the training reward
  optimizes toward; it does not relax what counts as a valid transport in reporting.

New config: `config/rl_gain_scheduling_sac_gains_alpha05_reward_v2.yaml` (base:
`config/rl_gain_scheduling_sac_gains_alpha05.yaml`, unmodified, +
`reward.hold_drift_relative_to_target: true`).

New tests (`tests/mujoco/test_gain_scheduling_env.py`):
`test_hold_drift_relative_to_target_absent_by_default` (config without the key behaves as before);
`test_hold_drift_relative_to_target_flag_removes_convergence_penalty` (a synthetic
slow-but-safely-converging trace, built directly rather than via `env.step()` to isolate the
scoring logic from physics, scores strictly higher with the flag on than off). Both pass, along
with the full pre-existing 24-test file, `tests/unit -m unit` (94 passed), and
`tests/mujoco -m mujoco -k "not slow"` (85 passed) -- no regressions.

### Pilot (80,016 timesteps, `--run-name sac_gains_alpha05_reward_v2_pilot`)

Ran to completion in ~72s (fps ~1010-1050). No NaNs. `ent_coef` decayed smoothly
(comparable trajectory to the original pilot). `critic_loss` stable (0.13-1.5, no blowup).
`actor_loss` trended more negative over training (-44 to -58), consistent with healthy SAC
Q-value growth. Judged sane -- proceeded to the full run.

### Full run (3,000,000 timesteps, `--run-name sac_gains_alpha05_reward_v2`)

Ran to completion: exit code 0, verified `model.num_timesteps == 3000000` after loading the
saved model (not just trusting the log). fps 947-965 throughout (~3040s / ~51 min total). No
NaNs; `critic_loss` fluctuated (spiked to ~1.1 mid-run, ~0.06-0.19 elsewhere) but never diverged;
`ent_coef` decayed smoothly to ~0.0046-0.0049 by the end (healthy, not a premature collapse).
Same cosmetic post-save multiprocessing-tempdir-cleanup `OSError` seen in every other run in this
project appears in the log after "Saved final model" -- unrelated to training, model save and
`SubprocVecEnv` teardown both completed first. Final model:
`outputs/rl_gain_scheduling/sac_gains_alpha05_reward_v2/models/sac_gain_scheduler_final.zip`.

### Evaluation -- real result: this specific fix did not help, and looks worse on this run

`rl_gain_scheduling/eval_gain_scheduler.py --model-path .../sac_gain_scheduler_final.zip --algo
sac --config config/rl_gain_scheduling_sac_gains_alpha05_reward_v2.yaml --alphas 0.5 --deltas
-0.20 -0.15 -0.10 -0.05 0.05 0.10 0.15 0.20`, run twice (once per `--baseline-config`), the exact
same grid as the prior (reward v1) experiment.

| Comparison | learned valid | baseline valid |
|---|---|---|
| vs `ur5e_mujoco_torque_osc_tuned_adaptive_lambda.yaml` | **0/8 (0%)** | 7/8 (88%) |
| vs `ur5e_mujoco_torque_osc_tuned_wrist_orient.yaml` | **0/8 (0%)** | 8/8 (100%) |

This is *worse* than the prior (reward v1) SAC gains-mode attempt's 2/8. Per-cell diagnosis
(`outputs/rl_gain_scheduling/eval/sac_gains_alpha05_reward_v2_vs_adaptive_lambda/learned/runs/*/summary.json`):

| dx (m) | termination_reason | hold_failure_reason | hold_phase_final_x_error_m | move_phase_max_abs_orientation_error_rad |
|---|---|---|---|---|
| -0.20 | `\|axis_error\| grew for 100 consecutive steps` | (guard trip) | -0.021 | 0.167 |
| -0.15 | duration_complete | `hold_phase_target_tracking` | -0.0001 | 0.113 |
| -0.10 | duration_complete | `hold_phase_target_tracking` | -0.0004 | 0.047 |
| -0.05 | duration_complete | `hold_phase_target_tracking` (+ move-phase tracking) | -0.0017 | 0.014 |
| +0.05 | duration_complete | `hold_phase_target_tracking` | ~0.0000 | 0.011 |
| +0.10 | `\|axis_error\| grew for 100 consecutive steps` | (guard trip) | ~0.0000 | 0.019 |
| +0.15 | `\|axis_error\| grew for 100 consecutive steps` | (guard trip) | ~0.0000 | 0.028 |
| +0.20 | `\|axis_error\| grew for 100 consecutive steps` | (guard trip) | ~0.0000 | 0.037 |

Two findings from this table:

1. **On the 4 cells that still run to `duration_complete`, the original mechanism this fix
   targeted is gone**: `hold_phase_final_x_error_m` is still ~0 (as before), but now they *also*
   fail `hold_phase_x_drift_from_hold_start_m` in 3/4 cases anyway (the flag only changes the
   *reward*, not the *evaluation* metric used for `valid_move_and_hold`, by design -- see the
   fix description above). This is expected and not a problem by itself: the fix was never meant
   to change what counts as "valid" in reporting.
2. **The real, unwelcome change: `|axis_error|` guard trips roughly doubled** (4/8 cells now trip
   it, vs 2/8 in the prior reward-v1 run) and now includes previously-safe negative-dx cells
   (`dx=-0.20`) as well as three positive-dx cells. This is a materially different, more
   guard-prone policy, not just a re-scored version of the same one.

**Plausible (not proven) causal account**, consistent with everything both this experiment and
the prior one measured: the `hold_phase_x_drift_from_hold_start_m` term in the *old* reward
formula, even though it was misclassifying safe convergence as drift, also happened to act as a
conservatism pressure discouraging aggressive gain schedules that converge fast but flirt with
the instability boundary the `|axis_error|`-growth guard polices (AGENTS.md sec 3's documented
"the ceiling is directional" asymmetry -- less restoring authority in `-X`). Removing that
pressure let this training run's policy explore more aggressive schedules; SAC's own stochastic
optimization trajectory (fixed algorithmic seed, but a materially different reward landscape from
step one, so the entire 3M-step trajectory diverges, not just the last few episodes) landed on
one that trips the instability guard more often, not less. This is a real, evidenced trade-off
this specific run exhibits, not confirmed as *the* mechanism (a single training run, seed=0 only,
cannot separate "this reward change causes more guard trips" from "this particular random
trajectory happened to land somewhere worse" -- distinguishing those would need multiple seeds,
out of scope for the remaining time here).

## Recommendation

- **Do not adopt `reward.hold_drift_relative_to_target` / the reward-v2 config.** It did not
  improve `valid_move_and_hold` on the canonical grid -- it went from 2/8 to 0/8, worse than
  every prior RL attempt at this problem including the very first (0/8 PPO gains-mode).
- **The Step-1 diagnosis itself stands as a real, useful finding independent of the fix's
  outcome**: the "undershoot" framing in `docs/status/sac_gains_mode_experiment_2026-07-29.md`
  was incomplete. The policy from that run was not stuck or satisfied early -- it converged to
  within ~1mm of target in every failing negative-dx case; the failure was a hold-phase-drift
  metric (correctly designed for its original hardware-relevant purpose: catching wander after a
  move is supposed to be settled) misclassifying a still-converging tail as instability. Anyone
  revisiting this problem should not re-diagnose "undershoot" from `move_phase_achieved_x_delta_m`
  alone -- pull `hold_phase_final_x_error_m` and the raw trace first.
- **Fixing the training *signal* for this specific metric mismatch, at least via this
  narrowly-scoped substitution, trades one failure mode for a worse one.** The old reward
  (unintentionally) suppressed a fragility that the new reward exposed. A more careful fix would
  need to address the *guard-adjacent aggressiveness* trade-off directly (e.g. a smoothness/
  guard-margin-aware term calibrated specifically for the harder `-X` direction) rather than
  simply removing the one term that happened to be suppressing it -- out of scope for the
  remaining time in this task.
- **Six-plus real RL attempts now (4 PPO, 2 SAC prior, this SAC reward-v2 variant), across two
  algorithms, two action-space designs, and now two reward formulations, have not beaten the
  fixed-gain controller on this problem.** The non-RL `wrist_orientation_task` structural fix
  (`docs/status/wrist_orientation_task_2026-07-29.md`) remains the best available solution to the
  original `height_alpha=0.5` directional-asymmetry bug; this result adds further evidence for
  that conclusion, and does not change it.
- Given the remaining time budget, the honest per-instructions call here is to **stop pursuing
  further reward-shaping variants on this exact problem** (Step 2's negative result is itself the
  answer, not a reason to keep iterating) rather than force a third reward attempt without a new,
  concrete, well-evidenced hypothesis to test. The task's optional "generalization" pivot
  (randomized `height_alpha` gain-scheduling) was not attempted given the remaining time and the
  greater evidentiary value of finishing this diagnosis honestly and completely with real trained
  results rather than adding a seventh partially-explored attempt.

## Files changed

- `rl_gain_scheduling/gain_scheduling_env.py` -- `_episode_end_quality_score` gained the opt-in
  `reward.hold_drift_relative_to_target` flag (default `false`, byte-identical to before).
- `tests/mujoco/test_gain_scheduling_env.py` -- two new tests:
  `test_hold_drift_relative_to_target_absent_by_default`,
  `test_hold_drift_relative_to_target_flag_removes_convergence_penalty`.
- `config/rl_gain_scheduling_sac_gains_alpha05_reward_v2.yaml` -- new config (base:
  `config/rl_gain_scheduling_sac_gains_alpha05.yaml`, unmodified, + the new reward flag).
- This file.
- Not touched: `AGENTS.md`, any `README.md`, `hardware/` (simulation-only task),
  `transport_metrics.py` (shared code, deliberately left unmodified -- see Fix-attempt section),
  any file belonging to the other two SAC experiments from earlier today (left as-is).

## Tests run

- `pytest tests/mujoco/test_gain_scheduling_env.py -q` -- 24 passed (22 pre-existing + 2 new).
- `pytest tests/unit -m unit -q` -- 94 passed.
- `pytest tests/mujoco -m mujoco -q -k "not slow"` -- 85 passed.

## Tests not run

- No hardware-in-the-loop or real-RTDE tests (simulation-only task).
- Full repo-wide `pytest -q` not re-run; the unit + mujoco subsets above cover everything this
  change could plausibly affect (`gain_scheduling_env.py` and its own test file). The one known
  pre-existing failure unrelated to this work
  (`tests/hardware/test_direct_torque_transport_timing.py::test_transport_records_timing_and_deadline_loop`)
  was not re-verified since this task never touches that path.

## Rollback

```
git revert <this-task's-commit-sha(s)>
```
or, to remove without a revert commit:
```
git checkout <prior-sha> -- rl_gain_scheduling/gain_scheduling_env.py tests/mujoco/test_gain_scheduling_env.py
rm config/rl_gain_scheduling_sac_gains_alpha05_reward_v2.yaml docs/status/rl_undershoot_instability_diagnosis_2026-07-29.md
```
The `hold_drift_relative_to_target` flag defaults to `false`, so simply not setting it in any
config (the state of every config in this repo except the new one above) is itself a full
functional rollback of the reward-shaping change without touching any code. The trained
model/logs under `outputs/rl_gain_scheduling/sac_gains_alpha05_reward_v2*/` and
`outputs/rl_gain_scheduling/eval/sac_gains_alpha05_reward_v2_vs_*/` are gitignored artifacts, not
tracked -- no rollback needed for those.

