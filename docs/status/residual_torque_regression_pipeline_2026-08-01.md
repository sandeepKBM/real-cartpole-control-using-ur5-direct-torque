# Offline residual-torque regression: phase-1 pipeline, 2026-08-01

## Context

`docs/status/nonlinear_controller_research_2026-07-31.md` (last night's research survey)
ranked options for giving the OSC controller
(`controller_core/x_axis_cartesian_impedance.py`) more representational capacity to close the
remaining sim-to-real gap. Its top recommendation, explicitly preferred over a 7th RL attempt
(this repo has a documented 6/6 real RL gain-scheduling failure record, root-caused to a
deceptive reward landscape, not an algorithm problem -- see `docs/CURRENT_STATUS.md`): **offline
supervised residual-torque regression**, using data the repo's residual-observer pipeline
already computes. The user was asked whether to launch long RL training runs and explicitly
chose this alternative instead, specifically because it is an offline fit on already-safe
logged data rather than online trial-and-error under a reward function.

This document covers **phase 1 only**: data pipeline, fit, and honest evaluation. No real-time
controller integration. That is explicitly out of scope here, per the task and per the research
doc's own sequencing (real payoff is gated on real-hardware data volume that does not exist yet).

## 1. What data existed already, and what didn't

The residual-observer math (`controller_core/dynamics_residual.py`:
`predict_joint_acceleration`, `joint_acceleration_residual`) and its online wiring
(`hardware/residual_observer_worker.py`, and the inline synchronous path inside
`hardware/direct_torque_transport.py`) are **only wired into the real-hardware `direct_torque`
control loop** -- confirmed by grepping `tools/ur5e_mujoco_torque_experiments.py` and
`tools/ur5e_move_hold_transport.py` (the sim rollout engine and its sweep driver) for
`qdd_residual`/`residual`: no hits. Checked local trace data directly:

- `outputs/hardware_transport/direct_torque_*` (real-hardware traces, dated 2026-07-28) predate
  the residual observer's existence (landed 2026-07-29) -- no `qdd_residual` field, confirmed by
  inspecting a trace row's keys.
- Every sim sweep trace checked (`friction_ff_at_neg45_2026-07-31/*`, `pose_sweep_*`, etc.) also
  lacks the field, for the reason above -- the sim engine never computes it.
- Real-hardware trace data physically lives on `thinkrobot`, not this machine (`westeros`), per
  `AGENTS.md` and confirmed by there being no newer real-hardware traces locally than
  2026-07-28. `tools/pull_hardware_logs_ssh.sh` (landed 2026-08-01) is the intended path to pull
  it once a real session logs the observer's fields, but no such session has happened yet.

**Conclusion: no existing local trace data (sim or real) had the fields this pipeline needs.**
This is not a blocker, though -- sim traces already log everything required to reconstruct the
same quantity *post-hoc*: per-step `q`, `qd`, `tau` (the true final physical torque delivered to
`data.ctrl` that cycle -- `simulation/ur5e_mujoco_torque.py`'s `"tau"` trace field, not
`tau_controller`/`tau_applied` which are pre-clip/pre-filter), and `qfrc_bias` (MuJoCo's own
`C(q,qd)qd + g(q)`, which excludes joint friction/damping by construction -- exactly the
"effective uncompensated torque" this pipeline wants to see, since friction is what should show
up as residual). `M(q)` is recomputed post-hoc via `PinocchioUR5eDynamics` (already validated
<1e-8 Nm mass-matrix parity vs MuJoCo). No new trace fields, no new instrumentation -- pure
algebra on data that already exists.

### New sim data generated (modest, per the task's explicit scope)

Checked `uptime`/`nproc` before generating anything (load average ~0.3-0.9 on 72 cores, idle).
Generated 16 short move-hold rollouts via `tools/ur5e_move_hold_transport.py --config
config/ur5e_mujoco_torque_osc_tuned.yaml` (friction feedforward **off**, so the sim's real
Coulomb+viscous joint friction, added 2026-07-31, is present and uncompensated -- the actual
residual signal of interest), across 4 poses for basic coverage variety:

- unrotated `height_alpha` 0.2, 0.3, 0.5 (`hardware/poses.py::q_for_height_alpha`)
- the -45 deg base-rotation clearance pose (`HEIGHT_ALPHA_0_5_CLEARANCE_Q`), kept to small
  displacement (dx <= 0.04 m) to stay inside its known-safe envelope (AGENTS.md sec 3: this pose
  fails via Y-drift at larger displacement -- not relevant to this pipeline, just avoided to keep
  the data clean rather than dominated by guard-trip transients)

Each pose: dx in {0.02, ~0.05} m, hold in {1, 2} s, move duration 1 s, torque scale 1.0, seed 0.
Output under `outputs/ur5e_mujoco_torque_transport/residual_pipeline_datagen_2026-08-01/`
(gitignored, not committed). **19,517 total rows** after target reconstruction, from 16 runs.
This is pipeline validation data, not a real data-collection campaign -- explicitly scoped that
way per the task.

## 2. Pipeline built

- `tools/analysis/residual_data.py` -- loads `trace.jsonl` rows, replays them through
  `JointAccelEstimator` (`hardware/joint_accel_estimator.py`, same class the real observer uses)
  to get `qdd_measured`, calls `predict_joint_acceleration`/`joint_acceleration_residual`
  (reused directly from `controller_core/dynamics_residual.py`, not reimplemented), then
  `tau_residual = M(q) @ qdd_residual`. Returns one `ResidualDatasetRun` per trace file, kept
  separate (never flattened into one array without a run index).
- `tools/analysis/fit_residual_torque_model.py` -- CLI: loads a set of trace files, splits
  **by run** (not by row -- rows 2 ms apart in the same rollout are extremely autocorrelated; a
  row-level split would leak near-duplicate samples between train/test and inflate the score),
  featurizes with `controller_core.residual_torque_model.all_joint_features`, fits one OLS
  weight vector per joint (`numpy.linalg.lstsq` -- no scipy/sklearn needed for plain OLS), and
  evaluates held-out performance against a zero-residual (do-nothing) baseline.
- `controller_core/residual_torque_model.py` -- the deterministic-cost, numpy-only inference
  side (feature basis + `compute_residual_torque(weights, q, qd)`). **Not imported or called by
  `x_axis_cartesian_impedance.py` anywhere** -- see section 4.

### Model form (first cut, deliberately simple)

Per-joint feature vector (6 features, using only that joint's own `q_j`/`qd_j`, not the full
12-dim state -- a deliberate simplicity choice given how little data exists, see section 5):

```
phi_j(q_j, qd_j) = [1, qd_j, tanh(qd_j / 0.05), qd_j*|qd_j|, sin(q_j), cos(q_j)]
tau_residual_j_pred = w_j . phi_j(q_j, qd_j)
```

The `tanh`/linear/quadratic velocity terms mirror the existing `friction_feedforward` term's
functional form (Coulomb + viscous); `sin`/`cos` position terms catch any leftover
position-dependent bias. 36 total scalar weights (6 joints x 6 features).

## 3. Honest evaluation results

Train/test split: 12 train runs (14,517 rows) / 4 test runs (5,000 rows), seed 0, held-out runs
were the two largest-displacement (`dx=0.05m`) runs at `height_alpha` 0.2 and 0.5 (both hold
durations) -- picked by the random split, not cherry-picked.

| joint | rmse zero-baseline | rmse model | R2 vs zero | mean &#124;residual&#124; (test) | corr(&#124;residual&#124;, &#124;qd&#124;) |
|---|---|---|---|---|---|
| 0 shoulder_pan | 0.655 | 0.012 | +1.000 | 0.540 | +1.00 |
| 1 shoulder_lift | 3.111 | 1.107 | +0.873 | 2.531 | +0.75 |
| 2 elbow | 1.775 | 0.032 | +1.000 | 1.455 | +1.00 |
| 3 wrist_1 | 0.168 | 0.004 | +0.999 | 0.132 | +1.00 |
| 4 wrist_2 | 0.025 | 0.002 | +0.994 | 0.018 | +1.00 |
| 5 wrist_3 | 0.248 | 0.003 | +1.000 | 0.192 | +1.00 |

(units: Nm; full numbers including train-set metrics in
`outputs/residual_pipeline_eval_2026-08-01/report.json`, gitignored, reproducible by rerunning
the fit script below)

**Does the fit meaningfully beat the zero-residual baseline?** Yes, clearly, on every joint,
including the held-out set -- RMSE drops by 1-2 orders of magnitude on 5 of 6 joints. Joint 1
(shoulder_lift, torque limit 150 Nm per `config/ur5e_mujoco_torque_osc_tuned.yaml`) is the
weakest: R2=0.873 held-out, mean residual magnitude still ~2.5 Nm (~1.7% of its torque limit)
after the model's correction. This is a real, not-fully-explained gap on the joint that carries
the most gravity-dependent load -- plausibly friction there depends on more than its own
(q, qd) (e.g. load-dependent normal force through the gearbox), which this per-joint,
own-state-only feature basis cannot represent by construction.

**What does the residual actually look like?** Dominated by joint velocity: `corr(|residual|,
|qd|)` is 0.75-1.00 across every joint. This is exactly what you'd expect if the residual is
mostly picking up the sim's own Coulomb+viscous joint friction (`frictionloss`/`damping` in
`assets/ur5e_torque/ur5e_torque.xml`, added 2026-07-31) -- which `qfrc_bias` deliberately
excludes, so it shows up entirely as "unmodeled" from this pipeline's point of view. This
matches AGENTS.md's own framing of `friction_feedforward` and is a useful sanity check that the
target-construction math is doing what it claims.

### Honest caveat: this result is easier than it looks

The near-perfect fit on 5/6 joints is **expected to be somewhat inflated**, not a strong claim
about real unmodeled dynamics: the sim's actual disturbance source (Coulomb+viscous friction) is
itself a smooth, deterministic function of `qd` that closely matches the chosen feature family
(`tanh` + linear + quadratic velocity terms were chosen *because* they mirror that exact
functional form). Real hardware's unmodeled dynamics will not be this clean -- stick-slip,
hysteresis, presliding compliance, sensor noise, load-dependent effects are all things this
static basis can't represent as cleanly (this is exactly the LuGre/asymmetric-Coulomb gap
`docs/hardware/LUGRE_FRICTION_MODEL_PLAN.md` and the 2026-08-01 literature review already flag
for the *static* friction-feedforward term this residual model closely resembles).

**What this result actually validates is the pipeline mechanics**: target construction is
algebraically correct (confirmed independently by the analytic unit tests in
`tests/mujoco/test_residual_data_pipeline.py`, which inject a known missing torque and verify
`tau_residual` reconstructs it exactly), the run-level train/test split avoids the obvious
leakage trap, and the evaluation code correctly detects a strong signal when a strong signal is
present. It does **not** validate that this feature basis or this fit would generalize to real
hardware, where the true residual source and its dependence on state are currently unknown and
almost certainly richer than "smooth function of one joint's own (q, qd)".

### Small-sample caveat

16 rollouts, 4 poses, all 1-3 s long, all sim-only, all from the same controller config and
gain set. This is not a claim of coverage across the operating envelope, and the honest 12/4
run train/test split (rather than a larger k-fold) reflects how little independent data exists.

## 4. Deterministic-cost inference: designed, tested, NOT wired into the controller

`controller_core/residual_torque_model.py::compute_residual_torque(weights, q, qd)` is a
plain numpy function: build the fixed 6x6 feature matrix, one `einsum` dot product, return a
`(6,)` torque correction. Fixed shape and flop count regardless of input values (no
data-dependent branching), matching `docs/status/direct_torque_controller_phase_profiling_2026-07-31.md`'s
stated ~0.5-0.7 ms real headroom in the 500 Hz `direct_torque` loop. Measured on this machine
(westeros, not thinkrobot -- absolute numbers don't transfer across machines per that same
profiling doc, only the relative statement that this is a small fraction of headroom): **~34
us/call**, warmed up, 20,000 reps.

Unit-tested in `tests/unit/test_residual_torque_model.py` (11 tests): feature-vector shape and
values, `tanh` saturation, zero-velocity zeroing of velocity-dependent terms, NaN rejection,
per-joint state isolation (joint j's features use only `q[j]`/`qd[j]`), zero-weight ->
zero-output, a manual-dot-product cross-check, weight-shape validation, and fixed-output-shape
across a range of inputs.

**Deliberately not wired into `x_axis_cartesian_impedance.py`'s `compute()`.** Judgment call per
the task's own instruction to weigh whether tonight's data justifies real wiring: it does not.
The fit quality that exists is on 16 short sim rollouts at one gain setting, with a feature basis
that (per the caveat above) is suspected to be flattering itself against sim's own friction
model. Wiring this into the control path now would repeat exactly the failure pattern this
whole approach was chosen to avoid -- trusting an insufficiently-validated learned correction
near real torque and letting the safety guards catch it after the fact, which is what six RL
attempts already did. If this is ever promoted, it must follow the same pattern as
`friction_feedforward`/`wrist_orientation_task`: a new `CartesianImpedanceConfig` flag,
default off, zero behavior change for every existing config, and a byte-identical regression
test -- none of that exists yet, on purpose.

## 5. What would be needed next

1. **A real-hardware trajectory corpus with the residual observer enabled**, collected across
   the operating envelope (multiple poses, displacements, hold durations) -- not existing yet at
   any scale (one 2026-07-28 lab session predates the observer; nothing since has run
   `direct_torque` with `enable_residual_observer=True` on the real arm as far as local
   artifacts show). `tools/pull_hardware_logs_ssh.sh` is ready to pull it once collected.
2. **Refit on real data and re-run this exact evaluation** -- if `corr(|residual|, |qd|)` stays
   as strong on real data as it is here, that's real evidence the static feature basis is
   sufficient; if it's much weaker, or joint 1's gap widens, that's evidence for a richer
   feature set (cross-joint terms, or a dynamic/hysteretic state as in the LuGre plan) before
   this is worth pursuing further.
3. **Only then**, if the held-out fit quality on real data clears a bar comparable to what
   `friction_feedforward` already achieved (2.5-9x steady-state error reduction, see AGENTS.md
   sec 3), consider the flag-gated `controller_core` wiring described in section 4, with its own
   4-category rigor sweep before any real-hardware exposure -- matching this repo's established
   promotion pattern for every prior controller addition.

## Files changed

- `controller_core/residual_torque_model.py` (new) -- numpy-only feature basis + inference,
  not imported anywhere else in `controller_core/`.
- `tools/analysis/__init__.py`, `tools/analysis/residual_data.py`,
  `tools/analysis/fit_residual_torque_model.py` (new) -- offline data pipeline + fit/eval CLI.
- `tests/unit/test_residual_torque_model.py` (new, 11 tests).
- `tests/mujoco/test_residual_data_pipeline.py` (new, 5 tests).
- `docs/status/residual_torque_regression_pipeline_2026-08-01.md` (this file).
- Not committed (gitignored, per `AGENTS.md`): the 16 generated sim traces under
  `outputs/ur5e_mujoco_torque_transport/residual_pipeline_datagen_2026-08-01/` and the fit
  report/weights under `outputs/residual_pipeline_eval_2026-08-01/` -- reproducible by rerunning
  the commands in this doc.
- No file in `controller_core/x_axis_cartesian_impedance.py`, `controller_core/safety.py`, or
  `hardware/safety.py` was touched.

## Tests run

- `python -m pytest -q -m "unit or mujoco"`: **244 passed, 3 xfailed, 250 deselected** -- matches
  the pre-existing baseline exactly (this change added 16 new passing tests: 11 unit + 5 mujoco;
  net delta from the prior baseline is exactly that addition, no regressions).

## Tests not run

- `tests/hardware/` (mocked RTDE, `-m hardware`) -- not touched by this change, not re-run.
- No real-hardware command was run (none is available in this environment).

## Rollback

```
git rm controller_core/residual_torque_model.py
git rm -r tools/analysis/
git rm tests/unit/test_residual_torque_model.py tests/mujoco/test_residual_data_pipeline.py
git rm docs/status/residual_torque_regression_pipeline_2026-08-01.md
```
(or `git revert <this commit>`). No existing file was modified, so a plain revert is
lossless -- nothing else depends on any of these new files yet.

## Reproduce

```
python tools/analysis/fit_residual_torque_model.py \
  --trace-root outputs/ur5e_mujoco_torque_transport/residual_pipeline_datagen_2026-08-01 \
  --test-fraction 0.25 --seed 0 \
  --output-json outputs/residual_pipeline_eval_2026-08-01/report.json \
  --output-weights outputs/residual_pipeline_eval_2026-08-01/weights.npy
```
(requires regenerating the sim traces first, via `tools/ur5e_move_hold_transport.py` -- traces
are gitignored, not part of this commit; see section 1 for the exact commands used.)
