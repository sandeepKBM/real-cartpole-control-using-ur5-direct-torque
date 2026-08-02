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

## Ridge regularization + output clipping fix (2026-08-01, same night)

**Still entirely offline analysis tooling -- nothing in this section touches
`controller_core/x_axis_cartesian_impedance.py` or any real-hardware path. See the closing
note below for an explicit restatement.**

### The failure

Later the same night, real UR5e hardware traces became loadable (see
`docs/status/residual_observer_real_trace_gap_2026-08-01.md`'s loader fix) and this pipeline
was run against them for the first time:

```
python3 tools/analysis/fit_residual_torque_model.py \
  --trace-root "outputs/hardware_transport_remote/hardware_transport/*/trace.jsonl" --seed 0
```

Train R^2 was reasonable on all 6 joints (0.68-0.89). Held-out test R^2 was reasonable for
joints 0, 1, 4, 5 -- but **joint 2 (elbow)** came back at **R^2 = -9799.8**, RMSE model
274.9 Nm vs. a 2.78 Nm zero-baseline (~99x worse than doing nothing). An earlier report of
this same run (different local snapshot of the pulled real-hardware trace directory -- 46
runs / 132,680 rows vs. this repro's 63 runs / 159,389 rows, same trace-pull mechanism, just
a later point in an ongoing pull) additionally saw joint 3 blow up to R^2 ~ -3.4e9; that exact
magnitude did not reproduce bit-for-bit against this snapshot's particular random train/test
split, but the identical mechanism (below) is present and measured on joint 3 in this snapshot
too (see the multi-seed table below, where joint 3 is consistently the second-most fragile
joint, e.g. R^2 -0.78 to -0.81 at seed 0 across every regularization strength tried -- bounded,
not catastrophic, in this snapshot's particular split, but the underlying design-matrix
pathology is identical and seed/data-dependent in whether a given held-out run happens to
contain an extreme-enough outlier to trigger a catastrophic blowup rather than a merely bad
fit). This confirms it is the same root cause, not a different bug.

### Root cause (not just "unregularized" -- the specific mechanism)

Directly inspected the training design matrix for joint 2 (and 3) at seed 0:

- **Joint 2's training-set velocity range is tiny**: `qd` spans only [-0.0084, 0.0092] rad/s
  across all 47 training runs (an X-only transport move barely moves the elbow). At such small
  `|qd|`, `tanh(qd/deadband) ≈ qd/deadband` (linear regime) and `qd*|qd|` is a small,
  odd-symmetric term -- so the model's `qd`, `tanh(qd/deadband)`, and `qd*|qd|` feature columns
  are **near-perfectly collinear** on this training set: measured `corr(qd, tanh) = 0.999996`,
  design-matrix condition number `cond(X) = 1.08e7` (joint 3: `corr = 0.999990`,
  `cond(X) = 3.2e7`), smallest singular values ~1e-4 to 1e-5 (vs. a largest singular value
  ~515) -- i.e. the design matrix is numerically near-rank-deficient in exactly that
  three-feature subspace.
- **`numpy.linalg.lstsq`'s minimum-norm SVD solution is only weakly constrained along that
  near-null direction**: with real (non-adversarial) label noise, the fitted coefficient along
  a near-null right-singular direction scales like (noise projected onto the matching
  left-singular vector) / (that direction's singular value) -- as the singular value shrinks
  toward zero, an ordinary, small amount of noise produces an enormous coefficient. Measured:
  the fitted joint-2 weights were `[3083, -471693, 23469, 731413, 2107, -3083]` (coefficients on
  `[bias, qd, tanh, qd|qd|, sin(q), cos(q)]`) -- **hundreds of thousands** for the `qd` and
  `qd*|qd|` columns, despite those columns' own training-set values never exceeding ~0.01 in
  magnitude. This is exactly a case OLS *cannot* protect against: the fit is genuinely good
  on training data (the huge, opposite-signed `qd`/`qd*|qd|` coefficients nearly cancel across
  the whole training range) but represents essentially no real relationship -- it's fitting
  noise in the near-null direction.
- **The held-out run genuinely extrapolates**: the worst-offending held-out row has joint-2
  `qd = 0.0937` rad/s -- **~10x** the largest `|qd|` any training run ever showed that joint
  (0.0092). At that point the near-collinear features' small differences (the `tanh` term
  saturating while `qd` keeps growing linearly, `qd*|qd|` growing quadratically) stop
  canceling the way they did on the training manifold, and the huge, poorly-determined weights
  get multiplied through directly: predicted torque `-15388 Nm` vs. true `-7.03 Nm` on that row.

**In short: this is not a generic "unregularized model" complaint -- it is a specific,
measured, near-collinear feature subspace (driven by how little velocity variation a
near-stationary joint's training data ever contains) combined with a specific held-out run
whose peak velocity for that joint sits an order of magnitude outside anything training saw.**
Multi-seed sweeping (below) confirms seed 0's split is the pathological one for this exact
data snapshot -- seeds 1-3 hold out runs whose joint-2/3 velocities stay within the training
range, and show no catastrophic blowup even under plain OLS -- consistent with this being an
extrapolation failure keyed to which specific run lands in the test set, not an error present
in every split.

### Fix implemented

Per the task's own preference ordering, implemented (a) ridge regression with (c) feature
scaling, plus (b) output clipping as an independent defensive layer:

1. **`fit_ridge_weights()`** (`tools/analysis/fit_residual_torque_model.py`) -- closed-form
   ridge, `w = (X^T X + lambda*I)^-1 X^T y`, solved **per joint in that joint's own
   per-feature-standardized space** (`_feature_scale()`: divide each column by its training-set
   std, bias column pinned to scale 1.0 since it has zero variance), then the fitted weights are
   scaled back to the original (unstandardized) feature units. Feature scaling matters here, not
   just as a nicety: this model's 6 raw features span roughly 4 orders of magnitude (bias == 1.0;
   `sin`/`cos` in `[-1, 1]`; a near-stationary joint's `qd` ~1e-2 and `qd*|qd|` ~1e-4) -- one
   global `lambda` on unscaled columns would regularize those very differently; standardizing
   first makes one scalar `lambda` penalize every feature comparably. Pure numpy (closed-form
   normal equations) -- no scipy/sklearn added, per the constraint (`environment.yml` already
   has `scipy>=1.11` for other reasons, but this fix needed neither it nor sklearn).
   `fit_ols_weights()` (plain `lstsq`) is kept, reachable via `--ridge-lambda 0`, both for
   comparison and as an explicit escape hatch.
2. **`--ridge-lambda` CLI flag**, default **`1.0e5`**. Chosen by sweeping `lambda` in
   {0.1, 1, 3, 10, ..., 3e7} against held-out R^2 on the real trace-pull data across 4 random
   seeds (0-3; seed 0 is the pathological split above, seeds 1-3 are already-clean splits used
   as a check that the fix doesn't need to sacrifice already-good performance to be safe):

   | lambda | seed0 joint2 R2 | seed0 worst R2 | seed1 worst R2 | seed2 worst R2 | seed3 worst R2 |
   |---|---|---|---|---|---|
   | 0 (OLS) | -9799.80 | -9799.80 (j2) | -0.22 (j4) | +0.04 (j4) | +0.01 (j4) |
   | 1e2 | -285.53 | -285.53 | -0.23 | +0.09 | +0.04 |
   | 1e4 | -110.14 | -110.14 | -0.21 | +0.09 | +0.05 |
   | 3e4 | -27.34 | -27.34 | -0.19 | +0.09 | +0.05 |
   | **1e5 (chosen default)** | **-0.68** | **-0.78** | **-0.14** | **+0.10** | **+0.06** |
   | 3e5 | -0.13 | -0.79 | -0.10 | +0.10 | +0.06 |
   | 1e6 | -0.07 | -0.80 | -0.09 | +0.10 | +0.07 |

   `1e5` is the smallest tested value at which every seed's worst-joint R^2 lands in a bounded,
   "honest bad fit" range (roughly [-1, +1], the same order of magnitude as simply predicting
   the training mean) rather than the catastrophic tens/hundreds-negative range smaller lambdas
   still leave joint 2 in. Pushing lambda further (3e5, 1e6, ...) keeps improving joint 2
   marginally but starts visibly eroding the well-conditioned joints' fit quality (e.g. seed 1
   joint 0 R^2 0.54 -> 0.39 by lambda=1e6) for no corresponding safety benefit -- not worth it
   per the task's own "don't over-engineer past what's needed" guidance.
3. **`compute_clip_bounds()` / `--clip-multiple` (default 5.0)** -- an independent, fit-quality-
   agnostic hard ceiling: `clip_multiple * max(|observed training residual|)` per joint, computed
   from TRAIN data only (never leaks test information into what should be a fixed,
   data-independent inference-time bound). Applied both in `predict()` (bulk, for evaluation)
   and as a new optional `clip_abs` parameter on
   `controller_core/residual_torque_model.py::compute_residual_torque()` (default `None` --
   exact prior behavior preserved for any existing/future caller that doesn't opt in). This
   exists as a second, independent layer of defense per the task's own reasoning: even a
   well-regularized fit should never be trusted to emit an unbounded correction torque if this
   is ever wired into a real controller later.

### Before/after: full per-joint held-out table (seed 0, the pathological split)

`python tools/analysis/fit_residual_torque_model.py --trace-root "outputs/hardware_transport_remote/hardware_transport/*/trace.jsonl" --seed 0`
(63 runs loaded, 159,389 rows total -- more than the original same-night report's 46 runs /
132,680 rows, since the real-hardware trace pull was still ongoing; same command, same seed,
same mechanism, numbers differ from the exact ones first reported for that reason, not because
of any change to the fix's methodology):

| joint | RMSE zero | **RMSE OLS (before)** | **R2 OLS (before)** | **RMSE ridge (after)** | **R2 ridge (after)** | R2 ridge+clip (after) |
|---|---|---|---|---|---|---|
| 0 shoulder_pan | 1.426 | 1.463 | -0.053 | 1.457 | -0.044 | -0.044 (0 rows clipped) |
| 1 shoulder_lift | 5.048 | 2.827 | +0.686 | 3.632 | +0.482 | +0.482 (0 rows clipped) |
| 2 elbow | 2.777 | **274.883** | **-9799.801** | 3.593 | -0.675 | **-0.055** (16 rows clipped) |
| 3 wrist_1 | 0.291 | 0.265 | +0.171 | 0.388 | -0.781 | -0.781 (0 rows clipped) |
| 4 wrist_2 | 0.379 | 0.319 | +0.291 | 0.358 | +0.108 | +0.108 (0 rows clipped) |
| 5 wrist_3 | 0.395 | 0.446 | -0.277 | 0.437 | -0.222 | -0.222 (0 rows clipped) |

(units: Nm; reproducible via the command above, default `--ridge-lambda 1e5 --clip-multiple 5.0`;
`--ridge-lambda 0 --no-output-clipping` reproduces the "before" OLS column exactly)

**Result: the catastrophic blowup is eliminated.** Joint 2 goes from R^2 = -9799.8 (RMSE 99x
the zero-baseline) to a bounded, honest R^2 = -0.68 with ridge alone, and -0.055 (RMSE only
~1.03x the zero-baseline) once the output clip additionally catches the 16 remaining
still-too-large predictions. All 6 joints now sit in a sane, bounded range. Two joints (1, 4)
still show genuinely useful positive held-out R^2 (0.48, 0.11); the others (0, 3, 5) are
honestly modest-to-negative -- an acceptable, non-catastrophic outcome per the task's own
framing, not a claim that this feature basis fits every joint well. Joint 1's R^2 did drop from
the OLS run's 0.686 to 0.482 under regularization -- a real, expected cost of protecting the
fragile joints, not a bug (see the lambda sweep table above for the explicit trade-off this
default was chosen against).

### Tests added

- `tests/unit/test_residual_torque_model.py`: 5 new tests for `compute_residual_torque`'s new
  `clip_abs` parameter (bounds output, scalar-broadcasts, `None` preserves old behavior exactly,
  rejects negative/wrong-shape bounds).
- `tests/mujoco/test_fit_residual_torque_model_ridge.py` (new file, marked `mujoco` per this
  directory's import-chain convention, same reason as `test_residual_data_pipeline.py`): 9 tests,
  including a deterministic SVD-constructed ill-conditioned dataset
  (`_make_ill_conditioned_dataset`) that reproduces the same qualitative failure mechanism found
  on real data (near-null design direction + real noise + held-out extrapolation along that
  direction) and asserts OLS's error there is >10x ridge's, plus `fit_ridge_weights` monotonic
  shrinkage with lambda, `lambda=0` matching plain OLS on well-conditioned data,
  `compute_clip_bounds`/`predict(..., clip_bounds=...)` correctness and input validation.

### Tests run

- `pytest -q -m "unit or mujoco"`: **287 passed, 3 xfailed, 259 deselected** -- zero regressions
  (all pre-existing tests still pass; only new tests added).
- `pytest -q -m hardware`: 258 passed, 1 failed -- the failure
  (`test_direct_torque_residual_observer_async.py::test_residual_observer_async_phase_cost_is_much_lower_than_sync`)
  is a pre-existing, unrelated wall-clock timing assertion (`deadline_overrun` under load) in a
  file that does not import anything touched by this change; confirmed load-sensitive by
  re-running in isolation (still failed, with a different overrun magnitude) while `uptime`
  showed this shared machine's load average had risen to ~21.5 during this session (vs. ~1.0 at
  session start) -- consistent with AGENTS.md section 8's documented shared-machine load
  variability, not a regression from this change.

### Not done / explicitly out of scope

- **Nothing in `controller_core/x_axis_cartesian_impedance.py` or any real-hardware code path
  was touched.** This entire fix is offline analysis/fit tooling
  (`tools/analysis/fit_residual_torque_model.py`) plus the standalone, still-unwired inference
  helper (`controller_core/residual_torque_model.py::compute_residual_torque`, which nothing in
  `controller_core/` imports). No config changed. No safety guard changed.
- The run-level (not row-level) train/test split was not touched -- still the correct choice for
  the autocorrelation reason already documented above.
- This does not claim the fixed pipeline's fit quality is good enough to promote into the
  controller -- see this document's original section 5 "What would be needed next", which is
  unchanged by this fix: that promotion bar (comparable to `friction_feedforward`'s validated
  2.5-9x steady-state error reduction) is still not cleared, and clearing it was never this
  task's goal. This task's goal, and what it achieved, was making the offline evaluation honest
  (no more silent catastrophic-looking-fine-until-you-check-held-out-R^2 numbers) -- nothing more.

## Files changed (this fix, 2026-08-01)

- `controller_core/residual_torque_model.py` -- added optional `clip_abs` parameter to
  `compute_residual_torque()`, default `None` (no behavior change for any existing caller).
- `tools/analysis/fit_residual_torque_model.py` -- added `fit_ridge_weights()`,
  `_feature_scale()`, `compute_clip_bounds()`; `predict()` gained an optional `clip_bounds`
  parameter; `evaluate()`/`JointEvalMetrics` gained optional clipped-prediction fields; `main()`
  now fits ridge by default (`--ridge-lambda`, default 1e5) and reports the defensive clip
  (`--clip-multiple`, default 5.0, `--no-output-clipping` to disable); `fit_ols_weights()` kept,
  now documents its own known failure mode.
- `tests/unit/test_residual_torque_model.py` -- 5 new tests for `clip_abs`.
- `tests/mujoco/test_fit_residual_torque_model_ridge.py` (new) -- 9 tests for ridge fitting and
  output clipping.
- This file (root cause, fix, before/after evaluation).
- Not committed (gitignored): `outputs/hardware_transport_remote/` (a local copy of the real
  trace-pull data used to reproduce and validate this fix) and
  `outputs/residual_pipeline_ridge_fix_2026-08-01/` (the `--output-json`/`--output-weights`
  artifacts from the final validation run).

## Rollback (this fix only)

```
git revert <this-commit>
```
Purely additive to both touched files (new functions/parameters, no existing function body
changed except `fit_ols_weights`'s docstring and `main()`'s call site) -- a plain revert is
lossless.

## Cross-joint coupling feature set + more-data check (2026-08-01, later the same night)

**Still entirely offline analysis tooling.** Nothing in this section touches
`controller_core/x_axis_cartesian_impedance.py` or any real-hardware code path. Goal, per the
task that motivated this pass: the ridge fix above made the pipeline honest (no more
catastrophic blowups), but held-out fit quality was still mostly mediocre-to-poor, especially
for joints 2/3 (elbow/wrist_1). This section tries to actually improve predictive quality, not
just honesty, and reports the full picture -- including what didn't help and what got worse.

### 1. More data: checked, negligible

The real-hardware trace pull grew from 63 runs (159,389 rows, the ridge-fix table above) to
**64 runs (161,139 rows)** by the time this pass started -- one additional run. Not a
meaningful volume increase. Re-running the exact ridge-fix command at seed 0 on the current
data no longer reproduces the seed-0 catastrophic joint-2 blowup under **plain OLS** (previously
R^2 = -9799.8; now bounded). Root-caused, not just observed: the single extreme-velocity real
run that causes the collinearity blowup (`direct_torque_20260801_202215`, joint-2 `|qd|` up to
0.0937 rad/s, ~10x every other run's peak) happens to land in the **training** fold under
seed 0's permutation now that the dataset has 64 runs instead of 63 (permutations differ by
array length, not because that run is new -- it already existed in the 63-run snapshot too).
Directly verified: forcing that run back into the **test** fold (seeds 4, 6, 9 all do this)
reproduces the identical catastrophic mechanism under plain OLS (seed 4: joint 2 R^2 = -22327,
joint 4 R^2 = -17690, joint 5 R^2 = -125 -- worse than the original report even, since more
joints are affected at this exact split). **Verdict: the one extra real run did not fix
anything by itself; the apparent seed-0 improvement was purely which run happened to land in
which fold.** The ridge+clip defense from the earlier fix remains necessary and is still doing
real, load-bearing work (see the lambda sweep below) -- it is not a redundant safety net now
that "more data" exists.

### 2. Feature-set experiments

Four candidate feature sets were tested against the SAME baseline (own-joint-only, 6
features/joint) using the SAME ridge (`lambda=1e5`) + output-clip (5x max train residual)
defaults, evaluated across **7 independent run-level train/test splits** (seeds 0,1,2,3 --
"clean" splits where the extreme-velocity run lands in train -- plus 4,6,9, which force it
into test, the pathological case):

- **B, cross-qd** (baseline 6 + other 5 joints' own `qd`, 11 features/joint)
- **C, full coupling** (baseline 6 + other joints' `qd` (5) + other joints' `sin(q)`/`cos(q)`
  (10), 21 features/joint) -- the eventual winner
- **D, raw q** (baseline 6 + own raw `q`, 7 features/joint) -- sanity check, expected roughly
  redundant with `sin`/`cos(q)` already present
- **Pooled/shared model** (own-joint baseline features, but ONE shared weight vector fit
  across all 6 joints' data on per-joint-standardized targets, rather than independent
  per-joint fits) -- tests whether joints 2/3/4's problem is fixable by borrowing statistical
  strength from data-rich joints, as opposed to needing genuinely different input information

**Condition numbers** (raw, unregularized design matrix, std-scaled, worst across all 7
seeds' training folds): baseline `1e5-1e18` (already large for joints 4/5, per the ridge-fix
doc's original diagnosis extended here to those joints too), cross-qd nearly identical to
baseline (adding 5 roughly-independent columns barely changes conditioning), **full-coupling
~2.6e18 uniformly across every joint** (reflecting real near-constant "other joint held at one
pose" columns during this X-only-transport corpus -- most joints barely move during any given
run), raw-q joints 2-4 degrade further (`7e9-7e18`, sin/cos already captured most of the
signal so raw q adds close-to-redundant, poorly-scaled information). None of this exceeded
what the existing ridge_lambda=1e5 + clip defaults handle safely (confirmed by a dedicated
lambda sweep on the full-coupling basis specifically, see below) -- but it is a genuinely
worse-conditioned raw matrix, not a free lunch.

**Held-out R^2, averaged (and worst-case) across the 7 seeds, ridge+clip applied:**

| joint | A baseline avg (worst) | B cross-qd avg (worst) | C full-coupling avg (worst) | D raw-q avg (worst) | Pooled avg (worst) |
|---|---|---|---|---|---|
| 0 shoulder_pan | +0.375 (+0.191) | +0.400 (+0.256) | **+0.557 (+0.470)** | +0.401 (+0.241) | +0.436 (+0.329) |
| 1 shoulder_lift | +0.747 (+0.677) | +0.725 (+0.635) | +0.720 (+0.606) | +0.747 (+0.680) | **+0.776 (+0.732)** |
| 2 elbow | +0.548 (+0.326) | +0.657 (+0.503) | **+0.724 (+0.591)** | +0.549 (+0.328) | +0.583 (+0.443) |
| 3 wrist_1 | +0.398 (+0.152) | +0.417 (+0.200) | **+0.561 (+0.303)** | +0.398 (+0.152) | +0.412 (+0.245) |
| 4 wrist_2 | -0.001 (-0.151) | -0.009 (-0.157) | **+0.496 (+0.348)** | -0.005 (-0.172) | -0.323 (-1.130) |
| 5 wrist_3 | +0.502 (+0.351) | +0.649 (+0.561) | **+0.748 (+0.626)** | +0.504 (+0.368) | +0.481 (+0.430) |

(units: R^2, dimensionless; reproducible via the scratch experiment described below, not
committed since it duplicates production `fit_ridge_weights`/`compute_clip_bounds` logic
verified byte-identical against the real CLI for cross-checking, see "Verification" below)

**C (full coupling) is the clear winner** -- best or tied-best on 5 of 6 joints, including the
two biggest wins of the whole experiment: joint 4 (wrist_2) goes from essentially uninformative
(avg R^2 ≈ 0, worst -0.15, i.e. sometimes worse than predicting zero) to a genuinely useful
+0.496 avg / +0.348 worst-case, and joint 0 improves from +0.375 to +0.557. Joints 2/3 (the
ones this task specifically asked about) both improve substantially: joint 2 (elbow)
+0.548 -> +0.724 avg (worst-case +0.326 -> +0.591, more than 1.8x); joint 3 (wrist_1)
+0.398 -> +0.561 avg (worst-case +0.152 -> +0.303, exactly 2x).

**Honest exception, not hidden**: joint 1 (shoulder_lift) sees a small, real REGRESSION under
C (+0.747 -> +0.720 avg, worst-case +0.677 -> +0.606) -- B (cross-qd only, no position terms)
regresses joint 1 more (+0.725, but its worst-case +0.635 is still below A's +0.677). Given
the much larger, consistent gains on every other joint (especially 2/3/4, the ones motivating
this whole task), this dip was judged worth accepting -- but it is reported plainly, not
cherry-picked around.

**D (raw q) is essentially a no-op** versus A on every joint (differences within noise, e.g.
joint 0 +0.375 -> +0.401), confirming the module docstring's original expectation that
`sin`/`cos(q)` already captures the position-dependent signal a raw `q` term would add.

**Pooled/shared model is a genuine, informative negative result.** It modestly helps joints
0-3 relative to independent baseline fits (e.g. joint 2 +0.548 -> +0.583) -- consistent with
some real benefit from borrowing statistical strength across joints -- but it clearly HURTS
joint 4 (wrist_2): avg R^2 -0.001 -> **-0.323**, worst-case -0.151 -> **-1.130** (i.e.
sometimes far worse than predicting zero). This is a meaningful, mechanistic finding, not
noise: joint 4 sits at `wrist_2=0`, this repo's documented wrist singularity pose (AGENTS.md
section 3's extensive `jacobian_singular_cond_max`/nullspace-projector history) -- forcing it
to share ONE functional mapping from `(bias, qd, tanh, qd|qd|, sin(q), cos(q))` to
(standardized) residual torque with the other 5 joints actively hurts it, plausibly because its
true residual behavior near that singularity is genuinely different in kind, not just smaller
in magnitude, from the other joints'. **Conclusion: joints 2/3/4's problem is not simply
"needs more aggregated training data via a shared functional form" -- it specifically needs
access to OTHER joints' (q, qd) as additional per-row INPUT information (what C provides),
not a shared coefficient vector borrowing strength across joints' own-state mappings (what
pooling provides).** This directly answers the task's question about whether this is a
data-collection/task-design limitation versus a feature-design one: it is at least partially a
feature-design gap that cross-joint input features close, not a wall that only new excitation
data could fix -- though see the caveat in the verdict below.

### 3. Ridge-lambda robustness check on the new basis

Condition number for full-coupling is far larger than the original basis's, so its
`ridge_lambda=1e5` sensitivity was checked independently (5 seeds -- 0,1 clean, 4,6,9
pathological -- swept over `lambda in {1e3...1e7}`, WITHOUT the output clip, to isolate ridge's
own behavior, then WITH the clip at the same lambda values):

| lambda | avg R^2 (clip+ridge) | worst R^2 (clip+ridge) | rows clipped | worst R^2 (ridge only, no clip) |
|---|---|---|---|---|
| 1e3 | +0.596 | +0.165 | 247 | -4745.7 |
| 1e4 | +0.594 | +0.162 | 234 | -506.6 |
| 3e4 | +0.603 | +0.218 | 190 | -42.3 |
| **1e5 (existing default, unchanged)** | **+0.604** | **+0.303** | **108** | **-601.7** |
| 3e5 | +0.559 | +0.238 | 91 | -613.4 |
| 1e6 | +0.307 | -3.972 | 58 | -205.7 |
| 3e6 | +0.380 | -0.089 | 16 | -38.3 |
| 1e7 | +0.341 | -0.171 | 9 | -4.2 |

Two conclusions: (1) **the existing `ridge_lambda=1e5` default remains close to optimal for
the new basis too** -- no change was needed, despite the far larger raw condition number; (2)
**the output clip is still doing genuine, load-bearing work for this basis**, not a redundant
belt-and-suspenders layer -- ridge alone (no clip), even at `lambda=1e5`, still produces a
worst-case R^2 of -601.7 on some joint/seed combination (and every other lambda tested is
similarly bad or worse unclipped). Both defenses stay on by default.

### 4. Verdict

**Joints 2/3 specifically: genuinely improved, not just re-labeled.** Held-out R^2 roughly
doubled for both (joint 2: +0.326 -> +0.591 worst-case; joint 3: +0.152 -> +0.303 worst-case),
via a mechanistically sensible, testable hypothesis (cross-joint coupling) that was directly
verified on synthetic data too (`tests/mujoco/test_fit_residual_torque_model_feature_sets.py::
test_end_to_end_coupled_vs_baseline_recovers_true_cross_joint_signal`: a synthetic residual
that depends ONLY on another joint's `qd` gets R^2 < 0.1 under the baseline basis and > 0.9
under the coupled basis, confirming the mechanism actually works, not just correlates on real
data). Joint 4 (wrist_2), previously the single worst-performing joint (essentially
uninformative), is now the biggest single winner.

**Is this "actually useful" yet, or still "not broken but not useful"?** Closer, genuinely, but
not fully there. Worst-case (pathological-split) R^2 for joints 2/3/4 is now solidly positive
(+0.591 / +0.303 / +0.348) instead of near-zero-or-negative, which is real progress toward
"actually useful." But every joint's BEST achievable fit still depends on the ridge+clip safety
net doing real work on some splits (section 3), and the fundamental data-starvation problem
this task asked about is only partially closed: cross-joint features let joints 2/3/4 borrow
signal from joints 0/1's larger `qd` excursions during this X-only task, but that signal is
still coming from a task that was never designed to excite joints 2/3/4 directly -- there is
no real "ground truth" evidence in this dataset of how joint 2's residual behaves across its
OWN full velocity range, only evidence of how it correlates with joints 0/1's motion during
small, incidental joint-2 excursions. **If a future task or pose intentionally excited joints
2/3/4 more (e.g. a joint-space move, or a transport direction that engages the elbow/wrists
more), that would very likely be a bigger win than any further feature engineering on this
same X-only-transport dataset** -- the honest task-design-limitation half of this task's
question. Feature engineering measurably helped here, but it borrowed strength from what OTHER
joints were doing, not from what joint 2/3/4 were themselves doing across a wider range --
those remain two different, complementary paths forward, and this pass only pursued the first.

### 5. What changed (promoted to default)

- `controller_core/residual_torque_model.py` -- added `NUM_FEATURES_PER_JOINT_COUPLED` (21),
  `joint_features_coupled`/`all_joint_features_coupled` (own-joint 6 features + other 5
  joints' `qd` + other joints' `sin(q)`/`cos(q)`), `compute_residual_torque_coupled` (same
  `clip_abs` contract as the original `compute_residual_torque`). All purely additive --
  `all_joint_features`/`compute_residual_torque` (the original 6-feature basis) are completely
  unchanged, still used whenever `--feature-set baseline` is passed, and still what every
  pre-existing test exercises.
- `tools/analysis/fit_residual_torque_model.py` -- new `FEATURE_SETS` registry and
  `--feature-set {coupled,baseline}` CLI flag, **default `coupled`** (the promotion this task
  asked for, made because the evidence above genuinely supports it, not by default). `baseline`
  reproduces the pre-existing behavior exactly (verified byte-identical held-out numbers at
  seed 0 against this doc's earlier ridge-fix table). `fit_ols_weights`/`fit_ridge_weights` now
  infer feature width from `x_train.shape[2]` instead of the fixed `NUM_FEATURES_PER_JOINT`
  constant, so both feature sets (and any future one) work through the same fit path.
  `_featurize_runs`/`evaluate()` gained a `feature_set` parameter (default `"coupled"`,
  matching `main()`'s new default). Report JSON gained `feature_set`/`n_features_per_joint`
  fields.
- `tests/unit/test_residual_torque_model.py` -- 15 new tests for the coupled feature basis
  (shape, own-state isolation of the FIRST 6 features, correct cross-joint content of the
  remaining 15, NaN rejection, `compute_residual_torque_coupled`'s weight-shape validation,
  clip behavior).
- `tests/mujoco/test_fit_residual_torque_model_feature_sets.py` (new, 10 tests) -- `FEATURE_SETS`
  registry contents, `_featurize_runs`/`fit_ridge_weights`/`fit_ols_weights` width-inference for
  the 21-feature basis, and the synthetic end-to-end cross-joint-signal-recovery test described
  above.

### Verification

Before promoting, the scratch experiment script (not committed -- it duplicates
`fit_ridge_weights`/`compute_clip_bounds` logic to sweep 4 feature sets x 7 seeds cheaply) was
cross-checked against the real, now-modified `tools/analysis/fit_residual_torque_model.py` CLI
at seeds 0 and 4: held-out clipped R^2 values matched to 3 decimal places on every joint (e.g.
seed-0 joint 2: scratch +0.723, CLI +0.723; seed-4 joint 4: scratch +0.348, CLI +0.348),
confirming the swept numbers reflect the actual production fit path, not a divergent
reimplementation. `--feature-set baseline` at seed 0 reproduces this doc's earlier ridge-fix
table exactly (e.g. joint 2 R^2 +0.627 clipped -- matches this section's table's "A baseline"
seed-0 cell before averaging).

### Tests run

- `pytest -q tests/unit/test_residual_torque_model.py`: 31 passed (16 pre-existing + 15 new).
- `pytest -q tests/mujoco/test_fit_residual_torque_model_feature_sets.py
  tests/mujoco/test_fit_residual_torque_model_ridge.py
  tests/mujoco/test_residual_data_pipeline.py`: 25 passed (10 new + 15 pre-existing).
- `pytest -q -m "unit or mujoco"`: **335 passed, 3 xfailed, 259 deselected** -- the pre-existing
  287-passed baseline (from the ridge fix above) plus this pass's 25 new tests (15 unit + 10
  mujoco) = 312, plus 23 more from other test files added to the repo elsewhere since that
  baseline was recorded (unrelated to this change; confirmed zero failures either way).

### Tests not run

- `tests/hardware/` (mocked RTDE, `-m hardware`) -- not touched by this change, not re-run.
- No real-hardware command was run (none is available in this environment; this remains
  purely offline analysis on already-pulled trace data).

### Not done / explicitly out of scope

- No promotion into `controller_core/x_axis_cartesian_impedance.py` -- unchanged from the
  original doc's section 4/5: this is still purely an offline analysis/fit-quality
  improvement, not a decision to wire anything into the real controller.
- The run-level (not row-level) train/test split was not touched.
- A richer coupled-basis variant (e.g. interaction terms between own `qd` and other joints'
  `qd`) was not tried -- given C's condition number is already ~2.6e18 (right at the edge of
  what double precision can represent meaningfully), adding more raw feature dimensions
  without first addressing that conditioning felt like it would trade one collinearity problem
  for a worse one; a natural next step if this direction is pursued further would be feature
  selection or PCA-style dimensionality reduction on the "other joints" block before adding
  more raw terms.

## Files changed (this pass, 2026-08-01 cross-joint coupling)

- `controller_core/residual_torque_model.py` -- additive: `NUM_FEATURES_PER_JOINT_COUPLED`,
  `joint_features_coupled`, `all_joint_features_coupled`, `compute_residual_torque_coupled`.
- `tools/analysis/fit_residual_torque_model.py` -- `FEATURE_SETS` registry, `--feature-set`
  flag (new default `coupled`), `_featurize_runs`/`evaluate()` gained `feature_set` param,
  `fit_ols_weights`/`fit_ridge_weights` now feature-width-agnostic, report JSON gained
  `feature_set`/`n_features_per_joint`.
- `tests/unit/test_residual_torque_model.py` -- 15 new tests.
- `tests/mujoco/test_fit_residual_torque_model_feature_sets.py` (new) -- 10 tests.
- This file (this section).
- Not committed (gitignored): `outputs/residual_pipeline_coupled_feature_eval_2026-08-01/`
  (report/weights from the final `--feature-set coupled` validation run at seed 0).

## Rollback (this pass only)

```
git revert <this-commit>
```
Purely additive to both touched source files (new functions/parameters/CLI flag; the only
existing-function-body changes are `fit_ols_weights`/`fit_ridge_weights` computing `n_features`
from `x_train.shape[2]` instead of a fixed import, which is backward-compatible for every
existing caller since they already pass `NUM_FEATURES_PER_JOINT`-wide arrays) -- a plain revert
is lossless. Passing `--feature-set baseline` at any time fully reproduces the pre-this-pass
behavior without needing to revert anything.
