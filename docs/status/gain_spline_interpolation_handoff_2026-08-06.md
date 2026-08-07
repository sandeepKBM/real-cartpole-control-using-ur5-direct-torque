# Gain-schedule interpolation ("splines across the X frame") — build + results handoff

**Date:** 2026-08-06
**Lane:** velocity control (`controller_core.cartesian_velocity_controller`, `ik_seeded_resolution` mode)
**Status:** built, tested, validated in sim. Not real-hardware validated. Nothing committed.
**Headline:** best schedule scores **122/128** on the standard grid vs. a **104/128** fixed-gain
baseline, against a measured per-cell oracle ceiling of **54/56**. Read §4.3 before quoting that
number — the honest generalisation figure is 13/16 held-out, where the fixed-gain baseline is 16/16.

This document is written for an engineer who has read `AGENTS.md` and has no other context.

---

## 1. The problem this was built for

`velocity_gain_tuning/optimize.py` searches for ONE gain vector
(`kp_x, kp_rot, ik_joint_gain, pinv_damping, qp_task_weight, ik_max_joint_deviation_rad`)
that must serve all four pose scenarios in `velocity_gain_tuning/poses.py` and the full signed
displacement range at once. Fourteen real searches on 2026-08-06
(`outputs/velocity_gain_tuning/search_result_*.json`) plateaued at 90–108 / 128 cells regardless
of search budget, bound width, population seeding, cell reweighting, or which
redundancy-resolution mechanism was in the action space. The current reproducible best is
**104/128 (81.25%)**, `search_result_nullspace_v2_20260806_194402.json`. (The historical
108/128 used `ik_posture_gain`, a field since deleted from `CartesianVelocityConfig` — it is
not reproducible; see `docs/status/nullspace_v2_search_results_2026-08-06.md`.)

That plateau is the signature of a genuine multi-pose/multi-displacement Pareto conflict, not a
search-budget shortfall. The requested response was to make the gains a smooth function of
(pose, commanded displacement) instead of a constant.

---

## 2. The engineering decision: DE-per-cell + PCHIP, **not** RL

The brief asked whether the existing `rl_gain_scheduling/` package (PPO gain scheduling for the
separate **torque**-control lane) should be adapted. **It should not.** Reasons, all from this
repo's own recorded evidence:

* `docs/CURRENT_STATUS.md` records **six** real training attempts (`run1_200k`,
  `run2_continued_2.2M`, `reward_v2_2M`, `reward_v3_2M`, `reward_v4/v5_height0.5`, and the
  2026-07-29 3M-step config-mismatch-fix run). **None** beat the fixed-gain baseline on its own
  eval grid; results ranged from 0/20 to 5/8 against a 7/8 baseline.
* The 2026-07-25 read-only audit root-caused the failure as **exploration / deceptive local
  optimum**, not reward magnitude: every dense reward term except `x_error` is minimised by not
  moving, the terminal bonus only pays for a fully completed move, and the path between "sit
  still" and "clean move" runs through "move clumsily," which scores worse than either endpoint.
  Plus `ent_coef=0` → policy-variance collapse. Those are properties of the **sequential-RL
  framing**, not of the torque lane. Porting the framing ports the failure mode.
* `AGENTS.md` §3 explicitly recommends against another RL attempt for that lane.
* `velocity_gain_tuning/__init__.py` had already made the same call for the global search, and
  that call was vindicated: `differential_evolution` found usable gains on its first run, where
  six RL attempts found none.

Chosen instead — a two-stage pipeline built entirely from machinery already validated in this
repo:

1. **Stage 1 (search).** Run the *same* `differential_evolution` search, but independently per
   (pose, signed-dx) knot cell, with a two-episode single-cell objective. Each cell is a small,
   well-posed problem instead of a four-way compromise.
2. **Stage 2 (fit).** Fit a **shape-preserving PCHIP interpolant** through the per-cell optima,
   per pose, per action dimension, in **action space** (not physical gain space).
3. **Stage 3 (score).** Evaluate on the exact 128-cell grid every historical
   `search_result_*.json` was scored on, using the unmodified `summarize_safety`.

A third, decision-relevant thing falls out of Stage 1 for free: the per-cell results are an
**oracle upper bound** on *any* (pose, dx)-conditioned gain policy — spline, regression, or a
hypothetical RL policy that observes only (pose, dx). Nothing that sees only (pose, dx) can
beat, at cell *c*, the best gain vector for cell *c*. Measuring that bound cost one 13-minute
DE sweep. Measuring it via RL would have cost six training runs.

### Why PCHIP, why action space, why clamp

* **Action space, not gain space.** `action_to_gains` log-scales `pinv_damping`,
  `qp_task_weight` and `ik_max_joint_deviation_rad` (5–10 orders of magnitude each). Linear
  interpolation between `qp_task_weight = 1e2` and `1e9` in physical space puts the midpoint at
  `5e8` — essentially the upper knot. In action space the midpoint is the geometric mean. Action
  space also makes staying inside validated physical bounds a single `clip(-1, 1)`.
* **PCHIP, not a natural cubic spline.** A natural cubic through unevenly spaced, noisy knots
  overshoots — it can emit a gain more extreme than anything ever evaluated. PCHIP's interpolated
  value always lies between its two bracketing knots, so every point the schedule emits is inside
  the hull of two validated settings.
* **Clamp, never extrapolate.** Out-of-range queries return the nearest end knot. The knot grid
  already spans ±2.0× each pose's `max_dx_hint_m`, well past its known safe boundary.
* **Poses are discrete lookups, not interpolated.** The four scenarios differ in several joints
  at once; there is no single scalar to interpolate along and inventing one would be a fabricated
  axis. Real limitation — see §7.

---

## 3. What was built

All new files. **No existing file was modified.**

| Path | What it is |
|---|---|
| `velocity_gain_tuning/scheduling/__init__.py` | Package rationale — the full RL-vs-DE decision, written for future readers |
| `velocity_gain_tuning/scheduling/cells.py` | Knot-cell grid (`build_cells`) and continuation chains (`build_chains`); `DEFAULT_KNOT_FRACTIONS` |
| `velocity_gain_tuning/scheduling/schedule.py` | `ScheduleKnot`, `GainSchedule` (PCHIP fit, clamping, `drop_failed_knots`, `smooth_actions`), JSON save/load. **Simulator-free** — imports only numpy/scipy |
| `velocity_gain_tuning/scheduling/search.py` | `cell_fitness`, `search_cell`, `search_cells` (independent + chained modes), CLI |
| `velocity_gain_tuning/scheduling/evaluate.py` | `evaluate_schedule`, `evaluate_fixed_action`, `compare_summaries`, CLI comparing baseline vs. N schedule fit variants |
| `tools/run_gain_schedule_search_remote.sh` | ilab launcher with this repo's remote-compute hygiene (BLAS pinning, foreground execution) |
| `tests/unit/test_gain_schedule_interpolation.py` | 35 tests — pure-numpy interpolation/chaining layer |
| `tests/mujoco/test_gain_schedule_search.py` | 29 tests — objective, search, continuation, scheduled evaluation, knot-coverage split |

Generated artifacts (gitignored, under `outputs/velocity_gain_tuning/scheduling/`):
`knots_20260806.json` (independent), `knots_chained_w0_20260806.json`,
`knots_chained_w05_20260806.json` (**the best one**), `eval_20260806.json`,
`eval_chained_w0_20260806.json`, `eval_chained_w05_20260806.json`, plus `search_*.log`.

### Grid definition

* **Knots** (`DEFAULT_KNOT_FRACTIONS`): ±{0.3, 0.6, 0.9, 1.1, 1.3, 1.6, 2.0} × each scenario's
  `max_dx_hint_m` → 7 per direction × 2 directions × 4 poses = **56 cells**.
* **±1.0× is deliberately NOT a knot.** It *is* in the evaluation grid, so 16 of the 128
  evaluation cells are genuine **held-out** interpolation tests. This turned out to be the single
  most informative design choice in the whole exercise (see §4).
* **Cell objective**: mean episode reward over the nominal 1.0 s move *and* the
  `FAST_MOVE_DURATION_S = 0.02 s` near-step move at that same displacement — because this lane's
  real speed governor is `ik_joint_gain`, not `move_duration_s` (see `optimize.FAST_MOVE_DURATION_S`).
* **Every cell seeded** with the best global vector, so each cell's result answers "can
  cell-specific gains beat the global vector *here*?"
* **Evaluation grid**: `optimize.run_search`'s own `eval_dx_fractions`
  (±{0.3,0.6,0.9,1.0,1.1,1.3,1.6,2.0}) at both move durations × 4 poses = **128 cells**, scored
  with the unmodified `summarize_safety` and unmodified guards
  (|qd| ≤ 3.0 rad/s, orthogonal drift ≤ 0.05 m, orientation error ≤ 0.25 rad).

---

## 4. Results — and the honest headline

### 4.1 Per-cell oracle (Stage 1, independent search, 56 cells, 13.1 min on ilab3 @ 48 workers)

**53/56 knot cells have a guard-clean best gain vector.** The three that do not:

| pose | dx | guard |
|---|---|---|
| `hanging_alpha_0_5` | −0.370 m (−2.0×) | `orientation_guard 0.2510 > 0.25` (slow **and** fast) |
| `neg40_wrist2offset` | +0.058 m (+2.0×) | `orientation_guard 0.2517 > 0.25` (slow only) |
| `neg45_wrist2offset` | +0.058 m (+2.0×) | `joint_velocity_guard 3.4221 > 3.0` (slow only) |

All three are at ±2.0× the empirical boundary and all three miss by <1%. **This is a real,
first-class finding: (pose, dx)-conditioned gain scheduling has a genuine oracle ceiling of
roughly 94–95% on this grid — the residual failures are not reachable by any gain policy of this
class, RL included.** Conversely, the headroom between the 104/128 fixed-gain baseline and that
ceiling is large and real, so scheduling *is* the right lever here (unlike the torque lane's
gain-tuning-exhausted failures in `AGENTS.md` §3).

### 4.2 Fitted schedules vs. the fixed-gain baseline (128-cell grid, identical process/env)

| variant | pass | slow | fast | worst \|qd\| | worst ori |
|---|---|---|---|---|---|
| `baseline_global_fixed` (nullspace_v2, 104/128) | **104/128** | 50/64 | 54/64 | 7.693 | 0.2528 |
| `schedule_raw` (independent knots, PCHIP) | **117/128** | 57/64 | 60/64 | 10.638 | 0.2517 |
| `schedule_dropfail` | 115/128 | 57/64 | 58/64 | 10.638 | 0.2522 |
| `schedule_smooth3` (3-wide moving average) | 81/128 | 36/64 | 45/64 | 27.136 | 0.2530 |
| `schedule_dropfail_smooth3` | 81/128 | 36/64 | 45/64 | 27.136 | 0.2536 |
| `schedule_smooth5` | 82/128 | 35/64 | 47/64 | **122.368** | 0.2526 |

`+13 cells` over the baseline, and past the historical (unreproducible) 108/128 high-water mark.

### 4.3 **The headline caveat: the raw schedule memorises its knots and generalises worse than a constant**

Splitting the 128 cells by whether the displacement was a searched knot or the held-out ±1.0×:

| variant | at searched knots (112 cells) | at held-out ±1.0× (16 cells) |
|---|---|---|
| `baseline_global_fixed` | 88/112 (78.6%) | **16/16 (100%)** |
| `schedule_raw` | **108/112 (96.4%)** | **9/16 (56%)** |
| `schedule_dropfail` | 106/112 (94.6%) | 9/16 (56%) |

The fixed global vector passes *every* held-out cell. The schedule fails 7 of 16 — including a
`joint_velocity_guard: 10.64 > 3.0` at `neg45_wrist2offset, −1.0×`. **Taken at face value, the
+13 headline is almost entirely knot memorisation, and the interpolation itself is a
regression.** Anyone reporting "gain scheduling gets 117/128" without this split is
misreporting the result. This is what motivated §4.5's continuation fix, which recovers most
(not all) of the gap. The split is now printed by the evaluate CLI so it cannot be skipped.

### 4.4 Root cause of 4.3 — measured, not guessed

The per-cell objective has **many near-equivalent optima scattered across the action box**, and
independent searches at neighbouring displacements land on different ones. Measured on the
independent knot table: **mean Euclidean distance between adjacent knots' action vectors is
1.288**, against a maximum possible 4.899 across the `[-1,1]^6` box — i.e. adjacent knots are
typically ~26% of the whole search space apart. Per pose: `hanging_alpha_0_5` 1.525 (max 2.715),
`neg40` 1.226, `neg45` 1.191, `unrotated` 1.210.

A point interpolated between two such optima is not itself a good gain vector. The smoothing
ablation is the clean confirmation: perturbing *every* knot toward its neighbours' mean (a 3-wide
average) collapses the score from 117 to 81 and drives peak |qd| from 10.6 to 27.1 rad/s
(122.4 rad/s at width 5). Averaging two good gain vectors reliably produces a bad one.

**Conclusion: splining is only valid if the knot SEQUENCE is coherent, and coherence has to be
produced during the search — it cannot be recovered afterwards by smoothing.**

### 4.5 The fix that was implemented and tested: continuation-chained search

`cells.build_chains` groups cells into 8 (pose, direction) chains ordered outward from the
smallest |dx|. `search_cells(chained=True)` walks each chain sequentially, seeding each cell's DE
population with the previous cell's solution and — with `--continuity-weight w > 0` — adding a
`w·‖action − prev_action‖²` continuation penalty to that cell's objective. Chains are split by
direction (not run straight through dx=0) because this repo's +X/−X asymmetry is real and
recurring (`AGENTS.md` §7); the two chains still meet near dx=0, where they should agree.

Two chained sweeps were run as a clean ablation at the same per-cell budget as the independent
sweep (maxiter=25, popsize=10, seeded from `nullspace_v2`), `w = 0.0` (seeding only) and
`w = 0.5` (seeding + penalty; scale reference: the objective's range is O(10), a guard trip costs
−20, and `‖·‖²` tops out at 24 over the box, so `w = 0.5` makes crossing the whole box cost about
as much as a guard trip).

**It worked, and the mechanism check confirms why.** Mean adjacent-knot action distance:

| knot table | mean adjacent-knot distance | max | oracle |
|---|---|---|---|
| independent (`knots_20260806.json`) | 1.288 | 2.715 | 53/56 |
| chained, `w = 0.0` (`knots_chained_w0_...`) | 1.195 | 2.218 | 53/56 |
| chained, `w = 0.5` (`knots_chained_w05_...`) | **0.352** | 1.962 | **54/56** |

Scores on the same 128-cell grid, same baseline, same process:

| variant | pass | slow | fast | at knot (112) | **held-out (16)** | worst \|qd\| |
|---|---|---|---|---|---|---|
| `baseline_global_fixed` | 104/128 | 50/64 | 54/64 | 88/112 | **16/16** | 7.693 |
| independent → `schedule_raw` | 117/128 | 57/64 | 60/64 | 108/112 | 9/16 | 10.638 |
| chained `w=0.0` → `schedule_raw` | 116/128 | 57/64 | 59/64 | 107/112 | 9/16 | 5.582 |
| **chained `w=0.5` → `schedule_raw`** | **122/128** | 60/64 | 62/64 | **109/112** | **13/16** | **4.184** |
| chained `w=0.5` → `schedule_dropfail` | 121/128 | 60/64 | 61/64 | 108/112 | 13/16 | 5.478 |
| chained `w=0.5` → `schedule_smooth3` | 85/128 | 38/64 | 47/64 | 72/112 | 13/16 | 33.531 |

Three things to take from this:

* **Seeding alone is not continuation.** `w = 0.0` (previous solution injected into the
  population, no penalty) moved the adjacent-knot distance only 1.288 → 1.195 and left held-out
  generalisation at exactly 9/16. DE simply explores the rest of the box and finds an
  equally-scoring distant optimum. The *penalty* is doing the work, not the seed. Useful negative
  result — do not "simplify" the penalty away.
* **The penalty is close to free.** Adjacent knots are 3.7× closer, yet the oracle went *up*
  (53/56 → 54/56) and every headline number improved. Constraining the search to a continuous
  branch did not cost per-cell quality at this weight.
* **A real gap remains.** 13/16 held-out still loses to a constant's 16/16. All three residual
  held-out failures are `joint_velocity_guard` at −1.0× (`unrotated` 3.05, `neg45` 4.18 slow and
  3.28 fast) — i.e. the interpolated point between the −0.9× and −1.1× knots is still faster than
  either. The two remaining knot-cell failures are the oracle-infeasible ones from §4.1.

**Best result of this pass: `knots_chained_w05_20260806.json` fitted `raw`, 122/128 (95.3%)**,
versus a 104/128 fixed-gain baseline and a 54/56 oracle ceiling — but read the held-out column
before quoting the 122.

Note also that smoothing is *still* destructive even on a coherent chain (85/128, peak |qd|
33.5 rad/s). Post-hoc averaging of gain vectors is simply not a valid operation in this space,
whatever the knots look like. Do not re-litigate this.

---

## 5. How to reproduce / extend

All commands from the repo root, in the `mujoco_ur5e` conda env.

```bash
source /common/users/ss5772/miniforge3/etc/profile.d/conda.sh && conda activate mujoco_ur5e
```

### Stage 1 — search the knots (expensive; use ilab)

Check `uptime`/`nproc` on `ilab1-4.cs.rutgers.edu` first and pick the least loaded host.
`tools/run_gain_schedule_search_remote.sh` applies the BLAS pinning and foreground-execution
hygiene from `AGENTS.md` §8 (it deliberately omits `set -u` — this env's conda `activate.d` hooks
reference unset vars and abort under nounset).

**Recommended (reproduces the best result, 122/128):** the chained `w=0.5` command below.
The independent command is kept because it is what the oracle bound was measured with and it is
3–4× faster for a quick look.

```bash
# Independent per-cell search (56 cells, ~13 min at 48 workers) -- fast, but produces knots
# too incoherent to interpolate between (see section 4.4)
ssh ilab3.cs.rutgers.edu "bash /common/users/ss5772/real_Cartpole/tools/run_gain_schedule_search_remote.sh \
  48 /common/users/ss5772/real_Cartpole/outputs/velocity_gain_tuning/scheduling/knots_NEW.json \
  --maxiter 25 --popsize 10 --seed 0 \
  --seed-from-json outputs/velocity_gain_tuning/search_result_nullspace_v2_20260806_194402.json"

# Continuation-chained search -- THE ONE THAT WORKS (8 sequential chains of 7 cells, so only
# 8 workers are usable; 45 min at w=0.0, 58 min at w=0.5)
ssh ilab3.cs.rutgers.edu "bash /common/users/ss5772/real_Cartpole/tools/run_gain_schedule_search_remote.sh \
  8 /common/users/ss5772/real_Cartpole/outputs/velocity_gain_tuning/scheduling/knots_chained_NEW.json \
  --maxiter 25 --popsize 10 --seed 0 --chained --continuity-weight 0.5 \
  --seed-from-json outputs/velocity_gain_tuning/search_result_nullspace_v2_20260806_194402.json"
```

Run the ssh command in the foreground of one persistent local background job (this repo's
`nohup`-on-ilab lingering gotcha, `AGENTS.md` §8). Chained mode reports progress **per completed
chain**, not per cell — expect ~1.5 h of silence per chain at full budget; that is not a hang.

`--scenarios` and `--knot-fractions` subset the grid for cheap experiments.

### Stage 3 — score a knot table (cheap, ~5 min locally)

```bash
python -m velocity_gain_tuning.scheduling.evaluate \
  --schedule-json outputs/velocity_gain_tuning/scheduling/knots_NEW.json \
  --baseline-json outputs/velocity_gain_tuning/search_result_nullspace_v2_20260806_194402.json \
  --variants raw dropfail smooth3 \
  --output-json outputs/velocity_gain_tuning/scheduling/eval_NEW.json
```

Always pass `--baseline-json`: a schedule number without the baseline it is meant to beat,
measured in the same process against the same env instance, is not a result.

### The knot vs. held-out split is printed automatically

`print_comparison` now emits `at_knot` and `held_out` columns alongside the aggregate, computed
from the knot fractions in the schedule table itself (so it is correct for a custom
`--knot-fractions` grid too), and `knot_coverage` is written into the output JSON. **Read the
`held_out` column, not the aggregate** — an interpolant passes through its own knots by
construction, so its aggregate is always part memorisation. See
`evaluate.knot_coverage_split`'s docstring for the measurement that made this non-optional.

### Using a schedule from other code

```python
from velocity_gain_tuning.scheduling.schedule import GainSchedule
sched = GainSchedule.load("outputs/velocity_gain_tuning/scheduling/knots_NEW.json")
gains = sched.gains_for("neg45_wrist2offset", target_x_delta_m=-0.02)
```

`schedule.py` and `cells.py` import nothing beyond numpy/scipy (there is a test asserting this),
so a hardware lane can consume a schedule without loading a simulator.

---

## 6. Test coverage added

Per `AGENTS.md` §5 ("new modules/packages ship with pytest coverage, no exceptions").

* `tests/unit/test_gain_schedule_interpolation.py` — **35 tests**, ~4 s. Knot-grid invariants
  (bidirectional, symmetric, ±1.0 held out); `build_chains` (direction split, outward ordering,
  exact coverage, determinism); `smooth_actions`; PCHIP behaviour (exact at knots, exact on a
  linear ramp, **never overshoots the bracketing knots**, clamps instead of extrapolating, output
  always inside `[-1,1]`, continuous through dx=0, scenarios independent); degenerate cases
  (single knot, empty, mixed dims, duplicate dx); `drop_failed_knots` semantics including the
  all-knots-failed fallback; JSON round-trip and fit-option overrides; and a subprocess test
  asserting the module import pulls in **neither mujoco nor gymnasium**.
* `tests/mujoco/test_gain_schedule_search.py` — **29 tests**, ~7 min. `cell_fitness` is exactly
  the negative mean of the two episodes it claims to average, is deterministic, genuinely includes
  the fast-move dimension, has the right sign for DE, and threads `env_config` through; the
  continuity penalty is exactly `w‖Δ‖²`, opt-in, and zero at `prev_action`; `search_cell` returns
  a consistent knot, its `passed` flag matches a direct replay, its recorded fitness is the *pure*
  objective (comparable across continuity weights), seeding works, and a large penalty
  demonstrably pulls the solution toward `prev_action`; `search_cells` returns sorted knots in
  both modes, and parallel matches serial (`@slow`); `evaluate_schedule` with a constant schedule
  is **bit-for-bit** the existing fixed-gain evaluator, `evaluate_fixed_action` matches
  `evaluate.evaluate_gains`, a non-constant schedule really does produce different behaviour per
  cell, and the standard grid is 128 cells over 4 poses.

Also covered: `knot_coverage_split` / `knot_fractions_from_schedule` (the at-knot vs. held-out
split is separated correctly, per-bucket pass counts reconcile with the raw results, it is opt-in
in `compare_summaries`, and `print_comparison` labels both columns).

**Tests run:** the two files above (**64 passed**, 6 min 47 s); plus a regression check of
`tests/mujoco/test_velocity_gain_tuning.py`, `tests/unit/test_cartesian_velocity_controller.py`,
`tests/hardware/test_velocity_transport.py` — **74 passed**, unchanged.
Two tests are marked `@slow` (parallel-vs-serial search equivalence, both modes) and run by
default; `-m "not slow"` skips them.
**Tests NOT run:** the full `python -m pytest -q` suite (the unrelated torque-lane tests were not
re-run; no existing file was modified, so no regression path exists through them).

---

## 7. What's next, prioritised

1. **Close the residual held-out gap.** The chained `w=0.5` schedule is 122/128 overall but
   13/16 held-out against a constant's 16/16, and all three misses are `joint_velocity_guard` at
   −1.0× — the interpolated point between the −0.9× and −1.1× knots is faster than either
   endpoint. Two cheap, targeted things to try, in order: (a) raise the continuity weight (see
   item 3) so those two knots sit closer still; (b) densify knots near ±1.0× specifically, which
   is a two-cell-per-pose search, not a full sweep. Do **not** reach for smoothing (§4.5).
2. **Add more held-out fractions.** One held-out fraction (±1.0×) over 16 cells is a thin
   generalisation test. Add ±0.75× / ±1.2× as evaluation-only fractions so generalisation is
   measured over ~48 cells rather than 16.
3. **Sweep the continuity weight.** Only `w ∈ {0.0, 0.5}` were tried, at one seed each. The jump
   from 0.0 to 0.5 took held-out from 9/16 to 13/16 *and* raised the oracle from 53/56 to 54/56 —
   there is no evidence 0.5 is near the optimum, and the measured trend says try higher (1.0, 2.0)
   next. Cheapest remaining lever: one ~58-minute ilab sweep per value.
4. **Try a genuinely smooth policy class instead of an interpolant.** Fit a low-order polynomial
   or small ridge regression of action on (dx, |dx|, sign) per pose, trained on *all* knots at
   once. Unlike PCHIP it cannot memorise knots, so its knot-cell and held-out-cell scores are
   directly comparable, which makes it a far more honest test of whether a smooth (pose, dx) → gain
   map exists at all. This is the natural next step and is squarely in the
   "supervised regression, not RL" direction `AGENTS.md` §3 already recommends for the torque lane.
5. **Pose interpolation is not implemented.** Poses are discrete lookups. A schedule cannot be
   queried at a pose it was not fitted at. If continuous pose coverage is needed, the honest route
   is a shared parametric fit over a real pose descriptor (e.g. base rotation and `wrist_2` offset)
   rather than an interpolation over an invented pose index.
6. **Nothing here has touched real hardware.** All numbers are from the kinematic-only
   `VelocityTransportEnv`. Before any real-arm use, note that a schedule changes gains *between*
   commanded moves; the real `speedL` path has never been driven with time-varying gains, and
   `hardware/x_transport.py` has no schedule plumbing at all.
7. **Known rough edge:** `search_cells(chained=True)` reports progress only per completed chain.
   Streaming per-cell progress out of a sequential chain worker would need a `multiprocessing`
   queue; it was not worth it for this pass.

---

## 8. Rollback

Nothing was committed and no existing file was modified. To remove everything:

```bash
rm -rf velocity_gain_tuning/scheduling \
       tests/unit/test_gain_schedule_interpolation.py \
       tests/mujoco/test_gain_schedule_search.py \
       tools/run_gain_schedule_search_remote.sh \
       docs/status/gain_spline_interpolation_handoff_2026-08-06.md \
       outputs/velocity_gain_tuning/scheduling
```

(`git status` will confirm the tree is clean afterwards; `outputs/` is gitignored, so that
directory is not git-recoverable and is listed here explicitly.)
