# Weighting orientation into the velocity controller's IK cost — `orientation_priority`

**Date:** 2026-08-06/07
**Lane:** velocity control (`controller_core.cartesian_velocity_controller`, `ik_seeded_resolution`)
**Status:** built, unit + MuJoCo tested, validated in sim on the standard 128-cell grid.
Not real-hardware validated. Default OFF. **Nothing committed.**

**Headline:** a new, flag-gated, default-off mechanism scores **111/128** on the standard grid
against the **104/128** fixed-gain baseline at the *same* gain vector — **+7 cells fixed, 0 broken**,
past the historical (unreproducible) 108/128 high-water mark.
**But it does NOT close the failure it was built for.** `hanging_alpha_0_5`'s −X orientation
failures are essentially untouched: +1 of its 5 failing cells, and that one is a *positive*-X fast
move. Six of the seven recovered cells are at a different pose entirely
(`unrotated_wrist2offset`, which goes 26/32 → **32/32**). Read §5 before quoting the +7.

---

## 1. The problem, restated

`compute_ik_seeded` solves, per Newton iteration, a box-constrained QP

```
min  reg*||dq||^2 + task_w*||J_task @ dq - task_err||^2
```

where `J_task` is the X/Y/Z position rows plus whichever rotation rows `task_dim_r{x,y,z}` enable.
`task_dim_rx`/`task_dim_ry` **default to `False`** and nothing in the repo turns them on for this
mode — so pitch/roll error is **not in the cost function at all**, while
`velocity_gain_tuning/envs/velocity_transport_env.py::step()` *does* trip an `orientation_guard` on
the full 3-axis swing-twist norm at 0.25 rad. Orientation is enforced but never optimised.

Prior evidence that this is structural, not a tuning gap:

* the pure minimum-norm, **zero-null-space-component** task step at `hanging_alpha_0_5` already
  induces real rx/ry rotation — the coupling is in the Jacobian's **row space**, so no null-space
  mechanism (`ik_max_joint_deviation_rad`, or the removed soft posture pull) can reach it;
* the per-cell `differential_evolution` oracle
  (`docs/status/gain_spline_interpolation_handoff_2026-08-06.md` §4.1), searching the **full 6-D
  gain space independently per cell**, found no guard-clean solution anywhere for
  `hanging_alpha_0_5 @ −0.370 m`.

---

## 2. Step 1 — the zero-code control experiment

Simply flipping `task_dim_rx`/`task_dim_ry` on, at `hanging_alpha_0_5`, with the baseline gain
vector (`search_result_nullspace_v2_20260806_194402.json`):

| dx (m) | move (s) | rx/ry OFF | rx/ry ON |
|---|---|---|---|
| +0.2035 | 0.02 | `orientation_guard 0.2523`, 65.6% tracked | **PASS**, ori **0.0000**, 100% tracked |
| +0.185 | 1.0 | PASS, ori 0.0855 | PASS, ori **0.0000** |
| +0.2405 | 1.0 | PASS, ori 0.0100 | PASS, ori **0.0000** |
| +0.370 | 1.0 | PASS, ori 0.2439 | PASS, ori **0.0000** |
| −0.185 | 1.0 | PASS, ori 0.0150 | PASS, ori **0.0000** |
| −0.296 | 1.0 | `orientation_guard 0.2518`, 94.0% | `orthogonal_drift_guard 0.0500`, 50.5% |
| −0.370 | 1.0 | `orthogonal_drift_guard 0.0504`, 67.0% | `orthogonal_drift_guard 0.0506`, 73.7% |

**Very strong signal, and not the one expected.** Weighting orientation in does not buy a graded
trade-off against X-tracking — where the pose is reachable it drives orientation error to *exactly*
zero at *100%* X-tracking with *lower* peak |qd| (0.575 vs 0.909 rad/s at dx=+0.370). It is a
strict improvement, not a trade. The trade only appears past the reach boundary, and there it is
catastrophic rather than graceful: the arm **retreats** in X (achieved goes −0.196 → −0.178 →
−0.149 over the hold) and drifts in Z until the drift guard fires.

That justified building something; it also made "one fixed weight, applied everywhere" clearly
wrong, since always-on rx/ry costs 28 cells elsewhere on the full grid (§5).

## 2b. Why the −X cases fail: a reachability check, not a controller property

Multi-start damped-least-squares IK (30 seeds, 600 iterations, proper quaternion rotvec error —
note that swing-twist per-axis components are *not* a valid large-error Newton descent direction,
which broke a first version of this check) against the full 6-DOF pose `(x0+dx, y0, z0, R0)`:

| dx (m) | full-6D residual | 4-row (x/y/z/rz) residual |
|---|---|---|
| −0.185 | 0.00000 m / 0.00000 rad | 0.00000 m |
| −0.240 | 0.00000 m / 0.00000 rad | 0.00000 m |
| −0.296 | 0.01032 m / 0.00957 rad | 0.00000 m |
| **−0.370** | **0.04698 m / 0.32014 rad** | **0.03357 m** |
| +0.185 / +0.296 / +0.370 | 0.00000 m / 0.00000 rad | 0.00000 m |

**`hanging_alpha_0_5 @ −0.370 m` is outside the reachable workspace.** Not "hard for this
controller" — *unreachable*, and unreachable by 3.4 cm even after dropping orientation from the task
entirely. This is a first-class finding in its own right: the oracle's infeasible cell is infeasible
because the arm cannot get there, matching the torque lane's precedent ("dx=0.25 m breaks via
Z-drift — a genuine workspace/reach limit, not a controller defect", `AGENTS.md` §3). No cost
function, gain schedule, or RL policy will ever pass that cell.

`−0.296 m` *is* reachable (to 1 cm), so it remains genuinely open — see §5.

---

## 3. What was built

`CartesianVelocityConfig.orientation_priority` (default `False`), a **hierarchical task-priority**
mechanism, not a weight ramp:

1. Solve the position-only IK exactly as today → `q_position_only`.
2. Solve it **again** with the disabled rotation axes **promoted to co-primary task rows**
   (weight `qp_task_weight * orientation_priority_weight`, default 1.0 = equal weight) →
   `q_promoted`.
3. `blend = smooth_falloff(|p_des − FK(q_promoted)|, residual_tol_m, residual_falloff_m, power)`
4. `q_target = q_position_only + blend * (q_promoted − q_position_only)`

i.e. **promote orientation exactly where promoting it is free, demote it where it costs position.**
The gate criterion is the promoted solve's own position residual — the only thing that actually
distinguishes the two regimes §2 found.

Properties, all deliberate and all covered by tests:

* **Path-independence preserved exactly.** Both solves are seeded from `q_rest`; the gate reads the
  promoted solve's residual, never `q_current`. `q_target` stays a deterministic function of
  `(q_rest, p_des, quat0)` — the entire reason `ik_seeded_resolution` exists.
* **Bit-for-bit prior behaviour when off**, when `orientation_priority_weight == 0`, when all three
  rotation axes are already selected, and past the falloff threshold (`smooth_falloff` returns
  *exactly* 0.0, so the promoted solve is discarded rather than blended in weakly).
* Cost: one extra `ik_iterations`-step solve per cycle when enabled (~0.23 ms measured), i.e. ~6%
  rather than ~3% of the 8 ms / 125 Hz budget.

### 3b. Two real bugs found while building it, both worth recording

**(a) Scheduling on the IK iterate's own orientation error does not work.** The first design was
exactly what the brief proposed: add the rotation rows as a softly-weighted term whose weight ramps
up as orientation error approaches the 0.25 rad guard. It made things *worse*, and instrumenting the
inner loop showed why — the ramp reads an **unconverged Newton transient**. At `dx=+0.185` the plain
solve's iterate error goes `0 → 0.186 → 0.029 → 0.018 → 0.016` (converging fine), but iteration 1's
transient 0.186 crosses the activation threshold, the term engages, and from there orientation
*diverges*: `0.186 → 0.163 → 0.244 → 0.358 → 0.500`. Final `q_target` orientation error 0.0157 rad
without the mechanism, **0.5720 rad with it.**

Related: at the searched gains the ramp's *magnitude* is irrelevant — `1e-4` and `1.0` give
identical results — because in the null space of the 4-row task the position term contributes
nothing at all, so the orientation term competes only against `reg = pinv_damping² ≈ 1.5e-10`
against `task_w ≈ 3.06e8`, a **relative regularisation of ~5e-19**. Any nonzero orientation weight
fully determines the redundant component. Corollary worth its own note: with the mechanism off, that
redundant component is not merely unweighted — it is set by little more than the linear solver's own
rounding.

A hierarchical null-space-only variant (Siciliano-style secondary task, `dq += N·z`) was also
prototyped and also fails: it diverges with more iterations (orientation error up to 3.67 rad at 40
iterations with the deviation clip removed), because the swing-twist per-axis error it descends on
is not a valid large-angle descent direction — the same defect that broke the reachability check in
§2b.

**(b) `ik_max_joint_deviation_rad`'s null-space clip must be computed against the task ACTUALLY
being solved.** Clipping the promoted solve against the *position-only* null space silently
discards the promoted solve's entire orientation correction, because that correction lives precisely
in the position task's null space. Before this fix the mechanism measured as doing nothing at
`hanging_alpha_0_5` and as *breaking* `+0.370` and `−0.2405`; after it, those pass. With
`extra_rot` empty the expression is bit-for-bit the previous one.

### 3c. The blend band had to be made very tight — measured, not assumed

Sweeping the gate band across the full 128-cell grid, same gain vector throughout:

| `residual_tol_m` | `residual_falloff_m` | power | pass | fixed | broken |
|---|---|---|---|---|---|
| — (mechanism off) | — | — | 104/128 | – | – |
| 1e-4 | 5e-4 | 2 | **111/128** | +7 | **0** |
| 5e-5 | 2e-4 | 2 | 111/128 | +7 | 0 |
| 1e-9 | 1e-9 (pure step) | 2 | 111/128 | +7 | 0 |
| 2e-4 | 2e-3 | 2 | 107/128 | +7 | −4 |
| 2e-3 | 1e-2 | 2 | 99/128 | +7 | −12 |
| 5e-4 | 5e-2 | 2 | 95/128 | +7 | −16 |
| 2e-4 | 2e-2 | 1 | 100/128 | +7 | −11 |
| 2e-3 | 1e-1 | 1 | 85/128 | +7 | −26 |

Clean, monotone, and against the brief's expectation: **the smooth blend is the part that hurts.**
Cause, traced rather than guessed — a partial blend emits a `q_target` neither solve endorses, and
worse, the blend weight sweeps through the band *during* a move as the commanded target advances,
migrating `q_target` between two different IK branches mid-move. The joint-space P law chases that
migration at `ik_joint_gain ≈ 47.9`: **all 12 cells the wide band broke were `joint_velocity_guard`
trips at 3.00–3.22 rad/s against a 3.0 limit**, all at `neg40`/`neg45`, all −X, all slow moves, all
of them cells the tight band leaves untouched. Defaults are therefore `(1e-4, 5e-4)` — very nearly a
hard accept/reject, with the small band kept only as robustness against floating-point noise in the
residual, not for blending.

---

## 4. Before/after — the full 128-cell grid

Same gain vector (`search_result_nullspace_v2_20260806_194402.json`, this lane's reproducible
fixed-gain best), same unmodified guards (|qd| ≤ 3.0 rad/s, orthogonal drift ≤ 0.05 m, orientation
error ≤ 0.25 rad), same `evaluate_gains`/`summarize_safety`, ±{0.3, 0.6, 0.9, 1.0, 1.1, 1.3, 1.6,
2.0} × each pose's `max_dx_hint_m` at both 1.0 s and 0.02 s moves.

| pose | baseline (off) | `orientation_priority` | `task_dim_rx/ry` always on |
|---|---|---|---|
| `hanging_alpha_0_5` | 27/32 | **28/32** | 28/32 |
| `unrotated_wrist2offset` | 26/32 | **32/32** | 32/32 |
| `neg40_wrist2offset` | 27/32 | **27/32** (0 fixed, 0 broken) | 8/32 (19 broken) |
| `neg45_wrist2offset` | 24/32 | **24/32** (0 fixed, 0 broken) | 8/32 (16 broken) |
| **total** | **104/128** | **111/128** | **76/128** |

Cells fixed (7): `unrotated_wrist2offset` at +0.0980/1.0, −0.0539/{1.0, 0.02}, −0.0784/1.0,
−0.0980/{1.0, 0.02}; and `hanging_alpha_0_5` at +0.2035/0.02.
Cells broken: **none.**

The third column is why the mechanism earns its complexity rather than being a one-line config
change: the residual gate is worth **+19 cells at `neg40` and +16 at `neg45`** over promoting
orientation unconditionally, and it recovers *every* cell that unconditional promotion recovers.

---

## 5. Honest verdict on the actual target

**It does not close `hanging_alpha_0_5`'s structural −X gap.** Of that pose's five failing cells:

| cell | baseline | with mechanism |
|---|---|---|
| +0.2035 / 0.02 s | `orientation_guard 0.2506` | **PASS**, 100% tracked, ori 0.0000 |
| −0.296 / 1.0 s | `orientation_guard 0.2518` | still fails |
| −0.296 / 0.02 s | `orientation_guard 0.2528` | still fails |
| −0.370 / 1.0 s | `orthogonal_drift_guard 0.0504` | still fails |
| −0.370 / 0.02 s | `orientation_guard 0.2504` | still fails |

Both `−0.370` cells are **provably unreachable** (§2b) and should be reclassified as such rather
than treated as an open controller problem. The two `−0.296` cells are the genuinely open ones: the
target pose *is* reachable to ~1 cm with full orientation held, but `compute_ik_seeded`'s
6-iteration Newton solve, seeded from `q_rest`, does not find it — the promoted solve's position
residual there reaches 1.32 m at the worst point along the commanded path, so the gate (correctly)
demotes and behaviour falls back to today's. **That is a solver-convergence limit, not a
cost-function limit** — a different, newly-identified problem from the one this work targeted, and
the most concrete remaining lead. Whether more IK iterations, a continuation seed, or a multi-start
solve closes it is untested here.

The brief's framing — that closing this would require accepting a real X-tracking trade-off —
turned out to be **wrong in an interesting way**. Where orientation can be held at all it is held
for free (exactly zero error, 100% tracking, *lower* peak joint velocity); where it cannot, no
trade-off exists to make, because the pose is out of reach. The row-space coupling is real, and the
premise that it forces a trade-off is what the measurements do not support.

---

## 6. Scope, limits, and what is deliberately not claimed

* Default is **OFF**. Nothing in the repo turns it on. No existing config was modified, no committed
  behaviour changed; every historical `outputs/velocity_gain_tuning/search_result_*.json` remains
  reproducible bit-for-bit (pinned by tests).
* **Not real-hardware validated.** This is a kinematic-only sim result
  (`LocalMujocoDynamics` FK/Jacobian, no `mj_step`, no dynamics), on a control mode
  (`ik_seeded_resolution`) that has itself never run on the real arm.
* Deliberately **not** added as an `ACTION_FIELDS` dimension: widening the action space would force
  every historical action vector to be padded and silently reinterpreted (the exact corruption
  `ik_max_joint_deviation_rad`'s bound-ordering comment documents). It is env/controller config, so
  "same gains, mechanism toggled" stays apples-to-apples.
* Promotion to a default, and any interaction with `velocity_gain_tuning/scheduling/`'s per-cell
  schedule (whose 56-cell oracle was measured *without* this mechanism and whose ceiling it may
  therefore move), are left as human decisions and are **not** acted on here.
* A re-run of the per-cell DE oracle with `orientation_priority` on is the obvious next measurement
  and was not run.

---

## 7. Files, tests, reproduction

Changed (all additive, all default-off):

| path | change |
|---|---|
| `controller_core/cartesian_velocity_controller/math_utils.py` | new `smooth_falloff` |
| `controller_core/cartesian_velocity_controller/config.py` | 5 new `orientation_priority*` fields + YAML parsing |
| `controller_core/cartesian_velocity_controller/modes.py` | Newton loop extracted to `_ik_newton_solve`; promoted-solve + gate block in `compute_ik_seeded`; deviation-clip basis fix (§3b(b)) |
| `velocity_gain_tuning/envs/velocity_transport_env.py` | 5 new env-config fields, threaded into the controller config |
| `tests/unit/test_cartesian_velocity_controller.py` | +15 tests |
| `tests/mujoco/test_orientation_priority.py` | new, 5 tests |
| `tools/evaluate_orientation_priority.py` | new, the 3-arm 128-cell before/after CLI |

Reproduce the headline table:

```bash
source /common/users/ss5772/miniforge3/etc/profile.d/conda.sh && conda activate mujoco_ur5e
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
python tools/evaluate_orientation_priority.py --scenario hanging_alpha_0_5 \
    --output-json outputs/velocity_gain_tuning/orientation_priority/op_hanging_alpha_0_5.json
```
(one call per scenario; ~35 s each, all four in parallel is fine).

---

## 8. Test status

* `pytest tests/unit -q` — **378 passed** (includes the 15 new `orientation_priority` /
  `smooth_falloff` tests).
* `pytest tests/mujoco/test_orientation_priority.py tests/mujoco/test_velocity_gain_tuning.py -q`
  — **38 passed** (5 new).
* `pytest tests/hardware -q` — 2-5 failures, **all pre-existing and flaky**, all in the unrelated
  `direct_torque` timing lane (`test_direct_torque_residual_observer_async.py`,
  `test_direct_torque_transport_pre_trip_trend.py`; `deadline_overrun` / golden-trace assertions).
  Confirmed by re-running them on a stashed (clean) tree, where a *different* subset fails run to
  run -- this host was under load ~58 during testing. Not investigated further; unrelated to this
  work.
* Not run: `tests/mujoco` in full (the torque-lane files are slow and untouched by this change).
