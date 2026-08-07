# Linear MPC-as-correction around a nominal trajectory — feasibility test

**Date:** 2026-08-07
**Lane:** velocity control (`controller_core.cartesian_velocity_controller`, `ik_seeded_resolution`)
**Status:** feasibility test only. **No production code changed, nothing committed.**
Prototype: `tools/diagnostics/linear_mpc_nominal_trajectory_prototype.py` (new, standalone,
imported by nothing). `modes.py` untouched.

## Verdict: **NO-GO**

A linear-MPC correction around an already-reasonable nominal trajectory — the formulation from
Lee et al. (arXiv 2209.11880), built specifically to avoid the two failure modes that killed the
earlier SQP-horizon prototype (`docs/status/mpc_feasibility_2026-08-07.md`) — is genuinely
different in kind (no re-linearization, provably no worse than baseline when its new term is
off) but still does not clear the bar: it does not fix either documented target failure, and its
compute cost is at least as bad as the already-rejected approach's.

| # | Finding | Evidence |
|---|---|---|
| 1 | Structurally sound, unlike the SQP-horizon | Exact (0.0 rad) reduction to production at N=1/rate=0/cond=0 on all three cases, including one case (`hanging_-0.296_fast`) where a first implementation attempt was NOT exact (2.2e-2 rad) — bug found and fixed (see Sec 2) |
| 2 | Zero regression when the new term is off | `rate=off, cond=off` at N=2/3/5 is byte-identical to baseline on ALL THREE cases — the rejected SQP-horizon was NOT this well-behaved (it degraded `hanging` even before its barrier was touched) |
| 3 | Does not fix `hanging_alpha_0_5 @ -0.296m` | `rate=on` reproduces the SAME qualitative failure as the rejected prototype: X-achievement degrades (-0.2783m -> -0.1823m at N=2) while orientation stays pinned at/near the 0.25 rad guard — independent confirmation this is a structural row-space coupling, not fixable by any redundancy-resolution-style correction |
| 4 | Does not fix `neg45_wrist2offset` wrist-singularity trip | A 5x7 (N x rate_w) sweep plus a conditioning-barrier variant: **0/35 pass**, non-monotone, sometimes WORSE (qd trip up to 10.5 rad/s vs baseline 3.365) |
| 5 | Compute cost is at least as bad as the rejected approach, for a different reason | N=2 costs ~2.2x baseline (median 3.49-3.58ms vs 1.58ms on idle ilab4) because nominal generation calls the full 6-iteration production solve N times — derated (x3.4, same factor as the prior doc) this is ~12ms, exceeding the ENTIRE 8ms/125Hz period, worse than the rejected SQP-horizon's cheapest working N=2/sqp=4 setting (2.66ms local / 9.0ms derated) |

---

## 1. The formulation (and why it differs in kind from the rejected SQP-horizon)

`docs/status/mpc_feasibility_2026-08-07.md` (read in full before writing this) found a
short-horizon SQP extension of `compute_ik_seeded` diverges (position residual 4.5e-4m ->
1.44m in one case) because it re-linearizes the WHOLE horizon from a cold `q_rest` start every
outer iteration, amplifying the single-step solve's own near-singular Hessian
(`cond(H) ~= 1.3e17` at the searched gains) with each re-linearization. Its own Sec 4c also found
the wrist-singularity guard trip is not a Jacobian-conditioning event at all — it is a discrete,
cycle-to-cycle JUMP in `q_target` (the IK solve's null-space branch can flip near a singularity),
and states plainly: "A horizon that re-solves from q_rest every cycle contains no term
penalising inter-cycle q_target motion."

This prototype is built to be structurally different on both counts:

1. **Nominal, not a cold horizon.** `build_nominal_trajectory()` calls the UNMODIFIED, imported
   `_ik_newton_solve` from `modes.py` (not a reimplementation) once per horizon stage, at N
   future profile setpoints along the same min-jerk trajectory the real controller already
   commands — literally "the controller's own output, used as the plan," per the task brief.
2. **Linearize ONCE.** `linear_mpc_correct()` takes the Jacobian at each nominal stage as FIXED
   and solves exactly one box-constrained QP over the stacked per-stage corrections
   `delta_0..delta_{N-1}`, via the same `build_weighted_least_squares_qp`/`solve_box_qp` pair
   production uses. There is no outer iteration loop anywhere in this file — nothing here can
   diverge the way the SQP horizon did, by construction.
3. **The new ingredient.** An explicit inter-stage `q_target` CONTINUITY cost (`rate_w`), aimed
   directly at Sec 4c's diagnosed mechanism, plus an optional linearized (not iteratively
   re-evaluated) `sigma_min` hinge barrier (`cond_weight`), carried over from the rejected
   prototype's own "genuinely new ingredient" per the task brief, to test on its own merits here
   too.
4. **Receding horizon.** Only the corrected stage-0 target is applied each cycle; it becomes
   `q_target_prev` for the next cycle's continuity term — a real, acknowledged departure from
   `ik_seeded_resolution`'s path-independence, flagged the same way the earlier feasibility
   doc's own "q_target rate limiting" lead was flagged (Sec 8 there).

## 2. Selfcheck — exact reduction, and a real bug found along the way

`--mode selfcheck` asserts N=1/rate=0/cond=0 reproduces `_ik_newton_solve` exactly. First attempt
was NOT exact: `hanging_-0.296_fast` differed by 2.2e-2 rad. Root cause: the null-space deviation
clip (`ik_max_joint_deviation_rad`) was re-derived as an absolute band against `q_rest`, using a
Jacobian evaluated AFTER production's own last internal clip — a different basis than the one
that internal clip used mid-iteration, so re-projecting a zero correction through it was not
idempotent for a case deep in the deviation band. Fixed by clipping the CORRECTION's own
null-space magnitude instead (trivially idempotent at `delta=0` regardless of basis). After the
fix, all three cases reduce exactly (0.0 rad) — a genuine correctness property the rejected
SQP-horizon could not offer (its own N=1 reduction needed a specific reg/box-value match and
still wasn't claimed exact past floating point in the same unconditional sense).

```
wrist_sing_neg45         max|q_mpc(N=1,rate=0) - q_baseline| = 0.000e+00
hanging_-0.296_slow      max|q_mpc(N=1,rate=0) - q_baseline| = 0.000e+00
hanging_-0.296_fast      max|q_mpc(N=1,rate=0) - q_baseline| = 0.000e+00
```

## 3. Profiling — idle ilab4, warmup=50, reps=300, median + p90

Host: ilab4.cs.rutgers.edu, idle (`load average: 1.43, 1.24, 1.41` on 96 cores; westeros was not
checked this session but the prior feasibility doc found it consistently loaded, ~55/32 cores).
`OPENBLAS/OMP/MKL/NUMEXPR_NUM_THREADS=1`. Baseline (1.587ms mean / 1.579ms median) matches the
prior feasibility doc's independently-measured 1.571ms almost exactly — good cross-check that
both measurements are sound.

| solver | median (ms) | p90 (ms) | x baseline | derated (x3.4) |
|---|---|---|---|---|
| production `ik_seeded` (baseline) | 1.579 | 1.616 | 1.00x | 5.4ms (already 1.5x over the 3.5ms budget, per the prior doc) |
| N=2, rate=off, cond=off | 3.489 | 3.507 | 2.21x | 11.9ms |
| N=2, rate=on, cond=off | 3.528 | 3.549 | 2.23x | 12.0ms |
| N=2, rate=on, cond=on | 3.567 | 3.606 | 2.26x | 12.1ms |
| N=3, rate=on, cond=off | 5.202 | 5.226 | 3.29x | 17.7ms |
| N=5, rate=on, cond=off | 8.599 | 8.670 | 5.45x | 29.2ms |

**Why this is at least as expensive as the rejected approach, for a different reason.** The QP
correction step itself is cheap (the on/off deltas above are all <3%) — nearly all of the cost is
`build_nominal_trajectory()` calling the FULL 6-iteration production solve N independent times.
N=2 here (3.49ms) already costs more than the rejected SQP-horizon's own N=2/sqp=4 setting
(2.66ms, per the prior doc's Sec 6) — that prototype used only 4 partial outer iterations total,
cheaper per-iteration than 2 full 6-iteration Newton solves. **Avoiding SQP re-linearization
does not avoid the compute problem**: calling ANY full per-cycle solve N times for the nominal
is itself over budget at N=2, before the correction step adds anything. This is confirmed,
board-and-nail evidence for the prior doc's Sec 7(c) point that 3-7x more compute headroom is a
precondition, independent of formulation.

## 4. Validation — `hanging_alpha_0_5 @ -0.296m` (slow 1.0s and fast 0.02s)

Same reproducible gains (`search_result_nullspace_v2_20260806_194402.json`, 104/128) as the
prior feasibility doc, so these numbers are directly comparable to its Sec 4a.

```
hanging_-0.296_slow  baseline_env                 orientation_guard: 0.2518 > 0.25   ach=-0.2783 ori=0.2518 qd=1.306
hanging_-0.296_slow  mpc N=2 rate=off cond=off    orientation_guard: 0.2518 > 0.25   ach=-0.2783 ori=0.2518 qd=1.306   (byte-identical to baseline)
hanging_-0.296_slow  mpc N=2 rate=on  cond=off    orientation_guard: 0.2529 > 0.25   ach=-0.1823 ori=0.2529 qd=0.893
hanging_-0.296_slow  mpc N=3 rate=on  cond=off    orientation_guard: 0.2517 > 0.25   ach=-0.1924 ori=0.2517 qd=0.895
hanging_-0.296_slow  mpc N=5 rate=on  cond=off    orientation_guard: 0.2528 > 0.25   ach=-0.2154 ori=0.2528 qd=1.022
hanging_-0.296_fast  baseline_env                 orientation_guard: 0.2528 > 0.25   ach=-0.1266 ori=0.2528 qd=0.819
hanging_-0.296_fast  mpc N=5 rate=on  cond=off    orientation_guard: 0.2504 > 0.25   ach=-0.1229 ori=0.2504 qd=0.730
```

`rate=off, cond=off` is byte-identical to baseline at every N tested — a real structural
improvement over the rejected prototype (which degraded X-achievement even with its barrier
weight at zero, purely from repeated re-linearization noise). But turning the new ingredient on
reproduces the SAME qualitative failure the rejected prototype found: X-achievement degrades
(-0.2783m -> -0.1823m, 34% less of the target reached) while orientation stays pinned at/just
above the guard (0.2529 vs 0.2518 — worse, not better). The one near-miss
(`hanging_-0.296_fast`, N=5: 0.2504 rad, 0.0004 above the 0.25 threshold) is noise-level, not a
genuine fix. This is **independent confirmation, via a completely different mechanism**, of the
structural finding already documented in `modes.py`'s own docstring: "hanging's
orientation-vs-X-tracking coupling lives in the TASK (row) space itself, not the null space — a
fundamentally different, structural phenomenon that no null-space-projected mechanism ... can
fix without a REAL X-tracking accuracy trade-off." A linear-MPC correction is, in the end,
another null-space/redundancy-resolution mechanism at its core, so this result is exactly what
that finding predicts.

## 5. Validation — `neg45_wrist2offset`, dx=-0.029m, `ik_max_joint_deviation_rad=0.01` (the 161.57 rad/s case)

Baseline: `joint_velocity_guard 3.3645 > 3.0`, achieving -0.0179 of -0.029m (62%) — matches the
prior feasibility doc and `docs/status/nullspace_v2_search_results_2026-08-06.md`'s
root-caused wrist_2=0 crossing exactly.

**Rate-weight sweep, N in {2,3,5} x rate_w in {0, 1e2..1e7} — 0/21 pass:**

```
N=2 rate_w=0e+00  joint_velocity_guard: 3.3645 > 3.0   ach=-0.0179
N=2 rate_w=1e+04  joint_velocity_guard: 3.3431 > 3.0   ach=-0.0179   (best case: 0.6% lower trip, still fails)
N=2 rate_w=1e+05  joint_velocity_guard: 6.9399 > 3.0   ach=-0.0191   (2x WORSE, |w2|min collapses to 0.00058)
N=2 rate_w=1e+06  joint_velocity_guard: 3.4574 > 3.0   ach=-0.0180
N=2 rate_w=1e+07  joint_velocity_guard: 6.1632 > 3.0   ach=-0.0175   (WORSE again)
```
(N=3, N=5 show the same non-monotone pattern, worst case 10.5 rad/s at N=5/rate_w=1e7.)

**Conditioning barrier (`cond_weight`, linearized once, alone and combined with rate) — 0/14
pass, and actively counterproductive**:

```
mpc N=2 rate=off cond=on   joint_velocity_guard: 5.0154 > 3.0   ach=-0.0187   |w2|min=0.00433 (vs baseline 0.01288 -- CLOSER to the singularity)
mpc N=2 rate=on  cond=on   joint_velocity_guard: 4.7051 > 3.0   ach=-0.0181   |w2|min=0.00854
```

Exactly the rejected prototype's own Sec 4b finding, reproduced independently here: the barrier
does what it is kinematically designed to do (it is a linearized version of the same term), but
the guard trips anyway, and the corrected trajectory goes DEEPER toward the singularity than
baseline, not away from it.

**dx band scan at N=3/rate_w=1e6** (mirroring the rejected prototype's own decisive control,
Sec 4b) — no -X band recovers at all, unlike the rejected prototype (which recovered a real,
if untrustworthy-by-saturation, 4-cell band at its one working setting):

```
dx=-0.0203  baseline:joint_vel ach=-0.0179   mpc:joint_vel ach=-0.0181
dx=-0.0261  baseline:joint_vel ach=-0.0179   mpc:joint_vel ach=-0.0181
dx=-0.0290  baseline:joint_vel ach=-0.0179   mpc:joint_vel ach=-0.0181
dx=-0.0319  baseline:joint_vel ach=-0.0179   mpc:joint_vel ach=-0.0181
dx=-0.0377  baseline:joint_vel ach=-0.0179   mpc:joint_vel ach=-0.0181
dx=-0.0464  baseline:joint_vel ach=-0.0179   mpc:joint_vel ach=-0.0181
```

**Why the targeted fix didn't work.** The continuity term was built specifically to counter the
diagnosed cycle-to-cycle `q_target` jump, but `ik_max_joint_deviation_rad=0.01` — the SAME tight
bound that (per `docs/status/nullspace_v2_search_results_2026-08-06.md`) causes this failure by
removing the null space's escape route from the wrist_2=0 crossing in the first place — is also
enforced on top of the correction itself (Sec 2's fix keeps this bound in force). There is very
little room left in a 0.01 rad null space for a continuity correction to counteract a real branch
flip; the correction mostly either does nothing (rate_w small) or fights the task term into a
different, sometimes worse, failure (rate_w large). This mechanism did not find a genuinely
different, better path here — it hit a wall for a reason distinct from the rejected prototype
(insufficient authority under the very constraint that causes the failure, not saturation or
non-convergence), but the practical outcome — no pass, ever — is the same.

## 6. Does this differ in kind from the rejected SQP-horizon, or hit the same wall?

**Genuinely different in kind, on robustness**: no re-linearization loop anywhere (cannot
diverge the way the SQP prototype did), exact reduction to baseline at N=1 (proven, not just
close), and zero regression on all three cases when the new term is off (the SQP prototype
regressed `hanging` even with its barrier weight at zero, from pure re-linearization noise). This
is a real, structural improvement in soundness.

**Hits the same wall on efficacy, for the SAME reason on `hanging`** (row-space coupling, not a
linearization-quality or convergence problem — confirmed independently here) **and a DIFFERENT
reason on the wrist case** (insufficient null-space authority under the very deviation bound that
causes the failure, not saturation/divergence — but still zero passes across a 35-point sweep).

**On compute, this formulation is at least as bad, for a different reason**: not SQP iteration
count, but N independent full production solves to build the nominal. Any future formulation in
this family needs both a materially cheaper way to build the nominal (fewer Newton iterations
per stage was not tried here, flagged as a possible lead, not validated) and a demonstrated fix
for at least one target case before compute is worth optimizing further.

## 7. Scope and limits, deliberately not claimed

* Kinematic-only sim (`LocalMujocoDynamics` FK/Jacobian, no `mj_step`), same as the prior
  feasibility doc — nothing here is hardware-validated.
* No full 128-cell grid run — per the task's own instruction, a clear negative at Sec 4/5 did
  not warrant one, and no (N, rate_w, cond_weight) setting earned it.
* Only the two documented target cases and their neighborhoods (rate/cond sweeps, one dx band
  scan) were tested — a broader gain re-search (e.g. jointly re-tuning `task_w`/`reg`/`rate_w`
  together rather than reusing the fixed 104/128 gains) was not attempted and might behave
  differently, though Sec 5's non-monotone, sometimes-worse pattern is not encouraging.
* Cheaper nominal-generation (fewer Newton iterations per stage, e.g. `ik_iterations=1` or `3`
  instead of 6) was not tried; Sec 3's `_ik_newton_solve 6it` cost dominates the N-scaling and is
  the obvious next lever if this family were revisited, but nothing here validates it would help
  efficacy too.

## 8. Files, reproduction, rollback

Created (2, both new, nothing else touched — plus the worktree's git branch was fast-forwarded
to sync with `feature/ur5e-mujoco-torque-control`, see note below):

| path | what |
|---|---|
| `tools/diagnostics/linear_mpc_nominal_trajectory_prototype.py` | standalone prototype + all measurement modes |
| `docs/status/linear_mpc_nominal_trajectory_2026-08-07.md` | this document |

```bash
PYTHONPATH=. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
  /common/users/ss5772/miniforge3/envs/mujoco_ur5e/bin/python3 \
  tools/diagnostics/linear_mpc_nominal_trajectory_prototype.py --mode selfcheck

# profiling/validation should run on an idle host (ilab4 was used here, load ~1.4; westeros
# was not checked, the prior feasibility doc found it consistently loaded)
python tools/diagnostics/linear_mpc_nominal_trajectory_prototype.py --mode profile --reps 300 --warmup 50
python tools/diagnostics/linear_mpc_nominal_trajectory_prototype.py --mode validate --rate-weight 1e6
python tools/diagnostics/linear_mpc_nominal_trajectory_prototype.py --mode rate_sweep --best-n 3 --best-rate-weight 1e6
python tools/diagnostics/linear_mpc_nominal_trajectory_prototype.py --mode validate --rate-weight 1e6 --cond-weight 1e6 --sweep-cond
```

**Rollback** — nothing else was touched, no config or production module changed, nothing
committed:

```bash
rm tools/diagnostics/linear_mpc_nominal_trajectory_prototype.py docs/status/linear_mpc_nominal_trajectory_2026-08-07.md
```

**Worktree note**: this session's worktree was created from a stale base (an old `main` commit,
`57c1290`, predating `feature/ur5e-mujoco-torque-control`'s recent history and missing both this
session's prior feasibility doc and the `velocity_gain_tuning`/`ik_seeded_resolution` code this
work depends on). Brought current via `git merge --ff-only feature/ur5e-mujoco-torque-control`
(a pure fast-forward — the worktree branch had zero unique commits of its own, confirmed via
`git merge-base` before merging) to `12382f1`, matching the task's stated starting commit. No
destructive git operations were used. To undo just that sync (not recommended, would remove the
`velocity_gain_tuning` package and everything else this session's context describes):
`git reset --hard 57c1290` (only safe if nothing else has been committed on this branch since).

**Tests run:** none — no production code was modified, so the suite is unchanged. Correctness is
established by `--mode selfcheck` (exact N=1 reduction to production, all three cases) and by
`--mode validate`'s `baseline_env` vs `baseline_replica` rows matching to every reported digit.
**Tests not run:** `pytest` (unit / mujoco / hardware) — untouched by this work, and per the
task's own instruction (`AGENTS.md` §5's coverage requirement applies to kept work; nothing here
is being kept/promoted, so no new tests were added).
