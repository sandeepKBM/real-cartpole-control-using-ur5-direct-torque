# Short-horizon (MPC) extension of `ik_seeded_resolution` — feasibility scoping

**Date:** 2026-08-07
**Lane:** velocity control (`controller_core.cartesian_velocity_controller`, `ik_seeded_resolution`)
**Status:** feasibility scoping only. **No production code changed, nothing committed.**
Prototype lives at `tools/diagnostics/mpc_feasibility_prototype.py`; `modes.py` untouched.

## Verdict: **NO-GO**

A short-horizon receding-horizon extension of `compute_ik_seeded` should **not** be built.
Five independent measurements each block it, and the two strongest ones are about
*efficacy* and *well-posedness*, not compute — so "buy a faster control PC" does not rescue it.

| # | Blocker | Evidence |
|---|---|---|
| 1 | Does not fix `hanging_alpha_0_5 @ −0.296 m` | 0 of ~40 (N, barrier, sqp) settings pass; every one fails the orientation guard at 0.2505–0.2539 rad, and the horizon monotonically *reduces* X achievement (−0.278 m → −0.212 m) |
| 2 | Fixes the wrist case only at one non-reproducible-by-design setting | Exactly 2 of 128 (N, w_cond, σ_floor) combinations pass; the passing one is non-monotone in N (N=1/3/4 all fail where N=2 passes) and works by *saturation*, with q_target jumps up to **12.7 rad** |
| 3 | The horizon SQP does not converge | Position residual non-monotone at every trust radius tried; with the barrier on and no trust region it diverges outright (4.5e-4 m at sqp=4 → **1.44 m** at sqp=6) |
| 4 | Root cause is upstream and would break any trajectory optimiser | The underlying QP Hessian at the searched gains has **cond ≈ 1.3e17** (reg/task_w ≈ 5e-19); two algebraically identical formulations of the *same* single-step solve diverge by 6.3e-2 rad after 6 Newton iterations |
| 5 | Wrong mechanism, and compute doesn't fit anyway | The guard trip is a **q_target jump**, not a Jacobian blowup — and a from-`q_rest` horizon has no term penalising inter-cycle q_target motion. Separately, no N ≥ 2 fits the budget (§2) |

---

## 1. The real compute budget

**8.0 ms period** (`hardware/velocity_transport.py`, `rate_hz=125.0`).

`DeadlineMonitor` (`hardware/safety.py`) aborts on `work − period > max_deadline_ms` (3.0 ms)
for 3 consecutive cycles, i.e. sustained work up to ~11 ms. That is the **abort ceiling, not a
design budget** — a loop doing 11 ms of work is running at 91 Hz while `speed_l` is being
commanded with `time_s = dt_s * 1.5 = 12 ms` semantics.

The right margin standard is this repo's own: `UR5eSafetyLimits.max_deadline_fraction_of_period
= 0.5`, added 2026-07-29 on the explicit finding that a flat 3.0 ms is "too loose for that
loop's own budget." Applying the same 50 %-of-period standard to the 125 Hz loop:

* total cycle work ≤ **4.0 ms**
* non-controller per-cycle work ≈ **0.5 ms**. From the only real-hardware phase measurement in
  the repo (`docs/status/clock_timing_late_cycles_2026-07-28.md`, thinkrobot): `read_state`
  0.064 + `build_state` 0.050 + `safety` 0.076 + command-send 0.070 = **0.26 ms**; the velocity
  loop additionally runs `x_profile_target`, `StaleStateMonitor.record`,
  `CartesianMoveMonitor.check`, `is_robot_safety_normal`, and a trace-row append with 4
  `.tolist()` calls, so 0.5 ms is a conservative round-up.

> **Controller compute budget ≈ 3.5 ms, held at p99** (a deadline miss is a per-cycle event, so
> the mean is the wrong statistic).

### Profiling host and derating

Profiled on **ilab4** (AMD EPYC 7352, idle, load 1.4–1.7, `OPENBLAS/OMP/MKL/NUMEXPR_NUM_THREADS=1`).
westeros was at load ~55/32 cores and is not usable for timing. The real deploy target is
thinkrobot (the lab control PC), which is not reachable from here.

Cross-machine calibration, same function, same config: `XAxisCartesianImpedanceController.compute()`
under `config/ur5e_mujoco_torque_osc_tuned.yaml` measures **0.196 ms** on ilab4 (n=8000; this also
cross-checks against the 2026-07-29 perf audit's 0.2075 ms on westeros) versus a real thinkrobot
`controller_mean_ms` of **0.669 ms**. → **derating ≈ 3.4×**.

*Caveat, stated plainly:* the real number is **n = 1 control cycle** on different input data.
Treat 3.4× as order-of-magnitude, not precise. A second available cross-machine pair points the
other way (`local_dynamics` 0.123 ms real vs. 0.209 ms on ilab4 for the nearest equivalent), so
real-hardware timing is not predictable from ilab4 better than roughly ±3×. **Because of that,
every efficacy-relevant cost below is also given as a ratio to today's baseline solve, which is
machine-independent.**

> Equivalent ceiling measured on ilab4: **≈ 1.0 ms**.

---

## 2. Baseline profile — and a stale documented number

`modes.py`'s docstring states "~0.23 ms for a full 6-iteration solve — ~3 % of the 8 ms /
125 Hz budget." **That is now stale by ~7×.** It predates `ik_max_joint_deviation_rad`
(added 2026-08-06), which the current best gain vector enables.

Measured on idle ilab4, `neg45_wrist2offset`, production `_ik_newton_solve`, gains from
`search_result_nullspace_v2_20260806_194402.json`:

| configuration | mean | p95 | p99 |
|---|---|---|---|
| `ik_iterations=1` | 0.274 ms | — | — |
| `ik_iterations=3` | 0.807 ms | 0.838 | 0.855 |
| **`ik_iterations=6` (the deployed value)** | **1.571 ms** | **1.583** | **1.593** |
| `ik_iterations=10` | 2.635 ms | 2.657 | 2.751 |
| `ik_iterations=6`, null-space clip disabled | 1.211 ms | — | — |
| `ik_iterations=6` × 2 (`orientation_priority` on) | 3.143 ms | — | — |

Per-primitive (ilab4): `fk_and_jacobian` 0.029 ms · `swing_twist_axis_error` 0.021 ms ·
`null_space_basis` 0.021 ms · `build_weighted_least_squares_qp` 0.014 ms · `solve_box_qp` (n=6)
0.039 ms · 6×6 SVD 0.011 ms. By cProfile cumulative time the 6-iteration solve is 31 %
`swing_twist_axis_error` (18 calls, each re-normalising quaternions), 20 % `solve_box_qp` (its
fixed 80-iteration projected-gradient loop), 15 % `fk_and_jacobian`. The
`ik_max_joint_deviation_rad` clip costs **0.36 ms/cycle** (23 %).

### Standalone finding worth a human decision, independent of MPC

At 1.571 ms on ilab4, derated 3.4× → **≈5.3 ms on the real control PC**, which is **~1.5× over
the 3.5 ms budget and ~66 % of the entire 8 ms period**. `ik_seeded_resolution` has never been
run on real hardware (`docs/status/task_priority_orientation_hanging_2026-08-06.md` §6). It may
not fit the 125 Hz loop as currently configured. `ik_iterations=3` (0.807 ms → ~2.7 ms derated)
would fit; so would disabling the null-space clip. Flagged, **not acted on** — it is a real-time
budget question for a human, and the derating factor carries the ±3× caveat above.

---

## 3. The prototype

`tools/diagnostics/mpc_feasibility_prototype.py` (new, standalone, imported by nothing).
Formulation — a **continuation horizon**:

```
q_0 = q_rest,  q_{i+1} = q_i + u_i,          i = 0 .. N-1
stage targets p_1..p_N interpolate linearly from FK(q_rest) to the commanded p_des
             (stage N lands exactly on p_des)

min   sum_i  task_w * ||J_task(q_i) (q_i - q_i^nom) - task_err_i||^2
           + reg    * ||du_i||^2
           + w_cond * max(0, sigma_floor - sigma_min(J_full(q_i)))^2     <-- the new term
s.t.  joint box (production's own +-2pi), optional SQP trust region
```

solved SQP-style (relinearise the whole horizon, one stacked box QP over `(u_0..u_{N-1})` via
the same `build_weighted_least_squares_qp` / `solve_box_qp` pair production uses, repeat).
`sigma_min` gradients by forward differences (6 extra `fk_jacobian_fn` calls per active stage
per outer iteration). Returns `q_N` — the same object today's solve returns, so the downstream
joint-space P law is untouched.

`sigma_min` of the **full 6×6** Jacobian is the right barrier target (not the reduced task
Jacobian): the documented failure is a blowup of `pinv(J_full) @ xd_cmd`, both in the eval env's
joint-velocity estimate and, on real hardware, inside the robot's own `speedL` resolution. The
controller's own reduced-task matrix stays well-conditioned throughout (cond ~15).

**Validity checks, both passed:**

1. *Episode-harness equivalence.* `--mode validate` runs each case through both the real
   `VelocityTransportEnv` and the prototype's replica loop with the production solver. All four
   cases match to every reported digit (e.g. `wrist_sing_neg45`: `joint_velocity_guard 3.3645`,
   `ach=-0.0179`, `ori=0.1037`, `qd=3.365` — identical). The replica is a faithful harness.
2. *N=1 reduction.* With `w_cond=0`, the N=1 horizon is bit-identical to production
   `_ik_newton_solve` for 1–3 Newton iterations (max |Δq| = 4.4e-16) and thereafter differs only
   inside the numerically-undetermined null space (see §5).

Two formulation choices had to be fixed to get that reduction, both recorded in the file:
regularising the SQP *step* (not the cumulative `u`), and reusing production's exact joint box
rather than a symmetric one.

---

## 4. Does it change the outcome? Measured before/after

All at the reproducible fixed-gain best (104/128), same guards (|qd| ≤ 3.0 rad/s, orthogonal
drift ≤ 0.05 m, orientation ≤ 0.25 rad), same env config.

### 4a. `hanging_alpha_0_5 @ −0.296 m` — **no effect, at any setting**

| variant | move 1.0 s | move 0.02 s |
|---|---|---|
| baseline | `orientation 0.2518`, ach −0.2783 | `orientation 0.2528`, ach −0.1266 |
| N=1 horizon | `orientation 0.2507`, ach −0.2552 | `orientation 0.2522`, ach −0.1289 |
| N=2 | `orientation 0.2523`, ach −0.2261 | `orientation 0.2534`, ach −0.1257 |
| N=3 | `orientation 0.2521`, ach −0.2210 | `orientation 0.2539`, ach −0.1244 |
| N=5 | `orientation 0.2511`, ach −0.2119 | `orientation 0.2511`, ach −0.1241 |

Identical with the barrier on (`w=1e4/σ=0.03` and `w=1e6/σ=0.05` both tried). Note `sigma_min`
along these trajectories is 0.07–0.18 and |wrist_2| ≥ 1.35 rad — **this pose is nowhere near the
wrist singularity**, so a conditioning barrier has nothing to act on, exactly as
`task_priority_orientation_hanging_2026-08-06.md` §2b predicts. The horizon strictly *degrades*
X achievement while leaving orientation pinned at the guard. `−0.370 m` also unchanged
(and is separately proven unreachable). **Clean negative.**

### 4b. `neg45_wrist2offset`, dx −0.029 m, 1.0 s, `ik_max_joint_deviation_rad = 0.01`

Baseline: `joint_velocity_guard 3.3645 > 3.0`, achieving −0.0179 of −0.029 m (62 %).

*Barrier hyper-parameter sweep*, N ∈ {2,3,5,8} × w_cond ∈ {1e4,1e6,1e8,1e10} × σ_floor ∈
{0.03,0.05,0.10,0.20}:

* at `sqp=6`: **0 / 64 pass** (best 3.067 rad/s, still over)
* at `sqp=4`: **2 / 64 pass**

The barrier *does* do what it was designed to do kinematically — |wrist_2|min rises from 0.0129
(baseline) to as much as 0.070 — and **the guard trips anyway**, at up to 37.8 rad/s. Avoiding
the singularity is not what makes this case pass.

*The decisive control — does the win need the horizon?* Fixing the one working barrier setting
(`w=1e4`, `σ_floor=0.03`, `sqp=4`) and varying **only N**:

| dx (m) | baseline | N=1 | **N=2** | N=3 | N=4 |
|---|---|---|---|---|---|
| −0.0203 | PASS | PASS | PASS | PASS | PASS |
| −0.0261 | joint_vel | joint_vel | **PASS** (−0.0250) | joint_vel | joint_vel |
| −0.0290 | joint_vel | joint_vel | **PASS** (−0.0289) | joint_vel | joint_vel |
| −0.0319 | joint_vel | PASS | **PASS** (−0.0318) | joint_vel | joint_vel |
| −0.0377 | joint_vel | joint_vel | **PASS** (−0.0364) | joint_vel | joint_vel |
| −0.0464 | joint_vel | joint_vel | joint_vel | joint_vel | joint_vel |

N=2 recovers a contiguous −X band (4 cells, 96–100 % tracking, no +X regression, verified at
+0.029/+0.0464 m). That is the strongest positive signal found — and it should **not** be
trusted, for four measured reasons:

1. **Non-monotone in N.** N=1, N=3 and N=4 fail exactly where N=2 passes. There is no
   horizon-length dial; N=2 is a special point with no mechanism behind it.
2. **It works by saturation, not by solving anything.** The passing N=2 run's `q_target` jumps
   up to **12.7 rad between consecutive cycles**; it survives only because the controller's
   linear-speed clamp and the damped pinv bound the resulting command.
3. **It goes deeper into the singularity than baseline** (`sigma_min` 0.00022 vs. 0.0129) — the
   opposite of the barrier's stated purpose.
4. **It does not hold at fast moves.** At `move_duration_s = 0.02` only dx=−0.0261 is recovered.

### 4c. What actually trips the wrist guard — a q_target jump, not a Jacobian blowup

At the trip step: `max|qd| = 3.365` while `ik_joint_gain × max|q_target − q_current| = 47.92 ×
0.0729 = 3.49`. The joint-velocity command *is* the P law; the damped pinv contributes
essentially nothing (`sigma_min = 0.0056`, but `qd_estimate_damping = 1e-3` bounds it). At
`ik_joint_gain = 47.92`, **a q_target jump of only 0.0626 rad on its own saturates the 3.0 rad/s
guard.**

`q_target` cycle-to-cycle jump distribution over the episode:

| solver | p50 | p99 | max | outcome |
|---|---|---|---|---|
| baseline | 0.00379 | 0.04830 | **0.06541** | TRIP 3.365 @ t=0.584 |
| horizon N=2 barrier | 0.00000 | 3.46 | **12.69** | "complete" (by saturation) |
| horizon N=3 barrier | 0.01127 | 19.04 | **19.06** | TRIP 3.272 |
| horizon N=5 barrier | 0.01936 | 0.046 | 0.050 | TRIP 3.624 |

Baseline's max jump (0.0654 rad) is just past the 0.0626 rad saturation point — that *is* the
failure. **A horizon that re-solves from `q_rest` every cycle contains no term penalising
inter-cycle `q_target` motion**, so it cannot address this mechanism; measurably, it makes it
2–3 orders of magnitude worse.

---

## 5. Why the horizon is not well-posed here

The horizon SQP **does not converge**. Terminal position residual (m) by outer iteration,
N=3, barrier on, `w=1e6`, `σ_floor=0.05` (baseline 6-Newton-iteration reference: 1.6e-5 m):

| trust radius | sqp=1 | 2 | 3 | 4 | 6 | 10 | 20 |
|---|---|---|---|---|---|---|---|
| ∞, barrier **off** | 2.9e-3 | 7.1e-5 | 1.7e-4 | 4.8e-4 | 1.9e-5 | 2.9e-5 | 1.1e-4 |
| ∞, barrier **on** | 2.9e-3 | 9.8e-4 | 4.5e-4 | 5.5e-4 | **1.4e+0** | 1.3e+0 | 1.1e+0 |
| 0.2, barrier on | 5.3e-1 | 2.4e-1 | 4.0e-1 | 3.9e-1 | 4.0e-1 | 5.8e-4 | 3.3e-1 |
| 0.05, barrier on | 1.4e-1 | 9.7e-2 | 7.6e-2 | 1.4e-2 | 1.0e-1 | 1.2e-1 | 8.5e-2 |

Without a trust region the barrier makes it **diverge** (residual → 1.44 m, `q_target` 19 rad
from the reference). *With* a trust region it never converges — the residual chatters and never
settles, and a trust region tight enough to stabilise it (0.05, 0.01) also prevents the solve
from reaching the target at all. **There is no (trust radius, sqp) setting where the barrier is
active and the solve converges.** All §4 numbers therefore use `sqp=4`, the last iteration
before divergence — i.e. the horizon results above are the *best case*, taken from a solver that
is one step from blowing up.

**The root cause is upstream of the horizon.** At the searched gains
(`pinv_damping ≈ 1.23e-5` → `reg ≈ 1.5e-10` against `qp_task_weight ≈ 3.06e8`, a relative
regularisation of ~5e-19), the QP Hessian at convergence has **cond(H) = 1.27e17** — past double
precision's 4.5e15. Direct consequence, measured: two algebraically identical formulations of
the *same single-step production solve*, differing only in the numeric values of a
never-binding box, agree to 4.4e-16 for 3 Newton iterations and then diverge to **6.3e-2 rad**
by iteration 6. This quantifies
`task_priority_orientation_hanging_2026-08-06.md` §3b's remark that the redundant component "is
set by little more than the linear solver's own rounding."

A horizon stacks 6N such indeterminate variables and adds a non-smooth hinge whose active set
flips between outer iterations. It cannot be well-posed while its own subproblem is not.

---

## 6. Compute cost vs. N

Idle ilab4, means. "Derated" = ×3.4 (§1), against the **3.5 ms budget** and the 8.0 ms period.

Barrier on, `sqp=4` (the setting §4 actually used):

| solver | ilab4 | ×baseline | derated | vs. 3.5 ms budget |
|---|---|---|---|---|
| production `compute_ik_seeded` | 1.558 ms | 1.00× | 5.3 ms | already **1.5× over** |
| horizon N=1 | 2.152 ms | 1.38× | 7.3 ms | over |
| horizon **N=2** (the only one that helped) | 2.659 ms | 1.71× | 9.0 ms | over — **exceeds the whole 8 ms period** |
| horizon N=3 | 4.269 ms | 2.74× | 14.5 ms | over |
| horizon N=5 | 6.843 ms | 4.39× | 23.3 ms | over |

Full grid over (N, sqp, barrier), barrier off / on:

| | sqp=2 | sqp=3 | sqp=6 |
|---|---|---|---|
| N=1 | 0.717 / — | 1.015 / — | 1.877 / 2.006 |
| N=2 | 1.141 / 1.239 | 1.638 / 1.786 | 3.130 / 3.439 |
| N=3 | 1.566 / 1.723 | 2.276 / 2.518 | 4.346 / 4.840 |
| N=5 | 2.422 / 4.534 | 3.523 / 7.437 | 8.739 / 16.240 |

Cost scales roughly linearly in `N × sqp` (the FK/Jacobian calls dominate), with the barrier
adding a further ~6 FK calls per active stage per outer iteration — visible as the barrier's
cost premium growing from ~9 % at N=2 to ~87 % at N=5/sqp=6.

**The affordable and the useful do not overlap.** The only setting with any positive signal
(N=2, sqp=4, barrier) costs 2.66 ms on ilab4 → ~9.0 ms derated, more than the entire control
period. The only entries that would fit a 1.0 ms ilab4 ceiling (N=1/sqp=2, N=1/sqp=3) are below
today's solve in iteration count and helped nothing. Even taking the derating factor's ±3×
uncertainty at its most favourable, N ≥ 2 with a converged solve does not fit.

---

## 7. What would have to be true for this to become a GO

In dependency order — (a) is a hard prerequisite, not an optimisation:

**(a) The QP's redundant component must be well-determined first.** With `reg/task_w ≈ 5e-19`
and `cond(H) ≈ 1.3e17`, no trajectory optimiser over that null space is well-posed, at any
horizon length or compute budget. This needs a real regulariser — `pinv_damping` meaningfully
above 1.2e-5, or an explicit weighted posture term — which is a **gain-space change with its own
128-cell revalidation cost**, and the searches have repeatedly driven `pinv_damping` to the
small end precisely because it buys task accuracy. Until this is fixed, §5's non-convergence
will reappear in any horizon formulation.

**(b) The formulation would have to target the measured mechanism.** §4c shows the failure is
inter-cycle `q_target` discontinuity. Addressing it requires a horizon that includes
`q_current` and an explicit per-cycle rate constraint `|q_{i+1} − q_i| ≤ qd_max · dt`. That is a
genuinely different formulation from the one scoped here — and it **reintroduces path
dependence**, which is the entire reason `ik_seeded_resolution` exists (it was built to fix
`reduced_task_dims`' path-dependent multistability). That trade would need its own decision.

**(c) Roughly 3–7× more compute headroom**, via any of: a lower control rate (62.5 Hz / 16 ms
period would make N=2–3 fit); a compiled or vectorised solve (today's cost is Python/numpy
overhead — 31 % `swing_twist_axis_error`, 20 % `solve_box_qp`'s fixed 80-iteration loop — so
3–5× is plausibly recoverable without changing any math); or faster control hardware.

Note that (c) alone is worthless without (a) and (b).

---

## 8. Cheaper leads this surfaced — reported, not recommended, not implemented

* **`q_target` rate limiting.** Baseline solve plus a 0.005 rad/cycle limit on `q_target`
  completes the wrist case at **100 % of target** (ach −0.0290 for a −0.029 m command) where
  baseline trips at 3.365 rad/s. This follows directly from §4c. **Do not read this as a fix:**
  it is one case, it is non-monotone (0.05, 0.02 and 0.01 rad/cycle all still trip, and 0.01 is
  *worse* than 0.05), and it is a stateful filter, so it breaks the strict path-independence
  `ik_seeded_resolution` is built on. It needs its own 128-cell evaluation before it means
  anything.
* **The conditioning barrier does not need a horizon.** It is expressible in the existing
  single-step QP for ~0.6 ms extra (N=1: 2.15 vs. 1.56 ms). N=1 recovered one cell at 1.0 s and
  one at 0.02 s here. If singularity avoidance is wanted, that is the cheap place to put it —
  and it should be evaluated on its own merits on the full grid, not as a horizon.
* **The stale 0.23 ms figure and the real-time budget question in §2** are the most actionable
  items in this document and are independent of everything else here.

---

## 9. Scope and limits, deliberately not claimed

* Kinematic-only sim (`LocalMujocoDynamics` FK/Jacobian, no `mj_step`, no dynamics), on a
  control mode that has never run on real hardware. Nothing here is hardware-validated.
* Only two poses tested (`neg45_wrist2offset`, `hanging_alpha_0_5`), because those are the two
  documented structural failures this was scoped against. No 128-cell grid run was done — a
  no-go verdict did not warrant one, and none of the prototype's settings earned it.
* The derating factor (§1) rests on a single real-hardware control cycle. Every efficacy-relevant
  cost is therefore also stated as a ratio to today's baseline.
* The prototype's SQP is a first, straightforward implementation. A better-engineered
  trajectory optimiser (proper merit function, filter line search, active-set handling for the
  hinge) would converge more reliably than §5 shows — but §4a (no effect on `hanging`), §4c
  (wrong mechanism), §5's `cond(H) = 1.3e17` root cause, and §6 (no affordable N) are all
  independent of solver quality, and each blocks a GO on its own.

---

## 10. Files, reproduction, rollback

Created (2, both new, nothing else touched):

| path | what |
|---|---|
| `tools/diagnostics/mpc_feasibility_prototype.py` | standalone prototype + all measurement modes |
| `docs/status/mpc_feasibility_2026-08-07.md` | this document |

```bash
source /common/users/ss5772/miniforge3/etc/profile.d/conda.sh && conda activate mujoco_ur5e
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONPATH=/common/users/ss5772/real_Cartpole
P=tools/diagnostics/mpc_feasibility_prototype.py

python $P --mode selfcheck                                  # N=1 reduction check (§3)
python $P --mode breakdown                                  # §2 baseline profile + hot spots
python $P --mode profile --reps 300                         # §6 cost grid
python $P --mode validate --sqp-iterations 4                # §4a, §4b, harness equivalence
python $P --mode sweep --case wrist_sing_neg45 --sqp-iterations 4   # §4b 64-cell barrier sweep
python $P --mode mechanism --cond-weight 1e4 --sigma-floor 0.03     # §4c jump probe + N control
```

Timing modes must be run on an idle host (ilab4 was used; westeros was at load ~55).

**Rollback** — nothing else was touched, no config or production module changed, nothing
committed:

```bash
rm tools/diagnostics/mpc_feasibility_prototype.py docs/status/mpc_feasibility_2026-08-07.md
```

**Tests run:** none — no production code was modified, so the suite is unchanged. The
prototype's own correctness is established by `--mode selfcheck` (N=1 reduction to production)
and by `--mode validate`'s `baseline_env` vs. `baseline_replica` rows matching to every reported
digit on all four cases.
**Tests not run:** `pytest` (unit / mujoco / hardware) — untouched by this work.
