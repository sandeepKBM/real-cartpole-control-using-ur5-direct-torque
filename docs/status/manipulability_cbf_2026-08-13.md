# Manipulability Control Barrier Function (CBF) — built, sim-validated, default off

**Date:** 2026-08-13
**Scope:** simulation only. No `hardware/*.py` touched. No existing config's gains changed.
No existing named config had the flag enabled.
**Method source:** *Safe, Task-Consistent Manipulation with Operational Space Control
Barrier Functions* (OSCBF), arXiv:2503.06736 — Sec. V-B1 (singularity-avoidance CBF),
Eq. 5 (high-order CBF), Eq. 11–15 (operational-space / rigid-body dynamics). That paper is
a 7-DOF Franka Panda result; **nothing robot-specific was borrowed, only the method.**

---

## 1. What was built, and why it is a different kind of thing

This repo already had three singularity mechanisms, and every one of them is **reactive or
offline**:

| mechanism | when it acts | what it does |
|---|---|---|
| `jacobian_singular_cond_max` | after `cond(J)` is already bad | scales the whole wrench by one scalar (isotropic) |
| `svd_singularity_filtering` (SCI) | after a singular value is already small | per-direction damped-least-squares back-off |
| offline pose pre-filtering | before the run | pick a start pose with good `cond(J)` |

None of them constrains the arm's **motion**. A CBF does: it adds an inequality row to a
per-cycle QP so the commanded torque is *required* to keep a barrier non-negative, which
bounds the **rate of approach** to the singular set rather than the **magnitude** of the
response once you are in it. It is complementary to, not a replacement for, SCI — it does
nothing about authority already lost in a direction that is already singular.

Note also that `mu(q)` (Yoshikawa manipulability, the **product** of the singular values)
is a different quantity from `cond(J)` (the **ratio** of the extremes), not a rescaling of
it. On this arm `mu` is very nearly *linear* in the distance from the singular set, which
is what makes it usable as a barrier (measured, `assets/ur5e_torque/scene.xml`, sweeping
`wrist_2` at `HEIGHT_ALPHA_0_5_Q`):

```
wrist_2 (rad)   0.000     0.005     0.020     0.050     0.200    1.200
mu             1.2e-18   9.4e-05   3.8e-04   9.4e-04   3.7e-03  1.8e-02
cond(J)        7.3e+16   9.1e+02   2.3e+02   9.1e+01   2.9e+01  2.3e+01
```

---

## 2. The derivation (the part that had to be right)

State `z = (q, qd)`; QP decision variable `u = tau` (joint torque — the level this repo's
plant interface accepts, and the level `hard_constraint_qp.py` already established for a
genuine linear inequality here).

**1. Barrier** (OSCBF Sec V-B1):

```
mu(q) = prod_i sigma_i( J(q) ) = sqrt( det( J J^T ) )
h(z)  = mu(q) - eps                                       (eps > 0)
```

`mu` is computed from `svd(J)` directly, not from `det(J)`: equal in exact arithmetic for a
square `J`, but `prod(svd)` has no catastrophic cancellation and is non-negative by
construction.

**2. Relative degree.** `h` depends on `q` only, so `hdot = grad_mu(q) . qd` contains no
`tau` — relative degree 2 w.r.t. a torque input. (The paper says exactly this: velocity
control is relative degree 1, torque control is 2.) Resolved with its Eq. 5 high-order CBF,

```
h2(z) = hdot(z) + alpha1( h(z) )
Lf h2(z) + Lg h2(z) u  >=  -alpha2( h2(z) )
```

which for **linear** class-K `alpha1(s) = a1 s`, `alpha2(s) = a2 s` expands to

```
hddot + (a1 + a2) hdot + a1 a2 h  >=  0                    (*)
```

Homogeneous solutions decay as `exp(-a1 t)`, `exp(-a2 t)`: `{h >= 0}` is forward-invariant,
and from `h < 0` the constraint *drives h back up* rather than merely holding — which
matters, because this repo's real transport start pose literally sits at `mu = 1.2e-18`.

**3. Affine in tau.** With `g(q) := grad_mu(q)` and `H_mu(q) := d^2 mu/dq^2`,

```
hddot = d/dt ( g . qd ) = qd^T H_mu qd + g . qddot
M(q) qddot + b(q,qd) = tau   =>   qddot = M^-1 ( tau - b )
=> hddot = g^T M^-1 tau - g^T M^-1 b + qd^T H_mu qd
```

Only the first term contains `tau`, and it is linear in it.

**4. The constraint row.** Substituting into (\*) and rearranging into the `A x <= b` form
`solve_constrained_box_qp` accepts:

```
A = -( g^T M^-1 )                                          shape (1, 6)
b = -g^T M^-1 b_dyn + qd^T H_mu qd
    + (a1 + a2) (g . qd) + a1 a2 (mu - eps)                scalar
```

**5. The QP** (OSCBF's structure: minimum deviation from the nominal control law, subject
to the CBF and the input limits):

```
min_tau  || tau - tau_nominal ||^2
s.t.     A tau <= b                    (the CBF row)
         tau_lo <= tau <= tau_hi       (the torque-headroom box)
```

### Design forks, and how they were called

- **Torque or task acceleration?** *Torque.* It is what the plant interface takes, it is
  what `hard_constraint_qp.py` already does, and it lets the filter sit at the very end of
  the pipeline where it constrains the torque that is actually commanded (including every
  bias term) rather than a task fragment that later gets scaled by backtracking.
- **Hard constraint or soft penalty?** *Hard.* A soft penalty is what every gain-based
  mechanism in this repo already is, and this session's whole premise is that those are
  exhausted. A hard row also gives a real infeasibility signal instead of silently
  mis-answering.
- **Does `box_qp.py` need extending?** **No — and this was checked, not assumed.**
  `solve_box_qp` supports box bounds only, but `controller_core/constrained_box_qp.py`
  (added 2026-08-03 for `hard_constraint_qp.py`) already wraps it with general linear
  inequality rows via dual ascent, and a CBF row is one more instance of exactly that
  class. The objective is built with `box_qp.build_weighted_least_squares_qp`, the existing
  shared builder. **No new solver was written and no existing solver was modified.**
- **`mu` from `J` or from `J_task`?** *Full 6x6 `J`, always.* A `split_base_wrist_task` /
  `reduced_task_dims` / `task_lock_*` `J_task` is rank-deficient **by construction**
  (`mu == 0` identically), which would make the barrier meaningless.
- **Full Hessian or directional second derivative?** *Directional.* `hddot` needs exactly
  the scalar `qd^T H_mu qd`, which is a second difference along `qd/||qd||` — 2 extra
  Jacobian evaluations instead of ~36 to build and contract the full Hessian.
- **Where does `grad_mu` come from?** A `jacobian_fn` (`q -> J(q)`) passed to the
  controller's constructor. There is no way around this: `grad_mu` depends on `dJ/dq`, and
  the per-cycle state contract carries `J(q)` at the current `q` only. It cannot ride on the
  state dict either — `state_types.as_impedance_robot_state` normalizes to plain arrays and
  would drop a callable. Enabling the flag without one **raises**.

### What is approximated (stated, not assumed away)

- `b_dyn` is the controller's own gravity term `g` this cycle. That is exactly right in
  both lanes: where the adapter compensates gravity externally (the MuJoCo lane, which does
  not put `gravity_torque` on the state) `g == 0` and the plant really does see
  `qddot = M^-1 tau`; where the controller compensates it itself, `g` is inside
  `tau_preclip` and must come back out. **Coriolis is omitted** — the same standing
  approximation `hard_constraint_qp.py` and the rest of this repo already make.
- `mu` is **not differentiable** where a singular value crosses zero (it behaves like
  `c|wrist_2|`), so a finite-difference curvature whose step straddles the kink measures the
  kink, not a curvature. The failure direction is benign — the term comes back large and
  positive, which makes the row trivially satisfiable, i.e. it degrades to *no constraint*,
  not to a wrong correction. Asserted as a locked-in documented limitation in
  `tests/unit/test_manipulability_cbf.py::test_curvature_is_meaningless_within_a_step_of_the_singular_kink`.
- The MuJoCo adapter's torque rate limiter (800/160 Nm/s) throttles how fast the
  correction actually reaches the plant.

---

## 3. Files changed

| file | what |
|---|---|
| `controller_core/manipulability_cbf.py` | **new, 470 lines.** Full derivation in the module docstring; `manipulability`, `manipulability_gradient`, `manipulability_directional_curvature`, `manipulability_cbf_constraint_row`, `manipulability_cbf_filter`, `ManipulabilityCBFResult`. Pure numpy. |
| `.../x_axis_cartesian_impedance/config.py` | +6 fields (`manipulability_cbf`, `_epsilon`, `_alpha1`, `_alpha2`, `_fd_step`, `_curvature_step`) with the full docstring block; +6 lines in `from_controller_yaml_section`; 2 new parser imports. |
| `.../x_axis_cartesian_impedance/parsing.py` | `_parse_manipulability_cbf_epsilon`, `_parse_manipulability_cbf_alpha` — both **loud** (raise on `<= 0`, NaN, inf, unparseable) rather than falling back, because a silently-defaulted safety mechanism is the exact failure class this module's other parsers exist to reject. |
| `.../x_axis_cartesian_impedance/controller.py` | keyword-only `jacobian_fn=None` on `__init__`; two guards (`jacobian_fn` missing, `mass_matrix` missing); one `if use_manipulability_cbf:` block between backtracking and the final clip; 7 new output kwargs. |
| `.../x_axis_cartesian_impedance/output.py` | +7 default-inert diagnostic fields. |
| `simulation/ur5e_mujoco_torque.py` | `make_mujoco_jacobian_fn()` (own scratch `MjData`, `mj_kinematics`+`mj_comPos`+`mj_jacSite` only); `build_controller(..., jacobian_fn=None)` keyword-only passthrough; `build_initial_state_and_adapter` builds and passes one. |
| `tools/diagnostics/manipulability_cbf_sim_check.py` | **new.** `--mode profile` / `--mode rollout`, real model, CBF on vs off, +X and -X. |
| `tests/unit/test_manipulability_cbf.py` | **new, 48 tests.** |
| `tests/mujoco/test_manipulability_cbf_closed_loop.py` | **new, 11 tests.** |
| `config/ur5e_mujoco_torque_osc_tuned_manipulability_cbf.yaml` | **new.** The tuned config plus *only* the CBF block. |

---

## 4. Golden-value diff: default behavior is byte-identical

Not an assertion — an actual before/after run. The pre-change `controller_core` was
reconstructed by reverse-applying exactly this change's edits, and a driver was run under
both trees:

- **75 real configs** from `config/*.yaml` that parse as impedance configs (the entire
  directory; the one new demo config that *intentionally* enables the flag is excluded and
  named as excluded),
- **12 deterministic randomized states each** — including exact `wrist_2 == 0`,
  `wrist_2 == 1e-4`, and rank-deficient Jacobians — **900 `compute()` calls**,
- every field of `CartesianImpedanceOutput` compared as **raw float64 bytes**.

```
configs=75  compute_calls=900  pre-existing_fields_compared=45000  MISMATCHES=0
new fields present only in 'after': ['manipulability', 'manipulability_cbf_active',
  'manipulability_cbf_delta_tau_norm', 'manipulability_cbf_feasible',
  'manipulability_cbf_h', 'manipulability_cbf_h_dot', 'manipulability_cbf_slack']
```

---

## 5. Closed-loop results on the REAL model

`assets/ur5e_torque/scene.xml`, `config/ur5e_mujoco_torque_osc_tuned.yaml` gains, MuJoCo
`qfrc_bias` gravity, no Coriolis feedforward, 2.0 s min-jerk move + 1.0 s hold, `eps = 1e-3`,
`a1 = a2 = 10`. Everything except the flag is byte-identical between the two runs of a pair.

### 5.1 The wrist singularity (`wrist_2 -> 0`) — the case this repo keeps fighting

`HEIGHT_ALPHA_0_5_Q` with `wrist_2 = 0.10 rad`, **world-Y** transport `dx = -0.10 m`:

| | min `mu` | min &#124;wrist_2&#124; | max `cond(J)` | CBF cycles active | max Δτ | tracking |
|---|---|---|---|---|---|---|
| **CBF off** | 1.69e-04 | **0.00567 rad** | 8.89e+02 | – | – | 0.500 |
| **CBF on** | **1.84e-03** | **0.0930 rad** | **5.27e+01** | 421 | 0.97 Nm | 0.309 |

The baseline walks `wrist_2` 94% of the way to the singularity. The CBF holds it within 7%
of its start value, for **under 1 Nm** of torque correction: **10.9x** higher `mu` floor,
**16.9x** lower peak `cond(J)`. Never infeasible; `h` never went negative at all here
(min `h` = +8.4e-4).

Both runs trip the orientation guard at ~0.25 rad — that is a **pre-existing** failure of
Y-axis transport at this pose (the baseline trips too, earlier). The CBF neither causes nor
fixes it.

### 5.2 A second, genuinely different UR singularity (found while building this)

`HEIGHT_ALPHA_0_5_WRIST2_OFFSET_Q` (`wrist_2 = 0.20`), **world-X** transport `dx = +0.15 m`:

| | min `mu` | min &#124;wrist_2&#124; | max `cond(J)` | CBF cycles active | max Δτ | tracking |
|---|---|---|---|---|---|---|
| **CBF off** | 1.77e-06 | 0.197 | 5.64e+04 | – | – | 0.977 |
| **CBF on** | **6.77e-04** | 0.200 | **2.48e+02** | 688 | 4.93 Nm | 0.920 |

**383x** higher `mu` floor, **227x** lower peak `cond(J)`. Instrumenting the baseline shows
this is *not* a `wrist_2` event: `wrist_2` stays at ~0.20 rad the whole time while
`shoulder_lift` moves −0.835 → −0.987 rad and `shoulder_lift + elbow + wrist_1` approaches
−π — the **wrist-alignment** singularity family (the wrist-3 axis lining up with the
shoulder-pan axis). A `wrist_2` threshold would have missed it entirely; `mu` caught it.
This is real evidence the barrier is a general kinematic quantity, not a `wrist_2` proxy.

Cost of the intervention here: the CBF run trips the axis-error-growth guard at t≈2.2 s
(the baseline does not). That is an honest trade — the filter is *refusing* to let the
controller buy tracking with manipulability — not a bug, but it is why this is not
proposed as a default.

### 5.3 Exact no-op where nothing is approached (both directions, AGENTS.md §7)

Same pose, `dx = +0.05 m`, `-0.10 m`, `-0.15 m`: **0 CBF cycles active, `Δτ == 0`, and
`min_mu` / `achieved_delta` / `max_tau` / step count / guard outcome all exactly equal to
the baseline's.** `+X` approaches the alignment singularity at this pose and `-X` does not —
a real directional asymmetry, consistent with everything else this repo has found.

### 5.4 Composition with the reduced-joint / reduced-row mechanisms

Checked rather than assumed (`dx = +0.15 m`, same pose). **All three compose; none needs a
guard or a raise:**

| overlay | baseline min `mu` / max `cond` | with CBF | active | infeasible |
|---|---|---|---|---|
| `split_base_wrist_task` (3-row) | 1.60e-06 / 6.24e+04 | 5.51e-04 / 3.10e+02 | 1078 | 0 |
| `split_base_wrist_task_dims=[0]`, joints `[1,2,3]` (1-row) | 1.45e-06 / 7.05e+04 | 6.91e-04 / 2.68e+02 | 1121 | 0 |
| `reduced_task_dims` (xyz only) | 2.54e-06 / 4.21e+04 | 4.75e-04 / 3.51e+02 | 986 | 0 |

The one interaction worth stating explicitly (and stated in the flag's docstring): because
`mu` is read from the full `J` and the filter acts on the final torque, the CBF **may
command torque on a joint the reduced task deliberately excludes**. That is correct for a
safety filter — it is not the task — but it means those flags' "structurally cannot route
through it" property applies to the *task* torque, not to the CBF's correction.

### 5.5 Barrier violation is bounded, never a collapse

`h` dips slightly below zero in the harder cells (min `h` = −1.2e-4 at `dx = 0.10 m`,
−3.2e-4 at `dx = 0.15 m`, i.e. `mu` ≈ 0.68·`eps` at worst) and never below `−eps`.
Expected: a continuous-time CBF enforced on a discrete grid through a rate-limited
actuator. `cbf_infeasible_steps == 0` in every cell tested.

---

## 6. Cost — the real blocker for hardware

Measured on this host, conda env `mujoco_ur5e`, this model:

| | µs/cycle |
|---|---|
| one `jacobian_fn` call (`mj_kinematics`+`mj_comPos`+`mj_jacSite`) | 29.6 |
| `grad_mu` (12 central-difference evals) | 568 |
| directional curvature (2 evals) | 154 |
| **filter, constraint inactive** (short-circuits before the QP) | **785** |
| **filter, constraint active** (adds the dual-ascent QP) | **3560** |
| full `compute()`, flag off | 320 |
| full `compute()`, flag on and active | ~4800 |

**~4.8 ms/cycle against the real `direct_torque` loop's 2 ms budget.** This is sim-only
work so nothing is blocked today, but **as implemented this cannot run on the 500 Hz real
loop.** The QP is ~78% of the added cost (`solve_constrained_box_qp`'s bracket-and-bisect
root-find issues up to ~65 inner `solve_box_qp` calls for a single row); a one-row
`||tau - tau_nom||^2` QP with a box has a near-closed-form solution, so this is an
optimization problem, not a design dead end. Not attempted here.

---

## 7. Tests

- `tests/unit/test_manipulability_cbf.py` — **48 tests.** `mu` vs `|det J|` and
  `sqrt(det J J^T)`; `grad_mu` vs a closed form (`mu(q) = |sin q4|`) at four
  configurations; directional curvature vs its closed form and its quadratic-form scaling
  (2x `qd` → 4x the term — the check a first-derivative-shaped mistake fails); the
  constraint row against a fully hand-computed case; a **sign check** that a torque
  increasing `mu` satisfies the row and the opposite one violates it; exact no-op far from
  a singularity (`np.array_equal`, not `allclose`); **no-op when `mu < eps` but moving
  away fast** — the property that distinguishes a CBF from a threshold; activation and
  correction direction when driven in; output satisfies its own row; infeasibility reported
  and the box respected; gradient-plateau degenerate row skipped; the kink limitation;
  config defaults, YAML round-trip, loud parser rejections; both controller guards; and
  flag-on-far-from-singularity output byte-identical to flag-off.
- `tests/mujoco/test_manipulability_cbf_closed_loop.py` — **11 tests** on the real model,
  covering §5.1–5.3 and 5.5 plus a check that the documented `mu` table still matches the
  model (so the default `eps` cannot silently stop meaning what the docstring says), and an
  explicit *premise* test that the baseline really does reach the singularity (without which
  the comparison would be vacuous).
- Full suite, conda env `mujoco_ur5e` (`/common/users/ss5772/miniforge3/envs/mujoco_ur5e/bin/python`
  — note the bare `miniforge3/bin/python3` lacks `pinocchio`/`optuna`/`gymnasium` and
  produces ~60 spurious failures):
  - `tests/unit` + `tests/hardware`: **944 passed, 1 failed** — the failure is
    `test_residual_observer_async_phase_cost_is_much_lower_than_sync`, a wall-clock timing
    assertion in the concurrently-modified hardware lane.
  - `tests/mujoco`: **345 passed, 5 failed, 3 xfailed** (16m32s) — all 5 failures are in
    `tests/mujoco/test_velocity_gain_tuning.py`, the velocity-control lane, which imports
    nothing this change touches (`grep` over `velocity_gain_tuning/` finds no reference to
    `XAxisCartesianImpedanceController`, `CartesianImpedanceConfig`, or `manipulability`;
    its only import from `simulation/ur5e_mujoco_torque.py` is `x_profile_target`, which is
    unmodified). Both failure sets are pre-existing work from the session's other track.
- **NOT run:** this repo's 4-category rigor sweep (canonical grid / long holds / large
  displacements / torque-scale robustness) at any `height_alpha`, with the flag on.

---

## 8. What is NOT done

1. **No real hardware, at all.** And §6's timing says it could not run there as written.
2. **No 4-category rigor sweep.** The evidence here is 4 pose/axis/direction cells plus 3
   composition overlays, not this repo's standard envelope.
3. **`eps = 1e-3` is a measured default for the `height_alpha_0_5` pose family only**
   (`mu ≈ 0.019·|wrist_2|` there; `mu` grows ~4x faster at `MEGA_SEARCH_WINNER_Q`). It is
   not a universal constant and must be re-read off §1's table for any new pose family.
4. **`a1`/`a2` were not tuned.** 10.0/10.0 is a reasoned first choice (~0.1 s reaction
   horizon), not a search result.
5. **The tracking cost was not characterized.** §5.2 shows a cell where the CBF's refusal
   to spend manipulability trips a guard the baseline does not. Whether that is a tuning
   problem (`a1`/`a2`, `eps`) or structural is unknown.
6. **Coriolis omitted** from `b_dyn` (§2), consistent with the rest of the repo, never
   quantified for this specific use.
7. **Not enabled in any existing named config**, and
   `config/ur5e_mujoco_torque_osc_tuned.yaml` is unmodified.
8. **The real-hardware lane cannot use it at all today, and fails loudly if asked to.**
   `hardware/direct_torque_transport.py:260` and `hardware/position_transport.py:174`
   construct `XAxisCartesianImpedanceController(impedance_cfg)` with no `jacobian_fn`
   (correctly — `hardware/*.py` was out of scope and untouched), so a config with
   `manipulability_cbf: true` handed to either would raise the "constructed without
   jacobian_fn" `ValueError` at the first `compute()`. That is the intended failure
   direction (loud, not a silently disabled safety filter), but it means wiring a real
   kinematic model into the hardware lane is a prerequisite for any hardware trial — on top
   of the timing work in §6.

---

## 9. Rollback

Everything except two files is either new or purely additive.

```bash
cd /common/users/ss5772/real_Cartpole

# 1. Delete the new files (nothing else imports them).
rm -f controller_core/manipulability_cbf.py \
      tools/diagnostics/manipulability_cbf_sim_check.py \
      tests/unit/test_manipulability_cbf.py \
      tests/mujoco/test_manipulability_cbf_closed_loop.py \
      config/ur5e_mujoco_torque_osc_tuned_manipulability_cbf.yaml \
      docs/status/manipulability_cbf_2026-08-13.md

# 2. Revert the five edited files. NOTE: these files also carry other,
#    unrelated work from this session, so do NOT `git checkout` them --
#    remove only the manipulability-CBF blocks:
#      controller_core/x_axis_cartesian_impedance/config.py
#        - the `# Manipulability Control Barrier Function (CBF).` comment block
#          and its 6 fields
#        - the 6 `manipulability_cbf*=` lines in from_controller_yaml_section
#        - the 2 `_parse_manipulability_cbf_*` names in the `.parsing` import
#      controller_core/x_axis_cartesian_impedance/parsing.py
#        - `_parse_manipulability_cbf_epsilon`, `_parse_manipulability_cbf_alpha`
#      controller_core/x_axis_cartesian_impedance/output.py
#        - the trailing `# Manipulability-CBF diagnostics` block (7 fields)
#      controller_core/x_axis_cartesian_impedance/controller.py
#        - the `from ..manipulability_cbf import ...` line
#        - `jacobian_fn` on __init__ (restore the one-line signature)
#        - the `use_manipulability_cbf` guard block
#        - the `# Manipulability CBF (default off; ...)` block before `tau = tau_preclip`
#        - the 7 `manipulability_cbf*=` output kwargs
#      simulation/ur5e_mujoco_torque.py
#        - `make_mujoco_jacobian_fn`
#        - `jacobian_fn` kwarg on build_controller and its passthrough
#        - the `jacobian_fn=make_mujoco_jacobian_fn(...)` argument in
#          build_initial_state_and_adapter
#        - `Callable` in the typing import

# 3. Re-verify (should be exactly the pre-change suite):
/common/users/ss5772/miniforge3/envs/mujoco_ur5e/bin/python -m pytest tests/unit -q
```

If the change is committed first, the rollback is simply `git revert <sha>` — the whole
feature is confined to one commit's worth of the files above.
