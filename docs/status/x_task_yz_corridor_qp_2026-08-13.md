# Reduced-task (X + orientation) torque QP with a Y/Z corridor — design, and three bug fixes

**Date:** 2026-08-13
**Code:** `controller_core/x_task_yz_corridor_qp/`
**Configs:** `config/ur5e_mujoco_torque_x_task_yz_corridor_qp.yaml` (mechanisms off),
`config/ur5e_mujoco_torque_x_task_yz_corridor_qp_enabled.yaml` (corridor + manipulability CBF on)
**Tests:** `tests/unit/test_x_task_yz_corridor_qp.py`,
`tests/mujoco/test_x_task_yz_corridor_qp_closed_loop.py`
**Diagnostic driver:** `tools/diagnostics/x_task_yz_corridor_qp_sim_check.py`
**Pose:** `ARM_Q0 = [-2.3688, -2.1801, -1.8838, -0.7962, 0.004714693, 0.0206]`
(from `tools/diagnostics/x_task_yz_corridor_ik_feasibility_check.py`), `cond(J) = 1395.8`,
`mu = 4.326e-4`. **Sim-only scope** — see §8.

---

## 1. Architecture

### 1.1 The reduced task

```
J_reduced = vstack([J[0:1, :], J[3:6, :]])          # (4, 6): world X + 3 orientation rows
H         = 2 (J_reducedᵀ W J_reduced + reg·I)      # W = diag(kp_x, kp_rot, kp_rot, kp_rot)
tau_des   = J_reducedᵀ wrench + tau_damping + tau_posture + tau_yz_soft + gravity
min over tau of  0.5 (tau - tau_des)ᵀ H (tau - tau_des)   s.t.  box, A_ineq tau <= b_ineq
```

Y and Z are excluded **by construction**: they are not rows of `J_reduced`, so no Y/Z state can
touch `H` or `tau_task_nominal`. Their only authority is the low-priority `tau_yz_soft` bias in
the linear term (gains 5.0/2.0, an order of magnitude below `kp_x`) plus the corridor rows at the
walls. Asserted byte-identically in the unit tests, not approximately.

### 1.2 Y/Z corridor as a high-order CBF

For `h = y_max − y(q)`, with `hdot = −J_y qd` exact and
`hddot ≈ −J_y M⁻¹ (tau − bias)` (dropping `Jdot_y qd`, the same standing approximation
`hard_constraint_qp.py` states), the HOCBF condition `hddot + (a1+a2) hdot + a1 a2 h ≥ 0`
becomes two linear rows per axis. Half-width 0.05 m, sized ~18% above the largest validated
natural Y transient at a real transport pose (0.0423 m, `neg45_y_axis_diagnosis_and_fix_2026-08-01.md`).

### 1.3 Manipulability CBF

`manipulability_cbf_constraint_row(...)` called directly (never `manipulability_cbf_filter`,
which is a closed single-row solve and is not composable). `epsilon = 3.0e-4`, sized below
`mu(ARM_Q0) = 4.326e-4` so the row is a genuine barrier at the start pose rather than a
permanent recovery push.

### 1.4 One solve

All rows stacked into `solve_constrained_box_qp` once, box = torque headroom ∩
`_velocity_implied_torque_bounds`.

---

## 2. Bug 1 — shoulder_pan drifted 4.3–13.2 deg during ordinary X transport

### 2.1 The bug

The 4-row task on a 6-joint arm leaves **2 genuinely redundant DOF**, and nothing in the
original design said which joints were allowed to absorb them — only the soft posture spring
(`kp_posture = 25`). Measured at ARM_Q0, `..._enabled.yaml`, shoulder_pan range over the move:

| dx (m) | move (s) | shoulder_pan range |
|---|---|---|
| −0.06 | 1.5 | 5.20 deg |
| +0.06 | 0.1 | 4.32 deg |
| +0.12 | 1.5 | 12.16 deg |
| −0.12 | 1.5 | 8.93 deg |
| +0.20 | 1.5 | 13.15 deg |

This is not incidental. At ARM_Q0 the world-X row of `J` gives shoulder_pan the **largest
coefficient of any joint** (0.2366, vs elbow's −0.2346), so the X task actively preferred to
swing the base. ARM_Q0 pins shoulder_pan at −135.7 deg for real wall/base clearance, so a
finite-stiffness spring is not an acceptable guarantee.

Half of the orientation error was the same phenomenon: at the dx = +0.12 peak,
`e_rot = (0.2072, 0.0364, −0.1202)`, and the `rz` component (49.6% of the norm) is exactly the
12.16 deg of pan drift — shoulder_pan's `rz` Jacobian coefficient at this pose is **1.0**.

### 2.2 Mechanism selection — measured, not argued

Two candidates were implemented and measured head to head at ARM_Q0 with both mechanisms on:

| mechanism | dx | track | pan range | ori max | \|qd\| max | guard |
|---|---|---|---|---|---|---|
| none (before) | −0.06 | 0.814 | 5.20 deg | 0.1417 | 0.148 | — |
| none (before) | +0.12 | 0.889 | 12.16 deg | 0.2423 | 0.304 | — |
| (a) column zeroing | −0.06 | 0.690 | 0.03 deg | 0.1500 | 0.243 | — |
| (a) column zeroing | +0.12 | 0.545 | 1.28 deg | 0.1036 | 0.531 | Z drift |
| (b) box pin only | −0.06 | 0.526 | 0.00 deg | 0.2515 | **2.111** | orientation |
| (b) box pin only | +0.12 | 0.286 | 0.00 deg | 0.0752 | **1.928** | Z drift |

**(a) alone is not a guarantee.** Zeroing the joint's column of `J_reduced` removes only
`J_reducedᵀ wrench`; `tau_damping`, `tau_posture`, `tau_yz_soft` and any deviation the QP makes
under an active row all still reach it. With the corridor rows on, shoulder_pan still moved
0.03 deg (dx = −0.06) and 2.12 deg (dx = +0.12), peak commanded pan torque 6.09 Nm — the
corridor rows route torque straight into the joint the zeroed column was protecting.

**(b) alone is a guarantee but a bad controller.** With the column left in place,
`H = 2(J_rᵀ W J_r + reg I)` still couples the pinned coordinate to the free ones, and because
`W` is dominated by `kp_x` the QP responds to the pin by re-optimizing the other five joints to
restore the lost `j_x · tau` projection. That is a **force-space** quantity, not the
acceleration `j_x M⁻¹ tau` that actually moves the tool, so the "compensation" over-drives the
wrists rather than reproducing the motion. Ablation isolates the trigger precisely — it is the
**corridor rows**, not the manipulability row:

| pin-only config | dx | \|qd\| max | ori max | guard |
|---|---|---|---|---|
| corridor off, CBF off | −0.06 / +0.12 | 0.328 / 0.479 | 0.121 / 0.138 | Z drift |
| corridor off, CBF **on** | −0.06 / +0.12 | 0.327 / 0.479 | 0.121 / 0.138 | Z drift |
| corridor **on**, CBF off | −0.06 / +0.12 | 1.973 / 1.720 | 0.251 / 0.252 | orientation |
| corridor **on**, CBF on | −0.06 / +0.12 | 2.111 / 1.928 | 0.252 / 0.075 | orientation / Z |

Raising `kp_rot` does not rescue pin-only (kp_rot = 200 → tracking 0.115; kp_rot = 400 → 0.140,
both still tripping), nor does raising `posture_regularization` (0.35 → 5 → 50 tames `|qd|` to
0.427 but leaves orientation at 0.2501). That is what identified the **Hessian coupling**, not
the rotational gains, as the cause.

### 2.3 The fix: (a) AND (b), together

They are complementary, not rivals — each removes the other's defect.

* **(a)** zeroes the excluded joint's column of `J_reduced`. Consequence:
  `H[free, excluded] == 0` exactly (the only surviving entry in that coordinate is the diagonal
  Tikhonov `2·reg`), so the pinned coordinate no longer couples into the free ones and the pin
  costs the free joints nothing — they keep `tau_des`, damping and all.
* **(b)** pins the box shut, `tau_lo[i] = tau_hi[i] = tau_hold[i]`, where
  `tau_hold = tau_damping + tau_posture + gravity`. `solve_box_qp` clips every iterate into the
  box, so `tau_preclip[i] == tau_hold[i]` **bit-for-bit**, independent of the Hessian, of which
  rows are active, and of the solver's iteration budget.

Config field: `task_excluded_joints`, default `(0,)` (shoulder_pan), `()` to disable, at most 2
indices (4 task rows on 6 joints ⇒ 2 redundant DOF; `_parse_task_excluded_joints` raises
otherwise). Pinned to the **bias**, not to zero: at ARM_Q0 gravity torque on shoulder_pan is
exactly 0.0 (vertical axis), but that is a property of this joint at this mounting, not a
general one, and the excluded joint keeps an active spring/damper hold at `q_rest`.

Kinematic screening (the check `split_base_wrist_active_joints`' docstring says to do):
dropping shoulder_pan takes `cond(J_reduced)` from 10.1 to 519 but keeps **rank 4**, and the
least-squares joint velocity for 1 m/s of world X with zero angular velocity is
**5.15 rad/s vs 4.87 rad/s** with all six joints, residual 3e-15. The task is genuinely
feasible without shoulder_pan; the degradation measured above is a property of the
torque-transpose control law, not of the kinematics.

Solver accuracy under a pin, measured rather than assumed (`solve_box_qp` is projected gradient
with a fixed 80-iteration budget): at ARM_Q0's Hessian the free coordinates agree with the
exact eliminated-variable solution to **1.6e-4 Nm** over 200 random `tau_des` (torque limits are
28–150 Nm); on a deliberately ill-conditioned synthetic fixture (`cond(H) ≈ 3.1e3`) 80
iterations is **not** enough — ~8.9 Nm out, converging by ~2000 — but still captures >99% of the
available objective improvement. With the column zeroed the question is moot for the pinned
coordinate, which is exact either way.

---

## 3. Bug 2 — `kd_rot` was never unit-converted

`config/ur5e_mujoco_torque_osc_tuned.yaml`'s gains are **acceleration**-domain: that controller
runs `task_space_inertia_shaping`, so the applied wrench is `Λ(q) · (kp·e + kd·edot)`. This
controller has no `Λ` term at all — its gains are plain force/torque-domain. `kp_x`/`kd_x` were
correctly converted through a measured `Λ_xx`; `kp_rot`/`kd_rot` were copied over unchanged.

Re-measured at ARM_Q0 with the tuned config's own `lambda_regularization = 0.1`, reproducing the
documented `Λ_xx` exactly as a methodology check:

```
Λ = (J M⁻¹ Jᵀ + 0.1 I)⁻¹
Λ_xx          = 5.92977          (documented factor: 5.9298 — reproduced)
Λ[3:6, 3:6]   = [[0.12063,  0.07049, -0.35566],
                 [0.07049,  0.14517, -0.31995],
                 [-0.35566, -0.31995, 3.27179]]
diag          = [0.12063, 0.14517, 3.27179]
eigenvalues   = [0.05873, 0.13426, 3.34461]
```

**Derived factor = trace(Λ_rot)/3 = 1.1792**, giving `kd_rot = 10.0 × 1.1792 = 11.79`.

Justification for that particular scalar summary, since several are defensible:

* The `kp_x` conversion used `Λ[0,0]` — the **diagonal entry of the task row in the world/task
  basis**, not an eigenvalue. The direct analogue for the three orientation rows is their
  diagonal `[Λ_33, Λ_44, Λ_55]`.
* `kd_rot` is a single scalar applied identically to all three rows, so the scalar that
  preserves the total rotational weighting is the **mean of that diagonal** — equivalently
  `trace(Λ_rot)/3`, since `Σ (kd_rot · 1) over 3 rows = kd_rot · trace(I)`.
* It is emphatically **not** 5.9298: the rotational block is ~5x smaller than `Λ_xx`, so reusing
  the translational factor would have overshot by that ratio.

The factor is a local property but a stable one: re-measured at ARM_Q0 plus 8 random
±0.15 rad perturbations of it, `mean diag(Λ_rot)` ranges 0.771–1.179 (median 1.086) while
`Λ_xx` ranges 4.39–6.61 — i.e. the 1.1792 used here is the high end of a narrow band, not a
knife-edge value.

**Named limitation.** `Λ_rot` is strongly anisotropic here — the diagonal spans 0.121 to 3.272,
a factor of **27**, and the eigenvalues span 0.059 to 3.345, a factor of **57**. A single scalar
`kd_rot` cannot represent that; a per-axis rotational gain vector could, and is a real follow-on,
but it is a new mechanism rather than a unit fix and was deliberately not built here. The
derived 11.79 is therefore treated as the **centre of the search range in §4**, not as the
final answer.

---

## 4. Bug 3 — `kp_rot = 0` was a stale assumption from a different architecture

`kp_rot = 0` in `osc_tuned.yaml` is deliberate and well documented: at the wrist singularity the
task rotation PD is *unstable* through the eps-regularized `Λ`, positive feedback regardless of
magnitude, so orientation is held by the posture anchor instead. **That reasoning is specific to
a Λ-weighted operational-space law.** This controller is a plain weighted-least-squares QP with
no `Λ` anywhere, so the instability mechanism does not exist here — and the value was carried
over without re-examination.

The consequence is worse than a missing restoring term. `task_weights = diag(kp_x, kp_rot,
kp_rot, kp_rot)` with a floor of `1e-6`, so at `kp_rot = 0` the Hessian carries **essentially no
weight on the three orientation rows**: `H ≈ 2(kp_x · j_xᵀ j_x + reg I)` is effectively rank-1
plus Tikhonov, and the QP is free to discard orientation-directed torque (including the wrist
damping in `tau_des`) at a cost of only `2·reg`.

### 4.1 The search

`scipy.optimize.differential_evolution` (this repo's standing rule — never RL) over
`(kp_rot, kd_rot)` in `[0, 400] × [0, 60]`, `popsize=8`, `maxiter=12`, `seed=0`,
`mutation=(0.5, 1.0)`, `recombination=0.7`, `polish=False`, `updating="deferred"`. Run **after**
`task_excluded_joints` landed, because the two fixes interact: the exclusion is what makes the
orientation rows' Hessian weight matter.

Objective, summed over `dx ∈ {−0.06/1.5 s, +0.06/0.1 s, +0.12/1.5 s}` at ARM_Q0:

```
max orientation error
  + 3.0 · max(0, tracking_ref − tracking)     # ref = the PRE-FIX run, never rewarded above it
  + 10.0 · guard_tripped                      # disqualifying, not merely bad
  + 2.0 · max(0, hold-phase max|qd| − 0.10)   # sustained hold-phase velocity == oscillation
  + 1.0 · max(0, max|qd| − 1.5)               # the 0.1 s cell legitimately reaches ~1.4
```

Converged in 6 iterations / **112 evaluations** (2367 s wall, 24 workers), `tol=0.01` reached.
Deterministic: an earlier identical run reproduced steps 1–3 to the last digit.

**Result: `kp_rot = 35.121920`, `kd_rot = 52.787714`.**

| cell | tracking | max orientation | shoulder_pan range | max \|qd\| | guard |
|---|---|---|---|---|---|
| dx = −0.06 / 1.5 s | 0.703 | **0.0797** | 0.0018 deg | 1.703 | — |
| dx = +0.06 / 0.1 s | 0.684 | **0.0536** | 0.0015 deg | 1.787 | — |
| dx = +0.12 / 1.5 s | 0.287 | **0.0483** | 0.0020 deg | 1.773 | Z drift |

### 4.2 Reading the result

**`kd_rot` landed at 52.79, i.e. 4.5x the unit-derived 11.79. That is not a contradiction of
§3, it is the expected consequence of also fixing bug 3.** The conversion factor answers "what
does the old number mean in this controller's units" and gets the scale right (10 → 11.79). It
cannot answer "what damping does a PD pair need once `kp_rot` is no longer zero" — the old value
was a damping-ONLY term with no proportional partner. Both fixes were needed; neither derivation
substitutes for the other.

**The `kp_rot > 0` question is answered affirmatively, and non-monotonically.** An independent
hand-probe over the same combined mechanism, run separately from the search:

| kp_rot / kd_rot | dx = −0.06 | dx = +0.06 / 0.1 s | dx = +0.12 |
|---|---|---|---|
| 0 / 11.79 | 0.655, 0.2453, — | 0.663, 0.2501, **ori trip** | 0.294, 0.0896, Z trip |
| 25 / 11.79 | 0.749, 0.1861, — | 0.710, 0.1856, — | 0.305, 0.1184, Z trip |
| 50 / 11.79 | 0.691, 0.1838, — | 0.676, 0.1667, — | 0.307, 0.1783, Z trip |
| 100 / 11.79 | 0.550, 0.2509, **ori trip** | 0.676, 0.1933, — | 0.311, 0.2123, Z trip |
| 50 / 24 | 0.686, 0.1333, — | 0.680, 0.1013, — | 0.337, 0.2073, Z trip |
| **35.1 / 52.8 (search)** | **0.703, 0.0797, —** | **0.684, 0.0536, —** | 0.287, 0.0483, Z trip |

(`tracking, max orientation rad, guard`.) `kp_rot` too large is as bad as zero — 100 trips the
orientation guard at dx = −0.06 m — and the best row is the one with the *most* damping, not the
most stiffness. The search's answer beats every hand-probed point on orientation by ~1.7-5x.

### 4.3 The hold-phase `|qd|` is a settling transient, not oscillation

Peak `|qd|` at the tuned gains occurs during the HOLD phase (1.703 rad/s at dx = −0.06 m), which
would normally read as a limit cycle. Traced directly and it is not: over all 500 hold-phase
cycles the fastest joint (wrist_1) has **zero velocity sign changes**, decaying monotonically
1.250 → 0.297 rad/s, while the orientation error rises smoothly and decelerates
(0.0553 → 0.0798 rad, no overshoot, no ringing). It is a slow one-directional post-move
settling, well inside the 3.0 rad/s guard. Named as a limitation rather than dismissed: it is
slower settling than the pre-fix controller had (`|qd|` 0.148 there), and it is the direct cost
of routing the task through five joints instead of six.

---

## 5. Before / after on the five-case matrix

All cells at ARM_Q0 through `config/ur5e_mujoco_torque_x_task_yz_corridor_qp_enabled.yaml`
(corridor + manipulability CBF on), 1.0 s hold, `gravity_source=mujoco_qfrc`, Coriolis
feedforward off — the same pipeline
`tools/diagnostics/x_task_yz_corridor_qp_sim_check.py::run_rollout` uses.

**BEFORE** = `task_excluded_joints: []` with `kp_rot = 0`, `kd_rot = 10` — i.e. the controller
exactly as it was when the bugs were found, reproduced rather than quoted.

**AFTER** = the shipped config: `task_excluded_joints: [0]`, `kp_rot = 35.121920`,
`kd_rot = 52.787714`. Everything else identical.

| dx / move | | tracking | shoulder_pan range | max orientation | max \|qd\| | guard |
|---|---|---|---|---|---|---|
| −0.06 m / 1.5 s | before | 0.814 | 5.20 deg | 0.1417 | 0.148 | — |
| | **after** | 0.702 | **0.00 deg** | **0.0798** | 1.703 | — |
| +0.06 m / 0.1 s | before | 0.823 | 4.32 deg | 0.1167 | 1.411 | — |
| | **after** | 0.684 | **0.00 deg** | **0.0536** | 1.787 | — |
| +0.12 m / 1.5 s | before | 0.889 | 12.16 deg | 0.2423 | 0.304 | — |
| | **after** | 0.294 | **0.00 deg** | **0.0456** | 1.749 | **Z drift** |
| −0.12 m / 1.5 s | before | 0.772 | 8.93 deg | 0.2503 | 0.485 | **orientation** |
| | **after** | 0.475 | **0.00 deg** | **0.1273** | 2.096 | **— (recovered)** |
| +0.20 m / 1.5 s | before | 0.586 | 13.15 deg | 0.2439 | 0.538 | Z drift |
| | **after** | 0.180 | **0.00 deg** | **0.0705** | 1.927 | Z drift |

Reading it straight:

* **Bug 1 is fully fixed, in every cell.** shoulder_pan's range goes from 4.32–13.15 deg to
  0.0015–0.0020 deg (printed as 0.00) — a 2000–8000x reduction — and its peak commanded torque
  from 5.79–11.90 Nm to ≤0.01 Nm. The pin held bit-exactly on every cycle of every rollout
  (`excluded_joint_pin_violations == 0`).
* **Bugs 2/3 are fixed, in every cell.** Orientation error improves everywhere, by 1.8x
  (dx = −0.06) to **5.3x** (dx = +0.12). The three cells that previously sat AT the 0.25 rad
  guard (0.2423 / 0.2503 / 0.2439) now sit at 0.0456 / 0.1273 / 0.0705.
* **dx = −0.12 m is recovered**: it used to trip the orientation guard at 0.2503 rad and now
  completes cleanly at 0.1273.
* **X-tracking degrades, and dx = +0.12 m is a genuine regression.** Small/moderate cells lose
  ~14% of tracking (0.814 → 0.702, 0.823 → 0.684). The large cells lose much more, and
  dx = +0.12 m goes from completing to tripping the Z-drift guard. Completed cells are 3/5 both
  before and after — but not the same three.

**This trade is structural, not a tuning gap, and it is the honest headline of this pass.** At
ARM_Q0 shoulder_pan has the largest Jacobian coefficient of any joint in world X (0.2366),
world Y (−0.578, 2.5x the next joint) and `rz` (1.0). Holding it out of the task removes the
primary actuator for the transport axis, for the axis the corridor constrains, and for a third
of the orientation task at once. Three independent attempts to buy the tracking back all failed
or traded sideways: a derived `kp_x` compensation (§8), `posture_regularization` up to 50, and
the full `(kp_rot, kd_rot)` search itself. The guarantee is what was asked for and it is
delivered exactly; the cost is real and is stated rather than tuned away.

**Practical floor:** with `task_excluded_joints: [0]`, trust `|dx| ≤ 0.06 m` at this pose —
which is exactly the range the corridor half-width's own calibration evidence covers anyway
(§1.2). For `|dx| ≥ 0.12 m`, either accept the Z-drift trip or set `task_excluded_joints: []`
and accept 8–13 deg of base swing; there is no setting of the gains in this architecture that
gives both.

---

## 6. Swing-up benchmark at ARM_Q0 — SUPERSEDED, DO NOT USE THESE NUMBERS

> **RETRACTED 2026-08-14.** Every number in this section was measured against the **wrong
> target angle** and none of it is trustworthy. A separate investigation found that
> `find_inverted_angle` — corrected earlier on 2026-08-13 to read the arm pose from `data` — was
> being called by every caller with a bare `mujoco.MjData(model)`, i.e. an **all-zeros arm
> pose**. Equilibria therefore resolved to `(-1.5621, +1.5795)` rad instead of the correct
> `(-3.0145, +0.1271)` rad — off by ~1.45 rad, with the hanging and inverted labels very nearly
> swapped. The fix changes real outcomes, not just labels: an energy-shaping trial that
> previously tripped `|Y-Y0| > 0.03 m` at step 328 now completes all 1000 steps clean, and an LQR
> 0.02 rad perturbation went from `peak_theta_err` 1.7316 rad (falling away) to 0.02366 rad
> (actually balancing).
>
> Consequences for this document: the "1.8643 rad energy-shaping" and "1.5049 rad multi-kick"
> figures below are **not valid baselines**, the comparison between them is void, and nothing
> new should be compared against either. The whole benchmark needs re-running against the
> corrected equilibrium before any of it means anything. The text is kept only so the retraction
> is legible and the numbers are not quietly re-used from an earlier draft.
>
> **This does not touch the transport work.** §2–§5 and §7–§10 — the three bug fixes, the
> five-case matrix, the gain searches, the byte-identical refactor proof and the closed-loop
> tests — run entirely through `x_task_yz_corridor_qp_sim_check.run_rollout` on the bare arm and
> never resolve a pendulum equilibrium at all.

### 6.1 Energy-shaping, matched methodology (superseded)

Driver: `pendulum_swingup_energy_shaping.run_energy_swingup_trial` with its own
`config_path` / `controller_kind` parameters, its own 6-parameter bounds
`(k_e, a_max, k_pos, k_vel, kick_amplitude_m, kick_duration_s)`, its own cost (min distance
from inverted, `+5.0` for a guard trip, `+5.0` for `cond(J)` growth > 10x), same seed, same
`popsize=8 / maxiter=12` budget, same 6 s search trial and 10 s re-validation. The only
difference between the two arms is `(controller_kind, config)`. The budget is smaller than the
shipped script's `popsize=14 / maxiter=30` because the corridor QP costs far more per cycle —
but it is reduced **identically on both sides**, which is what makes the comparison fair.

**Baseline (`impedance`, `config/ur5e_mujoco_torque_osc_tuned.yaml`):**

| | |
|---|---|
| best cost (6 s trial) | 1.9682 |
| re-validated at 10 s, min distance from inverted | **1.8643 rad** (need < 0.35) |
| flipped | no |
| guard fired | no |
| `cond(J)` growth ratio | 1.019 (no singularity walk) |
| max X excursion | 0.0295 m |
| evaluations | 624, 407.5 s wall |

`reached_singularity` reports `True` only because ARM_Q0's own `cond(J) = 1396` already exceeds
the diagnostic script's `SINGULARITY_COND_THRESHOLD = 1000` **at rest, before any motion** — a
known property of this pose already flagged in
`docs/status/transport_axis_generalization_and_pendulum_axis_2026-08-12.md`, not a property of
this trial. The growth ratio (1.019) is the discriminating number and it is clean.

**New controller (`x_task_yz_corridor_qp`, corridor + CBF on): DELIBERATELY ABANDONED, not
completed.** The search was launched under identical settings and stopped at generation 7 of 12
(best cost so far 1.9744, vs the baseline's 1.9682 — i.e. tracking slightly *worse* than
baseline and nowhere near a flip). It was stopped on a deliberate priority call, not because it
crashed:

* with `task_excluded_joints: [0]` the usable displacement at this pose collapsed to ~4 cm of
  actual achieved motion (§5), and energy-shaping swing-up needs large kicks — the remaining
  ~5 generations would have spent an hour confirming a no-flip that the transport numbers
  already predict;
* the previously documented energy-shaping ceiling at this pose was traced to the **arm** and
  its drift guards, not to the swing-up law
  (`transport_axis_generalization_and_pendulum_axis_2026-08-12.md`), and the bug-fix pass made
  the arm's usable range smaller, not larger.

**No conclusion is claimed from the partial run.** The honest statement is: the new controller
was not shown to beat the previously documented energy-shaping ceiling, and it was not shown
to fail to either — the comparison is unfinished by choice, and is now doubly moot because the
target angle it was scored against was wrong (see the §6 retraction). Re-run
`swingup_compare.py new` if the X+Z work (§10) restores enough range to make it worth an hour.

---

### 6.2 Multi-kick (superseded)

`tools/diagnostics/pendulum_swingup_multi_kick.py`, run unmodified at its shipped budget
(`popsize=16, maxiter=30`, **3** parameters: `kick_amplitude_m`, `kick_duration_s`,
`phi_trigger_rad`) against the same baseline controller:

| | |
|---|---|
| best cost (8 s trial) | 1.5222 |
| re-validated at 15 s, min distance from inverted | **1.5049 rad** |
| best parameters | amplitude 0.0567 m, duration 0.1293 s, trigger 0.6808 rad (39.0 deg) |
| kicks delivered | 28 |
| guard fired | no |

**~~The finding: a 3-parameter multi-kick law gets substantially closer to inverted (1.5049 rad)
than a 6-parameter energy-shaping law does (1.8643 rad), on the same arm, at the same pose.~~**
**VOID** — both figures were measured against the wrong equilibrium (see the retraction at the
top of §6). The relative comparison is not rescued by both arms sharing the bug: the two laws
were scored by distance to a target angle that was itself ~1.45 rad wrong, so neither the
ordering nor the magnitudes survive.
That is the opposite of what search-budget intuition predicts — energy shaping has twice the
parameters and a physically motivated law — and it is consistent with the previously documented
~2.24 rad energy-shaping ceiling being a property of the **arm**, not of the swing-up law's
expressiveness (`docs/status/transport_axis_generalization_and_pendulum_axis_2026-08-12.md`
traced that ceiling to the controller tripping its own drift guard on any kick large enough to
matter). Neither number is a flip (need < 0.35 rad); both are ceilings.

---

## 7. Test coverage

| suite | result |
|---|---|
| `tests/unit/test_x_task_yz_corridor_qp.py` | 76 passed (12 new for `task_excluded_joints`) |
| `tests/unit` + `tests/hardware` | 1021 passed |
| `tests/mujoco/test_x_task_yz_corridor_qp_closed_loop.py` | 39 passed (5 new tests, 4 updated) |

New unit coverage asserts the mechanism, not just its effect: the excluded column really is
zeroed in the Hessian (`H[0, free] == 0` exactly) while an unexcluded run's is not; the pin is
bit-exact under no rows, corridor rows, and simultaneous box saturation + walls; the pin cannot
widen the torque box; the pin costs the free coordinates nothing; two-joint exclusion; and the
parser rejects out-of-range, negative, duplicate, non-integer and >2-index sets.

New closed-loop coverage: shoulder_pan range < 0.5 deg and commanded pan torque < 1.0 Nm across
`dx ∈ {±0.02, ±0.06}`; orientation error < 0.10 rad with no guard trip on the same set;
`excluded_joint_pin_violations == 0` on every cycle of every rollout; and X-tracking still ahead
of the 6D OSC controller.

Four pre-existing closed-loop tests encoded claims the fixes legitimately changed and were
**corrected rather than loosened**, each with the reason and the new measurement in its
docstring: (1) "Y moves less than OSC" became "Y moves more" — shoulder_pan is the dominant Y
actuator and is now excluded (0.0120 vs 0.0069 at dx = +0.02 m), with the corridor-containment
bound kept; (2) the corridor's exact-no-op claim was narrowed to `±0.02 m` and a new test
asserts it now genuinely engages at `±0.06 m` (649/728 active cycles, from 0); (3) the
composition claim's manipulability ratio bound went 50x → 20x against a measured 27x (the
absolute `mu > 3.0e-4` floor, which is the meaningful half, is unchanged); (4)
`test_neither_mechanism_alone_completes_the_large_move_but_together_they_do` was replaced by
`test_the_large_move_is_now_outside_the_envelope_for_every_flag_combination`, which asserts the
regression explicitly so it cannot be silently inherited.

---

---

## 8. Scope and known limitations

**Sim-only.** The corridor + manipulability QP's measured per-cycle cost is over the 500 Hz
`direct_torque` budget (2.0 ms); the flags-off reduced-task QP is under it. That is asserted as
an honesty test in `test_qp_cost_is_measured_and_is_over_the_500hz_budget` rather than left to a
doc that could go stale. Nothing here has been run on real hardware.

**`task_excluded_joints` is pose-specific.** The index set is screened against ARM_Q0 only
(rank 4, `cond(J_reduced)` 10.1 → 519, 5.15 vs 4.87 rad/s for 1 m/s of world X). Nothing checks
at runtime that the remaining columns span the task — the same caveat
`split_base_wrist_active_joints` documents at length in the impedance controller. Repeat the
screening before moving this to another pose.

**Excluding shoulder_pan costs real X and Y authority at this pose, and that is not a defect of
the fix — it is the price of the guarantee.** At ARM_Q0 shoulder_pan has the largest Jacobian
coefficient of any joint in *three* of the rows that matter: world X (0.2366), world Y (−0.578,
2.5x the next joint) and `rz` (1.0). Holding it out of the task therefore removes the primary
actuator for the transport axis, for the axis the corridor constrains, and for one third of the
orientation task simultaneously. The measured consequences are reported honestly in §5 rather
than tuned away.

**A `kp_x` compensation for that lost authority was derived and measured, and is NOT adopted —
it trades one failure for another.** The principled factor is
`(j_x M⁻¹ j_xᵀ)_full / (j_x M⁻¹ j_xᵀ)_free = 1.6100` at ARM_Q0 — the X acceleration per unit of
task torque along the task direction, computed in the same spirit as `Λ_xx` — giving
`kp_x = 2400 × 1.61 = 3864`, `kd_x = 386`. Measured against the same three cells:

| gains | dx = −0.06 / 1.5 s | dx = +0.06 / 0.1 s | dx = +0.12 / 1.5 s |
|---|---|---|---|
| kp_x 2400, kp_rot 0 | 0.655, 0.2453, — | 0.663, 0.2501, **ori trip** | 0.294, 0.0896, Z trip |
| kp_x 2400, kp_rot 25 | 0.749, 0.1861, — | **0.710, 0.1856, —** | 0.305, 0.1184, Z trip |
| kp_x 3864, kp_rot 0 | 0.567, 0.2509, **ori trip** | 0.588, 0.2514, **ori trip** | 0.294, 0.1241, Z trip |
| kp_x 3864, kp_rot 50 | **0.821, 0.1928, —** | 0.606, 0.1629, **Z trip** | 0.297, 0.1683, Z trip |

(cells are `tracking fraction, max orientation error rad, guard`.)

Two things fall out, both worth stating because both are counter-intuitive:
1. The compensation is **useless or harmful on its own**. At `kp_rot = 0` it makes every cell
   worse and adds an orientation trip at dx = −0.06 m. More X force with no orientation
   authority to absorb it is not an improvement.
2. Paired with `kp_rot = 50` it **does** do what it was derived to do — dx = −0.06 m tracking
   goes to 0.821, i.e. fully back to the pre-fix 0.814 — but it costs the fast
   dx = +0.06 m / 0.1 s cell, which goes from clean to a Z-drift guard trip.

Since the fast cell is part of the matrix these bugs were found on, `kp_x` is left at 2400 and
the search below is run there. Recorded in full so the derivation is not re-attempted from
scratch and so the trade is visible rather than hidden inside a chosen number.

**A per-axis rotational gain is the obvious next mechanism, and was deliberately not built.**
`Λ_rot`'s diagonal spans a factor of 27 at ARM_Q0 (§3), so a single scalar `kd_rot` cannot
match all three orientation rows. Replacing the scalar with a 3-vector is a new mechanism, not
a unit fix, and belongs in its own change with its own validation.

**`posture_regularization` is a real third knob that this search did not use.** Raising it from
0.35 to 50 cut the pin-only divergence's peak `|qd|` from 2.111 to 0.427 rad/s while leaving
the orientation error unchanged at 0.2501 rad — i.e. it damps the QP's freedom to wander from
`tau_des` without addressing the orientation error itself. The search in §4 was kept to
`(kp_rot, kd_rot)` as scoped; widening it to include `posture_regularization` is a reasonable
follow-on with real supporting evidence.

**The dropped `Jdot_y qd` term in the corridor HOCBF** remains the standing approximation
(§1.2): the barrier is slightly optimistic when the Jacobian's Y row rotates fast, so the
corridor can be transiently overshot rather than being a hard invariant. Whether including it
changes behavior has still not been measured.

**Not an infeasibility problem.** The pin-only divergence was checked against the obvious
suspect — the dual ascent failing to reach a feasible point and driving multipliers to
`dual_max` — and rejected: `infeasible_steps == 0` on every cell measured, before and after.
The large torque deviations are the genuine constrained optimum, not a solver blow-up.

---

## 9. Files changed

* `controller_core/x_task_yz_corridor_qp/{config,controller,output,parsing}.py` — the
  `task_excluded_joints` mechanism, `tau_hold` on the output, the index parser, and (§10) the
  `task_axis_rows`/`corridor_axis_rows` generalization.
* `config/ur5e_mujoco_torque_x_task_yz_corridor_qp{,_enabled}.yaml` — `task_excluded_joints: [0]`,
  `kp_rot`/`kd_rot`.
* `config/ur5e_mujoco_torque_x_z_task_y_corridor_qp_enabled.yaml` — NEW (§10), X+Z tracked /
  Y bounded.
* `tools/diagnostics/x_task_yz_corridor_qp_sim_check.py` — additive: shoulder_pan range /
  deviation / torque and `excluded_joint_pin_violations` on `RolloutResult`, plus a `panDeg`
  column. No behavior change.
* `tests/unit/test_x_task_yz_corridor_qp.py`, `tests/mujoco/test_x_task_yz_corridor_qp_closed_loop.py`.

**Not touched, as scoped:** `controller_core/x_axis_cartesian_impedance/*`, `torque_task_qp.py`,
`hard_constraint_qp.py`, `manipulability_cbf.py`, `controller_core/safety.py`. No safety guard
was weakened anywhere — the only threshold that moved is a TEST bound, and it moved tighter
(orientation 0.15 → 0.10 rad).

---

## 10. Configurable task/corridor axis rows — implemented, validation UNFINISHED

### 10.1 Why Z, and why not Y

§5's regression is entirely a **Z** phenomenon: with `task_excluded_joints: [0]`, every large-move
failure at ARM_Q0 trips `|Z-Z0| > 0.06 m`, in both directions, with Z sitting at 94–100% of its
corridor half-width. Meanwhile `shoulder_lift` — the joint best placed to hold height — moves
0.11–1.33 deg across those same moves while the now-locked `shoulder_pan` used to do 9–13 deg of
the work. A constrained-IK feasibility sweep found continuous, well-conditioned solutions holding
Z inside 0.05 m across the full ±0.20 m X range, so **Z is holdable while X moves; it simply was
not being asked to be held** — a corridor only pushes back at the wall, and by then the arm is
already there. Promoting Z from a bounded axis to a tracked one is the direct response.

**Y is deliberately left as a corridor axis.** `neg45_y_axis_diagnosis_and_fix_2026-08-01.md`
established across three independent investigations that no P, D or I gain in this controller
family can hold Y without breaking X-tracking at this pose family, because Y is kinematically
coupled to X here. Adding Y as a task row would re-litigate a settled question.

### 10.2 The mechanism

`task_axis_rows` (default `[0]`) and `corridor_axis_rows` (default `[1, 2]`) replace the
hardcoded `vstack([J[0:1,:], J[3:6,:]])` and the hardcoded Y/Z corridor pair. Both are indices
into the world translation rows (0 = X, 1 = Y, 2 = Z); the three orientation rows are always task
rows, since there is no corridor formulation for orientation here.

They are validated **as a pair** (`_parse_axis_row_sets`), because the only interesting failure
mode — an axis appearing in both sets, i.e. the controller fighting itself with two mechanisms
designed as alternatives — is invisible to independent per-field validators. Also rejected, all
loudly: indices outside 0–2, duplicates, an empty task, an X in `corridor_axis_rows` (the
half-widths are the per-axis `y_`/`z_corridor_half_width_m` fields and there is no X one), and
`task_axis_rows` without the transport axis (every target, tolerance and guard in this lane
describes X, so an untracked X would mean the arm never drives toward its target while every
metric still said it should).

Two details worth stating because they are easy to get silently wrong:

* **The soft centering bias is applied to the corridor axes only.** An axis promoted to a task
  row draws its authority from the task; leaving the bias on would double-count it, and do so
  with a gain deliberately sized to be negligible. `y_error`/`z_error` are still *reported* for
  both, since a trace consumer wants distance-from-start regardless of which mechanism holds it.
* **`yz_corridor_active_rows` keeps its fixed `(y_max, y_min, z_max, z_min)` shape** whatever the
  row sets are; a tracked axis contributes no rows and reports `False` for both of its slots. A
  trace stays readable across configs.

### 10.3 The default is bit-identical, and that is proven rather than asserted

`test_default_row_sets_reproduce_the_x_only_controller_bit_identically` re-states the
pre-generalization formulas literally and requires `np.array_equal` — not `allclose` — on the
Hessian, `tau_task_nominal`, `tau_yz_soft` and the QP's own `tau_preclip`. This matters because
every closed-loop number in §4 and §5 was validated against the old code path; without a
byte-identical guarantee they would all silently stop applying.

`tests/unit/test_x_task_yz_corridor_qp.py`: **89 passed** (13 new for the row sets).

### 10.4 Gains re-derived from the measured row-restricted Λ — bug 2's lesson applied

A 5-row task sees a different Λ than a 4-row task, so the X-only numbers were **not** reused.
Measured at ARM_Q0 with `lambda_regularization = 0.1`, using the operator each row set actually
sees, `Λ_red = (J_red M⁻¹ J_redᵀ + εI)⁻¹`:

| row set | `Λ_red` diagonal | translation | rotation mean |
|---|---|---|---|
| X-only `(0,3,4,5)` | `[5.01960, 0.09752, 0.10084, 1.64419]` | 5.01960 | 0.61418 |
| X+Z `(0,2,3,4,5)` | `[5.25945, 3.25520, 0.09892, 0.12832, 1.67327]` | `[5.25945, 3.25520]` | 0.63350 |

`cond(J_reduced)` at ARM_Q0 is 10.51 with all six joints and 704.28 with `shoulder_pan` excluded,
rank 5 either way — screened and feasible, the same check §2 applied to the X-only row set.

| gain | derivation | value |
|---|---|---|
| `kp_x` | 2400 × 5.25945/5.01960 — corrects for the row-set change only, same basis as X-only | 2514.7 |
| `kd_x` | 240 × same ratio | 251.5 |
| `kp_z` | 120 × 3.25520 — from **osc_tuned's own hard-task Z gains**, exactly how `kp_x`'s 400 → 2400 was derived. Deliberately NOT from the 5.0 soft-bias value, which is a different mechanism at ~1e-2 of a task gain | 390.6 |
| `kd_z` | 20 × 3.25520 | 65.1 |
| `kp_rot`/`kd_rot` | 35.121920/52.787714 × 0.63350/0.61418 = 36.23/54.45 — used as the search **centre**, not the answer | see 10.5 |

Config: `config/ur5e_mujoco_torque_x_z_task_y_corridor_qp_enabled.yaml`. Identical to the X-only
enabled config in every mechanism, guard, corridor calibration and `task_excluded_joints` — a
unit test asserts that field-by-field, so the two configs cannot silently drift apart on anything
except the row sets and the gains re-derived for them.

### 10.5 STATUS: validation incomplete — do not treat the early search signal as a result

The `(kp_rot, kd_rot)` search for the X+Z row set (same objective, same three cells, bounds
widened to `[0,150] × [0,120]` around the converted centre) reached **generation 6 of 12 with a
best objective of 15.3666** and then could not be completed: the host's home filesystem hit its
disk quota (`EDQUOT`), which blocked all command execution.

For calibration, the objective adds a flat **10.0 per guard-tripped cell** across three cells.
The X-only search **converged** at 20.0831, and its per-cell detail shows that decomposes as
**exactly one** tripped cell — `dx = +0.12 m`, Z-drift, cell cost 12.1272 = 10.0 trip + 1.75
tracking shortfall + 0.05 orientation + 0.33 oscillation — plus 3.8244 and 4.1316 from the two
clean cells. (An earlier draft of this section said "two trips"; that was an arithmetic error,
corrected here.)

**The X+Z 15.3666 cannot be decomposed the same way, and therefore proves nothing yet.** Two
readings fit it arithmetically: one tripped cell plus 5.37 of other penalty, or zero tripped
cells plus 15.37 of tracking/orientation/oscillation penalty — and the second is entirely
plausible, because tracking Z necessarily spends X authority, and the tracking-shortfall term
alone is weighted 3.0 against a `dx = +0.12 m` reference of 0.889. The per-cell breakdown that
would separate these is only written at convergence, which this run did not reach.

**So this is a partial-run number, not a finding, and it is deliberately not written up as one.**
The X-only value is converged and the X+Z value is truncated at half the generations, so they are
not comparable on equal terms even before the decomposition problem. The claim "tracking Z
removes the Z-drift trips" requires the five-case matrix (`xz_matrix.py`, written and ready) to
show it per-cell, and that has not been run.

Remaining to close this out, in order: finish the search and apply its gains; run the X-only vs
X+Z five-case matrix with measured per-cycle QP cost (the row count should drop from 5 inequality
rows to 3 when Z stops needing corridor rows — predicted, not yet measured); re-run the X-only
closed-loop suite to confirm in closed loop what 10.3 proves in isolation; add X+Z closed-loop
coverage; then the full suite.

### 10.6 Per-cycle QP cost: promoting Z makes the QP ~4.9x cheaper when it matters

Predicted in §10.1 and now **measured**. Promoting Z from a bounded axis to a tracked one deletes
its two HOCBF rows, so the inequality set drops from five rows (Y max/min, Z max/min,
manipulability) to three (Y max/min, manipulability). The task Jacobian gains a row at the same
time, so a saving was not guaranteed a priori.

Isolated measurement (`scratchpad/qp_cost_xz.py`, one fixed state at ARM_Q0, 400 `compute()`
calls, `qd = 0.05` so the manipulability CBF's curvature term is really paid for, BLAS threads
pinned to 1 on an otherwise-quiet host):

| row set | walls | ineq rows | active corridor rows | mean QP | p95 QP |
|---|---|---|---|---|---|
| X-only | tight | 5 | 2 | **285.222 ms** | 381.296 ms |
| X+Z | tight | 3 | 1 | **58.761 ms** | 97.164 ms |
| X-only | wide | 5 | 0 | 0.459 ms | 0.479 ms |
| X+Z | wide | 3 | 0 | 0.342 ms | 0.349 ms |

**X+Z is 4.85x cheaper with the corridor rows genuinely ACTIVE** (285.222 → 58.761 ms) and 1.34x
cheaper when they are present but inactive. The active case is the one a real-time budget has to
cover, and it is where nearly all of the saving is: the expense is the dual bisection, and X+Z
runs it over one active row instead of two.

This does **not** change the scope verdict. Both row sets remain far outside the 2.00 ms 500 Hz
`direct_torque` budget with rows active (`over_budget_fraction` 1.00 for both), so the controller
stays sim-only. Promoting Z narrows the gap by ~4.9x; it does not close it.

> The in-rollout `qp_mean_ms` figures in `scratchpad/xz_matrix.json` (25–180 ms, and *higher* for
> X-only than X+Z in some cells and lower in others) are **load artifacts**, not costs. They were
> taken while a 24-way parallel search was running with BLAS threads unpinned. A single rollout
> re-timed with threads pinned on a quiet host drops from ~166 s to 33 s. Do not quote them.

### 10.7 The generalization is bit-identically inert for the X-only row set — in closed loop

§10.3 proves bit-identity at one synthetic state. This proves it over five full 1250-step
**closed-loop** rollouts, where any residual difference would compound through the changing
Jacobian, friction, the torque clip and the guards.

The baseline is `scratchpad/after_matrix.json`'s `AFTER` block, generated 2026-08-13 21:33 —
before the generalization landed (`controller.py` mtime 22:27; `scratchpad/controller_SHIPPED.py`
is that pre-generalization file, 504 lines, with no `task_axis_rows` in it). The X-only config it
ran against has not been touched since (mtime 21:24), so re-running the same five cells isolates
the refactor and nothing else.

`scratchpad/xonly_inertness.py`, comparing at **full precision** rather than the 4 decimals the
log happened to print, over all 19 recorded fields per cell (tracking, orientation, Y/Z drift,
`|qd|`, torque, manipulability, guard reason, corridor/CBF/infeasible step counts, pan torque):

**95 / 95 fields bit-identical across all 5 cells** — `steps`, `achieved_delta_m`,
`tracking_fraction`, `ori_max_rad`, `max_abs_y_drift_m`, `max_abs_z_drift_m`, `max_abs_qd_radps`,
`guard_reason`, everything. Not `allclose`: exact equality.

This is the single most important regression check in the change, because every closed-loop
number in §4 and §5 was measured against the pre-generalization code path.

## 11. Tool-face task frame, and the cart-speed envelope at ARM_Q0 (2026-08-14)

### 11.1 The frame, verified off the compiled model

`attachment_site`'s rotation at ARM_Q0, columns = tool axes in world:

| axis | world direction | angle from vertical |
|---|---|---|
| tool X | `[-0.0941, -0.0851, +0.9919]` | **7.29 deg** |
| tool Y | `[+0.7103, +0.6924, +0.1268]` | **82.72 deg** |
| tool Z | `[-0.6976, +0.7164, -0.0047]` | **89.73 deg** |

Pumping quality `kappa = |drive_axis x hinge|` with the hinge taken as tool Z:
**world X = 0.7165, tool Y = 1.0000, tool X = 1.0000, tool Z = 0.0000** — so tool Y delivers
**1.3958x** the pendulum torque per unit cart motion, reproducing the 40% claim exactly. The
blocked axis in the tool mapping (tool Z) is the one that provably does nothing to the pendulum.

### 11.2 The "106x" figure reconciled — and correctly scoped

The `|qd| <= 3.0` kinematic ceiling `v_max = 3.0 / max|pinv(J_rows) e_dir|` reproduces the
quoted number **exactly**, but only for a **full 6-D task**:

| task | joints | v_max along world X |
|---|---|---|
| full 6-D (X,Y,Z + orientation) | all six | **0.00960 m/s** |
| full 6-D | pan locked | 1.38372 m/s |
| **X+Z tracked, Y free, orientation HELD** (the configured task) | pan locked | **0.7093 m/s** |
| X+Z tracked, Y free, orientation FREE | pan locked | 0.7278 m/s |

**The 106x unlock belongs to a full 6-D task, not to the configured one.** Once Y is left free
the pseudo-inverse already has a free direction to exploit, so the ceiling is 0.709 m/s and
relaxing orientation on top buys only **1.03x** (pan locked) or 1.8x (all six) — not 106x. Most
of that unlock was already banked by freeing Y. Ceilings along tool Y are 1.0084 (orientation
held) / 1.0283 m/s (free), i.e. **1.41x the world-X ceiling**, matching the kappa argument.

### 11.3 `task_frame` — implementation, and the live/frozen question settled by measurement

`task_frame: "world" | "tool"` (default `"world"`) pre-rotates the Jacobian's three POSITION rows
by `R_tool^T` before row selection; orientation rows are untouched. Every position quantity that
feeds a row — `p`, `v`, `p0`, the targets, the corridor bounds and the corridor row's `J` row —
is rotated by the SAME `R` as the row it drives. `task_frame_update: "live" | "frozen" |
"hybrid"` selects which `R` the tracked rows and the corridor row each see; it is rejected at
parse time if set while `task_frame` is `"world"`, since it would provably do nothing.

**The world path is skipped entirely (`R is None`, not `eye(3)`), so it is byte-identical, not
merely close** — re-verified: **95/95 fields bit-identical** across the five closed-loop cells
after this change, and the 89 unit tests still pass.

**Measured answer to the live-vs-frozen question: it does not matter at this displacement.** Over
a representative move (dx=+0.06 m, 1.5 s move + 1 s hold) the tool rotates only **2.2947 deg**
total (tool-Y swing 2.2938 deg), and all three modes return **identical** results to four
decimals (tracking 0.4050, orientation 0.0146 rad, `|qd|` 0.134, max Z 0.0048 m, zero
corridor-active steps, no trip). Frozen is therefore the sane default. This finding is scoped to
~2 deg of rotation; under relaxed orientation at higher speed the modes could still diverge.

### 11.4 HEADLINE — max clean cart speed, and the speed-vs-orientation tradeoff

140 rollouts, dx = 0.10 m, move duration swept 2.0 s down to 0.18 s (min-jerk peak command
0.094 → 1.042 m/s). "Clean" = no guard trip. `w` scales `(kp_rot, kd_rot)`; `kp_rot` **is** the
orientation weight in the QP Hessian, so `w = 0` is exactly "orientation relaxed".

| frame | pan | w=0.0 | w=0.1 | w=0.5 | w=1.0 | w=3.0 |
|---|---|---|---|---|---|---|
| world | locked `[0]` | — | — | — | 0.297 @ 0.95° | — |
| world | free `[]` | **0.668 @ 13.84°** | 0.655 @ 13.40° | 0.573 @ 8.89° | 0.533 @ 7.40° | 0.466 @ 5.04° |
| tool | locked `[0]` | **0.497 @ 4.84°** | 0.489 @ 3.27° | 0.480 @ 1.41° | 0.473 @ 0.89° | 0.476 @ **0.55°** |
| tool | free `[]` | — | — | — | 0.102 @ 4.36° | 0.478 @ 1.37° |

(m/s @ orientation error at that speed; "—" = no clean cell at any speed.)

Three results that decide the design:

1. **Fastest overall is world + pan free + orientation fully relaxed: 0.6683 m/s — but it costs
   13.84 deg of orientation error.** The world-frame tradeoff is steep and monotone: 0.466 → 0.668
   m/s (+43%) for 5.04 → 13.84 deg (2.7x).
2. **The tool frame is nearly flat in `w`** — 0.473–0.497 m/s across the entire weight sweep,
   only 5% spread. At *matched* orientation error (~5 deg) the tool frame is **faster** than the
   world frame (0.497 vs 0.466 m/s) while holding the pan pin. At w=3.0 it still does 0.476 m/s
   at **0.55 deg**, roughly 9x tighter orientation than the world frame's best-case comparable.
3. **`task_excluded_joints` is frame-dependent, and this is the decision the numbers exist for.**
   In the **world** frame locking `shoulder_pan` costs **2.25x** speed (0.297 vs 0.668 m/s). In
   the **tool** frame it costs *nothing* — pan locked (0.4973) is marginally **better** than pan
   free (0.4779), 0.96x. **The tool frame lets the wall-clearance guarantee be kept for free**,
   which in the world frame is an expensive safety-vs-speed trade.

**Which guard binds first as speed rises**: world + pan locked → orientation guard (low `w`) then
`z_drift`; world + pan free → orientation at `w=0`, `axis_error` growth at `w=3`; tool + pan free
→ `y_drift` at every low `w`; **tool + pan locked → nothing trips anywhere in the sweep.**

> **Frame mismatch worth flagging, not yet fixed.** `ImpedanceSafetyMonitor`'s drift guards are
> **world-frame**, so a tool-frame corridor bounds tool Z while the guard measures world Y/Z.
> That mismatch is exactly why `tool + pan free` trips `y_drift`: the controller is not bounding
> the quantity the guard is watching. The guards were not weakened to work around this.

**Caveat, load-bearing:** the tool-frame gains are NOT re-derived from the tool row set's own
row-restricted `Λ` — `kp_y/kd_y` were simply promoted to task values and `kp_z/kd_z` demoted to
the bias, mirroring the world config. That is the same "gains do not transfer" step §10.4 did for
X+Z, and it is **outstanding** for this row set; the tool-frame speeds above are therefore a
floor, not a tuned result.

## 12. LOCKED CONFIGURATION — ARM_Q0, live tool frame, tool-Y transport (2026-08-14)

Fixed by decision, not by search: ARM_Q0 exactly as-is; `task_frame: "tool"` with
`task_frame_update: "live"`; tracked = tool X + tool Y (`task_axis_rows: [0, 1]`); corridor =
tool Z (`corridor_axis_rows: [2]`), which IS the hinge axis so motion along it does nothing to
the pole; `task_excluded_joints: [0]`; **all safety guards enforced, unmodified**. The
world-frame arm is retired; §10–§11's world numbers are kept only as the historical baseline.

### 12.1 Max cart speed along tool Y, and the speed-vs-orientation tradeoff

42 rollouts, dx = 0.07 m commanded along the world direction of tool Y, move duration 1.0 s →
0.12 s (min-jerk peak command 0.131 → 1.094 m/s). `w` scales `(kp_rot, kd_rot)` off
(149.814710, 100.220113); `kp_rot` **is** the orientation weight in the QP Hessian.

| w | kp_rot | max clean tool-Y speed | at T | orientation err | \|qd\| | first guard |
|---|---|---|---|---|---|---|
| 0.00 | 0.00 | **0.4885 m/s** | 0.16 s | 1.193 deg | 1.444 | none |
| 0.05 | 7.49 | 0.4780 | 0.16 | 1.424 | 1.407 | none |
| 0.20 | 29.96 | 0.4860 | 0.16 | 1.113 | 1.430 | none |
| 0.50 | 74.91 | 0.4823 | 0.16 | 0.739 | 1.429 | none |
| 1.00 | 149.81 | 0.4767 | 0.16 | 0.463 | 1.408 | none |
| 2.00 | 299.63 | 0.4685 | 0.16 | **0.257** | 1.379 | none |

**ZERO guard trips in all 42 runs.** `shoulder_pan` never moved more than **0.0145 deg** and the
pin reported no violations, so the wall-clearance guarantee held throughout.

**The headline result is that in the tool frame there is essentially no speed-vs-orientation
tradeoff left.** Speed varies only **4.3%** (0.4685 → 0.4885 m/s) across the whole weight sweep
while orientation error varies **4.6x** (0.257 → 1.193 deg). Relaxing orientation buys almost
nothing, and holding it costs almost nothing — so the right choice is to HOLD it. This is a
qualitative change from the world frame, where relaxing orientation bought +43% speed at 2.7x the
orientation error (§11.4). Choosing the frame, not relaxing orientation, is what bought the
performance.

**What actually binds is closed-loop bandwidth, not a guard.** Speed saturates at T = 0.16 s:
commanding T = 0.12 s (peak 1.094 m/s) yields a *lower* achieved speed (0.424–0.461 m/s) than
T = 0.16 s. Tracking sits at 0.78–0.81 throughout. Achieved 0.4885 m/s is **47.5%** of the
`|qd| <= 3.0` kinematic ceiling along tool Y (1.0283 m/s, pan locked, orientation free).

> **Displacement is capped by a frame mismatch, and the guards were NOT relaxed to hide it.**
> `ImpedanceSafetyMonitor`'s drift guards are WORLD-frame, measuring distance from the world-X
> axis through the start pose. Tool Y is `[0.7103, 0.6924, 0.1268]` in world, so 0.7039 of every
> metre along it registers as "orthogonal drift" — capping a clean tool-Y move at ~0.085 m against
> the 0.06 m tolerance, **regardless of the controller**. dx was therefore held at 0.07 m
> (0.0493 m of orthogonal drift, ~18% margin) and speed pushed via duration instead. Expressing
> the drift guards in the task frame is the correct fix and is NOT done here: it lives in
> `controller_core/safety.py`, which the real-hardware lane shares.

### 12.2 The `kp_rot` bound, resolved: an INTERIOR optimum at ~300

The old search pinned at `kp_rot = 149.81` on a 150.0 bound *and* stopped on its iteration cap.
Re-measured in the locked config with the bound widened **64x** (to 9588), at the speed-optimal
durations:

| kp_rot | speed @ T=0.16 | orientation | verdict |
|---|---|---|---|
| 149.81 | 0.4767 | 0.4634 deg | the old pinned value |
| **299.63** | **0.4685** | **0.2572 deg** | **optimum — best orientation at full speed, no trips** |
| 599.26 | 0.4778 | 0.9525 deg | orientation degrades |
| 1198.52 | 0.4129 | 0.6503 deg | **starts tripping** (`axis_error` growth) |
| 4794.07 | 0.2488 | 0.2110 deg | trips `y_drift`, speed collapsing |
| 9588.14 | 0.0689 | 0.0660 deg | degenerate: tightest orientation because the arm barely moves |

**The optimum moved well inside the widened bound — `kp_rot ≈ 300`, `kd_rot ≈ 200`, about 2x the
old pin.** So the 150 bound genuinely was binding, and the answer is the first of the two cases:
an interior optimum, not a runaway. It is *not* monotone — past ~600 orientation degrades, past
~1200 guards start tripping, and the apparently-excellent 0.066 deg at kp_rot 9588 is the
degenerate regime where the arm is too stiff to move (0.0689 m/s, tracking 0.20). Reporting only
the orientation number there would have been a trap.

### 12.3 Test status

`tests/unit/test_x_task_yz_corridor_qp.py`: **106 passed** (+17 new for `task_frame`: parser
validation, the world-default skip, the `R_tool^T` rotation asserted against a hand-computed
Hessian, reset snapshotting, all three update modes, and tool-frame corridor bounds).

`tests/mujoco/test_x_task_yz_corridor_qp_closed_loop.py` + unit: **138 passed, 1 failed.** The
failure is `test_tracking_z_holds_z_tighter_than_bounding_it_where_the_move_completes[-0.12]`
(X+Z 0.03532 m vs X-only 0.03085 m). It is a **world-frame X+Z** claim, on the arm this section
retires, and it is a genuine measurement change rather than a flake — left failing and reported
rather than deleted or loosened, since deciding the fate of the world-frame tests is a scope call.
