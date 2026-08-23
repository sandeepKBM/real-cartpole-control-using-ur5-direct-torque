# OLD_POSE transport envelope — measured, not assumed from conditioning alone

**Date:** 2026-08-14
**Scope:** simulation only (`assets/ur5e_torque/scene.xml`). No `hardware/*.py` touched, no
`controller_core/x_task_yz_corridor_qp/` touched, no file under `config/` modified (two
existing configs were *used*, not changed: `config/ur5e_mujoco_torque_osc_tuned.yaml` and
`config/ur5e_mujoco_torque_osc_tuned_manipulability_cbf.yaml`).
**Question:** this repo's entire X-transport corpus was built at `ARM_Q0`, a pose that sits
essentially on the UR5e wrist singularity. `OLD_POSE` is 201x better conditioned. Does that
conditioning advantage actually translate into a bigger, cleaner transport envelope, or is it
a number that doesn't cash out?

**Headline answer: no, not for this controller, without qualification.** OLD_POSE's clean
symmetric (`+X` **and** `-X`) transport window is **0.10-0.15 m**, roughly **2-4x larger**
than the ~0.04-0.06 m ARM_Q0 supports — genuinely better. But the picture is not "smaller
displacements are strictly easier": at OLD_POSE, **every canonical-grid displacement (0.01-0.06 m)
fails in both directions**, for a reason that has nothing to do with conditioning (steady-state
Coulomb-friction undershoot, the same mechanism already documented elsewhere in this repo), and
above 0.15 m a **genuine, sharp, direction-asymmetric Z-drift failure** appears in `+X` only.
The manipulability CBF is a measured, complete no-op here (see §4).

---

## 1. The two poses, verified

```
ARM_Q0    = [-2.3688, -2.1801, -1.8838, -0.7962, 0.004714693, 0.0206]
            cond(J) = 1395.76   sigma_min = 1.4853e-03   wrist_2 = +0.270 deg

OLD_POSE  = [0.0, -1.091985784398452, 2.0935362786892546, -2.7685637962327356,
             1.5620693866337145, 0.0]
            cond(J) =    6.928   sigma_min = 2.6026e-01   wrist_2 = +89.50 deg
```

Re-verified directly from `simulation.ur5e_mujoco_torque.load_model` /
`make_mujoco_jacobian_fn` on the real model — matches the numbers this task started from
exactly (both to the digits given).

## 2. Method

Reused this repo's existing rollout/scoring pipeline, not a new one:
- `tools/ur5e_move_hold_transport.py` (subprocesses `tools/ur5e_mujoco_torque_experiments.py
  --mode controller-rollout`, the one file in this repo with the per-step loop) for every
  run, with `--start-q-rad` set to `OLD_POSE`.
- `transport_metrics.compute_valid_move_hold_metrics` (tolerance = `max(5mm, 25% of target)`,
  the real pass/fail this repo uses everywhere) is what `tools/ur5e_move_hold_transport.py`
  calls internally — **not reimplemented**, and confirmed by reading the source before use.
- Grids follow `tools/ur5e_pose_sweep_transport.py`'s own canonical
  `CATEGORY_GRIDS` (the closest thing this repo has to a single "4-category rigor sweep"
  definition) for `canonical_grid` and `torque_scale_robustness`; `long_holds` and
  `large_displacements` were adapted (see §7 "what wasn't tested" for the exact deltas from
  that script's grids and why).
- **Both `+X` and `-X` swept for every single cell** (AGENTS.md §7's standing rule), reported
  separately below, never averaged.
- Two controller arms, config unmodified in both cases:
  `config/ur5e_mujoco_torque_osc_tuned.yaml` ("osc") and
  `config/ur5e_mujoco_torque_osc_tuned_manipulability_cbf.yaml` ("osc+cbf" — the same tuned
  config plus only the CBF block, per that file's own header).
- `min mu` / `max cond(J)` per run were computed from each run's own `trace.jsonl`
  `jacobian_pre_step` field (already logged by the experiment runner for every cycle) via
  `controller_core.manipulability_cbf.manipulability()` — the same helper
  `tools/diagnostics/manipulability_cbf_sim_check.py` uses. This is a diagnostic reduction
  over already-logged raw Jacobians, not new scoring logic.

124 closed-loop runs total (62 per arm): `canonical_grid` 16, `long_holds` 8,
`large_displacements` 24, `torque_scale_robustness` 14.

---

## 3. Results

`osc` and `osc+cbf` are **byte-identical** on every single one of the 62 matched runs (see
§4) — one table below per category is sufficient for both arms.

### 3.1 canonical_grid (dx in {0.01,0.02,0.03,0.04} m x2 directions, hold {1,2} s, move 1.0 s)

**0/16 pass, both arms.** Every cell fails via `move_phase_target_tracking` /
`hold_phase_target_tracking` — the controller never saturates, never trips a safety guard, and
runs to `duration_complete` every time; it simply doesn't reach the target.

| dx | dir | hold_s | track_frac | max_Y_m | max_Z_m | max_orient_rad | min_mu | max_cond | max_qd | max_tau_Nm | PASS |
|---|---|---|---|---|---|---|---|---|---|---|---|
| -0.040 | -X | 1.0 | 0.561 | 0.0004 | 0.0089 | 0.0460 | 7.292e-02 | 6.934e+00 | 0.203 | 26.61 | fail |
| -0.040 | -X | 2.0 | 0.561 | 0.0007 | 0.0089 | 0.0460 | 7.292e-02 | 6.944e+00 | 0.203 | 26.61 | fail |
| -0.030 | -X | 1.0 | 0.447 | 0.0004 | 0.0087 | 0.0343 | 7.292e-02 | 6.928e+00 | 0.137 | 26.61 | fail |
| -0.030 | -X | 2.0 | 0.447 | 0.0006 | 0.0087 | 0.0343 | 7.292e-02 | 6.928e+00 | 0.137 | 26.61 | fail |
| -0.020 | -X | 1.0 | 0.269 | 0.0003 | 0.0045 | 0.0143 | 7.292e-02 | 6.928e+00 | 0.053 | 26.61 | fail |
| -0.020 | -X | 2.0 | 0.269 | 0.0005 | 0.0045 | 0.0143 | 7.292e-02 | 6.928e+00 | 0.053 | 26.61 | fail |
| -0.010 | -X | 1.0 | 0.220 | 0.0002 | 0.0016 | 0.0049 | 7.292e-02 | 6.928e+00 | 0.013 | 26.61 | fail |
| -0.010 | -X | 2.0 | 0.220 | 0.0003 | 0.0022 | 0.0071 | 7.292e-02 | 6.928e+00 | 0.013 | 26.61 | fail |
| +0.010 | +X | 1.0 | 0.222 | 0.0002 | 0.0016 | 0.0050 | 7.137e-02 | 6.954e+00 | 0.014 | 28.91 | fail |
| +0.010 | +X | 2.0 | 0.222 | 0.0003 | 0.0022 | 0.0074 | 7.089e-02 | 6.963e+00 | 0.014 | 28.91 | fail |
| +0.020 | +X | 1.0 | 0.280 | 0.0003 | 0.0047 | 0.0151 | 6.949e-02 | 7.001e+00 | 0.055 | 31.18 | fail |
| +0.020 | +X | 2.0 | 0.280 | 0.0005 | 0.0047 | 0.0151 | 6.862e-02 | 7.028e+00 | 0.055 | 31.18 | fail |
| +0.030 | +X | 1.0 | 0.459 | 0.0003 | 0.0111 | 0.0411 | 6.651e-02 | 7.113e+00 | 0.141 | 32.66 | fail |
| +0.030 | +X | 2.0 | 0.459 | 0.0006 | 0.0111 | 0.0411 | 6.561e-02 | 7.153e+00 | 0.141 | 32.66 | fail |
| +0.040 | +X | 1.0 | 0.578 | 0.0003 | 0.0124 | 0.0558 | 6.364e-02 | 7.253e+00 | 0.209 | 32.91 | fail |
| +0.040 | +X | 2.0 | 0.578 | 0.0006 | 0.0124 | 0.0558 | 6.275e-02 | 7.302e+00 | 0.209 | 32.91 | fail |

### 3.2 long_holds (dx in {0.03,0.06} m x2 directions, hold {10,30} s, move 1.0 s)

**0/8 pass, both arms.** Note `track_frac` is identical at hold=10 and hold=30 for every dx —
the shortfall is a **non-decaying steady-state offset**, not an incomplete-move artifact.

| dx | dir | hold_s | track_frac | max_Y_m | max_Z_m | max_orient_rad | min_mu | max_cond | max_qd | max_tau_Nm | PASS |
|---|---|---|---|---|---|---|---|---|---|---|---|
| -0.060 | -X | 10.0 | 0.674 | 0.0021 | 0.0093 | 0.0685 | 7.292e-02 | 7.047e+00 | 0.319 | 29.54 | fail |
| -0.060 | -X | 30.0 | 0.674 | 0.0030 | 0.0093 | 0.0685 | 7.292e-02 | 7.047e+00 | 0.319 | 30.71 | fail |
| -0.030 | -X | 10.0 | 0.447 | 0.0015 | 0.0087 | 0.0343 | 7.292e-02 | 6.928e+00 | 0.137 | 27.27 | fail |
| -0.030 | -X | 30.0 | 0.447 | 0.0018 | 0.0087 | 0.0343 | 7.292e-02 | 6.928e+00 | 0.137 | 28.56 | fail |
| +0.030 | +X | 10.0 | 0.459 | 0.0014 | 0.0111 | 0.0411 | 6.466e-02 | 7.192e+00 | 0.141 | 32.66 | fail |
| +0.030 | +X | 30.0 | 0.459 | 0.0018 | 0.0111 | 0.0411 | 6.466e-02 | 7.192e+00 | 0.141 | 32.66 | fail |
| +0.060 | +X | 10.0 | 0.698 | 0.0017 | 0.0144 | 0.0826 | 5.627e-02 | 7.730e+00 | 0.327 | 33.23 | fail |
| +0.060 | +X | 30.0 | 0.698 | 0.0027 | 0.0144 | 0.0826 | 5.627e-02 | 7.730e+00 | 0.327 | 33.23 | fail |

### 3.3 large_displacements (dx up to 0.30 m x2 directions, hold {1,2} s, move 1.0 s)

**14/24 pass, both arms** — and the pattern is the interesting part (see §5).
`guard` column shows the actual `termination_reason` for the failing cells; `duration_complete`
means it ran the full window and still failed on tracking/tolerance, not a guard trip.

| dx | dir | hold_s | track_frac | max_Y_m | max_Z_m | max_orient_rad | min_mu | max_cond | max_qd | max_tau_Nm | PASS | guard (trip time) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| -0.300 | -X | 1.0 | 0.810 | 0.0242 | 0.0095 | 0.1961 | 7.292e-02 | 1.023e+01 | 1.639 | 40.55 | **PASS** | duration_complete |
| -0.300 | -X | 2.0 | 0.810 | 0.0242 | 0.0095 | 0.1961 | 7.292e-02 | 1.033e+01 | 1.639 | 40.59 | **PASS** | duration_complete |
| -0.250 | -X | 1.0 | 0.817 | 0.0178 | 0.0094 | 0.1741 | 7.292e-02 | 9.103e+00 | 1.346 | 37.31 | **PASS** | duration_complete |
| -0.250 | -X | 2.0 | 0.817 | 0.0178 | 0.0094 | 0.1741 | 7.292e-02 | 9.180e+00 | 1.346 | 37.56 | **PASS** | duration_complete |
| -0.200 | -X | 1.0 | 0.816 | 0.0114 | 0.0093 | 0.1502 | 7.292e-02 | 8.284e+00 | 1.066 | 33.95 | **PASS** | duration_complete |
| -0.200 | -X | 2.0 | 0.816 | 0.0114 | 0.0093 | 0.1502 | 7.292e-02 | 8.341e+00 | 1.066 | 34.54 | **PASS** | duration_complete |
| -0.150 | -X | 1.0 | 0.802 | 0.0052 | 0.0092 | 0.1254 | 7.292e-02 | 7.683e+00 | 0.797 | 30.61 | **PASS** | duration_complete |
| -0.150 | -X | 2.0 | 0.802 | 0.0052 | 0.0092 | 0.1254 | 7.292e-02 | 7.724e+00 | 0.797 | 31.56 | **PASS** | duration_complete |
| -0.100 | -X | 1.0 | 0.762 | 0.0006 | 0.0092 | 0.0975 | 7.292e-02 | 7.249e+00 | 0.533 | 27.49 | **PASS** | duration_complete |
| -0.100 | -X | 2.0 | 0.762 | 0.0011 | 0.0092 | 0.0975 | 7.292e-02 | 7.276e+00 | 0.533 | 28.46 | **PASS** | duration_complete |
| -0.050 | -X | 1.0 | 0.629 | 0.0004 | 0.0091 | 0.0579 | 7.292e-02 | 6.967e+00 | 0.264 | 26.61 | fail | duration_complete |
| -0.050 | -X | 2.0 | 0.629 | 0.0008 | 0.0091 | 0.0579 | 7.292e-02 | 6.980e+00 | 0.264 | 26.61 | fail | duration_complete |
| +0.050 | +X | 1.0 | 0.650 | 0.0003 | 0.0134 | 0.0698 | 6.080e-02 | 7.420e+00 | 0.270 | 33.09 | fail | duration_complete |
| +0.050 | +X | 2.0 | 0.650 | 0.0006 | 0.0134 | 0.0698 | 5.992e-02 | 7.477e+00 | 0.270 | 33.09 | fail | duration_complete |
| +0.100 | +X | 1.0 | 0.796 | 0.0004 | 0.0180 | 0.1248 | 4.738e-02 | 8.530e+00 | 0.537 | 33.68 | **PASS** | duration_complete |
| +0.100 | +X | 2.0 | 0.796 | 0.0007 | 0.0180 | 0.1248 | 4.654e-02 | 8.619e+00 | 0.537 | 33.68 | **PASS** | duration_complete |
| +0.150 | +X | 1.0 | 0.847 | 0.0023 | 0.0231 | 0.1718 | 3.528e-02 | 1.013e+01 | 0.787 | 34.14 | **PASS** | duration_complete |
| +0.150 | +X | 2.0 | 0.847 | 0.0023 | 0.0231 | 0.1718 | 3.451e-02 | 1.026e+01 | 0.787 | 34.14 | **PASS** | duration_complete |
| +0.200 | +X | 1.0 | 0.873 | 0.0080 | 0.0292 | 0.2187 | 2.578e-02 | 1.219e+01 | 1.021 | 34.54 | fail | **`\|Z-Z0\|>0.03m`** @ 1.236s |
| +0.200 | +X | 2.0 | 0.873 | 0.0080 | 0.0292 | 0.2187 | 2.578e-02 | 1.219e+01 | 1.021 | 34.54 | fail | **`\|Z-Z0\|>0.03m`** @ 1.236s |
| +0.250 | +X | 1.0 | 0.793 | 0.0130 | 0.0300 | 0.2265 | 2.226e-02 | 1.331e+01 | 1.250 | 34.91 | fail | **`\|Z-Z0\|>0.03m`** @ 0.792s |
| +0.250 | +X | 2.0 | 0.793 | 0.0130 | 0.0300 | 0.2265 | 2.226e-02 | 1.331e+01 | 1.250 | 34.91 | fail | **`\|Z-Z0\|>0.03m`** @ 0.792s |
| +0.300 | +X | 1.0 | 0.696 | 0.0152 | 0.0300 | 0.2246 | 2.070e-02 | 1.390e+01 | 1.472 | 35.27 | fail | **`\|Z-Z0\|>0.03m`** @ 0.702s |
| +0.300 | +X | 2.0 | 0.696 | 0.0152 | 0.0300 | 0.2246 | 2.070e-02 | 1.390e+01 | 1.472 | 35.27 | fail | **`\|Z-Z0\|>0.03m`** @ 0.702s |

### 3.4 torque_scale_robustness (dx = +-0.04 m, hold 2.0 s, torque_limit_scale 0.10-1.00)

**0/14 pass, both arms.** At the tuned config's full gains, `scale >= 0.25` is limited purely
by the same friction undershoot as §3.1/§3.2 (never a guard). At `scale = 0.10` the arm cannot
generate enough torque fast enough and **both directions** trip the identical `|Z-Z0|>0.03 m`
guard within 0.18 s — a real, symmetric torque-budget failure, not a direction effect.

| dx | dir | scale | track_frac | max_Y_m | max_Z_m | max_orient_rad | min_mu | max_cond | max_qd | max_tau_Nm | PASS | guard (trip time) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| -0.040 | -X | 0.10 | 0.234 | 0.0001 | 0.0300 | 0.0587 | 7.292e-02 | 6.946e+00 | 0.693 | 53.98 | fail | `\|Z-Z0\|>0.03m` @ 0.182s |
| -0.040 | -X | 0.25 | 0.561 | 0.0007 | 0.0089 | 0.0460 | 7.292e-02 | 6.944e+00 | 0.203 | 26.61 | fail | duration_complete |
| -0.040 | -X | 0.40-1.00 | 0.561 | 0.0007 | 0.0089 | 0.0460 | 7.292e-02 | 6.944e+00 | 0.203 | 26.61 | fail | duration_complete |
| +0.040 | +X | 0.10 | -0.226 | 0.0002 | 0.0301 | 0.0583 | 7.292e-02 | 6.946e+00 | 0.696 | 55.74 | fail | `\|Z-Z0\|>0.03m` @ 0.182s |
| +0.040 | +X | 0.25-1.00 | 0.578 | 0.0006 | 0.0124 | 0.0558 | 6.275e-02 | 7.302e+00 | 0.209 | 32.91 | fail | duration_complete |

(`-X` and `+X` rows at scale 0.40/0.55/0.70/0.85/1.00 are each 5 byte-identical rows,
collapsed here; full per-row data in the raw sweep output, see §8.)

---

## 4. Manipulability CBF at OLD_POSE: a measured, complete no-op

`config/ur5e_mujoco_torque_osc_tuned_manipulability_cbf.yaml` sets `manipulability_cbf_epsilon
= 1.0e-3`. Across all 62 matched cells, `min_mu` never dropped below **2.070e-2** — **20.7x**
the epsilon — and `max cond(J)` never exceeded **13.90** (vs. `ARM_Q0`'s 1395.76). Diffed the
two arms field-by-field (`tracking_frac`, `max_Y`, `max_Z`, `max_orient`, `max_qd`, `max_tau`,
`valid_move_and_hold`) across all 62 matched configurations: **zero differences, anywhere.**
`osc` and `osc+cbf` pass/fail identically in every one of the four category tables above.

This is exactly the "at cond=6.93 it may be a complete no-op" hypothesis in the task brief,
confirmed rather than assumed: OLD_POSE never gets close enough to the singular set, in any
tested cell, for the CBF's constraint row to ever bind.

---

## 5. Two surprises

### 5.1 Small displacements fail, medium ones pass, large ones fail differently — not monotonic

Naively one expects pass/fail to be monotonic in `|dx|`. It is not, and the reason is visible
directly in the data: `move_hold`'s tolerance is `max(5mm, 25% of |target|)`, and OLD_POSE's
`osc` config (no `friction_feedforward`) has a **roughly dx-independent absolute tracking
shortfall** at this pose (`achieved != target` by a few cm regardless of target size, visible
in `move_phase_achieved_x_delta_m` across every cell). For `dx <= 0.06 m` that absolute
shortfall is 40-78% of the target — outside the 25% tolerance band, so every canonical/long-hold
cell fails on `..._target_tracking`. For `dx` in 0.10-0.15 m, the same absolute shortfall is
only 15-24% of a bigger target — inside tolerance, so those cells pass. Above `dx ~ 0.15-0.20 m`
a second, unrelated failure mode (§5.2) takes over. Confirmed this is friction, not a pose
defect: re-running the `canonical_grid` dx=0.02 cell with `--friction-multiplier 0.0` (i.e. the
plant's Coulomb/viscous joint friction disabled) recovers 97% tracking and a clean pass
(`move_hold_quality_score` 0.61, `duration_pass`/`safety_pass` both true) — see §6.

### 5.2 A sharp, direction-asymmetric Z-drift failure above ~0.15-0.20 m — real, not investigated further

`+X` fails via `|Z-Z0| > 0.03 m` at `dx=0.20/0.25/0.30 m` (guard trips inside the move phase,
0.70-1.24 s in), while `-X` at the identical magnitudes (`-0.20/-0.25/-0.30 m`) completes the
full duration cleanly, `max_Z` staying at ~0.0093-0.0095 m — roughly **3x under the guard**
versus `+X`'s 0.0292-0.0300 m, which sits **right at** the 0.03 m ceiling. `max_qd` and
`track_frac` are comparable between the two directions at matched `|dx|`, so this is not simply
"the fast direction is harder" — it is a genuine kinematic/dynamic Z-coupling asymmetry, the
same *family* of finding AGENTS.md §3 already documents for the directional-ceiling and -45°
Y-drift cases at other poses, now observed at OLD_POSE too. **Not root-caused here** — this
sweep's job was to measure the envelope, not to diagnose why; flagging it rather than guessing.

---

## 6. Supplementary check: is the small-dx failure the known, already-fixed friction gap?

Not part of the requested two-arm comparison, but cheap and directly relevant to interpreting
§3.1/§3.2/§5.1 correctly: reran `canonical_grid` at OLD_POSE with the existing
`config/ur5e_mujoco_torque_osc_tuned_friction_ff.yaml` (already-validated
`friction_feedforward` fix, used unmodified, same as the other two configs).

| dx | dir | hold | track_frac (friction_ff) | PASS |
|---|---|---|---|---|
| -0.04 / +0.04 | both | 1,2 | 0.858 / 0.871 | **PASS** |
| -0.03 / +0.03 | both | 1,2 | 0.833 / 0.855 | **PASS** |
| -0.02 / +0.02 | both | 1,2 | 0.663 / 0.714 | fail |
| -0.01 / +0.01 | both | 1,2 | 0.317 / 0.321 | fail |

**8/16 recovered** (0.03-0.04 m now pass symmetrically), up from 0/16 with the plain tuned
config. 0.01-0.02 m stay marginal even with the fix — plausibly the friction feedforward's own
`qd` deadband (documented elsewhere as tuned for a different pose family), not re-tuned here.
This says the canonical-grid failure at OLD_POSE is the *same* known friction gap this repo
already has a fix for, not something specific to OLD_POSE's kinematics — but it was run with
one config, one category, informational only, and is **not** part of this task's two required
arms.

---

## 7. What was tested, what wasn't

**Tested:** all four categories, both directions, both required arms (`osc`, `osc+cbf`),
124 real closed-loop MuJoCo runs, `torque_limit_scale` down to 0.10.

**Not tested / known gaps:**
- `long_holds` used `hold in {10, 30} s`, not the 4-point `{4,10,20,30}` grid
  `tools/ur5e_pose_sweep_transport.py` uses elsewhere, to keep total sweep wall-time bounded;
  the 10s/30s cells already show the steady-state (no drift between them), so a 4/20s point
  would very likely just interpolate, but this was not run.
- `large_displacements` used `dx up to 0.30 m` at `hold in {1,2}s` only (not the 4-point long
  holds); the 0.10-0.15 m "sweet spot" (§5.1) was **not** re-tested with long holds, so whether
  it stays clean at 10-30 s of hold is unverified, not assumed.
- `move_duration` was fixed at 1.0 s everywhere (matching `CATEGORY_GRIDS`'s canonical_grid /
  large_displacements / torque_scale_robustness choice) — not swept.
- §5.2's Z-drift asymmetry was measured, not root-caused (no joint-space/Jacobian trace
  instrumentation was added this pass).
- §6 is one supplementary category, one config, not a full second 4-category sweep.
- No real-hardware implications drawn or claimed — this is a sim-only conditioning/envelope
  measurement, and OLD_POSE is not validated as a real-hardware start pose by this work.
- `gravity_source=pinocchio` / `coriolis_feedforward=true` as set by both named configs (their
  own YAML, unmodified) were used as-is, not varied.

---

## 8. Raw data

- Per-category `summary.json`/`summary.csv`/`run_log.jsonl`/`per_run_traces/` for both arms:
  `/common/home/ss5772/.tmp/claude-1905239669/-common-users-ss5772-real-Cartpole/886470ab-6030-4821-89db-b8c0c4fe2cb5/scratchpad/old_pose_sweep/{osc,osc_cbf}/{canonical_grid,long_holds,large_displacements,torque_scale_robustness}/`
  (scratch space, not in git — regenerate with the commands in §9 if needed for follow-up).
- Combined, per-run table with derived `min_mu`/`max_cond`/`min_sigma_min`:
  `.../old_pose_sweep/aggregated_rows.json` (124 rows) and the aggregation script
  `.../old_pose_sweep/aggregate.py` used to build every table above.
- §6 supplementary run: `.../old_pose_sweep/osc_frictionff/canonical_grid/`.

## 9. Reproduction / rollback

Nothing under version control was changed except this doc — no rollback needed for the repo.
To reproduce a category (example: `osc`, `canonical_grid`):

```bash
cd /common/users/ss5772/real_Cartpole
python tools/ur5e_move_hold_transport.py \
  --config config/ur5e_mujoco_torque_osc_tuned.yaml \
  --output-root <out_dir> --seed 0 \
  --start-q-rad 0.0 -1.091985784398452 2.0935362786892546 -2.7685637962327356 \
                1.5620693866337145 0.0 \
  --target-x-deltas 0.01 0.02 0.03 0.04 -0.01 -0.02 -0.03 -0.04 \
  --move-durations 1.0 --hold-durations 1.0 2.0 --torque-limit-scales 1.0 --no-plot
```

Swap `--config` for `config/ur5e_mujoco_torque_osc_tuned_manipulability_cbf.yaml` for the CBF
arm; swap `--target-x-deltas`/`--hold-durations`/`--torque-limit-scales` per §3's category
headers to reproduce the other three categories.
