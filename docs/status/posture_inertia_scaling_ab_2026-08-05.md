# posture_inertia_scaling: tested, REGRESSES — 2026-08-05

**Verdict: hypothesis rejected. Do not enable `controller.posture_inertia_scaling`.**
Kept in the codebase as a default-off, zero-regression-when-off addition (same precedent as
`ki_y`, AGENTS.md section 3), so the negative result is preserved rather than re-derived.

**What was tested.** `tau_posture = M(q) @ (kp*(q_rest-q) - kd*qd)` instead of the plain
joint-space PD, applied before the nullspace projection — adapted from
ian-chuang/homestri-ur5e-rl's `JointPositionController`. Motivation was a real semantic
inconsistency: with `task_space_inertia_shaping` on, the TASK gains are acceleration gains while
the POSTURE gains were still torque gains, so identical numeric `kp_posture` meant different
physical things in the two terms and a given posture error commanded the same torque in
shoulder_lift as in wrist_3. Hypothesis was that this contributed to the -45deg / directional
posture-authority asymmetry.

**Result.** Regression on the canonical grid (9/24 -> 3/24 valid move-and-hold), no change
anywhere else, and **no improvement at the -45deg pose** — the case it was designed to address
(0/24 both arms). The six lost cells are hold-phase X-drift misses, **not** safety trips: no
guard fired first in any changed cell. Mechanism: the candidate holds
`hold_phase_final_x_error_m` inside tolerance but pushes
`hold_phase_x_drift_from_hold_start_m` across `hold_x_drift_tol` on the two marginal cells.

**The confound was checked, not waved away.** Because the flag changes the effective units of
`kp_posture`/`kd_posture`, a small sensitivity check scaled them by the pose's
`mean(diag(M(q))) = 0.8236` and one point below it (25/6 -> 20.6/4.9 -> 16.0/3.8). Neither
recovered a single cell (3/24 in both). That does not prove no retune could work — it is a
2-point check, not a search — but it removes the cheapest explanation.

**Caveat on absolute numbers, stated plainly:** the baseline's own canonical pass rate (9/24)
is well below the 8/8 this repo documents for the tuned configs, so this sweep is not running
at the exact validated operating point (grid, start pose, and the post-2026-07-31 friction
model all differ from the original validation). The A/B comparison remains valid — both arms
used identical grid, seed, and pose — but the absolute rates should not be read as a
re-validation of the baseline configs.

Sim-only. No hardware validation. Raw runs under `outputs/homestri_ab/` (gitignored).

---

# `posture_inertia_scaling` A/B report

## Setup
- Interpreter: `/common/users/ss5772/miniforge3/envs/mujoco_ur5e/bin/python3`
- Entry point: `tools/ur5e_move_hold_transport.py`
- Seed: `0` for every run
- Output namespace: `outputs/homestri_ab/20260806T011611Z_envfix`
- Canonical start-pose mean diag(M): `0.823588208061`
- Run-record note: every completed sim reported `outcome=success`; the transport scorer in `summary.json` is the pass/fail signal used below (`valid_move_and_hold`, `move_failure_reason`, `hold_failure_reason`).
- The first B-pair runs without explicit `--start-q-rad` were superseded; the report below uses only the explicit -45° reruns (`_startqneg45`).

Exact explicit -45° start q used for the corrected B runs: `[-0.7853981633974483, -0.8353981633974483, -1.2, -0.9853981633974483, 0.0, 0.0]`.

## Verdict
`posture_inertia_scaling` **regresses** overall.
- On the canonical active-origin pose, valid move-and-hold cells drop from `9/24` to `3/24`.
- The six lost cells are all hold-phase X-drift misses, not safety trips, and the small posture-gain retunes do not recover them.
- The large and negative A grids keep the same valid counts, but the corrected actual -45° pose does not improve at all, and the candidate adds a few extra hold-phase target-tracking misses on the large grid.

## Comparison
| Pose / grid | Base valid | Cand valid | Base move fails | Cand move fails | Base hold fails | Cand hold fails | Transport verdict |
| --- | --- | --- | --- | --- | --- | --- | --- |
| A canonical active-origin | 9/24 | 3/24 | 6/24 | 6/24 | 15/24 | 21/24 | regression |
| A large | 24/24 | 24/24 | 0/24 | 0/24 | 0/24 | 0/24 | no validity change |
| A negative | 12/12 | 12/12 | 0/12 | 0/12 | 0/12 | 0/12 | no validity change |
| B actual -45 canonical | 0/24 | 0/24 | 24/24 | 24/24 | 21/24 | 21/24 | no validity change |
| B actual -45 large | 0/24 | 0/24 | 24/24 | 24/24 | 18/24 | 21/24 | no validity change; 3 hold-subtype flips |

## Changed Cells
Cells are grouped when the three torque-limit scales behave identically. `First guard` is `none` in every changed cell below: no safety guard fired first; the differences are purely transport-scoring failures.

### Canonical active-origin validity flips
| Cell group | First guard | Base -> cand valid | Failure reason change | Base -> cand hold final x err | Base -> cand hold x drift | Base -> cand move final x err | Base -> cand max orientation | Base -> cand max abs Y | Base -> cand max abs qd | Relevant tolerance |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dx0p03, move=1 s, hold=1 s, scales=0.5 / 0.75 / 1 | none | True -> False | none -> hold_phase_target_tracking | 0.00131642 -> 0.00203139 | 0.00442002 -> 0.00457761 | 0.00573644 -> 0.006609 | 0.0323671 -> 0.0315163 | 1.01811e-05 -> 8.66311e-06 | 0.0528412 -> 0.0497275 | move_x_tol=0.0075, hold_x_drift_tol=0.0045 |
| dx0p04, move=1 s, hold=2 s, scales=0.5 / 0.75 / 1 | none | True -> False | none -> hold_phase_target_tracking | 0.000315536 -> 0.000641946 | 0.00557087 -> 0.00637678 | 0.00588641 -> 0.00701872 | 0.0432873 -> 0.0416679 | 1.60958e-05 -> 1.37369e-05 | 0.0746807 -> 0.0708502 | move_x_tol=0.01, hold_x_drift_tol=0.006 |

Mechanism note: the candidate keeps `hold_phase_final_x_error_m` inside the `move_x_tol` bound, but `hold_phase_x_drift_from_hold_start_m` crosses `hold_x_drift_tol` on the two marginal canonical cells.

### Actual -45° large-grid subtype flips
| Cell group | First guard | Base -> cand valid | Failure reason change | Base -> cand move final x err | Base -> cand hold final x err | Base -> cand hold x drift | Base -> cand max orientation | Base -> cand max abs Y | Base -> cand max abs qd | Relevant tolerance |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dx0p05, move=2 s, hold=1 s, scales=0.5 / 0.75 / 1 | none | False -> False | move_phase_target_tracking -> move_phase_target_tracking + hold_phase_target_tracking | 0.0165191 -> 0.0177655 | 0.012228 -> 0.0132486 | 0.00429106 -> 0.00451696 | 0.0666597 -> 0.0615717 | 0.0375966 -> 0.0365331 | 0.0619364 -> 0.0579924 | move_x_tol=0.0125, hold_x_drift_tol=0.0075 |

Mechanism note: on the proper -45° large grid the candidate does not change validity, but it does push `hold_phase_final_x_error_m` above `move_x_tol` on the `dx=0.05 m` cells, so `hold_phase_target_tracking` appears in addition to the already-failing move phase.

### No-validity-change grids
- A large: `24/24` valid for both baseline and candidate.
- A negative: `12/12` valid for both baseline and candidate.
- B actual -45 canonical: `0/24` valid for both baseline and candidate; all 24 cells fail move-phase target tracking.

## Sensitivity Check
The canonical active-origin start pose has `mean(diag(M(q))) = 0.823588208061`. I used that to run two small reductions of the posture gains on the same canonical grid.

| kp_posture / kd_posture | Valid / total | Move fails | Hold fails | Recovered? |
| --- | --- | --- | --- | --- |
| 20.6 / 4.9 | 3/24 | 6/24 | 21/24 | no recovery |
| 16.0 / 3.8 | 3/24 | 6/24 | 21/24 | no recovery |

Both retunes keep the same six canonical failures. That is, the candidate does not recover when posture gains are reduced to the mean-diagonal-scaled point or a smaller follow-up point.

## What I Could Not Verify
- No hardware validation was performed.
- The B YAML filenames are not sufficient by themselves to guarantee the rotated pose; the explicit `--start-q-rad` rerun was required to verify the actual -45° start pose.
- The reported sensitivity check is intentionally small; it shows no recovery, but it is not a full retune search.
- The candidate flag changes gain units, so the comparison is still confounded by unretuned posture gains. The small retune check did not remove that confound.

## Output Roots
- Canonical A/B runs and A negative: `outputs/homestri_ab/20260806T011611Z_envfix/`
- Corrected actual -45° runs: `outputs/homestri_ab/20260806T011611Z_envfix/*_startqneg45/`
- Sensitivity checks: `outputs/homestri_ab/20260806T011611Z_envfix/B2_cand_canonical_kp20p6/` and `outputs/homestri_ab/20260806T011611Z_envfix/B2_cand_canonical_kp16/`
