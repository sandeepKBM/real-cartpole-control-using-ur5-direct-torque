# Null-Space Bound Search Results (2026-08-06)

## Summary

Two automated differential-evolution gain searches for `CartesianVelocityController` using the newly introduced `ik_max_joint_deviation_rad` mechanism both produced results worse than a pre-existing reference baseline. The reference result is now unreproducible: the mechanism it depended on (`ik_posture_gain` and `ik_posture_activation_error_rad`) has been deleted from the codebase. Both new searches avoided exploiting the new mechanism tightly, matching a documented historical pattern for its soft-pull predecessor.

---

## The `ik_max_joint_deviation_rad` Mechanism

`ik_max_joint_deviation_rad` is a hard bound (radians) on how far the *redundant (null-space)* part of `compute_ik_seeded`'s per-iteration IK solve may wander from the posture reference `q_rest`, enforced exactly via null-space-basis coordinate clipping. See `controller_core/cartesian_velocity_controller/config.py` for the full docstring (lines 66–94).

It replaces two deleted fields:
- `ik_posture_gain`: soft, task-weight-relative quadratic pull toward `q_rest`
- `ik_posture_activation_joint_dev_rad`: gate threshold for that pull

The old soft mechanism was evaluated by searches but almost never converged to using it even when correctly scaled and gated. The hard bound requires no learning: the null-space coordinate is *provably* confined to `[-max_dev, max_dev]`, and task achievement `(J_task @ dq)` is *provably* unaffected by however aggressively it clips. See `velocity_gain_tuning/envs/velocity_transport_env.py` lines 84–138 for the full design history.

**IMPORTANT SCOPE LIMIT** (documented in both `config.py` and `velocity_transport_env.py`): This mechanism only helps failures that are genuinely *redundant (null-space)* phenomena — confirmed effective for `neg40_wrist2offset` and `neg45_wrist2offset`'s wrist_2 runaway. It *cannot* help `hanging_alpha_0_5`'s -X orientation failure: a direct linear-algebra check found that failure's orientation coupling lives in the *task (row) space itself*, not the null space. No null-space-projected mechanism can fix it without real X-tracking accuracy trade-offs.

---

## Exact Search Results

| Metric | Reference (gated_posture_widerdx) | nullspace_v2 | nullspace_v2_seeded108 |
|--------|-----------------------------------|--------------|------------------------|
| **Date** | 2026-08-06 T 08:48 | 2026-08-06 T 19:44 | 2026-08-06 T 20:14 |
| **Passes** | 108 / 128 | 104 / 128 | 103 / 128 |
| **Pass Rate** | 84.38% | 81.25% | 80.47% |
| **Worst Orientation** | 0.2519 rad | 0.2528 rad | 0.2539 rad |
| **Worst Joint Velocity** | 6.70 rad/s | 7.69 rad/s | 18.14 rad/s |
| **Mechanism** | `ik_posture_gain=0.816` `ik_posture_activation_error_rad=0.229` | `ik_max_joint_deviation_rad=0.312 rad` | `ik_max_joint_deviation_rad=1.198 rad` |

---

## Reproducibility and Historical Record

**Critical Finding**: The reference result (108/128) includes `ik_posture_gain` and `ik_posture_activation_error_rad` in its gains dictionary. Both of these fields have been **deleted from `CartesianVelocityConfig`** and are no longer parseable. There is **currently no way to construct a `CartesianVelocityConfig` object that matches those exact gains and reproduces that exact behavior**.

As of this date, there is **no validated "best known" gain vector for the current controller code**. The true current best from the new searches is **104/128 (81.25%)** with `ik_max_joint_deviation_rad=0.312 rad`.

---

## Observed Search Behavior

Both new searches produced results that pushed `ik_max_joint_deviation_rad` toward its loose/near-off end of the action-space range:
- `nullspace_v2`: 0.312 rad (loose — within the 2.0 rad "essentially never bind" range)
- `nullspace_v2_seeded108`: 1.198 rad (very loose)

For comparison, the tight end is 0.01 rad, and the removed soft mechanism's validated effective values were 0.15 rad (for `neg40`/`neg45`) and 0.229 rad (for `hanging_alpha_0_5`).

This pattern matches the documented history of the replaced soft mechanism: "repeated gain searches over this exact 2D sub-range — even ones deliberately seeded with forced-nonzero starting values — almost never converged to using it: the optimizer kept landing on ik_posture_gain~0 or values indistinguishable from just lowering ik_joint_gain instead" (`velocity_transport_env.py` lines 90–94).

**Notably**: The second search (`nullspace_v2_seeded108`) produced a concerning **18.14 rad/s worst-case joint velocity** on a scenario (`unrotated_wrist2offset`, fast move) that the reference result handled cleanly at 4.19 rad/s, suggesting worse robustness despite comparable overall pass rates.

---

## A manual sweep (2026-08-06, same day) found tight bounds monotonically WORSE, and one case root-caused directly

A follow-up manual sweep fixed the 108/128-equivalent gains (kp_x/kp_rot/ik_joint_gain/pinv_damping/qp_task_weight unchanged) and varied ONLY `ik_max_joint_deviation_rad` across `{0.01, 0.02, 0.03, 0.05, 0.08, 2.0}` against the full 96-episode evaluation grid (`dx_fractions`/`fast_move_dx_fractions` only, not the full 128 -- see `evaluate_gains`'s defaults). Result: pass rate degrades MONOTONICALLY as the bound tightens (62/96 at 0.01 rad up to 88/96 at 2.0/loose) -- the opposite of the earlier direct-trace validation for `neg40`/`neg45` at a specific dx. The tightest setting's worst-case joint velocity was **161.57 rad/s** (`neg45_wrist2offset`, `dx=-0.029m`, `move_duration_s=1.0` -- a SLOW move, ruling out fast-move stress as the cause).

**Root-caused directly** (not just observed): this specific 161.57 rad/s case is `wrist_2` crossing exactly through 0 (the known UR wrist singularity this repo's torque-control lane has separately documented and fixed via `jacobian_singular_cond_max`) at the exact termination step. Re-running the IDENTICAL scenario/dx/gains with only the 6th action dimension varied confirms this is CAUSAL, not coincidental: loose (`action[5]=-1.0`) and mid (`action[5]=0.0`) both complete the full episode cleanly (no guard trip); only tight (`action[5]=1.0`, 0.01 rad) crosses the singularity and blows up. Mechanism: at loose/unconstrained settings, the redundant (null-space) part of the IK solve is free to steer `wrist_2` AWAY from 0 as the arm moves in -X; the tight bound forces `q` to stay within 0.01 rad of `q_rest` in the null-space directions, removing that escape route, so the task-only (row-space) component alone drives `wrist_2` straight through the singularity where `pinv(J)` is unbounded.

This is a DIFFERENT mechanism from the 18.14 rad/s spike documented above (that one was independently confirmed unaffected by disabling `ik_max_joint_deviation_rad` -- pure `pinv(J)` singularity blowup at an extreme fast-move duration, unrelated to the clip). This one is a genuine new failure mode CAUSED by the null-space clip: tightly confining redundant motion can remove a naturally-occurring singularity-avoidance path that the unconstrained solve was relying on, specifically for -X moves at these wrist2-offset poses (consistent with this repo's general finding that wrist-singularity effects and X-direction asymmetry are real and pose/direction-specific, not sampling noise). **This refines, rather than contradicts, both the manual sweep's aggregate numbers (real, monotonic) and the earlier direct-trace validation that tight bounds fixed `neg40`/`neg45` cleanly at a DIFFERENT dx (`+0.0464m`, a +X move that does not approach the singularity)** -- the mechanism's effect depends on whether the specific (pose, dx) combination's natural redundant drift happens to be singularity-avoiding, which is not uniform across the evaluation grid.

---

## Notes

This document is a factual record of the search outcomes for a human decision about next steps. No recommendation is made here — the scope of this finding is to establish baseline numbers and highlight that the referenced baseline is now unreproducible by any current code.
