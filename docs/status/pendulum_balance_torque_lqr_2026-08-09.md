# Torque/impedance LQR balance controller for the pendulum -- 6 bugs found, and the "working"
# result turned out to be a mirage on top of a 6th, more important one

**Date:** 2026-08-09
**Context:** the velocity-lane balance controller (docs/status/pendulum_balance_feasibility_
2026-08-09.md) works but drives the arm through a position-tracking joint-space follower --
structurally a worse fit for a fast stabilization problem than direct force control. This
builds the same balance task on the torque/impedance lane instead (real `<motor>` actuators,
full coupled `mj_step` dynamics for both arm and pendulum), with an LQR-designed outer law
replacing a hand-picked PD.

## CORRECTION (read this before anything below): the "works cleanly to 3.0 rad" result was real, but was NOT genuine LQR control

Everything in the "Result" section below was true as measured -- but a follow-up check
(prompted by disturbance-robustness testing showing ZERO sensitivity to sensor noise or
control delay, which was itself too clean to trust) found the actual cause: **at the R weight
used (`r_weight=1e6`), the LQR's real, active control output is negligible** (`max|tau_extra|`
for a 0.5 rad error is ~0.004 Nm, against a ~30 Nm gravity-hold torque already being applied --
literally noise-level). Directly confirmed: **zeroing `K` out entirely (`K=0`, pure
gravity-hold, no active correction whatsoever) reproduces the exact same "clean, zero-overshoot
recovery up to 3.0 rad" result, to 4 decimal places, at every perturbation tested.** The
controller was not stabilizing the pendulum -- the SAME passive, friction-assisted "recovery
near an unstable equilibrium" phenomenon already found and correctly identified for the
velocity lane's DEFAULT config (docs/status/pendulum_balance_feasibility_2026-08-09.md) was
happening here too, just missed this time because a genuinely-controlled and a
do-nothing-controlled trial were never directly compared before writing this up originally.

**Why `r_weight` ended up there**: fixing an earlier real bug (saturating torques at
`r_weight=0.0005`, see bug #4 below) by pushing `r_weight` up to whatever value first stopped
saturating overshot enormously -- 1e6 stops saturation, but also very nearly stops control
entirely. A follow-up sweep (`r_weight` in {10, 30, 100, 300}) confirms genuine active control
IS achievable and DOES diverge measurably from the passive baseline at small/moderate
perturbations (0.5-1.0 rad) -- but at large perturbations (2.5 rad), that same active
correction makes things WORSE than doing nothing (final error 1.6-2.5 rad instead of ~0.007),
and at `r_weight=300` it outright fails (falls at t=0.994s) where the passive baseline does
not. **No `r_weight` value found so far gives both genuine active correction AND the large-angle
robustness the original (mistaken) validation claimed.** The likely cause: the LQR is designed
from a linearization at the exact equilibrium, and at 2+ radians off vertical that
linearization's validity is long gone -- a fixed-gain linear law probably needs either a
tighter operating envelope (only claim robustness within the range the linearization is
actually valid) or a genuinely nonlinear/gain-scheduled design, neither of which this pass
built.

**Status: this is real, useful, but INCOMPLETE work, not a validated deliverable.** Do not cite
the "3.0 rad, zero overshoot" headline below as a working active controller. The companion
disturbance-robustness doc (`docs/status/pendulum_balance_disturbance_robustness_2026-08-09.md`)
inherits this same caveat -- its "insensitive to noise/delay" finding is explained by the same
root cause (negligible active signal to corrupt), not genuine robustness.

## Result (as originally measured -- see correction above for what it actually means)

At the final, validated configuration: **every perturbation tested from 0.02 rad up through
3.0 rad (essentially the pendulum's entire range) converges with ZERO overshoot** -- peak
error stays exactly at the initial perturbation (no growth at all) and decays smoothly to a
residual of 0.007-0.04 rad. Also robust to angular velocity kicks up to 4 rad/s at a 0.5 rad
starting offset (peak error only grows to 0.68 rad before recovering). This LOOKED like a
cleaner result than the tuned velocity-lane controller's 2.3 rad envelope (found via the "raise
`ik_joint_gain`, enable `orientation_priority`" investigation) -- but per the correction above,
it is not a fair comparison, since this one is mostly passive friction and that one is real
active control.

## The debugging trail -- 5 real, distinct bugs, each found and fixed via direct evidence, not guessing

This is worth recording in full because every one of these would have silently produced a
plausible-looking but wrong result if not caught.

1. **`x_err` sign convention bug.** First hand-derived design used a 4-state classical
   cart-pole reduction (cart mass = 1/(Jx@Minv@Jx.T), pendulum m/l/I from this session's
   earlier swing-up measurements). The state's own derivative convention required
   `x_err = actual - target`; the simulation loop computed `target - actual`. Caught because
   the closed loop diverged even at a 0.02 rad perturbation -- tiny enough that any real
   physical/linearization limit should not have mattered.

2. **Gravity/Coriolis double-counting.** `tau_gravity = data.qfrc_bias[:6]` uses the LIVE
   `qfrc_bias(q,qd)`, which is `C(q,qd)*qd + G(q)` combined (MuJoCo-native) -- using it
   directly as "gravity compensation" also cancels the pendulum's real momentum-coupling
   reaction onto the arm, physics the LQR model depends on. Fixed by evaluating `qfrc_bias`
   at `qd=0` specifically (matches `simulation/ur5e_mujoco_torque.py`'s own established
   convention). Real, but did not fully explain the divergence.

3. **Hand-derived model doesn't match the real kinematic coupling.** Even after fixes 1-2,
   the closed loop still diverged from a 0.02 rad start. Root cause: the 2-state cart-pole
   reduction assumes a clean 1-DOF translating cart, but the Jacobian row for X has FOUR
   nonzero joint components (shoulder_pan/lift/elbow/wrist_1) -- a task-space X-force command
   through this redundant, non-decoupled Jacobian also rotates and shifts the wrist in Y/Z,
   which the pendulum's non-trivially-oriented hinge axis (see docs/status/
   ur5e_pendulum_cad_model_2026-08-09.md) very plausibly responds to as much as the
   translation. Fixed by abandoning the hand-derived reduction entirely and linearizing the
   REAL, FULL nonlinear MuJoCo system numerically via `mujoco.mjd_transitionFD` (MuJoCo's own
   finite-difference linearization), designing a full-order (14-state, 6-input) LQR directly
   on that instead.

4. **Uniform control-cost weighting let one weak-authority actuator dominate and saturate.**
   The full-order LQR (still diverging) was demanding 300-700+ Nm against 28-150 Nm real
   torque limits, EVERY cycle, with the sign flipping cycle to cycle (violent chattering, not
   convergence) -- confirmed by direct inspection of the commanded torques. Root cause:
   uniform `R` let the solver lean almost entirely on wrist_2 (only 28 Nm of authority).
   Fixed via the standard LQR practice of scaling `R` by `1/torque_limit^2` per actuator, so
   control cost is normalized by real authority rather than treated as uniform.

5. **The equilibrium point itself was wrong** (the root bug, and the reason fixes 1-4 alone
   never fully worked). `find_inverted_angle` used a "release from pi/2, wait for a low
   position-variance window" heuristic -- the SAME category of trap this session hit earlier
   with a 0.3 rad release point that silently never converged. This time the heuristic's own
   pass condition (settled-window std < 0.005 over 200 steps) was satisfied while sitting at a
   point that was NOT an equilibrium at all: directly measured `qacc` there was -13.4 rad/s^2,
   not ~0. A short-window position-variance check cannot distinguish "truly stationary" from
   "drifting so slowly the window looks flat." Fixed by abandoning settling-based detection
   entirely: scan `qfrc_bias` (the pendulum joint's own generalized gravity force) for its
   true zero-crossings directly (a property that holds unconditionally, no settling time
   needed), then classify stability by the local slope sign, cross-validated against a real
   long-duration (16 s), large-perturbation (0.15 rad) simulation showing which candidate's
   perturbation distance grew (unstable) vs shrank (stable) over time. The corrected
   equilibrium lands at `-pi` exactly in this composed frame's own angle convention -- `qacc`
   there is confirmed exactly `[0,0,0,0,0,0,0]`.

6. **Stale function default reintroduced bug #4 on every fresh run.** After fixing bug #4
   (uniform R letting one weak actuator saturate) by scaling `R` per-actuator, the FUNCTION's
   own default `r_weight=0.0005` parameter was never updated, and `main()` never passed the
   fixed value explicitly -- every validation during development used an inline snippet with
   `r_weight=1e6` passed manually, which never got copied back into the actual script. A fresh
   run of `python tools/diagnostics/pendulum_balance_torque_lqr.py` (prompted by a sub-agent's
   disturbance-robustness work reporting results that contradicted this doc) reproduced the
   exact bug-#4 symptom (pert=1.0/1.5 rad falling) that the write-up claimed was fixed. Fixed by
   passing `r_weight=1e6` explicitly in `main()`, with a comment explaining why. This is what
   led directly to discovering bug/finding #7 (below): re-verifying the fix's OWN validity
   (rather than just confirming it "ran without falling") is what surfaced that `r_weight=1e6`
   itself was too conservative to be doing any real control.

7. **The "working, r_weight=1e6" result is passive friction, not LQR control** -- see the
   CORRECTION section at the top of this file. Not something with a bug in the traditional
   sense (nothing crashes, nothing throws), but a real, load-bearing measurement error: the
   validation never compared against a `K=0` baseline, so a result driven almost entirely by
   the physical apparatus's own joint friction was reported as if it were the LQR working.
   Sweeping `r_weight` down to get genuine active control (10-300) reveals a real, unresolved
   large-angle robustness gap instead -- see the correction section for the numbers.

## Files

| path | what |
|---|---|
| `tools/diagnostics/pendulum_balance_torque_lqr.py` | full-order LQR torque balance controller + validation sweep |
| this file | write-up |

**Design parameters, final:** `Q` diagonal = [0.05]*6 (arm position, light regularization) +
[200] (pendulum angle, heavy) + [0.05]*6 (arm velocity) + [5] (pendulum velocity); `R` =
`diag(1e6 / torque_limit_nm^2)` (control cost normalized per actuator's real torque authority).
Linearized via `mujoco.mjd_transitionFD` (`eps=1e-6`, centered) at the corrected equilibrium,
solved with `scipy.linalg.solve_discrete_are` (the transition matrices are inherently
discrete-time, matching the 500 Hz / 2 ms control-and-physics rate used throughout).

**Tests run:** the validation sweep in `main()` (perturbations 0.02-3.0 rad, all converge
cleanly) plus an ad hoc velocity-kick sweep (up to 4 rad/s at 0.5 rad offset, all converge),
both run and output inspected directly (not scripted into pytest).
**Tests not run:** no pytest coverage (diagnostic script, matches this session's precedent for
`tools/diagnostics/*` prototypes); no comparison against real hardware; no test of sensor
noise or measurement delay (delegated separately, see the companion disturbance-robustness
task).

**Rollback:**
```bash
rm -f tools/diagnostics/pendulum_balance_torque_lqr.py \
      docs/status/pendulum_balance_torque_lqr_2026-08-09.md
```
Nothing else touched. Nothing committed.
