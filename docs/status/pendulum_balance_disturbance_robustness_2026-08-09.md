# Sensor-noise and control-delay robustness of the pendulum balance controllers

**Date:** 2026-08-09
**Context:** both balance controllers built this session (velocity-lane:
docs/status/pendulum_balance_feasibility_2026-08-09.md; torque-lane LQR: docs/status/
pendulum_balance_torque_lqr_2026-08-09.md) were validated only against an idealized
simulation -- exact angle perturbation, perfect noiseless state feedback, zero control-loop
delay. This tests two realistic degradations against the torque-lane LQR controller: sensor
noise (matching the real planned 14-bit absolute encoder) and measurement/control delay.

## Read this first: the base controller under test was not doing real active control

The disturbance sweep below was run against `pendulum_balance_torque_lqr.py`'s LQR at
`r_weight=1e6`. A separate finding, made AFTER this sweep completed (see the CORRECTION section
at the top of `docs/status/pendulum_balance_torque_lqr_2026-08-09.md`), is that at this
`r_weight` the LQR's actual control output is negligible (~0.004 Nm against a ~30 Nm
gravity-hold torque) -- zeroing the gain matrix `K` out entirely reproduces the exact same
"clean recovery" result. **This sweep's headline finding -- complete insensitivity to sensor
noise and control delay, at every tested magnitude -- is fully explained by this: there was
essentially no real active control signal for noise or delay to corrupt in the first place.**
This is not a "the controller is robust" finding. It is a "the controller with real authority
has not yet been tested against these disturbances" gap. Re-running this sweep against a
genuinely active configuration (`r_weight` in the 10-300 range, which measurably diverges from
a passive baseline -- see the other doc) is the correct next step, not yet done.

The results below are kept and reported honestly (not discarded) because the METHOD -- the
noise model, the delay-line implementation, the statistical pass-rate framing -- was
independently verified correct (see "Method validation" below) and remains directly reusable
once a genuinely-active baseline controller exists.

## Method validation (this part is trustworthy regardless of the caveat above)

`tools/diagnostics/pendulum_balance_disturbance_robustness.py` reuses the base controller's
design functions (`find_inverted_angle`, `linearize_and_design_lqr`, `static_gravity_torque`)
but implements its own simulation loop, since the base script has no hook to inject noise into
the state fed to the control law or to delay torque application.

Before trusting anything from it, it cross-checks its own clean (no-noise, no-delay) path
against `pendulum_balance_torque_lqr.run_torque_balance_trial` directly at 5 perturbations
(0.02 to 2.0 rad) -- confirmed exact agreement (`SURVIVED`/`FELL` match at every point) after
fixing the same stale-`r_weight`-default bug documented in the companion file (this script
independently hit the identical bug, since it also called `linearize_and_design_lqr` without
passing `r_weight` explicitly -- found and fixed the same way).

A direct instrumented check of the delay line (comparing `used_state` at `delay_cycles=0` vs
`delay_cycles=50`) confirms the delay mechanism itself works correctly: at `delay_cycles=50`,
the state used at simulation step 50 is exactly the TRUE state from step 0, verified by direct
value comparison, not just behavior. The delay line's fill/pop logic is real and correct.

## Results (real, but see the caveat above for what they actually mean)

**Sensor noise**, baseline perturbations 0.5 rad and 1.8 rad: quantization at 14/12/10/8/6 bits,
plus additional Gaussian noise up to std=2.0 rad (an enormous, unrealistic magnitude -- larger
than the perturbation itself) on top of a finite-differenced (not directly measured) velocity
estimate -- zero effect on outcome at any tested level, `final_theta_err` unchanged to 4 decimal
places in every case, 10/10 seeds surviving at every noise magnitude in the pass-rate sweep.

**Control delay**, same baseline perturbations: 0 to 100 control cycles (0 to 200 ms, at the
500 Hz control rate) -- zero effect on outcome at any tested delay, again `final_theta_err`
unchanged to 4 decimal places throughout. 200 ms is notably LARGER than the pendulum's own
natural instability time constant (~144 ms, from this session's earlier analysis) -- a
genuinely active, fast-responding controller should very plausibly show real degradation well
before that point. That it does not here is itself indirect evidence supporting the "this
controller isn't really doing active work" finding, not evidence of genuine delay tolerance.

**Combined noise+delay**: same null result, consistent with the above.

## What this means going forward

This script and its verified noise/delay-injection method are ready to reuse. What is missing:
a genuinely active balance controller config to run it against. The companion doc's `r_weight`
sweep (10-300) found real active control at small/moderate perturbations but a new,
unresolved large-angle failure mode -- re-running THIS disturbance sweep at one of those
`r_weight` values, restricted to a perturbation range where that config is actually known-good
(not yet precisely characterized), is the concrete next step.

## Files

| path | what |
|---|---|
| `tools/diagnostics/pendulum_balance_disturbance_robustness.py` | noise/delay injection harness, method verified correct |
| this file | write-up, including the caveat that the base controller tested was not doing real active control |

**Tests run:** the script's own baseline cross-check (5 perturbations, exact match against the
base script) and full noise/delay/combined sweep (all output captured and reviewed above).
**Tests not run:** the same sweep against a genuinely-active `r_weight` configuration (the
actually-useful next step); no pytest coverage (diagnostic script, matches session precedent);
no real-hardware comparison.

**Rollback:**
```bash
rm -f tools/diagnostics/pendulum_balance_disturbance_robustness.py \
      docs/status/pendulum_balance_disturbance_robustness_2026-08-09.md
```
Nothing else touched. Nothing committed.
