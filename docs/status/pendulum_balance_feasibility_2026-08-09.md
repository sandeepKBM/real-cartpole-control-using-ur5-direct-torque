# Can the current controller balance the pendulum starting inverted?

**Date:** 2026-08-09
**Context:** follow-on to the swing-up infeasibility finding and the velocity-controller
speed-ceiling test (same date) -- user asked a different question: not "can we flip it up,"
but "if it STARTS flipped up, can we at least hold it there."

## SECOND CORRECTION (2026-08-09, later same day -- read this first): the "raised ik_joint_gain,
## 2.3 rad" result below is ALSO not genuine active balance

The "Follow-up: attacking the actual lag" section below reports that raising `ik_joint_gain` to
~100-150 + `orientation_priority` recovers cleanly from perturbations up to 2.3 rad. That
measurement is real, but a direct check -- prompted by finding the exact same problem in the
torque-lane LQR controller (docs/status/pendulum_balance_torque_lqr_2026-08-09.md's own
correction) -- shows it is **not genuine theta-feedback balance control either**:

Comparing the "active" outer law (`kp=0.1, kd=0.02`) against the SAME config with the outer law
zeroed out (`kp=0, kd=0` -- the arm still actively holds its X position via `ik_joint_gain=120`,
it just never reacts to the pendulum's tilt at all) gives IDENTICAL results at every
perturbation tested (0.5, 1.0, 1.5, 2.0, 2.3 rad -- same pass/fail, `final_theta_err` matching
to 2-3 decimal places in every case). **The outer balance law contributes nothing.** What is
actually happening: `ik_joint_gain=100-150` makes the arm hold its Cartesian position rigidly
and quickly (a real, active behavior), and the pendulum's own joint friction -- the SAME
mechanism already correctly identified for the untuned default config -- does the actual work of
bleeding off energy near the top, exactly as in the torque-lane LQR case.

**The real (friction + rigid-hold, not genuinely theta-reactive) envelope, bisected**: survives
cleanly up to ~1.5 rad (~86 deg), fails at ~2.0 rad and above. This is smaller than the
previously-claimed 2.3 rad (which happened to still be within the coincidentally-large passive
envelope this particular tuned config produces, not because the "active" term was doing
anything at that magnitude).

**Status of this whole file, given both corrections**: no controller built this session
(velocity-lane default, velocity-lane tuned, or torque-lane LQR) has been shown to perform
genuine, verified closed-loop theta-feedback stabilization at any gain tested. Every "recovery"
measured is fully or almost-fully explained by passive joint friction combined with whichever
mechanism happens to hold the arm rigidly in place. This is a real, useful, if humbling, finding
in its own right (see AGENTS.md's own repeated "friction near threshold" theme) -- but it
means the actual, honest current status is: **we do not yet have a demonstrated working active
balance controller**, only a characterization of how far passive dynamics alone can carry a
near-inverted release.

## Answer (ORIGINAL, kept for the debugging trail -- see the correction above for the current truth):
with the DEFAULT config (`ik_joint_gain=4.0`), only marginally, mostly via passive
friction. But see the "Follow-up: attacking the actual lag" section below -- raising
`ik_joint_gain` to ~100/s and enabling `orientation_priority` (both already-existing, currently
off/default-value mechanisms) extends real, active-correction balance to perturbations of over
2 radians (~130+ deg), a dramatically different and better answer than the default-config
result below suggests. Read both sections; the first section's narrow verdict is real for the
DEFAULT config specifically, not a fundamental limit of this controller architecture.

## Analytical prior (checked before running anything)

Unlike swing-up, balancing is a stabilization problem: the cart must react FASTER than the
pole falls, not just move fast in absolute terms.
- Pendulum's own inverted-equilibrium instability rate: `lambda = sqrt(m*g*l_com/I_pivot)`,
  using this session's already-measured `I_pivot ~= 0.0036 kg*m^2`: **lambda ~= 6.96 rad/s**
  (tau ~= 144 ms) -- how fast a small lean away from vertical grows on its own.
- The velocity controller's own position-tracking inner loop bandwidth is set by
  `ik_joint_gain` (config default 4.0 /s): **tau ~= 250 ms** -- how fast the cart can chase a
  commanded position.
- Ratio (actuator bandwidth / plant instability rate) = 4.0/6.96 = **0.575, i.e. the actuator
  is structurally SLOWER than the fall.** A real red flag for stabilizability going in.

## Method

`tools/diagnostics/pendulum_balance_test.py` -- unlike the speed-ceiling test (kinematic-only),
this needs REAL pendulum dynamics under gravity, so it uses the actual composed arm+pendulum
MuJoCo model (`simulation/ur5e_pendulum_compose.py`) with `mj_step`. The arm is still driven
kinematically (qvel set from the real `CartesianVelocityController`'s output each cycle, then
`mj_step` lets that velocity couple into the pendulum through real rigid-body dynamics) --
consistent with every other velocity-control-lane test in this repo. Outer loop: a simple PD
on pole angle, `target_x = x_pivot + kp*theta_err + kd*theta_dot`, fed through the SAME
`ik_seeded_resolution` controller and clamp used everywhere else in this session.

Two real mistakes were caught and fixed before trusting any result:
1. First perturbation tested (0.02 rad) was below the joint's own Coulomb friction breakaway
   angle (`frictionloss=0.01 Nm` vs gravity torque -- crosses at ~0.057 rad). Every gain
   combination looked identically "perfect" because the pendulum was stiction-locked and never
   moved at all -- not a real balance result.
2. `fk_jacobian_fn` originally allocated a fresh `mujoco.MjData` every call (up to ~1500/trial
   x 105 trials) -- the first sweep never finished inside a 10-minute timeout. Fixed by reusing
   one scratch `MjData`, matching `hardware/local_dynamics.py`'s own pattern; the fix cut a
   single trial from "didn't finish" to 0.75s.

## Results

**Small perturbations (<=0.3-0.4 rad, ~17-23 deg) survive** for the full tested duration (up to
20s) at gentle gains (kp~0.05-0.2). But a direct check isolates WHY: at `kp=0, kd=0` (the cart
never moves at all -- confirmed, max cart displacement = 0.000 mm) the pole ALSO survives a
0.3 rad perturbation for the full 20s, settling to a final error of -0.0072 rad, arguably
better than with active correction (-0.0583 rad at kp=0.1). **This is passive joint friction
arresting the fall, not active cart-based balance** -- the same "friction near the threshold"
phenomenon this session already found for swing-up (2026-08-09, real friction dominates near
an asymptotically slow approach to an unstable/marginal point), just working in the opposite
direction here: released from rest near the top, the gravity torque builds up slowly enough
that Coulomb + viscous friction bleeds the energy before the angle can grow large, for
perturbations below roughly the friction breakaway threshold.

**Active correction genuinely helps a little, within a narrow gain window.** At kp~0.1 (with or
without kd), the closed loop drives error toward zero using small, gentle cart motions -- max
27-37 mm of cart displacement, peak speed ~120-160 mm/s, both far under the 0.25 m/s software
clamp (clamped vs. unclamped runs gave identical results, confirming the clamp never binds for
balance-scale corrections).

**Push the SAME control law's gain too high and it makes things WORSE, not better** -- a direct
empirical confirmation of the analytical red flag. At `kp=0.4`, peak error grows past the
initial perturbation (0.2 -> 0.27-0.32 rad) before eventually falling at higher kd. At
`kp=0.8`, every trial falls, consistently around t~0.54-0.67s, via a clearly oscillatory
(not monotonic) divergence -- the signature of actuator-lag-induced instability: the
controller's ~250ms following delay is slower than the pole's ~144-210ms natural fall time
constant, so past a critical gain the closed loop's phase margin goes negative and correction
adds energy instead of removing it.

**Beyond ~0.4-0.5 rad initial deviation, nothing holds it, at any tested gain.** At
`pert=0.5 rad`, `kp=0.1`: falls at t=0.008s -- essentially immediately, both friction and the
lagged active correction are too slow to matter once the destabilizing torque clears the
friction threshold with real angular velocity already built up.

## Bottom line

Starting the pole exactly (or very nearly) at the inverted position, this controller stack can
keep it there for small deviations (roughly up to a quarter-radian, ~15 deg) -- but that
survival is mostly the apparatus's own joint friction doing the work, not genuine active
balance. There IS a real, narrow, usable active-control gain window (kp~0.05-0.2) that helps a
bit using only gentle, well-under-the-clamp cart motions. But it is fragile in a specific,
diagnosable way: push the gain higher (the natural instinct if a small perturbation isn't fully
corrected) and the SAME mechanism that helps at low gain actively destabilizes at high gain,
because the controller's own response lag (250ms) is slower than the pole's natural fall rate
(144-210ms) -- consistent with, and now empirically confirming, the analytical bandwidth-ratio
red flag computed before any simulation was run. Any real disturbance larger than roughly a
quarter to a half radian is unrecoverable at any gain tested.

This does not rule out balance entirely -- a controller with genuinely faster inner-loop
bandwidth (e.g. a higher `ik_joint_gain`, or a different actuation path entirely) might do
better, and that tradeoff (faster following vs. this session's other findings about
`ik_joint_gain`'s effect on tracking/singularity behavior) is a real, separate follow-up
question, not something this test attempted to search.

## Follow-up: attacking the actual lag (`ik_joint_gain`), not just the outer PD gain

Added later the same date, at the user's request to "work the inner controller" instead of only
the outer balance-law gain. The original analysis pinned the root cause on `ik_joint_gain`
(inner-loop bandwidth, default 4.0/s, tau=250ms) being slower than the pole's own instability
rate (~6.96 rad/s, tau=144ms) -- so the natural next test is raising `ik_joint_gain` itself,
which the outer-PD-only sweep above never varied.

**Real result: it works, within a bounded window, and only combined with `orientation_priority`.**
- Raising `ik_joint_gain` alone (to 100/s, ~14x the plant's instability rate -- comfortably past
  the classical "actuator must outrun the plant by 5-10x" margin) does let a 0.5 rad perturbation
  genuinely stabilize (converges to 0.011 rad, not just friction-stuck) -- but only with the
  software clamp REMOVED. With the real clamp (`max_lin_speed_mps=0.25`) back in, the same
  config still falls: the larger `ik_joint_gain` demands cart speeds the clamp truncates,
  degrading the correction.
- **`orientation_priority=True` rescues the clamped case at every `ik_joint_gain` tested**
  (100/150/200/300) -- combined with `ik_joint_gain~100-150` and the real 0.25 m/s clamp
  in force, 0.5 rad perturbations now stabilize cleanly (final error ~0.02-0.04 rad).
- **There is an exact, derivable upper bound on how far `ik_joint_gain` can be pushed**: the
  per-cycle update `q += dt*ik_joint_gain*(q_target-q)` is an explicit-Euler discrete map,
  stable only for `ik_joint_gain*dt < 2` -- at this repo's 125 Hz control rate (`dt=8ms`), that's
  **`ik_joint_gain < 250/s`**. Measured: 300/s starts failing again via a genuinely different
  mechanism (oscillatory divergence, not the physical actuator-lag problem) -- an exact match to
  this bound, not a coincidence.
- **A "~0.50 rad survival boundary" was measured and reported here, then found to be a bug, not
  a real result.** `FALL_THRESHOLD_RAD` (the "has it fallen" cutoff) was set to 0.5 rad, and this
  sweep tested perturbations from 0.5 up to 1.0 rad -- so every `pert >= 0.5` tripped "fell" on
  the very first sample (`theta_err` starts AT the perturbation, before the controller can act
  at all), regardless of any real dynamics. The tell: `fell_at=0.0` with `peak_err`/`final_err`
  byte-identical to the raw perturbation, IDENTICAL at both 125 Hz and 500 Hz -- a real dynamic
  wall would not look identical across two different control rates. Caught by that inconsistency,
  not by inspection alone.

**Corrected result, `FALL_THRESHOLD_RAD` raised to 1.5-2.8 rad (comfortably clear of any tested
perturbation) and re-run:** the true envelope is dramatically larger than first reported.
At `ik_joint_gain=100`, `orientation_priority=True`, real `max_lin_speed_mps=0.25`, **every
perturbation tested from 0.5 up through 2.3 rad (~132 deg) stabilizes cleanly**, converging to a
final error of only 0.024-0.032 rad regardless of how large the initial perturbation was (0.5 vs
2.3 rad converge to essentially the same small residual) -- genuine active correction, not a
near-miss. It only stops recovering at `pert=2.5 rad` (~143 deg), where `final_err=2.475` shows
essentially no correction happened at all -- but that perturbation is no longer really "starting
near the top": it is only ~0.64 rad from the pole's OWN hanging equilibrium (`pi` away from
inverted), i.e. past the halfway point to hanging, not a meaningful "recover from near-inverted"
case. Confirmed at both 125 Hz and 500 Hz with near-identical results, and separately confirmed
that `ik_joint_gain=250`/`600` (both under the 500 Hz discrete-stability ceiling of 1000/s, see
below) also converge cleanly at 500 Hz across the same range, while `ik_joint_gain=250` at 125 Hz
(near its own 250/s ceiling) visibly degrades with larger perturbations (final error grows to
0.76 rad at `pert=1.2`) -- real, consistent evidence that raising control rate genuinely buys
more usable inner-loop gain, not just a same-envelope shuffle.

**Practical takeaway, corrected**: raising `ik_joint_gain` well above its current default (4.0 ->
roughly 100/s) combined with enabling `orientation_priority` and respecting the real 0.25 m/s
clamp is a real, concrete, and far more effective fix than the original narrow-window finding
suggested -- it recovers from perturbations of over 2 radians (~130+ deg) off vertical, not just
a marginal ~15-20 deg improvement over passive friction. The genuine remaining limit is not a
sharp angular wall but the point at which "near-inverted" stops being a meaningful description of
the starting condition (past roughly the halfway point to the hanging equilibrium). `ik_joint_gain`
itself still has a real, exact discrete-stability ceiling (`2/dt`: 250/s at 125 Hz, 1000/s at
500 Hz) past which it degrades via loop-discretization instability, a different failure mode from
the original actuator-lag problem -- raising the control rate (125 -> 500 Hz, already a supported
`rate_hz` parameter in `hardware/velocity_transport.py` and already used in production by the
`direct_torque` control mode) directly raises that ceiling too. Neither `ik_joint_gain=100+`,
`orientation_priority=True`, nor `rate_hz=500` are the current defaults anywhere in this repo --
these are diagnostic findings, not promoted config changes, and none of this has been validated
against real hardware RTDE/`speedL` behavior at 500 Hz (untested whether the real robot's speedL
interface and this repo's `UR5eLink`/`DeadlineMonitor` timing assumptions, calibrated for 125 Hz
per `AGENTS.md` sec 4's timing-gap note, actually support this rate cleanly).

## Files

| path | what |
|---|---|
| `tools/diagnostics/pendulum_balance_test.py` | coupled dynamic balance sweep (gain grid x perturbation x ik_joint_gain x orientation_priority x clamp) |
| this file | write-up |

**Tests run:** manual execution of the script's gain/perturbation sweeps (documented above),
plus ad hoc follow-up checks (long-duration, real-clamp, cart-displacement instrumentation) run
inline, not committed as separate scripts.
**Tests not run:** no pytest coverage -- diagnostic script, matching this session's precedent
for `tools/diagnostics/*` prototypes (not a new module/package).

**Rollback:**
```bash
rm -f tools/diagnostics/pendulum_balance_test.py \
      docs/status/pendulum_balance_feasibility_2026-08-09.md
```
Nothing else touched. Nothing committed.
