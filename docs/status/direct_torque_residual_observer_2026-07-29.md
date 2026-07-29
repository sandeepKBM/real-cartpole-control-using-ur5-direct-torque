# Direct-torque dynamics residual observer, 2026-07-29

## Scope and status

**Diagnostic-only.** This lands a per-cycle joint-space dynamics residual observer for
`hardware/direct_torque_transport.py` (the `direct_torque` control mode only -- not
`position`, which has no real commanded joint torque in the sense this observer needs, and
not `urscript`, whose control law runs entirely on-robot with no local Python dynamics loop
to hook into). It is logged to `trace_rows` as four new fields for post-hoc analysis and
future tuning. **It does not read into, weaken, widen, or otherwise touch any existing trip
condition.** `hardware/safety.py` (`CartesianMoveMonitor`) and `controller_core/safety.py`
(`ImpedanceSafetyMonitor`) are untouched by this change -- confirmed by `git diff` before
committing (see "Verification" below).

## Motivation

The 2026-07-28 real-hardware session found `CartesianMoveMonitor`'s TCP acceleration
estimate has an inherent noise floor (median ~1.74 m/s^2 at rest, from a stationary RTDE
capture -- `tools/analyze_state_noise_capture.py`). That was already fixed via gap-windowed
differencing + an EMA low-pass filter (`CartesianMoveLimits.accel_gap_cycles` /
`speed_lowpass_alpha`), which reduces noise sensitivity broadly but has no way to
distinguish "this acceleration reading is a real physical event" from "this is measurement
noise" -- it is still pure finite-differencing with no model of the robot behind it.

This repo now has fast, validated access to the robot's own rigid-body dynamics at 500 Hz:
`hardware/local_dynamics.py::LocalPinocchioFastDynamics` (J(q), M(q) in ~0.05-0.08 ms) and
`controller_core/model_dynamics.py::PinocchioUR5eDynamics` (`.bias(q, qd)` =
C(q,qd)qd + g(q), parity <1e-6 Nm vs MuJoCo `qfrc_bias`). That makes it possible to predict
what acceleration the *known, commanded* torque should physically produce each cycle, and
compare it to what is actually measured. A residual that is fully explained by known
dynamics + commanded torque is real, intended motion; a residual that is not is a
candidate anomaly -- either a sensor/estimation artifact, or a genuine physical event
(external collision, joint fault) that the existing guards already exist to catch.

**Why this is not wired into a trip condition today:** a real collision or fault typically
produces MORE extreme accelerations than smooth commanded motion, not less -- so "this
looks physically implausible, therefore ignore it" would be backwards for a safety guard.
A residual observer is a plausible foundation for a future, carefully-validated smarter
trip condition (e.g. "a clean residual history could justify a modestly higher
acceleration ceiling"), but landing that untuned now would risk exactly the mistake this
project's own history warns against (AGENTS.md's do-not-recreate list; the retracted advice
to raise `CartesianMoveMonitor`'s threshold before understanding the real wrist-singularity
divergence it was masking). This change stops at the diagnostic.

## Design

### Predicted acceleration

`qdd_pred = M(q)^-1 @ (tau_total_physical - bias(q, qd))`
(`controller_core/dynamics_residual.py::predict_joint_acceleration`, pure numpy,
`np.linalg.solve`, no simulator dependency).

`tau_total_physical` must be the TRUE total torque delivered to the joints, not just the
controller's own output. On the real UR5e's `direct_torque` mode, PolyScope's
`directTorque()` call auto-adds gravity compensation to whatever Python sends (AGENTS.md:
"Never add gravity torque in Python when using directTorque() -- PolyScope adds it"), so
`hardware/direct_torque_transport.py` adds `residual_dynamics.gravity(q)` back onto `tau`
(the Python-side commanded torque already sent to `link.direct_torque()`) before calling
`predict_joint_acceleration`. Algebraically this cancels with the `g(q)` term inside
`bias(q, qd)`, so the formula is insensitive to any small residual mismatch between this
Pinocchio model's gravity and PolyScope's own internal one, as long as both are evaluated
consistently.

`M(q)` reuses whichever mass matrix this cycle's control step already computed (from
whichever `dynamics_source` is active -- `rtde`, `local`, or `local_pinocchio`), rather than
recomputing it. `bias(q, qd)` and `gravity(q)` are computed from a dedicated
`PinocchioUR5eDynamics` instance, constructed unconditionally (independent of
`dynamics_source`, which only controls the *controller's* own J/M source) so the residual's
own dynamics model is identical regardless of which `dynamics_source` a given run uses.

### Measured acceleration

`hardware/joint_accel_estimator.py::JointAccelEstimator` -- ports the *technique*, not the
code, of `hardware.safety.CartesianMoveMonitor`'s TCP acceleration estimate (gap-windowed
finite differencing + optional EMA low-pass, `CartesianMoveLimits.accel_gap_cycles` /
`speed_lowpass_alpha`). One difference: TCP acceleration there is a *double* finite
difference of raw position (position -> speed -> accel), so the gap-widening trick is
applied to the intermediate speed signal. `qd` (RTDE `getActualQd()`) is already a single,
directly-measured signal, so `qdd` here needs only ONE differentiation -- the gap-widening
trick is applied at that single step (a `qd` sample from `gap_cycles` cycles back, divided
by the real elapsed time across that window), then an optional EMA layer
(`lowpass_alpha`) for further smoothing. At the defaults (`gap_cycles=1`,
`lowpass_alpha=1.0`) this reduces to the original single-cycle, unfiltered finite
difference. Real elapsed time reuses the transport loop's already-computed `interval_ns`
(cycle-start-to-cycle-start), matching `dt_s` as a floor, the same real-elapsed-time
convention `CartesianMoveMonitor` already uses.

### Residual

`joint_acceleration_residual = qdd_measured - qdd_predicted`, per joint (6-vector). Both
the 6-vector (`qdd_residual`) and its L2 norm (`qdd_residual_norm`) are logged: the vector
for diagnosing *which* joint an anomaly localizes to (useful for fault attribution), the
norm for a quick scalar trend line / threshold sweep in post-hoc analysis. `qdd_pred` and
`qdd_measured` are logged too, so the full prediction/measurement/residual chain is
reconstructable from the trace alone.

New `trace_rows` fields (flat 6-element lists, matching the existing `q`/`qd`/
`tau_controller` convention, or `None` when unavailable): `qdd_pred` (always populated),
`qdd_measured`, `qdd_residual`, `qdd_residual_norm` (the latter three `None` for the first
`residual_qdd_gap_cycles` cycles while the estimator's gap window fills, or whenever
`enable_residual_observer=False`).

`run_x_transport_direct_torque()` gained three new optional parameters:
`enable_residual_observer` (default `True`), `residual_qdd_gap_cycles` (default `1`,
matches `CartesianMoveLimits.accel_gap_cycles`'s default), `residual_qdd_lowpass_alpha`
(default `1.0`, matches `speed_lowpass_alpha`'s default). No new CLI flags were added
(`tools/ur5e_direct_torque_x_transport.py` is unchanged) -- future tuning of the gap/alpha
values, if it becomes useful, is a small follow-up, not part of this change.

## Validation (simulation -- no real hardware available for this task)

Both checks use real MuJoCo rollouts via a hand-rolled step loop reusing
`simulation.ur5e_mujoco_torque`'s `MujocoUR5eTorqueAdapter` with
`gravity_mode="gravity_comp"`, `coriolis_feedforward=False` -- this reproduces the real
`direct_torque` mode's actual physical semantics (gravity auto-compensated, Coriolis not)
rather than the sim lane's usual full-bias compensation. Config:
`config/ur5e_mujoco_torque_osc_tuned.yaml`, start pose `HEIGHT_ALPHA_0_5_Q`, dt=0.002s
(500 Hz, matching `direct_torque`'s real rate). Full rollout/assertion code:
`tests/mujoco/test_direct_torque_residual_observer.py`.

### Check 1: clean move-hold rollout (no disturbance)

200 steps (0.4 s: 0.1 s min-jerk move + hold), `gap_cycles=3`. 198 residual samples once the
gap window fills:

| | value (rad/s^2) |
|---|---|
| median | 1.2e-8 |
| mean | 4.6e-6 |
| p95 | 3.2e-5 |
| max | 7.8e-5 |

The residual stays within numerical-precision range of zero throughout the clean move
*and* the hold, confirming the model correctly explains ordinary commanded motion (not just
a static hold). Test: `test_residual_stays_small_on_clean_move_hold_rollout` (asserts
`median < 0.05`, `max < 0.5` -- generous relative to the measured values above, to avoid a
brittle bound on exact floating-point reproducibility).

### Check 2: injected external disturbance

340 steps (0.68 s), same move-hold profile, `gap_cycles=3`. A 30 N force on
`wrist_3_link` (`data.xfrc_applied`, a plausible collision-scale load -- well under the
UR5e's ~49 N rated payload weight) injected for steps 150-180 (0.30-0.36 s, inside the hold
phase, chosen so it's unrelated to the move itself). This force is invisible to both the
Pinocchio dynamics model and the commanded torque -- exactly the "unmodeled physical event"
this observer is meant to flag.

| window | steps | qdd_residual_norm peak (rad/s^2) |
|---|---|---|
| baseline (pre-disturbance hold) | 80-140 | 3.7e-7 |
| disturbed (window + estimator gap lag) | 153-180 | 23.9 |
| recovered (well after disturbance ends) | 260-340 | 0.57 |

Ratios: disturbed/baseline ~6.4e7x; disturbed/recovered ~42x. The residual rises sharply
and specifically during the disturbed window (peaking ~24 rad/s^2, ~64 million times the
clean baseline) and clearly decays afterward -- by steps 260-280 alone it is already down to
~0.03 rad/s^2 (~800x below peak), continuing to fall through step 340. It does **not**
fully return to the ~1e-7 clean-baseline level within this window; that is expected, real
physics, not an observer defect -- a genuine kinetic-energy injection followed by the
controller's own corrective motion is a real, gradually-decaying mechanical transient, not
an instantaneous return to a quiescent, near-double-precision-noise residual. Test:
`test_residual_detects_and_recovers_from_injected_disturbance` (asserts
`disturbed_peak > 1000 * baseline_peak` and `recovered_peak < disturbed_peak / 20`, both
with wide margin below the measured ~6.4e7x / ~42x).

## What landed

- `controller_core/dynamics_residual.py` (new) -- pure-numpy `predict_joint_acceleration`,
  `joint_acceleration_residual`. No simulator dependency, matches `controller_core`'s
  numpy-only invariant.
- `hardware/joint_accel_estimator.py` (new) -- `JointAccelEstimator`, the gap-windowed +
  EMA qd -> qdd estimator described above.
- `hardware/direct_torque_transport.py` -- wiring only: construct a `PinocchioUR5eDynamics`
  + `JointAccelEstimator` before `link.connect()` (fails fast on a bad
  `residual_qdd_gap_cycles`/`residual_qdd_lowpass_alpha` before ever touching the robot,
  same convention as `gain_overrides`), reset after `state0` is read, compute and log the
  four new fields once per cycle (reusing this cycle's already-computed `mass_matrix` and
  `interval_ns`). No change to the guard-check block (`safety.check`, `move_monitor.check`,
  `is_robot_safety_normal`) or its ordering relative to `link.direct_torque()`.
- `tests/unit/test_dynamics_residual.py`, `tests/hardware/test_joint_accel_estimator.py`,
  `tests/hardware/test_direct_torque_residual_observer_trace.py` (wiring/shape checks via
  the existing mocked-link pattern from `test_direct_torque_transport_diagnostics.py`),
  `tests/mujoco/test_direct_torque_residual_observer.py` (the real-dynamics validation
  above).

## Verification

`git diff` for this change touches no trip-condition code: `hardware/safety.py` and
`controller_core/safety.py` are absent from the diff entirely. Confirmed by inspection
before committing (`git diff --stat` / `git diff -- hardware/safety.py
controller_core/safety.py` both empty).

## Future work (not implemented here)

The "a clean residual history could justify a modestly higher acceleration ceiling" idea
discussed during scoping is explicitly **not** implemented. Responsibly evolving this
observer into an actual guard improvement would need, at minimum: a real-hardware
validation campaign (this task only had simulation available), a principled way to bound
false-negative risk (a real fault that happens to look "dynamics-consistent" -- e.g. a slow
joint-friction degradation -- must not be masked by a residual-based relaxation), an
explicit decision on which existing trip condition (if any) it should modulate and how, and
sign-off following this repo's own precedent for that kind of guard change (see the "found,
not yet fixed" items in AGENTS.md SS4 for the standard of scrutiny expected before touching
`CartesianMoveMonitor`/`ImpedanceSafetyMonitor` trip logic).

## Tests

- `tests/unit/test_dynamics_residual.py` -- 8 tests, pure math.
- `tests/hardware/test_joint_accel_estimator.py` -- 9 tests.
- `tests/hardware/test_direct_torque_residual_observer_trace.py` -- 2 tests (mocked-link
  wiring: fields present/populated, `enable_residual_observer=False` disables cleanly).
- `tests/mujoco/test_direct_torque_residual_observer.py` -- 2 tests (the real-dynamics
  validation above).
- Full suite: `/common/users/ss5772/miniforge3/envs/mujoco_ur5e/bin/python -m pytest -q`.

## Rollback

`git checkout -- hardware/direct_torque_transport.py && rm controller_core/dynamics_residual.py hardware/joint_accel_estimator.py tests/unit/test_dynamics_residual.py tests/hardware/test_joint_accel_estimator.py tests/hardware/test_direct_torque_residual_observer_trace.py tests/mujoco/test_direct_torque_residual_observer.py docs/status/direct_torque_residual_observer_2026-07-29.md`

(or `git revert <commit>` once committed). Default behavior with `enable_residual_observer`
left at its new default (`True`) adds four new, always-`None`-safe trace fields and one
extra `PinocchioUR5eDynamics` instance's worth of per-cycle compute; nothing about the
control law, timing loop, or any trip condition changes.
