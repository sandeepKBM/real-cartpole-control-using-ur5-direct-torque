# Friction calibration from real qdd_residual data (2026-08-02)

## What this is

A real-hardware-grounded correction to one previously-guessed sim friction value, extracted
from data already collected this session (`direct_torque_20260802_190759`), not a new
real-hardware test.

## Method

`direct_torque_20260802_190759` (wrist2-offset pose, `config/
ur5e_mujoco_torque_osc_tuned_split_base_wrist.yaml`, `target_x_delta=0.02m`) ran cleanly to
completion (no guard trip) but achieved only 59% of target, settling into a flat, non-decaying
hold-phase plateau. The residual observer (`controller_core/dynamics_residual.py`, already
wired into `hardware/direct_torque_transport.py`, logging `qdd_measured`/`qdd_pred` every cycle)
was active for this run.

Key insight: at the plateau, `qdd_measured` is essentially exactly zero (`~1e-30`, genuinely
stationary) and `qd` is essentially zero too, so the Coriolis/centrifugal term in the equation
of motion vanishes. Real physics at that instant reduces to `M(q)*qdd_measured = tau_applied -
tau_friction_real` (gravity already cancelled by PolyScope's `directTorque()`, per this repo's
own established convention — see `dynamics_residual.py`'s module docstring). Since
`qdd_measured ~= 0` for the whole joint vector, this holds **without needing to invert `M`**:
`tau_friction_real ~= tau_applied`, componentwise, at that exact cycle. The plateau region's own
`tau_applied` values are therefore a direct, model-independent measurement of the real static
friction magnitude actively holding each joint — not an estimate derived through the dynamics
model, a direct read of it.

## Measurement (plateau window t=2.5-4.0s, 749 samples, std given to show how flat/clean it is)

| joint | `tau_applied` mean (Nm) | std (Nm) | current sim `frictionloss` (Nm) | verdict |
|---|---|---|---|---|
| shoulder_pan | +1.776 | 0.003 | 5.0 | below guess — no info, guess unchanged |
| **shoulder_lift** | **-6.084** | **0.011** | **5.0** | **exceeds guess — real evidence for a raise** |
| elbow | -3.131 | 0.006 | 5.0 | below guess — no info, guess unchanged |
| wrist_1 | +0.0004 | 0.0003 | 1.0 | unloaded (`split_base_wrist_task`) — no info |
| wrist_2 | -0.0003 | 0.0004 | 1.0 | unloaded — no info |
| wrist_3 | +0.0001 | 0.0003 | 1.0 | unloaded — no info |

Only `shoulder_lift` has real evidence justifying a change: its measured holding torque
(6.084 Nm) exceeds the current guessed `frictionloss` (5.0 Nm) by ~22%. The other five joints'
measurements are all *below* their current guessed values, which only establishes a lower bound
the existing guess already satisfies — this data cannot tell us whether their true friction is
higher, lower, or about right, so they are left unchanged rather than guessed at further.

Important caveat: this is a **lower bound** on shoulder_lift's real static friction (Fs), not a
precise breakaway measurement — the joint may not have been driven to its exact breakaway edge
in this one run. `frictionloss` was set to 6.1 Nm (a small margin above the 6.084 measurement,
not a speculative multiplier) rather than the raw measured value, to stay conservative given the
lower-bound nature of the evidence.

## Change made

`assets/ur5e_torque/ur5e_torque.xml`: `shoulder_lift_joint`'s `frictionloss` overridden from the
`size3` class default (5.0 Nm, shared with `shoulder_pan`/`elbow`) to a per-joint 6.1 Nm. The
other two `size3` joints and all `size1` (wrist) joints are untouched.

## Validation

- `pytest -q tests/mujoco/test_ur5e_mujoco_torque.py` — 29/29 pass (model loads, all existing
  invariants hold).
- `pytest -q -m mujoco` (full suite) — 160 passed, 3 pre-existing xfailed. One real, expected
  golden-value regression found and fixed: `tests/mujoco/test_ur5e_mujoco_torque_experiments_
  refactor_parity.py::test_controller_rollout_matches_pre_refactor_golden_values` — a full
  controller-rollout golden trace at a *different* pose/config
  (`config/ur5e_mujoco_torque_osc_tuned.yaml`) that also loads shoulder_lift, so its exact
  trajectory legitimately shifted (`achieved_x_delta_m` 0.00668 -> 0.00597 m against a 0.01 m
  target over a 0.3s move — more friction-limited than before, the correct direction given
  friction was raised). Re-derived by re-running the exact command and updating the golden
  values + docstring, following the exact precedent this same test already established on
  2026-07-31 for the original frictionloss addition.
- `pytest -q -m unit` — 206/206 pass, unaffected (pure-numpy `controller_core` tests don't touch
  the MJCF).

## Honest limit: this does NOT close the sim-to-real gap

Re-ran the exact original evidence scenario (wrist2-offset pose, `split_base_wrist_task`,
`dx=0.02m`) with the corrected friction: **98.3% achieved in sim, essentially unchanged from
98.6% before the correction** — versus 59% on the real robot. A single-joint, ~22% Coulomb-value
correction is real and evidence-grounded, but far too small to explain the actual sim-to-real
severity gap on its own. The remaining gap most likely needs either: more calibration points
across other joints/poses (this session only produced one clean, unambiguous plateau), a
genuinely different friction model (this is a static Coulomb correction, not a stick-slip/
breakaway dynamic — LuGre and Karnopp were both explored earlier this session and found not to
help the truly-stuck case, see `docs/status/karnopp_stiction_friction_model_2026-08-02.md`), or
factors unrelated to friction entirely (e.g. sim's idealized zero-latency control loop versus
real RTDE timing/jitter effects — see the companion timing-injection work this same session).

## Not done here

- No change to `friction_ff_coulomb_nm` (the controller's own feedforward *compensation*
  coefficient, currently still 5.0 Nm for shoulder_lift, same source as the old frictionloss
  guess) — raising the plant's real friction value without also raising the controller's
  compensation estimate means the real robot would now be *even more* under-compensated at this
  joint than before. Worth a deliberate follow-up decision, not bundled into this change.
- No additional real-hardware data collection to get more calibration points on other joints.
- No dynamic/stick-slip model change — this is a refinement to the existing static Coulomb
  value only.
