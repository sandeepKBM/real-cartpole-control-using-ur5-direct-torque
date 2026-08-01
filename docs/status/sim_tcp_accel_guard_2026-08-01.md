# Sim-side TCP-accel/speed guard: port, ground truth, and smoke test (2026-08-01)

## Problem

Real UR5e hardware testing tonight at the height_alpha=0.5 zero-degree transport pose
repeatedly tripped `hardware/safety.py::CartesianMoveMonitor`'s TCP-acceleration guard across
several `direct_torque` runs (`accel_duration_scurve`/`min_jerk_move_hold` profiles), traced to
the arm sitting at the `wrist_2≈0` kinematic singularity. `tools/ur5e_mujoco_torque_experiments.py`
(the sim per-step engine) only ever called `adapter.safety_monitor.check(...)`
(`controller_core/safety.py::ImpedanceSafetyMonitor` -- drift/orientation/`|qd|`/axis-growth, no
TCP-speed/accel check at all), so every sim run of the same scenario completed cleanly -- not
because the controller is fine there, but because sim structurally could not detect this failure
mode. This doc covers porting the guard, validating it against real trace data, and running the
exact failing scenario in sim with the guard now active.

## What was read

- `hardware/safety.py`: `CartesianMoveLimits` (speed/accel ceilings, `accel_gap_cycles`/
  `speed_lowpass_alpha` noise filtering, `accel_max_consecutive_violations`/`accel_hard_multiple`
  graduated tolerance, `NOISE_ROBUST_GUARD_OVERRIDES`) and `CartesianMoveMonitor.check()` (gap-
  windowed low-pass-filtered finite-difference speed/accel estimate, off-axis drift, orientation,
  `|qd|`, waypoint jump, axis-error-growth checks). Confirmed by reading `check()` directly: it
  only ever reads `tcp_pose[:3]` (position) -- the orientation components (`tcp_pose[3:6]`) are
  never touched.
- `tools/ur5e_mujoco_torque_experiments.py`'s per-step loop: `post_safety =
  adapter.safety_monitor.check(...)` (previously the only guard call) and the
  `termination_reason`/trace-row/`summary.json` plumbing around it.
- `hardware/__init__.py` / `hardware/link.py`: confirmed the RTDE binding import
  (`rtde_control`/`rtde_receive`) is fully lazy, deferred to `_load_rtde_classes()` inside
  connection-opening methods, never at module import time. Verified empirically: importing
  `hardware.safety` pulls in zero RTDE-named modules (`sys.modules` diff before/after). So the
  import chain is clean.

## Design decision: import, but only when the flag is used

`hardware.safety.CartesianMoveMonitor`/`CartesianMoveLimits`/`NOISE_ROBUST_GUARD_OVERRIDES` are
reused verbatim via a **local import inside the opt-in code path only** (`_resolve_tcp_accel_guard_limits()`
and the `if bool(args.enable_tcp_accel_guard):` block in `run()`), not a top-level import. Two
reasons: (1) it is genuinely clean (no RTDE side effects), so reimplementing ~200 lines of
already-thoroughly-tested logic (`tests/hardware/test_hardware_safety.py`) would just be a second,
divergent copy to keep in sync; (2) the module's own docstring states "This runner ... never
imports hardware, RTDE, or URScript code" -- a local, flag-gated import keeps that literally true
for every default/existing run (nothing is imported unless `--enable-tcp-accel-guard` is passed),
while still getting reuse when a caller opts in.

## What was built

- `tools/ur5e_mujoco_torque_experiments.py`:
  - New flags, all default off/`None`: `--enable-tcp-accel-guard`, `--tcp-accel-guard-noise-robust`
    (applies `NOISE_ROBUST_GUARD_OVERRIDES`), and per-field overrides mirroring
    `tools/ur5e_direct_torque_x_transport.py`'s CLI (`--max-tcp-accel-mps2`, `--max-tcp-speed-mps`,
    `--accel-gap-cycles`, `--speed-lowpass-alpha`, `--accel-max-consecutive-violations`,
    `--accel-hard-multiple`, `--speed-max-consecutive-violations`, `--speed-hard-multiple`).
  - `_resolve_tcp_accel_guard_limits()`: same preset-then-explicit-override merge convention as
    the real-hardware CLI's `resolve_move_limit_overrides()`.
  - When enabled: `CartesianMoveMonitor.set_start()` is called once after `state0` is built, then
    `move_monitor.check()` is called every step right after the existing
    `adapter.safety_monitor.check(...)` call, fed `tcp_pose = concat(ee_pos, zeros(3))` (sim has no
    axis-angle orientation representation matching real `tcp_pose[3:6]`; since `check()` never
    reads those components -- confirmed above and locked down by
    `test_ee_pos_zero_padding_convention_check_only_reads_first_three` -- zero-padding is exact,
    not an approximation).
  - A trip is folded into the same `row["safety_ok"]`/`row["safety_reason"]`/`termination_reason`
    fields the existing guard uses, prefixed `"tcp_accel_guard: "` to match the real hardware's own
    message format (e.g. `"tcp_accel_guard: TCP acceleration 0.72 m/s^2 > 0.5 m/s^2 for 3
    consecutive cycles"`). No new summary schema -- `RunLogger`'s existing
    `_find_first_safety_violation()` (reads `row["safety_ok"] is False`) picks this up for free, no
    `observability/run_logger.py` changes needed. New keys (`tcp_accel_guard_enabled`,
    `tcp_accel_guard_limits`, `tcp_accel_guard_tripped` in `summary.json`;
    `tcp_accel_guard_ok`/`tcp_accel_guard_reason` per trace row) are added **only** inside the
    `if move_monitor is not None:` branches, so a run without the flag gets zero new keys anywhere.
  - Known minor gap, not addressed here: `transport_metrics.transport_failure_category()` has no
    dedicated `tcp_accel_guard` bucket, so a guard-tripped run's `failure_category` currently falls
    through to `"duration"` or `"other"` depending on the run's other metrics. A small, separate
    follow-up if this needs its own bucket for sweep-level reporting.
- `tests/mujoco/test_tcp_accel_guard.py` (9 tests, all passing): CLI-override-merge unit tests,
  the zero-padding-convention lock-down, a synthetic clean-then-spike sequence through the exact
  padding/call convention used in the integration (the base class's own spike/clean/gap/lowpass/
  consecutive-violation coverage already lives in `tests/hardware/test_hardware_safety.py` and
  isn't duplicated), and three `@pytest.mark.slow` end-to-end subprocess tests: disabled-by-default
  produces no new keys, a gentle enabled move doesn't trip, and a deliberately near-zero ceiling
  trips deterministically through the real per-step loop.

## Zero-regression check

Ran the same `controller-rollout` command (impedance, dx=0.01m, 0.05s, seed 0) both against the
pre-change file (`git stash`) and the post-change file with the new flag unset. `summary.json`
(excluding path fields, which differ only because they point at different tmp output dirs) and
every row of `trace.jsonl` (25 rows) compared **exactly equal**. Confirmed separately by the
`test_guard_disabled_by_default_produces_no_new_summary_keys` test.

## Ground-truth validation against real trace data

Pulled real trip traces from tonight
(`outputs/hardware_transport_remote/hardware_transport/direct_torque_20260801_*`). Key structural
finding first: `hardware/direct_torque_transport.py` appends to `trace_rows` **after**
`move_monitor.check()` passes and the torque command is sent -- so the cycle that actually trips
the guard is, by construction, never written to `trace.jsonl`. This is true of every real trip
trace pulled, not something introduced by this validation. The `pre_trip_trend` diagnostic
(`summary.json`) does capture the tripping cycle's own raw single-cycle TCP speed (its window is
appended *before* the guard check), but not the smoothed/gap-windowed accel value the guard
actually failed on.

Given that, two complementary checks:

1. **`direct_torque_20260801_180745`** (real `termination_reason`: `"TCP acceleration 0.8261 m/s^2
   > 0.5 m/s^2 for 3 consecutive cycles"`): replaying this run's own logged `tcp_pose` sequence
   through the **unmodified, imported** `CartesianMoveMonitor` with `NOISE_ROBUST_GUARD_OVERRIDES`
   (the preset the "for 3 consecutive cycles" message format implies was active) reproduces a real
   trip using only real recorded data -- `"TCP acceleration 0.6231 m/s^2 > 0.5 m/s^2 for 3
   consecutive cycles"` at replay step 92 of 96. Same order of magnitude, identical message
   pattern and mechanism. The replay trips a few cycles before the real robot's own recorded end
   of trace, plausibly because the real per-cycle `dt_s` fed into the monitor at runtime had jitter
   this offline replay (uniform nominal `dt_s`) cannot reproduce -- `CartesianMoveLimits`'
   `accel_gap_cycles`/`speed_lowpass_alpha` fields exist precisely because this class of finite-
   difference noise is `dt`-sensitive.
2. **`direct_torque_20260801_182256`** (real: `"TCP acceleration 0.7214 m/s^2 > 0.5 m/s^2 for 3
   consecutive cycles"`, the exact run quoted in the task) and **`direct_torque_20260801_175602`**
   (real: `"TCP acceleration 0.6918 m/s^2 ..."`): replaying the full available trace with the same
   preset does not itself cross the trip threshold (expected, since the triggering cycle is
   structurally absent -- see above), but the consecutive-violation counter builds to 2/3 in the
   final few available cycles of both traces, i.e. one more violating cycle (exactly the missing,
   untraced one) would trip it. This is consistent with, not contradicting, the real outcome.

Net: the ported class, unmodified, computes accelerations in the same regime (order 0.5-1 m/s^2,
same graduated-tolerance message shape) as the real guard did on real data, and the one case with
a complete-enough trace to fully replay does reproduce an actual trip. Exact bit-for-bit
reproduction of the quoted 0.7214 number is not possible from any trace in this repo, for the
structural reason above, not a defect in the port.

## Smoke test: tonight's exact failure scenario, in sim, with the guard now active

Command: `--mode controller-rollout --controller-kind impedance --gravity-mode gravity_comp
--config config/ur5e_mujoco_torque_osc_tuned_wrist_orient_friction_ff.yaml --trajectory-profile
accel_duration_scurve --target-accel 0.02 --move-duration 4.0 --duration 6.0 --start-q-rad 0.0
-0.835398 -1.2 -0.985398 0.0 0.0 --enable-tcp-accel-guard --tcp-accel-guard-noise-robust`.

**Result: the guard did not trip.** `success: true`, `termination_reason: duration_complete`,
`tcp_accel_guard_tripped: false`, full 3000-step/6s rollout. Recomputing the *raw, unfiltered*
(gap=1, no low-pass) single-cycle TCP acceleration and speed directly from the trace's `ee_pos`
sequence -- i.e. ignoring the noise-robust smoothing entirely, the most sensitive possible view --
gives a peak of **0.34 m/s^2** and **0.045 m/s**, both still under the 0.5 m/s^2 / 0.05 m/s
ceilings. There is no spike anywhere in the trajectory, smoothed or raw.

**This is not a guard-implementation gap** -- the same tight-ceiling smoke test in
`tests/mujoco/test_tcp_accel_guard.py` confirms the ported guard trips deterministically and
immediately given a real position discontinuity in this exact code path. The guard is fully
capable of detecting this failure class in sim now; sim's own dynamics at this pose/profile simply
never produces the transient real hardware does.

This is consistent with, and independently corroborates, the same-night finding already recorded
in `AGENTS.md` section 3 (LuGre item): real hardware testing of `accel_duration_scurve`/
`accel_duration_triangular` found a real stick-slip breakaway signature (measured accel spiking to
~3.3-3.5x commanded, ~0.37-0.38s into the move) that the sim's current friction model --
`friction_feedforward`'s static tanh/viscous form, with no memory of "how long has this joint been
stuck" -- cannot represent. The guard now exists and works; the sim-vs-real dynamics-fidelity gap
that keeps it from firing on this scenario is a separate, already-identified, unimplemented
problem (`docs/hardware/LUGRE_FRICTION_MODEL_PLAN.md`), not something this change can or should
paper over.

## Files changed

- `tools/ur5e_mujoco_torque_experiments.py` -- new opt-in flags + `_resolve_tcp_accel_guard_limits()`
  + per-step guard wiring, all additive and default-off.
- `tests/mujoco/test_tcp_accel_guard.py` -- new, 9 tests.
- `docs/status/sim_tcp_accel_guard_2026-08-01.md` -- this file.

No changes to `hardware/safety.py`, `controller_core/`, or any existing config.

## Tests run

- `pytest tests/mujoco/test_tcp_accel_guard.py -v` -- 9/9 passed (6 fast + 3 `slow`).
- `pytest tests/mujoco/test_ur5e_mujoco_torque.py -q -m "not slow"` -- 28/28 passed (no
  regression in the file this change touches).
- `pytest -q -m "not slow"` (full repo) -- 515 passed, 5 deselected, 3 xfailed, zero failures.
- Manual zero-regression diff (`git stash` baseline vs. post-change, flag unset): summary.json
  and trace.jsonl byte-identical except output-path fields.

## Tests not run

- Full repo suite including `slow`-marked tests (`python -m pytest -q`, no `-m` filter) -- not
  run in full given time/scope; the `slow` tests in the new file specifically were run and passed
  (see above).
- Real-hardware validation -- out of scope (sim-only task, no hardware access).

## Rollback

`git checkout -- tools/ur5e_mujoco_torque_experiments.py && rm tests/mujoco/test_tcp_accel_guard.py
docs/status/sim_tcp_accel_guard_2026-08-01.md`
