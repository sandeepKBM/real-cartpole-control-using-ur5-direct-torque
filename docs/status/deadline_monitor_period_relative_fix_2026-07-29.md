# DeadlineMonitor period-relative cap for direct_torque (Fix 2a) — 2026-07-29

Context: `docs/status/timing_safety_gaps_audit_2026-07-28.md` traced a real
hardware incident (`direct_torque` mode, 500 Hz / 2 ms period) where 4-of-5
cycles ran late by up to ~2 ms (total cycle time up to ~2x the nominal 2 ms
period), yet `DeadlineMonitor` never tripped. Root cause, confirmed by that
doc's reading of the code and by
`test_deadline_monitor_ignores_clean_cycles`/
`test_deadline_monitor_tolerates_isolated_transient_overrun`: the flat
`UR5eSafetyLimits.max_deadline_ms = 3.0` default is calibrated for the three
125 Hz loops (8 ms period, a real fraction of it) but is ~150% of the 500 Hz
loop's entire 2 ms period, so a ~2 ms overrun there never even registers as
"late." The doc proposed three fixes (2a/2b/2c) and ranked 2a highest
priority as the minimal, targeted one. **This change implements Fix 2a
only.** Fix 2b (sliding-window overrun-rate trip) and Fix 2c (fail-closed
staleness when `robot_timestamp_s` is unavailable) remain open future work —
2b in particular needs dedicated attention because of a real interaction
risk with `test_deadline_monitor_tolerates_isolated_transient_overrun`'s
exact-50%-alternating overrun/clean pattern, which the source doc flagged as
not yet resolved.

`DeadlineMonitor` itself (`hardware/safety.py`) needed no changes — it
already accepts an arbitrary float threshold via its constructor. The gap
was entirely at the `direct_torque_transport.py` call site.

## The fix

1. `hardware/safety.py` — added a new `UR5eSafetyLimits` field:
   `max_deadline_fraction_of_period: float = 0.5`, extended into the
   existing positive/finite-float validation loop in `validate()` alongside
   `max_deadline_ms` and the other float fields (same style, no new
   validation path).

2. `hardware/direct_torque_transport.py` — at the `DeadlineMonitor`
   instantiation (previously `DeadlineMonitor(safety_limits.max_deadline_ms)`),
   changed to:
   ```python
   effective_deadline_ms = min(
       safety_limits.max_deadline_ms,
       safety_limits.max_deadline_fraction_of_period * dt_s * 1000.0,
   )
   deadline_monitor = DeadlineMonitor(effective_deadline_ms)
   ```
   `dt_s` is the loop's already-computed real period (`1.0 / frequency_hz`,
   asserted ~500 Hz earlier in the same function) — no new period
   computation was added.

## Before / after behavior

- **Before**: `direct_torque`'s `DeadlineMonitor` always used the flat 3.0 ms
  default regardless of loop period. A ~2 ms overrun (this loop's own
  period) sat under that floor and was invisible to the monitor.
- **After**: at `dt_s = 0.002` (500 Hz), `effective_deadline_ms =
  min(3.0, 0.5 * 2.0) = 1.0` ms — the reported incident's ~2 ms overruns now
  register immediately, and a sustained pattern trips within a few cycles
  (`max_consecutive_overruns` default 3).

## No-op verification for period ≥ 6 ms

By construction: `max_deadline_fraction_of_period * dt_s * 1000.0 ≥
max_deadline_ms` exactly when `0.5 * period_ms ≥ 3.0`, i.e. `period_ms ≥
6.0`. For any such period, `min(3.0, ...)` always resolves to the unchanged
`3.0` ms default — so this file's own behavior is provably unchanged for any
hypothetical loop period ≥ 6 ms, and only actually tightens the threshold
for the real 2 ms/500 Hz case. The three 125 Hz loops
(`position_transport.py`, `urscript_transport.py`, `motion.py`, 8 ms period)
don't go through this code path at all — each instantiates its own
`DeadlineMonitor` in its own file, untouched by this change — so they are
completely unaffected, both by construction (different files) and by the
formula (8 ms ≥ 6 ms floor anyway).

## Tests

Added to `tests/hardware/test_deadline_and_staleness.py`:

- `test_deadline_monitor_trips_on_reported_incident_shape` — constructs
  `DeadlineMonitor(1.0)` (the new effective 500 Hz cap) and feeds it the
  reported incident shape (4-of-5 cycles late, overruns up to ~2 ms,
  repeating 5-cycle blocks); asserts it now trips within the first block
  (`cycles_run <= 5`).
- `test_max_deadline_fraction_of_period_config_wiring` — direct unit test of
  the `min()` formula: `UR5eSafetyLimits(max_deadline_ms=3.0,
  max_deadline_fraction_of_period=0.5)` combined with `dt_s=0.002` (500 Hz)
  yields the tightened 1.0 ms cap; combined with `dt_s=0.008` (125 Hz)
  yields the unchanged 3.0 ms default.

All pre-existing tests in that file pass unchanged, including
`test_deadline_monitor_ignores_clean_cycles` and
`test_deadline_monitor_tolerates_isolated_transient_overrun` (constructed
directly with explicit values, independent of the call-site change).

## Test results

- `tests/hardware/test_deadline_and_staleness.py`: 21/21 passed (was 19
  before the two new tests were added).
- `tests/hardware/` full suite: 173 passed, 1 pre-existing unrelated failure
  (`test_direct_torque_transport_timing.py::test_transport_records_timing_and_deadline_loop`
  — a `dominant_phase` assertion that doesn't include `"local_dynamics"` in
  its expected set; confirmed pre-existing by running the identical test
  against the unmodified code via `git stash`, same failure). A handful of
  other pre-existing failures in that directory
  (`test_suggest_gains.py`, `test_direct_torque_coriolis.py`,
  `test_direct_torque_gain_overrides.py`, etc.) are environment/dependency
  issues (missing `optuna`, unrelated fixture setup) reproducible on
  unmodified code, not caused by this change.
- `pytest -m unit`: 94 passed, 259 deselected (unaffected, as expected —
  this change touches only the hardware lane).

## Files changed

- `hardware/safety.py`
- `hardware/direct_torque_transport.py`
- `tests/hardware/test_deadline_and_staleness.py`
- `docs/status/deadline_monitor_period_relative_fix_2026-07-29.md` (this file)

## Still open (not implemented here, per original doc's scoping)

- **Fix 2b** — sliding-window overrun-rate trip in `DeadlineMonitor` itself,
  to close the "non-consecutive overrun rate" hole (a bad pattern can evade
  the consecutive counter if a clean cycle lands mid-streak). Needs careful,
  dedicated attention to its interaction with
  `test_deadline_monitor_tolerates_isolated_transient_overrun`'s exact
  50%-alternating pattern before implementation.
- **Fix 2c** — fail-closed `StaleStateMonitor` mode when
  `robot_timestamp_s` is unavailable during motion. Lower priority; no
  evidence real UR5e RTDE ever lacks `getTimestamp()`.
