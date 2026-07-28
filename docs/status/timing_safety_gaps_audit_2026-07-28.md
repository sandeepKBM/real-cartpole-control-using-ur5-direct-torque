# Timing/staleness safety-gap audit — 2026-07-28

Context: AGENTS.md §4's "Found, not yet fixed" list (written after the 2026-07-25 audit)
names two gaps — `max_deadline_ms` never enforced, and no cycle-to-cycle staleness
detection. Today a real hardware run in `direct_torque` mode hit a genuine overrun (4/5
cycles late, up to 2x the nominal 2ms period) that neither gap caught. This doc audits the
CURRENT code (not the AGENTS.md text, which is stale on this point) and finds: **both
mechanisms already exist and are wired into all four motion loops** (commit `85498a0`,
2026-07-25, same day as the audit — AGENTS.md was apparently never updated after that
commit landed). The real, still-open problem is narrower and more precise than "never
enforced": `DeadlineMonitor` is wired in everywhere but its threshold/consecutive-trip
design is, **by explicit test-asserted intent**, incapable of catching today's exact
pattern. Staleness detection (`StaleStateMonitor`) is unrelated to today's incident and
appears to work as designed.

## 1. `max_deadline_ms` enforcement — current state

**Not actually unenforced.** `hardware/safety.py:187-271` defines `DeadlineMonitor`, and it
is instantiated and called every cycle in all four motion loops:

- `hardware/direct_torque_transport.py:170,184-188` — `deadline_monitor.record(lateness_ns)`
  called at the top of the loop, before `read_state()`; on trip, `estop.trip(...)` + break.
- `hardware/position_transport.py:183,279-283` — called after the per-cycle work, fed
  `overrun_ns = max(0, elapsed_s - dt_s)`.
- `hardware/urscript_transport.py:214,287-292` — same pattern, inside the Python
  supervisor thread (the control law itself runs on-robot in URScript; the supervisor is
  what can notice its own budget or the telemetry has stalled).
- `hardware/motion.py:150` — same pattern for the `servoL` bounded-move CLI path.

All four read `safety_limits.max_deadline_ms` — either from `link.limits`
(`UR5eSafetyLimits`, default `3.0` ms, `hardware/safety.py:94`) or a bare
`UR5eSafetyLimits()` — and none of them expose a CLI flag or config key to override it
(confirmed: no `max_deadline_ms` hit in `tools/*.py` outside the archived
`hardware_rtde_v1` lane). So every loop uses the same flat **3.0 ms** default regardless of
its own control period (2 ms @ 500 Hz for `direct_torque`; 8 ms @ 125 Hz for
`position`/`urscript`/`motion`).

### Why today's incident wasn't caught — traced precisely, not assumed

`DeadlineMonitor.record()` (`hardware/safety.py:250-271`) has two trip conditions:
1. A **single** cycle overrunning by `>= hard_overrun_multiple * max_deadline_ms` (default
   `5.0 * 3.0 = 15 ms`) trips immediately.
2. `max_consecutive_overruns` (default `3`) cycles **in a row**, each overrunning by
   `> max_deadline_ms` (3.0 ms), trips. Any single clean cycle (`overrun_ns <=
   max_deadline_ns`) resets the streak to 0 (`safety.py:269-270`).

Today's incident: 4/5 cycles late, "up to 2x" the nominal 2 ms period — i.e. total cycle
time up to ~4 ms, so the *overrun* (work/lateness beyond the 2 ms period) tops out around
**2 ms**. That is below the 3.0 ms `max_deadline_ms` threshold on every single cycle, so
`overrun_ns > self._max_deadline_ns` is `False` throughout and `_consecutive_overruns`
never leaves 0 — the monitor has nothing to count, let alone trip on. This isn't a
hypothesis: `tests/hardware/test_deadline_and_staleness.py:57-61`
(`test_deadline_monitor_ignores_clean_cycles`) asserts exactly this — a 2.9 ms overrun is
explicitly treated as "clean" by design, forever. And even if the overrun did cross 3.0 ms
on some cycles, `test_deadline_monitor_tolerates_isolated_transient_overrun`
(lines 64-70) asserts that a 4 ms overrun interleaved with clean cycles must **never** trip,
"even if this repeats forever" — which is structurally the same shape as a 4-out-of-5
pattern unless the late cycles happen to land 3-in-a-row. So the monitor is working exactly
as coded and tested; the calibration (a flat 3.0 ms absolute tolerance, chosen once and
reused for a loop whose entire nominal period is 2 ms — i.e. 150% of period, requiring
~250% of period before an overrun even registers) is what's wrong for the 500 Hz
`direct_torque` loop specifically.

Note also (`direct_torque_transport.py:161-188,291-313`): the value fed to
`deadline_monitor.record()` is **start-lateness carried into the next cycle**
(`cycle_start_ns - next_deadline_ns`), not that cycle's own `work_ns` (computed later at
line 292 and logged into the trace as `cycle_work_ms`/`lateness_ms`, and into
`tracker.add_sample`, but never passed to `deadline_monitor`). Because `next_deadline_ns`
advances by a fixed `period_ns` every iteration regardless of overrun (no deadline reset), a
single cycle's work overrun *does* eventually surface as next-cycle lateness — so this
isn't a wiring bug, just a one-cycle-lagged measurement, worth knowing when reading the
data but not the root cause of the miss.

### The data existed, just wasn't acted on

`hardware/timing.py`'s `TimingTracker` (used by `direct_torque_transport.py`) already
computes `late_cycles`, `max_consecutive_late_cycles`, `max_lateness_ms`,
`overrun_count_threshold_ns`, etc. (`timing.py:129-159`), stored under
`summary["timing"]` (`direct_torque_transport.py:374`) and written to `summary.json`. So
today's overrun pattern is almost certainly visible after the fact in `summary.json`'s
`timing` block — but: (a) this hardware-lane driver does **not** go through
`observability/run_logger.py::RunLogger` — only `tools/ur5e_move_hold_transport.py` does
(confirmed by grep across `hardware/*.py` and the direct-torque/position/urscript CLI
entrypoints) — so it isn't part of the standard `run_record.json` contract AGENTS.md's
Observability section describes, and (b) nothing reads or acts on it live; it's purely a
post-hoc artifact, not a guard.

## 2. Staleness detection — current state

**Also already implemented and wired**, and appears unrelated to today's incident
(no evidence a frozen stream occurred; this is audited for completeness/correctness).

- `ConnectionHealth` (`hardware/safety.py:139-184`): tracks `record_success()` /
  `record_failure()` / `is_alive()`. `record_success()` **is** called on every successful
  `read_state()`, in both `hardware/link.py:213` and `hardware/direct_torque_link.py:129` —
  so health bookkeeping does happen during motion, contrary to a literal reading of the
  AGENTS.md line. But `record_failure()` and `is_alive()` are still called from exactly one
  place in the whole repo: `tools/ur5e_connect.py:69-70` (the `--watch` idle loop). No
  motion loop calls either. In practice this doesn't matter much: every motion loop instead
  catches `RTDEStateError` directly and calls `estop.trip(...)` immediately (fail-fast on
  first exception, e.g. `direct_torque_transport.py:336-338`,
  `position_transport.py:212-215`), which is stricter than `ConnectionHealth`'s
  N-failures-before-trip grace window — so `ConnectionHealth`'s consecutive-failure counting
  is simply unused outside the watch loop, not a live safety hole.

- `hardware/link.py::UR5eLink.read_state()` (lines 159-214): raises `RTDEStateError` on
  exception, wrong shape, or NaN/Inf. It reads `robot_timestamp_s` via `getTimestamp()`
  (lines 187-193) but **never compares it to anything** — no wall-clock check, no
  cycle-to-cycle diff. Its "never stale" guarantee is purely the raise-on-exception case, as
  AGENTS.md describes.

- `StaleStateMonitor` (`hardware/safety.py:274-339`) is the actual fix: compares
  `robot_timestamp_s` across cycles against the host clock, trips after
  `max_frozen_cycles` (default 5) consecutive reads where the host clock advances but the
  robot clock doesn't. It is called every cycle in all four loops:
  `direct_torque_transport.py:171,196-200`, `position_transport.py:184,219-223`,
  `urscript_transport.py:215,227-231` (inside the supervisor thread), `motion.py:151,173-183`.

**Concrete trace for `direct_torque` mode** (the mode the task asks to trace explicitly): if
`ur_rtde` returned the same stale `robot_timestamp_s` every call without raising, `q`/`qd`/
`tcp_pose` would still come back numeric (not NaN) from the frozen buffer, so
`read_state()` would keep succeeding. But `stale_monitor.record(link_state.robot_timestamp_s,
link_state.host_stamp_ns)` at line 196 would see the robot timestamp fail to advance for 5
consecutive host-clock-advancing reads and return a `stale_state:` reason at line 197-200,
triggering `estop.trip(...)` and breaking the loop within ~5 cycles (~10 ms at 500 Hz). **So
yes, this specific stalled-but-non-raising scenario is currently caught** in the
`direct_torque` loop, and by the identical mechanism in the other three. The one residual
caveat: if the robot/simulator doesn't expose `getTimestamp()` at all
(`robot_timestamp_s=None`), `StaleStateMonitor.record()` treats that as "can't verify" and
never trips (lines 316-322) — a real gap only for a backend that lacks the robot clock
entirely (real UR5e RTDE exposes it; not confirmed here whether URSim's mock always does).

## 3. Concrete fix proposals

### Fix 2a (higher priority) — make `DeadlineMonitor`'s threshold period-relative for `direct_torque`

Today's miss is a calibration problem in one specific loop (500 Hz / 2 ms period), not an
absence of the mechanism. Minimal, scoped, non-silent fix, following the existing pattern
of `max_tcp_accel_mps2_override` (explicit opt-in, one file, documented) already used in
this same file:

- In `hardware/direct_torque_transport.py`, at line 170, replace
  `deadline_monitor = DeadlineMonitor(safety_limits.max_deadline_ms)` with a period-aware
  cap, e.g.:
  ```python
  # 500 Hz => 2 ms period; the flat 3.0 ms UR5eSafetyLimits default tolerates up to a
  # 250%-of-period cycle before counting as an overrun at all (see
  # tests/hardware/test_deadline_and_staleness.py's ignores_clean_cycles /
  # tolerates_isolated_transient_overrun tests for why that's intentional at 3.0 ms in
  # isolation) -- too loose for this loop's own budget. Cap to half the period here only;
  # the other three (125 Hz / 8 ms) loops keep the flat default unchanged.
  effective_deadline_ms = min(safety_limits.max_deadline_ms, 0.5 * dt_s * 1000.0)
  deadline_monitor = DeadlineMonitor(effective_deadline_ms)
  ```
  This makes the tolerance 1.0 ms for the 2 ms loop (a 3 ms total cycle trips the
  consecutive path; today's up-to-4ms/2ms-overrun cycles now exceed 1.0 ms and start
  counting immediately).
- Add a `UR5eSafetyLimits` field, e.g. `max_deadline_fraction_of_period: float = 0.5`, so
  the ratio is a named, tested constant rather than a magic `0.5` inline, and the other
  three loops could opt in later without inventing a second convention.
- New test in `tests/hardware/test_deadline_and_staleness.py`: feed the monitor
  `DeadlineMonitor(1.0)` (i.e. the post-fix 2 ms-loop value) the exact reported shape — a
  4/5-late, up-to-2ms-overrun sequence — and assert it now trips within a few cycles.

### Fix 2b — close the "non-consecutive overrun rate" hole in `DeadlineMonitor` itself

Independent of 2a: even at a tighter threshold, `DeadlineMonitor`'s only two trip
conditions are "N in a row" or "one huge one." A sustained-but-not-contiguous bad pattern
(exactly today's 4-of-5, if the good cycle lands mid-streak) can still evade the
consecutive counter indefinitely — `test_deadline_monitor_tolerates_isolated_transient_overrun`
proves this is by design today. Add a sliding-window rate trip, reusing the `deque` already
imported in `hardware/safety.py:30`:

- `DeadlineMonitor.__init__` (`hardware/safety.py:225-244`): add parameters
  `overrun_rate_window: int = 10, max_overrun_rate: float = 0.5`, and
  `self._recent_overruns: deque[bool] = deque(maxlen=overrun_rate_window)`.
- `DeadlineMonitor.record()` (`hardware/safety.py:250-271`): after the existing
  consecutive-count update (and before the final `return None`), append
  `overrun_ns > self._max_deadline_ns` to `self._recent_overruns`, and if the deque is full
  and `sum(self._recent_overruns) / len(self._recent_overruns) > self.max_overrun_rate`,
  return a new `f"deadline_overrun: {sum(...)}/{len(...)} of last cycles exceeded
  max_deadline_ms"` reason — same `estop.trip(...)` handling at every call site, no loop
  changes needed since all four already treat any non-`None` return as fatal.
- Test: alternating overrun/clean cycles at a rate above `max_overrun_rate` must trip; below
  it must not (guards against over-tripping on genuinely intermittent, healthy jitter).

### Fix 2c (lower priority, staleness) — trip fail-closed when `robot_timestamp_s` is unavailable during motion

Since `StaleStateMonitor.record(None, ...)` currently means "can't verify, never trips"
(`hardware/safety.py:316-322`), and this is meant to run during live-torque motion where a
stalled stream is exactly the danger case: add an opt-in stricter mode,
`StaleStateMonitor(..., require_robot_clock: bool = False)`; when `True` and
`robot_timestamp_s` is `None`, trip after `max_frozen_cycles` cycles instead of resetting
silently. Leave the default `False` (current behavior, since URSim may not expose
`getTimestamp()` and this must not break simulator-only bring-up). Lower priority because
there's no evidence real UR5e RTDE ever lacks `getTimestamp()`, and this isn't what caused
today's incident.

## 4. Priority recommendation

**Fix 2a first.** Today's real incident was a deadline-overrun pattern that the *already-
shipped* `DeadlineMonitor` structurally cannot catch at its current threshold, in the exact
mode (`direct_torque`, live torque on real hardware) where an undetected sustained overrun
is most dangerous — stale-relative-to-clock torque commands, not a `servoL` waypoint a
lower-level controller can reject. It's the smallest possible change (one `min()` at one
call site, scoped to the one loop that had the incident, leaving the other three loops'
proven-fine defaults untouched — consistent with AGENTS.md's "add new named configs instead
of mutating shared ones" and "don't combine controller and timing changes" rules) and it
directly targets the measured failure (overruns topping out around 2 ms, under the current
3 ms floor).

Fix 2b is real and should follow soon after — it's a second, independent way the *current*
design (even fixed by 2a) can still be defeated by a lucky good cycle breaking up a bad
streak — but it's a genuine design addition (new field, new state, new test matrix for
false-positive risk on healthy jitter), not a one-line calibration fix, so it's correctly
sequenced second.

Fix 2c is lowest priority: staleness detection already works for the concrete "frozen RTDE
stream" scenario the AGENTS.md gap described (traced precisely above), it wasn't implicated
in today's incident, and its remaining edge case (`robot_timestamp_s is None`) has no
evidence of occurring on real hardware in this project.

## Files read (no code modified — this is a read-only investigation)

- `hardware/safety.py` (full)
- `hardware/direct_torque_transport.py` (full)
- `hardware/position_transport.py` (full)
- `hardware/urscript_transport.py` (full)
- `hardware/motion.py` (lines 100-220)
- `hardware/link.py` (lines 155-220, plus grep of full file)
- `hardware/timing.py` (full)
- `observability/run_logger.py` (lines 300-345)
- `tests/hardware/test_deadline_and_staleness.py` (lines 1-100)
- `git log`/`git show` on `hardware/safety.py` and the four transport files
