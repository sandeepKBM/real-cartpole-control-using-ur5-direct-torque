# direct_torque_transport.py tail-latency investigation (2026-07-31)

Profiling/diagnosis only, no real hardware access. Grounded in the real
timing data from tonight's two lab runs (residual observer on vs off) and in
`hardware/direct_torque_transport.py`'s actual per-cycle loop, `hardware/
timing.py::TimingTracker`, `hardware/safety.py::DeadlineMonitor`, and
synthetic local benchmarks reproducing the loop's real object-creation
pattern (see `/tmp`-equivalent scratch scripts, numbers reproduced below).

## Real timing data under investigation

| metric | Run A (residual observer ON) | Run B (residual observer OFF) |
|---|---|---|
| work mean/p95/p99/max | 1.185/1.266/1.382/2.087 ms | 1.280/1.391/1.475/1.710 ms |
| sleep mean/p95/p99/max | 0.495/0.575/0.818/2.920 ms | 0.581/0.800/0.886/3.041 ms |
| lateness mean/p95/p99/max | 0.323/0.358/0.372/1.517 ms | 0.140/0.159/0.168/0.195 ms |
| outcome | DeadlineMonitor tripped (3 consecutive >1ms late) at ~4.8s | ran to ~7.2s, tripped an unrelated Y-drift guard |

`effective_deadline_ms` for this 500 Hz/2 ms-period loop is `min(3.0, 0.5 *
2ms*1000) = 1.0 ms` (`hardware/safety.py` `DeadlineMonitor` construction in
`direct_torque_transport.py`), matching "3 consecutive cycles >1ms late."

## 1. Why does MAX sit 3-4x above p99, and why does it differ Run A vs B?

Two distinct, code-grounded mechanisms, not one:

**(a) Lateness mean/p95/p99 are all systematically higher in Run A, not just
the max — this is the residual observer's real, measured cost never being
budgeted against the deadline.** In the loop, the diagnostic residual
observer (`residual_dynamics.coriolis()` + `predict_joint_acceleration` +
`residual_accel_estimator.update()`, lines ~433-444) runs **after**
`time.sleep()` has already been called and after `phases.record("sleep_ns",
...)` has already recorded that cycle's *computed* sleep target. Its cost
*is* measured (`residual_observer_ns`), but that measurement is purely
informational — nothing in the loop feeds it back into `next_deadline_ns` or
`sleep_ns`. The entire real cost of that block is added to how late the
*next* cycle's `cycle_start_ns` capture is, which shows up **only** as
`lateness_ns` on that next cycle. Since this runs every cycle (not
occasionally), it shifts the whole lateness distribution, not just the tail:
Run A's mean lateness (0.323 ms) is ~2.3x Run B's (0.140 ms) — almost exactly
the gap you'd expect from a real, always-on, unbudgeted per-cycle cost. This
also explains why Run A alone tripped `DeadlineMonitor`: 3 consecutive
cycles where that unbudgeted cost pushed lateness past the 1 ms threshold.

**(b) The MAX-vs-p99 outlier itself (present in `sleep`, in *both* runs,
independent of the residual-observer flag) is a deterministic, one-time
scheduling artifact on cycle 0, confirmed by direct calculation:**

```python
next_deadline_ns = monotonic_ns() + tracker.period_ns   # set once, before the loop
...
while ...:
    cycle_start_ns = monotonic_ns()
    lateness_ns = max(0, cycle_start_ns - next_deadline_ns)   # for cycle 0, ~0
    ...
    next_deadline_ns += tracker.period_ns   # unconditional += every cycle, including cycle 0
    sleep_ns = max(0, next_deadline_ns - cycle_end_ns)
```

`next_deadline_ns` is initialized to `now + period` *before* the loop, then
unconditionally incremented by `+= period` again at the bottom of cycle 0's
own body before `sleep_ns` is computed. So cycle 0 targets `now + 2*period`
instead of `now + period` — its sleep is genuinely ~1 extra period long.
Reproducing the exact arithmetic with realistic work numbers:

```
work=1.185ms (Run A mean) -> cycle0 sleep_ns=2.815ms  (steady-state would be 0.815ms)
work=1.280ms (Run B mean) -> cycle0 sleep_ns=2.720ms  (steady-state would be 0.720ms)
```

This lands almost exactly on both runs' reported sleep MAX (2.920 ms Run A,
3.041 ms Run B) — well within normal cycle-0-vs-mean work variance. It is a
one-time, self-contained artifact (verified it does not propagate: cycle 1's
`cycle_start_ns` lands back on-schedule afterward), present in every run of
this loop regardless of the residual-observer flag, and matches the "same
~3ms ballpark in both runs" pattern in the data. It is **not** GC, not RTDE,
not the residual observer — it is the deadline-init/increment arithmetic
double-counting one period on cycle 0 only.

Not fully resolved from available evidence: Run A's single 1.517 ms lateness
max (~4x its own already-elevated p99) — consistent with either an
occasional larger-than-typical residual-observer cost that cycle, or
ordinary OS-level scheduling preemption of the Python process (this repo's
control loops are not RT-kernel/RT-priority). Cannot be disambiguated further
without the actual `latency_phases.residual_observer` percentile breakdown
from that specific run, or real-hardware access — neither was available here.

## 2. Is `gc.disable()` fully effective? Refcounting cascades? List resize cost?

`gc.disable()` (already applied, `hardware/direct_torque_transport.py:268`)
only disables CPython's cyclic/generational collector — reference counting,
CPython's primary reclaim mechanism, is unconditional and cannot be disabled.
Quantified with a synthetic benchmark reproducing the loop's real per-cycle
object-creation pattern (the `trace_rows.append()` dict + ~11 `.tolist()`
calls + a `TimingSample` dataclass append, run 3600 times matching Run B's
real ~7.2s/3600-row trace length):

```
gc.disable() (matches real conditions): mean=8.9us  p99=16.3us  max=46.5us   (max/p99 = 2.85x)
gc left enabled (comparison only):      mean=9.3us  p99=41.0us  max=646.5us  (max/p99 = 15.8x)
```

With GC left enabled, outliers cluster at near-uniform ~648-650-cycle
intervals (646, 1294, 1942, 2590, 3238) — the signature of CPython's
generational gen0 threshold (default 700 allocations) periodically
re-scanning the growing tracked-container set, exactly the mechanism the
existing `gc.disable()` comment already identifies. **This is direct,
quantified confirmation that `gc.disable()` is real and effective**: it
removes a ~600+us periodic spike (>10x today's largest measured refcounting
outlier) that would otherwise recur roughly every 1.3s at 500 Hz. With
`gc.disable()` active (matching tonight's real runs), pure refcounting
churn tops out at 46.5us — three orders of magnitude below the ms-scale
lateness spikes observed live, so refcounting cascades are a real but small
effect already folded into ordinary per-cycle variance, not a hidden
explanation for the ms-scale outliers.

List over-allocation resize cost, isolated (pure `list.append()`, no dict
work, 4000 iterations, `gc.disable()`):

```
mean=171ns  p99=491ns  max=3710ns
```

Confirmed negligible (microseconds, not the "rare but real" concern the
question raised) — CPython's list growth factor keeps resize copies cheap
even as `trace_rows` reaches ~3500-3600 entries.

## 3. Other unmeasured contributors: `tracker.add_sample()` / `trace_rows.append()`

Confirmed real: both calls run *after* `time.sleep()` and after
`phases.record("sleep_ns", ...)`, so their cost was previously invisible to
`latency_phases` and leaked into the next cycle's `lateness_ns`, exactly like
the residual observer (§1a) but smaller in magnitude — bench-measured at
9-46us (gc-disabled), confirmed on a real (mocked-RTDE) run of this exact
code path post-fix: `tracker_sample` mean 0.017ms/p95 0.028ms/max 0.030ms,
`trace_append` mean 0.026ms/p95 0.043ms/max 0.058ms (150-cycle mocked run).
Real, but far too small (tens of microseconds) to be the dominant driver of
the observed ms-scale spikes — the residual observer (§1a, ~183us/cycle
*mean*, unbudgeted every cycle) is the larger, systematic unmeasured cost.

## 4. Fix implemented (low-risk, additive-only instrumentation)

**What**: added two new `PhaseLatencyRecorder` fields, `tracker_sample_ns`
and `trace_append_ns`, and wrapped the *existing* `tracker.add_sample()` and
`trace_rows.append()` calls in `monotonic_ns()` timers at their current,
unchanged position in the loop (`hardware/direct_torque_transport.py`,
`hardware/latency.py`). This makes both real-hardware evidence-backed costs
identified in §3 visible in `latency_phases` for every future run, instead of
requiring a synthetic benchmark to infer them.

**Why this and not more**: this is purely additive (two `monotonic_ns()`
calls + two `phases.record()` calls, the exact pattern already used 8 times
in this function) — it does not reorder anything, does not change what is
computed or when, does not touch `next_deadline_ns`/`sleep_ns`/
`DeadlineMonitor` math, and does not touch the control law or the torque
command (already sent earlier in the cycle via `link.direct_torque()`). New
fields are deliberately excluded from `dominant_phase`'s existing candidate
tuple in `hardware/latency.py`, so they cannot change that field's existing
output or interact with `tests/hardware/test_direct_torque_transport_timing.py`'s
fixed expected-value set.

**What was found but *not* fixed, flagged for human review instead**:
1. **The residual observer's real cost is structurally unbudgeted against
   the deadline** (§1a) — this is the dominant, systematic driver of Run A's
   elevated lateness and its `DeadlineMonitor` trip. The correct fix (moving
   the residual-observer computation, and/or the two bookkeeping calls, to
   before the sleep/deadline computation so their cost is subtracted from
   that cycle's sleep budget instead of leaking into the next cycle) is a
   real change to the loop's timing/scheduling semantics — it would change
   when `DeadlineMonitor` trips, which is safety-relevant control-loop
   timing logic, not pure instrumentation. Per this task's explicit
   instruction, that is reported here for a deliberate decision rather than
   implemented tonight.
2. **The cycle-0 double-period-sleep arithmetic bug** (§1b) — confirmed,
   deterministic, present every run, harmless in practice (one extra ~1
   period of stillness before the second control cycle; does not propagate;
   torque for cycle 0 is already sent before this code runs), but fixing it
   means editing `next_deadline_ns` init/increment logic in the live loop —
   also flagged rather than changed tonight.

## Tests

`/common/users/ss5772/miniforge3/envs/mujoco_ur5e/bin/python -m pytest -q
tests/ -m "not slow"` — **430 passed, 1 deselected** (matches the stated
baseline exactly, no regressions). Also spot-checked
`tests/hardware/test_direct_torque_transport_timing.py` and
`tests/hardware/test_latency.py` individually (2 passed), and ran a live
mocked-RTDE 150-cycle transport to confirm the new `tracker_sample`/
`trace_append` phases populate with real, sane numbers (above).

## Files changed

- `hardware/latency.py` — two new `PhaseLatencyRecorder` fields
  (`tracker_sample_ns`, `trace_append_ns`), included in `summary()`'s phases
  dict, excluded from `dominant_phase`'s candidate set.
- `hardware/direct_torque_transport.py` — wrapped the existing
  `tracker.add_sample()` and `trace_rows.append()` calls with
  `monotonic_ns()` timers at their unchanged position; no reordering, no
  control-law or deadline-math change.

## Rollback

`git revert <this commit>` (or `git checkout <prior-commit> --
hardware/latency.py hardware/direct_torque_transport.py`) — the change is
additive-only and isolated to those two files.
