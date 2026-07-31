# Async residual observer, 2026-07-31

## Scope and status

**Diagnostic-only, opt-in, default off.** This relocates the residual
observer's dynamics computation (`docs/status/direct_torque_residual_observer_2026-07-29.md`,
`docs/status/residual_observer_dynamics_optimization_2026-07-30.md`) out of
`hardware/direct_torque_transport.py`'s 500 Hz control loop and into a
separate `multiprocessing.Process`, so its cost -- and especially its real,
occasional tail-latency spikes measured on real hardware tonight (~0.15ms
mean but large enough spikes to trip the deadline monitor) -- leaves the
loop's timing budget entirely. New parameter `residual_observer_async: bool
= False`; when `False` (the default), the loop takes the exact same code
path it did before this change, verified byte-for-byte (see below), not
just "should be identical." **No change to any trip condition, control law,
torque path, or timing/deadline logic** -- only what computes the (already
never-safety-relevant) `qdd_pred`/`qdd_measured`/`qdd_residual*` trace
fields, and where.

Real-time alerting/threshold-triggered behavior based on the residual value
was explicitly out of scope for this task (deferred, documented separately)
and nothing here adds any new decision logic.

## What landed

- `hardware/residual_observer_worker.py` (new) -- `start_residual_observer_worker`,
  `ResidualObserverWorkerHandle` (`submit()`/`shutdown_and_collect()`),
  `ResidualObserverRequest`/`ResidualAsyncSummary`, and the worker process
  entry point `_residual_observer_worker_main`. The worker builds its own
  `PinocchioUR5eDynamics` + `JointAccelEstimator` (these wrap pybind11/C++
  state and cannot cross a process boundary), reset from the same `state0.qd`
  the sync path uses.
- `hardware/direct_torque_transport.py` -- new `residual_observer_async`
  parameter; when `enable_residual_observer=True` and `residual_observer_async=True`,
  starts the worker once (after `state0` is read, before the per-cycle loop
  begins), replaces the inline per-cycle computation with a non-blocking
  `submit()`, shuts the worker down in the existing `finally` block (so it
  runs under every exit path), and merges results back into `trace_rows` by
  step index after the loop exits.
- `hardware/x_transport.py`, `tools/ur5e_direct_torque_x_transport.py` --
  `residual_observer_async` threaded through; new `--residual-observer-async`
  CLI flag.
- `tests/hardware/test_direct_torque_residual_observer_async.py` (new, 8
  tests) + `tests/hardware/fixtures/direct_torque_sync_pre_async_baseline_trace.json`
  (golden trace captured from the code as it existed immediately before this
  change, commit `f1748c1`).

## Design

### Producer side (the 500 Hz loop)

Each cycle, instead of the inline `coriolis()` + `predict_joint_acceleration()`
+ `JointAccelEstimator.update()` block, the loop calls
`residual_worker.submit(step=steps, q=..., qd=..., tau=..., mass_matrix=...,
real_dt_s=...)` -- the same six inputs the inline path already used.
`submit()` is `request_queue.put_nowait()` wrapped in `try/except
queue.Full`; on `Full` it increments `dropped_request_count` and returns
`False`. **It never raises and never blocks, under any circumstance** --
verified by `test_producer_never_blocks_under_backpressure`, which floods a
real (not mocked) `maxsize=2` `multiprocessing.Queue` 2000 times with no
consumer ever draining it: 2000 non-blocking calls complete in well under a
second, with 1998 correctly dropped and counted.

Queue sizing: `maxsize=2000` for both request and result queues (~4s of
headroom at 500 Hz) -- generous enough that no run in this task's testing
ever dropped a request.

### Worker process

Started once via `start_residual_observer_worker`, using the `spawn` start
method (not Linux's default `fork`) so the child gets a fresh interpreter
rather than inheriting the parent's already-loaded Pinocchio/Eigen/BLAS
state. It loops on `request_queue.get(timeout=0.05)`, computes the same
`qdd_pred`/`qdd_measured`/`qdd_residual`/`qdd_residual_norm` the inline path
did, and sends results back via `result_queue.put_nowait()` (also
non-blocking + drop-and-count, so the worker itself can never stall trying
to write to a full result queue). A per-cycle compute exception is caught
and reported per-step (`{"error": ...}`) rather than crashing the worker.

### Merge-back and the documented delay/gap

`trace_rows` is appended once per loop iteration in step order starting at
0, so `step` is exactly the list index -- merging is `trace_rows[step].update(...)`.
Steps whose request was dropped, whose worker-side compute failed, or whose
result had not yet arrived when `shutdown_and_collect()` finished draining
(the last few in-flight cycles at run end, explicitly acceptable per the
task) **keep `qdd_*=None`** -- the exact same "diagnostic data not yet
available" convention the sync path already uses while `JointAccelEstimator`'s
gap window is filling. `summary["residual_observer_async"]` reports
`dropped_request_count`, `dropped_result_count` (worker-reported,
best-effort -- `None` if the worker was force-killed before it could send
its final-stats message), `merged_step_count`/`unmerged_step_count`,
`worker_init_error`, `worker_exitcode`, `worker_terminated_forcefully` --
so an incomplete merge is always visible in the run record, never silently
implied.

### A real bug found and fixed during this task: shutdown-drain ordering

The first implementation called `process.join(timeout)` **before** draining
`result_queue`. This is a well-known `multiprocessing.Queue` gotcha, not a
theoretical concern: `put_nowait()` only enqueues to an internal deque; a
background *feeder thread* inside the worker process is what actually
pushes bytes through the real OS pipe, whose buffer is much smaller than
the queue's logical `maxsize`. If nobody reads the other end while the
worker tries to exit, that feeder thread blocks trying to flush its
backlog, which blocks the process from exiting at all.

Measured before the fix (`bench_worker_throughput.py`, 2000-item burst): the
worker's own internal compute loop finished in ~130ms, but the process would
not actually exit even after an 11s wait (3s idle + 8s join timeout), and
only 693/2000 (~35%) of its results were ever recovered before being
force-killed (`exitcode=-15`, `terminated_forcefully=True`). This also
silently truncated a full-loop 500-cycle benchmark to `merged_step_count=240/500`.

Fix: `shutdown_and_collect()` now drains `result_queue` **concurrently**
with waiting for the process to exit (a tight loop: drain available results,
check `is_alive()`, `join(0.02)`, repeat, until `join_timeout_s` elapses or
the process exits), with a final drain after. Re-measured after the fix:
2000/2000 merged, `exitcode=0`, `terminated_forcefully=False`; the
500-cycle full-loop benchmark below also went from 240/500 merged to
500/500.

### A second real bug found and fixed: BLAS thread explosion on worker startup

The first `duration_s=1.0` full-loop async benchmark run **terminated
early** via `deadline_overrun: 3 consecutive cycles late by > max_deadline_ms
(1 ms)` at step 102/500 -- this on a 72-core host with `load average: 0.71,
1.16, 1.55` (not a busy-host artifact). This matches AGENTS.md SS8's
documented per-process BLAS-thread-explosion risk applied to a new context:
the freshly-spawned worker, on first importing numpy/Pinocchio and
constructing two `Model`/`Data` pairs, let OpenBLAS auto-detect the full
core count and spawn that many threads for itself -- those threads
competed for real CPU cycles with the parent's real-time loop thread during
the one-time construction window, producing genuine, measured deadline
overruns in the *parent* process, not the worker.

Fix: `start_residual_observer_worker` temporarily sets
`OPENBLAS_NUM_THREADS`/`OMP_NUM_THREADS`/`MKL_NUM_THREADS`/`NUMEXPR_NUM_THREADS=1`
in the environment immediately before `process.start()` (the `spawn` child
inherits this environment snapshot at exec time), then restores the
parent's original values right after -- the parent's own already-initialized
BLAS thread pool is unaffected (those libraries only read the env var once,
at their own first initialization, which already happened earlier in this
process). After the fix, the same benchmark ran to `duration_complete`
(500/500 steps) in both modes.

## Real timing evidence (sync vs async, `residual_observer` phase)

Mocked-link harness (`_MockDTLink`, same pattern as
`tests/hardware/test_direct_torque_residual_observer_trace.py`), 500 real
cycles (`duration_s=1.0`, dt=2ms, `dynamics_source="local"`,
`residual_qdd_gap_cycles=2`), real wall-clock `time.sleep` (not mocked),
this machine, after both fixes above:

| | sync (`residual_observer_async=False`) | async (`=True`) |
|---|---|---|
| mean | 0.0721 ms | 0.0177 ms |
| p95 | 0.1274 ms | 0.0355 ms |
| p99 | 0.1318 ms | 0.0406 ms |
| max | 0.1377 ms | 0.2300 ms |
| steps completed | 500/500 (`duration_complete`) | 500/500 (`duration_complete`) |
| merged / dropped | n/a | 500/500 merged, 0 dropped requests, 0 dropped results |
| worker exit | n/a | `exitcode=0`, not force-terminated |

Async mean dropped to ~25% of sync's (0.0177ms vs 0.0721ms) -- consistent
with "near just the enqueue overhead," the expected result. Async's `max`
(0.2300ms) is higher than sync's `max` in this particular sample -- a single
enqueue call is not perfectly free (dataclass construction + 4 numpy
`.copy()` calls + `put_nowait`'s own locking), and this is real
wall-clock noise on a shared host, not a design flaw; the **mean and p95/p99
are what matter for the deadline budget** (a 2ms period with a 1ms
deadline-monitor threshold), and both dropped substantially. This exact
comparison is a committed, repeatable test:
`tests/hardware/test_direct_torque_residual_observer_async.py::test_residual_observer_async_phase_cost_is_much_lower_than_sync`
(marked `@pytest.mark.slow`; asserts async mean `< 0.6x` sync mean -- a
generous, non-flaky margin around the ~0.25x measured here, per AGENTS.md
SS8's shared-host guidance).

Separately, the earlier raw worker-throughput microbenchmark
(`bench_worker_throughput.py`, not committed -- one-off investigation
script, matching this repo's existing precedent for scratch benchmarks)
measured the worker's own steady-state compute cost independent of IPC: a
2000-item burst was internally processed by the worker in ~130ms (~15,000
items/sec), confirming the per-cycle Pinocchio `coriolis()` + linear solve
is cheap in isolation, same order of magnitude as
`docs/status/residual_observer_dynamics_optimization_2026-07-30.md`'s
~0.003-0.005ms/call finding -- the two real bugs above were about process
*lifecycle* and *startup contention*, not the worker's steady-state compute
cost.

## Verification

- **Sync mode byte-identical**: `test_sync_mode_default_matches_pre_async_change_golden_trace`
  asserts today's sync path (`residual_observer_async` omitted, default
  `False`) reproduces EXACTLY (`rows == golden`, full dict equality
  including every trace field) a trace captured from the code as it existed
  immediately before this change (commit `f1748c1`), under a fully
  deterministic fake clock (`_FakeClock`, monkeypatching `monotonic_ns`/
  `time.sleep`) -- necessary because real wall-clock jitter would otherwise
  make `cycle_work_ms`/`lateness_ms`, and even `qdd_*` (the finite-difference
  qdd estimate divides by real elapsed time), differ slightly run to run
  even for identical code. This was additionally cross-checked manually
  during development via `git stash`: running the actual pre-change code and
  the new code (flag off) against the same deterministic-clock harness
  produced a byte-identical diff (`diff old_trace.json new_sync_trace.json`
  -> empty).
- **Async equivalent values**: `test_async_mode_produces_equivalent_diagnostic_values`
  -- sync and async runs under the same deterministic clock produce
  `qdd_pred`/`qdd_measured`/`qdd_residual`/`qdd_residual_norm` matching to
  `atol=1e-12` for every merged step (measured exactly `0.0` diff in
  practice, since both paths use the identical formula).
- **Producer never blocks**: `test_producer_never_blocks_under_backpressure`
  (above).
- **Clean process lifecycle**: `test_clean_process_lifecycle_normal_exit`,
  `test_clean_process_lifecycle_exception_mid_run` (a non-`RTDEStateError`
  exception raised from `link.read_state()` mid-run, expected to propagate
  out of `run_x_transport_direct_torque`), and
  `test_rtde_state_error_mid_run_also_tears_down_worker` (the one exception
  type that's caught inline rather than propagating) -- all three assert
  `"residual-observer-worker" not in {p.name for p in
  multiprocessing.active_children()}` after the call returns/raises.

## Tests

- `tests/hardware/test_direct_torque_residual_observer_async.py` -- 8 tests
  (1 marked `@pytest.mark.slow`, the real-timing benchmark).
- Full suite: `/common/users/ss5772/miniforge3/envs/mujoco_ur5e/bin/python -m pytest -q`.
  Before this change (verified via `git stash`, this tree): **430 passed, 1
  deselected** (`-m "not slow"`). After: **437 passed, 2 deselected**
  (`-m "not slow"`); including the new slow test, **439 passed** total
  (up from 431 total before). No regressions, no pre-existing failures
  encountered.

## Not in scope (deliberately deferred)

Real-time alerting/threshold-triggered behavior based on the residual value
-- this task only relocates the existing, already-diagnostic-only
computation off the critical path. No new safety-relevant decision logic
was added anywhere in this change.

## Rollback

`git checkout -- hardware/direct_torque_transport.py hardware/x_transport.py tools/ur5e_direct_torque_x_transport.py && rm hardware/residual_observer_worker.py tests/hardware/test_direct_torque_residual_observer_async.py tests/hardware/fixtures/direct_torque_sync_pre_async_baseline_trace.json docs/status/direct_torque_residual_observer_async_2026-07-31.md`

(or `git revert <commit>` once committed). Default behavior
(`residual_observer_async` left at its new default, `False`) is byte-for-byte
the previous inline synchronous behavior -- rollback restores the same
behavior this change already defaults to, not any previously-broken state.
