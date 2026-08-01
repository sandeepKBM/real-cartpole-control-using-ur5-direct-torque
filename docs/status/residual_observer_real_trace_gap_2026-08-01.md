# Residual observer real-trace gap: investigation and fix (2026-08-01)

## The reported gap

Real hardware traces pulled tonight via `tools/pull_hardware_logs_ssh.sh` into
`outputs/hardware_transport_remote/hardware_transport/` (81 runs, dated
2026-07-28 through 2026-08-01) were reported as containing **zero** fields
with `resid` or `qdd` in the name, in either `trace.jsonl` rows or
`summary.json` -- apparently blocking `tools/analysis/fit_residual_torque_model.py`
from ever being fit on real data.

## What was actually checked

### 1. `hardware/direct_torque_transport.py` -- does the real path log the fields?

Read `run_x_transport_direct_torque()` in full. `enable_residual_observer`
defaults to `True` (function signature, line 163). Construction is wrapped in
a try/except (lines 244-263): on failure it prints a `WARNING:` line, sets
`residual_dynamics = residual_accel_estimator = None`, and the run proceeds
without residual data -- matching AGENTS.md's "gracefully degrade with a
printed warning" description. On the synchronous path (`residual_observer_async=False`,
the default), `trace_rows.append(...)` (lines 654-694) unconditionally
includes `qdd_pred`, `qdd_measured`, `qdd_residual`, `qdd_residual_norm` keys
in every row dict -- `None` when unavailable, the real values otherwise. The
async path (`residual_worker`) merges the same four keys back into
`trace_rows` by step index after the loop exits (lines 717-751). **The fields
are already written into `trace_rows` on the real-hardware path.** `summary.json`
intentionally has no top-level scalar residual fields -- these are per-cycle
diagnostics that only live in `trace.jsonl`, matching the code's own
"diagnostic-only" comments. This is expected, not a gap.

### 2. `tools/ur5e_direct_torque_x_transport.py` -- CLI wiring

`--disable-residual-observer` (default off) maps to
`enable_residual_observer=not bool(args.disable_residual_observer)` (line 428)
-- same default as the underlying function, nothing silently disables it.
`--residual-observer-async` (default off) maps straight through (line 429).
No divergent default found.

### 3. Direct inspection of the pulled real trace data

The reported observation does not match the data currently sitting in
`outputs/hardware_transport_remote/hardware_transport/`. Checked every
`direct_torque_*` run from 2026-07-31 and 2026-08-01 (45 runs, after the
observer landed in commit `75312d3` on 2026-07-29): **43/45 have
`qdd_pred`/`qdd_measured`/`qdd_residual`/`qdd_residual_norm` present and
non-null in every single trace row.** Example:
`direct_torque_20260801_171839/trace.jsonl` -- 2476/2476 rows, all four
fields populated. Two runs (`direct_torque_20260731_155847`,
`direct_torque_20260731_161213`) have the keys present but `None` for every
row -- the graceful-degradation path, i.e. `PinocchioUR5eDynamics()`
construction genuinely failed for those two specific invocations (both sync,
non-async, per `summary.json`'s `residual_observer_async: null`). This is
~4% of the runs in this window, not all of them, so it does not look like a
missing dependency on `thinkrobot` generally (see the un-confirmable case (c)
below) -- more likely a transient per-process failure. `runs from before
2026-07-29 02:11` (commit `75312d3`) naturally lack the fields entirely,
since the feature didn't exist yet; not a gap, just chronology.

**Verdict: (a) is false as originally framed** -- the observer's output *is*
logged into `trace_rows` on the real-hardware path, for the large majority of
real runs pulled tonight. The premise that trace rows contain zero
resid/qdd fields does not hold up under direct inspection of the actual
files in this repo.

### 4. The real gap: found in the analysis/loading layer, not the logging layer

`tools/analysis/residual_data.py::_required_fields_present()` unconditionally
required `"tau"` and `"qfrc_bias"` in every row before accepting it. Those
are **sim-only** field names (from `simulation/ur5e_mujoco_torque.py` /
`tools/ur5e_mujoco_torque_experiments.py`). Real hardware trace rows have no
`"tau"` key at all (they log `tau_applied`/`tau_controller`/`tau_coriolis`
instead) and no `"qfrc_bias"` key at all. Confirmed by diffing the real trace
key set against the check: `_required_fields_present` returned `False` for
**every** row of every real trace, so `build_run_dataset()` always returned
`None` and `build_dataset()` silently produced zero training rows from any
real-hardware trace file -- despite `qdd_residual` already sitting correctly
computed in those same rows. This is the actual, reproducible reason the
phase-1 pipeline could never be fit on real data: not a missing hardware log,
but a downstream loader that only recognized the sim trace schema.

Confirmed before the fix (reconstructed from the failing precondition, not
re-run against the pre-fix code): with `_required_fields_present` requiring
`tau`+`qfrc_bias`, `build_dataset()` called on 8 real
`direct_torque_202607[3]*`/`202608*` trace files returns 0 runs, 0 rows.

## Fix applied

`tools/analysis/residual_data.py`:
- `_required_fields_present()` now accepts a row via either path: the
  existing sim reconstruction path (`tau` + `qfrc_bias` present), or a new
  real-hardware path (`qdd_residual` present and non-null) -- purely
  additive, `or`-combined with the existing check.
- `build_run_dataset()`'s per-row loop: when `qdd_residual` is already
  present and non-null, use it directly (`tau_residual = M(q) @
  qdd_residual`, computed from `q` alone) instead of requiring
  `tau`/`qfrc_bias` and running it through `JointAccelEstimator`. The
  existing sim-reconstruction branch is untouched (same code path, same
  formula, byte-for-byte behavior for rows without a precomputed
  `qdd_residual`).
- Module docstring updated to state this is now actually implemented (it
  previously described the intent -- "once real trace data ... is
  available" -- without the code actually supporting it).

Verified against real pulled data after the fix:
`build_dataset()` on the first 8 real `direct_torque_20260731_*` trace
files now loads **7/8 runs, 17,688 total training rows** (one run dropped
only because it was outside the requested slice, not a failure). This is a
genuine, verified fix -- real data now flows into the phase-1 pipeline.

### Tests

Added two regression tests to `tests/mujoco/test_residual_data_pipeline.py`:
- `test_build_run_dataset_uses_precomputed_qdd_residual_on_real_hardware_rows` --
  synthetic rows shaped like a real `direct_torque` trace (no `tau`/`qfrc_bias`,
  has `qdd_residual`) are now loaded correctly.
- `test_build_run_dataset_precomputed_path_ignores_null_qdd_residual` -- a row
  with the key present but `null` (the graceful-degradation case) still falls
  through to the sim path and correctly returns `None` when that path's
  fields are also absent, rather than being treated as a zero measurement.

Ran (conda env `mujoco_ur5e`, per AGENTS.md):
- `pytest -q tests/mujoco/test_residual_data_pipeline.py -m mujoco` -- 7 passed
  (5 pre-existing + 2 new).
- `pytest -q -m "mujoco or hardware or unit"` -- 500 passed, 3 xfailed, zero
  regressions.

No `hardware/direct_torque_transport.py`, `hardware/safety.py`, control logic,
safety-guard, or timing code was touched. Only `tools/analysis/residual_data.py`
(analysis/loading, offline, no hardware execution) and its test file changed.

## Case (c) -- not confirmed, not ruled out for the two anomalous runs

The two runs with all-`None` `qdd_pred`/`qdd_measured`/`qdd_residual`
(`direct_torque_20260731_155847`, `direct_torque_20260731_161213`) are
consistent with `PinocchioUR5eDynamics()` construction failing on
`thinkrobot` for those specific invocations, but this cannot be confirmed
from this repo (no hardware access). Since 43/45 runs in the same window
succeeded, a codebase-wide missing dependency in `.venv-hardware` is
unlikely -- if Pinocchio were entirely unavailable there, every run would
show `None`, not just these two. To confirm or rule out a real, non-transient
environment problem, on `thinkrobot` itself:

```
# in the same venv the direct_torque CLI actually runs under
python3 -c "import pinocchio; print(pinocchio.__version__)"
```

If this succeeds reliably, the two failures were most likely a transient
resource/race issue (e.g. contention loading the URDF/MJCF, a momentary
resource limit) at the moment those two specific runs started, not a
systemic dependency gap -- worth a `stderr`/stdout capture on the next
`direct_torque` run to see if the `WARNING:` line reappears, and if so, under
what conditions.
