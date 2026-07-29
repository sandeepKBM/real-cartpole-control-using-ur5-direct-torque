# Performance/latency audit — 2026-07-29

Scope per task: find real, measured performance issues in the 500Hz `direct_torque`
hot path (`hardware/direct_torque_transport.py` + `controller_core/x_axis_cartesian_impedance.py`'s
`compute()`, both safety-critical — report only, no direct edits) and in
simulation/training throughput (`simulation/ur5e_mujoco_torque.py`,
`tools/ur5e_mujoco_torque_experiments.py`, `rl_gain_scheduling/`). Same investigation
standard as `docs/status/local_dynamics_speedup_investigation_2026-07-29.md`: real
before/after numbers, not speculation.

## Verdict

**Found and fixed one real, measured redundant-allocation bug affecting every
MuJoCo-sim training/eval loop in the repo: `compute_gravity_torque()` was allocating
a brand-new `mujoco.MjData` (plus a full `mj_forward`+`mj_inverse` pass on it) on
*every single call* from `simulation/ur5e_mujoco_torque.py::build_mujoco_state()`,
instead of reusing a persistent scratch buffer the codebase already has a working
pattern for elsewhere.** Measured ~11-12x per-call speedup (0.70ms -> 0.06ms) and a
resulting ~2.1x speedup on `GainSchedulingEnv.step()` (2.87ms -> 1.38ms mean, the
literal RL-training inner loop). Fixed directly (behavior-preserving, opt-in
parameter, bit-identical output verified, full test suite green).

Also profiled `controller_core/x_axis_cartesian_impedance.py::compute()` (the 500Hz
hot path) with realistic bounded state — **not fixed, safety-critical carve-out,
reported for human review only**: it is healthy relative to the 2ms/500Hz budget
(mean 127-207us = 6-10%, p99 172-298us = 9-15%, depending on which P3 flags are on),
but `np.linalg.cond(J)` (an SVD) is unconditionally computed every cycle and is the
single largest line-item even in the historical/default config where its result
(`singular_scale`) is essentially always 1.0 away from the wrist singularity.

## Finding 1 (fixed): `compute_gravity_torque` fresh-`MjData`-per-call

### The bug

`mujoco_ur5e_tools.py:76`:
```python
scratch = scratch_data if scratch_data is not None else mujoco.MjData(model)
```
`simulation/ur5e_mujoco_torque.py:291` (before this fix) called it with no
`scratch_data`:
```python
gravity_torque = compute_gravity_torque(model, data, joint_ids) if gravity_compensation else None
```
inside `build_mujoco_state()`, the shared per-step state builder used by:
- `rl_gain_scheduling/gain_scheduling_env.py::GainSchedulingEnv.step()` — calls it
  **twice per env step** (pre-state and post-state), always with
  `gravity_compensation=True`. This is the literal PPO/SAC training inner loop.
- `tools/ur5e_mujoco_torque_experiments.py::run()` — the single-run rollout engine
  every sweep driver in the repo subprocesses (per AGENTS.md §2), called once or
  twice per simulated step whenever `gravity_mode == "gravity_comp"` (3 call sites:
  lines ~758, ~822, ~911 pre-fix).

The codebase already has the correct pattern, just not applied here:
`simulation/ur5e_mujoco_torque.py`'s `MujocoUR5eTorqueAdapter.__init__` allocates
`self._gravity_scratch = mujoco.MjData(model)` **once** and reuses it in
`_gravity_torque()` — but that fallback path is essentially dead in the common flow
(`state.gravity_torque` is already populated by `build_mujoco_state`, so the
adapter's own cached-scratch path is rarely exercised); the actually-hot free
function never got the same treatment.

### Benchmark (same machine, same q samples, 200-call warmup discarded, 2800-5000 timed calls)

`compute_gravity_torque(model, q, joint_ids)` (fresh `MjData` every call, the
pre-fix behavior) vs the same call with a persistent `scratch_data=` reused across
calls:

| | fresh `MjData` (before) | reused scratch (after) | speedup |
|---|---|---|---|
| mean | ~0.69-0.72 ms | ~0.06 ms | ~11-12x |
| p50 | ~0.67-0.70 ms | ~0.05-0.06 ms | ~12x |
| p95 | ~0.78-0.89 ms | ~0.08-0.09 ms | ~9-10x |
| p99 | ~0.85-0.97 ms | ~0.10-0.11 ms | ~9x |

(3 independent runs each direction; range shown, not a single sample.)

`build_mujoco_state(..., gravity_compensation=True)` end-to-end (includes
`mj_forward`, `mj_jacSite`, gravity, `expand_mass_matrix`): **mean 0.964ms** before
the fix — i.e. the gravity-torque allocation alone was ~65-70% of this function's
total cost.

**`GainSchedulingEnv.step()` end-to-end** (real RL env, real config
`config/rl_gain_scheduling.yaml`, 3000 timed steps after 200-step warmup, one
`mujoco.mj_step` + two `build_mujoco_state` calls per env step, measured via
`git stash`/`git stash pop` on the identical machine/process for a true before/after):

| | before | after | speedup |
|---|---|---|---|
| mean | 2.8685 ms | 1.3769 ms | ~2.08x |
| p50 | 2.7371 ms | 1.2370 ms | ~2.21x |
| p95 | 3.6096 ms | 2.3024 ms | ~1.57x |

This directly improves PPO/SAC training wall-clock throughput for every
`rl_gain_scheduling/` run — no other code in `env.step()` changed.

### Numerical equivalence verified before implementing

Before touching any hot-loop caller, confirmed `compute_gravity_torque`'s output is
**bit-identical** with vs. without a reused scratch (the function fully zeroes
`qpos`/`qvel`/`qacc`/`qfrc_applied`/`xfrc_applied`/`ctrl` before every
`mj_forward`/`mj_inverse`, so a reused buffer carries no state across calls): 20
random `(q, qd)` samples, `build_mujoco_state`'s full output
(`gravity_torque`, `q`, `jacobian`, `mass_matrix`) compared via `np.array_equal` —
all identical. Full test suite:
`/common/users/ss5772/miniforge3/envs/mujoco_ur5e/bin/python -m pytest -q` — **338
passed, 1 pre-existing failure** (`tests/hardware/test_direct_torque_transport_timing.py::test_transport_records_timing_and_deadline_loop`,
unrelated `dominant_phase` assertion gap already documented as pre-existing/known —
confirmed via `git stash` this is unaffected by these changes).

### What changed

- `simulation/ur5e_mujoco_torque.py`:
  - `build_mujoco_state(...)` — new optional kwarg `gravity_scratch_data:
    mujoco.MjData | None = None`, threaded through to `compute_gravity_torque(...,
    scratch_data=gravity_scratch_data)`. Default `None` preserves the exact prior
    behavior (fresh scratch every call) for any caller that doesn't opt in — purely
    additive, no existing call site's behavior changes unless it passes the new
    argument.
  - `build_initial_state_and_adapter(...)` — same new optional kwarg, forwarded to
    its own `build_mujoco_state` call.
- `rl_gain_scheduling/gain_scheduling_env.py`:
  - `GainSchedulingEnv.__init__` allocates `self._gravity_scratch =
    mujoco.MjData(self.model)` once (identical pattern/lifetime to the existing
    `self.model`/`self.data`; safe under `SubprocVecEnv` since SB3 constructs a
    fresh env per subprocess via `env_fns`, never pickles a live env instance).
  - `reset()`'s `build_initial_state_and_adapter(...)` call and both `step()`
    `build_mujoco_state(...)` calls (pre-state and post-state) now pass
    `gravity_scratch_data=self._gravity_scratch`.
- `tools/ur5e_mujoco_torque_experiments.py`:
  - `run()` allocates `gravity_scratch = mujoco.MjData(model)` once, right after
    `load_model(...)`.
  - All 4 `build_mujoco_state`/`build_initial_state_and_adapter` call sites in the
    main run loop now pass `gravity_scratch_data=gravity_scratch` (the
    `gravity_compensation=False` calls in the separate `_run_zero_torque_gravity`
    diagnostic path were left untouched — they never call `compute_gravity_torque`
    at all, so there's nothing to cache there).

### Not fixed (out of scope, flagged for a deliberate call)

- `rl_gain_scheduling/_scratch_accel_profile_demo.py` (2 `build_mujoco_state(...,
  gravity_compensation=True)` calls) has the same gap. Left alone: it's explicitly
  named/documented as a one-off out-of-distribution stress-test script, not a
  repeated training/sweep driver — lower value to fix, and per this repo's own
  "don't gold-plate" instruction, not touched.
- `tools/audit_ur5e_mujoco_gravity_torque.py` already passes `scratch_data=` — no
  gap there, confirmed by grep before starting.

## Finding 2 (report only, safety-critical carve-out): `compute()`'s `np.linalg.cond(J)`

**Not modified** — `controller_core/x_axis_cartesian_impedance.py`'s `compute()` is
explicitly carved out from direct edits by this task (safety- and timing-relevant).
Reported for human review.

### Measurement (realistic bounded synthetic state: `x_err` in [-3, +3] cm, small
q/qd jitter — an earlier pass with an unbounded synthetic `x_err` produced a
misleading ~350-500x inflated number by accidentally driving the backtracking loop;
caught via `task_backtrack_iters` sanity-check before drawing any conclusion, so
that number is not reported)

`compute()` cost on this machine, 8000 timed calls after 300-call warmup, 6x6
`J`/`M`:

| config | mean | p50 | p95 | p99 | % of 2ms budget (mean / p99) |
|---|---|---|---|---|---|
| default (P3 flags off, historical) | 126.8 us | 121.0 us | 160.6 us | 172.2 us | 6.3% / 8.6% |
| `config/ur5e_mujoco_torque_osc_tuned.yaml` (task-space shaping + nullspace posture on) | 207.5 us | 212.9 us | 284.5 us | 298.0 us | 10.4% / 14.9% |

**Not a current crisis** — well under the 2ms period at both mean and p99, on top of
whatever margin `read_state`/`local_dynamics`/`safety`/`direct_torque` phases
consume elsewhere in the same cycle (the one real hardware sample in
`docs/status/clock_timing_late_cycles_2026-07-28.md` had `controller_mean_ms:
0.669` on different hardware — not directly comparable, but the same order of
magnitude, i.e. this class of cost is real but currently within budget).

`cProfile` on the default (historical) config, realistic bounded data, 20000 calls:
`np.linalg.cond(J)` (`numpy/linalg/_linalg.py:1914`, internally an SVD) is **~46.6
us/call — the single largest line-item, ~37% of the ~127us mean call cost** — even
in the default config where `jacobian_singular_cond_max=1e5` means `singular_scale`
is 1.0 for any pose that isn't within a few orders of magnitude of the exact
wrist_2=0 singularity, i.e. for the vast majority of a transport move this SVD's
only real consumer is a diagnostic field (`output.jacobian_cond`) and a scale factor
that never deviates from 1.0. It is computed unconditionally at
`x_axis_cartesian_impedance.py:435`, before the `if use_shaping or use_nullspace`
branch, so it runs identically whether or not any P3 flag is on.

**Recommendation for human review, not implemented**: if this ever needs headroom
back, `cond(J)`'s cost is inherent to needing the 2-norm condition number (which
`np.linalg.cond` computes via SVD internally — there's no cheaper exact 2-norm cond
for a dense 6x6 without an SVD or eigendecomposition of `J.T@J`, and the latter
squares the conditioning making it numerically worse for exactly the near-singular
case this exists to detect). A genuine option would be computing it only when
`jacobian_cond` is actually consumed downstream (diagnostics logging is optional per
call site) or only every N cycles with a stale-but-conservative fallback between
updates — but that's a real behavior/timing-semantics change to a file this task is
carved out from touching directly, so left as a documented option, not a patch.

## Finding 3 (checked, found negligible — no action)

Hypothesized `hardware/direct_torque_transport.py`'s per-cycle `trace_rows.append()`
(13 `.tolist()` calls + dict construction, happening *after* the loop's
deadline-tracked sleep, so its cost is invisible to `work_ns`/`DeadlineMonitor` and
only surfaces as next-cycle start-lateness — consistent with
`docs/status/timing_safety_gaps_audit_2026-07-28.md`'s own note about this
lagged-measurement property) might be a real hidden cost. Measured directly with an
isolated microbenchmark of the identical dict-construction pattern: **mean 7.55 us,
p95 9.31 us — 0.38% of the 2ms budget. Negligible, no action taken.**

`hardware/latency.py::PhaseLatencyRecorder.record()` (called ~10x/cycle via
`getattr(self, name)` + list append) was also considered; not benchmarked in
isolation since it's an even smaller subset of the same negligible pattern already
measured above (dict/list operations, not `.tolist()`+numpy work) — no action.

## Files changed

- `simulation/ur5e_mujoco_torque.py` — `build_mujoco_state`, `build_initial_state_and_adapter`
- `rl_gain_scheduling/gain_scheduling_env.py` — `GainSchedulingEnv.__init__`, `reset`, `step`
- `tools/ur5e_mujoco_torque_experiments.py` — `run()`

No changes to `hardware/safety.py`, `controller_core/safety.py`,
`hardware/direct_torque_transport.py`'s control loop, or
`controller_core/x_axis_cartesian_impedance.py::compute()` — all safety-critical
per this task's carve-out; findings there are report-only (Findings 2 and 3 above).

## Tests

- `/common/users/ss5772/miniforge3/envs/mujoco_ur5e/bin/python -m pytest -q` — 338
  passed, 1 pre-existing failure (unrelated, confirmed via `git stash`).
- Targeted numerical-equivalence check (not a committed test file, ad hoc script):
  20 random `(q, qd)` samples, `build_mujoco_state` output with vs. without a reused
  gravity scratch — bit-identical (`np.array_equal`) on `gravity_torque`, `q`,
  `jacobian`, `mass_matrix`.
- Real before/after throughput benchmark on `GainSchedulingEnv.step()` via
  `git stash`/`git stash pop` on the same process/machine (numbers above).

## Rollback

```
git checkout -- simulation/ur5e_mujoco_torque.py rl_gain_scheduling/gain_scheduling_env.py tools/ur5e_mujoco_torque_experiments.py
rm docs/status/performance_audit_2026-07-29.md
```
(or `git revert <commit>` once committed). Default behavior for every existing
caller that doesn't pass the new `gravity_scratch_data`/`gravity_scratch` argument
is unchanged (verified bit-identical) — rollback only removes the opt-in fast path,
not any previously-working behavior.
