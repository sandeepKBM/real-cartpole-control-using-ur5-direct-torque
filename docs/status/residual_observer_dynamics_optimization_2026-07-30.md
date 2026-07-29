# Residual-observer dynamics computation optimization, 2026-07-30

## Verdict

**Real, verified, but modest win -- and it does not explain the 1.457 ms real-hardware
outlier by itself.** The residual observer's per-cycle `gravity(q)` call was provably
redundant (its result cancels algebraically against an identical `g(q)` term buried inside
`bias(q, qd)`) -- confirmed both algebraically and numerically (worst-case diff ~2e-14
across 200 random poses, ~13 orders of magnitude below the tolerance this diagnostic
already runs at). Removing it and replacing the two-call `gravity() + bias()` formula with
a single `coriolis(q, qd)` call (computed via a dedicated zero-gravity Pinocchio
model/data pair, not `bias() - gravity()`) measured **~1.5-2x faster on this machine**
(0.0050 ms -> 0.0033 ms mean per residual-observer cycle in a 5000-call microbenchmark).
`computeAllTerms` -- the strongest a priori candidate for a combined-computation win --
was measured **~2.5x SLOWER** than the current two-call path, not faster: it computes
Jacobians, center-of-mass, centroidal-momentum, and kinetic/potential energy terms this
observer never uses, and that extra work costs more than the redundant gravity call it
would have saved. This is an honest negative result for that specific API, not a missed
opportunity.

**Scale check against the real finding:** the real-hardware measurement that motivated
this investigation was `residual_observer_max_ms: 1.457` -- a *max*, not a mean, and
nearly the entire 2 ms/cycle budget. This optimization's real, measured effect is on the
order of **0.0017 ms saved per cycle** (mean, this cluster machine). That is roughly
three orders of magnitude smaller than the 1.457 ms outlier. In other words: the
redundant-gravity-call finding was real and worth fixing, but it is very unlikely to be
what produced the 1.457 ms spike on real hardware -- that outlier's cause (GC pause,
thread scheduling/preemption, cache effects, or something else entirely) is still
unexplained and out of this task's scope (per the task's explicit instruction not to
pursue async/threaded restructuring here).

## What was verified

### 1. The algebraic cancellation, from the actual current code

`hardware/direct_torque_transport.py` (before this change):
```python
tau_true_total = tau + residual_dynamics.gravity(link_state.q)
bias = residual_dynamics.bias(link_state.q, link_state.qd)
qdd_pred = predict_joint_acceleration(mass_matrix, tau_true_total, bias)
```
`controller_core/dynamics_residual.py::predict_joint_acceleration` computes
`M^-1 @ (tau_total_physical - bias)`. Substituting:
```
tau_true_total - bias = (tau + g(q)) - (C(q,qd)qd + g(q)) = tau - C(q,qd)qd
```
confirmed against the *current* `controller_core/model_dynamics.py`:
`PinocchioUR5eDynamics.gravity()` calls `pin.computeGeneralizedGravity(model, data, q)`;
`PinocchioUR5eDynamics.bias()` calls `pin.rnea(model, data, q, qd, zeros)` (RNEA with
zero acceleration, which by the manipulator equation equals `C(q,qd)qd + g(q)` exactly).
Both `g(q)` terms are evaluated from the *same* `q`, but via two structurally different
Pinocchio code paths (`computeGeneralizedGravity` vs. `rnea` with `qd=0` internally) --
so the cancellation is only exact in real-number algebra, not automatically bit-identical
in floating point. Measured (200 random poses, this env): max abs diff between
`computeGeneralizedGravity(q)` and `rnea(q, 0, 0)` was exactly `0.0` on the sampled poses
tested -- effectively bit-identical here, but this was verified empirically rather than
assumed. Production-code note: `hardware/direct_torque_transport.py`'s residual observer
only ever made **two** `residual_dynamics.*` calls per cycle (`gravity()` + `bias()`),
not three -- the mass matrix used in `predict_joint_acceleration` is reused from whichever
`dynamics_source` the controller itself already computed that cycle (RTDE readback, MuJoCo-
or Pinocchio-backed local dynamics), not recomputed by `residual_dynamics`. The task
prompt's "3-call" framing (`crba`/`rnea`/`computeGeneralizedGravity`) matches what was
benchmarked as reference points below, not literally the residual observer's own per-cycle
call count.

### 2. Real per-call timing (this machine, 5000 warmed-up calls, 200-call warmup discarded,
same q/qd samples across all paths, `time.perf_counter()`)

| path | mean (ms) | p50 | p95 | p99 |
|---|---|---|---|---|
| A: `gravity()` + `bias()` (current, 2 calls) | 0.0050 | 0.0050 | 0.0051 | 0.0052 |
| B: `computeAllTerms` (1 call, does much more) | 0.0125 | 0.0125 | 0.0128 | 0.0130 |
| C: zero-gravity-model `rnea` -> `C(q,qd)@qd` (proposed, 1 call) | 0.0033 | 0.0032 | 0.0033 | 0.0034 |
| `crba` alone (mass matrix) | 0.0027 | 0.0026 | 0.0027 | 0.0027 |
| `bias()` alone (`rnea`) | 0.0032 | 0.0032 | 0.0033 | 0.0033 |
| `gravity()` alone (`computeGeneralizedGravity`) | 0.0021 | 0.0021 | 0.0022 | 0.0022 |

Path C (the implemented fix) is ~34% faster than path A (current) in mean time, and
consistently the fastest way to get `C(q,qd)@qd`. Path B (`computeAllTerms`) is ~2.5x
*slower* than the current two-call path -- confirmed not a win for this specific need.

Repro: `bench_residual_dynamics.py`, not committed (scratch investigation script,
matching `local_dynamics_speedup_investigation_2026-07-29.md`'s precedent of not adding a
new top-level tool for a one-off benchmark); directly calls
`pin.computeGeneralizedGravity`/`pin.rnea`/`pin.crba`/`pin.computeAllTerms` against the
same `assets/ur5e_torque/ur5e_torque.xml` model, `numpy.random.default_rng(42)` samples,
200 warmup + 5000 timed calls per path. The committed, repeatable version of this
comparison is `tests/hardware/test_residual_observer_dynamics_optimization.py::
test_coriolis_single_call_is_not_slower_than_old_two_call_path` (best-of-3, generous
1.15x non-regression margin to avoid flakiness on this shared cluster host -- see
AGENTS.md SS8).

### 3. Numerical equivalence (old two-call formula vs. new one-call formula)

Across 200 random `(q, qd, tau)` samples (`tests/hardware/
test_residual_observer_dynamics_optimization.py`):
- `coriolis(q, qd)` (new, single zero-gravity `rnea` call) vs. `bias(q, qd) - gravity(q)`
  (old, two-call formula): worst abs diff **< 1e-9** (measured ~2e-14 in the standalone
  script above; the committed test uses a looser 1e-9 bound as a stable, non-brittle
  regression gate).
- Full `qdd_pred` (`predict_joint_acceleration` output) old formula
  (`tau + gravity(q)`, `bias(q,qd)`) vs. new formula (`tau`, `coriolis(q,qd)`): worst abs
  diff **< 1e-9** across the same 200 samples (measured ~7e-15 for a representative
  sample).
- Existing parity test `tests/mujoco/test_pinocchio_parity.py::
  test_coriolis_is_bias_minus_gravity` (asserts `coriolis(q,qd) + gravity(q) == bias(q,qd)`
  at `atol=1e-12`) still passes with the new implementation -- confirms the public
  `coriolis()` contract is unchanged, only its internal computation is.

## What landed

- **`controller_core/model_dynamics.py`**: `PinocchioUR5eDynamics.__init__` now also
  builds a second, dedicated zero-gravity `Model`/`Data` pair
  (`self._model_zero_gravity`/`self._data_zero_gravity`, one-time `__init__` cost, not
  hot-path) by re-parsing the same MJCF and zeroing `model.gravity`. `coriolis(q, qd)`'s
  body changed from `self.bias(q, qd) - self.gravity(q)` (two Pinocchio calls) to a single
  `pin.rnea(self._model_zero_gravity, self._data_zero_gravity, q, qd, zeros)` call.
  **Public API unchanged**: `gravity()`, `bias()`, `mass_matrix()`, `coriolis()` keep their
  exact signatures and documented return values -- only `coriolis()`'s internal
  implementation changed, and only to something faster and no less accurate. All existing
  callers (`hardware/local_dynamics.py::LocalPinocchioFastDynamics.coriolis` -- used in the
  production `coriolis_feedforward` path, not just diagnostics --, `simulation/
  ur5e_mujoco_torque.py`'s Coriolis feedforward path) get this speedup automatically with
  no call-site changes.
- **`hardware/direct_torque_transport.py`**: the residual observer's per-cycle block now
  computes `coriolis_term = residual_dynamics.coriolis(link_state.q, link_state.qd)` and
  calls `predict_joint_acceleration(mass_matrix, tau, coriolis_term)` directly (no
  `gravity()` call, no `tau + gravity(q)` reconstruction). Net effect: one fewer Pinocchio
  call per cycle in the diagnostic block, on top of `coriolis()` itself being faster.
- **`controller_core/dynamics_residual.py`**: `predict_joint_acceleration`'s docstring
  gained a note documenting that callers may pass any additive-term-cancelling pair
  (e.g. raw `tau` + pure `coriolis(q, qd)`, no gravity in either) instead of
  `tau_true_total` + full `bias`, since the function only ever uses their difference. No
  code/signature change.
- **`tests/hardware/test_residual_observer_dynamics_optimization.py`** (new): correctness
  (`coriolis()` vs. old `bias()-gravity()` formula, and full `qdd_pred` old-vs-new formula,
  both across 200 random poses) + a `@pytest.mark.slow` benchmark test with a generous
  non-regression margin.

## Tests

- `tests/hardware/test_residual_observer_dynamics_optimization.py` (new) -- 4 tests
  (correctness x3, benchmark x1).
- `tests/unit/test_dynamics_residual.py` -- 8 tests, unaffected (pure math, no
  `PinocchioUR5eDynamics` dependency).
- `tests/mujoco/test_pinocchio_parity.py` -- 6 tests, including
  `test_coriolis_is_bias_minus_gravity`, still passing against the new `coriolis()`
  implementation.
- `tests/hardware/test_direct_torque_residual_observer_trace.py`,
  `tests/mujoco/test_direct_torque_residual_observer.py`,
  `tests/hardware/test_local_pinocchio_dynamics.py`, `tests/hardware/test_local_dynamics.py`
  -- all still pass (the last two exercise `coriolis()` through
  `LocalPinocchioFastDynamics`, confirming the speedup lands there too with no behavior
  change).
- Full suite: `/common/users/ss5772/miniforge3/envs/mujoco_ur5e/bin/python -m pytest -q`
  -- confirmed via direct run for this change; see commit message for the exact
  pass/fail counts. One known pre-existing unrelated failure carried over from
  `local_dynamics_speedup_investigation_2026-07-29.md`
  (`tests/hardware/test_direct_torque_transport_timing.py::
  test_transport_records_timing_and_deadline_loop`, `dominant_phase` allow-list doesn't
  include `"local_dynamics"` -- pre-existing test/code drift, unrelated to this change,
  reconfirmed present on this tree via `git stash` before concluding it wasn't a
  regression).

## Recommendation

Land this as a real, verified micro-optimization -- it is correct, non-breaking, and
measurably faster with no downside found. It should **not** be reported or relied on as
"the fix" for the 1.457 ms real-hardware residual-observer spike: that outlier is roughly
1000x larger than the total saving measured here, so its cause is still open. The natural
next step for that investigation (explicitly out of scope for this task) is profiling on
real hardware with finer-grained sub-phase timing around the residual-observer block
(allocation/GC behavior, `JointAccelEstimator.update`'s internal cost, thread/scheduling
effects) rather than further Pinocchio API micro-tuning -- the Pinocchio call itself is
demonstrably cheap (microseconds) on this machine.

## Rollback

`git checkout -- controller_core/dynamics_residual.py controller_core/model_dynamics.py hardware/direct_torque_transport.py && rm tests/hardware/test_residual_observer_dynamics_optimization.py docs/status/residual_observer_dynamics_optimization_2026-07-30.md`

(or `git revert <commit>` once committed). Default behavior (`enable_residual_observer`
default `True`, `dynamics_source` unaffected) is otherwise unchanged -- rollback restores
the previous two-call formula, not any previously-broken behavior.
