# URScript singular-value (cond(J)) wrench scaling: closing the last known Mode-3 gap

Context: `assets/urscript/x_axis_osc_inner.script.template` (Mode 3, `urscript` control mode)
is a hand-ported reimplementation of `controller_core.x_axis_cartesian_impedance
.XAxisCartesianImpedanceController` that runs directly on the UR5e's PolyScope. Two of three
known divergences from the validated Python controller were fixed 2026-07-26 (nullspace
posture projection, geometric torque backtracking), leaving one deliberate, documented gap:
singular-value (cond(J)) wrench scaling was omitted entirely, because URScript has no
built-in SVD. This note closes that gap.

**Scope, stated plainly: this is Python-vs-Python numerical parity work only.** There is no
robot access tonight. Nothing here has been run on PolyScope, URSim, or a real UR5e. The tests
below prove the ported arithmetic agrees with the ground-truth controller in Python; they say
nothing about on-robot timing, PolyScope interpreter behavior, or real motion. Do not treat a
passing test here as hardware validation.

## What was ported

`XAxisCartesianImpedanceController.compute()` (`controller_core/x_axis_cartesian_impedance.py`):

```python
cond = float(np.linalg.cond(J))                       # exact 2-norm cond via SVD (LAPACK)
singular_scale = 1.0
if cond > self.cfg.jacobian_singular_cond_max > 0.0:
    singular_scale = float(self.cfg.jacobian_singular_cond_max / cond)
...
wrench_scaled = wrench_effective * singular_scale      # after optional Lambda shaping
tau_task_nominal = J.T @ wrench_scaled                 # before backtracking/clip
```

URScript has no SVD or eigendecomposition primitive, and `docs/status/performance_audit_2026
-07-29.md` (Finding 2, same day) independently confirms there is no cheaper *exact* way to get
the 2-norm condition number of a dense 6x6 without an SVD or an eigendecomposition of `J.T@J`
-- and that squaring the matrix (`J.T@J`) squares the conditioning, which is numerically
dangerous for exactly the near-singular case this term exists to detect.

The port (`assets/urscript/x_axis_osc_inner.script.template`) therefore **estimates** cond(J)
rather than computing it exactly:

- `jacobi_eigenvalues_sym6(A, sweeps)`: a from-scratch, trig-free cyclic Jacobi eigenvalue
  algorithm for a 6x6 symmetric matrix (only `abs`/`sqrt`, no `sin`/`cos`/`atan2`), using the
  standard `t = sign(theta)/(|theta|+sqrt(theta^2+1))` rotation formula. Full sweeps over all
  15 off-diagonal pairs.
- `sigma_max(J)` = sqrt(largest eigenvalue of `J^T*J`).
- `sigma_min(J)` = `1 / sqrt(largest eigenvalue of inv(J)^T*inv(J))`, i.e. the reciprocal of
  the dominant eigenvalue of the *inverse*, not the smallest eigenvalue of `J^T*J` directly.
  This is the numerically important design choice: reading off `J^T*J`'s smallest eigenvalue
  directly loses all precision once cond(J) exceeds roughly 1e8 (`J^T*J`'s own conditioning is
  cond(J)^2, which blows past float64's ~1e16 dynamic range). Routing through `inv(J)`'s own
  *dominant* eigenvalue instead keeps both eigenproblems individually well-conditioned,
  regardless of how ill-conditioned `J` itself is.
- `cond_j = sqrt(lam_max(J^T*J) * lam_max(inv(J)^T*inv(J)))`, then the same
  `singular_scale` branch as Python, applied at the same point in the pipeline (right after
  the wrench, before wrench shaping, before `J.T`).

New wiring (`hardware/urscript_gen.py`): `UrscriptOscParams.jacobian_singular_cond_max`
(sourced from the YAML config's `controller.jacobian_singular_cond_max`, default `1.0e5` --
matches `CartesianImpedanceConfig`'s own default exactly, and was previously not read by the
generator at all) and `UrscriptOscParams.singular_scale_jacobi_sweeps` (generation-only
constant, default `8`, no Python-config equivalent since `np.linalg.cond` is exact). Both bake
into new template placeholders `{{JACOBIAN_SINGULAR_COND_MAX}}` /
`{{SINGULAR_SCALE_JACOBI_SWEEPS}}`, following the same dataclass-field +
`load_params_from_yaml` + placeholder pattern used for `use_nullspace` / `task_resample_*`.

**Behavior-relevant note found while wiring this up**: `DEFAULT_CONFIG`
(`config/ur5e_mujoco_torque_osc_tuned.yaml`) does not set `jacobian_singular_cond_max` in its
YAML, so it falls back to the class default of `1.0e5` -- the *base* tuned config, not the
`_no_singular_scale` variant. Before this change the template unconditionally omitted the term
for every config it was asked to render, silently matching the `_no_singular_scale` config's
intended direction only by coincidence. It now actually reads the value out of whatever config
is passed to the generator, so `_no_singular_scale.yaml` (`jacobian_singular_cond_max: 1.0e18`)
and the base tuned config (implicit `1.0e5`) now render genuinely different scripts.

## Numerical accuracy of the estimate (measured, not asserted)

Randomized 6x6 Jacobians with prescribed exact condition number (same `_jac_with_cond` helper
the existing parity tests already use), comparing the Jacobi-based estimate against
`np.linalg.cond`:

| true cond(J) | estimate rel. error (8 sweeps) |
|---|---|
| 50 | ~4e-15 |
| 1e3 | ~4e-14 |
| 1e5 | ~2e-12 |
| 1e7 | ~4e-10 (worst case over 200 randomized trials up to 1e7: ~2e-9) |
| 1e10 | ~1e-6 |
| 1e13 | ~4e-4 (starting to degrade) |
| 5e16 (exact wrist singularity) | ~92% (unusable) |

The default `jacobian_singular_cond_max=1e5` means the term's real operating range is
"cond(J) somewhere from ~1e5 up to however far a transport move gets from the wrist-singular
start pose before the safety guard would trip" -- empirically nowhere near 1e13+. The
degradation at extreme cond(J) is a known, structural limitation of this approach (squaring
still happens inside `J^T*J`/`inv(J)^T*inv(J)` individually, just not across the max/min
comparison), documented here rather than hidden.

6 sweeps was already sufficient for machine-precision convergence at cond(J) up to 1e7 in a
separate sweep-count experiment; 8 is used for margin. Convergence was not otherwise tuned or
cherry-picked -- these are the numbers from the first algorithm design that avoided the
J^T*J-only precision trap.

## Test changes and results

`tests/hardware/test_urscript_parity.py`:
- Added `_jacobi_eigenvalues_sym6` / `_urscript_cond_estimate`: Python transcriptions of the
  new template functions, same algorithm, same loop structure (not calling `numpy`'s
  eigensolver) -- consistent with how `_urscript_backtrack_task_scale` already transcribes the
  backtracking helper.
- `_urscript_reference_tau` gained `cond_max` (default `1.0e18`, i.e. off, so every
  pre-existing call site that doesn't pass it is unaffected) and `jacobi_sweeps` (default `8`)
  params, and now applies `wrench_scaled = wrench_eff * singular_scale` before `J.T` in the
  same position as the Python source.
- `test_gap_singular_scaling` (previously asserted the gap *exists*) replaced with two tests
  asserting real parity, matching the module's existing `atol=1e-8` discipline rather than
  introducing a looser tolerance:
  - `test_singular_scaling_clean_regime_matches_python` (20 randomized well-conditioned
    states, cond(J)=50): `singular_scale == 1.0` exactly on both sides, `atol=1e-8` on
    `tau_task_nominal` and final `tau`.
  - `test_singular_scaling_near_singular_matches_python` (cond(J)=1e7, the same case the old
    gap test used, at DEFAULT_CONFIG's real `jacobian_singular_cond_max=1e5` threshold so
    scaling actually engages): `atol=1e-8` on `tau_task_nominal` and final `tau`, plus a
    direct check that the estimated `cond(J)` itself is within `rel=1e-6` of
    `np.linalg.cond`. Measured on this exact case: cond(J) relative error ~5.7e-11,
    `tau_task_nominal` max abs difference ~9.8e-12 Nm. Also re-verifies the `cond_max=1e18`
    (disabled) direction still agrees, as the old test did.
- `tests/hardware/test_urscript_gen.py`: added assertions that the rendered script bakes
  `cond_max = 100000` and `jacobi_sweeps = 8` from `DEFAULT_CONFIG`.

### Before

```
tests/hardware/test_urscript_parity.py::test_gap_singular_scaling  <- asserted the GAP exists
```

### After

```
$ python -m pytest -q tests/hardware/test_urscript_parity.py tests/hardware/test_urscript_gen.py
..........
10 passed in 0.99s
```

All 10 tests pass, including the two new/flipped singular-scaling parity tests. No other test
in the file needed a tolerance change.

### Broader regression check

`python -m pytest -q tests/hardware tests/unit`: 281 passed, 1 failed
(`test_direct_torque_transport_timing.py::test_transport_records_timing_and_deadline_loop`).
That failure is unrelated to this change -- it asserts on a `dominant_phase` timing label
(`'local_dynamics'` not yet in the test's allowed set) inside `hardware/x_transport.py` /
`hardware/direct_torque_link.py`, both of which were being actively edited by a concurrent
agent during this session (confirmed via live `git status` diffs showing those files changing
mid-session) and are explicitly out of scope for this task. Not touched, not investigated
further here.

## Known caveats (stated explicitly, not buried)

- **No real-hardware validation of any kind tonight.** No robot, no URSim, no PolyScope
  execution. This closes a *numerical-parity* gap only.
- **The estimate is not the algorithm** -- `np.linalg.cond` (LAPACK SVD) and
  `jacobi_eigenvalues_sym6` (from-scratch Jacobi on `J^T*J` / `inv(J)^T*inv(J)`) are different
  algorithms that happen to agree closely in the operating range that matters. They are not
  guaranteed to agree outside the measured range (see the accuracy table above).
- **Per-cycle compute cost on real hardware was not measured or budgeted.** Two full Jacobi
  eigendecompositions (8 sweeps x 15 pairs each) plus a 6x6 matrix inverse were added to the
  on-robot inner loop. `docs/status/performance_audit_2026-07-29.md` (same day) measured
  Python's single `np.linalg.cond` SVD call at ~46.6us on this machine; the URScript
  PolyScope interpreter's cost model for the equivalent work is unknown and not benchmarked
  here. If Mode 3 is ever run for real, its cycle time should be checked against
  `get_steptime()` headroom before trusting it at 500Hz -- this is a flagged follow-up, not
  something this task addressed.
