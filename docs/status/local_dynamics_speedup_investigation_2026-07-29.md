# Investigation: does Pinocchio actually speed up `hardware/local_dynamics.py`'s 500Hz hot path?

Context: task tracker item #55, "local_dynamics.py speedup attempt failed -- needs real
investigation, not a quick fix." No written record of an earlier attempt survives (no branch,
stash, or commit references it) -- but the code carries a real clue: the class is named
`LocalMujocoDynamics`, computes everything via MuJoCo (`mj_forward`/`mj_jacSite`/
`expand_mass_matrix`), and is aliased as `LocalPinocchioDynamics = LocalMujocoDynamics` at the
bottom of `hardware/local_dynamics.py`, with the `coriolis()` docstring explicitly flagging
"despite the LocalPinocchioDynamics back-compat alias name" it's MuJoCo, not Pinocchio. That
strongly implies someone previously intended a real Pinocchio-backed fast path and aliased it
back to MuJoCo without writing up why.

## Verdict

**Yes -- Pinocchio offers a real, ~10x speedup, and a working implementation now exists,
landed opt-in behind `dynamics_source="local_pinocchio"`.** The earlier attempt's most likely
blocker: `controller_core/model_dynamics.py::PinocchioUR5eDynamics` had no Jacobian method at
all (only `gravity`/`coriolis`/`mass_matrix`/`bias`), and a naive Jacobian implementation using
Pinocchio's standard API produces a **silently wrong, mirrored-on-X/Y result** for this specific
MJCF (worst-case error ~2.0 rad or m, i.e. completely wrong, not a small numerical drift) unless
a specific world-frame correction is applied -- see Root Cause below. That failure mode is
exactly the kind of thing that would make someone give up and alias back to MuJoCo rather than
ship a torque-command Jacobian that's silently wrong on a real robot.

## Benchmark (same machine, same q samples, 5000 warmed-up calls, 200-call warmup discarded)

`LocalMujocoDynamics.jacobian_and_mass_matrix()` (existing hot path) vs a Pinocchio path
(`computeJointJacobians` + `getFrameJacobian(..., LOCAL_WORLD_ALIGNED)` + `crba`, world-corrected
-- see below), both drawing q uniformly from the model's joint ranges:

| | MuJoCo (current) | Pinocchio (new) | speedup |
|---|---|---|---|
| mean | ~0.50-0.53 ms | ~0.05-0.08 ms | ~7-10x |
| p50 | ~0.46-0.50 ms | ~0.04-0.05 ms | ~10x |
| p95 | ~0.67-0.69 ms | ~0.06 ms | ~11x |
| p99 | ~0.78 ms | ~0.07-0.16 ms | ~5-10x |

(4 independent runs; numbers above are the observed range across runs, not a single sample.)
At 500 Hz the period budget is 2 ms. The MuJoCo path already consumes ~25-39% of that budget
per cycle (mean/p99); the Pinocchio path consumes ~2.5-8%. `docs/status/clock_timing_late_cycles_2026-07-28.md`'s
single real-hardware sample (`local_dynamics_mean_ms: 0.123`) is lower than this cluster
machine's MuJoCo measurement (different hardware, not directly comparable), but the *relative*
comparison -- same process, same machine, same q samples -- is what the speedup claim rests on,
and it reproduced consistently across 4 runs (7-11x).

Repro: the benchmark script is not committed (per this task's "stay scoped" instruction, no new
top-level tool was added); it directly calls `LocalMujocoDynamics.jacobian_and_mass_matrix` and
a `pin.computeJointJacobians`/`getFrameJacobian`/`crba` pipeline with identical q samples from
`numpy.random.default_rng(42)`, 200 warmup + 5000 timed calls, `time.perf_counter()` per call.

## Root cause of the likely earlier blocker: a real Pinocchio MJCF-parser gap

`pin.buildModelFromMJCF` (Pinocchio 4.0.0, the `mujoco_ur5e` conda env) does **not** apply the
MJCF's root body's own `quat` to the loaded frame tree. This UR5e MJCF's root body is
`<body name="base" quat="0 0 0 -1" ...>` (a 180-degree rotation about Z) --
`assets/ur5e_torque/ur5e_torque.xml:63`. MuJoCo applies it (`mj_forward`'s `base` body xmat
comes back as `diag(-1,-1,1)`); Pinocchio's `base` frame comes back with an **identity**
placement relative to `universe`.

This is invisible to every existing gravity/coriolis/mass-matrix parity test in
`tests/mujoco/test_pinocchio_parity.py` (validated to <1e-8 Nm gravity / <1e-6 Nm bias / <1e-8
mass-matrix, per AGENTS.md) because a 180-degree rotation about Z leaves the Z axis itself
unchanged, and gravity in this model is along Z -- so joint-space torque quantities come out
identical regardless of whether that root rotation is applied. It is **not** invisible to a
world-frame Cartesian quantity like a site Jacobian: before correction, a naive
`getFrameJacobian(..., LOCAL_WORLD_ALIGNED)` at `attachment_site` differs from MuJoCo's
`mj_jacSite` by up to ~2.0 (rad or m, mixed units across the 6 rows) across 200 full-joint-range
samples -- X and Y rows sign-flipped, Z rows matching exactly, exactly the signature of a
missing `diag(-1,-1,1)` correction.

Fix (`controller_core/model_dynamics.py::_root_body_quat_wxyz` +
`PinocchioUR5eDynamics.__init__`): parse the MJCF's `<worldbody><body quat="w x y z">` directly
(no `mujoco` import needed -- keeps `model_dynamics.py` numpy+pinocchio-only, its existing
invariant), convert via `pin.Quaternion(w, x, y, z).toRotationMatrix()`, and left-multiply the
6-row Jacobian output by `block_diag(R, R)`. After this correction, worst-case error across 200
full-joint-range samples drops to ~3e-15 (verified both in an ad hoc script and in the new
`tests/mujoco/test_pinocchio_parity.py::test_jacobian_parity`, tolerance `1e-6`).

## What landed

- **`controller_core/model_dynamics.py`**: `PinocchioUR5eDynamics.jacobian(q, site_name=...)`
  -- new method, world-frame-corrected as above, parity-tested to <1e-6 against MuJoCo's
  `mj_jacSite` (200 samples, full joint range). `DynamicsProvider` protocol itself is untouched
  (still gravity/coriolis/mass_matrix/bias only) -- the Jacobian is UR5e-specific site geometry,
  not part of that simulator-independent contract.
- **`hardware/local_dynamics.py`**: new `LocalPinocchioFastDynamics` class -- a genuinely new,
  explicitly-named implementation wrapping the extended `PinocchioUR5eDynamics`, with the same
  public interface as `LocalMujocoDynamics` (`jacobian`, `mass_matrix`,
  `jacobian_and_mass_matrix`, `coriolis`, `jacobian_mass_and_coriolis`). The legacy
  `LocalPinocchioDynamics = LocalMujocoDynamics` alias is **untouched** -- still means
  MuJoCo-identical numerics, as documented at its definition; nothing reading that name changes
  behavior. `DYNAMICS_SOURCES` extended from `{"rtde", "local"}` to
  `{"rtde", "local", "local_pinocchio"}`.
- **`hardware/direct_torque_transport.py`**: `dynamics_source="local_pinocchio"` selects
  `LocalPinocchioFastDynamics` in `run_x_transport_direct_torque`; `coriolis_feedforward`'s
  existing "requires local dynamics" gate now accepts either `local` or `local_pinocchio`.
  Default (`dynamics_source="rtde"` at the function level; CLI default `"local"`) is unchanged.
- **`tools/ur5e_direct_torque_x_transport.py`**, **`tools/ur5e_direct_torque_height_latency_test.py`**:
  `--dynamics-source` choices extended to include `local_pinocchio`; CLI default remains `local`.

### Why a new `dynamics_source` value instead of reusing `"local"` with a flag

`dynamics_source="local"` already has one flag-gated behavior modifier
(`coriolis_feedforward`) whose validation is keyed off `dynamics_source`. Overloading `"local"`
with a second, independent "which engine actually computes it" flag would mean two
independently-togglable dimensions collapsed into a single string plus a bolt-on flag, and would
require every future reader of `dynamics_source == "local"` in this file (and the CLI tools) to
also check a second flag to know what's actually running. A distinct value keeps
`normalize_dynamics_source`/`DYNAMICS_SOURCES` as the single source of truth for "which of these
three fully-distinct engine+source combinations is this run using," matches the existing
`{"rtde", "local"}` two-value pattern (just extended, not restructured), and reads directly in
logs/summaries (`summary.json`'s `"dynamics_source"` field) without cross-referencing a second
field.

## Tests

- `tests/mujoco/test_pinocchio_parity.py::test_jacobian_parity` (new) -- 200 samples, <1e-6 tol,
  same rigor as the existing gravity/bias/mass tests in that file.
- `tests/hardware/test_local_pinocchio_dynamics.py` (new) -- alias-unchanged check,
  `DYNAMICS_SOURCES`/`normalize_dynamics_source` coverage, `LocalPinocchioFastDynamics` J/M/C
  parity against `LocalMujocoDynamics`, combined-call consistency, and an end-to-end
  `run_x_transport_direct_torque(..., dynamics_source="local_pinocchio", coriolis_feedforward=True)`
  run against a mocked direct-torque link confirming the wiring actually exercises the new path
  and produces a nonzero Coriolis term.
- Full suite: `/common/users/ss5772/miniforge3/envs/mujoco_ur5e/bin/python -m pytest -q` --
  317 passed, 1 pre-existing failure
  (`tests/hardware/test_direct_torque_transport_timing.py::test_transport_records_timing_and_deadline_loop`,
  confirmed via `git stash` to fail identically on the pre-change tree: it asserts
  `dominant_phase` is one of a fixed set that does not include `"local_dynamics"`, a phase name
  that already existed in `direct_torque_transport.py` before this investigation touched
  anything -- unrelated pre-existing test/code drift, not a regression introduced here).

## Rollback

`git checkout -- controller_core/model_dynamics.py hardware/direct_torque_transport.py hardware/local_dynamics.py tests/mujoco/test_pinocchio_parity.py tools/ur5e_direct_torque_height_latency_test.py tools/ur5e_direct_torque_x_transport.py && rm tests/hardware/test_local_pinocchio_dynamics.py docs/status/local_dynamics_speedup_investigation_2026-07-29.md`

(or `git revert <commit>` once committed). Default behavior (`dynamics_source="local"` or
`"rtde"`) is byte-identical to before this change -- rollback is only needed to remove the new
opt-in path itself, not to restore any previously-working behavior.
