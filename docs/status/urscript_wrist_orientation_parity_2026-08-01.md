# URScript wrist-orientation task: closing the newest Mode-3 feature-parity gap

Context: `assets/urscript/x_axis_osc_inner.script.template` (Mode 3, `urscript` control mode) is
a hand-ported reimplementation of `controller_core.x_axis_cartesian_impedance
.XAxisCartesianImpedanceController` that runs directly on the UR5e's PolyScope. Three known
divergences were already closed (nullspace posture projection and geometric torque
backtracking, 2026-07-26; singular-value (cond(J)) wrench scaling, 2026-07-29 — see
`docs/status/urscript_singular_scaling_parity_2026-07-29.md`, the style/rigor template this
note follows). Since then, `wrist_orientation_task` — the mechanism that fixed the real
height_alpha=0.5 directional-ceiling bug tonight
(`config/ur5e_mujoco_torque_osc_tuned_wrist_orient_fixed.yaml`, see AGENTS.md sec 3) — was added
to the Python controller and had **zero** presence in `hardware/urscript_gen.py` or the
template. This note closes that gap.

**Scope, stated plainly: this is Python-vs-Python numerical parity work only.** There is no
robot access. Nothing here has been run on PolyScope, URSim, or a real UR5e. The tests below
prove the ported arithmetic agrees with the ground-truth controller in Python; they say nothing
about on-robot timing, PolyScope interpreter behavior, or real motion. Do not treat a passing
test here as hardware validation.

## What was ported

`XAxisCartesianImpedanceController.compute()` (`controller_core/x_axis_cartesian_impedance.py`):

```python
tau_orient_wrist = np.zeros(6)
if use_wrist_orientation_task:
    J_rot = J[3:6, :]
    m_wrist = kp_rot_wrist * e_rot - kd_rot_wrist * omega
    tau_orient_wrist = (J_rot.T @ m_wrist) * WRIST_ORIENTATION_MASK
```

reusing `e_rot = orientation_error_vec_wxyz(quat_ref, quat)` and `omega` already computed for
the (currently zero-gain) `kp_rot`/`kd_rot` wrench term, then summed into the same joint-space
bias as `tau_posture` (flows through the existing geometric backtracking + hard clip
unconditionally, no bypass).

Unlike `kp_rot` (damping-only rotation on the URScript side, no orientation-error term at all —
the generator refuses nonzero `kp_rot`), this term needed **new quaternion math the template
previously had no reason to have**: `get_actual_tcp_pose()` returns a UR rotation vector
`[rx, ry, rz]`, not a quaternion, and the Python side's `e_rot` is quaternion-based
(`orientation_error_vec_wxyz`). Four new URScript helpers were added
(`assets/urscript/x_axis_osc_inner.script.template`, before `def osc_x_axis_transport():`):

- `rotvec_to_quat(rv)` — mirrors `controller_core.kinematics_utils.rotvec_to_quat_wxyz` exactly
  (the same convention `hardware/direct_torque_link.py` already uses on the real-hardware
  Python side to build `ee_quat` from `tcp[3:6]`).
- `quat_norm`, `quat_conj`, `quat_mul` — the Hamilton-product primitives
  `orientation_error_vec_wxyz` is built from.
- `orientation_error_vec(quat_des, quat_cur)` — `q_err = conj(q_des) * q_cur` (normalized,
  sign-flipped so `w >= 0`), `e = 2*vec(q_err)`, matching
  `controller_core.kinematics_utils.orientation_error_vec_wxyz` line for line.

Unlike the cond(J) port, **none of this is an approximation** — quaternion arithmetic is exact
floating-point math on both sides, no eigendecomposition or other iterative estimate involved.
Parity numbers below are near machine precision as a result (~1e-15), not the ~1e-9–1e-11
tolerances the singular-scaling port needed.

New wiring (`hardware/urscript_gen.py`):
- `UrscriptOscParams.use_wrist_orientation_task` (bool, default `False`),
  `kp_rot_wrist`/`kd_rot_wrist` (float, default `0.0`), sourced in `load_params_from_yaml` from
  the same YAML keys the Python side reads (`controller.wrist_orientation_task`,
  `controller.gains.kp_rot_wrist`/`kd_rot_wrist`).
- `render_urscript()`: three new placeholders, `{{USE_WRIST_ORIENT}}` (baked `"1"`/`"0"`),
  `{{KP_ROT_WRIST}}`, `{{KD_ROT_WRIST}}`.
- The fixed `WRIST_ORIENTATION_MASK` (`[0, 0, 0, 1.25/1.55, 1.0, 1.25/1.55]`, joint order
  shoulder_pan…wrist_3) is **not** a config field on the Python side either — it's baked as a
  literal in the template (`wrist_mask = [0.0, 0.0, 0.0, 0.8064516129032258, 1.0,
  0.8064516129032258]`, the exact float64 value of `1.25/1.55`), matching the Python source's
  own hardcoded constant.
- Template wiring follows the exact style precedent of `use_nullspace`/`use_lambda`: baked
  runtime `use_wrist_orient` 0/1 flag, term computed unconditionally as a zero vector when off,
  summed unconditionally into `tau_nominal` — not compile-time conditional templating.

### On "byte-identical for the default config" — clarifying the standard applied

The task asked for the default config's rendered output to be "byte-identical" to before. This
codebase's own established meaning of that phrase for this exact kind of change is **numerical**,
not literal source-text identity — see `docs/status/wrist_orientation_task_2026-07-29.md`'s own
line "Default-off path is byte-identical to before (unit test
`test_wrist_orientation_task_off_by_default_and_zero_when_disabled`)", which is an
`np.testing.assert_allclose` check, not a text diff, and every prior URScript port
(nullspace, singular scaling) changed the template's literal text unconditionally while
preserving *behavior* for configs that didn't opt in. This port follows the same convention:
the rendered text for `config/ur5e_mujoco_torque_osc_tuned.yaml` (DEFAULT_CONFIG, which does not
set `wrist_orientation_task`) does change (new helper functions, new baked params, new
unconditional-but-zero term in the loop — see the diff below), but the **arithmetic it produces
is unaffected**: `use_wrist_orient=0` skips the `if` block entirely, `tau_orient_wrist` stays the
zero-initialized vector, and `tau_nominal[i] = tau_task[i] + tau_damp[i] + tau_post[i] +
tau_orient_wrist[i]` is a no-op addition. Verified directly:

```
$ git diff (template only, default-config generated script)
37a38,48   +  new header bullet documenting the port
210a222,276  +  4 new quaternion helper function defs (unused when off)
228a295,302  +  use_wrist_orient=0 / kp_rot_wrist=0 / kd_rot_wrist=0 / wrist_mask baked (unused when off)
248a323    +  quat0 = rotvec_to_quat(...) (captured but never read when off)
376a452,472 +  the gated if-block (skipped entirely, tau_orient_wrist stays zero)
380c476    tau_nominal[i] = ... + tau_post[i]  ->  ... + tau_post[i] + tau_orient_wrist[i]  (adds 0.0)
```

If a stricter, literal-text-diff reading of "byte-identical" was actually intended, that would
require compile-time-conditional templating (omitting the new code entirely from the rendered
text when the flag is off) rather than the runtime-flag pattern every other port in this file
uses — flagging this interpretation explicitly so it can be corrected if wrong, rather than
silently picking one.

## Numerical parity (measured)

`tests/hardware/test_urscript_parity.py`, new section 6:

- `test_wrist_orientation_task_default_config_bakes_flag_off` — DEFAULT_CONFIG bakes
  `use_wrist_orient = 0`, `kp_rot_wrist = 0`, `kd_rot_wrist = 0`.
- `test_wrist_orientation_task_fixed_config_bakes_flag_on` —
  `config/ur5e_mujoco_torque_osc_tuned_wrist_orient_fixed.yaml` bakes `use_wrist_orient = 1`,
  the real validated gains (`kp_rot_wrist=0.0, kd_rot_wrist=10.0`), and the new helper
  functions are present in the rendered text.
- `test_wrist_orientation_task_off_matches_pre_existing_behavior` — flag off with nonzero gains
  still set (proves the flag itself gates the term, not just the gains happening to be zero):
  `tau_orient_wrist == 0` and `tau` matches a reference run that never had the term, on both the
  Python controller and the URScript reference oracle, 20 randomized states, `atol=1e-12`/`1e-9`.
- `test_wrist_orientation_task_matches_python` — the main new test: `compute()` with
  `wrist_orientation_task=True` vs. the Python transcription of the new URScript math
  (`_urscript_reference_tau(use_wrist_orient=True, ...)`), across 4 poses × 2 gain pairs = 8
  cases: 3 randomized poses plus **`hardware.poses.HEIGHT_ALPHA_0_5_Q` specifically** — the
  exact pose whose directional-ceiling failure this term fixed tonight — each run with both the
  shipped damping-only gains (`kp_rot_wrist=0, kd_rot_wrist=10`) and a synthetic nonzero
  `kp_rot_wrist=15` so the proportional (`e_rot`) branch of the formula is actually exercised
  (the shipped config deliberately uses `kp_rot_wrist=0` for stability, so a damping-only test
  alone would never catch a bug in that branch — see
  `docs/status/wrist_orientation_task_2026-07-29.md`'s own instability finding for why it's 0).

Measured (script run once for this doc, seed 42, `nullspace_posture=True` + `task_space_inertia_shaping=True`
matching the real fixed config):

| pose | kp_rot_wrist | kd_rot_wrist | max\|Δtau_orient_wrist\| (Nm) | max\|Δtau\| (Nm) | max\|tau_orient_wrist\| (Nm) |
|---|---|---|---|---|---|
| random | 0.0 | 10.0 | 0.0 | 3.6e-15 | 0.232 |
| random | 15.0 | 6.0 | 0.0 | 3.6e-15 | 0.541 |
| random | 0.0 | 10.0 | 0.0 | 8.9e-16 | 0.086 |
| random | 15.0 | 6.0 | 0.0 | 4.4e-16 | 0.115 |
| random | 0.0 | 10.0 | 0.0 | 3.6e-15 | 0.573 |
| random | 15.0 | 6.0 | 0.0 | 1.8e-15 | 0.038 |
| **height_alpha=0.5** | 0.0 | 10.0 | 0.0 | 7.1e-15 | 0.055 |
| **height_alpha=0.5** | 15.0 | 6.0 | 7.5e-16 | 4.4e-16 | 0.178 |

Overall max difference across all 8 cases: `tau_orient_wrist` 7.5e-16 Nm, final `tau` 7.1e-15 Nm
— machine precision, as expected for exact (non-approximated) arithmetic on both sides. This is
tighter than the `atol=1e-8`/`1e-9` this module's other tests use for the wrench-shaping/
nullspace terms (which involve matrix inversion) and far tighter than the cond(J) estimate's
documented ~1e-11 Nm (an actual approximation, not exact math).

## Test results

```
$ python -m pytest -q tests/hardware/test_urscript_parity.py -v
10 passed in 0.85s   (4 pre-existing test functions unaffected + 6 new: 2 config-bake checks,
                      1 off-path-unchanged check, 1 main matches_python parity test — note some
                      of the 10 are parametrized-by-loop within a single test function, not
                      separate pytest items; 10 is the actual collected+passed count)

$ python -m pytest -q tests/hardware/test_urscript_gen.py
4 passed
```

Broader regression check (`python -m pytest -q -m "unit or mujoco or hardware"`, excluding two
test modules that fail to import in this environment for unrelated reasons —
`test_gain_scheduling_env.py`/`test_train_gain_scheduler.py` need `gymnasium`/`stable_baselines3`,
not installed here): **23 failed, 419 passed, 4 skipped, 1 xfailed, 5 errors**. Every one of
those 23 failures + 5 errors was verified pre-existing on the unmodified branch (same failures,
same count, reproduced via `git stash` before re-running the identical failing subset) — all
trace to `ModuleNotFoundError` for `optuna` (`test_auto_tune_gains.py`, `test_suggest_gains.py`,
`test_residual_data_pipeline.py`) or `pinocchio` (`test_noise_injection.py`,
`test_ur5e_mujoco_torque_experiments_refactor_parity.py`,
`test_direct_torque_residual_observer_async.py`,
`test_direct_torque_residual_observer_trace.py`, `test_direct_torque_residual_observer.py`),
neither installed in this conda environment. Zero regressions caused by this change.

## Known caveats (stated explicitly, not buried)

- **No real-hardware validation of any kind.** No robot, no URSim, no PolyScope execution. This
  closes a *feature-parity* gap only — URScript mode (Mode 3) as a whole still has zero
  real-hardware or URSim execution ever, per AGENTS.md sec 4's standing note. Do not read this
  as "URScript now validated for wrist orientation," only as "the math it would run, if it were
  ever run, now agrees with the Python controller in Python."
- **Per-cycle compute cost on real hardware was not measured or budgeted**, same caveat as the
  singular-scaling port: the new quaternion math (one `sqrt`, a handful of trig-free multiplies)
  is cheap relative to the existing two Jacobi eigendecompositions, but nothing here proves the
  combined per-cycle cost fits the real-time budget on PolyScope.
- **`controller_core/x_axis_cartesian_impedance.py` was read closely but not modified** — no bug
  found worth flagging during this pass; the implementation matches its own docstring exactly.
- Out of scope by the task's own instruction and not touched: `friction_feedforward`,
  `y_integral_action`, `lambda_diagonal_shaping`, `lambda_adaptive_regularization`. None of these
  have any URScript-side representation either; that gap is unchanged by this work.

## Files changed

- `hardware/urscript_gen.py` — `UrscriptOscParams` gained `use_wrist_orientation_task`,
  `kp_rot_wrist`, `kd_rot_wrist`; `load_params_from_yaml` reads them from
  `controller.wrist_orientation_task`/`controller.gains.kp_rot_wrist`/`kd_rot_wrist`;
  `render_urscript` bakes three new placeholders.
- `assets/urscript/x_axis_osc_inner.script.template` — new header bullet; 4 new quaternion
  helper functions (`rotvec_to_quat`, `quat_norm`, `quat_conj`, `quat_mul`,
  `orientation_error_vec`); new baked params (`use_wrist_orient`, `kp_rot_wrist`,
  `kd_rot_wrist`, `wrist_mask`); `quat0` captured alongside `x0`/`y0`/`z0`; new gated
  `tau_orient_wrist` computation per cycle; `tau_nominal` sum now includes it (adds `0.0` when
  the flag is off).
- `tests/hardware/test_urscript_parity.py` — new imports (`rotvec_to_quat_wxyz`,
  `WRIST_ORIENTATION_MASK`, `HEIGHT_ALPHA_0_5_Q`, `WRIST_ORIENT_CONFIG`); new transcription
  helpers (`_urscript_rotvec_to_quat`, `_urscript_quat_norm`, `_urscript_quat_conj`,
  `_urscript_quat_mul`, `_urscript_orientation_error_vec`); `_urscript_reference_tau` gained
  `use_wrist_orient`/`kp_rot_wrist`/`kd_rot_wrist`/`quat0` params (all default off/`None`, so
  every pre-existing call site is unaffected); `_py_config`/`_run_python` gained the same
  optional params, and `_run_python` now derives `ee_quat` from `tcp[3:6]` via
  `rotvec_to_quat_wxyz` instead of a hardcoded identity quaternion (verified a no-op for every
  pre-existing test, since `e_rot` was previously only reachable through `kp_rot`, always 0 in
  this module's configs); 4 new test functions (one of them looping over 8 pose/gain
  combinations internally).
- `docs/status/urscript_wrist_orientation_parity_2026-08-01.md` — this file.

No `controller_core/`, `hardware/safety.py`, or existing config file was modified.

## Tests run

- `python -m pytest -q tests/hardware/test_urscript_parity.py tests/hardware/test_urscript_gen.py`
  — 14 passed.
- `python -m pytest -q -m "unit or mujoco or hardware" --ignore=tests/mujoco/test_gain_scheduling_env.py --ignore=tests/mujoco/test_train_gain_scheduler.py`
  — 419 passed, 23 failed / 5 errors (all pre-existing, verified via `git stash`), 4 skipped,
  1 xfailed.

## Tests not run

- No hardware-in-the-loop, RTDE, URSim, or PolyScope execution of any kind (out of scope; no
  robot/URSim access, and this task is explicitly scoped to a code port).
- No sim rollout / MuJoCo controller-rollout tests specific to `wrist_orientation_task` were
  re-run beyond the existing suite (`tests/unit/test_impedance_dynamics.py`,
  `tests/mujoco/test_wrist_orientation_task.py`) — this task did not touch
  `controller_core/x_axis_cartesian_impedance.py`, so those are unaffected by construction and
  were left to the pre-existing broader regression run above.

## Rollback

```
git revert <this-commit-sha>
```
or, to remove without a revert commit:
```
git checkout <previous-sha> -- hardware/urscript_gen.py assets/urscript/x_axis_osc_inner.script.template tests/hardware/test_urscript_parity.py
rm docs/status/urscript_wrist_orientation_parity_2026-08-01.md
```
The new `UrscriptOscParams` fields default to `use_wrist_orientation_task=False`,
`kp_rot_wrist=0.0`, `kd_rot_wrist=0.0`, and every existing named config (none of which set
`wrist_orientation_task` except `..._wrist_orient.yaml`/`..._wrist_orient_fixed.yaml`, which
were not created or modified by this task) is numerically unaffected — so simply not rendering
a script from a wrist-orientation config is itself a full functional rollback without touching
any code.
