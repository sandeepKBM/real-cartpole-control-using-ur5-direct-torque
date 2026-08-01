# LuGre dynamic friction feedforward -- implementation + sim validation

**Status:** implemented per `docs/hardware/LUGRE_FRICTION_MODEL_PLAN.md` Part B (controller-side
feedforward only). Sim-validated through plan sec 6 items 1-3 (smoke test, a targeted
canonical-grid regression check, explicit comparison vs the static model). Item 2's full
4-category/3-height_alpha sweep was NOT run (see "How far through validation" below). **Not
real-hardware validated. Not a recommended replacement for `friction_feedforward`'s existing
static model** -- see the negative result below.

## 1. What was built

`controller_core/x_axis_cartesian_impedance.py` (`CartesianImpedanceConfig`):
- `friction_model: Literal["static", "lugre"] = "static"` -- nested under the existing
  `friction_feedforward` bool per the plan's own reasoning (only consulted when that bool is
  True; default preserves every existing config's behavior byte-for-byte).
- Six new per-joint LuGre parameter arrays: `lugre_sigma0_nm_per_rad`,
  `lugre_sigma1_nm_s_per_rad`, `lugre_sigma2_nm_s_per_rad`, `lugre_fc_nm`, `lugre_fs_nm`,
  `lugre_vs_radps`, parsed in `from_controller_yaml_section` with the same per-joint-name-dict
  pattern as `friction_ff_coulomb_nm`.
- `self._friction_z` -- persistent per-joint bristle-deflection state, initialized in
  `__init__`, zeroed in `reset_from_state()` (same lifecycle as `_q_rest`/`_y_integral`),
  **not** touched by `set_gains()` (matches that method's documented contract).
- `_lugre_step(qd, dt)` -- implements the plan's exact ODE:
  `dz/dt = qd - |qd|*z/g(qd)`, `g(qd) = Fc + (Fs-Fc)*exp(-(qd/vs)^2)`,
  `tau_friction = sigma0*z + sigma1*dz/dt + sigma2*qd`, explicit Euler at the real per-cycle
  `dt_s` (falls back to 1/500s when absent, matching `y_integral_action`'s identical fallback).
  Wired into `compute()`'s existing `friction_feedforward` branch point, flowing through the
  same backtrack/clip pipeline as the static term -- no bypass.
- `CartesianImpedanceOutput` gained `friction_model_used: str` and `friction_z: np.ndarray`
  (both auto-exposed via `vars(output)` to `controller_output` in sim traces; also added
  `tau_friction_ff`/`friction_z` explicitly to the two trace-row dict literals in
  `tools/ur5e_mujoco_torque_experiments.py` so they land in `trace.jsonl`, matching this repo's
  established practice of adding targeted per-cycle instrumentation for a specific diagnosis).
- New config `config/ur5e_mujoco_torque_osc_tuned_friction_ff_lugre.yaml` (copy of
  `..._friction_ff.yaml` with `friction_model: lugre` + the six LuGre fields). Existing
  `..._friction_ff.yaml` untouched.
- New unit test `tests/unit/test_lugre_friction.py` (11 tests, pure numpy, no MuJoCo): default-off
  behavior, z persistence/reset lifecycle, zero-velocity no-op, a "held against a wall"
  monotonic-rise-then-plateau ODE test (plan sec 6 item 4), Stribeck-bound invariant, odd
  velocity-sign symmetry, `dt_s` fallback, backtrack/clip flow-through, YAML parsing. All pass.

## 2. Placeholder parameter values and reasoning

No authoritative UR5e-class LuGre table exists (confirmed by the plan's own literature pass).
Values chosen, and why:

- `lugre_fc_nm` = `friction_ff_coulomb_nm` exactly (5.0/1.0 Nm size3/size1) -- same physical
  quantity (Coulomb/kinetic floor).
- `lugre_sigma2_nm_s_per_rad` = `friction_ff_viscous` exactly (0.4/0.15) -- same role (viscous
  coefficient).
- `lugre_fs_nm` = 1.3x `lugre_fc_nm` (6.5/1.3) -- a modest, explicitly-a-guess breakaway
  multiplier (Fs must be > Fc).
- `lugre_sigma0_nm_per_rad` = 1.0 uniformly. **This required a real derivation, not a literature
  lookup**, because the plan's ODE is written literally as given (`g(qd)` in Nm, NOT divided by
  sigma0 the way textbook LuGre normalizes it -- confirmed this is what the plan's own sec 3.2
  code block specifies, and the task brief explicitly asked for the ODE "exactly as specified").
  Under that literal form, a standalone numeric check (before touching the config) showed `z` is
  bounded within roughly `+/-g(qd)` (i.e. ~`+/-Fs` at low speed) **regardless of sigma0** -- sigma0
  alone then sets `sigma0*z`'s torque scale. sigma0=1.0 keeps that steady-state torque comparable
  in magnitude to Fc/Fs themselves; sigma0=100 on the same trajectory produced an unphysical ~15
  Nm jump in the same check. This is a deliberate, documented deviation from "typical published
  large sigma0" literature values, which apply to the textbook meters-normalized form, not this
  plan's literal one.
- `lugre_sigma1_nm_s_per_rad` = 2x the matching sigma2/viscous value (0.8/0.3) -- a modest, bounded
  addition on top of the dominant sigma0*z term; not independently identified (plan sec 4 flags
  sigma1 as needing a dedicated stick-slip test or qualitative tuning -- neither was done).
- `lugre_vs_radps` = 0.02 uniformly, within the plan's own cited Stribeck-velocity sweep range
  (0.01-0.2 rad/s).

## 3. Smoke test result (plan sec 6 item 1) -- honest finding

Single dx=0.04m move(1s)-hold(2s) run, `config/ur5e_mujoco_torque_osc_tuned.yaml`'s gains,
`--gravity-source mujoco_qfrc` (pinocchio is not installed in this session's Python env --
confirmed via `python -c "import pinocchio"` failing outside this change entirely; the
`--gravity-source`/native-MuJoCo-Coriolis code path already exists and is exercised identically
by both this smoke test and the pre-existing test suite, so this is not new surface):

| variant | `achieved_x_delta_m` (target 0.04) | `hold_phase_final_x_error_m` | `move_hold_quality_score` |
|---|---|---|---|
| no feedforward (baseline) | 0.03974 | 2.57e-4 | 0.290 |
| static (`friction_feedforward`) | 0.03995 | 4.61e-5 | 0.492 |
| LuGre (this work) | 0.03978 | 2.17e-4 | 0.292 |

**LuGre is nearly indistinguishable from the no-feedforward baseline, and clearly worse than the
static model, at these placeholder parameters.** Not because of instability -- the opposite:
inspected the actual `friction_z`/`tau_friction_ff` trace (now logged per-cycle) and found `z`
stays 1-2 orders of magnitude below `Fc`/`Fs` throughout the whole 3s run (e.g. shoulder_lift
joint: `z` = -0.0326 at hold-start, -0.0346 at hold-end, vs. `Fc`=5.0/`Fs`=6.5 Nm), growing
monotonically with a **decaying, converging** growth rate (Δz per 0.5s: 0.00107 → 0.00057 →
0.00030 → 0.00015) -- clean, bounded, non-oscillating convergence, exactly the "not diverging"
half of the plan's own stability caveat. No new `|qd|` guard trips, no growing/unbounded `z`, no
regression vs. the static model's own guard behavior.

**Root cause of the weakness, derived and confirmed, not guessed**: under the plan's literal
(non-sigma0-normalized) ODE, the relaxation time constant governing how fast `z` approaches
`g(qd)` is approximately `g(qd)/|qd|`. With Nm-scale `g` (~5-6.5) and realistic low hold-phase
velocities (this run's hold-phase `|qd|` peaked at 0.0085 rad/s), that time constant is on the
order of **hundreds of seconds** -- far beyond any real transport-move duration (1-10s). `z`
never gets anywhere near its Stribeck-curve asymptote in a realistic move-hold window, so the
resulting feedforward torque (`tau_friction_ff` peaked around 0.03-0.1 Nm in this run) is far
below the static model's coulomb floor (0.4-5 Nm), which is why it barely moves any tracking
metric. This is a genuine property of the plan's literal formula combined with physically-sane
`sigma0`/`Fc`/`Fs` choices, not a bug in the implementation -- verified independently via a
standalone offline ODE simulation (varying `qd` from 0.01 to 5.0 rad/s) before this smoke test,
which predicted exactly this slow-relaxation behavior and its magnitude.

## 4. Regression check (plan sec 6 items 2-3, partial)

Ran `tools/ur5e_pose_sweep_transport.py --categories canonical_grid` at `height_alpha=0.5` (the
pose AGENTS.md documents as most friction-sensitive) for baseline / static / LuGre, using scratch
copies of each config with `gravity_source: mujoco_qfrc` (pinocchio unavailable in this env; scratch
configs only, nothing in `config/` was touched for this). Per-cell (`dx` x `hold_duration`)
`valid_move_and_hold` result:

| dx | hold | baseline | static | LuGre |
|---|---|---|---|---|
| 0.01 | 1.0 | fail | **pass** | fail |
| 0.01 | 2.0 | fail | fail | fail |
| 0.02 | 1.0 | fail | **pass** | fail |
| 0.02 | 2.0 | fail | **pass** | fail |
| 0.03 | 1.0 | pass | pass | pass |
| 0.03 | 2.0 | fail | **pass** | fail |
| 0.04 | 1.0 | pass | pass | pass |
| 0.04 | 2.0 | pass | pass | pass |

Totals: baseline 3/8, static 7/8, LuGre 3/8. **LuGre's per-cell pass/fail pattern is
byte-identical to the no-feedforward baseline in all 8 cells** (confirmed via `valid_move_phase`/
`valid_hold_phase` too, not just the aggregate flag) -- zero regressions vs. baseline, but also
zero of the four rescues the static model provides. This corroborates the smoke test's finding at
a second, independent measurement point using the project's own standard rigor-sweep tooling.

**Not run**: the full 3-`height_alpha` (0.2/0.3/0.5) x 4-category (adding `long_holds`,
`large_displacements`, `torque_scale_robustness`) sweep the plan's sec 6 item 2 specifies (~114
runs total per config). Given the canonical_grid result already shows LuGre is functionally a
no-op at these parameters, and the task explicitly prioritized the smoke test + stability check as
the must-have over rushing to the full sweep, the full sweep was not run this session. The
canonical_grid result here stands in as the regression check (item 3): LuGre never underperforms
baseline anywhere tested, so there is no full-sweep-only regression risk uniquely exposed by the
untested categories that this evidence would predict.

## 5. LuGre vs. static: does it actually behave differently at low/zero velocity in sim?

Per the plan's own explicit non-goal #5 (sim's plant is still static Coulomb+viscous, not LuGre,
so this validates "does the feedforward degrade tracking/safety," never "does it correctly cancel
dynamic friction"): yes, LuGre's `z` state is visibly different in kind from the static model's
instantaneous `tanh(qd/deadband)` -- it is a genuine slow-building history-dependent quantity,
non-zero and monotonically evolving even while `qd` itself is decaying toward zero during hold
(the shoulder_lift `z` trace above keeps growing through the whole hold phase even as `qd` falls
from 0.003 to 0.00004 rad/s). But at these placeholder parameters that difference is currently
**too weak, on transport-move timescales, to produce any measurable tracking or safety benefit**
in this sim's plant. The static model remains the working, validated default
(`friction_feedforward` with `friction_model: "static"`, unchanged).

## 6. What would need to change before LuGre is worth retrying

Not implemented this session (would be new tuning work, out of scope for "implement per plan" +
"do not combine with gain tuning"): either (a) a much larger `sigma0` sized specifically so
`sigma0 * z` reaches meaningful torque before `z` itself saturates (i.e. accepting a torque
overshoot risk that would need its own stability check), or (b) revisiting the ODE to normalize
`g(qd)` by `sigma0` the way textbook LuGre does, giving `z` a genuinely fast (sub-second, at
realistic joint speeds) relaxation timescale the way the plan's own literature review describes.
Both are real design decisions requiring their own justification and validation pass, not
something to silently choose here.

## Files changed

- `controller_core/x_axis_cartesian_impedance.py` (LuGre config fields, `_friction_z` state,
  `_lugre_step()`, `compute()` branch, YAML parsing, new output fields).
- `tools/ur5e_mujoco_torque_experiments.py` (added `tau_friction_ff`/`friction_z` to the two
  trace-row dict literals -- purely additive, no existing key changed).
- `config/ur5e_mujoco_torque_osc_tuned_friction_ff_lugre.yaml` (new).
- `tests/unit/test_lugre_friction.py` (new, 11 tests).
- `docs/status/lugre_friction_feedforward_2026-08-01.md` (this file).

## Tests run

- `python -m pytest -q tests/unit/test_lugre_friction.py` -- 11/11 pass.
- `python -m pytest -q -m "unit or mujoco" --ignore=tests/mujoco/test_gain_scheduling_env.py --ignore=tests/mujoco/test_train_gain_scheduler.py`
  -- 202 passed, 4 skipped, 1 xfailed, 7 failed, 7 errors. The two `--ignore`d files fail to
  collect in this session's env (`gymnasium`/`stable_baselines3` not installed) unrelated to this
  change. All 7 failed + 7 errors trace to `ModuleNotFoundError: No module named 'pinocchio'`
  (`controller_core/model_dynamics.py`) -- confirmed pre-existing (this session's Python env has
  no `pinocchio` at all, unrelated to any file touched here) and identical in count before and
  after this change.

## Rollback

`git revert` the commit, or manually: remove the `friction_model`/`lugre_*` fields and
`_lugre_step`/`_friction_z` from `controller_core/x_axis_cartesian_impedance.py`, remove the two
new trace-row keys from `tools/ur5e_mujoco_torque_experiments.py`, delete
`config/ur5e_mujoco_torque_osc_tuned_friction_ff_lugre.yaml`, `tests/unit/test_lugre_friction.py`,
and this file. No existing config or `controller_core/` default behavior changes with `friction_model`
defaulting to `"static"`, so a partial rollback (keeping the code but never selecting `"lugre"`) is
also safe and requires no file changes at all.
