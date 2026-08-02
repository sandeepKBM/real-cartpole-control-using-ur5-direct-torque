# Karnopp stick-slip friction feedforward -- design, evidence, and validation

**Status:** implemented, unit-tested, sim-smoke-tested (canonical_grid at height_alpha=0.5
only). **Not real-hardware validated. Not a recommended replacement for the existing static
`friction_feedforward` default** -- an opt-in third `friction_model` option, same posture as
`"lugre"`.

**CORRECTION, 2026-08-02 (same day, independent verification pass):** the original stuck-branch
implementation described in this doc (`tau_stuck = clip(driving_torque, -fs, fs)`) had a real,
severe bug -- `driving_torque` is built from terms (`tau_task_nominal`, `tau_damping`,
`tau_posture`, `tau_orient_wrist`, `g`) that are ALSO summed into `tau_bias` separately by the
caller, so adding a clipped copy of them as feedforward roughly DOUBLED the real commanded
torque on any stuck joint. Confirmed by direct measurement: full-controller `tau` output under
`friction_model="karnopp"` came out at exactly 2x `"static"`'s output for an identical stuck
state and realistic (non-zeroed) gains. The bug was invisible to every test in this file's
original suite because they all use `_zero_gain_config()`, which zeroes `kp_x/kd_x/kp_posture/
kd_posture/kd_joint` -- exactly the terms whose duplication caused it.

Beyond the correctness bug, the underlying premise doesn't hold up either: real static friction
on the physical robot already self-adjusts to hold a stuck joint, up to its real breakaway
limit, without any help -- that IS what "stuck" means. Sending additional feedforward torque
changes neither the robot's real behavior (real friction just absorbs more) nor the
`qdd_residual` diagnostic gap that motivated this feature (that gap reflects an incomplete
rigid-body dynamics MODEL used for prediction, which more commanded torque cannot fix -- fixing
it needs a friction-aware model for the *predictor*, not more feedforward from the controller).
A naive sign flip (cancel instead of match) is also not right -- see `_karnopp_step`'s docstring
in `controller_core/x_axis_cartesian_impedance.py` for the full reasoning.

**Fix applied:** the stuck branch now contributes zero feedforward. This removes the real bug
but does NOT close the qd~=0-forever gap LuGre also cannot close -- that remains open, and needs
a separately-validated design (most plausibly: a friction-aware bias term fed to the residual
observer's own predictor, not a controller-output change) rather than a quick patch. Re-ran the
canonical_grid smoke test (§6 below) with the fix: **karnopp now scores 6/8, matching the static
baseline exactly** (same two cells fail, same `hold_phase_target_tracking` reason) -- the
originally-reported 8/8 was the double-counting bug producing an illusory improvement, not a
real one. Regression tests added: `test_karnopp_stuck_branch_contributes_zero_feedforward`,
`test_karnopp_stuck_matches_static_zero_velocity_output_exactly`, and, critically, a new
full-controller test with realistic (non-zeroed) gains --
`test_karnopp_stuck_produces_identical_full_controller_output_to_static` -- that explicitly
guards against the exact regression found (`karnopp.tau == 2 * static.tau`). All 14 tests in
this file pass; full `pytest -q -m unit` (195 tests) and the sliding-regime tests are unaffected
by this fix (only the stuck branch changed). The §6 table and narrative below are left as
originally written for the historical record, but are now KNOWN INCORRECT -- see this
correction for the honest result.

## 0. Important: this task's premise needed correcting first

The task brief described `docs/hardware/LUGRE_FRICTION_MODEL_PLAN.md` as "a plan, not yet
implemented" and asked for a stiction-capable friction model to be built from scratch. That
premise is stale. Reading the actual repo state (not just the brief) found:

- LuGre is **already fully implemented**, in `controller_core/x_axis_cartesian_impedance.py`,
  landed commit `58ede2c` ("Add opt-in LuGre dynamic friction feedforward (Part B of the plan
  doc)"), 2026-08-01 18:36 -0400 -- config field `friction_model: Literal["static", "lugre"]`,
  persistent `_friction_z` state, `_lugre_step()`, YAML parsing, `config/
  ur5e_mujoco_torque_osc_tuned_friction_ff_lugre.yaml`, and an 11-test suite
  (`tests/unit/test_lugre_friction.py`), all present before this session started.
- That same commit's own message, and `docs/status/lugre_friction_feedforward_2026-08-01.md`
  (also already present), **already found and documented** that LuGre is functionally a no-op
  at hold-phase velocities: a canonical_grid sweep showed LuGre's per-cell pass/fail pattern
  byte-identical to no-feedforward-at-all baseline (3/8 both), while the existing static model
  scored 7/8 on the same sweep -- LuGre provided zero of the static model's four rescues.
  Root cause, already derived in that doc: the ODE's relaxation time constant is
  approximately `g(qd)/|qd|`, which is on the order of **hundreds of seconds** at realistic
  hold-phase velocities (measured hold-phase `|qd|` peaked at 0.0085 rad/s in that doc's own
  test) -- far beyond any real transport-move window.

This is not a tangential detail -- it is the load-bearing fact that shapes everything below.
Section 1 shows the new real-hardware evidence this task supplied is a second, independent,
real-world confirmation of the *exact same* structural limitation, not a new problem.

## 1. New evidence, independently re-derived (not trusted from the task brief)

Trace analyzed: `outputs/hardware_transport_remote/hardware_transport/direct_torque_20260802_172059/`
(real UR5e hardware, `config/ur5e_mujoco_torque_osc_tuned_split_base_wrist.yaml`,
`friction_feedforward: true`, `friction_model` left at its default `"static"`). Re-derived
directly from `trace.jsonl` (3190 rows) with a standalone numpy script, not copied from the
task brief:

| quantity | brief's claim | independently re-derived | verdict |
|---|---|---|---|
| wrist_1 `qd` range | `[-0.000, 0.000]` | `[-1.15e-4, +8.5e-5]` rad/s | matches (display rounding) |
| wrist_1 total `q` motion | "never moved at all" | 0.000126 rad total range | matches |
| `qdd_residual` (wrist_1) growth | "grew to -4.35 rad/s^2" | first=0.018, **last=-5.39**, min=-6.97 rad/s^2 | same sign/order of magnitude, **numerically more extreme** than quoted, not less |
| corr(t, `qdd_residual`) | -0.80 | **-0.797** | matches |
| corr(qd, `qdd_residual`) | -0.23 | **-0.225** | matches |
| run length | "8-second run" | trace covers **6.38s** of a planned 8s-move + 2s-hold (10s total) | the run tripped a real safety guard early -- see below, not a discrepancy in the friction claim |

**One thing the brief didn't mention, found while verifying**: this specific run's
`termination_reason` is `"TCP acceleration 1.7275 m/s^2 > 0.5 m/s^2 for 3 consecutive cycles"`
(`success: false`, `valid_move_and_hold: false`, from `summary.json`) -- the real run was cut
short by `CartesianMoveMonitor`'s TCP-acceleration guard, not a clean completion. The
`qdd_residual` trace analyzed above runs right up to that trip. Whether the growing wrist_1
static-friction holding torque is causally connected to that TCP-accel trip (e.g. some other
joint's compensating motion accelerating once wrist_1 finally lets go) was **not investigated**
this session -- flagged here as a real, open question for whoever picks up real-hardware
follow-up, not resolved by this document.

Also independently checked (not in the brief): per-joint `|qd|` statistics across the whole
trace, used below to ground the new model's default thresholds in real telemetry rather than a
pure guess:

| joint | `|qd|` median | `|qd|` p90 | `|qd|` max |
|---|---|---|---|
| shoulder_pan | 0.0 | 0.0 | 0.0013 |
| shoulder_lift (the joint actually doing the commanded move) | 0.034 | 0.050 | 0.058 |
| elbow | 0.0 | 0.0 | 0.0019 |
| wrist_1 (the stuck joint) | 0.0 | 0.0 | **0.000115** |
| wrist_2 | 0.0 | 0.0 | 0.0089 |
| wrist_3 | 0.0 | 0.0 | 0.0065 |

**Verdict on the evidence: real, correctly characterized (modulo the two cosmetic
discrepancies above, both non-material), and it is a second independent confirmation of a
structural gap already found and documented the day before, not a new discovery on its own.**

## 2. Why LuGre (already built) doesn't close this gap, re-derived independently

`tests/unit/test_lugre_friction.py::test_lugre_zero_velocity_leaves_z_and_torque_at_zero`
already proves, algebraically: at `qd == 0` exactly, both terms of
`dz/dt = qd - |qd|*z/g(qd)` vanish, so `z` (and therefore `tau_friction_ff = sigma0*z +
sigma1*dz/dt + sigma2*qd`) never departs from its current value. Wrist_1's `|qd|` never
exceeded `1.15e-4` rad/s for the entire trace -- at that velocity, even accepting some nonzero
`qd`, the ODE's own relaxation time constant `g(qd)/|qd|` (with `g` order 1-6.5 Nm for this
repo's joints) is on the order of **tens of thousands of seconds**, i.e. far past "hundreds of
seconds" the 2026-08-01 doc already found at a 70x larger `|qd|` (0.0085 rad/s). **If LuGre had
been active in this exact real run instead of the static model, it would have produced
essentially zero compensation too** -- this is a direct, quantitative implication of the
already-existing z-dynamics, not a new finding requiring new code to demonstrate.

## 3. Design choice: Karnopp stick-slip switching, not a LuGre retune

Three options were on the table (per the task brief): retune/fix LuGre, build the
Clochiatti-style asymmetric-Coulomb model, or something else with its own justification.

**Rejected: retuning LuGre's parameters.** The relaxation-time problem is structural, not a
parameter-tuning problem -- `g(qd)/|qd|` diverges as `qd -> 0` for *any* choice of
`sigma0`/`sigma1`/`sigma2`/`Fc`/`Fs`/`vs` (none of those parameters appear in the relaxation
time formula at all). A joint that is truly, persistently at `qd ~= 0` for its whole hold phase
cannot be helped by retuning this ODE.

**Rejected (for this task): the Clochiatti asymmetric-Coulomb model.** That model's own
distinguishing feature (asymmetric friction keyed to *power-flow direction* through the
harmonic drive, not just velocity sign) targets a different phenomenon -- a friction floor that
differs between "motor driving load" and "load driving motor" -- and is still fundamentally a
function of velocity/direction, inheriting the same qd~=0 blind spot this evidence is about. It
also requires joint-current-based system identification this repo has no tooling for yet. Kept
as a candidate for a future session, not built here.

**Chosen: Karnopp stick-slip switching** (D. Karnopp, "Computer simulation of stick-slip
friction in mechanical dynamic systems," ASME J. Dyn. Sys. Meas. Control, 1985 -- a classical,
textbook-standard model; this session did not run a fresh adversarially-verified literature
pass on it the way `docs/status/literature_review_dynamics_and_sensor_noise_identification_
2026-08-01.md` did for its own claims, so treat the model's currency/name/year as reliable
textbook knowledge, not a freshly-verified citation). Rationale:

- It is a **velocity-hysteresis switching model, not a continuously-integrated ODE** -- no
  relaxation-time problem exists because there is no relaxation dynamics to speak of. Below a
  "stuck" velocity band, the model simply asserts the joint isn't sliding and cancels whatever
  net non-friction torque is being applied, up to a static ceiling.
- Critically, the compensation magnitude is driven by the **already-computed net driving
  torque** (task + damping + posture + wrist-orientation + gravity, all already available as
  local variables at the point in `compute()` where friction feedforward is applied) -- **not
  by qd**. This is exactly the missing ingredient: a joint whose commanded torque is growing
  while its velocity stays at zero (precisely wrist_1's real trace) now gets a feedforward term
  that grows with it, instead of one that requires velocity to exist at all.
- Simple, bounded, and easy to reason about: no new ODE integration, no new stability question
  (Section 6's smoke test's zero guard trips confirm this empirically too).

## 4. What was built

`controller_core/x_axis_cartesian_impedance.py` (`controller_core/` stays numpy-only, no new
imports):

- `friction_model: Literal["static", "lugre", "karnopp"]` -- extends the existing Literal,
  default unchanged (`"static"`).
- New config fields `karnopp_qd_stick_enter_radps` / `karnopp_qd_stick_exit_radps`
  (per-joint arrays, defaults 0.005 / 0.02 rad/s -- grounded in this session's own real-trace
  `|qd|` statistics above: wrist_1 (truly stuck) never exceeded 1.15e-4 rad/s, shoulder_lift
  (actually moving) had median 0.034 / p90 0.050 rad/s; these thresholds sit comfortably
  between the two with room for the hysteresis band). Reuses `lugre_fc_nm`/`lugre_fs_nm`
  (kinetic Coulomb floor / static breakaway ceiling) and `friction_ff_viscous` for the sliding
  regime rather than duplicating those fields.
- `_parse_friction_model()` module-level helper -- factored out of
  `from_controller_yaml_section` so a third literal value didn't need a nested ternary; matches
  the pre-existing permissive convention (unrecognized string silently falls back to `"static"`
  rather than raising).
- Persistent per-joint hysteresis latch `self._karnopp_stuck` (bool array), same lifecycle
  class as `self._friction_z`/`self._y_integral`: initialized `True` (stuck) in `__init__`,
  reset to `True` in `reset_from_state()`, **not** touched by `set_gains()` (matches that
  method's documented contract).
- `_karnopp_step(qd, driving_torque)` -- the switching law itself (see its docstring in the
  file for the full derivation). No `dt` needed (unlike `_lugre_step`): the only state is which
  regime each joint is latched into.
- Wired into `compute()`'s existing `friction_feedforward` `if/elif` chain, alongside `"lugre"`
  and the static branch -- flows through the identical downstream backtrack/clip pipeline, no
  bypass.
- **One structural change required to wire this in**: the gravity-torque extraction (`g = ...`)
  was moved to *before* the friction-feedforward block (previously it ran after). This is a
  reordering of two mutually independent computations -- gravity doesn't depend on anything the
  friction block computes, and the pre-existing static/lugre branches never read `g` -- verified
  byte-identical via `tests/unit/test_karnopp_friction.py::test_gravity_reorder_does_not_change_
  static_model_output` and the full pre-existing suite passing unchanged (Section 7).
- `CartesianImpedanceOutput` gained one new diagnostic field, `friction_karnopp_stuck` (float
  0/1 per joint, exposing the hysteresis latch state for post-hoc trace analysis).

New config: `config/ur5e_mujoco_torque_osc_tuned_split_base_wrist_karnopp_friction.yaml` --
byte-identical to `config/ur5e_mujoco_torque_osc_tuned_split_base_wrist.yaml` (the exact config
that produced the real evidence above) except `friction_model: "karnopp"` plus the two new
`karnopp_qd_stick_*` blocks. The base config is untouched.

New tests: `tests/unit/test_karnopp_friction.py`, 15 tests, pure numpy:
- flag-off / default-value regression (`friction_model` stays `"static"` unless explicitly
  set; the gravity-reorder regression check above).
- directional correctness: stuck branch cancels driving torque up to `Fs` with matching sign;
  sliding branch matches `fc*sign(qd) + viscous*qd` exactly; odd velocity-sign symmetry.
- hysteresis: a velocity in the dead zone between enter/exit holds the previous latch state
  (no chatter); `reset_from_state()` resets the latch to all-stuck.
- the core requirement-5 claim, built as two synthetic scenarios (no real hardware available):
  a single joint held at `qd == 0` while a driving torque (`gravity_torque`, standing in for
  "whatever the rest of the controller is applying") ramps linearly -- the static model's
  residual equals the full ramp at every step (0% compensated, by construction of
  `tanh(0/deadband) == 0`), Karnopp's residual is ~0 throughout (near-exact cancellation, up to
  `1e-8` numerical tolerance) until the ramp is deliberately pushed past `Fs`, where the
  residual saturates but is still strictly smaller than the static model's.

## 5. Existing evidence this design should NOT claim to fully resolve

- **No real breakaway-velocity calibration exists for this arm.** `karnopp_qd_stick_enter_
  radps`/`_exit_radps` are grounded in *this session's own* real-trace `|qd|` statistics (a
  genuine improvement over a pure guess), but that is one run, one pose, one direction -- not a
  dedicated breakaway measurement. Same unresolved gap the LuGre plan doc already flags for its
  own parameters (`docs/hardware/LUGRE_FRICTION_MODEL_PLAN.md` sec 4).
- **`lugre_fc_nm`/`lugre_fs_nm` (reused here) are still the original 2026-07-31 placeholder
  values**, not independently validated for the Karnopp use case.
- This model still cannot distinguish "genuinely stuck against static friction" from "at rest
  because the task is actually satisfied" -- both look identical as `driving_torque` near zero
  and `qd` near zero. In practice this is harmless (a near-zero driving torque produces a
  near-zero compensation either way), but it's worth stating plainly: this is not a
  fault-detection mechanism, just a compensation term.

## 6. Sim smoke test (canonical_grid, height_alpha=0.5 -- the exact pose/config that produced
   the real evidence)

Ran `tools/ur5e_pose_sweep_transport.py --height-alphas 0.5 --categories canonical_grid --seed
0` against both the base config (friction_feedforward on, `friction_model` at its default
`"static"`) and the new karnopp config, both built on `config/
ur5e_mujoco_torque_osc_tuned_split_base_wrist.yaml`'s gains/flags unchanged:

| dx (m) | hold (s) | static: `valid_move_and_hold` | karnopp: `valid_move_and_hold` | changed? |
|---|---|---|---|---|
| 0.01 | 1.0 | **fail** (`hold_phase_target_tracking`) | **pass** | yes |
| 0.01 | 2.0 | **fail** (`hold_phase_target_tracking`) | **pass** | yes |
| 0.02 | 1.0 | pass | pass | no |
| 0.02 | 2.0 | pass | pass | no |
| 0.03 | 1.0 | pass | pass | no |
| 0.03 | 2.0 | pass | pass | no |
| 0.04 | 1.0 | pass | pass | no |
| 0.04 | 2.0 | pass | pass | no |

**Totals: static 6/8, karnopp 8/8 -- zero regressions, two rescues, both via
`hold_phase_target_tracking`** (matching the 6/8 number this exact base config's own prior
validation doc, `docs/status/split_base_wrist_impedance_2026-08-01.md`, already reports for
this pose). Every run in both sweeps terminated via `duration_complete` -- no guard trips, no
new instability introduced.

Per-cell detail on the two rescued cells (both from each run's own `summary.json`):

| | dx=0.01, hold=1.0 (static -> karnopp) | dx=0.01, hold=2.0 (static -> karnopp) |
|---|---|---|
| `hold_phase_final_x_error_m` | 8.12e-4 -> **3.20e-4** | 2.53e-4 -> **6.37e-5** |
| `move_hold_quality_score` | 0.247 -> **0.346** | 0.234 -> **0.335** |
| `hold_failure_reason` | `hold_phase_target_tracking` -> `none` | `hold_phase_target_tracking` -> `none` |

This is a real, positive, honestly-measured result at the smallest-displacement cells -- the
regime where the required steady-state holding torque is small enough to stay comfortably under
`Fs`, and where the static model's velocity-only compensation contributes essentially nothing
during a settled hold (`qd` already near zero there too).

**What was NOT run** (time-scoped, not hidden): the full 4-category (`long_holds`,
`large_displacements`, `torque_scale_robustness` in addition to `canonical_grid`) x
3-`height_alpha` (0.2/0.3/0.5) sweep this repo's established rigor-sweep pattern uses for a
promotion decision (AGENTS.md sec 3). This is a smoke test consistent with the task's own
"if time allows" framing for sim validation, not a full validation pass -- do not treat 8/8 on
one category/one pose as proof this generalizes.

## 7. Test results (full, honest)

- `python -m pytest -q tests/unit/test_karnopp_friction.py` -- **15/15 pass**.
- `python -m pytest -q -m unit` -- **194/194 pass**, 439 deselected. No pre-existing failures
  in this session's environment (unlike the 2026-08-01 LuGre session, this environment does
  have `pinocchio` installed -- confirmed no `ModuleNotFoundError` anywhere).
- `python -m pytest -q -m mujoco` -- **160 passed, 3 xfailed** (pre-existing expected-fail
  cases, unrelated to this change), 470 deselected, ~77s.
- `-m hardware` was **not run** -- out of scope per the task's explicit instruction not to
  exercise anything that could touch a real robot beyond what's already mocked.

No test failures, no regressions, in either the pre-existing suite or the new file.

## 8. Explicitly NOT done (flagged for a human decision, not silently skipped)

- **No real-hardware validation of any kind.** Matches this repo's own established discipline
  for every new friction model landed so far (static, then LuGre, now this) -- sim-only until a
  deliberate real-lab decision.
- **No real parameter calibration** -- `karnopp_qd_stick_enter_radps`/`_exit_radps` are grounded
  in one real trace's telemetry (better than a pure guess, per Section 4) but not a dedicated
  breakaway-velocity measurement; `Fc`/`Fs` are unchanged 2026-07-31 placeholders.
- **No investigation of whether wrist_1's stiction is causally connected to this exact run's
  TCP-acceleration guard trip** (Section 1) -- flagged as a real, open question, not resolved
  here.
- **No full 4-category / 3-height_alpha sweep** -- only the one category/pose smoke test in
  Section 6.
- **No attempt to build or compare the Clochiatti asymmetric-Coulomb alternative** -- noted as
  rejected-for-this-session in Section 3, not evaluated empirically.

## Files changed

- `controller_core/x_axis_cartesian_impedance.py` (karnopp config fields, `_karnopp_stuck`
  state, `_karnopp_step()`, `compute()` branch + gravity reorder, `_parse_friction_model()`
  helper, new output field, YAML parsing).
- `config/ur5e_mujoco_torque_osc_tuned_split_base_wrist_karnopp_friction.yaml` (new).
- `tests/unit/test_karnopp_friction.py` (new, 15 tests).
- `docs/status/karnopp_stiction_friction_model_2026-08-02.md` (this file).

## Tests run

See Section 7.

## Rollback

`git diff controller_core/x_axis_cartesian_impedance.py` and revert manually, or `git checkout
-- controller_core/x_axis_cartesian_impedance.py` (this file was the only pre-existing file
touched). Delete the two new files
(`config/ur5e_mujoco_torque_osc_tuned_split_base_wrist_karnopp_friction.yaml`,
`tests/unit/test_karnopp_friction.py`) and this doc for a full rollback. `friction_model`
defaults to `"static"` everywhere else, so a partial rollback (keeping the code, never
selecting `"karnopp"`) changes no existing config's behavior and requires no file changes at
all -- verified by the full pre-existing suite passing unchanged (Section 7).
