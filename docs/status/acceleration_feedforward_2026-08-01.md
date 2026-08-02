# Acceleration feedforward for the X-axis Cartesian impedance law (2026-08-01)

## Motivation

`controller_core/x_axis_cartesian_impedance.py`'s X-axis torque law was pure PD on
position+velocity error (`Fx = kp_x*x_err + kd_x*(x_vel_des - vx)`) with no acceleration
feedforward at all -- confirmed by grep before starting this work. A pure PD law is always
reactively chasing tracking error that has already appeared rather than anticipating the
commanded trajectory, which was discussed directly with the user the same night real-hardware
`direct_torque` traces showed tracking lag and per-cycle jitter. The
`accel_duration_triangular`/`accel_duration_scurve` trajectory profiles
(`simulation/ur5e_mujoco_torque.py::x_profile_target`) already compute a reference velocity
analytically; this change adds the matching reference acceleration and threads it into the
controller as an opt-in feedforward term.

## Design

### Step 1 -- reference acceleration

`simulation/ur5e_mujoco_torque.py` gained `x_profile_accel(profile, target_x_delta, t_s,
duration_s, move_duration_s=None, target_accel_mps2=None)`, a **separate function**, not a
third return value on `x_profile_target()` itself. `x_profile_target` has ~15 existing call
sites (this repo's sim experiment engine, `hardware/position_transport.py`,
`rl_gain_scheduling/gain_scheduling_env.py`, three test files) that all unpack exactly
`x, v = x_profile_target(...)`; widening that tuple would force an unrelated edit at every one
of them for a feature only two call sites (the sim experiment engine,
`hardware/direct_torque_transport.py`) actually need today. This is a deliberate deviation from
a literal reading of "thread it through `x_profile_target()`'s return value" in favor of a
smaller, equally-correct blast radius -- `x_profile_target` itself is untouched, byte-for-byte.

`x_profile_accel` is defined as the exact time-derivative of the **velocity**
`x_profile_target` returns, not a fresh re-derivation from the position formula. For
`min_jerk`/`min_jerk_move_hold` those are the same thing (the shipped position/velocity closed
forms are self-consistent; min-jerk's second derivative is analytically zero at both endpoints,
so there is no discontinuity to worry about at the move/hold boundary). For
`accel_duration_scurve` they are **not**: the shipped `x_vel` is `accel*(1-cos(omega*t))`
(`omega = 2*pi/move_duration_s`), whose own closed-form derivative is
`accel*omega*sin(omega*t)` -- not the simpler `accel*sin(2*pi*t/T)` this repo's other comments
use to describe the profile (that simpler form is the derivative of a *different*, un-shipped
velocity profile, and differs from the one `kd_x` actually tracks by a constant factor of
`move_duration_s/(2*pi)`). This was caught by deriving both forms by hand and cross-checking
against the shipped `test_scurve_matches_closed_form_kinematics` test's own worked example
(`v_mid = accel*(1-cos(pi)) = 2*accel`, confirming the shipped, undivided form). `x_profile_accel`
returns the derivative of the velocity the controller actually tracks -- the only choice
consistent with what a feedforward term is supposed to anticipate. Nothing about
`x_profile_target` was changed to arrive at this; it is a pre-existing, real (if harmless)
inconsistency between that function's own position and velocity closed forms for the scurve
profile, noted here and left alone (fixing it would be a trajectory-shape change, out of scope
and explicitly not bundled with this controller change).

Threaded through, purely additively:
- `simulation/ur5e_mujoco_torque.py`: `MujocoUR5eState.target_x_accel` (default `0.0`), included
  unconditionally in `as_robot_state()`'s output dict (same treatment as `target_x`/
  `target_x_vel`); `build_mujoco_state()` gained a `target_x_accel` kwarg.
  `tools/ur5e_mujoco_torque_experiments.py`'s per-step loop computes `target_x_accel_now` via
  `x_profile_accel()` alongside the existing `x_profile_target()` call and passes it into all
  three `build_mujoco_state()` call sites (pre-state, the safety-violation trace branch, and the
  normal post-state), and into both trace-row dicts for visibility.
- `hardware/direct_torque_link.py`: `UR5eDirectTorqueLink.compose_robot_state()`/
  `build_robot_state()` gained an optional `target_x_accel: float | None = None` kwarg (`None`
  by default -> `"target_x_accel": None` in the returned dict, same contract as the existing
  `dt_s` kwarg). `hardware/direct_torque_transport.py`'s per-cycle loop computes
  `target_x_accel` via `x_profile_accel()` alongside its existing `x_profile_target()` call and
  passes it into `link.compose_robot_state(...)`.
- `controller_core/state_types.py`: `RobotState`'s `target_x_accel` key added to the TypedDict
  and to **both** `as_robot_state()` and `as_impedance_robot_state()`'s optional-key whitelists.
  This is the exact bug class already found and fixed once tonight for `dt_s` -- a field present
  in the raw dict but silently dropped because the state-normalization function's whitelist
  didn't mention it, so no controller code downstream could ever see it even though the adapter
  had already put it there. Verified, not just inspected: `tests/unit/test_core.py`'s
  `test_as_robot_state_passes_through_target_x_accel`, `tests/unit/test_impedance.py`'s
  `test_as_impedance_robot_state_passes_through_target_x_accel` (the layer-2 check --
  `as_impedance_robot_state` is the function `compute()` calls internally), and
  `tests/mujoco/test_ur5e_mujoco_torque.py`'s
  `test_build_mujoco_state_target_x_accel_survives_to_impedance_normalization` (full sim-side
  path: `build_mujoco_state()` -> `MujocoUR5eState.as_robot_state()` -> `as_impedance_robot_state()`,
  asserting the same nonzero value survives every hop). All three pass.

### Step 2 -- the feedforward term

New `CartesianImpedanceConfig.acceleration_feedforward: bool = False` (default off, matching
every other P3-era flag's convention). When on, adds a mass-weighted term to the task wrench
before the `J_task.T @ wrench` mapping:

```
wrench_task[axis] += Lambda_diag[axis] * target_<axis>_accel
```

where `Lambda_diag` is the diagonal of the **same** task-space inertia matrix
`Lambda = (J_task M^-1 J_task^T + eps*I)^-1` this file already builds for
`task_space_inertia_shaping`/`nullspace_posture` -- reused, not recomputed.
`acceleration_feedforward` alone is now also a trigger for that block to run (previously only
`task_space_inertia_shaping or nullspace_posture`), so the flag works correctly even in a config
that has neither of those two on. Only X has a real reference today (`target_x_accel`); Y/Z
feedforward is wired generically (`target_y_accel`/`target_z_accel` looked up via a plain
`st.get(..., 0.0)`, defaulting to zero) so adding those trajectory generators later is a one-line
change here, not a redesign -- no new plumbing for Y/Z was added anywhere else, per the
instruction not to over-build.

**Graceful no-op, not a silent wrong answer**, when the state doesn't actually carry a real mass
matrix (`mass_matrix_provided` False): the existing shaping/nullspace code already falls back to
an identity `M` in that case (documented, historical behavior for those two flags), but doing the
same for the feedforward term would silently feed a fake ~1 kg effective mass into it -- exactly
the "silently produce wrong torques" failure mode this flag must not have. Checked in practice for
`direct_torque`'s three `dynamics_source` values (`rtde`/default, `local`, `local_pinocchio`): all
three already fetch and supply a real mass matrix for `task_space_inertia_shaping`'s sake, so this
no-op path is a defensive guard, not an expected-to-trigger one on that real-hardware lane.
`acceleration_feedforward_active` in `CartesianImpedanceOutput` reports whether the term was
actually applied a given cycle, and `wrench_accel_ff` reports the raw contribution, both plumbed
into `trace.jsonl` for `controller-rollout` runs.

**Regression contract**: with `acceleration_feedforward=False` (default), the new "also compute
Lambda" trigger condition is unchanged in truth value and the feedforward block itself is a no-op
(`if use_accel_ff and ...`), so output is provably byte-identical to before this change on any
input, including one with a nonzero `target_x_accel` in the state -- this is a *proof*, not a
coincidence: `tests/unit/test_acceleration_feedforward.py::
test_flag_off_is_byte_identical_regardless_of_target_x_accel` feeds a nonzero
`target_x_accel=3.7` with the flag off and asserts `tau` matches a state with no `target_x_accel`
key at all, `atol=1e-14`.

New named config: `config/ur5e_mujoco_torque_osc_tuned_split_base_wrist_accel_ff.yaml`, built from
tonight's best real-hardware-validated config
(`config/ur5e_mujoco_torque_osc_tuned_split_base_wrist.yaml`) plus
`acceleration_feedforward: true`. No existing config file was modified.

### Unit tests (`tests/unit/test_acceleration_feedforward.py`, pure numpy)

- Flag off by default (`CartesianImpedanceConfig().acceleration_feedforward is False`).
- Flag-off byte-identical regression, both against itself with/without `target_x_accel` in the
  state, and against a reference controller that never mentions the flag at all.
- Directional correctness at an identity Jacobian/mass matrix: `tau[0]` is positive for
  `target_x_accel=+2.0`, negative and exactly antisymmetric for `-2.0`, and scales linearly with
  the commanded magnitude (`wrench_accel_ff[0]` at `target_x_accel=4.0` is exactly `2x` the value
  at `2.0`).
- Graceful no-op when `mass_matrix` is absent from the state even with the flag on and a nonzero
  `target_x_accel`: `tau` unchanged, `acceleration_feedforward_active` reports `False`.
- YAML parsing round-trip (`from_controller_yaml_section`).

`tests/mujoco/test_accel_duration_profile.py` gained matching coverage for `x_profile_accel`
itself: zero for `step`/`ramp`; closed-form vs. central-finite-difference-of-velocity agreement
for `min_jerk`/`min_jerk_move_hold`/`accel_duration_triangular`/`accel_duration_scurve`;
triangular's `+accel`/`-accel` bang-bang sign and hold-phase zero; scurve's exact closed form
(`accel*omega*sin(omega*t)`) and its C0-continuity at the move/hold boundary (unlike triangular);
sign flip under negative `target_accel_mps2`; and the same `move_duration_s`/`target_accel_mps2`
required-argument error contract as `x_profile_target`.

**All new tests pass. Full suite (`pytest -q -m "unit or mujoco or hardware"`): 556 passed, 3
xfailed (pre-existing), zero regressions** -- including after fixing eight pre-existing
`tests/hardware/test_direct_torque_*.py`/`test_deadline_and_staleness.py`/
`test_local_pinocchio_dynamics.py` mock `UR5eDirectTorqueLink.compose_robot_state()` shims, which
had a fixed keyword-only signature and broke (`TypeError: got an unexpected keyword argument
'target_x_accel'`) the moment `direct_torque_transport.py`'s real call site started passing the
new kwarg. Each mock now accepts and forwards `target_x_accel=None`, matching the real class.

## Sim comparison

Representative move at the real `height_alpha=0.5`, zero-degree transport pose
(`hardware/poses.py::HEIGHT_ALPHA_0_5_Q`, `q = [0, -0.835398, -1.2, -0.985398, 0, 0]`), comparing
`config/ur5e_mujoco_torque_osc_tuned_split_base_wrist.yaml` against the new `..._accel_ff.yaml`
variant, `controller-rollout` mode, `--trajectory-profile accel_duration_scurve`, seed 0.

### At tonight's real-hardware-tested accel magnitude (`target_accel=0.02`, `move_duration=4.0s`)

Both runs pass cleanly (`valid_move_and_hold: true`, zero guard trips, zero torque clipping).
Comparing per-cycle traces during the move phase:

| metric | baseline | accel_ff | delta |
|---|---|---|---|
| move-phase RMS \|x_error\| | 0.002618 m | 0.002667 m | +1.9% (worse) |
| move-phase mean \|x_error\| | 0.002292 m | 0.002324 m | +1.4% (worse) |
| move-phase `tau_controller[0]` delta std (jitter proxy) | 0.006655 | 0.006619 | -0.5% (~unchanged) |
| peak \|wrench_accel_ff[0]\| | 0 (inactive) | 0.078 N | -- |

**Honest verdict: no measurable benefit at this accel magnitude, and a very slightly worse
tracking error** -- both differences are at or below what looks like noise-floor level (well under
1 mm), and the jitter proxy is essentially unchanged. This is explained by the physics, not a bug:
`effective_mass_x` (`Lambda_diag[0]` at this pose) is a few kg-equivalent, so at
`target_accel=0.02 m/s^2` the feedforward force peaks at ~0.08 N -- negligible next to `kp_x=400`,
which alone produces ~0.4 N from a 1 mm position error. At the actual accel magnitude used in
tonight's real-hardware `accel_duration_scurve` tests, the feedforward term is real (confirmed
active every cycle, `wrench_accel_ff` nonzero) but too small to matter either way.

### At a larger accel (`target_accel=0.3`, `move_duration=2.0s` -- the magnitude AGENTS.md's
own prior scurve-vs-triangular robustness note used)

| metric | baseline | accel_ff | delta |
|---|---|---|---|
| move-phase mean \|x_error\| | 0.01727 m | 0.01689 m | -2.2% (better) |
| move-phase peak \|x_error\| | 0.03570 m | 0.03648 m | +2.2% (worse) |
| move-phase RMS \|x_error\| | 0.02147 m | 0.02162 m | +0.7% (worse) |
| move-phase peak \|qd\| | 0.287 rad/s | 0.274 rad/s | -4.5% (better) |
| move-phase `tau_controller[0]` delta std (jitter proxy) | 0.0104 | 0.0285 | **+174% (worse)** |
| hold-phase max \|`tau_controller`\| | 3.75 Nm | 0.55 Nm | **-85% (much better)** |
| `move_tracking_score` | 0.9589 | 0.9657 | +0.7% (better) |

**Honest verdict: a real, measurable, but genuinely mixed effect, not a clean win.** The clearest
benefit is at the move->hold transition: peak controller torque needed during the hold phase drops
~7x (3.75 -> 0.55 Nm), meaning the controller has much less residual correction to do once the
move ends -- the feedforward is doing real anticipatory work, exactly as the physics predicts. Mean
tracking error during the move also improves slightly. But the per-cycle `tau_controller[0]`
command jitter nearly triples during the move, and peak instantaneous tracking error is very
slightly worse, not better. This is the specific risk flagged before running this comparison
("feedforward amplifying noise... if that reference itself isn't smooth") -- `x_profile_accel`'s
scurve closed form is analytically continuous, so this isn't a discontinuity artifact; it is more
likely the feedforward force's own continuous time-variation, combined with this pose's
already-known Jacobian/Lambda sensitivity near the wrist_2=0 singularity (see AGENTS.md sec 3's
extensive history of Lambda-related leaks at this exact pose), showing up as extra command-to-command
variation that a pure PD term (which only reacts to slowly-varying position/velocity error) doesn't
produce.

### Small canonical-grid-style smoke (regression check, not the full 4-category rigor sweep)

`tools/ur5e_move_hold_transport.py`, `min_jerk_move_hold` profile (dx-driven, not accel-driven),
`--target-x-deltas 0.02 0.04 --move-durations 1.0 --hold-durations 1.0 2.0`, same pose, seed 0:

- Baseline: **4/4** valid move-and-hold.
- accel_ff: **3/4** -- `dx=0.02, hold=2.0` regressed from pass to a `hold_phase_target_tracking`
  tolerance miss (`hold_phase_max_abs_x_error_m`: 0.00285 m baseline -> 0.00364 m accel_ff, both
  genuinely tiny in absolute terms, but this pose's tolerance for that case is tight enough that
  the ~28% relative increase crosses it).

This is a real, small, honestly-reported regression, not a safety-guard trip (no drift/orientation/
velocity guard fired; it is a tracking-tolerance miss). Investigated the mechanism directly from
`trace.jsonl` before writing this up: `min_jerk`'s own second derivative is analytically zero at
both endpoints (`d^2s/da^2 = 60a - 180a^2 + 120a^3`, which is exactly `0` at `a=0` and `a=1`), and
the trace confirms `wrench_accel_ff[0]` decays smoothly through the move->hold boundary with no
step -- **this specific case is not a reference-discontinuity artifact**, ruling out the most
obvious hypothesis. The more mundane explanation, consistent with the accel=0.3 scurve comparison
above, is that the feedforward term measurably perturbs the closed-loop trajectory shape during the
move (even smoothly), and at this pose's already-tight tolerances that is enough to flip one
marginal case.

## Overall verdict

`acceleration_feedforward` is implemented correctly, flag-gated, byte-identical when off, and its
state plumbing is verified end-to-end (not just inspected) to survive the same silent-drop bug
class already found once tonight for `dt_s`. But **the sim evidence does not support turning it on
by default, or claiming it "measurably reduces tracking lag/jitter" as the physics motivated it
to**: at the actual accel magnitude used in tonight's real-hardware tests it is too small to matter
either way; at a larger accel it trades a real hold-phase settling benefit for a real ~3x increase
in move-phase torque-command jitter and a small canonical-grid regression. Per the constraints for
this pass, no gain retuning was attempted to try to close this gap (that would be combining a
controller change with a retune in the same pass). `config/
ur5e_mujoco_torque_osc_tuned_split_base_wrist_accel_ff.yaml` is saved as a validated, sim-only,
opt-in experiment config for follow-up investigation -- **not** a recommended default, and **not**
real-hardware tested (no real hardware access in this pass; this needs its own careful, small-first
real-lab check like every other change in this repo, per AGENTS.md sec 4/7).
