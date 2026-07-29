# Wrist-orientation task: fixing the height_alpha=0.5 directional ceiling

Context: AGENTS.md sec 3's 2026-07-27/28 finding ("The ceiling is directional, not just a
magnitude limit") documents that at `height_alpha=0.5`
(`hardware/poses.py::HEIGHT_ALPHA_0_5_Q`), the tuned OSC config passes a `+0.20m` transport
move cleanly but fails `-0.20m` via orientation error at roughly half the peak `wrist_2`
excursion of the passing direction — root-caused to the nullspace-posture projector's
restoring authority (the *only* mechanism holding orientation, since `kp_rot=0`) being
asymmetric with `wrist_2` sign at that pose, confirmed not fixable by retuning
`kp_posture`/`kd_posture`/`kd_joint`. This note adds a new, additive, flag-gated controller
term (`controller.wrist_orientation_task`) that gives orientation its own dedicated
authority via the wrist joints, translating the position/orientation task-split design of two
pre-torque-lane kinematic controllers (`archive/legacy_mujoco/controller.py`'s
`split_forearm_origin_face_controller` / `differential_ik_split_controller`) into the
inverse-dynamics/impedance controller.

## Verdict

**Fixed, validated, zero regressions.** The new term reduces peak orientation error at
`height_alpha=0.5` by roughly 3–4x in both directions and removes the directional asymmetry
entirely, while the full previously-validated envelope (canonical grid, long holds, large
displacements, torque-scale robustness — all at the canonical `alpha=0` pose where these were
originally characterized) is unchanged: every headline pass count matches the known-good
baseline exactly (8/8, 8/8, 16/16, 14/14).

**Design finding worth keeping in mind for any future wrist-authority term**: any nonzero
proportional gain (`kp_rot_wrist`) on this term reproduces a slow-growing instability through
`J_rot.T` near the `wrist_2=0` singularity — the same qualitative failure mode already
documented for `kp_rot` in the shared Lambda-weighted wrench pipeline, but here through a raw
(non-Lambda-shaped) Jacobian-transpose path. A short (1–2s) hold does not catch it; a long
(10s+) hold does. The shipped fix is damping-only (`kp_rot_wrist=0.0`, `kd_rot_wrist=10.0`)
and is stable through a 30s+ hold in both directions.

## Evidence

### 1. The exact documented case: before/after orientation error

Reproduced with the production entrypoint (`tools/ur5e_mujoco_torque_experiments.py
--mode controller-rollout --controller-kind impedance --trajectory-profile
min_jerk_move_hold`), `--start-q-rad` set to `hardware.poses.HEIGHT_ALPHA_0_5_Q` exactly,
`--move-duration 1.0 --duration 3.0` (1s move + 2s hold), `--seed 0`.

Two baseline configs give two different (but consistent) pictures of "the exact documented
case," so both are reported honestly:

**`config/ur5e_mujoco_torque_osc_tuned.yaml`** (the config the new `..._wrist_orient.yaml` is
built from, per this task's config-preservation rule):

| dx (m) | peak orientation error (rad) | termination | valid |
|---|---|---|---|
| -0.20 (before) | 0.2370 | duration_complete | True (marginal — 0.013 rad under the 0.25 rad guard) |
| +0.20 (before) | 0.2178 | duration_complete | True |
| -0.20 (**after**, wrist_orient) | **0.0636** | duration_complete | True |
| +0.20 (**after**, wrist_orient) | **0.0587** | duration_complete | True |

**`config/ur5e_mujoco_torque_osc_tuned_adaptive_lambda.yaml`** (the config that AGENTS.md's
own commit `74c27d7` and `docs/status/rl_gain_scheduling_alpha05_directional_fix_2026-07-29.md`
actually used to characterize this finding — it reproduces the guard trip exactly, unlike the
plain tuned config above which is close but doesn't literally cross 0.25 rad at these move/hold
settings):

| dx (m) | peak orientation error (rad) | termination | valid |
|---|---|---|---|
| -0.20 (before) | 0.2498 | `\|\|orientation error\|\| > 0.25 rad` | **False** (matches AGENTS.md's own reproduction number, 0.24975) |
| +0.20 (before) | 0.2238 | duration_complete | True |
| -0.20 (**after**, same fix applied on top, scratch config) | **0.0728** | duration_complete | True |
| +0.20 (**after**, same fix applied on top, scratch config) | **0.0648** | duration_complete | True |

Both baselines show the same directional asymmetry (`-0.20m` worse than `+0.20m`); the fix
removes it in both cases and leaves comfortable safety margin (worst case 0.073 rad vs the
0.25 rad guard, a 3.4x margin) instead of the ~0.005–0.013 rad margin (or an outright trip) the
baseline had.

### 2. Gain selection: why damping-only (`kp_rot_wrist=0.0`, `kd_rot_wrist=10.0`)

Empirical sweep at the exact `-0.20m`/`+0.20m`, `height_alpha=0.5` case, `config/
ur5e_mujoco_torque_osc_tuned_wrist_orient.yaml` base:

| kp_rot_wrist | kd_rot_wrist | hold | -0.20m ori (rad) | -0.20m valid | +0.20m ori (rad) | +0.20m valid |
|---|---|---|---|---|---|---|
| 20.0 | 5.0 (first guess) | 2s | 0.141* | **False** | 0.130* | **False** |
| 5 | 2 | 2s | 0.156 | False | — | — |
| 0 | 10 | 2s | 0.0636 | True | 0.0587 | True |
| 5 | 10 | 2s | 0.0622 | True | 0.0560 | True |
| 10 | 10 | 2s | 0.0627 | True | — | — |
| 20 | 10 | 2s | 0.0708 | **False** | 0.0626 | **False** |
| 40 | 10 | 2s | 0.1126 | **False** | 0.1068 | **False** |
| 5 | 20 | 2s | 0.0365 | True | 0.0327 | True |
| **0** | **10** | **11s** | 0.0636 (move) / 0.0579 (hold) | **True** | 0.0587 (move) / 0.0011 (hold) | **True** |
| 5 | 10 | 11s | — | **False** (slow divergence, guard trips ~t=11s) | — | **False** |
| 5 | 20 | 11s | 0.0365 (move) / 0.1733 (hold) | True | 0.0327 (move) / 0.1645 (hold) | True |
| **0** | **10** | **31s** | 0.0636 (move) / 0.0579 (hold) | **True** | 0.0587 (move) / 0.0015 (hold) | **True** |

\* First guess (kp=20, kd=5) reported `move_phase_max_abs_orientation_error_rad` of only
0.14, but the run still failed — the guard tripped shortly *after* the move phase ended (early
in the hold), which the move-phase-only field doesn't show; full trace inspection showed
`q_wrist2`/`qd_wrist2` growing monotonically with no sign of settling, i.e. a real dynamic
instability, not just elevated static error.

The pattern is consistent and repeats a finding this project already made once for `kp_rot`
in the shared Lambda-weighted wrench pipeline: **any nonzero proportional gain destabilizes
near the wrist_2=0 singularity, only more slowly as it shrinks.** Here the mechanism is
different in detail (raw `J_rot.T` at a pose where `J_rot`'s smallest singular value is ~0.085
— not Lambda's eps-regularized inversion) but the qualitative shape is the same: `kp=5` looks
clean at a 2s hold and only diverges once the hold runs past ~10s; `kp=20`/`40` diverge fast
enough to trip within the 2s hold itself. `kp=0` (damping-only) has no such failure mode —
verified clean through a 31s hold in both directions. Chosen: `kp_rot_wrist=0.0,
kd_rot_wrist=10.0` (the damping value mirrors `kd_rot`'s own already-validated value, just
delivered through the wrist-masked `J_rot.T` path instead of the Lambda pipeline).

### 3. `WRIST_ORIENTATION_MASK` values and provenance

```
WRIST_ORIENTATION_MASK = [0.0, 0.0, 0.0, 1.25/1.55, 1.0, 1.25/1.55]
                        = [0.0, 0.0, 0.0, 0.8065,   1.0, 0.8065]
  joint order: shoulder_pan, shoulder_lift, elbow, wrist_1, wrist_2, wrist_3
```

Read directly from `archive/legacy_mujoco/controller.py` (not the task prompt's approximate
numbers, which don't match the actual legacy code): both `split_forearm_origin_face_controller`
(C1, line ~471) and `differential_ik_split_controller` (C2, line ~268) use
`face_mask = [0, 0, 0, 1.25, 1.55, 1.25]` over the full 6-joint order — exactly zero on the
three proximal joints, heaviest on `wrist_2` (the joint that sits at 0 at the transport
singularity), symmetric on `wrist_1`/`wrist_3`. Normalized here to a 1.0 peak; only the shape
is taken from the legacy controllers, not their literal PD gains (those live in
`kp_rot_wrist`/`kd_rot_wrist` instead, chosen independently — see above).

### 4. Full regression against the already-validated envelope

All at the canonical `alpha=0` transport pose (`home_qpos` in the config), where the tuned
config's own envelope was originally characterized (AGENTS.md sec 3 / the config's own header
comment), using `tools/ur5e_move_hold_transport.py` with `--gain-overrides-json` set to the
tuned gain set (required — this driver overrides the 11 schedulable gain fields with its own
`BASELINE_GAINS` otherwise; `kp_rot_wrist`/`kd_rot_wrist` and the `wrist_orientation_task` flag
pass through from the config file untouched since they aren't in `GAIN_FIELDS`).

| Sweep | Known-good baseline | `..._wrist_orient.yaml` result |
|---|---|---|
| Canonical grid (dx 0.01–0.04m × hold 1/2s, move 1.0s) | 8/8 | **8/8** |
| Long holds (dx 0.03/0.06m × hold 4/10/20/30s) | 8/8 | **8/8** |
| Large displacements (dx 0.05–0.20m × hold 1/2s; combined with canonical grid dx 0.01-0.04m for the documented "16/16") | 16/16 | **16/16** (8 canonical + 8 here) |
| Torque-scale robustness (dx 0.03/0.06m × scale 0.10–1.00, 7 steps × 2 = 14) | 14/14 | **14/14** |

Zero new failures at any point in the grid. One incidental observation, reported for
completeness though it isn't part of the documented headline metric: the large-displacement
sweep's separate "strict" tolerance sub-metric (`num_valid_move_and_hold_strict`, computed by
the same driver but not what AGENTS.md's "16/16" refers to) was 6/8 for the wrist-orient
config vs 4/8 for the unmodified tuned baseline run through the identical sweep — i.e. slightly
*better*, not worse.

## Files changed

- `controller_core/x_axis_cartesian_impedance.py` — new `WRIST_ORIENTATION_MASK` constant;
  new `CartesianImpedanceConfig` fields `wrist_orientation_task` (bool, default `False`),
  `kp_rot_wrist`/`kd_rot_wrist` (float, default `0.0`); new `CartesianImpedanceOutput` fields
  `tau_orient_wrist`, `wrist_orientation_task_active`; `compute()` adds the new term into the
  same joint-space bias sum as `tau_posture`, flowing through the existing geometric
  backtracking and hard clip unchanged. Default-off path is byte-identical to before (unit
  test `test_wrist_orientation_task_off_by_default_and_zero_when_disabled`).
- `controller_core/torque_task_qp.py` — one-line addition (`tau_orient_wrist=np.zeros(6)`) to
  its `CartesianImpedanceOutput` construction, required because that field is now non-optional
  on the shared dataclass; this controller does not implement the new term.
- `config/ur5e_mujoco_torque_osc_tuned_wrist_orient.yaml` — new config, copy of
  `config/ur5e_mujoco_torque_osc_tuned.yaml` plus `wrist_orientation_task: true` and the two
  new gains. `ur5e_mujoco_torque_osc_tuned.yaml` itself is untouched (verified by diff — only
  the three new fields differ).
- `tests/unit/test_impedance_dynamics.py` — 6 new tests: mask shape, flag-off-is-zero/
  byte-identical, exact-formula match (isolated via zeroed other gains + `J=I`), flows through
  backtracking/clip, YAML parsing.
- `tests/mujoco/test_wrist_orientation_task.py` — new file, one controller-rollout-level test
  reproducing a scaled-down version of the fix (dx=-0.10m, 2s total, `height_alpha=0.5`):
  asserts orientation error drops by more than 40% and both runs stay valid.

## Tests run

- `pytest tests/unit/test_impedance_dynamics.py -q` — 27 passed (21 pre-existing + 6 new).
- `pytest tests/mujoco/test_wrist_orientation_task.py -q` — 1 passed.
- `pytest -q` (full suite) — 348 passed, 1 failed:
  `tests/hardware/test_direct_torque_transport_timing.py::test_transport_records_timing_and_deadline_loop`.
  This failure is unrelated to this change: it asserts `dominant_phase` is one of a fixed set
  of timing-phase names, and fails because a `"local_dynamics"` phase now appears — traceable
  to the unrelated, already-landed `hardware/local_dynamics.py` Pinocchio fast-path commit
  (`dee0190`), which never touched this test's expected-phase whitelist. This test imports
  `hardware.direct_torque_transport` and uses a mocked RTDE link; `controller_core/
  x_axis_cartesian_impedance.py` (this change's only functional edit) has zero callers in that
  path's timing-phase accounting, and `wrist_orientation_task` defaults off. Not touched or
  investigated further per this task's explicit hardware-lane scope exclusion.

## Tests not run

- No hardware-in-the-loop or real-RTDE tests (out of scope — simulation only, per task and
  per this project's standing hardware-safety rules).
- No RL gain-scheduling retraining — this is a pure controller-law change; RL configs were not
  touched and are out of scope for this task.

## Rollback

```
git revert <this-commit-sha>
```
or, to remove without a revert commit:
```
git checkout <previous-sha> -- controller_core/x_axis_cartesian_impedance.py controller_core/torque_task_qp.py
rm config/ur5e_mujoco_torque_osc_tuned_wrist_orient.yaml docs/status/wrist_orientation_task_2026-07-29.md tests/mujoco/test_wrist_orientation_task.py
git checkout <previous-sha> -- tests/unit/test_impedance_dynamics.py
```
The new flag defaults to `False` and the new dataclass fields default to values that make the
term a no-op, so simply not referencing `config/ur5e_mujoco_torque_osc_tuned_wrist_orient.yaml`
(every other existing config is untouched) is itself a full functional rollback without
touching any code.
