# Intermediate-pose rigor sweep: height_alpha in {0.2, 0.3} with wrist_orientation_task

Context: `controller.wrist_orientation_task` (commit `bd5bba3`, config
`config/ur5e_mujoco_torque_osc_tuned_wrist_orient.yaml`,
`docs/status/wrist_orientation_task_2026-07-29.md`) was validated with the full
four-category rigor (canonical grid, long holds, large displacements,
torque-scale robustness) at exactly two poses: the canonical `alpha=0` pose and
`height_alpha=0.5` (the pose it targets). Poses in between never got that
rigor -- `docs/status/alpha_0.1_singularity_investigation_2026-07-28.md`
covered `alpha=0.1` with smaller ad-hoc checks, and `alpha=0.2`/`0.3` had
nothing comparable. This note runs the same four categories, unmodified, at
`height_alpha in {0.2, 0.3}` using the wrist-orient config, to check whether
today's fix helps, hurts, or is neutral away from the pose it was designed
for.

## Verdict

**Clean pass, zero regressions, no evidence of directional/asymmetric
degradation at either intermediate pose.** All 76 runs (2 alphas x 38 runs/
alpha) are valid: every category hits the same headline pass count the
canonical `alpha=0` pose and `height_alpha=0.5` already established (8/8,
8/8, 8/8, 14/14), at both `alpha=0.2` and `alpha=0.3`. Worst-case orientation
error across all 76 runs is 0.054 rad (hold-phase worst is far tighter,
0.0019 rad) against the 0.25 rad guard -- a >4.6x margin, comparable to the
0.073 rad worst case reported for `height_alpha=0.5` itself. No guard trips,
no near-misses, no directional asymmetry of the kind found at
`height_alpha=0.5` (AGENTS.md sec 3's "-0.20m fails, +0.20m passes" finding)
appears anywhere in this grid: every `target_x_delta` sign, direction, and
magnitude combination in all four categories passed at both alphas.

**Practical read: `wrist_orientation_task` is safe to treat as generalizing
across the alpha=0-0.5 range**, at least for the parameter grids these four
categories cover. This does not by itself resolve the *unrelated*
`alpha=0.1` real-hardware velocity-overshoot finding from
`docs/status/alpha_0.1_singularity_investigation_2026-07-28.md` -- that
investigation used the plain tuned config (not wrist-orient) and a different
failure signature (velocity overshoot, not orientation-guard trips); this
sweep does not re-run that specific check with wrist-orient and should not be
read as having done so.

## Method

New wrapper `tools/ur5e_pose_sweep_transport.py` computes each pose via
`hardware.poses.q_for_height_alpha(alpha)` and subprocesses
`tools/ur5e_move_hold_transport.py` once per (alpha, category) pair with
`--start-q-rad` set to that pose. It also auto-extracts the named config's own
`controller.gains` and forwards them via `--gain-overrides-json` -- required
because `ur5e_move_hold_transport.py` otherwise silently substitutes its own
`BASELINE_GAINS` (kp_x=80, ...) instead of the tuned config's real gains
(kp_x=400, ...); see `docs/status/bug_audit_2026-07-29.md` and
`docs/status/wrist_orientation_task_2026-07-29.md` sec 4, which hit the same
footgun. Verified by inspecting every subprocess command line printed to
stderr during the run (see below) -- `kp_x=400.0` etc. are present in every
invocation.

Category grids (reconstructed from AGENTS.md sec 3 / the wrist-orient doc's
own parameter *shapes*, since none of the three non-canonical categories'
exact literal commands are preserved verbatim anywhere in this repo, confirmed
before building this tool):

| Category | target_x_deltas | move_durations | hold_durations | torque_limit_scales | rows |
|---|---|---|---|---|---|
| canonical_grid | 0.01, 0.02, 0.03, 0.04 | 1.0 | 1.0, 2.0 | 1.0 | 8 |
| long_holds | 0.03, 0.06 | 1.0 | 4.0, 10.0, 20.0, 30.0 | 1.0 | 8 |
| large_displacements | 0.05, 0.10, 0.15, 0.20 | 1.0 | 1.0, 2.0 | 1.0 | 8 |
| torque_scale_robustness | 0.03, 0.06 | 1.0 | 2.0 | 0.10, 0.25, 0.40, 0.55, 0.70, 0.85, 1.00 | 14 |

Command actually run (one example; alpha and category varied per invocation):

```
python tools/ur5e_pose_sweep_transport.py \
  --height-alphas 0.2 \
  --config config/ur5e_mujoco_torque_osc_tuned_wrist_orient.yaml \
  --categories canonical_grid \
  --seed 0 \
  --output-root outputs/ur5e_mujoco_torque_transport/pose_sweep_alpha_0.2_0.3_wristorient
```
repeated for `--categories long_holds`, `large_displacements`,
`torque_scale_robustness`, and `--height-alphas 0.3`, all sharing the same
`--output-root` (results land in `alpha_<token>/<category>/`).

## Results

### Pass/fail table (alpha x category, N/M)

| height_alpha | canonical_grid | long_holds | large_displacements | torque_scale_robustness |
|---|---|---|---|---|
| 0.2 | 8/8 | 8/8 | 8/8 | 14/14 |
| 0.3 | 8/8 | 8/8 | 8/8 | 14/14 |

Every single row is a clean pass. Total: 76/76 runs valid across both poses.

### Worst-case metrics per (alpha, category), for honesty beyond pass/fail

All termination reasons were `duration_complete` (no guard ever fired) in
every one of the 76 runs.

| height_alpha | category | worst hold-phase orientation err (rad) | worst move-phase orientation err (rad) | worst \|qd\| (rad/s) | worst hold-phase Y drift (m) | worst hold-phase Z drift (m) |
|---|---|---|---|---|---|---|
| 0.2 | canonical_grid | 0.0002 | 0.0104 | 0.185 | 0.0000 | 0.0001 |
| 0.2 | long_holds | 0.0006 | 0.0157 | 0.250 | 0.0000 | 0.0002 |
| 0.2 | large_displacements | 0.0019 | 0.0526 | 0.827 | 0.0004 | 0.0030 |
| 0.2 | torque_scale_robustness | 0.0003 | 0.0209 | 0.640 | 0.0000 | 0.0002 |
| 0.3 | canonical_grid | 0.0003 | 0.0108 | 0.199 | 0.0000 | 0.0003 |
| 0.3 | long_holds | 0.0006 | 0.0163 | 0.272 | 0.0000 | 0.0003 |
| 0.3 | large_displacements | 0.0017 | 0.0538 | 0.892 | 0.0004 | 0.0037 |
| 0.3 | torque_scale_robustness | 0.0019 | 0.0317 | 0.870 | 0.0002 | 0.0045 |

Guards for reference (`config/ur5e_mujoco_torque_osc_tuned_wrist_orient.yaml`):
`max_orientation_error_rad: 0.25`, `max_abs_y_drift_m: 0.03`,
`max_abs_z_drift_m: 0.03`, `max_joint_velocity_radps: 3.0`. Every observed
worst case sits well inside these (orientation worst 0.054 rad = 4.6x margin;
drift worst 0.0045 m = 6.7x margin; qd worst 0.89 rad/s = 3.4x margin). The
largest values cluster in `large_displacements`/`torque_scale_robustness` at
the largest `target_x_delta` (0.20 m and 0.06 m respectively), as expected --
not a pose-specific anomaly, the same pattern the canonical pose shows.

### Comparison against the already-validated poses

| Pose | canonical_grid | long_holds | large_displacements | torque_scale_robustness |
|---|---|---|---|---|
| alpha=0 (config header, wrist-orient) | 8/8 | 8/8 | 16/16* | 14/14 |
| height_alpha=0.5 (this fix's target case) | 0.0587-0.0728 rad worst ori, both directions valid | up to 31s, valid | n/a (fix validated via +-0.20m directly) | n/a |
| height_alpha=0.2 (this sweep) | 8/8 | 8/8 | 8/8 | 14/14 |
| height_alpha=0.3 (this sweep) | 8/8 | 8/8 | 8/8 | 14/14 |

\* the documented "16/16" combines canonical_grid's 8 rows with 8 more
large-displacement rows; this sweep's `large_displacements` category alone is
the 8-row extension, matching that convention.

No degradation trend visible between alpha=0, 0.2, 0.3, and 0.5 -- if
anything the intermediate poses show slightly *lower* worst-case orientation
error than the alpha=0.5 case, though the parameter grids aren't identical
(alpha=0.5's headline number is specifically the +-0.20m directional test, not
this sweep's full 8-row large-displacement grid) so this is not a strict
apples-to-apples ranking claim.

## Files changed

- `tools/ur5e_pose_sweep_transport.py` (new, commit `a7a57de`) -- the sweep
  wrapper described above.
- `tests/unit/test_ur5e_pose_sweep_transport.py` (new, commit `a7a57de`) -- 11
  unit tests covering pose computation (matches
  `hardware.poses.q_for_height_alpha` directly, matches
  `HEIGHT_ALPHA_0_5_Q`, rejects out-of-range alpha), category-grid row-count
  shapes, gain extraction from a config YAML (only forwards `GAIN_FIELDS`,
  confirms `kp_rot_wrist`/`kd_rot_wrist` are correctly excluded), and
  subprocess command construction (start-q-rad forwarding, gain-overrides-json
  forwarding, `--no-plot` toggling, category grid forwarding) -- no
  subprocess/simulation calls inside the tests themselves.
- `docs/status/intermediate_pose_sweep_alpha_0.2_0.3_2026-07-29.md` (this
  file).
- No existing file was modified. `hardware/poses.py`,
  `tools/ur5e_move_hold_transport.py`,
  `config/ur5e_mujoco_torque_osc_tuned_wrist_orient.yaml`, and
  `controller_core/x_axis_cartesian_impedance.py` were only read, never
  edited.

## Tests run

- `pytest tests/unit/test_ur5e_pose_sweep_transport.py -q` -- 11 passed.
- `pytest -q -m unit` -- 105 passed (no regressions in the pure-numpy unit
  suite from adding the new files).
- Real simulation sweep itself (`tools/ur5e_pose_sweep_transport.py`, 76
  total child runs across 2 alphas x 4 categories) -- all 76 valid, 0
  failures, 0 guard trips; run via the conda `mujoco_ur5e` environment
  python, not the bare system python (the system python lacks `pinocchio`,
  which this config requires via `gravity_source: pinocchio`,
  `coriolis_feedforward: true`).

## Tests not run

- Full repo `pytest -q` (all markers) was not re-run in this task; only the
  `unit` marker subset relevant to the new files. No hardware, mujoco-marked,
  or urscript-parity tests were touched or exercised, consistent with this
  task's explicit scope (new files under `tools/`/`docs/status/` plus one new
  test file; hardware-lane files were off-limits as other agents were working
  on them concurrently).
- No hardware-in-the-loop or real-RTDE tests -- simulation only, per this
  project's standing hardware-safety rules.
- `alpha=0.1`'s real-hardware velocity-overshoot investigation was not
  re-run with the wrist-orient config -- out of scope for this task (see
  Verdict section caveat above).

## Rollback

```
git rm tools/ur5e_pose_sweep_transport.py tests/unit/test_ur5e_pose_sweep_transport.py docs/status/intermediate_pose_sweep_alpha_0.2_0.3_2026-07-29.md
```
or, to remove without deleting:
```
git checkout <sha-before-a7a57de> -- tools/ tests/unit/ docs/status/
```
No existing file was modified by this task, so rollback is purely additive
removal -- nothing else needs to change. Sweep output artifacts live under
`outputs/ur5e_mujoco_torque_transport/pose_sweep_alpha_0.2_0.3_wristorient/`
(gitignored, not git-recoverable, safe to delete independently of any git
rollback).
