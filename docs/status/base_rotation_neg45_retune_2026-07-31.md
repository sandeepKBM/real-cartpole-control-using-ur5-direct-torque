# -45° base-rotation retune/validation for UR5e OSC transport (sim-only)

**Status:** Sim reproduces the real -45° Y-drift failure cleanly. Gain retuning across every
axis explored (x/y/z, rot/posture, kd_joint) does **not** fix it — a real negative result,
consistent with the two live real-hardware fix attempts that also had zero effect. No new
config promoted. This is flagged as a likely structural/kinematic effect of the rotated pose,
requiring a human decision on next steps (a controller-design question, not a gain retune),
per this task's hard constraint against touching `controller_core/` in this pass.
**Date:** 2026-07-31. Simulation-only throughout; the real UR5e (172.16.71.77) was never
contacted.

## 0. Context

Real-hardware testing the same night added a -45° `shoulder_pan` base rotation
(`hardware.poses.HEIGHT_ALPHA_0_5_CLEARANCE_Q`) as the new default real-lab start pose for
wall/base clearance. At that pose, `config/ur5e_mujoco_torque_osc_tuned.yaml`'s unmodified,
already-validated gains reproducibly tripped `ImpedanceSafetyMonitor`'s Y-drift guard
(`|Y-Y0| > 0.03 m`) on a dx=0.20m move, 4+ separate real attempts, near-identical magnitude
each time (~0.030m), with near-45°-diagonal real TCP motion. Two live fixes
(kp_y/kd_y +50%, `lambda_diagonal_shaping: true`) had zero measurable effect. This doc's job:
confirm whether sim reproduces this at all, and if so, whether a gain search (the same kind
of search that produced the existing tuned config) can fix it.

## 1. What was added

`tools/tune_ur5e_residual_impedance_transport.py` gained a `--start-q-rad` flag (nargs=6),
mirroring the flag already on its sibling `tools/ur5e_move_hold_transport.py`. It threads
straight into the same `--start-q-rad` flag on the child `tools/ur5e_mujoco_torque_experiments.py`
process, exactly like the sibling tool already does. Purely additive: omitting the flag
reproduces prior behavior exactly (verified by inspection — the new code path is only
entered when the argument is not `None`). No `controller_core/` changes.

## 2. Environment note (found the hard way)

Every sim invocation in this investigation used
`/common/users/ss5772/miniforge3/envs/mujoco_ur5e/bin/python` explicitly. The base conda
env lacks `pinocchio`, which `config/ur5e_mujoco_torque_osc_tuned.yaml` requires
(`gravity_source: pinocchio`, `coriolis_feedforward: true`) — using the wrong interpreter
fails immediately with `ModuleNotFoundError: No module named 'pinocchio'`.

## 3. Mid-task hazard: concurrent edit to the shared model file

While this investigation's baseline sweeps were running, a **different, concurrent process**
(not this task — evidenced by unrelated untracked files appearing during this session:
`docs/status/direct_torque_controller_phase_profiling_2026-07-31.md`,
`docs/status/real_lab_session_2026-07-31.md`, `tools/box_sync_hardware_logs.sh`,
`tools/box_sync_pull_hardware_logs.sh`, and later modifications to three `tests/mujoco/*`
files) edited the tracked, shared `assets/ur5e_torque/ur5e_torque.xml` **in place**,
uncommitted, adding joint friction (`frictionloss`/`damping` on the `size3`/`size1` joint
default classes). The edit was non-atomic: one of this investigation's own sim subprocesses
caught the file mid-write (`xml.etree.ElementTree.ParseError: not well-formed`).

This file is explicitly called out in this repo's own working notes as "the centerpiece —
never delete or silently regenerate it," and it was clearly mid-edit by someone else's
in-progress, unrelated work. To avoid (a) corrupting their edit and (b) getting inconsistent
physics across this investigation's own before/after runs, this investigation did **not**
touch, stash, or revert the live file. Instead it froze an isolated, read-only copy of the
pre-edit model (`git show HEAD:assets/ur5e_torque/ur5e_torque.xml`, meshdir rewritten to an
absolute path to avoid relative-include depth issues) at
`outputs/_scratch_scene_orig_model/ur5e_torque_orig/{scene.xml,ur5e_torque.xml}` and re-ran
**every** sweep in this doc against that frozen scene via `--scene`. `outputs/` is gitignored,
so this scratch copy is not a git concern. All numbers below are from that frozen,
internally-consistent, pre-friction-edit model — i.e. they measure the same model family this
repo's entire existing OSC validation history (AGENTS.md §3) was built on, not whatever the
concurrent friction-modeling work eventually lands.

## 4. Baseline: does sim reproduce the real failure?

Same 4-category rigor sweep AGENTS.md §3 documents for the un-rotated pose (`canonical_grid`,
`long_holds`, `large_displacements`, `torque_scale_robustness` — identical grids to
`tools/ur5e_pose_sweep_transport.py`'s `CATEGORY_GRIDS`, 38 runs each), run via
`tools/ur5e_move_hold_transport.py --config config/ur5e_mujoco_torque_osc_tuned.yaml
--start-q-rad <pose> --seed 0 --no-plot` (config's own gains used as-is), at both:

- **un-rotated**: `hardware.poses.HEIGHT_ALPHA_0_5_Q` = `[0, -0.8354, -1.2, -0.9854, 0, 0]`
- **-45°**: `hardware.poses.HEIGHT_ALPHA_0_5_CLEARANCE_Q` = `[-0.7854, -0.8354, -1.2, -0.9854, 0, 0]`

### Pass/fail counts (num_valid_move_and_hold / num_runs)

| category | un-rotated | -45° |
|---|---|---|
| canonical_grid (dx 0.01-0.04m, hold 1-2s) | 8/8 | 8/8 |
| long_holds (dx 0.03/0.06m, hold 4-30s) | 8/8 | **4/8** |
| large_displacements (dx 0.05-0.20m, hold 1-2s) | 8/8 | **0/8** |
| torque_scale_robustness (dx 0.03/0.06m, scale 0.10-1.00) | 12/14 | **6/14** |
| **Total** | **36/38 (94.7%)** | **18/38 (47.4%)** |

### The mechanism, quantified

- Every -45° failure except `torque_scale=0.10` (both poses fail there identically via
  `|Z-Z0| > 0.03 m` — a known, unrelated torque-budget-saturation case, same as the
  historical un-rotated finding) trips the exact same guard the real hardware tripped:
  `|Y-Y0| > 0.03 m` (`ImpedanceSafetyMonitor`'s Y-drift guard).
- The trip magnitude is essentially constant, ~0.0300-0.0302 m, across dx=0.05m through
  dx=0.20m in `large_displacements` — i.e. it saturates right at the guard threshold
  regardless of how far the move target is, matching the real-hardware report of tripping
  "at nearly identical magnitude every time (~0.030m)" almost exactly.
- In `canonical_grid` (8/8 pass — the drift never *quite* crosses 0.03m there), move-phase Y
  drift grows ~linearly with dx: 0.0054m (dx=0.01) → 0.0124m (0.02) → 0.0197m (0.03) →
  0.0276m (0.04). Extrapolating that line crosses the 0.03m guard right around dx≈0.043-0.05m
  — exactly where `large_displacements` (dx≥0.05m) starts failing 100% of the time. A clean,
  reproducible, quantitative dose-response curve, not sim noise.
- `long_holds`: dx=0.03m passes at every hold duration (4-30s); dx=0.06m fails at every hold
  duration, always during the **move phase itself** (`hold_phase_max_abs_y_drift_m≈0` in every
  failing row) — this is a transient coupling during the X-move, not a drift-over-time hold
  issue.

**Verdict: sim clearly reproduces the real -45° Y-drift/diagonal-coupling failure** — same
guard, same trip magnitude, same "saturates immediately once past a dx threshold" signature.
One honest quantitative difference: sim's failure onset (dx≈0.05m at 1.0s move duration) is
smaller than the real-hardware report (dx=0.20m, longer move durations) — plausibly a
duration/dynamics difference not investigated further here, since it doesn't change the
qualitative verdict this doc's decision rule turns on (does sim reproduce this at all — yes).

## 5. Gain search: can retuning fix it?

### 5a. Staged search (`tools/tune_ur5e_residual_impedance_transport.py`, now flag-enabled)

Ran the tool at the -45° pose against the frozen scratch scene, with trimmed grids (existing
CLI flags only, no code/search-space changes) to keep total runs in the tens instead of
hundreds while still exercising the actual failure displacements in Stage C:

```
tools/tune_ur5e_residual_impedance_transport.py \
  --config config/ur5e_mujoco_torque_osc_tuned.yaml \
  --start-q-rad -0.7853981633974483 -0.8353981633974483 -1.2 -0.9853981633974482 0.0 0.0 \
  --seed 0 --hold-durations 2.0 --small-target-x-deltas 0.02 --small-durations 1.0 \
  --validation-target-x-deltas 0.03 0.06 0.10 --validation-durations 1.0 2.0 \
  --torque-limit-scales 1.0 --stage-a-top-k 3 --stage-b-top-k 2 --stage-c-top-k 2 --no-plot
```

- **Stage A** (12 candidates: baseline + `BASE_STAGE_A_VARIANTS` — 3 kp_x/kd_x pairs, 4
  kp_y/kd_y pairs up to kp_y=130/kd_y=25, 4 kp_z/kd_z pairs), evaluated at dx=0 hold-only —
  top-3 advanced by hold-quality score.
- **Stage B** (12 candidates: 3 parents × 4 `BASE_STAGE_B_VARIANTS` rot/posture pairs),
  evaluated at a small dx=0.02m move — top-2 advanced.
- **Stage C** (12 rows: 2 finalists × dx{0.03,0.06,0.10} × dur{1,2}s) — the only stage that
  actually exercises the failure displacement. **Both finalists still failed identically at
  dx=0.06 and dx=0.10**, tripping `|Y-Y0| > 0.03 m` at ~0.030m, same as baseline:

  | candidate | gains changed from baseline | dx=0.03 | dx=0.06 | dx=0.10 |
  |---|---|---|---|---|
  | `001_04_rot30_post20` | kp_rot 0→30, kd_rot 10→8, kp_posture 25→2, kd_posture 6→0.8 | pass (1s) / fail-orientation (2s) | **fail (Y-drift)** | **fail (Y-drift)** |
  | `002_01_rot10_post025` | kp_rot 0→10, kd_rot 10→3, kp_posture 25→0.25, kd_posture 6→0.2 | pass | **fail (Y-drift)** | **fail (Y-drift)** |

  Note both Stage C finalists happened to retain `kp_y=80, kd_y=15` (unchanged from
  baseline) — Stage A's y-gain variants (up to kp_y=130/kd_y=25) were evaluated only at
  dx=0 (hold-only, where y-gain barely matters) and were never selected forward to a Stage
  C validation against the real failure. This is an honest gap in the staged tool's funnel
  for *this specific* problem (it was designed for the original un-rotated-pose tuning
  work, where hold-quality and small-move metrics were the right early filters). Closed
  directly below.

### 5b. Supplementary direct smoke tests (outside the staged tool's built-in variant space)

`docs/hardware/AUTO_TUNING_PLAN.md`'s established restricted safe search space calls out
`kp_x`/`kd_x` and `kd_joint` specifically — `kd_joint` isn't covered by either
`BASE_STAGE_A_VARIANTS` or `BASE_STAGE_B_VARIANTS`. Also closed the Stage-A/C gap above by
testing the largest y-gain variant directly at the failure condition. All run via
`tools/ur5e_move_hold_transport.py --gain-overrides-json ... --target-x-deltas 0.06 0.10
--hold-durations 2.0` at the -45° pose, frozen scene:

| candidate | dx=0.06 | dx=0.10 |
|---|---|---|
| `kd_joint: 6.0` (baseline 4.0) | fail, Y-drift 0.03004m | fail, Y-drift 0.03017m |
| `kd_joint: 8.0` (2×) | fail, Y-drift 0.03016m | fail, Y-drift 0.03003m |
| `kd_joint: 8.0` + `kp_posture: 37.5, kd_posture: 9.0` (1.5×) | fail, Y-drift 0.03011m | fail, Y-drift 0.03027m |
| `kp_y: 130.0, kd_y: 25.0` (matches the real live +50% attempt) | fail, Y-drift 0.03008m | fail, Y-drift 0.03013m |

Every single candidate across both the staged search and the supplementary smoke tests trips
the identical guard at a magnitude statistically indistinguishable from the unmodified
baseline (0.0300-0.0303m across all of them). **Zero measurable effect from any gain change
tried, across x/y/z/rot/posture/kd_joint.**

## 6. Verdict and recommendation

**No new config promoted.** Nothing found in this search — spanning the staged tool's full
built-in variant space (x/y/z, rot/posture) plus direct kd_joint and y-gain tests matching
`AUTO_TUNING_PLAN.md`'s restricted safe space and the real-hardware live-tuning attempts —
moves this failure at all. This is a genuine, reproducible negative result in sim, and it
**matches** the real hardware's own two independent live fix attempts (kp_y/kd_y +50%,
`lambda_diagonal_shaping`), both of which also had zero effect. Three independent lines of
evidence (real hardware ×2, sim ×5 gain variants across every schedulable axis this repo's
own tuning methodology considers) now agree: **this is not a gain-magnitude problem.**

The trip-at-a-near-constant-~0.030m-regardless-of-target-displacement signature, combined
with the flat non-response to a 2× damping change and a 50%+ stiffness change on the exact
axis being violated, is consistent with a **structural kinematic/Jacobian effect of the -45°
rotated pose** — most plausibly a nullspace-posture-projector or wrench-shaping asymmetry
introduced by the base rotation, in the same family as (though not yet confirmed identical
to) the already-documented "ceiling is directional" finding in AGENTS.md §3, where a
`kp_posture`/`kd_posture`/`kd_joint` sweep at a different rotated-relative-to-nominal case
"barely moved the outcome" for the same underlying reason (the nullspace-posture projector's
own authority, not the task-space gains, is what's asymmetric). Confirming that specific
mechanism here would need directly measuring `cond(J)`, `Λ`, and the nullspace-projector norm
along the -45° move — genuinely a controller-design investigation, not a gain retune, and is
explicitly out of scope for this task's hard constraint against `controller_core/` changes.

**Recommendation for a human decision:** do not attempt further gain retuning for the -45°
pose without first confirming the mechanism (the sweep above already spent the "cheap" search
budget this repo's own methodology allows). Options from here are a controller-design
question (e.g. extending the existing diagonal-Λ/adaptive-regularization work, or a
pose-dependent posture re-anchoring authority fix) — not something this sim-only,
gain-tuning-scoped task should decide or implement unilaterally.

## 7. Test suite / regression check

No `controller_core/` or shared-config files were modified by this task (only the additive
`--start-q-rad` CLI flag on `tools/tune_ur5e_residual_impedance_transport.py`, which is inert
unless explicitly passed). `config/ur5e_mujoco_torque_osc_tuned.yaml` is untouched. No new
named config was created (nothing validated beat baseline). The concurrent session's
in-progress changes (`assets/ur5e_torque/ur5e_torque.xml` friction addition and three
`tests/mujoco/*` files) are unrelated to this task and were left exactly as found — not
staged, not committed, not reverted.

## 8. Files / artifacts from this investigation

- `tools/tune_ur5e_residual_impedance_transport.py` — `--start-q-rad` flag added (committed).
- `outputs/ur5e_mujoco_torque_transport/base_rotation_neg45_baseline/`,
  `base_rotation_unrotated_baseline/`, `base_rotation_neg45_gain_search/`,
  `base_rotation_neg45_kdjoint_smoke/` — raw sweep outputs (gitignored, not git-recoverable;
  regenerate via the commands in §4/§5 against the frozen scene at
  `outputs/_scratch_scene_orig_model/ur5e_torque_orig/scene.xml` if needed, or against
  `assets/ur5e_torque/scene.xml` directly once the concurrent friction work lands and is
  itself validated).
