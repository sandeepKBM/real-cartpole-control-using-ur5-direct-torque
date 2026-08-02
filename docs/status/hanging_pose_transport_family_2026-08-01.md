# A "hanging"/elbow-down transport pose family that avoids the wrist_2=0 singularity by
# construction (2026-08-01)

## Why this exists

Tonight's session found the ENTIRE canonical transport pose family
(`hardware/poses.py::ACTIVE_ORIGIN_Q`/`LOWER_B_Q`/`q_for_height_alpha(alpha)`, every
`alpha` in `[0,1]`) sits at `wrist_2=0`, a genuine UR-family kinematic singularity, for its
whole range (measured `cond(full 6x6 J)` 1e16-2.5e17 throughout). `split_base_wrist_task`
(`docs/status/split_base_wrist_impedance_2026-08-01.md`) fixed the *consequence* of that in
the controller (real-hardware validated, 9/9) but the user judged it a diagnostic
workaround, not a fix acceptable for real lab deployment, given two still-open real issues
(orientation-error growth with exposure time, an unexplained `-X` transient -- both
documented in `docs/status/session_compilation_2026-08-01_night.md`). This doc covers the
more fundamental alternative: a genuinely different pose family that never sits at the
singularity in the first place.

**Bottom line up front**: a working, additive "hanging" pose family was designed,
kinematically verified, and given a first-pass gain configuration that reaches 36/38
(94.7%) on the standard 4-category rigor sweep -- comparable to or better than every
documented number for the old pose family at the equivalent friction-affected sim model.
`cond(J)` across the new family's whole range is 7.04-15.41, twelve to sixteen orders of
magnitude better than the old family's 1e16-2.5e17. **This is sim-only. No real-hardware or
physical-clearance check has been done for this genuinely different arm posture -- do not
test it live without one, see "Real-hardware readiness" below.**

## 1. Design

### 1.1 Search method

The old family's shape (`ACTIVE_ORIGIN_Q` to `LOWER_B_Q`) is a near-fully-extended arm
reaching up and out, held there against gravity, that happens to keep `wrist_2` exactly at
0 throughout. The new family instead uses an elbow-down "hanging" shape: shoulder_lift
steeply negative, elbow bent well away from both 0 (fully extended) and +-pi (fully
folded), `wrist_2` fixed at +pi/2 (never 0).

Method (`mujoco.MjModel.from_xml_path("assets/ur5e_torque/scene.xml")`, forward kinematics
+ `mujoco.mj_jacSite` at the real `attachment_site`, exactly how the controller's own
Jacobian is computed):

1. Grid search over `(shoulder_lift, elbow, wrist_1)` at `wrist_2 in {+pi/2, -pi/2}`,
   keeping only points with `cond(full 6x6 J) < 50` and `0.3 <= site_z <= 1.3` m
   (16,684 candidates found).
2. Bucketed by site Z height (0.05 m bins); confirmed cond(J) stays in the single-to-
   low-double digits from z~0.30 m all the way to z~1.10 m -- this is a wide, well-
   conditioned region, not a narrow lucky point.
3. Picked two endpoints (`wrist_2 = +pi/2` fixed for consistency) and refined each with
   `scipy.optimize.minimize` (Nelder-Mead) to match `ACTIVE_ORIGIN_Q`'s/`LOWER_B_Q`'s own
   site-frame Z heights (1.08 m / 0.537 m) as closely as practical while keeping cond(J)
   low.
4. Verified the *linear interpolation* between the two refined endpoints (not just the
   endpoints themselves) stays well-conditioned across its whole range -- the actual claim
   `q_for_hanging_height_alpha` needs to support.

### 1.2 The two endpoints (`hardware/poses.py`)

| | `HANGING_ORIGIN_Q` (alpha=0, "tall") | `HANGING_LOWER_Q` (alpha=1, "low") |
|---|---|---|
| q (rad) | `[0, -1.791994, 0.812668, -1.288057, +pi/2, 0]` | `[0, -1.491612, 1.990426, -2.630057, +pi/2, 0]` |
| site pos (x,y,z) m | (-0.138, -0.134, **1.044**) | (-0.409, -0.134, **0.537**) |
| cond(full 6x6 J) | 15.41 | 7.04 |
| gravity-comp torque, max\|tau\| (Nm) | 9.04 | 16.81 |

Compare to the old family's endpoints: `ACTIVE_ORIGIN_Q` site z=1.08 m, cond=2.49e17,
max\|tau\|=0.00 Nm (a genuine equilibrium at the singularity, not a real "lower effort"
reading); `LOWER_B_Q` site z=0.537 m, cond=2.13e17, max\|tau\|=26.48 Nm. The new family's Z
range (0.537-1.044 m) closely matches the old family's (0.537-1.08 m) -- the "tall" end is
3.6 cm short of an exact match; not adjusted further since cond(J) was prioritized once
close. Gravity torque is comparable or lower throughout (9-17 Nm hanging vs. 0-26 Nm old;
the old family's headline 0-Nm reading is a singularity artifact, not evidence it is
lower-effort overall) -- consistent with, though not strong independent confirmation of,
the "hanging is lower-effort" framing that motivated this design.

### 1.3 `cond(J)` sweep across the whole range -- the key claim

21-point linear interpolation `q_for_hanging_height_alpha(alpha)`, `alpha` in
`[0, 0.05, ..., 1.0]`, static forward-kinematics `cond(full 6x6 J)`:

| alpha | 0.00 | 0.10 | 0.20 | 0.30 | 0.40 | 0.50 | 0.60 | 0.70 | 0.80 | 0.90 | 1.00 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| cond(J) | 15.41 | 13.38 | 11.91 | 10.77 | 9.85 | 9.10 | 8.49 | 7.98 | 7.57 | 7.26 | 7.04 |

Full 21-point range: **min 7.04, max 15.41** -- monotonically decreasing from the "tall" to
the "low" end, no spike anywhere in between. Compare to the old family's own 11-point sweep
this session ran on the identical model: `cond(J)` ranges 2.13e17 to 2.59e17 with no clean
trend (singular throughout). This directly confirms the design claim: the new family avoids
the wrist singularity across its *entire* range, not just at hand-picked points.

`tests/mujoco/test_hanging_pose_family.py::test_hanging_family_cond_j_stays_well_conditioned_across_full_range`
locks this down as a regression test (bound 100, ~6.5x the measured max) directly against
the real MJCF, mirroring how `tests/unit/test_split_base_wrist_task.py` locks down that
fix's cond(J) claim.

**Dynamic confirmation (not just static FK)**: `jacobian_cond` from the per-cycle trace of
an actual closed-loop dx=0.20 m move-hold run (the largest displacement tested, 1000 steps,
friction-feedforward config) stayed in **8.91-11.17** throughout the real 500 Hz rollout --
confirms the static sweep's claim holds under real controller-driven motion, not just at
rest.

### 1.4 Reachability verification (task-space coverage)

Task requirement: world-X translation, Z-hold, tool-orientation-hold, comparable coverage
to the old family. Checked via a differential-IK probe (task-space PD toward a shifted
target position + a small-angle orientation-hold term, `pinv(J)`, 400 iterations, clipped
to real joint ranges) from three representative poses -- not a substitute for the real
closed-loop controller test in section 2, but independent, purely kinematic evidence:

| start pose | target dx | achieved dx | final cond(J) | at joint limit? |
|---|---|---|---|---|
| HANGING_ORIGIN_Q | +0.20 m | 0.181 m | 2266 | no |
| HANGING_ORIGIN_Q | -0.20 m | -0.226 m | 36.9 | no |
| HANGING_LOWER_Q | +0.20 m | 0.211 m | 11.7 | no |
| HANGING_LOWER_Q | -0.20 m | -0.199 m | 8.9 | no |
| HANGING_ALPHA_0_5_Q (midpoint) | +0.25 m | 0.272 m | 17.0 | no |
| HANGING_ALPHA_0_5_Q (midpoint) | -0.25 m | -0.275 m | 23.0 | no |

All six directions reach within ~10% of a 0.20-0.25 m target with no joint-limit
contact and no `cond(J)` blowup (worst case 2266 at the "tall" end's `+X` probe, still
13+ orders of magnitude better than the old family's baseline). This matches or exceeds the
old family's own documented `large_displacements` envelope (dx up to 0.20 m).
`tests/mujoco/test_hanging_pose_family.py::test_hanging_family_endpoints_reach_expected_site_height`
locks down the Z-height claim as a regression test.

## 2. Implementation (additive only)

- `hardware/poses.py`: added `HANGING_ORIGIN_Q`, `HANGING_LOWER_Q`, `HANGING_ALPHA_0_5_Q`,
  `q_for_hanging_height_alpha(alpha)`. `ACTIVE_ORIGIN_Q`, `LOWER_B_Q`,
  `HEIGHT_ALPHA_0_5_Q`, `HEIGHT_ALPHA_0_5_CLEARANCE_Q`, `q_for_height_alpha` all unchanged
  (verified byte-identical, `tests/hardware/test_poses.py::test_hanging_pose_family_does_not_mutate_existing_constants`).
- `rl_gain_scheduling/gain_scheduling_env.py`: mirrored `HANGING_ORIGIN_Q`/`HANGING_LOWER_Q`
  additively (per this session's own finding that this file's duplicate
  `ACTIVE_ORIGIN_Q`/`LOWER_B_Q` had already silently diverged once for the original pair --
  keeping the two files in sync for the new pair). Not wired into the env's own
  observation/reset/training logic; RL retraining is explicitly out of scope for this task
  (six documented prior RL gain-scheduling failures, `docs/CURRENT_STATUS.md`).
- New configs (neither modifies any existing config file):
  - `config/ur5e_mujoco_torque_osc_hanging_pose.yaml` -- `config/ur5e_mujoco_torque_osc_tuned.yaml`'s
    gains and structural flags verbatim (`task_space_inertia_shaping`, `nullspace_posture`,
    `posture_reanchor_on_settle`, `lambda_regularization: 0.1`,
    `jacobian_singular_cond_max: 1.0e18`), only `home_qpos` changed to
    `HANGING_ALPHA_0_5_Q`.
  - `config/ur5e_mujoco_torque_osc_hanging_pose_friction_ff.yaml` -- the above plus
    `friction_feedforward: true` (byte-identical coulomb/viscous/deadband values to
    `config/ur5e_mujoco_torque_osc_tuned_friction_ff.yaml`).
- New tests: `tests/hardware/test_poses.py` (+6 tests: endpoint interpolation, midpoint,
  wrist_2 invariant, range validation, non-mutation of old constants) and
  `tests/mujoco/test_hanging_pose_family.py` (+3 tests: cond(J) bound across the full
  range, Z-height reachability, a sanity anchor confirming the old family is still
  singular on this same model/test infrastructure).

## 3. Gain-tuning attempt and validation

Per the task instructions, this is deliberately a first pass, not a tuning campaign: start
from `ur5e_mujoco_torque_osc_tuned.yaml`'s existing gains unchanged, run the standard
4-category rigor sweep (`canonical_grid`/`long_holds`/`large_displacements`/
`torque_scale_robustness`, same grids as `tools/ur5e_pose_sweep_transport.py`), at the new
family's `alpha=0.5` midpoint (`HANGING_ALPHA_0_5_Q`), same seed=0 methodology used
throughout this session's other configs.

### 3.1 Plain gains (no retuning at all)

| category | result |
|---|---|
| canonical_grid | 4/8 |
| long_holds | 5/8 |
| large_displacements | 2/8 |
| torque_scale_robustness | 12/14 |
| **total** | **23/38 (60.5%)** |

Every failure's `move_failure_reason`/`hold_failure_reason` was `*_target_tracking`
undershoot (e.g. dx=0.02 m achieving 0.0155 m mid-move, 77%), not a guard trip -- the same
signature AGENTS.md sec 3 already documents for the old family post-friction-model
(`config/ur5e_mujoco_torque_osc_tuned.yaml`'s own height_alpha=0.5 pass rate "roughly
halved, 36/38 -> 19/38" once real joint friction was added to the sim model 2026-07-31,
with zero controller-side compensation). **This 23/38 is already a fair, direct,
apples-to-apples improvement over the old family's documented 19/38 at the identical
config, identical friction-affected plant, identical rigor-sweep methodology** -- same
gains, same lack of friction compensation, different pose family only.

### 3.2 Plus `friction_feedforward` (one targeted, already-validated addition)

Since the plain-gains failures were friction-undershoot, not guard trips or a
pose-specific problem, `friction_feedforward` (`docs/status/`-documented, already validated
at the old family's own height_alpha=0.2/0.3/0.5) was tried as a second, quick iteration --
not a new tuning campaign, just reusing an existing, well-understood fix for the exact
failure signature observed:

| category | result |
|---|---|
| canonical_grid | 8/8 |
| long_holds | 8/8 |
| large_displacements | 8/8 |
| torque_scale_robustness | 12/14 |
| **total** | **36/38 (94.7%)** |

**Direct comparison to the old family's own documented friction_ff number at
height_alpha=0.5**: old family 33/38 (86.8%); new hanging family **36/38 (94.7%)** --
better, using the identical fix, at a directly comparable displacement/hold/torque-scale
grid. It also matches the old family's own *pre-friction-model* historical ceiling (36/38,
frictionless era) -- i.e. the new pose family with friction feedforward performs as well as
the best result this repo has ever recorded for the old family at any point in its history,
friction included.

Worst-case orientation error across all 38 runs: 0.175 rad (`large_displacements`,
dx=+0.20 m), comfortably under the 0.25 rad guard and lower than the old family's own
comparable worst case (0.2497-0.25 rad at its own directional-ceiling dx=0.20 m failure,
AGENTS.md sec 3) -- consistent with, though not proof of, the singularity-avoidance design
also helping orientation-holding margin, since orientation is held via the same
nullspace-posture mechanism that the old family's directional-ceiling investigation found
was itself degraded near the wrist singularity.

### 3.3 Honest residual failures (both configs)

Both configs fail the identical two cases: `torque_scale_robustness` at `torque_scale=0.1`
(10% of nominal torque budget), `dx in {0.03, 0.06}`, `hold=2.0s`. `dx=0.03`: undershoot
(`achieved=0.0176` vs `target=0.03`, tracking failure, not a guard trip). `dx=0.06`: a
genuine `|Y-Y0| > 0.03 m` guard trip. This matches the old family's own already-documented
pattern -- AGENTS.md sec 3 notes "One shared failure (alpha=0.5, torque_scale=0.1) is
identical in both configs -- a real torque-budget limit, unrelated to [the pose/singularity
fix being validated]." Read as a genuine torque-budget floor at 10% scale, not a
pose-specific defect: the family's own gravity-comp torque (9-17 Nm) is a meaningful
fraction of a 10%-scaled 15/2.8 Nm limit (150/28 Nm nominal x 0.1), leaving little headroom
for tracking authority on top of just holding position against gravity.

### 3.4 What was NOT done (explicitly out of scope for this first pass)

- No gain retuning beyond reusing the existing `friction_feedforward` fix verbatim.
- No sweep across other `hanging_alpha` values (only the alpha=0.5 midpoint was validated;
  the reachability probe in sec 1.4 gives some independent evidence the endpoints also
  work, but that is not the same as a full rigor sweep at each one).
- No attempt to close the two remaining `torque_scale=0.1` failures.
- No RL retraining (explicitly excluded by the task).

## 4. Honest verdict

**This is a real, viable path, not a dead end** -- the central claim (cond(J) stays
well-conditioned across the whole new family, both statically and under real closed-loop
motion) is directly verified with numbers, and a first-pass gain config reaches 36/38, at
or above every historical number this repo has for the old family, including its
frictionless-era ceiling. The design did not trade away reach (sec 1.4) or torque margin
(sec 1.2) to get there. No new singularity or unexpected failure mode was found anywhere in
this pass -- the two residual failures (torque_scale=0.1) match a pre-existing, pose-agnostic
torque-budget limit already documented for the old family, not a new problem introduced by
this design.

What would still need attention before calling this "finished" (not attempted here, out of
scope for a first pass): validation across the full `hanging_alpha` range rather than just
the midpoint; whether the `-45°`/wall-clearance base-rotation problem the old family needed
`HEIGHT_ALPHA_0_5_CLEARANCE_Q` for would recur or behave differently in this new posture (a
"hanging" arm's swept volume near the base is a different shape and has never been checked
against the real lab's obstacles); and whether the directional-ceiling and orientation-growth
issues that motivated abandoning `split_base_wrist_task` as sufficient are actually absent
here or merely not yet triggered by the grids tested.

## 5. Real-hardware readiness -- explicit, prominent flag

**This pose family has NEVER been tested on real hardware and has NO physical clearance
verification.** Per this repo's own established discipline (`hardware/poses.py`'s own
comment on `HEIGHT_ALPHA_0_5_CLEARANCE_Q` needing re-verification any time the setup
changes, and the fact that pose was only adopted after being "visually confirmed twice on
the real robot"), a genuinely different arm posture -- elbow-down rather than
near-fully-extended -- sweeps through completely different physical space near the base,
shoulder, and elbow than anything previously run in the physical lab. Before this pose
family is ever commanded on the real UR5e:

1. A slow, supervised, `position`-mode-only visual clearance check at both endpoints and
   the midpoint, with a human present and ready to e-stop, checking for collision with the
   table, mount, cables, and any nearby obstacles the old "tall" family's swept volume never
   came near.
2. Only after that: a small-first `direct_torque` test (as this session's own real-hardware
   log did for `split_base_wrist_task`), starting from the smallest displacement in the
   canonical grid.
3. This config is explicitly NOT a replacement for the real-hardware default
   (`hardware/poses.py::HEIGHT_ALPHA_0_5_CLEARANCE_Q` /
   `config/ur5e_mujoco_torque_osc_tuned_wrist_orient_fixed_neg45_pose.yaml`) without its own
   separate human decision, exactly like every other unvalidated config in this repo's
   history.

## Files changed

- `hardware/poses.py` (additive)
- `rl_gain_scheduling/gain_scheduling_env.py` (additive)
- `config/ur5e_mujoco_torque_osc_hanging_pose.yaml` (new)
- `config/ur5e_mujoco_torque_osc_hanging_pose_friction_ff.yaml` (new)
- `tests/hardware/test_poses.py` (additive)
- `tests/mujoco/test_hanging_pose_family.py` (new)
- `docs/status/hanging_pose_transport_family_2026-08-01.md` (this file)

## Tests run

- `python -m pytest -q tests/hardware/test_poses.py tests/mujoco/test_hanging_pose_family.py`
  -- 10 passed.
- `python -m pytest -q` (full suite) -- 578 passed, 3 xfailed (pre-existing, unrelated),
  zero regressions.

## Rollback

`git revert <this commit's hash>` -- every change here is additive (new constants, new
files); reverting removes the new pose family and configs without touching anything the
old family or any other config depends on.
