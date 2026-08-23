# Transport-axis generalization, split-controller joint restriction, SCI, and pendulum hinge-axis investigation (2026-08-12)

Real-hardware session at the mega-search-winner-adjacent pose family, driven by a real,
physical constraint discovered live: `wrist_2` cannot be moved away from its current
near-zero value (the classic UR wrist gimbal-lock configuration) on this specific robot.
Everything below traces from that one constraint. All controller/pendulum work is
**sim-only** unless explicitly marked otherwise; nothing below has been validated on real
hardware beyond the connectivity/pose-probing already logged elsewhere.

## 1. The real-hardware starting point

Live probes (`tools/ur5e_connect.py --robot-ip 172.16.71.77 --once`) found the arm sitting
at `q = [-2.3688, -2.1801, -1.8838, -0.7962, ~0, 0.0206]` -- `wrist_2` within a fraction of
a degree of 0, i.e. essentially parked at the UR wrist singularity (both `wrist_2=0` and
`wrist_2=180deg` are singular; only `wrist_2~=90deg` is well-conditioned, confirmed
repeatedly across this and prior sessions). `cond(J6)` at this exact pose measures
1395.76-3823 depending on the precise decimal (varies with how many degrees off zero the
live probe caught it).

This pose was previously validated only bare-arm at `MEGA_SEARCH_WINNER_Q`
(`hardware/poses.py`) -- the arm is not actually sitting at that exact pose live, it is
sitting at a nearby, wrist_2-singular variant of it.

## 2. Real, structural finding: which 3-joint subsets can do X-transport, and why most can't

Original ask: hold `shoulder_pan` fixed (base clearance), and use exactly
`{shoulder_lift, elbow, wrist_1}` for a 1D front/back transport task.

**This specific 3-joint set is structurally rank-deficient at every pose, not just this
one** -- confirmed by sweeping `cond(J_position[:, {shoulder_lift,elbow,wrist_1}])` at 8+
random poses plus every named pose in `hardware/poses.py`: cond is always ~1e15-1e16 (rank
2 of 3), never a pose-dependent fluke. Reason: `shoulder_lift`, `elbow`, and `wrist_1`
rotate about axes that stay mutually parallel throughout the arm's motion (the classic UR
"shoulder-elbow-wrist-pitch" planar sub-chain). Three parallel-axis revolute joints can
only ever span a 2D subspace of 3D linear velocity -- a cross-product identity
(`axis x lever_arm` is always perpendicular to `axis`), not a numerical coincidence.

Full 10-combination sweep of 3-joint sets excluding `shoulder_pan`, across multiple poses:
only sets that include `wrist_2` are ever well-conditioned (median cond single digits to
low tens); every set including `wrist_3`, and the original `{shoulder_lift, elbow,
wrist_1}`, is singular everywhere. **Any usable pan-free 3-joint set for full 3D position
control needs wrist_2 active, not frozen** -- which also happens to solve the original
motivating problem better than freezing wrist_2 would have, since an active wrist_2 gets
driven toward its own well-conditioned region (~90deg) instead of being stuck wherever it
started.

**But this changes for a 1-dimensional task.** The above is about spanning all of 3D
position space. A front/back transport task is 1D. The single world-X row of the position
Jacobian, restricted to just `{shoulder_lift, elbow, wrist_1}`, has real magnitude at the
real pose (0.2353) -- essentially identical to the verified full-rank set's own X-row
magnitude (0.2351 for `{elbow, wrist_1, wrist_2}`). The missing 3rd rank dimension was
never needed for a 1D task; world-X lies well inside the 2D plane these three joints do
span.

## 3. Controller feature: combined row + column task restriction (`split_base_wrist_task_dims`)

Pre-existing `split_base_wrist_task` (`controller_core/x_axis_cartesian_impedance/`)
restricted the translation task to a subset of joint COLUMNS but always used all 3 position
ROWS -- a 3x3 system, unusable for a rank-2 column set like section 2's original ask.
`reduced_task_dims` restricted ROWS but always used all 6 joint columns. The two were
mutually exclusive (never tested together).

New field `split_base_wrist_task_dims: tuple[int,...] | None` (rows 0=X/1=Y/2=Z; `None` =
all 3, today's unchanged default) makes the split's `J_task` `len(rows) x len(cols)`
instead of always `3 x len(cols)` -- e.g. 1x3 for X-only over 3 joints.

Verified generalization of every interacting mechanism (not assumed):
`task_space_inertia_shaping` and `nullspace_posture` both generalize cleanly to a
non-square `J_task` (verified against independently-recomputed references to 1e-12).
`svd_singularity_filtering` (SCI, section 4) also generalizes exactly. `singular_scale`'s
`cond()` metric is identically 1.0 for any 1-row block, so it silently never engages for a
1-row task -- **documented as a real consequence, not silently wrong**, matching the
convention `reduced_task_dims` already used. `lambda_adaptive_regularization`,
`acceleration_feedforward`, and `y_integral_action` are explicitly **guarded to raise**
when they would be ambiguous or actively wrong at 1 row, rather than silently producing a
bad result.

Golden-value proof: default behavior (field unset) is byte-identical before/after (full
deterministic-rollout diff empty, matching hashes).

**Real closed-loop numbers at the actual real-hardware pose**, columns
`{shoulder_lift(1), elbow(2), wrist_1(3)}`, rows `{X(0)}`
(`config/ur5e_mujoco_torque_osc_tuned_split_base_wrist_lift_elbow_wrist1_x_only.yaml`):

| dx | hold | tracking | max\|Y\| | max\|Z\| | guard |
|---|---|---|---|---|---|
| +0.02m | 2s | 89.2% | 0.0174 | 0.0237 | - |
| -0.02m | 2s | 86.3% | 0.0168 | 0.0259 | - |
| +0.02m | 20s | 99.1% | 0.0193 | 0.0262 | - |
| -0.02m | 20s | 88.7% | 0.0173 | 0.0262 | - |
| +0.03m | 2s | 77.6% | 0.0227 | 0.0300 | **`\|Z-Z0\|>0.03m` @1.72s** |
| +0.05m | 2s | 47.1% | 0.0229 | 0.0300 | **`\|Z-Z0\|>0.03m` @1.06s** |
| +0.10m | 2s | 24.2% | 0.0236 | 0.0301 | **`\|Z-Z0\|>0.03m` @0.78s** |
| +0.20m | 2s | 12.3% | 0.0240 | 0.0302 | **`\|Z-Z0\|>0.03m` @0.60s** |

Compare to the same 3 columns under the OLD, unusable 3-row mechanism: only 53-67%
tracking even at the small displacement, never really arriving -- the row+column fix is
the real difference, confirmed, not noise.

**Real validated envelope at this pose, for THIS X-only-row config, is ~+-0.02m.** The
guard trips at essentially the same ~0.030m Z-drift ceiling regardless of target size -- a
bigger target just reaches it faster (0.60s at 0.20m vs 1.72s at 0.03m), not a bigger
displacement before tripping. A sim-only extension test (guard intentionally left running
past its trip point, for visualization) showed drift does NOT plateau near the guard line
if not stopped -- by the end of a full 2s move toward a 0.20m target, Y drift reached
0.184m and Z drift 0.141m. The guard is doing real protective work; this is not evidence
the "true" envelope is close to 0.03m, it demonstrates the opposite.

**Superseded by section 3a below**: the actual fix was not a gain or an eps schedule --
it was making Z an active task row instead of holding it. See 3a for the corrected
numbers (+-0.031m) and why that's the exact kinematic ceiling, not just a better result.

Y/Z ARE genuinely posture-held, verified by ablation not just asserted: held joints move
3.3x less with the posture spring on vs off; at dx=-0.02m, Z drift stays 0.0262m with the
spring and walks straight past the 0.03m guard without it.

### 3a. What actually caps the range at this pose (corrected 2026-08-13)

**This section originally claimed the nullspace-projector damping leak (below) was
"very likely the dominant reason" for the ~0.02m ceiling. That was wrong, and a
follow-up gain-search agent caught it rather than let it stand.** The corrected finding,
from 2,460 real closed-loop rollouts plus a proper derivation, replaces the hypothesis
below with a proven cause. Left the original analysis in place (folded into "the damping
leak was real, just not the bottleneck" below) rather than deleting it, since the leak
itself is real and the fix for it shipped -- it just doesn't explain the ceiling.

**The actual fix: make Z an active task row instead of holding it.**
`{shoulder_lift, elbow, wrist_1}` were already known (section 2) to span a real 2D
subspace of Cartesian velocity at this pose, not just 1D. The X-only config only ever
asked the controller to use 1 of those 2 available dimensions, then relied on the posture
spring to suppress motion in the other -- fighting a real, physically-available direction
instead of asking for it. Setting `split_base_wrist_task_dims: [0, 2]` (X and Z both
active task rows, Y still posture-held) with the *exact same gains* as the X-only config
-- no search, no retuning -- takes the range from +-0.02m to **+-0.031m**, a 55% increase.
New config: `config/ur5e_mujoco_torque_osc_tuned_split_base_wrist_lift_elbow_wrist1_xz_kinematic_max.yaml`.

**This is provably the exact kinematic ceiling, not just the best result found.** At this
pose (`shoulder_pan = -135.7 deg`), the one truly unreachable direction for this 3-joint
set is the world X/Y diagonal -- any real motion along the available 2D plane drags Y at a
fixed ratio of 0.975 (Y_row = 1.0255 * X_row within that plane, the same coupling
identified in section 2). Y still has its own 0.03m drift guard. Solving
`0.975 * dx = 0.03` gives `dx = 0.0308m` -- and the measured guard-trip ceiling lands
right on that number. There is nothing left on the table at this pose; the only way past
+-0.031m is a different pose or a loosened Y guard, not a better controller.

**The gain search itself was a genuine, honest negative result.** 2,460 real closed-loop
rollouts across 6 gain dimensions (posture and task gains, X-only config) found no
combination that extends the +-0.02m ceiling -- confirming empirically, not just by
assumption, that the X-only ceiling was structural (a missing task dimension) rather than
a tuning problem. This is real, useful evidence, and it's the reason the search agent kept
looking past "better gains" and found the actual fix instead.

**The nullspace-projector damping leak described below was real, and the fix for it
shipped, but it does not move the ceiling.** Original analysis, still accurate as far as
it goes: the `nullspace_posture` term holds the frozen joints via a plain joint-space
spring, projected through the nullspace of `J_task` via a fixed `eps=0.1` regularization
constant everywhere in this controller. At this pose, with few task rows, the real physics
scale (`J_task @ M^-1 @ J_task.T = 0.0975`) is *smaller* than `eps` itself, so the
projector only cancelled ~49% of the posture spring's leak into the task direction instead
of nearly all of it. The follow-up work built a proper, mathematically-derived fix
(`nullspace_inertia_adaptive_regularization`/`nullspace_inertia_eps_ratio` --
schedules `eps` against the actual block-norm scale rather than the degenerate `cond()`
metric this doc originally proposed working around) and verified it cuts the leak from
49% down to under 5%, checked to 9 decimal places against the derived math. It genuinely
improves tracking quality (86% -> 95%) at a fixed target. **But applied to the X-only
config, it does not extend the ~0.02m envelope at all** -- direct evidence the leak was
never the bottleneck; the missing Z task row was. Kept as a real, validated,
default-off tracking-quality improvement, independent of the range question.

## 4. Singularity-Consistent Inversion (SCI): per-direction damping instead of uniform

Existing near-singularity handling (`lambda_regularization * I` inside
`Lambda=(J M^-1 J^T + eps I)^-1`, and the scalar `singular_scale`) is uniform across all
task directions -- at a true 1-DOF-lost singularity, this also throws away authority in the
5 directions that are still fine. New opt-in `svd_singularity_filtering` (default off)
replaces this with a per-singular-value damped inverse
(`sigma_i* = (sigma_i^2+lambda_i^2)/sigma_i`, continuous at a threshold, `lambda_max`
chosen so the fully-lost-direction case reproduces today's exact damping and never exceeds
it).

**Real result at the reference singularity pose** (`HEIGHT_ALPHA_0_5_Q`, wrist_2=0,
cond=7.28e16): uniform damping delivers 72.77% of a commanded task acceleration and leaks
0.253 into cross-axes; SCI delivers 100.00% with cross-axis leak at machine epsilon
(2.3e-14). Closed-loop tracking improved from ~91-95% to ~96-98% at every displacement
tested, zero new guard trips.

**Real result at the user's actual pose** (messier conditioning, cond=1396, not a clean
single lost direction): tracking still improves (36.8%->51.9% at dx=0.02m; 55.1%->62.0% at
dx=0.05m) but Y-drift consistently gets WORSE, and at dx=0.05m SCI actually TRIPS the
`|Y-Y0|>0.03m` guard that uniform damping does not trip. **Not a clean win here** -- SCI
commands more total torque (not automatically conservative everywhere), confirmed directly:
at a WELL-conditioned pose (`MEGA_SEARCH_WINNER_Q`, cond=6.9) SCI commands up to 1.78x more
torque than uniform damping, since the existing tuned gains were implicitly tuned around
uniform damping's extra conservatism. **Turning SCI on requires its own gain-retuning pass,
not a drop-in swap.** Shipped default-off; a demonstration config exists
(`config/ur5e_mujoco_torque_osc_tuned_sci_svd_filtering.yaml`) explicitly labeled
unvalidated beyond the specific numbers above.

Golden-value proof: default off is byte-identical (5x re-verified). URScript lane does not
read this flag -- would silently run plain uniform damping on real hardware while sim shows
SCI; not patched (`hardware/` out of scope for that change).

## 5. Pendulum hinge-axis investigation

The pendulum attachment (`assets/ur5e_pendulum/pendulum_attachment.xml`) had its hinge
axis set to `local X` ("parallel to the tool face") -- confirmed via the file's own history
to be a placeholder never derived from the real CAD assembly's mate geometry, chosen only
because it happened to be horizontal at whichever pose was in use at the time.

**Physically standard convention is perpendicular-to-face** (`local Z`, the mounting-stack
direction, matching how a shaft housing bolted to a flat plate normally protrudes).

**First attempt (old pose, `MEGA_SEARCH_WINNER_Q`-derived `ARM_Q0`): rejected.** At that
pose local Z is 11.25deg off vertical -- gravity produces near-zero torque about it
(measured: released 90deg off hanging, creeps 9.9deg and stops dead vs local X's clean
71.6deg swing-and-settle). A 2200-evaluation DE search over gains/kick shape found no
combination producing a real flip (best 2.24 rad from inverted, need <0.35).

**Second attempt (the real-hardware pose from section 1): the axis genuinely works here.**
At this pose local Z is 89.73deg off vertical (essentially perfectly horizontal) -- a
complete flip from the old pose, not a coincidence: whichever axis happens to be
horizontal swaps as the arm's overall orientation changes. Landed changes:
- `pendulum_hinge` axis: `1 0 0` -> `0 0 1`.
- Rod re-oriented from along-hinge (`local +Z`, would have been collinear -- an offset
  crank that never visibly swings) to perpendicular (`local +X`), housing re-oriented to
  match. Length/diameter/mass of all parts unchanged -- direction only.
- Mass and inertia about the hinge came out **exactly identical** before/after (same parts,
  rotated together as a rigid assembly). `R_COM_M` also came out effectively unchanged
  (0.029183 m both ways) -- verified 3 independent ways again (qfrc_bias sine fit, residual
  1.3e-16; free-release period; geometric cross-check).
- New equilibria (rigorous `find_inverted_angle`): inverted=+0.1271 rad, hanging=-3.0145
  rad (old: +0.196/-2.945 -- do not reuse old values).
- Wrist clearance actually IMPROVED (4.45cm vs old 1.6cm, arm-side). Floor clearance
  dropped to 6.34cm (was 26.8cm) -- thinner margin, not contact, flagged for a real check
  before any hardware use.

**No genuine flip achieved at this pose either -- but for a different, precisely
identified reason.** Broadened DE search (~2200 evals): best 2.24 rad from inverted, same
ceiling as the old pose. This time the pendulum's OWN dynamics are fine (gravity-torque
margin 2.79x frictionloss, real if ~30% weaker kick coupling than the old pose's near-
perfect alignment). **The actual blocker is arm-side**: the controller trips its own
`|Y-Y0|>0.03m` guard on any kick large enough to matter, tracing directly to this repo's
already-documented base-rotation X-Y authority trade-off (AGENTS.md sec 3) --
`shoulder_pan~=-135.7deg` at this pose sits deep in that same territory. Separately, this
pose's own arm-side `cond(J6)=1395.76` already exceeds the diagnostic scripts'
`SINGULARITY_COND_THRESHOLD=1000` at rest, before any motion -- flagged as a real "is this
threshold still right, or is this pose just not acceptable" question for a human, not
silently resolved either way.

A real, separate bug found and fixed along the way: the cond(J)/excursion safety tracking
in `pendulum_swingup_energy_shaping.py::run_energy_swingup_trial` was dead code -- allocated
and commented as protection, never actually computed or returned, so `objective()` never
saw it. Now wired in and penalized in the search.

Artifacts: `outputs/pendulum_renders/new_pose_local_z_axis_{wide,closeup,closeup_edge_on,
closeup_inverted}.png`, `annotated_swingup_attempt_new_pose_axis_no_flip.mp4` (deliberately
not named "flip"). `outputs/pendulum_renders/split_controller_pendulum_drift.mp4` --
pendulum-attached visualization of the split-controller 0.20m attempt, guard intentionally
not stopping the sim (sim-only visualization request), HUD marks the real trip point.

**Known-broken elsewhere from this axis change, not fixed (out of scope)**: the velocity-
control lane's `pendulum_balance_test.py` hardcodes the OLD pose family
(`[0,-pi/2,pi/2,-pi/2,-pi/2,0]`), where the new axis maps to world -Z exactly -- 0.0 Nm
gravity torque there now. `pendulum_balance_disturbance_robustness.py` depends on it
transitively.

## 6. Open items for a human decision

1. `SINGULARITY_COND_THRESHOLD=1000` vs this real pose's resting `cond(J6)=1396` -- either
   the pose needs a different validated threshold with real evidence behind it, or this
   pose family should be considered unacceptable for arm-side reasons independent of the
   pendulum. Not resolved.
2. **Closed, corrected 2026-08-13** (section 3a): nullspace-projector `eps` scheduling
   was built and validated (`nullspace_inertia_adaptive_regularization`) -- real tracking
   improvement, but does NOT extend the envelope as originally hypothesized. The envelope
   fix was `split_base_wrist_task_dims: [0, 2]` (X+Z active) instead, +-0.031m, proven to
   be the exact kinematic ceiling at this pose. Real-hardware validation of the (X,Z)
   config and its gain-search-derived redistribution (lower Cartesian damping, higher
   joint damping) is the remaining open item -- needs its own small-first real-lab check,
   same as everything else in this doc.
3. SCI needs its own gain-retuning pass before being enabled anywhere real (section 4).
4. Floor clearance at the new pendulum pose (6.34cm) -- worth a physical check before any
   real-hardware pendulum mounting.
5. Hinge axis remains an inference from mounting convention, not confirmed against the real
   CAD assembly's actual mate geometry (SolidWorks files unreadable in this environment).
6. Velocity-lane pendulum scripts now silently broken by the axis change (section 5) --
   not touched, flagged only.

## 7. Rollback

Each piece is independently revertible; see the individual `git checkout` commands logged
in-session for `controller_core/x_axis_cartesian_impedance/{config,controller,output,
parsing}.py` (transport-axis + split-dims + SCI, all in the same 4 files -- revert
hunk-by-hunk if only one feature should go), `assets/ur5e_pendulum/pendulum_attachment.xml`
+ `tools/diagnostics/pendulum_swingup_energy_shaping.py` +
`tools/diagnostics/pendulum_balance_torque_lqr.py` +
`tests/mujoco/test_ur5e_pendulum_compose.py` (pendulum axis work), and the various new
`tests/`/`tools/diagnostics/`/`config/*.yaml` files, all safe to `rm` individually since
none are committed.
