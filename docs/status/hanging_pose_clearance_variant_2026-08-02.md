# Hanging-pose family: does a -45deg base-rotation clearance variant reintroduce the old
# family's Y-drift coupling? (2026-08-02)

## Why this exists

`docs/status/hanging_pose_transport_family_2026-08-01.md` (last night's work) built a new
"hanging"/elbow-down transport pose family (`hardware/poses.py::HANGING_ORIGIN_Q`/
`HANGING_LOWER_Q`/`q_for_hanging_height_alpha`/`HANGING_ALPHA_0_5_Q`) that structurally avoids
the `wrist_2=0` kinematic singularity the OLD pose family (`ACTIVE_ORIGIN_Q`/`LOWER_B_Q`/
`q_for_height_alpha`) sits at across its entire range. That doc's own "what would still need
attention" section flagged, unresolved: *"whether the `-45deg`/wall-clearance base-rotation
problem the old family needed `HEIGHT_ALPHA_0_5_CLEARANCE_Q` for would recur or behave
differently in this new posture."*

The old family's `-45deg` rotation (`hardware/poses.py::HEIGHT_ALPHA_0_5_CLEARANCE_Q`) was
needed for real wall/base clearance in the physical lab (visually confirmed on the real robot,
2026-07-31) but subsequently caused a real, extensively-investigated problem (AGENTS.md sec 3):
real hardware reproducibly tripped `ImpedanceSafetyMonitor`'s `|Y-Y0| > 0.03 m` guard at
dx=0.20m with the TCP moving in a near-45-degree diagonal, and three independent investigations
(gain sweep, orientation/nullspace mechanism family, P/D/I-authority + instrumentation) all
concluded this is a structural controller-architecture limitation at that rotated pose, not a
fixable gain problem, only partially mitigated by an evidence-scoped drift-tolerance increase
that itself remains real-hardware-unvalidated.

This doc characterizes whether the same base rotation, applied to the NEW hanging pose family,
reintroduces the same problem. **This is sim-only characterization, not a fix pass** (per task
scope) and **this pose is NOT real-hardware-ready** (see the prominent flag at the end).

## 1. The new pose constant

`hardware/poses.py::HANGING_ALPHA_0_5_CLEARANCE_Q` -- built by mirroring
`HEIGHT_ALPHA_0_5_CLEARANCE_Q`'s exact pattern (copy the family's own `alpha=0.5` anchor point,
override `shoulder_pan` to `-0.7853981633974483` rad, i.e. `-45deg`) onto
`HANGING_ALPHA_0_5_Q` -- the hanging family's own `alpha=0.5` midpoint, which is the point that
family's own first-pass gain-tuning and rigor sweep already anchored to (the natural analogue
of how `HEIGHT_ALPHA_0_5_CLEARANCE_Q` anchors to the old family's own validated
`height_alpha=0.5` point rather than a raw endpoint):

```
HANGING_ALPHA_0_5_Q            = [ 0.000000, -1.641803,  1.401547, -1.959057,  1.570796,  0.0]
HANGING_ALPHA_0_5_CLEARANCE_Q  = [-0.785398, -1.641803,  1.401547, -1.959057,  1.570796,  0.0]
```

Purely additive: `HANGING_ALPHA_0_5_Q`, `HANGING_ORIGIN_Q`, `HANGING_LOWER_Q`,
`q_for_hanging_height_alpha`, and every existing pose constant (including
`HEIGHT_ALPHA_0_5_CLEARANCE_Q` itself) are unchanged -- verified by
`tests/hardware/test_poses.py::test_hanging_alpha_0_5_clearance_does_not_mutate_existing_constants`.
Mirrored into `rl_gain_scheduling/gain_scheduling_env.py` per that file's own existing
duplicate-definition pattern (it did not previously mirror `HANGING_ALPHA_0_5_Q` at all; added
here only as the intermediate needed to build the clearance constant, matching the same
"kept in sync, not wired into training" caveat already on `HANGING_ORIGIN_Q`/`HANGING_LOWER_Q`
in that file).

## 2. cond(J) sweep across the rotated variant's range

**Result: kinematically, rotation changes nothing.** `shoulder_pan` rotates the entire
downstream kinematic chain rigidly about the base Z axis -- a symmetry that should leave
`cond(full 6x6 J)` exactly unchanged at any pose, verified directly:

- Sanity check at `HANGING_ALPHA_0_5_Q` with `shoulder_pan` swept through `{0, -45deg, 0.5,
  -1.2, pi}` rad: `cond(J) = 9.0991368...` at every single value (differences at the 1e-13
  level, floating-point noise).
- 21-point sweep across the full hanging-family range (`q_for_hanging_height_alpha(alpha)` for
  `alpha` in `[0, 0.05, ..., 1.0]`), `shoulder_pan` forced to `-45deg` at every point, compared
  directly against the un-rotated sweep from last night's doc:

  | alpha | 0.00 | 0.10 | 0.20 | 0.30 | 0.40 | 0.50 | 0.60 | 0.70 | 0.80 | 0.90 | 1.00 |
  |---|---|---|---|---|---|---|---|---|---|---|---|
  | cond(J), rotated | 15.41 | 13.38 | 11.91 | 10.77 | 9.85 | 9.10 | 8.49 | 7.98 | 7.57 | 7.26 | 7.04 |

  Identical to last night's un-rotated table to at least 4 significant figures; max abs diff
  across all 21 points, rotated vs. un-rotated: `1.24e-14`. Min/max across the rotated sweep:
  **min 7.04, max 15.41** -- the same numbers as the un-rotated family, comfortably below the
  `COND_BOUND=100` regression threshold and many orders of magnitude below the old family's
  `1e16-2.5e17` floor.
- Locked down as regression tests: `tests/mujoco/test_hanging_pose_family.py::
  test_hanging_alpha_0_5_clearance_stays_well_conditioned` (single-point, asserts exact
  rotation-invariance to `1e-6`) and `::test_hanging_clearance_rotation_invariant_across_full_range`
  (21-point sweep, same assertion at every alpha).

**So the static kinematic story is clean: the hanging family's singularity-avoidance is
completely unaffected by base rotation.** This is expected and, on its own, would suggest no
new problem -- but it is exactly the wrong question to stop at, since the old family's own
`-45deg` failure was **not** a cond(J)/singularity problem (its cond(J) was already ~1e17
regardless of rotation) -- it was a dynamic controller-architecture coupling. The rigor sweep
below checks that directly.

## 3. Rigor sweep at the rotated pose

Config: `config/ur5e_mujoco_torque_osc_hanging_pose_friction_ff.yaml` (the hanging family's own
validated config, 36/38 at the un-rotated `HANGING_ALPHA_0_5_Q`, per last night's doc) --
gains unchanged, only the start pose swapped via `--start-q-rad` to
`HANGING_ALPHA_0_5_CLEARANCE_Q`. This is the fair, apples-to-apples comparison: same gains, same
friction compensation, only the base rotation differs.

### 3.1 canonical_grid (dx 0.01-0.04m x hold 1/2s x scale 1.0, 8 runs)

`outputs/ur5e_mujoco_torque_transport/hanging_clearance_canonical_grid_2026-08-02/summary.json`

| dx (m) | hold (s) | valid | achieved_x (m) | y_max (m) | z_max (m) | orient_max (rad) | termination |
|---|---|---|---|---|---|---|---|
| 0.01 | 1/2 | **True** | 0.0060-0.0065 | 0.0052 | 0.0041 | 0.012-0.013 | duration_complete |
| 0.02 | 1/2 | False | 0.0133-0.0137 | 0.0125 | 0.0091 | 0.026 | duration_complete |
| 0.03 | 1/2 | False | 0.0209-0.0213 | 0.0205 | 0.0143 | 0.041 | duration_complete |
| 0.04 | 1/2 | False | 0.0287-0.0289 | 0.0287 | 0.0194 | 0.057 | duration_complete |

**Result: 2/8** (vs. the un-rotated pose's 8/8 at the identical config, per last night's doc) --
a real regression. Critically, **none of these 6 failures are guard trips** -- every one
terminates `duration_complete` and fails only on `move_phase_target_tracking`/
`hold_phase_target_tracking` (X-displacement undershoot vs. the tolerance `max(0.005m, 0.25 *
target)`, `transport_metrics.py::_target_x_tolerance`). This is a **different failure signature**
from the old family's canonical hard `|Y-Y0| > 0.03 m` guard trip.

**But the underlying coupling is the same phenomenon**, visible directly in the numbers: at
dx=0.04, `y_max` (0.0287m) is essentially equal to `achieved_x` (0.0287m) -- the ratio
`y_max / achieved_x` climbs from 0.87 at dx=0.01 to 1.00 at dx=0.04. The controller is moving the
TCP along a near-45-degree diagonal in response to a pure-world-X command, exactly the signature
AGENTS.md documents for the old family ("the TCP moving in a near-45deg diagonal (X and Y
displacement nearly equal)"). It just doesn't (yet, at this displacement) grow large enough in
absolute terms to trip the flat 0.03m guard -- `y_max` peaks at 0.0287m within this grid, just
under the 0.03m threshold.

### 3.2 Extended dx sweep, to find where (if anywhere) the guard actually trips

Since canonical_grid alone left the guard-trip question open (all 8 runs terminated
`duration_complete`), two follow-up single-category sweeps were run to characterize the larger-
displacement behavior (task explicitly asked for "at least canonical_grid" -- this extends that
minimum with real evidence rather than leaving an open question):

`outputs/ur5e_mujoco_torque_transport/hanging_clearance_extra_dx_2026-08-02/summary.json`
(dx=0.05/0.06/0.10, hold=2.0):

| dx (m) | valid | achieved_x (m) | y_max (m) | z_max (m) | orient_max (rad) |
|---|---|---|---|---|---|
| 0.05 | **True** | 0.0448 | 0.0260 | 0.0189 | 0.088 |
| 0.06 | **True** | 0.0542 | 0.0216 | 0.0165 | 0.111 |
| 0.10 | **True** | 0.0915 | 0.0154 | 0.0172 | 0.210 |

All three pass -- the X-tracking tolerance's relative term (`0.25 * target`) grows faster than
the absolute tracking error here, so these clear the bar despite `y_max` still being non-trivial.
Note `y_max` is now *shrinking* in absolute terms as dx grows (0.026 to 0.0154m) even as
`achieved_x` grows -- the near-1:1 diagonal signature seen at small dx does not persist to this
range; orientation error is growing instead (0.088 to 0.210 rad, approaching the 0.25 rad guard).
This non-monotonic pattern (Y-coupling ratio peaks around dx=0.04-0.05, then recedes) is reported
honestly and is not fully explained here -- characterization only, not a mechanism claim.

`outputs/ur5e_mujoco_torque_transport/hanging_clearance_large_dx_2026-08-02/summary.json`
(dx=0.15/0.20, hold=1.0):

| dx (m) | valid | achieved_x (m) | y_max (m) | orient_max (rad) | termination |
|---|---|---|---|---|---|
| 0.15 | False | 0.1274 | 0.0224 | 0.2496 | `||orientation error|| > 0.25 rad` |
| 0.20 | False | 0.1372 | 0.0301 | 0.2341 | **`|Y-Y0| > 0.03 m`** |

**At dx=0.20m, the rotated hanging pose trips the exact same `|Y-Y0| > 0.03 m` guard that hit
the old family** -- direct confirmation that the coupling is real and does eventually manifest
as the same hard failure mode, not just a softer tracking miss. At dx=0.15m a different guard
(orientation, 0.25 rad ceiling) fires first.

## 4. Verdict

**The hanging pose family's structural singularity-avoidance does NOT survive base rotation
unscathed.** cond(J) is completely rotation-invariant (as expected -- this was never really in
question given the underlying rigid-body symmetry), but that was answering the wrong half of the
old family's problem. The dynamic X-Y coupling that caused the old family's real-hardware guard
trips **is reintroduced** by the `-45deg` rotation on the hanging family too, evidenced two ways:
(1) a near-1:1 X:Y diagonal-motion signature at small displacements (dx=0.02-0.04m), and (2) an
actual `|Y-Y0| > 0.03 m` guard trip at dx=0.20m, identical in kind to the old family's own
documented failure.

**However, this is a real, quantified, partial improvement, not a wash**: the hanging family's
guard-trip onset in sim is dx~0.15-0.20m, vs. the old family's documented sim onset of
dx~0.05-0.06m -- roughly 3x further out. The canonical_grid category itself (dx up to 0.04m)
shows only soft tracking-tolerance misses in the hanging family, not hard guard trips, whereas
the old family's canonical_grid-range failures were already hard guard trips at the identical
rotation. Whether this margin is "enough" for any given real deployment is a judgment call for a
human, not something this pass adjudicates.

## 5. What's different about the hanging posture's kinematics (brief characterization, not a fix)

The old family's `-45deg` coupling was investigated at length in AGENTS.md as inseparable from
its `wrist_2=0` near-singularity (cond(J) ~1e17 throughout) -- the posture's own site-history
text repeatedly frames the Y-coupling alongside the singularity as related failure modes of the
same underlying pose family. The hanging family's cond(J) is 7-15 throughout, on the rotated
variant exactly as on the un-rotated one (section 2) -- i.e. **the hanging family's Y-coupling
reproduces at a pose that is definitively NOT near-singular, kinematically well-conditioned by
a wide margin**. Since a well-conditioned Jacobian means pure-world-X motion (zero Y, zero Z,
zero orientation error) is kinematically reachable via a small joint-velocity combination at
this pose -- there is no hard kinematic obstruction here, unlike the old family -- the diagonal
motion the controller actually produces must be coming from the controller's task-space
weighting / nullspace-authority architecture itself, not from any kinematic necessity. This is
new evidence disentangling the two: the old family's investigation could never fully separate
"is this the singularity, or is this the rotation" because both were always present together;
the hanging family's clean cond(J) here isolates the rotation as sufficient on its own to
reproduce a Y-coupling problem, with no singularity involved. Consistent with AGENTS.md's own
already-drawn conclusion for the old family ("no P, D, or I gain in this controller architecture
can hold Y without breaking X-tracking at this pose/displacement... structural, not a search
gap") -- this looks like the same controller-architecture property, now shown to also apply at a
well-conditioned rotated pose, not a phenomenon tied to being near a singularity. No further
mechanism investigation (Lambda-coupling traces, nullspace-projector Frobenius norms, etc., the
tools AGENTS.md's own prior investigations used) was done here -- out of scope for this
characterization-only pass.

## 6. Real-hardware readiness -- explicit, prominent flag

**This pose has NEVER been tested on real hardware and has NO physical clearance verification,
exactly like the rest of the hanging pose family it is built on.** Unlike
`HEIGHT_ALPHA_0_5_CLEARANCE_Q` (visually confirmed twice on the real robot before being adopted
as a default), `HANGING_ALPHA_0_5_CLEARANCE_Q` is a sim-only characterization artifact:

1. The hanging posture's swept volume near the base is a fundamentally different shape from the
   old "tall" family's (elbow-down vs. near-fully-extended) -- rotating that different shape by
   `-45deg` near the base/wall has never been visually checked in the physical lab.
2. Even setting clearance aside, this doc's own finding (section 3-4) is that the rotation
   reintroduces a real, guard-trip-capable Y-drift coupling failure mode in sim -- exactly the
   kind of problem that, per this repo's own established history with the old family, does not
   reliably announce itself gently in transition from sim to real hardware (the old family's own
   real-vs-sim dose-response gap for this same failure was never fully explained).
3. Before this pose is ever commanded on the real UR5e: a slow, supervised, `position`-mode-only
   visual clearance check at minimum (per the hanging family's own existing real-hardware
   readiness section), AND a deliberate human decision about whether the guard-trip margin found
   here (dx~0.15-0.20m onset vs. the old family's dx~0.05-0.06m) is acceptable for the intended
   use, AND its own small-first `direct_torque` validation exactly like every other pose change
   in this project's history.
4. This pose is explicitly NOT a replacement for any existing real-hardware default
   (`hardware/poses.py::HEIGHT_ALPHA_0_5_CLEARANCE_Q` remains the real-hardware default for
   `direct_torque` transport, unmodified) without its own separate human decision.

## Files changed

- `hardware/poses.py` (additive: `HANGING_ALPHA_0_5_CLEARANCE_Q`)
- `rl_gain_scheduling/gain_scheduling_env.py` (additive: mirrored `HANGING_ALPHA_0_5_Q` +
  `HANGING_ALPHA_0_5_CLEARANCE_Q`)
- `tests/hardware/test_poses.py` (+2 tests)
- `tests/mujoco/test_hanging_pose_family.py` (+2 tests)
- `docs/status/hanging_pose_clearance_variant_2026-08-02.md` (this file)
- No existing config file, `controller_core/` file, or existing pose constant modified.

## Tests run

- `python -m pytest -q tests/hardware/test_poses.py tests/mujoco/test_hanging_pose_family.py`
  (via the `mujoco_ur5e` conda env) -- 14 passed.
- `python -m pytest -q` (full suite) -- 606 passed, 3 xfailed, 1 failed. The 1 failure
  (`tests/hardware/test_direct_torque_residual_observer_async.py::
  test_residual_observer_async_phase_cost_is_much_lower_than_sync`) is an unrelated,
  pre-existing timing-deadline flake (`deadline_overrun: single cycle late by 5.24 ms > 5x
  max_deadline_ms`) reproduced in isolation on this shared, loaded machine (`uptime` load
  average ~10 on 72 cores during this session) -- confirmed unrelated to `hardware/poses.py` or
  `rl_gain_scheduling/gain_scheduling_env.py` by inspection; not touched by this change.

## Rollback

`git revert <this commit's hash>` -- every change here is additive (one new pose constant, its
mirror, and new tests); reverting removes the new constant and its tests without touching the
hanging pose family, the old pose family, or any config this doc's sweeps ran against.
