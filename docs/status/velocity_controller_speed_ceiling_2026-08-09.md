# ik_seeded_resolution: empirical max speed + orientation-loss tolerance

**Date:** 2026-08-09
**Context:** follow-on to the pendulum swing-up velocity analysis (same date) -- user asked
to empirically TEST the velocity controller's real achievable speed rather than trust the
configured cap, and to characterize how much orientation error/loss is tolerable before
infeasibility.

## Mechanism correction found before the test could even be designed

`controller.py`'s `max_lin_speed_mps`/`max_ang_speed_radps` clamp (lines 147-151) is applied
**unconditionally after the mode dispatch** (confirmed by reading the code directly, not
inferred) -- unlike `kp_x`/`kp_rot`, which are read only inside specific mode branches and
simply never touched by `ik_seeded_resolution`, this speed cap is a real, universal post-hoc
clamp that DOES apply to `ik_seeded_resolution`'s output. 0.25 m/s (software default,
`CartesianVelocityConfig.max_lin_speed_mps`) and 0.05 m/s (hardware default,
`hardware/safety.py`'s `max_tcp_speed_mps`) are genuine enforced ceilings, not bypassed.

A second, new finding surfaced while building the test: **`ik_seeded_resolution` ignores
`target_ee_vel`/velocity-feedforward entirely.** `compute_ik_seeded` (`modes.py`) never reads
`xd_full` -- only `p_des`, `quat0`, `q_rest`, `q_current`. It solves a fresh position-IK target
`q_target` from `p_des` every cycle (seeded from `q_rest`, never `q_current` -- the
path-independence property this mode exists for), then drives
`qd_joint = ik_joint_gain * (q_target - q_current)`, `xd_cmd = jac_current @ qd_joint`. So "how
fast can it move" is governed by how far the ABSOLUTE POSITION TARGET is from the current
position (dx) and `ik_joint_gain` (config default 4.0/s), not by any commanded velocity value
at all. This extends the existing "kp_x/kp_rot are no-ops for this mode" finding to velocity
feedforward too. The first version of the test script commanded pure `target_ee_vel` and
measured near-zero achieved speed as a result -- a real bug in the test's own design, caught
and fixed before reporting any numbers.

## Method

`tools/diagnostics/velocity_controller_speed_ceiling_test.py` -- kinematic-only simulation
(`hardware.local_dynamics.LocalMujocoDynamics` for FK/Jacobian at arbitrary q, same fidelity
level as `velocity_gain_tuning/envs/velocity_transport_env.py`, appropriate because real
`speedL` is resolved to joint velocities by the robot firmware's own Jacobian-based IK, not by
rigid-body dynamics), at the `hanging_alpha_0_5` pose (`velocity_gain_tuning/poses.py`, a known
large-safe-range scenario). Three passes:
1. `min_jerk_move_hold` profile, dx/move_duration_s swept from calm to extreme (commanded
   peak position-target speed up to 17 m/s), clamp REMOVED (`max_lin_speed_mps=1000.0`).
2. Same sweep, clamp at the REAL configured default (0.25 m/s).
3. `step` profile (instantaneous target jump -- the most aggressive possible input), dx swept
   toward the UR5e's reach limit, clamp REMOVED -- isolates the joint-space P-follower's own
   peak per-step velocity ceiling from how fast the target itself moves.

## Results

**Pass 1 (unclamped, min-jerk):** achieved EE speed plateaus at ~0.79-0.88 m/s regardless of
how aggressively the position target itself moves (commanded peak target speed from 0.75 up to
17.28 m/s made no difference past a point) -- a controller-dynamics saturation, not a guard
trip. Confirms the target-velocity value is structurally irrelevant to this mode, exactly as
the mechanism correction above predicts.

**Pass 2 (clamped at 0.25 m/s, min-jerk):** achieved speed never exceeds 0.25 m/s once dx is
large enough for the clamp to bind (dx=0.2m cases all saturate at exactly 0.2500 m/s); for
small dx (0.06m) the clamp never engages since the controller's own dynamics already stay
under it -- both passes agree exactly there, a real consistency check that the harness is
correct.

**Pass 3 (step profile, unclamped) -- the real ceiling test:** orientation error is the FIRST
guard to bind, not joint velocity or drift. Bisected precisely: the orientation-guard boundary
(0.25 rad, the same threshold used throughout this session) sits at **dx ~= 0.1906 m**, where
peak achieved EE speed is **0.7829 m/s** and peak `|qd|` is only 1.39 rad/s (well under the 3.0
rad/s joint-velocity guard -- confirms orientation, not joint speed, is what actually limits
this controller). Past dx~0.2m the orientation guard trips almost immediately (marginal
overshoot, 0.2501-0.2549 rad) up through dx=0.4m; at dx>=0.5m the joint-velocity guard also
trips, but on the very first control cycle (demanded instantaneous joint rates of 6-25 rad/s)
-- i.e. that guard would matter only past a displacement the orientation guard has already
ruled out.

## Answer to "can it move as fast as the swing-up needs (~5.42 m/s)?"

No, at every tested configuration, empirically:

| Configuration | Real achievable EE speed ceiling | Shortfall vs 5.42 m/s |
|---|---|---|
| Software clamp at real configured default (0.25 m/s) | 0.25 m/s | ~21.7x short |
| Software clamp REMOVED entirely (best case, hypothetical) | 0.78 m/s (orientation-guard bound) | ~6.9x short |
| Real hardware default (`max_tcp_speed_mps=0.05`) | 0.05 m/s | ~108x short |

Even in the most permissive configuration tested -- the software speed cap disabled
completely, which is not a real deployable state, just an isolation test -- the controller's
own position-tracking dynamics and the orientation safety guard together cap real achievable
speed at under 0.8 m/s, still roughly 7x short of the swing-up requirement. This empirically
confirms (not just re-derives from config values) the earlier feasibility verdict: a
single-impulse flip is not achievable with this controller architecture at any tested setting.

## Answer to "how much orientation loss can we handle before infeasible?"

The threshold is sharp, not gradual: orientation error stays comfortably low (0.12 rad at
dx=0.1m, ~48% of the guard) up to dx~0.19m, then crosses the existing 0.25 rad guard almost
exactly at dx~0.1906m and stays pinned near that boundary for any larger step (the guard trips
before error can grow much past it). There is no meaningfully wide "degraded but usable" band
between comfortable and infeasible at this pose -- the practical envelope is bounded by
essentially the same 0.25 rad threshold already used everywhere else in this session
(`max_orientation_error_rad`), not a new number this test discovered. The corresponding speed
at that boundary (0.78 m/s) is therefore also the real practical speed ceiling.

## Follow-up: does `orientation_priority` extend the ceiling?

Added later the same date, at the user's request ("how to make the controller move faster").
`CartesianVelocityConfig.orientation_priority` (default off, `controller_core/
cartesian_velocity_controller/config.py`, full evidence in `docs/status/
task_priority_orientation_hanging_2026-08-06.md`) was already validated on the standard
128-cell grid to drive orientation error to exactly 0.0 "for free" within the reachable
workspace, at `hanging_alpha_0_5` specifically for +X moves (it does NOT help −X past
~−0.296m, a separate solver-convergence limit, or −0.370m, which is provably unreachable).

Re-ran this test's own +X step-profile bisection with `orientation_priority=True`: orientation
error stays at exactly 0.0000 throughout (confirms the existing evidence, on this exact
harness). The binding constraint shifts entirely to `joint_velocity_guard` (3.0 rad/s), and the
ceiling moves from **dx=0.1906m / 0.783 m/s (orientation-guard bound)** to **dx=0.342m /
1.492 m/s (joint-velocity-guard bound)** — roughly a **1.9x** improvement in achievable +X peak
speed at this pose, using an already-built, already-tested, currently-default-off mechanism.
Still ~3.6x short of the 5.42 m/s swing-up requirement, and this is the +X direction only (the
mechanism's own −X limitations, documented separately, mean this improvement is asymmetric —
consistent with this session's standing "always sweep both +X and -X" rule).

Not tested here: whether raising `max_lin_speed_mps`/`ik_joint_gain` on TOP of
`orientation_priority` pushes the new joint-velocity-guard-bound ceiling any further, or
whether the real hardware software clamp (currently 0.25 m/s, override available via
`--max-tcp-speed-mps`, see this repo's real-hardware CLIs) would need raising too to realize
this in deployment. `orientation_priority` remains default OFF and unpromoted, per the original
doc's own "promotion to default is a human decision" note -- not changed here.

## Files

| path | what |
|---|---|
| `tools/diagnostics/velocity_controller_speed_ceiling_test.py` | the 3-pass empirical sweep + bisection script |
| this file | write-up |

**Tests run:** manual execution of the script's 3 passes (visually inspected output above) plus
an ad hoc bisection (not committed as a separate script, run inline). No pytest coverage added
-- this is a diagnostic script, matching the precedent of every other `tools/diagnostics/*`
prototype this session (e.g. `mpc_feasibility_prototype.py`, the Kalman-filter prototypes), not
a new module/package.
**Tests not run:** none applicable -- no existing file was modified, no regression surface.

**Rollback:**
```bash
rm -f tools/diagnostics/velocity_controller_speed_ceiling_test.py \
      docs/status/velocity_controller_speed_ceiling_2026-08-09.md
```
Nothing else touched. Nothing committed.
