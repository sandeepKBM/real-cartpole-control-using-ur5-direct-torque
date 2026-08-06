# Maximizing X-frame distance and speed: the pose is the whole game — 2026-08-06

**Objective:** maximize X-frame travel distance and X-frame speed (velocity/`speedL` lane).
**Sim-only (kinematic), no hardware validation.**

## Result

| Metric | Current default pose (-40°, wrist_2=0.2) | **HANGING_ORIGIN_Q** | Gain |
|---|---|---|---|
| Max clean X distance | **0.04 m** | **0.18 m** | **4.5x** |
| Absolute X wall | ~0.04 m | ~0.22 m | 5.5x |
| Max clean X speed | ~0.45 m/s | **~1.70 m/s** | **3.8x** |
| Peak abs qd at dx=0.10 | 3.09 rad/s (trips guard) | **0.13 rad/s** | 23x headroom |

No control-law change, no gain change, no guard weakened. Only the start pose differs.

## How the old limit was actually characterized

Earlier sweeps fixed `move_duration=1.0 s` and concluded dx>=0.05 m "fails on the
joint-velocity guard". Two follow-ups showed that framing was wrong:

1. **Longer moves do not help.** At the old pose, peak `|qd|` stays ~3.0-3.3 rad/s at
   move durations 1.0 / 2.0 / 4.0 s, and at 4.0 s / dx=0.05 it is *worse* (9.83 rad/s). So the
   spike is not commanded Cartesian speed — it is `pinv(J)` amplification as the configuration
   drifts toward the wrist_2 singularity.
2. **The limit is a fixed distance, not a fixed dx.** Commanding dx = 0.05 / 0.10 / 0.20 / 0.30 m
   from the old pose, the arm trips after achieving **0.0386 / 0.0396 / 0.0384 / 0.0362 m** — the
   same ~4 cm of real travel every time, at trip times 1.33 / 0.90 / 0.65 / 0.54 s. It is a hard
   **kinematic reachability wall of that pose**, invariant to commanded distance and duration.
   No amount of control tuning moves it.

## Why the hanging family wins

`HANGING_ORIGIN_Q` (`hardware/poses.py`) holds `wrist_2` at **+pi/2** by construction, so pure-X
translation never drives it toward the `wrist_2=0` singularity. The measured consequence is the
23x joint-velocity headroom above, which is what buys BOTH more distance and more speed.

**Note the cond(J) hypothesis was refuted, not confirmed.** A separate sweep measured max cond(J)
as non-monotone and a poor single predictor (worst conditioning at dx=0.045 m still failed on
*orientation*, while higher-dx runs had lower cond(J) and failed on *joint velocity*). Start-pose
cond(J) improves only 29.2 -> 24.7 across a wrist_2 bias sweep that barely moved the envelope. The
mechanism is singularity *proximity along the traversed path*, not a scalar conditioning number
at the start.

## Speed detail (HANGING_ORIGIN_Q, dx=0.10 m)

| move duration | approx peak v | peak abs qd | outcome |
|---|---|---|---|
| 0.50 s | 0.38 m/s | 0.53 | complete |
| 0.25 s | 0.75 m/s | 1.14 | complete |
| 0.15 s | 1.25 m/s | 1.98 | complete |
| 0.11 s | **1.70 m/s** | 2.72 | **complete** |
| 0.10 s | 1.88 m/s | 3.03 | joint_velocity_guard |

Requires raising the `max_lin_speed_mps` **command clamp** (default 0.25 m/s) — that is a command
limit, not a safety guard. `max_joint_velocity_radps` stayed 3.0 and `max_orientation_error_rad`
stayed 0.25 throughout; the ceiling above is the real guard binding, not a relaxed one.

## What binds now (the next problem, not this one)

Distance is no longer joint-velocity limited — it is limited by the **orientation guard**
(0.25 rad), which grows with dx: 0.141 @ 0.12 m, 0.182 @ 0.15 m, 0.226 @ 0.18 m, 0.250 @ 0.20 m.

**`kp_posture` is ruled out as a lever for it**: swept 1.0 / 3.0 / 5.0 / 10.0 at dx = 0.20 / 0.25 /
0.30, the achieved distance is identical to 4 decimal places (0.2005, 0.2202) at every gain. The
rx/ry drift that `reduced_task_dims` leaves to the null space is not something the posture term
can fight at this pose.

## What I could not verify

- No real-hardware validation of any of this; the sim is kinematic-only with a plain
  Moore-Penrose pseudo-inverse, not the damped singularity-robust IK real UR firmware likely uses.
- Only +X was swept here; -X untested at the hanging poses.
- `HANGING_ORIGIN_Q` places the TCP at a genuinely different location/attitude than the current
  default. Reported, not judged — whether that pose is acceptable for the task is a separate call.
- The ~0.22 m wall at the hanging pose was bracketed, not bisected to a tight threshold.
