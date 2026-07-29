# Investigation: is `height_alpha=0.1` near a kinematic singularity, and does alpha=0.2/0.3 fix the reported real-hardware velocity overshoot?

Context: on 2026-07-28, live real-hardware testing at `height_alpha=0.1`
(`hardware/poses.py::q_for_height_alpha`) with `shoulder_pan` overridden to
`-0.7853981633974483` rad and `wrist_2` overridden to `0.1` rad reportedly
produced real TCP peak velocities up to ~2.7x the planned min-jerk peak for
`direct_torque`-mode moves. Those specific alpha=0.1 trials were pasted into
chat, not saved to `hardware_captures/` or anywhere else in this repo — this
investigation could not independently corroborate the reported ratios against
any saved artifact (the same caveat this project's own
`docs/status/clock_timing_late_cycles_2026-07-28.md` raised about a
differently-described claim from the same day: "the specific ... run
described ... does not exist anywhere in `hardware_captures/`, in git
history, or in any tool that was run that day"). Everything below is
independently reproducible simulation evidence, not a re-verification of the
real-hardware numbers.

## Verdict

**Is alpha=0.1 near a real, identifiable singularity? Yes — but it's the elbow/full-extension
singularity, not a wrist or shoulder singularity, and by alpha=0.1 the pose has already moved
substantially off it.** Confidence: high (numerically exact SVD evidence, see below).

**Does alpha=0.2 or alpha=0.3 fix the reported real-hardware overshoot in simulation? No —
and more importantly, the overshoot doesn't reproduce in simulation at alpha=0.1 either.**
Confidence: high for "doesn't reproduce in this sim pipeline at any of the three alphas
tested"; the question of *why* the real hardware showed overshoot remains genuinely open —
this investigation can rule out "the OSC controller's own physics, run at this exact pose
through this project's validated production sim, produces the reported overshoot" but cannot
positively identify the real cause from simulation alone.

Both real numbers this task's premise and one existing status doc lean on (the alpha=0.1
overshoot ratios, and a claimed "4/5 cycles late" timing anomaly in
`docs/status/timing_safety_gaps_audit_2026-07-28.md`) are narrative claims from the same
session that this project's own precedent (`clock_timing_late_cycles_2026-07-28.md`) shows
can fail to match saved data. Treat the timing explanation as a plausible, previously-flagged
candidate — not a confirmed one.

## Evidence

### 1. Kinematic conditioning across alpha

Computed via `simulation.ur5e_mujoco_torque.load_model` / `build_mujoco_state`
(`simulation/ur5e_mujoco_torque.py:202`, `:254`, Jacobian assembled at
`simulation/ur5e_mujoco_torque.py:288`) — the same MuJoCo `mj_jacSite` call the production
controller and adapter use, not a reimplementation — at
`q = 0.9*ACTIVE_ORIGIN_Q + alpha_frac*LOWER_B_Q` (`hardware/poses.py:8-24`) with
`shoulder_pan=-0.7853981633974483` and `wrist_2=0.1` applied at every alpha, matching
today's exact tested pose at alpha=0.1:
`[-0.7853981633974483, -1.4237166941154069, -0.24, -1.4537166941154069, 0.1, 0.0]`.
`cond(J) = np.linalg.cond(J)` matches exactly how the production controller already computes
it (`controller_core/x_axis_cartesian_impedance.py:378`).

| alpha | cond(J) | manipulability sqrt(det(JJᵀ)) | min singular value | elbow (rad) | wrist_2 (rad) |
|---|---|---|---|---|---|
| 0.00 | 5.3e16 (numerically singular) | ~0 | 1.8e-16 | 0.000 | 0.100 |
| 0.01 | 1086 | 1e-6 | 0.00199 | -0.024 | 0.100 |
| 0.05 | 216 | 2.8e-5 | 0.00996 | -0.120 | 0.100 |
| 0.10 | **110** | 1.1e-4 | 0.0196 | -0.240 | 0.100 |
| 0.15 | 76 | 2.5e-4 | 0.0283 | -0.360 | 0.100 |
| 0.20 | **61** | 4.3e-4 | 0.0352 | -0.480 | 0.100 |
| 0.25 | 54 | 6.4e-4 | 0.0399 | -0.600 | 0.100 |
| 0.30 | **50** | 8.9e-4 | 0.0427 | -0.720 | 0.100 |
| 0.40 | 47.5 | 1.4e-3 | 0.0449 | -0.960 | 0.100 |
| 0.50 | 47.3 | 1.9e-3 | 0.0448 | -1.200 | 0.100 |

Full sweep (19 points, alpha 0.00-0.50) and raw SVDs: see the analysis script referenced
below.

**wrist_2 is pinned at exactly 0.1 rad at every single alpha** — both `ACTIVE_ORIGIN_Q[4]`
and `LOWER_B_Q[4]` are 0, so the interpolation itself never moves `wrist_2`, and the
`wrist_2=0.1` override then sets it identically regardless of alpha. The well-documented
wrist singularity (`wrist_2=0`, AGENTS.md §3) is therefore constant across this entire sweep
and cannot be what differentiates alpha=0.1 from alpha=0.2/0.3 — it's a fixed background
factor, not the variable one.

**Shoulder-singularity check**: horizontal distance from the shoulder-pan (base-z) axis to
the end-effector stays 0.233-0.259 m across the whole sweep (world frame; base body confirmed
at `[0,0,0]`), never approaching zero — rules out a shoulder singularity (wrist center on the
base rotation axis) as any part of the story here.

**Elbow/full-extension singularity, confirmed**: at alpha=0.0 the `elbow` joint is exactly 0
rad (`ACTIVE_ORIGIN_Q[2]=0`) — upper-arm and forearm links colinear — and the Jacobian's two
smallest singular values collapse to ~1.8e-16 / 4.1e-17, i.e. **numerically exact rank-4**,
independent of the wrist_2 override (which is already at 0.1, off the wrist singularity, at
this same point). As alpha increases the elbow bends away from full extension
(elbow: 0 → -0.024 → ... → -1.2 rad) and cond(J) collapses from 5e16 to 1086 (alpha=0.01) to
110 (alpha=0.10) to ~47-50 by alpha≥0.4, where it plateaus. This is the textbook elbow/reach
singularity, not wrist or shoulder.

**Is alpha=0.1 a real outlier relative to 0.2/0.3?** Yes, quantitatively: cond(J) at 0.10
(110) is ~1.8x that at 0.20 (61) and ~2.2x that at 0.30 (50); the min singular value at 0.10
(0.0196) is a little over half of 0.20's (0.0352) and under half of 0.30's (0.0427). Real,
substantial, monotonic — but nowhere near the alpha=0.0 case, and the curve is smooth/
monotonic across the whole range, not a sharp isolated spike at 0.1 specifically (alpha=0.05
is actually worse than 0.10). The user's premise that 0.1 is "worse than 0.2/0.3" is
correct; "spike specific to 0.1" is not — it's a monotonic tail of the alpha=0 elbow
singularity.

### 2. Move-hold rollout at alpha=0.1 / 0.2 / 0.3, `config/ur5e_mujoco_torque_osc_tuned.yaml`

Run via the production entrypoint (`--mode controller-rollout --controller-kind impedance`,
`--trajectory-profile min_jerk_move_hold`), `--start-q-rad` set to the exact pose above at
each alpha, `--transport-axis-index 0`, `--seed 0`, gravity_source/coriolis_feedforward as
set by the config (pinocchio / on). Peak velocity ratio = peak `|ee_lin_vel[x]|` (achieved) /
peak `|target_x_vel|` (planned min-jerk), both read straight from the trace the tool already
writes — no reimplemented trajectory math.

**dx=0.03 m** (move-duration 3.0s, 8.0s) — valid at all three alphas, `safety_pass: true`:

| alpha | move_dur | planned peak v (m/s) | achieved peak v (m/s) | ratio |
|---|---|---|---|---|
| 0.10 | 3.0 | 0.01875 | 0.01548 | 0.825 |
| 0.10 | 8.0 | 0.00703 | 0.00576 | 0.819 |
| 0.20 | 3.0 | 0.01875 | 0.01538 | 0.820 |
| 0.20 | 8.0 | 0.00703 | 0.00569 | 0.810 |
| 0.30 | 3.0 | 0.01875 | 0.01553 | 0.828 |
| 0.30 | 8.0 | 0.00703 | 0.00611 | 0.869 |

No overshoot at any alpha — consistently ~0.80-0.87x undershoot (the closed-loop tracking
slightly lags the planned min-jerk peak, matching the "closed-loop bandwidth limit" behavior
already documented for this gain set in AGENTS.md §3). No meaningful alpha-dependence.

**dx=0.10 m** (move-duration 2.5s, 4.5s, 8.0s) — **fails identically at all three alphas**,
all 9 runs `valid_move_and_hold: false`, `safety_pass: false`,
`termination_reason: "|Y-Y0| > 0.03 m"` (the config's own `max_abs_y_drift_m: 0.03`,
`config/ur5e_mujoco_torque_osc_tuned.yaml:109`), tripping partway through the move
(sim_time 1.0-3.4s depending on move duration):

| alpha | move_dur | planned peak v | achieved peak v (pre-trip) | ratio | outcome |
|---|---|---|---|---|---|
| 0.10 | 2.5 | 0.0726 | 0.0608 | 0.838 | Y-drift trip |
| 0.10 | 4.5 | 0.0401 | 0.0336 | 0.838 | Y-drift trip |
| 0.10 | 8.0 | 0.0225 | 0.0189 | 0.841 | Y-drift trip |
| 0.20 | 2.5 | 0.0727 | 0.0593 | 0.816 | Y-drift trip |
| 0.20 | 4.5 | 0.0402 | 0.0328 | 0.815 | Y-drift trip |
| 0.20 | 8.0 | 0.0226 | 0.0184 | 0.815 | Y-drift trip |
| 0.30 | 2.5 | 0.0734 | 0.0586 | 0.798 | Y-drift trip |
| 0.30 | 4.5 | 0.0406 | 0.0323 | 0.796 | Y-drift trip |
| 0.30 | 8.0 | 0.0228 | 0.0181 | 0.793 | Y-drift trip |

Two findings from this table:
- **No velocity overshoot reproduces at alpha=0.1 in simulation, for either move size** —
  every ratio is <1 (undershoot), the opposite of the ~1.4-2.7x reported from real hardware.
  Whatever produced the real overshoot, it is not reproduced by this project's own validated
  controller physics run noise-free in sim at the exact reported pose and config.
- **alpha=0.2/0.3 does not fix the one real failure mode this sim does show** — the 0.10m
  move Y-drift trip is present, at essentially the same severity, at all three alphas. Better
  Jacobian conditioning did not help here. This gain set (`osc_tuned`) has only ever been
  validated at height_alpha=0.0 and 0.5 (AGENTS.md §3); this alpha 0.1-0.3 band with these
  pose overrides is genuinely outside its validated envelope, and a 0.10m move at this depth
  is not currently safe to expect to pass regardless of alpha in this range.

## Recommendation for tomorrow's lab session

- **Do not expect alpha=0.2 or alpha=0.3 to resolve the reported overshoot** — this
  investigation found no simulation evidence that better Jacobian conditioning fixes it, and
  no simulation evidence that conditioning caused it in the first place at alpha=0.1.
- If testing proceeds at this pose family anyway, **0.03m moves are the only size validated
  by this investigation** (clean tracking, `safety_pass: true`, at alpha=0.1, 0.2, and 0.3
  identically — no alpha in this set shows an advantage for that move size). Suggested pose
  if you want the mildest available conditioning within this family:
  alpha=0.3, `--start-q-rad -0.7853981633974483 -1.1295574287564276 -0.72 -1.2195574287564277 0.1 0.0`.
- **Do not attempt a 0.10m move at alpha=0.1, 0.2, or 0.3 with `config/ur5e_mujoco_torque_osc_tuned.yaml`** —
  simulation shows all nine tested combinations trip the Y-drift guard before completing the
  move, independent of alpha. This is a real, unresolved gap in this gain set's validated
  envelope for this pose region, not something alpha selection fixes.
- **The overshoot itself is still unexplained.** The most credible documented alternative
  candidate is real hardware/timing behavior specific to the `direct_torque` control loop
  (`docs/status/timing_safety_gaps_audit_2026-07-28.md` describes a genuine, code-verified
  `DeadlineMonitor` calibration gap for the 500 Hz loop — a flat 3.0ms deadline against a 2ms
  nominal period, unable to catch a 4/5-late-cycles pattern) — but that doc's own headline
  claim of "today a real hardware run ... hit a genuine overrun" is not independently
  grounded in any saved artifact either, by the same standard this doc holds the alpha=0.1
  overshoot claim to. Recommend capturing real, saved (not pasted-into-chat) `direct_torque`
  timing data (`summary.json`'s `timing` block, already computed by
  `hardware/timing.py::TimingTracker`) alongside any repeat of the alpha=0.1 move tomorrow,
  so the next investigation has real artifacts to work from instead of two more unverifiable
  narrative claims.

## Reproducing this analysis

Jacobian/conditioning sweep and move-hold validation were done with two small scripts that
import `simulation.ur5e_mujoco_torque.{load_model,build_mujoco_state}` and
`hardware.poses.q_for_height_alpha` directly (no reimplemented FK/Jacobian/trajectory math)
and shell out to `tools/ur5e_mujoco_torque_experiments.py` for the move-hold runs; they were
kept in a scratch directory rather than committed, per this task's "do not modify any other
files" scope. The exact commands, e.g. for the alpha=0.1 baseline dx=0.03m/move=3.0s case:

```
python tools/ur5e_mujoco_torque_experiments.py --mode controller-rollout --controller-kind impedance \
  --config config/ur5e_mujoco_torque_osc_tuned.yaml \
  --start-q-rad -0.7853981633974483 -1.423716694115407 -0.24 -1.453716694115407 0.1 0.0 \
  --target-x-delta 0.03 --trajectory-profile min_jerk_move_hold --move-duration 3.0 --duration 5.0 \
  --transport-axis-index 0 --seed 0 --no-plot
```

reproduce any alpha by substituting the `q_for_height_alpha(alpha)` vector with the same two
overrides (`shoulder_pan=-0.7853981633974483`, `wrist_2=0.1`) applied.
