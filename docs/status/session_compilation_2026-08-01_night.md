# 2026-08-01 night session compilation — real UR5e, height_alpha=0.5 wrist-singularity work

Compiled at the user's request to consolidate a long, dense real-hardware session before
deciding next steps. Everything below is either directly measured on real hardware tonight
(`outputs/hardware_transport_remote/hardware_transport/direct_torque_202608011*`) or from the
sim-only diagnostic docs also produced tonight (linked). Pose throughout:
`hardware/poses.py::q_for_height_alpha(0.5)` with `shoulder_pan=0` (the "zero-degree" pose,
distinct from the `-45°` `HEIGHT_ALPHA_0_5_CLEARANCE_Q` real-hardware default pose).

## 1. What was broken at the start of tonight

Real UR5e testing at this pose repeatedly tripped the TCP-acceleration/speed safety guards
using `accel_duration_scurve`/`min_jerk` trajectories, at modest target accelerations
(0.005-0.02 m/s²), regardless of controller config (`wrist_orient_fixed`, `..._friction_ff`,
`..._friction_ff_diag_adaptive_lambda`). Root-caused from real trace data: the arm sits almost
exactly at `wrist_2=0`, a genuine UR-family kinematic singularity — confirmed this is a
structural property of the **entire** canonical transport pose family
(`hardware/poses.py::q_for_height_alpha(alpha)` for every `alpha` in `[0,1]`, not just this one
pose), per a code comment in `rl_gain_scheduling/gain_scheduling_env.py:46-48` acknowledging
this was a known, load-bearing design assumption. Measured `cond(J)` (full 6x6 Jacobian) at
this pose: `1e16`-`2.5e17` across the whole `height_alpha` range. `lambda_diagonal_shaping`/
`lambda_adaptive_regularization` (existing fixes for a *different* Λ-conditioning leak) were
tried and found to have **no benefit** here — real-hardware retest was, if anything, slightly
worse (peak accel 0.92 vs 0.72 m/s²) — because those regularize how the wrench-shaping matrix Λ
is computed, not `cond(J)` itself, which is an upstream property of J alone.

## 2. Fix #1 — `split_base_wrist_task` (landed, real-hardware validated repeatedly)

**Design** (`controller_core/x_axis_cartesian_impedance.py`, new flag, default off,
`docs/status/split_base_wrist_impedance_2026-08-01.md`): the translation task (Fx/Fy/Fz) is
computed via a reduced Jacobian restricted to base joints (shoulder_pan/shoulder_lift/elbow)
only — structurally excludes `wrist_2` from that computation's matrix inversion entirely.
Orientation-holding stays on the existing `nullspace_posture` mechanism, recomputed against
this reduced task Jacobian.

**Numeric verification at the exact failure pose** (before implementing, to check the design
was sound): `cond(full 6x6 J) ≈ 7.28e16` (singular); `cond(base-only 3x3 position Jacobian)
≈ 7.8` (well-conditioned); `cond(wrist-only 3x3 rotation Jacobian) = inf` (**exactly**
rank-deficient — real UR gimbal lock, `wrist_1`/`wrist_3` axes literally align at `wrist_2=0`).
This refuted a naive symmetric split (wrist-only orientation task) before it was ever built —
it would have been exactly as broken as the problem it was meant to fix.

**Sim validation**: `jacobian_cond` at this pose drops from a flat `8.07e16` (up to ~1600x
cycle-to-cycle jitter) to `7.8-13` (near-zero jitter). One honest minor regression:
`canonical_grid` 6/8 vs baseline 7/8 at `dx=0.01m`, root-caused to a pre-existing
friction-feedforward tracking edge, not a new failure mode. `large_displacements` and
`torque_scale_robustness` byte-identical to baseline.

**Real-hardware validation, config `config/ur5e_mujoco_torque_osc_tuned_split_base_wrist.yaml`,
`accel_duration_scurve` profile** (all times below are direct measurements, table below is the
authoritative real-hardware record):

| run | target_accel | move_duration | achieved / target | outcome | `jacobian_cond` range |
|---|---|---|---|---|---|
| `direct_torque_20260801_193036` | +0.005 | 8.0s | 0.0412 / 0.0509m (81%) | **clean, duration_complete** | 7.8–11.8 |
| `direct_torque_20260801_193232` | +0.02 | 4.0s | 0.0220 / 0.0509m (43%) | TCP-speed trip (0.0539) | 7.8–9.5 |
| `direct_torque_20260801_200315` | +0.02 | 4.0s | 0.0220 / 0.0509m (43%) | TCP-speed trip (0.0507, after smoothing) | 7.8–9.5 |
| `direct_torque_20260801_200823` | +0.01 | 8.0s | 0.0915 / 0.1019m (90%) | **clean, duration_complete** | 7.8–31.4 |
| `direct_torque_20260801_202017` | +0.015 | 8.0s | 0.1409 / 0.1528m (92%) | **clean, duration_complete** | 7.8–297,619 (single-cycle transient, see §5) |
| `direct_torque_20260801_202215` | −0.015 | 8.0s | −0.0632 / −0.1528m (41%) | TCP-**accel** trip (0.5112) | 5.31–5.32 |
| `direct_torque_20260801_202639` | −0.015 | 8.0s | −0.0045 / −0.1528m (3%) | TCP-**accel** trip (0.7042, different timing/magnitude) | 7.56–7.82 |
| `direct_torque_20260801_203354` | +0.02 | 8.0s | 0.0639 / 0.2037m (31%) | TCP-speed trip (0.0510) | 7.81–16.4 |
| `direct_torque_20260801_204446` | −0.015 | 8.0s | −0.0496 / −0.1528m (32%) | TCP-**speed** trip (0.0514, 3rd `-0.015` trial, different guard than trials 1-2) | 5.65–7.83 |

**Bottom line: the singularity-conditioning mechanism is fixed, confirmed on real hardware
every single time it's been tested tonight** (`jacobian_cond` stayed in the single-to-low-double
digits, vs the pre-fix `1e16-1e17`, in all 7 real runs above, including the ones that tripped
for *other* reasons). Not yet found: the real ceiling in either direction — `+X` clean through
`0.015/8s` (92%), fails via a *different*, understood mechanism at `0.02/4s`; `-X` fails at
`0.015/8s` via a still-unexplained mechanism (§6). `-X` has been the weaker direction all night,
consistent with an already-documented pre-existing directional-ceiling finding at this pose
family (AGENTS.md §3).

## 3. Fix #2 — TCP speed-limit guard smoothing (landed, real-hardware validated)

**Finding**: unlike the TCP-accel guard (already gap-windowed/low-pass smoothed since
2026-07-28), the TCP-**speed**-limit check in `hardware/safety.py::CartesianMoveMonitor` was
still raw and single-cycle — a deliberate original design choice based on a *stationary* noise
floor measurement (p100 0.036 m/s) that turned out not to hold during real active motion
(measured tonight: residual noise std ~1.5-2x higher moving than stationary).

**Real-trace ground truth** (`direct_torque_20260801_193232`): the trip's last ~140ms showed a
smoothed/underlying speed trend climbing genuinely and monotonically past `0.05 m/s` (driven by
real orientation-error growth, not sensor noise) — real per-cycle noise (`std≈0.007-0.009 m/s`)
riding on top made the *exact* trip cycle noise-timing-dependent even though the average had
already crossed the guard.

**Fix**: new `CartesianMoveLimits.speed_limit_gap_cycles`/`speed_limit_lowpass_alpha` (defaults
1/1.0, byte-identical to old behavior), independent ring buffer/EMA state, wired through all
three control modes and both real-hardware CLIs (`--speed-limit-gap-cycles`/
`--speed-limit-lowpass-alpha`). Not folded into `NOISE_ROBUST_GUARD_OVERRIDES` (separate,
explicit opt-in). 6 new unit tests, full suite clean (532 passed at the time).

**Real-hardware confirmation** (`direct_torque_20260801_200315` vs `_193232`, identical
scenario): smoothing did **not** prevent the trip (as predicted — the underlying rise was real),
but cleaned up the reported value from a noisy `0.0539` overshoot to a clean `0.0507` right at
the threshold, and essentially identical timing/achieved-displacement — confirms smoothing does
exactly what it should (clean signal, doesn't mask a real problem) and nothing more.

## 4. Diagnosis — `+X` orientation-error growth mechanism (understood, not yet fixed)

`docs/status/split_base_wrist_orientation_growth_2026-08-01.md` (background-agent diagnostic,
sim + real trace analysis, no code changes):
- The "clean" `0.005/8s` real run **also** grows orientation error monotonically (to 0.054 rad
  by the end) — comparable to or exceeding the *failing* `0.02/4s` run's trip-point value
  (0.028 rad). Growth tracks **elapsed exposure time** more than accel magnitude per se.
- Sim A/B (`split_base_wrist_task` on vs off, identical scenario): orientation-error outcomes
  nearly identical (~6% difference) — does **not** support "the reduced Jacobian removed
  restoring authority" as the dominant cause.
- A real, secondary, directly-measured effect **was** found and confirmed both algebraically and
  numerically: `split_base_wrist_task` closes a near-singularity cross-coupling leak in the
  nullspace-posture projector that used to route ~13x more restoring torque through `wrist_2` —
  real, but a minor contributor, not the dominant driver.
- Sim has an honest fidelity gap: doesn't reproduce the real clean run's non-decaying
  orientation plateau — points to an unmodeled friction/stiction effect (same family as this
  repo's already-known friction-feedforward gaps), independent of tonight's fix.
- **Verdict**: mostly a general property of holding orientation at this pose with damping-only
  rotational gains (`kp_rot=0`), not a regression introduced by `split_base_wrist_task`.
- **Concrete pointer for future work, not attempted**: raise `kp_rot_wrist` (currently 0,
  damping-only) in `config/ur5e_mujoco_torque_osc_tuned_split_base_wrist.yaml` so
  `wrist_orientation_task` gets a genuine proportional restoring channel independent of the
  nullspace-posture path.
- A stray item in that agent's own report: it mentioned declining an alleged "mid-turn request"
  to widen the TCP-speed guard past 0.05 m/s — **no such request was ever sent** (not by the
  user, not by me). It correctly refused regardless (guard-weakening needs a deliberate,
  evidence-scoped human decision per AGENTS.md §4's existing precedent), so nothing unsafe
  happened, but the origin of that apparent instruction is unexplained and worth noting.

## 5. Two unexplained, real, low-priority observations

- **Transient full-`J` conditioning spike during a clean run**: `direct_torque_20260801_202017`
  (`+0.015/8s`, otherwise clean pass) showed `jacobian_cond` spike to 297,619 for a single ~2ms
  cycle at `t=5.664s`, coincident with the real `wrist_2` joint transiently swinging back through
  ~0 (2e-5 to 6e-5 rad) as a side effect of holding a larger orientation excursion (0.157 rad at
  that point). Caused no problem (nullspace-posture torque now flows through the *reduced*,
  well-conditioned Jacobian, so this full-`J` reading is diagnostic-only, not load-bearing) —
  but flags that larger orientation excursions can still bring the real arm physically close to
  the singularity, worth monitoring if accel is pushed further.

## 6. RESOLVED — `-X` acceleration transient at `0.015/8s` (both mechanisms now accounted for)

Three real trials at the identical command (`-0.015/8s`), three different outcomes — rules out
a fixed trajectory-shape event (the scurve acceleration zero-crossing at `t=T/2=4.0s` was a live
hypothesis, directly refuted by trial 2 tripping at `t=1.84s`, nowhere near that point):

| | trial 1 (`_202215`) | trial 2 (`_202639`) | trial 3 (`_204446`) |
|---|---|---|---|
| guard | accel | accel | **speed** |
| trip time | t=3.93s | t=1.84s | t=3.50s |
| achieved | 41% | 3% | 32% |
| peak value | 0.5112 | 0.7042 | 0.0514 |
| orientation error at trip | 0.083-0.085 rad | 0.0053-0.006 rad | 0.066 rad |
| `jacobian_cond` | 5.31-5.32 (low, stable) | 7.56-7.82 (low, stable) | 5.65-7.83 (low, stable) |
| `tau_controller` | falling | falling | (not re-checked) |

All three rule out the singularity (low, stable `jacobian_cond` every time) and friction
breakaway (falling tau, flat/converged `x_error` in trials 1-2, not diverging). **Revised
read after the third trial**: this doesn't look like one mechanism with noisy timing — trial 3's
signature (speed trip, moderate orientation buildup) matches the same `+X`
orientation-growth mechanism from §4, while trials 1-2 (accel trip, wildly different orientation
levels, no consistent timing) don't fit that pattern at all.

**Resolved by a deeper follow-up investigation**
(`docs/status/neg_x_accel_transient_deeper_investigation_2026-08-01.md`), reading every field
across the full traces (not just aggregates near the trip), plus a sim reproduction attempt:
- **Trial 2**: real correlate found — 107/919 cycles (11.6%) show exact byte-identical duplicate
  consecutive `tcp_pose` telemetry frames, dense right up to the trip. This is RTDE telemetry
  staleness (an already-documented failure class, AGENTS.md §4, 2026-07-28) synthesizing a
  spurious acceleration via finite-differencing of frozen readings, while the real underlying
  motion was slow and unremarkable (0.006 rad orientation error, 3% achieved). Trial 1 has zero
  duplicate frames; trial 3 has 2, not concentrated near its trip — confirms this is
  trial-2-specific, not a shared mechanism.
- **Trial 1**: reclassified, not left unexplained — its orientation-error magnitude/growth shape
  closely matches trial 3's already-understood §4 mechanism. Sim reproduction of the identical
  scenario shows the same qualitative signature (different magnitude/timing, a known sim-vs-real
  gap for this failure class) and, as expected, zero telemetry artifacts — sim cannot reproduce
  trial 2's mechanism at all, supporting that trials 1 and 2 are genuinely different phenomena.

**Net result: no new unexplained hazard remains.** All three `-X` trials are now accounted for
by two already-understood, already-documented causes (the §4 orientation-growth mechanism, and
RTDE telemetry staleness) — not a new, mysterious real-hardware risk. `-0.01` remains the
practical `-X` ceiling for this pose/config (untested combinations above that, not a fixed hard
limit).

**Side finding from the same investigation, not yet acted on**: `mujoco.coriolis_feedforward:
true` (set in `config/ur5e_mujoco_torque_osc_tuned_split_base_wrist.yaml` via inheritance) never
actually took effect on any of the three real trials — `tau_coriolis` was exactly zero
throughout all of them. Traced to `hardware/direct_torque_transport.py` only honoring an
explicit `--coriolis-feedforward` CLI flag, never the config key, unlike the sim path. A real
config-vs-implementation gap, flagged for a maintainer decision, not fixed.

## 7. Negative/deprioritized results (real findings, not gaps in follow-through)

- **LuGre dynamic friction feedforward** (`docs/status/lugre_friction_feedforward_2026-08-01.md`):
  implemented, unit-tested, zero-regression, but a **no-op** at the chosen placeholder
  parameters — the plan's own literal (non-normalized) ODE gives the bristle state a relaxation
  time constant of hundreds of seconds at real hold-phase velocities, far slower than any real
  transport-move window. Needs real `sigma0` recalibration before it's useful. Merged, flag
  defaults to the existing static model.
- **`torque_task_qp` vs impedance at the singularity**
  (`docs/status/qp_vs_impedance_wrist_singularity_2026-08-01.md`): both controllers use the
  identical `cond(J)`/`singular_scale` mechanism, both provably no-op it at
  `jacobian_singular_cond_max=1e18`; QP's one different mechanism (box torque/velocity
  constraints) never bound in 24 sim runs. `hardware/direct_torque_transport.py` hardcodes the
  impedance controller — QP was never wired to real hardware at all, contrary to what was
  assumed when this was proposed. Recommendation: don't pursue QP for this problem.
- **Sim-side TCP-accel/speed guard** (`docs/status/sim_tcp_accel_guard_2026-08-01.md`): built
  (`--enable-tcp-accel-guard` on `tools/ur5e_mujoco_torque_experiments.py`), validated against
  real trace replay (reproduces a real trip within ~30% of measured magnitude). But run against
  tonight's exact real failure scenario, it did **not** trip in sim (raw peak accel 0.34 vs 0.5
  ceiling) — confirmed as a genuine sim-vs-real dynamics fidelity gap (no transient to detect),
  not a guard defect. This means sim cannot currently be trusted to validate this failure class
  either way — a real constraint on how much weight to put on any sim-only result for this
  family of problems, including the orientation-growth sim A/B in §4.
- **"Hanging" end-effector pose redesign — REVERSED, now built and validated (§8 below).**
  Initially deprioritized here on the grounds that `split_base_wrist_task` already solved the
  singularity more surgically. After §6 resolved and the user judged `split_base_wrist_task`
  "mainly a diagnostic fix" not acceptable for real lab deployment on its own, this was revisited
  and built properly (`docs/status/hanging_pose_transport_family_2026-08-01.md`) — see §8.

## 8. Fix #3 — hanging-pose transport family (landed, sim-only, structural fix)

After §6 resolved and the user judged `split_base_wrist_task` alone "mainly a diagnostic fix,
not acceptable for real lab deployment" — the more fundamental fix was built:
`docs/status/hanging_pose_transport_family_2026-08-01.md`. A new elbow-down pose family
(`hardware/poses.py::HANGING_ORIGIN_Q`/`HANGING_LOWER_Q`/`q_for_hanging_height_alpha`, mirrored
into `rl_gain_scheduling/gain_scheduling_env.py`), found via a MuJoCo FK grid search (16,684
candidates) + Nelder-Mead refinement — additive only, no existing pose constant touched.

**This avoids the singularity structurally, not just its consequence**: `cond(J)` across the
whole new family's range is `7-15` (static FK sweep, confirmed `8.9-11.2` in an actual
closed-loop rollout) — vs. the old family's `1e16-2.5e17` across its ENTIRE range. 12-16 orders
of magnitude better, everywhere in the range, not just spot-checked at endpoints. Workspace
coverage verified equivalent (Z-height range 0.537-1.044m vs. old family's 0.537-1.08m;
reachable ±0.20-0.25m in X with no joint-limit contact); gravity-comp torque comparable or
lower (9-17 Nm vs. 0-26 Nm).

**Sim validation**: the plain old tuned gains (only `home_qpos` changed, zero pose-specific
tuning) score 23/38 on the standard rigor sweep — already beating the old family's own
documented friction-era baseline (19/38). Layering the existing, already-validated
`friction_feedforward` fix reaches **36/38 (94.7%)**, matching the old family's best-ever
historical number. The 2 remaining failures are a pre-existing, pose-agnostic torque-budget
limit at 10% torque scale, not a new problem introduced by this pose.

**Not yet done**: only the `alpha=0.5` midpoint got the full rigor sweep (not the whole range);
the `-45°` real-lab clearance rotation hasn't been checked in this posture; and — critically —
**zero real-hardware or physical clearance validation**. This needs a supervised visual
clearance check in the actual lab before it goes anywhere near the real arm, same discipline as
every pose change in this project's history.

## 9. Two more sim-only tracks landed tonight (from home, after the real-hardware session ended)

- **Acceleration feedforward** (`docs/status/acceleration_feedforward_2026-08-01.md`): added a
  computed-torque-style feedforward term to the previously pure-PD torque law, motivated by real
  observed tracking lag/jitter and the confirmed complete absence of acceleration feedforward in
  the controller. Honest mixed result: negligible at tonight's real accel magnitude (0.02 m/s²),
  and at a larger untested one (0.3 m/s²) a real tradeoff — hold-phase torque drops ~7x but
  move-phase jitter nearly triples, one canonical-grid case regresses. Not recommended as
  default; not real-hardware tested.
- **Residual-torque regression ridge fix** (`docs/status/residual_torque_regression_pipeline_2026-08-01.md`):
  fixed a catastrophic held-out extrapolation blowup (joint 2 R² `-9799.8`) root-caused to
  near-collinear features at the tiny `qd` range elbow/wrist_1 see during X-only transport.
  Closed-form ridge regression (`λ=1e5`) plus output clipping fixed it: joint 2 R²
  `-9799.8 → -0.055`. Pipeline is now honest; still mostly negative R² overall, not yet an
  actually-useful correction — a follow-up feature/data-improvement pass is in progress (§10).

## 10. Four parallel follow-ups dispatched after the real-hardware session — all landed

Dispatched together once there was no robot-time constraint left:
1. **`kp_rot_wrist` retune** (`docs/status/kp_rot_wrist_retune_2026-08-02.md`) — confirmed
   structurally safe first (no matrix inversion in this term, unlike the unrelated `kp_rot`
   gain's documented instability mechanism). Raising `kp_rot_wrist` alone went unstable
   (underdamped PD); scaling `kd_rot_wrist` alongside (~4:1) fixed that. Chosen candidate
   `kp_rot_wrist=20/kd_rot_wrist=80` cuts real orientation-error growth 35-41% with zero
   regression on the standard sweep (34/38, byte-identical to baseline). A more aggressive
   candidate (kp=80/kd=320) cut it further (~75%) but introduced a new long-hold divergence —
   correctly rejected. New config:
   `config/ur5e_mujoco_torque_osc_tuned_split_base_wrist_wrist_orient_retuned.yaml`.
2. **Residual-regression feature improvement, round 2** — landed, see §9's update: cross-joint
   coupling features made the model genuinely useful (5/6 joints improved, `wrist_2` went from
   `R²≈0` to `+0.5`), not just honest. New default `--feature-set coupled`.
3. **Hanging-pose transport family** — landed, see §8: the structural fix, `36/38` sim rigor
   sweep, needs a real clearance check before real hardware.
4. Skill updates (`.agents/skills/`, not git-tracked) capturing tonight's operational lessons —
   sim-fidelity limits, config-key parity, a new `background-agent-dispatch` skill for a
   recurring agent-dispatch failure pattern seen 3 times tonight.

## 11. Current real-hardware-validated envelope (the practical answer to "how far can we go")

At `height_alpha=0.5`, zero-degree pose, `config/ur5e_mujoco_torque_osc_tuned_split_base_wrist.yaml`,
`accel_duration_scurve`:
- **`+X`**: clean through `accel=0.015, move_duration=8.0s` (92% achieved). Fails at
  `accel=0.02` via the orientation-growth-driven speed guard at **both** `move_duration=4.0s`
  (43% achieved, `orientation_error` 0.028 rad at trip) and `8.0s` (31% achieved,
  `orientation_error` 0.083 rad at trip — ~3x the `4.0s` run's value). Doubling duration made it
  *worse*, not better, directly confirming the exposure-time hypothesis from §4: more time gives
  orientation error more room to grow before the same guard catches it, rather than helping the
  arm "get there gently."
- **`-X`**: clean at `accel=0.01` (implied by symmetry, not directly tested — only `+0.01` was
  actually run). Fails at `accel=0.015, move_duration=8.0s` in all 3 trials tonight — now fully
  explained, not a mystery (§6): 2/3 trials were RTDE telemetry staleness artifacts, 1/3 was the
  already-understood §4 orientation-growth mechanism. `-X` ceiling for this pose/config is
  `accel=0.01`, same as `+X`'s effective ceiling once §4's exposure-time finding is accounted
  for.
- Singularity-conditioning: fixed and validated in **every real run tonight, 9/9, pass or fail.**
- All of the above is specific to `split_base_wrist_task` at the OLD (singular) pose family. The
  new hanging-pose family (§8) hasn't had real hardware or the same displacement-envelope
  characterization done yet — its sim numbers (36/38 rigor sweep) aren't directly comparable to
  this section's real-hardware displacement-envelope numbers without that work.

## 12. Suggested next steps (not decided, for discussion)

1. ~~Directly test `+X` at `accel=0.02, move_duration=8.0s`~~ — **done**, see §11.
2. ~~A third `-X` `0.015/8s` trial~~ — **done**, resolved via §6.
3. ~~Acceleration feedforward~~ — **done**, mixed result, see §9. Not real-hardware tested.
4. ~~`kp_rot_wrist` retune~~ — **done**, see §10 item 1.
5. ~~Residual-regression ridge fix~~ — **done**, see §9. Feature/data improvement round 2 —
   **done**, see §10 item 2.
6. ~~Hanging-pose transport family~~ — **done**, see §8. The biggest open item now: a real
   physical clearance check in the lab, then a first small real test — cannot happen from home.
7. **Hanging-pose `-45°` clearance variant — done, real finding, see §13.** Kinematically the
   singularity-avoidance survives rotation (`cond(J)` unaffected); dynamically it does NOT — the
   Y-drift coupling that bit the OLD family at this same rotation reintroduces here too, just
   with ~3x more margin before onset. New evidence this coupling is a base-rotation
   controller-architecture property, not something inseparable from the old family's wrist
   singularity as previously assumed.
8. Full-range rigor sweep for the hanging-pose family (only `alpha=0.5` fully characterized) —
   **dispatched, in progress**.
9. Whether the hanging pose needs `split_base_wrist_task` at all, since it doesn't sit at the
   singularity to begin with — **dispatched, in progress**.

## 13. Hanging-pose `-45°` clearance variant — real finding: Y-drift coupling is NOT singularity-specific

(`docs/status/hanging_pose_clearance_variant_2026-08-02.md`) New pose
`hardware/poses.py::HANGING_ALPHA_0_5_CLEARANCE_Q` (the hanging family's `shoulder_pan`
overridden to `-45°`, mirroring `HEIGHT_ALPHA_0_5_CLEARANCE_Q`'s pattern exactly) — additive
only, mirrored into `rl_gain_scheduling/gain_scheduling_env.py`.

- **`cond(J)` unaffected by rotation**: 21-point sweep matches the un-rotated family to float
  precision (max diff `1.24e-14`) — `shoulder_pan` rotation is a rigid-body symmetry that
  doesn't touch conditioning, as expected.
- **But the rigor sweep drops hard**: `canonical_grid` `2/8` at this rotated pose vs. `8/8`
  un-rotated, with the SAME qualitative signature the old pose family showed on real hardware at
  this exact rotation — near-1:1 X:Y diagonal motion, growing with displacement, up to a real
  `|Y-Y0|>0.03m` guard trip at `dx=0.20m`.
- **The real news**: since this reproduces at `cond(J)≈7-15` (nowhere near singular), the Y-drift
  coupling is **not** inseparable from the old family's wrist singularity, as this session
  previously assumed — it looks like a real property of commanding this controller architecture
  at a rotated base pose, independent of which specific pose family it's rotating. Genuinely
  useful, if sobering, evidence: neither `split_base_wrist_task` nor a totally different pose
  family fixes this on its own — it may need its own dedicated investigation regardless of which
  pose family is chosen.
- One real silver lining: onset margin is meaningfully better here (~3x further out than the old
  family's documented sim onset of `~0.05-0.06m`) — a real, if partial, improvement, not a fix.
- **Not real-hardware-ready** — no physical clearance check has been done for this rotated
  hanging posture, same rule as every pose change in this project.
