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

## 6. Open, unexplained — `-X` acceleration transient at `0.015/8s`, plus a likely second overlapping mechanism

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
levels, no consistent timing) don't fit that pattern at all. Most likely **two distinct real
mechanisms both active near this `-X` magnitude**, with which one trips first varying run to
run — consistent with a genuine physical effect with real run-to-run stochasticity (e.g.
stick-slip has exactly this character), not a deterministic trajectory-shape or guard-noise
artifact. The accel-transient piece (trials 1-2) remains unexplained; the speed-trip piece
(trial 3) is plausibly just the already-understood §4 mechanism showing up in this direction
too.

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
- **"Hanging" end-effector pose redesign** — considered and deprioritized (not built). Was
  motivated by avoiding `wrist_2=0` entirely, but that problem is already solved more surgically
  by `split_base_wrist_task`; the two remaining open problems (§4, §6) have no evidence tying
  them to the pose choice specifically. Would be a much larger, more invasive change (duplicated
  pose definitions in two files, invalidates all tuned gains, needs fresh real-world clearance
  re-verification) with no clear problem it currently solves. Revisit only if future evidence
  ties either open issue to sitting near `wrist_2=0` specifically.

## 8. In progress, not yet reported

- **Acceleration feedforward** (background agent, sim-only build): adding a computed-torque-style
  feedforward term (`Fx += effective_mass * target_x_accel`) to the currently-pure-PD impedance
  law, motivated by real observed tracking lag/jitter and the complete absence of any
  acceleration feedforward in the current controller (confirmed by direct grep — `target_accel_mps2`
  only reaches the trajectory generator, never the torque law). New flag-gated option, new named
  config layered on `split_base_wrist_task`. Not yet complete as of this compilation.

## 9. Current real-hardware-validated envelope (the practical answer to "how far can we go")

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
  actually run). Fails at `accel=0.015, move_duration=8.0s` in all 3 trials tonight, via two
  likely-distinct real mechanisms (§6) — an unexplained accel transient (2/3 trials) and the
  already-understood §4 orientation-growth speed trip (1/3 trials). `-X` ceiling for this
  pose/config is `accel=0.01`, same as `+X`'s effective ceiling once §4's exposure-time finding
  is accounted for.
- Singularity-conditioning: fixed and validated in **every real run tonight, 9/9, pass or fail.**

## 10. Suggested next steps (not decided, for discussion)

1. ~~Directly test `+X` at `accel=0.02, move_duration=8.0s`~~ — **done**
   (`direct_torque_20260801_203354`): worse than `4.0s`, confirming the exposure-time
   hypothesis (see §9). `+X` ceiling for this pose/config is now `accel=0.015` at any tested
   duration; `0.02` fails regardless of duration.
2. ~~A third `-X` `0.015/8s` trial~~ — **done** (`direct_torque_20260801_204446`): revealed a
   likely second, distinct mechanism rather than resolving the first (see §6's revised read).
   `-0.01` is the practical `-X` ceiling for now.
3. Once the acceleration-feedforward agent reports, real-hardware test it (small, careful first
   step per this session's own established discipline) — motivated by the general jitter/tracking-lag
   discussion, not by either open failure mode specifically.
4. `docs/status/split_base_wrist_orientation_growth_2026-08-01.md`'s `kp_rot_wrist` retune
   pointer is real, scoped, future work — not attempted tonight (would need its own sim
   validation pass before real hardware, per this repo's own gain-tuning discipline).
5. **Dispatched** (background agent, sim/offline-only): fix the residual-torque regression's
   catastrophic held-out extrapolation blowup (2 of 6 joints, R² deeply negative) found earlier
   tonight — regularization/output-bounding, not a redesign. This repo's own top-ranked
   "make the controller smarter" direction (`docs/status/nonlinear_controller_research_2026-07-31.md`),
   explicitly not another RL attempt.
