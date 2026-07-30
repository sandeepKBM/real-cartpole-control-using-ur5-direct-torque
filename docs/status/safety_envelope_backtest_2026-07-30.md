# Smooth/state-aware safety envelope backtest — 2026-07-30

Context: a proposal is on the table to replace/augment `hardware/safety.py`'s rigid,
flat-threshold guards (`CartesianMoveMonitor`'s TCP speed/acceleration ceilings in
particular) with a "smooth funnel"-style envelope — a bound that shrinks continuously as
some risk metric grows, instead of tripping the instant a fixed number is crossed. The
counter-argument already accepted as the standard to beat: every REAL safety incident
found in this project's history was a wrong-number or missing-check bug, not a
rigid-vs-smooth shape problem, and a badly-tuned smooth interpolation between two
correctly-tuned regimes can create a "hole" in the envelope — looser than either
endpoint — that's worse than a rigid ceiling and harder to catch by inspection.

This doc is the result of the agreed falsifiable test: replay REAL recorded guard trips
through (a) the current rigid `CartesianMoveMonitor`/baseline as ground truth and (b) two
concrete smooth/state-aware candidates, and count avoided trips vs. missed trips. **Zero
misses on the one confirmed-genuine catch was a hard, pre-agreed disqualifying bar.**

Harness: `experiments/safety_envelope_backtest.py` (this worktree,
`experiments/safety-envelope-study` branch). Run with:
`/common/users/ss5772/miniforge3/envs/mujoco_ur5e/bin/python3 experiments/safety_envelope_backtest.py`

## 1. Data used

All 21 runs in `outputs/hardware_transport/` (main repo) whose `summary.json` has a
`termination_reason != "duration_complete"` — every one of them is a **real** live UR5e
RTDE session from `thinkrobot`, 2026-07-28 (the first-ever live motion/torque tests on
this project's real arm; see `hardware_captures/2026-07-28_thinkrobot_172.16.71.77/README.md`
for 6 of the 21, narratively described; the other 15 are the same session's later attempts,
recovered directly from `outputs/hardware_transport/*/summary.json` + `trace.jsonl`, not
previously written up). **No synthetic or hand-built trace data was used anywhere in this
backtest** — every number below comes from real RTDE telemetry.

One real, load-bearing data limitation, discovered while building the harness and
documented in the script itself: production's control loop only appends a trace row for a
cycle *after* that cycle's `move_monitor.check()` call returns `ok=True`. The cycle whose
`check()` call actually failed and ended the run is therefore **never written to
`trace.jsonl`** — confirmed generally (not just spot-checked) across all 21 runs:
`summary["steps"]` always equals `len(trace_rows)` exactly, and `summary["sim_time_s"]`
always equals `steps * dt_s`. The harness reconstructs that one missing cycle per run by
holding the last logged row's joint velocity constant for one more `dt_s` and mapping it
to a Cartesian delta via the real linear Jacobian (`controller_core.model_dynamics.
PinocchioUR5eDynamics`, the same dynamics provider already parity-tested elsewhere in this
codebase) — a documented, physically-motivated extrapolation, not a fabricated number.

A second, unexpected limitation: several of the longer real runs (roughly half of the 21)
turn out to have used `CartesianMoveMonitor`'s noise-robust `accel_gap_cycles`/
`speed_lowpass_alpha` CLI overrides (added same-day per `hardware/safety.py`'s own
docstring) — but neither value is recorded anywhere in `summary.json`. Replaying those
runs with the class defaults (`gap=1, alpha=1.0`) produces a **false** self-check failure:
a genuine several-micron real telemetry jitter, double-differenced by an unfiltered
estimator, produces a spurious multi-m/s^2 spike at cycle 2 that real production never saw
(it logged hundreds of clean cycles past that point). The script calibrates the smallest
filtering that reconciles baseline replay with the real recorded pass history, and flags
every run where this was needed. This recovered 19 of 21 runs to a clean self-check; **2
runs (`direct_torque_20260728_162159`, `direct_torque_20260728_162206`) could not be
reconciled within a wide grid search and are excluded from the candidate comparison as
inconclusive** rather than reported on faith. `162206`'s real trip was a `DeadlineMonitor`
overrun in any case (not a `CartesianMoveMonitor` accel/speed event), so it was out of
scope for these two candidates regardless.

Net: **19 of 21 real trip runs are usable for a trustworthy candidate comparison.**

## 2. Candidates tested

Both candidates are implemented by mutating the real, imported `CartesianMoveMonitor`'s
`limits` object in place before each `check()` call — `check()` re-reads
`self.limits.max_tcp_accel_mps2`/`max_tcp_speed_mps` fresh every call, so this drives novel
threshold logic through the monitor's actual, unmodified internal math (position history,
EMA smoothing, gap windowing) rather than reimplementing any of it, per the task's
constraint.

**Candidate A — CBF-style, move-phase-timing risk metric.** Full baseline ceiling
throughout the commanded min-jerk move (real intended motion legitimately accelerates
there); once the move ends and the axis target goes static, the ceiling shrinks smoothly
over a 0.5 s settle window down to 20% of baseline. Rationale: a TCP-accel spike once
nothing should be moving is categorically more suspicious than the same spike mid-move.

**Candidate B — cond(J)-scaled threshold.** Threshold = base x `scale(cond(J))`, using the
*exact* log(cond(J))-space linear interpolation shape already validated in this codebase
for `controller_core/x_axis_cartesian_impedance.py`'s `lambda_adaptive_regularization`
(`_scheduled_lambda_regularization`). The breakpoints were **not** copied from that module
(`1e4`/`1e8`) — those were tuned for a Tikhonov regularization epsilon on the task-space
inertia matrix, a different physical quantity — but re-picked from this dataset's own
observed cond(J) range: ~1e2 at ordinary transport poses, ~5.8e5 at the real wrist_2=0
singularity. `cond(J)` is computed per-cycle from real logged `q` via
`PinocchioUR5eDynamics.jacobian`.

## 3. The disqualifying check, first

`position_20260728_150847` is the one run with independently-documented physical evidence
of genuine danger (not just "a number crossed a line"): its trace shows Z climbing
monotonically over the last 3 logged samples and wrist_1/wrist_3 `|qd|` growing
near-exponentially (~1.6-1.8x per 8 ms step) while `wrist_2` sits within ~1e-4 rad of
exactly 0 — the UR wrist singularity, driven via `servoL`'s firmware IK with none of this
codebase's singularity protections active in position mode. Per the pre-agreed rule, any
candidate that lets this one slide is disqualified regardless of every other number.

| | Result |
|---|---|
| Baseline (real, recorded) | **TRIPS** — `TCP acceleration 1.3545 m/s^2 > 1.2 m/s^2` |
| Candidate A (cbf_move_timing) | **AVOIDED — DISQUALIFYING MISS** |
| Candidate B (cond_j_scaled) | **TRIPS** (correctly) — cond(J) reached 5.78e5; candidate B actually trips 7 cycles *earlier* than baseline on this run (cycle 2, t=0.016s, once cond(J) is already ~5.8e5) |

**Candidate A is disqualified outright.** The move-phase-timing risk metric is the wrong
axis for this danger: the divergence is a structural/kinematic property of the pose (a
true wrist singularity), not a timing property of the move, and the entire trip happens
*during* the commanded move (well before Candidate A's settle-window tightening would ever
engage) — so Candidate A stays at the loose, full-ceiling ratio the whole time and never
catches it. This is a direct, concrete instance of the counter-argument's core worry: a
smooth design can look better in aggregate (see SS4 — it has the *higher* raw "avoided"
count of the two) while hiding a real correctness gap.

## 4. Full results (19 applicable runs)

| Candidate | Real trips avoided/graduated | Disqualifying misses |
|---|---|---|
| A: cbf_move_timing | 18 / 19 | **1 (position_20260728_150847) -- DISQUALIFIED** |
| B: cond_j_scaled | 15 / 19 | 0 |

Breakdown of the 7 runs where the real trip was a single logged cycle with essentially
zero real motion (`qd` ~1e-4 rad/s, tau ~1e-3 Nm) -- i.e. the timing/noise-estimator
artifacts this project's own `hardware/safety.py` docstrings already identify as the
motivating bug for the noise-robust accel filtering (real elapsed-time vs. assumed-`dt_s`
double-differencing, amplifying noise by ~1/dt^2): **both candidates correctly avoid all 7**
(`direct_torque_151331`, `151512`, `152908`, `153128`, `153225`, `154811`,
`position_145240`). Neither candidate is doing anything clever here -- both are simply
loose enough at cycle 0-1 (before either risk metric has anything to react to) that they
don't inherit baseline's over-tight single-cycle math. This is the least interesting 7/19
of the result.

Of the remaining 12 applicable runs (longer, real-motion runs where baseline eventually hit
a flat speed/accel ceiling after hundreds of clean cycles, or where an early real trip has
some genuine kinematic content): Candidate A avoided 11/12, Candidate B avoided 8/12 -- but,
critically, one of B's 4 "still trips" is the correct disqualifying-check case above, so
B's genuine shortfall on this subset is 3/12, not 4/12.

## 5. Candidate B's own new problem, found by this same backtest

The backtest surfaced a second finding that complicates a clean "B wins": on the position-
mode runs, which per `hardware_captures/.../README.md` and `AGENTS.md` SS3 all start at
`wrist_2=0.0` (**this codebase's own documented, deliberately-used transport start pose** --
`config/ur5e_mujoco_torque_osc_tuned.yaml`'s canonical operating point, not an edge case),
Candidate B's cond(J)-scaled threshold is tightened to ~20% of baseline from cycle 2
onward (t~16ms) in *every* such run -- including two (`position_145539`, and likely
`position_150316`) where the existing README explicitly documents the trip as benign
(`max_abs_qd_radps` only 0.018 rad/s, "all drift/orientation metrics negligible" --
`position_150316`'s `qd_max` is higher, 0.082 rad/s, and less clearly benign, but nowhere
near `position_150847`'s confirmed-divergent 0.84 rad/s). Because cond(J) is a static
property of the pose, not a measure of whether anything is actually going wrong yet,
Candidate B cannot distinguish "this is the real, singular, but intentionally-used
operating pose, functioning normally" from "this is the real operating pose, and it's about
to diverge" -- it tightens by the same amount in both cases. In practice this candidate, as
tuned here, would make it very hard to ever complete an ordinary move that starts at this
pose without a near-immediate trip -- a concrete instance of the *other* half of the
counter-argument's worry: not a "hole" looser than either endpoint, but a wall tighter than
either endpoint at a real, legitimate, frequently-visited operating point.

## 6. Verdict (Candidates A/B; superseded in part by SS8 below -- see there for Candidate C)

**Mixed, and net negative for both concrete designs tested -- do not build either as
specified.** The falsifiable test the proposal accepted in advance was: zero misses on
the genuine catch, full stop. Candidate A fails that test outright, despite topping
Candidate B on the aggregate "avoided" count (18/19 vs 15/19) -- which is itself the
clearest evidence this backtest could have produced *for* the counter-argument: a smooth
design's higher aggregate score is not evidence of safety, and picking the design with the
better top-line number without the disqualifying check would have shipped the wrong one.
Candidate B clears the disqualifying bar and gives real signal that a cond(J)-aware
mechanism can help, but as tuned here it introduces its own state-aware-envelope-specific
failure mode (uniformly over-tight at a real, named, intentionally-used singular operating
pose) that a flat rigid ceiling never has, since a flat ceiling's behavior doesn't depend on
which pose you're at.

This is not evidence that a smooth/state-aware envelope is structurally impossible to build
correctly -- Candidate B's core idea (shrink with cond(J), same functional form already
validated for `lambda_adaptive_regularization`) is not disproven, only this specific
static, pose-only version of it. A version that also incorporates *recent growth rate*
(e.g. is `|qd|` or drift accelerating away from a settled value, not just "is the pose
currently ill-conditioned") is the natural next design to test, since that is exactly the
distinguishing signal between `position_150847` (genuinely, monotonically worsening) and
`position_145539`/`150316` (transient, bounded) that a static cond(J) reading cannot see.
That is future work, not a claim this doc is making evidence for yet.

**Recommendation: do not replace `hardware/safety.py`'s rigid guards with either candidate
as currently specified.** If this line of work continues, the next falsifiable version
should test a growth-rate-aware metric, using this same harness and the same disqualifying
check, before any of this touches a live-motion code path.

## 7. Explicitly out of scope / not touched

Per the task constraints, nothing in `hardware/safety.py`, any `hardware/*_transport.py`,
or `hardware/motion.py` was modified -- this is pure offline analysis in
`experiments/safety_envelope_backtest.py`. Two follow-up items surfaced by this work, for a
deliberate decision rather than a silent patch:

- The noise-robust `accel_gap_cycles`/`speed_lowpass_alpha` values actually used for ~10 of
  the 21 real runs analyzed here are unrecoverable after the fact -- they were never written
  to `summary.json` even though the CLI supports overriding them. If real-hardware runs
  continue to use these overrides, they should be logged into the run record
  (`observability/run_logger.py` or the transport summary schema), the same way
  `torque_limit_scale` and other CLI-driven parameters already are.
- `direct_torque_20260728_162159` and `_162206` could not be reconciled to the real
  recorded pass history within a wide `(gap, alpha)` grid search; both are excluded from
  this analysis as inconclusive rather than forced through. Worth a closer look with the
  actual CLI invocation history if it's recoverable from session logs, but not blocking for
  this verdict.

## 8. Follow-up (same day): Candidate C -- growth-rate-aware threshold

SS6's recommended next step was tested directly: a candidate that conditions on the RATE OF
CHANGE of a risk quantity, not its instantaneous magnitude, to see whether that specifically
fixes Candidate B's over-tight-at-a-static-singular-pose failure mode (SS5) without
reintroducing Candidate A's disqualifying miss (SS3). Implemented as `QdGrowthRateCandidate`
in the same `experiments/safety_envelope_backtest.py`, same real 21-run dataset, same
disqualifying-check discipline, extended rather than rewritten.

**Design.** Risk metric: `|qd|` (joint-velocity Euclidean norm) -- not `cond(J)`. Chosen over
`d(cond(J))/dt` for two concrete, empirical reasons: (1) `|qd|` is *exactly* the quantity
documented as growing in the one real disqualifying case -- "wrist_1/wrist_3 joint
velocities grow near-exponentially step over step (~0.31 -> ~0.55 -> ~0.84 rad/s, roughly
1.6-1.8x per 8ms step)" (`hardware_captures/2026-07-28_.../README.md` item 4) -- so it is a
direct leading indicator of the real failure mode, not a proxy for one; (2) it needs no
Jacobian/pinocchio call, so it's available on every real cycle (`getActualQd()`) at zero
extra compute, unlike a per-cycle `cond(J)` history. `growth_rate()` computes the
geometric-mean per-cycle multiplicative growth factor over a 3-cycle window, both endpoints
floored at 5e-3 rad/s (well above the real ~1e-4 rad/s stationary noise floor measured in
`stationary_noise_capture_154018_stats.json`, so flat noise never reads as "growth"). Flat or
shrinking `|qd|` (growth_rate <= 0.05/cycle) gets the full baseline ceiling; sustained growth
at or above 0.5/cycle (picked below the ~0.6-0.8/cycle documented in the real disqualifying
case, so it tightens before matching that growth) gets the 20% floor; linear interpolation
in growth-rate space between the two.

**Disqualifying check, run first, same as before.** `position_20260728_150847`: baseline
TRIPS; **Candidate C also TRIPS** on the reconstructed final cycle (growth rate measured up
to 3.442/cycle -- far past `r_high`). **No disqualifying miss.** Candidate C also trips
*within the logged rows* at cycle 6 (t=0.048s) of this run -- i.e. it independently
red-flags the real divergence before the run even reaches its undocumented final cycle,
using only the growth trend, not a lucky static threshold.

**Does it fix Candidate B's specific failure mode?** Checked directly against real logged
telemetry for the two runs that exposed it:

| Run | Real trip | Candidate B (`cond_j_scaled`) on logged rows | Candidate C (`qd_growth_rate`) on logged rows |
|---|---|---|---|
| `position_20260728_145539` (README: confirmed benign, `qd_max`=0.018 rad/s, drift/orientation negligible) | `TCP acceleration 0.9042 > 0.5` | **nuisance-trips at cycle 2** (`0.3770 > 0.10`) | **never trips** -- correctly rides through on the growth-rate metric |
| `position_20260728_150316` (no README writeup; `qd_max`=0.082 rad/s, higher than the confirmed-benign case, well below the confirmed-divergent 0.84 rad/s -- genuinely ambiguous) | `TCP acceleration 1.0749 > 0.5` | trips at cycle 2 (`0.2815 > 0.10`) | trips at cycle 9 (`0.1381 > 0.10`) |

On the one run with unambiguous ground truth (`145539`, independently documented as benign
before this backtest existed), **Candidate C correctly does not nuisance-trip where
Candidate B does** -- direct evidence the growth-rate reformulation targets the intended
failure mode, not just a retuned constant. On the one ambiguous run (`150316`, no prior
documentation either way), both candidates still trip; Candidate C's trip is later (cycle 9
vs. cycle 2) and, unlike Candidate B's, is driven by an actual measured trend rather than a
pose-static value -- consistent with genuine caution on a case this doc cannot itself
classify as safe.

**Aggregate numbers (19 applicable runs, same set as SS4):**

| Candidate | Real trips avoided/graduated | Disqualifying misses |
|---|---|---|
| A: cbf_move_timing | 18 / 19 | 1 -- DISQUALIFIED |
| B: cond_j_scaled | 15 / 19 | 0 |
| C: qd_growth_rate | 15 / 19 | 0 |

Candidate C ties Candidate B's raw count -- it does not "win" on the aggregate number, and
this doc is not claiming it does. The improvement is qualitative, not quantitative: on the
7 single-cycle noise-artifact runs (SS4) both C and B (and A) already agreed and avoided all
7 -- that subset was never the interesting one. On the 12 real-motion runs, C's behavior is
now *causally tied* to the real risk signal (it rode through the one case independently
confirmed safe, and still caught the one case confirmed dangerous, using the same
underlying mechanism), where B's was coincidental (B tripped at cycle 2 in literally every
position-mode run regardless of outcome, because it only ever looks at the static pose).
That is the property SS6 asked the next design to have, and it is the property measured
here, directly, against real telemetry -- not inferred.

**Updated verdict.** Candidate A remains disqualified (SS3, unchanged). Candidate B remains
not recommended as specified (SS5, unchanged). **Candidate C clears the disqualifying bar,
does not reproduce Candidate B's over-tight-at-a-static-pose failure mode on the one case
with clear ground truth, and is the first of the three designs tested whose per-run
behavior is defensible on a mechanistic basis rather than by threshold-tuning luck.** This
is still a single day's backtest against 21 runs from one real session, all at one specific
transport pose family -- not a green light to wire this into `hardware/safety.py` untested
on hardware. Recommended before any live-motion use: (1) validate against a second, distinct
real-hardware capture session (different pose, different day) to check the 3-cycle window
and 0.05/0.5 growth-rate breakpoints generalize rather than being fit to this one night's
data; (2) resolve `position_20260728_150316`'s ambiguity with real ground truth (was it
actually fine, or an early/slower version of the same divergence?) rather than leaving both
candidates' agreement on it uninterpreted; (3) get a considered answer on the
`|qd|`-vs-`d(cond(J))/dt` design choice from someone who can reason about the wrist-
singularity dynamics directly, since this doc picked `|qd|` for practical/empirical reasons,
not because the alternative was tested and lost.
