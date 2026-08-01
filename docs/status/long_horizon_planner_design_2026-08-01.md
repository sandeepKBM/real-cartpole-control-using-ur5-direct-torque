# Long-horizon (receding-horizon) planner — design/research doc, 2026-08-01

**Scope**: design/research only, per explicit instruction. No code, no new configs, no
`controller_core/`/`hardware/` edits. This document answers whether and how to add a
planner that looks ahead and injects corrective action early, motivated by the user's
stated "maximum movement and faster movement" goal and the near-future need to handle
Coriolis/centrifugal coupling at higher speed and abrupt reversals.

**Relationship to `docs/status/nonlinear_controller_research_2026-07-31.md`**: that survey
covered representational-capacity upgrades to the *reactive* controller (residual learning,
LuGre friction, Koopman lifting, learned gain-scheduling) and explicitly did not cover
receding-horizon planning/MPC — its scope was "higher order of representation" for
`compute()` itself, not a new planning layer above it. This document is new ground, not a
duplicate. Where the two surveys reach the same background facts (real-time budget, the
directional-ceiling root cause, the RL gain-scheduling failure history), this doc cites
them rather than re-deriving them.

## 0. What already exists that this proposal reuses

1. **Trajectory generation is already separated from tracking.**
   `simulation/ur5e_mujoco_torque.py::x_profile_target()` (lines 656-732) is a closed-form
   function `(profile, x0, target_x_delta, t_s, duration_s, move_duration_s,
   target_accel_mps2) -> (x_des, x_vel_des)`. It has exactly two real call sites in the
   control loops: `hardware/direct_torque_transport.py:434` and
   `tools/ur5e_mujoco_torque_experiments.py:785` (plus `hardware/position_transport.py:223`
   for the `position` mode and `rl_gain_scheduling/gain_scheduling_env.py:304` for the RL
   env). In every case the call happens once per 500 Hz cycle, inside the loop, and its
   two outputs are written straight into `robot_state["target_x"]` /
   `["target_x_vel"]`, which `XAxisCartesianImpedanceController.compute()` reads at
   `controller_core/x_axis_cartesian_impedance.py:459` (`x_des = float(st["target_x"])`)
   with no notion of *how* that number was produced. This is the real, load-bearing fact
   the rest of this design leans on: **the 500 Hz loop already treats "what's the current
   reference" as a black-box function call**, so a planner only has to win the right to be
   that function call's implementation for the next few seconds — it does not need to run
   at 500 Hz itself, and it does not need to touch `x_axis_cartesian_impedance.py` at all.

2. **A real dynamics model is already available and cheap enough to reuse for
   prediction.** `controller_core/model_dynamics.py::PinocchioUR5eDynamics` provides
   `gravity(q)`, `coriolis(q, qd)` (pure `C(q,qd)@qd`, dedicated zero-gravity RNEA call,
   lines 164-188), `mass_matrix(q)`, `bias(q, qd)`, all parity-verified against MuJoCo to
   <1e-8 Nm / <1e-6 bias / <1e-8 mass matrix (module docstring, lines 1-15). This is
   exactly the model a predictive rollout needs (`qdd = M(q)^-1 (tau - bias(q,qd))`), and
   it already has a `jacobian()` method (lines 204-221) for mapping task-space plans to
   joint space.

3. **A residual signal exists but is diagnostic-only.**
   `controller_core/dynamics_residual.py` computes `qdd_residual = qdd_measured -
   qdd_predicted` every cycle; `docs/status/nonlinear_controller_research_2026-07-31.md`
   §1 already flags that `tau_residual = M(q) @ qdd_residual` is a training target sitting
   right there, unused beyond logging. Relevant to a planner only as a *future* input (a
   learned residual could improve the planner's prediction model later) — not a
   dependency for a first version.

4. **The real-time budget is characterized, not assumed.**
   `docs/status/direct_torque_controller_phase_profiling_2026-07-31.md`: real hardware
   500 Hz loop, 2 ms period, measured `controller_mean_ms=0.615` (p95 0.677, max 0.801),
   total real-cycle cost mean 1.28 ms / p99 1.47 ms of the 2 ms budget — **roughly
   0.5-0.7 ms of real headroom** before the deadline monitor. This number is why a
   planner cannot run inside the 500 Hz loop at all (see §1.2) — there is not enough slack
   for even a single QP solve, let alone an iterative one, on every cycle.

5. **A "guard margin" concept already exists in this codebase, but not on the path this
   proposal touches.** `controller_core/recoverability_monitor.py` /
   `controller_core/safety_filter.py` implement a `x_warning_margin_m`-based command
   governor with graduated intervention levels — but both files are explicitly scoped to
   "the constrained cart-pole scaffold" (their own docstrings), part of the legacy/generic
   `controller_interfaces.py` abstraction, and have zero call sites in the UR5e OSC path
   (`grep` for `recoverability_monitor`/`safety_filter` imports outside tests returns only
   `controller_core/__init__.py` and each other). They are not currently wired to
   `ImpedanceSafetyMonitor` or the UR5e transport loops at all. Useful prior art for the
   *shape* of a margin-based cost term (§1.4 below), not reusable machinery.

6. **What a "predictive catch" would look like on real data**: `hardware/safety.py:413`
   region (`docs/status/safety_envelope_backtest_2026-07-30.md`) and this week's
   `pre_trip_trend` capture (commits `f1a0791`, `467fe52`, landed in
   `hardware/direct_torque_transport.py`) exist specifically because guard trips were
   being diagnosed by hand from `trace.jsonl` after the fact. `pre_trip_trend` computes a
   first-third-vs-last-third rising/falling/stable classification over a 60-cycle window
   (`PRE_TRIP_TREND_WINDOW_CYCLES`) of `qd`, TCP speed, `x_error`, `tau_controller` L1
   norm, orientation error, and (as of `467fe52`) Y/Z drift — captured every cycle,
   read only at trip time. This is exactly the kind of signal a receding-horizon
   predictor would need to reproduce *before* the trip, not after — it is strong evidence
   that "the guard margin was closing for several cycles before the trip" is a real,
   already-observed pattern in this system, not a hypothetical.

## 1. Right-sized design for THIS system

The framing to avoid: "should we do MPC." The actual design question, given §0, is
narrower: **what's the smallest thing that (a) reuses `x_profile_target()`'s existing
call-site contract unchanged, (b) never touches the 500 Hz loop's per-cycle budget, and
(c) uses the dynamics model that's already validated and already sitting in the process?**

### 1.1 Architecture: a slow outer loop feeding a fixed-shape reference into the existing tracker

```
[500 Hz OSC loop, unchanged]  <-- reads robot_state["target_x"/"target_x_vel"] every cycle,
       |                          exactly as it does today from x_profile_target()
       |
[reference buffer]  <-- small array of (t, x_des, x_vel_des) samples, written by the
       |                 planner, read by the 500 Hz loop via interpolation
       |
[planner, ~10-50 Hz]  <-- runs on its OWN cadence, in the SAME process (no IPC latency
                           budget needed if run between cycles / on a slower tick;
                           multiprocessing only if profiling shows it must be), consumes
                           current (q, qd), rolls forward with PinocchioUR5eDynamics,
                           re-optimizes a short local segment, writes the buffer
```

Concretely: add a **new profile value** to `x_profile_target()`'s existing `profile`
string dispatch (today: `step`/`ramp`/`min_jerk`/`min_jerk_move_hold`/
`accel_duration_triangular`/`accel_duration_scurve`) — e.g. `planned_segment` — that,
instead of evaluating a closed-form polynomial, looks up `(x_des, x_vel_des)` from the
current reference buffer by linear interpolation on `t_s`. This is a few-line addition to
a function that already dispatches on a string and already returns exactly this tuple
shape; it changes nothing about how either transport loop *calls* the function
(`hardware/direct_torque_transport.py:434`, `tools/ur5e_mujoco_torque_experiments.py:785`
keep calling `x_profile_target(...)` with the same signature). The planner becomes a
producer that overwrites the buffer asynchronously; the 500 Hz loop stays a pure consumer,
identical in structure to today.

This is deliberately the minimum-invasiveness option consistent with AGENTS.md §7 ("do
not change training/eval logic and controller logic in the same commit," "never touch the
real-time control path without extremely explicit justification") — it adds a lookup
branch to a pure function, not a rewrite of the loop.

### 1.2 Replan rate — justified against the measured budget, not guessed

Two independent numbers bound this:

- **Not 500 Hz.** §0.4's ~0.5-0.7 ms of headroom is barely enough for `compute()`'s own
  small matrix ops (§0.4's cited sub-op table: `cond(J)` SVD ~24 µs, Λ-shaping ~31 µs,
  nullspace ~10 µs — all *individually* cheap only because they're 6x6). A receding-horizon
  solve — even a cheap one — reasonably costs 100 µs-few ms depending on horizon/iterations,
  which does not fit in the remaining headroom without risking exactly the deadline-overrun
  failure mode already observed once this week (`docs/status/real_lab_session_2026-07-31.md`
  §5: one real `DeadlineMonitor` trip from an unrelated ~0.15 ms diagnostic addition).
- **A reasonable target: 20-50 Hz (20-50 ms replan period).** Justification: (a) the
  trajectory profiles this system actually uses are 1-30 s moves (AGENTS.md §3's validated
  envelope: canonical grid holds 1-2 s, long holds up to 30 s) — a reference that updates
  every 20-50 ms is >>20x faster than the timescale of the moves themselves, so it can track
  a changing plan without visibly chunky reference updates; (b) it leaves at least 10-25 full
  500 Hz cycles between replans, comfortably enough wall-clock slack (20-50 ms) for even an
  unoptimized Python/numpy short-horizon solve to run *outside* the hot loop without
  contending with its 2 ms deadline; (c) it's slow enough that, if it does run as a
  separate process/thread (see §1.5), any IPC latency is a rounding error relative to the
  replan period itself — unlike the residual observer's async move
  (`docs/status/real_lab_session_2026-07-31.md` §3), which mattered because it was
  originally *inside* the 2 ms budget.
  This is a design-time estimate reasoned from the two numbers above, not a value measured
  from a running planner (none exists yet) — the offline prototype in §4a is exactly where
  it gets validated or revised.

### 1.3 Horizon length

Given the replan rate above, a **0.3-0.6 s horizon** (i.e. 6-30 replan steps at 20-50 Hz,
or equivalently a few hundred ms of the underlying 500 Hz reference) is the right order of
magnitude: long enough to see a guard margin start closing before the reactive guard's own
window would (`ImpedanceSafetyMonitor`'s growth guards use a 100-consecutive-step counter
at 500 Hz = 0.2 s, `controller_core/safety.py:19-20`,
`max_x_error_growth_steps`/`max_axis_error_growth_steps`, both default 100 — so a horizon
shorter than ~0.2 s literally cannot see further ahead than the reactive guard already
does), short enough that a short-horizon iterative solve (§4b) stays cheap and the
open-loop dynamics prediction doesn't drift too far from reality between replans. This is
not long-horizon in the "plan the whole 10 s move" sense — it's a rolling short lookahead,
re-anchored every 20-50 ms to the real measured state, which also limits how much a
model-mismatch (the very real, currently-uncompensated real-vs-sim friction/stiction gap
documented in AGENTS.md §3's 2026-07-31 entries) can compound before it's corrected by the
next replan.

### 1.4 Dynamics model and cost function

**Model**: `PinocchioUR5eDynamics` (§0.2), already validated, already in-process for
`coriolis_feedforward`. No new dynamics code needed for the model itself.

**Cost function**, over the horizon, per candidate reference segment:
- **Tracking term**: `(x(t) - x_command(t))^2` against the *original* min-jerk/accel-profile
  target the user actually asked for (the planner corrects around the user's intended move,
  it doesn't replace it) — otherwise "maximum movement" regresses to "never move," the
  exact failure this repo already suffered six times in the RL gain-scheduling attempts
  (`docs/status/nonlinear_controller_research_2026-07-31.md` §4, "collapsed to never move,"
  0/20 through 0/8 across multiple reward formulations). This is the single most important
  design constraint given this repo's own history: **any objective term that can be trivially
  minimized by not moving is a proven failure mode here**, so the cost function must be
  structured so the tracking term dominates unless a margin is genuinely closing.
- **Guard-margin term, soft**: for each of `ImpedanceSafetyMonitor`'s hard thresholds
  (`max_abs_y_drift_m=0.03`, `max_abs_z_drift_m=0.03`, `max_abs_orthogonal_drift_m=0.03`,
  `max_orientation_error_rad=0.25`, `max_joint_velocity_radps=1.5` —
  `controller_core/safety.py:14-18`), a penalty that only activates within some fraction
  of the threshold (e.g. quadratic ramp starting at 60-70% of the hard limit, matching the
  existing precedent for a *warning* margin distinct from the hard limit —
  `recoverability_monitor.py`'s `x_warning_margin_m` pattern, §0.5 — even though that file
  isn't reusable code here, its margin-before-limit shape is the right one to copy). This
  is the mechanism that would let the planner steer away from a closing margin before
  `ImpedanceSafetyMonitor.check()` ever has to return `ok=False` — genuinely different from
  today, where nothing sees the margin until the hard threshold is already crossed.
- **Control-effort term**: `||tau||^2` or `||qdd||^2` penalty, standard, keeps the plan
  from chasing tracking/margin terms with unrealistic accelerations — also a hedge against
  reproducing the exact TCP-acceleration-spike failure this repo already found and fixed
  once (AGENTS.md §4, "Fixed and promoted to default, 2026-07-30": the `singular_scale`
  freeze-then-cram bug produced "a genuine... TCP acceleration spike 53x the nominal
  min-jerk profile's theoretical peak"). A planner is a new place that exact failure shape
  could reappear if effort isn't penalized.

### 1.5 Handoff without touching the 500 Hz budget

Two structurally different implementation choices, in increasing order of complexity —
recommend starting with the first:

- **(a) Same-process, cooperative scheduling.** The planner runs between control cycles
  (e.g. every Nth cycle, checked cheaply the way `deadline_monitor`/`stale_monitor` are
  already checked every cycle at the top of the loop
  — `hardware/direct_torque_transport.py`'s loop body) or on a fixed wall-clock cadence
  using the same `monotonic_ns()` pattern the loop already uses for `cycle_start_ns`. Risk:
  if a single replan iteration ever overruns, it steals a whole 500 Hz cycle. Needs a hard
  time budget *per replan call* (not just per horizon step) with a bail-out that reuses the
  previous buffer unchanged if exceeded — cheap to add given the loop already computes
  `lateness_ns` every cycle.
- **(b) Separate process, mirroring the residual observer's precedent.** Given this repo
  already solved "run something dynamics-heavy off the 500 Hz loop safely" once this week
  (`hardware/residual_observer_worker.py`, commit `56d230c`, moved the diagnostic residual
  compute into a `multiprocessing.Process`, fixed a real `Queue` feeder-thread deadlock and
  a real OpenBLAS-thread-stealing bug along the way — `docs/status/real_lab_session_2026-07-31.md`
  §3), this is a proven pattern to copy rather than reinvent, if profiling under (a) shows
  cooperative scheduling isn't enough headroom. The residual observer is diagnostic-only
  (never feeds control), so its concurrency bugs were low-stakes to discover; a planner
  process feeding the reference buffer is control-relevant, so the interprocess handoff
  needs its own staleness check (same idea as `StaleStateMonitor`, applied to "is the
  reference buffer's last-write time within N replan periods of now" — if not, fall back to
  the original closed-form `x_profile_target()` profile, never to a frozen/undefined
  buffer).

Either way, the 500 Hz loop's own per-cycle cost is unchanged: it still does one buffer
lookup (interpolation, O(1)) in place of one closed-form polynomial evaluation — not a
measurable difference from what it already does.

## 2. Would this help the two gain-tuning-exhausted failure modes?

**Directional ceiling** (+X passes further than -X at the same pose, AGENTS.md §3,
2026-07-27/28 entry) and **-45° Y-drift coupling** (AGENTS.md §3, 2026-07-31 entry) share
one documented root cause family: **the nullspace-projected posture term's own restoring
authority is asymmetric with pose/direction** — for the directional ceiling, the
projector's Frobenius norm "grows during the +0.20 m move... but shrinks monotonically
during -0.20 m move"; for the -45° case, "every failure tripping the identical guard at
the identical ~0.030 m magnitude" with "no candidate that fixes it" found across "a full
staged BO gain search plus targeted kd_joint smoke tests." Both are explicit that gain
retuning was tried and structurally can't fix this, because the asymmetry is in *how much
authority the projector has available*, not in what gain multiplies that authority.

**Honest assessment: a planner does not fix this, and here's the specific reason why, not
just "prediction is generally good."** The planner as scoped in §1 changes *what reference
the tracker chases* — it does not change the tracker's actuation mechanism. Both failure
modes are actuation-authority problems: even with a perfectly-anticipated reference (in the
limit, even if the planner somehow knew the exact moment the guard margin would close), the
same nullspace projector with the same shrinking Frobenius norm is still the only thing
available to hold orientation once whatever reference is commanded there. A planner
operating through `x_profile_target()`'s `(x_des, x_vel_des)` interface only ever commands
where the *task* (X-axis) target should be — it has no channel to reallocate authority
between the task-space wrench and the nullspace posture term, which is precisely the
resource that's asymmetric. **The one documented fix that actually worked for the
directional ceiling was structural** — `wrist_orientation_task`, a separate joint-space PD
term masked to the wrist chain, isolated from the shared wrench pipeline (`0.22-0.25 rad
-> 0.06-0.07 rad`, `docs/status/nonlinear_controller_research_2026-07-31.md` §0 row 6) —
which is a different *mechanism* (a new torque path), not a different reference. A
planner is closer in kind to "smarter gains/smarter reference" than to "new torque
mechanism," and this repo's own §4 ranked recommendation already makes exactly this
argument against gain-scheduling for the same bug: "A gain-scheduling layer, however it's
trained, is very unlikely to beat a fix that already directly addresses the actual
mechanism." The same logic applies here.

**Where a planner plausibly *would* help, modestly**: the guard-margin cost term (§1.4)
could make the planner slow down, shorten, or reshape the *commanded* move as a margin
closes — i.e. trade some of the "maximum movement" goal for staying inside the currently-
known-safe envelope, rather than committing to the full commanded displacement and hoping.
That is a real, if modest, value: it could reduce how often the -45° pose's dx=0.20 m case
crosses fully into a hard trip, by choosing (e.g.) a smaller effective displacement or a
slower approach automatically, informed by the predicted margin trajectory — but that is
qualitatively "the planner requests less of the thing that's known to fail," not "the
planner makes the failing thing succeed." AGENTS.md is explicit that the practical floor
for the -45° pose is "dx<=~0.04m passes cleanly... dx>=0.06m (sim)/dx>=0.20m (real) is a
known, reproducible failure with no current fix" — a margin-aware planner is a plausible
way to *automatically discover and respect that floor per-pose* rather than requiring a
human to already know it, which has real operational value, but it is not a fix for the
floor itself.

## 3. Coriolis / high-speed / abrupt-reversal connection

**Today's `coriolis_feedforward` (`hardware/direct_torque_transport.py:447-448`,
`local_dynamics.jacobian_mass_and_coriolis(link_state.q, link_state.qd)`) is a pure
one-shot feedforward**: every cycle it evaluates `C(q, qd) @ qd` at the *currently measured*
`qd` and adds that as `tau_coriolis` on top of the controller's torque
(`hardware/direct_torque_transport.py:483`, `tau = tau_controller + tau_coriolis`). It is
reactive in the precise sense that it can only compensate for the Coriolis/centrifugal
coupling that the *current* velocity state already implies — it has no model of where `qd`
is headed. AGENTS.md §3 already notes this term was validated as "historical lane never
compensated C(q,qd)qd; measured negligible below ~0.5 rad/s" — i.e. its value has only ever
been checked at low speed, exactly where Coriolis effects are smallest and a one-shot
term is closest to sufficient.

**Where a predictive planner using the same `C(q,qd)` model could concretely do more,
ahead of a fast reversal**: `C(q, qd)@qd` is quadratic in `qd` and its *sign* structure
depends on the sign of `qd` itself — at a reversal, `qd` crosses zero and its sign (and
therefore the coupling torque's direction on other joints) flips. A one-shot feedforward
computed from the *current* `qd` necessarily lags this: at the instant just before a
reversal, `qd` is still (small, but) signed in the old direction, so `tau_coriolis` reflects
the old-direction coupling right up until `qd` has actually started changing sign — by
construction, it cannot anticipate the flip, only react to it a cycle (or more, given
sensor/estimation lag) after it starts. A planner rolling `PinocchioUR5eDynamics` forward
over the §1.3 horizon, using the *planned* `qd(t)` trajectory (not just the measured
current one), evaluates `C(q(t), qd(t))@qd(t)` at future points on the planned reversal —
i.e. it can compute the sign-flipped coupling torque *before* the reversal happens and bias
the reference/feedforward schedule accordingly, rather than only after the real robot's
`qd` has already crossed zero. This is a genuine, mechanistically distinct advantage over
the one-shot term for exactly the abrupt-reversal case, *because* reversal is precisely
where "coupling depends on current qd" (reactive) and "coupling depends on qd a few cycles
from now" (predictive) diverge most — away from a reversal, `qd`'s sign is stable and the
one-shot term is a much closer approximation to what's actually needed a cycle later.

**Honest caveat, and the cheaper alternative that should be tried first**: this advantage
is real but has never been measured in this repo — `coriolis_feedforward`'s only validation
to date is "negligible below ~0.5 rad/s," i.e. *at* low speed, and this system's own
canonical validated envelope (AGENTS.md §3) is built from move durations of 1 s+ (1-30 s
holds), which are far from the "abrupt reversal at speed" regime this section is about. A
smaller, much cheaper experiment should come first and would likely capture most of the
value: **actually test `coriolis_feedforward` at the higher speeds and reversal profiles
the user is asking about, in sim, before building any planner.** If the one-shot term
already tracks well through a reversal at the target speeds (plausible, since even a
one-cycle-lagged correction at 500 Hz is only a 2 ms lag relative to typical UR5e
joint-acceleration timescales), a planner's predictive advantage may be marginal in
practice despite being real in theory. If that sim test instead shows a measurable
tracking or guard-margin degradation specifically around reversals that the one-shot term
can't close, *that* result is the concrete justification for building the predictive
version — not the a priori argument alone. This test does not exist yet in this repo's
history (no sim run at elevated speed/reversal profiles with `coriolis_feedforward` is
cited anywhere in AGENTS.md or docs/status/) and is cheap: it needs only a reversal
trajectory profile (the `accel_duration_triangular`/`accel_duration_scurve` profiles added
in commit `c1b52fe` already support arbitrary accel/duration — a reversal is just two back
to back opposite-sign segments) and the existing `--coriolis-feedforward` flag, no new
code.

## 4. Implementation cost/risk, staged

### (a) Offline/sim-only prototype — what it needs, before any real-time integration

- A pure-Python/numpy prototype (like `tools/diagnostics/kalman_tcp_accel_filter_prototype.py`'s
  precedent — offline, reads real/sim data, computes no decision wired into any real guard
  path) that: (1) reconstructs the §1.4 cost function, (2) runs the chosen short-horizon
  solver (§4b) against `PinocchioUR5eDynamics`, (3) replays it against the **already-known
  failure cases** — the -45° dx=0.20m Y-drift trip and the directional-ceiling dx=0.20-0.25m
  cases both have committed sim reproductions per AGENTS.md §3 — to see, offline, whether
  the planner's chosen reference differs meaningfully from the closed-form one and whether
  that difference would have kept the guard margin from closing, *before* touching any
  control loop.
- This is the step that should settle §2's prediction (planner likely doesn't fix the
  structural asymmetry) and §3's prediction (planner may help reversals, cheaper test
  should go first) with actual numbers instead of the reasoning-only arguments above.
- Success criterion for even continuing past this stage: the offline prototype needs to
  beat the *already fixed* baselines (`wrist_orientation_task` for the directional ceiling,
  a tuned `coriolis_feedforward` for reversals) on the same cases, not just beat "no fix at
  all" — otherwise it's solving an already-solved problem with more machinery.

### (b) Making it fast enough to matter — solver choice

- **Full nonlinear MPC (multi-shooting, general NLP solver)**: almost certainly overkill
  and too slow to justify even at 20-50 Hz on this hardware, and brings in a new heavy
  dependency (an NLP solver) this repo doesn't currently have — not recommended as the
  first cut.
- **iLQR / short-horizon DDP**: a real, well-understood middle ground — quadratic-ish
  cost (§1.4's terms are all near-quadratic away from the margin ramps), local
  linearization of `PinocchioUR5eDynamics` each iteration, horizon of 6-30 steps (§1.3).
  Feasible in pure numpy, no new heavy dependency, and there is a close structural
  precedent already in this exact codebase for "local iterative linear-algebra solve, no
  external solver" — `controller_core/torque_task_qp.py` + `box_qp.py` already do
  something in this spirit (a QP-based torque law) for the *current-cycle* torque
  allocation, so the team already has practice building and validating this class of
  numerical routine to this repo's standards.
- **The cheapest option, and the one to build and validate FIRST**: a "look N steps ahead
  with the existing dynamics model and nudge the reference if a predicted guard margin is
  closing" heuristic — no optimization loop at all, just forward-simulate the *existing*
  closed-form profile with `PinocchioUR5eDynamics`, check whether any margin in §1.4's list
  crosses its soft threshold within the horizon, and if so scale back the commanded
  displacement/duration (a single scalar knob, not a full re-plan) before handing it to
  `x_profile_target()`. This captures most of §2's honestly-assessed real value (steering
  away from a closing margin) with a small fraction of the engineering/validation cost of
  iLQR, and is a much smaller, much more reviewable diff against this repo's existing
  pattern of small, additive, flag-gated changes. **Recommended starting point if this
  line of work is pursued at all.**

### (c) Real-hardware validation requirements

Whatever the design, real-hardware rollout has to clear the same bar every other
control-relevant change in this repo has: typed-confirmation motion gating (already
enforced, `--i-understand-this-moves-the-robot`), flag-gated/opt-in (not a default-on
change to `x_profile_target()`'s existing profiles), the existing `ImpedanceSafetyMonitor`
+ `CartesianMoveMonitor` guard stack must remain the actual backstop (per
`docs/status/nonlinear_controller_research_2026-07-31.md` §5's explicit warning: "relying
on the guards to catch a bad [new component's] output after the fact... is exactly what
happened six times" in the RL history), and the same 4-category rigor sweep this repo
already uses for every controller change before any real-arm exposure. Given this repo's
"friction feedforward validated in sim 2026-07-31, still not validated on real hardware as
of this doc" precedent, expect a real planner to need its own dedicated real-lab session
after full sim validation, not a same-night landing.

## 5. Recommendation

**Do not build this next.** Sequence, in order of actual leverage for this repo's stated
goals ("maximum movement and faster movement," handling Coriolis/reversals):

1. **Let the two currently-running background efforts land first** (a safety-constrained
   nullspace-handling search and a Kalman-filtering sensor-noise investigation — both
   referenced in this session's brief; neither has a landed doc as of this writing,
   `docs/status/kalman_filtering_sensor_noise_2026-08-01.md` is referenced by
   `tools/diagnostics/kalman_tcp_accel_filter_prototype.py` but does not yet exist on disk).
   The nullspace search in particular targets the exact mechanism (§2) that a planner
   cannot reach through `x_profile_target()`'s interface — if it produces a structural fix,
   it directly obsoletes part of the motivation for a margin-aware planner (the
   directional-ceiling/-45° cases). No need to duplicate or race that work.
2. **Validate `coriolis_feedforward` at actual high-speed/reversal profiles in sim before
   building anything predictive** (§3) — cheap (existing flag, existing
   `accel_duration_triangular`/`scurve` profiles, no new code), and is the test that turns
   this document's Coriolis argument from "plausible reasoning" into "measured need or
   measured non-need." This is genuinely a few hours of sim runs, not a new subsystem.
3. **Only if step 2 shows a real, reversal-specific gap the one-shot feedforward can't
   close**, prototype the cheapest option from §4b (the margin-lookahead heuristic, not
   iLQR, not full MPC) offline against the known failure cases from §4a, and re-assess.

**Strongest argument for pursuing this anyway, stated honestly**: the trajectory-generator/
tracker split (§0.1) really is an unusually clean seam for exactly this kind of addition —
most of the engineering risk other systems would have (rewriting a real-time loop, new
solver dependencies, unclear handoff boundaries) is genuinely absent here, and the
guard-margin-aware cost framing is a mechanism this system has never had before (today
nothing sees a margin until `ImpedanceSafetyMonitor` already returns `ok=False`). If the
goal is specifically "catch problems before the reactive guard has to," this is a
real, structurally-motivated way to do that, not a speculative one.

**Strongest argument against, stated honestly**: neither of this system's two headline,
gain-tuning-exhausted real failures would plausibly be fixed by it (§2's specific,
mechanism-level argument, not a hand-wave), the Coriolis case it's most directly motivated
by has a cheaper test that hasn't been run yet and could show the reactive term is already
sufficient, and this repo has a recent, direct precedent for what happens when a new
correction layer is added before its target problem is well-characterized: six RL
gain-scheduling attempts that all failed against the same structural bug a planner would
also not reach. Building a planner now would be adding real complexity (a new
async/offline component, a new numerical solver, a new real-hardware validation cycle)
ahead of two cheaper, more targeted pieces of evidence that would each independently tell
us whether it's even likely to help. **The juice isn't worth the squeeze yet — not because
the idea is bad, but because this repo already has two cheaper experiments that would each
answer "is this worth it" more directly than building it speculatively would.**

## What was NOT done, and why

No code was written or modified. No new config files were added. `controller_core/` and
`hardware/` were read-only for this task. No sim runs were executed (the recommendation in
§5.2 is a proposed next experiment, not one run as part of this document). This is
consistent with the task's explicit "design/research only" scope.

## Rollback

N/A — this commit adds only this document.
