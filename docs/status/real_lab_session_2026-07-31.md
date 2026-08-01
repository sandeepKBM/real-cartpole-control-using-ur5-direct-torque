# Real-lab session 2026-07-31 — findings

Real UR5e (172.16.71.77), `direct_torque` control mode, operated from `thinkrobot`. All
numbers below are from real robot runs, not simulation, unless explicitly marked (sim).

## 1. Root cause found: -45° base rotation, not a controller/gain defect

**Background**: `HEIGHT_ALPHA_0_5_CLEARANCE_Q` (shoulder_pan = -45°) was added this session
as the new default start pose for real transport, needed for real wall/base clearance
(visually confirmed twice). `hardware/x_transport.py` and `tools/ur5e_move_joints.py` were
updated to default to it.

**Symptom**: a dx=0.20m X-axis transport at the -45°-rotated pose reproducibly tripped
`ImpedanceSafetyMonitor`'s Y-drift guard (`|Y-Y0| > 0.03 m`) on 4+ separate real attempts, at
almost identical magnitude every time (~0.0300-0.0300 m), with real TCP motion at close to a
45° diagonal (X and Y displacement nearly equal magnitude, e.g. one run: Δx=+0.0300m,
Δy=-0.0300m).

**Two targeted fixes tried live, both had zero effect** on the trip point:
- `kp_y: 80→120, kd_y: 15→22.5` (50% increase) — trip still at 0.030025 m.
- `config/ur5e_mujoco_torque_osc_tuned_diagonal_lambda.yaml`
  (`lambda_diagonal_shaping: true`, the fix already validated for an analogous Λ_xz leak at
  this same pose family per AGENTS.md §3) — trip still at 0.030020 m, same near-45° diagonal.

**Confirming test**: the same dx=0.20m move at shoulder_pan=0 (un-rotated), same plain
`config/ur5e_mujoco_torque_osc_tuned.yaml`, no gain overrides, 10s move duration: **succeeded
cleanly** — `achieved_x_delta_m=0.189` (94.5% of target), `move_phase_max_abs_y_drift_m=8.2e-6`
(essentially zero), `success: true`, `safety_pass: true`.

**Conclusion**: the -45° base rotation itself — a real kinematic/Jacobian-conditioning effect
of that specific rotated pose — is the actual cause, not a controller gain or Λ-shaping defect.
Two follow-up real runs at shoulder_pan=0 with shorter move durations (4s, 6s) "failed," but on
an unrelated guard (`TCP speed > 0.05 m/s`, since 0.20m/4s sits right at that ceiling) — their
own Y-drift stayed negligible (~7-8e-6 m), so they don't contradict the above.

**Status**: user has decided to keep the -45° rotation for real use (needed for wall clearance,
wants faster/better transport there) rather than reverting to 0°. A background task was
launched same night to add `--start-q-rad` support to `tools/tune_ur5e_residual_impedance_transport.py`
(previously had none) and re-validate/retune the controller specifically at -45° in sim — see
that task's own findings doc once it lands (not yet complete as of this doc).

## 2. Real friction/stiction sim-to-real gap (open, unquantified)

Multiple real runs this session showed substantial commanded torque (5-8 Nm) producing far
less real displacement than sim would predict, with steady-state hold-phase torque **not**
decaying toward zero — a force-balance-against-friction signature, not a transient. Confirmed
via direct grep: `assets/ur5e_torque/ur5e_torque.xml`'s joint `<default>` class sets
`armature="0.1"` but has **no `frictionloss` or `damping` attribute anywhere** — real joint
friction is completely unmodeled in this sim. `kp_x=600` (vs default 400) measurably improved
small-displacement authority (dx=0.04m: 55%→72% of target) but was never validated at large
displacement before being combined with other changes, and is not a fix for the underlying gap.

A separate background task was launched to add real friction modeling
(`frictionloss`/`damping` in the MJCF, informed by tonight's torque/displacement signatures) —
see that task's own findings doc once it lands.

## 3. Async residual observer (landed, commit `56d230c`, not pushed)

Diagnostic-only residual observer (predicts `qdd` from known dynamics + commanded torque,
compares to measured `qdd`, never read by any safety guard) was moved off the 500Hz
`direct_torque` control loop into a separate `multiprocessing.Process`, opt-in via
`--residual-observer-async` (default off, byte-identical to the prior sync path — proven via a
deterministic fake-clock trace diff, not eyeballed). Real measured win: `residual_observer`
phase mean 0.0721ms (sync) → 0.0177ms (async), ~75% cut. Two real bugs found and fixed during
validation: (1) `shutdown_and_collect` joining the worker before draining its result queue — a
documented `multiprocessing.Queue` feeder-thread deadlock gotcha, truncated results to 240/500
in first full test; (2) the spawned worker let OpenBLAS grab all 72 cores, stealing real-time
CPU from the parent loop — pinned to 1 thread for the child. Full suite: 430→437 passed
(`-m "not slow"`), 439 total. Clean process teardown verified on normal exit, mid-run exception,
and mid-run `RTDEStateError`.

**Real-time alerting on residual threshold** (proposed by the user as a possible follow-on:
"hits the alert only when it finds something off") is explicitly deferred — documented here as
a future-work item, not implemented. Concretely: today the async worker only merges results
back into `trace_rows` post-run for offline inspection; it does not evaluate the residual
against any threshold live, and nothing currently escalates a large `qdd_residual_norm` back to
the main control loop or an operator-visible signal during the run. Any implementation would
need: (a) a threshold config value (not yet chosen — needs real data on what residual
magnitude is actually anomalous vs. normal tracking-transient noise, e.g. by looking at the
`qdd_residual_norm` distribution across tonight's own real traces first), (b) a decision on
what "alert" means in practice (log line only vs. something that could plausibly trip a guard
— the latter would be a real safety-relevant design decision requiring its own review, not a
quick add), and (c) a non-blocking path back from the worker process to the main loop faster
than the current post-run merge (e.g. a small dedicated alert queue polled once per cycle,
separate from the bulk result queue, kept cheap enough not to reintroduce the latency this
change just removed).

## 4. Direct-torque loop timing/inefficiency investigation (2 background tasks)

- **Controller-phase profiling** (task #99, no code changed): no single operation dominates
  `compute()`'s ~0.28ms mean cost on westeros; ~31% attributable to identifiable matrix math
  (cond(J)/SVD, quaternion orientation pipeline — flagged as having 3 redundant normalizations,
  untouched; Lambda-shaping; nullspace), ~69% aggregate small-call Python overhead. Confirmed
  `cond(J)` is still needed as real telemetry even where `singular_scale` itself is inert.
  Doc: `docs/status/direct_torque_controller_phase_profiling_2026-07-31.md`.
- **Tail-latency investigation** (task #100, commit `f1748c1`): found the (then-still-sync)
  residual observer ran after `time.sleep()`, so its cost leaked into the *next* cycle's
  measured lateness — informed the async-observer work in §3 above. Found a separate, harmless
  cycle-0 double-period-sleep scheduling bug (flagged for human review, not fixed — touches
  real safety-relevant timing logic). Confirmed `gc.disable()` (already applied) is genuinely
  effective: 46.5μs max with it vs 646.5μs without.

## 5. Deadline-overrun trip, real, resolved same night

One real run tripped `DeadlineMonitor` ("3 consecutive cycles late by > 1 ms; latest 1.52 ms")
with the (then-synchronous) residual observer enabled. Disabling it via
`--disable-residual-observer` eliminated the trip (max_lateness_ms: 1.404→0.195), confirming
the hypothesis before any code changed. Superseded by the async-observer fix in §3, which
addresses the same root cause without needing to disable the diagnostic entirely.

## Rollback

- Async residual observer: `git revert 56d230c` (not pushed yet, so a plain revert or dropping
  the commit both work).
- Nothing in this doc set changed any real-hardware safety threshold.
