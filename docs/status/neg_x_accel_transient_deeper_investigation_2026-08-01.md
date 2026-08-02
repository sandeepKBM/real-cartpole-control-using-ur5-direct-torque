# -X acceleration transient, deeper investigation (2026-08-01, night follow-up)

Follow-on to `docs/status/session_compilation_2026-08-01_night.md` §6. That doc's first pass
checked `jacobian_cond`, `tau_controller`, `x_error`, and `orientation_error_norm` only, over
the last 10-60 rows of three real-hardware trials at the identical command
(`config/ur5e_mujoco_torque_osc_tuned_split_base_wrist.yaml`, `accel_duration_scurve`,
`target_accel=-0.015`, `move_duration=8.0s`, `q=[0.0, -0.835398, -1.2, -0.985398, 0.0, 0.0]`),
and concluded trial 3 (speed trip) plausibly matches the already-understood `+X`
orientation-growth mechanism (§4 of that doc) while trials 1-2 (both accel trips) remained
genuinely unexplained. This pass reads every field in every row of all three full traces, adds
sim reproduction, and reaches a materially different conclusion: **trials 1 and 2 are not one
unexplained phenomenon — they are two different mechanisms, one of which (trial 1) is not
actually a mystery.**

Traces: `outputs/hardware_transport_remote/hardware_transport/direct_torque_20260801_{202215,202639,204446}/trace.jsonl` (trials 1/2/3 respectively). All analysis below is offline trace/sim
analysis; no `controller_core/`, `hardware/safety.py`, or existing config was touched. Ad-hoc
analysis scripts used for this pass live only in the session scratchpad, not the repo (nothing
here was judged reusable enough to add to `tools/diagnostics/`).

## 1. What was checked beyond the first pass

Full-trace (not just the trip window), all fields present in the real `direct_torque` trace
schema, for all three trials:

- `cycle_work_ms` / `lateness_ms` (control-loop timing) — full trace, not just near-trip.
- `qdd_residual`/`qdd_residual_norm` (per-joint and aggregate), `qdd_measured` vs `qdd_pred`.
- `tau_coriolis`, `coriolis_feedforward_active`.
- `task_backtrack_iters`, `task_scale`, `singular_scale`.
- Per-joint `q`/`qd` (not just the aggregate norm already checked).
- `torque_saturation_percentage` from each `summary.json`.
- Each run's `pre_trip_trend` block (`summary.json`) — a 60-cycle window of `qd_max_radps`,
  `tcp_speed_mps`, `x_error_m`, `tau_controller_l1`, `orientation_error_norm_rad`, `y_drift_m`,
  `z_drift_m`, added by the 2026-07-31/08-01 pre-trip diagnostic capture (AGENTS.md §4) — not
  read at all in the first pass.
- Raw `ee_pos`/`tcp_pose` telemetry: checked for exact-duplicate consecutive rows (frozen RTDE
  frames) across each entire trace.
- Sim reproduction of the exact scenario (`tools/ur5e_mujoco_torque_experiments.py --mode
  controller-rollout`, same config/pose/profile/accel/duration, `--enable-tcp-accel-guard
  --tcp-accel-guard-noise-robust`), with a field-by-field comparison against the real traces.

## 2. Ruled out again, on more direct evidence than the first pass

- **Wrist singularity**: `jacobian_cond` stays in the same low, stable 5.1-7.8 band in all three
  full traces (not just near the trip) and in the sim reproduction. Confirmed, not just at the
  trip cycle.
- **Task-space saturation / geometric backtracking**: `task_scale` and `singular_scale` are
  pinned at exactly `1.0` for literally every cycle in all three traces; `task_backtrack_iters`
  is `0` for every cycle in all three traces; `torque_saturation_percentage: 0.0` in all three
  `summary.json`. No headroom pressure anywhere, at any point in any run — this rules out
  saturation-driven torque discontinuities as a candidate for any of the three trials, not just
  near their respective trips.
- **Control-loop timing overrun**: `cycle_work_ms` (mean ~0.73-0.78ms, max ~1.2-1.3ms) and
  `lateness_ms` (mean ~0.17-0.20ms, max ~0.34ms) are statistically indistinguishable across all
  three trials, including a small periodic ~0.3ms lateness ripple that is present **equally in
  all three trials** (not unique to 1 or 2) — most likely a background OS/scheduler artifact,
  not something correlated with the accel-guard trips. No spike, ramp, or anomaly in either
  field anywhere near any trip. A genuine per-cycle compute-time or scheduling glitch producing
  a bad torque command is ruled out as an explanation for trials 1 or 2.
- **The scurve `t=T/2=4.0s` jerk-extremum hypothesis, re-examined more rigorously**: trial 1
  trips at `t=3.93s`, only 70ms before `T/2=4.0s`, close enough that the first pass's
  cross-trial-timing refutation (trial 2 trips at `t=1.84s`) doesn't fully settle whether trial 1
  specifically is a `T/2` artifact. Checked directly this time: `target_accel` (finite-differenced
  from the trace's own `target_x`) shows no discontinuity, kink, or localized feature anywhere
  near `t=3.93-4.0s` — it is smoothly, monotonically decaying through the entire window exactly
  as the `sin(2*pi*t/T)` profile predicts, with no anomaly. Trial 3's trip (`t=3.50s`, 500ms from
  `T/2`) shows the same smooth growth shape. This closes the `T/2` hypothesis on direct evidence
  from the trajectory shape itself, not just cross-trial timing dispersion.
- **Coriolis feedforward**: `tau_coriolis` is exactly `[0,0,0,0,0,0]` and
  `coriolis_feedforward_active` is `False` for every single cycle in all three real traces
  (5906 rows combined) — ruled out as a factor in any of the three trials, but see §4 for why,
  which is itself a real finding worth flagging.

## 3. Trial 2 — explained: real RTDE telemetry staleness, not a physical event

Checked exact-duplicate consecutive `ee_pos`/`tcp_pose` rows (a frozen/re-served RTDE frame)
across each full trace:

| trial | duplicate-frame count | % of trace | last duplicate vs. trip time |
|---|---|---|---|
| 1 (`_202215`) | 0 / 1967 | 0.00% | n/a |
| 2 (`_202639`) | **107 / 919** | **11.64%** | t=1.834s (trip at t=1.836s — 1 cycle before) |
| 3 (`_204446`) | 2 / 1750 | 0.11% | t=0.132s (not concentrated near trip at t=3.50s) |

Trial 2's `pre_trip_trend.tcp_speed_mps` (60-cycle window, matches
`CartesianMoveMonitor.check()`'s own single-cycle position-delta/dt formula) shows the smoking
gun directly: `[..., 0.0609, 0.0, 0.0345, 0.0528, 0.019, 0.0, 0.074]` — repeated exact-zero
readings (stale frame, no reported displacement) immediately followed by an oversized jump
(the next fresh frame "catching up" on two cycles' worth of real displacement at once). This
zero-then-spike pattern occurs 12 times in just the last 60 cycles before the trip. Trials 1 and
3's `tcp_speed_mps` windows are smooth and monotonic with no zeros at all (see the compilation
doc's own excerpt, confirmed against the full 60-cycle arrays here).

This is a self-consistent explanation for every other trial-2-specific oddity already noted in
the compilation doc: orientation error at trip is only 0.0053-0.006 rad (an order of magnitude
below trials 1 and 3), `x_error` is flat/converged (not diverging), `tau_controller` is falling,
and only 3% of the target displacement was achieved — nothing physically dramatic was actually
happening in trial 2. A stale-then-fresh position pair, double-differenced for the accel guard's
own calculation, synthesizes a large spurious acceleration reading from what was, physically, a
slow, unremarkable, still-early-in-the-move motion. This matches the already-documented UR RTDE
behavior in AGENTS.md §4 (2026-07-28 finding: "two real RTDE read stalls, most likely the
documented UR behavior of the robot controller deprioritizing telemetry under its own load, not
a bug in this codebase") — trial 2 is a third occurrence of the same class of event, just dense
enough this time (11.6% of an admittedly short, 919-cycle run) to trip a live safety guard
instead of merely being noted as a stall.

## 4. Trial 1 — reclassified: same mechanism as trial 3, not a separate mystery

The first pass grouped trials 1 and 2 together as "both unexplained accel trips" and contrasted
them against trial 3 ("understood... orientation-growth mechanism"). Comparing trial 1 against
trial 3 directly (not just against trial 2) shows they look alike, not different:

| | trial 1 (`_202215`, accel trip) | trial 3 (`_204446`, speed trip, understood) |
|---|---|---|
| orientation error at trip | 0.083-0.085 rad | 0.066 rad |
| orientation growth shape | smooth, monotonic over the whole move | smooth, monotonic over the whole move |
| `qd_norm` shape before trip | dips to a local minimum (0.016 rad/s at t=3.826s), then ramps smoothly to 0.094 rad/s by t=3.930s | smoothly increasing |
| duplicate RTDE frames | 0 | 2 (not concentrated near trip) |

Trial 1's `qd_norm` minimum-then-ramp shape (0.016 -> 0.094 rad/s over ~100ms, monotonic, no
noise/discontinuity — verified row-by-row) is a real, smooth kinematic event, not a sensor
artifact (zero duplicate frames anywhere in this trace). Its raw single-cycle TCP speed
(`pre_trip_trend.tcp_speed_mps`) already touches 0.0516 m/s — just over the 0.05 m/s **speed**
guard's own threshold — in an isolated cycle before the trip, but apparently didn't sustain 3
consecutive over-threshold cycles before the **acceleration** guard's independent 3-consecutive
criterion fired first. That is a mundane guard-race explanation (both guards were approaching
their thresholds together, driven by the same growing-orientation/growing-speed event; which one
crosses its own consecutive-cycle criterion first is a matter of a few cycles of noise-timing) —
consistent with, not contradicting, the session compilation's own §6 remark that "which one trips
first varying run to run" is expected of a shared underlying cause.

**Revised read: trial 1 is very likely the same `-X` counterpart of the already-understood §4
orientation-growth mechanism as trial 3, just crossing the accel guard a few cycles before the
speed guard would have, rather than a third, distinct phenomenon.** No timing glitch, no
saturation, no Coriolis anomaly (it's off — see §5), nothing else in trial 1's full trace stands
out as anomalous relative to trial 3's.

## 5. Real, but non-differentiating, side finding: Coriolis feedforward is off on all three real trials, despite the config

`config/ur5e_mujoco_torque_osc_tuned_split_base_wrist.yaml` sets `mujoco.coriolis_feedforward:
true`. All three real traces show `coriolis_feedforward_active: false` and `tau_coriolis`
identically zero every cycle. Traced to the actual wiring
(`hardware/direct_torque_transport.py`, `hardware/x_transport.py`,
`tools/ur5e_direct_torque_x_transport.py`): the real-hardware `direct_torque` path only enables
Coriolis feedforward via an explicit `--coriolis-feedforward` CLI flag (`default=False`) — it
never reads `mujoco.coriolis_feedforward` from the config at all. This is different from the sim
path (`tools/ur5e_mujoco_torque_experiments.py`), whose own `--coriolis-feedforward` help text
says it "overrides `mujoco.coriolis_feedforward`" — implying sim honors the config value by
default when the flag is omitted, which a same-scenario sim reproduction run here confirms
behaviorally (see §6). **This means the config's stated Coriolis-feedforward intent silently does
not take effect on real hardware unless the CLI flag is separately passed** — a real
config-vs-implementation gap, worth a maintainer decision (either wire the config field through
on the real-hardware path, or note the CLI-flag-only real-hardware behavior explicitly in the
config's own comments). It does **not** differentiate trials 1/2/3 from each other (off in all
three identically), and P2's own historical note (AGENTS.md §3: "measured negligible below ~0.5
rad/s") suggests it's unlikely to be a first-order driver of any of these trips regardless
(`|qd|` never exceeds ~0.098 rad/s in any of the three real trials).

## 6. Sim reproduction of the -X scenario

Ran the identical config/pose/profile/accel/duration in sim (`tools/ur5e_mujoco_torque_experiments.py --mode controller-rollout --controller-kind impedance --config
config/ur5e_mujoco_torque_osc_tuned_split_base_wrist.yaml --start-q-rad 0.0 -0.835398 -1.2
-0.985398 0.0 0.0 --trajectory-profile accel_duration_scurve --target-accel -0.015
--move-duration 8.0 --duration 10.0 --transport-axis-index 0 --enable-tcp-accel-guard
--tcp-accel-guard-noise-robust --seed 0 --no-plot`):

- **Sim also trips the (opt-in, diagnostic-only) TCP-accel guard**: `TCP acceleration 0.6487
  m/s^2 > 0.5 m/s^2 for 3 consecutive cycles` at `t=4.574s`, 60% of target achieved — a
  different magnitude and timing from both real trials (trial 1: t=3.93s/41%/0.5112 peak; trial
  2: t=1.84s/3%/0.7042 peak), consistent with the already-documented sim-vs-real fidelity gap
  for this failure class (session compilation §7: the `+X` sim TCP-accel-guard smoke test also
  didn't reproduce the real magnitude). Sim should not be trusted to predict exact trip
  timing/magnitude for this class of event.
- **Qualitatively, however, sim's trip signature matches trial 1's family, not trial 2's**:
  `orientation_error_norm` grows smoothly and monotonically to 0.1245 rad by the trip (same
  shape as trials 1 and 3); `qd_norm` sits near a local minimum (~0.032 rad/s at t=4.534s) and
  then ramps smoothly to 0.098 rad/s over the next 40ms right as the trip fires — the same
  minimum-then-ramp shape independently found in trial 1's real trace (§4). `jacobian_cond`
  stays in the same low 5.1-7.8 band throughout. Zero duplicate consecutive `ee_pos` rows in the
  sim trace, as expected (sim has no RTDE telemetry to go stale) — sim cannot and does not
  reproduce anything resembling trial 2's signature.
- **Practical read**: sim can reproduce a qualitative version of the trial-1/trial-3 family
  mechanism (a genuine, control-software/dynamics-attributable orientation-growth-driven
  velocity transient), even though it cannot match real magnitude/timing. Sim categorically
  cannot reproduce trial 2's mechanism, because that mechanism is a real RTDE hardware/telemetry
  artifact with no sim-side equivalent — this is itself supporting evidence that trial 2 is a
  hardware/telemetry phenomenon, not a control-software bug, since a real software bug in the
  shared `controller_core`/trajectory code would be expected to show up in sim too.

## 7. Bottom line

- **Trial 2 is explained**: real RTDE telemetry staleness (11.6% duplicate consecutive telemetry
  frames throughout the run, dense right up to the trip) synthesizes a spurious TCP-acceleration
  reading via finite differencing, while the real underlying motion was slow and unremarkable
  (orientation error only 0.006 rad, 3% achieved). Not a controller-software or dynamics issue;
  matches previously-documented UR RTDE staleness behavior (AGENTS.md §4, 2026-07-28).
- **Trial 1 is very likely not a separate mystery**: its orientation-error magnitude and growth
  shape, and its `qd_norm` minimum-then-ramp signature, match trial 3's already-understood `+X`
  (here `-X`) orientation-growth mechanism closely, and a sim reproduction shows the same
  qualitative `qd_norm` and orientation-error signature. It most likely tripped the accel guard
  instead of the speed guard on this particular run due to which guard's own 3-consecutive-cycle
  criterion happened to fire first, not because of a distinct underlying cause.
- **Revised verdict for session-compilation §6**: there is no longer a genuinely unexplained
  accel-transient mechanism specific to `-X` at this pose/config. What looked like "two
  overlapping mechanisms, trials 1-2 vs trial 3" is better described as **one mechanism (the §4
  orientation-growth effect, already understood, showing up via whichever guard's cycle-count
  criterion trips first) plus one unrelated, previously-documented hardware artifact (RTDE
  telemetry staleness) that happened to coincide with trial 2**.
- **New, real, separately-actionable finding**: `config/ur5e_mujoco_torque_osc_tuned_split_base_wrist.yaml`'s `mujoco.coriolis_feedforward: true` does not take effect on the real-hardware
  `direct_torque` path (needs an explicit `--coriolis-feedforward` CLI flag not used in these
  three runs) — confirmed empirically (`tau_coriolis` identically zero, `coriolis_feedforward_active: false`, all three real traces) and by reading the wiring. Does not affect this
  investigation's conclusions but is worth a maintainer decision.

## 8. What would still improve confidence

- A real-hardware retest of this exact scenario with `--noise-robust-guards`
  (`accel_gap_cycles`/`speed_lowpass_alpha`) would directly test whether trial-2-style
  telemetry-staleness spurious trips go away, without changing the class of `-X` transport
  ceiling the §4 mechanism itself imposes.
- The §4 orientation-growth mechanism itself remains unfixed (its own concrete pointer,
  `kp_rot_wrist` retuning, is still open per AGENTS.md/session compilation §10 item 4) — this
  investigation only reclassifies trial 1 as an instance of that already-known problem, it does
  not solve it.
