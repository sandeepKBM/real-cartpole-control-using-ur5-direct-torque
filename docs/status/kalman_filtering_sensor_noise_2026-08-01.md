# Kalman filtering for TCP accel-guard sensor noise -- research, design, offline prototype (2026-08-01)

## Task and constraints

Research/design only, plus a small offline prototype. **No changes to `hardware/safety.py`'s
guard logic or defaults.** No live hardware access exists right now regardless. This doc plus
`tools/diagnostics/kalman_tcp_accel_filter_prototype.py` are the only artifacts.

## 1. What the current approach actually is, and what was actually measured (not assumed)

`hardware.safety.CartesianMoveMonitor` estimates TCP acceleration as a **double finite
difference of raw RTDE position**: `speed = |Δpos|/dt`, `accel = |Δspeed|/dt`. This amplifies
raw position noise by ~1/dt^2 (documented in the class's own docstring: ~15,600x at 125 Hz,
~250,000x at `direct_torque`'s 500 Hz).

**Real measured noise floor** (`hardware_captures/2026-07-28_thinkrobot_172.16.71.77/
stationary_noise_capture_154018_stats.json`, cross-checked directly against the raw capture
file `outputs/hardware_state_noise/capture_20260728_154018.jsonl` -- both present in this
checkout, 4730 real samples, 10s, robot stationary, height_alpha=0.5/wrist_2=0.1 pose):

| quantity | measured |
|---|---|
| `tcp_pos_std_m` (x, y, z) | `[8.88e-06, 9.86e-06, 3.25e-06]` |
| `q_std_rad_max` | `1.695e-05` |
| `qd_std_radps` (all joints) | exactly `0.0` -- RTDE's own `qd` reads a hard deadband at rest, not raw sensor noise |
| sample rate | nominal 500 Hz; **median measured inter-sample dt in this specific capture is 2.119 ms (471.9 Hz)**, not exactly 500 Hz -- real jitter in the capture pipeline itself, distinct from the guard's own logic |
| accel-estimate noise floor (median, `gap=1`) | 1.74 m/s^2 at rest -- already ~3.5x the 0.5 m/s^2 production default (AGENTS.md SS3, friction-modeling note) |

**Current production mitigation** (`CartesianMoveLimits`, all opt-in, defaults are a no-op
reproducing the original single-cycle behavior exactly):
- `accel_gap_cycles` -- speed sample formed from position N cycles back, not 1; shrinks accel
  noise by ~1/N^2 since position noise between any two ~independent samples doesn't grow with
  the gap but the time-span denominator does.
- `speed_lowpass_alpha` -- EMA smoothing on the gap-windowed speed before differencing for
  accel.
- `accel_max_consecutive_violations` / `accel_hard_multiple` -- `DeadlineMonitor`-style
  graduated tolerance: N consecutive over-threshold cycles, or one single-cycle spike over
  `hard_multiple x threshold`, either trips immediately.
- `NOISE_ROBUST_GUARD_OVERRIDES` (validated preset, `--noise-robust-guards`):
  `accel_gap_cycles=5`, `speed_lowpass_alpha=0.2`, `accel_max_consecutive_violations=3`,
  `accel_hard_multiple=5.0` (same for speed).

**What was already found and validated about this preset** (`docs/status/
safety_envelope_backtest_2026-07-30.md`, `experiments/safety-envelope-study` branch --
not on this branch by name, fetched via `git show` for this task):
- Graduated tolerance **alone** (no gap/lowpass filtering) still spuriously trips 30/30 seeds
  on a profile constructed to be genuinely physically clean at real measured RTDE noise
  magnitudes -- consecutive violations happen anyway because double-differenced noise samples
  share a position sample and are short-range correlated, not IID.
  Only **graduated tolerance + `accel_gap_cycles=5`/`speed_lowpass_alpha=0.2` together**
  closes it (0/30 spurious, still 30/30 correct on the genuine-catch case).
- A separate, unplanned finding in that same backtest: real closed-loop **controller torque-
  tracking chatter** (not RTDE sensor noise at all) can reach ~3.6 m/s^2 on the accel estimate
  for an ordinary `dx=0.02m/T=1.0s` move with **zero injected sensor noise** -- ~31x the
  theoretical kinematic peak. This is a distinct problem from what this task/doc addresses
  (position-sensor noise); a Kalman filter on RTDE position measurements does nothing about
  controller-output jitter, which isn't a measurement-noise problem in the first place. Noted
  here so the recommendation below isn't misread as covering that case too.
- Three smooth/state-aware redesigns (CBF-style move-timing, `cond(J)`-scaled threshold,
  `|qd|`-growth-rate-aware threshold) were separately backtested against 21 real hardware
  trips and none were adopted -- one was outright disqualified (missed the one confirmed-real
  divergence), the others didn't clearly beat the existing gap/EMA + graduated-tolerance
  combination. Kalman filtering is evaluated here on its own merits against that same bar, not
  assumed better because it's more "principled."

## 2. Every derivative/noisy-signal site found in the real-hardware path

Grepped `hardware/` and `controller_core/` for finite-difference patterns (`/ dt`, `np.diff`,
manual differencing), not assumed:

| site | signal | filtered today? | verdict |
|---|---|---|---|
| `hardware/safety.py::CartesianMoveMonitor` | TCP position -> speed -> accel | yes (gap/EMA/graduated, opt-in) | **primary candidate, this doc's focus** |
| `hardware/joint_accel_estimator.py::JointAccelEstimator` | joint `qd` -> `qdd` | yes -- identical gap/EMA technique, ported deliberately from `CartesianMoveMonitor` per its own docstring | diagnostic-only (feeds `controller_core/dynamics_residual.py`'s residual observer, logged to trace, **no safety trip condition anywhere** -- confirmed via both modules' docstrings). Same lag-vs-noise tradeoff as SS3/SS4 below would apply if this were ever Kalman-filtered; out of scope for this task ("the best-documented, most concrete candidate" = the TCP guard), noted as the natural second candidate if this work continues. |
| RTDE `qd` itself (`getActualQd()`) | joint velocity | **not differentiated by this codebase at all** -- it's RTDE's own directly-reported estimate, consumed as-is by `check_joint_state`/`CartesianMoveMonitor`'s `|qd|>qd_max` checks | Real capture shows `qd_std_radps=0.0` at rest -- RTDE appears to apply its own internal deadband/filter already. Matches this task's own background note: real trace diagnosis found `qd` smooth and monotonic at the moment guard trips actually happened. **Not a Kalman-filtering candidate** -- there's no derivative computed here for this codebase to smooth, and the one real signal quality question (does RTDE's own `qd` filtering introduce its own lag) is outside this codebase's control. |
| `controller_core/filters.py::TorqueCommandFilter` | `dx/dt` inside a **torque-command** rate-limit filter | N/A | Output-side rate limiting on a commanded torque, not differentiation of a *measured* signal. Not a noise-smoothing candidate. |
| `controller_core/lqr_controller.py`/`mpc_controller.py`/`transport_lqr.py` (`_raw_acceleration`) | 1D cart-pole-lane acceleration models | N/A | `CartPoleLQRController`/`CartPoleFallbackController` -- confirmed via `grep -rl` across `hardware/*.py`: **not imported anywhere in the real-hardware path**. Legacy/archived-lane code, out of scope. |
| `hardware/position_transport.py` | same TCP position->accel issue, `position` control mode | same `CartesianMoveMonitor` instance, shared code path | covered by the same analysis as the primary candidate |

**Conclusion: exactly one live, safety-relevant candidate** (`CartesianMoveMonitor`'s TCP
accel chain), one diagnostic-only secondary candidate (`JointAccelEstimator`), and no others.

## 3. Kalman filter design

**Model**: constant-acceleration (white-noise-jerk), one independent 1D filter per Cartesian
axis (x, y, z decoupled -- matches the real noise floor's own per-axis independence in the
stats file, and needs no cross-axis coupling since `CartesianMoveMonitor`'s speed/accel are
themselves built from per-axis position).

- State `x = [p, v, a]`.
- `F = [[1, dt, dt^2/2], [0, 1, dt], [0, 0, 1]]` (standard CA transition).
- `Q = jerk_psd * [[dt^5/20, dt^4/8, dt^3/6], [dt^4/8, dt^3/3, dt^2/2], [dt^3/6, dt^2/2, dt]]`
  -- exact discretization of continuous white-noise-jerk process noise (Bar-Shalom, Li &
  Kirubarajan). `jerk_psd` is the one free design parameter.
- `H = [1, 0, 0]` (position-only measurement), `R = measured per-axis position variance`
  (real: `(8.88e-6)^2`, `(9.86e-6)^2`, `(3.25e-6)^2` m^2 -- **the real measured number, not a
  guess**).
- Standard predict/update; since the measurement is scalar, the update's "matrix inversion" is
  a scalar division, no `np.linalg.inv` needed.

**Lighter-weight comparison point**: an alpha-beta-gamma (g-h-k) fixed-gain filter, with gains
derived as the **steady-state Kalman gain** of the equivalent CA-KF (a standard, principled
way to pick alpha/beta/gamma rather than guessing, per Kalata's g-h-k/Kalman correspondence)
-- same asymptotic behavior, no per-cycle covariance propagation, cheaper.

**Two accel metrics reported**: (1) `accel_scalar_mps2 = |Δ(||v||)/dt|`, matching
`CartesianMoveMonitor`'s own definition exactly (one differentiation of the filter's already-
smoothed speed, for direct backward-compatible comparison); (2) `accel_vector_mps2 =
||a_xyz||`, read straight from the filter state with **zero extra differencing** -- arguably
the more principled metric, since it doesn't re-introduce a raw derivative on top of already-
filtered output. Metric (2) is what all comparisons below use unless noted, since (1) is
markedly worse in every case (see script output: `kalman_scalar_speed_diff` p99=1.72 vs
`kalman_vector_norm` p99=0.48 m/s^2 at `jerk_psd=10`, same run).

**Real-time cost** (this machine/westeros -- see the same caveat `docs/status/
direct_torque_controller_phase_profiling_2026-07-31.md` makes explicitly: absolute numbers
aren't portable to thinkrobot's CPU, relative comparison is what transfers):

| method | measured cost |
|---|---|
| 3-axis constant-acceleration KF | 76.9 us/cycle |
| 3-axis alpha-beta-gamma | 21.4 us/cycle |
| current heuristic-equivalent op (norm + subtract/divide) | 2.65 us/cycle |

Budget context: the 500 Hz `direct_torque` loop's own phase-profiling
(`direct_torque_controller_phase_profiling_2026-07-31.md`) measured mean total `compute()`
cost at 0.280 ms of the 2 ms period -- **~1.7 ms of headroom**. Even the full KF's 77 us is
~4% of that headroom; alpha-beta-gamma's 21 us is ~1%. **Compute cost is not a blocker for
either design.**

## 4. Prototype and results

Script: `tools/diagnostics/kalman_tcp_accel_filter_prototype.py`. Runs standalone (no RTDE
import -- `hardware.safety` is numpy/dataclasses-only, confirmed by reading it, so this is
import-safe without `ur_rtde`/`rtde_control` installed). Uses the **real** captured stationary
RTDE data at `outputs/hardware_state_noise/capture_20260728_154018.jsonl` (present in this
checkout, 4730 real samples) -- falls back to synthesized Gaussian noise at the documented std
only if that file is missing, and prints which one it used. The "current heuristic" columns
call the real, unmodified `hardware.safety.CartesianMoveMonitor`/`CartesianMoveLimits`
directly (not reimplemented), with drift/orientation/waypoint-jump/qd checks defanged via
generous limits so only the accel channel is being compared.

Run: `python tools/diagnostics/kalman_tcp_accel_filter_prototype.py --jerk-psd 10.0 --seeds 30`
(≈14s on this machine). `jerk_psd=10` was chosen because it's the largest value in a coarse
grid search (`--tune-jerk-psd`) whose stationary-noise p99 (0.478 m/s^2) still sits under the
real production `max_tcp_accel_mps2=0.5` default -- i.e. the tightest jerk_psd that could
plausibly replace the current tightest real threshold without immediately failing on noise
alone. A second, more heavily-smoothed point (`jerk_psd=0.1`) is reported too, to show the
other end of the real tradeoff.

### (a) Noise suppression, real stationary capture (zero true motion), m/s^2

| method | mean | p99 | p99.9 | max |
|---|---|---|---|---|
| current, `gap=1/alpha=1.0` (unfiltered default) | 2.233 | 8.169 | 10.989 | 12.636 |
| **current, `gap=5/alpha=0.2` (shipped `NOISE_ROBUST_GUARD_OVERRIDES` preset)** | 0.076 | **0.272** | 0.419 | **0.553** |
| KF vector-norm, `jerk_psd=10` | 0.186 | 0.478 | 0.606 | 0.661 |
| KF vector-norm, `jerk_psd=0.1` | -- | **0.073** | -- | 0.110 |
| alpha-beta-gamma vector-norm, `jerk_psd=10` | 0.467 | 1.229 | 1.507 | 1.789 |

**At a jerk_psd tuned to the current threshold's own headroom, the KF's noise floor is worse
than the already-shipped `gap=5/alpha=0.2` heuristic** (p99 0.478 vs 0.272 m/s^2). Only at a
much heavier-smoothing `jerk_psd=0.1` does the KF clearly beat it (p99 0.073, ~3.7x tighter)
-- at a real lag cost, see (b). The alpha-beta-gamma fixed-gain filter is worse than the full
KF at the same `jerk_psd` (no online covariance adaptation), and worse than the current
heuristic too.

### (b) Tracking lag, real genuine move (`dx=-0.20m`, `T=1.0s` min-jerk, analytic peak
1.1547 m/s^2 at `t=0.500s`), real bootstrapped noise (seed 0)

| method | peak reported | % of true peak | peak location vs true |
|---|---|---|---|
| current, `gap=1/alpha=1.0` | 15.767 m/s^2 | 1366% | dominated by noise, not a real tracking signal |
| current, `gap=5/alpha=0.2` | 1.458 m/s^2 | 126% | 324 ms **early** |
| KF vector-norm, `jerk_psd=10` | 1.465 m/s^2 | 127% | 350 ms **late** |
| KF vector-norm, `jerk_psd=0.1` | 1.192 m/s^2 | 103% | **300 ms late**, consistently across `jerk_psd` 0.1-1 (checked directly, not just this one seed) |
| alpha-beta-gamma vector-norm, `jerk_psd=10` | 2.369 m/s^2 | 205% | 350 ms late |

**Neither approach cleanly tracks the true peak's timing or magnitude.** The current
heuristic's own smoothing produces a comparable-magnitude (~126%) overshoot, just early
instead of late. The KF's overshoot shrinks toward ~103% as `jerk_psd` drops (heavier
smoothing), but the **lag grows to ~300 ms and stays there** -- confirmed by directly scanning
`jerk_psd` in [0.1, 0.5, 1.0, 3.0, 10.0]: peak time sits at `t≈0.80s` (true peak `t=0.5s`) for
every value in the smooth, well-behaved regime, and only becomes erratic (noise-dominated,
peak lands anywhere) once `jerk_psd` gets large enough that the noise floor approaches the
signal. **This is the real "not a free lunch" cost the task asked to check for, and it's a
serious one for this application**: the one real, independently-documented divergence event
in this codebase's history (AGENTS.md/backtest SS3 -- `position_20260728_150847`'s wrist_1/
wrist_3 `|qd|` growing ~1.6-1.8x **every 8 ms cycle**) escalates on a timescale of tens of
milliseconds, not hundreds. A ~300 ms lag, needed to get the KF's noise floor meaningfully
below the current heuristic's, would very plausibly be too slow to catch that exact case in
time -- the KF was never tested against that specific trace (out of scope for this offline
prototype, see SS5), but the lag number alone is reason for real caution before trusting it on
a fast-divergence event.

### (c) Pass/fail replay, 30 seeds/profile, real bootstrapped noise

Two of the backtest's three profiles are reconstructable here without the full MuJoCo
controller-chatter simulation (the third, `canonical` at `base_accel=0.5`, was disqualified in
the original backtest by controller torque-tracking chatter, not sensor noise -- out of scope,
noted not silently dropped):

| profile | current `gap=1/alpha=1.0` | current `gap=5/alpha=0.2`+graduated | KF vector-norm (`jerk_psd=10`) | KF vector-norm (`jerk_psd=0.1`) | alpha-beta-gamma (`jerk_psd=10`) |
|---|---|---|---|---|---|
| `canonical_headroom` (clean, noise-free-safe, `base_accel=4.5`) | 30/30 spurious | **0/30** | **0/30** | **0/30** | **0/30** |
| `large_displacement` (genuine catch, `base_accel=0.8`) | 30/30 (correct) | 30/30 (correct) | 30/30 (correct) | 30/30 (correct) | 30/30 (correct) |

**Every non-naive method -- the already-shipped heuristic and every Kalman/alpha-beta-gamma
variant tested -- achieves identical pass/fail outcomes on both reconstructable profiles.**
No regression, no improvement, over what's already in production.

## 5. Recommendation

**Do not implement Kalman filtering as a replacement for the current `accel_gap_cycles`/
`speed_lowpass_alpha`/graduated-tolerance mechanism.** On the three requested axes:

1. **Noise suppression**: at a jerk_psd tuned to be comparably tight to the current shipped
   preset, the KF is *worse* (p99 0.478 vs 0.272 m/s^2). It can be tuned tighter (0.073 at
   `jerk_psd=0.1`), but only by trading away timing accuracy -- point 2.
2. **Tracking lag**: real, measured, and large (~300 ms in the noise-suppression-favorable
   regime) -- a genuine regression risk given this codebase's own documented real divergence
   escalates on an 8 ms/cycle timescale.
3. **Pass/fail agreement**: identical to the already-shipped, already-validated heuristic on
   both profiles this prototype could reconstruct. No demonstrated behavioral improvement.

Compute cost is **not** the blocker (77 us/cycle vs ~1.7 ms of measured headroom) -- if
anything is a point in the KF's favor it's that it's affordable, not that it's needed.

The KF's one real advantage found here is **interpretability/tunability**: one physically-
motivated parameter (`jerk_psd`, tied to a real measured `R`) mapping out an explicit,
measured noise-vs-lag Pareto curve, versus two coupled ad-hoc constants
(`accel_gap_cycles`/`speed_lowpass_alpha`) whose combined effect had to be validated
empirically against 21 real trip traces. That is a real, if modest, engineering-quality
argument -- but it is not a demonstrated performance win on any of the three metrics this task
asked about, and AGENTS.md's own stated culture (validated findings over speculative
complexity) is exactly the standard this doesn't clear. **This matches, rather than
contradicts, the existing backtest's own verdict on the three previously-tested smooth/
state-aware redesigns**: a more principled-looking mechanism is not automatically a better
one, and the falsifiable comparison is what should decide it, not architecture aesthetics.

**If this line of work continues anyway** (e.g. if the interpretability argument alone is
judged worth it, or if a future real-hardware capture shows the lag is more tolerable than
this offline analysis suggests), the concrete next steps, none of which are done here:
- Validate the KF against the actual 21-run real-hardware trip dataset the existing heuristic
  was validated against (`experiments/safety_envelope_backtest.py`,
  `experiments/safety-envelope-study` branch), not just this script's two reconstructable
  synthetic profiles -- in particular, the one confirmed-genuine divergence
  (`position_20260728_150847`) that the lag finding above raises real concern about.
- Test whether the KF's `jerk_psd` can be **scheduled** (loose during a commanded move, tight
  at rest/hold) the way `accel_gap_cycles`/`speed_lowpass_alpha` already effectively are via
  CLI presets -- this could plausibly recover the noise-suppression win without the fast-move
  lag cost, but is untested here and is a real design task, not a parameter tweak.
- If wired in for real: this would replace `CartesianMoveMonitor`'s internal `_prev_speed_mps`/
  `_pos_history` bookkeeping with per-axis filter state, needs its own unit tests mirroring
  `tests/hardware/test_hardware_safety.py`'s coverage, and per this task's hard constraint,
  needs an explicit, clearly-flagged design proposal and sign-off before touching
  `hardware/safety.py` at all -- not done here.

## 6. Tests run

- `python -m pytest -q -m hardware`: 248 passed, 1 failed
  (`test_direct_torque_residual_observer_async.py::
  test_residual_observer_async_phase_cost_is_much_lower_than_sync`) -- a real-time-budget
  timing assertion (`deadline_overrun: 3 consecutive cycles late by > max_deadline_ms`) on this
  shared, currently-loaded host (AGENTS.md SS8: `westeros` load can spike unpredictably).
  **Pre-existing and unrelated**: no file this failure touches
  (`hardware/direct_torque_transport.py`, `hardware/residual_observer_worker.py`) was modified
  by this task; nothing in this task's diff runs anywhere near that test's code path. Not
  investigated further, per this doc's own scope (research/design/offline prototype only).
- `python tools/diagnostics/kalman_tcp_accel_filter_prototype.py` (default args, `--jerk-psd
  10.0 --seeds 30`, `--tune-jerk-psd`): all run cleanly end to end, no import errors, no
  crashes, real-data path confirmed (`[data] REAL stationary capture, 4730 samples`).

## 7. Files changed / rollback

- Added: `tools/diagnostics/kalman_tcp_accel_filter_prototype.py` (new offline diagnostic
  tool, imports `hardware.safety` read-only, no production file touched).
- Added: this doc.
- No existing file modified. Rollback: `git rm tools/diagnostics/kalman_tcp_accel_filter_prototype.py
  docs/status/kalman_filtering_sensor_noise_2026-08-01.md` (or `git revert` the commit).

---

## 8. Follow-up (2026-08-01, same day): parallel/bucketed architecture -- does a diagnostic-only KF branch help?

**The original verdict above (SS5, "don't implement Kalman filtering as a replacement for the
current heuristic") is unchanged and stays as the recommendation for that framing.** This
section tests a materially different proposal, raised after reviewing SS1-7: don't pick one KF
tuning to *replace* the accel-guard heuristic (that's the tradeoff already shown above -- lower
noise floor costs real lag, and lag is a genuine hazard on the trip-gating path). Instead: keep
the current heuristic as the **sole, unmodified, real-time trip authority** (so a genuine fast
divergence still trips exactly as it does today -- zero regression, zero added lag on the
safety-critical path), and run a **separate, more heavily-smoothed KF branch alongside it purely
as a diagnostic/trend channel that gates nothing**, where the ~300ms lag measured above is
irrelevant by construction, since it's aimed at slow/gradual trends whose own timescale is much
longer than 300ms -- specifically, the `y_drift_m`/`z_drift_m`/`orientation_error_norm_rad`
pre-trip trend window `hardware/direct_torque_transport.py::_build_pre_trip_trend`/
`_classify_trend` already captures (commit `467fe52`, landed after a real -0.15m return-leg trip
needed manual trace re-parsing to see this trend by hand).

Script: `tools/diagnostics/kalman_parallel_trend_prototype.py`. Reuses
`ConstantAccelKalmanFilter1D` and the real capture-loading from the original prototype
(no duplicated math); imports the real, unmodified `_classify_trend`/`PRE_TRIP_TREND_WINDOW_CYCLES`
from `hardware.direct_torque_transport`.

**No real trace.jsonl for the actual -0.15m return-leg incident exists in this checkout** --
only the commit-message/AGENTS.md prose description survived (`outputs/hardware_transport/`
only has the 2026-07-28 runs already used in SS1; nothing dated 2026-07-31). The slow-creep
profile used below is an **explicitly synthetic reconstruction** matching that description
(y_drift ramping to the real 0.03 m off-axis-drift guard over a
`PRE_TRIP_TREND_WINDOW_CYCLES=60`-cycle / ~127ms window at the real capture's measured 2.119ms/
cycle rate -- consistent with the "~118ms" figure in the incident description), with real
noise bootstrapped from the same stationary capture used throughout this doc (not a hand-picked
number): `y_drift` std `9.86e-06 m`, `z_drift` std `3.25e-06 m`, an `orientation_error_norm`
noise-floor proxy (real rotation-vector residual norm, since no real `orientation_error_norm`
capture exists -- an honestly-labeled approximation, not production's exact quaternion-error
computation) std `1.94e-05 rad`.

### 8.1 Architecture is genuinely independent (confirmed, not just asserted)

`bucket_demo()` re-runs the original `large_displacement` (dx=-0.20m/T=1.0s) fast-move backtest
through the real, unmodified `CartesianMoveMonitor`: **30/30 correct on both `current_default`
and `current_graduated_filtered`, byte-identical to SS4(c)'s original numbers.** The script
asserts this and would raise if it ever changed. Nothing about adding a parallel KF branch can
regress this, since the branch is a separate object over the same read-only input stream with no
return path into the monitor's own decision.

### 8.2 Does the parallel KF branch help detect the slow y_drift/z_drift/orientation creep?

**Short answer: no, not measurably -- for a specific, checked reason, not by assumption.**

First check: at the synthetic incident's own headline magnitude (`y_drift` 0->0.031m,
`z_drift` 0->0.012m, `orientation_error_norm` 0->0.10 rad over the 60-cycle window), both raw
and KF-smoothed `_classify_trend()` on the full window correctly report "rising" **50/50 seeds,
every channel, both methods** -- this magnitude isn't the hard case; a sensitivity sweep down
toward the real noise floor is needed to find one.

**Sensitivity sweep** (`y_drift_m`, true final magnitude as a multiple of its own real measured
per-cycle noise std `9.86e-06 m`, full 60-cycle window, 50 seeds/point, real bootstrapped noise):

| true final magnitude | raw "rising" rate | KF "rising" rate |
|---|---|---|
| 0 (null -- no real drift at all) | 29/50 | 30/50 |
| 0.5x std (~5 um) | 49/50 | 46/50 |
| 1x-2.5x std | 50/50 | 50/50 |
| 3x-1000x std | 50/50 | 50/50 |

**Raw `_classify_trend` is already saturated at essentially its maximum possible sensitivity by
0.5-1x the real noise std** -- i.e. a true drift of a few *micrometers* over 60 cycles, three to
four orders of magnitude below the 0.03 m guard this trend window exists to give early warning
about. There is no real-world-relevant magnitude range in which the raw signal fails to detect
the trend but the KF-smoothed signal succeeds. At the one point tested below full saturation
(0.5x std), the KF is if anything *slightly worse* (46/50 vs 49/50, within noise of each other
but never better) -- consistent with the KF's own transient/lag interacting with the first-third/
last-third split, not a real advantage.

**Why, root-caused**: unlike the TCP accel estimate (a *double* finite difference that amplifies
noise by ~1/dt^2, the actual reason the original heuristic needed engineering), `y_drift_m`/
`z_drift_m`/`orientation_error_norm_rad` are **not derivatives at all** -- they're direct,
single-shot geometric quantities (drift-from-start position, orientation error), computed once
per cycle with no differencing. Their real noise floor (`~1e-5 to 2e-5`) is already 3-4 orders of
magnitude below the guard thresholds these trends are meant to anticipate, so there was never a
noise-suppression problem here for a KF to solve.

**A separate, unplanned finding, found as a side effect of building this test, not assumed
going in**: the null-case row above (29/50 raw, 30/50 KF report "rising"; the rest mostly
"falling", not "stable") shows `_classify_trend`'s `deadband_frac`-based relative-change check
essentially **never** reports "stable" for a channel whose own mean sits near zero -- which
describes `y_drift_m`/`z_drift_m` by definition (they're computed as an offset *from* the start
position, so they're expected to hover near zero for a stable run). `rel_change`'s denominator
(`max(|first_mean|, |last_mean|, 1e-9)`) collapses toward the `1e-9` floor whenever both means are
near zero, so even sub-noise-floor fluctuations produce a relative change that blows past the 5%
deadband. **This is a real weakness in the existing, unmodified `_classify_trend` heuristic,
independent of Kalman filtering** -- and critically, **Kalman smoothing does not fix it** (29/50
vs 30/50 is not a meaningful difference): smoothing reduces variance, but the failure mode here
is a scale-normalization issue in `_classify_trend`'s own math, not a noise-magnitude issue a
lower-variance input signal would resolve. Per this task's own scope (diagnostic-only,
`hardware/direct_torque_transport.py` is not `hardware/safety.py` and gates no trip, but still
out of scope to silently patch here) this is **flagged, not fixed**: `_build_pre_trip_trend`
only ever runs once, at trip time, on a run that already tripped some *other* guard, so this
specific weakness doesn't cause a false trip by itself -- but it does mean the "stable" label in
`pre_trip_trend`'s output for a near-zero-mean channel should not be trusted as evidence of a
truly flat trend without checking the raw values, since "rising"/"falling" is nearly the default
output there regardless of whether anything real is happening.

### 8.3 Revised recommendation

**The reframing is real and changes the risk profile, but the parallel branch still doesn't earn
its complexity for the specific slow-drift/trend-detection use case tested.** Concretely:

- The safety argument for going parallel is sound and is conceded: a diagnostic-only branch that
  never gates a trip genuinely has zero real-time/lag risk, confirmed in 8.1. If a future use
  case needs heavy smoothing for its own sake with no lag constraint, "run it in parallel, don't
  replace the trip authority" is the right shape for that, and the original SS5 objection (lag is
  a hazard) legitimately does not apply to a branch that never gates anything.
- But for *this* concrete target -- helping `_classify_trend` read the `y_drift`/`z_drift`/
  `orientation_error_norm` pre-trip window -- there is no measurable win: those channels were
  never noise-limited in the first place (they're not derivatives), the raw signal already
  detects a real trend at a few micrometers/microradians (far below anything operationally
  relevant), and the one real weakness found in that detection path (`_classify_trend`'s
  near-zero-baseline deadband collapse) is a *different* bug that a KF branch does not fix.
- **Do not build a parallel KF trend branch for `y_drift_m`/`z_drift_m`/`orientation_error_norm_rad`
  specifically** -- there's nothing here for it to improve. If `_classify_trend`'s deadband issue
  is worth fixing, that's a separate, small, targeted fix (e.g. an absolute-magnitude floor on top
  of the relative deadband) -- flagged here for a deliberate decision, not implemented, and not a
  Kalman-filtering question at all.
- The one place in this codebase where a parallel/diagnostic-only KF branch *would* face a real
  tradeoff worth solving is the one this doc already covers in SS1-7: the TCP accel estimate
  itself, which genuinely is noise-limited by a double differentiation. Item 1's `bucket_demo()`
  demonstrates that running the SS7 KF variant (`jerk_psd=0.1`) alongside the unchanged heuristic
  is mechanically trivial and safe -- but per SS5, that specific signal's own KF variant still
  doesn't outperform the current heuristic even as an independent diagnostic overlay, since the
  question there was never "is it safe to compute in parallel" (yes, trivially) but "does it show
  something the current heuristic's own trip decision and `pre_trip_trend` capture don't already
  show" -- and SS1-7 already found comparable noise/lag tradeoffs, not a clear win, even before
  this parallel framing.

### 8.4 Tests / files

- `python tools/diagnostics/kalman_parallel_trend_prototype.py` -- runs cleanly end to end, no
  import errors, no crashes, asserts the fast-move sanity check itself (would raise on any
  regression).
- `python -m pytest -q tests/hardware/test_direct_torque_transport_pre_trip_trend.py
  tests/hardware/test_hardware_safety.py`: 46 passed (both files this script imports from,
  confirming nothing about running it disturbs their own test coverage -- no production file
  was modified).
- Added: `tools/diagnostics/kalman_parallel_trend_prototype.py`. No existing file modified.
  Rollback: `git rm tools/diagnostics/kalman_parallel_trend_prototype.py` (or `git revert` the
  follow-up commit) plus reverting this section of the doc.
