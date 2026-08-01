# Real-hardware test queue — 2026-08-01 5hr lab session

Written the night before (2026-07-31/08-01) from `westeros`, with no hardware access, to make
tomorrow's in-lab session execution-only: copy a command, type `MOVE`, paste the JSON result back
for analysis. Nothing here removes the typed-confirmation safety step — every command below still
prompts for it (no `--yes`).

**Updated later the same night** after two real fixes landed for what section 2 originally called
"no fix attempted" — read section 2 fresh even if you read an earlier version of this doc. Two
background tasks were still running when this update was written (a harder-plant comparison
campaign, a URScript `wrist_orientation_task` parity port) — check `git log --oneline -10` for
commits after `3f77057` before relying on this doc being fully current.

**Before anything else**, run this on `thinkrobot` and paste the tail — I don't have visibility into
tonight's real run history (`outputs/hardware_transport/` only exists on `thinkrobot`, not `westeros`),
so I can't tell from here exactly which configs got real coverage tonight vs. only sim coverage:

```bash
cd ~/real-cartpole-control-using-ur5-direct-torque
python3 -c "
import csv
with open('outputs/hardware_transport/run_log.csv') as f:
    rows = list(csv.DictReader(f))
for r in rows[-30:]:
    print(r.get('created_at_utc'), r.get('config_path'), r.get('trajectory_profile'),
          r.get('target_x_delta_m'), r.get('outcome'), r.get('failure_category'))
"
```
(If `run_log.csv` doesn't exist at that path, it may be per-sweep — check `outputs/hardware_transport/*/run_log.csv`.)

## 0. M0 — the one real prerequisite blocking the auto-tuning ("auto experiment") system

`docs/hardware/AUTO_TUNING_PLAN.md` is a real, sim-gated Bayesian-optimization (Optuna TPE) system
for auto-calibrating controller gains on the real robot — not a plan, actually **built and tested**
(`tools/ur5e_suggest_gains.py` + `tools/ur5e_auto_tune_gains.py`, batches 4 candidates per round,
sim-gates every one automatically, requires a typed `CONFIRM` before even printing real commands,
verified to never import `hardware.*` — statically checked, not just claimed). It has never touched
real hardware, for exactly one reason: **M0**, its own hard prerequisite, has never been done. M0 is
small and mechanical — just proving the unmodified, sim-validated gains move the real arm once in
each control mode, before any search is layered on top:

1. `tools/ur5e_connect.py --once` — confirm the link works at all.
2. `position` mode, `--distance-m 0.02`, typed `MOVE`. (First-ever real motion this repo makes in a
   session, if not already done.)
3. `direct_torque` mode, same tiny displacement, fixed validated gains, no exploration.
4. `urscript` mode, same tiny displacement — **this is exactly section 1 below**, do that test and
   M0 step 4 is satisfied at the same time, no separate run needed.
5. Only after all three pass at `dx=0.02m`: escalate to the sim-validated envelope (`dx=0.10-0.20m`)
   with the *unmodified* gains, still no auto-tuning yet.

**Once M0 is done**, `tools/ur5e_auto_tune_gains.py` becomes usable for real — that's the actual
unlock, not a separate future task. Do not run it before M0 completes; there's no scored trial data
for it to build on yet, and per the plan's own explicit decision, autonomy stays human-gated (you
approve every batch before anything moves) even after M0 — this was never meant to be unattended.

## 1. URScript first real test — never run on hardware before tonight's plumbing fix

Motivation: `hardware/urscript_transport.py` has full Python-math parity (`tests/hardware/test_urscript_parity.py`,
~1e-11 relative error) but **zero real-hardware or URSim execution ever**. The generated script was
hand-checked before this queue was written (`cond_max=1e18` — the promoted singular-scale fix — gains
match `config/ur5e_mujoco_torque_osc_tuned.yaml`, `kp_rot=0`, stop-register checked every cycle).

```bash
cd ~/real-cartpole-control-using-ur5-direct-torque
python3 tools/ur5e_urscript_x_transport.py --robot-ip 172.16.71.77 \
  --i-understand-this-moves-the-robot
```
Defaults: `dx=0.02m`, `move-duration=1.0s`, `duration=3.0s`, `config/ur5e_mujoco_torque_osc_tuned.yaml`.

**What to paste back**: the full JSON summary + whether `outputs/hardware_urscript/*/supervisor_trace.jsonl`
was written. **Watch for specifically**: does `sendCustomScript()` actually execute the motion (not just
return `true`)? Does the 125Hz Python supervisor's guard checks behave sanely (no spurious trip, but
also — critically — does it actually catch a real fault if one occurs)? Any PolyScope-side error/popup
countrolling this is something only you can see.

**If clean**, repeat with the noise-robust preset (now wired, commit `b586f23`) to confirm the flag
plumbs through:
```bash
python3 tools/ur5e_urscript_x_transport.py --robot-ip 172.16.71.77 \
  --target-x-delta 0.05 --move-duration 1.0 --duration 3.0 \
  --noise-robust-guards --i-understand-this-moves-the-robot
```

**Do not** jump to a large displacement (e.g. the 0.20m/12s scale used in tonight's direct_torque test)
on URScript until at least one small run has been visually + numerically confirmed clean — this is a
new code path touching the real robot for the first time, independent of direct_torque's own track record.

## 2. Negative-X direction — TWO real, sim-validated fixes now exist, neither real-hardware tested

Original real state (unchanged as fact): `+0.20m` passes cleanly; `-0.20m` (historical) and
`-0.15m` (tonight) both failed on real hardware. This turned out to be TWO separate failures,
both root-caused and both fixed in sim since this doc was first written:

**Fix A — the un-rotated directional ceiling** (this is almost certainly what the real `-0.20m`/
`-0.15m` failures above actually were, since those ran at the un-rotated height_alpha=0.5 pose):
`config/ur5e_mujoco_torque_osc_tuned_wrist_orient_fixed.yaml` — `wrist_orientation_task` combined
with the already-promoted singular-scale fix. Sim: 8/8 vs baseline 6/8 at both `dx=+0.20m` and
`dx=-0.20m`, worst-case orientation error roughly halved. Zero regressions anywhere tested
(`docs/status/nullspace_envelope_search_2026-08-01.md`). **Start here** for a real retest:

```bash
python3 tools/ur5e_direct_torque_x_transport.py --robot-ip 172.16.71.77 \
  --control-mode direct_torque --skip-joint-move \
  --target-x-delta -0.05 --move-duration 1.0 --duration 3.0 \
  --config config/ur5e_mujoco_torque_osc_tuned_wrist_orient_fixed.yaml \
  --i-understand-this-moves-the-robot
```
Small first (`-0.05m`), matching this repo's own start-small discipline — this config's real
behavior is completely unknown, only sim-validated. If clean, escalate toward `-0.20m` in a couple
of steps, not directly.

**Fix B — the -45° pose's Y-drift** (only relevant if you're testing from the real-hardware
default `HEIGHT_ALPHA_0_5_CLEARANCE_Q` pose, not the un-rotated one): root-caused as a genuine
structural X-Y authority trade-off in the controller (no P/D/I gain fixes it without breaking
X-tracking — three independent investigations agree). You explicitly directed and reviewed a
deliberate, evidence-scoped safety-tolerance change for this pose specifically:
`config/ur5e_mujoco_torque_osc_tuned_wrist_orient_fixed_neg45_pose.yaml` raises
`max_abs_orthogonal_drift_m` from 0.03m to 0.05m (real margin above the largest measured natural
transient, not a blanket loosening — `controller_core/safety.py`'s class default is untouched).
Sim: 32/38, but **only validated up to `dx=0.06m`** — `dx≥0.10m` deliberately still fails/blocked,
don't expect it to pass. `docs/status/neg45_drift_tolerance_validation_2026-08-01.md` has the full
numbers. Real dose-response at this pose has an unexplained sim-vs-real gap already documented
(real historically tripped at `dx=0.20m`, sim onset `dx=0.05-0.06m`) — so this needs its own small
real test, separate from Fix A:

```bash
python3 tools/ur5e_direct_torque_x_transport.py --robot-ip 172.16.71.77 \
  --control-mode direct_torque \
  --target-x-delta 0.04 --move-duration 1.0 --duration 3.0 \
  --config config/ur5e_mujoco_torque_osc_tuned_wrist_orient_fixed_neg45_pose.yaml \
  --i-understand-this-moves-the-robot
```
(No `--skip-joint-move` here — let it drive to the -45° pose fresh.) Small positive-X first to
confirm the config behaves sanely at all on real hardware before ever trying the negative
direction or anything past `dx=0.06m`.

Neither fix has ANY real-hardware validation yet — both are pure sim results. Treat both real
tests above as first-ever, start-small runs, not confirmations of something already known-safe.

Still available if you'd rather sidestep this entirely for a quick real round-trip: **joint-space
return** (`moveJ` back to the start pose) instead of fighting the Cartesian `-X` OSC path —
untested tonight but architecturally simple; ask if you want a small script for this.

## 3. Accel/duration trajectory profiles — partially characterized

`accel_duration_scurve`: real-tested clean at `accel=0.02, move_duration=4.0`; **failed** (speed guard)
at `move_duration=6.0` and `10.0`, despite peak commanded velocity being duration-independent by the
profile's own closed-form math — an open, only-partially-explained real finding (see the qd/x_error/tau
trend pulls done tonight). `accel_duration_triangular` has closed-form/unit test coverage but **no real
hardware run at all** yet.

```bash
# Confirm the known-clean point still holds:
python3 tools/ur5e_direct_torque_x_transport.py --robot-ip 172.16.71.77 \
  --control-mode direct_torque --skip-joint-move \
  --trajectory-profile accel_duration_scurve --target-accel 0.02 \
  --move-duration 4.0 --duration 6.0 \
  --config config/ur5e_mujoco_torque_osc_tuned.yaml \
  --noise-robust-guards --i-understand-this-moves-the-robot

# First-ever real triangular-profile test, same conservative accel/duration:
python3 tools/ur5e_direct_torque_x_transport.py --robot-ip 172.16.71.77 \
  --control-mode direct_torque --skip-joint-move \
  --trajectory-profile accel_duration_triangular --target-accel 0.02 \
  --move-duration 4.0 --duration 6.0 \
  --config config/ur5e_mujoco_torque_osc_tuned.yaml \
  --noise-robust-guards --i-understand-this-moves-the-robot
```

## 4. Friction feedforward — confirm real coverage before repeating anything

Sim validation is solid (height_alpha 0.2/0.3/0.5, see AGENTS.md §3). Real-hardware coverage tonight is
uncertain from where I'm sitting (see the run_log pull at the top of this doc) — **check that first**
before re-running what might be a duplicate. If it turns out untested, this is the config:
`config/ur5e_mujoco_torque_osc_tuned_friction_ff.yaml`. **Confirmed final number at the -45° pose**:
17/38 vs. 18/38 baseline — roughly neutral (not a fix, not a full regression), so it's not
disqualified there, but don't expect it to help either.

## 5. New tools tonight that are sim-only and should NOT be pointed at real hardware yet

- `config/ur5e_mujoco_torque_osc_tuned_y_integral.yaml` (`y_integral_action`/`ki_y`) — built and
  tested while diagnosing the -45° failure, confirmed to have **zero effect** at its committed
  gentle dose (byte-identical to baseline). Not a real candidate; no reason to test it live.
- `--asymmetric-coulomb-friction` (new opt-in plant-side friction model on
  `tools/ur5e_mujoco_torque_experiments.py`/`tools/ur5e_move_hold_transport.py`) — a **sim-only
  plant realism addition** for stress-testing controllers against more adversarial physics before
  ever going real. It changes MuJoCo simulation behavior, not anything on the real robot — there
  is no real-hardware equivalent to run.
- `tools/analysis/fit_residual_torque_model.py` (phase-1 offline residual-torque regression) —
  offline data analysis only, not a controller change, nothing to run live.
- URScript `wrist_orientation_task` port — if it landed by the time you read this (check
  `docs/status/urscript_wrist_orientation_parity_2026-08-01.md`), it is Python-vs-Python
  numerical parity ONLY, same standing caveat as the rest of URScript mode: zero real-hardware
  execution. Do not treat it as ready for a real `wrist_orient_fixed` config test via URScript
  without independently confirming that doc says so explicitly.

## Standing rules for tomorrow (carried over, not new)

- Typed `MOVE` confirmation every time — never `--yes` on a real move.
- Don't widen `--*-max-consecutive-violations` past validated presets based on an in-the-moment "sensors
  seem noisy" read — tonight's real trace evidence showed real trips were smooth/sustained, not noise.
- After any guard trip: `pre_trip_trend` in `summary.json` now auto-captures qd/speed/x_error/tau/
  orientation/y_drift/z_drift trend — read that before re-running blind.
- If a background process/agent is used for any part of tomorrow's analysis, don't let it end its turn
  with something still running unchecked — this bit us multiple times tonight.
