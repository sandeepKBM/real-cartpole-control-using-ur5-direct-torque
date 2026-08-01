# Real-hardware test queue — 2026-08-01 5hr lab session

Written the night before (2026-07-31/08-01) from `westeros`, with no hardware access, to make
tomorrow's in-lab session execution-only: copy a command, type `MOVE`, paste the JSON result back
for analysis. Nothing here removes the typed-confirmation safety step — every command below still
prompts for it (no `--yes`).

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

## 2. Negative-X direction — currently 0-for-2 real, no fix attempted

Real state: `+0.20m` passes cleanly; `-0.20m` (historical) and `-0.15m` (tonight) both failed —
diagnosed as a real, slow orientation/Z-drift creep (not noise, not instability), consistent with
AGENTS.md's documented directional-ceiling / nullspace-projector-asymmetry finding. No fix has been
designed. **Do not** re-attempt a negative-X `direct_torque` move expecting a different outcome without
a real change first — this is a known, reproducible, structural failure, not a fluke.

Practical options for tomorrow if you need a real round-trip:
- **Joint-space return** (`moveJ` back to the start pose) instead of fighting the Cartesian `-X` OSC
  path — untested tonight but architecturally simple; ask me to wire a small script for this if wanted.
- Or treat root-causing this as its own dedicated block of the 5 hours, with me pulling `pre_trip_trend`
  (now captures `y_drift_m`/`z_drift_m` automatically, commit `467fe52`) after each attempt instead of a
  manual trace pull.

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
`config/ur5e_mujoco_torque_osc_tuned_friction_ff.yaml`. Do **not** use it at the -45° pose without
`--start-q-rad` override caution — corrected result tonight: 17/38 vs. 18/38 baseline (roughly neutral,
not a fix, not a full regression either — see `docs/status/` for tonight's corrected numbers once
written up).

## Standing rules for tomorrow (carried over, not new)

- Typed `MOVE` confirmation every time — never `--yes` on a real move.
- Don't widen `--*-max-consecutive-violations` past validated presets based on an in-the-moment "sensors
  seem noisy" read — tonight's real trace evidence showed real trips were smooth/sustained, not noise.
- After any guard trip: `pre_trip_trend` in `summary.json` now auto-captures qd/speed/x_error/tau/
  orientation/y_drift/z_drift trend — read that before re-running blind.
- If a background process/agent is used for any part of tomorrow's analysis, don't let it end its turn
  with something still running unchecked — this bit us multiple times tonight.
