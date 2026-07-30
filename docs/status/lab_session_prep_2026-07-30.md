# Prep for the next real lab session (as of end of 2026-07-30 work)

Concrete run-sheet, not a narrative — exact commands and what to check. Written the night
before, for tomorrow.

## 1. What actually changed since the last real lab session

- **The default tuned config now has a real, validated fix that likely explains most of last
  session's TCP-accel guard trips.** `config/ur5e_mujoco_torque_osc_tuned.yaml`
  (`jacobian_singular_cond_max: 1.0e18`) — the controller no longer freezes for ~half of
  every move at the transport-start wrist singularity. This is the single biggest thing to
  actually test for real tomorrow; it's only been validated in MuJoCo sim (304 runs, zero
  regressions), never on the real robot. Old behavior preserved at
  `config/ur5e_mujoco_torque_osc_tuned_singular_scale_enabled.yaml` if you ever need to
  reproduce it.
- **New `--noise-robust-guards` flag** on `tools/ur5e_move.py` and
  `tools/ur5e_direct_torque_x_transport.py` (all three `--control-mode` values now, including
  `urscript`) — applies the validated combination that actually closed the real-noise
  spurious-trip gap (graduated tolerance + `accel_gap_cycles=5`/`speed_lowpass_alpha=0.2`
  together; graduated tolerance ALONE was tested and does NOT work, see
  `docs/status/safety_envelope_backtest_2026-07-30.md` §9 if you want the evidence). Use this
  flag by default for any real move tomorrow rather than the old individual flags, unless
  you have a specific reason to override one field.
- **`urscript` mode now has full guard-tuning parity** with `position`/`direct_torque` — all 7
  `CartesianMoveLimits` override flags (including `--noise-robust-guards`) now reach it. This
  was never true before tonight.
- **`urscript` supervisor's polling-rate timing bug fixed** — it previously used a fixed sleep
  guess instead of adapting to real per-cycle work, which could silently drift the actual
  polling rate below `monitor_hz` without any guard catching it.
- **A real, still-open question deferred to tomorrow**: does the URScript `cond(J)` Jacobi
  solver actually fit PolyScope's real-time budget on real hardware? Never benchmarked. See
  §3 below for how to actually get evidence on this without needing URSim/Docker access.
- **Still NOT validated for real**: `config/ur5e_mujoco_torque_osc_tuned_wrist_orient.yaml`
  (the other real controller fix, for orientation drift) may have the exact same freeze bug
  as the base config had — a sim validation sweep for this specific config was kicked off
  tonight; check `docs/status/disable_singular_scale_wrist_orient_validation_2026-07-30.md`
  for its outcome before trusting that config's timing behavior tomorrow. If that doc doesn't
  exist yet when you read this, the validation didn't finish — don't assume it's fine.

## 2. Recommended first real commands tomorrow, in order

**Pre-flight (receive-only, cannot move the robot):**
```
python tools/check_hardware_deps.py
python tools/ur5e_probe_connection.py --robot-ip <IP> --duration-s 5.0
python tools/ur5e_connect.py --robot-ip <IP> --once
```

**First real motion — deliberately conservative, direct_torque, new default config + validated guard preset:**
```
python tools/ur5e_direct_torque_x_transport.py --robot-ip <IP> --control-mode direct_torque \
  --target-x-delta 0.01 --move-duration 2.0 --duration 4.0 \
  --config config/ur5e_mujoco_torque_osc_tuned.yaml \
  --noise-robust-guards \
  --i-understand-this-moves-the-robot
```
(drop `--yes` so you get the typed `MOVE` confirmation; same physical-safety checks as
before — pendant singularity warning, base clearance, visual judgment call each time.)

**If that's clean, this is the real test of tonight's finding** — try a move/duration profile
that would have hit the freeze hard before (short move_duration, e.g. `--move-duration 0.3`)
and confirm it now actually completes instead of achieving ~0 displacement.

## 3. How to actually get evidence on the Jacobi-solver question tomorrow, without URSim

URSim access is blocked from this machine (no Docker permission) and there's no reason to
delay this to a separate URSim setup — the real robot itself gives better evidence anyway.
Run a real `urscript` mode transport and watch the EXISTING timing telemetry, which already
does exactly this kind of "is the loop keeping up" check:
```
python tools/ur5e_direct_torque_x_transport.py --robot-ip <IP> --control-mode urscript \
  --target-x-delta 0.02 --move-duration 1.0 --duration 3.0 \
  --noise-robust-guards \
  --i-understand-this-moves-the-robot
```
Check the resulting `summary.json`'s timing/deadline fields and whether `DeadlineMonitor`/
`StaleStateMonitor` tripped or showed elevated lateness during the run — if the on-robot
Jacobi solver doesn't fit PolyScope's real-time budget, this is the most direct real signal
of it, without needing to build any new instrumentation. This is `urscript` mode's first-ever
real-hardware execution in this project — go in expecting to find something, not expecting
a clean pass.

## 4. Safety discipline, same as always

- Typed `MOVE` confirmation every time, no `--yes` shortcuts for a first attempt at anything new.
- Check the pendant for wrist-singularity warnings before confirming any move at the
  transport-start pose family — this hasn't changed, the controller fix doesn't remove the
  physical singularity, only the pathological freeze response to it.
- Re-verify base/wall clearance visually each time — a base-rotation override value from a
  previous session is not guaranteed safe at a different pose.
- `E-stop latch is one-way` — a tripped run needs a new process, not a retry in the same one.
