# Hardware scratch scripts (archived)

One-off RTDE / URSim bring-up probes that used to live under `tools/`.
They are **not** part of the active hardware lane.

Kept for archaeology only — prefer:

- `tools/_rtde_control_probe.py` — minimal control + `directTorque` check
- `tools/_check_ursim_remote.py` — dashboard status
- `docs/hardware/URSIM_REMOTE_CONTROL.md` — fieldbus / remote-control checklist

Moved here 2026-07-14 from `tools/`:

- `_find_rtde_script.py`
- `_rtde_flags.py`
- `_rtde_control_try_flags.py`
- `_rtde_control_with_play.py`
- `_rtde_control_after_play.py`
- `_rtde_control_sweep.py`
- `_pin_jac_parity.py`
- `_bench_controller_iter.py`
