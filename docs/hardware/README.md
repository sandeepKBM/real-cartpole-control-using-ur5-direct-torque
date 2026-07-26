# Hardware lane — learn this end to end

**Start here.** This is the map of every file that matters for real-UR5e / URSim control.
Deep dives live in [`HARDWARE_GUIDE.md`](HARDWARE_GUIDE.md). Operational checklists are linked below.

**Last updated:** 2026-07-14

---

## Learning order (read + run in this sequence)

| Step | What | File(s) to read | Command to run |
|------|------|-----------------|----------------|
| 0 | Mental model: 3 control modes | this README §Modes | — |
| 1 | Safety primitives | `hardware/safety.py` | `pytest tests/hardware/test_hardware_safety.py -q` |
| 2 | RTDE link (receive + servoL) | `hardware/link.py` | `python tools/ur5e_connect.py --robot-ip <IP> --once` |
| 3 | Bounded position move | `hardware/motion.py` + `tools/ur5e_move.py` | tiny `--distance-m 0.02` move |
| 4 | Named poses + joint move | `hardware/poses.py`, `joint_motion.py` | `tools/ur5e_move_joints.py --pose height_alpha_0_5 ...` |
| 5 | Mode router | `hardware/control_mode.py`, `x_transport.py` | — |
| 6 | **Mode 1 — position + shadow OSC** | `position_transport.py` | `--control-mode position` (default) |
| 7 | Direct-torque link | `direct_torque_link.py` | `--probe-only` with mode `direct_torque` |
| 8 | **Mode 2 — live torque** | `direct_torque_transport.py` | `--control-mode direct_torque` on **real** arm |
| 9 | Local J+M + latency | `local_dynamics.py`, `latency.py`, `timing.py` | `--dynamics-source local` |
| 10 | **Mode 3 — URScript** | `urscript_gen.py`, `urscript_transport.py` + [`URSCRIPT_INNER_LOOP.md`](URSCRIPT_INNER_LOOP.md) | `--control-mode urscript` |
| 11 | Dashboard / PolyScope | `dashboard.py` + checklists below | `tools/_check_ursim_remote.py` |

Shared controller math (not under `hardware/` but required for torque modes):

- `controller_core/x_axis_cartesian_impedance.py` — OSC law
- `controller_core/safety.py` — `ImpedanceSafetyMonitor` (torque path)
- `config/ur5e_mujoco_torque_osc_tuned.yaml` — gains + safety limits
- `transport_metrics.py` — move/hold pass criteria
- `simulation/ur5e_mujoco_torque.py` — `x_profile_target` (min-jerk profile)

---

## Three control modes

| Mode | Flag | Moves via | OSC torques | Works on URSim motion? |
|------|------|-----------|-------------|------------------------|
| **1 Position** | `--control-mode position` (default) | `servoL` @ 125 Hz | Shadow only (`tau_shadow` in trace) | **Yes** |
| **2 Direct torque** | `--control-mode direct_torque` | `directTorque()` @ 500 Hz | Live | **No** (API only) |
| **3 URScript** | `--control-mode urscript` | On-robot script @ 500 Hz | Live on PolyScope | Prefer real arm |

Bring-up order: **1 → 2 → 3**.

Primary CLIs (both accept `--control-mode`):

```bash
# Mode 1 — component test (URSim or real)
python tools/ur5e_direct_torque_x_transport.py --robot-ip <IP> \
  --control-mode position --target-x-delta 0.02 \
  --i-understand-this-moves-the-robot --yes

# Full pipeline (joint move → transport)
python tools/ur5e_direct_torque_height_latency_test.py --robot-ip <IP> \
  --control-mode position --target-x-delta 0.02 \
  --i-understand-this-moves-the-robot --yes
```

---

## Every relevant file (canonical list)

### Core library — `hardware/`

| File | Role | Learn? |
|------|------|--------|
| `safety.py` | Limits, `ConnectionHealth`, one-way `EStopLatch`, `CartesianMoveMonitor` | **Required** |
| `link.py` | `UR5eLink` — RTDE receive, `servoL`, `moveJ`, honest `read_state()` | **Required** |
| `motion.py` | One bounded Cartesian `servoL` move (`ur5e_move.py`) | **Required** |
| `dashboard.py` | Port 29999 power-on / remote-query helpers | **Required** |
| `poses.py` | Named joint poses (`HEIGHT_ALPHA_0_5_Q`, …) | Required for transport |
| `joint_motion.py` | `moveJ` to a named pose before transport | Required for transport |
| `control_mode.py` | Normalize `position` / `direct_torque` / `urscript` | Required |
| `x_transport.py` | Router that dispatches the three modes | **Required** |
| `position_transport.py` | Mode 1: min-jerk X via `servoL` + optional shadow OSC | **Required** |
| `direct_torque_link.py` | `UR5eDirectTorqueLink` — `directTorque` + RTDE J/M | Mode 2 |
| `direct_torque_transport.py` | Mode 2: 500 Hz OSC X move+hold | Mode 2 |
| `local_dynamics.py` | MuJoCo J(q), M(q) instead of RTDE dynamics calls | Mode 2 perf |
| `timing.py` | Loop period / lateness stats | Mode 2 |
| `latency.py` | Per-phase RTDE/controller timers | Mode 2 |
| `urscript_gen.py` | Render URScript from tuned YAML | Mode 3 |
| `urscript_transport.py` | Deploy script + Python supervisor | Mode 3 |
| `logging.py` | Thin re-export of `controller_core.logging_utils` | Optional |
| `__init__.py` | Public exports | Optional |

### CLI — `tools/` (active)

| Script | Moves? | Purpose |
|--------|--------|---------|
| `ur5e_connect.py` | **No** | Receive-only state (`--once` / `--watch`) |
| `ur5e_move.py` | Yes | Single-axis bounded `servoL` (first real motion) |
| `ur5e_move_joints.py` | Yes | `moveJ` to named pose |
| `ur5e_direct_torque_x_transport.py` | Yes | **Main** X transport CLI (`--control-mode`) |
| `ur5e_direct_torque_height_latency_test.py` | Yes | Joint move + transport + latency report |
| `ur5e_urscript_x_transport.py` | Yes | Dedicated Mode 3 entrypoint |
| `_bootstrap.py` | No | Repo-root path helper for tools |
| `_check_ursim_remote.py` | No | Dashboard status snapshot |
| `_clear_dashboard_mode_lock.py` | No | Undo dashboard mode lock |
| `_ursim_wait_and_power_on.py` | No | Wait for URSim dashboard after Docker start |
| `_rtde_control_probe.py` | Probe | Minimal control + `directTorque` API check |
| `_direct_torque_pulse_test.py` | Pulse | Short torque pulse (URSim → expect zero motion) |
| `reset_ursim_docker.sh` | No | Factory-reset URSim container |

### Assets / config / tests

| Path | Role |
|------|------|
| `assets/urscript/x_axis_osc_inner.script.template` | Mode 3 URScript template |
| `config/ur5e_mujoco_torque_osc_tuned.yaml` | Gains + safety (MuJoCo **and** hardware) |
| `tests/hardware/test_*.py` | Mocked RTDE unit tests — run before any live move |

### Docs (this folder)

| Doc | When |
|-----|------|
| [`HARDWARE_GUIDE.md`](HARDWARE_GUIDE.md) | Module deep dive, RTDE primer, troubleshooting |
| [`POLYSCOPE_PANEL_CHECKLIST.md`](POLYSCOPE_PANEL_CHECKLIST.md) | **At the robot** — teach-pendant switches |
| [`URSIM_REMOTE_CONTROL.md`](URSIM_REMOTE_CONTROL.md) | URSim Docker / remote control / fieldbus |
| [`URSCRIPT_INNER_LOOP.md`](URSCRIPT_INNER_LOOP.md) | Mode 3 details |
| [`AUTO_TUNING_PLAN.md`](AUTO_TUNING_PLAN.md) | Design-only plan for sim-gated Bayesian gain calibration on real hardware — not implemented yet |

### Not hardware (do not confuse)

These `tools/ur5e_*` scripts are **MuJoCo simulation**, not RTDE:

- `ur5e_mujoco_torque_experiments.py`, `ur5e_move_hold_transport.py`, `ur5e_x_frame_envelope.py`, …

### Archived / scratch (do not extend)

| Path | What |
|------|------|
| `archive/superseded/hardware_rtde_v1/` | Pre-2026-07-07 RTDE lane (ROS2 node bug: stale state) |
| `archive/superseded/hardware_scratch/` | One-off RTDE flag/sweep/parity scripts moved out of `tools/` |
| `docs/archive/hardware_v1/` | Historical warnings / staged bring-up notes |

---

## Minimal verification (no motion)

```bash
pytest tests/hardware/ -q
python tools/_check_ursim_remote.py
python tools/ur5e_connect.py --robot-ip 127.0.0.1 --once
python tools/ur5e_direct_torque_x_transport.py --robot-ip 127.0.0.1 --probe-only
```

---

## Data flow (one picture)

```
tools/ur5e_* ──► hardware/x_transport.run_x_transport(control_mode=...)
                      │
          ┌───────────┼───────────────────┐
          ▼           ▼                   ▼
   position_transport  direct_torque_transport  urscript_transport
   (servoL + shadow)   (directTorque @ 500 Hz)  (on-robot script)
          │                   │                       │
          └─────────┬─────────┴───────────────────────┘
                    ▼
              safety.py  +  link / direct_torque_link  +  dashboard
                    ▼
              PolyScope / UR5e (or URSim)
```
