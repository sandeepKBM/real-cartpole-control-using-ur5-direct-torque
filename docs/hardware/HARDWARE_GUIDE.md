# Hardware lane — complete learning guide

**Audience:** you, on the lab floor, without an AI assistant.  
**Last updated:** 2026-07-08  
**Code root:** `hardware/` + `tools/ur5e_*.py`

This document explains **what every hardware file does**, **how data flows**, **how to
run things safely**, and **what URSim can vs cannot validate**.

Shorter operational checklists live in:

- [`URSIM_REMOTE_CONTROL.md`](URSIM_REMOTE_CONTROL.md) — Docker / PolyScope bring-up
- [`README.md`](README.md) — file index and quick commands

---

## 1. Mental model (read this first)

The repo has **two different ways** to move a real UR5e. They are **not interchangeable**.

| Path | Robot API | Who does IK / dynamics | Controller in Python | Gravity comp |
|------|-----------|------------------------|----------------------|--------------|
| **Position** (`ur5e_move.py`) | `servoL` @ 125 Hz | **Robot firmware** | None (waypoints only) | N/A |
| **Direct torque** (`ur5e_direct_torque_x_transport.py`) | `directTorque()` @ **500 Hz** | **Your OSC controller** | `XAxisCartesianImpedanceController` | **Inside PolyScope** — do **not** add in Python |

```
                    ┌─────────────────────────────────────────┐
                    │           Your Python process            │
                    └─────────────────────────────────────────┘
                          │                    │
              receive-only│                    │control + receive
                          ▼                    ▼
              ┌──────────────────┐   ┌──────────────────────────┐
              │  RTDEReceive      │   │ RTDEReceive + RTDEControl │
              │  (port 30004)     │   │ (30004 + control script)  │
              └────────┬─────────┘   └────────────┬─────────────┘
                       │                          │
                       │    Dashboard (29999)     │
                       └──────────┬───────────────┘
                                  ▼
                    ┌─────────────────────────────────────────┐
                    │     PolyScope on UR5e / URSim Docker     │
                    └─────────────────────────────────────────┘
```

**Golden rules**

1. **`read_state()` never lies with stale data** — on failure it **raises** `RTDEStateError`.
2. **E-stop latch is one-way** — after `estop.trip()`, start a **new Python process**.
3. **Remote control must be ON** for RTDE **control** (not always for receive-only).
4. **URSim does not simulate `direct_torque()` motion** — API works, physics does not.
5. **Never add gravity torque on hardware** when using `directTorque()` — PolyScope adds it.

---

## 2. Directory map

### Core library (`hardware/`)

| File | Responsibility |
|------|----------------|
| `safety.py` | All limits, `ConnectionHealth`, `EStopLatch`, `CartesianMoveMonitor` |
| `link.py` | `UR5eLink` — RTDE receive + optional `servoL` control |
| `motion.py` | `move_cartesian_bounded()` — one safe Cartesian move via `servoL` |
| `direct_torque_link.py` | `UR5eDirectTorqueLink` — RTDE + `directTorque()` + Jacobian/M |
| `direct_torque_transport.py` | 500 Hz OSC X move+hold loop for real robot |
| `dashboard.py` | TCP dashboard on port 29999 (power on, remote check, mode unlock) |
| `logging.py` | Re-exports `controller_core.logging_utils` |
| `timing.py` | Loop period / lateness stats for staged bring-up |

Legacy / staging (not the main path): `ur5e_rtde_bridge.py`, `ur5e_control_session.py`,
`ur5e_stages.py`, `ros_topics.py`, `safety_limits.py`.

### CLI entrypoints (`tools/`)

| Script | Moves robot? | Purpose |
|--------|--------------|---------|
| `ur5e_connect.py` | **No** (by design) | Receive-only state read (`--once` / `--watch`) |
| `ur5e_move.py` | Yes (`servoL`) | Bounded single-axis Cartesian move |
| `ur5e_direct_torque_x_transport.py` | Yes (`directTorque`) | Tuned OSC X transport |
| `_check_ursim_remote.py` | No | Dashboard status snapshot |
| `_clear_dashboard_mode_lock.py` | No | Undo `set operational mode` lock |
| `_ursim_wait_and_power_on.py` | No | Wait for dashboard after Docker start |
| `reset_ursim_docker.sh` | No | Factory-reset URSim container |
| `_rtde_control_probe.py` | Probe only | Minimal control + directTorque test |
| `_direct_torque_pulse_test.py` | Pulse | 15 Nm joint test (URSim: expect zero motion) |

### Config

| File | Used by |
|------|---------|
| `config/ur5e_mujoco_torque_osc_tuned.yaml` | MuJoCo **and** hardware direct-torque (gains + safety) |

### Tests

| File | What it checks |
|------|----------------|
| `tests/hardware/test_link.py` | `UR5eLink` connect, read, servoL signature |
| `tests/hardware/test_motion.py` | Waypoint planning, move abort on safety |
| `tests/hardware/test_hardware_safety.py` | Limits, monitor, e-stop latch |
| `tests/hardware/test_direct_torque_helpers.py` | `rotvec_to_quat_wxyz` for TCP pose |

Run: `pytest tests/hardware/ -q`

---

## 3. RTDE primer (what the code assumes)

### Ports (URSim Docker / real robot)

| Port | Service |
|------|---------|
| 29999 | Dashboard server (plain TCP text commands) |
| 30001 | Primary client (URScript) |
| 30002 | Secondary client |
| 30003 | RTDE control script channel |
| 30004 | RTDE data (receive interface; control sync also uses this) |
| 6080 | noVNC teach pendant (URSim only) |

### Python package

```bash
pip install "ur-rtde>=1.6"
# imports:
from rtde_receive import RTDEReceiveInterface
from rtde_control import RTDEControlInterface
```

### `UR5eState` (shared dataclass)

Defined in `hardware/link.py`, used by both link classes:

| Field | Type | Meaning |
|-------|------|---------|
| `q` | `(6,)` rad | Joint positions |
| `qd` | `(6,)` rad/s | Joint velocities |
| `tcp_pose` | `(6,)` | `[x,y,z, rx,ry,rz]` — position m, orientation **rotation vector** |
| `host_stamp_ns` | int | `time.monotonic_ns()` when read completed |
| `robot_timestamp_s` | float or None | Controller clock if available |
| `safety_status` | int or None | Bitfield from `getSafetyStatusBits()` |

**TCP orientation is rotation vector (axis-angle), not quaternion.**  
`UR5eDirectTorqueLink.build_robot_state()` converts to `wxyz` quaternion via
`controller_core.kinematics_utils.rotvec_to_quat_wxyz()`.

### Exceptions

| Exception | When | Your action |
|-----------|------|-------------|
| `RTDELinkError` | Connect failed, wrong API, no `servoL`/`directTorque` | Fix network, remote control, ur_rtde version |
| `RTDEStateError` | Bad read, NaN, wrong shape, command before connect | Record failure in `ConnectionHealth`; abort or reconnect per policy |
| `EStopTripped` | `estop.raise_if_tripped()` after latch fired | **New process** — do not reset latch |

---

## 4. Module deep dive

### 4.1 `hardware/safety.py`

Two **families** of checks:

#### A. Absolute limits (`UR5eSafetyLimits`)

Manufacturer-scale ceilings — always apply:

- Joint limits `q_lower` / `q_upper` (from `controller_core.safety_utils`)
- `qd_max_radps` per joint
- `tcp_speed_max_mps`, `tcp_jump_max_m`
- `state_stale_max_s` — max age of last good read

Functions: `check_joint_state()`, `check_tcp_pose()`.

#### B. Move monitor (`CartesianMoveMonitor`)

Used only by **`servoL` moves** (`hardware/motion.py`). Pure geometry on TCP pose:

- Off-axis drift (Y/Z while moving X)
- Orientation error vs start
- TCP speed / acceleration from finite differences
- Waypoint jump detection
- Axis tracking error growth during hold

Defaults are **tighter than MuJoCo** because this was the first real-hardware Cartesian move.

#### C. Connection health (`ConnectionHealth`)

```python
health.record_success(host_stamp_ns)  # on every good read_state()
health.record_failure()               # returns True when streak >= 3 → fatal
health.is_alive()                     # False if stale or tripped
```

**Policy is caller-defined:**

- `ur5e_connect.py --watch` → reconnect with backoff
- `motion.py` / `direct_torque_transport.py` → **abort immediately** (no mid-motion reconnect)

#### D. E-stop latch (`EStopLatch`)

```python
estop.trip("reason string")   # permanent for this process
estop.raise_if_tripped()      # raises EStopTripped
```

**There is no `reset()`.** Design intent: after any safety stop, human inspects robot and
restarts script.

#### E. Impedance safety (direct torque only)

`direct_torque_transport.py` uses `controller_core.safety.ImpedanceSafetyMonitor` — same
family as MuJoCo (Y/Z drift, orientation, joint velocity). Loaded from YAML
`controller.safety` section.

---

### 4.2 `hardware/link.py` — `UR5eLink`

**Purpose:** honest RTDE wrapper for **position streaming**.

```python
link = UR5eLink(robot_ip, frequency_hz=125.0)
link.connect(with_control=False)   # receive only
link.connect(with_control=True)    # also opens control + verifies servoL signature

state = link.read_state()        # raises RTDEStateError on any problem

link.servo_l(pose, speed=0.25, acceleration=1.2, time_s=0.012,
             lookahead_time=0.1, gain=300.0)
link.servo_stop()
link.safe_stop("reason")
link.disconnect()
```

**Why servoL signature is verified at connect:**

Older `ur_rtde` versions could swap `speed` and `acceleration` positionally with no error.
The code checks parameter **names** match:

`(pose, speed, acceleration, time, lookahead_time, gain)`

**`servoL` semantics:**

- You send a **TCP pose** `[x,y,z,rx,ry,rz]` every cycle.
- Robot firmware runs its own IK and joint controllers.
- `time` should be **> control period** (code uses `dt * 1.5` at 125 Hz).

---

### 4.3 `hardware/motion.py` — bounded Cartesian move

**Trajectory:** quintic min-jerk along one world axis:

```python
s(tau) = 10*tau^3 - 15*tau^4 + 6*tau^5   # tau in [0,1]
```

Peak velocity: `v_peak = 1.875 * |distance| / duration`

**Control loop (every 1/125 s):**

1. `servo_l(waypoint)`
2. `read_state()`
3. `monitor.check(...)` — any failure → `safe_stop` + `estop.trip`
4. Sleep remainder of period

**Does not:**

- Run impedance controller
- Use Jacobian
- Send torques

---

### 4.4 `hardware/direct_torque_link.py` — `UR5eDirectTorqueLink`

**Purpose:** same receive path as `UR5eLink`, but control via **`directTorque()`**.

```python
link = UR5eDirectTorqueLink(robot_ip, frequency_hz=500.0)
link.connect()                    # always opens BOTH receive and control

state = link.read_state()
J = link.get_jacobian()           # 6x6 from RTDE
M = link.get_mass_matrix()        # 6x6 from RTDE

link.direct_torque(tau_nm, friction_comp=True)
```

**`build_robot_state()`** maps RTDE data into the dict expected by
`XAxisCartesianImpedanceController`:

```python
{
  "time", "q", "qd",
  "ee_pos", "ee_quat",           # from TCP pose
  "ee_lin_vel", "ee_ang_vel",    # J @ qd
  "jacobian", "mass_matrix",
  "target_x", "target_x_vel",
  "transport_axis_index": 0,     # world X
}
```

**No `gravity_torque` key** — intentional. PolyScope compensates gravity inside
`direct_torque()`.

**`safe_stop()`** tries `stopJ(2.0)`, `servoStop`, `stopScript` then disconnects.

**500 Hz requirement:** `direct_torque()` must be called every **2 ms**. If you miss
calls, PolyScope reverts to position control mode.

---

### 4.5 `hardware/direct_torque_transport.py`

**Purpose:** run the **same tuned OSC controller** as MuJoCo on real hardware.

**Flow:**

```
load YAML → XAxisCartesianImpedanceController + ImpedanceSafetyMonitor
     ↓
connect → read_state → controller.reset_from_state()
     ↓
loop at 500 Hz until duration_s:
     read_state()
     x_profile_target("min_jerk_move_hold", ...)  → target_x, target_x_vel
     build_robot_state()
     output = controller.compute(robot_state)   → tau (6,)
     safety.check(...)
     link.direct_torque(tau)
     append trace row
     sleep ~0.95 * dt
     ↓
finally: zero torque → safe_stop
     ↓
write trace.jsonl + summary.json
     ↓
summarize_move_hold_trace() + compute_valid_move_hold_metrics()
```

**Success criteria** (`summary["success"]`):

- No e-stop
- `termination_reason == "duration_complete"`
- `valid_move_and_hold` from transport metrics (tracking, drift, hold quality)

**Config requirements:**

- `hardware.rtde_frequency_hz` ≈ **500** in YAML
- Uses `controller.gains` and `controller.safety` from tuned OSC YAML

**Important MuJoCo vs hardware difference:**

| | MuJoCo | Hardware direct torque |
|---|--------|------------------------|
| Gravity | Python adds `tau_gravity` | **Omit** — PolyScope handles |
| Jacobian/M | Pinocchio / MuJoCo | RTDE `getJacobian()` / `getMassMatrix()` |
| Physics | Sim integrates | Real robot / **not URSim** |

---

### 4.6 `hardware/dashboard.py`

Plain TCP to port **29999**. First message from server is discarded (welcome banner).

```python
from hardware.dashboard import (
    dashboard_command,
    query_remote_control,
    power_on_and_release,
    clear_operational_mode,
)

dashboard_command("192.168.1.10", "robotmode")
query_remote_control("192.168.1.10")  # True/False
clear_operational_mode("192.168.1.10")  # undo dashboard mode lock
```

**Common dashboard commands:**

| Command | Effect |
|---------|--------|
| `power on` | Power on arm |
| `brake release` | Release brakes |
| `is in remote control` | `true` / `false` |
| `get operational mode` | `MANUAL`, `AUTOMATIC`, `NONE` |
| `set operational mode automatic` | **Locks** mode in PolyScope until cleared |
| `clear operational mode` | Returns mode control to teach pendant |
| `play` | Start program (often fails with empty program) |

**Remote control cannot be enabled via dashboard** — teach pendant only.

---

## 5. CLI tools — step by step

### 5.1 `tools/ur5e_connect.py` — safe first step

**Cannot move the robot** — does not import `motion.py`.

```bash
# One-shot state
python tools/ur5e_connect.py --robot-ip <IP> --once

# Continuous monitor (reconnects on failure)
python tools/ur5e_connect.py --robot-ip <IP> --watch --frequency-hz 125
```

**Interpret output:**

```
q  = [...] rad          # joint positions
qd = [...] rad/s        # should be ~0 when still
tcp_pose = [x,y,z,rx,ry,rz]
```

URSim @ 127.0.0.1, real robot @ static IP on your lab VLAN.

---

### 5.2 `tools/ur5e_move.py` — first real motion (position mode)

**Recommended sequence:**

```bash
# 1. Dry run — no network
python tools/ur5e_move.py --robot-ip <IP> --axis y --direction left \
  --distance-m 0.02 --dry-run

# 2. Tiny move
python tools/ur5e_move.py --robot-ip <IP> --axis y --direction left \
  --distance-m 0.02 --i-understand-this-moves-the-robot

# 3. Larger move after confirming axis sign
python tools/ur5e_move.py --robot-ip <IP> --axis y --direction left \
  --distance-m 0.15 --i-understand-this-moves-the-robot --yes
```

**Axis convention:** `--direction left` = **+distance** along `--axis`. If the robot
moves the wrong way, swap `left`/`right` or pick a different axis — depends on mounting.

**Guards:**

- `--i-understand-this-moves-the-robot` required
- Interactive `Type MOVE` unless `--yes`
- Peak velocity checked against `CartesianMoveLimits.max_tcp_speed_mps`

---

### 5.3 `tools/ur5e_direct_torque_x_transport.py` — torque mode

**Prerequisites:** see [`URSIM_REMOTE_CONTROL.md`](URSIM_REMOTE_CONTROL.md)

```bash
# Probe — zero torque 1 s (validates API on URSim)
python tools/ur5e_direct_torque_x_transport.py --robot-ip <IP> --probe-only \
  --skip-dashboard-power-on

# Small X move — USE REAL ROBOT for visible motion
python tools/ur5e_direct_torque_x_transport.py --robot-ip <IP> \
  --target-x-delta 0.02 --move-duration 1.0 --duration 3.0 \
  --i-understand-this-moves-the-robot --yes
```

**Outputs:** `outputs/hardware_direct_torque/x_transport_<timestamp>/`

- `trace.jsonl` — per-step q, tau, errors
- `summary.json` — pass/fail metrics

**Key summary fields:**

| Field | Good value (0.02 m move) |
|-------|---------------------------|
| `achieved_x_delta_m` | ~0.02 |
| `valid_move_and_hold` | `true` |
| `max_abs_qd_radps` | < safety limit |
| `move_failure_reason` | absent |

---

## 6. URSim vs real robot

### What URSim **can** validate

- Docker networking, dashboard, power on
- Remote control enablement
- RTDE receive (`ur5e_connect.py`)
- RTDE control socket sync (PolyScope version 5.25 in logs)
- `directTorque()` returns without error at 500 Hz

### What URSim **cannot** validate

- **Any joint motion from `direct_torque()`** — official UR docs: torque commands have
  no effect in URSim
- Tuned gain behavior, tracking error, drift — use **MuJoCo** or **real arm**

### Simulation toggle in PolyScope

Turning **Simulation ON** in the teach pendant does **not** enable torque physics in
URSim for `direct_torque()`. We confirmed zero `qd` and zero `dq` after 15 Nm for 1 s.

---

## 7. Lab-day checklist (real UR5e)

### Before you leave your desk

- [ ] `pytest tests/hardware/ -q` passes
- [ ] `ur_rtde >= 1.6` in venv: `pip show ur-rtde`
- [ ] Config copied: `config/ur5e_mujoco_torque_osc_tuned.yaml`
- [ ] Read this guide + `URSIM_REMOTE_CONTROL.md` if using Docker practice

### At the robot

1. **E-stop accessible**, workspace clear, speed slider reasonable
2. `python tools/ur5e_connect.py --robot-ip <IP> --once` — sane `q`, `tcp_pose`
3. PolyScope: **Remote control ON** (for torque path)
4. **Position path first:** `ur5e_move.py` with 0.02 m
5. **Torque path:** `--probe-only` then `0.02 m` X transport
6. Inspect `summary.json` before increasing to `0.10 m`

### If something goes wrong

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| `RTDE connect failed` + sync timeout | Remote off, fieldbus conflict | Remote ON; disable PROFINET/EtherNet/IP |
| `is in remote control: false` | Not in Remote profile | PolyScope UI |
| `controlled by dashboard server` | Stale `set operational mode` | `_clear_dashboard_mode_lock.py` |
| `RTDE input registers already in use` | PROFINET/EtherNet/IP | Disable in Services + Installation |
| Probe OK, zero motion on URSim | Expected | Use real robot |
| Probe OK, zero motion on **real** robot | 500 Hz loop broken, protective stop | Check `qd`, safety popup, reduce delta |
| `EStopTripped` | Any safety violation | New Python process; inspect arm |

---

## 8. Architecture diagram — direct torque path

```mermaid
sequenceDiagram
    participant CLI as ur5e_direct_torque_x_transport
    participant DT as direct_torque_transport
    participant Link as UR5eDirectTorqueLink
    participant Ctrl as XAxisCartesianImpedanceController
    participant Safe as ImpedanceSafetyMonitor
    participant RTDE as RTDE / PolyScope

    CLI->>Link: connect()
    Link->>RTDE: RTDEReceive + RTDEControl
    DT->>Link: read_state()
    DT->>Ctrl: reset_from_state()
    loop 500 Hz
        DT->>Link: read_state()
        DT->>DT: x_profile_target(min_jerk_move_hold)
        DT->>Link: build_robot_state()
        DT->>Ctrl: compute() → tau
        DT->>Safe: check()
        DT->>Link: direct_torque(tau)
        Link->>RTDE: directTorque(tau, friction_comp=True)
    end
    DT->>Link: safe_stop()
```

---

## 9. Design decisions (why the code looks this way)

### Why two link classes?

`servoL` and `directTorque()` need different connect policies, stop methods, and state
enrichment (Jacobian/M only for torque). Merging would blur safety boundaries.

### Why `read_state()` raises instead of returning last good value?

A previous ROS2 node kept stale state after RTDE failures and continued moving. Raising
forces every caller to handle failure explicitly.

### Why no reconnect mid-motion?

Reconnecting during motion could mask a real fault (protective stop, cable pull). Idle
watch mode may reconnect; motion modes abort.

### Why motion opt-in flags?

`--i-understand-this-moves-the-robot` + typed `MOVE` prevents accidental agent/script
motion on a real cobot.

### Why reuse MuJoCo YAML on hardware?

Same tuned gains (`kp_x=400`, `kd_x=40`, `kp_rot=0`, etc.) were validated in simulation.
Hardware path omits sim-only gravity and uses RTDE for J, M.

---

## 10. Config YAML — what hardware reads

From `config/ur5e_mujoco_torque_osc_tuned.yaml`:

```yaml
hardware:
  rtde_frequency_hz: 500.0    # direct_torque_transport enforces ~500

controller:
  gains:
    kp_x: 400.0
    kd_x: 40.0
    kp_rot: 0.0               # task rot unstable at wrist singularity — see YAML comments
    kd_rot: 10.0
    kp_posture: 25.0
    kd_posture: 6.0
    kd_joint: 4.0
    # ... kp_y, kp_z, etc.
  safety:
    max_abs_y_drift_m: 0.03
    max_abs_z_drift_m: 0.03
    max_orientation_error_rad: 0.25
    max_joint_velocity_radps: 3.0
  task_space_inertia_shaping: true
  nullspace_posture: true
  posture_reanchor_on_settle: true
```

MuJoCo-only keys (`mujoco.scene_xml`, `gravity_mode`, etc.) are ignored by hardware code.

---

## 11. Extending the hardware lane (if you need to)

### Add Y or Z torque transport

1. Copy `direct_torque_transport.py` pattern
2. Set `transport_axis_index` in `build_robot_state()` (or extend it)
3. Use appropriate profile from `simulation/ur5e_mujoco_torque.py`
4. Keep **500 Hz** and **no gravity comp in Python**

### Add ROS2 bridge

See `hardware/ros_topics.py` and `ros2_ws/` — not the active path for bring-up.
Prefer proving bare RTDE first.

### Tune gains on hardware

Start from MuJoCo-validated YAML. Change one gain at a time. Log every run to
`outputs/hardware_direct_torque/`. Compare `summary.json` fields to MuJoCo baselines.

---

## 12. Quick reference commands

```bash
# Venv (WSL example)
source ~/ur5e_repo/.venv/bin/activate
cd /mnt/c/Users/sandr/Downloads/real-cartpole-control-using-ur5-direct-torque

# Tests
pytest tests/hardware/ -q

# Status
python tools/_check_ursim_remote.py

# Receive only
python tools/ur5e_connect.py --robot-ip 127.0.0.1 --once

# Torque probe
python tools/ur5e_direct_torque_x_transport.py --robot-ip 127.0.0.1 \
  --probe-only --skip-dashboard-power-on

# Factory reset URSim
bash tools/reset_ursim_docker.sh
python tools/_ursim_wait_and_power_on.py
```

---

## 13. Related reading

| Document | Topic |
|----------|-------|
| [`URSIM_REMOTE_CONTROL.md`](URSIM_REMOTE_CONTROL.md) | Docker, PolyScope, passwords, fieldbus |
| [`README.md`](README.md) | Short index |
| `config/ur5e_mujoco_torque_osc_tuned.yaml` | Gain tuning rationale (comments) |
| `controller_core/x_axis_cartesian_impedance.py` | OSC math |
| `AGENTS.md` §4 | Project-wide hardware guardrails |
| [UR `direct_torque()` manual](https://www.universal-robots.com/manuals/EN/HTML/SW5_25/Content/prod-scriptmanual/all_scripts/direct_torque.htm) | Official API semantics |

---

*When in doubt: connect read-only first, move small second, read `summary.json` third.*
