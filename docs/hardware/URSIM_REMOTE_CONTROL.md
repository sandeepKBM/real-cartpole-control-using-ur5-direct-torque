# URSim bring-up for direct-torque RTDE control (PolyScope 5.25)

This guide covers everything needed to run
`tools/ur5e_direct_torque_x_transport.py` against **URSim Docker** on Windows/WSL.
It documents failures we hit during bring-up (June–July 2026) so you do not have
to rediscover them.

PolyScope API:
[`direct_torque()`](https://www.universal-robots.com/manuals/EN/HTML/SW5_25/Content/prod-scriptmanual/all_scripts/direct_torque.htm)

Python path: `ur_rtde >= 1.6` → `RTDEControlInterface.directTorque()`

---

## Quick checklist (do in this order)

| Step | Where | What |
|------|--------|------|
| 1 | Docker | Start URSim with all ports published |
| 2 | PolyScope UI | Power on, release brake, confirm safety |
| 3 | Top-right mode | **Manual → Automatic** (required before enabling remote control) |
| 4 | Settings → System → Remote control | Tap **Enable** |
| 5 | Top-right mode | **Automatic → Remote** |
| 6 | Top-right mode | Switch to **Local Control** to edit security |
| 7 | Settings → Security → Services | Unlock with admin password **`easybot`** |
| 8 | Services | **Disable** PROFINET, EtherNet/IP, Modbus TCP |
| 9 | **Installation** tab → Fieldbus | **Disable** PROFINET / EtherNet/IP adapters there too |
| 10 | Services | Keep **RTDE**, Dashboard, Primary/Secondary/Real-Time Client **enabled** |
| 11 | Apply / restart | Use **Apply and Restart** or `docker restart ursim` |
| 12 | Repeat steps 3–5 | Re-enable **Remote** after restart |
| 13 | WSL | Run verification commands below |

---

## 1. Start URSim (Docker)

```bash
docker run -d --name ursim \
  -p 5900:5900 -p 6080:6080 -p 29999:29999 \
  -p 30001:30001 -p 30002:30002 -p 30003:30003 -p 30004:30004 \
  -e UR_ROBOT_TYPE=UR5 \
  universalrobots/ursim_e-series
```

Teach pendant UI:

**http://localhost:6080/vnc.html?host=localhost&port=6080** → **Connect**

| Port | Use |
|------|-----|
| 6080 | Browser VNC UI |
| 29999 | Dashboard server |
| 30003 | RTDE control script channel |
| 30004 | RTDE data (receive + control sync) |

Run Python from **WSL** with the repo venv:

```bash
cd /mnt/c/Users/sandr/Downloads/real-cartpole-control-using-ur5-direct-torque
source ~/ur5e_repo/.venv/bin/activate
```

---

## 2. Power on

1. If prompted: **Confirm Safety Configuration**
2. Bottom-left: **Power on** → **Release brake**
3. Status should be **Normal** (green)

Optional dashboard check:

```bash
python -c "from hardware.dashboard import power_on_and_release; print(power_on_and_release('127.0.0.1'))"
```

---

## 3. Enable remote control

### Error: cannot enable from Manual mode

```text
Remote Control cannot be enabled from Manual mode.
To ensure safe usage, Remote Control can only be enabled from Automatic or Remote mode.
```

**Fix:** top-right **Manual → Automatic** first, then enable remote control.

### Correct UI sequence

1. Top-right: **Manual → Automatic**
2. ☰ **Settings → System → Remote control → Enable**
3. Top-right: **Automatic → Remote**

### Dashboard verification (must be `True` before motion)

```bash
python tools/_check_ursim_remote.py
# or:
python -c "from hardware.dashboard import query_remote_control; print(query_remote_control('127.0.0.1'))"
```

---

## 4. Security / Services (admin unlock)

Security menus are **greyed out** until you unlock with the admin password.

| Item | Value |
|------|--------|
| Default admin password | **`easybot`** |
| URSim fallback | blank / empty (tap Unlock with no text) |

Steps:

1. Top-right: **Remote → Local Control** (to edit security)
2. ☰ **Settings → Security → Services**
3. Bottom of screen: enter **`easybot`** → **Unlock**

### Services to **enable**

- Dashboard Server
- Primary Client Interface
- Secondary Client Interface (optional)
- Real-Time Client Interface
- **Real-Time Data Exchange (RTDE)**

### Services to **disable** (critical for RTDE control)

- **PROFINET**
- **EtherNet/IP**
- **Modbus TCP**

On URSim 5.25 these are enabled by default. RTDE **receive** may still work, but
`RTDEControlInterface` / `directTorque()` will fail while fieldbus holds RTDE
input registers.

Also open the **Installation** tab (top bar, not Settings) → **Fieldbus** and
disable PROFINET / EtherNet/IP adapters there. Disabling only in Services is not
always enough.

After changing fieldbus: **Apply and Restart** PolyScope, or:

```bash
docker restart ursim
```

Then repeat power-on and remote-control steps.

---

## 5. Dashboard operational-mode lock (do not get stuck)

If PolyScope shows:

```text
Unable to change operational mode. It is currently controlled by the dashboard server.
```

Something sent `set operational mode ...` over the dashboard (our early debug
script did this once). Release it from WSL:

```bash
python tools/_clear_dashboard_mode_lock.py
```

Expected response:

```text
No longer controlling the operational mode. Current operational mode: 'NONE'.
```

**Do not** use `set operational mode` during normal bring-up unless you intend
to hold the mode from an external PLC.

---

## 6. Verification commands

### 6a. Dashboard status

```bash
python tools/_check_ursim_remote.py
```

Healthy pre-motion output:

```text
robotmode: Robotmode: RUNNING
is in remote control: true
get operational mode: NONE   # or MANUAL/AUTOMATIC if not dashboard-locked
safetymode: Safetymode: NORMAL
query_remote_control: True
```

### 6b. RTDE receive-only

```bash
python tools/ur5e_connect.py --robot-ip 127.0.0.1 --once
```

Should print joint positions and TCP pose.

### 6c. RTDE control + directTorque probe

```bash
python tools/_rtde_control_probe.py
# or:
python tools/ur5e_direct_torque_x_transport.py --robot-ip 127.0.0.1 --probe-only --skip-dashboard-power-on
```

Success:

```text
PROBE OK: receive + control + directTorque() path works.
```

### 6d. Fieldbus conflict sweep (diagnostic)

```bash
python tools/_rtde_control_sweep.py
```

At 125 Hz this often surfaces the explicit fieldbus error:

```text
One of the RTDE input registers are already in use!
Currently you must disable the EtherNet/IP adapter, PROFINET or any MODBUS unit
configured on the robot.
```

---

## 7. Run direct-torque X transport

**Probe** (zero torque, 1 s):

```bash
python tools/ur5e_direct_torque_x_transport.py --robot-ip 127.0.0.1 --probe-only --skip-dashboard-power-on
```

**Small live move** (start here):

```bash
python tools/ur5e_direct_torque_x_transport.py --robot-ip 127.0.0.1 \
  --target-x-delta 0.02 --move-duration 1.0 --duration 3.0 \
  --i-understand-this-moves-the-robot --yes
```

**Full MuJoCo parity move** (0.10 m):

```bash
python tools/ur5e_direct_torque_x_transport.py --robot-ip 127.0.0.1 \
  --target-x-delta 0.10 --move-duration 1.0 --duration 3.0 \
  --i-understand-this-moves-the-robot --yes
```

Outputs: `outputs/hardware_direct_torque/x_transport_<timestamp>/`

---

## Troubleshooting table

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| `is in remote control: false` | Remote not enabled or not in Remote profile | Steps 3–5 |
| `Remote Control cannot be enabled from Manual mode` | Still in Manual | Switch to Automatic first |
| Security / Services greyed out | Admin lock | Password **`easybot`** → Unlock |
| `controlled by the dashboard server` | Dashboard holds operational mode | `python tools/_clear_dashboard_mode_lock.py` |
| Receive works, control sync timeout, `PolyScope major version: 0` | Fieldbus conflict or RTDE service off | Disable PROFINET/EtherNet/IP in **Services + Installation**, restart |
| `RTDE input registers are already in use` | PROFINET / EtherNet/IP / Modbus still active | Disable in Installation → Fieldbus, restart URSim |
| `Failed to start RTDE data synchronization` | Remote off **or** fieldbus conflict | Check remote + fieldbus |
| `Failed to execute: play` | No program loaded / wrong mode | Not required for `directTorque()` probe; ignore for probe-only |

---

## Control notes

- Call `direct_torque()` at **500 Hz** (every 2 ms).
- **Do not** add gravity compensation in Python; PolyScope adds it inside
  `direct_torque()`.
- After a safety stop, **restart the Python process** (e-stop latch is one-way).
- `ur_rtde` 1.6.3+ is required (`pip show ur-rtde`).

### URSim limitation: `direct_torque()` does not move the arm

URSim accepts `directTorque()` RTDE calls (probe passes, torques are sent), but
**torque commands have no physical effect in the simulator**. This is documented
in Universal Robots' own client-library examples:

> On URSim torque commands don't have any effect.

Symptoms on URSim even with **Simulation ON** and remote control enabled:

- `PROBE OK` (API path works)
- `achieved_x_delta_m: 0.0`, `max_abs_qd_radps: 0.0` on transport runs
- `tools/_direct_torque_pulse_test.py` shows `max_abs_dq: 0.0` after 15 Nm for 1 s

Use URSim to validate **RTDE connect, remote control, and dashboard bring-up**.
Use **MuJoCo** (`tools/ur5e_mujoco_torque_experiments.py`) or a **physical UR5e**
to validate actual torque motion.

---

## Helper scripts (repo)

| Script | Purpose |
|--------|---------|
| `tools/reset_ursim_docker.sh` | **Factory reset**: remove container and start fresh (no volumes) |
| `tools/_ursim_wait_and_power_on.py` | Wait for dashboard after start, power on |
| `tools/_check_ursim_remote.py` | Dashboard status snapshot |
| `tools/_clear_dashboard_mode_lock.py` | Undo dashboard operational-mode lock |
| `tools/_rtde_control_probe.py` | Minimal control + directTorque test |
| `tools/_rtde_control_sweep.py` | Frequency / flag sweep + fieldbus error surfacing |
| `tools/ur5e_direct_torque_x_transport.py` | Full tuned OSC X transport CLI |
| `hardware/dashboard.py` | Dashboard helpers (`query_remote_control`, `clear_operational_mode`) |

---

## Factory reset (worst case: wrong admin password / stuck state)

Your `ursim` container has **no host volume mounts**, so removing it wipes all
PolyScope state (passwords, installations, fieldbus config).

```bash
# From Git Bash / WSL / PowerShell:
bash tools/reset_ursim_docker.sh

# Or manually:
docker rm -f ursim
docker run -d --name ursim \
  -p 5900:5900 -p 6080:6080 -p 29999:29999 \
  -p 30001:30001 -p 30002:30002 -p 30003:30003 -p 30004:30004 \
  -e UR_ROBOT_TYPE=UR5 \
  universalrobots/ursim_e-series
```

Then wait and power on:

```bash
python tools/_ursim_wait_and_power_on.py
```

On a **fresh** container the admin password is the factory default:

| Password | When |
|----------|------|
| **`easybot`** | Default on new / factory-reset URSim |
| blank (empty) | Tap Unlock with no text if `easybot` fails |

Linux root login inside the container (rarely needed): user `root`, password `easybot`.

After reset, redo the full checklist at the top (remote control, disable
fieldbus in **Installation → Fieldbus**, then Remote again).
