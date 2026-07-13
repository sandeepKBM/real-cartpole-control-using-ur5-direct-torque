# PolyScope panel checklist — before running hardware control

Use this at the teach pendant **before** you run Python (`ur5e_connect.py`,
`ur5e_move.py`, or `ur5e_direct_torque_x_transport.py`).

**Real UR5e:** same UI as URSim, but torque motion **actually works** on hardware.  
**URSim Docker:** use this for RTDE bring-up only; `direct_torque()` will not move the arm.

---

## Quick checklist (print this)

### Every time (real robot or URSim)

| Step | PolyScope panel | Setting |
|------|-----------------|--------|
| 1 | Bottom-left status | **Normal** (green) — power on, brakes released |
| 2 | Bottom-right speed | **100%** or your lab’s approved value |
| 3 | ☰ → Settings → System → **Remote control** | **Enable** (blue border on Enable) |
| 4 | Top-right mode button | Switch to **Remote** (not Manual, not Local only) |
| 5 | From laptop | `python tools/_check_ursim_remote.py` → must show `query_remote_control: True` |

### Security / Services (do once after fresh install or factory reset)

| Step | Panel path | Action |
|------|------------|--------|
| 6 | Top-right → **Local Control** (temporarily) | Needed to edit security |
| 7 | ☰ → Settings → **Security** → **Services** | Enter admin password **`easybot`** → **Unlock** |
| 8 | Services list | **Enable:** RTDE, Dashboard, Primary Client, Real-Time Client |
| 9 | Services list | **Disable:** PROFINET, EtherNet/IP, Modbus TCP |
| 10 | **Installation** tab (top bar) → **Fieldbus** | Disable PROFINET / EtherNet/IP adapters there too |
| 11 | Top-right | Switch back to **Remote** |

### Before **direct torque** on a **real robot**

| Step | Panel | Action |
|------|-------|--------|
| 12 | Workspace | Clear — no hands in reach |
| 13 | E-stop | Accessible |
| 14 | Remote | Stay in **Remote** for whole Python run |
| 15 | Do **not** touch | Manual / Local / Freedrive while script runs |

### URSim only (Docker practice)

| Item | Note |
|------|------|
| Simulation toggle (bottom-right) | ON or OFF — **does not matter** for `direct_torque()`; URSim still won’t move |
| noVNC “CCCC” text | Font glitch — button still works; or use VNC port **5900** |
| After `docker restart` | Redo power on + Remote control (steps 1–5) |

---

## What to switch OFF / disable

These **block RTDE control** or cause register conflicts:

- **PROFINET** — OFF in Services **and** Installation → Fieldbus  
- **EtherNet/IP** — OFF in Services **and** Installation → Fieldbus  
- **Modbus TCP** — OFF (recommended)  
- **Manual mode** when enabling Remote control — use **Automatic** first, then Enable remote, then **Remote**

You do **not** need External Control URCap for this repo’s `directTorque()` path.

---

## What to switch ON / enable

- **Remote control** (Settings → System)  
- **Remote** profile (top-right) — not just “feature enabled”  
- **RTDE** service (Security → Services)  
- **Dashboard Server** (for `hardware/dashboard.py` checks)  
- Robot **powered on** + **brakes released**

---

## Mode button (top-right) — what each means

| What you see | Meaning | Use when |
|--------------|---------|----------|
| **Manual** (hand) | Teach/local jogging | **Not** for external RTDE control |
| **Automatic** (play/spiral) | Needed to **enable** remote control in settings | Step before Enable |
| **Remote** | External PC owns RTDE control | **Running Python scripts** |
| **Local Control** | Edit security settings | Unlock Services, then go back to Remote |

**Order that worked for us:**

1. Manual → **Automatic**  
2. Settings → Remote control → **Enable**  
3. Automatic → **Remote**  
4. (For Services) Remote → **Local** → unlock Services → back to **Remote**

---

## Admin password

| Situation | Password |
|-----------|----------|
| Fresh URSim / factory default | **`easybot`** |
| If that fails | Try **blank** (empty) + Unlock |
| You changed it earlier | Use your password — **cannot recover** admin password |

---

## If PolyScope blocks you

| Message / symptom | Fix |
|-------------------|-----|
| “Remote Control cannot be enabled from Manual mode” | Top-right → **Automatic**, then Enable |
| “Controlled by the dashboard server” | Laptop: `python tools/_clear_dashboard_mode_lock.py` |
| Security menu greyed out | Security → enter **`easybot`** at bottom → **Unlock** |
| Top-right shows garbled “CCCC” | Click it anyway, or use real VNC client on port 5900 |
| Python: `is in remote control: false` | Repeat steps 3–4 above |

---

## What to do **during** the Python run

**Do:**

- Keep PolyScope in **Remote**  
- Keep laptop script running until it exits cleanly  
- Watch for protective stop / red status  

**Do not:**

- Switch to Manual or Local mid-run  
- Press Freedrive  
- Stop script with Ctrl+C without expecting `safe_stop` — script tries to stop arm in `finally`  
- Re-run immediately after e-stop without **restarting Python** (e-stop latch is one-way in code)

---

## What to do **after** the run

| If | Then |
|----|------|
| Run finished OK | Script calls `safe_stop` / disconnect — you can switch to Manual for jogging |
| Safety stop / fault | Clear fault on pendant, inspect arm, **restart Python process** |
| Want to jog by hand | Switch **Remote → Manual** on top-right |
| Done for the day | Optional: Remote → Local, or power off from pendant |

---

## Verify from laptop (before any motion)

```bash
# 1. Remote control
python tools/_check_ursim_remote.py
# expect: query_remote_control: True

# 2. Receive-only
python tools/ur5e_connect.py --robot-ip <IP> --once

# 3. Torque API (no motion test on URSim)
python tools/ur5e_direct_torque_x_transport.py --robot-ip <IP> --probe-only --skip-dashboard-power-on
# expect: PROBE OK
```

---

## Real robot — first motion order

1. Panel checklist above (Remote ON)  
2. `ur5e_connect.py --once`  
3. **Position test first:** `ur5e_move.py` with `0.02 m` (safer, firmware IK)  
4. **Then torque:** `ur5e_direct_torque_x_transport.py` with `--target-x-delta 0.02`  
5. Read `outputs/hardware_direct_torque/.../summary.json` before going to `0.10 m`

---

## Related docs

- [`HARDWARE_GUIDE.md`](HARDWARE_GUIDE.md) — full code + RTDE guide  
- [`URSIM_REMOTE_CONTROL.md`](URSIM_REMOTE_CONTROL.md) — Docker, factory reset, fieldbus detail
