# Hardware lane — documentation index

**Start here if you are learning the hardware code:**

## [`HARDWARE_GUIDE.md`](HARDWARE_GUIDE.md) — full learning guide

Module-by-module explanation, RTDE primer, both control paths (`servoL` vs
`directTorque`), safety system, CLI tools, lab checklist, troubleshooting, and
URSim vs real-robot limits.

## Operational guides

| Document | When to use |
|----------|-------------|
| [`POLYSCOPE_PANEL_CHECKLIST.md`](POLYSCOPE_PANEL_CHECKLIST.md) | **At the robot** — what to switch on/off on the teach pendant before Python |
| [`URSIM_REMOTE_CONTROL.md`](URSIM_REMOTE_CONTROL.md) | URSim Docker, PolyScope remote control, admin password, fieldbus, factory reset |
| [`ur5e_rtde_minimal_test_plan.md`](ur5e_rtde_minimal_test_plan.md) | Older staged RTDE bring-up notes |
| [`ur5e_direct_torque_warning.md`](ur5e_direct_torque_warning.md) | Historical warning (superseded by direct-torque lane) |

## Code layout (quick)

| Path | Role |
|------|------|
| `hardware/safety.py` | Limits, connection health, e-stop, Cartesian monitor |
| `hardware/link.py` | `UR5eLink` — receive + `servoL` |
| `hardware/motion.py` | Bounded Cartesian move |
| `hardware/direct_torque_link.py` | `UR5eDirectTorqueLink` — `directTorque()` @ 500 Hz |
| `hardware/direct_torque_transport.py` | OSC X move+hold on real robot |
| `hardware/dashboard.py` | Dashboard port 29999 helpers |
| `tools/ur5e_connect.py` | Receive-only (cannot move) |
| `tools/ur5e_move.py` | Position-mode move |
| `tools/ur5e_direct_torque_x_transport.py` | Torque-mode X transport |

## Minimal verification (WSL)

```bash
source ~/ur5e_repo/.venv/bin/activate
cd /mnt/c/Users/sandr/Downloads/real-cartpole-control-using-ur5-direct-torque

pytest tests/hardware/ -q
python tools/_check_ursim_remote.py
python tools/ur5e_connect.py --robot-ip 127.0.0.1 --once
python tools/ur5e_direct_torque_x_transport.py --robot-ip 127.0.0.1 --probe-only --skip-dashboard-power-on
```

## Two control paths

| Path | Tool | Robot API | Motion in URSim? |
|------|------|-----------|------------------|
| Position | `ur5e_move.py` | `servoL` @ 125 Hz | Yes (firmware IK) |
| Direct torque | `ur5e_direct_torque_x_transport.py` | `directTorque()` @ 500 Hz | **No** (API only) |

Validate torque **motion** in MuJoCo or on a **physical UR5e**.
