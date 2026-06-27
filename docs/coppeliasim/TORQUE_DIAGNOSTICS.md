# CoppeliaSim Torque Diagnostics

CoppeliaSim-only tooling to diagnose why the low-level torque / impedance controller
hits safety limits. **MuJoCo is not used or modified by this path.**

## Control path (diagnosis summary)

| Layer | File | Role |
|-------|------|------|
| Entry | `simulation/launch_coppeliasim_x_axis_headless.sh` | Starts CoppeliaSim + Python ZMQ runner |
| WSL launcher | `simulation/run_torque_y_transport_wsl.sh` | WSL-specific launcher for Y-axis transport |
| Runner | `simulation/run_coppeliasim_x_axis_headless.py` | Stepped sim loop, reference generation, trace/summary |
| ZMQ adapter | `ros2_ws/.../coppeliasim_adapter.py` | `configure_force_torque_mode()`, `apply_torque()` |
| Cartesian impedance | `controller_core/x_axis_cartesian_impedance.py` | `J.T` wrench + posture + damping, headroom backtracking |
| Task torque QP | `controller_core/torque_task_qp.py` | Box QP on `tau` with torque + velocity-implied bounds |
| IK joint PD | `simulation/run_coppeliasim_x_axis_headless.py::ik_joint_pd_torque` | Joint PD for transport / diagnostic modes |
| Torque filter | `controller_core/filters.py` | Low-pass + per-joint rate limit |
| Safety monitor | `controller_core/safety.py` | Drift / velocity / NaN / joint-limit E-stop |
| Diagnostics | `simulation/coppelia_torque_diagnostics.py` | Per-step logs, JSON summary, plots |

### CoppeliaSim actuation mode

On connect the adapter calls:

1. `sim.setJointMode(handle, sim.jointmode_dynamic, 0)` — dynamic mode
2. `sim.setObjectInt32Param(handle, sim.jointintparam_motor_enabled, 1)` — motor on
3. `sim.setObjectInt32Param(handle, sim.jointintparam_ctrl_enabled, 0)` — **internal joint PID off**

Torque is sent each step via:

- **Primary:** `sim.setJointTargetForce(handle, tau, signed=True)` when available
- **Fallback:** `sim.setJointTargetVelocity(handle, ±large_v)` + `sim.setJointMaxForce(handle, |tau|)`  
  This is velocity-mode with force cap, not true torque mode. Config: `coppeliasim.torque_application.prefer_signed_target_force`.

Joint velocity `qd` is estimated by unwrapping successive `getJointPosition` samples and dividing by `getSimulationTimeStep()` (falls back to `getJointVelocity` if dt is unknown).

## How CoppeliaSim is used in this project

### Architecture overview

CoppeliaSim serves as the physics simulator for a UR5 robot arm under external
direct-torque control. The control loop runs in Python on the host (or WSL),
communicating with CoppeliaSim via the ZMQ Remote API. The simulation runs in
**stepped mode** — Python calls `sim.step()` to advance one physics tick, reads
joint states and EE pose, computes torques, and sends them back before the next
step.

```
┌─────────────────────────────────────────────────────┐
│  Python Controller (run_coppeliasim_x_axis_headless)│
│                                                     │
│  1. sim.step()           — advance physics          │
│  2. adapter.read_*()     — joint state, EE pose     │
│  3. compute gravity comp — MuJoCo qfrc_bias proxy   │
│  4. IK solver + joint PD — reference tracking       │
│  5. Cartesian Z PID      — gravity error correction  │
│  6. adapter.apply_torque — send tau to joints        │
│  7. repeat at sim_dt = 5 ms (200 Hz)                │
└───────────────────────┬─────────────────────────────┘
                        │ ZMQ Remote API
                        ▼
┌─────────────────────────────────────────────────────┐
│  CoppeliaSim Edu V4.10.0 rev0 (Ubuntu 22.04/24.04) │
│                                                     │
│  - UR5.ttm model loaded at runtime                  │
│  - Physics engine: Bullet (default)                 │
│  - All 6 joints in dynamic torque mode              │
│  - Optional vision sensor for video capture         │
└─────────────────────────────────────────────────────┘
```

### CoppeliaSim installation paths

| Component | Path |
|-----------|------|
| Runtime (WSL) | `/home/kbm/coppelia_runtime/CoppeliaSim_Edu_V4_10_0_rev0_Ubuntu22_04` |
| Runtime (HPC) | `third_party/coppelia_runtime/CoppeliaSim_Edu_V4_10_0_rev0_Ubuntu24_04` |
| UR5 model | `<runtime>/models/robots/non-mobile/UR5.ttm` |
| ZMQ Python client | `<runtime>/programming/zmqRemoteApi/clients/python/src/` |
| Python deps (WSL) | `~/.local/lib/python3.10/site-packages/` |
| Env setup script | `simulation/env_wsl_local.sh` |

### Launching CoppeliaSim

CoppeliaSim is always started from a shell script that:

1. Sources `simulation/env_wsl_local.sh` to resolve paths
2. Kills any stale CoppeliaSim on the ZMQ port
3. Starts CoppeliaSim as a background process with ZMQ RPC enabled:
   ```bash
   ./coppeliaSim.sh \
     -GzmqRemoteApi.rpcPort=23000 \
     -GzmqRemoteApi.cntPort=23001
   ```
4. Polls `ss -ltn` until the RPC port is listening
5. Waits additional seconds for the ZMQ API to initialize
6. Starts the Python controller in the foreground

### Key environment variables

| Variable | Purpose |
|----------|---------|
| `COPPELIA_ROOT` | Path to CoppeliaSim installation |
| `COPPELIA_PYDEPS` | Path to Python ZMQ client packages |
| `REAL_CARTPOLE_ENABLE_VIDEO_SMOKE` | Set to `0` for controller runs (disables smoke add-on) |
| `DISPLAY` | Required for vision sensor rendering (`:0` on WSLg) |
| `QT_QPA_PLATFORM` | Unset (defaults to `xcb` on WSLg for working rendering) |

### Python ZMQ connection pattern

```python
from coppeliasim_zmqremoteapi_client import RemoteAPIClient

client = RemoteAPIClient("127.0.0.1", port)
sim = client.require("sim")

# Set physics timestep before starting
sim.setFloatParam(sim.floatparam_simulation_time_step, 0.005)
sim.setStepping(True)
sim.startSimulation()

# Load model
ur5_handle = sim.loadModel(model_path)

# Configure joints for torque control
for joint_handle in joint_handles:
    sim.setJointMode(joint_handle, sim.jointmode_dynamic, 0)
    sim.setObjectInt32Param(joint_handle, sim.jointintparam_motor_enabled, 1)
    sim.setObjectInt32Param(joint_handle, sim.jointintparam_ctrl_enabled, 0)

# Control loop
for step in range(num_steps):
    q = [sim.getJointPosition(h) for h in joint_handles]
    ee_pos = sim.getObjectPosition(ee_handle, -1)
    # ... compute torques ...
    for h, tau in zip(joint_handles, torques):
        sim.setJointTargetForce(h, tau, True)  # signed=True
    sim.step()
```

### UR5 model object hierarchy

```
/UR5                          (model root)
├── /UR5/joint                (shoulder_pan,  joint 0)
│   └── /UR5/link
│       └── /UR5/link/joint   (shoulder_lift, joint 1)
│           └── /UR5/link/link
│               └── .../joint (elbow,         joint 2)
│                   └── ...
│                       └── .../joint (wrist_1, joint 3)
│                           └── ...
│                               └── .../joint (wrist_2, joint 4)
│                                   └── ...
│                                       └── .../joint (wrist_3, joint 5)
└── /UR5/connection           (end-effector / tool frame)
```

### CoppeliaSim UR5 link masses (extracted 2026-06-25)

| Link | Joint | Mass (kg) | Notes |
|------|-------|-----------|-------|
| 0 | shoulder_pan | 5.500 | Base link |
| 1 | shoulder_lift | 13.700 | Heaviest link |
| 2 | elbow | 8.094 | |
| 3 | wrist_1 | 5.500 | |
| 4 | wrist_2 | 6.500 | |
| 5 | wrist_3 | 1.829 | Lightest |

Total model mass: ~41.1 kg (heavier than MuJoCo UR5e at ~21.3 kg).

## Enable diagnostics on a single run

```bash
cd /common/users/ss5772/real_Cartpole

# Start CoppeliaSim (terminal 1)
bash simulation/launch_coppeliasim_x_axis_headless.sh --no-video \
  --enable-coppelia-torque-diagnostics \
  --save-controller-logs \
  --save-controller-plots \
  --torque-diagnostics-mode hold_soft \
  --impedance-gain-scale 0.05 \
  --config ros2_ws/src/ur5_x_axis_controller_ros/config/controller_coppelia_bringup.yaml
```

Or pass flags only to the Python runner if CoppeliaSim is already up:

```bash
python simulation/run_coppeliasim_x_axis_headless.py \
  --host 127.0.0.1 --port 23000 \
  --config ros2_ws/src/ur5_x_axis_controller_ros/config/controller_coppelia_bringup.yaml \
  --enable-coppelia-torque-diagnostics \
  --save-controller-logs \
  --save-controller-plots \
  --torque-diagnostics-mode hold_soft \
  --impedance-gain-scale 0.05 \
  --duration 3 --no-video
```

### Config flags (CLI or YAML `diagnostics:` section)

| Flag | Default | Purpose |
|------|---------|---------|
| `enable_coppelia_torque_diagnostics` | `false` | Master switch |
| `save_controller_logs` | `true` when diagnostics on | JSONL + summary |
| `save_controller_plots` | `true` when diagnostics on | PNG plots |
| `diagnostics_output_dir` | `outputs/control_runs/coppelia_torque_diagnostics` | Output root |
| `impedance_gain_scale` | `1.0` | Scale Cartesian/IK gains |
| `reference_smoothing_enabled` | `false` | Limit `q_des` / `x_des` steps |
| `max_reference_step` | `0.02` | Max reference jump per step |
| `max_reference_velocity` | `0.05` | Max reference velocity |

## WSL Y-axis transport command

```bash
cd /mnt/c/Users/sandr/Downloads/real-cartpole-control-using-ur5-direct-torque

# Full 20-second Y-axis transport with video
bash simulation/run_torque_y_transport_wsl.sh

# Quick 10-second test without video
DURATION=10 NO_VIDEO=1 bash simulation/run_torque_y_transport_wsl.sh

# Tune gravity and Z feedback gains
GRAVITY_SCALE=1.0 CART_Z_KP=200 CART_Z_KD=40 CART_Z_KI=50 \
  bash simulation/run_torque_y_transport_wsl.sh
```

### Tunable parameters (env vars)

| Variable | Default | Purpose |
|----------|---------|---------|
| `GRAVITY_SCALE` | `1.0` | Scale MuJoCo qfrc_bias for Coppelia gravity feedforward |
| `GRAVITY_COMP_SOURCE` | `mujoco` | Source for gravity model |
| `IK_JOINT_KP` | `140` | IK joint PD proportional gain (transport phase) |
| `IK_JOINT_KD` | `25` | IK joint PD derivative gain (transport phase) |
| `CART_Z_KP` | `200` | Cartesian Z stiffness via J^T (N/m) |
| `CART_Z_KD` | `40` | Cartesian Z damping via J^T (N/(m/s)) |
| `CART_Z_KI` | `50` | Cartesian Z integral gain (N/(m·s)) |
| `TARGET_DY` | `0.04` | Y-axis displacement target (m) |
| `A_AXIS_MAX` | `0.008` | Max Y-axis acceleration (m/s²) |
| `V_AXIS_MAX` | `0.008` | Max Y-axis velocity (m/s) |
| `DURATION` | `20` | Total run time (s) |
| `MOTION_HOLD_WARMUP` | `0.5` | Warmup hold time before transport (s) |
| `NO_VIDEO` | `0` | Set to `1` for faster headless runs |

## Smoke test ladder

### On this cluster (Singularity container — recommended)

The host libc is too old for CoppeliaSim directly. Use the container wrapper, which
sets `XVFB_RUN_BIN`, `LD_LIBRARY_PATH`, and starts/stops CoppeliaSim per test:

```bash
cd /common/users/ss5772/real_Cartpole

# Full ladder (each test launches CoppeliaSim via launch_coppeliasim_x_axis_headless.sh)
bash simulation/run_coppelia_torque_diagnostics_container.sh

# Subset only
PORT=23270 DURATION=2 bash simulation/run_coppelia_torque_diagnostics_container.sh \
  passive hold_soft sinusoid_joint tiny_x_motion ref_discontinuity
```

Outputs: `outputs/control_runs/coppelia_torque_diagnostics/smoke_ladder_summary.json`

### Two-terminal workflow (CoppeliaSim already running)

```bash
# Terminal 1: CoppeliaSim
bash simulation/launch_coppeliasim_x_axis_headless.sh --probe-only

# Terminal 2: runner-only ladder (CoppeliaSim must stay up)
python simulation/run_coppelia_torque_diagnostics_smoke.py \
  --host 127.0.0.1 --port 23000 \
  --config ros2_ws/src/ur5_x_axis_controller_ros/config/controller_coppelia_bringup.yaml
```

### Self-contained per test (no pre-started CoppeliaSim)

```bash
python simulation/run_coppelia_torque_diagnostics_smoke.py --use-launcher \
  --config /common/users/ss5772/real_Cartpole/ros2_ws/src/ur5_x_axis_controller_ros/config/controller_coppelia_bringup.yaml \
  --tests hold_soft
```

### Individual tests

| Test | Command snippet |
|------|-----------------|
| 1. Passive | `--torque-diagnostics-mode passive` |
| 2. Zero torque | same as passive (zero command, log motion) |
| 3. Soft hold | `--torque-diagnostics-mode hold_soft --impedance-gain-scale 0.05` |
| 4. Gain sweep | `python simulation/run_coppelia_torque_diagnostics_smoke.py --tests gain_sweep` |
| 5. Sinusoid joint | `--torque-diagnostics-mode sinusoid_joint --torque-diagnostics-joint-index 5` |
| 6. Tiny X motion | `--torque-diagnostics-mode tiny_x_motion` |
| 7. Ref discontinuity | `--tests ref_discontinuity` (runs `ref_step` vs `ref_smooth`) |

Dry-run command list:

```bash
python simulation/run_coppelia_torque_diagnostics_smoke.py --dry-run
```

## Outputs

| Artifact | Location |
|----------|----------|
| Per-step JSONL | `{diagnostics_output_dir}/{run_label}.jsonl` |
| Run summary | `{diagnostics_output_dir}/{run_label}_summary.json` |
| Plots (13 PNGs) | `{diagnostics_output_dir}/{run_label}_01_*.png` … `_13_*.png` |
| Smoke ladder rollup | `{diagnostics_output_dir}/smoke_ladder_summary.json` |
| Runner trace (unchanged) | `outputs/control_runs/{trace-name}` |

## Pass / fail criteria

Immediate pass (hold / sinusoid / tiny-X diagnostic modes):

- No torque clipping
- Torque-rate clipping rare or zero
- `max_tau_fraction < 0.50` of configured limit
- Stable `dt`, no NaN/Inf
- No joint-limit or workspace guardrail trips

Passive / zero-torque: no NaN/Inf (motion from gravity alone is OK).

Gain sweep: documents torque usage vs gain scale; fails if `max_tau_fraction >= 0.50`.

## Likely causes of limit hits (check plots/logs)

1. **High Kp/Kd** — P/D panels (`09_tau_P_vs_D`) and gain sweep
2. **Discontinuous `q_des` / `x_des`** — compare `ref_step` vs `ref_smooth` runs
3. **Noisy `qd` (finite difference)** — spikes in `04_qd_error` and D-term torque
4. **`dt` mismatch / skipped steps** — plot `13_dt_and_frequency`
5. **Cartesian backtracking at limit** — `tau_raw` vs `tau_after_saturation` in JSONL
6. **Rate limiter** — `08_torque_rate_usage_fraction`
7. **Wrong actuation mode** — check `coppelia_api_per_joint` in JSONL (should be `setJointTargetForce`)

## Gravity compensation calibration

### Phase 1: Single-joint probe (2026-06-24)

Initial probe (`tools/probe_gravity_sign.py`) applied single-joint torques to
shoulder_lift and measured EE Z drift after 100 steps at dt=0.005s.

| Shoulder-lift torque | EE dz after 100 steps | Interpretation |
|----------------------|-----------------------|----------------|
| 0 Nm | -0.17 m | freefall |
| +20 Nm | -0.39 m | undershoot, still falls |
| +40 Nm | +0.035 m | close, slight overshoot |
| +56 Nm (1.5x MuJoCo) | -0.0015 m | near-perfect balance |
| +75 Nm (2.0x MuJoCo) | -0.006 m | slightly over, acceptable |

**Initial conclusion (incorrect):** Scale MuJoCo bias by 1.5x.

### Phase 2: Multi-joint probe (2026-06-25)

Full 6-joint gravity probe (`tools/probe_coppelia_gravity_native.py`) revealed
the single-joint test was misleading — applying only shoulder_lift torque causes
other joints to compensate, masking the true gravity balance.

**Multi-joint MuJoCo bias sweep** (all 6 joints, 50 steps at dt=0.005s):

| MuJoCo scale | EE dz (m) | Shoulder-lift tau (Nm) | Assessment |
|--------------|-----------|------------------------|------------|
| 1.0 | +0.007 | +37.46 | **Near-perfect** |
| 1.5 | -0.754 | +56.18 | **Catastrophic collapse** |
| 1.8 | -0.072 | +67.42 | Large drift |
| 2.0 | +0.015 | +74.91 | Good, slight overshoot |
| 2.5 | +0.036 | +93.64 | Moderate overshoot |

**Key finding:** MuJoCo `qfrc_bias` at scale=1.0 provides excellent
multi-joint gravity balance for the CoppeliaSim UR5. Scales 1.5 and 1.8 cause
catastrophic Z collapse because partial sign cancellation across joints
amplifies errors on some joints while over-compensating others.

**Why 1.0 works despite different model masses:** The MuJoCo UR5e has
different link masses than CoppeliaSim's UR5 (~21 kg vs ~41 kg), but
`qfrc_bias` is computed at the given joint configuration and includes full
Coriolis/centrifugal terms. At this particular seed pose, the bias values happen
to align well with the CoppeliaSim gravity loads.

### CoppeliaSim UR5 link masses

Extracted via `sim.getShapeMass()` and `sim.getShapeInertia()`:

| Link | Joint | Mass (kg) | COM world (seed) |
|------|-------|-----------|-------------------|
| 0 | shoulder_pan | 5.500 | (-0.018, 0.000, +0.097) |
| 1 | shoulder_lift | 13.700 | (-0.135, +0.008, +0.318) |
| 2 | elbow | 8.094 | (-0.013, -0.205, +0.502) |
| 3 | wrist_1 | 5.500 | (-0.097, -0.373, +0.502) |
| 4 | wrist_2 | 6.500 | (-0.109, -0.403, +0.413) |
| 5 | wrist_3 | 1.829 | (-0.174, -0.406, +0.403) |

### Native RNEA gravity (attempted and rejected)

A recursive Newton-Euler gravity computation using the extracted link masses and
COM positions was attempted but produces incorrect results. The native RNEA
reports shoulder_lift gravity as -68.1 Nm and elbow as -72.7 Nm, which do not
match empirical balance tests. The CoppeliaSim UR5.ttm model likely has
compound shapes or non-trivial inertia frames that the simple mass-weighted COM
approach does not capture correctly. The MuJoCo bias proxy at scale=1.0 remains
the best available gravity model.

### Other findings

1. `sim.getJointForce()` returns 0.0 for all joints in this motor+dynamic config.
   Do **not** rely on Coppelia-measured force for gravity learning.
2. MuJoCo `qfrc_bias` must be **negated** (sign=-1) for CoppeliaSim: MuJoCo
   reports -37.5 Nm for shoulder_lift, Coppelia needs +37.5 Nm (positive).
3. The CoppeliaSim default `sim_dt = 0.05s` (20 Hz) is too slow for torque
   control. The runner now requests `sim_dt = 0.005s` (200 Hz) before start.
4. The original code had a **double gravity bug**: the QP controller adds gravity
   internally, and the runner added it again externally. This is now fixed.

## Transport phase: Z-axis stability

### Problem: Z oscillation during Y-axis transport

When the arm sweeps along world-Y under IK joint PD, the joint configuration
changes, causing the MuJoCo gravity bias to vary:

| Joint | Bias range during sweep (Nm) | Impact |
|-------|------------------------------|--------|
| shoulder_lift | -37.5 to -34.3 | 3.2 Nm variation |
| elbow | -5.8 to +1.6 | **7.4 Nm variation** |

This dynamic mismatch drives Z-axis drift during transport.

### Solution: Two-part fix

1. **Higher warmup hold gains** (kp=300, kd=40):
   Reduces joint steady-state offset from gravity model error. At kp=100, a 15 Nm
   gravity mismatch causes 0.15 rad joint error. At kp=300, this drops to 0.05 rad.

2. **Cartesian Z PID via J^T** (`--cartesian-z-kp`, `--cartesian-z-kd`, `--cartesian-z-ki`):
   Direct Cartesian Z feedback that corrects gravity model error regardless of
   joint configuration. Applied during both warmup hold and IK PD transport:
   ```
   f_z = kp_z * (z_target - z_current) - kd_z * z_vel + ki_z * ∫(z_error)dt
   tau_z = J_z^T * f_z
   ```

### Progress log

| Date | Gravity scale | Warmup kp | Cart Z PID | Z drift (m) | Y disp (m) | Orient (°) | Warmup |
|------|---------------|-----------|------------|-------------|------------|------------|--------|
| 06-24 | 1.5 | 100 | off | 0.104 | 0.224 | 34.2 | hold_timeout |
| 06-25a | 1.0 | 100 | off | 0.086 | 0.234 | 15.0 | hold_timeout |
| 06-25b | 1.0 | 300 | off | 0.034 | 0.109 | 5.9 | **z_stable** |
| 06-25c | 1.0 | 300 | kp=200,kd=40,ki=50 | TBD | TBD | TBD | TBD |

## MuJoCo

This diagnostics lane does **not** modify MuJoCo files, add MuJoCo dependencies, or run MuJoCo comparisons. The MuJoCo UR5e model is used **only** as a gravity estimation proxy via `qfrc_bias` at the current joint configuration. It is loaded read-only by `build_mujoco_gravity_estimator()` and never modified.

## Diagnostic tools

| Tool | Purpose |
|------|---------|
| `tools/probe_gravity_sign.py` | Single-joint gravity sign/scale probe |
| `tools/probe_coppelia_gravity_native.py` | Extract UR5 link masses + multi-joint gravity sweep |
| `tools/run_coppelia_gravity_probe.sh` | Launcher for native gravity probe |
| `tools/analyze_z_oscillation.py` | Post-hoc trace analysis of Z drift during transport |
| `tools/diagnose_trace.py` | Deep trace diagnostic for warmup hold failure |
| `tools/read_summary.py` | Quick summary JSON reader |
| `tools/check_mujoco_gravity2.py` | Direct MuJoCo qfrc_bias query |

## RL Y-axis transport (PPO)

Model-based Z hold (gravity scale 1.0 + Cartesian Z PID) reduced drift but did not achieve reliable full Y sweeps. Active work moved to learned control:

- Full guide: [`docs/coppeliasim/RL_Y_TRANSPORT.md`](RL_Y_TRANSPORT.md)
- Train: `bash simulation/launch_rl_training_wsl.sh`
- Eval: `bash simulation/run_rl_eval_wsl.sh`
