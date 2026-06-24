# CoppeliaSim Torque Diagnostics

CoppeliaSim-only tooling to diagnose why the low-level torque / impedance controller
hits safety limits. **MuJoCo is not used or modified by this path.**

## Control path (diagnosis summary)

| Layer | File | Role |
|-------|------|------|
| Entry | `simulation/launch_coppeliasim_x_axis_headless.sh` | Starts CoppeliaSim + Python ZMQ runner |
| Runner | `simulation/run_coppeliasim_x_axis_headless.py` | Stepped sim loop, reference generation, trace/summary |
| ZMQ adapter | `ros2_ws/.../coppeliasim_adapter.py` | `configure_force_torque_mode()`, `apply_torque()` |
| Cartesian impedance | `controller_core/x_axis_cartesian_impedance.py` | `J.T` wrench + posture + damping, headroom backtracking |
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

## MuJoCo

This diagnostics lane does **not** modify MuJoCo files, add MuJoCo dependencies, or run MuJoCo comparisons. Optional gravity compensation in the runner still uses MuJoCo only when `use_gravity_compensation: true` in YAML (disabled in `controller_coppelia_bringup.yaml`).
