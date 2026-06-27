# real_Cartpole Documentation

This workspace is currently focused on one active problem:

**make the UR5 arm move in CoppeliaSim using the existing single-axis torque controller.**

Workspace rule: CoppeliaSim is the only active runtime for current development. MuJoCo code, videos, and notes are historical reference only unless a task explicitly says otherwise.

**New (2026-06-25):** RL-based Y-axis transport training via PPO — see [RL Y-Transport Controller](coppeliasim/RL_Y_TRANSPORT.md).

The historical folder name is misleading. Treat this as a UR5 / UR5e control workspace with ROS 2 scaffolding and an active CoppeliaSim torque-control bring-up; MuJoCo work is archived.

## Start Here

Read these in order:

1. [Current Status](CURRENT_STATUS.md)
2. [CoppeliaSim: two pipelines (smoke vs ZMQ)](coppeliasim/PIPELINES.md)
3. [UR5 controller, from first principles](coppeliasim/CONTROLLER_FIRST_PRINCIPLES.md)
4. [MuJoCo acceleration controller guardrails](coppeliasim/MUJOCO_ACCELERATION_GUARDRAILS.md) (historical reference only)
5. [Lab workspace guardrails extracted from MuJoCo](simulation/lab_workspace_guardrails.md) (diagnostic only)
6. [CoppeliaSim vision API notes (handle → get, buffers)](coppeliasim/COPPELIASIM_VISION_NOTES.md)
7. [External ZMQ controller bring-up](coppeliasim/EXTERNAL_ZMQ_CONTROLLER_BRINGUP.md)
8. [CoppeliaSim RPC Controller System](coppeliasim/RPC_CONTROLLER_SYSTEM.md)
9. [CoppeliaSim RPC Controller TODO](coppeliasim/RPC_CONTROLLER_TODO.md)
10. [RL Y-Transport Controller (PPO)](coppeliasim/RL_Y_TRANSPORT.md)
11. [Torque diagnostics & CoppeliaSim usage (WSL)](coppeliasim/TORQUE_DIAGNOSTICS.md)
12. [Project Plan](PROJECT_PLAN.md)
13. [Workspace Map](WORKSPACE_MAP.md)

## Active Code Anchors

| Area | Path | Purpose |
| --- | --- | --- |
| Portable controller | `controller_core/` | Simulator-independent torque controller, filtering, safety, tracing. |
| Active CoppeliaSim scripts | `simulation/` | Headless launchers, RPC runner, smoke runner, Lua add-on. |
| ROS 2 bridge scaffold | `ros2_ws/src/ur5_x_axis_controller_ros/` | ROS wrapper, CoppeliaSim adapter, config. |
| CoppeliaSim runtime | `third_party/coppelia_runtime/` | Repo-local CoppeliaSim install. |
| Coppelia Python deps | `third_party/coppelia_pydeps/` | Shared Python dependency anchor for ZMQ RPC. |
| RL training (PPO) | `rl/` | Gymnasium env, PPO trainer, eval script, `config.yaml`. |
| Current outputs | `outputs/control_runs/` | Traces, logs, frame dumps, probe summaries. |
| Current videos | `demonstration_videos/ur5e_coppeliasim/` | CoppeliaSim validation videos. |
| Archived MuJoCo reference | `mujoco_menagerie/`, `simulation/run_x_acceleration_transport.py`, `controller_core/transport_lqr.py` | Historical reference only; do not use for active controller development. |

## Known-Good Command

The render-only CoppeliaSim smoke test is the known-good baseline:

```bash
cd /common/users/ss5772/real_Cartpole
bash simulation/launch_coppeliasim_video_smoke.sh
```

It verifies launch, model load, rendering, upright frame capture, and MP4 encoding. It does not verify torque-controller motion.

## Known-Good Visible Motion Command

The controller-motion visible-video path moves the UR5 with a small simulator-side joint-space controller and captures via the proven Lua/PNG route:

```bash
cd /common/users/ss5772/real_Cartpole
bash simulation/launch_coppeliasim_controller_video.sh
```

It writes `demonstration_videos/ur5e_coppeliasim/coppeliasim_ur5_controller_video.mp4`. This proves visible motion capture, but not yet stable external ZMQ torque control.

## Known-Good Origin Acquisition Command

The origin-to-acceleration path is the required first stage before broader
acceleration transport. It starts from an offset joint pose, recovers the
grounded reference origin pose at `EE_TARGET_Z_M=0.65`, holds for 1 second,
keeps the camera fixed, scans the reachable Coppelia world-X interval, clamps
the requested displacement to that interval, and runs a one-direction
acceleration profile:

```bash
cd /common/users/ss5772/real_Cartpole
bash simulation/launch_coppeliasim_origin_acquisition_video.sh
```

It writes:

```text
demonstration_videos/ur5e_coppeliasim/coppeliasim_ur5_origin_acquisition.mp4
outputs/control_runs/coppelia_origin_acquisition_state/coppeliasim_ur5_origin_acquisition_summary.json
```

Latest verified metrics: `success=true`, `base_on_ground=true`,
`requested_target_dx_m=0.060`, `target_dx_m=0.020`,
`target_dx_was_clamped_by_range_scan=true`,
`acceleration_final_orientation_error_deg=2.784`.

## Diagnostic Fixed-Z Acceleration Transport Command

The acceleration-transport path moves the UR5 across Coppelia world X from the
MuJoCo fixed-Z transport seed while holding Y, Z, and end-effector orientation
nearly fixed. It writes a MuJoCo-style metrics JSON with pass/fail guardrails:

```bash
cd /common/users/ss5772/real_Cartpole
bash simulation/launch_coppeliasim_acceleration_transport_video.sh
```

It writes:

```text
demonstration_videos/ur5e_coppeliasim/coppeliasim_ur5_acceleration_transport.mp4
outputs/control_runs/coppelia_acceleration_transport_state/coppeliasim_ur5_acceleration_transport_summary.json
```

This command is now diagnostic, not a completed port. Treat the run as accepted
only when the summary JSON reports `success=true` and `base_on_ground=true`.

## Active In-Progress Command

The current controller/RPC path is:

```bash
cd /common/users/ss5772/real_Cartpole
bash simulation/launch_coppeliasim_x_axis_headless.sh
```

This path now defaults to Python-owned stepping and zero-torque warmup; legacy marker handoff remains available only via `--legacy-marker-handoff`.

## External Capture Wrapper

The live ZMQ lane now also has a dedicated video-capable wrapper:

```bash
cd /common/users/ss5772/real_Cartpole
bash simulation/launch_coppeliasim_x_axis_offscreen_capture.sh
```

It uses the time-based capture add-on in `simulation/ur5_external_controller_capture_addon.lua`. The launcher resolves `FFMPEG_BIN` and `XVFB_RUN_BIN` explicitly and is intended to run inside the repo's Ubuntu 24.04 Singularity image on this cluster.

Current status: use `simulation/launch_coppeliasim_x_axis_offscreen_capture_container.sh` as the top-level entrypoint inside Singularity. It applies the verified narrow XKB/Xvfb bind set and then calls the plain capture launcher.

To run the actual X transport motion through the capture lane, set `RUNNER_EXTRA_ARGS` on the plain launcher or the container wrapper, for example:

```bash
RUNNER_EXTRA_ARGS='--accel-x-transport --accel-torque-policy ik_joint_pd --target-dx 0.005 --duration 3 --settle-duration 1' \
  bash simulation/launch_coppeliasim_x_axis_offscreen_capture_container.sh
```

## Documentation Organization

Project-owned Markdown now lives under `docs/`, except for root [AGENTS.md](../AGENTS.md), which remains at the workspace root because agents load it as the working playbook.

Historical MuJoCo, MoveIt, SLSQP, and older agent notes are in [archive](archive/). They are useful context, but they are not the current active path.

The nested `mujoco_menagerie/` repository keeps its own upstream documentation in place.
