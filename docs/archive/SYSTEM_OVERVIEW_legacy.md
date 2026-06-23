# Real_Cartpole System Overview

This document summarizes how the UR5 / UR5e stack in `real_Cartpole` is organized, what is working today, and what the intended runtime behavior is supposed to be.

The short version:
- The repo now has a verified, controller-free CoppeliaSim video smoke path.
- The controller/RPC path exists, but the bootstrap behavior is still being stabilized.
- The important runtime assets now live inside the repo under `third_party/coppelia_runtime/` and `third_party/coppelia_pydeps/`.

## Big Picture

There are two main execution paths:

1. Render-only smoke path
   - Starts CoppeliaSim headless under `xvfb-run`.
   - Loads the UR5 model and a camera from a Lua add-on.
   - Captures a short upright PNG sequence.
   - Encodes an MP4 with `ffmpeg`.
   - Does not depend on the controller, ROS 2, or RPC.

2. Controller / RPC path
   - Starts CoppeliaSim headless on a GPU node.
   - Launches the Python controller runner.
   - Uses the CoppeliaSim ZMQ Remote API to read robot state and apply torques.
   - Can also render frames and write traces/videos.
   - This path is the one still under active bootstrap cleanup.

## What Is Working Today

The verified path is the render-only smoke test:

- Launcher: `simulation/launch_coppeliasim_video_smoke.sh`
- Add-on: `simulation/ur5_video_smoke_addon.lua`
- Runtime: `third_party/coppelia_runtime/CoppeliaSim_Edu_V4_10_0_rev0_Ubuntu24_04`
- Output video: `demonstration_videos/ur5e_coppeliasim/coppeliasim_ur5_video_smoke.mp4`

That path is intentionally controller-free. It is the best proof that:
- the local CoppeliaSim runtime is valid,
- the UR5 model loads,
- the camera pose works,
- the display wrapper works,
- and `ffmpeg` can turn the captured PNGs into a usable MP4.

## How The System Is Supposed To Work

### Intended render-only workflow

The smoke launcher should:

1. Confirm the repo-local CoppeliaSim runtime exists.
2. Start CoppeliaSim under `xvfb-run -a`.
3. Auto-load `simulation/ur5_video_smoke_addon.lua` from `addOns/`.
4. Let the add-on load the UR5 model and create the camera.
5. Capture 40 frames.
6. Encode the final MP4 with `ffmpeg`.

This path should stay independent from RPC and from the controller.

### Intended controller / RPC workflow

The controller path is meant to work like this:

1. Start CoppeliaSim headless on a GPU-capable node.
2. Open the ZMQ Remote API port.
3. Let the Python runner connect with `RemoteAPIClient`.
4. Load the scene or model.
5. Attach the adapter to the UR5 in CoppeliaSim.
6. Start stepping and run the impedance controller loop.
7. Record trace data and, optionally, video frames.

The intended end state is a clean attach model:
- CoppeliaSim comes up independently.
- The controller attaches when ready.
- The controller does not need to “hold” CoppeliaSim alive with a file gate.

## Current RPC Bootstrap State

The current RPC bring-up uses a temporary handshake:

- Launcher: `simulation/launch_coppeliasim_x_axis_headless.sh`
- Runner: `simulation/run_coppeliasim_x_axis_headless.py`
- Add-on: `simulation/ur5_video_smoke_addon.lua`
- Marker file: `outputs/control_runs/coppelia_video_smoke_state/rpc_bootstrap_ready.txt`

Current behavior:

- The launcher starts CoppeliaSim with `-h` and the ZMQ RPC port enabled.
- The Python runner sleeps briefly, writes the ready marker, then retries `RemoteAPIClient`.
- The Lua add-on watches for that marker and only then tries to start the simulation.

This is the present implementation, but it is not yet the final clean design.
The stable target is still a simulator-first attach flow, not a file-gated bootstrap.

## Component Map

### `controller_core/`

Portable controller logic that is not tied to ROS or CoppeliaSim.

What it does:
- Implements the x-axis Cartesian impedance controller.
- Applies safety checks and torque filtering.
- Keeps the control math reusable across simulator and real hardware work.

Key file:
- `controller_core/x_axis_cartesian_impedance.py`

### `ros2_ws/src/ur5_x_axis_controller_ros/`

ROS 2 wrapper around the portable controller core.

What it does:
- Provides the ROS 2 controller node.
- Provides the CoppeliaSim adapter and bridge nodes.
- Loads controller configuration from YAML.

Key files:
- `ros2_ws/src/ur5_x_axis_controller_ros/ur5_x_axis_controller_ros/controller_node.py`
- `ros2_ws/src/ur5_x_axis_controller_ros/ur5_x_axis_controller_ros/coppeliasim_bridge_node.py`
- `ros2_ws/src/ur5_x_axis_controller_ros/ur5_x_axis_controller_ros/coppeliasim_adapter.py`

### `simulation/`

Launchers and bootstrap code for CoppeliaSim.

What it does:
- Starts the simulator in headless mode.
- Manages the smoke test and the controller/RPC test.
- Holds the Lua add-on that loads the UR5 model and camera.

Key files:
- `simulation/launch_coppeliasim_video_smoke.sh`
- `simulation/run_coppeliasim_video_smoke.py`
- `simulation/ur5_video_smoke_addon.lua`
- `simulation/launch_coppeliasim_x_axis_headless.sh`
- `simulation/run_coppeliasim_x_axis_headless.py`
- `simulation/ur5_video_smoke_addon_shim.lua`

### `third_party/coppelia_runtime/`

Repo-local CoppeliaSim runtime.

What it does:
- Stores the extracted CoppeliaSim build used by the repo.
- Removes the old dependency on the temp-folder mirror.

Important path:
- `third_party/coppelia_runtime/CoppeliaSim_Edu_V4_10_0_rev0_Ubuntu24_04`

### `third_party/coppelia_pydeps/`

Shared Python dependency anchor for CoppeliaSim.

What it does:
- Supplies the Python client dependencies needed by the ZMQ Remote API client.
- Keeps the simulator Python path self-contained inside the repo.

### `mujoco_menagerie/universal_robots_ur5e/`

Model assets and related MuJoCo menagerie content.

What it does:
- Holds robot model resources used by the broader workspace.
- Not the active CoppeliaSim runtime itself, but still part of the UR5 ecosystem.

### `demonstration_videos/ur5e_coppeliasim/`

Video outputs.

What it does:
- Stores the generated MP4s from CoppeliaSim runs.
- This is the main place to look for visual validation artifacts.

### `outputs/control_runs/`

Run artifacts and debug state.

What it does:
- Stores JSONL traces, logs, frame dumps, and bootstrap markers.
- Keeps runtime state out of `/tmp`, which has been a failure point on this node.

## Current Data Flow

### Render-only smoke

`launch_coppeliasim_video_smoke.sh`
-> starts CoppeliaSim under `xvfb-run`
-> auto-loads `ur5_video_smoke_addon.lua`
-> loads UR5 and camera
-> captures 40 PNGs
-> writes MP4 with `ffmpeg`

### Controller / RPC

`launch_coppeliasim_x_axis_headless.sh`
-> starts CoppeliaSim under `xvfb-run`
-> copies the add-on into the runtime `addOns/` directory
-> starts `run_coppeliasim_x_axis_headless.py`
-> runner waits on the RPC bootstrap marker
-> runner connects through ZMQ
-> controller reads state and applies torques
-> runner can also capture video and trace rows

## Known Caveats

- `-h` is the right CoppeliaSim mode for this headless vision workflow, but it is not a substitute for a working OpenGL / Xvfb path.
- `xvfb-run -a` is the wrapper that has been reliable in this environment.
- GPU access matters for reliable simulator rendering on these nodes.
- The current RPC bootstrap is still the part most likely to change.
- The repo root is not a git repository.
- `mujoco_menagerie/` contains a nested git repo.

## Practical Rule Of Thumb

If you are debugging:

1. Prove the render-only smoke path first.
2. Then attach the controller/RPC path.
3. Keep the controller out of first-pass rendering bugs.
4. Keep the runtime paths inside `third_party/coppelia_runtime/` and `third_party/coppelia_pydeps/`.

If you only need to verify that the simulator is alive, use the render-only smoke path.
If you need torques, state reads, or controller integration, use the RPC path.

