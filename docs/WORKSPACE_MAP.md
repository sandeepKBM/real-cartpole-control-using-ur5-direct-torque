# Workspace Map

This is a quick orientation map for `/common/users/ss5772/real_Cartpole`.

## Important Reality Check

- The root folder is not a git repository.
- `mujoco_menagerie/` is a nested git repository.
- The folder name is historical. Current work is UR5 / UR5e control, especially CoppeliaSim torque control.
- Root `AGENTS.md` is the agent playbook and intentionally remains outside `docs/`.

## Active Source Areas

### `controller_core/`

Portable controller code.

Important files:

- `x_axis_cartesian_impedance.py`
- `filters.py`
- `safety.py`
- `logging_utils.py`
- `tests/test_impedance.py`

### `simulation/`

Simulator launchers and direct runner scripts.

Important files:

- `launch_coppeliasim_video_smoke.sh`
- `launch_coppeliasim_x_axis_headless.sh`
- `launch_coppeliasim_x_axis_offscreen_capture.sh`
- `run_coppeliasim_video_smoke.py`
- `run_coppeliasim_x_axis_headless.py`
- `ur5_video_smoke_addon.lua`
- `ur5_external_controller_capture_addon.lua`
- `plot_coppeliasim_trace.py`

### `ros2_ws/src/ur5_x_axis_controller_ros/`

ROS 2 package and CoppeliaSim adapter.

Important files:

- `config/controller.yaml`
- `ur5_x_axis_controller_ros/coppeliasim_adapter.py`
- `ur5_x_axis_controller_ros/controller_node.py`
- `ur5_x_axis_controller_ros/coppeliasim_bridge_node.py`

### `third_party/coppelia_runtime/`

Repo-local CoppeliaSim runtime.

Expected anchor:

```text
third_party/coppelia_runtime/CoppeliaSim_Edu_V4_10_0_rev0_Ubuntu24_04
```

Expected official model:

```text
models/robots/non-mobile/UR5.ttm
```

### `third_party/coppelia_pydeps/`

Repo-local Python dependency anchor for CoppeliaSim ZMQ RPC.

### `outputs/control_runs/`

Runtime artifacts:

- logs,
- JSONL traces,
- summary JSON files,
- frame dumps,
- bootstrap marker/state files.

### `demonstration_videos/`

Rendered validation videos.

Current CoppeliaSim output area:

```text
demonstration_videos/ur5e_coppeliasim/
```

## Documentation Areas

### `docs/`

Current documentation landing page and active status docs.

### `docs/coppeliasim/`

Active CoppeliaSim controller/RPC docs.

### `docs/archive/`

Historical notes from earlier MuJoCo, MoveIt, SLSQP, and workspace studies.

### `docs/ros2/`

Moved ROS 2 package documentation.

### `docs/controller_core/`

Moved portable controller documentation.

### `docs/simulation/`

Simulation-script documentation.
