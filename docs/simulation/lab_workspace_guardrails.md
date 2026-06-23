# Lab Workspace Guardrails

This repo now carries a diagnostic-only workspace guardrail model extracted from the external Einksul MuJoCo visualization repo:

- external checkout: `/common/users/ss5772/external/mujocoSim`
- local config: [`config/lab_workspace_guardrails.yaml`](/common/users/ss5772/real_Cartpole/config/lab_workspace_guardrails.yaml)
- shared checker / overlay module: [`simulation/workspace_guardrails.py`](/common/users/ss5772/real_Cartpole/simulation/workspace_guardrails.py)

This is **not** a real robot safety layer. It is for simulation, video overlays, and offline trajectory inspection only.

## What Was Inspected

External files and what they define:

- [`README.md`](https://github.com/Einksul/mujocoSim/blob/main/README.md#L3-L6)
  - Launch flow for the MuJoCo ROS 2 visualization stack.
  - Topic flow includes `mj_real_desired.yaml`, trajectory IK service, and trajectory playback.
- [`src/mujoco_interface/launch/mj_real_desired.yaml`](https://github.com/Einksul/mujocoSim/blob/main/src/mujoco_interface/launch/mj_real_desired.yaml#L5-L38)
  - Topic names:
    - `/ur5e/joints/real`
    - `/ur5e/joints/desired`
    - `/ur5e/trajectory/real`
    - `/ur5e/trajectory/desired`
  - Loads `scene_real_desired.xml`.
- [`src/mujoco_interface/models/scene_real_desired.xml`](https://github.com/Einksul/mujocoSim/blob/main/src/mujoco_interface/models/scene_real_desired.xml#L20-L40)
  - Workspace primitives:
    - floor plane at `pos="0 0 -1.22"`
    - slanted wall plane at `pos="0 0.55 0"` with quaternion
    - tools obstacle box
    - desk / PC-side obstacle box
- [`src/mujoco_interface/include/mujoco_interface/mj_sim.hpp`](https://github.com/Einksul/mujocoSim/blob/main/src/mujoco_interface/include/mujoco_interface/mj_sim.hpp#L99-L203)
  - Publishes `/viz/collision` as `std_msgs/Bool`.
  - Collision status is `mj_data->ncon > 0`.
  - This is diagnostic-only in the external repo and remains diagnostic-only here.
- [`src/mujoco_interface/include/mujoco_interface/mj_visualizer.hpp`](https://github.com/Einksul/mujocoSim/blob/main/src/mujoco_interface/include/mujoco_interface/mj_visualizer.hpp#L116-L198)
  - Draws trajectories as spheres + line segments in the MuJoCo visualizer.
  - Uses world-frame rendering (`mjFRAME_WORLD`).
- [`src/mujoco_interface/include/mujoco_interface/trajectory_subscriber.hpp`](https://github.com/Einksul/mujocoSim/blob/main/src/mujoco_interface/include/mujoco_interface/trajectory_subscriber.hpp#L32-L44)
  - Subscribes to `geometry_msgs/PoseArray` trajectory topics and forwards them to the visualizer.
- [`src/mujoco_interface/src/mujoco_real_desired.cpp`](https://github.com/Einksul/mujocoSim/blob/main/src/mujoco_interface/src/mujoco_real_desired.cpp#L24-L43)
  - Wires the real/desired joint topics and trajectory topics into the simulator and visualizer.
- [`src/control/src/tool_to_joint_traj.cpp`](https://github.com/Einksul/mujocoSim/blob/main/src/control/src/tool_to_joint_traj.cpp#L70-L176)
  - Publishes desired joint states and desired tool trajectories from a trajectory file.

## Coordinate Frame and Units

- Frame: `mujoco_world`
- Units: meters and radians
- The extracted boundary values are stored exactly as they appear in the MuJoCo scene XML, with the overlay/checker using the same world-frame geometry.

## What Was Extracted

The local YAML config preserves the exact scene primitives:

- floor plane
- wall plane
- tools-side box obstacle
- desk / PC-side box obstacle

Unknown / unresolved items are left as TODOs in the config:

- door-side boundary
- robot base exclusion
- cartpole rail safe range

Those unresolved items were not explicitly encoded in the external repo files inspected.

## How To Check a Trajectory

Run the offline checker on a JSON log that contains trajectory points:

```bash
python3 tools/check_trajectory_guardrails.py \
  --log outputs/control_runs/your_log.json \
  --guardrail-config config/lab_workspace_guardrails.yaml \
  --output logs/guardrail_report.json
```

Optional outputs:

- `--csv logs/guardrail_report.csv`
- `--render-overlay logs/guardrail_overlay.png`
- `--desired-log ...` if your log has a separate desired trajectory with TCP positions

The checker reports:

- inside / near_boundary / outside / unknown
- first near-boundary sample
- first violation sample
- number of violating samples
- worst signed distance when computable

## How To Render Guardrails In Video

The current MuJoCo video runners accept optional guardrail flags:

- `--draw-guardrails`
- `--guardrail-config config/lab_workspace_guardrails.yaml`
- `--guardrail-margin-m 0.02`
- `--show-boundary-labels`

The overlay is a 2D top-down inset drawn on top of the rendered frame. It shows:

- boundary footprints
- the current TCP point
- the desired TCP point when available
- the trajectory path when available

## ROS Topics

The repo also includes an optional asynchronous ROS publisher for guardrail diagnostics:

- `/ur5e/workspace_guardrails`
- `/ur5e/workspace_guardrail_status`

These are diagnostic visualization topics only. They are not a safety stop path.

## Limitations

- This model is diagnostic only.
- `/viz/collision` remains a visualization signal only.
- The guardrail overlay is not a substitute for robot-side safety limits.
- Coordinate-frame mismatches should be treated as conservative warnings / failures.
- If a trajectory log does not include TCP positions, the checker cannot infer workspace clearance without additional kinematics.

