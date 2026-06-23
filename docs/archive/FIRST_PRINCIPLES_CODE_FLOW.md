# First-Principles Code Flow

This document explains the active `real_Cartpole` UR5e simulation stack from the bottom up.

It is the current read path for understanding:

- what the MuJoCo model contains
- how the origin target is defined
- how the controller uses that target
- how a full run turns into a video and JSON artifact

This is not a generic UR5e tutorial. It is a guide to this codebase as it exists now.

## What The System Is

At a high level, the active workflow is:

1. Define a robot model in MuJoCo XML.
2. Pick a reference point on the robot that will count as the "origin" target.
3. Pick a tool-frame orientation target in world coordinates.
4. Search for a joint configuration that satisfies that orientation while pushing the origin point as high as possible.
5. Start from a disturbed random pose near that target.
6. Run an outer-loop Python controller that keeps correcting:
   - the forearm-origin position
   - the tool orientation
7. Let MuJoCo's built-in joint servos execute those setpoints.
8. Save a rendered video and a JSON summary.

The active files are:

- [`simulation/controller.py`](./simulation/controller.py)
- [`simulation/probe_origin_pose.py`](./simulation/probe_origin_pose.py)
- [`simulation/run_origin_stabilization.py`](./simulation/run_origin_stabilization.py)
- [`simulation/render_pose_orbit.py`](./simulation/render_pose_orbit.py)
- [`mujoco_menagerie/universal_robots_ur5e/ur5e.xml`](./mujoco_menagerie/universal_robots_ur5e/ur5e.xml)

## First Principles

### MuJoCo Concepts

The XML model is made of a few core object types:

- `body`: a rigid link with its own local coordinate frame
- `joint`: a degree of freedom that connects a body to its parent
- `geom`: visible geometry or collision geometry
- `site`: a massless marker frame used for measurement, targeting, or attachments
- `actuator`: the control input that drives a joint

This project relies heavily on `site`s.

Two sites matter most:

- `forearm_tip_site`: the point used as the origin reference
- `attachment_site`: the tool frame used for end-effector orientation control

### Why Sites Matter

The code intentionally separates:

- "Where should the arm be?" from
- "How should the tool be oriented?"

That split is the core architectural decision in the current controller.

The origin is not defined from the end-effector face anymore.
It is defined from a stable point near the distal end of the forearm.

The end-effector orientation is controlled separately through the rotated tool frame at `attachment_site`.

## Model Layer

The relevant chain in [`ur5e.xml`](./mujoco_menagerie/universal_robots_ur5e/ur5e.xml) is:

`upper_arm_link -> forearm_link -> wrist_1_link -> wrist_2_link -> wrist_3_link -> attachment_site`

Important pieces:

- `forearm_link` contains `forearm_tip_site`
- `wrist_3_link` contains `attachment_site`
- `attachment_site` has a quaternion, so it is not aligned with the raw `wrist_3_link` frame

That means:

- if you control `wrist_3_link` directly, the visible tool may not behave the way you expect
- if you control `attachment_site`, you are controlling a rotated tool frame that better matches the visible end-effector geometry

This distinction was the source of several earlier orientation confusions.

## Actuation Model

The MuJoCo actuators are not being given direct torque commands from Python.

The Python code writes `ctrl` setpoints.
MuJoCo then applies its own joint servo behavior to pull the joints toward those setpoints.

So the Python controller is an outer loop.
It says:

- "move the desired joint positions this way"

MuJoCo's actuators then do the low-level tracking.

That is why `controller.py` looks like a setpoint shaper, not a torque controller.

## Current Active Target

The current active target joint vector is defined in [`simulation/controller.py`](./simulation/controller.py) as `VERTICAL_FACE_ORIGIN_Q`.

As of the current code, it is:

```text
[0.0, -1.984713, 1.410664, 0.574171, 1.571001, -3.141593]
```

Its associated origin reference point is approximately:

```text
forearm_tip_site = [-0.158232, -0.007000, 0.764980] m
```

That target was produced by searching for the highest `forearm_tip_site` world `z` while holding the desired `attachment_site` orientation.

## Controller Layer

[`simulation/controller.py`](./simulation/controller.py) defines three separate kinds of things:

- target constants
- helper functions
- control policies

### Constants

The important constants are:

- `ACTIVE_ORIGIN_Q`
- `TARGET_SITE_X_AXIS_WORLD`
- `TARGET_SITE_Y_AXIS_WORLD`
- `TARGET_SITE_Z_AXIS_WORLD`

Together they define:

- the desired joint-space target pose
- the desired tool orientation in world coordinates

The tool orientation target is expressed as a rotation matrix built from those three target axes.

### Orientation Error Helper

`tool_target_orientation_omega(...)` compares:

- current tool-frame axes from `attachment_site`
- desired tool-frame axes in world coordinates

It returns a small-angle angular correction vector.

Conceptually:

- if the tool frame is tilted the wrong way, this function produces a corrective angular motion
- the controller then maps that correction back into joint-space through the rotational Jacobian

### Random Start Sampling

`sample_random_configuration(...)` generates a disturbed starting pose near the active target.

The intent is:

- reproducible
- nontrivial
- not arbitrarily extreme

This gives the controller something real to recover from without testing useless pathological poses every time.

### Legacy Controller

`reverse_nested_x_servo_controller(...)` is the older controller.

It mixed:

- posture recovery
- a small site `x` correction
- tool orientation guidance

It is useful for understanding the project's evolution, but it is not the main active path anymore.

### Active Split Controller

`split_forearm_origin_face_controller(...)` is the active controller.

It explicitly splits the task:

- forearm-origin position control
- tool-face orientation control

The logic is:

1. Start from a small posture regularization term.
2. Add origin guidance using the position Jacobian of `forearm_tip_site`.
3. Add tool-orientation guidance using the rotational Jacobian of `attachment_site`.
4. Optionally add a wrist posture bias.
5. Clip the per-step command change.
6. Write the new setpoint vector.

The key idea is that the wrist is no longer allowed to redefine the origin target just because the tool needs to rotate.

## Origin Probe Layer

[`simulation/probe_origin_pose.py`](./simulation/probe_origin_pose.py) answers:

"What joint vector should count as the canonical target?"

It has two search modes:

- `peak-height`
- `vertical-face`

### Peak-Height Mode

This mode ignores the tool-orientation objective and simply maximizes the world `z` coordinate of `forearm_tip_site`.

### Vertical-Face Mode

This is the important active mode.

It scores a candidate pose by:

- rewarding larger `forearm_tip_site` height
- heavily penalizing any tool orientation error

So the search only accepts poses that match the target tool frame, and among those it prefers the one with the highest origin.

### Probe Workflow

The probe does:

1. Load the MuJoCo model.
2. Resolve the reference site ids.
3. Generate seed poses.
4. Sample random configurations.
5. Evaluate each candidate with forward kinematics.
6. Refine the best candidate with coordinate search.
7. Print a report and optionally save JSON.

That JSON is the cleanest place to inspect:

- candidate `q`
- origin position
- tool-frame world position
- tool-frame world axes
- joint anchors and axes

## Runtime Experiment Layer

[`simulation/run_origin_stabilization.py`](./simulation/run_origin_stabilization.py) is the main execution script.

This file performs the end-to-end experiment.

### Startup

At startup it:

1. Parses CLI arguments.
2. Loads the UR5e scene.
3. Resolves:
   - `forearm_tip_site`
   - `attachment_site`
4. Builds actuator limits.
5. Samples a seeded disturbed start pose around `ACTIVE_ORIGIN_Q`.

### Target Construction

It then creates a separate `target_data` object and forward-propagates `ACTIVE_ORIGIN_Q`.

This is how it measures the target values:

- `target_origin_site_pos`
- `target_tool_site_pos`
- `target_tool_rot`

That step is important because the controller does not guess world positions.
It always derives them from the target joint vector via forward kinematics.

### Main Loop

Each rendered frame contains many smaller simulation steps.

On each simulation step the runner:

1. Reads current `q` and `qvel`
2. Computes the Jacobian of `forearm_tip_site`
3. Computes the Jacobian of `attachment_site`
4. Reads the current tool rotation matrix
5. Calls `split_forearm_origin_face_controller(...)`
6. Writes the returned `ctrl`
7. Calls `mujoco.mj_step(...)`

So the data flow is:

`current state -> Jacobians + site transforms -> controller -> new setpoint -> MuJoCo servo response`

### Metrics

After each rendered frame, the runner records:

- full max joint error
- forearm-origin joint error
- wrist/tool-face joint error
- base pan error
- forearm-origin site error
- tool-site position drift
- tool orientation error

It also renders the overlay text you see in the video.

### Success Conditions

The run computes two success flags:

- `success`
- `success_full_joint_match`

`success` is the task-level objective:

- forearm origin is correct
- tool orientation is correct
- settling happened

`success_full_joint_match` is stricter:

- the entire joint vector also matches the target closely

These can differ if the tool reaches an equivalent orientation with a different wrist configuration.

## Output Layer

Every run produces:

- an MP4 video in [`demonstration_videos/ur5e_cartpole`](./demonstration_videos/ur5e_cartpole)
- a JSON summary in [`outputs/control_runs`](./outputs/control_runs)

The JSON summary is the best compact description of a run.

Read these fields first:

- `origin_target_q`
- `target_origin_site_world`
- `target_tool_site_world`
- `target_tool_rotation_world`
- `final_q`
- `final_origin_site_error_m`
- `final_tool_orientation_error_deg`
- `success`
- `success_full_joint_match`

## Pose Inspection Layer

[`simulation/render_pose_orbit.py`](./simulation/render_pose_orbit.py) is a helper, not part of the control loop.

It does:

1. Load a saved run summary JSON.
2. Read either `final_q` or `origin_target_q`.
3. Load the MuJoCo scene.
4. Set the robot to that single static pose.
5. Orbit a free camera around it.
6. Save a video.

Use this when you want to inspect geometry and orientation from multiple viewpoints without rerunning the controller.

## Current End-To-End Reading Order

If you want to understand the system in the cleanest order, read the files in this sequence:

1. [`mujoco_menagerie/universal_robots_ur5e/ur5e.xml`](./mujoco_menagerie/universal_robots_ur5e/ur5e.xml)
2. [`simulation/controller.py`](./simulation/controller.py)
3. [`simulation/probe_origin_pose.py`](./simulation/probe_origin_pose.py)
4. [`simulation/run_origin_stabilization.py`](./simulation/run_origin_stabilization.py)
5. one JSON summary in [`outputs/control_runs`](./outputs/control_runs)
6. [`simulation/render_pose_orbit.py`](./simulation/render_pose_orbit.py)

That order matches the real data flow:

`model -> targets -> controller -> experiment -> artifacts -> inspection`

## Common Confusions

### "Is the origin the end effector?"

Not in the active architecture.

The origin is defined from `forearm_tip_site`.
The end-effector orientation is defined from `attachment_site`.

### "Is wrist_2_joint locked?"

Not as a raw joint angle lock.

The controller locks the tool orientation in world coordinates.
The wrist joints are free to move however they need to in order to satisfy that task.

### "Why can the wrist move even when the tool is correct?"

Because multiple joint configurations can produce the same tool orientation.

When that happens:

- task success can still be true
- full joint match can be false

### "Why is the arm not maximally extended?"

Because the runner is normally a target-tracking experiment, not a height optimizer.

Height optimization happens in the probe stage.
Tracking happens in the run stage.

## Current Practical Commands

Probe the highest origin under the active tool orientation:

```bash
cd /common/users/ss5772/real_Cartpole
source /common/users/ss5772/miniforge3/etc/profile.d/conda.sh
conda activate mujoco_ur5e
xvfb-run -a python simulation/probe_origin_pose.py --mode vertical-face --seed 7
```

Run the full stabilization experiment:

```bash
cd /common/users/ss5772/real_Cartpole
source /common/users/ss5772/miniforge3/etc/profile.d/conda.sh
conda activate mujoco_ur5e
xvfb-run -a python simulation/run_origin_stabilization.py --seed 7
```

Render an orbit video around a saved pose:

```bash
cd /common/users/ss5772/real_Cartpole
source /common/users/ss5772/miniforge3/etc/profile.d/conda.sh
conda activate mujoco_ur5e
xvfb-run -a python simulation/render_pose_orbit.py \
  --summary-json outputs/control_runs/ur5e_attachment_preferred_face_high_origin_seed7_50fps.json \
  --pose-key final_q
```

## What To Read Next

After this document, the most useful next read is a specific run summary JSON together with the matching video.

That gives you:

- the exact target pose
- the exact final pose
- the exact measured errors
- the exact visual behavior

This is the fastest way to connect the code to what the robot actually did.
