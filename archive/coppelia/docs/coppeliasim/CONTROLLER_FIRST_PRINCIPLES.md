# UR5 Controller, From First Principles

This note explains the active UR5 torque controller end to end, in the order
you should read it.

It also separates the live controller from the Lua video demo, because those
two paths are easy to mix up and they do different jobs.

## The Short Answer

If you want the live controller, start here:

- [controller_core/x_axis_cartesian_impedance.py](/common/users/ss5772/real_Cartpole/controller_core/x_axis_cartesian_impedance.py)
- [simulation/run_coppeliasim_x_axis_headless.py](/common/users/ss5772/real_Cartpole/simulation/run_coppeliasim_x_axis_headless.py)
- [ros2_ws/src/ur5_x_axis_controller_ros/ur5_x_axis_controller_ros/coppeliasim_adapter.py](/common/users/ss5772/real_Cartpole/ros2_ws/src/ur5_x_axis_controller_ros/ur5_x_axis_controller_ros/coppeliasim_adapter.py)
- [ros2_ws/src/ur5_x_axis_controller_ros/config/controller.yaml](/common/users/ss5772/real_Cartpole/ros2_ws/src/ur5_x_axis_controller_ros/config/controller.yaml)
- [simulation/controller.py](/common/users/ss5772/real_Cartpole/simulation/controller.py)

If you want the Lua video demo, that is:

- [simulation/ur5_mujoco_like_y_torque_addon.lua](/common/users/ss5772/real_Cartpole/simulation/ur5_mujoco_like_y_torque_addon.lua)

## What The Controller Is Doing

At the lowest level, a robot arm is just a state vector:

- joint angles `q`
- joint velocities `qd`
- end-effector position
- end-effector orientation
- Jacobian `J`

The controller takes that state, compares it to a target, and outputs joint
torques.

The key mental model is:

```text
state -> error -> wrench -> J^T wrench -> joint torques -> simulator
```

That is the first-principles version of the live controller.

## Direct Torque Versus Position Servo

This project uses direct torque control, not a pure position servo.

That means the controller does not say "put joint 3 at angle X and let the
simulator do the rest." Instead it computes torques `tau` and sends those
torques into the simulator.

The controller is still closed-loop, though. It reads the current pose and
velocity every step and updates the torques from the current error.

That distinction matters:

- position servo: command a pose, let a built-in servo do the work
- direct torque: compute `tau` yourself, then the physics engine integrates it

## The Live Torque Law

The live task-space controller lives in
[controller_core/x_axis_cartesian_impedance.py](/common/users/ss5772/real_Cartpole/controller_core/x_axis_cartesian_impedance.py).

Its logic is:

1. lock a reference end-effector pose from the current state
2. compute error in X, Y, Z, and orientation
3. turn that error into a task-space wrench
4. map the wrench to joint torques with `J^T`
5. add joint damping and a soft posture term
6. clip torques to the model effort limits

The important line is the Jacobian transpose idea:

```text
tau_task = J^T * wrench
```

The rest of the terms keep the arm stable:

- `tau_damping` removes joint velocity
- `tau_posture` keeps the arm near its starting configuration
- `tau_gravity` can be added when gravity compensation is enabled

The output is then clipped to the per-joint effort limits from the model
config.

## The Acceleration Transport Path

The acceleration-driven path lives mostly in
[simulation/controller.py](/common/users/ss5772/real_Cartpole/simulation/controller.py)
and the main runner
[simulation/run_coppeliasim_x_axis_headless.py](/common/users/ss5772/real_Cartpole/simulation/run_coppeliasim_x_axis_headless.py).

This path is a little different from the pure Cartesian impedance controller.

The flow is:

1. take a signed world-axis acceleration command
2. turn it into a bounded 1D motion profile
3. solve a differential IK problem for a joint reference
4. convert that joint reference into joint torques with joint PD
5. send those torques to CoppeliaSim

So the acceleration path is still torque control, but the torque comes from
joint-space tracking of an IK-generated reference.

## The Lua Video Demo

The Lua file
[simulation/ur5_mujoco_like_y_torque_addon.lua](/common/users/ss5772/real_Cartpole/simulation/ur5_mujoco_like_y_torque_addon.lua)
is a simulator-side demo path.

It is useful because it is self-contained and makes a visible MP4, but it is
not the same as the live ZMQ controller.

Its role is:

- preload the UR5 model in CoppeliaSim
- build a reference motion in Lua
- compute torques in Lua
- save frames and encode video

If you are trying to understand the live controller, treat this as a separate
example, not the main control stack.

## What Each File Is For

- `controller_core/x_axis_cartesian_impedance.py`: the task-space torque law
- `simulation/controller.py`: the acceleration transport policy and IK split
- `simulation/run_coppeliasim_x_axis_headless.py`: the main orchestrator
- `ros2_ws/src/ur5_x_axis_controller_ros/ur5_x_axis_controller_ros/coppeliasim_adapter.py`: simulator bridge
- `ros2_ws/src/ur5_x_axis_controller_ros/config/controller.yaml`: gains and limits
- `simulation/ur5_mujoco_like_y_torque_addon.lua`: Lua video demo

## Reading Order

If you want to build a correct mental model quickly, read in this order:

1. [docs/coppeliasim/PIPELINES.md](/common/users/ss5772/real_Cartpole/docs/coppeliasim/PIPELINES.md)
2. [controller_core/x_axis_cartesian_impedance.py](/common/users/ss5772/real_Cartpole/controller_core/x_axis_cartesian_impedance.py)
3. [simulation/controller.py](/common/users/ss5772/real_Cartpole/simulation/controller.py)
4. [simulation/run_coppeliasim_x_axis_headless.py](/common/users/ss5772/real_Cartpole/simulation/run_coppeliasim_x_axis_headless.py)
5. [ros2_ws/src/ur5_x_axis_controller_ros/ur5_x_axis_controller_ros/coppeliasim_adapter.py](/common/users/ss5772/real_Cartpole/ros2_ws/src/ur5_x_axis_controller_ros/ur5_x_axis_controller_ros/coppeliasim_adapter.py)
6. [ros2_ws/src/ur5_x_axis_controller_ros/config/controller.yaml](/common/users/ss5772/real_Cartpole/ros2_ws/src/ur5_x_axis_controller_ros/config/controller.yaml)
7. [simulation/ur5_mujoco_like_y_torque_addon.lua](/common/users/ss5772/real_Cartpole/simulation/ur5_mujoco_like_y_torque_addon.lua)

## What To Look For While Reading

Ask these questions as you go:

1. What is the target state?
2. What is the measured state?
3. What error is being computed?
4. What part of the error becomes force or torque?
5. Where are the limits applied?
6. Who owns the simulation step?
7. Is this live control or a video demo?

If you can answer those seven questions, the code stops feeling like a pile of
plumbing and starts feeling like a control system.

## Where To Change Things

- To change the torque law, edit
  [controller_core/x_axis_cartesian_impedance.py](/common/users/ss5772/real_Cartpole/controller_core/x_axis_cartesian_impedance.py)
- To change gains and effort limits, edit
  [ros2_ws/src/ur5_x_axis_controller_ros/config/controller.yaml](/common/users/ss5772/real_Cartpole/ros2_ws/src/ur5_x_axis_controller_ros/config/controller.yaml)
- To change simulator wiring, edit
  [ros2_ws/src/ur5_x_axis_controller_ros/ur5_x_axis_controller_ros/coppeliasim_adapter.py](/common/users/ss5772/real_Cartpole/ros2_ws/src/ur5_x_axis_controller_ros/ur5_x_axis_controller_ros/coppeliasim_adapter.py)
- To change the control loop and mode selection, edit
  [simulation/run_coppeliasim_x_axis_headless.py](/common/users/ss5772/real_Cartpole/simulation/run_coppeliasim_x_axis_headless.py)
- To change the Lua video-only demo, edit
  [simulation/ur5_mujoco_like_y_torque_addon.lua](/common/users/ss5772/real_Cartpole/simulation/ur5_mujoco_like_y_torque_addon.lua)

## One Important Caveat

The live controller and the Lua video demo both send torques, but they are not
the same controller.

The live path is the one you should study for control design. The Lua path is
the one you should study for visualization and capture.
