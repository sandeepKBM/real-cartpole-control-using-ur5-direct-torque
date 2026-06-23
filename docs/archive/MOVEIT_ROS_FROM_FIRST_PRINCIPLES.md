# MoveIt And ROS For real_Cartpole: A First-Principles Guide

## Current Status

As of March 20, 2026, this document is background and support guidance, not the primary execution notebook.

The currently validated control artifact is:

- [`ur5e_origin_stabilization_fixedbase_detailed_seed7.mp4`](/common/users/ss5772/real_Cartpole/demonstration_videos/ur5e_cartpole/ur5e_origin_stabilization_fixedbase_detailed_seed7.mp4)
- [`ur5e_origin_stabilization_fixedbase_detailed_seed7.json`](/common/users/ss5772/real_Cartpole/outputs/control_runs/ur5e_origin_stabilization_fixedbase_detailed_seed7.json)

The current primary path is:

- MuJoCo for origin recovery, rendering, metrics, and controller development

The retained ROS/MoveIt path is now:

- robot description
- frame management
- optional kinematics support
- later hardware integration support

It is no longer the main place where progress is being validated.

## What I Changed

I added a bootstrap `ROS 2` and `MoveIt 2` workspace under:

- `ros2_ws/src/real_cartpole_description`
- `ros2_ws/src/real_cartpole_moveit_config`

The goal of these packages is very specific:

1. describe the UR5e arm in a ROS-friendly robot model
2. define the peak-height origin pose we already measured in MuJoCo
3. keep a MoveIt-compatible frame and kinematics scaffold available if you need ROS-side IK later

I also kept the physics-side MuJoCo work in place. The new ROS and MoveIt files do not replace MuJoCo. They add a separate frame and kinematics track beside the control work.

## The Big Idea

Your project really has two different problems inside it.

### Problem A: Kinematics, frames, and planning

This is the question:

- Where is the robot base?
- What does `X`, `Y`, and `Z` mean?
- What is the end-effector frame?
- What joint angles reach a given end-effector pose?
- How do I move from one joint configuration to another safely?

This is where `ROS 2` and `MoveIt 2` are strong.

### Problem B: Dynamics and control

This is the question:

- If the end effector accelerates, what torques are needed?
- How does the attached pole swing?
- What happens under gravity, inertia, damping, and controller feedback?
- Should I use `LQR`, `RL`, or something else?

This is where `MuJoCo` is strong.

That is why the stack is now split:

- `MoveIt 2` for robot description, frames, and optional kinematics metadata
- `MuJoCo` for dynamics and controller development

## What ROS 2 Actually Is

To a beginner, ROS can look like a giant robot library. It is better to think of it as a communication and organization system for robot software.

ROS 2 gives you:

- **nodes**: separate programs that each do one job
- **topics**: streams of messages, like joint states or poses
- **services**: request-response calls
- **actions**: long-running tasks, like executing a trajectory
- **parameters**: structured configuration
- **tf2**: a way to keep track of coordinate frames

If you build without ROS, every script has to invent its own assumptions about frames, controllers, and state flow. That usually becomes messy very quickly.

If you build with ROS, the robot description, frames, and controllers have a common language.

## Why Frames Matter So Much

Earlier, you asked a very important question:

- what are `X`, `Y`, and `Z` here?

That is not a minor detail. It is one of the central questions in robotics.

If you say:

- move the end effector in `+X`

you must immediately ask:

- `+X` of the world frame?
- `+X` of the robot base frame?
- `+X` of the tool frame?

Without a frame tree, different parts of a project silently mean different things by the same symbol.

`tf2` solves that by making frames explicit.

In this scaffold, the important frames are:

- `world`
- `base_link`
- the UR5e intermediate links
- `wrist_3_link`
- `attachment_site_link`

The current measured origin pose from MuJoCo is already written in [`PROJECT_PLAN.md`](/common/users/ss5772/real_Cartpole/PROJECT_PLAN.md). The MoveIt configuration now carries that same pose as a named state called `origin_peak`.

## What MoveIt 2 Does

MoveIt is the layer that sits on top of the robot description and answers practical motion-planning questions.

MoveIt gives you:

- IK solvers
- collision checking
- planning groups
- named robot states
- motion planning
- trajectory execution interfaces
- RViz integration

For your immediate task, MoveIt is useful because it can do this workflow cleanly:

1. define the UR5e arm and tool frame
2. declare a named state for the origin
3. choose an IK solver
4. plan from a random valid joint state to that origin

That is much cleaner than hand-writing this logic in ad hoc scripts.

## What `ros2_control` Does

MoveIt can plan trajectories, but something still has to execute them.

That is the job of `ros2_control`.

You can think of `ros2_control` as the low-level execution layer that exposes robot joints as controllable interfaces.

For this scaffold, I used a fake hardware pattern:

- the URDF includes a `ros2_control` block
- that block uses `mock_components/GenericSystem`
- this lets the robot behave like a controllable arm even without real hardware

That is important because it lets you prototype:

- named states
- planning
- trajectory execution

before you have a real UR5e driver connected.

## Why I Did Not Depend On The Official UR Packages Yet

Normally, for a production ROS workspace, I would prefer to reuse the official Universal Robots description packages.

I did not do that here for one reason:

- there is no visible ROS installation or existing UR package set on this machine right now

So I created a **bootstrap description package** instead.

That means:

- the kinematic chain and joint names are coherent
- the frames are explicit
- the MoveIt scaffold has something real to attach to
- but this is still a starter description, not the final industrial-quality UR integration

Later, once you have a proper ROS and MoveIt installation, you should strongly consider swapping this bootstrap description for the official UR stack.

## The New Packages

### 1. `real_cartpole_description`

This package defines the robot itself.

Main file:

- [`ur5e_cartpole.urdf.xacro`](/common/users/ss5772/real_Cartpole/ros2_ws/src/real_cartpole_description/urdf/ur5e_cartpole.urdf.xacro)

What it contains:

- a `world` frame
- a `base_link`
- the UR5e chain:
  - `shoulder_link`
  - `upper_arm_link`
  - `forearm_link`
  - `wrist_1_link`
  - `wrist_2_link`
  - `wrist_3_link`
- a fixed tool frame:
  - `attachment_site_link`
- a `ros2_control` fake-hardware block

Important design choice:

- the geometry is simplified

That means the visuals and collisions are not an exact copy of the industrial UR meshes. The goal here is to get a clean planning scaffold, not a perfect CAD model.

### 2. `real_cartpole_moveit_config`

This package tells MoveIt how to think about the robot.

Main files:

- [`real_cartpole.srdf`](/common/users/ss5772/real_Cartpole/ros2_ws/src/real_cartpole_moveit_config/config/real_cartpole.srdf)
- [`kinematics.yaml`](/common/users/ss5772/real_Cartpole/ros2_ws/src/real_cartpole_moveit_config/config/kinematics.yaml)
- [`joint_limits.yaml`](/common/users/ss5772/real_Cartpole/ros2_ws/src/real_cartpole_moveit_config/config/joint_limits.yaml)

What it defines:

- the planning group `ur5e_arm`
- the tool end-effector
- a fixed virtual joint from `world` to `base_link`
- disabled self-collision pairs for the simplified model
- the named joint state `origin_peak`
- the MoveIt kinematics solver:
  - `kdl_kinematics_plugin/KDLKinematicsPlugin`

This is the critical file for your solver choice:

- [`kinematics.yaml`](/common/users/ss5772/real_Cartpole/ros2_ws/src/real_cartpole_moveit_config/config/kinematics.yaml)

That file is where the MoveIt arm group is told which kinematics plugin is actually available in the current Jazzy environment.

## Planner Demo Removal

The earlier `real_cartpole_moveit_demos` package and OMPL planner launch path were removed.

Reason:

- the project direction is control and stability first
- the random-to-origin task is better expressed as a stabilization problem in MuJoCo
- keeping planner-only code around was adding maintenance cost without helping the current experiments

The control-side replacement now lives in:

- [`run_origin_stabilization.py`](/common/users/ss5772/real_Cartpole/simulation/run_origin_stabilization.py)

The current source tree no longer keeps that demo package.

## How The Origin Was Carried Into MoveIt

We already measured the peak-height origin in MuJoCo. That produced the joint vector:

- `shoulder_pan_joint = 0.000000`
- `shoulder_lift_joint = -1.570784`
- `elbow_joint = -0.000007`
- `wrist_1_joint = -2.356201`
- `wrist_2_joint = -1.570784`
- `wrist_3_joint = 0.000000`

Instead of hiding those values in code, I stored them as a named state in the SRDF:

- `origin_peak`

That matters because named states are easy for beginners to reason about.

Instead of saying:

- “go to six floating-point joint values”

you can say:

- “go to `origin_peak`”

That is a much better mental model for learning and debugging.

## Why The Control-First Path Won

The real project question is not “can a planner find a path?” It is “can the arm and attached system be driven and stabilized from a disturbed state?”

That is why the current working experiment is in MuJoCo instead of MoveIt planning:

- the MuJoCo model already contains strong joint servos
- the random-to-origin task can be written as a direct stabilization problem
- the result you care about is convergence, settling time, and stability rather than path search

That decision is now reflected in the verified fixed-base detailed run, which converges successfully in MuJoCo while preserving a ROS-side scaffold for later use.

## What Is Still Approximate

This scaffold is useful, but it is not the final production setup.

Important approximations:

1. the URDF geometry is simplified  
It is designed to bootstrap planning, not to match the real UR5e meshes perfectly.

2. the collision pairs are manually relaxed  
Because the geometry is simplified, I disabled some adjacent collisions so the approximate links do not overconstrain the bootstrap model.

3. the retained ROS files are not the primary execution path  
The current validated experiment path is the MuJoCo controller workflow.

4. the ROS side is now optional support code  
It is there for frames, robot description, and future kinematics utility work, not as the main control loop.

## What I Could Not Fully Validate Here

I could not yet claim full hardware-grade ROS integration.

The control work is validated first in MuJoCo, and that is now the intended primary path.

What has been validated on the ROS side:

- `real_cartpole_control` builds
- the dry-run `origin_hold` launch path starts
- bounded origin commands are computed correctly from `/joint_states`

## How To Use This Once ROS 2 And MoveIt 2 Are Installed

Because this machine is Ubuntu 24.04, the most natural ROS 2 target is typically `Jazzy`.

Once ROS 2, MoveIt 2, `ros2_control`, `xacro`, and the needed controller packages are installed, the normal workflow is:

```bash
cd /common/users/ss5772/real_Cartpole/ros2_ws
colcon build
source install/setup.bash
```

To run the control-first simulation baseline:

```bash
xvfb-run -a python simulation/run_origin_stabilization.py
```

To bring up the real-arm ROS controller in safe dry-run mode:

```bash
ros2 launch real_cartpole_control origin_hold.launch.py
```

## What I Recommend You Do Next

### Near term

1. keep the ROS description and control scaffold in sync with the validated MuJoCo origin
2. use MuJoCo to do the next `x`-range and `x`-tracking experiments
3. only return to ROS-side work when the simulation controller needs a hardware-facing interface

### After that

1. replace the bootstrap URDF with the official UR description packages if available
2. confirm that the `attachment_site_link` frame matches the tool frame you want physically
3. connect the ROS origin-hold node to the actual UR driver interfaces
4. reuse the same fixed-base origin definition and command conventions on the real arm side

### Longer term

1. use MoveIt for:
   - IK
   - frame management
   - later hardware execution plumbing
2. use MuJoCo for:
   - cartpole dynamics
   - end-effector acceleration and torque studies
   - controller studies after the current deterministic baseline

## Why ROS Helps Here, In One Sentence

ROS helps because it turns your project from “a set of disconnected scripts with hidden assumptions” into “a robot system with explicit frames, explicit models, explicit controllers, and reusable planning tools.”

## Official References

These are the main official references behind the stack choice:

- ROS 2 `tf2` overview: <https://docs.ros.org/en/rolling/Concepts/Intermediate/About-Tf2.html>
- MoveIt kinematics configuration tutorial: <https://moveit.picknik.ai/main/doc/examples/kinematics_configuration/kinematics_configuration_tutorial.html>
- MoveIt move group interface tutorial: <https://moveit.picknik.ai/main/doc/examples/move_group_interface/move_group_interface_tutorial.html>
- MoveIt Python API tutorial: <https://moveit.picknik.ai/main/doc/examples/motion_planning_python_api/motion_planning_python_api_tutorial.html>
- MoveIt Servo tutorial: <https://moveit.picknik.ai/humble/doc/examples/realtime_servo/realtime_servo_tutorial.html>
- `ros2_control` getting started: <https://control.ros.org/humble/doc/getting_started/getting_started.html>
