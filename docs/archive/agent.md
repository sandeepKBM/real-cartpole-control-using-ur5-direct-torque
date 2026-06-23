# real_Cartpole Agent Guide

## Mission

Build a simulation-first workflow for a UR5e arm carrying or attaching a cartpole in MuJoCo, then use that model to answer a narrow control question:

- how far the end effector can translate along the world or task `x` axis
- while keeping the target `y`, `z`, and end-effector orientation fixed
- and how that `x` translation maps to joint motion, joint limits, and eventually joint torques

The first controller target is not general 6D manipulation. It is controlled left-to-right motion of the UR5e end effector from a defined origin point.

## Current Project Goal

The current project focus is:

1. keep the MuJoCo UR5e plus cartpole model as the primary control sandbox
2. preserve a ROS 2 plus MoveIt scaffold for frames and later hardware integration
3. use the measured peak-height origin as the canonical reference
4. keep the base-attached rotation joint fixed during the current recovery experiments
5. compute the reachable end-effector `x` limits at fixed `y`, `z`, and orientation
6. extend the validated origin-recovery controller into left-to-right `x` motion

## Current State Snapshot

Verified current state:

- `ORIGIN_PEAK_Q` is measured and stored in [`controller.py`](/common/users/ss5772/real_Cartpole/simulation/controller.py)
- the active simulation runner is [`run_origin_stabilization.py`](/common/users/ss5772/real_Cartpole/simulation/run_origin_stabilization.py)
- the active controller is `reverse_nested_x_servo_controller(...)`
- `shoulder_pan_joint` is currently locked at the origin orientation
- the current reference run converges successfully and is saved as:
  - [`ur5e_origin_stabilization_fixedbase_detailed_seed7.mp4`](/common/users/ss5772/real_Cartpole/demonstration_videos/ur5e_cartpole/ur5e_origin_stabilization_fixedbase_detailed_seed7.mp4)
  - [`ur5e_origin_stabilization_fixedbase_detailed_seed7.json`](/common/users/ss5772/real_Cartpole/outputs/control_runs/ur5e_origin_stabilization_fixedbase_detailed_seed7.json)

Verified metrics for that run:

- reach-chain error `0.001114 rad`
- wrist-orientation error `0.002382 rad`
- base-pan error `0.000000 rad`
- site error `0.000068 m`
- reach settle time `5.32 s`
- full-joint settle time `6.2 s`

## Important Clarification

Inverse kinematics alone is not enough for the full control problem.

- IK answers: "what joint configuration reaches a target pose?"
- the mapping from desired end-effector `x` acceleration to joint torques is a dynamics problem
- the likely stack is:
  - IK or constrained kinematics for pose feasibility
  - Jacobian-based differential kinematics for `x`-only motion
  - inverse dynamics or operational-space control for joint torque commands

Any solver evaluation should keep that separation clear.

## Working Assumptions

- The reference robot is `UR5e`.
- MuJoCo is the first environment used for model validation and controller development.
- ROS integration should follow the simulation model instead of being developed independently.
- The cartpole is initially treated as an attachment to the end effector with known mass and inertia.
- The main task coordinate is end-effector motion along one `x` axis from a chosen origin.
- End-effector `y`, `z`, and orientation are held constant during the first kinematic and control studies.
- The current validated recovery controller uses a fixed base-pan joint and a reverse-priority reach chain.
- Hardware deployment is out of scope until the simulation model is repeatable and measurable.

## Source Areas

Use these existing directories as the main project anchors:

- `simulation/`
- `mujoco_menagerie/universal_robots_ur5e/`
- `demonstration_videos/`
- `README_UR5E.md`
- `CONTROL_DESIGN_NOTEBOOK.md`

## Primary Questions To Resolve

### 1. Origin configuration

Define one canonical origin state that includes:

- world frame
- robot base frame
- end-effector frame
- joint configuration `q0`
- end-effector pose at `q0`
- cartpole pose at `q0`

This origin must become the shared reference for every later experiment.

Current status:

- resolved for the present model
- the remaining question is not the origin itself but which frame should define the project `x` direction for the next motion study

### 2. Reachable `x` interval at fixed pose constraints

Given the origin pose, determine:

- minimum reachable `x`
- maximum reachable `x`
- joint-limit bottlenecks
- singularity or near-singularity regions
- self-collision or cartpole-collision regions

Current status:

- still open
- this is now the main next experiment

### 3. Solver stack for `x`-only control

The solver must support:

- fixed `y`, `z`, and orientation constraints
- differential motion in `x`
- mapping from Cartesian targets to joint velocities or accelerations
- torque-level control through dynamics

Candidate families to evaluate:

- MuJoCo Jacobian plus inverse dynamics as the first baseline
- constrained differential IK
- operational-space control
- external robotics libraries only if the built-in stack is insufficient

Current status:

- the current origin-recovery baseline does not need a full external IK stack
- the next extension should stay in MuJoCo first and only use ROS/MoveIt support if the `x` study needs additional frame or IK utilities

## Near-Term Deliverables

- a maintained origin-configuration specification
- one sweep script for feasible end-effector `x` at fixed pose constraints
- a joint-angle versus end-effector-`x` reference table or plot
- a baseline dynamics mapping from desired `x` acceleration to joint torques
- one controller extension that tracks controlled `x` motion from the validated origin-recovery state

## Execution Rules

- Prefer simulation-first validation before ROS or hardware complexity.
- Keep the first experiments narrow: one translation axis, one orientation target, one origin.
- Log every assumption that changes the reference frame, joint limits, or attachment model.
- Separate kinematics, dynamics, and controller evaluation in the documentation.
- Record failures explicitly, especially unreachable poses and unstable controller regions.

## Success Criteria

The first meaningful success is:

- a reproducible origin configuration
- a verified reachable `x` range for the end effector
- a measured mapping between end-effector `x` motion and UR5e joint bending
- one controller that moves the end effector left-to-right while keeping the other pose constraints approximately fixed

Current progress:

- reproducible origin configuration: complete
- validated origin-recovery controller: complete
- reachable `x` range: pending
- `x`-to-joint-bending map: pending
- left-to-right controller from origin: pending

The second success is:

- stable cartpole transport or regulation behavior under that motion

## Document Ownership

- `agent.md` defines the working mission and constraints for the project
- `PROJECT_PLAN.md` tracks the phased execution plan
- `CONTROL_DESIGN_NOTEBOOK.md` is the current implementation reference for the active controller path
