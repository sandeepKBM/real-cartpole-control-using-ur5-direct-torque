# Old Controller Reuse

This repo already contains useful controller math, but only some of it is appropriate for the new UR5e hardware staging lane.

## Reused

Pure math and safety helpers:

- `controller_core/x_axis_cartesian_impedance.py`
- `controller_core/safety.py`
- `controller_core/safety_utils.py`
- `controller_core/filters.py`
- `controller_core/logging_utils.py`
- `controller_core/state_types.py`
- `controller_core/x_axis_controller.py`

These are useful because they are simulator-independent and do not send motion by themselves.

## Not reused directly

Simulation-only torque application and CoppeliaSim control paths are not hardware-safe:

- `simulation/controller.py`
- `simulation/external_zmq_controller_common.py`
- CoppeliaSim direct torque lanes

Those paths assume a simulator and cannot be treated as a real-arm safety layer.

## How the old controller should be used on hardware

If you want to reuse the old positional / differential-IK logic for the real arm, convert it to bounded joint-position or joint-velocity targets first.

Preferred first hardware targets:

- `servoJ` with tiny joint-space targets
- `speedJ` only if it is demonstrably safer for the specific stage

Do not send the simulation torque output straight to the UR5e.

## Progression path

1. Receive-only RTDE.
2. Zero-hold `servoJ`.
3. Tiny bounded `servoJ` motion.
4. Bounded position / velocity targets from old controller logic.
5. Cartpole-specific behavior only after the arm-only stages pass.
6. Direct torque only after a documented robot-side control loop exists.
