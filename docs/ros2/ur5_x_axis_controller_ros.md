# ur5_x_axis_controller_ros

This ROS 2 package originally hosted a CoppeliaSim torque/force-mode Cartesian impedance
controller stack. That CoppeliaSim-facing stack (`controller_node.py`,
`coppeliasim_bridge_node.py`, `coppeliasim_adapter.py`, the `config/controller_coppelia_*.yaml`
family, the CoppeliaSim scene setup / tuning / JSONL-trace-plotting sections that used to make
up most of this document) was archived 2026-07-03 to
`archive/coppelia/ros2/ur5_x_axis_controller_ros/` and is not runnable in place. See
`archive/coppelia/README.md` for resurrect notes and `docs/archive/coppelia_port_stage1_audit.md`
for the original MuJoCo-baseline audit that preceded the CoppeliaSim port.

The live Cartesian impedance control law (`controller_core/x_axis_cartesian_impedance.py`)
that stack used is unarchived and unchanged — it's the same law the active MuJoCo lane uses
today; see `AGENTS.md` §3 for its current (Pinocchio/operational-space-upgraded) state.

What remains live in this package is the **real-hardware staging pipeline**, described below.

## Hardware readiness audit

This workspace does **not** yet contain a full real UR5e RTDE control loop — only staged,
guardrailed probes. What is present:

- `real_cartpole_control` publishes bounded `JointTrajectory` commands to a mock
  `ros2_control` system in the URDF. It is safe-by-default and is not an RTDE driver.
- The repo has safety monitors, torque clipping, and rate limiting, but no `servoJ`,
  `speedJ`, `moveJ`, `moveL`, or unguarded RTDE connection code in the active source tree.

What is missing for full hardware readiness:

- a real UR5e RTDE client / URScript wrapper beyond the staged probes below
- a 500 Hz fail-stop watchdog on the hardware side
- explicit robot-side safe-stop commands (`stopJ`, `servoStop`, or equivalent)
- a documented PolyScope/firmware compatibility target

Safe pre-hardware diagnostic:

```bash
python3 tools/diagnose_rtde_timing.py \
  --dry-run \
  --frequency 500 \
  --duration 60 \
  --output timing_report.json
```

Use the dry-run report to check whether the host can sustain a 2 ms budget in pure Python
before any real-arm work. That result does **not** prove RTDE readiness by itself.

Staged hardware lane (see `AGENTS.md` §4 for the guardrail guarantees — do not weaken any of
these):

- `hardware/ur5e_rtde_bridge.py` wraps optional RTDE receive/control clients.
- `tools/ur5e_receive_only.py` reads state only.
- `tools/ur5e_servoj_zero_hold.py` and `tools/ur5e_servoj_tiny_motion.py` require explicit
  motion opt-in.
- `tools/ur5e_direct_torque_probe.py` is zero-only by default and refuses nonzero torque
  regardless of flags.
- The optional ROS visualization publishers live behind the staged scripts and never block
  the hard loop.

## Staged hardware pipeline node

For a cleaner ROS-facing entrypoint, the package includes a dedicated staged hardware
pipeline wrapper:

- `ros2_ws/src/ur5_x_axis_controller_ros/ur5_x_axis_controller_ros/ur5e_hardware_pipeline_node.py`
- `ros2_ws/src/ur5_x_axis_controller_ros/launch/run_ur5e_hardware_pipeline.launch.py`

It is designed to keep the three hardware phases separate:

- `connection_smoke` — connect, read state, and publish capability metadata
- `basic_servoj_hold` — bounded `servoJ` hold with explicit motion opt-in
- `basic_servoj_tiny` — tiny bounded `servoJ` perturbation with explicit motion opt-in
- `direct_torque_probe` — direct torque probe lane, zero-only by default

The launch defaults remain safe:

- `motion_opt_in=false`
- `allow_nonzero_direct_torque=false`
- `direct_torque_zero_only=true`

Example connection-only launch:

```bash
source /opt/ros/humble/setup.bash
source /common/users/ss5772/real_Cartpole/ros2_ws/install/setup.bash
ros2 launch ur5_x_axis_controller_ros run_ur5e_hardware_pipeline.launch.py \
  robot_ip:=<UR5E_IP> \
  stage:=connection_smoke \
  motion_opt_in:=false \
  allow_nonzero_direct_torque:=false
```

This wrapper is a staging scaffold, not evidence that direct torque is ready on hardware. It
exists so the eventual real-arm path can be engineered in small, auditable steps without
conflating connection smoke, bounded motion, and direct torque policy.
