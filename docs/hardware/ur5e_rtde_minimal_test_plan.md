# UR5e RTDE Minimal Test Plan

This plan is intentionally conservative. It keeps the default path at receive-only / no motion until each prior stage passes.

## Stage 0: Dry-run, no robot

Use the local timing diagnostic first:

```bash
python3 tools/diagnose_rtde_timing.py \
  --dry-run \
  --frequency 500 \
  --duration 60 \
  --output logs/timing_report.json
```

Pass criteria:
- JSON report is written.
- p95/p99/max are recorded.
- No robot connection is attempted.

## Stage 1: Receive-only RTDE

Read state only. No motion commands.

```bash
python3 tools/ur5e_receive_only.py \
  --robot-ip <UR5E_IP> \
  --frequency 500 \
  --duration 30 \
  --output logs/receive_only.json
```

Pass criteria:
- RTDE receive connects.
- Joint state and TCP pose are finite.
- Deadline misses are either zero or explicitly reported and fail the run.
- No command channel is opened unless the library requires it.

## Staged hardware pipeline node

The repo also has a thin ROS 2 wrapper for the staged hardware flow in
`ros2_ws/src/ur5_x_axis_controller_ros/ur5_x_axis_controller_ros/ur5e_hardware_pipeline_node.py`
with launch file
`ros2_ws/src/ur5_x_axis_controller_ros/launch/run_ur5e_hardware_pipeline.launch.py`.

It centralizes the three hardware lanes we want to keep separate:

- `connection_smoke` - connection and state snapshot only, no motion
- `basic_servoj_hold` - bounded `servoJ` hold, explicit motion opt-in required
- `basic_servoj_tiny` - tiny bounded `servoJ` perturbation, explicit motion opt-in required
- `direct_torque_probe` - zero-only direct torque probe by default, nonzero torque remains blocked unless explicitly enabled later

Safe first launch example:

```bash
source /opt/ros/humble/setup.bash
source /path/to/real_Cartpole/ros2_ws/install/setup.bash
ros2 launch ur5_x_axis_controller_ros run_ur5e_hardware_pipeline.launch.py \
  robot_ip:=<UR5E_IP> \
  stage:=connection_smoke \
  motion_opt_in:=false \
  allow_nonzero_direct_torque:=false
```

Pass criteria:

- the bridge reports receive/control capability metadata
- joint and TCP state are finite
- no motion command is issued
- direct torque remains blocked by default

## Stage 2: Zero-hold `servoJ`

Hold the current joint state exactly. Motion opt-in required.

```bash
python3 tools/ur5e_servoj_zero_hold.py \
  --robot-ip <UR5E_IP> \
  --frequency 500 \
  --duration 5 \
  --gain 100 \
  --lookahead-time 0.1 \
  --velocity 0.05 \
  --acceleration 0.05 \
  --max-deadline-ms 3.0 \
  --i-understand-this-moves-the-robot \
  --output logs/servoj_zero_hold.json
```

Pass criteria:
- `servoJ` holds the current joint pose.
- `servoStop()` / `stopJ()` / `stopScript()` are called on exit.
- Any deadline miss, NaN, stale state, or disconnect fails the run.

## Stage 3: Tiny bounded `servoJ` motion

Small sinusoidal perturbation around the current pose.

```bash
python3 tools/ur5e_servoj_tiny_motion.py \
  --robot-ip <UR5E_IP> \
  --frequency 500 \
  --duration 3 \
  --joint-index 0 \
  --amplitude-rad 0.005 \
  --max-amplitude-rad 0.01 \
  --gain 100 \
  --lookahead-time 0.1 \
  --velocity 0.05 \
  --acceleration 0.05 \
  --max-deadline-ms 3.0 \
  --i-understand-this-moves-the-robot \
  --output logs/servoj_tiny_motion.json
```

Pass criteria:
- Motion stays tiny and smooth.
- Command delta limits are enforced.
- Log includes actual-vs-commanded joint error.

## Stage 4: Reuse old controller only through bounded joint targets

Reuse the older positional / differential-IK logic only as a source of bounded joint targets or velocity targets.

Do not port simulation torque commands directly to the real arm.

## Stage 5: Cartpole behavior only after all prior stages pass

Only after the arm-only tests are stable should any cartpole-specific behavior be attempted.

## Stage 6: Direct torque only after extra gates

Direct torque should remain refused unless all of the following are documented and approved:
- supervisor approval
- PolyScope / URSoftware compatibility checklist
- robot-side watchdog plan
- zero-torque probe passes
- a robot-side real-time loop is in place

At the moment this repo only provides a guarded zero-only probe and refuses nonzero direct torque.
