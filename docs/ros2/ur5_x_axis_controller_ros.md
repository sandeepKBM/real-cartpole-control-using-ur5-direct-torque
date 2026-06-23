# ur5_x_axis_controller_ros

ROS 2 package for **UR5.ttm** in **CoppeliaSim** using **torque/force mode** and a **backend-independent Cartesian impedance** controller (`controller_core/x_axis_cartesian_impedance.py`).

## Control law (summary)

At first valid measurement the controller stores `x0,y0,z0`, tool quaternion `quat0`, and joint rest posture `q_rest`. Then each cycle:

- `x_des` comes from `/target_x`, **slew-limited** by `target_x_step_max_m` and `target_x_velocity_limit_mps` (see `config/controller.yaml`).
- `y_des=y0`, `z_des=z0`, `quat_des=quat0`.
- Cartesian PD on translation + orientation error, mapped through `tau_task = J^T * wrench`.
- Joint damping `tau_damp = -kd_joint * qd`.
- Posture `tau_post = Kp_post*(q_rest - q) - Kd_post*qd`.
- Optional `tau_gravity` from `/ur5/gravity_torque` when `use_gravity_compensation: true`.
- Torque saturation, **low-pass**, and **rate limit** on the final command.

## Legacy positional transport family

The ROS 2 node also has a diagnostic fallback that reuses the older
`simulation/controller.py` transport logic:

- Set `controller.family: legacy_xz_transport_pd`.
- The node calls `simulation.controller.differential_ik_xz_transport_controller(...)`.
- The resulting reduced-chain joint target is converted to joint torques with a local joint-PD wrapper.
- `/target_x` is treated as an absolute world-X target and is slew-limited by `target_x_step_max_m` and `target_x_velocity_limit_mps`.
- The task frame for this lane should stay on `mujoco_attachment_dummy` with the `/UR5/connection` EE proxy and its alternates.

Verified diagnostic launch command:

```bash
source /common/users/ss5772/miniforge3/etc/profile.d/conda.sh
conda activate mujoco_ur5e
source /common/users/ss5772/real_Cartpole/ros2_ws/install/setup.bash
export PYTHONPATH=/common/users/ss5772/real_Cartpole:${PYTHONPATH}
ros2 launch ur5_x_axis_controller_ros run_controller.launch.py \
  config_path:=/common/users/ss5772/real_Cartpole/ros2_ws/src/ur5_x_axis_controller_ros/config/controller_coppelia_legacy_xz_transport_relaxed.yaml
```

Observed result on this build:

- startup seeding completed cleanly
- the bridge resolved all six joints and the EE proxy
- publishing a small absolute `/target_x` step moved the EE in world X
- the relaxed ROS-side safety gate stayed green during the verified step

Use this family for ROS 2 bring-up verification. Keep the Cartesian impedance
family as the torque-controller baseline.

## Files

| Path | Role |
|------|------|
| `config/controller.yaml` | Single source of truth: `controller`, `coppeliasim`, `safety`, `topics`, `logging`. |
| `ur5_x_axis_controller_ros/config_loader.py` | Load YAML (PyYAML). |
| `ur5_x_axis_controller_ros/controller_node.py` | Impedance + filter + safety + JSONL trace + `/ur5/controller_debug` + `/ur5/safety_status`. |
| `ur5_x_axis_controller_ros/coppeliasim_bridge_node.py` | ZMQ bridge: state publishers + torque application. |
| `ur5_x_axis_controller_ros/coppeliasim_adapter.py` | Handle resolution (fails loudly), `setJointTargetForce` with fallback, Jacobian API or FD. |

## CoppeliaSim scene (Stage 6 checklist)

1. Model browser → load `models/robots/non-mobile/UR5.ttm`.
2. Save scene as e.g. `scenes/ur5_x_axis_torque_test.ttt`.
3. Each of the six joints: **dynamic**, **force/torque** control, **position control disabled**; remove/disable child scripts that command the same joints.
4. Default tool object in YAML: `ee_object_name: /UR5/UR5_connection` (adjust if your tree differs).
5. Enable **ZMQ Remote API** (default port `23000`).
6. Start simulation, then start ROS.

## Jacobian (Stage 7)

Priority in `config/controller.yaml` under `coppeliasim.jacobian.source`:

- `auto` — try `sim.getJacobian`; on failure use **numerical** Jacobian (small joint perturbations, acceptable at 20–50 Hz for first demos).
- `api` — API only.
- `numerical` — FD only.

The bridge publishes a **6×6** row-major Jacobian on `/ur5/jacobian` (3 linear + 3 angular rows).

## End-to-end commands (Stage 8)

**Terminal 1 — build**

```bash
cd /path/to/real_Cartpole/ros2_ws
source /opt/ros/humble/setup.bash
export PYTHONPATH="$PYTHONPATH:$(pwd)/.."   # so `controller_core` imports
colcon build --packages-select ur5_x_axis_controller_ros
source install/setup.bash
```

**Terminal 2 — CoppeliaSim**

Open CoppeliaSim, load your `.ttt` scene, press **Play**.

**Terminal 3 — launch**

```bash
source /opt/ros/humble/setup.bash
source /path/to/real_Cartpole/ros2_ws/install/setup.bash
pip install coppeliasim-zmqremoteapi-client numpy pyyaml
ros2 launch ur5_x_axis_controller_ros run_controller.launch.py
```

Optional: bridge off (controller only, if you replay bagged topics):

```bash
ros2 launch ur5_x_axis_controller_ros run_controller.launch.py run_bridge:=false
```

**Terminal 4 — inspect**

```bash
ros2 topic list
ros2 topic echo /joint_states --once
ros2 topic echo /ur5/ee_pose --once
ros2 topic echo /ur5/controller_debug --once
```

**Tiny X targets (do not start large)**

```bash
ros2 topic pub --once /target_x std_msgs/msg/Float64 "{data: 0.01}"
# then if stable:
ros2 topic pub --once /target_x std_msgs/msg/Float64 "{data: 0.02}"
ros2 topic pub --once /target_x std_msgs/msg/Float64 "{data: -0.02}"
```

## JSONL trace (Stage 10)

Set in `config/controller.yaml`:

```yaml
logging:
  trace_jsonl_path: "/tmp/ur5_trace.jsonl"
```

Plot:

```bash
python simulation/plot_coppeliasim_trace.py /tmp/ur5_trace.jsonl --out /tmp/plot
```

## Tuning procedure (Stage 9)

**Pass 0 — read-only:** bridge only; verify `q`, `ee_pos`, `ee_quat` look sane when you jog the arm manually.

**Pass 1 — zero torque:** publish `Float64MultiArray` zeros on `/ur5/torque_command`; expect no violent motion (gravity sag is normal without gravity comp).

**Pass 2 — posture + damping:** keep `target_x` at initial `x0`; increase `kp_posture` / `kd_joint` slowly if the arm drifts or oscillates.

**Pass 3 — Y/Z/orientation hold:** keep `target_x = x0`; tune `kp_y,kd_y,kp_z,kd_z,kp_rot,kd_rot` until the EE holds pose.

**Pass 4 — tiny X:** `target_x = x0 + 0.01` m; check Y/Z drift `< 3 cm`, orientation `< 0.25 rad`, bounded torques.

**Pass 5 — larger X:** only after Pass 4 is solid; then `±0.02` m; raise `kp_x` last.

**Pass 6 — MuJoCo parity:** record JSONL / Coppelia trace and compare X, Y/Z drift, torques, joint motion (see `simulation/compare_mujoco_vs_coppeliasim.py` for a starting point).

### Tuning rules (from spec)

- First run with `torque_limits_mode: "initial"` only.
- If height drops, **raise `kp_z` before `kp_x`**.
- If orientation wanders, **raise `kp_rot` before `kp_x`**.
- If joints thrash, **raise damping / posture** or **lower Cartesian gains**.

## Acceptance criteria (Stage 11)

See project spec: X tracks toward target; Y/Z drift small; orientation stable; posture similar; torques bounded; no limit violations; trace saved for comparison.

## Stage 1 audit

MuJoCo baseline audit (no behavior change): `docs/archive/coppelia_port_stage1_audit.md`.

## Hardware readiness audit

This workspace does **not** yet contain a real UR5e RTDE hardware loop.

What is present:

- `real_cartpole_control` publishes bounded `JointTrajectory` commands to a
  mock `ros2_control` system in the URDF. It is safe-by-default and is not an
  RTDE driver.
- `ur5_x_axis_controller_ros` is the active torque-control stack for CoppeliaSim
  over ZMQ. Its control rate is 100 Hz, not 500 Hz.
- The repo has safety monitors, torque clipping, and rate limiting, but no
  `servoJ`, `speedJ`, `moveJ`, `moveL`, or RTDE connection code in the active
  source tree.

What is missing for hardware:

- a real UR5e RTDE client / URScript wrapper
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

Use the dry-run report to check whether the host can sustain a 2 ms budget in
pure Python before any real-arm work. That result does **not** prove RTDE
readiness by itself.

New staged hardware lane:

- `hardware/ur5e_rtde_bridge.py` wraps optional RTDE receive/control clients.
- `tools/ur5e_receive_only.py` reads state only.
- `tools/ur5e_servoj_zero_hold.py` and `tools/ur5e_servoj_tiny_motion.py` require explicit motion opt-in.
- `tools/ur5e_direct_torque_probe.py` is zero-only by default and refuses nonzero torque in this patch.
- The optional ROS visualization publishers live behind the staged scripts and never block the hard loop.

## Staged hardware pipeline node

For a cleaner ROS-facing entrypoint, the repo now includes a dedicated staged
hardware pipeline wrapper:

- `ros2_ws/src/ur5_x_axis_controller_ros/ur5_x_axis_controller_ros/ur5e_hardware_pipeline_node.py`
- `ros2_ws/src/ur5_x_axis_controller_ros/launch/run_ur5e_hardware_pipeline.launch.py`

It is designed to keep the three hardware phases separate:

- `connection_smoke` - connect, read state, and publish capability metadata
- `basic_servoj_hold` - bounded `servoJ` hold with explicit motion opt-in
- `basic_servoj_tiny` - tiny bounded `servoJ` perturbation with explicit motion opt-in
- `direct_torque_probe` - direct torque probe lane, zero-only by default

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

This wrapper is a staging scaffold, not evidence that direct torque is ready on
hardware. It exists so the eventual real-arm path can be engineered in small,
auditable steps without conflating connection smoke, bounded motion, and direct
torque policy.
