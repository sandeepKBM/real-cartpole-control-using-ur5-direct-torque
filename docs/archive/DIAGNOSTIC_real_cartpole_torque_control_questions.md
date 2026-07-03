# Diagnostic: real_Cartpole torque-control questions

> **SUPERSEDED / HISTORICAL** — moved to `docs/archive/` 2026-07-03. This diagnostic's own
> recommended next steps (Pinocchio parity check → gravity source swap → Coriolis feedforward
> → operational-space upgrade) have since been implemented in full (P0-P3, see `AGENTS.md`
> §3) and the resulting controller was tuned and extensively validated. Read `AGENTS.md` §3
> for the current answer to the question this document asks; kept as the origin story for
> that work.

Scope: code/log inspection only. No controller changes, no safety-limit changes, no hardware execution, no RL, no CoppeliaSim runtime changes.

Evidence sources used:
- `docs/README.md`
- `docs/CURRENT_STATUS.md`
- `.codex_graph/context_pack.md`
- `simulation/launch_coppeliasim_x_axis_headless.sh`
- `simulation/run_coppeliasim_x_axis_headless.py`
- `simulation/controller.py`
- `controller_core/x_axis_cartesian_impedance.py`
- `controller_core/torque_task_qp.py`
- `controller_core/safety.py`
- `controller_core/filters.py`
- `controller_core/kinematics_utils.py`
- `mujoco_ur5e_tools.py`
- `simulation/ur5e_mujoco_torque.py`
- `tools/ur5e_mujoco_torque_experiments.py`
- `tools/ur5e_move_hold_transport.py`
- `tools/tune_ur5e_impedance_transport.py`
- `tools/tune_ur5e_residual_impedance_transport.py`
- `tools/ur5e_x_frame_envelope.py`
- `tools/audit_ur5e_mujoco_gravity_torque.py`
- `ros2_ws/src/ur5_x_axis_controller_ros/launch/run_controller.launch.py`
- `ros2_ws/src/ur5_x_axis_controller_ros/ur5_x_axis_controller_ros/controller_node.py`
- `ros2_ws/src/ur5_x_axis_controller_ros/ur5_x_axis_controller_ros/coppeliasim_bridge_node.py`
- `ros2_ws/src/ur5_x_axis_controller_ros/ur5_x_axis_controller_ros/coppeliasim_adapter.py`
- `ros2_ws/src/ur5_x_axis_controller_ros/ur5_x_axis_controller_ros/messages.py`
- `ros2_ws/src/ur5_x_axis_controller_ros/config/controller.yaml`
- `ros2_ws/src/ur5_x_axis_controller_ros/config/controller_coppelia_y_transport_torque.yaml`
- `ros2_ws/src/ur5_x_axis_controller_ros/config/controller_coppelia_legacy_xz_transport.yaml`
- `ros2_ws/src/ur5_x_axis_controller_ros/config/controller_coppelia_legacy_xz_transport_relaxed.yaml`
- `ros2_ws/src/ur5_x_axis_controller_ros/config/controller_coppelia_slow_seeded_probe.yaml`
- `ros2_ws/src/real_cartpole_description/urdf/ur5e_cartpole.urdf.xacro`
- `outputs/control_runs/coppelia_lqr_ratefix_coppeliasim_x_axis_offscreen_capture_summary.json`
- `outputs/control_runs/coppelia_lqr_smoke16_coppeliasim_x_axis_offscreen_capture_summary.json`
- `outputs/control_runs/coppelia_torque_diagnostics/03_hold_soft_impedance_summary.json`
- `outputs/control_runs/coppelia_torque_diagnostics/06_tiny_x_motion_summary.json`
- `outputs/ur5e_mujoco_torque_transport/residual_impedance_tuning_20260630_034451/best_settings.json`
- `outputs/ur5e_mujoco_torque_transport/move_hold_transport_20260630_074824/summary.json`

## 1. Exact control stack

There are two relevant stacks in this repository.

### A. Active MuJoCo true-torque / residual-torque lane

This is the current simulation-only development lane.

Call chain:
- `tools/ur5e_mujoco_torque_experiments.py::main()`
  - loads the MuJoCo UR5e torque scene via `simulation/ur5e_mujoco_torque.py::load_model()`
  - instantiates the controller via `simulation/ur5e_mujoco_torque.py::build_controller()`
  - steps the MuJoCo scene and writes torques into `data.ctrl[:6]`
- `tools/ur5e_move_hold_transport.py` and `tools/tune_ur5e_residual_impedance_transport.py`
  - build move / hold trajectories and feed them into the same MuJoCo torque adapter
- `simulation/ur5e_mujoco_torque.py::MujocoUR5eTorqueAdapter`
  - computes `tau_controller`
  - computes `tau_gravity` when `gravity_mode == "gravity_comp"`
  - returns `tau_applied = tau_controller + tau_gravity`
  - shapes and clips the result before it is written to MuJoCo
- `tools/ur5e_mujoco_torque_experiments.py` writes the final torque to MuJoCo:
  - `data.ctrl[:6] = np.asarray(tau, dtype=np.float64).reshape(6)`

Exact low-level controller implementations in this lane:
- `controller_core/x_axis_cartesian_impedance.py::XAxisCartesianImpedanceController`
- `controller_core/torque_task_qp.py::TorqueTaskQPController`
- `simulation/ur5e_mujoco_torque.py::ZeroTorqueController`

Safety and filtering in this lane:
- `controller_core/safety.py::ImpedanceSafetyMonitor`
- `controller_core/filters.py::TorqueCommandFilter`
- `transport_metrics.py::compute_valid_transport_metrics()`
- `transport_metrics.py::compute_valid_move_hold_metrics()`

Config files:
- `config/ur5e_mujoco_torque_transport.yaml`
- `config/ur5e_mujoco_torque.yaml`

### B. Legacy CoppeliaSim / ROS torque lane

This is a separate historical / reference path, still present in the tree.

Call chain:
- `simulation/launch_coppeliasim_x_axis_headless.sh`
  - starts CoppeliaSim
  - launches `simulation/run_coppeliasim_x_axis_headless.py`
- `simulation/run_coppeliasim_x_axis_headless.py::main()`
  - loads runtime config
  - constructs a controller (`XAxisCartesianImpedanceController`, `TorqueTaskQPController`, or legacy differential-IK family)
  - applies safety checks
  - sends torques to the adapter
- `ros2_ws/src/ur5_x_axis_controller_ros/launch/run_controller.launch.py`
  - starts the bridge node and `controller_node`
- `ros2_ws/src/ur5_x_axis_controller_ros/ur5_x_axis_controller_ros/controller_node.py::ControllerNode`
  - computes torque commands and publishes them on the ROS side
- `ros2_ws/src/ur5_x_axis_controller_ros/ur5_x_axis_controller_ros/coppeliasim_adapter.py::CoppeliaSimURAdapter.apply_torque()`
  - final torque sink to CoppeliaSim

The legacy high-level transport helper is:
- `simulation/controller.py::differential_ik_xz_transport_controller()`
- In the ROS node, that legacy family is wrapped by `controller_node.py::_joint_pd_torque()`

Config files:
- `ros2_ws/src/ur5_x_axis_controller_ros/config/controller.yaml`
- `ros2_ws/src/ur5_x_axis_controller_ros/config/controller_coppelia_y_transport_torque.yaml`
- `ros2_ws/src/ur5_x_axis_controller_ros/config/controller_coppelia_legacy_xz_transport.yaml`
- `ros2_ws/src/ur5_x_axis_controller_ros/config/controller_coppelia_legacy_xz_transport_relaxed.yaml`
- `ros2_ws/src/ur5_x_axis_controller_ros/config/controller_coppelia_slow_seeded_probe.yaml`

### Practical read

The repo does not have a single global controller. The active MuJoCo torque lane is the main development path; the CoppeliaSim/ROS lane is still a separate stack and should not be conflated with the MuJoCo residual-torque diagnostics.

## 2. Exact low-level torque law

### A. Cartesian impedance controller

File:
- `controller_core/x_axis_cartesian_impedance.py::XAxisCartesianImpedanceController.compute()`

The code implements a Cartesian impedance torque law plus posture damping and optional gravity compensation:

```text
Fx = kp_x * (x_des - x) + kd_x * (x_dot_des - x_dot)
Fy = kp_y * (y_des - y) - kd_y * y_dot
Fz = kp_z * (z_des - z) - kd_z * z_dot
M  = kp_rot * e_rot - kd_rot * omega
wrench = [Fx, Fy, Fz, Mx, My, Mz]

tau_task_nominal = J^T * (singular_scale * wrench)
tau_damping      = -kd_joint * qd
tau_posture      = kp_posture * (q_rest - q) - kd_posture * qd
tau_gravity      = g   # if provided in state

tau_nominal = tau_task_nominal + tau_damping + tau_posture + tau_gravity
tau_preclip = task_backtrack_scale * tau_nominal
tau = clip(tau_preclip, -tau_max_nm, +tau_max_nm)
```

Exact variable mapping:
- `kp_x`, `kd_x`, `kp_y`, `kd_y`, `kp_z`, `kd_z`, `kp_rot`, `kd_rot`, `kp_posture`, `kd_posture`, `kd_joint` come from `CartesianImpedanceConfig`
- `J` is the 6x6 body/world Jacobian from the robot state
- `e_rot` is `orientation_error_vec_wxyz(quat_ref, quat)`
- `omega` is end-effector angular velocity
- `qd` is joint velocity
- `q_rest` is the stored posture reference
- `g` is `state["gravity_torque"]` when present
- `tau_max_nm` is the per-joint torque limit vector
- `singular_scale` reduces wrench magnitude near Jacobian singularities
- `task_backtrack_scale` is the additional backtracking scale used to keep the nominal torque within headroom

What is not present:
- no mass matrix
- no Coriolis / centrifugal compensation
- no inverse dynamics
- no desired joint acceleration
- no torque rate limiting inside this controller class itself

### B. Torque QP controller

File:
- `controller_core/torque_task_qp.py::TorqueTaskQPController.compute()`

The QP controller uses the same task-space wrench and the same posture / damping / gravity terms:

```text
tau_des = J^T * (singular_scale * wrench) + tau_damping + tau_posture + gravity
```

Then it solves a box-constrained QP with torque bounds (and optionally velocity-implied torque bounds):

- `solve_box_qp(hessian, linear, tau_lo, tau_hi)`
- final torque is clipped to `[-tau_max_nm, +tau_max_nm]`

What is not present:
- no mass matrix
- no Coriolis / centrifugal compensation
- no inverse dynamics

### C. Legacy joint-PD family

File:
- `ros2_ws/src/ur5_x_axis_controller_ros/ur5_x_axis_controller_ros/controller_node.py::_joint_pd_torque()`

Law:

```text
tau_raw = kp * (q_ref - q) - kd * qd
tau = clip(tau_raw, -tau_limit, +tau_limit)
```

If gravity compensation is enabled in that path, gravity torque is added separately before final filtering / application.

### D. Active MuJoCo residual-torque lane

File:
- `simulation/ur5e_mujoco_torque.py::MujocoUR5eTorqueAdapter.apply_torque_components()`

The low-level applied torque is:

```text
tau_applied = tau_controller + tau_gravity
tau_shaped  = filter_and_clip(tau_applied)
```

The adapter logs:
- `tau_controller`
- `tau_controller_clipped`
- `tau_gravity`
- `tau_applied`
- `tau_applied_clipped`
- `tau_filtered`
- `tau_clipped`
- `tau_controller_clip_fraction`
- `tau_applied_clip_fraction`
- `gravity_mode`
- `gravity_compensation_active`

The actual write to MuJoCo is done by the runner:
- `tools/ur5e_mujoco_torque_experiments.py` writes `data.ctrl[:6] = tau`

### E. Torque filtering / clipping

File:
- `controller_core/filters.py::TorqueCommandFilter.apply_with_diagnostics()`

The filter is a first-order low-pass followed by a per-joint slew rate limit:

- low-pass parameter: `lowpass_alpha`
- rate limit parameter: `rate_limit_nm_per_sec`
- per-step maximum delta: `delta_max = rate_limit_nm_per_sec * dt`

So the practical output torque is not just the raw controller output; it is low-pass filtered and slew-rate limited before final clipping.

## 3. Is gravity compensation present and correct?

### Confirmed: gravity compensation is present

Yes, in both the MuJoCo lane and the CoppeliaSim/ROS lane.

MuJoCo helper:
- `mujoco_ur5e_tools.py::compute_gravity_torque()`

This helper:
- copies the current `q`
- sets `qvel = 0`
- sets `qacc = 0`
- zeros applied forces
- runs MuJoCo forward and inverse dynamics on a scratch `MjData`
- returns `scratch.qfrc_bias[dof_adr]` for each joint

That means the gravity term is model-based and computed from the intended static state, not from a hardcoded torque table.

CoppeliaSim live helper:
- `simulation/run_coppeliasim_x_axis_headless.py::build_mujoco_gravity_estimator()`
- `simulation/run_coppeliasim_x_axis_headless.py::compute_mujoco_gravity_bias()`
- `simulation/run_coppeliasim_x_axis_headless.py::coppelia_gravity_feedforward()`

That path also uses MuJoCo bias torque as the feedforward source.

### Confirmed: gravity compensation is added before final application

In the active MuJoCo residual lane:
- `tau_applied = tau_controller + tau_gravity`
- then `shape_torque()` filters and clips
- then the result is written to `data.ctrl`

In the CoppeliaSim runner:
- gravity feedforward is added before the final `adapter.apply_torque(...)`

### Confirmed: sign and static-hold behavior were audited

The repository contains a dedicated audit tool:
- `tools/audit_ur5e_mujoco_gravity_torque.py`

It compares:
- `raw_zero`
- `plus_gravity_comp`
- `minus_gravity_comp`
- `direct_bias_force`
- `inverse_dynamics_hold`

The current MuJoCo audit and regression tests indicate:
- the correct gravity sign for the MuJoCo residual-torque lane is the `plus_gravity_comp` variant
- the static-state gravity computation is consistent with the bias-force helper
- hold diagnostics exist and are validated by tests

Relevant tests:
- `tests/test_ur5e_mujoco_torque.py::test_gravity_torque_utility_returns_finite_vector`
- `tests/test_ur5e_mujoco_torque.py::test_gravity_comp_hold_reports_applied_gravity_torque`
- `tests/test_ur5e_mujoco_torque.py::test_gravity_torque_audit_helpers_are_consistent`
- `tests/test_ur5e_mujoco_torque.py::test_gravity_torque_audit_cli_smoke`

### Important nuance

The CoppeliaSim configs still carry a `gravity_compensation_sign` setting because that path can be calibrated independently. The MuJoCo audit evidence says the MuJoCo residual-torque lane is now sign-correct; I did not find evidence that the current free-space MuJoCo failures are caused by a gravity sign bug.

## 4. Are joint order, signs, and units consistent?

### Joint order

Canonical UR5 joint order is consistent across the code inspected:

- `shoulder_pan_joint`
- `shoulder_lift_joint`
- `elbow_joint`
- `wrist_1_joint`
- `wrist_2_joint`
- `wrist_3_joint`

Files that encode this order:
- `controller_core/x_axis_cartesian_impedance.py`
- `controller_core/torque_task_qp.py`
- `mujoco_ur5e_tools.py`
- `simulation/ur5e_mujoco_torque.py`
- `ros2_ws/src/ur5_x_axis_controller_ros/ur5_x_axis_controller_ros/messages.py`
- `ros2_ws/src/ur5_x_axis_controller_ros/ur5_x_axis_controller_ros/coppeliasim_adapter.py`

Evidence of reordering:
- `ros2_ws/src/ur5_x_axis_controller_ros/ur5_x_axis_controller_ros/messages.py::joint_state_to_arrays()`
  - reorders ROS `JointState` messages into the canonical order

### Signs and frames

Confirmed:
- joint positions are in radians
- joint velocities are in rad/s
- torques are in Nm
- world-frame positions / velocities are used for the Cartesian target and state
- quaternion convention is `[w, x, y, z]`

Relevant helpers:
- `controller_core/kinematics_utils.py::orientation_error_vec_wxyz()`
- `controller_core/kinematics_utils.py::world_linear_jacobian()`
- `ros2_ws/src/ur5_x_axis_controller_ros/ur5_x_axis_controller_ros/messages.py::pose_to_pos_quat()`

### End-effector site / frame

The MuJoCo torque lane validates:
- `site_name="attachment_site"`

The CoppeliaSim adapter can mirror that convention with:
- `task_frame.mode = "mujoco_attachment_dummy"`
- created dummy name: `real_cartpole_mujoco_attachment_site`

This is important because the target profiles and metrics are world-X transport tasks, not a local wrist-frame task.

### Evidence for consistency

I did not find evidence of a joint-index remap bug in the code inspected.
I also did not find evidence of a radians-versus-degrees mismatch.
I did not find evidence of a world-frame / local-frame mismatch in the controller / metric pipeline.

## 5. Why are safety limits being hit?

### Exact safety limits

File:
- `controller_core/safety.py::ImpedanceSafetyConfig`

Default guard thresholds include:
- `max_abs_y_drift_m = 0.03`
- `max_abs_z_drift_m = 0.03`
- `max_abs_orthogonal_drift_m = 0.03`
- `max_orientation_error_rad = 0.25`
- `max_joint_velocity_radps = 1.5`
- `max_x_error_growth_steps = 100`
- `max_axis_error_growth_steps = 100`
- `emergency_stop_on_nan = True`
- `emergency_stop_on_joint_limit = True`

### Where they are enforced

- `controller_core/safety.py::ImpedanceSafetyMonitor.check()`
- `simulation/run_coppeliasim_x_axis_headless.py` calls safety before applying torque
- `simulation/ur5e_mujoco_torque.py::MujocoUR5eTorqueAdapter.apply_torque_components()` checks safety and logs the reason
- `tools/ur5e_mujoco_torque_experiments.py` classifies validity using the transport metrics after each rollout

### What the logs say

Representative CoppeliaSim failure:
- `outputs/control_runs/coppelia_lqr_ratefix_coppeliasim_x_axis_offscreen_capture_summary.json`
- failure reason: `|qd| > 1.5 rad/s`
- `max_abs_tau_nm` was tiny (`~8e-4 Nm`)
- `torque_saturation_ok = True`

Another representative CoppeliaSim failure:
- `outputs/control_runs/coppelia_lqr_smoke16_coppeliasim_x_axis_offscreen_capture_summary.json`
- failure reasons included:
  - joint limit violation
  - `|qd| > 1.5 rad/s`
  - `|Y-Y0| > 0.03 m`
  - `|Z-Z0| > 0.03 m`
  - orientation error above threshold
- `max_abs_tau_nm = 0.0`
- `torque_saturation_ok = True`

Representative MuJoCo residual-torque failures in the recent transport tuning:
- dominant failure reasons were `move_phase_target_tracking`, `hold_phase_target_tracking`, `hold_phase_incomplete`, and `|axis_error| grew for 100 consecutive steps`
- torque saturation was not the dominant limiter
- the best valid runs had `torque_saturation_percentage = 0.0`

### Earliest abnormal signal

The earliest recurring abnormal signal is not torque saturation. It is usually one of:
- joint velocity growth (`|qd|`)
- target/axis error growth
- hold-phase drift after the move settles

That points to stability / damping / model mismatch / safety-envelope interaction, not a raw torque-capacity problem.

## 6. Is the impedance controller too primitive?

### Confirmed from code

Yes. The current controller is basically:
- Cartesian impedance via `J^T`
- plus posture PD
- plus joint damping
- plus optional gravity compensation
- plus Jacobian singularity scaling / backtracking
- plus hard torque clipping and torque filtering in the caller

It does **not** use:
- the mass matrix
- Coriolis / centrifugal compensation
- inverse dynamics
- desired joint or task accelerations

### Why that matters

For a 6-DOF torque plant like UR5e, a pure Cartesian PD/J^T controller can work in a small neighborhood, but it is fragile when:
- gravity is substantial
- the configuration is near a Jacobian conditioning problem
- transport has to move and then hold
- safety limits are tight

### Most likely missing term

The first missing dynamics term that would matter is likely:
- Coriolis / inertia-aware compensation

However, that is still a hypothesis. The code inspection alone confirms only that the controller is simple, not that a specific missing term is the sole failure cause.

### Hypothesis vs confirmed

Confirmed:
- the controller is structurally simple
- it lacks full model-based dynamics

Hypothesis:
- the missing dynamics are a major reason that longer move-and-hold transport fails

## 7. What would Pinocchio help diagnose?

### Is Pinocchio already used in this repo?

I did not find Pinocchio imports or references in the relevant controller / simulation / tools / ROS paths.

### Is there a URDF/model file suitable for Pinocchio?

Yes, there is at least one likely source:
- `ros2_ws/src/real_cartpole_description/urdf/ur5e_cartpole.urdf.xacro`

There is also a MuJoCo conversion script:
- `scripts/convert_ur5e_urdf_to_mjcf.py`

I did **not** verify a generated URDF export from the xacro in this pass, but the xacro source is a plausible Pinocchio input once expanded.

### What joint state data is available?

Available already:
- `q`
- `qd`
- canonical joint order
- end-effector pose / twist
- target pose / twist

### What could Pinocchio compute?

In principle, yes:
- gravity torque `rnea(q, 0, 0)`
- inverse dynamics `rnea(q, dq, ddq)`
- mass matrix `crba`
- Jacobians

### Most useful comparison

The diagnostic comparison that would be most useful is:

1. controller torque command
2. Pinocchio gravity torque `rnea(q, 0, 0)`
3. optional Pinocchio inverse dynamics `rnea(q, dq, ddq)`
4. simulator / model-reported torque (`qfrc_bias`, `qfrc_actuator`)
5. safety torque limits

### Is it feasible now?

Feasible in principle, yes.
But because the repository already has a working MuJoCo gravity audit and hold diagnostics, I would do one more simpler simulation-only diagnostic before introducing Pinocchio.

## 8. What safe diagnostic experiments already exist?

### Safe simulation-only

These are safe to use without touching hardware:

- `tools/ur5e_mujoco_torque_experiments.py`
  - simulation-only MuJoCo torque experiments
  - includes `gravity-comp-hold`, `gravity-comp-hold-long`, `residual-impedance-hold`, `controller-rollout`, `x-transport-minjerk`, `zero-torque`, `safety-clipping`
- `tools/ur5e_move_hold_transport.py`
  - move-and-hold transport sweeps in MuJoCo
- `tools/tune_ur5e_impedance_transport.py`
  - staged impedance tuning in MuJoCo
- `tools/tune_ur5e_residual_impedance_transport.py`
  - staged residual-impedance tuning in MuJoCo
- `tools/ur5e_x_frame_envelope.py`
  - valid-transport envelope study
- `tools/audit_ur5e_mujoco_gravity_torque.py`
  - gravity sign / actuator mapping / clamping / hold quality audit
- `simulation/run_coppeliasim_x_axis_headless.py`
  - CoppeliaSim-only controller runner
- `simulation/run_coppelia_torque_diagnostics_smoke.py`
  - CoppeliaSim-only torque diagnostics smoke ladder

### Potentially hardware-touching / caution

These should be treated as hardware-oriented and not run in this diagnostic task:

- `tools/ur5e_hardware_smoke_test.py`
- `tools/ur5e_servoj_zero_hold.py`
- `tools/ur5e_direct_torque_probe.py`
- `ros2_ws/src/ur5_x_axis_controller_ros/launch/run_ur5e_hardware_pipeline.launch.py`
- `ros2_ws/src/ur5_x_axis_controller_ros/ur5_x_axis_controller_ros/ur5e_hardware_pipeline_node.py`

### No separate current LQR torque test

I saw historical CoppeliaSim `lqr_*` output folders under `outputs/control_runs/`, but I did not identify a current first-class LQR torque-controller script in the MuJoCo torque lane.

## 9. Most likely failure causes

Ranked from most likely to least likely, based on the code and the representative logs.

### 1) Controller is structurally too primitive / missing dynamics compensation

- Evidence:
  - Cartesian impedance is J^T PD plus posture damping plus optional gravity
  - no mass matrix
  - no Coriolis compensation
  - no inverse dynamics
  - long-hold / move-hold failures persist even when torque saturation is not the limiter
- Confidence: high
- Safe verification:
  - compare short gravity-hold traces against the gravity audit
  - inspect whether drift appears after the move settles while torques remain below limits

### 2) Safety limits are tripping before the controller can settle

- Evidence:
  - multiple representative failures stop on `|qd| > 1.5 rad/s`
  - drift and axis-error-growth guards appear in the failures
  - torque saturation is often not the first failure mode
- Confidence: high
- Safe verification:
  - run zero-command / gravity-comp hold on the same start pose
  - log the first time `qd`, Y/Z drift, and axis error cross threshold

### 3) Gain / damping design is not matched to the robot dynamics

- Evidence:
  - the controller is a simple impedance law with fixed gains
  - move-hold runs show a short clean transport envelope but no longer stable hold
- Confidence: medium
- Safe verification:
  - narrow gain sweeps around the current best impedance gains
  - separate move-phase and hold-phase metrics

### 4) Trajectory / hold interaction is causing the failure

- Evidence:
  - move-hold diagnostics show cases where the move phase is acceptable but the hold phase drifts
  - the failure mode changes when move duration is shortened
- Confidence: medium
- Safe verification:
  - hold the current pose with zero motion
  - use a move-and-hold profile that freezes the target after arrival

### 5) Gravity compensation bug or sign mismatch

- Evidence against:
  - MuJoCo gravity torque audit exists
  - static-hold gravity torque helper is consistent with the bias torque helper
  - recent MuJoCo audit indicates the `plus_gravity_comp` sign is correct
- Evidence for:
  - the CoppeliaSim config still exposes a gravity sign parameter and a different runtime sign convention could exist there
- Confidence: low for the current MuJoCo lane
- Safe verification:
  - compare static gravity torque, live bias torque, and controller torque on a hold pose

### 6) Wrong joint order / sign / units

- Evidence against:
  - canonical UR5 joint order is repeated consistently
  - radians / Nm / world-frame pose data are used consistently
  - no evidence of a remapping bug was found
- Confidence: low
- Safe verification:
  - compare single-joint torque perturbations against the actuator mapping audit

### 7) Torque saturation causing instability

- Evidence against:
  - several representative failures have very low torque fractions
  - the best valid transport runs had zero saturation
- Confidence: low
- Safe verification:
  - inspect `tau_controller_clip_fraction`, `tau_applied_clip_fraction`, and `torque_saturation_percentage`

### 8) Degrees / radians mismatch

- Evidence against:
  - the code and configs use radians and rad/s consistently
- Confidence: low
- Safe verification:
  - a units mismatch would show up immediately in the hold audits; it did not

### 9) Simulator / model mismatch

- Evidence:
  - possible, but not yet the leading explanation
  - the MuJoCo audit shows the torque plumbing itself is consistent
- Confidence: medium-low
- Safe verification:
  - compare controller torque to MuJoCo `qfrc_bias` / `qfrc_actuator`
  - later, compare to Pinocchio if needed

### 10) Initial pose too close to a guardrail

- Evidence:
  - some failures are sensitive to the start pose and move duration
- Confidence: medium-low
- Safe verification:
  - reuse the same start pose in a zero-command gravity hold and in a very small move-hold command

## 10. Safest next diagnostic steps

These stay simulation-only and do not change the controller.

### Step 1: Run a short static gravity-hold comparison on the transport start pose

- Goal:
  - verify whether the start pose itself is stable under zero residual torque plus gravity compensation
- File / script:
  - `tools/audit_ur5e_mujoco_gravity_torque.py`
  - or `tools/ur5e_mujoco_torque_experiments.py --mode gravity-comp-hold`
- Log:
  - `q`, `qd`, `ee_pos`, `ee_quat`
  - `tau_controller`, `tau_gravity`, `tau_applied`
  - max / mean absolute torque
  - drift in X / Y / Z / orientation
  - failure reason and first failure time
- Expected outcome if the issue is gravity / model mismatch:
  - drift appears even with zero residual command, or the gravity torque sign / magnitude does not hold the pose
- Why safe:
  - zero residual torque, simulation only, no hardware, no controller rewrite

### Step 2: Compare move-phase and hold-phase traces separately using the existing move-hold runner

- Goal:
  - determine whether the failure is move tracking or hold stability
- File / script:
  - `tools/ur5e_move_hold_transport.py`
- Log:
  - `move_phase_final_x_error_m`
  - `hold_phase_final_x_error_m`
  - `hold_phase_x_drift_from_hold_start_m`
  - `max_abs_y_drift_m`
  - `max_abs_z_drift_m`
  - `max_abs_orientation_error_rad`
  - `max_abs_qd_radps`
  - `tau_controller`, `tau_gravity`, `tau_applied`
  - `torque_saturation_percentage`
- Expected outcome if the issue is damping / hold instability:
  - move phase is acceptable, then drift begins during hold with no torque saturation
- Why safe:
  - simulation only, no hardware, no runtime changes

### Step 3: Compare controller torque against model torque terms on one short hold run

- Goal:
  - determine whether the controller output is consistent with the model’s gravity / bias load
- File / script:
  - `tools/audit_ur5e_mujoco_gravity_torque.py`
  - plus the MuJoCo trace fields already emitted by `simulation/ur5e_mujoco_torque.py`
- Log:
  - controller torque command
  - MuJoCo gravity torque
  - MuJoCo bias torque
  - `qfrc_actuator` if available
  - clipping fractions
  - safety reason
- Expected outcome if the issue is missing dynamics compensation:
  - the controller torque remains small relative to the model load, or the sign/magnitude of the residual torque does not stabilize the hold
- Why safe:
  - read-only diagnostic, no new controller behavior, no hardware

## Final assessment

### Confirmed facts

- The active MuJoCo lane is true torque, not a hidden position servo.
- Gravity compensation is present and now audited in the MuJoCo lane.
- Joint order and units are consistent across the code inspected.
- The torque law is a Cartesian impedance / J^T mapping plus posture damping and optional gravity.
- The controller does not include mass-matrix, Coriolis, or inverse-dynamics compensation.
- Safety limits are real and are often hit before torque saturation becomes the bottleneck.

### Main hypothesis

The current failure mode is more likely due to a structurally simple low-level torque controller plus tight safety envelopes than to a high-level target-generation bug.

### What is *not* the leading explanation

- collision geometry contamination
- contact saturation
- joint-order mismatch
- radians/degrees mismatch
- raw torque capacity

Those are lower-probability explanations given the code and logs inspected here.

