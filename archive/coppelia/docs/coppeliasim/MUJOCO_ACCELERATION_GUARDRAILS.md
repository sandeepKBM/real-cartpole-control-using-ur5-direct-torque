# MuJoCo Acceleration Controller Guardrails

This is the behavior contract CoppeliaSim must satisfy before we call the
acceleration transport port correct.

## What MuJoCo Actually Does

The MuJoCo acceleration controller is not a direct torque controller. It is a
position-servo controller:

- input is one scalar signed world-X acceleration, `a_x_cmd`;
- the controller clips that acceleration and integrates an internal
  `v_x_state`;
- weighted differential IK maps desired world-X velocity, fixed world-Z, and
  fixed tool orientation to joint position setpoints;
- `shoulder_pan_joint` is locked to the start pose;
- MuJoCo's position actuators then realize those joint setpoints;
- estimated servo torque is logged only as a guardrail.

The recommended acceleration run starts from
`outputs/control_runs/fixed_z_x_transport_firstpass_z0.540_seed1.json`, not from
the high shoulder-side origin. That tucked fixed-Z pose exists because the
shoulder-side origin is nearly extended and has strong X/Z coupling.

## Required Invariants

Before transport starts, Coppelia must first pass origin acquisition:

- from a randomized start state, recover the origin pose;
- the end effector/tool frame must face the same reference direction as the
  established MuJoCo target orientation;
- the final origin position error must be below 15 mm;
- the final orientation error must be below 3 deg;
- the base must remain on the ground.
- the reachable Coppelia world-X interval must be scanned at the requested
  height and reference orientation before commanding acceleration.

This first stage is now implemented and verified in:

```text
simulation/ur5_origin_acquisition_video_addon.lua
simulation/launch_coppeliasim_origin_acquisition_video.sh
```

Latest verified Coppelia summary at `EE_TARGET_Z_M=0.65`:

```text
success=true
base_on_ground=true
requested_target_dx_m=0.060
target_dx_m=0.020
target_dx_was_clamped_by_range_scan=true
x_reachable_min_m=-0.115575
x_reachable_max_m=-0.075575
final_origin_position_error_m=0.000303
acceleration_final_position_error_m=0.000508
acceleration_final_orientation_error_deg=2.784
```

That path is a simulator-side Lua/IK position-setpoint proof, not direct torque
control. It exists to lock down the required pre-transport pose/orientation
behavior and the local reachable X interval before acceleration transport is
judged. If the requested X displacement is outside the scanned interval, the
path clamps to the reachable displacement and reports that explicitly.

Only after that should acceleration transport be tested. A Coppelia transport
run is valid only if all of these are true:

- the UR5 base stays on the ground; do not lift the whole model to make the
  video look better;
- the same end-effector/task frame convention is used as MuJoCo's
  `attachment_site`;
- the reference orientation is fixed for the whole run;
- world Y drift remains small;
- world Z drift remains small;
- measured world-X displacement matches the commanded signed X displacement;
- the motion profile reports actual peak speed, acceleration, and joint speed;
- a failed IK/path solve is reported as failure, not as a successful video.

## Current Coppelia Gap

The official Coppelia `UR5.ttm` frame/axis convention does not map directly to
the MuJoCo UR5e fixed-Z transport joint vector. Reusing MuJoCo's transport
`start_q` in Coppelia with the base on the ground currently produces little
usable world-X motion under the fixed-Y/Z/orientation constraints.

Important frame note from the current Lua implementation:

- the current Coppelia origin/acceleration add-on moves the resolved EE proxy
  along **Coppelia world X**, not along the EE object's local X axis;
- it builds targets as `{target_x + dx, target_y, target_z}` after reading the
  EE pose in `sim.handle_world`;
- this means world Y and world Z are held fixed while the reference orientation
  is held fixed;
- it does **not** compute `target_position + dx * ee_local_x_axis`;
- this matches the intended MuJoCo fixed-Z transport direction only if the
  controlled Coppelia task frame is the MuJoCo-equivalent `attachment_site`.

Clean MuJoCo comparison showed these missing pieces in the Coppelia port:

- Coppelia currently controls `/UR5/UR5_connection` or joint 6 as an EE proxy;
  MuJoCo controls custom task frames: `forearm_tip_site` for origin acquisition
  and `attachment_site` for tool transport/orientation.
- Coppelia derives the target orientation from the Coppelia pose at `Q_ORIGIN`;
  MuJoCo uses the explicit `TARGET_SITE_ROTATION_WORLD` matrix.
- Coppelia's default origin pose is the fixed-Z transport `start_q`; MuJoCo's
  first origin stage targets `ACTIVE_ORIGIN_Q = [0, -pi/2, 0, -pi/2, 0, 0]`.
- Coppelia pre-solves IK waypoints and plays them with a trapezoidal profile;
  MuJoCo runs an online controller: scalar world-X acceleration -> clipped
  velocity state -> differential IK -> position-servo setpoints.
- Coppelia currently solves all six joints; MuJoCo locks shoulder pan and solves
  the reduced chain.
- Coppelia direct `sim.setJointPosition` bypasses the servo dynamics/force-limit
  guardrails that MuJoCo logs through its position actuators.

Therefore the tiny scanned Coppelia X span is not yet evidence of the real UR5
workspace. It is evidence of the current frame/orientation/controller mismatch.
The next correct fix is to create/resolve Coppelia task dummies equivalent to
MuJoCo's `forearm_tip_site` and `attachment_site`, then redo the fixed-Z range
and acceleration tests with MuJoCo's target rotation and reduced-DOF controller
semantics.

The Coppelia fixed-Z video path now writes explicit guardrail fields:

- `base_on_ground`
- `x_tracking_ok`
- `single_axis_y_ok`
- `fixed_z_ok`
- `orientation_ok`
- `success`
- `failure_reasons`

Until `success` is true with `base_on_ground=true`, the Coppelia fixed-Z
acceleration video is diagnostic output, not a completed MuJoCo-equivalent port.
