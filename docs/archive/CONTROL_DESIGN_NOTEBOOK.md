# Control Reference

## Scope

This file is an implementation reference for the current `real_Cartpole` control stack.

It is not a tutorial. It records:

- what has been implemented
- what has been measured (where artifacts exist)
- which files are authoritative
- which pieces are legacy or alternate

## Active Objective

Current active objective (simulation):

- drive the arm from a reproducible random start near the canonical pose to a **shoulder-side vertical-face** reference
- keep **`shoulder_pan_joint` fixed** at the target value (typically `0`)
- hold **`forearm_tip_site`** world position at the pose implied by the target joint vector (proximal / forearm-origin task)
- hold **`attachment_site`** to a **fixed world-frame tool orientation** (blue side-panel / “face normal” convention; see `controller.py` comments)
- optional **`wrist_2_joint`** bias via `--wrist2-offset-deg` without changing the origin sites or the orientation target
- produce reproducible rendered experiments with saved JSON metrics

Deferred objective:

- real cartpole stabilization
- learned control / RL
- planner-based motion generation

Those are not part of the active simulation path.

## Authoritative Files

Simulation control:

- [`controller.py`](/common/users/ss5772/real_Cartpole/simulation/controller.py)
- [`run_origin_stabilization.py`](/common/users/ss5772/real_Cartpole/simulation/run_origin_stabilization.py)
- [`probe_origin_pose.py`](/common/users/ss5772/real_Cartpole/simulation/probe_origin_pose.py)

MuJoCo model:

- [`ur5e.xml`](/common/users/ss5772/real_Cartpole/mujoco_menagerie/universal_robots_ur5e/ur5e.xml)
- [`scene_ur5e_cartpole.xml`](/common/users/ss5772/real_Cartpole/mujoco_menagerie/universal_robots_ur5e/scene_ur5e_cartpole.xml)

ROS control scaffold:

- [`origin_hold_controller.py`](/common/users/ss5772/real_Cartpole/ros2_ws/src/real_cartpole_control/real_cartpole_control/origin_hold_controller.py)
- [`origin_hold.launch.py`](/common/users/ss5772/real_Cartpole/ros2_ws/src/real_cartpole_control/launch/origin_hold.launch.py)
- [`origin_hold.yaml`](/common/users/ss5772/real_Cartpole/ros2_ws/src/real_cartpole_control/config/origin_hold.yaml)

## Constrained world-X envelope (no cartpole)

For **`attachment_site` world X** with **tool orientation** fixed to `TARGET_SITE_ROTATION_WORLD` and **`shoulder_pan_joint` locked at 0**, a numerical kinematic study (SLSQP over joint limits) gives a total span on the order of **~1.8 m** for the current menagerie scene—see [`CONSTRAINED_EE_X_WORKSPACE.md`](/common/users/ss5772/real_Cartpole/CONSTRAINED_EE_X_WORKSPACE.md) and [`simulation/study_constrained_ee_x_workspace.py`](/common/users/ss5772/real_Cartpole/simulation/study_constrained_ee_x_workspace.py). At the canonical origin, **per-joint “mm per degree”** only makes sense for **raw** Jacobian entries (orientation not held) or for a **joint mix** in the rotational null space; the note explains why isolated per-joint X motion under orientation lock is ill-posed.

## Plant Facts

MuJoCo uses `general` actuators with affine bias. From [`ur5e.xml`](/common/users/ss5772/real_Cartpole/mujoco_menagerie/universal_robots_ur5e/ur5e.xml):

**`size3` (shoulder pan, shoulder lift, elbow)**

```text
gainprm = 2000
biasprm = 0 -2000 -400
```

Operationally (per MuJoCo’s affine general actuator):

```text
tau ~= 2000 * (ctrl - q) - 400 * qdot
```

**`size1` (wrist 1–3)**

```text
gainprm = 500
biasprm = 0 -500 -100
```

```text
tau ~= 500 * (ctrl - q) - 100 * qdot
```

The Python layer commands **joint setpoints** (`ctrl`) each step; it is an outer-loop policy on top of these built-in servos, not a raw torque controller.

## Origin and Tool-Frame Convention

**Active joint target** (`ACTIVE_ORIGIN_Q` in code) is `SHOULDER_SIDE_FACE_DIRECTION_ORIGIN_Q`:

```text
[ 0, -pi/2, 0, -pi/2, 0, 0 ] rad
```

**Legacy** peak-height joint vector (kept as `LEGACY_PEAK_HEIGHT_Q` for comparison and probe seeds):

```text
[ 0.0, -1.570784, -0.000007, -2.356201, -1.570784, 0.0 ] rad
```

At `ACTIVE_ORIGIN_Q`, MuJoCo forward kinematics give **`forearm_tip_site`** near `[0, -0.007, 0.98]` m and **`attachment_site`** position near `[0, -0.234, 1.08]` m in a typical load of the menagerie scene (exact numbers depend on the loaded `scene_*.xml`); re-measure with `probe_origin_pose.py` if the scene or XML changes.

**World-frame tool orientation target** (`TARGET_SITE_ROTATION_WORLD`):

- site **x** → world **-X**
- site **y** → world **-Z**
- site **z** → world **-Y** (face normal / panel convention as documented in `controller.py`)

`probe_origin_pose.py` can search origins under modes such as `vertical-face`, `peak-height`, etc., and optionally write JSON (`peak_q_rad` / geometry) for `--target-q-json` overrides on the runner.

## Controllers in `controller.py`

### Primary (wired in `run_origin_stabilization.py`): `split_forearm_origin_face_controller`

Design:

1. **`shoulder_pan_joint`**: not driven by the incremental policy; **`ctrl[0] = q_target[0]`** each step (fixed base rotation).
2. **Forearm-origin task**: Jacobian-transpose correction on **`forearm_tip_site`** position error using `origin_jacobian_pos` (columns mapped to the six actuated joints). Mask emphasizes **`shoulder_lift`** and **`elbow`**; wrist columns are not used for the origin position task in this path.
3. **Tool-face task**: orientation correction via `tool_target_orientation_omega` and **`attachment_site`** rotational Jacobian, masked toward the wrist joints.
4. Optional **`wrist_posture_target`**: small posture bias on the wrist joints when the runner passes a modified target (e.g. `wrist_2` offset).

Implemented gains (see source for exact arrays):

- Forearm posture/damping on indices 1–2 only for the baseline posture term; wrist posture term is zero unless `wrist_posture_target` is set.
- Origin guidance gain `2.2` with joint mask `[0, 1.35, 1.55, 0, 0, 0]`.
- Face orientation guidance `0.50` with mask `[0, 0, 0, 1.25, 1.55, 1.25]`.
- Hard per-step limits: `[0.0, 0.0048, 0.0052, 0.0058, 0.0030, 0.0026]` rad (index 0 clamped to 0 delta because pan is set explicitly).

### Alternate (not used by the current runner): `reverse_nested_x_servo_controller`

Reserved outer-loop policy that:

- prioritizes **wrist_1 → elbow → shoulder_lift** with gating
- adds a small **world-x** task on **`attachment_site`** via the translational Jacobian row for x
- adds **world-frame orientation** guidance via `tool_target_orientation_omega` and rotational Jacobian

Current numeric parameters live only in `controller.py` (posture/damping, priority scales, `max_delta`, orientation weights). To use it in experiments, the runner would need to be switched back to call this function and to pass the same Jacobian/site arguments it used historically.

### Shared helpers

- `sample_random_configuration(...)`: random start near a center pose (default center `ACTIVE_ORIGIN_Q`).
- `tool_target_orientation_omega(...)`: builds an angular “error” vector from dot-aligned tool axes vs. the fixed world target.

## Random Start Policy

Implemented in `sample_random_configuration` and invoked from `run_origin_stabilization.py` with:

- `center = target_q` (default `ACTIVE_ORIGIN_Q`, or from `--target-q-json`)
- `span_scale` default `0.35` (CLI `--span-scale`)
- **`min_distance=0.9`** rad (runner hard-codes this; the function default alone is `0.8` if called elsewhere)
- `shoulder_pan` of the sampled start is **overwritten** to match `target_q[0]`

## Runner Behavior

Primary runner: [`run_origin_stabilization.py`](/common/users/ss5772/real_Cartpole/simulation/run_origin_stabilization.py).

Responsibilities:

- load **`scene_ur5e_cartpole.xml`** if it exists, else **`scene.xml`**
- sample a seeded start
- apply **`split_forearm_origin_face_controller`** every simulation sub-step
- render a video and write JSON under `outputs/control_runs/`

Default output stem:

```text
ur5e_forearm_origin_shoulder_side_face_seed{seed}
```

Appends e.g. `_wrist2_offset{deg}deg` or `_{fps}fps` when those options differ from defaults.

### Metrics in the JSON summary

Key fields (non-exhaustive):

- `controller_name`, `origin_name`, `scene_xml`, `seed`, `duration_s`, `fps`
- `origin_target_q`, `random_start_q`, `final_q`
- `target_origin_site_world`, `final_origin_site_world`, `final_origin_site_error_m`
- `target_tool_site_world`, `final_tool_site_world`, `final_tool_site_position_error_m`
- `final_forearm_origin_joint_error_rad` — max abs error over `FOREARM_ORIGIN_INDICES` (shoulder lift, elbow)
- `final_tool_face_joint_error_rad` — max abs error over `TOOL_FACE_INDICES` (wrist 1–3)
- `final_tool_orientation_error_deg` — rotation mismatch vs. `target_tool_rotation_world`
- `final_base_pan_error_rad`, `final_max_joint_error_rad`
- `origin_settle_time_s`, `full_joint_settle_time_s`, `orientation_settle_time_s`, `settle_time_s` (alias for origin settle)
- `success`, `success_full_joint_match`

**Success predicates** (see `run_origin_stabilization.py`):

- **`success`**: forearm-origin joint error ≤ `0.03` rad, tool orientation error ≤ `3` deg, origin site error ≤ `0.015` m, and both origin- and orientation-settle times computed (hold ~0.5 s within tolerances on the traces).
- **`success_full_joint_match`**: max joint error ≤ `0.03` rad, same orientation and origin tolerances, and full-joint settle time computed.

Settle-time traces use `compute_settle_time` with tolerances documented in source (origin vs. orientation vs. full-joint pairs).

### Video overlay (current)

Includes forearm/tool XYZ, errors, tool RPY, face orientation error, forearm vs. wrist joint-error splits, base pan, full joint error, and link alignment angles vs. the target posture.

## Verified Experimental Artifacts

Historical runs referenced older filenames (`ur5e_origin_stabilization_*`, `fixedbase_detailed`, etc.). The **current** controller naming and JSON schema differ.

A **current-schema** saved summary on this machine (re-run the command below to regenerate video if missing):

- [`ur5e_shoulder_side_face_direction_seed7_50fps.json`](/common/users/ss5772/real_Cartpole/outputs/control_runs/ur5e_shoulder_side_face_direction_seed7_50fps.json)

Recorded in that file (seed 7, 50 fps, 10 s):

- `final_forearm_origin_joint_error_rad`: `0.000587` (approx.)
- `final_tool_face_joint_error_rad`: `0.011587` (approx.)
- `final_tool_orientation_error_deg`: `0.00154` (approx.)
- `final_origin_site_error_m`: `4.9e-7` (approx.)
- `final_tool_site_position_error_m`: `0.00116` (approx.)
- `origin_settle_time_s` / `orientation_settle_time_s`: `1.8` s
- `full_joint_settle_time_s`: `8.32` s
- `success`: `true`
- `success_full_joint_match`: `true`

Older experiments (uniform shaper, reverse-nested-only, fixed-base detailed) remain valid **as history** but do not describe the active pipeline; treat their metrics and filenames as superseded unless you restore that runner configuration.

## ROS Status

Unchanged in intent: scaffold only.

Implemented / smoke-tested:

- [`real_cartpole_control`](/common/users/ss5772/real_Cartpole/ros2_ws/src/real_cartpole_control)
- bounded incremental commands from `/joint_states`

Not implemented:

- planner-driven demos on hardware
- real robot validation of this origin law
- cartpole-aware ROS controller
- torque-level real-arm control

## Cleanup Status

Legacy sway / notebook / MoveIt demo paths have been pruned from the active control story.

Remaining primary simulation entrypoints:

- [`controller.py`](/common/users/ss5772/real_Cartpole/simulation/controller.py)
- [`probe_origin_pose.py`](/common/users/ss5772/real_Cartpole/simulation/probe_origin_pose.py)
- [`run_origin_stabilization.py`](/common/users/ss5772/real_Cartpole/simulation/run_origin_stabilization.py)

## Current Run Command

```bash
cd /common/users/ss5772/real_Cartpole
source /common/users/ss5772/miniforge3/etc/profile.d/conda.sh
conda activate mujoco_ur5e
xvfb-run -a python simulation/run_origin_stabilization.py --seed 7
```

Optional: `--target-q-json path/to/probe_report.json` to override `origin_target_q`, or `--wrist2-offset-deg` for a wrist bias experiment.

## Next Steps

1. **Limb angle vs. X motion:** extend [`study_constrained_ee_x_workspace.py`](/common/users/ss5772/real_Cartpole/simulation/study_constrained_ee_x_workspace.py) with batched pose sampling if you need statistics across configuration space; see notes in [`CONSTRAINED_EE_X_WORKSPACE.md`](/common/users/ss5772/real_Cartpole/CONSTRAINED_EE_X_WORKSPACE.md).
2. Regenerate and commit reference **video** artifacts if the repo should ship them (JSON may exist without `.mp4` after cleanup).
3. Align top-level README / plan docs with this notebook so they only describe the **split forearm + face** workflow.
4. **ROS:** [`ros2_ws/ROS_SYSTEM_FLOWCHART.md`](/common/users/ss5772/real_Cartpole/ros2_ws/ROS_SYSTEM_FLOWCHART.md). Then: hardware bring-up, **cartpole-aware** simulation, or **`reverse_nested_x_servo_controller`** comparison as priorities dictate.
