# Current Status

Last updated: 2026-06-25 (RL PPO pipeline added for Y-axis transport; model-based Z PID still insufficient for full sweeps)

## Active Objective

The current goal is stable **Y-axis end-effector transport** in CoppeliaSim under direct joint torques.

Primary paths:

1. **RL (active development):** PPO policy trained in CoppeliaSim — `rl/`, `simulation/launch_rl_training_wsl.sh`, [RL_Y_TRANSPORT.md](coppeliasim/RL_Y_TRANSPORT.md)
2. **Model-based (baseline / comparison):** Cartesian impedance + IK joint PD — `simulation/run_torque_y_transport_wsl.sh`, [TORQUE_DIAGNOSTICS.md](coppeliasim/TORQUE_DIAGNOSTICS.md)

MuJoCo transport / LQR experiments are archived reference material only.

The controller we care about right now is the portable Cartesian impedance torque controller:

```text
controller_core/x_axis_cartesian_impedance.py
```

The active CoppeliaSim runner is:

```text
simulation/run_coppeliasim_x_axis_headless.py
```

The active launcher is:

```text
simulation/launch_coppeliasim_x_axis_headless.sh
```

## What Is Done

- Repo-local CoppeliaSim runtime is in place under `third_party/coppelia_runtime/`.
- Repo-local Python dependency anchor is in place under `third_party/coppelia_pydeps/`.
- Official CoppeliaSim UR5 model exists at `models/robots/non-mobile/UR5.ttm` inside the runtime.
- Render-only CoppeliaSim smoke test works.
- The smoke test captures 40 upright PNG frames and encodes a 2 second MP4.
- The controller-motion visible-video smoke path works via a simulator-side Lua joint-space controller.
- The controller-motion video path captures 80 upright PNG frames and encodes a 4 second MP4.
- The grounded Coppelia origin-to-acceleration path works as the first stage before broader acceleration transport.
- The path starts from an offset joint pose, drives the end effector to the reference origin pose, holds for 1 second, keeps the camera fixed, scans reachable Coppelia world-X range, clamps the requested displacement to the reachable range, runs an acceleration profile, captures video, and writes pass/fail JSON metrics.
- The origin-acquisition video now renders a visible RGB triad on the end effector and records `ee_triad_visible=true` in the summary.
- The MuJoCo-like Coppelia sweep can now run along the green axis (`MUJOCO_LIKE_SWEEP_AXIS=y`) with a visible RGB triad on the grounded base as well as the end effector.
- The Coppelia X-range height sweep command works and reports the best fixed-Z height for reachable world-X span under the current fixed Y/reference-orientation guardrails.
- The fixed-Z acceleration-transport visible-video path is now guarded against false positives. It captures frames and writes metrics, but the Coppelia port is not accepted unless `success=true` with the base on the ground.
- The Lua add-on can load the UR5 model and create a camera.
- The portable torque controller exists and has small unit/smoke tests.
- The CoppeliaSim adapter exists and can resolve joints, read state, read Jacobian, and apply torque commands.
- The direct Python runner exists, avoiding ROS for first-pass HPC/headless debugging.
- The controller runner now closes its `RemoteAPIClient` cleanly on probe-only and shutdown exits, avoiding the previous silent cleanup bug.
- The controller runner now logs joint-mode readback, compares API vs numerical Jacobians, supports `--zero-torque-test`, and can run bidirectional torque-pulse diagnostics.
- The ROS 2 bridge now seeds a known-good joint pose before starting simulation, stops any already-running simulation first, and treats wrapped joint readback as a warning instead of a fatal startup mismatch on this Coppelia build.
- The ROS 2 legacy positional controller family `legacy_xz_transport_pd` is now verified on the live CoppeliaSim bridge/controller path via `ros2_ws/src/ur5_x_axis_controller_ros/config/controller_coppelia_legacy_xz_transport_relaxed.yaml`.
  - In the verified diagnostic run, publishing an absolute `/target_x` step from about `x=-0.111855` to `x=-0.101855` moved the EE to about `x=-0.089835` with the relaxed safety gate remaining green.
  - The conservative legacy preset still exists, but the relaxed preset is the one that actually demonstrated motion on this build.
- The direct-torque acceleration transport path now has an `ik_joint_pd` policy: a signed world-X acceleration command feeds the transport differential-IK policy, IK produces a reduced-chain joint target with shoulder pan locked, and Coppelia receives direct joint torques from a conservative joint-PD torque law.
- The Coppelia adapter can now create an explicit `mujoco_attachment_dummy` task frame instead of silently controlling the raw `/UR5/UR5_connection` proxy.
- The direct-torque summary now reports frame-reference metadata, local fixed-Z X IK capability, X tracking, single-axis Y drift, fixed-Z drift, orientation hold, joint-configuration sanity, torque saturation, and explicit failure reasons.
- A diagnostic-only lab workspace guardrail model extracted from the external Einksul MuJoCo visualizer now exists in `config/lab_workspace_guardrails.yaml`, with shared checker and overlay helpers in `simulation/workspace_guardrails.py` and an offline trajectory checker in `tools/check_trajectory_guardrails.py`.
- The guardrail path is simulation / visualization only. It adds optional boundary overlays and offline trajectory checks, plus non-blocking ROS 2 status publishers, but it is not a real-arm safety layer.
- The live transport start pose is now policy-specific. The direct Cartesian-impedance branch keeps the better-conditioned fixed-Z seed, while the `ik_joint_pd` surrogate can use the Coppelia-derived `q_start_rad` from the successful origin-acquisition summary in `outputs/control_runs/coppelia_origin_acquisition_state/coppeliasim_ur5_origin_acquisition_summary.json`.
- A separate diagnostic config now exists at `/common/users/ss5772/real_Cartpole/ros2_ws/src/ur5_x_axis_controller_ros/config/controller_after_stable.yaml`. It uses the same gains but switches to the higher `after_stable` torque caps; that improves joint-configuration drift, but the fixed-axis and orientation guardrails still fail on the current controller structure.
- The controller-ready signal mismatch is resolved: Python now publishes both int and string ready signals to avoid signal-type mismatch.
- The probe-only controller startup now passes on the working RPC handshake:
  - `MAX_SIM_SECONDS=120 bash simulation/launch_coppeliasim_x_axis_headless.sh --probe-only --no-video --task-frame-mode mujoco_attachment_dummy`
  - `probe_passed=true`, `torque_mode_verified=true`, `dynamic_mode_verified=true`
  - this Coppelia build does not expose `sim.getJacobian`, so API-vs-numerical Jacobian comparison is skipped at probe time
  - `sim.getJointMode()` returns a tuple here, and the adapter now parses the first element
- The controller launcher now defaults to Python-owned stepping: the shell starts CoppeliaSim in the background, Python connects directly over ZMQ, enables stepping, resolves handles, starts simulation, and runs the live controller. For `--accel-x-transport`, the runner now skips the passive zero-torque warmup because it was letting the arm free-fall into the joint-speed safety stop before the first transport command. Legacy marker handoff remains behind `--legacy-marker-handoff` for compatibility only.
- The controller launcher now fails fast if the requested ZMQ port is already listening, instead of letting Coppelia hide `Address already in use` in the simulator log while Python retries.
- Conservative controller config exists in `ros2_ws/src/ur5_x_axis_controller_ros/config/controller.yaml`.
- The external-capture launcher now exists:
  - `simulation/ur5_external_controller_capture_addon.lua`
  - `simulation/launch_coppeliasim_x_axis_offscreen_capture.sh`
  - the add-on captures on a time cadence instead of every sensing tick, which keeps frame spacing stable when simulator timing jitters
  - the launcher now resolves `ffmpeg` via `FFMPEG_BIN` and the display wrapper via `XVFB_RUN_BIN` or the bundled temp-path fallback
  - the launcher now forwards `RUNNER_EXTRA_ARGS` to the Python runner, so it can run the actual X-transport motion command as well as the default step/probe modes
- The verified Singularity wrapper for the offscreen capture lane now exists:
  - `simulation/launch_coppeliasim_x_axis_offscreen_capture_container.sh`
  - it applies the narrow XKB/Xvfb bind set and is the preferred top-level entrypoint inside `/common/users/ss5772/containers/aha_u2404.sif`
- The actual `--accel-x-transport` capture lane now runs through that wrapper and produces MP4 output, but the current controller tuning still trips the safety guard after a short move.
  - `seeded_probe_cartesian_long1` and `seeded_probe_cartesian_mujoco_neg01` used `--accel-torque-policy cartesian_impedance` with the seeded probe config. Both runs still failed the live envelope: the direct impedance lane moved the arm the wrong way in X, `|Y-Y0|` reached about `0.320 m`, `|Z-Z0|` reached about `0.717 m`, and the run exited on the Y/Z safety stop while the orientation and joint-configuration checks also failed.
  - `seeded_probe_ikpd_long1`, `seeded_probe_ikpd_long2_neg002`, and `seeded_probe_ikpd_lowgain_long1` used `--accel-torque-policy ik_joint_pd`. Those runs were closer in one respect because orientation stayed locked, and the lower-gain pass reduced torque saturation to about 20 percent, but they still failed on Z drift and joint excursion. The best `ik_joint_pd` sweep so far reached about `9.1 mm` net X displacement before failing safety.
  - `run_external_zmq_validation_ladder.sh` / `run_hpc_zmq_attach_probe.sh` are not the primary live probe path on this build. In plain `xvfb_resident_plain` mode the simulator can exit before attach, even with `dfltscn.ttt`, so use the headless probe or the offscreen capture wrapper instead.
  - This is now a controller-tuning / controller-family blocker, not a container/display blocker.
- The ROS 2 diagnostic probe preset now exists at `ros2_ws/src/ur5_x_axis_controller_ros/config/controller_coppelia_slow_seeded_probe.yaml`.
  - it seeds the UR5 to `[0.2, -1.15, 1.55, -1.8, -1.45, 0.35]`
  - it relaxes the safety envelope for a debug run and disables the joint-limit latch because this Coppelia build can report a wrapped revolute angle just beyond `2π`
  - when the controller is started and a `Float64` target is published on `/target_x`, the arm now moves instead of tripping immediately
  - one confirmed run moved the EE from about `x=-0.068669` to `x=0.004001` without a safety stop
- The ROS 2 legacy positional diagnostic preset now exists at `ros2_ws/src/ur5_x_axis_controller_ros/config/controller_coppelia_legacy_xz_transport_relaxed.yaml`.
  - it selects `family: legacy_xz_transport_pd`
  - it uses the MuJoCo-style dummy task frame and the `/UR5/connection` EE proxy with alternates
  - it drives the old differential-IK transport family inside the ROS 2 node, then wraps the resulting joint target in joint-PD torques
  - in the verified run, a single small `/target_x` step moved the EE in world X while the relaxed safety guard remained green
- A new hardware staging lane now exists in `hardware/` and `tools/`:
  - receive-only RTDE probe
  - zero-hold `servoJ`
  - tiny bounded `servoJ` motion
  - guarded direct-torque zero-only probe
  - these scripts are default-safe, but they are not yet validated on a live UR5e
- The point-to-point `ik_joint_pd` surrogate is still not a success condition. The latest settle-hold experiments were a mistake for that policy because they used the wrong hold behavior for the transport family; the best signal remains the no-settle near-miss above.
- Do not treat the existence of a generated MP4 as proof of success. The capture wrapper can still encode a short video even when the summary JSON reports `success=false`; always trust the summary fields.
- Controller code map:
  - live controller family: `python_zmq_external_cartesian_impedance` from [simulation/external_zmq_controller_common.py](/common/users/ss5772/real_Cartpole/simulation/external_zmq_controller_common.py)
  - general Cartesian impedance controller: [controller_core/x_axis_cartesian_impedance.py](/common/users/ss5772/real_Cartpole/controller_core/x_axis_cartesian_impedance.py)
  - actual X-transport mode used in the latest capture run: [simulation/run_coppeliasim_x_axis_headless.py](/common/users/ss5772/real_Cartpole/simulation/run_coppeliasim_x_axis_headless.py) with `--accel-x-transport --accel-torque-policy ik_joint_pd`
  - direct impedance probe mode remains available in the same runner with `--accel-torque-policy cartesian_impedance`; that is the next controller path to validate for final torque behavior
- Current best transport diagnosis:
  - `ik_joint_pd` remains the closest-to-working lane so far, but it still fails the live summary checks.
  - The runner is now using the Coppelia-derived transport start pose for that surrogate.
  - The live stepping interval reported by the summary is `sim_dt_s = 0.05`, so the effective loop is much slower than the nominal `control_rate_hz = 100`.
  - The settle window is not a free lunch for this scene: passive warmup and some hold styles can push `|qd|` past the safety limit before the first nonzero `target_axis_accel` is observed.
  - Do not use a passive zero-torque warmup in transport mode; it can push `|qd|` past the safety limit before the first nonzero `target_axis_accel` is observed.
  - The direct impedance lane remains experimental. The latest hold-reference rework did not produce a stable X traverse on Coppelia UR5.
- Safety threshold code map:
  - defaults live in [controller_core/safety.py](/common/users/ss5772/real_Cartpole/controller_core/safety.py) via `ImpedanceSafetyConfig`
  - the live runner constructs that config in [simulation/run_coppeliasim_x_axis_headless.py](/common/users/ss5772/real_Cartpole/simulation/run_coppeliasim_x_axis_headless.py)
  - the ROS 2 config mirror is [ros2_ws/src/ur5_x_axis_controller_ros/config/controller.yaml](/common/users/ss5772/real_Cartpole/ros2_ws/src/ur5_x_axis_controller_ros/config/controller.yaml)
- The host shell is not sufficient for this Coppelia build. The runtime needs the repo's Ubuntu 24.04 Singularity image:
  - `/common/users/ss5772/containers/aha_u2404.sif`
  - the host shell hits `GLIBCXX_3.4.32` / `GLIBC_2.38` mismatches on the CoppeliaSim binary

## Two Coppelia Pipelines (read this)

Rendering and ZMQ control are **not the same program path**. See
[docs/coppeliasim/PIPELINES.md](coppeliasim/PIPELINES.md) and
[docs/coppeliasim/COPPELIASIM_VISION_NOTES.md](coppeliasim/COPPELIASIM_VISION_NOTES.md)
(Coppelia manual: `handleVisionSensor` before `getVisionSensorImg`, create-options bit 0 = explicit, ZMQ may return a list of bytes).

- **Smoke:** Lua add-on (or the optional ZMQ `run_coppeliasim_video_smoke.py`) → PNGs → MP4. Baseline for **visible** arm.
- **Controller/RPC:** `launch_coppeliasim_x_axis_headless.sh` starts CoppeliaSim in the background, then runs the Python ZMQ runner. The default path is Python-owned stepping with no Lua motion and no marker-file handoff; `--legacy-marker-handoff` keeps the older compatibility path available. The runner also supports `--compare-jacobian/--no-compare-jacobian`, `--zero-torque-test`, and `--torque-pulse-bidirectional` for richer bring-up diagnostics.

## What Is In Progress

Immediate milestone completed: Coppelia origin acquisition and range-scanned
one-direction acceleration now come before any broader acceleration transport
claim. The current verified sequence starts already in the transport plane, so
the visible clip is:

```text
transport-plane start at the requested height
-> brief hold
-> scan reachable X range for fixed Y/Z/reference orientation
-> clamp the requested X displacement to the reachable range
-> run strong one-direction acceleration transport
```

The MuJoCo acceptance contract for this is now documented in
[docs/coppeliasim/MUJOCO_ACCELERATION_GUARDRAILS.md](coppeliasim/MUJOCO_ACCELERATION_GUARDRAILS.md).

Important caveat: the current Coppelia Lua origin/acceleration path moves the
resolved EE proxy along **Coppelia world X**, not along the EE object's local X
axis. That is acceptable only after the controlled Coppelia task frame is made
equivalent to MuJoCo's `attachment_site`. Right now the path still uses a raw
Coppelia EE proxy (`/UR5/UR5_connection` or joint 6 fallback), so the small
scanned X range is diagnostic and should not be interpreted as the real MuJoCo
workspace port.

The current Coppelia origin-acquisition command is:

```bash
cd /common/users/ss5772/real_Cartpole
bash simulation/launch_coppeliasim_origin_acquisition_video.sh
```

The launcher defaults are now the stronger profile with the EE triad enabled:

- `EE_TARGET_Z_M=0.4`
- `TARGET_DX_M=0.06`
- `A_X_MAX_MPS2=3.0`
- `V_X_MAX_MPS=0.6`

Latest verified summary at `EE_TARGET_Z_M=0.4`:

```text
success: true
base_on_ground: true
requested_target_dx_m: 0.060
target_dx_m: 0.025
target_dx_was_clamped_by_range_scan: true
ee_triad_visible: true
start_at_transport_plane: true
x_reachable_min_m: -0.115575017
x_reachable_max_m: -0.0705750173
final_origin_position_error_m: 0.0000937
acceleration_final_position_error_m: 0.0010861
acceleration_final_orientation_error_deg: 2.881
peak_abs_ee_vx_mps: 0.2347
peak_joint_speed_rad_s: 0.8298
frames: 106
```

Outputs:

```text
outputs/control_runs/coppelia_origin_acquisition_frames/
outputs/control_runs/coppelia_origin_acquisition_state/coppeliasim_ur5_origin_acquisition_summary.json
demonstration_videos/ur5e_coppeliasim/coppeliasim_ur5_origin_acquisition.mp4
```

The current Coppelia X-range height sweep command is:

```bash
cd /common/users/ss5772/real_Cartpole
bash simulation/launch_coppeliasim_x_range_sweep.sh
```

Latest verified sweep, using `SWEEP_Z_MIN_M=0.35`, `SWEEP_Z_MAX_M=0.95`,
`SWEEP_Z_STEP_M=0.025`, `RANGE_SCAN_STEP_M=0.0025`, found:

```text
best_height_m: 0.400
best_x_span_m: 0.045
best_x_min_m: -0.115575
best_x_max_m: -0.070575
```

The max span is a tie at several heights with the current scan resolution:
`0.400`, `0.425`, `0.550`, `0.575`, `0.600`, and `0.700` m all report
about `0.045 m` reachable X span.

The current green-axis MuJoCo-like sweep command is:

```bash
cd /common/users/ss5772/real_Cartpole
bash simulation/launch_coppeliasim_mujoco_like_y_sweep_video.sh
```

Latest verified sweep:

```text
success: true
base_on_ground: true
sweep_axis_index: 2
sweep_axis_label: y
base_triad_visible: true
ee_triad_visible: true
axis_net_displacement_m: 0.3486423
axis_tracking_error_m: -0.00135770049
acceleration_final_position_error_m: 0.00122073936
acceleration_final_orientation_error_deg: 0.0023251412
frames: 744
```

Outputs:

```text
outputs/control_runs/coppelia_mujoco_like_y_sweep_frames/
outputs/control_runs/coppelia_mujoco_like_y_sweep_state/coppeliasim_ur5_mujoco_like_y_sweep_summary.json
demonstration_videos/ur5e_coppeliasim/coppeliasim_ur5_mujoco_like_y_sweep.mp4
```

The current MuJoCo-like Y torque video command is:

```bash
cd /common/users/ss5772/real_Cartpole
bash simulation/launch_coppeliasim_mujoco_like_y_torque_video.sh
```

Latest verified torque-video summary:

```text
success: true
base_on_ground: true
transport_axis_tracking_ok: true
fixed_axes_ok: true
orientation_ok: true
torque_saturation_ok: true
frame_reference_ok: true
target_axis_net_displacement_m: 0.149700342
max_orientation_error_deg: 0.655846542
peak_abs_tau_nm: 3.01971679
frames: 185
```

The current default target is `TARGET_DX_M=0.35`. For the same frame, the
measured green-axis sweep limit is about `0.3486423 m` net displacement, so
the video path now asks for the full measured travel instead of stopping at
the old 0.15 m clip. The Lua add-on now clips only to the UR5e model effort
limits from `mujoco_menagerie/universal_robots_ur5e/ur5e.xml`.

The current controller/RPC path uses **Python-owned stepping by default**. The shell starts CoppeliaSim in the background, Python connects directly over ZMQ, enables stepping, resolves handles, starts simulation, and runs a short zero-torque warmup before the live controller. The older Lua release-marker bootstrap remains only behind `--legacy-marker-handoff`. The remaining work is to keep this startup path boring, then prove stable torque-driven motion and, when video is on, that summary JSON shows healthy `first_frame_std_rgb` (not all-black buffers).

The current blocker for the new external-capture launcher is the display/toolchain environment inside the 24.04 container. The launcher code is in place, but the exact bind set for `xauth`/`Xvfb`/video support still needs to be pinned down in this shell before the launcher can be treated as verified here.

For a first-principles walkthrough of the controller stack, start at
[docs/coppeliasim/CONTROLLER_FIRST_PRINCIPLES.md](coppeliasim/CONTROLLER_FIRST_PRINCIPLES.md).

For the acceleration-transport controller path, the intended direct-torque
command is now:

```bash
cd /common/users/ss5772/real_Cartpole
bash simulation/launch_coppeliasim_x_axis_headless.sh \
  --accel-x-transport \
  --accel-torque-policy ik_joint_pd \
  --target-dx 0.005 \
  --duration 3 \
  --settle-duration 1
```

In this mode, the external command is only signed world-X acceleration. The
runner uses differential IK to compute the joint target for the other degrees of
freedom, locks shoulder pan, and applies direct torques with a joint-PD law.
The summary must show `uses_direct_torque_control=true`,
`uses_position_servo_setpoints=false`, `frame_reference_ok=true`,
`x_tracking_ok=true`, `single_axis_y_ok=true`, `fixed_z_ok=true`,
`orientation_ok=true`, `joint_configuration_ok=true`, and
`torque_saturation_ok=true` before the run is accepted.

The current implementation has the pieces wired together in the default Python-owned launch path, but the full success case has not yet been made clean and boring in every environment:

```text
CoppeliaSim starts and stays resident
-> RPC is ready
-> Python connects
-> UR5 handles resolve
-> torque mode is configured
-> controller reads state and Jacobian
-> controller sends torque
-> UR5 moves in controlled world-X direction
-> trace/video prove bounded, stable motion
```

## What Is Not Proven Yet

- Stable torque-driven UR5 motion in CoppeliaSim.
- A full `--accel-x-transport --accel-torque-policy ik_joint_pd ...` run that satisfies the summary guardrails.
- The startup probe is now clean; the remaining gap is motion, not RPC bootstrapping.
- The direct-torque Lua add-on experiment still only reaches the first sensing tick at `t=0`; it does not advance into a continuous step loop in this add-on form.
- Correct torque sign and magnitude semantics for `sim.setJointTargetForce(handle, tau, true)` on this UR5 model.
- Whether all six joints are truly in dynamic force/torque mode with position control disabled.
- Whether the CoppeliaSim API Jacobian is reliable on this build.
- Whether the numerical Jacobian fallback is fast and accurate enough for first controller tests.
- Whether zero torque causes acceptable gravity sag or an unstable/drop behavior.
- Whether tiny X-target control survives safety limits without immediate stop.
- Whether the new `mujoco_attachment_dummy` task frame is numerically aligned with MuJoCo's `attachment_site` on a live Coppelia run.
- Whether the local fixed-Z X capability report predicts enough reachable motion at the selected Z height before a larger target is attempted.
- Whether gravity compensation is needed before X tracking can be meaningfully evaluated.

## Current Fragility

- **HPC / display:** Run Coppelia on a **Slurm GPU or interactive** node inside the repo's Ubuntu 24.04 Singularity image; see [AGENTS.md](../AGENTS.md). The host shell is not sufficient for the current CoppeliaSim binary. OpenGL may be Mesa **llvmpipe** (CPU) or a hardware GL stack.

- **Video over ZMQ:** Pixels are not guaranteed the same as the Lua PNG path until `handle`→`get` order and **buffer** decoding are correct; see vision notes. Use `--video-camera smoke` (default) for the same world camera as the smoke test.

- **Probe:** The launcher no longer uses the old `-s300000` sim auto-stop. Use `--probe-only` before applying torque in new environments.

The `--probe-only` path is the safest way to inspect handle resolution, EE pose,
Jacobian reads, and startup timing without commanding torque.

## RPC Connection Guardrails

- Treat the launcher and the runner as a timed handshake, not a fire-and-forget process.
- Confirm the actual Python interpreter before a live probe; the shared Coppelia wheels are `cp312`, so the live runner must match that ABI.
- Confirm the ZMQ port is free before every live probe:

```bash
ps -ef | rg 'zmqRemoteApi.rpcPort=23000|coppeliaSim'
```

If a stale Coppelia process is present, stop it intentionally or use a different
port, for example `PORT=23010`.

- Preserve the Python-owned launch order: the shell starts CoppeliaSim first, Python starts second, Python enables stepping and starts simulation, and then the live controller loop begins. If you explicitly enable `--legacy-marker-handoff`, the older Lua release-marker choreography applies only in that compatibility mode.
- Use `--probe-only` first, then verify joint-mode readback and Jacobian sanity before touching torque gains.
- If anything fails, inspect `outputs/control_runs/coppeliasim_x_axis_headless.log` first and sort the failure into port conflict, import mismatch, simulator shutdown, missing ready signal, or missing `connect` attempts.
- After a startup fix, rerun the exact launcher command and confirm that one log marker advanced before changing controller logic.

## Working Visible Motion Video

For a visible CoppeliaSim video where the UR5 actually moves, use:

```bash
cd /common/users/ss5772/real_Cartpole
bash simulation/launch_coppeliasim_controller_video.sh
```

This uses `simulation/ur5_controller_video_addon.lua` to drive a bounded
joint-space trajectory inside CoppeliaSim and capture PNG frames through the
known-good Lua/PNG rendering route. Output:

```text
outputs/control_runs/coppelia_controller_video_frames/
demonstration_videos/ur5e_coppeliasim/coppeliasim_ur5_controller_video.mp4
```

This proves visible arm motion and video capture. It does **not** prove stable
external ZMQ Cartesian torque control yet.

## Working Acceleration Transport Video

For a diagnostic fixed-Z Coppelia video with speed metrics and guardrails, use:

```bash
cd /common/users/ss5772/real_Cartpole
bash simulation/launch_coppeliasim_acceleration_transport_video.sh
```

Output:

```text
outputs/control_runs/coppelia_acceleration_transport_frames/
outputs/control_runs/coppelia_acceleration_transport_state/coppeliasim_ur5_acceleration_transport_summary.json
demonstration_videos/ur5e_coppeliasim/coppeliasim_ur5_acceleration_transport.mp4
```

The summary now includes acceptance fields:

```text
base_on_ground
x_tracking_ok
single_axis_y_ok
fixed_z_ok
orientation_ok
success
failure_reasons
```

The MuJoCo contract is documented in
[docs/coppeliasim/MUJOCO_ACCELERATION_GUARDRAILS.md](coppeliasim/MUJOCO_ACCELERATION_GUARDRAILS.md).
The Coppelia path must not lift the base to satisfy height. It must use a
grounded model and pass the fixed-Y/Z/orientation/X-tracking guardrails before
it is considered a proper port.

## Current Success Criteria

The next meaningful success is not a big demo. It is a tiny, instrumented motion proof:

- CoppeliaSim starts from the launcher.
- RPC readiness is detected without a fixed long sleep.
- Python connects and logs each startup phase.
- UR5 model loads or attaches deterministically.
- All six joints and the EE object resolve.
- A read-only state/Jacobian probe succeeds.
- The probe includes API-vs-numerical Jacobian deltas and joint-mode readback.
- A zero-torque test is stable enough to interpret.
- A tiny `target_dx`, for example `0.005 m`, produces motion in the expected X direction.
- Y/Z drift, orientation error, joint velocity, and torque saturation remain bounded.
- JSONL trace and summary are written.

## Current Commands

Known-good render smoke:

```bash
cd /common/users/ss5772/real_Cartpole
bash simulation/launch_coppeliasim_video_smoke.sh
```

Known-good visible motion video:

```bash
cd /common/users/ss5772/real_Cartpole
bash simulation/launch_coppeliasim_controller_video.sh
```

Known-good grounded origin acquisition video:

```bash
cd /common/users/ss5772/real_Cartpole
bash simulation/launch_coppeliasim_origin_acquisition_video.sh
```

Diagnostic acceleration transport video:

```bash
cd /common/users/ss5772/real_Cartpole
bash simulation/launch_coppeliasim_acceleration_transport_video.sh
```

Latest verified summary for that command:

```text
success: true
base_on_ground: true
x_tracking_ok: true
single_axis_y_ok: true
fixed_z_ok: true
orientation_ok: true
x_net_displacement_m: 0.00931395539
max_orientation_error_deg: 1.86170679
```

The Coppelia attachment proxy that matches this run is the tool target rotated
`+90 deg` about local Z on this UR5.ttm build. The launcher now records the
start/end pose matrices so the attachment-frame guardrail can stay aligned with
the actual Coppelia axes instead of a guessed MuJoCo flip.

Active controller/RPC attempt:

```bash
cd /common/users/ss5772/real_Cartpole
bash simulation/launch_coppeliasim_x_axis_headless.sh
```

Suggested tiny first controller run after port cleanup:

```bash
cd /common/users/ss5772/real_Cartpole
bash simulation/launch_coppeliasim_x_axis_headless.sh --duration 3 --settle-duration 1 --target-dx 0.005
```

Useful diagnostics once the launcher is up:

```bash
cd /common/users/ss5772/real_Cartpole
bash simulation/launch_coppeliasim_x_axis_headless.sh --probe-only
bash simulation/launch_coppeliasim_x_axis_headless.sh --zero-torque-test
bash simulation/launch_coppeliasim_x_axis_headless.sh --torque-pulse --torque-pulse-bidirectional
```
