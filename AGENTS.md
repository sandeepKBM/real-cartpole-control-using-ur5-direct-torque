# AGENTS.md

This file is the working playbook for agents operating in `/common/users/ss5772/real_Cartpole`.

Update it whenever a new workflow becomes reliable, a failure mode becomes clear, or a path/env assumption changes.

## Project Reality

- The folder name is historical. This is primarily a UR5 / UR5e control workspace, not an active cartpole project.
- Current active work is CoppeliaSim only. Do not use MuJoCo simulation/controller paths for active debugging or development in this workspace.
  - make the UR5 arm move in CoppeliaSim using the existing single-axis torque controller.
  - the render-only CoppeliaSim video smoke path is done and should remain a separate baseline.
  - the controller-driven CoppeliaSim motion path is still in progress.
  - MuJoCo-related code, videos, and notes are historical reference only unless a task explicitly says otherwise.
- The most important current paths are:
  - `controller_core/`
  - `simulation/`
  - `ros2_ws/src/ur5_x_axis_controller_ros/`
- The diagnostic lab workspace guardrail workflow now lives in:
  - `config/lab_workspace_guardrails.yaml`
  - `simulation/workspace_guardrails.py`
  - `tools/check_trajectory_guardrails.py`
  - `tools/render_guardrail_overlay.py`
  - optional `--draw-guardrails` flags on the simulation runners
  - Treat it as simulation / visualization only. Do not wire `/viz/collision` into real-arm emergency stop logic.
- The repo root is not a git repo. Do not assume `git status` will work here.
- There is a nested git repo under `mujoco_menagerie/`.
- `third_party/coppelia/` is a historical symlink to the old external mirror.
  - Use `third_party/coppelia_runtime/` for active CoppeliaSim launches.
- Project-owned Markdown documentation now lives under `docs/`, except this root `AGENTS.md`.
  - Start at `docs/README.md`.
  - Current status is `docs/CURRENT_STATUS.md`.
  - **Smoke (Lua/PNG) vs ZMQ controller:** `docs/coppeliasim/PIPELINES.md`.
  - **Coppelia API for vision (handle → get, buffers):** `docs/coppeliasim/COPPELIASIM_VISION_NOTES.md`.
  - **MuJoCo acceleration guardrails:** `docs/coppeliasim/MUJOCO_ACCELERATION_GUARDRAILS.md` (historical reference only; do not use for active controller development).
  - The active RPC/controller plan is `docs/coppeliasim/RPC_CONTROLLER_TODO.md`.
  - Historical MuJoCo/MoveIt/SLSQP notes are archived under `docs/archive/`.

## Current Headless CoppeliaSim Setup

- Target simulator build:
  - `third_party/coppelia_runtime/CoppeliaSim_Edu_V4_10_0_rev0_Ubuntu24_04`
- Shared Python dependency anchor for CoppeliaSim:
  - `third_party/coppelia_pydeps`
- Required Python packages available via the shared pydeps anchor:
  - `coppeliasim-zmqremoteapi-client`
  - `numpy`
  - `pyyaml`
- Official model path:
  - `third_party/coppelia_runtime/CoppeliaSim_Edu_V4_10_0_rev0_Ubuntu24_04/models/robots/non-mobile/UR5.ttm`

## Important Controller Paths

- Portable controller core:
  - `controller_core/x_axis_cartesian_impedance.py`
- ROS 2 controller node:
  - `ros2_ws/src/ur5_x_axis_controller_ros/ur5_x_axis_controller_ros/controller_node.py`
- CoppeliaSim bridge and adapter:
  - `ros2_ws/src/ur5_x_axis_controller_ros/ur5_x_axis_controller_ros/coppeliasim_bridge_node.py`
  - `ros2_ws/src/ur5_x_axis_controller_ros/ur5_x_axis_controller_ros/coppeliasim_adapter.py`

## Files Added During Headless Bring-Up

- Controller-coupled headless runner:
  - `simulation/run_coppeliasim_x_axis_headless.py`
- Controller-coupled launcher:
  - `simulation/launch_coppeliasim_x_axis_headless.sh`
- External-controller capture launcher:
  - `simulation/launch_coppeliasim_x_axis_offscreen_capture.sh`
- Singularity wrapper for the external-controller capture lane:
  - `simulation/launch_coppeliasim_x_axis_offscreen_capture_container.sh`
- Pure simulator/video smoke runner:
  - `simulation/run_coppeliasim_video_smoke.py`
- Pure simulator/video smoke launcher:
  - `simulation/launch_coppeliasim_video_smoke.sh`
- RPC-free UR5 smoke add-on:
  - `simulation/ur5_video_smoke_addon.lua`
- Controller-motion visible-video add-on:
  - `simulation/ur5_controller_video_addon.lua`
- Controller-motion visible-video launcher:
  - `simulation/launch_coppeliasim_controller_video.sh`
- Acceleration-transport visible-video add-on:
  - `simulation/ur5_acceleration_transport_video_addon.lua`
- Fixed-Z acceleration-transport visible-video add-on:
  - `simulation/ur5_fixed_z_acceleration_transport_addon.lua`
- Acceleration-transport visible-video launcher:
  - `simulation/launch_coppeliasim_acceleration_transport_video.sh`
- Origin-to-acceleration visible-video add-on:
  - `simulation/ur5_origin_acquisition_video_addon.lua`
- Origin-to-acceleration visible-video launcher:
  - `simulation/launch_coppeliasim_origin_acquisition_video.sh`
- Coppelia X-range height sweep launcher:
  - `simulation/launch_coppeliasim_x_range_sweep.sh`
- MuJoCo-like green-axis sweep launcher:
  - `simulation/launch_coppeliasim_mujoco_like_y_sweep_video.sh`

## Current Working Video Smoke Command

- From `/common/users/ss5772/real_Cartpole`, on a GPU/Xvfb-capable interactive node:
  - `bash simulation/launch_coppeliasim_video_smoke.sh`
- Verified GPU-node run:
  - Slurm job `131037` on `rlab7`but you can actually use any gpu in ilab and rlab (try to avoid rlab6&7 & 1 because other jobs might need them and we need gpu for the display head primarily not for gpu computations)
  - Rendered successfully with `xvfb-run -a` and CoppeliaSim `-h -vscriptinfos`
- Expected outputs:
  - 40 upright PNG frames in `outputs/control_runs/coppelia_video_smoke_frames/`
  - 2-second MP4 at `demonstration_videos/ur5e_coppeliasim/coppeliasim_ur5_video_smoke.mp4`
- The launcher watches for all 40 frames, stops CoppeliaSim, then runs ffmpeg.
  - This avoids waiting for CoppeliaSim to exit cleanly from the add-on callback.
- The launcher now writes state and logs under `outputs/control_runs/coppelia_video_smoke_state/`.
  - That avoids `/tmp` quota issues on this node.
  - The default fallback is `xvfb-run -a` with `coppeliaSim.sh -h -vscriptinfos`.
  - Raw `Xvfb` is only used when `COPPELIA_USE_RAW_XVFB=1` is set.

## Current Working Controller-Motion Video Command

- From `/common/users/ss5772/real_Cartpole`, on a GPU/Xvfb-capable interactive node:
  - `bash simulation/launch_coppeliasim_controller_video.sh`
- Verified run:
  - Captured 80 upright PNG frames in `outputs/control_runs/coppelia_controller_video_frames/`
  - Encoded a 4-second MP4 at `demonstration_videos/ur5e_coppeliasim/coppeliasim_ur5_controller_video.mp4`
  - The arm is visible and moves through a scripted Lua joint-space controller.
- This path intentionally uses the simulator-side Lua/PNG capture route because the ZMQ controller video path can still return black frames.
- This is a visible controller-motion validation path, not proof that the external Cartesian torque controller is stable.

## Current Diagnostic Acceleration-Transport Video Command

- From `/common/users/ss5772/real_Cartpole`, on a GPU/Xvfb-capable interactive node:
  - `bash simulation/launch_coppeliasim_acceleration_transport_video.sh`
- Verified run:
  - Captures upright PNG frames in `outputs/control_runs/coppelia_acceleration_transport_frames/`
  - Encodes an MP4 at `demonstration_videos/ur5e_coppeliasim/coppeliasim_ur5_acceleration_transport.mp4`
  - Wrote metrics JSON at `outputs/control_runs/coppelia_acceleration_transport_state/coppeliasim_ur5_acceleration_transport_summary.json`
- This path is now guarded against false positives:
  - `base_on_ground`
  - `x_tracking_ok`
  - `single_axis_y_ok`
  - `fixed_z_ok`
  - `orientation_ok`
  - `success`
  - `failure_reasons`
- Do not accept a Coppelia acceleration video as a correct port unless `success=true` and `base_on_ground=true`.
- This is a Coppelia simulator-side position/IK diagnostic path. It is not yet external ZMQ torque-controller parity.

## Current Working Origin-To-Acceleration Video Command

- From `/common/users/ss5772/real_Cartpole`, on a GPU/Xvfb-capable interactive node:
  - `bash simulation/launch_coppeliasim_origin_acquisition_video.sh`
- Verified run:
  - Starts already in the transport plane at `EE_TARGET_Z_M=0.4`.
  - Holds still briefly, keeps the camera fixed, scans reachable Coppelia world-X range at fixed Y/Z/reference orientation, clamps the requested X displacement to the reachable range, and then runs a one-direction strong acceleration profile.
  - Renders a visible RGB XYZ triad on the end effector.
  - Captures 106 upright PNG frames in `outputs/control_runs/coppelia_origin_acquisition_frames/`.
  - Encodes an MP4 at `demonstration_videos/ur5e_coppeliasim/coppeliasim_ur5_origin_acquisition.mp4`.
  - Writes metrics JSON at `outputs/control_runs/coppelia_origin_acquisition_state/coppeliasim_ur5_origin_acquisition_summary.json`.
- Latest verified metrics:
  - `success`: `true`
  - `base_on_ground`: `true`
  - `requested_target_dx_m`: `0.060`
  - `target_dx_m`: `0.025`
  - `target_dx_was_clamped_by_range_scan`: `true`
  - `x_reachable_min_m`: about `-0.115575017`
  - `x_reachable_max_m`: about `-0.0705750173`
  - `acceleration_final_position_error_m`: about `0.0010861`
  - `acceleration_final_orientation_error_deg`: about `2.881`
  - `peak_abs_ee_vx_mps`: about `0.2347`
  - `ee_triad_visible`: `true`
- This is a Coppelia simulator-side position/IK path, not external ZMQ torque-controller parity.
- The launcher defaults are now the stronger profile:
  - `EE_TARGET_Z_M=0.4`
  - `TARGET_DX_M=0.06`
  - `A_X_MAX_MPS2=3.0`
  - `V_X_MAX_MPS=0.6`
- This is the best verified option so far when the goal is a visibly larger X traverse with stronger acceleration while keeping the base grounded.

## Current Working MuJoCo-Like Green Sweep Video Command

- From `/common/users/ss5772/real_Cartpole`, on a GPU/Xvfb-capable interactive node:
  - `bash simulation/launch_coppeliasim_mujoco_like_y_sweep_video.sh`
- Verified run:
  - Sweeps along world `Y` (`MUJOCO_LIKE_SWEEP_AXIS=y`), which is the green axis in the triad overlay.
  - Renders visible RGB triads on both the grounded base and the end effector.
  - Captures 744 upright PNG frames in `outputs/control_runs/coppelia_mujoco_like_y_sweep_frames/`.
  - Encodes an MP4 at `demonstration_videos/ur5e_coppeliasim/coppeliasim_ur5_mujoco_like_y_sweep.mp4`.
  - Writes metrics JSON at `outputs/control_runs/coppelia_mujoco_like_y_sweep_state/coppeliasim_ur5_mujoco_like_y_sweep_summary.json`.
- Latest verified metrics:
  - `success`: `true`
  - `base_on_ground`: `true`
  - `sweep_axis_index`: `2`
  - `sweep_axis_label`: `y`
  - `base_triad_visible`: `true`
  - `ee_triad_visible`: `true`
  - `axis_net_displacement_m`: `0.3486423`
  - `axis_tracking_error_m`: `-0.00135770049`
  - `acceleration_final_position_error_m`: `0.00122073936`
  - `acceleration_final_orientation_error_deg`: `0.0023251412`
- This is a Coppelia simulator-side position/IK path, not external ZMQ torque-controller parity.

## Current Working X-Range Height Sweep Command

- From `/common/users/ss5772/real_Cartpole`, on a GPU/Xvfb-capable interactive node:
  - `bash simulation/launch_coppeliasim_x_range_sweep.sh`
- Verified run:
  - Sweeps `SWEEP_Z_MIN_M=0.35` to `SWEEP_Z_MAX_M=0.95` in `SWEEP_Z_STEP_M=0.025` increments.
  - Uses the current fixed Y/reference-orientation guardrails and scans Coppelia world-X reachability with `RANGE_SCAN_STEP_M=0.0025`.
  - Writes summary JSON at `outputs/control_runs/coppelia_origin_acquisition_state/coppeliasim_ur5_x_range_height_sweep_summary.json`.
- Latest verified metrics:
  - `best_height_m`: `0.400`
  - `best_x_span_m`: `0.045`
  - `best_x_min_m`: about `-0.115575`
  - `best_x_max_m`: about `-0.070575`
- The latest sweep has several tied max-span heights at the current scan resolution: `0.400`, `0.425`, `0.550`, `0.575`, `0.600`, and `0.700` m each report about `0.045 m` reachable X span.

## Current Controller / RPC Status

- **Launcher:** `simulation/launch_coppeliasim_x_axis_headless.sh` starts CoppeliaSim in the background and runs the Python runner in the foreground. The older keepalive add-on is only copied when `--legacy-marker-handoff` is explicitly requested.
- The launcher intentionally does **not** use Coppelia `-s` auto-stop by default.
- The launcher fails fast when the requested ZMQ port is already listening. Do not debug controller code if the log says `Address already in use`; clear the stale Coppelia process or choose a different `PORT`.
- Python remains the intended ZMQ control owner. In the legacy marker-handoff mode, the Lua bootstrap is only responsible for scene/model preload and waiting on the RPC release marker.
- The Lua bootstrap must not publish `real_cartpole_controller_ready` itself. Only Python should publish readiness after it connects, resolves handles, configures torque mode, and samples initial state.
- The default bootstrap is Python-owned stepping:
  - The shell launches CoppeliaSim in the background.
  - Python connects directly, enables stepping, resolves handles, starts simulation, and enters the controller loop.
  - The legacy marker-file choreography remains only behind `--legacy-marker-handoff`.
- The controller runner has `--probe-only` and optional `--torque-pulse`, `--video-camera smoke` (default, same camera logic as `run_coppeliasim_video_smoke.py` for visible arm).
- The controller runner also supports `--compare-jacobian/--no-compare-jacobian`, `--zero-torque-test`, and `--torque-pulse-bidirectional` for bring-up diagnostics.
- For direct-torque acceleration transport, prefer:
  - `--accel-x-transport --accel-torque-policy ik_joint_pd --target-dx 0.005 --duration 3 --settle-duration 1`
- In `ik_joint_pd` mode, the only external task command is signed world-X acceleration. Differential IK computes the reduced-chain joint target, shoulder pan stays locked, and Coppelia receives direct joint torques from conservative joint PD.
- The Coppelia adapter now supports `task_frame.mode: mujoco_attachment_dummy`, which creates an explicit task dummy mirroring MuJoCo's `attachment_site` convention instead of silently controlling the raw wrist connection.
- Do not accept an acceleration-transport run unless the summary reports `uses_direct_torque_control=true`, `uses_position_servo_setpoints=false`, `frame_reference_ok=true`, `x_tracking_ok=true`, `single_axis_y_ok=true`, `fixed_z_ok=true`, `orientation_ok=true`, `joint_configuration_ok=true`, and `torque_saturation_ok=true`.
- The external-capture wrapper is now tracked separately from the plain torque launcher:
  - `simulation/launch_coppeliasim_x_axis_offscreen_capture.sh`
  - `simulation/ur5_external_controller_capture_addon.lua`
  - it uses a time-based capture cadence and resolves `FFMPEG_BIN` / `XVFB_RUN_BIN`
  - it now forwards `RUNNER_EXTRA_ARGS` to the Python runner, so the same capture lane can run the actual X-transport command
  - the host shell is not sufficient for the current CoppeliaSim binary; use the repo's Ubuntu 24.04 Singularity image on this cluster
- The verified Singularity wrapper for the capture lane is:
  - `simulation/launch_coppeliasim_x_axis_offscreen_capture_container.sh`
  - it binds `/usr/bin/xkbcomp`, the host XKB tree to `/usr/lib/X11/xkb` and `/usr/share/X11/xkb`, and reuses `/common/home/ss5772/.tmp/container_bind_libs`
- Probe-only startup now passes on the working RPC handshake:
  - `MAX_SIM_SECONDS=120 bash simulation/launch_coppeliasim_x_axis_headless.sh --probe-only --no-video --task-frame-mode mujoco_attachment_dummy`
  - `probe_passed=true`, `torque_mode_verified=true`, `dynamic_mode_verified=true`, and `real_cartpole_controller_ready=1`
  - this Coppelia build does not expose `sim.getJacobian`, so the adapter skips the API-vs-numerical Jacobian comparison at probe time
  - `sim.getJointMode()` returns a tuple on this build; the adapter now parses the first element before checking dynamic mode
- The older `simulation/run_external_zmq_validation_ladder.sh` and `simulation/run_hpc_zmq_attach_probe.sh` paths are not the preferred live probe path here. In plain `xvfb_resident_plain` mode the simulator can exit before attach, even with `dfltscn.ttt`, so prefer `simulation/launch_coppeliasim_x_axis_headless.sh --probe-only --no-video --task-frame-mode mujoco_attachment_dummy` or the offscreen capture wrapper.
- The current ROS 2 diagnostic probe is:
  - seed config: `ros2_ws/src/ur5_x_axis_controller_ros/config/controller_coppelia_slow_seeded_probe.yaml`
  - bridge first, then controller
  - publish an absolute `Float64` on `/target_x` after the controller prints its first valid state
  - observed result: the EE moved from about `x=-0.068669` to `x=0.004001` without a ROS-side safety stop under that loose diagnostic gate
- The verified ROS 2 legacy positional fallback is:
  - seed config: `ros2_ws/src/ur5_x_axis_controller_ros/config/controller_coppelia_legacy_xz_transport_relaxed.yaml`
  - family: `legacy_xz_transport_pd`
  - it reuses the older `simulation/controller.py` differential-IK transport logic inside the ROS 2 node, then wraps the result in joint-PD torques
  - observed result: publishing a small absolute `/target_x` step moved the EE in world X while the relaxed ROS-side safety gate remained green
- Bridge startup now stops any already-running simulation before seeding the UR5 pose. Seed readback should be treated as a wrapped revolute-angle warning, not a fatal error, on repeated probes in this build.
- **Pixels:** ZMQ `getVisionSensorImg` may return `list` of bytes; the runner decodes that. The separate **smoke** script still uses **Lua/PNG** for the most reliable “known good” image baseline (`docs/coppeliasim/PIPELINES.md`).

## RPC Connection Guardrails

- Never assume the Python environment is interchangeable. The live runner currently needs Python 3.12, while `vlaism-ur5e-openvla` is Python 3.10.20 and the shared `third_party/coppelia_pydeps/` wheels are `cp312`.
- Before any live probe, confirm the exact `PYTHON_BIN` or `python3` that the launcher will use, and keep that path stable for the whole run.
- Before any live probe on port `23000`, check for stale simulator processes:
  - `ps -ef | rg 'zmqRemoteApi.rpcPort=23000|coppeliaSim'`
  - If one exists, use a free `PORT=...` or deliberately stop the stale Coppelia process before launching.
- Preserve startup order:
  - Default mode: the shell starts CoppeliaSim first, Python starts second, Python enables stepping and starts simulation, then the controller loop begins.
  - Legacy mode: if `--legacy-marker-handoff` is enabled, the older Lua release-marker choreography applies only in that compatibility path.
- Use `--probe-only` first in every new environment. Do not change torque gains until joint-mode readback is clean and the API-vs-numerical Jacobian comparison is finite.
- If a launch fails, read `outputs/control_runs/coppeliasim_x_axis_headless.log` first and classify the failure by marker:
  - `Address already in use`
  - `ImportError` or wheel mismatch
  - `QProcess: Destroyed while process ... is still running`
  - `controller ready signal observed`
  - missing or delayed `connect` attempts from the Python runner
- Current live failure marker after the 2026-04-30 fixes: stale Coppelia processes can occupy `rpcPort=23000`, producing `Address already in use` and causing Python to retry until timeout. Treat that as port hygiene, not torque-control tuning.
- After any startup change, rerun the exact launcher command and record which marker moved forward. Do not combine startup changes with torque tuning in the same iteration.
- Prefer startup fixes over controller logic fixes whenever the failure happens before the first successful connect.

## Do Not Recreate These RPC Bugs

- Do not make Lua publish `real_cartpole_controller_ready`; readiness belongs to Python only.
- Do not rely on `sysCall_thread`, `sysCall_sensing`, or `sysCall_nonSimulation` as the only keepalive for this controller add-on in headless foreground mode; use the marker-file handshake that keeps `sysCall_init` alive until Python releases RPC.
- Do not leave `PORT=23000` stale and then tune controller gains. `Address already in use` means the simulator never opened the RPC endpoint for this run.
- Do not switch the controller runner back to the Python 3.10 `vlaism-ur5e-openvla` env for live Coppelia; the shared Coppelia Python deps are `cp312`.
- Do not re-enable smoke/video add-ons in the controller launch; they can stop the simulator independently of the torque path.
- Do not accept a direct-torque acceleration run without the explicit summary guardrails passing.
- Do not leave `--accel-x-transport` in a passive zero-torque warmup. The arm can free-fall into the `|qd| > 1.5 rad/s` safety stop before the first transport command is issued; if a settle phase is needed, hold the pose with the controller instead of passive torque.
- Do not validate a transport start pose in `--probe-only` unless `--accel-x-transport` is also enabled; `Q_START_RAD` is only applied inside the transport branch.
- Do not transplant a MuJoCo workspace seed directly into Coppelia and treat it as equivalent. Coppelia-specific seeds need their own probe because the task-frame conditioning can change by orders of magnitude.
- Prefer a Coppelia-derived successful transport seed over a MuJoCo-derived seed when tuning `Q_START_RAD` for live x-transport.
- For the default live x-transport start pose, choose the seed by torque policy. `cartesian_impedance` should keep the better-conditioned fixed-Z transport seed, while `ik_joint_pd` can use a Coppelia-success seed from a passed summary (`q_start_rad`). Do not reuse one seed across both controller families.
- Do not tune the Cartesian impedance gains as if the loop were 100 Hz unless the live summary confirms it. The current Coppelia stepping path is effectively running at `sim_dt_s = 0.05`, so gains must be interpreted at that discrete-time rate.
- Do not treat a solver-side `all_task_feasible=true` / `task_feasible=true` result as proof of a working transport controller. The live summary must still pass the runtime safety and drift checks.
- Do not use `ik_joint_pd` as evidence that the direct Cartesian impedance path works. It is a joint-PD transport surrogate; use the actual impedance branch when validating the final torque controller.
- Do not hard-fail a ROS 2 seeded probe just because a revolute joint reads back at `2π + epsilon` after start. On this build, compare revolute seed/pose errors modulo `2π` or downgrade the mismatch to a warning in diagnostic configs.
- Do not treat an MP4 as proof of a successful live run. The capture wrapper can still encode a short video even when the summary JSON reports `success=false`; always inspect the JSON summary first.

## Best Practices For This Workspace

- Start with the simplest proof first.
  - For CoppeliaSim, prove: launch -> load model -> render frames -> write MP4.
  - Only after that should controller wiring happen.
- For the current controller/RPC journey, debug in this order:
  - RPC lifecycle and startup ownership.
  - read-only adapter probe.
  - torque sign/magnitude semantics.
  - Jacobian validity.
  - zero-torque dynamics.
  - tiny X motion.
  - controller tuning.
- The probe now emits joint-mode readback and API-vs-numerical Jacobian deltas, so inspect those before changing torque gains.
- Prefer an internal CoppeliaSim add-on for the very first rendering smoke test.
  - The add-on can load the UR5 model, create the camera, and save a PNG sequence with no external RPC client.
  - Use RPC only when an external process must control or inspect the simulator.
- Keep the controller out of first-pass smoke tests.
  - A video/rendering failure should not be mixed with torque-control debugging.
- Reuse the portable controller core for simulator integration work.
  - Avoid coupling new work to MuJoCo-only helper logic when targeting CoppeliaSim.
- Prefer reproducible scripts over one-off shell history.
  - Every working launch path should live in `simulation/` or another obvious repo path.
- Assume HPC display behavior is fragile.
  - Add explicit readiness checks.
  - Fail fast if the display server or simulator is not alive.
  - Do not allow Python clients to hang forever waiting for a missing RPC server.

## Headless / HPC Lessons Learned

- `ffmpeg` is available on the system and is the preferred video writer.
- Avoid adding heavy Python imaging dependencies unless truly needed.
  - A prior attempt to add `imageio` / `Pillow` hit quota issues.
- Use `ffmpeg` raw RGB piping for MP4 generation when possible.
- CoppeliaSim vision-sensor image buffers are returned left-to-right and bottom-to-top.
  - When saving PNGs from Lua, call `sim.transformImage(img, res, 4)` before `sim.saveImage(...)` so frames are upright.
  - When decoding from Python, keep `np.flipud(...)` before sending frames to ffmpeg or image viewers.
- CoppeliaSim should be run in emulated headless mode with `-h` for vision-sensor work.
  - Avoid true headless `-H` unless there is a specific reason and the rendering path is understood.
- When `xvfb-run` is available, prefer it for this smoke test.
  - Only fall back to raw `Xvfb` with `COPPELIA_USE_RAW_XVFB=1` if the wrapper itself is the problem.
- ZMQ port used so far:
  - `23000`
- Reliable no-RPC smoke loader pattern:
  - Keep the source add-on in `simulation/ur5_video_smoke_addon.lua`.
  - The launcher copies that file into `COPPELIA_ROOT/addOns/ur5_video_smoke_addon.lua` only for smoke runs.
  - The smoke add-on is opt-in only: `REAL_CARTPOLE_ENABLE_VIDEO_SMOKE=1` enables capture.
  - RPC/controller launches remove the copied smoke add-on so it cannot interfere with torque-control startup.
  - Export `COPPELIA_ROOT` into the CoppeliaSim process environment so the Lua source can locate the official `UR5.ttm` model during smoke runs.
  - Let CoppeliaSim auto-load the add-on from `addOns/` for smoke runs instead of passing an external absolute `-a` path.
  - `simulation/ur5_video_smoke_addon_shim.lua` and `simulation/ur5_video_smoke_startup.lua` are compatibility shims only.
  - This avoided the "script never loads, markers stay missing" failure mode.

## Known Findings From 2026-04-24

- The official Ubuntu-matched CoppeliaSim build `V4.10.0 rev0` was downloaded and unpacked successfully.
- The official `UR5.ttm` model loads successfully through the external ZMQ client path.
- A direct adapter smoke test already succeeded for:
  - joint state reads
  - EE pose reads
  - Jacobian reads
- The no-RPC Lua video smoke path produced visible robot-arm frames, but raw frames initially appeared upside down.
  - Root cause: CoppeliaSim's raw vision-sensor buffer origin is bottom-left.
  - Fix: `simulation/ur5_video_smoke_addon.lua` now flips the y-axis with `sim.transformImage(img, res, 4)` before saving.
- The no-RPC video smoke launcher now successfully encoded a 40-frame, 2-second MP4 from upright PNG frames.
  - Last verified command: `bash simulation/launch_coppeliasim_video_smoke.sh`
- The smoke scene is now cube-free again.
  - The rendered frame shows the UR5 arm only, with no calibration object.
- The smoke path now uses only the repo-local runtime tree under `third_party/coppelia_runtime/`.
  - The old external backing-root fallback is gone.
- Before the local-runtime copy was put in place, the smoke launcher stalled before the Lua smoke script wrote its first marker.
  - Observed failure signature: `No frames were captured.` with all three markers missing.
  - That issue is now resolved by the repo-local runtime tree and removal of the external backing-root fallback.
- The recoverable render-only path now uses the direct `simulation/ur5_video_smoke_addon.lua` source copied into auto-loaded `addOns/` startup.
  - The launcher no longer passes `-a` to CoppeliaSim.
  - The Lua source gets `COPPELIA_ROOT` from the launcher environment, so the official UR5 model path stays valid.
  - The add-on lifecycle markers are written under `outputs/control_runs/coppelia_video_smoke_state/`.
  - RPC/controller launches must explicitly disable smoke mode so the auto-loaded add-on does not close the simulator after the frame capture completes.
  - If `-h` exits immediately in controller mode, confirm the add-on is still auto-starting in keepalive mode and loading the default scene.
- The current Jacobian path appears to fall back to a deprecated compute route on this build.
  - This is acceptable for smoke testing, but should be cleaned up later.

## Known Findings From 2026-04-27

- The controller launcher no longer depends on `rpc_bootstrap_ready.txt`.
- The launcher now waits for the RPC port, then starts the Python runner immediately.
- The Python runner now logs startup phases and can stop after a probe-only adapter check.
- The remaining work is proving that the UR5 actually moves under torque control in CoppeliaSim.

## Known Findings From 2026-04-28

- Added and verified a controller-motion visible-video path:
  - `simulation/ur5_controller_video_addon.lua`
  - `simulation/launch_coppeliasim_controller_video.sh`
- The new add-on loads the official UR5 model, drives a bounded joint-space trajectory, captures PNGs inside CoppeliaSim, and encodes MP4 with ffmpeg.
- The verified output is:
  - `demonstration_videos/ur5e_coppeliasim/coppeliasim_ur5_controller_video.mp4`
- The controller-motion video path is separate from the external ZMQ torque-controller path.
  - Use it for visible movement/capture validation.
  - Continue using `simulation/launch_coppeliasim_x_axis_headless.sh` for torque-control bring-up.
- Added and verified an acceleration-transport visible-video path:
  - `simulation/ur5_acceleration_transport_video_addon.lua`
  - `simulation/launch_coppeliasim_acceleration_transport_video.sh`
- Initial attempt to reuse MuJoCo fixed-Z endpoints in Coppelia produced nearly zero Coppelia world-X span because the official Coppelia UR5 axes differ from the MuJoCo UR5e scene.
- A Coppelia-specific shoulder-pan video produced large visible motion, but it had unacceptable Y drift and did not preserve orientation.
- Lifting the whole Coppelia model made the active-origin video readable, but that is invalid because the base must remain on the ground.
- The acceleration path now reports guardrails explicitly; until they pass, the path is diagnostic rather than complete.

## Known Findings From 2026-04-30

- The controller runner now closes the actual `RemoteAPIClient` on probe-only and shutdown exits, which removes the previous silent cleanup bug in `simulation/run_coppeliasim_x_axis_headless.py`.
- This makes the ZMQ lifecycle less brittle, but it does not change the remaining control gap: torque semantics, Jacobian validity, zero-torque dynamics, and tiny X motion are still the real blockers.
- The runner now records joint-mode readback, API-vs-numerical Jacobian differences, zero-torque drift metrics, and bidirectional torque-pulse response summaries.
- The pure unit test `pytest -q tests/test_coppeliasim_adapter.py` passes and covers the new joint-mode / Jacobian summary helpers.
- Replaying `simulation/launch_coppeliasim_acceleration_transport_video.sh` now hits `pythonLauncher.py` with `ModuleNotFoundError: No module named 'zmq'` before the embedded wrapper can start. CoppeliaSim still launches, the `ur5_fixed_z_acceleration_transport_addon.lua` add-on still loads, and ffmpeg still re-encodes the captured frames into `demonstration_videos/ur5e_coppeliasim/coppeliasim_ur5_acceleration_transport.mp4`, but the JSON summary ends `success=false` with `ik_failed` and `x_tracking_error`. Treat the embedded `pythonLauncher.py` deps (`zmq`, `cbor2`) as a separate blocker from the Lua capture path.
- The direct-torque Lua add-on experiment (`simulation/ur5_mujoco_like_y_torque_addon.lua`) still does not advance past the first sensing tick at `t=0`. `sysCall_init` starts cleanly and the path plan is valid, but the callback loop never progresses to additional frames in this add-on form. Treat this as an add-on-execution-model blocker, not a controller-gain issue.

## Xvfb / Display Caveats

- Raw `Xvfb` startup has been unreliable in this environment.
- A direct raw test failed with:
  - `Fatal server error: Could not write pid to lock file in /tmp/.tX180-lock`
- Existing working X servers on this node have been observed from `xvfb-run`, not from the raw `Xvfb` launcher path.
- Practical implication:
  - Treat `xvfb-run` as the more trustworthy display wrapper until the raw lock-file issue is fully understood.
- The containerized external-capture path uses the narrow XKB/Xvfb bind set in `simulation/launch_coppeliasim_x_axis_offscreen_capture_container.sh`.
  - Do not bind the full host `/lib/x86_64-linux-gnu` into the container; that caused libc mismatches during bring-up.
- For the current video smoke test, `xvfb-run` wraps CoppeliaSim while the Lua add-on does the capture.
  - That keeps the display path and the capture path separate from controller/RPC work.
  - If `nvidia-smi -L` reports `No devices found`, do not treat GPU-backed validation on that shell as meaningful.

## Debugging Checklist For CoppeliaSim Smoke Tests

1. Confirm the repo-local CoppeliaSim anchor and pydeps anchor exist under `third_party/`.
2. Confirm `coppeliaSim.sh` exists under `third_party/coppelia_runtime/CoppeliaSim_Edu_V4_10_0_rev0_Ubuntu24_04`.
3. Confirm the official model exists at `third_party/coppelia_runtime/CoppeliaSim_Edu_V4_10_0_rev0_Ubuntu24_04/models/robots/non-mobile/UR5.ttm`.
4. Confirm Python can import `coppeliasim_zmqremoteapi_client`.
5. Confirm the display wrapper stays alive before starting the Python client.
6. Confirm CoppeliaSim stays alive after launch.
7. Confirm the ZMQ RPC port opens before attempting `RemoteAPIClient`.
8. Only then run the smoke or controller script.
9. If the Python client hangs, inspect display and simulator processes first before changing controller code.
10. For the no-RPC smoke test, verify PNG frames are written under `outputs/control_runs/coppelia_video_smoke_frames/` before trying to debug video encoding.

## Verified Acceleration Transport Mapping

- The current verified launcher is:
  - `bash simulation/launch_coppeliasim_acceleration_transport_video.sh`
- Latest successful summary:
  - `success=true`
  - `base_on_ground=true`
  - `x_tracking_ok=true`
  - `single_axis_y_ok=true`
  - `fixed_z_ok=true`
  - `orientation_ok=true`
- The useful motion is still a short diagnostic transport, not a full end-to-end sweep:
  - `target_dx_m=0.01`
  - `x_net_displacement_m≈0.009314`
- The observed Coppelia attachment proxy for this UR5.ttm build is the tool-frame target rotated +90 deg about local Z.
  - Use the logged `ee_start_world_matrix` / `ee_final_world_matrix` to keep the attachment-frame check aligned with the real Coppelia axes.

## Transport Sign Rules

- The point-to-point acceleration reference already returns a signed
  world-axis acceleration.
- Do not negate `target_axis_accel` again in `simulation/run_coppeliasim_x_axis_headless.py`
  unless the reference generator itself changes sign convention.
- If the motion goes the wrong way, inspect the reference generator and the
  task-frame mapping before changing controller gains or safety thresholds.
- `all_task_feasible=true` from the surrogate solver is only a feasibility
  hint. It does not mean the live Coppelia summary will pass safety, fixed-axis,
  orientation, or torque-saturation checks.
- The live stepping rate reported by the runner is `sim_dt_s = 0.05`. Do not
  tune the Cartesian impedance gains as if the loop were 100 Hz unless the live
  summary explicitly confirms that rate.
- If `target_axis_accel` is still zero in the trace when the run fails, the
  transport move never started. Do not attribute the final X displacement to the
  transport sign or the point-to-point profile until the first nonzero transport
  command has actually executed.
- When a run fails before the transport window opens, reduce the settle window
  or inspect the pre-motion posture torques before changing the transport solver
  sign or gains.

## Gravity Compensation Calibration (2026-06-25)

- **MuJoCo bias at scale=1.0 is optimal.** Multi-joint probe shows dz=+0.007m over 50 steps.
- **Scale=1.5 is catastrophic.** Multi-joint probe shows dz=-0.754m (arm collapses).
- Single-joint probes are misleading because other joints compensate. Always test with all 6 joints.
- CoppeliaSim UR5.ttm total mass ≈ 41.1 kg (vs MuJoCo UR5e ≈ 21.3 kg).
- Native RNEA gravity computation from extracted link masses was attempted and rejected (compound shapes produce incorrect torques).
- `sim.getJointForce()` returns 0.0 in dynamic torque mode on this build. Do not use for gravity learning.
- The runner now accepts `--gravity-scale` (default 1.0) and `--cartesian-z-kp/kd/ki` for Cartesian Z feedback via J^T.
- The warmup hold gains are now kp=300, kd=40 (from kp=100, kd=20) to reduce steady-state gravity model offset.

## WSL Y-Transport Launcher

- `simulation/run_torque_y_transport_wsl.sh` is the primary launcher on WSL.
- Key new env vars: `GRAVITY_SCALE`, `CART_Z_KP`, `CART_Z_KD`, `CART_Z_KI`.
- The launcher uses `simulation/env_wsl_local.sh` to resolve CoppeliaSim and Python paths.
- Documentation: `docs/coppeliasim/TORQUE_DIAGNOSTICS.md` (includes CoppeliaSim usage guide).

## RL Y-Transport (PPO, 2026-06-25)

- **Docs:** `docs/coppeliasim/RL_Y_TRANSPORT.md`
- **Code:** `rl/coppelia_y_transport_env.py`, `rl/train_ppo.py`, `rl/eval_policy.py`, `rl/config.yaml`
- **Train:** `bash simulation/launch_rl_training_wsl.sh` (env `TIMESTEPS=500000` for shorter runs)
- **Smoke:** `bash simulation/run_rl_smoke_test.sh`
- **Eval:** `bash simulation/run_rl_eval_wsl.sh`
- **Deps (WSL):** `python3 -m pip install stable-baselines3 gymnasium tensorboard`
- **CoppeliaSim launch:** Do **not** use `-h` on WSL — sim exits after ZMQ starts. Launch without scene arg (same as `run_torque_y_transport_wsl.sh`); Python loads `UR5.ttm` over ZMQ.
- **Outputs:** `outputs/rl_logs/`, `outputs/rl_models/ppo_y_transport.zip`, `outputs/rl_eval/eval_summary.json`
- **Why RL:** Model-based gravity + Cartesian Z PID held Z to ~34 mm but could not sustain full Y sweeps; RL learns compensation from state without explicit gravity model.
- **Warm-start:** Residual baseline uses hold PD + MuJoCo `qfrc_bias` gravity (same as model-based path); see `rl/baseline_controller.py`.

## Do Not Recreate These Gravity Bugs

- Do not set `gravity_scale` to 1.5 or 1.8 based on single-joint probes. Always test multi-joint.
- Do not use native RNEA gravity computed from `sim.getShapeMass()` + `sim.getShapeInertia()` — the CoppeliaSim UR5.ttm has compound shapes that make simple mass-weighted COM computation inaccurate.
- Do not rely on `sim.getJointForce()` in dynamic torque mode — it returns 0.0 on this build.
- Do not add gravity compensation twice. The QP controller adds it internally; the runner should add it only in the IK PD / warmup paths.

## Update Rule

Whenever a new reliable command, launcher, env path, or failure signature is discovered, append it here before moving on.
