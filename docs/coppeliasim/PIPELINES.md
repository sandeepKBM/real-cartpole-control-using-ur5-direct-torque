# CoppeliaSim pipelines: smoke vs ZMQ controller

Two different paths serve two different goals. They are both valid; do not assume they share the same code.

## A. Render-only smoke (known-good pixels)

**Goal:** prove Coppelia, Xvfb, UR5 model load, and **visible** offscreen OpenGL, without the ZMQ control loop.

| Step | What runs |
|------|------------|
| Script | `simulation/launch_coppeliasim_video_smoke.sh` |
| Coppelia | Loads a **Lua** add-on copied to `COPPELIA_ROOT/addOns/ur5_video_smoke_addon.lua` |
| Capture | Lua reads the scene **inside the simulator**, writes **PNGs** under `outputs/control_runs/coppelia_video_smoke_frames/` |
| Encode | Shell runs **ffmpeg** on PNGs → `demonstration_videos/ur5e_coppeliasim/coppeliasim_ur5_video_smoke.mp4` |
| ZMQ | Not required for the baseline capture (optional alternative: `run_coppeliasim_video_smoke.py` uses ZMQ but uses the same camera/joint idea as the Lua path) |

**When to use:** baseline “arm visible in frame,” CI smoke, or display/GPU node checks.

## B. ZMQ controller (torque / Cartesian impedance)

**Goal:** one Python process owns `sim.step()`, reads joint/EE state, **applies torques** from `controller_core`, and optionally records **MP4** from a vision sensor over the **remote API**.

| Step | What runs |
|------|------------|
| Script | `simulation/launch_coppeliasim_x_axis_headless.sh` |
| Process order | The shell starts **CoppeliaSim** in the **background** in resident plain mode, usually wrapped by `xvfb-run -a` when no trusted display is available, waits for the ZMQ port to listen, then runs the Python ZMQ runner in the **foreground**. |
| Add-ons | **Smoke** Lua is **removed** from `addOns/` so it cannot stop the sim after a capture. |
| Handshake | **Default:** Python connects directly, sets stepping, resolves handles, starts simulation, runs a short zero-torque warmup, and then enters the controller. The known-bad `-h/-vscriptinfos` launch shape is not used here. **Legacy only:** `--legacy-marker-handoff` restores the old Lua release-marker bootstrap for compatibility. |
| Python | `simulation/run_coppeliasim_x_axis_headless.py` — connects with `RemoteAPIClient`; the default bootstrap is Python-owned stepping, not Lua-owned release handoff. |
| Video (optional) | Creates a **vision sensor**; each frame: set pose → `step` → `handleVisionSensor` → `getVisionSensorImg` (see [COPPELIASIM_VISION_NOTES.md](COPPELIASIM_VISION_NOTES.md)). |
| Camera | `--video-camera smoke` (default) uses the same world trajectory as `run_coppeliasim_video_smoke.py` for a visible arm. `--video-camera ee` looks at the end effector. |

**When to use:** proving RPC, `sim.getJacobian`, torque commands, and logging (`outputs/control_runs/*.jsonl`). This path intentionally does not use Coppelia's `-s` auto-stop flag.

Before using this path, confirm only one simulator owns the selected ZMQ port.
If `outputs/control_runs/coppeliasim_x_axis_headless.log` contains
`Address already in use`, stop the stale simulator or rerun with a free
`PORT=...`; do not tune controller code until the RPC port is clean.
If you see the old `-h/-vscriptinfos` shape in a launch command for the live
external lane, stop and switch back to resident `xvfb-run -a` plain launch.

The current video-enabled external-capture wrapper for this lane is:

```text
simulation/launch_coppeliasim_x_axis_offscreen_capture.sh
```

It uses a time-based capture cadence in `simulation/ur5_external_controller_capture_addon.lua` so frame spacing stays stable even when simulator timing jitters. The launcher resolves `xvfb-run` via `XVFB_RUN_BIN` or the bundled temp-path fallback and resolves `ffmpeg` via `FFMPEG_BIN` or the system path. Run it inside the repo's Ubuntu 24.04 Singularity image, not the host shell.

If you are already inside the Singularity image, use the container wrapper as the top-level entrypoint:

```text
simulation/launch_coppeliasim_x_axis_offscreen_capture_container.sh
```

It injects the verified narrow XKB/Xvfb bind set before calling the plain external-capture launcher. The working bind set is intentionally narrow:

- `/usr/bin/xkbcomp:/usr/bin/xkbcomp`
- `/usr/share/X11/xkb:/usr/lib/X11/xkb`
- `/usr/share/X11/xkb:/usr/share/X11/xkb`
- `/common/home/ss5772/.tmp:/common/home/ss5772/.tmp`
- `LD_LIBRARY_PATH=/common/home/ss5772/.tmp/container_bind_libs`

The launcher also forwards `RUNNER_EXTRA_ARGS` to the Python runner. That is how you ask the capture lane to execute the actual X-transport controller instead of the default step/probe mode.

## C. Fixed-Z acceleration video (Lua/IK, guarded diagnostic)

**Goal:** test whether Coppelia can reproduce the MuJoCo-style acceleration
transport contract: X motion only, fixed height, fixed end-effector direction,
and base on the ground.

| Step | What runs |
|------|------------|
| Script | `simulation/launch_coppeliasim_acceleration_transport_video.sh` |
| Coppelia | Loads `simulation/ur5_fixed_z_acceleration_transport_addon.lua` as an auto-start Lua add-on |
| Origin | Verified MuJoCo fixed-Z transport seed from `outputs/control_runs/fixed_z_x_transport_firstpass_z0.540_seed1.json` by default; override with `Q_START_RAD` only for diagnostics |
| Path | Lua solves fixed-Y/Z/orientation IK waypoints along Coppelia world X |
| Controller | Acceleration-limited scalar X profile over the solved joint path |
| Capture | Lua writes PNGs under `outputs/control_runs/coppelia_acceleration_transport_frames/` |
| Metrics | JSON at `outputs/control_runs/coppelia_acceleration_transport_state/coppeliasim_ur5_acceleration_transport_summary.json` |

Current diagnostic defaults: `TARGET_DX_M=0.12`, `MOVE_DURATION_S=2.0`,
`V_X_MAX_MPS=0.35`, `A_X_MAX_MPS2=1.2`, `IK_WAYPOINTS=72`,
`MODEL_BASE_Z_OFFSET_M=0.0`.

This is not ZMQ torque parity. It is also not accepted just because it writes a
video. Acceptance requires `success=true` in the summary JSON. See
[MUJOCO_ACCELERATION_GUARDRAILS.md](MUJOCO_ACCELERATION_GUARDRAILS.md).

## D. Origin-to-acceleration video (Lua/IK, known-good first stage)

**Goal:** prove the pre-transport behavior and a bounded first acceleration
move: from an offset/randomized joint state, recover the grounded reference
origin pose and end-effector facing direction, pause, scan the reachable
Coppelia world-X range, then accelerate in one direction inside that range.

| Step | What runs |
|------|------------|
| Script | `simulation/launch_coppeliasim_origin_acquisition_video.sh` |
| Coppelia | Loads `simulation/ur5_origin_acquisition_video_addon.lua` as an auto-start Lua add-on |
| Origin | Grounded Coppelia reference pose at `EE_TARGET_Z_M=0.65` by default; override with `ORIGIN_Q_RAD` only for diagnostics |
| Start | Deterministic offset from the origin by default; override with `START_Q_RAD` for randomized probes |
| Controller | Lua position/IK stage that drives the end effector to the target pose, holds for 1 second, scans reachable X, and runs one-direction acceleration |
| Capture | Lua writes PNGs under `outputs/control_runs/coppelia_origin_acquisition_frames/` |
| Metrics | JSON at `outputs/control_runs/coppelia_origin_acquisition_state/coppeliasim_ur5_origin_acquisition_summary.json` |

Latest verified metrics:

```text
success=true
base_on_ground=true
requested_target_dx_m=0.060
target_dx_m=0.020
target_dx_was_clamped_by_range_scan=true
x_reachable_min_m=-0.115575
x_reachable_max_m=-0.075575
acceleration_final_position_error_m=0.000508
acceleration_final_orientation_error_deg=2.784
peak_abs_ee_vx_mps=0.149
```

This is still a simulator-side Lua/IK position-setpoint path, not ZMQ torque
parity. Its purpose is to make the first behavior precise and measurable, and
to avoid commanding an unreachable X displacement before the broader fixed-Z
acceleration controller is attempted.

## E. Lua direct-torque probe (internal diagnostic)

**Goal:** verify whether the UR5 can move under direct joint torque commands
inside CoppeliaSim, with optional frame or MP4 capture.

| Step | What runs |
|------|------------|
| Script | `simulation/launch_coppeliasim_lua_direct_torque_probe_video.sh` |
| Coppelia | Loads `simulation/ur5_lua_direct_torque_probe_addon.lua` as an auto-start Lua add-on |
| Control | Lua configures dynamic torque-capable joint mode if available, then applies a small direct torque to joint 0 |
| Capture | Lua writes PNGs under `outputs/control_runs/lua_direct_torque_probe/frames/` |
| Metrics | JSON at `outputs/control_runs/lua_direct_torque_probe/lua_direct_torque_probe_summary.json` |

This is **not** the external Python/ZMQ controller. A successful Lua torque
probe only proves that the simulator-side direct-torque lane can move the UR5
and, if rendering works, can produce a visual artifact. It does not validate
ZMQ attach, Python-owned stepping, or the external Cartesian impedance path.

### F. Lua acceleration-direction torque lane

**Goal:** keep the direct-torque proof internal to CoppeliaSim while exposing
acceleration direction as the only required user-facing input for Y-axis motion.

| Step | What runs |
|------|------------|
| Script | `simulation/launch_coppeliasim_lua_direct_torque_probe_video.sh` |
| Mode | `LUA_TORQUE_MODE=y_axis_accel_direction` |
| Input | `ACCEL_DIRECTION` chooses +Y versus -Y motion |
| Defaults | `ACCEL_MAGNITUDE_MPS2`, `TRAVEL_DISTANCE_M`, and torque limits are internal or optional overrides |
| Control | Lua computes direct joint torques from a Jacobian-based Y-axis task force when available |
| Capture | Lua writes PNGs under `outputs/control_runs/lua_direct_torque_probe/` and may encode MP4 if frames exist |
| Metrics | JSON at `outputs/control_runs/lua_direct_torque_probe/.../lua_direct_torque_probe_summary.json` |

This lane is still simulator-side Lua control. It does **not** validate the
external Python/ZMQ controller. It is useful because it separates the direct
torque question from the ZMQ attach/runtime question.

## Why two pipelines?

- **Smoking pixels through Lua/PNG** avoids all remote-API image marshalling issues; it is the most reliable *picture*.
- **Origin acquisition Lua/IK** is the accepted first-stage behavior before acceleration transport.
- **Fixed-Z acceleration Lua/IK** is a guarded diagnostic path until its summary reports `success=true`.
- **Controller work** must use ZMQ/ RPC for forces and state; **video** there is a bonus and must follow Coppelia’s `handle` → `get` contract and buffer decoding (see vision notes).
- **Bring-up order** for the live lane: HPC/ZMQ attach-only probe, zero-torque stepping probe, tiny single-joint torque probe, then Cartesian impedance.
- **Lua direct torque probe** is a separate simulator-side diagnostic; it can validate direct torque motion and rendering without touching the external ZMQ lane.

## Suggested “success” checklist

1. **Smoke MP4** looks correct (arm + scene).
2. **Resident attach-only** probe succeeds, then `--probe-only` completes and resolves handles.
3. **Resident** `--torque-pulse` moves joints (see `delta_q` in summary) and, with `--video-camera smoke`, `first_frame_std_rgb` in the JSON summary is **not** near zero.
4. Full run with `--target-dx` and bounded errors when tuning allows.
5. Lua direct-torque probe reports `controller_family = lua_internal_direct_joint_torque_probe` and, if video succeeds, produces a separate MP4 or frame sequence under `outputs/control_runs/lua_direct_torque_probe/`.
6. Lua acceleration-direction torque mode reports `controller_family = lua_internal_y_axis_accel_direction_direct_torque` and `required_user_inputs = ["ACCEL_DIRECTION"]`.

## Slurm / GPU

OpenGL in logs may show **Mesa llvmpipe** (CPU) or a hardware driver. A Slurm **GPU** node is often used for a stable **display** story on this cluster; it is not a substitute for correct `handle`/`get` order and buffer parsing in pipeline B. See [CURRENT_STATUS.md](../CURRENT_STATUS.md) and root [AGENTS.md](../../AGENTS.md) for node notes.
