# CoppeliaSim RPC Controller System

This document explains the current controller/RPC system as it exists now.

## Purpose

The goal is to run the existing UR5 single-axis torque controller against the official CoppeliaSim `UR5.ttm` model.

When debugging on HPC, run `simulation/run_hpc_zmq_attach_probe.sh` first; a listening port alone does not prove that `require("sim")` will attach cleanly.

On this cluster, the current CoppeliaSim build needs the repo's Ubuntu 24.04 Singularity image:

```text
/common/users/ss5772/containers/aha_u2404.sif
```

The host shell is not sufficient for this CoppeliaSim binary. If you launch it directly on the host, it fails with `GLIBCXX_3.4.32` / `GLIBC_2.38` library mismatches.

The controller path intentionally avoids ROS for first-pass debugging:

```text
CoppeliaSim
-> ZMQ Remote API
-> Python adapter
-> controller_core torque controller
-> torque commands back to CoppeliaSim
```

ROS 2 remains useful later, but it is not the fastest path for isolating the current CoppeliaSim/RPC issues.

## Main Files

| File | Role |
| --- | --- |
| `simulation/launch_coppeliasim_x_axis_headless.sh` | Starts CoppeliaSim in the **background** under a resident display wrapper, waits for the ZMQ port to listen, and then runs the Python runner in the foreground. The default path is Python-owned stepping; `--legacy-marker-handoff` restores the older Lua release-marker bootstrap only for compatibility. |
| `simulation/launch_coppeliasim_x_axis_offscreen_capture.sh` | Video-capable wrapper for the live ZMQ lane. Uses the time-based external capture add-on, resolves `XVFB_RUN_BIN` / `FFMPEG_BIN`, forwards `RUNNER_EXTRA_ARGS` to the Python runner, and encodes the captured PNGs to MP4. |
| `simulation/launch_coppeliasim_x_axis_offscreen_capture_container.sh` | Singularity top-level wrapper for the live ZMQ capture lane. Applies the verified narrow XKB/Xvfb bind set and then calls the plain external-capture launcher. |
| `simulation/run_coppeliasim_x_axis_headless.py` | ZMQ client: load `dfltscn.ttt` + `UR5.ttm`, `setStepping`, zero-torque warmup, torque mode, control loop, optional video via vision sensor. |
| `simulation/launch_coppeliasim_video_smoke.sh` | **Separate** pipeline: Lua add-on, PNG capture, MP4. Not used when launching the controller. |
| `simulation/ur5_video_smoke_addon.lua` | Installed by the smoke launcher only. **Deleted** from `addOns/` before a controller run. |
| `ros2_ws/.../coppeliasim_adapter.py` | Handles, state, Jacobian, torques. |
| `controller_core/x_axis_cartesian_impedance.py` | Control law. |
| `config/controller.yaml` | Gains, paths, safety. |

See [PIPELINES.md](PIPELINES.md) and [COPPELIASIM_VISION_NOTES.md](COPPELIASIM_VISION_NOTES.md) for how Coppelia expects `createVisionSensor` / `handleVisionSensor` / `getVisionSensorImg` to be used.

## Current Launch Sequence

```bash
bash simulation/launch_coppeliasim_x_axis_headless.sh   # pass-through args to the Python runner, e.g. --torque-pulse --no-video
```

1. Unsets smoke: remove `ur5_video_smoke_addon.lua` and keepalive shims from `COPPELIA_ROOT/addOns/`.
2. `export REAL_CARTPOLE_ENABLE_VIDEO_SMOKE=0` (so an auto-copied smoke add-on would not be used).
3. Start `coppeliaSim.sh` in the **background** under the resolved display wrapper, without `-h` or `-vscriptinfos`, and wait for the RPC port to listen.
4. Start `run_coppeliasim_x_axis_headless.py` in the foreground. The default path connects directly, sets stepping, resolves handles, starts simulation, and then runs a short zero-torque warmup before the controller loop.
5. Legacy marker handoff is available only when `--legacy-marker-handoff` is passed; in that mode the older Lua release-marker bootstrap is used for compatibility.
6. Run `--probe-only`, `--torque-pulse`, or the main impedance loop; optional **MP4** using `--video-camera smoke` (default) or `ee`.

**Smoke (different shell script):** `bash simulation/launch_coppeliasim_video_smoke.sh` — Lua/PNG/MP4 only. Does not replace the controller.

If you are already inside the Singularity image, prefer
`simulation/launch_coppeliasim_x_axis_offscreen_capture_container.sh` as the
top-level wrapper for offscreen capture. It supplies the bind set that the
plain launcher expects and avoids hand-assembling XKB/Xvfb mounts.

## Current Python Runner Behavior

1. `load_yaml_config` → `CoppeliaSimConfig`, connect after the launcher has already opened the port (see [COPPELIASIM_VISION_NOTES](COPPELIASIM_VISION_NOTES.md) for `RCVTIMEO` after connect).
2. `sim.loadModel` (UR5) if not present.
3. `make_vision_sensor` (unless `--no-video`): `createVisionSensor(1|2|4, …)` and `setExplicitHandling(1)`.
4. Optional `set_ur5_joints_for_video_framing` for a known-good on-screen arm pose when recording.
5. `setStepping`, `startSimulation`, adapter probe / controller.
6. Each video frame: place camera (`--video-camera smoke|ee`) → `sim.step` → `handleVisionSensor` → `getVisionSensorImg` → decode (bytes or list).
7. `write_video_ffmpeg` via raw RGB to libx264.
8. JSONL + summary; optional `first_frame_std_rgb` sanity check.

The runner now prints startup phases and stores them in the summary JSON so slow startup is easier to diagnose.

## Adapter Responsibilities

`coppeliasim_adapter.py` is the bridge between the simulator and the controller.

It owns:

- ZMQ RPC connection.
- UR5 joint handle resolution.
- EE object resolution.
- Joint force/torque mode setup.
- Joint position and velocity reads.
- EE pose and twist reads.
- Jacobian reads through `sim.getJacobian`.
- Numerical Jacobian fallback.
- Torque command application.

Torque application currently prefers:

```text
sim.setJointTargetForce(handle, tau, true)
```

If that fails, it falls back to:

```text
sim.setJointTargetVelocity(handle, sign * large_velocity)
sim.setJointMaxForce(handle, abs(tau))
```

## Controller Responsibilities

`controller_core/x_axis_cartesian_impedance.py` owns the control law.

At reset, it stores:

- initial X/Y/Z,
- initial EE orientation,
- initial joint posture.

Each compute cycle:

- tracks desired X,
- holds initial Y and Z,
- holds initial orientation,
- adds posture control,
- adds joint damping,
- optionally includes gravity torque if supplied,
- scales near singular Jacobians,
- clips torque to configured limits.

## Current Outputs

Trace and summary outputs are written under:

```text
outputs/control_runs/
```

Current default controller outputs:

```text
outputs/control_runs/coppeliasim_ur5_x_impedance_headless.jsonl
outputs/control_runs/coppeliasim_ur5_x_impedance_headless_summary.json
demonstration_videos/ur5e_coppeliasim/coppeliasim_ur5_x_impedance_headless.mp4
```

## Why the launcher is shaped this way

- **Python-owned stepping is the default.** The launcher starts CoppeliaSim in the **background**, Python connects directly, enables stepping, and owns the controller loop. The older keepalive add-on and RPC release-marker choreography remain only behind `--legacy-marker-handoff`.
- **No** `-s300000` one-shot: Python owns `sim.startSimulation` / `sim.stopSimulation` over RPC.
- **No** separate Lua for controller logic: single owner = `run_coppeliasim_x_axis_headless.py`.

## Current lifecycle

```text
xvfb/runtime wrapper + CoppeliaSim (background) + ZMQ
  -> launcher waits for port
  -> Python connects, setStepping, startSimulation
  -> zero-torque warmup
  -> optional vision sensor + torque loop
  -> trace / summary / optional MP4
```
