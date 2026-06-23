# External ZMQ Controller Bring-Up

This note is for the **live Python/ZMQ UR5 controller path** only.

**Important warning:** a successful Lua-rendered MP4 does **not** validate the
external Python/ZMQ controller. The Lua demo lane can work while the live
controller is still broken.
The old `-h -vscriptinfos` launch shape is known-bad in this environment: it
binds the RPC port but does not let `client.require("sim")` complete.

## Two different lanes

### Lua demo lane

This lane stays inside CoppeliaSim and is useful for visible motion and MP4
capture:

- `simulation/ur5_mujoco_like_y_torque_addon.lua`
- `simulation/launch_coppeliasim_mujoco_like_y_torque_video.sh`

It is a simulator-side render/demo path. It is **not** the live controller.

### Live Python/ZMQ lane

This lane is the actual external controller:

- `controller_core/x_axis_cartesian_impedance.py`
- `simulation/run_coppeliasim_x_axis_headless.py`
- `simulation/launch_coppeliasim_x_axis_headless.sh`
- `ros2_ws/src/ur5_x_axis_controller_ros/ur5_x_axis_controller_ros/coppeliasim_adapter.py`

For the live lane:

- Python owns stepping.
- Python owns simulation start.
- Python owns the torque loop.
- Lua motion must be disabled.
- The known-good startup mode is resident `xvfb-run -a` plain launch without
  `-h` or `-vscriptinfos`.

On this cluster, the live lane needs the repo's Ubuntu 24.04 Singularity image:

```text
/common/users/ss5772/containers/aha_u2404.sif
```

The host shell is too old for the current CoppeliaSim build; if you launch the
binary there, it fails with `GLIBCXX_3.4.32` / `GLIBC_2.38` errors. The
capture launcher now resolves `XVFB_RUN_BIN` and `FFMPEG_BIN` explicitly, so
the display and video tools can be supplied by the container or the bind
mounts instead of relying on a fragile PATH lookup.

### Containerized offscreen capture

For MP4 capture inside Singularity, use the dedicated wrapper:

```bash
cd /common/users/ss5772/real_Cartpole
bash simulation/launch_coppeliasim_x_axis_offscreen_capture_container.sh
```

This wrapper is the top-level entrypoint. It calls the plain external-capture
launcher after applying the narrow, working bind set:

- `/usr/bin/xkbcomp:/usr/bin/xkbcomp`
- `/usr/share/X11/xkb:/usr/lib/X11/xkb`
- `/usr/share/X11/xkb:/usr/share/X11/xkb`
- `/common/home/ss5772/.tmp:/common/home/ss5772/.tmp`
- `LD_LIBRARY_PATH=/common/home/ss5772/.tmp/container_bind_libs`

Do not bind the full host `/lib/x86_64-linux-gnu` into the container; that was
the failure mode that introduced libc mismatches during bring-up.

To run the actual X-transport motion through the same capture lane, set:

```bash
RUNNER_EXTRA_ARGS='--accel-x-transport --accel-torque-policy ik_joint_pd --target-dx 0.005 --duration 3 --settle-duration 1' \
  bash simulation/launch_coppeliasim_x_axis_offscreen_capture_container.sh
```

That keeps the video/capture plumbing separate from the controller mode while
still exercising the live torque path end-to-end.

Current status: this path now reaches the live controller, but the direct
impedance lane still does not clear the safety envelope and the split
`ik_joint_pd` lane is only a near-miss. The container and ZMQ attach path are
working; the remaining blocker is controller behavior and seed/tuning choice,
not launch plumbing.

There is now also a verified ROS 2 legacy positional fallback:

- `controller.family: legacy_xz_transport_pd`
- `ros2_ws/src/ur5_x_axis_controller_ros/config/controller_coppelia_legacy_xz_transport_relaxed.yaml`
- `ros2 launch ur5_x_axis_controller_ros run_controller.launch.py config_path:=...controller_coppelia_legacy_xz_transport_relaxed.yaml`

In the verified run, the bridge resolved the UR5 handles, seeded the known-good
pose, and a single `/target_x` step moved the EE in world X while the relaxed
safety gate stayed green. This is the best ROS 2 proof that the live Coppelia
attach path is healthy on this build.

Most recent signal:

- direct impedance settle-hold probes still fail on `qd`, Z drift, and/or
  torque saturation
- `ik_joint_pd` with a MuJoCo dummy task frame and no settle window is the
  closest near-miss so far
- the best historical near-miss is `ik_accel_probe17`
  - `--accel-torque-policy ik_joint_pd`
  - `--task-frame-mode mujoco_attachment_dummy`
  - `--settle-duration 0`
  - `--target-dx 0.002`
  - `--a-x-max 0.02`
  - `--v-x-max 0.02`
  - it still failed `fixed_axes`, but it kept orientation and joint config
    within bounds and stayed inside the torque envelope

Sign-convention note: the point-to-point acceleration reference already emits
a signed world-axis acceleration. Do not negate `target_axis_accel` again in
the caller unless the reference generator itself changes. The latest caller-side
sign flip was a mistake and is now treated as a debugging lesson, not a model
of the transport convention.

Timing note: in the latest failing probe, the transport command never actually
started before the safety stop because the run was still inside the settle
window. Do not attribute the final X displacement to the transport profile
until the first nonzero `target_axis_accel` has been observed in the trace.

Warmup note: `--accel-x-transport` should not spend multiple steps in a
passive zero-torque warmup. That phase can let the UR5 free-fall into the
`|qd| > 1.5 rad/s` safety stop before the first transport command is issued.
Use the controller hold phase instead of passive torque if you need to settle
the pose before moving.

Current default transport seed: the live runner now starts from the verified
fixed-Z transport pose in
`outputs/control_runs/fixed_z_x_transport_firstpass_z0.540_seed1.json`.
That avoids the older shoulder-side default, which was slightly past the wrist
joint limit and produced a near-singular Jacobian.

Diagnostic torque-cap preset:
`/common/users/ss5772/real_Cartpole/ros2_ws/src/ur5_x_axis_controller_ros/config/controller_after_stable.yaml`.
It keeps the same controller gains but uses the higher after-stable torque
caps. In live testing it improved joint-configuration drift, but it did not
clear the fixed-axis/orientation safety guards, so it remains a diagnostic lane
rather than the baseline.

ROS 2 seeded diagnostic probe:
`/common/users/ss5772/real_Cartpole/ros2_ws/src/ur5_x_axis_controller_ros/config/controller_coppelia_slow_seeded_probe.yaml`.
This is the preset that currently proves the ROS 2 bridge/controller path can
actually move the arm in CoppeliaSim under the seeded startup pose. The bridge
now stops any already-running simulation before seeding, seeds the joints while
simulation is stopped, and only then enables dynamic torque mode. Because this
build can report wrapped revolute angles just beyond `2π`, the diagnostic probe
disables the joint-limit latch.

Working ROS 2 probe sequence:

1. Start CoppeliaSim on `PORT=23000` in the repo-local Ubuntu 24.04 Singularity
   image.
2. Load `UR5.ttm` into the default scene.
3. Start `coppeliasim_bridge_node` with the seeded probe config above.
4. Start `controller_node` with the same config.
5. Publish a `std_msgs/msg/Float64` on `/target_x` with an absolute world-X
   target.

Observed result from the current probe:

- initial pose after seeding: about `x=-0.068669`
- target published on `/target_x`: `-0.035361`
- observed EE pose after the run: about `x=0.004001`
- no ROS-side safety stop was triggered during that diagnostic run

## Latest Motion Sweep Results

These are the latest live-motion attempts through the containerized capture
lane. They are useful diagnostics, but they are not success cases.

### Direct impedance sweep

`seeded_probe_cartesian_long1` and `seeded_probe_cartesian_mujoco_neg01` ran
with:

- `--accel-x-transport`
- `--accel-torque-policy cartesian_impedance`
- `--task-frame-mode mujoco_attachment_dummy`
- `--task-orientation-target mujoco`
- seeded probe config

Both runs failed the live envelope. The key failure signature was:

- the end effector moved the wrong way in X relative to the requested step
- `|Y-Y0|` grew to about `0.320 m`
- `|Z-Z0|` grew to about `0.717 m`
- the safety stop triggered on the Y/Z drift envelope
- orientation and joint-configuration checks also failed

### `ik_joint_pd` sweep

`seeded_probe_ikpd_long1`, `seeded_probe_ikpd_long2_neg002`, and
`seeded_probe_ikpd_lowgain_long1` ran with:

- `--accel-x-transport`
- `--accel-torque-policy ik_joint_pd`
- `--task-frame-mode mujoco_attachment_dummy`
- `--task-orientation-target mujoco`
- seeded probe config

These were closer in one respect because orientation stayed locked, but they
still failed on:

- `|Z-Z0| > 0.3 m`
- torque saturation
- joint excursion
- X direction remained inverted in the summary

The lower-gain pass reduced torque saturation to about 20 percent, but it
still failed the same Z-drift and joint-excursion envelope. The best
`ik_joint_pd` run so far reached about `9.1 mm` net X displacement before
failing safety.

### Validation ladder caveat

The older `simulation/run_external_zmq_validation_ladder.sh` and
`simulation/run_hpc_zmq_attach_probe.sh` paths are not the primary live probe
paths on this build. In plain `xvfb_resident_plain` mode the simulator can
exit before attach, even with `dfltscn.ttt`, so use the headless probe or the
offscreen capture wrapper instead.

## Code Map

If you want the code locations in one place:

- Live controller family: `python_zmq_external_cartesian_impedance` in [simulation/external_zmq_controller_common.py](/common/users/ss5772/real_Cartpole/simulation/external_zmq_controller_common.py)
- General Cartesian impedance controller: [controller_core/x_axis_cartesian_impedance.py](/common/users/ss5772/real_Cartpole/controller_core/x_axis_cartesian_impedance.py)
- Live X transport / acceleration runner: [simulation/run_coppeliasim_x_axis_headless.py](/common/users/ss5772/real_Cartpole/simulation/run_coppeliasim_x_axis_headless.py)
- Safety thresholds: [controller_core/safety.py](/common/users/ss5772/real_Cartpole/controller_core/safety.py) (`ImpedanceSafetyConfig`)
- Safety config mirror for ROS 2: [ros2_ws/src/ur5_x_axis_controller_ros/config/controller.yaml](/common/users/ss5772/real_Cartpole/ros2_ws/src/ur5_x_axis_controller_ros/config/controller.yaml)

Current default safety values are:

- `max_abs_y_drift_m = 0.03`
- `max_abs_z_drift_m = 0.03`
- `max_abs_orthogonal_drift_m = 0.03`
- `max_orientation_error_rad = 0.25`
- `max_joint_velocity_radps = 1.5`
- `max_x_error_growth_steps = 100`
- `max_axis_error_growth_steps = 100`
- `q_lower = -2π`, `q_upper = +2π`

## Ownership model

The live controller summary should make the ownership explicit:

- `controller_family = python_zmq_external_cartesian_impedance`
- `uses_direct_torque_control = true`
- `stepping_owner = python_zmq`
- `simulation_started_by = python`
- `lua_motion_enabled = false`

Marker-file handoff is kept only for compatibility and diagnostics. It is not
the default live path.

## Diagnosing HPC ZMQ attach failures

This is the first diagnostic to run when the live external lane behaves
strangely on a scheduler node.

Important rules:

- A ZMQ port being open does **not** prove that `require("sim")` will attach.
- The old `-h -vscriptinfos` path can bind the RPC port without servicing the
  first request in this environment.
- On HPC, `127.0.0.1` means the current compute node only.
- Python and CoppeliaSim must run inside the same scheduler allocation and
  preferably on the same node.
- Check both `rpcPort` and `cntPort`.
- `client.require("sim")` is the real attach test.
- Do not debug impedance gains if attach-only fails.
- If attach-only passes but zero-torque stepping fails, debug Python-owned
  stepping/startup.
- If zero-torque stepping passes but single-joint torque fails, debug joint
  dynamic mode / force-torque mode / torque API path.
- If single-joint torque passes, then debug Cartesian impedance.

Validation ladder:

1. HPC/ZMQ attach-only probe.
2. External ZMQ zero-torque stepping probe.
3. External ZMQ single-joint torque probe.
4. Cartesian impedance controller.

Do not treat the `ik_joint_pd` transport surrogate as a final controller
validation. It can report a feasible joint solve while the live Coppelia run
still violates the fixed-axis or orientation guardrails. The final success
criterion is the direct impedance lane passing the runtime summary checks.

Troubleshooting table:

| Symptom | Likely layer | Likely cause | Next action |
| --- | --- | --- | --- |
| RPC port closed | HPC/ZMQ attach | CoppeliaSim never started or crashed before listen | Check the Coppelia log and relaunch cleanly |
| RPC port open but `require("sim")` fails | HPC/ZMQ attach | ZMQ server is reachable but not attachable yet | Run `simulation/probe_hpc_zmq_attach.py` and inspect the attach summary |
| `require("sim")` works but stepping fails | Python-owned stepping/startup | Wrong ownership of `sim.startSimulation()` or `sim.step()` | Fix the live startup path before tuning torque |
| Stepping works but torque does not move joint | Joint dynamics / torque API | Wrong dynamic mode, motor mode, or force-torque path | Debug joint mode and direct-torque application |
| Lua demo works but external controller fails | Lane separation | Demo lane succeeded, live lane still broken | Keep the Lua MP4 separate from the live controller path |

## Bring-up order

Do not jump straight to Cartesian impedance. Prove the stack in order:

1. HPC/ZMQ attach-only probe.
2. Zero-torque stepping.
3. Tiny single-joint torque displacement.
4. Cartesian impedance only after both probes pass.

## Probe commands

### 1. HPC/ZMQ attach-only probe

This proves the remote API can be attached cleanly from the current HPC
environment before any stepping or torque is attempted.

```bash
cd /common/users/ss5772/real_Cartpole
bash simulation/run_hpc_zmq_attach_probe.sh
```

By default this now uses resident `xvfb-run -a` launch mode. The legacy
`-h/-vscriptinfos` path is reserved only for explicit diagnostics.

The attach-only diagnostic writes a summary under:

```text
outputs/control_runs/hpc_zmq_attach/hpc_zmq_attach_summary.json
```

### 2. Zero-torque stepping probe

This proves that Python can connect, enable stepping, start simulation, and
advance time while keeping torques at zero.

```bash
cd /common/users/ss5772/real_Cartpole
bash simulation/run_external_zmq_probes.sh
```

The probe runner also uses resident `xvfb-run -a` launch mode by default and
refuses to silently fall back to the broken headless path.

The zero-torque probe writes a summary under:

```text
outputs/control_runs/external_zmq_handshake/summary.json
```

### 2. Tiny single-joint torque probe

This proves that direct torque commands on one joint produce a measurable
response without needing a full controller loop.

The second probe writes a summary under:

```text
outputs/control_runs/external_zmq_single_joint_torque/summary.json
```

Acceptance for the probe pair is:

- attach-only summary reports `success=true`, `require_sim_ok=true`, and
  `get_simulation_state_ok=true`
- handshake summary reports `success=true`
- single-joint summary reports `success=true`
- the joint-0 displacement exceeds the configured minimum threshold

## Live controller startup

The default live controller path is Python-owned and should not rely on the
old release-marker handoff.
It also starts CoppeliaSim in resident `xvfb-run -a` mode by default.

If you use the video/capture wrapper instead of the bare torque launcher,
prefer `simulation/launch_coppeliasim_x_axis_offscreen_capture.sh`. It uses a
time-based frame cadence and is the current wrapper for external-controller
MP4 capture.

```bash
cd /common/users/ss5772/real_Cartpole
bash simulation/launch_coppeliasim_x_axis_headless.sh --probe-only --no-video
```

The live controller should print these startup lines:

```text
LIVE_EXTERNAL_CONTROLLER=1
controller_family=python_zmq_external_cartesian_impedance
stepping_owner=python_zmq
simulation_started_by=python
lua_motion_enabled=false
legacy_marker_handoff=false
```

If you must use the old release-marker choreography, pass:

```bash
--legacy-marker-handoff
```

That mode exists only for compatibility and debugging.
For a legacy runtime diagnosis, `--legacy-headless` may be used only as an
explicit fallback. It is known to fail attach in this environment and should
not be the default.

## After the probes pass

Only after both probes are clean should you run the Cartesian impedance
controller path.

Suggested next step:

```bash
cd /common/users/ss5772/real_Cartpole
bash simulation/launch_coppeliasim_x_axis_headless.sh \
  --no-video \
  --task-frame-mode mujoco_attachment_dummy
```

If that run is not boring and deterministic, keep debugging the startup and
handshake path before touching gains.
