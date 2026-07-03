# CoppeliaSim RPC Controller TODO

This is the active task list for getting the UR5 moving in CoppeliaSim with the existing torque controller.

## Guiding Principle

Do not debug everything at once.

The correct order is:

```text
HPC/ZMQ attach-only
-> RPC lifecycle
-> handle/state probe
-> torque semantics
-> Jacobian validity
-> zero-torque dynamics
-> tiny X motion
-> controller tuning
```

## Phase 0: Validate The HPC/ZMQ Attach Path

Current state:

- the attach-only diagnostic exists as `simulation/probe_hpc_zmq_attach.py`,
- the probe runner exists as `simulation/run_hpc_zmq_attach_probe.sh`,
- the probe records host, process, cwd, sys.path, and selected env vars,
- `client.require("sim")` is the real attach test.

Remaining work:

- prove the attach-only diagnostic on the current scheduler environment,
- confirm both `rpcPort` and `cntPort` are reachable and usable,
- confirm the summary makes the HPC failure mode obvious before any torque work.

Acceptance:

- the attach-only summary reports `success=true`,
- `require_sim_ok=true`,
- `get_simulation_state_ok=true`,
- the failure path is clear if `require("sim")` times out.

## Phase 1: Validate The Cleaned RPC Startup Path

Current state:

- the launcher now uses the repo-local CoppeliaSim runtime and backgrounds the simulator in the shell,
- the default controller bootstrap is Python-owned stepping, not a Lua release-marker handoff,
- the controller runner has startup phase logs,
- the controller runner has a `--probe-only` mode,
- the legacy marker-file handshake still exists, but only behind `--legacy-marker-handoff`.

Remaining work:

- keep the attach-only probe as the first step before any torque test,
- prove the cleaned startup path on a real run,
- confirm the simulator stays alive in the non-simulation bootstrap state until the Python runner is ready,
- confirm the runner can attach, start simulation, and load/control the model reliably,
- confirm the probe-only mode is the fastest way to inspect failures.

Tasks:

- Keep the startup phase logs and make sure they show the slowest step.
- Use the probe-only mode to inspect handle resolution, pose, and Jacobian before torque tests.
- Confirm the controller startup remains reliable without the idle-keepalive flow.
- Keep the Lua smoke add-on only for render-only smoke.

Acceptance:

- a failed attach-only probe exits with a clear reason within a bounded timeout,
- a failed RPC launch exits with a clear reason within a bounded timeout,
- a successful launch reaches first state read with the shell-launched Python runner,
- the simulator does not burn through an auto-started `-s` window before the Python client attaches,
- probe-only mode proves the adapter path without applying torque.

## Phase 2: Harden The Read-Only Adapter Probe

Purpose:

Before applying torque, prove that the adapter can read everything needed by the controller.

Tasks:

- Add a `--probe-only` mode to `run_coppeliasim_x_axis_headless.py`.
- Print resolved joint handles and EE handle.
- Read `q`, `qd`, EE pose, EE twist, and Jacobian.
- Validate finite values and expected shapes.
- Write a small probe summary JSON.

Acceptance:

- one command can prove the RPC adapter works without sending torque,
- the probe summary is enough to debug handle or Jacobian failures quickly.

## Phase 3: Verify Torque Semantics

Purpose:

Confirm that CoppeliaSim interprets signed torque commands the way the controller expects.

Tasks:

- Add a tiny single-joint torque test mode.
- Apply very small positive and negative torque to one joint at a time.
- Record joint velocity sign and magnitude.
- Compare `setJointTargetForce(handle, tau, true)` with the fallback large-velocity/max-force method.
- Confirm joints are dynamic, motors are enabled, and position control is disabled.

Acceptance:

- torque sign is known for each joint,
- unstable torque-mode setup is detected before running the Cartesian controller.

## Phase 4: Validate Jacobian Path

Purpose:

The controller depends on `tau = J.T @ wrench`. If the Jacobian is wrong, the controller will push in the wrong direction.

Tasks:

- Compare API Jacobian and numerical Jacobian at the same static pose.
- Check shape, finite values, row ordering, and sign.
- Perturb joint positions and confirm predicted EE delta roughly matches observed EE delta.
- Decide whether `coppeliasim.jacobian.source` should remain `auto`, force `api`, or force `numerical` for now.

Acceptance:

- a documented Jacobian source is selected for first controller motion tests.

## Phase 5: Zero-Torque Dynamics Test

Purpose:

Separate simulator dynamics issues from controller issues.

Tasks:

- Start CoppeliaSim.
- Configure torque mode.
- Apply zero torque for a short duration.
- Record joint drift, EE drift, and velocity.
- Decide whether gravity compensation or stronger posture control is needed before X motion.

Acceptance:

- zero-torque behavior is understood and documented.

## Phase 6: Tiny X Controller Test

Purpose:

Make the smallest meaningful proof that the controller can move the arm along X.

Suggested command after startup cleanup:

```bash
bash simulation/launch_coppeliasim_x_axis_headless.sh \
  --accel-x-transport \
  --accel-torque-policy ik_joint_pd \
  --duration 3 \
  --settle-duration 1 \
  --target-dx 0.005
```

Tasks:

- Start with `target_dx=0.005`.
- Keep conservative torque limits.
- Use `ik_joint_pd` first: the external command is signed world-X acceleration, differential IK solves the reduced-chain joint target, shoulder pan remains locked, and Coppelia receives direct joint torques.
- Confirm `task_frame.mode=mujoco_attachment_dummy` or explicitly override `--task-frame-mode`.
- Inspect `frame_reference`, `local_fixed_z_x_ik_capability`, and `failure_reasons` before tuning gains.
- Watch X error sign and reduction.
- Watch Y/Z drift.
- Watch orientation error.
- Watch joint velocities.
- Watch torque saturation.
- Save JSONL trace and summary.

Acceptance:

- X moves in the commanded direction,
- `uses_direct_torque_control=true` and `uses_position_servo_setpoints=false`,
- `frame_reference_ok=true`,
- `x_tracking_ok=true`,
- `single_axis_y_ok=true`,
- `fixed_z_ok=true`,
- `orientation_ok=true`,
- `joint_configuration_ok=true`,
- `torque_saturation_ok=true`,
- safety does not stop immediately,
- no uncontrolled joint thrash,
- trace gives enough evidence for tuning.

## Phase 7: Tuning After Motion Exists

Only tune gains after the lifecycle, torque semantics, and Jacobian are known.

Rules:

- If height drops, raise `kp_z` before `kp_x`.
- If orientation wanders, raise `kp_rot` before `kp_x`.
- If joints thrash, raise damping/posture or lower Cartesian gains.
- If torques saturate continuously, reduce target size or gains before increasing limits.
- Do not jump from 5 mm to 2 cm until the 5 mm case is boring.

## Open Questions

- Does this CoppeliaSim build's `sim.getJacobian` return the expected world-frame 6x6 Jacobian for `UR5_connection`?
- Does the UR5 model need explicit gravity compensation for stable torque-mode tests?
- Are there hidden child scripts in `UR5.ttm` that can fight torque commands?
- Is `UR5_connection` the best EE object for control, or should a separate tool frame/dummy be created?
- Should the controller path use a saved `.ttt` scene instead of loading `UR5.ttm` directly every time?
