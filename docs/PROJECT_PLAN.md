# Project Plan

Last updated: 2026-04-27

## Current Active Milestone

Make the UR5 move in CoppeliaSim using the existing torque controller.

Success is a small controlled X motion, not a polished demo.

## Milestone 0: Keep Known-Good Smoke Baseline

Status: done

Command:

```bash
bash simulation/launch_coppeliasim_video_smoke.sh
```

Purpose:

- prove CoppeliaSim launches,
- prove the UR5 model loads,
- prove the render path works,
- prove frames and MP4 writing work.

Do not mix this with torque-control debugging.

## Milestone 1: Clean Controller/RPC Startup

Status: active

The launcher/runner startup path has been refactored away from the old marker-file/sleep bootstrap.
What remains is validating that flow end-to-end on the target node and tightening any slow step.

Target:

```text
launcher starts CoppeliaSim without auto-starting simulation
-> bootstrap add-on loads default scene, UR5, spawns Python, and waits for readiness
-> RPC readiness check succeeds
-> Python connects
-> Python starts simulation
-> probe-only mode can inspect the adapter without torque
```

Primary doc:

```text
docs/coppeliasim/RPC_CONTROLLER_TODO.md
```

## Milestone 2: Read-Only CoppeliaSim Adapter Probe

Status: next

Before torque commands, prove:

- all six joint handles resolve,
- EE object resolves,
- joint state reads are finite,
- EE pose/twist reads are finite,
- Jacobian read returns a usable 6x6 matrix.

## Milestone 3: Torque Semantics Probe

Status: next

Before Cartesian impedance control, prove:

- signed torque commands move joints in understandable directions,
- torque magnitudes are not wildly mis-scaled,
- dynamic force/torque mode is actually active,
- position controllers or child scripts are not fighting the command.

## Milestone 4: Zero-Torque Dynamics

Status: next

Run torque mode with zero torque and measure drift.

This tells us whether the first controller run needs gravity compensation, stronger posture hold, a different start pose, or a saved scene with cleaner joint setup.

## Milestone 5: Tiny X Motion

Status: next

Run a very small X offset:

```bash
bash simulation/launch_coppeliasim_x_axis_headless.sh --duration 3 --settle-duration 1 --target-dx 0.005
```

Success:

- X moves in the intended direction,
- Y/Z drift stays bounded,
- orientation error stays bounded,
- joint velocities stay sane,
- torque saturation is interpretable,
- JSONL trace and summary are written.

## Milestone 6: Tune And Scale

Status: later

Only after tiny X motion works:

- increase duration,
- increase target from 5 mm toward 10 mm and 20 mm,
- compare API vs numerical Jacobian behavior,
- evaluate gravity compensation,
- generate final validation video and plot.

## Historical Context

Older MuJoCo, MoveIt, SLSQP, workspace, and origin-stabilization notes remain under:

```text
docs/archive/
```

Those notes are useful context but not the current active path.
