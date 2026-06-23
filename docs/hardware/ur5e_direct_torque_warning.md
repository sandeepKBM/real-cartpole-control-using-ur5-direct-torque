# UR5e Direct Torque Warning

Direct torque on a real UR5e is high risk. It should not be the default and it should not be exercised from a generic external Python loop.

## Current repo position

- The simulator path can compute torques, but that does not make the same path safe on hardware.
- The RTDE stack in this repo does not expose a real direct-torque hardware loop.
- The guarded probe in `tools/ur5e_direct_torque_probe.py` defaults to zero-only behavior and refuses nonzero torque unless multiple explicit safety flags are present.

## Why this is blocked

- Python timing jitter is not a real-time guarantee.
- A missed deadline, stale state, disconnect, NaN, or bad limit handling can create unsafe motion.
- Direct torque needs robot-side enforcement and watchdog behavior, not just host-side checks.

## Minimum future checklist before any nonzero direct torque

1. Record the exact PolyScope / URSoftware version.
2. Confirm the RTDE client / controller library version.
3. Confirm the robot-side watchdog action for missing updates.
4. Confirm the controller-stop path issues a safe stop on every failure mode.
5. Confirm the zero-torque probe is clean.
6. Confirm the robot-side loop owns the torque application and clamps output.

## Current status

- Zero-only probe: available as a guarded diagnostic.
- Nonzero direct torque: refused in this patch.
- Default behavior: no motion.
