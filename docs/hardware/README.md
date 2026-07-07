# Hardware Lane

Current as of 2026-07-07. Authoritative detail lives in `AGENTS.md` §4 — this is a short
pointer, not a duplicate.

## Layout

- `hardware/safety.py` — every numeric limit and safety decision (`UR5eSafetyLimits`,
  `ConnectionHealth`, `EStopLatch`, `CartesianMoveLimits`/`CartesianMoveMonitor`).
- `hardware/link.py` — `UR5eLink`: connection + live state. `read_state()` raises
  `RTDEStateError` rather than ever returning stale/default data.
- `hardware/motion.py` — `move_cartesian_bounded()`, the one bounded Cartesian move
  (quintic min-jerk `servoL` streaming, safety-checked every cycle).
- `tools/ur5e_connect.py` — connect + read state only (`--once` / `--watch`); cannot move
  the robot (never imports `hardware/motion.py`).
- `tools/ur5e_move.py` — the one move entrypoint; `--axis {x,y,z}` has no default,
  `--i-understand-this-moves-the-robot` + a typed `MOVE` confirmation required.

Direct torque control is out of scope entirely — the installed `rtde_control` library has no
working torque API, and there is no Jacobian/FK code that works from real robot state without
MuJoCo. See `docs/archive/hardware_v1/ur5e_direct_torque_warning.md` for the fuller historical
rationale (superseded framing, same conclusion).

## Procedural sequence for first real-robot contact

1. `tools/ur5e_connect.py --once` first, always — confirms the RTDE connection and that
   `read_state()` returns sane joint/TCP data before anything else.
2. `tools/ur5e_connect.py --watch` to confirm the connection holds and liveness/reconnect
   behavior works as expected.
3. `tools/ur5e_move.py` with a small `--distance-m 0.02` before ever requesting a larger move.

## What's verified vs. not

Verified today against the real Universal Robots `URControl` binary (via a local URSim
instance, not a mock): `hardware/link.py`'s connection path against the real RTDE protocol
server, and that its error handling correctly rejects a degenerate (not-fully-powered-on)
server response instead of accepting bad data. Also confirmed directly against the real
library: `RTDEReceiveInterface` has `getSafetyStatusBits` and not `getSafetyStatus` (matches
this code's fallback order), and `RTDEControlInterface.servoL`'s real signature is exactly
`(pose, speed, acceleration, time, lookahead_time, gain)` (matches `hardware/link.py`'s
`_EXPECTED_SERVOL_PARAMS`).

Not verifiable without the real robot or a fuller URSim stack (Dashboard server + real
power-on): real `servoL` streaming motion, whether 125Hz streaming holds up on real network
traffic, whether `getActualTCPPose()`'s rotation-vector convention matches the orientation-error
math here, and whether the physical mount makes a given `--axis` argument actually correspond
to true left/right.
