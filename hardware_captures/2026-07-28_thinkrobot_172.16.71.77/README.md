# Real-hardware capture — 2026-07-28

First-ever live motion tests (position mode) and first-ever live torque
command (direct_torque mode) on the real UR5e, robot IP `172.16.71.77`,
run from `thinkrobot` (a separate machine from westeros, not this repo's
usual dev host). Captured here (not under `outputs/`, which is gitignored)
because this is real reference data worth keeping for sim-to-real gap work
-- calibrating `env.noise.{q_noise_std_rad,qd_noise_std_radps,
torque_noise_std_nm}` (`rl_gain_scheduling/gain_scheduling_env.py`) and the
CLI's equivalent `--q-noise-std-rad`/`--qd-noise-std-radps`/
`--torque-noise-std-nm` flags (`tools/ur5e_mujoco_torque_experiments.py`)
against what real telemetry noise actually looks like, not guessed values.

Files here are pasted verbatim from the live session transcript (thinkrobot
output copied into chat), not re-generated -- some runs only have the
printed summary JSON (trace file not pasted), others have the full
`trace.jsonl`.

## Pose context

All runs below are at `height_alpha=0.5`
(`hardware/poses.py::HEIGHT_ALPHA_0_5_Q`), base joint (`shoulder_pan`)
rotated `-45°` (`-0.7853981633974483 rad`) from the canonical pose for
real-world wall clearance in the room `thinkrobot`'s robot sits in. The
`position_150847` and `direct_torque_151331` runs are at `wrist_2=0.0`
(exactly at the UR wrist singularity); `direct_torque_151512` is at
`wrist_2=0.1` (nudged off the singularity) -- see the finding below.

## Runs, in order, with what each one showed

1. **`position_145240_summary.json`** -- first live position-mode attempt,
   `dx=+0.02m`. Tripped the TCP-acceleration guard (`CartesianMoveMonitor`)
   at step 1 (0.008s), `0.7158 m/s^2 > 0.5 m/s^2`. `max_abs_qd_radps` only
   `9.3e-5 rad/s` -- essentially no real motion.
2. **`position_145539_summary.json`** -- retry, same command. Tripped at
   step 6 (0.048s) this time, `0.9042 m/s^2`. Still `max_abs_qd_radps`
   `0.018 rad/s` and all drift/orientation metrics negligible.
3. **`position_150508_summary.json`** -- retry with
   `--max-tcp-accel-mps2 1.2` (a since-added CLI override, see
   `hardware/safety.py`/`hardware/position_transport.py`). Tripped at step 7
   (0.056s), `1.3456 m/s^2 > 1.2 m/s^2`.
4. **`position_150847_trace.jsonl`** -- retry after a settle pause, same
   `1.2` threshold. This is the run that revealed the REAL finding: `Z`
   position climbs monotonically in the last 3 samples
   (`0.929981 -> 0.930001 -> 0.930107 -> 0.930251`, should hold flat for a
   pure-X move) and `wrist_1`/`wrist_3` joint velocities grow
   near-exponentially step over step (`~0.31 -> ~0.55 -> ~0.84 rad/s`,
   roughly 1.6-1.8x per 8ms step). Root cause: `wrist_2 stays within
   ~1e-4 rad of exactly 0` throughout -- this is the UR wrist singularity,
   and `position` mode drives the robot via `servoL`, i.e. the UR
   firmware's own internal Cartesian-to-joint IK solver, which has NONE of
   this codebase's singularity protections (those -- nullspace-posture
   projection, singular-value wrench scaling -- only exist in the
   torque-control path; `shadow_osc` in position mode computes them but
   never sends them to the robot). This was a REAL, worsening divergence,
   not sensor noise -- raising the accel threshold across attempts 1-4 was
   the wrong move (it let the divergence run longer each time before
   stopping) and was corrected mid-session.
5. **`direct_torque_151331_summary.json`** -- first-ever live torque
   command, still at `wrist_2=0.0` (singularity), default (unmodified)
   `0.5 m/s^2` threshold. Tripped at step 1 (0.002s, 500Hz),
   `13.9027 m/s^2 > 0.5 m/s^2`. Unlike the position-mode case: everything
   else in this record is near-zero (`max_abs_qd_radps=0.000134`,
   `tau_controller` max `~0.001 Nm`, zero drift/orientation) -- no real
   motion or force anywhere that could produce a real 13.9 m/s^2. This
   pointed at a software timing bug, not a physical one.
6. **`direct_torque_151512_trace.jsonl`** -- retry, `wrist_2` nudged to
   `0.1 rad` (off the singularity, via the run's own automatic joint-space
   `moveJ` startup step). Trace has exactly 1 row (`t=0.0`), everything
   near-zero again (`qd` max `~9e-5 rad/s`, `tau_controller` max
   `~0.0018 Nm`). Same signature as run 5: reproducible trip with zero real
   signal. Root cause identified and fixed in `hardware/safety.py`'s
   `CartesianMoveMonitor`: `set_start()` captures a reference position, but
   real wall-clock time between that call and the first `check()` call
   (real setup work in between -- e.g. `direct_torque_transport.py`'s
   `local_dynamics.jacobian_and_mass_matrix()` call) can be meaningfully
   longer than the caller-assumed nominal `dt_s` (`0.002s` at 500Hz, worse
   than position mode's `0.008s` at 125Hz). Dividing a tiny real position
   delta by too-small an assumed `dt_s` inflates the computed speed, and
   squares that error again for acceleration. Fixed by using real measured
   elapsed time whenever it's LONGER than the caller's `dt_s` (never
   shorter, so synthetic/test call sequences with no real sleep between
   `check()` calls are unaffected -- confirmed via the full test suite,
   293/294 passing, same 1 pre-existing unrelated failure as always).

## Why this matters for sim-noise calibration

The real per-sample jitter visible in `position_150847_trace.jsonl`'s early
rows (before the singularity divergence dominates, e.g. `t=0.0` to
`t=0.032`, where the move is genuinely near-stationary) is the closest
thing this project has to real `q`/`qd`/`tcp_pose` sensor noise magnitudes
-- useful for sanity-checking whether
`config/rl_gain_scheduling_noise_smoke.yaml`'s `q_noise_std_rad: 0.02` /
`qd_noise_std_radps: 0.02` (chosen without real reference data) are in the
right ballpark, too large, or too small relative to what real RTDE
telemetry actually looks like.
