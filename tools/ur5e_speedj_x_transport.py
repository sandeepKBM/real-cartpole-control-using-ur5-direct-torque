#!/usr/bin/env python3
"""Run RTDE JOINT velocity (speedJ) X transport on URSim / real UR5e.

Dedicated entrypoint for ``--control-mode joint_velocity`` (the mode
``hardware/joint_velocity_transport.py::run_x_transport_joint_velocity``
implements, dispatched via ``hardware/x_transport.py::run_x_transport``).
Modeled closely on ``tools/ur5e_velocity_x_transport.py``: same receive-only
``--probe-only``, same explicit motion opt-in, same typed-MOVE confirmation,
same guard-override flag family. Unlike that tool, this one streams RTDE
``speedJ`` (a JOINT velocity WE compute via a damped least-squares Jacobian
inverse), not ``speedL`` (a Cartesian velocity the robot's own firmware
inverts internally).

Why this mode exists: real-hardware testing of ``--control-mode velocity``
found the firmware's own Cartesian-velocity IK protective-stops at the
wrist-2 kinematic singularity (+X moved 100%, the mirrored -X singularity-
stopped). This mode bypasses that firmware IK step entirely -- the Jacobian
inverse is computed here via singularity-robust damped least-squares
(``controller_core/damped_least_squares.py``), which degrades GRACEFULLY
near a singularity (a bounded, if imperfectly-tracked, joint velocity)
instead of raising an IK error. It still uses the robot's own joint SERVO
loop (speedJ ramps toward each streamed setpoint under firmware control,
just like speedL does for Cartesian setpoints) -- only the IK step moved
from the firmware to us.

Default ``--config`` is ``config/ur5e_speedj_joint_velocity.yaml``, a
DEDICATED config with ``reduced_task_dims``/``split_base_wrist_task``/
``ik_seeded_resolution`` all false -- deliberately NOT
``config/ur5e_velocity_control.yaml`` (the speedL mode's default). That
config's ``reduced_task_dims: true`` makes ``CartesianVelocityController``
resolve the Cartesian task to a joint velocity internally before this tool's
own DLS step runs, which double-resolves (see
``hardware/joint_velocity_transport.py``'s module docstring and
``tests/hardware/test_joint_velocity_resolution_fix.py``). The transport
function itself refuses to run against a config with any of those three
flags set, so pointing ``--config`` back at ``ur5e_velocity_control.yaml``
fails fast rather than silently double-resolving.

*** SAFETY CAVEAT -- READ BEFORE USE ***
This mode is BRAND NEW and has ZERO real-hardware or URSim validation of any
kind as of this writing. Unlike ``--control-mode velocity`` (which at least
has one sim-characterized stable operating point), no operating point for
this mode has been characterized in sim OR on real hardware yet. The DLS
damping defaults (``--damping-lambda-max``/``--damping-sigma0``, both 0.05)
and the mandatory joint-velocity clamp (``--joint-velocity-clamp``, default
0.3 rad/s, well under the ~3.15-3.20 rad/s manufacturer ceiling and the
operative CartesianMoveMonitor 1.5 rad/s guard) are conservative STARTING
POINTS chosen by reasoning about the math, not validated numbers -- treat
every run as a first-of-its-kind probe, start with small ``--target-x-delta``
and short ``--duration``, and read the trace (``sigma_min``/
``dls_lambda_used``/``qd_clamp_hit`` are logged every cycle) rather than
trusting a clean exit code. Like ``--control-mode velocity``, this mode has
ZERO force compliance (no gravity comp, no Jacobian-weighted dynamics) --
appropriate only for pure point-to-point transport / range characterization,
not once a physical pole is mounted.

Examples:
  # Probe only (receive + read state, no motion):
  python tools/ur5e_speedj_x_transport.py --robot-ip 127.0.0.1 --probe-only

  # Live speedJ transport (URSim or real UR5e) -- small, short, first probe:
  python tools/ur5e_speedj_x_transport.py --robot-ip <IP> \\
    --target-x-delta 0.02 --move-duration 1.0 --duration 3.0 \\
    --i-understand-this-moves-the-robot --yes
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from _bootstrap import ensure_repo_root

ensure_repo_root()

from hardware.dashboard import power_on_and_release, query_remote_control  # noqa: E402
from hardware.joint_velocity_transport import (  # noqa: E402
    DEFAULT_DAMPING_LAMBDA_MAX,
    DEFAULT_DAMPING_SIGMA0,
    DEFAULT_JOINT_VELOCITY_CLAMP_RADPS,
)
from hardware.link import RTDELinkError, UR5eLink  # noqa: E402
from hardware.safety import NOISE_ROBUST_GUARD_OVERRIDES  # noqa: E402
from hardware.x_transport import run_x_transport  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
# Dedicated config, NOT config/ur5e_velocity_control.yaml -- that config's
# reduced_task_dims: true makes CartesianVelocityController resolve the
# Cartesian task to a joint velocity internally, which would make
# hardware/joint_velocity_transport.py's DLS step double-resolve instead of
# being the sole Cartesian-to-joint resolver. See
# config/ur5e_speedj_joint_velocity.yaml's own header for the measured
# residuals that motivated this, and
# hardware/joint_velocity_transport.py::run_x_transport_joint_velocity's
# fail-fast check for the same thing.
DEFAULT_CONFIG = REPO_ROOT / "config" / "ur5e_speedj_joint_velocity.yaml"


def resolve_move_limit_overrides(args: argparse.Namespace) -> dict[str, float | int]:
    """Same convention as tools/ur5e_velocity_x_transport.py's helper of the
    same name: --noise-robust-guards' preset (if set) applied first via
    dict.update(), then each explicit individual override flag (if not None)
    wins for its own field."""
    overrides: dict[str, float | int] = {}
    if args.noise_robust_guards:
        overrides.update(NOISE_ROBUST_GUARD_OVERRIDES)
    if args.max_tcp_accel_mps2 is not None:
        overrides["max_tcp_accel_mps2"] = float(args.max_tcp_accel_mps2)
    if args.max_tcp_speed_mps is not None:
        overrides["max_tcp_speed_mps"] = float(args.max_tcp_speed_mps)
    if args.accel_gap_cycles is not None:
        overrides["accel_gap_cycles"] = int(args.accel_gap_cycles)
    if args.speed_lowpass_alpha is not None:
        overrides["speed_lowpass_alpha"] = float(args.speed_lowpass_alpha)
    if args.speed_limit_gap_cycles is not None:
        overrides["speed_limit_gap_cycles"] = int(args.speed_limit_gap_cycles)
    if args.speed_limit_lowpass_alpha is not None:
        overrides["speed_limit_lowpass_alpha"] = float(args.speed_limit_lowpass_alpha)
    if args.accel_max_consecutive_violations is not None:
        overrides["accel_max_consecutive_violations"] = int(args.accel_max_consecutive_violations)
    if args.accel_hard_multiple is not None:
        overrides["accel_hard_multiple"] = float(args.accel_hard_multiple)
    if args.speed_max_consecutive_violations is not None:
        overrides["speed_max_consecutive_violations"] = int(args.speed_max_consecutive_violations)
    if args.speed_hard_multiple is not None:
        overrides["speed_hard_multiple"] = float(args.speed_hard_multiple)
    return overrides


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--robot-ip", required=True)
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument(
        "--target-x-delta",
        type=float,
        default=0.04,
        help=(
            "Commanded X displacement in meters. Default 0.04 for parity with "
            "velocity mode's own default -- NOT validated for this brand-new "
            "mode (see the module docstring's SAFETY CAVEAT). Start smaller "
            "for a first real/URSim probe."
        ),
    )
    p.add_argument("--move-duration", type=float, default=1.0)
    p.add_argument("--duration", type=float, default=3.0)
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument(
        "--rate-hz",
        type=float,
        default=125.0,
        help=(
            "Control-loop rate for the speedJ streaming loop AND the RTDE "
            "link's own frequency_hz (default 125.0). hardware/x_transport.py's "
            "joint_velocity dispatch branch constructs UR5eLink(robot_ip, "
            "frequency_hz=rate_hz) -- the two must match."
        ),
    )
    p.add_argument(
        "--speed-j-acceleration",
        type=float,
        default=1.2,
        help=(
            "speedJ's own 'acceleration' argument (rad/s^2), i.e. how hard the "
            "robot's firmware is allowed to ramp toward each streamed joint "
            "velocity command -- NOT a CartesianMoveMonitor guard threshold. "
            "Default 1.2, matching speedL mode's own default (units differ: "
            "rad/s^2 here vs m/s^2 there)."
        ),
    )
    p.add_argument(
        "--joint-velocity-clamp",
        type=float,
        default=DEFAULT_JOINT_VELOCITY_CLAMP_RADPS,
        help=(
            f"Mandatory hard per-joint clamp on the DLS-resolved joint "
            f"velocity command, in rad/s, applied BEFORE every speedJ() call "
            f"(default {DEFAULT_JOINT_VELOCITY_CLAMP_RADPS}). This is the last "
            "line of defense before a command reaches the robot -- DLS only "
            "bounds |qd| as the Jacobian approaches singularity, not in "
            "general. Conservative, unvalidated starting point -- see the "
            "module docstring's SAFETY CAVEAT."
        ),
    )
    p.add_argument(
        "--damping-lambda-max",
        type=float,
        default=DEFAULT_DAMPING_LAMBDA_MAX,
        help=(
            f"DLS variable-damping lambda_max (default {DEFAULT_DAMPING_LAMBDA_MAX}) -- "
            "see controller_core/damped_least_squares.py's module docstring for "
            "the Nakamura/Wampler formula. Larger values bound |qd| more "
            "aggressively near a singularity at the cost of more Cartesian "
            "tracking error there."
        ),
    )
    p.add_argument(
        "--damping-sigma0",
        type=float,
        default=DEFAULT_DAMPING_SIGMA0,
        help=(
            f"DLS variable-damping sigma0 (default {DEFAULT_DAMPING_SIGMA0}) -- "
            "the smallest-singular-value threshold below which damping ramps "
            "up. Larger values start damping farther from the true "
            "singularity."
        ),
    )
    p.add_argument(
        "--start-q-rad",
        type=float,
        nargs=6,
        default=None,
        metavar=("Q1", "Q2", "Q3", "Q4", "Q5", "Q6"),
        help=(
            "Optional six-joint pose in radians to moveJ to before transport, "
            "overriding the default HEIGHT_ALPHA_0_5_CLEARANCE_Q pose."
        ),
    )
    p.add_argument("--skip-joint-move", action="store_true")
    p.add_argument("--probe-only", action="store_true", help="Connect + read state only (no transport).")
    p.add_argument("--skip-dashboard-power-on", action="store_true")
    p.add_argument(
        "--i-understand-this-moves-the-robot",
        dest="motion_opt_in",
        action="store_true",
    )
    p.add_argument("--yes", action="store_true", help="Skip typed MOVE confirmation.")
    p.add_argument(
        "--max-tcp-accel-mps2",
        type=float,
        default=None,
        help="Explicit, opt-in override of CartesianMoveLimits.max_tcp_accel_mps2 (class default 0.5 m/s^2).",
    )
    p.add_argument(
        "--max-tcp-speed-mps",
        type=float,
        default=None,
        help="Explicit, opt-in override of CartesianMoveLimits.max_tcp_speed_mps (class default 0.05 m/s).",
    )
    p.add_argument(
        "--accel-gap-cycles",
        type=int,
        default=None,
        help="Override of CartesianMoveLimits.accel_gap_cycles (class default 1).",
    )
    p.add_argument(
        "--speed-lowpass-alpha",
        type=float,
        default=None,
        help="Override of CartesianMoveLimits.speed_lowpass_alpha (class default 1.0 = no filtering).",
    )
    p.add_argument(
        "--speed-limit-gap-cycles",
        type=int,
        default=None,
        help="Override of CartesianMoveLimits.speed_limit_gap_cycles (class default 1).",
    )
    p.add_argument(
        "--speed-limit-lowpass-alpha",
        type=float,
        default=None,
        help="Override of CartesianMoveLimits.speed_limit_lowpass_alpha (class default 1.0 = no filtering).",
    )
    p.add_argument(
        "--accel-max-consecutive-violations",
        type=int,
        default=None,
        help="Override of CartesianMoveLimits.accel_max_consecutive_violations (class default 1 = instant trip).",
    )
    p.add_argument(
        "--accel-hard-multiple",
        type=float,
        default=None,
        help="Override of CartesianMoveLimits.accel_hard_multiple (class default 5.0).",
    )
    p.add_argument(
        "--speed-max-consecutive-violations",
        type=int,
        default=None,
        help="Override of CartesianMoveLimits.speed_max_consecutive_violations (class default 1).",
    )
    p.add_argument(
        "--speed-hard-multiple",
        type=float,
        default=None,
        help="Override of CartesianMoveLimits.speed_hard_multiple (class default 5.0).",
    )
    p.add_argument(
        "--noise-robust-guards",
        action="store_true",
        help=(
            "Convenience flag applying the validated 6-parameter combination "
            "from docs/status/safety_envelope_backtest_2026-07-30.md (see "
            "hardware.safety.NOISE_ROBUST_GUARD_OVERRIDES). That validation "
            "was performed for other control modes -- applying it here is a "
            "reasonable prior, not a re-validation for speedJ streaming. "
            "Applied first; any individual override flag above still wins "
            "for that specific field."
        ),
    )
    return p.parse_args()


def _default_output_dir() -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return REPO_ROOT / "outputs" / "hardware_transport" / f"joint_velocity_{stamp}"


def main() -> int:
    args = parse_args()
    needs_motion = not args.probe_only
    if needs_motion and not args.motion_opt_in:
        print("Refusing motion without --i-understand-this-moves-the-robot", file=sys.stderr)
        return 2
    if needs_motion and not args.yes:
        typed = input("Type MOVE to run X transport (mode=joint_velocity): ").strip()
        if typed != "MOVE":
            print("Aborted.", file=sys.stderr)
            return 2

    if not args.skip_dashboard_power_on:
        print("Dashboard status:")
        for cmd, resp in power_on_and_release(args.robot_ip).items():
            print(f"  {cmd}: {resp}")
        time.sleep(2.0)

    remote = query_remote_control(args.robot_ip)
    print(f"is_in_remote_control: {remote}")
    if not remote and needs_motion:
        print(
            "\nRemote control is OFF. Enable it in PolyScope before motion:\n"
            "  docs/hardware/URSIM_REMOTE_CONTROL.md\n",
            file=sys.stderr,
        )
        return 2

    if args.probe_only:
        link = UR5eLink(args.robot_ip, frequency_hz=float(args.rate_hz))
        try:
            link.connect(with_control=False)
        except RTDELinkError as exc:
            print(f"RTDE connect failed: {exc}", file=sys.stderr)
            return 1
        state = link.read_state()
        print(f"PROBE OK (joint_velocity) q={state.q.round(4).tolist()} tcp_x={state.tcp_pose[0]:.4f}")
        link.disconnect()
        return 0

    output_dir = args.output_dir or _default_output_dir()
    start_q_rad = None if args.start_q_rad is None else np.asarray(args.start_q_rad, dtype=np.float64)
    move_limit_overrides = resolve_move_limit_overrides(args)
    try:
        result = run_x_transport(
            control_mode="joint_velocity",
            robot_ip=args.robot_ip,
            config_path=args.config,
            target_x_delta_m=float(args.target_x_delta),
            move_duration_s=float(args.move_duration),
            duration_s=float(args.duration),
            output_dir=output_dir,
            motion_opt_in=True,
            skip_joint_move=bool(args.skip_joint_move),
            start_q_rad=start_q_rad,
            rate_hz=float(args.rate_hz),
            speed_j_acceleration=float(args.speed_j_acceleration),
            joint_velocity_clamp_radps=float(args.joint_velocity_clamp),
            damping_lambda_max=float(args.damping_lambda_max),
            damping_sigma0=float(args.damping_sigma0),
            max_tcp_accel_mps2_override=move_limit_overrides.get("max_tcp_accel_mps2"),
            max_tcp_speed_mps_override=move_limit_overrides.get("max_tcp_speed_mps"),
            accel_gap_cycles_override=move_limit_overrides.get("accel_gap_cycles"),
            speed_lowpass_alpha_override=move_limit_overrides.get("speed_lowpass_alpha"),
            speed_limit_gap_cycles_override=move_limit_overrides.get("speed_limit_gap_cycles"),
            speed_limit_lowpass_alpha_override=move_limit_overrides.get("speed_limit_lowpass_alpha"),
            accel_max_consecutive_violations_override=move_limit_overrides.get(
                "accel_max_consecutive_violations"
            ),
            accel_hard_multiple_override=move_limit_overrides.get("accel_hard_multiple"),
            speed_max_consecutive_violations_override=move_limit_overrides.get(
                "speed_max_consecutive_violations"
            ),
            speed_hard_multiple_override=move_limit_overrides.get("speed_hard_multiple"),
        )
    except (RTDELinkError, ValueError) as exc:
        print(f"RTDE/start-pose failed: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(result.summary, indent=2))
    if result.trace_path is not None:
        print(f"trace: {result.trace_path}")
    return 0 if result.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
