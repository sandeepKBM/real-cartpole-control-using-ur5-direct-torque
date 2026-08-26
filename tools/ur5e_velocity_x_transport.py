#!/usr/bin/env python3
"""Run native RTDE Cartesian velocity (speedL) X transport on URSim / real UR5e.

Dedicated entrypoint for ``--control-mode velocity`` (the mode
``hardware/velocity_transport.py::run_x_transport_velocity`` implements, dispatched
via ``hardware/x_transport.py::run_x_transport``). Modeled closely on
``tools/ur5e_direct_torque_x_transport.py``: same receive-only ``--probe-only``,
same explicit motion opt-in, same typed-MOVE confirmation, same guard-override
flag family. Unlike that tool, this one always runs ``control_mode="velocity"``
-- there is no ``--control-mode`` flag here.

Call path: ``run_x_transport(control_mode="velocity", ...)`` ->
``hardware/x_transport.py``'s ``mode == "velocity"`` branch, which:
  1. builds a plain ``hardware.link.UR5eLink`` (NOT ``UR5eDirectTorqueLink`` --
     velocity mode has no torque/dynamics path and never needs the 500Hz
     direct-torque link),
  2. joint-moves to the start pose via that module's ``_joint_move_ur5e_link``
     (same ``HEIGHT_ALPHA_0_5_CLEARANCE_Q`` default as direct_torque/urscript,
     overridable with ``--start-q-rad``),
  3. calls ``hardware/velocity_transport.py::run_x_transport_velocity``, which
     itself calls ``link.verify_speedl_signature()`` right after
     ``connect(with_control=True)`` and before ever calling ``link.speed_l()``.

Confirmed by reading (not guessing) both dispatch functions -- see
hardware/x_transport.py's ``mode == "velocity"`` branch and
hardware/velocity_transport.py::run_x_transport_velocity.

*** SAFETY CAVEAT -- READ BEFORE USE ***
Velocity/speedL mode is NOT YET REAL-HARDWARE VALIDATED. Sim-only kinematic
characterization (config/ur5e_velocity_control.yaml's header, 2026-08-03)
found a real, UNRESOLVED, dx-dependent instability at the pose that config
was checked against: ``--target-x-delta 0.02`` (0.02 m) DIVERGES --
orientation error grows unboundedly during the hold phase, reaching ~1 rad
by t=6s with guards disabled; ``--target-x-delta 0.04`` (0.04 m, this CLI's
default) is the ONE point checked stable to a 10s hold with zero
orientation growth; ``--target-x-delta`` >= 0.06 m fails via the wrist_2
joint-velocity guard (a kinematic-singularity ceiling). The dx=0.02 vs
dx=0.04 bifurcation is NOT understood -- do not assume values between or
around these are safe merely by interpolation. This mode also has ZERO
force compliance (no gravity comp, no Jacobian-weighted dynamics) --
appropriate only for pure point-to-point transport / range
characterization, not once a physical pole is mounted.

Examples:
  # Probe only (receive + read state, no motion):
  python tools/ur5e_velocity_x_transport.py --robot-ip 127.0.0.1 --probe-only

  # Live speedL transport (URSim or real UR5e) -- default dx=0.04, the one
  # checked-stable point; see the safety caveat above before changing it:
  python tools/ur5e_velocity_x_transport.py --robot-ip <IP> \\
    --target-x-delta 0.04 --move-duration 1.0 --duration 3.0 \\
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
from hardware.link import RTDELinkError, UR5eLink  # noqa: E402
from hardware.safety import NOISE_ROBUST_GUARD_OVERRIDES  # noqa: E402
from hardware.x_transport import run_x_transport  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "config" / "ur5e_velocity_control.yaml"


def resolve_move_limit_overrides(args: argparse.Namespace) -> dict[str, float | int]:
    """Same convention as tools/ur5e_direct_torque_x_transport.py's helper of
    the same name: --noise-robust-guards' preset (if set) applied first via
    dict.update(), then each explicit individual override flag (if not None)
    wins for its own field. Trimmed to exactly the override kwargs
    hardware/velocity_transport.py::run_x_transport_velocity accepts --
    unlike direct_torque, it has no accel/speed *_variable_tolerance params,
    so those two flags are deliberately not offered here (they would parse
    but never reach the call, which is exactly the "flag parsed and then
    discarded" failure mode this repo's AGENTS.md warns about)."""
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
            "Commanded X displacement in meters. Default 0.04 -- the ONE point "
            "sim characterization found stable to a 10s hold (see the module "
            "docstring's SAFETY CAVEAT). 0.02 m is known to DIVERGE "
            "(unbounded orientation-error growth during hold); >= 0.06 m trips "
            "the wrist_2 joint-velocity guard. Velocity mode overall is NOT "
            "yet real-hardware validated -- do not assume other values are "
            "safe by interpolation."
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
            "Control-loop rate for the speedL streaming loop AND the RTDE "
            "link's own frequency_hz (default 125.0, matching "
            "run_x_transport_velocity's own default). hardware/x_transport.py's "
            "velocity dispatch branch constructs UR5eLink(robot_ip, "
            "frequency_hz=rate_hz) -- the two must match (a mismatched link "
            "frequency is a real desync, not cosmetic), so changing this flag "
            "changes both together; there is no separate flag for the link's "
            "own rate."
        ),
    )
    p.add_argument(
        "--speed-l-acceleration",
        type=float,
        default=1.2,
        help=(
            "speedL's own 'acceleration' argument (m/s^2), i.e. how hard the "
            "robot's firmware is allowed to ramp toward each streamed "
            "Cartesian velocity command -- NOT a CartesianMoveMonitor guard "
            "threshold. Default 1.2, matching run_x_transport_velocity's own "
            "default."
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
            "overriding the default HEIGHT_ALPHA_0_5_CLEARANCE_Q pose (the same "
            "default hardware/x_transport.py's velocity/direct_torque branches "
            "share)."
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
        help="Override of CartesianMoveLimits.accel_gap_cycles (class default 1). See tools/ur5e_direct_torque_x_transport.py's flag of the same name for the noise-floor rationale.",
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
            "hardware.safety.NOISE_ROBUST_GUARD_OVERRIDES). Applied first; any "
            "individual override flag above still wins for that specific field."
        ),
    )
    return p.parse_args()


def _default_output_dir() -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return REPO_ROOT / "outputs" / "hardware_transport" / f"velocity_{stamp}"


def main() -> int:
    args = parse_args()
    # Same ordering as tools/ur5e_direct_torque_x_transport.py::main(): the
    # motion opt-in check runs BEFORE dashboard power-on / connect, so
    # --probe-only never requires --i-understand-this-moves-the-robot and
    # never opens a control connection.
    needs_motion = not args.probe_only
    if needs_motion and not args.motion_opt_in:
        print("Refusing motion without --i-understand-this-moves-the-robot", file=sys.stderr)
        return 2
    if needs_motion and not args.yes:
        typed = input("Type MOVE to run X transport (mode=velocity): ").strip()
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
        # Receive-only probe -- connect(with_control=False), plain
        # disconnect(), no motion/stop command ever issued. Mirrors the
        # position-mode probe branch in tools/ur5e_direct_torque_x_transport.py.
        link = UR5eLink(args.robot_ip, frequency_hz=float(args.rate_hz))
        try:
            link.connect(with_control=False)
        except RTDELinkError as exc:
            print(f"RTDE connect failed: {exc}", file=sys.stderr)
            return 1
        state = link.read_state()
        print(f"PROBE OK (velocity) q={state.q.round(4).tolist()} tcp_x={state.tcp_pose[0]:.4f}")
        link.disconnect()
        return 0

    output_dir = args.output_dir or _default_output_dir()
    start_q_rad = None if args.start_q_rad is None else np.asarray(args.start_q_rad, dtype=np.float64)
    move_limit_overrides = resolve_move_limit_overrides(args)
    try:
        result = run_x_transport(
            control_mode="velocity",
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
            speed_l_acceleration=float(args.speed_l_acceleration),
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
