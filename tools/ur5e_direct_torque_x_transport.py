#!/usr/bin/env python3
"""Run tuned OSC X transport on URSim / real UR5e with selectable control mode.

Default ``--control-mode position`` streams the same min-jerk profile through
``servoL`` and runs OSC in **shadow** (logs ``tau_shadow``, no torques sent).
Use ``direct_torque`` on the real robot when ready for live torque.

Examples:
  # Component test (URSim or real arm — position / servoL):
  python tools/ur5e_direct_torque_x_transport.py --robot-ip 127.0.0.1 \\
    --control-mode position --target-x-delta 0.02 --move-duration 1.0 --duration 3.0 \\
    --i-understand-this-moves-the-robot --yes

  # Live direct torque (real UR5):
  python tools/ur5e_direct_torque_x_transport.py --robot-ip <IP> \\
    --control-mode direct_torque --dynamics-source local \\
    --target-x-delta 0.02 --i-understand-this-moves-the-robot --yes

  # Probe only (receive + read state):
  python tools/ur5e_direct_torque_x_transport.py --robot-ip 127.0.0.1 --probe-only
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

from _bootstrap import ensure_repo_root

ensure_repo_root()

from hardware.dashboard import power_on_and_release, query_remote_control  # noqa: E402
from hardware.direct_torque_link import UR5eDirectTorqueLink  # noqa: E402
from hardware.link import RTDELinkError, UR5eLink  # noqa: E402
from hardware.x_transport import run_x_transport  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "config" / "ur5e_mujoco_torque_osc_tuned.yaml"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--robot-ip", required=True)
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--control-mode", choices=("position", "direct_torque", "urscript"), default="position")
    p.add_argument("--target-x-delta", type=float, default=0.02)
    p.add_argument("--move-duration", type=float, default=1.0)
    p.add_argument("--duration", type=float, default=3.0)
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--skip-joint-move", action="store_true")
    p.add_argument("--no-shadow-osc", action="store_true", help="Position mode only: skip OSC shadow compute.")
    p.add_argument("--probe-only", action="store_true", help="Connect + read state only (no transport).")
    p.add_argument("--skip-dashboard-power-on", action="store_true")
    p.add_argument(
        "--i-understand-this-moves-the-robot",
        dest="motion_opt_in",
        action="store_true",
    )
    p.add_argument("--yes", action="store_true", help="Skip typed MOVE confirmation.")
    p.add_argument(
        "--dynamics-source",
        choices=("rtde", "local"),
        default="local",
        help="direct_torque only: rtde=PolyScope J+M; local=MuJoCo J+M from q.",
    )
    return p.parse_args()


def _default_output_dir(control_mode: str) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return REPO_ROOT / "outputs" / "hardware_transport" / f"{control_mode}_{stamp}"


def main() -> int:
    args = parse_args()
    if not args.skip_dashboard_power_on:
        print("Dashboard status:")
        for cmd, resp in power_on_and_release(args.robot_ip).items():
            print(f"  {cmd}: {resp}")
        time.sleep(2.0)

    remote = query_remote_control(args.robot_ip)
    print(f"is_in_remote_control: {remote}")
    if not remote and not args.probe_only:
        print(
            "\nRemote control is OFF. Enable it in PolyScope before motion:\n"
            "  docs/hardware/URSIM_REMOTE_CONTROL.md\n",
            file=sys.stderr,
        )
        return 2

    if args.probe_only:
        if args.control_mode == "direct_torque":
            link = UR5eDirectTorqueLink(args.robot_ip, frequency_hz=500.0)
            try:
                link.connect()
            except RTDELinkError as exc:
                print(f"RTDE connect failed: {exc}", file=sys.stderr)
                return 1
            state = link.read_state()
            print(f"PROBE OK (direct_torque) q={state.q.round(4).tolist()} tcp_x={state.tcp_pose[0]:.4f}")
            link.safe_stop("probe_complete")
            return 0
        link = UR5eLink(args.robot_ip, frequency_hz=125.0)
        link.connect(with_control=False)
        state = link.read_state()
        print(f"PROBE OK (position) q={state.q.round(4).tolist()} tcp_x={state.tcp_pose[0]:.4f}")
        link.disconnect()
        return 0

    if not args.motion_opt_in:
        print("Refusing motion without --i-understand-this-moves-the-robot", file=sys.stderr)
        return 2
    if not args.yes:
        typed = input(f"Type MOVE to run X transport (mode={args.control_mode}): ").strip()
        if typed != "MOVE":
            print("Aborted.", file=sys.stderr)
            return 2

    output_dir = args.output_dir or _default_output_dir(str(args.control_mode))
    try:
        result = run_x_transport(
            control_mode=str(args.control_mode),
            robot_ip=args.robot_ip,
            config_path=args.config,
            target_x_delta_m=float(args.target_x_delta),
            move_duration_s=float(args.move_duration),
            duration_s=float(args.duration),
            output_dir=output_dir,
            motion_opt_in=True,
            dynamics_source=str(args.dynamics_source),
            shadow_osc=not args.no_shadow_osc,
            skip_joint_move=bool(args.skip_joint_move),
        )
    except RTDELinkError as exc:
        print(f"RTDE failed: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(result.summary, indent=2))
    if result.trace_path is not None:
        print(f"trace: {result.trace_path}")
    return 0 if result.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
