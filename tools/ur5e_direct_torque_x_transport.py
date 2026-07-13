#!/usr/bin/env python3
"""Run the tuned OSC controller on URSim / real UR5e via PolyScope direct_torque().

This uses ``ur_rtde.RTDEControlInterface.directTorque()`` (PolyScope >=5.23).
PolyScope compensates gravity inside ``direct_torque()`` -- this script does
**not** add gravity torque on top of the controller output.

Prerequisites:
  1. URSim powered on and brakes released.
  2. **Remote control enabled** in PolyScope (see docs/hardware/URSIM_REMOTE_CONTROL.md).
  3. ``ur_rtde >= 1.6`` in your Python env.

Examples:
  # Probe only (no motion):
  python tools/ur5e_direct_torque_x_transport.py --robot-ip 127.0.0.1 --probe-only

  # Small live X move (start here):
  python tools/ur5e_direct_torque_x_transport.py --robot-ip 127.0.0.1 \\
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

from _bootstrap import ensure_repo_root

ensure_repo_root()

from hardware.dashboard import power_on_and_release, query_remote_control  # noqa: E402
from hardware.direct_torque_link import UR5eDirectTorqueLink  # noqa: E402
from hardware.direct_torque_transport import run_x_transport_direct_torque  # noqa: E402
from hardware.link import RTDELinkError  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "config" / "ur5e_mujoco_torque_osc_tuned.yaml"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--robot-ip", required=True)
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--target-x-delta", type=float, default=0.02)
    p.add_argument("--move-duration", type=float, default=1.0)
    p.add_argument("--duration", type=float, default=3.0)
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--probe-only", action="store_true", help="Connect, hold zero torque briefly, exit.")
    p.add_argument("--skip-dashboard-power-on", action="store_true")
    p.add_argument(
        "--i-understand-this-moves-the-robot",
        dest="motion_opt_in",
        action="store_true",
    )
    p.add_argument("--yes", action="store_true", help="Skip typed MOVE confirmation.")
    return p.parse_args()


def _default_output_dir() -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return REPO_ROOT / "outputs" / "hardware_direct_torque" / f"x_transport_{stamp}"


def main() -> int:
    args = parse_args()
    if not args.skip_dashboard_power_on:
        print("Dashboard status:")
        for cmd, resp in power_on_and_release(args.robot_ip).items():
            print(f"  {cmd}: {resp}")
        time.sleep(2.0)

    remote = query_remote_control(args.robot_ip)
    print(f"is_in_remote_control: {remote}")
    if not remote:
        print(
            "\nRemote control is OFF. Enable it in PolyScope before motion:\n"
            "  docs/hardware/URSIM_REMOTE_CONTROL.md\n"
            "  UI: http://localhost:6080/vnc.html?host=localhost&port=6080\n",
            file=sys.stderr,
        )
        if not args.probe_only:
            return 2

    link = UR5eDirectTorqueLink(args.robot_ip, frequency_hz=500.0)
    try:
        link.connect()
    except RTDELinkError as exc:
        print(f"RTDE connect failed: {exc}", file=sys.stderr)
        if not remote:
            print("Most likely cause: remote control not enabled.", file=sys.stderr)
        else:
            print(
                "Remote control is ON but RTDE control still failed. Common causes:\n"
                "  - PROFINET / EtherNet/IP still enabled (Services + Installation > Fieldbus)\n"
                "  - URSim not restarted after disabling fieldbus\n"
                "  See docs/hardware/URSIM_REMOTE_CONTROL.md and run:\n"
                "    python tools/_rtde_control_sweep.py",
                file=sys.stderr,
            )
        return 1

    state = link.read_state()
    print(f"Connected. q={state.q.round(4).tolist()} tcp_x={state.tcp_pose[0]:.4f} m")

    if args.probe_only:
        print("Holding zero direct torque for 1.0 s...")
        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline:
            link.direct_torque([0.0] * 6, friction_comp=True)
            time.sleep(0.002)
        link.safe_stop("probe_complete")
        print("PROBE OK: receive + control + directTorque() path works.")
        return 0

    if not args.motion_opt_in:
        print("Refusing motion without --i-understand-this-moves-the-robot", file=sys.stderr)
        link.safe_stop("missing_opt_in")
        return 2
    if not args.yes:
        typed = input("Type MOVE to command a live direct-torque X transport: ").strip()
        if typed != "MOVE":
            print("Aborted.", file=sys.stderr)
            link.safe_stop("user_abort")
            return 2

    output_dir = args.output_dir or _default_output_dir()
    result = run_x_transport_direct_torque(
        link,
        config_path=args.config,
        target_x_delta_m=float(args.target_x_delta),
        move_duration_s=float(args.move_duration),
        duration_s=float(args.duration),
        output_dir=output_dir,
        motion_opt_in=True,
    )
    print(json.dumps(result.summary, indent=2))
    if result.trace_path is not None:
        print(f"trace: {result.trace_path}")
    return 0 if result.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
