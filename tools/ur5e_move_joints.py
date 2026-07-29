#!/usr/bin/env python3
"""Move UR5e joints to a named pose via RTDE ``moveJ`` (direct-torque lane).

Use this as **step 1** before a direct-torque OSC transport when the arm is
not already at the desired start height.

Example (height_alpha=0.5, same as MuJoCo Test 2):
  python tools/ur5e_move_joints.py --robot-ip <IP> \\
    --pose height_alpha_0_5 \\
    --i-understand-this-moves-the-robot --yes
"""

from __future__ import annotations

import argparse
import json
import sys
import time

from _bootstrap import ensure_repo_root

ensure_repo_root()

from hardware.dashboard import power_on_and_release, query_remote_control  # noqa: E402
from hardware.direct_torque_link import UR5eDirectTorqueLink  # noqa: E402
from hardware.joint_motion import move_joints_to_pose, verify_joint_pose  # noqa: E402
from hardware.link import RTDELinkError  # noqa: E402
from hardware.poses import HEIGHT_ALPHA_0_5_Q, q_for_height_alpha  # noqa: E402
from hardware.safety import EStopLatch  # noqa: E402

POSE_NAMES = ("height_alpha_0_5", "active_origin")


def _resolve_pose(name: str, height_alpha: float | None, shoulder_pan_override_rad: float | None) -> list[float]:
    if height_alpha is not None:
        q = q_for_height_alpha(height_alpha).tolist()
    elif name == "active_origin":
        from hardware.poses import ACTIVE_ORIGIN_Q

        q = ACTIVE_ORIGIN_Q.tolist()
    elif name == "height_alpha_0_5":
        q = HEIGHT_ALPHA_0_5_Q.tolist()
    else:
        raise ValueError(f"unknown pose {name!r}")
    if shoulder_pan_override_rad is not None:
        # Base rotation only -- same real-world-wall-clearance pattern used
        # 2026-07-28 for the alpha=0.1 pose (shoulder_pan=-0.7853981633974483
        # rad = -45deg there). Rotating the base changes nothing about the
        # rest of the arm's configuration/reach shape, only which absolute
        # direction it points -- but the SAME angle does not guarantee the
        # SAME real clearance at a different pose (different shoulder_lift/
        # elbow/wrist angles change the arm's physical shape). Always
        # re-verify visually via --dry-run + eyeballing before a real move.
        q = list(q)
        q[0] = float(shoulder_pan_override_rad)
    return q


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--robot-ip", required=True)
    p.add_argument("--pose", default="height_alpha_0_5", choices=POSE_NAMES)
    p.add_argument("--height-alpha", type=float, default=None, help="Override --pose with 0..1 interpolation.")
    p.add_argument(
        "--shoulder-pan-override-rad",
        type=float,
        default=None,
        help=(
            "Override joint 0 (base rotation) only, on top of whichever pose is resolved -- "
            "for real-world wall/obstacle clearance, same pattern used 2026-07-28 "
            "(-0.7853981633974483 rad = -45deg for the alpha=0.1 pose). Does not change "
            "the rest of the arm's configuration. Always re-check with --dry-run first: the "
            "same angle does not guarantee the same clearance at a different pose."
        ),
    )
    p.add_argument("--speed-rad-s", type=float, default=0.5)
    p.add_argument("--acceleration-rad-s2", type=float, default=0.5)
    p.add_argument("--q-tolerance-rad", type=float, default=0.03)
    p.add_argument("--skip-dashboard-power-on", action="store_true")
    p.add_argument("--dry-run", action="store_true", help="Print target q only; no RTDE connection.")
    p.add_argument("--i-understand-this-moves-the-robot", dest="motion_opt_in", action="store_true")
    p.add_argument("--yes", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    target_q = _resolve_pose(args.pose, args.height_alpha, args.shoulder_pan_override_rad)
    print(f"target_q_rad={target_q}")

    if args.dry_run:
        return 0

    if not args.motion_opt_in:
        print("Refusing motion without --i-understand-this-moves-the-robot", file=sys.stderr)
        return 2
    if not args.yes:
        typed = input("Type MOVE to command a joint-space moveJ: ").strip()
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
    if not remote:
        print("Remote control is OFF — see docs/hardware/URSIM_REMOTE_CONTROL.md", file=sys.stderr)
        return 2

    link = UR5eDirectTorqueLink(args.robot_ip, frequency_hz=125.0)
    estop = EStopLatch()
    try:
        link.connect()
    except RTDELinkError as exc:
        print(f"RTDE connect failed: {exc}", file=sys.stderr)
        return 1

    result = move_joints_to_pose(
        link,
        estop,
        target_q_rad=target_q,
        motion_opt_in=True,
        speed_rad_s=float(args.speed_rad_s),
        acceleration_rad_s2=float(args.acceleration_rad_s2),
        q_tolerance_rad=float(args.q_tolerance_rad),
    )
    ok_verify, reason, final_q = verify_joint_pose(
        link,
        target_q_rad=target_q,
        q_tolerance_rad=float(args.q_tolerance_rad),
    )
    link.safe_stop("joint_move_complete")

    summary = {
        "ok": result.ok and ok_verify,
        "move_result": {
            "ok": result.ok,
            "reason": result.reason,
            "elapsed_s": result.elapsed_s,
            "final_q_rad": None if result.final_q_rad is None else result.final_q_rad.tolist(),
        },
        "verify_ok": ok_verify,
        "verify_reason": reason,
        "final_q_rad": final_q.tolist(),
        "target_q_rad": target_q,
    }
    print(json.dumps(summary, indent=2))
    return 0 if summary["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
