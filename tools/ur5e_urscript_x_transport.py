"""Generate URScript OSC inner-loop transport."""

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
from hardware.link import RTDELinkError  # noqa: E402
from hardware.poses import HEIGHT_ALPHA_0_5_Q  # noqa: E402
from hardware.urscript_gen import DEFAULT_CONFIG  # noqa: E402
from hardware.urscript_transport import run_urscript_x_transport  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Run tuned OSC X transport with a 500 Hz URScript inner loop on PolyScope "
            "(direct_torque V2, on-robot Jacobian/M). Python only supervises safety."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--robot-ip", required=True)
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--target-x-delta", type=float, default=0.02)
    p.add_argument("--move-duration", type=float, default=1.0)
    p.add_argument("--duration", type=float, default=3.0)
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--skip-joint-move", action="store_true")
    p.add_argument("--skip-dashboard-power-on", action="store_true")
    p.add_argument("--no-lambda", action="store_true", help="Kinematic J.T wrench only (faster, less matched to sim).")
    p.add_argument("--generate-only", action="store_true", help="Write generated .script and exit (no robot).")
    p.add_argument("--i-understand-this-moves-the-robot", dest="motion_opt_in", action="store_true")
    p.add_argument("--yes", action="store_true")
    return p.parse_args()


def _default_output_dir() -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return REPO_ROOT / "outputs" / "hardware_urscript" / f"x_transport_{stamp}"


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir or _default_output_dir()

    if args.generate_only:
        from hardware.urscript_gen import load_params_from_yaml, write_generated_script

        params = load_params_from_yaml(
            args.config,
            target_x_delta_m=float(args.target_x_delta),
            move_duration_s=float(args.move_duration),
            duration_s=float(args.duration),
            use_lambda=not args.no_lambda,
        )
        path = write_generated_script(params, output_dir / "x_axis_osc_inner.script")
        print(f"Wrote {path}")
        return 0

    if not args.motion_opt_in:
        print("Refusing motion without --i-understand-this-moves-the-robot", file=sys.stderr)
        return 2
    if not args.yes:
        typed = input("Type MOVE to run URScript inner-loop transport: ").strip()
        if typed != "MOVE":
            print("Aborted.", file=sys.stderr)
            return 2

    if not args.skip_dashboard_power_on:
        for cmd, resp in power_on_and_release(args.robot_ip).items():
            print(f"dashboard {cmd}: {resp}")
        time.sleep(2.0)

    if not query_remote_control(args.robot_ip):
        print("Remote control is OFF — see docs/hardware/URSIM_REMOTE_CONTROL.md", file=sys.stderr)
        return 2

    try:
        result = run_urscript_x_transport(
            robot_ip=args.robot_ip,
            config_path=args.config,
            target_x_delta_m=float(args.target_x_delta),
            move_duration_s=float(args.move_duration),
            duration_s=float(args.duration),
            output_dir=output_dir,
            motion_opt_in=True,
            skip_joint_move=bool(args.skip_joint_move),
            use_lambda=not args.no_lambda,
        )
    except RTDELinkError as exc:
        print(f"RTDE failed: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(result.summary, indent=2))
    if result.script_path is not None:
        print(f"script: {result.script_path}")
    return 0 if result.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
