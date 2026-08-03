#!/usr/bin/env python3
"""Find the real maximum X-axis travel bound at a given pose, using
position-mode (servoL) motion -- NOT direct_torque.

Why position mode: this is a boundary-finding sweep, deliberately pushing
past whatever soft displacement limit earlier direct_torque work assumed
(e.g. the +-0.20m this session used out of caution while characterizing
open-loop torque behavior). servoL is UR's own trajectory-bounded motion
primitive -- velocity/acceleration are generated and limited by the robot
controller itself, not by an open-loop torque command that could run away.
That's a materially different risk profile, which is why this tool does
NOT impose its own soft displacement ceiling the way
tools/autonomous_transport_explorer.py deliberately does for direct_torque.

What's still real and NOT removed here: joint limits
(hardware/x_transport.py's own start-pose validation, plus
move_joints_to_pose/verify_joint_pose's post-move checks), the robot's own
reported safety status (checked every cycle in all motion loops per
hardware/safety.py), e-stop, and CartesianMoveMonitor's TCP speed/accel
checks (still active in position mode -- hardware/position_transport.py
wires it in same as direct_torque). This tool relies on those, plus
whatever independent workspace/travel limit the user has configured on the
robot's own controller, as the actual safety net -- it does not duplicate
them with an extra software clamp on top.

Every trial is still individually approved -- see
tools/autonomous_transport_explorer.py's docstring for why that's not
loosened just because the motion is lower-risk. Escalates +X and -X
independently: keeps stepping a direction's |dx| up by X_STEP_M as long as
trials succeed, stops and reports the last clean value plus the failure
mode the moment one doesn't.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from tools.diagnostics.transport_trial_diagnostics import diagnose  # noqa: E402

# -45 deg base-rotation pose -- the pose this repo's real hardware actually
# uses (AGENTS.md sec 3/4), not the wrist2-offset "zero" pose most of
# tonight's earlier work targeted. Matches hardware/poses.py's
# HEIGHT_ALPHA_0_5_CLEARANCE_Q exactly (shoulder_pan overridden to -45deg) --
# recomputed directly from ACTIVE_ORIGIN_Q/LOWER_B_Q rather than imported,
# since this tool should work standalone without importing hardware/ (no
# RTDE dependency needed just to print a pose constant).
NEG45_START_Q = [-0.7853981633974483, -0.8353981633974483, -1.2, -0.9853981633974482, 0.0, 0.0]

DEFAULT_CONFIG = "config/ur5e_mujoco_torque_osc_tuned_wrist_orient_fixed_neg45_pose.yaml"
DEFAULT_LOG_PATH = REPO_ROOT / "outputs" / "hardware_transport" / "rail_bound_finder_log.jsonl"
RUNS_GLOB = str(REPO_ROOT / "outputs" / "hardware_transport" / "position_*")

# Escalation step and starting point. No upper clamp on how far this climbs
# by design (see module docstring) -- SANITY_CEILING_M is a "something is
# clearly wrong" backstop, not a real expected limit.
X_START_M = 0.05
X_STEP_M = 0.05
SANITY_CEILING_M = 1.00
MOVE_DURATION_S = 6.0

COMMON_FLAGS = [
    "--dynamics-source",
    "local",
    "--noise-robust-guards",
    "--accel-variable-tolerance",
    "--speed-variable-tolerance",
    "--i-understand-this-moves-the-robot",
    "--yes",
]


def build_command(dx_signed: float, *, config: str, robot_ip: str) -> list[str]:
    return (
        [
            sys.executable,
            "tools/ur5e_direct_torque_x_transport.py",
            "--robot-ip",
            robot_ip,
            "--control-mode",
            "position",
            "--config",
            config,
            "--start-q-rad",
        ]
        + [f"{v:.6f}" for v in NEG45_START_Q]
        + [
            "--target-x-delta",
            f"{dx_signed:.4f}",
            "--move-duration",
            f"{MOVE_DURATION_S:.2f}",
            "--duration",
            f"{MOVE_DURATION_S + 2.0:.2f}",
        ]
        + COMMON_FLAGS
    )


def find_latest_run_dir(after_mtime: float) -> Path | None:
    candidates = [d for d in glob.glob(RUNS_GLOB) if os.path.isdir(d)]
    fresh = [d for d in candidates if os.path.getmtime(d) >= after_mtime]
    if not fresh:
        return None
    fresh.sort(key=os.path.getmtime, reverse=True)
    return Path(fresh[0])


def append_log(log_path: Path, record: dict) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a") as f:
        f.write(json.dumps(record, default=str) + "\n")


def escalate_direction(sign: int, *, config: str, robot_ip: str, log_path: Path) -> float:
    label = "+X" if sign > 0 else "-X"
    last_clean = 0.0
    dx = X_START_M
    print(f"\n{'=' * 78}\nESCALATING {label} from the -45deg pose (no software displacement cap)\n{'=' * 78}")
    while dx <= SANITY_CEILING_M:
        cmd = build_command(sign * dx, config=config, robot_ip=robot_ip)
        print(f"\nNext step: {label} dx={dx:.3f}m (last clean: {last_clean:.3f}m)")
        print("  " + " ".join(cmd))
        resp = input("Approve this trial? [y]es / [s]top escalating this direction / [q]uit entirely: ").strip().lower()
        if resp in ("q", "quit"):
            print("Quitting entirely at user request.")
            raise SystemExit(0)
        if resp in ("s", "stop"):
            print(f"Stopping {label} escalation by request. Last confirmed clean: {last_clean:.3f}m")
            return last_clean
        if resp not in ("y", "yes", ""):
            print("Unrecognized response, treating as stop for this direction.")
            return last_clean

        start_mtime = time.time()
        proc = subprocess.run(cmd, cwd=str(REPO_ROOT))
        run_dir = find_latest_run_dir(after_mtime=start_mtime - 5.0)
        if run_dir is None:
            print("WARNING: could not find the run directory this trial produced. Treating as inconclusive, not advancing.")
            continue

        diag = diagnose(run_dir)
        success = diag.get("success")
        print(
            f"  result: success={success} guard={diag.get('guard_category')} "
            f"achieved_fraction={diag.get('achieved_fraction')}"
        )
        append_log(
            log_path,
            {"record_type": "rail_bound_trial", "direction": label, "dx": dx, "exit_code": proc.returncode, **diag},
        )

        if success:
            last_clean = dx
            dx += X_STEP_M
        else:
            print(
                f"\n{label} bound found: last clean = {last_clean:.3f}m, "
                f"first failure at {dx:.3f}m ({diag.get('guard_category')})"
            )
            return last_clean

    print(f"Hit the sanity ceiling ({SANITY_CEILING_M}m) without any failure -- stopping, this is unexpected.")
    return last_clean


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--robot-ip", required=True)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--log-path", type=Path, default=DEFAULT_LOG_PATH)
    parser.add_argument(
        "--direction", choices=("both", "+x", "-x"), default="both", help="Which direction(s) to escalate."
    )
    args = parser.parse_args()

    results = {}
    if args.direction in ("both", "+x"):
        results["+X"] = escalate_direction(1, config=args.config, robot_ip=args.robot_ip, log_path=args.log_path)
    if args.direction in ("both", "-x"):
        results["-X"] = escalate_direction(-1, config=args.config, robot_ip=args.robot_ip, log_path=args.log_path)

    print(f"\n{'=' * 78}\nFINAL RESULT\n{'=' * 78}")
    for label, bound in results.items():
        print(f"  {label}: {bound:.3f}m")
    print(f"Log: {args.log_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
