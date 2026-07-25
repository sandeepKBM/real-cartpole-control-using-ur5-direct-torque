#!/usr/bin/env python3
"""End-to-end hardware test: joint move → X transport with selectable control mode.

Default ``--control-mode position`` uses ``servoL`` (works on URSim + real UR5) and
runs the tuned OSC controller in **shadow** (logs ``tau_shadow``, does not command
torques). Switch to ``direct_torque`` or ``urscript`` for live torque on the real robot.

Examples:
  # Component test on URSim or real arm (position / servoL):
  python tools/ur5e_direct_torque_height_latency_test.py --robot-ip <IP> \\
    --control-mode position --target-x-delta 0.02 --move-duration 1.0 --duration 3.0 \\
    --i-understand-this-moves-the-robot --yes

  # Live direct torque (real UR5, wired PC):
  python tools/ur5e_direct_torque_height_latency_test.py --robot-ip <IP> \\
    --control-mode direct_torque --dynamics-source local \\
    --target-x-delta 0.02 --i-understand-this-moves-the-robot --yes

  # Offline Python OSC latency mock:
  python tools/ur5e_direct_torque_height_latency_test.py --robot-ip 127.0.0.1 \\
    --latency-only-mock --control-mode direct_torque
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from _bootstrap import ensure_repo_root

ensure_repo_root()

from hardware.dashboard import power_on_and_release, query_remote_control  # noqa: E402
from hardware.link import RTDELinkError  # noqa: E402
from hardware.poses import HEIGHT_ALPHA_0_5_Q  # noqa: E402
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
    p.add_argument("--latency-only-mock", action="store_true", help="Mock direct_torque loop (no robot).")
    p.add_argument("--skip-dashboard-power-on", action="store_true")
    p.add_argument("--dynamics-source", choices=("rtde", "local"), default="local")
    p.add_argument("--i-understand-this-moves-the-robot", dest="motion_opt_in", action="store_true")
    p.add_argument("--yes", action="store_true")
    return p.parse_args()


def _default_output_dir(control_mode: str) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return REPO_ROOT / "outputs" / "hardware_transport" / f"{control_mode}_{stamp}"


def _print_latency_report(summary: dict[str, Any]) -> None:
    timing = summary.get("timing", {})
    phases = summary.get("latency_phases", {})
    if not timing and not phases:
        print(f"\n=== Summary (mode={summary.get('control_mode', summary.get('backend', '?'))}) ===")
        print(f"  shadow_osc: {summary.get('shadow_osc', False)}")
        print(f"  success: {summary.get('success')}")
        return
    print("\n=== Latency report ===")
    print(f"  target period: {timing.get('target_period_s', 0) * 1000:.3f} ms")
    work = timing.get("work_duration", {})
    if work.get("mean_ms") is not None:
        print(f"  cycle work mean/p95/max: {work['mean_ms']:.3f} / {work.get('p95_ms', 0):.3f} / {work['max_ms']:.3f} ms")
    print(f"  late cycles: {timing.get('late_cycles', 0)} / {timing.get('cycle_count', 0)}")


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir or _default_output_dir(str(args.control_mode))
    pipeline_summary: dict[str, Any] = {
        "robot_ip": args.robot_ip,
        "config_path": str(args.config),
        "control_mode": str(args.control_mode),
        "target_q_rad": HEIGHT_ALPHA_0_5_Q.tolist(),
        "dynamics_source": str(args.dynamics_source),
        "steps": {},
    }

    if args.latency_only_mock:
        if args.control_mode != "direct_torque":
            print("--latency-only-mock requires --control-mode direct_torque", file=sys.stderr)
            return 2
        from hardware.direct_torque_transport import run_x_transport_direct_torque
        from hardware.link import UR5eState

        class _LatencyMockLink:
            def __init__(self) -> None:
                self._tcp_x = 0.4

            def connect(self) -> None:
                return None

            def read_state(self) -> UR5eState:
                return UR5eState(
                    q=HEIGHT_ALPHA_0_5_Q.copy(),
                    qd=np.zeros(6),
                    tcp_pose=np.array([self._tcp_x, -0.2, 0.3, 0.0, 3.14, 0.0]),
                    host_stamp_ns=time.monotonic_ns(),
                    robot_timestamp_s=None,
                    safety_status=None,
                )

            def get_jacobian(self) -> np.ndarray:
                return np.eye(6)

            def get_mass_matrix(self) -> np.ndarray:
                return np.eye(6)

            def direct_torque(self, tau_nm: np.ndarray, *, friction_comp: bool = True) -> None:
                self._tcp_x += float(tau_nm[0]) * 1e-6

            def build_robot_state(self, link_state, *, time_s, target_x, target_x_vel):
                from hardware.direct_torque_link import UR5eDirectTorqueLink

                return UR5eDirectTorqueLink.compose_robot_state(
                    link_state,
                    jacobian=self.get_jacobian(),
                    mass_matrix=self.get_mass_matrix(),
                    time_s=time_s,
                    target_x=target_x,
                    target_x_vel=target_x_vel,
                )

            def compose_robot_state(self, link_state, *, jacobian, mass_matrix, time_s, target_x, target_x_vel):
                from hardware.direct_torque_link import UR5eDirectTorqueLink

                return UR5eDirectTorqueLink.compose_robot_state(
                    link_state,
                    jacobian=jacobian,
                    mass_matrix=mass_matrix,
                    time_s=time_s,
                    target_x=target_x,
                    target_x_vel=target_x_vel,
                )

            def safe_stop(self, reason: str) -> None:
                return None

        result = run_x_transport_direct_torque(
            _LatencyMockLink(),  # type: ignore[arg-type]
            config_path=args.config,
            target_x_delta_m=float(args.target_x_delta),
            move_duration_s=float(args.move_duration),
            duration_s=float(args.duration),
            output_dir=output_dir,
            motion_opt_in=True,
            dynamics_source=str(args.dynamics_source),
        )
        pipeline_summary["steps"]["mock_transport"] = result.summary
        _print_latency_report(result.summary)
        (output_dir / "pipeline_summary.json").write_text(json.dumps(pipeline_summary, indent=2), encoding="utf-8")
        print(f"\nWrote {output_dir / 'pipeline_summary.json'}")
        return 0

    needs_motion = not args.probe_only
    if needs_motion and not args.motion_opt_in:
        print("Refusing motion without --i-understand-this-moves-the-robot", file=sys.stderr)
        return 2
    if needs_motion and not args.yes:
        typed = input(f"Type MOVE to run transport (mode={args.control_mode}): ").strip()
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
        print("Remote control is OFF — see docs/hardware/URSIM_REMOTE_CONTROL.md", file=sys.stderr)
        return 2

    if args.probe_only:
        from hardware.link import UR5eLink

        link = UR5eLink(args.robot_ip, frequency_hz=125.0)
        link.connect(with_control=False)
        st = link.read_state()
        print(f"PROBE OK q={st.q.round(4).tolist()} tcp_x={st.tcp_pose[0]:.4f}")
        link.disconnect()
        return 0

    print(f"\n--- Transport mode={args.control_mode} ---")
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

    pipeline_summary["steps"]["transport"] = result.summary
    _print_latency_report(result.summary)
    (output_dir / "pipeline_summary.json").write_text(json.dumps(pipeline_summary, indent=2), encoding="utf-8")
    print(json.dumps(result.summary, indent=2))
    if result.trace_path:
        print(f"trace: {result.trace_path}")
    return 0 if result.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
