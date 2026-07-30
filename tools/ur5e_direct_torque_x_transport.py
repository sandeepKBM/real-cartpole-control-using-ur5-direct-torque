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

import numpy as np

from _bootstrap import ensure_repo_root

ensure_repo_root()

from hardware.dashboard import power_on_and_release, query_remote_control  # noqa: E402
from hardware.direct_torque_link import UR5eDirectTorqueLink  # noqa: E402
from hardware.link import RTDELinkError, UR5eLink  # noqa: E402
from hardware.safety import NOISE_ROBUST_GUARD_OVERRIDES  # noqa: E402
from hardware.x_transport import run_x_transport  # noqa: E402
from transport_metrics import GAIN_FIELDS  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "config" / "ur5e_mujoco_torque_osc_tuned.yaml"


def _normalize_gain_overrides(raw: str) -> dict[str, float]:
    """Same convention as tools/ur5e_move_hold_transport.py's sim-side helper:
    ``raw`` is either a path to a JSON file or an inline JSON object string.
    Unknown keys are silently dropped, matching that tool -- only the 11
    schedulable gain fields are ever valid overrides."""
    path = Path(raw)
    text = path.read_text(encoding="utf-8") if path.exists() else raw
    payload = json.loads(text)
    if not isinstance(payload, dict):
        raise ValueError("--gain-overrides-json must decode to a JSON object")
    overrides: dict[str, float] = {}
    for key, value in payload.items():
        if key in GAIN_FIELDS:
            overrides[key] = float(value)
    return overrides


def resolve_move_limit_overrides(args: argparse.Namespace) -> dict[str, float | int]:
    """Combine --noise-robust-guards' preset (if set) with any explicit
    individual override flags, explicit always winning for a given field.

    The preset is applied first via dict.update(), then each explicit flag
    (if not None) overwrites its own key -- so a user who passes both
    --noise-robust-guards and e.g. --accel-gap-cycles 8 gets gap=8, not the
    preset's gap=5. See NOISE_ROBUST_GUARD_OVERRIDES in hardware/safety.py
    and docs/status/safety_envelope_backtest_2026-07-30.md for the evidence
    behind the preset values.
    """
    overrides: dict[str, float | int] = {}
    if args.noise_robust_guards:
        overrides.update(NOISE_ROBUST_GUARD_OVERRIDES)
    if args.max_tcp_accel_mps2 is not None:
        overrides["max_tcp_accel_mps2"] = float(args.max_tcp_accel_mps2)
    if args.accel_gap_cycles is not None:
        overrides["accel_gap_cycles"] = int(args.accel_gap_cycles)
    if args.speed_lowpass_alpha is not None:
        overrides["speed_lowpass_alpha"] = float(args.speed_lowpass_alpha)
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
    p.add_argument("--control-mode", choices=("position", "direct_torque", "urscript"), default="position")
    p.add_argument("--target-x-delta", type=float, default=0.02)
    p.add_argument("--move-duration", type=float, default=1.0)
    p.add_argument("--duration", type=float, default=3.0)
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument(
        "--start-q-rad",
        type=float,
        nargs=6,
        default=None,
        metavar=("Q1", "Q2", "Q3", "Q4", "Q5", "Q6"),
        help=(
            "Optional six-joint pose in radians to moveJ to before transport, "
            "overriding the default HEIGHT_ALPHA_0_5_Q pose. E.g. the alpha=0.1 "
            "pose used in sim validation: 0.0 -1.423717 -0.24 -1.453717 0.0 0.0"
        ),
    )
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
        choices=("rtde", "local", "local_pinocchio"),
        default="local",
        help=(
            "direct_torque only: rtde=PolyScope J+M; local=MuJoCo J+M from q "
            "(default, legacy behavior); local_pinocchio=opt-in Pinocchio-backed "
            "J+M fast path, ~10x lower per-call latency, see "
            "docs/status/local_dynamics_speedup_investigation_2026-07-29.md."
        ),
    )
    p.add_argument(
        "--gain-overrides-json",
        default=None,
        help=(
            "direct_torque only. Path to a JSON file or an inline JSON object of "
            "gain overrides (same 11 fields as controller_core's schedulable "
            "gains), applied via controller.set_gains() before the robot moves. "
            "For live retuning between trials without editing --config."
        ),
    )
    p.add_argument(
        "--max-tcp-accel-mps2",
        type=float,
        default=None,
        help=(
            "All three control modes. Explicit, opt-in override of "
            "CartesianMoveLimits.max_tcp_accel_mps2 (class default 0.5 m/s^2). Added "
            "2026-07-28: on real hardware the naive one-step-finite-difference accel "
            "estimate amplifies raw RTDE position noise (~1/dt^2 -- ~15,600x at "
            "position mode's 125Hz, ~250,000x at direct_torque's 500Hz) during a "
            "min-jerk move's near-zero-velocity onset, tripping spuriously (observed "
            "0.72 and 0.90 m/s^2 in position mode across two trials, trip point "
            "varying step 1 vs step 6, every other metric -- drift/orientation/qd -- "
            "negligible). Does not fix the underlying numerical issue; a deliberate, "
            "visible override for continuing real-hardware testing, not a silent "
            "threshold change."
        ),
    )
    p.add_argument(
        "--accel-gap-cycles",
        type=int,
        default=None,
        help=(
            "All three control modes. Explicit override of "
            "CartesianMoveLimits.accel_gap_cycles (class default 1 = original "
            "single-cycle behavior). Added 2026-07-28 after "
            "tools/analyze_state_noise_capture.py measured the accel estimate's own "
            "noise floor from a real stationary RTDE capture: median 1.74 m/s^2 at "
            "gap=1, already ~3.5x the 0.5 default. Using position from N cycles back "
            "(instead of 1) to form each speed sample fed into the accel estimate "
            "cuts noise sensitivity substantially without losing detection of a real, "
            "sustained fast motion -- see CartesianMoveLimits' docstring for the "
            "mechanism. Combine with --speed-lowpass-alpha and re-run "
            "analyze_state_noise_capture.py (it accepts the same two flags) against a "
            "real stationary capture before picking --max-tcp-accel-mps2."
        ),
    )
    p.add_argument(
        "--speed-lowpass-alpha",
        type=float,
        default=None,
        help=(
            "All three control modes. Explicit override of "
            "CartesianMoveLimits.speed_lowpass_alpha (class default 1.0 = no "
            "filtering). EMA smoothing factor in (0, 1] applied to the gap-windowed "
            "speed sample before differencing for the accel estimate -- smaller = "
            "more smoothing. See --accel-gap-cycles."
        ),
    )
    p.add_argument(
        "--accel-max-consecutive-violations",
        type=int,
        default=None,
        help=(
            "All three control modes. Explicit override of "
            "CartesianMoveLimits.accel_max_consecutive_violations (class default 1 "
            "= original instant-trip behavior). Added 2026-07-30 (commit 8eccd1d) -- "
            "DeadlineMonitor-style graduated tolerance: this many consecutive "
            "over-threshold cycles before tripping, tolerating an isolated noise "
            "spike while still catching a sustained real trend. See "
            "--noise-robust-guards below: real-hardware-noise-replay backtest "
            "evidence (docs/status/safety_envelope_backtest_2026-07-30.md) found "
            "this field ALONE, at its own validated value, does not eliminate "
            "spurious trips -- it must be combined with --accel-gap-cycles/"
            "--speed-lowpass-alpha to actually work."
        ),
    )
    p.add_argument(
        "--accel-hard-multiple",
        type=float,
        default=None,
        help=(
            "All three control modes. Explicit override of "
            "CartesianMoveLimits.accel_hard_multiple (class default 5.0). A single "
            "cycle at or above max_tcp_accel_mps2 * this multiple trips immediately "
            "regardless of --accel-max-consecutive-violations, catching a genuine "
            "one-shot catastrophic event. See --accel-max-consecutive-violations."
        ),
    )
    p.add_argument(
        "--speed-max-consecutive-violations",
        type=int,
        default=None,
        help=(
            "All three control modes. Explicit override of "
            "CartesianMoveLimits.speed_max_consecutive_violations (class default 1). "
            "Same graduated-tolerance mechanism as --accel-max-consecutive-violations, "
            "applied to the TCP speed check instead of acceleration."
        ),
    )
    p.add_argument(
        "--speed-hard-multiple",
        type=float,
        default=None,
        help=(
            "All three control modes. Explicit override of "
            "CartesianMoveLimits.speed_hard_multiple (class default 5.0). Same "
            "immediate-trip safety valve as --accel-hard-multiple, applied to the "
            "TCP speed check."
        ),
    )
    p.add_argument(
        "--noise-robust-guards",
        action="store_true",
        help=(
            "All three control modes. Convenience flag applying the full "
            "validated 6-parameter combination found to actually close the "
            "real-hardware noise-driven-spurious-trip gap (2026-07-30 backtest, "
            "docs/status/safety_envelope_backtest_2026-07-30.md section 9, "
            "experiments/safety-envelope-study branch): the graduated-tolerance "
            "fields ALONE still spuriously tripped 30/30 replayed seeds at real "
            "measured RTDE noise magnitudes; only pairing them with "
            "accel_gap_cycles/speed_lowpass_alpha filtering closed it (0/30 "
            "spurious), while still catching the real genuine-catch case "
            "(-0.20m/1.0s move, theoretical peak accel 1.1547 m/s^2). Preset "
            "values: accel_max_consecutive_violations=3, accel_hard_multiple=5.0, "
            "speed_max_consecutive_violations=3, speed_hard_multiple=5.0, "
            "accel_gap_cycles=5, speed_lowpass_alpha=0.2 (see "
            "hardware.safety.NOISE_ROBUST_GUARD_OVERRIDES). The preset is applied "
            "first; any individual override flag above still wins for that "
            "specific field."
        ),
    )
    p.add_argument(
        "--coriolis-feedforward",
        action="store_true",
        help=(
            "direct_torque + dynamics-source=local only. The robot firmware "
            "auto-compensates gravity but NOT Coriolis/centrifugal forces inside "
            "directTorque() (confirmed against UR's own docs) -- this adds that "
            "missing term via MuJoCo. Default off; negligible below ~0.5 rad/s "
            "joint velocity, matters more for faster moves."
        ),
    )
    p.add_argument(
        "--disable-residual-observer",
        action="store_true",
        help=(
            "direct_torque only. Disable the diagnostic-only dynamics residual "
            "observer (default on) -- see "
            "docs/status/direct_torque_residual_observer_2026-07-29.md. It never "
            "affects control/torque, only trace logging, and already degrades "
            "gracefully (with a printed warning) if its PinocchioUR5eDynamics "
            "dependency fails to initialize -- but pass this to skip that "
            "dependency entirely, e.g. on a host where it's known to be broken."
        ),
    )
    return p.parse_args()


def _default_output_dir(control_mode: str) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return REPO_ROOT / "outputs" / "hardware_transport" / f"{control_mode}_{stamp}"


def main() -> int:
    args = parse_args()
    # Opt-in check moved before dashboard power-on / connect (matches
    # tools/ur5e_direct_torque_height_latency_test.py's already-correct
    # order) -- previously power_on_and_release() ran unconditionally, and
    # --probe-only --control-mode direct_torque opened a real control
    # connection + issued a real stopJ() before this check ever ran.
    needs_motion = not args.probe_only
    if needs_motion and not args.motion_opt_in:
        print("Refusing motion without --i-understand-this-moves-the-robot", file=sys.stderr)
        return 2
    if needs_motion and not args.yes:
        typed = input(f"Type MOVE to run X transport (mode={args.control_mode}): ").strip()
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
        if args.control_mode == "direct_torque":
            # Receive-only probe -- mirrors the position-mode probe below
            # (connect(with_control=False), plain disconnect(), no
            # stop/motion command ever issued).
            link = UR5eDirectTorqueLink(args.robot_ip, frequency_hz=500.0)
            try:
                link.connect(with_control=False)
            except RTDELinkError as exc:
                print(f"RTDE connect failed: {exc}", file=sys.stderr)
                return 1
            state = link.read_state()
            print(f"PROBE OK (direct_torque) q={state.q.round(4).tolist()} tcp_x={state.tcp_pose[0]:.4f}")
            link.disconnect()
            return 0
        link = UR5eLink(args.robot_ip, frequency_hz=125.0)
        link.connect(with_control=False)
        state = link.read_state()
        print(f"PROBE OK (position) q={state.q.round(4).tolist()} tcp_x={state.tcp_pose[0]:.4f}")
        link.disconnect()
        return 0

    output_dir = args.output_dir or _default_output_dir(str(args.control_mode))
    start_q_rad = None if args.start_q_rad is None else np.asarray(args.start_q_rad, dtype=np.float64)
    gain_overrides = None
    if args.gain_overrides_json is not None:
        try:
            gain_overrides = _normalize_gain_overrides(args.gain_overrides_json)
        except (ValueError, json.JSONDecodeError) as exc:
            print(f"--gain-overrides-json invalid: {exc}", file=sys.stderr)
            return 2
    move_limit_overrides = resolve_move_limit_overrides(args)
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
            start_q_rad=start_q_rad,
            coriolis_feedforward=bool(args.coriolis_feedforward),
            gain_overrides=gain_overrides,
            max_tcp_accel_mps2_override=move_limit_overrides.get("max_tcp_accel_mps2"),
            accel_gap_cycles_override=move_limit_overrides.get("accel_gap_cycles"),
            enable_residual_observer=not bool(args.disable_residual_observer),
            speed_lowpass_alpha_override=move_limit_overrides.get("speed_lowpass_alpha"),
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
