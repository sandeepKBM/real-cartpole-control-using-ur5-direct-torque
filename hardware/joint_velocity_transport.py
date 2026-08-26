"""X move+hold via ``speedJ`` (RTDE JOINT velocity streaming, IK resolved by us).

Fourth-and-a-half hardware control mode. Sibling of
``hardware/velocity_transport.py`` (speedL, CARTESIAN velocity streaming) --
reuses the exact same ``CartesianVelocityController``/``CartesianVelocityConfig``
machinery to build the desired Cartesian velocity ``xd_cmd`` from the same
min-jerk X move+hold profile, so the two modes are directly comparable. The
ONLY difference is the last step: instead of handing ``xd_cmd`` to the
robot's firmware (``speedL``, which inverts its own Jacobian internally),
this module inverts the Jacobian ITSELF via a singularity-robust damped
least-squares (DLS) resolver (``controller_core/damped_least_squares.py``)
and streams the resulting JOINT velocity via ``speedJ``.

**REQUIRES a config with ``reduced_task_dims``/``split_base_wrist_task``/
``ik_seeded_resolution`` all false** -- i.e. ``config/ur5e_speedj_joint_velocity.yaml``,
NOT ``config/ur5e_velocity_control.yaml`` (that config's default
``reduced_task_dims: true``). Any of those three flags makes
``CartesianVelocityController`` resolve the Cartesian task to a joint
velocity INTERNALLY and return ``xd_cmd = J @ qd_internal`` -- feeding that
into DLS below double-resolves (``DLS(J, J @ qd_internal)``), which is a
near-exact no-op at a well-conditioned pose but an uncharacterized
double-damped result exactly at the singularity this mode exists to handle
(found by review 2026-08-26; see
``tests/hardware/test_joint_velocity_resolution_fix.py`` for the measured
residuals and ``config/ur5e_speedj_joint_velocity.yaml``'s header for the
full story). ``run_x_transport_joint_velocity`` below fails fast if a loaded
config has any of the three flags set, so this is enforced, not just
documented.

Why this exists: real-hardware testing of speedL found the firmware's own
Cartesian-velocity IK protective-stops at the wrist-2 kinematic singularity
(measured: a +X move completed 100%, the mirrored -X move singularity-
stopped instead of degrading). A DLS resolver computed here does not have
that failure mode -- as the Jacobian's smallest singular value shrinks, the
resolver smoothly increases damping and returns a BOUNDED joint velocity
instead of raising an IK error, at the cost of degraded Cartesian tracking
near the singularity. This still uses the robot's own joint SERVO loop
(speedJ ramps toward each streamed setpoint under firmware control, exactly
like speedL does for Cartesian setpoints) -- only the IK step moves from the
firmware to us. Like speedL mode, this has ZERO force compliance (no gravity
comp, no mass matrix, no torque) -- appropriate only for pure point-to-point
transport / range characterization, not once a physical pole is mounted.

Jacobian source: the SAME ``LocalMujocoDynamics`` (MuJoCo-backed, matches the
sim lane bit-for-bit) the direct_torque loop and velocity_transport.py's
``reduced_task_dims`` path already use -- computed fresh from the real
``q`` every cycle (mandatory here, unlike velocity_transport.py where it's
conditional on the controller config, since DLS needs a Jacobian every cycle
regardless of what the Cartesian-velocity law itself needs internally).

Shares the SAME safety stack as velocity_transport.py (CartesianMoveMonitor,
DeadlineMonitor, StaleStateMonitor, EStopLatch, robot safety-status check
every cycle) -- this is a new command path, not a new safety policy. On top
of that shared stack, the commanded joint velocity is HARD-CLAMPED per-joint
(``joint_velocity_clamp_radps``) before every ``speed_j()`` call -- mandatory
even with DLS's own bounded output, since DLS bounds ``|qd|`` only as
``sigma_min -> 0``; away from the singularity an aggressive Cartesian
velocity target can still resolve to a large joint velocity through a
well-conditioned Jacobian, and the clamp is the last line of defense before
that reaches the robot.
"""

from __future__ import annotations

import gc
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from controller_core.cartesian_velocity_controller import (
    CartesianVelocityConfig,
    CartesianVelocityController,
)
from controller_core.damped_least_squares import damped_least_squares_qd
from controller_core.kinematics_utils import rotvec_to_quat_wxyz
from simulation.ur5e_mujoco_torque import x_profile_target
from transport_metrics import compute_valid_move_hold_metrics, summarize_move_hold_trace

from .link import RTDEStateError, UR5eLink
from .local_dynamics import LocalMujocoDynamics
from .position_transport import _load_cartesian_limits
from .safety import (
    CartesianMoveMonitor,
    DeadlineMonitor,
    EStopLatch,
    StaleStateMonitor,
    UR5eSafetyLimits,
    is_robot_safety_normal,
)
from .transport_common import max_abs_qd_from_trace

# First-ever real-hardware default for this brand-new streaming mode -- see
# this module's docstring and tools/ur5e_speedj_x_transport.py's own caveat.
# Deliberately conservative (well under both the manufacturer ceiling
# ~3.15-3.20 rad/s and the operative CartesianMoveMonitor.qd_max_radps=1.5
# default): this clamp is the ONLY thing standing between an aggressive
# Cartesian target resolved through a well-conditioned Jacobian and a fast
# joint move, since DLS itself only bounds |qd| as sigma_min -> 0, not in
# general. Flagged as an open question pending real-hardware validation, NOT
# presented as a validated safe ceiling.
DEFAULT_JOINT_VELOCITY_CLAMP_RADPS = 0.3

# DLS variable-damping defaults -- see controller_core/damped_least_squares.py's
# module docstring for the formula. Both in singular-value units of the mixed
# linear/angular Jacobian (no single physical unit applies). Chosen as a
# starting point consistent with common DLS practice for a
# comparably-scaled 6-DOF arm Jacobian, NOT tuned or validated against this
# robot -- open question, see this module's/the CLI's docstrings.
DEFAULT_DAMPING_LAMBDA_MAX = 0.05
DEFAULT_DAMPING_SIGMA0 = 0.05


@dataclass
class JointVelocityTransportResult:
    ok: bool
    reason: str
    summary: dict[str, Any]
    trace_path: Path | None


def run_x_transport_joint_velocity(
    link: UR5eLink,
    *,
    config_path: Path,
    target_x_delta_m: float,
    move_duration_s: float,
    duration_s: float,
    output_dir: Path | None = None,
    motion_opt_in: bool,
    rate_hz: float = 125.0,
    speed_j_acceleration: float = 1.2,
    joint_velocity_clamp_radps: float = DEFAULT_JOINT_VELOCITY_CLAMP_RADPS,
    damping_lambda_max: float = DEFAULT_DAMPING_LAMBDA_MAX,
    damping_sigma0: float = DEFAULT_DAMPING_SIGMA0,
    max_tcp_accel_mps2_override: float | None = None,
    max_tcp_speed_mps_override: float | None = None,
    accel_gap_cycles_override: int | None = None,
    speed_lowpass_alpha_override: float | None = None,
    speed_limit_gap_cycles_override: int | None = None,
    speed_limit_lowpass_alpha_override: float | None = None,
    accel_max_consecutive_violations_override: int | None = None,
    accel_hard_multiple_override: float | None = None,
    speed_max_consecutive_violations_override: int | None = None,
    speed_hard_multiple_override: float | None = None,
) -> JointVelocityTransportResult:
    """Stream the same min-jerk move+hold X profile through ``speedJ``, with
    the Cartesian-to-joint resolution done here via DLS instead of by the
    robot's own firmware.

    Every cycle: read state -> CartesianVelocityController.compute() builds
    the desired Cartesian velocity xd_cmd (X feedforward + P correction,
    Y/Z/orientation held) exactly as velocity_transport.py does -> DLS
    resolves xd_cmd through the current Jacobian into a joint velocity ->
    that joint velocity is hard-clamped -> streamed via link.speed_j().
    """
    estop = EStopLatch()
    if not motion_opt_in:
        raise ValueError("motion_opt_in must be True for live joint-velocity transport")

    move_duration_s = float(move_duration_s)
    duration_s = float(duration_s)
    if move_duration_s <= 0.0 or duration_s <= 0.0:
        raise ValueError("move_duration_s and duration_s must be positive")
    if move_duration_s > duration_s:
        raise ValueError("move_duration_s must not exceed duration_s")
    if not np.isfinite(joint_velocity_clamp_radps) or joint_velocity_clamp_radps <= 0.0:
        raise ValueError("joint_velocity_clamp_radps must be positive and finite")

    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    velocity_cfg = CartesianVelocityConfig.from_controller_yaml_section(cfg.get("controller", {}) or {})
    # HARD REQUIREMENT, not a preference: this mode's whole premise (see
    # module docstring) is that CartesianVelocityController hands back a
    # PURE Cartesian velocity (compute_full_hold: `return xd_full`, no
    # Jacobian) and DLS below is the ONLY Cartesian-to-joint resolution
    # step. Any of these three flags makes the controller resolve the task
    # to a joint velocity INTERNALLY first (qd_internal) and return
    # xd_cmd = J @ qd_internal -- DLS would then compute
    # DLS(J, J @ qd_internal), a second, uncharacterized resolution stacked
    # on top of the first (measured residual vs. qd_internal: ~6.5e-17 at a
    # well-conditioned pose, i.e. DLS becomes a silent no-op there, but
    # ~1.3e-2 at the ARM_Q0 singularity this mode exists to handle -- a real,
    # unvalidated double-damped result exactly where correctness matters
    # most). Fail fast instead of silently double-resolving; see
    # config/ur5e_speedj_joint_velocity.yaml for the dedicated config that
    # keeps all three off.
    if velocity_cfg.reduced_task_dims or velocity_cfg.split_base_wrist_task or velocity_cfg.ik_seeded_resolution:
        raise ValueError(
            "config_path's controller.velocity_control must have reduced_task_dims, "
            "split_base_wrist_task, and ik_seeded_resolution all false/absent -- any of "
            "them makes CartesianVelocityController resolve the Cartesian task to a joint "
            "velocity internally, and DLS below would then double-resolve "
            "(DLS(J, J @ qd_internal)) instead of being the sole resolver. Use "
            "config/ur5e_speedj_joint_velocity.yaml (or an equivalent config with all "
            "three flags off), not config/ur5e_velocity_control.yaml."
        )
    controller = CartesianVelocityController(velocity_cfg)
    # Unlike velocity_transport.py, the Jacobian is MANDATORY here every
    # cycle regardless of velocity_cfg's own resolution mode -- DLS needs it
    # to convert xd_cmd into a joint velocity even when the Cartesian-
    # velocity law itself (e.g. compute_full_hold) never touches a Jacobian.
    local_dyn = LocalMujocoDynamics()

    monitor = CartesianMoveMonitor(
        _load_cartesian_limits(
            config_path,
            link.robot_ip,
            max_tcp_accel_mps2_override=max_tcp_accel_mps2_override,
            max_tcp_speed_mps_override=max_tcp_speed_mps_override,
            accel_gap_cycles_override=accel_gap_cycles_override,
            speed_lowpass_alpha_override=speed_lowpass_alpha_override,
            speed_limit_gap_cycles_override=speed_limit_gap_cycles_override,
            speed_limit_lowpass_alpha_override=speed_limit_lowpass_alpha_override,
            accel_max_consecutive_violations_override=accel_max_consecutive_violations_override,
            accel_hard_multiple_override=accel_hard_multiple_override,
            speed_max_consecutive_violations_override=speed_max_consecutive_violations_override,
            speed_hard_multiple_override=speed_hard_multiple_override,
        )
    )
    dt_s = 1.0 / float(rate_hz)

    link.connect(with_control=True)
    link.verify_speedj_signature()
    try:
        state0 = link.read_state()
    except RTDEStateError as exc:
        estop.trip(f"initial state read failed: {exc}")
        link.safe_stop(str(exc))
        return JointVelocityTransportResult(
            ok=False,
            reason=str(exc),
            summary={"success": False, "termination_reason": str(exc)},
            trace_path=None,
        )

    start_pose = state0.tcp_pose.copy()
    x0 = float(start_pose[0])
    monitor.set_start(start_pose, move_axis_index=0)
    init_state = {
        "time": 0.0,
        "q": state0.q,
        "qd": state0.qd,
        "ee_pos": start_pose[:3].copy(),
        "ee_quat": rotvec_to_quat_wxyz(start_pose[3:6]),
        "target_x": x0,
    }
    controller.reset_from_state(init_state)

    trace_rows: list[dict[str, Any]] = []
    termination_reason = "duration_complete"
    steps = 0
    t_s = 0.0
    last_state = state0

    safety_limits = getattr(link, "limits", None) or UR5eSafetyLimits()
    deadline_monitor = DeadlineMonitor(safety_limits.max_deadline_ms)
    stale_monitor = StaleStateMonitor()

    gc.disable()
    try:
        while t_s < duration_s - 1e-12:
            estop.raise_if_tripped()
            cycle_start = time.monotonic()

            target_x, target_x_vel = x_profile_target(
                "min_jerk_move_hold",
                x0,
                float(target_x_delta_m),
                t_s,
                duration_s,
                move_duration_s=move_duration_s,
            )
            target_ee_pos = start_pose[:3].copy()
            target_ee_pos[0] = float(target_x)
            target_ee_vel = np.array([target_x_vel, 0.0, 0.0], dtype=np.float64)
            waypoint = start_pose.copy()
            waypoint[0] = float(target_x)

            try:
                link_state = link.read_state()
            except RTDEStateError as exc:
                termination_reason = f"rtde_state_error: {exc}"
                estop.trip(termination_reason)
                break

            jacobian = local_dyn.jacobian(link_state.q)

            robot_state = {
                "time": t_s,
                "q": link_state.q,
                "qd": link_state.qd,
                "ee_pos": link_state.tcp_pose[:3].copy(),
                "ee_quat": rotvec_to_quat_wxyz(link_state.tcp_pose[3:6]),
                "target_x": float(target_x),
                "target_ee_pos": target_ee_pos,
                "target_ee_vel": target_ee_vel,
                "jacobian": jacobian,
            }
            xd_cmd = controller.compute(robot_state)

            dls_result = damped_least_squares_qd(
                jacobian,
                xd_cmd,
                lambda_max=float(damping_lambda_max),
                sigma0=float(damping_sigma0),
            )
            qd_cmd_unclamped = dls_result.qd
            qd_cmd = np.clip(qd_cmd_unclamped, -joint_velocity_clamp_radps, joint_velocity_clamp_radps)
            qd_clamp_hit = bool(np.any(np.abs(qd_cmd_unclamped) > joint_velocity_clamp_radps))

            try:
                link.speed_j(qd_cmd, acceleration=float(speed_j_acceleration), time_s=dt_s * 1.5)
            except RTDEStateError as exc:
                termination_reason = f"rtde_state_error: {exc}"
                estop.trip(termination_reason)
                break

            last_state = link_state

            stale_reason = stale_monitor.record(link_state.robot_timestamp_s, link_state.host_stamp_ns)
            if stale_reason:
                termination_reason = stale_reason
                estop.trip(termination_reason)
                break

            orientation_error_rad = float(np.linalg.norm(link_state.tcp_pose[3:] - start_pose[3:]))
            decision = monitor.check(
                q=link_state.q,
                qd=link_state.qd,
                tcp_pose=link_state.tcp_pose,
                target_tcp_pose=waypoint,
                orientation_error_rad=orientation_error_rad,
                axis_target_moving=bool(t_s <= move_duration_s),
                dt_s=dt_s,
            )
            if not decision.ok:
                termination_reason = decision.reason or "safety_stop"
                estop.trip(termination_reason)
                break

            if not is_robot_safety_normal(link_state.safety_status):
                termination_reason = f"robot_safety_status_abnormal: {link_state.safety_status}"
                estop.trip(termination_reason)
                break

            row: dict[str, Any] = {
                "time_s": t_s,
                "q": link_state.q.tolist(),
                "qd": link_state.qd.tolist(),
                "tcp_pose": link_state.tcp_pose.tolist(),
                "target_x": target_x,
                "target_x_vel": target_x_vel,
                "xd_cmd": xd_cmd.tolist(),
                "qd_cmd": qd_cmd.tolist(),
                "qd_cmd_unclamped": qd_cmd_unclamped.tolist(),
                "qd_clamp_hit": qd_clamp_hit,
                "sigma_min": dls_result.sigma_min,
                "dls_lambda_used": dls_result.lambda_used,
                "command_mode": "speedJ",
            }
            tcp = np.asarray(link_state.tcp_pose, dtype=np.float64).reshape(6)
            row["ee_pos"] = tcp[:3].tolist()
            row["ee_quat"] = rotvec_to_quat_wxyz(tcp[3:6]).tolist()
            row["x_error"] = float(target_x - tcp[0])
            row["orientation_error_norm"] = float(np.linalg.norm(tcp[3:] - start_pose[3:]))
            trace_rows.append(row)
            steps += 1
            t_s += dt_s

            elapsed_s = time.monotonic() - cycle_start
            overrun_ns = int(max(0.0, elapsed_s - dt_s) * 1e9)
            deadline_reason = deadline_monitor.record(overrun_ns)
            if deadline_reason:
                termination_reason = deadline_reason
                estop.trip(termination_reason)
                break

            sleep_s = dt_s - elapsed_s
            if sleep_s > 0:
                time.sleep(sleep_s)
    finally:
        gc.enable()
        link.speed_stop()
        link.safe_stop("transport_exit")

    achieved_x_delta_m = float(last_state.tcp_pose[0] - x0)
    max_abs_qd = max_abs_qd_from_trace(trace_rows)
    min_sigma_min = min((float(r["sigma_min"]) for r in trace_rows), default=None)
    max_dls_lambda_used = max((float(r["dls_lambda_used"]) for r in trace_rows), default=None)
    any_qd_clamp_hit = any(bool(r["qd_clamp_hit"]) for r in trace_rows)
    summary = {
        "backend": "speedJ_joint_velocity",
        "control_mode": "joint_velocity",
        "config_path": str(config_path),
        "target_x_delta": float(target_x_delta_m),
        "target_x_delta_m": float(target_x_delta_m),
        "transport_axis_index": 0,
        "move_duration_s": move_duration_s,
        "hold_duration_s": max(duration_s - move_duration_s, 0.0),
        "duration_s": duration_s,
        "total_duration_s": duration_s,
        "sim_time_s": min(t_s, duration_s),
        "frequency_hz": float(rate_hz),
        "steps": steps,
        "termination_reason": termination_reason,
        "achieved_x_delta_m": achieved_x_delta_m,
        "final_tcp_pose": last_state.tcp_pose.tolist(),
        "initial_ee_pos": state0.tcp_pose[:3].tolist(),
        "success": termination_reason == "duration_complete" and not estop.tripped,
        "velocity_guard_ok": max_abs_qd <= monitor.limits.qd_max_radps,
        "max_abs_qd_radps": max_abs_qd,
        "joint_limit_guard_ok": True,
        "torque_saturation_percentage": 0.0,
        "joint_velocity_clamp_radps": float(joint_velocity_clamp_radps),
        "damping_lambda_max": float(damping_lambda_max),
        "damping_sigma0": float(damping_sigma0),
        "min_sigma_min": min_sigma_min,
        "max_dls_lambda_used": max_dls_lambda_used,
        "any_qd_clamp_hit": any_qd_clamp_hit,
    }
    summary.update(
        summarize_move_hold_trace(
            trace_rows,
            initial_ee_pos=state0.tcp_pose[:3],
            move_duration_s=move_duration_s,
            total_duration_s=duration_s,
            transport_axis_index=0,
        )
    )
    summary.update(compute_valid_move_hold_metrics(summary))
    summary["success"] = bool(
        not estop.tripped
        and summary.get("valid_move_and_hold", False)
        and termination_reason == "duration_complete"
    )

    trace_path = None
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        trace_path = output_dir / "trace.jsonl"
        with trace_path.open("w", encoding="utf-8") as fh:
            for row in trace_rows:
                fh.write(json.dumps(row) + "\n")
        (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    ok = bool(summary["success"])
    reason = "" if ok else str(summary.get("hold_failure_reason") or summary.get("move_failure_reason") or termination_reason)
    return JointVelocityTransportResult(ok=ok, reason=reason, summary=summary, trace_path=trace_path)
