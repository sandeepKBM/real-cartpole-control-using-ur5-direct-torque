"""Joint-space repositioning for the direct-torque RTDE lane."""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from .direct_torque_link import UR5eDirectTorqueLink
from .link import RTDEStateError
from .safety import EStopLatch, UR5eSafetyLimits, check_joint_state


@dataclass
class JointMoveResult:
    ok: bool
    reason: str
    final_q_rad: np.ndarray | None
    elapsed_s: float


def move_joints_to_pose(
    link: UR5eDirectTorqueLink,
    estop: EStopLatch,
    *,
    target_q_rad: np.ndarray,
    motion_opt_in: bool,
    speed_rad_s: float = 0.5,
    acceleration_rad_s2: float = 0.5,
    q_tolerance_rad: float = 0.03,
    settle_timeout_s: float = 30.0,
) -> JointMoveResult:
    """Move to a named joint pose with ``moveJ`` before a direct-torque session."""
    estop.raise_if_tripped()
    if not motion_opt_in:
        return JointMoveResult(
            ok=False,
            reason="motion is blocked until motion_opt_in is enabled",
            final_q_rad=None,
            elapsed_s=0.0,
        )

    target_q = np.asarray(target_q_rad, dtype=np.float64).reshape(6)
    t0 = time.monotonic()
    try:
        try:
            link.direct_torque(np.zeros(6), friction_comp=True)
        except Exception:
            pass
        link.move_j(
            target_q,
            speed_rad_s=float(speed_rad_s),
            acceleration_rad_s2=float(acceleration_rad_s2),
        )
        deadline = time.monotonic() + float(settle_timeout_s)
        final_q = None
        while time.monotonic() < deadline:
            state = link.read_state()
            final_q = np.asarray(state.q, dtype=np.float64).reshape(6)
            if float(np.max(np.abs(final_q - target_q))) <= float(q_tolerance_rad):
                return JointMoveResult(
                    ok=True,
                    reason="",
                    final_q_rad=final_q,
                    elapsed_s=time.monotonic() - t0,
                )
            time.sleep(0.05)
        reason = f"joint move did not settle within {q_tolerance_rad} rad in {settle_timeout_s}s"
        estop.trip(reason)
        return JointMoveResult(ok=False, reason=reason, final_q_rad=final_q, elapsed_s=time.monotonic() - t0)
    except RTDEStateError as exc:
        reason = f"joint move failed: {exc}"
        link.safe_stop(reason)
        estop.trip(reason)
        return JointMoveResult(ok=False, reason=reason, final_q_rad=None, elapsed_s=time.monotonic() - t0)


def verify_joint_pose(
    link: UR5eDirectTorqueLink,
    *,
    target_q_rad: np.ndarray,
    q_tolerance_rad: float = 0.03,
    limits: UR5eSafetyLimits | None = None,
) -> tuple[bool, str, np.ndarray]:
    """Read state and check joint limits + pose tolerance."""
    limits = limits or link.limits
    state = link.read_state()
    q = np.asarray(state.q, dtype=np.float64).reshape(6)
    decision = check_joint_state(q, qd=state.qd, limits=limits)
    if not decision.ok:
        return False, decision.reason or "joint_state_check_failed", q
    err = float(np.max(np.abs(q - np.asarray(target_q_rad, dtype=np.float64).reshape(6))))
    if err > float(q_tolerance_rad):
        return False, f"joint pose error {err:.4f} rad exceeds tolerance {q_tolerance_rad}", q
    return True, "", q
