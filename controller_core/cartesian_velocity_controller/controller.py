"""CartesianVelocityController: shared per-cycle setup (position error,
swing-twist orientation error, xd_full, mutual-exclusivity check), dispatch
to exactly one resolution-mode function from ``modes.py``, then the shared
speed clamp. See this package's ``__init__.py`` and ``modes.py`` docstrings
for the design history behind the four modes.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..kinematics_utils import swing_twist_axis_error
from ..state_types import as_robot_state
from .config import CartesianVelocityConfig
from .modes import (
    compute_full_hold,
    compute_ik_seeded,
    compute_reduced_task_dims,
    compute_split_base_wrist,
)


class CartesianVelocityController:
    """xd_cmd = feedforward + kp * (target - actual), clamped to configured
    speed ceilings. Holds Y/Z at their reset-time values unless the caller
    supplies a moving target_ee_pos/target_ee_vel for them. Orientation
    holding depends on reduced_task_dims -- see the package docstring.

    posture_reanchor_on_settle (default on): fixes a real bug found
    2026-08-03 -- pulling the null-space posture term toward the pose
    captured at reset() is fine WHILE the arm is still at that pose, but
    once a move completes and the arm holds a NEW X position, q_rest (the
    OLD pose) is no longer consistent with the current task -- the posture
    term keeps trying to pull q back toward a configuration the task no
    longer allows, and since the null-space direction relative to a fixed
    target isn't guaranteed to monotonically shrink as the null-space
    projector itself evolves with q, this manifested as UNBOUNDED
    orientation drift during the hold phase (measured: growing from 0 to
    beyond the 0.25 rad guard over ~1s of an otherwise-static hold, for a
    move as small as dx=0.02m). Fix, mirroring
    XAxisCartesianImpedanceController's posture_reanchor_on_settle: once
    the primary task's position error drops under reanchor_pos_tol_m,
    re-capture q_rest at the CURRENT q -- the posture term then has nothing
    left to correct at the settled pose instead of chasing a stale target
    forever."""

    def __init__(self, config: CartesianVelocityConfig) -> None:
        self.cfg = config
        self._initialized = False
        self._p0 = np.zeros(3, dtype=np.float64)
        self._quat0 = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        self._q_rest = np.zeros(6, dtype=np.float64)
        self._reanchored = False
        self._settled_cycles = 0

    def reset_from_state(self, state: dict[str, Any]) -> None:
        st = as_robot_state(state)
        self._p0 = np.asarray(st["ee_pos"], dtype=np.float64).reshape(3).copy()
        self._quat0 = np.asarray(st["ee_quat"], dtype=np.float64).reshape(4).copy()
        self._q_rest = np.asarray(st["q"], dtype=np.float64).reshape(6).copy()
        self._reanchored = False
        self._settled_cycles = 0
        self._initialized = True

    @property
    def initialized(self) -> bool:
        return self._initialized

    def compute(self, state: dict[str, Any]) -> np.ndarray:
        """Returns a 6D Cartesian velocity command [vx,vy,vz,wx,wy,wz] in the
        axis-angle/rotvec convention RTDE's speedL/getActualTCPPose use --
        NOT a quaternion, deliberately unlike every torque-path controller
        in this package (hence returning a plain ndarray, not a
        CartesianImpedanceOutput)."""
        if not self._initialized:
            raise RuntimeError("Call reset_from_state() before compute().")
        st = as_robot_state(state)
        p = np.asarray(st["ee_pos"], dtype=np.float64).reshape(3)
        quat = np.asarray(st["ee_quat"], dtype=np.float64).reshape(4)
        p_des = np.asarray(st.get("target_ee_pos", self._p0), dtype=np.float64).reshape(3)
        v_ff = np.asarray(st.get("target_ee_vel", np.zeros(3)), dtype=np.float64).reshape(3)

        pos_err = p_des - p
        kp_lin = np.array([self.cfg.kp_x, self.cfg.kp_y, self.cfg.kp_z], dtype=np.float64)
        v_cmd = v_ff + kp_lin * pos_err

        # Swing-twist per-axis decomposition, NOT orientation_error_vec_wxyz's
        # 2*vec(q_err) -- see swing_twist_axis_error's docstring. That
        # small-angle vector approximation mixes all 3 axes together for a
        # compound rotation and is not exact for large angles; naively using
        # its row 2 as "the rz error" was found (2026-08-03) to create a
        # genuine, reproducible, unbounded-growth instability once the other
        # two axes accumulated real rotation from redundancy-resolution
        # null-space motion -- confirmed independent of kp_posture (identical
        # divergence with kp_posture=0.0, i.e. no null-space term at all),
        # so this is a bug in the task's own orientation-error signal, not a
        # gain-tuning problem.
        w_cmd = self.cfg.kp_rot * np.array(
            [
                swing_twist_axis_error(self._quat0, quat, 0),
                swing_twist_axis_error(self._quat0, quat, 1),
                swing_twist_axis_error(self._quat0, quat, 2),
            ],
            dtype=np.float64,
        )

        xd_full = np.concatenate([v_cmd, w_cmd]).astype(np.float64)

        modes_on = sum([self.cfg.reduced_task_dims, self.cfg.split_base_wrist_task, self.cfg.ik_seeded_resolution])
        if modes_on > 1:
            raise ValueError(
                "reduced_task_dims, split_base_wrist_task, and ik_seeded_resolution "
                "are mutually exclusive -- enable at most one."
            )

        if self.cfg.ik_seeded_resolution:
            fk_jacobian_fn = state.get("fk_jacobian_fn")
            q_current = np.asarray(st["q"], dtype=np.float64).reshape(6)
            xd_cmd = compute_ik_seeded(
                self.cfg, fk_jacobian_fn, q_current, p_des, self._quat0, self._q_rest
            )
        elif self.cfg.split_base_wrist_task:
            jacobian = st.get("jacobian")
            xd_cmd = compute_split_base_wrist(self.cfg, jacobian, v_cmd, xd_full)
        elif self.cfg.reduced_task_dims:
            jacobian = st.get("jacobian")
            q = np.asarray(st["q"], dtype=np.float64).reshape(6)
            xd_cmd, self._q_rest, self._reanchored, self._settled_cycles = compute_reduced_task_dims(
                self.cfg,
                jacobian,
                q,
                xd_full,
                pos_err,
                v_ff,
                self._q_rest,
                self._reanchored,
                self._settled_cycles,
            )
        else:
            xd_cmd = compute_full_hold(xd_full)

        v_cmd_out = xd_cmd[:3]
        w_cmd_out = xd_cmd[3:]
        v_norm = float(np.linalg.norm(v_cmd_out))
        if v_norm > self.cfg.max_lin_speed_mps and v_norm > 1.0e-9:
            v_cmd_out = v_cmd_out * (self.cfg.max_lin_speed_mps / v_norm)
        w_norm = float(np.linalg.norm(w_cmd_out))
        if w_norm > self.cfg.max_ang_speed_radps and w_norm > 1.0e-9:
            w_cmd_out = w_cmd_out * (self.cfg.max_ang_speed_radps / w_norm)

        return np.concatenate([v_cmd_out, w_cmd_out]).astype(np.float64)
