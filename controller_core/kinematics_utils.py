"""
J^T adapter and small kinematic helpers used by the simulator-independent
controller stack.

All inputs are world-frame 3xN position and rotation Jacobians as produced
by MuJoCo's ``mj_jacSite`` or CoppeliaSim's ``sim.getJacobian`` (after
reshaping; see the CoppeliaSim bridge notes).
"""

from __future__ import annotations

import numpy as np

from .state_types import ControlOutput


def cartesian_force_to_joint_torque(
    fx_newtons: float,
    jacobian_pos: np.ndarray,
    *,
    qd: np.ndarray | None = None,
    kd_joint: np.ndarray | float = 0.0,
    gravity_torque: np.ndarray | None = None,
    jacobian_rot: np.ndarray | None = None,
    tau_max: np.ndarray | None = None,
) -> ControlOutput:
    """Convert a task-space X force into a 6-vector of joint torques.

    The core identity is

        F_task = [Fx, 0, 0, 0, 0, 0]
        tau    = J_full.T @ F_task

    When ``jacobian_rot`` is ``None`` we just use the 3-row position Jacobian:

        tau = jacobian_pos.T @ [Fx, 0, 0]

    Optional joint damping (``-Kd_joint * qd``) and gravity compensation
    (``+tau_gravity``) are added afterwards, matching the Stage 6 spec.

    ``tau_max`` (shape ``(6,)``) is the per-joint saturation limit applied
    before the torque is returned. Hard saturation sets ``saturated=True``
    on the output so the caller can log it.
    """
    jacobian_pos = np.asarray(jacobian_pos, dtype=np.float64)
    if jacobian_pos.ndim != 2 or jacobian_pos.shape[0] != 3:
        raise ValueError(f"jacobian_pos must be shape (3, n); got {jacobian_pos.shape}")
    num_joints = jacobian_pos.shape[1]

    f_task_pos = np.array([float(fx_newtons), 0.0, 0.0], dtype=np.float64)
    tau = jacobian_pos.T @ f_task_pos

    if jacobian_rot is not None:
        jacobian_rot = np.asarray(jacobian_rot, dtype=np.float64)
        if jacobian_rot.shape != (3, num_joints):
            raise ValueError(
                f"jacobian_rot must be shape (3, {num_joints}); got {jacobian_rot.shape}"
            )
        # Zero angular task today; kept for future 6D extensions.
        tau = tau + jacobian_rot.T @ np.zeros(3, dtype=np.float64)

    if qd is not None:
        qd_arr = np.asarray(qd, dtype=np.float64).reshape(-1)
        if qd_arr.shape[0] != num_joints:
            raise ValueError(f"qd must have length {num_joints}; got {qd_arr.shape}")
        kd = np.asarray(kd_joint, dtype=np.float64).reshape(-1)
        if kd.shape[0] == 1:
            kd = np.full(num_joints, float(kd[0]), dtype=np.float64)
        if kd.shape[0] != num_joints:
            raise ValueError(
                f"kd_joint must be scalar or length {num_joints}; got {kd.shape}"
            )
        tau = tau - kd * qd_arr

    if gravity_torque is not None:
        g = np.asarray(gravity_torque, dtype=np.float64).reshape(-1)
        if g.shape[0] != num_joints:
            raise ValueError(f"gravity_torque length must be {num_joints}; got {g.shape}")
        tau = tau + g

    saturated = False
    if tau_max is not None:
        tau_max_arr = np.asarray(tau_max, dtype=np.float64).reshape(-1)
        if tau_max_arr.shape[0] != num_joints:
            raise ValueError(
                f"tau_max length must be {num_joints}; got {tau_max_arr.shape}"
            )
        tau_clipped = np.clip(tau, -tau_max_arr, +tau_max_arr)
        saturated = bool(np.any(np.abs(tau - tau_clipped) > 1e-12))
        tau = tau_clipped

    return ControlOutput(mode="torque", tau=tau, saturated=saturated)


def quat_to_rotmat(quat_wxyz: np.ndarray) -> np.ndarray:
    """Convert a ``[w, x, y, z]`` quaternion to a 3x3 rotation matrix.

    Used by adapters so both MuJoCo (which returns ``site_xmat``) and
    CoppeliaSim (which typically returns quaternions) can produce the same
    ``ee_quat`` in the RobotState.
    """
    q = np.asarray(quat_wxyz, dtype=np.float64).reshape(-1)
    if q.shape[0] != 4:
        raise ValueError(f"quaternion must have length 4; got {q.shape}")
    w, x, y, z = q
    n = w * w + x * x + y * y + z * z
    if n < 1e-12:
        return np.eye(3, dtype=np.float64)
    s = 2.0 / n
    return np.array(
        [
            [1.0 - s * (y * y + z * z), s * (x * y - z * w), s * (x * z + y * w)],
            [s * (x * y + z * w), 1.0 - s * (x * x + z * z), s * (y * z - x * w)],
            [s * (x * z - y * w), s * (y * z + x * w), 1.0 - s * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def rotmat_to_quat(rot: np.ndarray) -> np.ndarray:
    """Convert a 3x3 rotation matrix to a ``[w, x, y, z]`` quaternion."""
    r = np.asarray(rot, dtype=np.float64).reshape(3, 3)
    trace = r[0, 0] + r[1, 1] + r[2, 2]
    if trace > 0.0:
        s = 2.0 * np.sqrt(trace + 1.0)
        w = 0.25 * s
        x = (r[2, 1] - r[1, 2]) / s
        y = (r[0, 2] - r[2, 0]) / s
        z = (r[1, 0] - r[0, 1]) / s
    elif (r[0, 0] > r[1, 1]) and (r[0, 0] > r[2, 2]):
        s = 2.0 * np.sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2])
        w = (r[2, 1] - r[1, 2]) / s
        x = 0.25 * s
        y = (r[0, 1] + r[1, 0]) / s
        z = (r[0, 2] + r[2, 0]) / s
    elif r[1, 1] > r[2, 2]:
        s = 2.0 * np.sqrt(1.0 + r[1, 1] - r[0, 0] - r[2, 2])
        w = (r[0, 2] - r[2, 0]) / s
        x = (r[0, 1] + r[1, 0]) / s
        y = 0.25 * s
        z = (r[1, 2] + r[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + r[2, 2] - r[0, 0] - r[1, 1])
        w = (r[1, 0] - r[0, 1]) / s
        x = (r[0, 2] + r[2, 0]) / s
        y = (r[1, 2] + r[2, 1]) / s
        z = 0.25 * s
    return np.array([w, x, y, z], dtype=np.float64)


def quat_normalize_wxyz(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64).reshape(4)
    n = np.linalg.norm(q)
    if n < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    return q / n


def quat_conj_wxyz(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64).reshape(4)
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=np.float64)


def quat_multiply_wxyz(q_left: np.ndarray, q_right: np.ndarray) -> np.ndarray:
    """Hamilton product with ``[w, x, y, z]`` storage."""
    w1, x1, y1, z1 = np.asarray(q_left, dtype=np.float64).reshape(4)
    w2, x2, y2, z2 = np.asarray(q_right, dtype=np.float64).reshape(4)
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=np.float64,
    )


def orientation_error_vec_wxyz(quat_des: np.ndarray, quat_cur: np.ndarray) -> np.ndarray:
    """3-vector orientation error for PD (world-frame convention).

    Uses ``q_err = conj(q_des) * q_cur`` (rotation from desired to current in
    the usual multiplicative sense). For small errors ``e ≈ 2 * vec(q_err)``.
    """
    qd = quat_normalize_wxyz(quat_des)
    qc = quat_normalize_wxyz(quat_cur)
    q_err = quat_multiply_wxyz(quat_conj_wxyz(qd), qc)
    q_err = quat_normalize_wxyz(q_err)
    if q_err[0] < 0.0:
        q_err = -q_err
    return 2.0 * q_err[1:4]
