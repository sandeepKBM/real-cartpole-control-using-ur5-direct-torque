"""
Archived reverse-nested x-servo controller.

This controller is not part of the active simulation path anymore. It is kept
here only so the previous design can be inspected or revived without cluttering
the main active controller module.
"""

from __future__ import annotations

import numpy as np

from controller import tool_target_orientation_omega


def reverse_nested_x_servo_controller(
    q: np.ndarray,
    q_target: np.ndarray,
    ctrl_prev: np.ndarray,
    ctrl_lower: np.ndarray,
    ctrl_upper: np.ndarray,
    qvel: np.ndarray | None = None,
    site_x: float | None = None,
    site_x_target: float | None = None,
    jacobian_x: np.ndarray | None = None,
    site_rot: np.ndarray | None = None,
    target_site_rot: np.ndarray | None = None,
    jacobian_rot: np.ndarray | None = None,
) -> np.ndarray:
    """
    Archived reverse-priority outer-loop controller for the reach joints.
    """
    q = np.asarray(q, dtype=np.float64)
    q_target = np.asarray(q_target, dtype=np.float64)
    ctrl_prev = np.asarray(ctrl_prev, dtype=np.float64)
    q_error = q_target - q

    if qvel is None:
        qvel = np.zeros_like(q)
    else:
        qvel = np.asarray(qvel, dtype=np.float64)

    posture_gains = np.array([0.010, 0.0180, 0.0260, 0.0320, 0.0120, 0.0120], dtype=np.float64)
    damping_gains = np.array([0.010, 0.0280, 0.0240, 0.0185, 0.0110, 0.0110], dtype=np.float64)
    delta = posture_gains * q_error - damping_gains * qvel

    if site_x is not None and site_x_target is not None and jacobian_x is not None:
        jacobian_x = np.asarray(jacobian_x, dtype=np.float64)
        x_error = float(site_x_target - site_x)
        x_guidance = 0.9 * x_error * jacobian_x
        x_mask = np.zeros_like(q)
        x_mask[1:4] = 1.0
        delta += x_mask * x_guidance

    if site_rot is not None and jacobian_rot is not None:
        jacobian_rot = np.asarray(jacobian_rot, dtype=np.float64)
        orient_omega = tool_target_orientation_omega(site_rot, target_site_rot=target_site_rot)
        orientation_guidance = 0.22 * jacobian_rot.T @ orient_omega
        orientation_mask = np.array([0.0, 0.16, 0.24, 0.64, 1.20, 1.00], dtype=np.float64)
        delta += orientation_mask * orientation_guidance

    wrist_err = abs(q_error[3])
    elbow_err = abs(q_error[2])

    elbow_gate = 0.72 + 0.28 * np.clip(1.0 - wrist_err / 1.20, 0.0, 1.0)
    shoulder_gate = 0.58 + 0.42 * np.clip(1.0 - max(wrist_err, elbow_err) / 1.20, 0.0, 1.0)

    priority_scale = np.ones_like(q)
    priority_scale[1] = 0.88 * shoulder_gate
    priority_scale[2] = 1.10 * elbow_gate
    priority_scale[3] = 1.35
    priority_scale[0] = 0.0
    priority_scale[4] = 0.45
    priority_scale[5] = 0.35

    delta *= priority_scale

    max_delta = np.array([0.0020, 0.0042, 0.0048, 0.0058, 0.0023, 0.0023], dtype=np.float64)
    delta = np.clip(delta, -max_delta, max_delta)

    ctrl = ctrl_prev + delta
    ctrl[0] = q_target[0]
    return np.clip(ctrl, ctrl_lower, ctrl_upper)
