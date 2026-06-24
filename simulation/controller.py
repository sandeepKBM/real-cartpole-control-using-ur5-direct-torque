"""
Modular controller for UR5e arm.
Replace this module to implement different control strategies.
"""

import numpy as np
from typing import Any

# UR5e joint indices:
# 0=shoulder_pan, 1=shoulder_lift, 2=elbow, 3=wrist_1, 4=wrist_2, 5=wrist_3
FOREARM_ORIGIN_INDICES = np.array([1, 2], dtype=np.int64)
TOOL_FACE_INDICES = np.array([3, 4, 5], dtype=np.int64)

# Legacy peak-height origin measured from the MuJoCo model.
LEGACY_PEAK_HEIGHT_Q = np.array([
    0.0,
    -1.570784,
    -0.000007,
    -2.356201,
    -1.570784,
    0.0,
], dtype=np.float64)

# Active shoulder-side-face-direction origin:
# - shoulder_pan stays fixed at zero
# - forearm_tip_site defines the origin reference position
# - attachment_site defines the rotated tool-frame reference
# - the tool normal is aligned to the blue side panel on the shoulder/upper-arm assembly
# - that panel is the urblue `upperarm_3` mesh, whose visible face is along upper_arm_link +y
# - with shoulder_pan = 0 in the canonical straight-arm pose, upper_arm_link +y is world -Y
# - the canonical branch that matches this face direction is q=[0, -pi/2, 0, -pi/2, 0, 0]
SHOULDER_SIDE_FACE_DIRECTION_ORIGIN_Q = np.array([
    0.0,
    -1.5707963267948966,
    0.0,
    -1.5707963267948966,
    0.0,
    0.0,
], dtype=np.float64)

ACTIVE_ORIGIN_Q = SHOULDER_SIDE_FACE_DIRECTION_ORIGIN_Q.copy()

# Locked tool target for the current task:
# - attachment_site z is the pointing/face-normal direction and matches the blue side panel
# - attachment_site z points along world -Y
# - the chosen roll branch is the straight-arm canonical pose:
#   site x along world -X and site y along world -Z
TARGET_SITE_X_AXIS_WORLD = np.array([-1.0, 0.0, 0.0], dtype=np.float64)
TARGET_SITE_Y_AXIS_WORLD = np.array([0.0, 0.0, -1.0], dtype=np.float64)
TARGET_SITE_Z_AXIS_WORLD = np.array([0.0, -1.0, 0.0], dtype=np.float64)
TARGET_SITE_ROTATION_WORLD = np.column_stack(
    [TARGET_SITE_X_AXIS_WORLD, TARGET_SITE_Y_AXIS_WORLD, TARGET_SITE_Z_AXIS_WORLD]
)

AXIS_NAME_TO_INDEX = {"x": 0, "y": 1, "z": 2}
AXIS_INDEX_TO_NAME = ("x", "y", "z")


def axis_name_to_index(axis: str | int) -> int:
    """Normalize an axis selector to 0=x, 1=y, 2=z."""
    if isinstance(axis, str):
        key = axis.strip().lower()
        if key in AXIS_NAME_TO_INDEX:
            return AXIS_NAME_TO_INDEX[key]
        raise ValueError(f"Unknown transport axis: {axis!r}")
    idx = int(axis)
    if idx not in (0, 1, 2):
        raise ValueError(f"Transport axis index out of range: {axis!r}")
    return idx


def axis_index_to_name(axis: str | int) -> str:
    return AXIS_INDEX_TO_NAME[axis_name_to_index(axis)]


def orthogonal_axis_indices(axis: str | int) -> tuple[int, int]:
    idx = axis_name_to_index(axis)
    return tuple(i for i in range(3) if i != idx)  # type: ignore[return-value]


def tool_target_orientation_omega(
    site_rot: np.ndarray,
    target_site_rot: np.ndarray | None = None,
) -> np.ndarray:
    """
    Small-angle angular-velocity correction that aligns the tool to a fixed
    world-frame orientation target.

    `attachment_site` is the task frame. The controller does not switch to using
    wrist_2_joint as the reference frame, because the user-facing goal is a
    fixed world-frame tool orientation, not a specific intermediate joint angle.
    """
    current = np.asarray(site_rot, dtype=np.float64).reshape(3, 3)
    target = (
        TARGET_SITE_ROTATION_WORLD
        if target_site_rot is None
        else np.asarray(target_site_rot, dtype=np.float64).reshape(3, 3)
    )

    current_x = current[:, 0]
    current_y = current[:, 1]
    current_z = current[:, 2]
    target_x = target[:, 0]
    target_y = target[:, 1]
    target_z = target[:, 2]

    # Keep the visible tool normal aligned with the blue shoulder/upper-arm side panel:
    # - site z follows world -Y
    # - site x follows world -X
    # - site y follows world -Z
    return (
        1.20 * np.cross(current_z, target_z)
        + 1.10 * np.cross(current_y, target_y)
        + 0.40 * np.cross(current_x, target_x)
    )


def sample_random_configuration(
    rng: np.random.Generator,
    lower: np.ndarray,
    upper: np.ndarray,
    center: np.ndarray | None = None,
    span_scale: float = 0.35,
    min_distance: float = 0.8,
    max_tries: int = 256,
) -> np.ndarray:
    """
    Sample a nontrivial random joint configuration near a reference pose.

    The project is control-focused, so this deliberately avoids adversarial
    extreme poses while still producing a reproducible random start.
    """
    if center is None:
        center = ACTIVE_ORIGIN_Q

    center = np.asarray(center, dtype=np.float64)
    lower = np.asarray(lower, dtype=np.float64)
    upper = np.asarray(upper, dtype=np.float64)

    half_span = 0.5 * span_scale * (upper - lower)
    sample_lower = np.maximum(lower, center - half_span)
    sample_upper = np.minimum(upper, center + half_span)

    best = center.copy()
    best_distance = 0.0
    for _ in range(max_tries):
        candidate = rng.uniform(sample_lower, sample_upper)
        distance = float(np.linalg.norm(candidate - center))
        if distance >= min_distance:
            return candidate
        if distance > best_distance:
            best = candidate
            best_distance = distance

    return best


def _solve_weighted_delta(
    a_blocks: list[np.ndarray],
    b_blocks: list[np.ndarray],
    num_dofs: int,
    regularization: float = 1e-4,
) -> np.ndarray:
    """
    Solve a small stacked weighted least-squares problem for joint increments.

    This is the core linear solve used by the differential-IK controller.
    """
    if not a_blocks or not b_blocks:
        return np.zeros(num_dofs, dtype=np.float64)

    a = np.vstack(a_blocks)
    b = np.concatenate(b_blocks)

    if regularization > 0.0:
        a = np.vstack([a, np.sqrt(regularization) * np.eye(num_dofs, dtype=np.float64)])
        b = np.concatenate([b, np.zeros(num_dofs, dtype=np.float64)])

    delta, *_ = np.linalg.lstsq(a, b, rcond=None)
    return np.asarray(delta, dtype=np.float64)


def _joint_limit_repulsion_step(
    q: np.ndarray,
    ctrl_lower: np.ndarray,
    ctrl_upper: np.ndarray,
    guard_margin_rad: float = 0.02,
    gain: float = 0.0025,
) -> np.ndarray:
    """
    Softly bias the IK solve away from the joint limits.

    This is a gentle centering term, not a hard constraint. The hard safety
    clamp still happens after the solve, but this extra term makes the solver
    prefer configurations that keep more margin in reserve.
    """
    q = np.asarray(q, dtype=np.float64).reshape(6)
    ctrl_lower = np.asarray(ctrl_lower, dtype=np.float64).reshape(6)
    ctrl_upper = np.asarray(ctrl_upper, dtype=np.float64).reshape(6)

    lower = ctrl_lower + max(float(guard_margin_rad), 0.0)
    upper = ctrl_upper - max(float(guard_margin_rad), 0.0)
    if np.any(lower >= upper):
        lower = ctrl_lower.copy()
        upper = ctrl_upper.copy()

    center = 0.5 * (lower + upper)
    half_span = 0.5 * np.maximum(upper - lower, 1e-9)
    normalized = np.clip((q - center) / half_span, -0.999, 0.999)

    # The closer a joint gets to one edge, the stronger the push back toward the
    # interior. Keep the magnitude small so task tracking still dominates.
    distance_to_edge = np.maximum(1.0 - np.abs(normalized), 0.08)
    repulsion = -float(gain) * normalized / distance_to_edge
    return np.clip(repulsion, -0.01, 0.01)


# Legacy reverse_nested_x_servo_controller was moved to
# simulation/legacy_reverse_nested_controller.py so this module only contains
# the actively used control path.


def differential_ik_split_controller(
    q: np.ndarray,
    q_target: np.ndarray,
    ctrl_prev: np.ndarray,
    ctrl_lower: np.ndarray,
    ctrl_upper: np.ndarray,
    qvel: np.ndarray | None = None,
    origin_pos: np.ndarray | None = None,
    origin_target_pos: np.ndarray | None = None,
    origin_jacobian_pos: np.ndarray | None = None,
    tool_rot: np.ndarray | None = None,
    target_tool_rot: np.ndarray | None = None,
    tool_jacobian_rot: np.ndarray | None = None,
    wrist_posture_target: np.ndarray | None = None,
) -> np.ndarray:
    """
    Differential-IK reinterpretation of the active split controller.

    Task design:

    - solve for small joint increments on joints 1..5 with shoulder_pan fixed
    - forearm-tip position is the proximal task
    - tool orientation is the distal task
    - posture and wrist bias remain soft secondary objectives

    This keeps the current experiment structure but replaces the hand-tuned
    Jacobian-transpose blend with a small weighted least-squares IK solve.
    """
    q = np.asarray(q, dtype=np.float64)
    q_target = np.asarray(q_target, dtype=np.float64)
    ctrl_prev = np.asarray(ctrl_prev, dtype=np.float64)
    ctrl_lower = np.asarray(ctrl_lower, dtype=np.float64)
    ctrl_upper = np.asarray(ctrl_upper, dtype=np.float64)

    if qvel is None:
        qvel = np.zeros_like(q)
    else:
        qvel = np.asarray(qvel, dtype=np.float64)

    red = slice(1, 6)
    q_error_red = q_target[red] - q[red]
    qvel_red = qvel[red]
    num_red = 5

    origin_mask = np.array([1.35, 1.55, 0.0, 0.0, 0.0], dtype=np.float64)
    face_mask = np.array([0.0, 0.0, 1.25, 1.55, 1.25], dtype=np.float64)

    posture_gains = np.array([0.022, 0.024, 0.0, 0.0, 0.0], dtype=np.float64)
    damping_gains = np.array([0.032, 0.028, 0.020, 0.014, 0.012], dtype=np.float64)
    desired_posture_step = posture_gains * q_error_red - damping_gains * qvel_red

    if wrist_posture_target is not None:
        wrist_posture_target = np.asarray(wrist_posture_target, dtype=np.float64)
        wrist_error_red = wrist_posture_target[red] - q[red]
        wrist_posture_gains = np.array([0.0, 0.0, 0.010, 0.024, 0.010], dtype=np.float64)
        desired_posture_step += wrist_posture_gains * wrist_error_red

    desired_posture_step += _joint_limit_repulsion_step(q, ctrl_lower, ctrl_upper)[red]

    a_blocks: list[np.ndarray] = []
    b_blocks: list[np.ndarray] = []

    if origin_pos is not None and origin_target_pos is not None and origin_jacobian_pos is not None:
        origin_pos = np.asarray(origin_pos, dtype=np.float64)
        origin_target_pos = np.asarray(origin_target_pos, dtype=np.float64)
        origin_jacobian_pos = np.asarray(origin_jacobian_pos, dtype=np.float64)
        origin_error = origin_target_pos - origin_pos
        j_origin = origin_jacobian_pos[:, red] * origin_mask
        a_blocks.append(np.sqrt(4.0) * j_origin)
        b_blocks.append(np.sqrt(4.0) * (2.2 * origin_error))

    if tool_rot is not None and tool_jacobian_rot is not None:
        tool_jacobian_rot = np.asarray(tool_jacobian_rot, dtype=np.float64)
        orient_omega = tool_target_orientation_omega(tool_rot, target_site_rot=target_tool_rot)
        j_rot = tool_jacobian_rot[:, red] * face_mask
        a_blocks.append(np.sqrt(3.0) * j_rot)
        b_blocks.append(np.sqrt(3.0) * (0.50 * orient_omega))

    posture_weights = np.array([0.35, 0.35, 0.10, 0.16, 0.10], dtype=np.float64)
    a_blocks.append(np.diag(np.sqrt(posture_weights)))
    b_blocks.append(np.sqrt(posture_weights) * desired_posture_step)

    delta_red = _solve_weighted_delta(
        a_blocks=a_blocks,
        b_blocks=b_blocks,
        num_dofs=num_red,
        regularization=1e-4,
    )

    max_delta_red = np.array([0.0048, 0.0052, 0.0058, 0.0030, 0.0026], dtype=np.float64)
    delta_red = np.clip(delta_red, -max_delta_red, max_delta_red)

    ctrl = ctrl_prev.copy()
    ctrl[red] = _apply_joint_limit_guardrails(
        ctrl_prev[red],
        delta_red,
        ctrl_lower[red],
        ctrl_upper[red],
    )
    ctrl[0] = float(np.clip(q_target[0], ctrl_lower[0], ctrl_upper[0]))
    return _apply_joint_limit_guardrails(ctrl_prev, ctrl - ctrl_prev, ctrl_lower, ctrl_upper)


def differential_ik_xz_transport_controller(
    q: np.ndarray,
    ctrl_prev: np.ndarray,
    ctrl_lower: np.ndarray,
    ctrl_upper: np.ndarray,
    qvel: np.ndarray | None = None,
    tool_pos: np.ndarray | None = None,
    x_target: float | None = None,
    z_target: float | None = None,
    tool_jacobian_pos: np.ndarray | None = None,
    tool_rot: np.ndarray | None = None,
    target_tool_rot: np.ndarray | None = None,
    tool_jacobian_rot: np.ndarray | None = None,
    pan_target: float = 0.0,
    posture_target: np.ndarray | None = None,
) -> np.ndarray:
    """
    Differential-IK transport controller for the actual task of interest:

    - `attachment_site` world X tracks a moving reference
    - `attachment_site` world Z stays at a fixed target
    - `attachment_site` orientation stays locked in the world frame
    - `shoulder_pan_joint` stays fixed

    This is the square 5-constraint / 5-joint controller that matches the
    current intended hardware task more closely than the origin-hold controller.
    """
    q = np.asarray(q, dtype=np.float64)
    ctrl_prev = np.asarray(ctrl_prev, dtype=np.float64)
    ctrl_lower = np.asarray(ctrl_lower, dtype=np.float64)
    ctrl_upper = np.asarray(ctrl_upper, dtype=np.float64)

    if qvel is None:
        qvel = np.zeros_like(q)
    else:
        qvel = np.asarray(qvel, dtype=np.float64)

    red = slice(1, 6)
    qvel_red = qvel[red]
    num_red = 5

    if posture_target is None:
        posture_target = ctrl_prev
    posture_target = np.asarray(posture_target, dtype=np.float64)
    posture_error_red = posture_target[red] - q[red]

    damping_gains = np.array([0.040, 0.036, 0.024, 0.018, 0.016], dtype=np.float64)
    posture_gains = np.array([0.010, 0.010, 0.004, 0.006, 0.004], dtype=np.float64)
    desired_posture_step = posture_gains * posture_error_red - damping_gains * qvel_red
    desired_posture_step += _joint_limit_repulsion_step(q, ctrl_lower, ctrl_upper)[red]

    a_blocks: list[np.ndarray] = []
    b_blocks: list[np.ndarray] = []

    if tool_pos is not None and x_target is not None and z_target is not None and tool_jacobian_pos is not None:
        tool_pos = np.asarray(tool_pos, dtype=np.float64)
        tool_jacobian_pos = np.asarray(tool_jacobian_pos, dtype=np.float64)
        x_error = float(x_target - tool_pos[0])
        z_error = float(z_target - tool_pos[2])
        j_x = tool_jacobian_pos[0:1, red]
        j_z = tool_jacobian_pos[2:3, red]
        a_blocks.append(np.sqrt(5.0) * j_x)
        b_blocks.append(np.sqrt(5.0) * np.array([1.8 * x_error], dtype=np.float64))
        a_blocks.append(np.sqrt(4.0) * j_z)
        b_blocks.append(np.sqrt(4.0) * np.array([2.0 * z_error], dtype=np.float64))

    if tool_rot is not None and tool_jacobian_rot is not None:
        tool_jacobian_rot = np.asarray(tool_jacobian_rot, dtype=np.float64)
        orient_omega = tool_target_orientation_omega(tool_rot, target_site_rot=target_tool_rot)
        a_blocks.append(np.sqrt(4.0) * tool_jacobian_rot[:, red])
        b_blocks.append(np.sqrt(4.0) * (0.55 * orient_omega))

    posture_weights = np.array([0.28, 0.28, 0.10, 0.12, 0.10], dtype=np.float64)
    a_blocks.append(np.diag(np.sqrt(posture_weights)))
    b_blocks.append(np.sqrt(posture_weights) * desired_posture_step)

    delta_red = _solve_weighted_delta(
        a_blocks=a_blocks,
        b_blocks=b_blocks,
        num_dofs=num_red,
        regularization=1e-4,
    )

    max_delta_red = np.array([0.0052, 0.0054, 0.0060, 0.0032, 0.0030], dtype=np.float64)
    delta_red = np.clip(delta_red, -max_delta_red, max_delta_red)

    ctrl = ctrl_prev.copy()
    ctrl[red] = _apply_joint_limit_guardrails(
        ctrl_prev[red],
        delta_red,
        ctrl_lower[red],
        ctrl_upper[red],
    )
    ctrl[0] = float(np.clip(pan_target, ctrl_lower[0], ctrl_upper[0]))
    return _apply_joint_limit_guardrails(ctrl_prev, ctrl - ctrl_prev, ctrl_lower, ctrl_upper)


def split_forearm_origin_face_controller(
    q: np.ndarray,
    q_target: np.ndarray,
    ctrl_prev: np.ndarray,
    ctrl_lower: np.ndarray,
    ctrl_upper: np.ndarray,
    qvel: np.ndarray | None = None,
    origin_pos: np.ndarray | None = None,
    origin_target_pos: np.ndarray | None = None,
    origin_jacobian_pos: np.ndarray | None = None,
    tool_rot: np.ndarray | None = None,
    target_tool_rot: np.ndarray | None = None,
    tool_jacobian_rot: np.ndarray | None = None,
    wrist_posture_target: np.ndarray | None = None,
) -> np.ndarray:
    """
    Split controller:

    - `forearm_tip_site` position defines the origin task
    - `attachment_site` orientation defines the vertical-face task

    This keeps wrist motion from redefining the origin reference while still
    letting the distal chain hold the tool face perpendicular to the ground.
    """
    q = np.asarray(q, dtype=np.float64)
    q_target = np.asarray(q_target, dtype=np.float64)
    ctrl_prev = np.asarray(ctrl_prev, dtype=np.float64)
    q_error = q_target - q

    if qvel is None:
        qvel = np.zeros_like(q)
    else:
        qvel = np.asarray(qvel, dtype=np.float64)

    # The forearm origin is a proximal-arm task. Do not pull the wrist chain
    # back to a specific joint-space pose, because the wrist should remain free
    # to satisfy the fixed world-frame tool orientation target even when the
    # forearm origin definition changes.
    posture_gains = np.array([0.0, 0.022, 0.024, 0.0, 0.0, 0.0], dtype=np.float64)
    damping_gains = np.array([0.0, 0.032, 0.028, 0.020, 0.014, 0.012], dtype=np.float64)
    delta = posture_gains * q_error - damping_gains * qvel

    if origin_pos is not None and origin_target_pos is not None and origin_jacobian_pos is not None:
        origin_pos = np.asarray(origin_pos, dtype=np.float64)
        origin_target_pos = np.asarray(origin_target_pos, dtype=np.float64)
        origin_jacobian_pos = np.asarray(origin_jacobian_pos, dtype=np.float64)
        origin_error = origin_target_pos - origin_pos
        origin_guidance = 2.2 * origin_jacobian_pos.T @ origin_error
        origin_mask = np.array([0.0, 1.35, 1.55, 0.0, 0.0, 0.0], dtype=np.float64)
        delta += origin_mask * origin_guidance

    if tool_rot is not None and tool_jacobian_rot is not None:
        tool_jacobian_rot = np.asarray(tool_jacobian_rot, dtype=np.float64)
        orient_omega = tool_target_orientation_omega(tool_rot, target_site_rot=target_tool_rot)
        face_guidance = 0.50 * tool_jacobian_rot.T @ orient_omega
        face_mask = np.array([0.0, 0.0, 0.0, 1.25, 1.55, 1.25], dtype=np.float64)
        delta += face_mask * face_guidance

    # Optional non-standard wrist posture bias. This lets a run favor a
    # different wrist_2_link pose without redefining the origin site or the
    # world-frame tool orientation target.
    if wrist_posture_target is not None:
        wrist_posture_target = np.asarray(wrist_posture_target, dtype=np.float64)
        wrist_error = wrist_posture_target - q
        wrist_posture_gains = np.array([0.0, 0.0, 0.0, 0.010, 0.024, 0.010], dtype=np.float64)
        delta += wrist_posture_gains * wrist_error

    delta += _joint_limit_repulsion_step(q, ctrl_lower, ctrl_upper)

    max_delta = np.array([0.0, 0.0048, 0.0052, 0.0058, 0.0030, 0.0026], dtype=np.float64)
    delta = np.clip(delta, -max_delta, max_delta)

    ctrl = _apply_joint_limit_guardrails(ctrl_prev, delta, ctrl_lower, ctrl_upper)
    ctrl[0] = float(np.clip(q_target[0], ctrl_lower[0], ctrl_upper[0]))
    return _apply_joint_limit_guardrails(ctrl_prev, ctrl - ctrl_prev, ctrl_lower, ctrl_upper)


# Position-servo parameters extracted from
# `mujoco_menagerie/universal_robots_ur5e/ur5e.xml`:
#   size3 joints (pan, lift, elbow): Kp=2000, Kd=400, |tau| <= 150 N*m
#   size1 joints (wrist_1..3):       Kp=500,  Kd=100, |tau| <= 28  N*m
# These let the controller estimate the torque the MuJoCo position servo would
# command for a proposed `ctrl - q` offset and throttle the Cartesian velocity
# command so no joint is pushed into saturation.
SERVO_KP = np.array([2000.0, 2000.0, 2000.0, 500.0, 500.0, 500.0], dtype=np.float64)
SERVO_KD = np.array([400.0, 400.0, 400.0, 100.0, 100.0, 100.0], dtype=np.float64)
SERVO_FORCE_LIMIT = np.array([150.0, 150.0, 150.0, 28.0, 28.0, 28.0], dtype=np.float64)

# Per-joint setpoint-delta caps reused from the existing xz transport
# controller. With a typical scene timestep of ~2 ms these correspond to a
# joint-speed envelope of roughly 1.5-3 rad/s, well above the speeds needed
# for a 0.05 m/s tool translation in the nominal pose.
VELOCITY_MAX_DELTA_RED = np.array(
    [0.0052, 0.0054, 0.0060, 0.0032, 0.0030], dtype=np.float64
)


def velocity_x_transport_controller(
    q: np.ndarray,
    qvel: np.ndarray,
    ctrl_prev: np.ndarray,
    ctrl_lower: np.ndarray,
    ctrl_upper: np.ndarray,
    tool_pos: np.ndarray,
    tool_rot: np.ndarray,
    tool_jacobian_pos: np.ndarray,
    tool_jacobian_rot: np.ndarray,
    v_x_cmd: float,
    x_goal: float,
    z_hold: float,
    hold_y_target: float | None = None,
    target_tool_rot: np.ndarray | None = None,
    pan_target: float = 0.0,
    dt: float = 0.002,
    stop_tol_m: float = 2.0e-3,
    torque_headroom: float = 0.9,
    a_decel_m_s2: float = 0.35,
    k_x_hold_s_inv: float = 8.0,
    # Squared weights in the normal equations Jᵀ W J (same convention as
    # `wx**2`, etc.). X must be on the same order as Z / Ω or the arm barely
    # moves along world X.
    task_weight_x: float = 100.0,
    task_weight_y: float = 64.0,
    task_weight_z: float = 64.0,
    task_weight_omega: float = 36.0,
    hold_y_gain: float = 8.0,
) -> tuple[np.ndarray, dict]:
    """
    Velocity-commanded `attachment_site` transport along world X.

    Task (5 constraints on joints 1..5, shoulder_pan locked):

    - d/dt(tool_pos[0]) = v_x_cmd_effective   (signed, m/s)
    - d/dt(tool_pos[2]) = Kz * (z_hold - tool_pos[2])
    - d/dt(tool_orientation) = K_omega * tool_target_orientation_omega

    The x row uses weight `task_weight_x`; y/z/orientation rows can be added
    to keep the transport plane fixed while height and tool facing are not
    sacrificed for x tracking.

    **Braking:** Cruise speed applies when farther than braking distance
    ``s_brake = v_cmd^2 / (2 * a_decel)``. Inside that distance, commanded
    speed scales **linearly** with remaining distance to the goal so the
    commanded x velocity reaches **zero exactly at** ``x_goal`` (no hard
    switch that leaves residual motion and overshoot).

    Feasibility is enforced against the MuJoCo position servos defined in
    `ur5e.xml`:

    1. Joint-speed ceiling: `|q_dot_des[i]| <= VELOCITY_MAX_DELTA_RED[i] / dt`.
    2. Torque-headroom guard: the servo torque implied by
       `Kp*(ctrl - q) - Kd*qvel` must stay within `torque_headroom *
       SERVO_FORCE_LIMIT`.

    If either guard clips a joint, the Cartesian v_x command is reduced by the
    same ratio for that step. This keeps Cartesian x/z/orientation tracking
    self-consistent rather than silently drifting when one joint saturates.

    `goal_reached` is True when `|x_goal - x|` <= `stop_tol_m` (for settle
    timers). Inside that band, x velocity is `k_x_hold_s_inv * x_remaining`
    so the tool eases onto the goal instead of coasting past it.
    """
    q = np.asarray(q, dtype=np.float64)
    qvel = np.asarray(qvel, dtype=np.float64)
    ctrl_prev = np.asarray(ctrl_prev, dtype=np.float64)
    ctrl_lower = np.asarray(ctrl_lower, dtype=np.float64)
    ctrl_upper = np.asarray(ctrl_upper, dtype=np.float64)
    tool_pos = np.asarray(tool_pos, dtype=np.float64)
    tool_rot = np.asarray(tool_rot, dtype=np.float64).reshape(3, 3)
    tool_jacobian_pos = np.asarray(tool_jacobian_pos, dtype=np.float64)
    tool_jacobian_rot = np.asarray(tool_jacobian_rot, dtype=np.float64)

    red = slice(1, 6)
    num_red = 5

    # Remaining x distance, signed toward +X if the goal is to the right.
    x_remaining = float(x_goal - tool_pos[0])
    d = abs(x_remaining)
    goal_reached = d <= stop_tol_m
    sign_goal = float(np.sign(x_remaining)) if d > 1e-9 else 0.0

    v_cmd_mag = abs(float(v_x_cmd))
    s_brake = v_cmd_mag**2 / (2.0 * max(float(a_decel_m_s2), 1e-6))

    if d <= stop_tol_m:
        # Final hold: proportional x correction (m/s per m) so we settle on the
        # goal without a hard velocity discontinuity from the cruise law.
        v_x_effective = float(k_x_hold_s_inv * x_remaining)
    else:
        if d >= s_brake:
            v_mag = v_cmd_mag
        else:
            # Linear in distance: v=0 at the goal, full cruise at the braking radius.
            v_mag = v_cmd_mag * (d / max(s_brake, 1e-9))
        v_x_effective = float(sign_goal * v_mag)

    # Never cross the goal in one integration step (discrete-time safety).
    max_step_v = d / max(dt, 1e-6)
    if abs(v_x_effective) > max_step_v:
        v_x_effective = float(np.sign(v_x_effective) * max_step_v)

    # Secondary Cartesian constraints: z hold and orientation hold.
    if hold_y_target is not None:
        v_y_des = float(hold_y_gain * (float(hold_y_target) - float(tool_pos[1])))
    else:
        v_y_des = None
    Kz = 8.0
    K_omega = 2.5
    v_z_des = Kz * (float(z_hold) - float(tool_pos[2]))
    if target_tool_rot is None:
        target_tool_rot = TARGET_SITE_ROTATION_WORLD
    omega_des = K_omega * tool_target_orientation_omega(
        tool_rot, target_site_rot=target_tool_rot
    )

    # Weighted least squares: prioritize z and orientation over x.
    wx = float(np.sqrt(max(task_weight_x, 0.0)))
    wy = float(np.sqrt(max(task_weight_y, 0.0)))
    wz = float(np.sqrt(max(task_weight_z, 0.0)))
    wo = float(np.sqrt(max(task_weight_omega, 0.0)))

    jx = tool_jacobian_pos[0:1, red]
    jy = tool_jacobian_pos[1:2, red]
    jz = tool_jacobian_pos[2:3, red]
    jw = tool_jacobian_rot[:, red]

    a_blocks = [
        wx * jx,
    ]
    b_blocks = [
        wx * np.array([v_x_effective], dtype=np.float64),
    ]
    if v_y_des is not None:
        a_blocks.append(wy * jy)
        b_blocks.append(wy * np.array([v_y_des], dtype=np.float64))
    a_blocks.extend([
        wz * jz,
        wo * jw,
    ])
    b_blocks.extend([
        wz * np.array([v_z_des], dtype=np.float64),
        wo * omega_des,
    ])
    q_dot_des_red = _solve_weighted_delta(
        a_blocks=a_blocks,
        b_blocks=b_blocks,
        num_dofs=num_red,
        regularization=1e-3,
    )

    # Feasibility guard 1: per-joint speed ceiling derived from servo step cap.
    q_dot_ceiling = VELOCITY_MAX_DELTA_RED / max(dt, 1e-6)
    abs_ratio = np.abs(q_dot_des_red) / np.maximum(q_dot_ceiling, 1e-9)
    speed_ratio = float(np.max(abs_ratio))
    scale_speed = 1.0 if speed_ratio <= 1.0 else 1.0 / speed_ratio
    q_dot_des_red = q_dot_des_red * scale_speed

    # Proposed new control setpoint for joints 1..5.
    ctrl_red_new = ctrl_prev[red] + q_dot_des_red * dt
    ctrl_red_new = np.clip(ctrl_red_new, ctrl_lower[red], ctrl_upper[red])

    # Feasibility guard 2: estimated servo torque stays within a headroom
    # fraction of the per-joint forcerange. This is the PD model MuJoCo's
    # `general` position servos implement for this robot.
    kp_red = SERVO_KP[red]
    kd_red = SERVO_KD[red]
    limit_red = SERVO_FORCE_LIMIT[red] * float(torque_headroom)
    tau_est_red = kp_red * (ctrl_red_new - q[red]) - kd_red * qvel[red]
    abs_tau_ratio = np.abs(tau_est_red) / np.maximum(limit_red, 1e-9)
    torque_ratio = float(np.max(abs_tau_ratio))
    scale_torque = 1.0 if torque_ratio <= 1.0 else 1.0 / torque_ratio
    if scale_torque < 1.0:
        q_dot_des_red = q_dot_des_red * scale_torque
        ctrl_red_new = ctrl_prev[red] + q_dot_des_red * dt
        ctrl_red_new = np.clip(ctrl_red_new, ctrl_lower[red], ctrl_upper[red])
        tau_est_red = kp_red * (ctrl_red_new - q[red]) - kd_red * qvel[red]

    # Report effective Cartesian command after all feasibility scaling.
    total_scale = float(scale_speed * scale_torque)
    v_x_realized_cmd = float(v_x_effective * total_scale)

    ctrl = ctrl_prev.copy()
    ctrl[red] = ctrl_red_new
    ctrl[0] = float(pan_target)
    ctrl = np.clip(ctrl, ctrl_lower, ctrl_upper)

    # Full 6-DOF torque estimate for logging (pan is just its own PD).
    tau_est_full = np.zeros(6, dtype=np.float64)
    tau_est_full[0] = SERVO_KP[0] * (ctrl[0] - q[0]) - SERVO_KD[0] * qvel[0]
    tau_est_full[red] = tau_est_red

    diagnostics = {
        "v_x_cmd": float(v_x_cmd),
        "v_x_effective": float(v_x_effective),
        "v_x_realized_cmd": v_x_realized_cmd,
        "x_remaining_m": float(x_remaining),
        "goal_reached": bool(goal_reached),
        "braking_distance_m": float(s_brake),
        "hold_band_active": bool(d <= stop_tol_m),
        "speed_scale": float(scale_speed),
        "torque_scale": float(scale_torque),
        "speed_ratio": speed_ratio,
        "torque_ratio": torque_ratio,
        "q_dot_des_red": q_dot_des_red.tolist(),
        "tau_estimate_nm": tau_est_full.tolist(),
    }
    return ctrl, diagnostics


ACCEL_AXIS_DEFAULT_A_MAX_M_S2 = 0.15
ACCEL_AXIS_DEFAULT_V_MAX_M_S = 0.08
VELOCITY_MAX_DELTA_FULL = np.array(
    [0.0052, 0.0052, 0.0054, 0.0060, 0.0032, 0.0030], dtype=np.float64
)
JOINT_LIMIT_GUARD_MARGIN_RAD = 0.02


def _apply_joint_limit_guardrails(
    ctrl_prev: np.ndarray,
    delta: np.ndarray,
    ctrl_lower: np.ndarray,
    ctrl_upper: np.ndarray,
    guard_margin_rad: float = JOINT_LIMIT_GUARD_MARGIN_RAD,
) -> np.ndarray:
    """Keep IK setpoints inside the joint box with a small safety margin."""
    ctrl_prev = np.asarray(ctrl_prev, dtype=np.float64).reshape(-1)
    delta = np.asarray(delta, dtype=np.float64).reshape(-1)
    ctrl_lower = np.asarray(ctrl_lower, dtype=np.float64).reshape(-1)
    ctrl_upper = np.asarray(ctrl_upper, dtype=np.float64).reshape(-1)
    if not (
        ctrl_prev.size == delta.size == ctrl_lower.size == ctrl_upper.size
    ):
        raise ValueError(
            "ctrl_prev, delta, ctrl_lower, and ctrl_upper must have matching lengths"
        )

    margin = max(float(guard_margin_rad), 0.0)
    lower = ctrl_lower + margin
    upper = ctrl_upper - margin
    if np.any(lower > upper):
        lower = ctrl_lower.copy()
        upper = ctrl_upper.copy()

    ctrl_prev = np.clip(ctrl_prev, lower, upper)
    max_up = upper - ctrl_prev
    max_down = lower - ctrl_prev
    safe_delta = np.where(delta >= 0.0, np.minimum(delta, max_up), np.maximum(delta, max_down))
    return np.clip(ctrl_prev + safe_delta, lower, upper)


def acceleration_transport_controller(
    q: np.ndarray,
    qvel: np.ndarray,
    ctrl_prev: np.ndarray,
    ctrl_lower: np.ndarray,
    ctrl_upper: np.ndarray,
    tool_pos: np.ndarray,
    tool_rot: np.ndarray,
    tool_jacobian_pos: np.ndarray,
    tool_jacobian_rot: np.ndarray,
    a_axis_cmd: float,
    axis_state: float,
    transport_axis: str | int,
    fixed_position: np.ndarray,
    target_tool_rot: np.ndarray | None = None,
    dt: float = 0.002,
    a_axis_max_m_s2: float = ACCEL_AXIS_DEFAULT_A_MAX_M_S2,
    v_axis_max_m_s: float = ACCEL_AXIS_DEFAULT_V_MAX_M_S,
    torque_headroom: float = 0.9,
    joint_speed_limit_scale: float = 1.0,
    move_axis_weight: float = 120.0,
    hold_axis_weight: float = 100.0,
    orientation_weight: float = 64.0,
    hold_axis_gain: float = 8.0,
    posture_target: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """
    Axis-agnostic acceleration-commanded transport in the world frame.

    The caller selects a transport axis (`x`, `y`, or `z`). The controller:

    - integrates the signed scalar acceleration command into an internal
      scalar velocity state for that axis
    - holds the two orthogonal world axes near the supplied fixed position
    - holds the tool orientation in the world frame
    - uses all six joints in the weighted least-squares solve so the same
      transport routine can be reused for X, Y, or Z motion without a separate
      hardcoded branch

    This is the generic version of the existing X transport controller. The
    legacy `acceleration_x_transport_controller` remains available as a
    compatibility wrapper for older scripts and tests. The optional
    `joint_speed_limit_scale` tightens the per-joint speed ceiling before the
    feasibility checks, which is useful when the simulator's measured joint
    velocity guard is much stricter than the nominal MuJoCo servo envelope.
    """
    q = np.asarray(q, dtype=np.float64).reshape(6)
    qvel = np.asarray(qvel, dtype=np.float64).reshape(6)
    ctrl_prev = np.asarray(ctrl_prev, dtype=np.float64).reshape(6)
    ctrl_lower = np.asarray(ctrl_lower, dtype=np.float64).reshape(6)
    ctrl_upper = np.asarray(ctrl_upper, dtype=np.float64).reshape(6)
    tool_pos = np.asarray(tool_pos, dtype=np.float64).reshape(3)
    tool_rot = np.asarray(tool_rot, dtype=np.float64).reshape(3, 3)
    tool_jacobian_pos = np.asarray(tool_jacobian_pos, dtype=np.float64).reshape(3, 6)
    tool_jacobian_rot = np.asarray(tool_jacobian_rot, dtype=np.float64).reshape(3, 6)
    fixed_position = np.asarray(fixed_position, dtype=np.float64).reshape(3)
    joint_speed_limit_scale = max(abs(float(joint_speed_limit_scale)), 1e-6)

    axis_idx = axis_name_to_index(transport_axis)
    fixed_axes = orthogonal_axis_indices(axis_idx)

    a_max = max(abs(float(a_axis_max_m_s2)), 0.0)
    v_max = max(abs(float(v_axis_max_m_s)), 0.0)
    a_eff = float(np.clip(float(a_axis_cmd), -a_max, a_max))
    v_axis_pre = float(axis_state) + a_eff * float(dt)
    v_axis_target = float(np.clip(v_axis_pre, -v_max, v_max))

    if target_tool_rot is None:
        target_tool_rot = TARGET_SITE_ROTATION_WORLD
    omega_des = 2.5 * tool_target_orientation_omega(
        tool_rot, target_site_rot=target_tool_rot
    )

    translation_weights = np.full(3, float(hold_axis_weight), dtype=np.float64)
    translation_weights[axis_idx] = float(move_axis_weight)
    translation_des = np.zeros(3, dtype=np.float64)
    translation_des[axis_idx] = v_axis_target
    for idx in fixed_axes:
        translation_des[idx] = float(
            hold_axis_gain * (fixed_position[idx] - tool_pos[idx])
        )

    a_blocks: list[np.ndarray] = []
    b_blocks: list[np.ndarray] = []
    for idx in range(3):
        weight = float(np.sqrt(max(translation_weights[idx], 0.0)))
        a_blocks.append(weight * tool_jacobian_pos[idx : idx + 1, :])
        b_blocks.append(weight * np.array([translation_des[idx]], dtype=np.float64))

    a_blocks.append(np.sqrt(float(orientation_weight)) * tool_jacobian_rot)
    b_blocks.append(np.sqrt(float(orientation_weight)) * omega_des)

    if posture_target is None:
        posture_target = ctrl_prev
    posture_target = np.asarray(posture_target, dtype=np.float64).reshape(6)
    posture_error = posture_target - q
    posture_gains = np.array([0.008, 0.010, 0.010, 0.006, 0.006, 0.004], dtype=np.float64)
    posture_damping = np.array([0.020, 0.024, 0.024, 0.018, 0.018, 0.014], dtype=np.float64)
    desired_posture_step = posture_gains * posture_error - posture_damping * qvel
    posture_weights = np.array([0.08, 0.12, 0.12, 0.10, 0.08, 0.06], dtype=np.float64)
    a_blocks.append(np.diag(np.sqrt(posture_weights)))
    b_blocks.append(np.sqrt(posture_weights) * desired_posture_step)

    delta = _solve_weighted_delta(
        a_blocks=a_blocks,
        b_blocks=b_blocks,
        num_dofs=6,
        regularization=1e-3,
    )

    max_delta = VELOCITY_MAX_DELTA_FULL.copy()
    delta = np.clip(delta, -max_delta, max_delta)
    qdot_des = delta / max(float(dt), 1e-9)

    q_dot_ceiling = (VELOCITY_MAX_DELTA_FULL * joint_speed_limit_scale) / max(float(dt), 1e-6)
    abs_ratio = np.abs(qdot_des) / np.maximum(q_dot_ceiling, 1e-9)
    speed_ratio = float(np.max(abs_ratio))
    scale_speed = 1.0 if speed_ratio <= 1.0 else 1.0 / speed_ratio
    if scale_speed < 1.0:
        qdot_des = qdot_des * scale_speed
        delta = qdot_des * float(dt)

    ctrl = _apply_joint_limit_guardrails(ctrl_prev, delta, ctrl_lower, ctrl_upper)

    tau_est_full = SERVO_KP * (ctrl - q) - SERVO_KD * qvel
    limit_full = SERVO_FORCE_LIMIT * float(torque_headroom)
    abs_tau_ratio = np.abs(tau_est_full) / np.maximum(limit_full, 1e-9)
    torque_ratio = float(np.max(abs_tau_ratio))
    scale_torque = 1.0 if torque_ratio <= 1.0 else 1.0 / torque_ratio
    if scale_torque < 1.0:
        qdot_des = qdot_des * scale_torque
        delta = qdot_des * float(dt)
    ctrl = _apply_joint_limit_guardrails(ctrl_prev, delta, ctrl_lower, ctrl_upper)
    tau_est_full = SERVO_KP * (ctrl - q) - SERVO_KD * qvel

    total_scale = float(scale_speed * scale_torque)
    axis_state_next = float(v_axis_target * total_scale)
    axis_velocity_realized_cmd = axis_state_next

    diagnostics = {
        "transport_axis": axis_index_to_name(axis_idx),
        "transport_axis_index": int(axis_idx),
        "fixed_axis_indices": list(fixed_axes),
        "a_axis_cmd": float(a_axis_cmd),
        "a_axis_effective": float(a_eff),
        "axis_state_next": axis_state_next,
        "axis_velocity_target": float(v_axis_target),
        "axis_velocity_realized_cmd": float(axis_velocity_realized_cmd),
        "axis_velocity_saturated": bool(abs(v_axis_pre) > v_max + 1e-12),
        "speed_scale": float(scale_speed),
        "torque_scale": float(scale_torque),
        "joint_speed_limit_scale": float(joint_speed_limit_scale),
        "speed_ratio": float(speed_ratio),
        "torque_ratio": float(torque_ratio),
        "q_dot_des": qdot_des.tolist(),
        "tau_estimate_nm": tau_est_full.tolist(),
    }
    return ctrl, diagnostics


# Default safety caps for the acceleration-commanded X controller.
# Kept conservative so the arm never races near its servo limits purely from an
# outer-loop acceleration signal.
ACCEL_X_DEFAULT_A_MAX_M_S2 = 0.15
ACCEL_X_DEFAULT_V_MAX_M_S = 0.08


def acceleration_x_transport_controller(
    q: np.ndarray,
    qvel: np.ndarray,
    ctrl_prev: np.ndarray,
    ctrl_lower: np.ndarray,
    ctrl_upper: np.ndarray,
    tool_pos: np.ndarray,
    tool_rot: np.ndarray,
    tool_jacobian_pos: np.ndarray,
    tool_jacobian_rot: np.ndarray,
    a_x_cmd: float,
    v_x_state: float,
    z_hold: float,
    target_tool_rot: np.ndarray | None = None,
    pan_target: float = 0.0,
    dt: float = 0.002,
    a_x_max_m_s2: float = ACCEL_X_DEFAULT_A_MAX_M_S2,
    v_x_max_m_s: float = ACCEL_X_DEFAULT_V_MAX_M_S,
    torque_headroom: float = 0.9,
    task_weight_x: float = 100.0,
    task_weight_z: float = 64.0,
    task_weight_omega: float = 36.0,
) -> tuple[np.ndarray, dict]:
    """
    Acceleration-commanded `attachment_site` transport along world X.

    **Outer-loop input:** a single signed scalar ``a_x_cmd`` in m/s^2. Its sign
    picks direction along world X, its magnitude sets how aggressively the tool
    accelerates. There is no X goal, no path, and no velocity command from the
    caller.

    **Mechanism:** The controller carries an internal X velocity state
    ``v_x_state`` (m/s) that the caller passes in each step. Every call it

    1. clips ``a_x_cmd`` to ``[-a_x_max_m_s2, +a_x_max_m_s2]``
    2. integrates ``v_x_new = clip(v_x_state + a_eff * dt, +/- v_x_max_m_s)``
    3. solves the same weighted-least-squares differential IK used by
       ``velocity_x_transport_controller`` with ``v_x_new`` as the X target,
       ``Kz * (z_hold - z)`` as the Z target, and a small orientation-hold
       angular velocity.
    4. applies the **same** feasibility guards against the MuJoCo position
       servos defined in ``ur5e.xml``:

       - per-joint speed ceiling from ``VELOCITY_MAX_DELTA_RED / dt``
       - predicted servo torque ``Kp*(ctrl - q) - Kd*qvel`` inside
         ``torque_headroom * SERVO_FORCE_LIMIT``

       If anything is scaled down, the integrator state is shrunk by the same
       total factor so the controller cannot wind up velocity it could not
       realize this step.

    Shoulder pan stays locked at ``pan_target``; only joints 1..5 are driven.
    The returned ``diagnostics["v_x_state_next"]`` must be fed back on the next
    call as ``v_x_state``.
    """
    q = np.asarray(q, dtype=np.float64)
    qvel = np.asarray(qvel, dtype=np.float64)
    ctrl_prev = np.asarray(ctrl_prev, dtype=np.float64)
    ctrl_lower = np.asarray(ctrl_lower, dtype=np.float64)
    ctrl_upper = np.asarray(ctrl_upper, dtype=np.float64)
    tool_pos = np.asarray(tool_pos, dtype=np.float64)
    tool_rot = np.asarray(tool_rot, dtype=np.float64).reshape(3, 3)
    tool_jacobian_pos = np.asarray(tool_jacobian_pos, dtype=np.float64)
    tool_jacobian_rot = np.asarray(tool_jacobian_rot, dtype=np.float64)

    red = slice(1, 6)
    num_red = 5

    a_max = max(float(a_x_max_m_s2), 0.0)
    v_max = max(float(v_x_max_m_s), 0.0)

    a_eff = float(np.clip(float(a_x_cmd), -a_max, a_max))
    v_x_pre = float(v_x_state) + a_eff * float(dt)
    v_x_target = float(np.clip(v_x_pre, -v_max, v_max))

    # Tighter Z regulation than the velocity controller so the height rejects
    # disturbances fast even when the speed/torque limiter scales down the
    # whole task vector during brief near-singular configurations.
    Kz = 12.0
    K_omega = 2.5
    v_z_des = Kz * (float(z_hold) - float(tool_pos[2]))
    if target_tool_rot is None:
        target_tool_rot = TARGET_SITE_ROTATION_WORLD
    omega_des = K_omega * tool_target_orientation_omega(
        tool_rot, target_site_rot=target_tool_rot
    )

    wx = float(np.sqrt(max(task_weight_x, 0.0)))
    wz = float(np.sqrt(max(task_weight_z, 0.0)))
    wo = float(np.sqrt(max(task_weight_omega, 0.0)))

    jx = tool_jacobian_pos[0:1, red]
    jz = tool_jacobian_pos[2:3, red]
    jw = tool_jacobian_rot[:, red]

    a_blocks = [wx * jx, wz * jz, wo * jw]
    b_blocks = [
        wx * np.array([v_x_target], dtype=np.float64),
        wz * np.array([v_z_des], dtype=np.float64),
        wo * omega_des,
    ]
    q_dot_des_red = _solve_weighted_delta(
        a_blocks=a_blocks,
        b_blocks=b_blocks,
        num_dofs=num_red,
        regularization=1e-3,
    )

    q_dot_ceiling = VELOCITY_MAX_DELTA_RED / max(dt, 1e-6)
    abs_ratio = np.abs(q_dot_des_red) / np.maximum(q_dot_ceiling, 1e-9)
    speed_ratio = float(np.max(abs_ratio))
    scale_speed = 1.0 if speed_ratio <= 1.0 else 1.0 / speed_ratio
    q_dot_des_red = q_dot_des_red * scale_speed

    ctrl_red_new = ctrl_prev[red] + q_dot_des_red * dt
    ctrl_red_new = np.clip(ctrl_red_new, ctrl_lower[red], ctrl_upper[red])

    kp_red = SERVO_KP[red]
    kd_red = SERVO_KD[red]
    limit_red = SERVO_FORCE_LIMIT[red] * float(torque_headroom)
    tau_est_red = kp_red * (ctrl_red_new - q[red]) - kd_red * qvel[red]
    abs_tau_ratio = np.abs(tau_est_red) / np.maximum(limit_red, 1e-9)
    torque_ratio = float(np.max(abs_tau_ratio))
    scale_torque = 1.0 if torque_ratio <= 1.0 else 1.0 / torque_ratio
    if scale_torque < 1.0:
        q_dot_des_red = q_dot_des_red * scale_torque
        ctrl_red_new = ctrl_prev[red] + q_dot_des_red * dt
        ctrl_red_new = np.clip(ctrl_red_new, ctrl_lower[red], ctrl_upper[red])
        tau_est_red = kp_red * (ctrl_red_new - q[red]) - kd_red * qvel[red]

    total_scale = float(scale_speed * scale_torque)
    # Shrink the integrator back if we had to slow down, so we don't wind up
    # commanded velocity that the joints could not realize this step.
    v_x_state_next = float(v_x_target * total_scale)
    v_x_realized_cmd = v_x_state_next

    ctrl = ctrl_prev.copy()
    ctrl[red] = ctrl_red_new
    ctrl[0] = float(pan_target)
    ctrl = np.clip(ctrl, ctrl_lower, ctrl_upper)

    tau_est_full = np.zeros(6, dtype=np.float64)
    tau_est_full[0] = SERVO_KP[0] * (ctrl[0] - q[0]) - SERVO_KD[0] * qvel[0]
    tau_est_full[red] = tau_est_red

    diagnostics = {
        "a_x_cmd": float(a_x_cmd),
        "a_x_effective": a_eff,
        "v_x_state_next": v_x_state_next,
        "v_x_target": float(v_x_target),
        "v_x_realized_cmd": float(v_x_realized_cmd),
        "v_x_saturated": bool(abs(v_x_pre) > v_max + 1e-12),
        "speed_scale": float(scale_speed),
        "torque_scale": float(scale_torque),
        "speed_ratio": speed_ratio,
        "torque_ratio": torque_ratio,
        "q_dot_des_red": q_dot_des_red.tolist(),
        "tau_estimate_nm": tau_est_full.tolist(),
    }
    return ctrl, diagnostics
