#!/usr/bin/env python3
"""Shared infra for the TOOL-Y pumping cartpole swing-up + LQR pipeline.

CONFIGURATION (per this task's own explicit spec -- do not silently drift
from these three choices, all three were gotten wrong by a predecessor):
  - Pose:  ARM_Q0 = simulation.ur5e_pendulum_compose.DEFAULT_ARM_Q
           (wrist_2 ~ 0, the user's real-hardware pose).
  - Asset: assets/ur5e_pendulum/pendulum_attachment.xml (DEFAULT_PENDULUM_XML),
           hinge axis local Z, REAL measured friction (damping 1e-4,
           frictionloss 1e-3).
  - Pumping direction: TOOL Y (attachment_site's local Y column in world),
           NOT world X. At ARM_Q0, tool Y is 82.7 deg from vertical and
           PERPENDICULAR to the hinge axis (tool Z) -- alignment (kappa) 1.0.

WHY A ROTATED-FRAME WRAPPER, NOT A controller_core CHANGE.

controller_core/x_axis_cartesian_impedance is structurally axis-index based
(transport_axis_index in {0,1,2} selects one of world X/Y/Z as the "task"
axis; the other two are held at their reset value). Tool Y is a genuine 3D
diagonal (world components [0.71, 0.69, 0.13] at ARM_Q0), not any world
axis, so driving the arm along it needs the controller to track a moving
target along that diagonal -- which the axis-index architecture cannot
express directly.

Rather than extend the protected, heavily-validated controller_core module
(AGENTS.md: never combine controller-logic changes with other work; a
change there is real risk to every other transport lane), this module
instead ROTATES THE STATE fed to the existing, UNMODIFIED controller so
that its own "axis 0" (still transport_axis_index=0, the historical
default) IS the tool-Y direction:

  R = [ d | u1 | u2 ]   (3x3, world frame, orthonormal columns)

d = the chosen pumping direction (tool Y here), u1/u2 = two more world
unit vectors completing an orthonormal basis (chosen as tool X and tool Z
-- the OTHER two attachment_site columns, already orthonormal by
construction, no re-derivation needed).

Every per-cycle state handed to the controller/adapter has its
TRANSLATIONAL fields rotated by R^T before being passed in:
  ee_pos_rot     = R^T @ ee_pos_world
  ee_lin_vel_rot = R^T @ ee_lin_vel_world
  jacobian_rot[:3,:] = R^T @ jacobian_world[:3,:]   (rows 0:3 only; the
                                                      rotational rows 3:6
                                                      are untouched -- an
                                                      an orientation task
                                                      operates on ee_quat
                                                      directly, not on
                                                      this frame at all)

This is exact, not an approximation, because:
  1. tau = J.T @ wrench is LINEAR in J and wrench. Since R is orthonormal
     (R^T R = I), J_rot.T @ wrench_rot = (R^T J_trans).T @ wrench_trans_rot
     + J_rot_rows.T @ wrench_rot_rows = J_trans.T @ (R @ wrench_trans_rot)
     + (unchanged rot rows).T @ (unchanged). I.e. the joint torque the
     controller emits, when fed the rotated J and given a wrench it
     computes internally in the rotated basis, is EXACTLY the same
     real-world torque as if a controller understood arbitrary directions
     natively and had produced wrench_world = R @ wrench_trans_rot.
  2. task_space_inertia_shaping's Lambda = (J M^-1 J^T + eps I)^-1: since
     J_rot_trans = R^T J_world_trans, Lambda_rot = R^T Lambda_world R (a
     congruence transform under an orthonormal change of basis) -- exactly
     the physically-correct effective inertia along the new axes, not a
     distortion.
  3. nullspace_posture's projector N = I - J^+ J depends only on the ROW
     SPACE of J; premultiplying 3 of its 6 rows by an invertible 3x3 (R^T)
     does not change that row space, so N is IDENTICAL, not merely close.
  4. cond(J) / singular_scale / SCI: singular values of J_rot equal those
     of J_world exactly, because premultiplying by an orthonormal matrix
     preserves singular values. Conditioning is therefore unaffected --
     appropriately, since this is a relabelling of which linear
     combination of Cartesian directions is "task" vs "held", not a
     change to the arm's actual kinematic conditioning.
  5. Safety guards: ImpedanceSafetyMonitor reads state["ee_pos"] and is
     given move_axis=0 (unchanged default) by the SAME adapter code every
     other transport driver uses -- feeding it the rotated ee_pos makes
     its existing "hold axis 1/2 within 0.03 m of reset" checks measure
     drift PERPENDICULAR TO THE CHOSEN DIRECTION (i.e. along tool X and
     tool Z) instead of world Y/Z, with the SAME 0.03 m threshold --
     neither weakened nor bypassed, just correctly re-expressed in the
     frame the task is actually being driven in.

No line of controller_core/ or simulation/ur5e_mujoco_torque.py is
modified by this module. reset()/step() on the adapter are exactly the
calls every other transport script makes; only the ee_pos/ee_lin_vel/
jacobian VALUES handed to them differ.
"""

from __future__ import annotations

import functools
import sys
from pathlib import Path

import mujoco
import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from simulation.ur5e_pendulum_compose import (  # noqa: E402
    DEFAULT_ARM_Q,
    DEFAULT_PENDULUM_XML,
    PendulumConstants,
    compose_ur5e_pendulum_model,
    derive_pendulum_constants,
)
from simulation.ur5e_mujoco_torque import (  # noqa: E402
    MujocoUR5eState,
    build_initial_state_and_adapter,
    build_mujoco_state,
)
from tools.diagnostics.pendulum_balance_torque_lqr import (  # noqa: E402
    find_inverted_angle as _find_inverted_angle_rigorous,
)

# ---------------------------------------------------------------------------
# The three configuration choices, per this task's spec.
# ---------------------------------------------------------------------------
ARM_Q0 = DEFAULT_ARM_Q.copy()
PENDULUM_XML = DEFAULT_PENDULUM_XML
CONFIG_PATH = REPO_ROOT / "config" / "ur5e_mujoco_torque_osc_tuned.yaml"
CONTROLLER_KIND = "impedance"

RATE_HZ = 500.0
CONTROL_DT = 1.0 / RATE_HZ

# config/ur5e_mujoco_torque_osc_tuned.yaml specifies gravity_source: "pinocchio",
# but this conda env (mujoco_ur5e) does not actually have the `pinocchio` python
# package importable (verified: `import pinocchio` raises ModuleNotFoundError,
# despite environment.yml listing `pin>=3.1` and libpinocchio 4.0.0 being present
# as an unrelated pypi package with no importable module). Every script in this
# module therefore forces gravity_source="mujoco_qfrc" (the MuJoCo-native
# fallback already implemented in simulation/ur5e_mujoco_torque.py's adapter)
# rather than the config's nominal value, which would otherwise crash on
# ImportError as soon as coriolis_feedforward is exercised. This does not
# silently change the GRAVITY term (build_mujoco_state always computes
# state.gravity_torque via the MuJoCo-native path regardless of gravity_source,
# and the adapter uses that value directly whenever it is set -- gravity_source
# only ever controlled the CORIOLIS feedforward path here). AGENTS.md records
# Pinocchio-vs-MuJoCo gravity parity as <1e-8 Nm and mass-matrix parity as
# <1e-8, so this substitution is not expected to change results in a way that
# matters at this task's precision.
GRAVITY_SOURCE = "mujoco_qfrc"

JOINT_NAMES = (
    "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
    "wrist_1_joint", "wrist_2_joint", "wrist_3_joint",
)


def load_config(config_path: Path = CONFIG_PATH) -> dict:
    with Path(config_path).open() as fp:
        return yaml.safe_load(fp)


def build_model() -> mujoco.MjModel:
    return compose_ur5e_pendulum_model(pendulum_xml=PENDULUM_XML)


@functools.lru_cache(maxsize=1)
def default_constants() -> PendulumConstants:
    return derive_pendulum_constants(build_model(), ARM_Q0)


def wrap_pi(angle: float) -> float:
    return float(np.mod(angle + np.pi, 2 * np.pi) - np.pi)


def hinge_ids(model: mujoco.MjModel) -> tuple[int, int, int]:
    pend_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "/pendulum_hinge")
    if pend_jid < 0:
        pend_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "pendulum_hinge")
    hub_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "/pendulum_hub")
    if hub_bid < 0:
        hub_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pendulum_hub")
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "attachment_site")
    return pend_jid, hub_bid, site_id


def joint_ids_for(model: mujoco.MjModel) -> list[int]:
    return [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in JOINT_NAMES]


def find_hanging_and_inverted_angle(model, data, pend_qpos_adr: int) -> tuple[float, float]:
    inverted = _find_inverted_angle_rigorous(model, data, pend_qpos_adr)
    # BUG FOUND AND FIXED while smoke-testing this module: hanging must be
    # exactly pi away from inverted (mod 2pi) -- an earlier version of this
    # line, `mod(inverted + 2*pi + pi, 2*pi) - pi`, algebraically collapses
    # back to `wrap_pi(inverted)` (the extra +2*pi is a no-op under mod, and
    # +pi/-pi then cancel), silently returning hanging == inverted. Caught
    # because a swing-up trial reported "min|phi| = 0.000 deg at t=0.000s"
    # starting from what should have been the FAR equilibrium. Verified fix
    # against AGENTS.md's independently-recorded values at this exact pose
    # (inverted=+0.1271 rad, hanging=-3.0145 rad).
    hanging = wrap_pi(inverted + np.pi)
    return hanging, inverted


def resolve_equilibria(model: mujoco.MjModel, arm_q=None) -> tuple[float, float]:
    arm_q = ARM_Q0 if arm_q is None else np.asarray(arm_q, dtype=np.float64).reshape(6)
    data = mujoco.MjData(model)
    data.qpos[:6] = arm_q
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)
    pend_jid, _, _ = hinge_ids(model)
    return find_hanging_and_inverted_angle(model, data, model.jnt_qposadr[pend_jid])


def hinge_axis_world(model: mujoco.MjModel, data: mujoco.MjData, pend_jid: int, hub_bid: int) -> np.ndarray:
    """The hinge's actual world-frame axis, read from the joint/body
    kinematics -- NOT assumed to be any particular local axis of the tool
    frame. (A predecessor script hardcoded "local X of the tool frame",
    which was correct only for a different, retired asset with a local-X
    hinge; this asset's hinge is local Z. Reading it off the model makes
    this function correct for either.)"""
    xmat = np.asarray(data.xmat[hub_bid], dtype=np.float64).reshape(3, 3)
    axis_local = np.asarray(model.jnt_axis[pend_jid], dtype=np.float64)
    axis_world = xmat @ axis_local
    return axis_world / np.linalg.norm(axis_world)


def tool_frame_world(data: mujoco.MjData, site_id: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(tool_x, tool_y, tool_z) world unit vectors -- attachment_site's own
    rotation matrix columns."""
    R = np.asarray(data.site_xmat[site_id], dtype=np.float64).reshape(3, 3)
    return R[:, 0].copy(), R[:, 1].copy(), R[:, 2].copy()


def pumping_rotation_matrix(tool_x: np.ndarray, tool_y: np.ndarray, tool_z: np.ndarray) -> np.ndarray:
    """3x3 orthonormal world-frame basis, column 0 = pumping direction
    (tool Y), columns 1/2 = the two held directions (tool X, tool Z --
    already orthonormal, no re-derivation)."""
    R = np.column_stack([tool_y, tool_x, tool_z])
    ortho_err = float(np.max(np.abs(R.T @ R - np.eye(3))))
    if ortho_err > 1e-9:
        raise ValueError(f"pumping_rotation_matrix: columns not orthonormal (err={ortho_err:.3e})")
    return R


def rotate_state_inplace(state: MujocoUR5eState, R: np.ndarray) -> MujocoUR5eState:
    """Rotate the translational fields of ``state`` into R's frame (axis 0
    becomes the pumping direction). See module docstring for the exactness
    argument. Mutates and returns ``state``."""
    RT = R.T
    state.ee_pos = RT @ np.asarray(state.ee_pos, dtype=np.float64).reshape(3)
    state.ee_lin_vel = RT @ np.asarray(state.ee_lin_vel, dtype=np.float64).reshape(3)
    J = np.array(state.jacobian, dtype=np.float64).reshape(6, 6)
    J[:3, :] = RT @ J[:3, :]
    state.jacobian = J
    if state.target_ee_pos is not None:
        state.target_ee_pos = RT @ np.asarray(state.target_ee_pos, dtype=np.float64).reshape(3)
    if state.target_ee_vel is not None:
        state.target_ee_vel = RT @ np.asarray(state.target_ee_vel, dtype=np.float64).reshape(3)
    return state


def measure_cart_coupling_nm_per_mps2(
    model: mujoco.MjModel, arm_q: np.ndarray, inverted_angle: float, direction_world: np.ndarray,
) -> float:
    """The EXACT hinge generalized torque produced by one unit (1 m/s^2) of
    cart acceleration along ``direction_world`` (a world unit vector), at
    the given hinge angle: in the pivot's non-inertial frame, a cart
    acceleration ``a`` applies a pseudo-force F=-m*a*direction_world at the
    swinging subtree's COM, and the resulting generalized torque about the
    (model-measured) hinge axis n is Q = n . (r x F), r = COM - pivot.

    This is the function that makes kappa an EXPLICIT, measured quantity
    (never hard-coded to 1, never assumed from a shortcut formula like
    -mgr/g -- a prior version of this idea in this repo's history was
    checked against a closed-loop finite-difference probe and found to be
    off in both sign and magnitude for a different asset/pose; see this
    module's own header). ``direction_world`` here is tool Y; passing world
    X instead reproduces this task's own "kappa=0.7165" comparison figure.
    """
    data = mujoco.MjData(model)
    data.qpos[:6] = np.asarray(arm_q, dtype=np.float64).reshape(6)
    pend_jid, hub_bid, site_id = hinge_ids(model)
    qadr = model.jnt_qposadr[pend_jid]
    data.qpos[qadr] = float(inverted_angle)
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)

    n_axis = hinge_axis_world(model, data, pend_jid, hub_bid)
    hinge_pos = np.asarray(data.xanchor[pend_jid], dtype=np.float64).copy()
    com = np.asarray(data.subtree_com[hub_bid], dtype=np.float64).copy()
    m = float(model.body_subtreemass[hub_bid])
    r = com - hinge_pos
    d = np.asarray(direction_world, dtype=np.float64)
    d = d / np.linalg.norm(d)
    F = -m * d
    return float(np.dot(n_axis, np.cross(r, F)))


def build_rotated_initial_state_and_adapter(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    site_id: int,
    joint_ids: list[int],
    *,
    R: np.ndarray,
    controller_cfg: dict,
    gravity_mode: str,
    gravity_source: str = GRAVITY_SOURCE,
    coriolis_feedforward: bool = True,
    torque_limit_scale: float = 1.0,
):
    """build_initial_state_and_adapter(), then re-anchored in the rotated
    (tool-Y) frame: the adapter's own reset() captures _x0/_y0/_z0 and the
    safety monitor's _pos0 from whatever ee_pos it is given, so calling
    reset() a second time with the rotated ee_pos re-anchors everything
    (controller AND safety guards) in the pumping-direction frame. Cheap
    (no physics step) and idempotent -- reset() has no state carried over
    from the first call that this second call doesn't fully overwrite."""
    state, adapter = build_initial_state_and_adapter(
        model, data, site_id, joint_ids,
        controller_cfg=controller_cfg,
        transport_axis_index=0,
        target_x_delta=0.0,
        controller_kind=CONTROLLER_KIND,
        force_hold_current_pose=False,
        gravity_mode=gravity_mode,
        gravity_source=gravity_source,
        coriolis_feedforward=coriolis_feedforward,
        torque_limit_scale=torque_limit_scale,
    )
    rotate_state_inplace(state, R)
    adapter.reset(state)
    return state, adapter


def build_step_state(
    model: mujoco.MjModel, data: mujoco.MjData, *, site_id: int, joint_ids: list[int],
    time_s: float, target_x: float, target_x_vel: float, target_x_accel: float,
    reference_quat: np.ndarray, R: np.ndarray,
) -> tuple[MujocoUR5eState, np.ndarray]:
    """One control cycle's state, in WORLD frame first (returned separately
    for logging/HUD), then rotated in place for the controller/adapter.

    ``R`` is a FIXED (frozen-at-trial-start) rotation. Measured limitation
    (see build_step_state_live's docstring): the tool frame this R was
    captured from can rotate up to ~12 deg over a single guard-clean ~0.22 m
    tool-Y move at ARM_Q0, so a frozen R's "held" directions (tool X / tool Z
    at t=0) are no longer exactly the CURRENT tool frame's orthogonal
    directions by the end of a large move. Kept for the small-perturbation
    LQR-balance trials (state stays near the inverted equilibrium, negligible
    rotation) and as the deliberately-labeled frozen-frame baseline; large
    swing-up moves should use build_step_state_live instead."""
    state = build_mujoco_state(
        model, data, site_id=site_id, joint_ids=joint_ids,
        time_s=time_s, dt_s=CONTROL_DT,
        target_x=target_x, target_x_vel=target_x_vel, target_x_accel=target_x_accel,
        reference_quat=reference_quat, transport_axis_index=0,
        gravity_compensation=True,
    )
    ee_pos_world = np.asarray(state.ee_pos, dtype=np.float64).copy()
    rotate_state_inplace(state, R)
    return state, ee_pos_world


def build_step_state_two_axis(
    model: mujoco.MjModel, data: mujoco.MjData, *, site_id: int, joint_ids: list[int],
    time_s: float, target_x: float, target_x_vel: float, target_x_accel: float,
    target_y: float, target_y_vel: float,
    reference_quat: np.ndarray, R: np.ndarray,
) -> tuple[MujocoUR5eState, np.ndarray]:
    """Two-actively-driven-axis variant of build_step_state: same FIXED
    (frozen-at-trial-start) R as build_step_state -- see that function's
    docstring for the measured ~12 deg/0.22m frame-rotation caveat this
    inherits -- but ALSO drives the "y-role" (tool-X-role) axis via a
    genuine moving target instead of holding it at the frozen p0[y_axis].
    Requires controller config second_task_axis_enabled: true.

    A LIVE-frame version of this (build_step_state_live) was built and
    REJECTED: making R live-per-cycle fixes the CONTROLLER's target-tracking
    exactly (derivation in that function's docstring) but does nothing for
    ImpedanceSafetyMonitor's drift check, which compares the CURRENT rotated
    ee_pos against a FROZEN _pos0 captured at reset() under R(0) -- as R(t)
    rotates away from R(0), that comparison mixes two different physical
    projections and manufactures a spurious apparent drift that is not a
    real safety concern, tripping the guard EARLIER than the frozen-frame
    version (measured: t=0.348s vs t=0.714s for an identical commanded
    move) -- i.e. strictly worse, and for a reason that has nothing to do
    with real off-plane motion. Fixing that properly needs the guard's own
    reference re-expressed in the live frame too (a controller_core/safety.py
    change), which was judged too much additional protected-code surgery to
    complete safely in this pass. Kept only as a documented negative result;
    build_step_state_live remains importable for anyone who lands the
    guard-side fix later, but nothing in this pipeline calls it."""
    state = build_mujoco_state(
        model, data, site_id=site_id, joint_ids=joint_ids,
        time_s=time_s, dt_s=CONTROL_DT,
        target_x=target_x, target_x_vel=target_x_vel, target_x_accel=target_x_accel,
        reference_quat=reference_quat, transport_axis_index=0,
        gravity_compensation=True,
    )
    ee_pos_world = np.asarray(state.ee_pos, dtype=np.float64).copy()
    rotate_state_inplace(state, R)

    _orig_as_robot_state = state.as_robot_state

    def _as_robot_state_with_y():
        d = _orig_as_robot_state()
        d["target_y"] = float(target_y)
        d["target_y_vel"] = float(target_y_vel)
        return d

    state.as_robot_state = _as_robot_state_with_y  # type: ignore[method-assign]
    return state, ee_pos_world


def build_step_state_live(
    model: mujoco.MjModel, data: mujoco.MjData, *, site_id: int, joint_ids: list[int],
    time_s: float, p0_world: np.ndarray, s_y: float, s_y_vel: float,
    s_x: float = 0.0, s_x_vel: float = 0.0, reference_quat: np.ndarray,
) -> tuple[MujocoUR5eState, np.ndarray, np.ndarray]:
    """LIVE-tool-frame variant of build_step_state: tool_x(t)/tool_y(t)/
    tool_z(t) and R(t) are recomputed from the CURRENT ``data`` every call
    (not frozen at trial start), and the task-axis / second-task-axis
    targets are expressed so the controller's error signal is exact under a
    ROTATING frame, not just a fixed one.

    DERIVATION. We want the controller to track
        target_pos_world(t) = p0_world + s_y(t)*tool_y(t) + s_x(t)*tool_x(t)
    i.e. a point offset from the ORIGINAL (trial-start) world position by
    s_y(t) along the CURRENT tool-Y direction and s_x(t) along the CURRENT
    tool-X direction -- exactly what "live tool frame" pumping means: the
    pump directions track the arm's actual current orientation, not a stale
    snapshot.

    The controller computes x_err = target_x - [R(t)^T @ ee_pos_world(t)]_0
    = target_x - tool_y(t) . ee_pos_world(t) (since R(t)'s column 0 is
    tool_y(t) by construction). We want this to equal
    tool_y(t) . (target_pos_world(t) - ee_pos_world(t))
    = tool_y(t).p0_world + s_y(t) - tool_y(t).ee_pos_world(t)
    (using tool_y(t).tool_y(t)=1 and tool_y(t).tool_x(t)=0). Matching terms:
        target_x fed to the controller = s_y(t) + tool_y(t) . p0_world
    and symmetrically for the second task axis (tool X role):
        target_y fed to the controller = s_x(t) + tool_x(t) . p0_world
    This is exact for ANY R(t), not an approximation -- it is just
    "correctly re-express the ORIGINAL fixed reference point in whatever
    frame R(t) happens to be" rather than assuming R(t) stays constant.
    At t=0 (R(t)=R(0), ee_pos_world(0)=p0_world), this reduces to
    target_x(0) = s_y(0) + tool_y(0).p0_world = 0 + x_ref exactly, matching
    build_rotated_initial_state_and_adapter's own calibration.

    Requires the controller config to have
    ``second_task_axis_enabled: true`` (config/
    ur5e_mujoco_torque_osc_tuned_second_task_axis.yaml) for target_y to be
    consumed; with s_x=s_x_vel=0.0 always (the tool-Y-ONLY baseline case)
    this still matters -- it keeps the "held" tool-X-role direction tracking
    the LIVE frame's own zero-drift reference instead of a frozen one.

    The third (tool-Z-role / hinge-axis) direction is NOT corrected this way
    -- the controller has no equivalent target_z hook, and this direction is
    the hinge axis itself (motion along it does not couple into the
    pendulum), so residual staleness there is the one accepted
    approximation, unlike tool X/Y which are corrected exactly.

    Returns (state [rotated, ready for adapter.step], ee_pos_world,
    R(t) [for HUD/diagnostics]).
    """
    tool_x, tool_y, tool_z = tool_frame_world(data, site_id)
    R = pumping_rotation_matrix(tool_x, tool_y, tool_z)
    p0 = np.asarray(p0_world, dtype=np.float64).reshape(3)
    target_x = float(s_y) + float(tool_y @ p0)
    target_y = float(s_x) + float(tool_x @ p0)

    state = build_mujoco_state(
        model, data, site_id=site_id, joint_ids=joint_ids,
        time_s=time_s, dt_s=CONTROL_DT,
        target_x=target_x, target_x_vel=float(s_y_vel), target_x_accel=0.0,
        reference_quat=reference_quat, transport_axis_index=0,
        gravity_compensation=True,
    )
    ee_pos_world = np.asarray(state.ee_pos, dtype=np.float64).copy()
    rotate_state_inplace(state, R)
    # target_y/target_y_vel are NOT part of MujocoUR5eState.as_robot_state()'s
    # fixed field set (they are new, controller-only keys) -- stash them via
    # the same as_robot_state() dict the adapter reads, by monkeypatching a
    # thin wrapper rather than a new dataclass field, so every OTHER caller
    # of MujocoUR5eState is completely unaffected.
    _orig_as_robot_state = state.as_robot_state

    def _as_robot_state_with_y():
        d = _orig_as_robot_state()
        d["target_y"] = target_y
        d["target_y_vel"] = float(s_x_vel)
        return d

    state.as_robot_state = _as_robot_state_with_y  # type: ignore[method-assign]
    return state, ee_pos_world, R
