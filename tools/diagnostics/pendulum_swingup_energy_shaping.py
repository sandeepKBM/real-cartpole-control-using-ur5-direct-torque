#!/usr/bin/env python3
"""Energy-shaping (Astrom-Furuta style) swing-up for the pendulum at the arm
pose in ARM_Q0 below (as of 2026-08-12: the user's real-hardware UR5e
configuration, paired with the corrected local-Z hinge axis -- see ARM_Q0's
own comment, and note that pose sits on the wrist_2=0 arm singularity, a
separate concern from the pendulum's dynamics) -- replaces
pendulum_swingup_x_oscillation.py's fixed-frequency sinusoid, which searched
(amplitude, frequency) and found NOTHING better than ~5.5deg of arc: a fixed
frequency necessarily falls out of resonance as swing amplitude grows (a
pendulum's period is NOT amplitude-independent away from small angles), so
no (amplitude, frequency) pair can complete a full swing-up. This is a
different CONTROL STRUCTURE, not just different numbers: cart acceleration
is a live feedback function of the pendulum's instantaneous (angle,
angular velocity), not a pre-computed trajectory.

Law: a_cmd = -k_e * thetadot * cos(phi) * (E_top - E) - k_pos * (x - x0),
clipped to +-a_max. (The leading MINUS is load-bearing and was wrong until
2026-08-12 -- see the derivation + measurement at the a_energy line below.)
phi = theta - hanging_angle (0 at hanging, pi at
inverted); E is the pendulum's own rotational+gravitational energy
referenced to hanging = 0 (NOT total system energy -- the moving pivot's
own translational KE is deliberately excluded, matching the classical
derivation, which assumes the cart's kinematics -- not energy -- couples
into the pendulum). The k_pos term (not in the textbook-minimal version) is
a practical recentering term: pure energy shaping has no reason to return
the cart to x0 and can walk arbitrarily far from it.

Reuses the same torque-lane Cartesian impedance controller/config as every
other transport check this session -- a_cmd is integrated into
target_x/target_x_vel/target_x_accel each step, fed through the same
controller interface (build_mujoco_state -> adapter.step), same as the
sinusoid version.

3-parameter search (k_e, a_max, k_pos) via scipy.optimize.differential_
evolution, NOT RL, per this repo's own documented history of RL
gain-scheduling failures (AGENTS.md).
"""

from __future__ import annotations

import argparse
import dataclasses
import functools
import json
import sys
from pathlib import Path

import mujoco
import numpy as np
import yaml
from scipy.optimize import differential_evolution

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from simulation.ur5e_pendulum_compose import (  # noqa: E402
    DEFAULT_ARM_Q,
    DEFAULT_PENDULUM_XML,
    PendulumConstants,
    arm_q_for_pendulum_xml,
    compose_ur5e_pendulum_model,
    derive_pendulum_constants,
)
from simulation.ur5e_mujoco_torque import (  # noqa: E402
    build_initial_state_and_adapter,
    build_mujoco_state,
)
from tools.diagnostics.pendulum_balance_torque_lqr import find_inverted_angle as _find_inverted_angle_rigorous  # noqa: E402

# Switched 2026-08-12 to the wrist_orientation_task variant to test the one
# concrete, mechanistically-motivated fix identified for the orientation-
# holding ceiling: a real restoring spring for wrist orientation, isolated
# from the Λ-weighted task wrench pipeline that direct testing tonight
# showed is unstable even at well-conditioned poses (cond(J)~7-10, kp_rot
# sweep 0->50 caused monotonic guard-trip-earlier collapse). This config
# has wrist_orientation_task:true + kp_rot_wrist/kd_rot_wrist=10 already
# set (built earlier this session, never validated against the corrected
# pendulum model). Its mujoco.home_qpos field is stale/irrelevant here --
# these scripts set the arm pose from ARM_Q0 below, not from config.
CONFIG_PATH = REPO_ROOT / "config" / "ur5e_mujoco_torque_osc_hanging_pose_friction_ff_wrist_orient.yaml"
# Updated 2026-08-12 to the USER'S ACTUAL REAL-HARDWARE UR5e CONFIGURATION,
# read off the physical arm (wrist_2 wrapped into the model's valid range
# from a real 6.2879 rad probe). This replaces the 2026-08-11 mega
# pose-oscillation-stability-search winner
# ([0, -1.0920, 2.0935, -2.7686, 1.5621, 0], cond(J)=6.93), whose rationale
# was arm-side oscillation stability -- a real property, but one measured
# for a pose the physical apparatus is not actually in.
#
# WHY THIS POSE, measured on the composed model (see the joint element's
# comment in assets/ur5e_pendulum/pendulum_attachment.xml for the full
# derivation): the hinge axis was corrected the same date to local Z, the
# mounting-face normal, which is the physically standard direction for a
# shaft housing bolted to a flat plate. At THIS pose attachment_site's
# local Z lands at world [-0.6976, 0.7164, -0.0047]:
#   - 89.73 deg from VERTICAL -> gravity produces essentially full torque
#     about the hinge (sin = 0.99999); peak |qfrc_bias| = 0.02787 Nm, i.e.
#     2.79x this joint's own 0.01 Nm Coulomb frictionloss, so it is not
#     stiction-locked;
#   - 45.76 deg from world X -> |Zhat x xhat| = 0.7165, real (not zero)
#     coupling from the arm's X transport motion into hinge torque, though
#     WEAKER than the 1.000 the old pose+axis pair had. That is a genuine
#     cost of this pose, quantified in the swing-up results, not papered
#     over.
# The two changes are ONE decision: at this pose the OLD axis (local X) is
# 7.29 deg from vertical, i.e. unusable, and at the old pose local Z was
# measured unusable. Neither the axis nor the pose is valid alone.
#
# SEPARATE, DO NOT CONFLATE: this pose sits essentially on the classic UR
# wrist_2=0 singularity (wrist_2 = 0.0047 rad; cond(J6) = 1396 at the pose
# itself, vs 6.93 at the old pose). That is an ARM-side Jacobian-
# conditioning problem -- it limits how accurately the arm can execute the
# kick, and it is tracked explicitly by the cond(J) instrumentation in the
# trial loop below and in the render scripts. It has no effect on the
# pendulum's own gravity dynamics, which depend only on the hinge axis's
# orientation in world.
# Single source of truth is now simulation/ur5e_pendulum_compose.py's
# asset<->pose table (see PENDULUM_ASSET_ARM_Q there), so that pointing any of
# these scripts at the ALTERNATE long-rod asset also moves the pose it is only
# valid at. The value is unchanged; the name is kept because the render scripts
# and tests import it.
ARM_Q0 = DEFAULT_ARM_Q
RATE_HZ = 500.0
CONTROL_DT = 1.0 / RATE_HZ

# Absolute arm-side singularity ceiling on cond(J6), the SAME value
# render_energy_swingup_flip.py / render_annotated_swingup_flip.py already
# use. Deliberately NOT relaxed for the 2026-08-12 ARM_Q0 change even though
# that pose starts at cond(J6) = 1396, i.e. already above it: relaxing a
# singularity check to make a chosen pose look acceptable is exactly the
# failure mode this constant exists to prevent. The honest consequence is
# that every trial at this pose is reported untrustworthy on the arm-side
# criterion, and that is a real property of the pose, not a bookkeeping
# artifact -- read it alongside cond_j_growth_ratio below, which separates
# "this pose is poorly conditioned to begin with" from "this trial drove the
# arm into a WORSE singularity than it started in" (the specific failure the
# threshold was originally added to catch).
SINGULARITY_COND_THRESHOLD = 1000.0

# ---------------------------------------------------------------------------
# Pendulum physical constants -- DERIVED, never hand-cached (2026-08-13).
#
# These used to be four literals (M_TOTAL_KG, R_COM_M, I_PIVOT_KGM2, E_TOP)
# measured off the 0.12 m asset at the ARM_Q0 below. That was silently wrong
# for any other asset or pose, which blocked using the alternate long-rod
# asset at all: measured, the same literals are off by 7.9x in m*g*r and 2.8x
# in natural period when applied to assets/ur5e_pendulum/
# pendulum_attachment_longrod.xml at its own validated pose.
#
# They are now measured off the compiled model by
# simulation.ur5e_pendulum_compose.derive_pendulum_constants(model, arm_q);
# read that function for the method and for why fitting qfrc_bias is the only
# safe route. The two lessons the deleted comment block existed to record,
# preserved because both were real bugs (see git history around 2026-08-12 for
# the full text):
#   1. The moment arm is NOT the 3-D distance |subtree_com - hinge_site|. Only
#      the component PERPENDICULAR to the hinge axis makes torque, and this
#      apparatus has a large along-axis component ON PURPOSE (the rod is
#      clamped outboard along the axis for wrist clearance). Using the 3-D
#      distance inflated m*g*r by 1.83x, which made E_TOP 1.83x too large and
#      the phase-locked drive's seed period 35% too short.
#   2. Nor is it the perpendicular component alone. Gravity torque also scales
#      with sin(angle between the hinge axis and WORLD VERTICAL), so the
#      constant is a property of (asset, POSE), not of the asset. At one
#      earlier pose those two definitions differed by 5.1x.
# Fitting qfrc_bias(theta) gets both right automatically and cannot go stale.
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=8)
def _constants_for(pendulum_xml: str, arm_q: tuple[float, ...]) -> PendulumConstants:
    return derive_pendulum_constants(
        compose_ur5e_pendulum_model(pendulum_xml=pendulum_xml), np.asarray(arm_q)
    )


def default_constants() -> PendulumConstants:
    """Constants for the DEFAULT (asset, pose) pair, derived from the model."""
    return _constants_for(str(DEFAULT_PENDULUM_XML), tuple(float(v) for v in ARM_Q0))


_LEGACY_CONSTANT_FIELDS = {
    "M_TOTAL_KG": "m_total_kg",
    "R_COM_M": "r_com_m",
    "I_PIVOT_KGM2": "i_pivot_kgm2",
    "G": "g",
    "E_TOP": "e_top_j",
}


def __getattr__(name: str):
    """Legacy module-level constant names, now DERIVED from the default model
    on first access (PEP 562) instead of being hand-cached literals.

    They are kept because several render/diagnostic scripts import them by
    name, but they are deliberately no longer values you can edit here: the
    one source of truth is derive_pendulum_constants(model, arm_q). Lazy
    rather than eager so importing this module does not pay for a model
    compile; cached, so repeated access is free.

    Any code that runs against a NON-default asset or pose must take its
    constants from the ``constants`` argument threaded through the trial
    functions below, never from these names -- that is the entire point of
    the 2026-08-13 change."""
    field = _LEGACY_CONSTANT_FIELDS.get(name)
    if field is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(default_constants(), field)


def load_config(config_path: Path = CONFIG_PATH) -> dict:
    with Path(config_path).open() as fp:
        return yaml.safe_load(fp)


def resolve_equilibria(model, arm_q) -> tuple[float, float]:
    """(hanging, inverted) for ``model`` with the arm held at ``arm_q``.

    BUG FIXED 2026-08-13: every call site used to do a bare
    ``mujoco.MjData(model)`` and hand it straight to
    find_hanging_and_inverted_angle. Since find_inverted_angle was corrected
    (same date) to read the arm pose out of the ``data`` it is given rather
    than hardcoding ARM_Q0, that left every caller silently analysing the
    pendulum at the arm's ALL-ZEROS pose instead of the pose the trial then
    actually ran at. Measured on the default asset: hanging/inverted came back
    as (-1.5619, +1.5793) rad from the zeros pose against the true
    (-3.0145, +0.1272) rad at ARM_Q0 -- i.e. the swing-up target angle was off
    by ~1.45 rad and 'hanging' and 'inverted' were very nearly swapped, the
    exact class of error find_hanging_and_inverted_angle's own docstring
    already records being burned by once. Funnelling every caller through this
    one helper is what makes posing the arm impossible to forget."""
    data = mujoco.MjData(model)
    data.qpos[:6] = np.asarray(arm_q, dtype=np.float64).reshape(6)
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)
    pend_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "/pendulum_hinge")
    return find_hanging_and_inverted_angle(model, data, model.jnt_qposadr[pend_jid])


def find_hanging_and_inverted_angle(model, data, pend_qpos_adr: int) -> tuple[float, float]:
    """Fixed 2026-08-12 -- this used to release the pendulum from theta=0
    and call wherever it landed after 5000 steps "hanging," the exact
    "release and watch it settle" approach pendulum_balance_torque_lqr.py's
    own find_inverted_angle docstring already documented as unreliable for
    this system's slow/friction-dominated dynamics. Confirmed as a REAL bug
    here, not just theoretically risky: with the corrected (much lighter,
    more friction-dominated) pendulum, this method's "hanging" (0.177 rad)
    and "inverted" (-2.964 rad) were nearly swapped with the TRUE values
    (rigorously verified: hanging=-2.945 rad, inverted=+0.196 rad -- the
    crude method's "hanging" was within 0.02 rad of the TRUE unstable
    equilibrium). Every swing-up result computed against the old function
    was measuring closeness to the wrong reference -- likely showing the
    pendulum falling from near-inverted to near-hanging under gravity
    (trivial, needs no real control) mislabeled as a successful swing-up.
    Now reuses the same rigorous, twice-independently-validated
    qfrc_bias-zero-crossing + stability-classification method already in
    pendulum_balance_torque_lqr.py's find_inverted_angle, rather than a
    second, weaker implementation of the same idea."""
    inverted = _find_inverted_angle_rigorous(model, data, pend_qpos_adr)
    hanging = float(np.mod(inverted + np.pi + np.pi, 2 * np.pi) - np.pi)
    return hanging, inverted


def measure_pivot_coupling(model, arm_q, hanging_angle: float, drive_axis_world) -> float:
    """Q/a: hinge generalized force per unit pivot acceleration along
    ``drive_axis_world``, evaluated at the HANGING equilibrium.

    Q = -m * n_hat . (r x u), with n_hat the hinge axis, r = com - pivot, both
    in world. This is the coefficient the energy law's SIGN depends on, and it
    is a property of (pose, asset, drive axis) -- not a constant.

    WHY THIS IS MEASURED AND NOT HARDCODED (2026-08-16). The law shipped with
    ``a_energy = -k_e * ...``, a sign taken from ONE measurement at ONE pose in
    ONE direction. Measured across the configurations actually in use:

        w2=-90, realrod, world X   c0 = -0.001993   -k_e PUMPS  (Goal 1's flip)
        ARM_Q0, default, world X   c0 = +0.002035   -k_e DAMPS
        ARM_Q0, default, in-plane  c0 = +0.002841   -k_e DAMPS

    So the hardcoded sign was correct only where it was validated, and every
    ARM_Q0 swing-up ever run was fighting a damper -- the law removing exactly
    the energy it was written to add. Symptom: the pendulum decays to rest while
    the drive saturates its clip, and k_e becomes inert (k_e=0 and k_e=50 give
    identical results, because both end at rest).

    A wrong gain is mistuned; a wrong sign inverts the objective. Hence measured.
    """
    import mujoco as _mj

    data = _mj.MjData(model)
    jid = _mj.mj_name2id(model, _mj.mjtObj.mjOBJ_JOINT, "/pendulum_hinge")
    if jid < 0:
        jid = _mj.mj_name2id(model, _mj.mjtObj.mjOBJ_JOINT, "pendulum_hinge")
    data.qpos[:6] = np.asarray(arm_q, dtype=np.float64).reshape(6)
    data.qpos[model.jnt_qposadr[jid]] = float(hanging_angle)
    data.qvel[:] = 0.0
    _mj.mj_forward(model, data)
    n = data.xmat[model.jnt_bodyid[jid]].reshape(3, 3) @ model.jnt_axis[jid]
    n = n / np.linalg.norm(n)
    pivot = np.asarray(data.xanchor[jid], dtype=np.float64).copy()
    bid = int(model.jnt_bodyid[jid])
    mass, com = 0.0, np.zeros(3)
    for b in range(model.nbody):          # every body outboard of the hinge
        p, hit = b, False
        while p > 0:
            if p == bid:
                hit = True
                break
            p = int(model.body_parentid[p])
        if hit and model.body_mass[b] > 0:
            mass += float(model.body_mass[b])
            com += float(model.body_mass[b]) * np.asarray(data.xipos[b], dtype=np.float64)
    if mass <= 0:
        raise ValueError("no mass outboard of the pendulum hinge -- wrong joint?")
    com /= mass
    u = np.asarray(drive_axis_world, dtype=np.float64)
    u = u / np.linalg.norm(u)
    return float(-mass * float(n @ np.cross(com - pivot, u)))


def run_energy_swingup_trial(
    model,
    k_e: float,
    a_max: float,
    k_pos: float,
    k_vel: float,
    duration_s: float,
    hanging_angle: float,
    inverted_angle: float,
    kick_amplitude_m: float = 0.0,
    kick_duration_s: float = 0.0,
    config_path: Path | None = None,
    controller_kind: str = "impedance",
    arm_q=None,
    constants: PendulumConstants | None = None,
) -> dict:
    """kick_amplitude_m/kick_duration_s (added after the plain energy law
    alone was found to stall): the energy term a_energy = k_e*thetadot*...
    is exactly zero when thetadot=0, so starting from PERFECT rest it can
    only ever grow from whatever numerical residual leaks in -- confirmed
    real coupling exists (a deliberate fast single move reached thetadot=
    0.62 rad/s in 0.08s) but the feedback law alone never escapes near-zero
    thetadot on its own. A brief open-loop half-sine "seed kick" for
    t < kick_duration_s gets thetadot away from zero before the energy law
    takes over -- the standard practical fix for this exact bootstrap
    problem in energy-shaping swing-up.

    config_path=None / controller_kind="impedance" (both defaults) reproduce
    the pre-2026-08-13 behavior exactly: the module's own hardcoded
    CONFIG_PATH and the controller kind this call site hardcoded. Both are
    parameters so the swing-up benchmark can be run against a different
    controller FAMILY (e.g. "x_task_yz_corridor_qp"), which a config alone
    cannot select -- build_initial_state_and_adapter takes the kind
    separately. Mirrors the identical pair on
    pendulum_swingup_multi_kick.run_multi_kick_trial.

    arm_q=None / constants=None (both defaults, added 2026-08-13) reproduce
    the default asset exactly: the module's own ARM_Q0 and the constants
    derived from ``model`` at it. Pass BOTH explicitly when running a
    non-default pendulum asset -- the constants are a property of the
    (asset, pose) PAIR, and using the default asset's numbers against the
    long-rod asset is wrong by 7.9x in m*g*r (measured), which is a silently
    wrong energy target rather than an error."""
    config = load_config() if config_path is None else load_config(config_path)
    arm_q = ARM_Q0 if arm_q is None else np.asarray(arm_q, dtype=np.float64).reshape(6)
    if constants is None:
        constants = derive_pendulum_constants(model, arm_q)
    mgr = constants.mgr_nm
    i_pivot = constants.i_pivot_kgm2
    e_top = constants.e_top_j
    data = mujoco.MjData(model)
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "attachment_site")
    joint_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in [
        "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
        "wrist_1_joint", "wrist_2_joint", "wrist_3_joint",
    ]]
    pend_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "/pendulum_hinge")
    pend_qpos_adr = model.jnt_qposadr[pend_jid]
    pend_dof_adr = model.jnt_dofadr[pend_jid]

    data.qpos[:6] = arm_q
    data.qpos[pend_qpos_adr] = hanging_angle
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)

    state0, adapter = build_initial_state_and_adapter(
        model, data, site_id, joint_ids,
        controller_cfg=config["controller"],
        transport_axis_index=0,
        target_x_delta=0.0,
        controller_kind=str(controller_kind),
        force_hold_current_pose=False,
        gravity_mode=config["mujoco"].get("gravity_mode", "gravity_comp"),
        gravity_source=config["mujoco"].get("gravity_source", "pinocchio"),
        coriolis_feedforward=bool(config["mujoco"].get("coriolis_feedforward", True)),
        torque_limit_scale=1.0,
    )
    # GUARD FRAME. If the controller drives a rotated task axis, the SAFETY
    # MONITOR must measure drift in that same basis or it counts legitimate
    # on-axis travel as lateral drift. Measured at ARM_Q0 2026-08-16: the
    # in-plane drive axis is [0.716 0.698 0], so 0.698 of every metre travelled
    # lands in world Y; against the 0.06 m world-Y guard that caps useful travel
    # at 0.086 m, while the validated flip needed 0.187 m. The symptom is a
    # swing-up search that retreats to a 20x-too-weak k_e (13.58 vs 277.75)
    # because every stronger pump trips |Y-Y0| at ~0.77 s -- the guard, not the
    # physics, choosing the gain.
    #
    # controller_core.safety.validated_task_rotation is the SAME checker the
    # controller's own task_rotation goes through, so the two cannot disagree
    # about what basis they are in. Absent from a config => unchanged behavior.
    # configure_task_frame's own docstring says "Call before reset()" -- reset()
    # is what hands the rotation to safety_monitor.set_initial_position, and
    # build_initial_state_and_adapter already reset internally. So configure and
    # then RE-reset, and ASSERT the monitor actually took it: a first attempt
    # that skipped the re-reset was a silent no-op that reproduced the unfixed
    # numbers to four decimals, which is indistinguishable from "the fix did not
    # help" unless the frame is checked directly.
    # ENERGY-LAW SIGN, measured for THIS (pose, asset, drive axis). See
    # measure_pivot_coupling. sign(K) must equal sign(c0) or the law damps.
    _rot_cfg = (config.get("controller") or {}).get("task_rotation")
    _drive_axis = (np.asarray(_rot_cfg, dtype=np.float64).reshape(3, 3)[:, 0]
                   if _rot_cfg is not None else np.array([1.0, 0.0, 0.0]))
    _c0 = measure_pivot_coupling(model, arm_q, hanging_angle, _drive_axis)
    energy_sign = -1.0 if _c0 < 0.0 else 1.0

    _task_rot = _rot_cfg
    if _task_rot is not None:
        adapter.configure_task_frame(
            task_rotation=np.asarray(_task_rot, dtype=np.float64).reshape(3, 3))
        adapter.reset(state0)
        _applied = getattr(adapter.safety_monitor, "task_rotation", None)
        if _applied is None:
            raise RuntimeError(
                "config sets controller.task_rotation but the safety monitor is still "
                "resolving drift in WORLD axes. Running on would count on-axis travel "
                "as lateral drift and silently cap the search."
            )

    x0 = float(state0.ee_pos[0])

    n_steps = int(duration_s * RATE_HZ)
    target_x = x0
    target_x_vel = 0.0
    min_theta_dist = np.pi
    guard_fired = False
    guard_reason = None
    steps_done = 0
    # cond(J)/excursion tracking -- added 2026-08-12 after a real incident: a
    # manually-tuned (k_e=300, a_max=8) trial reached min_theta_dist=0.0028
    # (near-perfect) but only by overwhelming the recentering term enough to
    # drag the arm 1.12m from x0 into cond(J)=152603 (a real singularity),
    # NOT a genuine achievable flip. The controller's own safety guards
    # (Y/Z-drift, orientation, joint velocity) do not reliably catch this --
    # a Jacobian can be numerically singular without any single one of those
    # thresholds tripping first. Without this check the DE search below could
    # converge on that exact same fake "solution" and report it as real.
    #
    # BUG FIXED 2026-08-12 (later, during the hinge-axis/pose correction):
    # this instrumentation was DEAD CODE -- jacp/jacr/max_cond_j were
    # allocated here and never touched again, nothing was computed inside the
    # loop, nothing was returned, and objective() therefore never saw a
    # conditioning number. The comment above described a protection that did
    # not exist, which is worse than having no comment. It matters much more
    # now than when it was written: the current ARM_Q0 sits essentially ON the
    # wrist_2=0 singularity (cond(J6) = 1396 at the pose itself, vs 6.93 at
    # the previous pose), so "the search found a fake flip by walking into a
    # singularity" is no longer a remote hazard. Now actually computed every
    # step, returned, and penalised in objective().
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    max_cond_j = 0.0
    cond_j_start = None
    max_abs_x_dev = 0.0
    # Floor-strike tracking. See the in-loop comment for why the pole needs
    # explicit tracking while the arm does not.
    # The composed model PREFIXES attachment sites with "/" (mjcf attach), so the
    # bare name resolves to -1. Never index site_xpos with an unchecked result:
    # site_xpos[-1] silently returns the LAST site, which here happens to be the
    # tip site, so a wrong lookup can return plausible numbers instead of failing.
    tip_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "/pendulum_tip_site")
    if tip_site_id < 0:
        tip_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "pendulum_tip_site")
    min_tip_z = float("inf")
    arm_contact_steps = 0

    for step in range(n_steps):
        t = step * CONTROL_DT
        theta = float(data.qpos[pend_qpos_adr])
        thetadot = float(data.qvel[pend_dof_adr])
        phi = float(np.mod(theta - hanging_angle + np.pi, 2 * np.pi) - np.pi)
        E = 0.5 * i_pivot * thetadot * thetadot + mgr * (1.0 - np.cos(phi))

        if t < kick_duration_s and kick_duration_s > 1e-9:
            # Raised-cosine (Hann) position pulse: x0 -> x0+kick_amplitude -> x0.
            # A plain half-sine's VELOCITY is a cosine, which is nonzero (in
            # fact maximal) at both endpoints -- found empirically the hard
            # way (a large, undamped leftover velocity was dominating the
            # entire post-kick trial, exactly explaining the earlier
            # decay-after-kick result, not a genuine energy-pumping
            # failure). This profile's velocity is a sine, genuinely zero
            # at both t=0 and t=kick_duration_s, so the handoff to the
            # feedback law below sees no velocity discontinuity.
            omega_kick = 2.0 * np.pi / kick_duration_s
            target_x = x0 + 0.5 * kick_amplitude_m * (1.0 - np.cos(omega_kick * t))
            target_x_vel = 0.5 * kick_amplitude_m * omega_kick * np.sin(omega_kick * t)
            a_cmd = float(0.5 * kick_amplitude_m * omega_kick * omega_kick * np.cos(omega_kick * t))
        else:
            # SIGN FIXED 2026-08-12 -- this was `+k_e * thetadot * ...`, which
            # makes the law a DAMPER, not a pump. Derivation, then the direct
            # measurement that confirms it:
            #
            # The generalized force a horizontal pivot acceleration a_x exerts
            # on this hinge is Q = n . (r x (-m*a_x*xhat)) = -m*r_perp*cos(phi)*a_x
            # (n = hinge axis, r = com-minus-pivot). Measured directly on the
            # composed model: Q/a_x = -0.002841 kg m at phi=0, and it tracks
            # cos(phi) exactly (-0.002009 at +-45deg, 0.000000 at +-90deg).
            # Power into the pendulum is therefore
            #     P = Q*thetadot = -m*r_perp*cos(phi)*a_x*thetadot,
            # so PUMPING requires a_x*thetadot*cos(phi) < 0 -- the pivot must
            # accelerate AGAINST the bob's motion, the standard quadrature
            # (90deg-lagged) resonance condition, not with it.
            #
            # Confirmed by measurement, not just algebra: driving an open-loop
            # 6cm sinusoid at the true resonance through this same controller
            # path and integrating W = sum(Q*thetadot*dt) over the steady-state
            # window closes the pendulum's own energy budget to within 1%
            # (W=+0.093523 J vs dissipation+dE = +0.094434 J), and the measured
            # steady-state phase of thetadot relative to the achieved pivot
            # acceleration is +163.6deg -- i.e. very nearly ANTIPHASE, exactly
            # as the sign above requires.
            #
            # With the old `+`, k_e>0 gave P<0 on every step, and the search
            # bound for k_e is [1, 400] -- positive only -- so the search could
            # not have recovered the correct sign by flipping k_e either. The
            # bound is left positive and the sign is fixed here instead, so
            # "larger k_e = more pumping" now holds.
            # energy_sign is MEASURED, not assumed: -1.0 reproduces the shipped
            # law exactly wherever the shipped sign was correct (c0 < 0).
            a_energy = energy_sign * k_e * thetadot * np.cos(phi) * (e_top - E)
            a_recenter = -k_pos * (target_x - x0) - k_vel * target_x_vel
            a_cmd = float(np.clip(a_energy + a_recenter, -a_max, a_max))

            target_x_vel += a_cmd * CONTROL_DT
            target_x += target_x_vel * CONTROL_DT

        state = build_mujoco_state(
            model, data, site_id=site_id, joint_ids=joint_ids,
            time_s=t, dt_s=CONTROL_DT,
            target_x=target_x, target_x_vel=target_x_vel, target_x_accel=a_cmd,
            reference_quat=state0.ee_quat, transport_axis_index=0,
            gravity_compensation=True,
        )
        tau, diag = adapter.step(state=state)
        if not bool(diag.get("safety_ok", True)):
            guard_fired = True
            guard_reason = str(diag.get("safety_reason", ""))
            break

        data.ctrl[:6] = np.asarray(tau, dtype=np.float64).reshape(6)
        mujoco.mj_step(model, data)
        steps_done += 1

        mujoco.mj_jacSite(model, data, jacp, jacr, site_id)
        j6 = np.vstack([jacp[:, :6], jacr[:, :6]])
        cond_j = float(np.linalg.cond(j6))
        if cond_j_start is None:
            cond_j_start = cond_j
        max_cond_j = max(max_cond_j, cond_j)
        max_abs_x_dev = max(max_abs_x_dev, abs(float(data.site_xpos[site_id][0]) - x0))

        # FLOOR TRACKING (added 2026-08-14). The pendulum rod/hub/tool geoms are
        # declared contype="0" conaffinity="0", so the pole passes THROUGH the
        # floor with no contact force, no warning and no error -- a swing-up can
        # "succeed" having driven the rod through the table. Measured clearance at
        # ARM_Q0 is only 0.0634 m at the hanging equilibrium, which is also the
        # rod's lowest point, so this is a live hazard rather than a theoretical
        # one. The ARM's own geoms DO collide, so an arm strike is detectable via
        # data.ncon; the pole is not, and must be tracked explicitly.
        if tip_site_id >= 0:
            min_tip_z = min(min_tip_z, float(data.site_xpos[tip_site_id][2]))
        if int(data.ncon) > 0:
            arm_contact_steps += 1

        dist = abs(float(np.mod(theta - inverted_angle + np.pi, 2 * np.pi) - np.pi))
        min_theta_dist = min(min_theta_dist, dist)

    reached_singularity = max_cond_j > SINGULARITY_COND_THRESHOLD
    return {
        "k_e": k_e, "a_max": a_max, "k_pos": k_pos, "k_vel": k_vel,
        "kick_amplitude_m": kick_amplitude_m, "kick_duration_s": kick_duration_s,
        "duration_s": duration_s,
        "min_theta_dist_from_inverted_rad": min_theta_dist,
        "flipped": min_theta_dist < 0.35,
        "guard_fired": guard_fired, "guard_reason": guard_reason,
        "steps_completed": steps_done,
        "max_cond_j": max_cond_j,
        "cond_j_start": cond_j_start,
        "cond_j_growth_ratio": (max_cond_j / cond_j_start) if cond_j_start else None,
        "reached_singularity": reached_singularity,
        "min_tip_z_m": (None if min_tip_z == float("inf") else float(min_tip_z)),
        "tip_hit_floor": bool(min_tip_z != float("inf") and min_tip_z <= 0.0),
        "arm_contact_steps": int(arm_contact_steps),
        # "trustworthy" deliberately means BOTH: a flip that only happened
        # because the arm walked into a singularity is not a flip. Same
        # definition render_energy_swingup_flip.py / render_annotated_swingup_
        # flip.py already use, kept identical so the three agree.
        "trustworthy": bool(min_theta_dist < 0.35) and not reached_singularity,
        "max_abs_x_dev_m": max_abs_x_dev,
    }


# ---------------------------------------------------------------------------
# Shared run context + CLI. Reused verbatim by pendulum_swingup_multi_kick.py
# and pendulum_swingup_phase_locked.py, which already import from this module,
# so the three swing-up strategies cannot drift apart on how an asset/pose is
# selected.
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class PendulumRunContext:
    """Everything a swing-up trial needs that is NOT a searched gain.

    Deliberately a frozen dataclass of plain picklable scalars/tuples (never a
    compiled MjModel): differential_evolution's parallel workers receive it via
    functools.partial, and a model is not picklable. Each worker recomposes its
    own model from ``pendulum_xml``, exactly as the pre-2026-08-13 objective()
    already did.

    ``hanging_angle``/``inverted_angle``/``constants`` are resolved ONCE by
    :meth:`resolve` in the parent process and carried along, rather than being
    recomputed per candidate. They are deterministic functions of
    (asset, pose), so this is a pure speedup -- find_inverted_angle alone runs
    two 2000-step classification simulations."""

    pendulum_xml: str
    arm_q: tuple[float, ...]
    config_path: str
    controller_kind: str = "impedance"
    hanging_angle: float | None = None
    inverted_angle: float | None = None
    constants: PendulumConstants | None = None

    @property
    def arm_q_array(self) -> np.ndarray:
        return np.asarray(self.arm_q, dtype=np.float64)

    def build_model(self):
        return compose_ur5e_pendulum_model(pendulum_xml=self.pendulum_xml)

    def resolve(self) -> "PendulumRunContext":
        """Fill in constants/equilibria by measuring them off a freshly
        composed model at this context's own pose."""
        model = self.build_model()
        constants = derive_pendulum_constants(model, self.arm_q_array)
        hanging, inverted = resolve_equilibria(model, self.arm_q_array)
        return dataclasses.replace(
            self, constants=constants, hanging_angle=hanging, inverted_angle=inverted
        )

    def trial_kwargs(self) -> dict:
        return {
            "hanging_angle": self.hanging_angle,
            "inverted_angle": self.inverted_angle,
            "config_path": Path(self.config_path),
            "controller_kind": self.controller_kind,
            "arm_q": self.arm_q_array,
            "constants": self.constants,
        }


def add_common_pendulum_args(parser: argparse.ArgumentParser, *,
                             default_config: Path = CONFIG_PATH,
                             with_controller_kind: bool = True) -> None:
    parser.add_argument(
        "--pendulum-xml", default=str(DEFAULT_PENDULUM_XML),
        help="Pendulum attachment MJCF to compose onto the arm "
             "(default: the real 0.12 m apparatus).")
    parser.add_argument(
        "--start-q-rad", nargs=6, type=float, default=None,
        metavar=("Q1", "Q2", "Q3", "Q4", "Q5", "Q6"),
        help="Arm start pose, 6 joint angles in radians. Default: the pose "
             "registered for --pendulum-xml in "
             "simulation.ur5e_pendulum_compose.PENDULUM_ASSET_ARM_Q. A hinge "
             "axis is only usable at the pose it was measured at -- do not "
             "override this without re-measuring.")
    if with_controller_kind:
        parser.add_argument(
            "--controller-kind", default="impedance",
            help="Controller FAMILY passed to build_initial_state_and_adapter "
                 "(e.g. impedance, x_task_yz_corridor_qp). A config alone "
                 "cannot select this.")
    parser.add_argument("--config", default=str(default_config),
                        help="Controller YAML config.")
    parser.add_argument("--output-json", default=None,
                        help="Write the run's results to this path as JSON.")


def context_from_args(args) -> PendulumRunContext:
    pendulum_xml = str(Path(args.pendulum_xml).resolve())
    arm_q = (tuple(float(v) for v in args.start_q_rad) if args.start_q_rad is not None
             else tuple(float(v) for v in arm_q_for_pendulum_xml(pendulum_xml)))
    return PendulumRunContext(
        pendulum_xml=pendulum_xml,
        arm_q=arm_q,
        config_path=str(Path(args.config).resolve()),
        controller_kind=getattr(args, "controller_kind", "impedance"),
    )


def describe_context(ctx: PendulumRunContext) -> str:
    c = ctx.constants
    return (f"pendulum_xml={Path(ctx.pendulum_xml).name}  "
            f"arm_q={np.round(ctx.arm_q_array, 6).tolist()}\n"
            f"controller_kind={ctx.controller_kind}  config={Path(ctx.config_path).name}\n"
            f"hanging_angle={ctx.hanging_angle:.4f} rad  inverted_angle={ctx.inverted_angle:.4f} rad\n"
            f"derived: m*g*r={c.mgr_nm:.6f} Nm  I_pivot={c.i_pivot_kgm2:.6e} kg m^2  "
            f"m={c.m_total_kg:.6f} kg  r_eff={c.r_com_m:.6f} m\n"
            f"         omega={c.omega_natural_radps:.4f} rad/s  T_natural={c.t_natural_s:.4f} s  "
            f"E_top={c.e_top_j:.4f} J")


def write_output_json(path, payload) -> None:
    def _plain(o):
        if isinstance(o, dict):
            return {k: _plain(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [_plain(v) for v in o]
        if isinstance(o, np.ndarray):
            return _plain(o.tolist())
        if isinstance(o, PendulumConstants):
            return dataclasses.asdict(o)
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        if isinstance(o, Path):
            return str(o)
        return o
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(_plain(payload), indent=2, sort_keys=True))


def objective(x, ctx: PendulumRunContext | None = None, duration_s: float = 6.0):
    k_e, a_max, k_pos, k_vel, kick_amplitude_m, kick_duration_s = x
    if ctx is None:
        ctx = default_context()
    if ctx.constants is None:
        ctx = ctx.resolve()
    model = ctx.build_model()
    result = run_energy_swingup_trial(model, k_e, a_max, k_pos, k_vel, duration_s=duration_s,
                                       kick_amplitude_m=kick_amplitude_m, kick_duration_s=kick_duration_s,
                                       **ctx.trial_kwargs())
    cost = result["min_theta_dist_from_inverted_rad"]
    if result["guard_fired"]:
        cost += 5.0  # strictly worse than any survived outcome (max survived cost = pi < 5.0)
    # FLOOR STRIKE = HARD REJECT (2026-08-14). Penalised ABOVE the guard penalty
    # because a guard trip is a controller stopping itself, whereas this is the
    # hardware going through the table. The pendulum geoms are contype=0/
    # conaffinity=0, so a rod strike produces NO contact force and NO error --
    # without this term the search is free to "win" by driving the rod through
    # the floor. The arm's own geoms DO collide, so data.ncon catches an arm
    # strike; both are rejected here.
    if result.get("tip_hit_floor"):
        cost += 20.0
    if int(result.get("arm_contact_steps") or 0) > 0:
        cost += 20.0
    # Singularity penalty, added 2026-08-12 when the tracking above was found
    # to be dead code (see run_energy_swingup_trial). Keyed to the GROWTH in
    # cond(J) rather than to SINGULARITY_COND_THRESHOLD directly: at the
    # current ARM_Q0 the absolute threshold is already exceeded at t=0 by the
    # pose itself, so an absolute test adds the same constant to every
    # candidate and cannot discriminate at all. What must be rejected is a
    # candidate that reaches the inverted angle by dragging the arm into a
    # far WORSE conditioned configuration than it started in -- the exact
    # 1.12m / cond=152603 incident the tracking was added for (that one is a
    # 2.2e4x growth; 10x is a wide margin below it and well above the normal
    # few-x wander seen in clean trials). This is an additional rejection on
    # top of the absolute threshold, not a replacement for it.
    growth = result.get("cond_j_growth_ratio")
    if growth is not None and growth > 10.0:
        cost += 5.0
    return cost


def default_context() -> PendulumRunContext:
    """The default (asset, pose, config, controller) tuple -- what running this
    script with no flags means."""
    return PendulumRunContext(
        pendulum_xml=str(DEFAULT_PENDULUM_XML),
        arm_q=tuple(float(v) for v in ARM_Q0),
        config_path=str(CONFIG_PATH),
        controller_kind="impedance",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Energy-shaping (Astrom-Furuta style) swing-up search.")
    add_common_pendulum_args(parser)
    parser.add_argument("--maxiter", type=int, default=30)
    parser.add_argument("--popsize", type=int, default=14)
    parser.add_argument("--a-max-upper", type=float, default=3.0,
                        help="Upper bound on the searched a_max (m/s^2). Default 3.0 "
                             "preserves historical behavior. Raise it when a search "
                             "reports best a_max at the ceiling AND no guard fired -- "
                             "that means the bound, not safety, capped the result.")
    parser.add_argument("--search-backend", default="de", choices=["de", "optuna"],
                        help="de = differential_evolution (unchanged default). optuna = "
                             "TPE, typically a comparable optimum in ~10-20x fewer "
                             "rollouts on this 6-D space. A faster optimizer does NOT fix "
                             "a broken objective: when a result looks insensitive to a "
                             "parameter, SWEEP that parameter -- flat responses are how "
                             "this lane's real bugs were found (k_e=0 == k_e=50).")
    parser.add_argument("--n-trials", type=int, default=None,
                        help="optuna only: evaluation budget. Set EQUAL to the DE count "
                             "to compare optimizer quality rather than budget.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--duration-s", type=float, default=6.0,
                        help="Trial duration used inside the search objective.")
    parser.add_argument("--final-duration-s", type=float, default=10.0,
                        help="Duration of the final re-validation trial.")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    ctx = context_from_args(args).resolve()
    print(describe_context(ctx))

    # a_max's upper bound is CLI-overridable because it has been observed to
    # PIN: a 2026-08-14 search at wrist_2=-90 deg with friction feedforward
    # converged to a_max=2.990 against the 3.0 ceiling while firing NO safety
    # guard, i.e. the bound -- not the robot and not a safety limit -- was what
    # capped energy injection. When a run reports best a_max within ~1% of this
    # ceiling, re-run with a wider --a-max-upper before concluding anything
    # about feasibility.
    bounds = [
        (1.0, 400.0),   # k_e
        (0.3, float(args.a_max_upper)),  # a_max (m/s^2)
        (0.0, 20.0),    # k_pos
        (0.0, 10.0),    # k_vel
        (0.02, 0.15),   # kick_amplitude_m
        (0.1, 0.6),     # kick_duration_s
    ]
    print("=== searching (k_e, a_max, k_pos, k_vel, kick_amplitude_m, kick_duration_s) via differential_evolution ===")
    # workers=_de_workers() (2026-08-12): this file was MISSED when the
    # bounded-worker fix landed in pendulum_swingup_multi_kick.py and
    # pendulum_swingup_phase_locked.py -- it still had workers=-1, which claims
    # every core on this shared host and leaves the scipy>=1.17 forkserver
    # start-method hazard unpinned. Imported lazily inside main() because
    # pendulum_swingup_multi_kick imports FROM this module at import time; a
    # top-level import here would be circular.
    from tools.diagnostics.pendulum_swingup_multi_kick import _de_workers
    from tools.diagnostics.pendulum_search_backends import minimize as _minimize

    res = _minimize(
        functools.partial(objective, ctx=ctx, duration_s=args.duration_s),
        bounds, backend=args.search_backend, maxiter=args.maxiter,
        popsize=args.popsize, seed=args.seed, workers=_de_workers(),
        n_trials=args.n_trials)
    print(f"search backend={res.backend}  evaluations={res.nfev}")
    print(f"Best params: k_e={res.x[0]:.4f}, a_max={res.x[1]:.4f}, k_pos={res.x[2]:.4f}, "
          f"k_vel={res.x[3]:.4f}, kick_amplitude_m={res.x[4]:.4f}, kick_duration_s={res.x[5]:.4f}")
    print(f"Best cost: {res.fun:.4f}")

    model = ctx.build_model()
    best = run_energy_swingup_trial(model, res.x[0], res.x[1], res.x[2], res.x[3],
                                     duration_s=args.final_duration_s,
                                     kick_amplitude_m=res.x[4], kick_duration_s=res.x[5],
                                     **ctx.trial_kwargs())
    print(f"Best candidate, re-validated at {args.final_duration_s}s:", best)
    if args.output_json:
        write_output_json(args.output_json, {
            "context": {"pendulum_xml": ctx.pendulum_xml, "arm_q": list(ctx.arm_q),
                        "config_path": ctx.config_path, "controller_kind": ctx.controller_kind,
                        "hanging_angle": ctx.hanging_angle, "inverted_angle": ctx.inverted_angle,
                        "constants": ctx.constants},
            "best_params": {"k_e": res.x[0], "a_max": res.x[1], "k_pos": res.x[2],
                            "k_vel": res.x[3], "kick_amplitude_m": res.x[4],
                            "kick_duration_s": res.x[5]},
            "best_cost": float(res.fun),
            "best_trial": best,
        })
        print("wrote", args.output_json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
