#!/usr/bin/env python3
"""CURVED (2-D) pivot-pumping swing-up for the pendulum at Goal 1's pose
(AGENTS.md sec 0): ARM_Q0 with wrist_2 moved to -90 deg, the
local-X-hinge 0.12 m asset (assets/ur5e_pendulum/pendulum_attachment_realrod.xml).
Sim-only; wall-clock cost is not a constraint (user directive).

MECHANISM (re-derived and re-verified against the compiled model below, see
``compute_pump_basis`` and the module-level physics re-verification in this
file's own test coverage -- do not trust the numbers in AGENTS.md/the task
brief without re-checking, several of them were off when checked here, see
the "RE-VERIFIED PHYSICS" note below).

For a pivot accelerating with components (a_par, a_z) in a plane spanned by
(pump direction, vertical):
    I*thetaddot = -M*r*[ (g + a_z)*sin(phi) + a_par*cos(phi) ] - b*thetadot
    Edot        = -M*r*thetadot*[ a_par*cos(phi) + a_z*sin(phi) ]
                = -M*r*thetadot*(a . n_hat),   n_hat = (cos phi, sin phi)
phi = theta - hanging_angle (0 at hanging, +-pi at inverted). Vertical drive
is at MAXIMUM authority exactly where horizontal ("par") has NONE (rod
horizontal) and vice versa -- so the energy-maximizing drive direction is
PERPENDICULAR TO THE ROD, which rotates with phi as the pendulum swings.
The generalization of the scalar 1-D law
    a = -k_e*thetadot*cos(phi)*(E_top - E)
to 2-D is therefore a VECTOR law:
    a_vec = -k_e * thetadot * (E_top - E) * n_hat,   n_hat = (cos phi, sin phi)
which reduces to the exact 1-D law along the "par" (pump) component and adds
a genuinely new "vertical" component along sin(phi). This is a Lyapunov
construction, not a heuristic: Edot = -M*r*thetadot*(a_vec . n_hat)
= M*r*k_e*thetadot^2*(E_top-E) >= 0 for k_e>=0, E<=E_top -- energy can only
increase under this law, self-limiting to zero at E=E_top, for ANY
right-handed orthonormal (pump, vertical) basis, not just world X.

WHY A ROTATED-FRAME WRAPPER, NOT A NEW WORLD-AXIS CHOICE.

controller_core/x_axis_cartesian_impedance is structurally axis-index based:
one task axis (native moving target, kp_x/kd_x) plus, optionally
(``second_task_axis_enabled``), one more axis (kp_y/kd_y, target_y). The
THIRD axis is always held fixed at its reset value -- there is no
``target_z`` hook (confirmed by reading controller.py directly: z_des is
unconditionally ``p0[z_axis]`` in both the hold-pose and normal branches).
Measured at Goal 1's exact pose (see ``compute_pump_basis``), the true "par"
(perpendicular-to-hinge, horizontal) direction is a 45-degree-ish WORLD
DIAGONAL (components [0.7016, 0.7126, 0.0003]), not aligned with world X or
Y -- so neither ``transport_axis_index=0`` (X+Y movable, Z held) nor
``=2`` (Z+Y movable, X held) can reach "par + vertical" using WORLD axes as
the two movable directions: X and Z can never be simultaneously movable
(Z is only ever the task axis (axis=2) or the always-held axis; when Z is
the task axis, the second axis is forced to world Y, not X, by
``_axis_roles``).

The fix, reused rather than reinvented: tools/diagnostics/pendulum_toolY_common.py
already built and validated (tests/unit/test_second_task_axis.py,
tools/diagnostics/pendulum_toolY_swingup_search.py) an EXACT rotated-frame
technique for exactly this problem (see that module's own docstring for the
linear-algebra proof: tau = J.T @ wrench is exactly reproduced under an
orthonormal change of basis for translation, singular values and nullspace
row-space are unaffected). It rotates the state handed to the UNMODIFIED
controller so that R's column 0 becomes the task axis and (with
``second_task_axis_enabled: true``) column 1 becomes a second, independently
moving axis. That machinery does not care WHICH world direction each column
is -- the existing helpers (``pumping_rotation_matrix``,
``build_rotated_initial_state_and_adapter``, ``build_step_state_two_axis``)
are reused here verbatim with:
    R = [ par_hat | vert_hat | n_hat ]
i.e. column 0 = the TRUE par direction (full authority, not a world-axis
approximation), column 1 = vertical (also full authority -- it is built to
be exactly perpendicular to the hinge axis and to par_hat, and is measured
to be 0.9999954 aligned with world +Z at this pose, see
``compute_pump_basis``), column 2 = the hinge axis itself (held fixed --
motion along it does not couple into hinge torque at all, so holding it is
free and harmless). Unlike the (X,Y)-vs-(Z,Y) world-axis compromise this
gets BOTH true par and true vertical authority simultaneously, exactly,
with NO controller_core change and no approximation loss -- a strictly
better solution than picking one of the two achievable world-axis pairs.

RE-VERIFIED PHYSICS (2026-08-14, this file's own measurement -- see
tests/mujoco/test_pendulum_swingup_curved.py::test_reverified_physics_matches_module
for the pinned regression of these exact numbers):
    cond(J6)          = 7.2032          (task brief: 7.20, matches)
    hinge tilt         = 0.174 deg off horizontal (task brief: 0.2 deg, matches)
    m*g*r (mgr_nm)     = 0.0278701 Nm    (task brief: 0.02787, matches)
    I_pivot            = 2.37470e-4 kg m^2 (task brief: 2.3747e-4, matches)
    E_top              = 0.055740 J      (task brief: 0.0557, matches)
    T_natural          = 0.57998 s       (task brief: 0.580, matches)
    par_hat authority  = -2.84101e-3 Nm per m/s^2 * cos(phi)  (exactly M*r = mgr/g)
    vert_hat authority = -2.84101e-3 Nm per m/s^2 * sin(phi)  (EQUAL, to 6 sig figs,
                          not the ~0.7% the task brief measured for a different
                          world-axis pair -- expected, since par_hat/vert_hat are
                          constructed to be EXACTLY symmetric about the hinge axis)
    Lambda_uu (task inertia, arm-only 6x6 block) = 5.437 kg along par_hat,
                          3.539 kg along vert_hat (task brief quoted a single
                          "5.376 kg" for "the pump axis" without saying which
                          world axis that was measured on -- these differ
                          because Lambda is direction-dependent; NOT reproduced
                          exactly, reported as measured here instead)
    Fmax (wrist_1-limited) = 332.6 N along par_hat, 246.4 N along vert_hat
                          (task brief: 280.1 N -- same order, not reproduced
                          exactly; likely a torque-headroom-scaled or
                          different-pose figure)
    max accel (torque-limited) = 61.2 m/s^2 along par_hat, 69.6 m/s^2 along
                          vert_hat (task brief: 52.1 m/s^2 -- same order)
    tip site world z at hanging = 0.2077 m (task brief: "0.0634 m" -- a REAL,
                          material disagreement; measured here by a direct
                          37-point theta scan of "/pendulum_tip_site" world Z
                          confirming 0.2077 m at the true energy-minimum
                          hanging angle, floor plane at world Z=0. This file's
                          own trial loop tracks and reports min tip Z every
                          run regardless, per the task's own instruction --
                          it does not rely on this number being right.)

Reuses config/CLI conventions from pendulum_swingup_energy_shaping.py (never
modified in place -- only imported from) and the rotated-frame helpers from
pendulum_toolY_common.py (also never modified). Search via
scipy.optimize.differential_evolution, NOT RL -- see AGENTS.md's four
documented RL gain-scheduling failures (0/20, 0/20, 1/20 vs 100% fixed-gain).
"""

from __future__ import annotations

import argparse
import dataclasses
import functools
import sys
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import differential_evolution

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from simulation.ur5e_pendulum_compose import (  # noqa: E402
    REALROD_PENDULUM_XML,
    PendulumConstants,
    compose_ur5e_pendulum_model,
    derive_pendulum_constants,
)
from controller_core.config_provenance import check_config_pose  # noqa: E402
from tools.diagnostics.pendulum_swingup_energy_shaping import (  # noqa: E402
    PendulumRunContext,
    add_common_pendulum_args,
    load_config,
    resolve_equilibria,
    write_output_json,
)
from tools.diagnostics.pendulum_toolY_common import (  # noqa: E402
    build_rotated_initial_state_and_adapter,
    build_step_state_two_axis,
    hinge_axis_world,
    hinge_ids,
    measure_cart_coupling_nm_per_mps2,
    pumping_rotation_matrix,
)

RATE_HZ = 500.0
CONTROL_DT = 1.0 / RATE_HZ

# Goal 1's pose, verbatim from AGENTS.md sec 0 ("ANSWER: wrist_2 = -90 deg"):
# ARM_Q0 with wrist_2 moved off the wrist singularity. Written as a literal
# (not DEFAULT_ARM_Q with index 4 overwritten) so this file's pose cannot
# silently drift if DEFAULT_ARM_Q is ever edited for an unrelated reason.
GOAL1_ARM_Q = np.array([-2.3688, -2.1801, -1.8838, -0.7962, -1.5707963, 0.0206])

PENDULUM_XML = REALROD_PENDULUM_XML
CONFIG_PATH = REPO_ROOT / "config" / "ur5e_mujoco_torque_osc_tuned_friction_ff_second_task_axis.yaml"
CONTROLLER_KIND = "impedance"

# Same absolute ceiling pendulum_swingup_energy_shaping.py uses -- kept
# identical so "trustworthy" (flip achieved without walking into a worse
# singularity) means the same thing across both swing-up strategies.
SINGULARITY_COND_THRESHOLD = 1000.0

# "flipped" threshold: identical to pendulum_swingup_energy_shaping.py's, so
# min_theta_dist_from_inverted_rad is directly comparable to the 1-axis
# baseline this file is asked to beat.
FLIP_THRESHOLD_RAD = 0.35


# ---------------------------------------------------------------------------
# Pump basis: (par_hat, vert_hat, n_hat), measured off the compiled model.
# ---------------------------------------------------------------------------


def compute_pump_basis(model: mujoco.MjModel, arm_q: np.ndarray, hanging_angle: float
                        ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Exact orthonormal (par_hat, vert_hat, n_hat) at ``arm_q``.

    n_hat: the hinge's own world axis (read off the model, never assumed).
    par_hat = normalize(n_hat x Z_world): EXACTLY perpendicular to the hinge
        axis (the only property that matters for hinge torque), lying in the
        horizontal plane to the precision Z_world is exact (it does not
        depend on the hinge's own small tilt off horizontal at all).
    vert_hat = normalize(n_hat x par_hat): the EXACT third orthonormal
        vector (not a re-use of world Z, which would only be approximately
        orthogonal to n_hat/par_hat given the hinge's ~0.17 deg tilt) --
        measured to align with world +Z to 0.9999954 at Goal 1's pose, i.e.
        the approximation error from using a non-perfectly-horizontal hinge
        is negligible, but this construction has none regardless.

    Sign convention: chosen so the measured hinge-torque coupling at phi=0
    (theta=hanging_angle) along par_hat is NEGATIVE, matching the
    Astrom-Furuta sign convention this file's law derivation assumes
    (Q/a_par = -M*r*cos(phi), see module docstring). If the raw
    (n x Z, n x par) construction gives the opposite sign, BOTH par_hat and
    vert_hat are flipped together -- flipping only one would fix par's sign
    at the cost of breaking vert's (they are coupled: flipping one is a
    reflection of the (par, vert) plane, which flips a_vec's phase by pi
    relative to n_hat; flipping both is a rotation by pi, which preserves
    the sign of a_vec . n_hat identically for every phi). Measured at Goal
    1's pose, this flip is required, and running it turns vert_hat that
    would otherwise point DOWN into one pointing UP (+0.9999954 world Z) --
    i.e. the sign convention needed for a correct pump law and "bias the
    loop upward" (module docstring) are satisfied by the SAME flip here, not
    two independent choices.
    """
    data = mujoco.MjData(model)
    data.qpos[:6] = np.asarray(arm_q, dtype=np.float64).reshape(6)
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)
    pend_jid, hub_bid, _site_id = hinge_ids(model)
    n_hat = hinge_axis_world(model, data, pend_jid, hub_bid)

    z_world = np.array([0.0, 0.0, 1.0])
    par_hat = np.cross(n_hat, z_world)
    par_hat = par_hat / np.linalg.norm(par_hat)
    vert_hat = np.cross(n_hat, par_hat)
    vert_hat = vert_hat / np.linalg.norm(vert_hat)

    q0 = measure_cart_coupling_nm_per_mps2(model, arm_q, hanging_angle, par_hat)
    if q0 > 0.0:
        par_hat = -par_hat
        vert_hat = -vert_hat

    return par_hat, vert_hat, n_hat


# ---------------------------------------------------------------------------
# The 2-D law itself, as a pure function of scalars (no mujoco) -- this is
# what tests/unit/test_pendulum_swingup_curved_law.py exercises directly.
# ---------------------------------------------------------------------------


def curved_pump_accel(thetadot: float, phi: float, energy_j: float, e_top_j: float,
                       k_e: float, energy_sign: float = -1.0) -> tuple[float, float]:
    """a_vec = energy_sign * k_e * thetadot * (E_top - E) * (cos phi, sin phi).

    ``energy_sign`` defaults to -1.0, reproducing the shipped law bit-for-bit.
    It exists because that leading sign is NOT a constant: it encodes
    c0 = Q/a = -m * n_hat . (r x u), a property of (hinge axis, drive
    direction). Measured 2026-08-16, c0 = -0.001993 at Goal 1's pose (so -1.0
    is right there) but +0.002841 at the singular ARM_Q0 -- where this law as
    shipped would DAMP in BOTH components at once, since a_pump and a_vert
    share one cross-product geometry and therefore one sign.

    Returns (a_pump, a_vert), the "energy" component of the commanded
    acceleration BEFORE the position leash and the clip. Pure function of
    scalars -- reused identically by the trial loop and by the unit tests
    that check the Edot >= 0 / cos-vs-sin-scaling properties this docstring
    (and the module docstring above) derive."""
    common = float(energy_sign) * float(k_e) * float(thetadot) * (float(e_top_j) - float(energy_j))
    return common * np.cos(phi), common * np.sin(phi)


def pendulum_edot(thetadot: float, mgr_nm: float, g: float, a_pump: float, a_vert: float,
                   phi: float) -> float:
    """Edot = -M*r*thetadot*(a . n_hat), n_hat=(cos phi, sin phi). Used only
    by the unit tests to check the Lyapunov property of ``curved_pump_accel``
    directly against the module docstring's derivation; not called by the
    trial loop (which gets Edot for free from the simulator)."""
    m_r = float(mgr_nm) / float(g)
    return -m_r * float(thetadot) * (a_pump * np.cos(phi) + a_vert * np.sin(phi))


# ---------------------------------------------------------------------------
# Trial loop.
# ---------------------------------------------------------------------------


def _tip_site_id(model: mujoco.MjModel) -> int:
    for name in ("/pendulum_tip_site", "pendulum_tip_site"):
        sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
        if sid >= 0:
            return sid
    raise ValueError("pendulum_tip_site not found on composed model")


def run_curved_swingup_trial(
    model: mujoco.MjModel,
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
    arm_q: np.ndarray | None = None,
    constants: PendulumConstants | None = None,
    enable_vertical: bool = True,
    controller_kind: str | None = None
) -> dict:
    """CURVED 2-D pivot-pumping swing-up trial.

    k_pos/k_vel are SHARED between the pump and vertical axes (a deliberate
    simplification, not a physics requirement -- see module docstring;
    kept so this shares the exact same 6-parameter search space as
    pendulum_swingup_energy_shaping.py's 1-axis law, for a like-for-like DE
    search budget). The vertical leash is one-sided ("bias the loop
    upward", per the task brief): it only pulls back once the vertical
    reference has dropped BELOW its start value, never resists rising --
    this is exactly the "costs nothing" asymmetry the task brief describes,
    since sin(phi)'s CONTRIBUTION depends on phi's phasing, not on where the
    vertical loop is centered.

    ``enable_vertical=False`` (added after the first search result: DE found
    a candidate whose kick alone -- k_e barely used, verified by a k_e=0
    ablation reproducing the SAME near-zero min_theta_dist -- reached
    near-exact inversion, meaning the win was attributable to the ROTATED
    FRAME'S full-strength par_hat axis, not to the vertical/curved
    mechanism this file exists to test) zeroes a_vert unconditionally
    (energy AND leash terms both), reducing the trial to a genuine
    single-axis pump ALONG par_hat -- the fair ablation for isolating "does
    curving help GIVEN the same full-authority pump axis", as opposed to
    "does using the true par_hat axis (instead of world X's ~70% projection
    of it) help" -- a different, real, but separate effect."""
    from tools.diagnostics.pendulum_swingup_energy_shaping import load_config

    config = load_config(CONFIG_PATH if config_path is None else Path(config_path))
    controller_kind = CONTROLLER_KIND if controller_kind is None else str(controller_kind)
    arm_q = GOAL1_ARM_Q if arm_q is None else np.asarray(arm_q, dtype=np.float64).reshape(6)
    # MEASURED energy sign for THIS (pose, asset, pump direction). a_pump and
    # a_vert come from one cross-product geometry, so a single sign governs
    # both -- get it wrong and the 2-D pump damps in both components at once.
    # -1.0 (the shipped constant) is correct at Goal 1's pose and WRONG at the
    # singular ARM_Q0; see measure_pivot_coupling.
    from tools.diagnostics.pendulum_swingup_energy_shaping import measure_pivot_coupling
    # (energy sign is resolved AFTER the pump basis is built -- see below)
    if constants is None:
        constants = derive_pendulum_constants(model, arm_q)
    mgr = constants.mgr_nm
    i_pivot = constants.i_pivot_kgm2
    e_top = constants.e_top_j

    par_hat, vert_hat, n_hat = compute_pump_basis(model, arm_q, hanging_angle)
    # ENERGY SIGN, measured along par_hat -- the direction this law actually
    # drives. compute_pump_basis ALREADY flips (par_hat, vert_hat) together when
    # the coupling comes out positive, so this is guaranteed to resolve to -1.0
    # and reproduce the shipped law exactly. It is kept as a CHECK, not a fix:
    # if the basis construction ever stops self-correcting, this catches it
    # instead of silently inverting the pump.
    #
    # A previous attempt measured this along WORLD X and got +0.002035 -> +1,
    # which INVERTED a law that was already correct. It cost two full searches:
    # 1.2417 rad (plain OSC) and 1.9661 rad (2-axis), both worse than the
    # single-axis 1.5105 they were meant to beat. The single-axis lane genuinely
    # needed a measured sign; this one never did. Same fix, opposite verdict --
    # which is exactly why the axis it is measured along has to be the one the
    # law drives.
    _c0 = measure_pivot_coupling(model, arm_q, hanging_angle, par_hat)
    energy_sign = -1.0 if _c0 < 0.0 else 1.0
    R = pumping_rotation_matrix(tool_x=vert_hat, tool_y=par_hat, tool_z=n_hat)

    data = mujoco.MjData(model)
    joint_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in [
        "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
        "wrist_1_joint", "wrist_2_joint", "wrist_3_joint",
    ]]
    pend_jid, _hub_bid, site_id = hinge_ids(model)
    pend_qpos_adr = model.jnt_qposadr[pend_jid]
    pend_dof_adr = model.jnt_dofadr[pend_jid]
    tip_site_id = _tip_site_id(model)

    data.qpos[:6] = arm_q
    data.qpos[pend_qpos_adr] = hanging_angle
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)

    state0, adapter = build_rotated_initial_state_and_adapter(
        model, data, site_id, joint_ids,
        R=R,
        controller_cfg=config["controller"],
        gravity_mode=config["mujoco"].get("gravity_mode", "gravity_comp"),
        gravity_source=config["mujoco"].get("gravity_source", "pinocchio"),
        coriolis_feedforward=bool(config["mujoco"].get("coriolis_feedforward", True)),
        torque_limit_scale=1.0,
    )
    pump0 = float(state0.ee_pos[0])
    vert0 = float(state0.ee_pos[1])
    reference_quat = state0.ee_quat

    n_steps = int(duration_s * RATE_HZ)
    s_pump, s_pump_vel = 0.0, 0.0
    s_vert, s_vert_vel = 0.0, 0.0
    min_theta_dist = np.pi
    guard_fired = False
    guard_reason = None
    steps_done = 0
    min_tip_z = float(data.site_xpos[tip_site_id][2])
    floor_struck = False

    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    max_cond_j = 0.0
    cond_j_start = None
    max_abs_pump_dev = 0.0
    max_abs_vert_dev = 0.0

    for step in range(n_steps):
        t = step * CONTROL_DT
        theta = float(data.qpos[pend_qpos_adr])
        thetadot = float(data.qvel[pend_dof_adr])
        phi = float(np.mod(theta - hanging_angle + np.pi, 2 * np.pi) - np.pi)
        E = 0.5 * i_pivot * thetadot * thetadot + mgr * (1.0 - np.cos(phi))

        if t < kick_duration_s and kick_duration_s > 1e-9:
            # Seed kick along the PUMP axis only (matches
            # pendulum_swingup_energy_shaping.py's rationale: the energy law
            # is identically zero at thetadot=0, so it cannot bootstrap
            # itself off perfect rest -- and at phi=0 (hanging) cos(phi)=1,
            # sin(phi)=0, so pump is exactly the axis with authority at the
            # start anyway). Raised-cosine (Hann) position pulse: zero
            # velocity at both endpoints, no handoff discontinuity.
            omega_kick = 2.0 * np.pi / kick_duration_s
            s_pump = 0.5 * kick_amplitude_m * (1.0 - np.cos(omega_kick * t))
            s_pump_vel = 0.5 * kick_amplitude_m * omega_kick * np.sin(omega_kick * t)
            a_pump = float(0.5 * kick_amplitude_m * omega_kick * omega_kick * np.cos(omega_kick * t))
            a_vert = 0.0
        else:
            a_pump_energy, a_vert_energy = curved_pump_accel(
                thetadot, phi, E, e_top, k_e, energy_sign=energy_sign)

            a_pump_leash = -k_pos * s_pump - k_vel * s_pump_vel
            # One-sided vertical leash: only restores when the reference has
            # sagged BELOW its start; free to rise without bound (floor is
            # 6-20 cm below, ceiling is "effectively unlimited up" per the
            # task brief).
            if s_vert >= 0.0:
                a_vert_leash = -k_vel * s_vert_vel
            else:
                a_vert_leash = -k_pos * s_vert - k_vel * s_vert_vel

            a_pump = float(np.clip(a_pump_energy + a_pump_leash, -a_max, a_max))
            a_vert = float(np.clip(a_vert_energy + a_vert_leash, -a_max, a_max)) if enable_vertical else 0.0

            s_pump_vel += a_pump * CONTROL_DT
            s_pump += s_pump_vel * CONTROL_DT
            s_vert_vel += a_vert * CONTROL_DT
            s_vert += s_vert_vel * CONTROL_DT

        state, _ee_pos_world = build_step_state_two_axis(
            model, data, site_id=site_id, joint_ids=joint_ids,
            time_s=t, target_x=pump0 + s_pump, target_x_vel=s_pump_vel, target_x_accel=a_pump,
            target_y=vert0 + s_vert, target_y_vel=s_vert_vel,
            reference_quat=reference_quat, R=R,
        )
        tau, diag = adapter.step(state=state)
        if not bool(diag.get("safety_ok", True)):
            guard_fired = True
            guard_reason = str(diag.get("safety_reason", ""))
            break

        data.ctrl[:6] = np.asarray(tau, dtype=np.float64).reshape(6)
        mujoco.mj_step(model, data)
        steps_done += 1

        tip_z = float(data.site_xpos[tip_site_id][2])
        min_tip_z = min(min_tip_z, tip_z)
        if tip_z < 0.0:
            floor_struck = True

        mujoco.mj_jacSite(model, data, jacp, jacr, site_id)
        j6 = np.vstack([jacp[:, :6], jacr[:, :6]])
        cond_j = float(np.linalg.cond(j6))
        if cond_j_start is None:
            cond_j_start = cond_j
        max_cond_j = max(max_cond_j, cond_j)
        max_abs_pump_dev = max(max_abs_pump_dev, abs(s_pump))
        max_abs_vert_dev = max(max_abs_vert_dev, abs(s_vert))

        dist = abs(float(np.mod(theta - inverted_angle + np.pi, 2 * np.pi) - np.pi))
        min_theta_dist = min(min_theta_dist, dist)

    reached_singularity = max_cond_j > SINGULARITY_COND_THRESHOLD
    return {
        "k_e": k_e, "a_max": a_max, "k_pos": k_pos, "k_vel": k_vel,
        "kick_amplitude_m": kick_amplitude_m, "kick_duration_s": kick_duration_s,
        "duration_s": duration_s,
        "min_theta_dist_from_inverted_rad": min_theta_dist,
        "flipped": bool(min_theta_dist < FLIP_THRESHOLD_RAD),
        "guard_fired": guard_fired, "guard_reason": guard_reason,
        "steps_completed": steps_done,
        "max_cond_j": max_cond_j,
        "cond_j_start": cond_j_start,
        "cond_j_growth_ratio": (max_cond_j / cond_j_start) if cond_j_start else None,
        "reached_singularity": reached_singularity,
        "trustworthy": bool(min_theta_dist < FLIP_THRESHOLD_RAD) and not reached_singularity
                        and not floor_struck,
        "max_abs_pump_dev_m": max_abs_pump_dev,
        "max_abs_vert_dev_m": max_abs_vert_dev,
        "min_tip_world_z_m": min_tip_z,
        "floor_struck": floor_struck,
        "par_hat": par_hat.tolist(), "vert_hat": vert_hat.tolist(), "n_hat": n_hat.tolist(),
    }


# ---------------------------------------------------------------------------
# CLI / search, mirroring pendulum_swingup_energy_shaping.py's own.
# ---------------------------------------------------------------------------


def default_context() -> PendulumRunContext:
    return PendulumRunContext(
        pendulum_xml=str(PENDULUM_XML),
        arm_q=tuple(float(v) for v in GOAL1_ARM_Q),
        config_path=str(CONFIG_PATH),
        # Was a bare `controller_kind`, which exists nowhere in this scope --
        # the same discarded-flag bug fixed in context_from_args below, but
        # here it was a hard NameError that made this entrypoint (and its six
        # tests) fail on import-and-call rather than fail silently.
        controller_kind=CONTROLLER_KIND,
    )


def context_from_args(args) -> PendulumRunContext:
    """Local override of the shared helper (this lane defaults a different
    asset/pose). It must therefore repeat the shared helper's config <-> pose
    check: a script that builds its own context silently opts out of every
    guard the shared path added, which is precisely how a config gets run at
    the wrong pose. See controller_core.config_provenance."""
    pendulum_xml = str(Path(args.pendulum_xml).resolve()) if args.pendulum_xml else str(PENDULUM_XML)
    arm_q = (tuple(float(v) for v in args.start_q_rad) if args.start_q_rad is not None
             else tuple(float(v) for v in GOAL1_ARM_Q))
    config_path = str(Path(args.config).resolve())
    provenance = check_config_pose(
        load_config(Path(config_path)),
        arm_q,
        pendulum_xml,
        controller_kind=(args.controller_kind or CONTROLLER_KIND),
        config_name=Path(config_path).name,
        allow_mismatch=bool(getattr(args, "allow_pose_mismatch", False)),
    )
    return PendulumRunContext(
        pendulum_xml=pendulum_xml,
        arm_q=arm_q,
        config_path=config_path,
        # Read the FLAG. This previously referenced a bare `controller_kind`
        # that exists nowhere in scope, so --controller-kind parsed fine and was
        # then silently discarded -- the run would use plain OSC while the
        # command line said otherwise.
        controller_kind=(args.controller_kind or CONTROLLER_KIND),
        provenance=provenance,
    )


def objective(x, ctx: PendulumRunContext | None = None, duration_s: float = 6.0,
              enable_vertical: bool = True):
    k_e, a_max, k_pos, k_vel, kick_amplitude_m, kick_duration_s = x
    if ctx is None:
        ctx = default_context()
    if ctx.constants is None:
        ctx = ctx.resolve()
    model = ctx.build_model()
    result = run_curved_swingup_trial(
        model, k_e, a_max, k_pos, k_vel, duration_s=duration_s,
        hanging_angle=ctx.hanging_angle, inverted_angle=ctx.inverted_angle,
        kick_amplitude_m=kick_amplitude_m, kick_duration_s=kick_duration_s,
        config_path=Path(ctx.config_path), arm_q=ctx.arm_q_array, constants=ctx.constants,
        enable_vertical=enable_vertical,
    )
    cost = result["min_theta_dist_from_inverted_rad"]
    if result["guard_fired"]:
        cost += 5.0
    if result["floor_struck"]:
        cost += 5.0
    growth = result.get("cond_j_growth_ratio")
    if growth is not None and growth > 10.0:
        cost += 5.0
    return cost


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="CURVED (2-D) pivot-pumping swing-up search, Goal 1 pose.")
    add_common_pendulum_args(parser, default_config=CONFIG_PATH, with_controller_kind=True)
    # controller_kind is now selectable: the CURVED 2-D pump can run through the
    # 2-AXIS controller (x_task_yz_corridor_qp), whose in-plane config already
    # tracks task_axis_rows (0,2) -- the horizontal drive and the vertical, both
    # perpendicular to the hinge, which is exactly the pair curved pumping needs.
    # Plain OSC (the previous hardcoded default) can only drive WORLD X here,
    # wasting 28% of the horizontal component along the hinge at ARM_Q0.
    # add_common_pendulum_args hardcodes --pendulum-xml's default to
    # DEFAULT_PENDULUM_XML (the local-Z-hinge asset) regardless of
    # default_config -- override it here so an unqualified run uses Goal 1's
    # asset (REALROD_PENDULUM_XML) rather than silently falling back to the
    # wrong hinge geometry for this pose.
    parser.set_defaults(pendulum_xml=str(PENDULUM_XML))
    parser.add_argument("--maxiter", type=int, default=30)
    parser.add_argument("--popsize", type=int, default=14)
    parser.add_argument("--a-max-upper", type=float, default=15.0,
                        help="Upper bound on the searched a_max (m/s^2), applied to BOTH "
                             "pump and vertical components. Default 15.0 -- well above "
                             "pendulum_swingup_energy_shaping.py's historical 3.0 ceiling "
                             "(per the task brief's own instruction), still well under the "
                             "~61-70 m/s^2 torque-limited ceiling measured in this file's "
                             "module docstring.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--duration-s", type=float, default=6.0)
    parser.add_argument("--final-duration-s", type=float, default=10.0)
    parser.add_argument("--workers", type=int, default=None,
                        help="differential_evolution worker count. Default: min(8, 90 pct of "
                             "cores) -- see this file's own _de_workers().")
    parser.add_argument("--kick-amplitude-upper", type=float, default=0.15,
                        help="Upper bound on the searched kick_amplitude_m. Default 0.15 "
                             "matches pendulum_swingup_energy_shaping.py's own bound. Lower "
                             "this (e.g. 0.03-0.05) to force the SWING-UP LAW to do the work "
                             "instead of a large, resonantly-tuned single kick -- a real "
                             "confound found in this file's own dev process, see "
                             "run_curved_swingup_trial's enable_vertical docstring.")
    parser.add_argument("--search-backend", default="de", choices=["de", "optuna"],
                        help="de = differential_evolution (unchanged default). optuna = TPE.")
    parser.add_argument("--n-trials", type=int, default=None,
                        help="optuna only: evaluation budget.")
    parser.add_argument("--disable-vertical", action="store_true",
                        help="Ablation: force a_vert=0 always, reducing the trial to a "
                             "genuine single-axis pump along the TRUE par_hat direction "
                             "(full geometric authority, unlike world-X's ~70%% projection "
                             "of it at this pose). Isolates \"does curving help\" from "
                             "\"does using the true pump axis help\" -- see "
                             "run_curved_swingup_trial's own docstring.")
    return parser


def _de_workers(requested: int | None) -> int:
    import multiprocessing as mp
    import os
    try:
        mp.set_start_method("fork")
    except RuntimeError:
        pass
    if requested is not None:
        return max(1, int(requested))
    env = os.environ.get("DE_WORKERS")
    if env:
        return max(1, int(env))
    return max(1, min(8, int((os.cpu_count() or 2) * 0.9)))


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    ctx = context_from_args(args).resolve()
    print(f"pendulum_xml={Path(ctx.pendulum_xml).name}  arm_q={np.round(ctx.arm_q_array, 6).tolist()}")
    print(f"config={Path(ctx.config_path).name}")
    print(f"hanging_angle={ctx.hanging_angle:.4f} rad  inverted_angle={ctx.inverted_angle:.4f} rad")
    c = ctx.constants
    print(f"derived: m*g*r={c.mgr_nm:.6f} Nm  I_pivot={c.i_pivot_kgm2:.6e} kg m^2  "
          f"T_natural={c.t_natural_s:.4f} s  E_top={c.e_top_j:.4f} J")

    bounds = [
        (1.0, 400.0),   # k_e
        (0.3, float(args.a_max_upper)),  # a_max (m/s^2), shared pump/vert clip
        (0.0, 20.0),    # k_pos
        (0.0, 10.0),    # k_vel
        (0.02, float(args.kick_amplitude_upper)),   # kick_amplitude_m
        (0.1, 0.6),     # kick_duration_s
    ]
    workers = _de_workers(args.workers)
    enable_vertical = not args.disable_vertical
    print(f"=== searching (k_e, a_max, k_pos, k_vel, kick_amplitude_m, kick_duration_s) "
          f"via differential_evolution, workers={workers}, enable_vertical={enable_vertical} ===")
    from tools.diagnostics.pendulum_search_backends import minimize as _minimize

    res = _minimize(
        functools.partial(objective, ctx=ctx, duration_s=args.duration_s, enable_vertical=enable_vertical),
        bounds, backend=args.search_backend, maxiter=args.maxiter,
        popsize=args.popsize, seed=args.seed, workers=workers, n_trials=args.n_trials)
    print(f"search backend={res.backend}  evaluations={res.nfev}")
    print(f"Best params: k_e={res.x[0]:.4f}, a_max={res.x[1]:.4f}, k_pos={res.x[2]:.4f}, "
          f"k_vel={res.x[3]:.4f}, kick_amplitude_m={res.x[4]:.4f}, kick_duration_s={res.x[5]:.4f}")
    print(f"Best cost: {res.fun:.4f}")

    model = ctx.build_model()
    best = run_curved_swingup_trial(
        model, res.x[0], res.x[1], res.x[2], res.x[3],
        duration_s=args.final_duration_s,
        hanging_angle=ctx.hanging_angle, inverted_angle=ctx.inverted_angle,
        kick_amplitude_m=res.x[4], kick_duration_s=res.x[5],
        config_path=Path(ctx.config_path), arm_q=ctx.arm_q_array, constants=ctx.constants,
        enable_vertical=enable_vertical,
    )
    print(f"Best candidate, re-validated at {args.final_duration_s}s:", best)
    if args.output_json:
        write_output_json(args.output_json, {
            "context": {"pendulum_xml": ctx.pendulum_xml, "arm_q": list(ctx.arm_q),
                        "config_path": ctx.config_path,
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
    sys.exit(main())
