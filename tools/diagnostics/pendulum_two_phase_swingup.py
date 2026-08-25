#!/usr/bin/env python3
"""Energy-scheduled swing-up: slow stable kicks that sharpen continuously into
short sharp kicks as measured energy rises, arriving inside the LQR capture band.

WHY NOT THE EXISTING LAW. Energy shaping self-limits by construction -- its
drive carries a factor (E_top - E), so it goes to zero exactly as the pendulum
reaches the top. That is why it PARKS the pole there with thetadot ~ 0. The
117-cell capture envelopes then showed that arriving at rest is the WRONG
arrival: the band is not a ball around the origin but an anti-diagonal band in
the unstable mode

    s = thetadot + omega*phi          omega = sqrt(m g r / I) = 10.8334 rad/s

and the celebrated (phi=0.1875, thetadot=0) arrival scores |s| = 2.03 -- 1.7x
OVER the |s| <= 1.2 threshold. Capture needs phi and thetadot of OPPOSITE sign,
i.e. the pole still moving toward vertical. A self-limiting law structurally
cannot deliver that; a drive that gets SHARPER near the top can, because its
amplitude and sharpness there set the arrival velocity.

WHY ENERGY IS THE SCHEDULING VARIABLE, AND WHY THERE IS NO SWITCH. An earlier
draft of this file used a discrete pump -> push switch on an energy threshold.
That is the wrong shape twice over: it injects a step into a plant whose
closed-loop bandwidth (~0.5 s) is already the binding limitation, and it makes
the interesting decision a cliff that a search can only find by luck. Here the
drive is a CONTINUOUS function of the measured energy fraction e = E/E_top:

    blend(e) = 0.5 * (1 + tanh((e - e_center) / e_width))       in [0, 1]
    amplitude = a_slow  + (a_sharp  - a_slow ) * blend(e)
    deadband  = db_slow + (db_sharp - db_slow) * blend(e)

Low energy -> small amplitude and a LARGE tanh deadband, i.e. gentle,
well-damped, near-sinusoidal kicks that build amplitude without stressing the
drift guards. High energy -> large amplitude and a SMALL deadband, i.e. the
drive approaches bang-bang and becomes a short sharp kick placed exactly where
it does the most work. Nothing switches; the character of the motion slides.

THE PUMP IS PHASE-LOCKED BY CONSTRUCTION, NOT FREQUENCY-TUNED. Since

    Edot = c0 * cos(phi) * a * thetadot

the drive that maximises Edot pointwise subject to |a| <= amplitude is
a = amplitude * sign(c0 * cos(phi) * thetadot). That needs no frequency
parameter and automatically tracks the amplitude detuning which dropped the
fixed-frequency sweep's positive-work fraction from 90.9% to 70.5% as the swing
grew (pump_frequency_sweep.py, this session). The tanh deadband above is
exactly the smoothing of that sign, so "sharpness" and "phase-locking" are the
same knob rather than two.

WHY A SLOW PHASE AT ALL. Measured this session at wrist_2=-90: resonant pumping
at the pendulum's own 1.724 Hz delivered 115x the energy of the 1 Hz drive every
prior run used, for 1.44x the guard budget -- 82x better energy per unit guard
budget, no guard fired. Energy was never the shortage (AGENTS.md: available kick
energy at 1.023 m/s is ~1.5x E_top); PHASING and DISPLACEMENT were.

``c0`` IS MEASURED, NEVER ASSUMED. It is the coefficient the drive's SIGN
depends on, it is a property of (pose, asset, drive axis), and a sign taken from
one pose was already responsible for a law that provably could not remove
energy, removing it. See measure_pivot_coupling.

THE SCHEDULE IS A SWAPPABLE POLICY. ``EnergyBlendPolicy.blend`` maps observable
state -> a scalar in [0, 1]. An RL policy belongs in exactly this slot: a
continuous, state-dependent schedule is a real sequential-decision problem,
unlike using RL to pick static gains, which is a bandit and strictly worse than
the DE/optuna backends this file already uses (and which failed 0/20, 0/20, 1/20
here over ~4.4M steps). A continuous scalar action is also a far better-behaved
RL target than a binary switch. Any such policy must be judged against
``positive_work_fraction``, ~1.0 for the analytic law: without it, a policy that
pumps backwards is indistinguishable from one that has not learned yet.
"""

from __future__ import annotations

import argparse
import functools
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from controller_core.config_provenance import check_config_pose, describe_provenance  # noqa: E402
from simulation.ur5e_pendulum_compose import (  # noqa: E402
    compose_ur5e_pendulum_model,
    derive_pendulum_constants,
)
from simulation.ur5e_mujoco_torque import (  # noqa: E402
    build_initial_state_and_adapter,
    build_mujoco_state,
)
from tools.diagnostics.pendulum_energy_monitor import EnergyMonitor  # noqa: E402
from tools.diagnostics.pendulum_swingup_energy_shaping import (  # noqa: E402
    CONTROL_DT,
    RATE_HZ,
    load_config,
    measure_pivot_coupling,
    resolve_equilibria,
)
from tools.diagnostics.pendulum_lqr_cascade import wrap_pi  # noqa: E402

JOINT_NAMES = (
    "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
    "wrist_1_joint", "wrist_2_joint", "wrist_3_joint",
)
V_MAX_MPS = 1.00
AXIS_NAMES = ("world_X", "world_Y", "world_Z")

# DIRECT FLOOR GUARD. The pendulum geoms are contype=0/conaffinity=0, so the rod
# passes THROUGH the floor (world z = 0) with no contact force, no penetration
# warning, and no error -- floor penetration is SILENT. Worse, the shared
# ImpedanceSafetyMonitor's per-axis max_abs_z_drift_m is NOT consulted on the
# task_rotation path (controller_core/safety.py::check only applies the single
# max_abs_orthogonal_drift_m there), so a measured 4.9 cm downward EE drift --
# tip to 0.0144 m, 1.4 cm off the floor -- did NOT trip any guard. This is a
# positive guard on the ACTUAL hazard (tip world z), local to the sim rollout
# because the tip site needs mujoco and cannot live in the numpy-only monitor.
# 0.03 m matches the config's intended max_abs_z_drift_m=0.035 (hanging tip
# 0.063 - 0.035 = 0.028) rounded to a clean 3 cm hard floor margin.
FLOOR_MARGIN_M = 0.03


@dataclass(frozen=True)
class EnergyScheduleParams:
    """The drive's shape as a function of measured energy fraction."""

    a_slow: float           # kick amplitude at low energy, m/s^2
    a_sharp: float          # kick amplitude at high energy, m/s^2
    e_center: float         # energy fraction at which the schedule is halfway
    e_width: float          # how gradually it sharpens (in energy fraction)
    # Deadbands are SEARCHED, not hand-picked -- the defaults below are only a
    # sane starting point. Measured at (a_slow=2, a_sharp=8), 12 s, Goal-1 pose:
    #   db_slow=0.40 -> E_peak/E_top 0.0089, positive_work 0.380  (pumps BACKWARDS)
    #   db_slow=0.10 -> E_peak/E_top 0.2413, positive_work 0.904
    #   db_slow=0.02 -> E_peak/E_top 0.2501, positive_work 0.971
    # i.e. 28x the energy across a range that all "looked reasonable". A large
    # deadband cannot bootstrap: at small swing the normalised drive argument is
    # itself small, so tanh(arg/db) collapses and the leash sets the sign.
    db_slow: float = 0.02   # tanh deadband at low energy
    db_sharp: float = 0.01  # tanh deadband at high energy -> short, sharp
    # SEED KICK -- MANDATORY, not optional. The phase-locked drive is
    # proportional to tanh(c0*cos(phi)*thetadot/db), which is IDENTICALLY ZERO
    # at the hanging rest state (thetadot = 0). Without a seed the rollout is a
    # silent no-op: zero drive, zero energy, pole never leaves hanging, and no
    # error anywhere. AGENTS.md states this for the energy-shaping law ("a seed
    # kick is mandatory: at rest thetadot = 0 makes term 1 identically zero")
    # and it applies verbatim here. Caught by the first smoke run, which
    # returned max_blend = 0 and E_peak/E_top = 0.
    a_seed: float = 3.0     # open-loop seed acceleration, m/s^2
    t_seed_s: float = 0.25  # how long the seed runs before the law takes over
    # SELF-LIMITING TARGET. The blend above makes the drive STRONGER as energy
    # rises, which is the opposite of self-limiting: it drives hardest exactly
    # when the pole already has enough, and measured E_peak/E_top hit 2.54 --
    # the pole blew through the switch window at ~27 rad/s and the LQR could
    # never engage. Classic energy shaping avoids this with an (E_top - E)
    # factor; that factor is what makes the drive vanish at the top. Restoring
    # it as a ramp to `e_target` keeps the slow->sharp character while capping
    # the energy. e_target must sit only slightly above 1.0: arriving NEAR the
    # top SLOWLY requires E ~ E_top almost exactly (at e_target = 1.02 the pole
    # still crosses the top at ~3 rad/s; at 1.002, ~1 rad/s).
    e_target: float = 1.005   # stop pumping at this fraction of E_top
    e_ramp: float = 0.05      # width of the ramp-down approaching e_target
    # Leash gains carried from the validated Goal-1 flip. They belong to that
    # (controller, pose, role); re-derive if either changes -- see AGENTS.md 7.
    k_pos: float = 15.797
    k_vel: float = 8.180


class EnergyBlendPolicy:
    """Maps observable state -> blend in [0, 1]. Swap for a learned policy."""

    def __init__(self, e_center: float, e_width: float):
        self.e_center = float(e_center)
        self.e_width = max(float(e_width), 1e-6)

    def blend(self, *, energy_frac: float, phi_from_hanging: float,
              thetadot: float, t: float) -> float:
        z = (float(energy_frac) - self.e_center) / self.e_width
        return float(0.5 * (1.0 + np.tanh(z)))


def run_energy_scheduled_trial(
    model,
    params: EnergyScheduleParams,
    *,
    arm_q,
    hanging_angle: float,
    inverted_angle: float,
    constants,
    coupling_c0: float,
    config_path: Path,
    controller_kind: str,
    transport_axis_index: int = 0,
    duration_s: float = 12.0,
    s_capture: float = 1.2,
    # Tightened from 0.6: the LQR captures at phi = 5-10 deg and loses at 20
    # deg, so an 'arrival' recorded at 34 deg was never a catchable state.
    arrival_window_rad: float = 0.30,
    stop_at_arrival: bool = True,
    policy: EnergyBlendPolicy | None = None,
    track_history: bool = False,
    velocity_swingup: bool = False,
    lqr_K=None,
    lqr_a_max: float = 3.0,
    s_switch: float = 1.2,
    phi_switch_max_rad: float = 0.45,
    hold_s: float = 4.0,
    velocity_hold: bool = False,
    blend_lqr: bool = False,
    # Blend-weight shape (used only when blend_lqr=True). alpha = a_s * a_d, each
    # a tanh gate that is ~1 when its coordinate is small and decays to ~0 past
    # its center. Defaults sit at the discrete switch thresholds (s_switch=1.2,
    # phi_switch_max=0.45) so the unified law starts life close to the validated
    # switch, then can be tuned. The hard cutoff zeroes the LQR contribution well
    # before the pole is far from the top, as cheap insurance against the linear
    # law's bounded-but-wrong extrapolation leaking into the pump.
    blend_s_center: float = 1.2,
    blend_s_width: float = 0.5,
    blend_dist_center: float = 0.45,
    blend_dist_width: float = 0.20,
    blend_dist_cutoff: float = 0.9,
) -> dict:
    """One continuous rollout, guards ON, no discrete phase change anywhere.

    Reports the arrival state in the coordinate the LQR capture band is actually
    defined in (``s = thetadot + omega*phi``, phi from INVERTED), not just
    ``min_theta_dist_from_inverted`` -- that metric rewards a fast fly-through,
    which is exactly the arrival a catch cannot use.
    """
    config = load_config(config_path)
    arm_q = np.asarray(arm_q, dtype=np.float64).reshape(6)
    omega = float(constants.omega_natural_radps)
    e_top = float(constants.e_top_j)

    data = mujoco.MjData(model)
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "attachment_site")
    joint_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in JOINT_NAMES]
    pend_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "/pendulum_hinge")
    pend_qpos = model.jnt_qposadr[pend_jid]
    pend_dof = model.jnt_dofadr[pend_jid]
    tip_site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "/pendulum_tip_site")
    if tip_site < 0:
        tip_site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "pendulum_tip_site")

    data.qpos[:6] = arm_q
    data.qpos[pend_qpos] = hanging_angle
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)

    state0, adapter = build_initial_state_and_adapter(
        model, data, site_id, joint_ids,
        controller_cfg=config["controller"],
        transport_axis_index=transport_axis_index,
        target_x_delta=0.0,
        controller_kind=str(controller_kind),
        force_hold_current_pose=False,
        gravity_mode=config["mujoco"].get("gravity_mode", "gravity_comp"),
        gravity_source=config["mujoco"].get("gravity_source", "pinocchio"),
        coriolis_feedforward=bool(config["mujoco"].get("coriolis_feedforward", True)),
        torque_limit_scale=1.0,
    )
    # PHASE 1 = velocity tracking on the drive row. Same controller instance,
    # same QP, same CBF rows -- only the position term of that one row is
    # dropped, so the pump is fed the FIRST integral of the drive instead of the
    # second and carries one integrator less of lag. Switched off again at the
    # LQR handoff below, which keeps the seam a source switch.
    _inner = getattr(adapter, "controller", None)
    if velocity_swingup:
        if not hasattr(_inner, "task_velocity_rows"):
            raise RuntimeError(
                f"--velocity-swingup needs a controller with task_velocity_rows; "
                f"{type(_inner).__name__} has none. Use controller_kind="
                f"x_task_yz_corridor_qp.")
        _inner.task_velocity_rows = (int(transport_axis_index),)

    # The unified blend never resets task_velocity_rows, so the drive row stays
    # VELOCITY-tracked through the catch -- the same plant the pole-weighted LQR
    # was validated against. Position-hold has no continuous analog (the discrete
    # path re-syncs target_x at the switch instant, which a switch-free law has no
    # instant to do), so refuse it rather than silently run the LQR against a
    # different plant than its K was solved for.
    if blend_lqr and velocity_swingup and not velocity_hold:
        raise RuntimeError(
            "blend_lqr requires velocity_hold=True with velocity_swingup: the "
            "switch-free blend keeps the drive row velocity-tracked throughout, "
            "so there is no seam at which to re-sync a position-tracked catch.")

    x_ref = float(state0.ee_pos[transport_axis_index])
    policy = policy or EnergyBlendPolicy(params.e_center, params.e_width)

    mon = EnergyMonitor(
        i_pivot_kgm2=float(constants.i_pivot_kgm2), mgr_nm=float(constants.mgr_nm),
        e_top_j=e_top, coupling_c0=float(coupling_c0),
        hinge_damping=float(model.dof_damping[pend_dof]),
    )

    target_x = x_ref
    target_x_vel = 0.0
    state_prev_ee = None
    state_prev_vel = None
    guard_fired = False
    guard_reason = None
    guard_t = None
    best_abs_s = np.inf
    best_arrival = None
    min_dist_inverted = np.inf
    min_tip_z = np.inf
    max_blend = 0.0
    max_alpha = 0.0  # peak LQR blend weight (blend_lqr path only)
    t_blend_half = None
    prev_theta = None
    theta_travel = 0.0
    first_arrival = None
    first_arrival_abs_s = np.inf
    n_window_entries = 0
    in_window = False
    K = None if lqr_K is None else np.asarray(lqr_K, dtype=np.float64).reshape(1, 4)
    lqr_engaged_t = None
    lqr_switch_state = None
    held_to_end = None
    max_abs_phi_after_switch = 0.0
    fell_after_switch_t = None  # first t after handoff where |phi_inv| > 0.5 rad
    last_hold_t = None          # last t spent in the LQR hold branch
    peak_abs_a_cmd = 0.0
    history = [] if track_history else None

    for step in range(int(duration_s * RATE_HZ)):
        t = step * CONTROL_DT
        theta = float(data.qpos[pend_qpos])
        thetadot = float(data.qvel[pend_dof])
        phi_hang = wrap_pi(theta - hanging_angle)
        phi_inv = wrap_pi(theta - inverted_angle)
        energy = mon.energy(thetadot, phi_hang)
        e_frac = energy / e_top if e_top else 0.0

        s_val = thetadot + omega * phi_inv
        dist = abs(phi_inv)
        min_dist_inverted = min(min_dist_inverted, dist)

        # Unwrapped angular travel, so a SPINNING pendulum is visible as a
        # number instead of hiding behind a good-looking |s|.
        if prev_theta is not None:
            theta_travel += abs(wrap_pi(theta - prev_theta))
        prev_theta = theta

        # FIRST arrival, not the best of all approaches. `best_abs_s` used to be
        # min|s| over EVERY pass through the window, which is not a measure of
        # arrival quality at all -- it is a measure of how many chances the
        # trajectory bought. Measured on the search's own winner: 49.16 full
        # rotations, 49 separate passes through this window, |s| at successive
        # pass entries 1.66, 13.4, 16.7, 18.6, 19.8, 20.4 ... and a reported
        # "best" of 0.057 drawn from that lottery. The optimiser was doing
        # exactly what it was asked: maximise the number of tickets.
        if dist < arrival_window_rad:
            if not in_window:
                n_window_entries += 1
            if first_arrival is None:
                first_arrival = {"t": t, "phi_inv_rad": phi_inv,
                                 "thetadot": thetadot, "s": s_val,
                                 "energy_over_e_top": e_frac,
                                 "rotations_before_arrival": theta_travel / (2.0 * np.pi)}
                first_arrival_abs_s = abs(s_val)
                # STOP PUMPING AT ARRIVAL when there is no controller to hand
                # off to. Continuing to drive at full amplitude past the top is
                # not something any real run would do -- the LQR would already
                # have taken over -- and it produced the 49-rotation trajectory
                # that made this law look like a propeller. The rotations were
                # an artifact of evaluating swing-up in isolation for a fixed
                # duration, entirely AFTER the arrival being scored.
                if K is None and stop_at_arrival:
                    break
            in_window = True
        else:
            in_window = False
        if dist < arrival_window_rad and abs(s_val) < best_abs_s:
            best_abs_s = abs(s_val)
            best_arrival = {"t": t, "phi_inv_rad": phi_inv,
                            "thetadot": thetadot, "s": s_val,
                            "energy_over_e_top": e_frac}

        blend = policy.blend(energy_frac=e_frac, phi_from_hanging=phi_hang,
                             thetadot=thetadot, t=t)
        max_blend = max(max_blend, blend)
        if t_blend_half is None and blend >= 0.5:
            t_blend_half = t

        amplitude = params.a_slow + (params.a_sharp - params.a_slow) * blend
        # Self-limiting factor: 1 well below e_target, ramping to 0 at it.
        limiter = float(np.clip((params.e_target - e_frac)
                                / max(params.e_ramp, 1e-9), 0.0, 1.0))
        amplitude *= limiter
        deadband = params.db_slow + (params.db_sharp - params.db_slow) * blend

        # Maximum-rate energy injection subject to |a| <= amplitude: the sign of
        # c0*cos(phi)*thetadot, smoothed by `deadband`. Small deadband -> the
        # drive approaches bang-bang, i.e. a short sharp kick.
        # NORMALISED drive argument. The raw product c0*cos(phi)*thetadot has
        # magnitude ~|c0|*thetadot ~ 1e-3 (c0 = -0.002 Nm per m/s^2 here), so a
        # deadband expressed as 0.02-0.40 -- which reads like a sensible
        # fraction -- put tanh's argument at ~0.005 and attenuated the pump by
        # ~200x. The leash then set the sign of the commanded acceleration and
        # the drive REMOVED energy: a DE search over 240 candidates returned
        # positive_work_fraction = 0.1517 for its BEST result, with every
        # parameter setting in the family equally broken. Dividing by |c0| and
        # by omega makes the argument O(1) for thetadot ~ omega, so `deadband`
        # is a fraction of the natural scale and sign(c0) still carries the
        # measured direction. The RL oracle avoided this only because it used a
        # bare sign() with no deadband at all.
        drive_arg = (float(np.sign(coupling_c0)) * np.cos(phi_hang)
                     * thetadot / max(omega, 1e-9))
        if t < params.t_seed_s:
            # Seed: sign chosen from the MEASURED coupling so it adds energy at
            # this pose/asset/axis rather than removing it.
            a_cmd = params.a_seed * float(np.sign(coupling_c0))
        else:
            a_cmd = amplitude * float(np.tanh(drive_arg / max(deadband, 1e-9)))

        if blend_lqr and K is not None:
            # ---- UNIFIED SMOOTH-BLEND LAW: one continuous control law, no
            # discrete switch anywhere. u = (1-alpha)*u_energy + alpha*u_lqr,
            # with alpha a pure function of state that fades the energy pump out
            # and the LQR in as the pole nears the top.
            #
            # Energy (pump) candidate WITH its leash -- exactly what the swing-up
            # branch applies below; alpha fades this whole term (pump + leash)
            # out near the top so the leash cannot fight the catch.
            if velocity_swingup and state_prev_ee is not None:
                leash_x = float(state_prev_ee[transport_axis_index])
                leash_v = float(state_prev_vel[transport_axis_index])
            else:
                leash_x, leash_v = target_x, target_x_vel
            a_energy = a_cmd - params.k_pos * (leash_x - x_ref) - params.k_vel * leash_v

            # LQR candidate from the MEASURED end-effector state -- the same s_vec
            # the discrete hold uses. Its own clip bounds it to +-lqr_a_max even
            # when the linear law is extrapolated far from the top; alpha then
            # scales that bounded value to ~0 there so it never reaches the cart.
            x_act = float(state_prev_ee[transport_axis_index]) if state_prev_ee is not None else x_ref
            xd_act = float(state_prev_vel[transport_axis_index]) if state_prev_vel is not None else 0.0
            s_vec = np.array([x_act - x_ref, xd_act, phi_inv, thetadot], dtype=np.float64)
            a_lqr = float(np.clip(-(K @ s_vec)[0], -lqr_a_max, lqr_a_max))

            # Blend weight in [0,1]: ~0 far from the top (energy owns the cart),
            # ~1 at the top (LQR owns it). No latch -- alpha rises AND falls as
            # the pole passes through, and only once truly captured (dist->0,
            # |s|->0) does it stay pinned near 1 and hold. Zeroed during the seed
            # (pole at the bottom) and beyond the hard cutoff.
            if t < params.t_seed_s or dist > blend_dist_cutoff:
                alpha = 0.0
            else:
                a_s = 0.5 * (1.0 - float(np.tanh((abs(s_val) - blend_s_center)
                                                 / max(blend_s_width, 1e-9))))
                a_d = 0.5 * (1.0 - float(np.tanh((dist - blend_dist_center)
                                                 / max(blend_dist_width, 1e-9))))
                alpha = a_s * a_d
            a_cmd = (1.0 - alpha) * a_energy + alpha * a_lqr
            max_alpha = max(max_alpha, alpha)
            peak_abs_a_cmd = max(peak_abs_a_cmd, abs(a_cmd))

            # Reporting-only engagement analog of the discrete switch: first time
            # the LQR owns at least half the command AND the pole is genuinely
            # near the top. Drives the hold metrics; it does NOT gate control.
            if lqr_engaged_t is None and alpha >= 0.5 and dist <= phi_switch_max_rad:
                lqr_engaged_t = t
                lqr_switch_state = {"t": t, "phi_inv_rad": phi_inv,
                                    "thetadot": thetadot, "s": s_val,
                                    "energy_over_e_top": e_frac, "alpha": alpha}

            # Same velocity-tracked integration + step as the rest of the rollout.
            target_x_vel = float(np.clip(target_x_vel + a_cmd * CONTROL_DT,
                                         -V_MAX_MPS, V_MAX_MPS))
            target_x = target_x + target_x_vel * CONTROL_DT
            state = build_mujoco_state(
                model, data, site_id=site_id, joint_ids=joint_ids,
                time_s=t, dt_s=CONTROL_DT,
                target_x=target_x, target_x_vel=target_x_vel, target_x_accel=a_cmd,
                reference_quat=state0.ee_quat,
                transport_axis_index=transport_axis_index,
                gravity_compensation=True,
            )
            tau, diag = adapter.step(state=state)
            if not bool(diag.get("safety_ok", True)) and not guard_fired:
                guard_fired = True
                guard_reason = str(diag.get("safety_reason", ""))
                guard_t = t
            mon.step(thetadot=thetadot, phi=phi_hang, drive_accel=a_cmd, dt=CONTROL_DT)
            state_prev_ee = np.asarray(state.ee_pos, dtype=np.float64)
            state_prev_vel = np.asarray(state.ee_lin_vel, dtype=np.float64)
            data.ctrl[:6] = np.asarray(tau, dtype=np.float64).reshape(6)
            mujoco.mj_step(model, data)
            if tip_site >= 0:
                tip_z_now = float(data.site_xpos[tip_site][2])
                min_tip_z = min(min_tip_z, tip_z_now)
                if tip_z_now < FLOOR_MARGIN_M and not guard_fired:
                    guard_fired = True
                    guard_reason = f"tip world z {tip_z_now:.4f} < floor margin {FLOOR_MARGIN_M} m"
                    guard_t = t
                    break
            if lqr_engaged_t is not None:
                max_abs_phi_after_switch = max(max_abs_phi_after_switch, abs(phi_inv))
                last_hold_t = t
                if fell_after_switch_t is None and abs(phi_inv) > 0.5:
                    fell_after_switch_t = t
            if track_history:
                history.append({"t": t, "phase": "blend", "alpha": float(alpha),
                                "a_cmd": a_cmd, "a_energy": a_energy, "a_lqr": a_lqr,
                                "phi_inv_deg": float(np.degrees(phi_inv)),
                                "thetadot": thetadot, "s": s_val,
                                "energy_over_e_top": e_frac,
                                "orientation_error": float(diag.get("orientation_error_norm", 0.0)),
                                "ee": np.asarray(state.ee_pos, dtype=np.float64).tolist(),
                                "qpos6": np.asarray(data.qpos[:6], dtype=np.float64).tolist()})
            if lqr_engaged_t is not None and t - lqr_engaged_t >= hold_s:
                held_to_end = True
                break
            continue

        # SOURCE SWITCH, not a controller swap. Once the pole is inside the
        # capture band the LQR takes over writing the cart acceleration; the
        # low-level controller and the integrated reference are untouched.
        # target_x/target_x_vel are deliberately CARRIED ACROSS -- re-zeroing
        # them injects a step into the inner loop that looks like a catch
        # failure but is a handoff artifact (AGENTS.md's Goal-1 seam lesson).
        if K is not None and lqr_engaged_t is None:
            if abs(s_val) <= s_switch and dist <= phi_switch_max_rad:
                lqr_engaged_t = t
                # velocity_hold keeps the drive row in VELOCITY tracking through
                # the catch instead of switching to position tracking. At the
                # singular pose this is what makes the hold realizable: position
                # tracking adds lag that mis-phases the balancing cart-accel and
                # PUMPS the pole (energy climbs to 1.12 E_top, Z sags into the
                # floor guard); velocity tracking removes that integrator so the
                # LQR's command is realized cleanly (energy stays ~1.0). The env
                # var is kept as an override for older launch scripts.
                _keep_vel_hold = velocity_hold or bool(
                    int(os.environ.get("KEEP_VEL_HOLD", "0")))
                if velocity_swingup and not _keep_vel_hold:
                    # Back to position tracking for the catch. The LQR was
                    # derived against a position-tracked plant; leaving the row
                    # in velocity mode would hand it a different plant than the
                    # one its K was solved for -- the exact substitution this
                    # repo keeps getting caught by.
                    _inner.task_velocity_rows = ()
                    # RE-SYNC THE REFERENCE TO THE ARM (2026-08-23). In velocity
                    # mode nothing tracked target_x -- it is a free-running
                    # integration by-product that drifted far from the arm
                    # (measured: target_x ran to ~-0.3 m while the arm was
                    # elsewhere). Position-tracking that stale target now would
                    # slam the cart toward it, PUMP the pole through vertical and
                    # sag Z into the guard -- exactly the observed catch failure
                    # (thetadot grew -1.3 -> -21 approaching the top). Carrying
                    # target_x across is right for a POSITION-tracked swing-up
                    # (Goal-1 at wrist_2=-90, where it already tracked the arm);
                    # for a velocity swing-up the equivalent "no step at the
                    # seam" requires re-syncing target_x/target_x_vel to the
                    # measured arm state, NOT carrying the runaway across.
                    if state_prev_ee is not None:
                        target_x = float(state_prev_ee[transport_axis_index])
                    if state_prev_vel is not None:
                        target_x_vel = float(state_prev_vel[transport_axis_index])
                lqr_switch_state = {"t": t, "phi_inv_rad": phi_inv,
                                    "thetadot": thetadot, "s": s_val,
                                    "energy_over_e_top": e_frac}

        if lqr_engaged_t is not None:
            x_act = float(state_prev_ee[transport_axis_index]) if state_prev_ee is not None else x_ref
            xd_act = float(state_prev_vel[transport_axis_index]) if state_prev_vel is not None else 0.0
            s_vec = np.array([x_act - x_ref, xd_act, phi_inv, thetadot], dtype=np.float64)
            a_cmd = float(np.clip(-(K @ s_vec)[0], -lqr_a_max, lqr_a_max))
            max_abs_phi_after_switch = max(max_abs_phi_after_switch, abs(phi_inv))
            last_hold_t = t
            if fell_after_switch_t is None and abs(phi_inv) > 0.5:
                fell_after_switch_t = t
            target_x_vel = float(np.clip(target_x_vel + a_cmd * CONTROL_DT,
                                         -V_MAX_MPS, V_MAX_MPS))
            target_x = target_x + target_x_vel * CONTROL_DT
            state = build_mujoco_state(
                model, data, site_id=site_id, joint_ids=joint_ids,
                time_s=t, dt_s=CONTROL_DT,
                target_x=target_x, target_x_vel=target_x_vel, target_x_accel=a_cmd,
                reference_quat=state0.ee_quat,
                transport_axis_index=transport_axis_index,
                gravity_compensation=True,
            )
            tau, diag = adapter.step(state=state)
            if not bool(diag.get("safety_ok", True)) and not guard_fired:
                guard_fired = True
                guard_reason = str(diag.get("safety_reason", ""))
                guard_t = t
            mon.step(thetadot=thetadot, phi=phi_hang, drive_accel=a_cmd, dt=CONTROL_DT)
            state_prev_ee = np.asarray(state.ee_pos, dtype=np.float64)
            state_prev_vel = np.asarray(state.ee_lin_vel, dtype=np.float64)
            data.ctrl[:6] = np.asarray(tau, dtype=np.float64).reshape(6)
            mujoco.mj_step(model, data)
            if tip_site >= 0:
                tip_z_now = float(data.site_xpos[tip_site][2])
                min_tip_z = min(min_tip_z, tip_z_now)
                if tip_z_now < FLOOR_MARGIN_M and not guard_fired:
                    guard_fired = True
                    guard_reason = f"tip world z {tip_z_now:.4f} < floor margin {FLOOR_MARGIN_M} m"
                    guard_t = t
                    break
            if track_history:
                history.append({"t": t, "phase": "lqr", "a_cmd": a_cmd,
                                "phi_inv_deg": float(np.degrees(phi_inv)),
                                "thetadot": thetadot, "s": s_val,
                                "energy_over_e_top": e_frac,
                                "orientation_error": float(diag.get("orientation_error_norm", 0.0)),
                                "ee": np.asarray(state.ee_pos, dtype=np.float64).tolist(),
                            # ARM joints, so a replay renders the REAL arm
                            # motion. Replaying the pendulum angle onto a
                            # frozen arm would show the pole swinging with a
                            # motionless robot -- hiding the cart motion that
                            # is the entire drive mechanism.
                            "qpos6": np.asarray(data.qpos[:6], dtype=np.float64).tolist()})
            if t - lqr_engaged_t >= hold_s:
                held_to_end = True
                break
            continue

        # Leash: bounds net travel so the off-axis drift it couples into stays
        # inside the guard. The ON-axis component is guard-exempt (tracked_axes
        # = {move_axis}), which is why the validated Goal-1 flip could run to
        # max_abs_x_dev = 0.187 m guard-clean -- but the Y/Z it induces is not.
        # LEASH REFERENCE. In position mode the arm tracks target_x closely, so
        # the reference integral is a fair proxy for where the arm actually is.
        # In VELOCITY mode nothing tracks target_x -- it is only an integration
        # by-product -- so it runs away from the arm and the leash brakes
        # against a position the arm never occupied. Measured: peak |a_cmd|
        # collapsed 14.24 -> 3.42 m/s^2 and E_peak/E_top 0.9813 -> 0.0002, i.e.
        # the leash silently cancelled the entire pump. Drive it from the
        # MEASURED end-effector state instead, which is the quantity the drift
        # budget is actually spent in.
        if velocity_swingup and state_prev_ee is not None:
            leash_x = float(state_prev_ee[transport_axis_index])
            leash_v = float(state_prev_vel[transport_axis_index])
        else:
            leash_x, leash_v = target_x, target_x_vel
        a_cmd = a_cmd - params.k_pos * (leash_x - x_ref) - params.k_vel * leash_v
        peak_abs_a_cmd = max(peak_abs_a_cmd, abs(a_cmd))

        target_x_vel = float(np.clip(target_x_vel + a_cmd * CONTROL_DT, -V_MAX_MPS, V_MAX_MPS))
        target_x = target_x + target_x_vel * CONTROL_DT

        state = build_mujoco_state(
            model, data, site_id=site_id, joint_ids=joint_ids,
            time_s=t, dt_s=CONTROL_DT,
            target_x=target_x, target_x_vel=target_x_vel, target_x_accel=a_cmd,
            reference_quat=state0.ee_quat,
            transport_axis_index=transport_axis_index,
            gravity_compensation=True,
        )
        tau, diag = adapter.step(state=state)
        state_prev_ee = np.asarray(state.ee_pos, dtype=np.float64)
        state_prev_vel = np.asarray(state.ee_lin_vel, dtype=np.float64)

        if not bool(diag.get("safety_ok", True)) and not guard_fired:
            guard_fired = True
            guard_reason = str(diag.get("safety_reason", ""))
            guard_t = t

        mon.step(thetadot=thetadot, phi=phi_hang, drive_accel=a_cmd, dt=CONTROL_DT)

        data.ctrl[:6] = np.asarray(tau, dtype=np.float64).reshape(6)
        mujoco.mj_step(model, data)

        if tip_site >= 0:
            tip_z_now = float(data.site_xpos[tip_site][2])
            min_tip_z = min(min_tip_z, tip_z_now)
            if tip_z_now < FLOOR_MARGIN_M and not guard_fired:
                guard_fired = True
                guard_reason = f"tip world z {tip_z_now:.4f} < floor margin {FLOOR_MARGIN_M} m"
                guard_t = t
                break
        if track_history:
            history.append({"t": t, "phase": "swingup", "blend": blend,
                            "amplitude": amplitude, "deadband": deadband, "a_cmd": a_cmd,
                            "phi_inv_deg": float(np.degrees(phi_inv)),
                            "thetadot": thetadot, "s": s_val,
                            "energy_over_e_top": e_frac,
                            "orientation_error": float(diag.get("orientation_error_norm", 0.0)),
                            "ee": np.asarray(state.ee_pos, dtype=np.float64).tolist(),
                            # ARM joints, so a replay renders the REAL arm
                            # motion. Replaying the pendulum angle onto a
                            # frozen arm would show the pole swinging with a
                            # motionless robot -- hiding the cart motion that
                            # is the entire drive mechanism.
                            "qpos6": np.asarray(data.qpos[:6], dtype=np.float64).tolist()})

    b = mon.budget()
    # Use the FIRST arrival, not best-of-all-passes. best_abs_s is None once
    # the rollout breaks at first arrival, which made this field read False
    # on runs that arrived cleanly inside the band.
    capturable = bool(first_arrival is not None
                      and first_arrival_abs_s <= s_capture and not guard_fired)
    return {
        "params": dict(params.__dict__),
        "guard_fired": guard_fired, "guard_reason": guard_reason, "first_guard_t": guard_t,
        "max_blend": float(max_blend), "t_blend_half_s": t_blend_half,
        "blend_lqr": bool(blend_lqr), "max_alpha": float(max_alpha),
        "peak_abs_a_cmd_mps2": float(peak_abs_a_cmd),
        "min_theta_dist_from_inverted": float(min_dist_inverted),
        "best_abs_s": None if not np.isfinite(best_abs_s) else float(best_abs_s),
        "best_arrival": best_arrival,
        # The honest arrival metric: the FIRST time the pole entered the window.
        "first_arrival_abs_s": (None if not np.isfinite(first_arrival_abs_s)
                                else float(first_arrival_abs_s)),
        "first_arrival": first_arrival,
        # Spin diagnostics. total_rotations >> 1 means the pole is a propeller
        # and any |s| drawn from it is a lottery result, not an arrival.
        "total_rotations": float(theta_travel / (2.0 * np.pi)),
        "window_entries": int(n_window_entries),
        "capturable": capturable,
        "lqr_engaged_t": lqr_engaged_t,
        "lqr_switch_state": lqr_switch_state,
        "held_after_switch": held_to_end,
        "max_abs_phi_after_switch_rad": (
            float(max_abs_phi_after_switch) if lqr_engaged_t is not None else None),
        # How long the pole stayed upright (|phi_inv| <= 0.5 rad) after the
        # handoff, in seconds. This is the gradient a hold search needs: max|phi|
        # saturates at pi for ANY eventual fall and cannot tell a 4 s hold from an
        # instant one.
        "upright_duration_after_switch_s": (
            None if lqr_engaged_t is None else
            float((fell_after_switch_t if fell_after_switch_t is not None
                   else (last_hold_t if last_hold_t is not None else lqr_engaged_t))
                  - lqr_engaged_t)),
        "held_and_upright": bool(
            held_to_end and not guard_fired
            and max_abs_phi_after_switch <= 0.35),
        "e_peak_over_e_top": float(b.e_peak_j / e_top) if e_top else None,
        "positive_work_fraction": float(b.positive_work_fraction),
        "work_by_drive_j": float(b.work_by_drive_j),
        # Hinge dissipation, so the energy budget
        #     dE = work_by_drive - dissipated
        # can be CLOSED rather than estimated. Without it, an unexplained
        # shortfall is equally consistent with "hinge friction ate it" and
        # "the arm never delivered the command" -- two very different
        # diagnoses, and this file previously asserted the wrong one.
        "dissipated_j": float(b.dissipated_j),
        # E_initial/E_final are needed to close the budget CORRECTLY:
        #     E_final - E_initial = work_by_drive - dissipated
        # Both sides must be time-integrated quantities. Comparing E_PEAK
        # (an instantaneous maximum) against work and dissipation summed
        # over the whole run is a category error, and produced a bogus
        # 45%-unaccounted "residual" on the first attempt here.
        "e_initial_j": float(b.e_initial_j),
        "e_final_j": float(b.e_final_j),
        "min_tip_world_z_m": None if not np.isfinite(min_tip_z) else float(min_tip_z),
        "history": history,
    }


# Per-process model cache. A compiled MjModel is NOT picklable, so the search
# spec carries only plain data and each worker composes its own model once --
# exactly what PendulumRunContext does for the other pendulum searches. The
# first version of this file closed over the model in a lambda defined inside
# main(), which dies under any process pool with
# "Can't get local object 'main.<locals>.<lambda>'".
_MODEL_CACHE: dict[str, object] = {}


def _model_for(pendulum_xml: str):
    if pendulum_xml not in _MODEL_CACHE:
        _MODEL_CACHE[pendulum_xml] = compose_ur5e_pendulum_model(pendulum_xml=pendulum_xml)
    return _MODEL_CACHE[pendulum_xml]


# A swing-up may legitimately pass the top once on the way to being caught,
# but a trajectory that keeps going is a propeller. Rejecting above this is
# what stops the optimiser from buying more chances at a lucky |s|.
# NO ROTATION PENALTY -- deliberately removed after it broke a search.
# `total_rotations` accumulates ANGULAR PATH LENGTH, not revolutions, so a
# legitimate multi-swing pump-up racks it up without ever crossing the top: the
# pan-locked search's best candidate showed total_rotations = 19.00 with
# window_entries = 0, i.e. ~21 swings to just short of inverted and never over
# it. A threshold of 1.5 therefore rejected EVERY genuine arrival on that plant
# (they all need more than ~1.5 swings to pump up), making arrival more
# expensive than failure, and the optimiser correctly chose to never arrive
# (cost 100.3, min_dist 0.3001 -- missing the window by 1e-4 rad).
#
# The pathology being guarded against was passing through the top REPEATEDLY to
# buy chances at a lucky |s|. The right measure of that is `window_entries`, not
# path length -- and first-arrival scoring already makes entries-before-arrival
# zero by construction, so the penalty was redundant as well as wrong.
# `total_rotations` and `window_entries` remain REPORTED, as diagnostics.
#
# Pendulum natural frequency at this apparatus; used to put phi into the
# same units as thetadot in the arrival cost.
OMEGA_RADPS = 10.8334


def objective_spec(x, spec: dict) -> float:
    """Module-level, picklable objective for the process pool."""
    params = EnergyScheduleParams(a_slow=float(x[0]), a_sharp=float(x[1]),
                                  e_center=float(x[2]), e_width=float(x[3]),
                                  db_slow=float(x[4]), db_sharp=float(x[5]),
                                  e_target=float(x[6]))
    # SCORE-THE-HOLD MODE (2026-08-23). When the spec carries an LQR, run the
    # full flip->catch->hold and score the ACTUAL hold, not just arrival |s|.
    # Arrival |s| is necessary but not sufficient: at this pose the capture basin
    # is a knife-edge (the pole must reach vertical with |thetadot| inside the
    # LQR's LINEAR basin), and only end-to-end scoring finds a schedule that
    # lands there. Velocity-mode hold (KEEP_VEL_HOLD) is what makes the catch
    # realizable here, so it is expected to be set in the search environment.
    lqr_kwargs = {}
    scoring_hold = spec.get("lqr_K") is not None
    if scoring_hold:
        lqr_kwargs = dict(
            lqr_K=np.asarray(spec["lqr_K"], dtype=np.float64),
            lqr_a_max=float(spec["lqr_a_max"]),
            s_switch=float(spec.get("s_switch", 1.2)),
            phi_switch_max_rad=float(spec.get("phi_switch_max_rad", 0.30)),
            hold_s=float(spec.get("hold_s", 6.0)),
            blend_lqr=bool(spec.get("blend_lqr", False)),
            blend_s_center=float(spec.get("blend_s_center", 1.2)),
            blend_s_width=float(spec.get("blend_s_width", 0.5)),
            blend_dist_center=float(spec.get("blend_dist_center", 0.45)),
            blend_dist_width=float(spec.get("blend_dist_width", 0.20)),
            blend_dist_cutoff=float(spec.get("blend_dist_cutoff", 0.9)),
        )
    r = run_energy_scheduled_trial(
        _model_for(spec["pendulum_xml"]), params,
        arm_q=np.asarray(spec["arm_q"], dtype=np.float64),
        hanging_angle=spec["hanging_angle"], inverted_angle=spec["inverted_angle"],
        constants=spec["constants"], coupling_c0=spec["coupling_c0"],
        config_path=Path(spec["config_path"]), controller_kind=spec["controller_kind"],
        transport_axis_index=spec["transport_axis_index"],
        duration_s=spec["duration_s"], s_capture=spec["s_capture"],
        velocity_swingup=bool(spec.get("velocity_swingup", False)),
        velocity_hold=bool(spec.get("velocity_hold", False)),
        **lqr_kwargs,
    )
    if r["guard_fired"]:
        return 1e3 - float(r["first_guard_t"] or 0.0)
    if scoring_hold:
        # Ordering (lower = better): held (< -900) << catch-attempted, ranked by
        # how upright it stayed (100..103) << never arrived (200+). This gives a
        # gradient: learn to arrive, then to stay up longer, then to hold.
        if r.get("held_and_upright"):
            return -1000.0 + float(r.get("max_abs_phi_after_switch_rad") or 0.0)
        if r.get("lqr_engaged_t") is not None:
            # Rank catch-attempts by how LONG they stayed upright, not by max|phi|
            # (which saturates at pi for any eventual fall and gives no gradient).
            dur = float(r.get("upright_duration_after_switch_s") or 0.0)
            return 100.0 - dur
        return 200.0 + float(r["min_theta_dist_from_inverted"])
    if r["first_arrival"] is None:
        # Never reached the top: rank by how close it got, strictly worse than
        # anything that did arrive.
        return 1e2 + float(r["min_theta_dist_from_inverted"])
    # |s| AND |phi| at the FIRST arrival. |s| = 0 defines a MANIFOLD, not a
    # target: (phi=-0.62, thetadot=6.38) and (phi=-0.05, thetadot=0.5) both
    # score |s| ~ 0.3, but only the second is near the equilibrium the LQR
    # linearised about. Measured directly -- a handoff at |s| = 0.29 but
    # phi = -35 deg, thetadot = 6.38 rad/s LOST THE POLE COMPLETELY, while the
    # same LQR captures at phi = 5-10 deg with thetadot = 0. Small |s| is
    # necessary, not sufficient, and the linearisation stops being valid long
    # before |s| notices. omega converts phi into rad/s so the two terms are
    # commensurate; at thetadot = 0 this reduces to 2*omega*|phi|, monotonic in
    # how far from vertical the arrival is.
    fa = r["first_arrival"]
    # SWING-COUNT CONSTRAINT (2026-08-18). `rotations_before_arrival` is
    # accumulated ANGULAR PATH LENGTH over 2*pi, NOT revolutions -- read it as
    # arc: one clean hanging->inverted stroke is 180 deg = 0.5, "two swings" is
    # roughly 1.0-1.5. The unconstrained in-plane schedule sits at 2.63, i.e.
    # about five half-swings.
    #
    # A HARD REJECT, not a penalty. An earlier penalty version was removed
    # because a weight large enough to matter made ARRIVING more expensive than
    # never arriving, so the search learned to miss the capture window by 1e-4
    # rad. Rejecting instead leaves the arrival branch's ordering untouched and
    # only removes violating candidates -- ranked below every accepted arrival
    # but above a non-arrival, since a too-slow flip is still more informative
    # than no flip.
    max_rot = spec.get("max_rotations_before_arrival")
    if max_rot is not None and float(fa["rotations_before_arrival"]) > float(max_rot):
        return 50.0 + float(fa["rotations_before_arrival"])
    return float(r["first_arrival_abs_s"]) + OMEGA_RADPS * abs(float(fa["phi_inv_rad"]))


def objective(x, *, ctx) -> float:
    """Minimise |s| at closest approach -- NOT min_theta_dist_from_inverted.

    That metric rewards reaching the top at speed, i.e. a fly-through, which is
    the one arrival a catch cannot use. A guard trip is rejected outright rather
    than penalised, per this repo's rule that a result needing guards off is a
    negative result.
    """
    params = EnergyScheduleParams(a_slow=float(x[0]), a_sharp=float(x[1]),
                                  e_center=float(x[2]), e_width=float(x[3]))
    r = run_energy_scheduled_trial(ctx["model"], params, **ctx["trial_kwargs"])
    if r["guard_fired"]:
        return 1e3 - float(r["first_guard_t"] or 0.0)
    if r["first_arrival"] is None:
        # Never reached the top: rank by how close it got, strictly worse than
        # anything that did arrive.
        return 1e2 + float(r["min_theta_dist_from_inverted"])
    # |s| AND |phi| at the FIRST arrival. |s| = 0 defines a MANIFOLD, not a
    # target: (phi=-0.62, thetadot=6.38) and (phi=-0.05, thetadot=0.5) both
    # score |s| ~ 0.3, but only the second is near the equilibrium the LQR
    # linearised about. Measured directly -- a handoff at |s| = 0.29 but
    # phi = -35 deg, thetadot = 6.38 rad/s LOST THE POLE COMPLETELY, while the
    # same LQR captures at phi = 5-10 deg with thetadot = 0. Small |s| is
    # necessary, not sufficient, and the linearisation stops being valid long
    # before |s| notices. omega converts phi into rad/s so the two terms are
    # commensurate; at thetadot = 0 this reduces to 2*omega*|phi|, monotonic in
    # how far from vertical the arrival is.
    fa = r["first_arrival"]
    return float(r["first_arrival_abs_s"]) + OMEGA_RADPS * abs(float(fa["phi_inv_rad"]))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pendulum-xml", required=True)
    p.add_argument("--start-q-rad", type=float, nargs=6, required=True)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--controller-kind", default="impedance")
    p.add_argument("--transport-axis-index", type=int, default=0)
    p.add_argument("--duration-s", type=float, default=12.0)
    p.add_argument("--s-capture", type=float, default=1.2)
    p.add_argument("--search-backend", default="de", choices=["de", "optuna"])
    p.add_argument("--maxiter", type=int, default=25)
    p.add_argument("--popsize", type=int, default=16)
    p.add_argument("--n-trials", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--a-slow-max", type=float, default=6.0)
    p.add_argument("--a-sharp-max", type=float, default=20.0)
    p.add_argument("--allow-pose-mismatch", action="store_true")
    p.add_argument("--evaluate", type=float, nargs=7, default=None,
                   metavar=("A_SLOW", "A_SHARP", "E_CENTER", "E_WIDTH",
                            "DB_SLOW", "DB_SHARP", "E_TARGET"),
                   help="Skip the search and evaluate one schedule. ALL SEVEN "
                        "searched parameters, in the search's own order. It "
                        "previously took only the first four and silently used "
                        "defaults for db_slow/db_sharp/e_target, so a search "
                        "winner could not be reproduced through this flag at "
                        "all -- and a handoff run was reported as 'the searched "
                        "schedule' when it was a different one.")
    p.add_argument("--lqr-json", type=Path, default=None,
                   help="Run the LQR handoff after swing-up, using K/a_max from "
                        "this file (pendulum_lqr_cascade output).")
    p.add_argument("--s-switch", type=float, default=1.2)
    p.add_argument("--phi-switch-max-rad", type=float, default=0.45)
    p.add_argument("--hold-s", type=float, default=4.0)
    p.add_argument("--max-rotations", type=float, default=None,
                   help="Reject arrivals whose accumulated angular PATH LENGTH "
                        "before first arrival exceeds this many turns (arc/360deg). "
                        "One clean hanging->inverted stroke is 0.5; two swings is "
                        "about 1.0-1.5. Default None = no constraint.")
    p.add_argument("--velocity-swingup", action="store_true",
                   help="Track the drive row in VELOCITY during swing-up and switch "
                        "back to position tracking at the LQR handoff. Removes one "
                        "integrator from the drive path (the law produces an "
                        "acceleration that is otherwise double-integrated into a "
                        "position target), so the pump carries less lag -- which is "
                        "what a 1-2 stroke flip needs. Every CBF row, the torque box, "
                        "the joint exclusion and the posture weights are unchanged: "
                        "this drops one TERM from one task row, it is not a "
                        "different controller.")
    p.add_argument("--velocity-hold", action="store_true",
                   help="Keep the drive row in VELOCITY tracking through the LQR "
                        "catch instead of switching to position tracking. Required "
                        "for the hold at the singular ARM_Q0: position tracking's "
                        "lag mis-phases the balancing cart-acceleration and PUMPS "
                        "the pole (E climbs to 1.12 E_top, Z sags into the floor "
                        "guard); velocity tracking realizes the LQR command cleanly "
                        "(E stays ~1.0) so the catch holds. Default off preserves "
                        "the position-tracked catch the LQR K was derived against.")
    p.add_argument("--blend-lqr", action="store_true",
                   help="ONE unified high-level law instead of an energy->LQR "
                        "source switch: u=(1-alpha)*u_energy + alpha*u_lqr, alpha "
                        "a smooth state function that fades the pump out and the "
                        "LQR in near the top. No discrete switch anywhere. Requires "
                        "--velocity-hold (the switch-free blend keeps the drive row "
                        "velocity-tracked throughout, so there is no seam to "
                        "re-sync a position catch at). Needs --lqr-json.")
    p.add_argument("--blend-s-center", type=float, default=1.2)
    p.add_argument("--blend-s-width", type=float, default=0.5)
    p.add_argument("--blend-dist-center", type=float, default=0.45)
    p.add_argument("--blend-dist-width", type=float, default=0.20)
    p.add_argument("--blend-dist-cutoff", type=float, default=0.9)
    p.add_argument("--output-json", type=Path, default=None)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    arm_q = np.asarray(args.start_q_rad, dtype=np.float64)

    provenance = check_config_pose(
        load_config(Path(args.config)), arm_q, args.pendulum_xml,
        config_name=Path(args.config).name,
        allow_mismatch=bool(args.allow_pose_mismatch),
    )

    model = compose_ur5e_pendulum_model(pendulum_xml=str(args.pendulum_xml))
    hanging_angle, inverted_angle = resolve_equilibria(model, arm_q)
    constants = derive_pendulum_constants(model, arm_q)

    axis = int(args.transport_axis_index)
    # The PHYSICAL drive direction is column `axis` of controller.task_rotation
    # when the config sets one, not the world unit vector. c0 sets the energy
    # law's SIGN, so getting this wrong does not merely mis-scale the drive --
    # it damps instead of pumping. Same source the controller and the safety
    # monitor read, so all three agree on what "the drive axis" means.
    _rot_cfg = (load_config(Path(args.config)).get("controller") or {}).get("task_rotation")
    if _rot_cfg is not None:
        drive_axis = np.asarray(_rot_cfg, dtype=np.float64).reshape(3, 3)[:, axis]
    else:
        drive_axis = np.zeros(3)
        drive_axis[axis] = 1.0
    c0 = float(measure_pivot_coupling(model, arm_q, hanging_angle, drive_axis))

    print(f"pendulum={Path(args.pendulum_xml).name}  arm_q={np.round(arm_q, 6).tolist()}")
    print(f"config={Path(args.config).name}  kind={args.controller_kind}")
    print(describe_provenance(provenance))
    print(f"hanging={hanging_angle:.4f}  inverted={inverted_angle:.4f}  "
          f"omega={constants.omega_natural_radps:.4f}  E_top={constants.e_top_j:.6f} J")
    _axis_label = f"task row {axis} (rotated)" if _rot_cfg is not None else AXIS_NAMES[axis]
    print(f"drive axis {_axis_label} = {np.round(drive_axis, 6).tolist()}  c0={c0:+.6f} Nm per m/s^2")
    if abs(c0) < 1e-5:
        print("REFUSING TO RUN: this drive axis has no authority over the hinge "
              "(|c0| ~ 0). Pumping along it does nothing.")
        return 2

    trial_kwargs = dict(
        arm_q=arm_q, hanging_angle=hanging_angle, inverted_angle=inverted_angle,
        constants=constants, coupling_c0=c0, config_path=Path(args.config),
        controller_kind=str(args.controller_kind), transport_axis_index=axis,
        duration_s=float(args.duration_s), s_capture=float(args.s_capture),
        velocity_swingup=bool(args.velocity_swingup),
        velocity_hold=bool(args.velocity_hold),
    )
    lqr_kwargs = {}
    if args.lqr_json is not None:
        _l = json.loads(Path(args.lqr_json).read_text())["lqr"]
        lqr_kwargs = dict(
            lqr_K=np.asarray(_l["K"], dtype=np.float64),
            lqr_a_max=float(_l["a_max"]),
            s_switch=float(args.s_switch),
            phi_switch_max_rad=float(args.phi_switch_max_rad),
            hold_s=float(args.hold_s),
            blend_lqr=bool(args.blend_lqr),
            blend_s_center=float(args.blend_s_center),
            blend_s_width=float(args.blend_s_width),
            blend_dist_center=float(args.blend_dist_center),
            blend_dist_width=float(args.blend_dist_width),
            blend_dist_cutoff=float(args.blend_dist_cutoff),
        )
        if args.blend_lqr:
            print(f"UNIFIED BLEND LAW (no switch): alpha gates s@{args.blend_s_center}"
                  f"/{args.blend_s_width}, dist@{args.blend_dist_center}"
                  f"/{args.blend_dist_width}, cutoff {args.blend_dist_cutoff}")
        print(f"LQR handoff: K={_l['K']}  a_max={_l['a_max']:.4f}  "
              f"switch |s|<={args.s_switch} and |phi|<={args.phi_switch_max_rad}")
    ctx = {"model": model, "trial_kwargs": trial_kwargs}

    if args.evaluate is not None:
        best = EnergyScheduleParams(a_slow=args.evaluate[0], a_sharp=args.evaluate[1],
                                    e_center=args.evaluate[2], e_width=args.evaluate[3],
                                    db_slow=args.evaluate[4], db_sharp=args.evaluate[5],
                                    e_target=args.evaluate[6])
    else:
        from tools.diagnostics.pendulum_search_backends import minimize as _minimize
        bounds = [
            (0.2, float(args.a_slow_max)),    # a_slow
            (1.0, float(args.a_sharp_max)),   # a_sharp
            (0.20, 0.95),                     # e_center
            (0.02, 0.40),                     # e_width
            (0.005, 0.20),                    # db_slow  -- searched, see above
            (0.002, 0.10),                    # db_sharp
            (0.98, 1.10),                     # e_target -- self-limit, see above
        ]
        print(f"\n=== searching the energy schedule via {args.search_backend} ===")
        spec = {
            "pendulum_xml": str(args.pendulum_xml),
            "arm_q": arm_q.tolist(),
            "hanging_angle": float(hanging_angle),
            "inverted_angle": float(inverted_angle),
            "constants": constants,
            "coupling_c0": float(c0),
            "config_path": str(args.config),
            "controller_kind": str(args.controller_kind),
            "transport_axis_index": axis,
            "duration_s": float(args.duration_s),
            "s_capture": float(args.s_capture),
            "velocity_swingup": bool(args.velocity_swingup),
            "velocity_hold": bool(args.velocity_hold),
            "max_rotations_before_arrival": args.max_rotations,
        }
        if args.lqr_json is not None:
            # Score the full flip->catch->hold in the search, not just arrival.
            spec["lqr_K"] = [float(v) for v in np.asarray(_l["K"], dtype=float).ravel()]
            spec["lqr_a_max"] = float(_l["a_max"])
            spec["s_switch"] = float(args.s_switch)
            spec["phi_switch_max_rad"] = float(args.phi_switch_max_rad)
            spec["hold_s"] = float(args.hold_s)
            spec["blend_lqr"] = bool(args.blend_lqr)
            spec["blend_s_center"] = float(args.blend_s_center)
            spec["blend_s_width"] = float(args.blend_s_width)
            spec["blend_dist_center"] = float(args.blend_dist_center)
            spec["blend_dist_width"] = float(args.blend_dist_width)
            spec["blend_dist_cutoff"] = float(args.blend_dist_cutoff)
            print(f"SEARCH SCORES THE HOLD: LQR a_max={_l['a_max']:.3f} "
                  f"phi_switch<={args.phi_switch_max_rad} hold_s={args.hold_s}")
        res = _minimize(functools.partial(objective_spec, spec=spec), bounds,
                        backend=args.search_backend, maxiter=args.maxiter,
                        popsize=args.popsize, seed=args.seed,
                        workers=int(args.workers), n_trials=args.n_trials)
        best = EnergyScheduleParams(a_slow=float(res.x[0]), a_sharp=float(res.x[1]),
                                    e_center=float(res.x[2]), e_width=float(res.x[3]),
                                    db_slow=float(res.x[4]), db_sharp=float(res.x[5]),
                                    e_target=float(res.x[6]))
        print(f"best cost = {res.fun:.6f}")

    out = run_energy_scheduled_trial(model, best, **trial_kwargs, **lqr_kwargs)
    print(f"\na_slow={best.a_slow:.4f}  a_sharp={best.a_sharp:.4f}  "
          f"e_center={best.e_center:.4f}  e_width={best.e_width:.4f}")
    print(f"max blend          = {out['max_blend']:.4f}  "
          f"(t at blend=0.5: {out['t_blend_half_s']})")
    print(f"peak |a_cmd|       = {out['peak_abs_a_cmd_mps2']:.4f} m/s^2")
    print(f"E_peak/E_top       = {out['e_peak_over_e_top']:.4f}")
    print(f"positive_work_frac = {out['positive_work_fraction']:.4f}  "
          f"(analytic law ~1.0; <0.5 means the drive REMOVES energy)")
    print(f"min dist inverted  = {out['min_theta_dist_from_inverted']:.4f} rad")
    print(f"total rotations    = {out['total_rotations']:.2f}   window entries = {out['window_entries']}")
    print(f"FIRST-arrival |s|  = {out['first_arrival_abs_s']}  (capture band <= {args.s_capture})")
    print(f"  min |s| over ALL passes = {out['best_abs_s']}  (a lottery result when rotations >> 1; kept for comparison only)")
    print(f"CAPTURABLE         = {out['capturable']}")
    print(f"guard_fired        = {out['guard_fired']}  {out['guard_reason'] or ''}")
    if out.get("lqr_engaged_t") is not None:
        sw = out["lqr_switch_state"]
        print(f"LQR engaged at t   = {out['lqr_engaged_t']:.3f} s  "
              f"(phi={sw['phi_inv_rad']:+.4f} rad, thetadot={sw['thetadot']:+.4f}, s={sw['s']:+.4f})")
        print(f"max |phi| after sw = {out['max_abs_phi_after_switch_rad']:.4f} rad")
        print(f"HELD AND UPRIGHT   = {out['held_and_upright']}")
    elif out.get("lqr_switch_state") is None and lqr_kwargs:
        print("LQR never engaged  = swing-up never entered the switch window")
    print(f"min tip world z    = {out['min_tip_world_z_m']}")

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        payload = dict(out)
        payload.pop("history", None)
        payload.update({"arm_q": arm_q.tolist(), "config": str(args.config),
                        "pendulum_xml": str(args.pendulum_xml),
                        "coupling_c0": c0, "provenance": provenance.as_dict()})
        args.output_json.write_text(json.dumps(payload, indent=2))
        print(f"\nwrote {args.output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
