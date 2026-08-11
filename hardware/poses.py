"""Named joint poses shared by MuJoCo experiments and hardware bring-up."""

from __future__ import annotations

import numpy as np

# Canonical transport poses (see rl_gain_scheduling/gain_scheduling_env.py).
ACTIVE_ORIGIN_Q = np.array(
    [0.0, -1.5707963267948966, 0.0, -1.5707963267948966, 0.0, 0.0],
    dtype=np.float64,
)
LOWER_B_Q = np.array([0.0, -0.1, -2.4, -0.4, 0.0, 0.0], dtype=np.float64)

# height_alpha=0.5 — matches tools/_run_fixed_gain_test.sh Test 2.
HEIGHT_ALPHA_0_5_Q = (0.5 * ACTIVE_ORIGIN_Q + 0.5 * LOWER_B_Q).astype(np.float64)

# height_alpha=0.5 with a real-lab base-rotation for wall/obstacle clearance
# (shoulder_pan = -45 deg), visually confirmed twice on the real robot
# (2026-07-31) at this exact pose in this exact physical setup. Same
# precedent value used for the alpha=0.1 pose on 2026-07-28. This is now the
# default real-hardware start pose for height_alpha_0_5 (see
# tools/ur5e_move_joints.py / hardware/x_transport.py /
# hardware/urscript_transport.py) -- if the robot or room layout ever
# changes, re-verify clearance visually before trusting this again; override
# back to HEIGHT_ALPHA_0_5_Q (or a fresh --shoulder-pan-override-rad) if so.
HEIGHT_ALPHA_0_5_CLEARANCE_Q = HEIGHT_ALPHA_0_5_Q.copy()
HEIGHT_ALPHA_0_5_CLEARANCE_Q[0] = -0.7853981633974483


# height_alpha=0.5 with wrist_2 nudged off exactly 0 (added 2026-08-02). HEIGHT_ALPHA_0_5_Q
# sits exactly at wrist_2=0, the UR-family wrist singularity (cond(full 6x6 J) ~7.28e16 at
# this pose) -- confirmed by direct measurement that a single-joint offset resolves this far
# more cheaply than either controller-side split_base_wrist_task or the from-scratch
# "hanging" pose family above: at wrist_2=0.2 rad (~11 deg), cond(J) drops to ~29.2 (15
# orders of magnitude better), while site position shifts only ~2cm and tool orientation
# tilts only slightly (Z-axis [0,-1,0] -> [0.20,-0.98,0.02]) from HEIGHT_ALPHA_0_5_Q -- unlike
# the hanging-pose family, whose from-scratch search produced a tool orientation
# ([0.588,0,0.809] Z-axis) with no resemblance to the original at all. This is the minimal,
# single-joint fix; everything else is unchanged from HEIGHT_ALPHA_0_5_Q.
#
# Real-hardware validated in position mode (2026-08-02, position_20260802_170653):
# duration_complete, safety_pass=True, achieved 0.01998/0.02m (99.9%), orientation error
# <0.0004 rad throughout, zero real-time telemetry issues. direct_torque not yet tested.
HEIGHT_ALPHA_0_5_WRIST2_OFFSET_Q = HEIGHT_ALPHA_0_5_Q.copy()
HEIGHT_ALPHA_0_5_WRIST2_OFFSET_Q[4] = 0.2


# height_alpha=0.5 with the elbow flipped to the alternate IK branch (genuinely inverted/
# "hanging" arm shape, user-requested 2026-08-02) PLUS the same wrist_2 nudge as above,
# reaching the same tool position/orientation as HEIGHT_ALPHA_0_5_Q to within a small,
# deliberate margin. Found via numeric IK (damped least-squares) seeded to converge to the
# elbow-up branch (elbow=+1.098 rad here vs HEIGHT_ALPHA_0_5_Q's elbow=-1.2 rad -- a real
# sign flip, not a small tweak) targeting HEIGHT_ALPHA_0_5_Q's exact tool pose
# (pos=[-0.1215,-0.234,0.9279], Z-axis=[0,-1,0]) to <1e-6 residual before applying the
# wrist_2 offset.
#
# IMPORTANT, measured directly: the elbow-inverted branch alone (wrist_2=0) does NOT avoid
# the singularity -- wrist_2 converges back to ~0 on its own regardless of elbow branch,
# since the singularity is tied to this specific TARGET ORIENTATION, not to which elbow
# solution reaches it (cond(J)=1.67e8 at wrist_2=0 on this branch). The wrist_2 offset is
# still required. On this branch it needs a larger offset than HEIGHT_ALPHA_0_5_WRIST2_OFFSET_Q
# above for comparable conditioning (0.3 rad here vs 0.2 rad there): wrist_2=0.1 -> cond=208,
# 0.2 -> cond=104, 0.3 -> cond=70.2 (chosen value, ~14 orders of magnitude better than
# singular, position shift ~3cm, orientation Z-axis [0,-1,0] -> [0.206,-0.955,0.212]).
#
# Sim-only so far -- no real-hardware validation of this specific pose yet. When testing
# direct_torque mode with this pose, always pass it as --start-q-rad explicitly so the
# transport loop's own moveJ brings the arm here as the real starting origin before any
# torque command begins -- do not rely on skip-joint-move or an assumed prior position.
HEIGHT_ALPHA_0_5_ELBOW_INVERTED_WRIST2_OFFSET_Q = np.array(
    [-0.0, -2.023883, 1.098332, -1.414729, 0.3, -0.680517], dtype=np.float64
)


def q_for_height_alpha(alpha: float) -> np.ndarray:
    """Interpolate between active-origin (0) and lower-B (1) joint poses."""
    alpha = float(alpha)
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"height_alpha must be in [0, 1]; got {alpha}")
    return ((1.0 - alpha) * ACTIVE_ORIGIN_Q + alpha * LOWER_B_Q).astype(np.float64)


# --- "Hanging"/elbow-down transport pose family (added 2026-08-01) -----------------
#
# ACTIVE_ORIGIN_Q/LOWER_B_Q/q_for_height_alpha above sit at wrist_2=0 across their ENTIRE
# range (a genuine UR-family kinematic singularity; cond(full 6x6 J) measured 1e16-2.5e17
# throughout -- see docs/status/hanging_pose_transport_family_2026-08-01.md and the
# 2026-08-01 night session docs this follows from). This family is a from-scratch,
# additive alternative that avoids that singularity across its whole range by construction
# (wrist_2 held at +pi/2 throughout, never 0) instead of routing around its consequences in
# the controller (contrast with `split_base_wrist_task` in
# controller_core/x_axis_cartesian_impedance.py, a controller-side workaround for the SAME
# singularity that leaves ACTIVE_ORIGIN_Q/LOWER_B_Q unchanged).
#
# Design: elbow-down / "hanging" shape (shoulder_lift steeply negative, elbow bent well
# away from both 0 -- fully extended -- and +-pi -- fully folded; wrist_2 fixed at +pi/2)
# rather than the old family's near-fully-extended-arm shape. Found by a numeric grid
# search over (shoulder_lift, elbow, wrist_1) at wrist_2=+-pi/2 for cond(full 6x6 J) < 50
# and 0.3 <= site z <= 1.3 m, then each endpoint refined (scipy Nelder-Mead) to match
# ACTIVE_ORIGIN_Q's/LOWER_B_Q's own site-frame Z heights as closely as practical.
#
# HANGING_ORIGIN_Q ("tall" end, alpha=0): site pos (x,y,z) = (-0.138, -0.134, 1.044) m,
# cond(full 6x6 J) = 15.41, gravity-comp torque max |tau| = 9.04 Nm.
HANGING_ORIGIN_Q = np.array(
    [0.0, -1.791994, 0.812668, -1.288057, 1.5707963267948966, 0.0],
    dtype=np.float64,
)
# HANGING_LOWER_Q ("low" end, alpha=1): site pos (x,y,z) = (-0.409, -0.134, 0.537) m,
# cond(full 6x6 J) = 7.04, gravity-comp torque max |tau| = 16.81 Nm.
HANGING_LOWER_Q = np.array(
    [0.0, -1.491612, 1.990426, -2.630057, 1.5707963267948966, 0.0],
    dtype=np.float64,
)

# hanging_alpha=0.5 -- the pose used for this family's first-pass gain tuning and rigor
# sweep (mirrors how HEIGHT_ALPHA_0_5_Q anchors the old family's own validation).
HANGING_ALPHA_0_5_Q = (0.5 * HANGING_ORIGIN_Q + 0.5 * HANGING_LOWER_Q).astype(np.float64)


# hanging_alpha=0.5 with the same real-lab base-rotation (shoulder_pan = -45 deg) applied to
# HEIGHT_ALPHA_0_5_Q above -- built purely by mirroring that exact pattern onto this family's
# own alpha=0.5 midpoint (HANGING_ALPHA_0_5_Q), the point this family's own first-pass gain
# tuning and rigor sweep already anchor to (see the "hanging_alpha=0.5" comment above and
# docs/status/hanging_pose_transport_family_2026-08-01.md sec 3), analogous to how
# HEIGHT_ALPHA_0_5_CLEARANCE_Q anchors to the old family's own validated height_alpha=0.5
# point rather than one of its raw endpoints.
#
# UNLIKE HEIGHT_ALPHA_0_5_CLEARANCE_Q, this constant is NOT a real-hardware default and has
# NEVER been visually confirmed for wall/base clearance in the physical lab -- the hanging
# posture's swept volume near the base is a different shape from the old "tall" family's, and
# clearance for THIS specific rotated shape has never been checked. Added 2026-08-02 purely to
# characterize whether rotating shoulder_pan reintroduces the old family's real-hardware
# "-45 deg base-rotation Y-drift coupling" problem (AGENTS.md sec 3) on this structurally
# different, singularity-avoiding pose family. See
# docs/status/hanging_pose_clearance_variant_2026-08-02.md for the cond(J) sweep and rigor-
# sweep investigation this constant was built for. Do NOT treat this as real-hardware-ready;
# it requires its own dedicated visual clearance check before ever being commanded live,
# exactly like every other pose in this project's history.
HANGING_ALPHA_0_5_CLEARANCE_Q = HANGING_ALPHA_0_5_Q.copy()
HANGING_ALPHA_0_5_CLEARANCE_Q[0] = -0.7853981633974483


def q_for_hanging_height_alpha(alpha: float) -> np.ndarray:
    """Interpolate between the hanging-family origin (0) and lower (1) joint poses.

    Full cond(full 6x6 J) sweep across this range (21-point linear interpolation, measured
    2026-08-01): min 7.04, max 15.41 -- three to sixteen orders of magnitude better than the
    1e16-2.5e17 measured across ``q_for_height_alpha``'s whole range, and comfortably below
    the ``jacobian_singular_cond_max`` thresholds this repo treats as "well conditioned"
    elsewhere (e.g. the ~7.8 base-only sub-Jacobian in
    docs/status/split_base_wrist_impedance_2026-08-01.md). See
    docs/status/hanging_pose_transport_family_2026-08-01.md for the full sweep table,
    reachability verification, and gain-tuning/validation results.

    NOT a drop-in replacement for ``q_for_height_alpha`` -- sim-only, no real-hardware or
    physical-clearance validation exists for this pose family yet. Do not use on real
    hardware without a dedicated visual clearance check first (see that status doc).
    """
    alpha = float(alpha)
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"hanging_height_alpha must be in [0, 1]; got {alpha}")
    return ((1.0 - alpha) * HANGING_ORIGIN_Q + alpha * HANGING_LOWER_Q).astype(np.float64)


# --- Mega-pose-search winner (added 2026-08-11) ------------------------------------
#
# Found by tools/diagnostics/pose_oscillation_stability_search.py: a broad kinematic
# pre-filter (~200k random (shoulder_lift, elbow, wrist_1, wrist_2) samples, cond(J) < 30,
# site height in [0.3, 1.2]m) narrowed to 71 diverse representatives, then scored by a real
# dynamic test (6 forced alternating raised-cosine kicks on the bare arm, torque-lane
# impedance controller) -- this pose survived all 6 cleanly, cond(J)=6.93, best in the
# batch. Meaningfully better than the earlier hand-picked hanging-pose candidate on every
# axis independently validated the same session (tools/ur5e_move_hold_transport.py real
# sweeps, config/ur5e_mujoco_torque_osc_hanging_pose_friction_ff.yaml):
#   X-range:  clean -0.25m to +0.25m   (old candidate: roughly +-0.10-0.15m)
#   speed:    0.78 m/s (-X) / 1.10 m/s (+X)  (old candidate: 0.31-0.82 m/s)
# Also the pose the pendulum swing-up investigation used (tools/diagnostics/
# pendulum_swingup_*.py) -- best achieved there was ~40deg of real swing from hanging
# (not a full flip), see that work's own docs for the honest ceiling.
#
# Gains: for real hardware, use config/ur5e_mujoco_torque_osc_mega_search_winner.yaml,
# which keeps the ORIGINAL, thoroughly-validated kp_x=400/kd_x=40 (93-95% tracking
# accuracy at dx=+-0.20m, both directions, under this repo's real tolerance --
# transport_metrics.py: max(5mm, 25% of target)).
#
# A differential_evolution search (tools/diagnostics/x_transport_gain_scheduling_newpose.py)
# also found an 8-parameter live gain schedule AND a single FIXED (kp_x=110.2050,
# kd_x=33.6673) pair that both extend safety-guard survival out to 0.20-0.30m aggressive
# moves where the default gains fail outright. CORRECTED 2026-08-11 (previously overstated
# here): that result was validated using ONLY a safety-guard-trip check, NOT this repo's
# real accuracy tolerance -- cross-checked directly for the first time while building the
# hardware config above, and the single gain is genuinely marginal against it (dx=+0.20m
# passes barely at 4.4cm error vs. the 5cm bound; dx=-0.20m FAILS at 6.0cm error). Treat
# the schedule/single-gain result as a real but not-yet-properly-validated finding, not a
# real-hardware-ready alternative to the default gains.
#
# NOT YET REAL-HARDWARE VALIDATED -- sim-only, same as every other pose in this file before
# its own first real-lab session. In particular:
#   - No physical/visual clearance check has been done at this pose. It is a genuinely
#     different arm shape (elbow bent ~120deg, wrist_2 near +90deg) from every previously
#     real-hardware-cleared pose in this file -- do not assume the lab's existing table/
#     mount/cable clearance still holds. Do this FIRST, before any --i-understand-this-
#     moves-the-robot run, exactly like every other pose here required.
#   - shoulder_pan is 0 here (no base rotation). The old pose family needed a -45deg
#     rotation for real wall/base clearance in this specific lab
#     (HEIGHT_ALPHA_0_5_CLEARANCE_Q) -- whether this pose needs an analogous rotation for
#     the same physical reason is UNCHECKED. If the lab layout is unchanged, check this
#     pose's own swept volume near the base visually before trusting shoulder_pan=0.
#   - The single-gain/schedule search validated the BARE arm only, matching this session's
#     other X-transport work -- not with the pendulum attachment. If testing with the
#     pendulum mounted, re-validate (mass/inertia at the end effector changes the dynamics).
MEGA_SEARCH_WINNER_Q = np.array(
    [0.0, -1.091985784398452, 2.0935362786892546, -2.7685637962327356,
     1.5620693866337145, 0.0],
    dtype=np.float64,
)
