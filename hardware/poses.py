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
