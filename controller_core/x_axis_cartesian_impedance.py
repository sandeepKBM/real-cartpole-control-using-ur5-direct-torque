"""
Constrained Cartesian impedance / PD torque law for UR5 X transport.

Stabilizes X tracking while holding initial Y, Z, tool orientation, and a
rest joint posture. Pure numpy; no simulator imports.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

from .kinematics_utils import orientation_error_vec_wxyz
from .state_types import as_impedance_robot_state


JOINT_NAME_ORDER: tuple[str, ...] = (
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
)

# Fixed per-joint weighting for the optional wrist-orientation task term
# (``wrist_orientation_task``, see CartesianImpedanceConfig below). Shape
# matches ``JOINT_NAME_ORDER``. Values are the ``face_mask``/``orientation
# mask`` ratios from the two pre-torque-lane kinematic controllers that
# originated this task-partition idea (``archive/legacy_mujoco/controller.py``
# -- ``split_forearm_origin_face_controller`` (C1) and
# ``differential_ik_split_controller`` (C2), documented in
# ``docs/archive/SLSQP_CONTROLLER_REFERENCE.md``): both use
# ``face_mask = [0, 0, 0, 1.25, 1.55, 1.25]`` over
# ``[shoulder_pan, shoulder_lift, elbow, wrist_1, wrist_2, wrist_3]`` --
# exactly zero on the three proximal joints, heavily weighted on the wrist
# chain with wrist_2 (the joint that sits at 0 at the transport singularity)
# weighted highest. Normalized here to a 1.0 peak; only the *shape* is taken
# from the legacy controllers (an overall scale is absorbed by
# ``kp_rot_wrist``/``kd_rot_wrist``), not the literal legacy PD gains.
WRIST_ORIENTATION_MASK: np.ndarray = np.array(
    [0.0, 0.0, 0.0, 1.25 / 1.55, 1.0, 1.25 / 1.55], dtype=np.float64
)


def _parse_friction_model(raw: Any) -> Literal["static", "lugre", "karnopp"]:
    """Case-insensitive parse of the ``friction_model`` YAML value.

    Matches the pre-existing (2026-08-01) permissive convention: any value that
    isn't a recognized model name silently falls back to ``"static"`` rather
    than raising, so a typo in a config never breaks loading -- it just leaves
    the historical default behavior in place. Extended 2026-08-02 to recognize
    ``"karnopp"`` alongside the pre-existing ``"lugre"``.
    """
    value = str(raw).lower()
    if value == "lugre":
        return "lugre"
    if value == "karnopp":
        return "karnopp"
    return "static"


@dataclass
class CartesianImpedanceConfig:
    kp_x: float = 25.0
    kd_x: float = 8.0
    kp_y: float = 80.0
    kd_y: float = 15.0
    kp_z: float = 120.0
    kd_z: float = 20.0
    kp_rot: float = 20.0
    kd_rot: float = 5.0
    kp_posture: float = 2.0
    kd_posture: float = 0.5
    kd_joint: float = 0.8
    tau_max_nm: np.ndarray = field(
        default_factory=lambda: np.array([8.0, 8.0, 8.0, 2.5, 2.5, 2.5], dtype=np.float64)
    )
    jacobian_singular_cond_max: float = 1.0e5
    torque_headroom: float = 0.9
    task_resample_factor: float = 0.5
    task_resample_min_scale: float = 1.0 / 16384.0
    task_resample_max_iters: int = 14
    # Operational-space upgrades (P3). Both default off = historical behavior.
    # With shaping on, the PD wrench is interpreted as a desired task-space
    # acceleration and premultiplied by Lambda(q) = (J M^-1 J^T + eps I)^-1.
    # With nullspace_posture on, the posture PD torque is projected into the
    # task nullspace via the dynamically consistent pseudoinverse so it cannot
    # disturb the end-effector. Both need ``mass_matrix`` in the state dict;
    # if absent, M falls back to identity (kinematic pseudoinverse).
    task_space_inertia_shaping: bool = False
    nullspace_posture: bool = False
    lambda_regularization: float = 1.0e-6
    # Diagonal-only Lambda for the wrench shaping step (default off = historical
    # P3 behavior). Root cause of the Z-drift/orientation-coupling ceiling found
    # 2026-07-25: away from the wrist_2=0 singularity, Lambda = (J M^-1 J^T +
    # eps I)^-1 develops large off-diagonal terms (e.g. X-Z), so the shaped
    # wrench's Z-row picks up Lambda[2,0]*Fx even when z_err is ~0 -- the large
    # X-restoring force leaks into a spurious Z command. With this on, only
    # diag(Lambda) is used to shape the wrench (each task-space channel only
    # responds to its own raw-wrench component), eliminating that leak; the
    # nullspace-posture projector still uses the full (undiagonalized) Lambda,
    # since that math needs the true dynamically-consistent pseudoinverse, and
    # it is a separate, unaffected term (measured healthy across the same dx
    # sweep that exposed the wrench-shaping coupling).
    lambda_diagonal_shaping: bool = False
    # Adaptive lambda_regularization (default off = historical behavior: a
    # single static eps == lambda_regularization everywhere). Root cause found
    # 2026-07-25: a static eps cannot be correct both at the singularity and
    # away from it. At the exact wrist_2=0 singularity (cond(J)~5e16) a small
    # eps makes Lambda's diagonal blow up (measured Lambda[3,3] 9.8->88->670 as
    # eps drops 0.1->0.01->0.001) -- eps=0.1 tames that. But away from the
    # singularity (cond(J)~700, well within the well-conditioned range the arm
    # spends most of a transport move in), that same eps=0.1 corrupts the
    # nullspace-posture projector: at eps=0 the projector correctly nulls a
    # representative posture torque's task-space effect (0.0 leak, matching
    # theory for a full-rank task), but at eps=0.1 it leaks a real, sustained
    # 0.074 rad/s^2 task acceleration (0.015-0.05 rad/s^2 per rotational axis)
    # -- a direct, mechanistic explanation for the orientation drift that
    # eventually trips the safety guard at large X-displacement. With this
    # flag on, eps is interpolated in log(cond(J)) space between
    # lambda_regularization_far (used when cond(J) <= lambda_cond_low, i.e.
    # most of a transport move) and lambda_regularization (used unchanged as
    # the near-singularity ceiling when cond(J) >= lambda_cond_high), instead
    # of being a single fixed value -- but ONLY for the nullspace-posture
    # projector's Lambda. A first attempt also scheduled the wrench-shaping
    # Lambda and caused a real regression (joint velocity >3.0 rad/s in
    # previously-trivial cases at cond(J) values far from the singularity,
    # e.g. alpha=0/dx=0.15-0.20): reducing eps amplifies the wrench-shaping
    # Lambda's diagonal in ways the tuned gains were never validated against.
    # Wrench shaping therefore always uses the static lambda_regularization,
    # unaffected by this flag.
    lambda_adaptive_regularization: bool = False
    lambda_regularization_far: float = 1.0e-4
    lambda_cond_low: float = 1.0e4
    lambda_cond_high: float = 1.0e8
    # Wrench-shaping adaptive regularization (default off), DELIBERATELY
    # SEPARATE from lambda_adaptive_regularization above -- added 2026-08-02
    # after root-causing the -45deg Y-drift problem to exactly the failure
    # mode this file's own comment predicted: at this pose, the static
    # lambda_regularization=0.1 (sized for the wrist_2=0 singularity,
    # cond(J)~1e17) badly damages decoupling even when the Jacobian is
    # actually well-conditioned (verified: a unit-X task acceleration through
    # the real wrench-shaping Lambda predicts a_y=-0.27 instead of 0, with
    # eps=0.1, REGARDLESS of whether wrist_2 is at the singularity or offset
    # to 0.2 -- cond(J) 1e17 vs 29 made almost no difference to the broken
    # prediction). The comment above (lines ~130-134) explains why the
    # EXISTING nullspace-only scheduler can't just be pointed at wrench-
    # shaping too: naively reducing eps there (tried at eps_far=1.0e-4, this
    # flag's sibling's default) caused joint-velocity blowup measured at
    # cond(J)~1e3-1e4, well short of any real singularity -- confirmed again
    # here (2026-08-02 sim sweep at the wrist2-offset -45deg pose, cond~29):
    # eps=0.001 dropped Y-drift substantially but tripped max|qd| to 3.05
    # rad/s (guard is 3.0); eps=0.01-0.03 gave real, safe Y-drift improvement
    # (max|qd| 0.28-0.89 rad/s) without the blowup. wrench_lambda_regularization_far
    # defaults to 0.01, NOT 1.0e-4 -- deliberately far more conservative than
    # the nullspace scheduler's far-field value, informed directly by that
    # sweep rather than reused blind. Reuses lambda_cond_low/lambda_cond_high
    # (same log(cond)-space interpolation, see _scheduled_lambda_regularization)
    # rather than duplicating separate thresholds -- at cond(J)~29 (this
    # pose) that's already far below cond_low=1e4 so the schedule returns
    # wrench_lambda_regularization_far unclamped; at cond(J)~1e17 (the
    # original singular pose) it clips to lambda_regularization (0.1,
    # unchanged, safe) -- so enabling this flag without ALSO fixing the
    # wrist_2 singularity falls back to today's already-validated behavior,
    # not something new and unvalidated.
    wrench_lambda_adaptive_regularization: bool = False
    wrench_lambda_regularization_far: float = 0.01
    # Posture re-anchoring (default off = historical behavior). In move+hold
    # trajectories ``_q_rest`` stays the reset pose, so during the hold the
    # posture anchor fights the task force; at a task singularity that force
    # couple pumps unbounded self-motion drift. With this flag on, ``_q_rest``
    # is re-captured once when the x target is reached and the arm has settled
    # (|x_err| <= reanchor_x_tol_m and max|qd| <= reanchor_qd_tol_radps), and
    # re-armed whenever the x target moves on by more than the tolerance.
    posture_reanchor_on_settle: bool = False
    reanchor_x_tol_m: float = 2.0e-3
    reanchor_qd_tol_radps: float = 0.05
    # Wrist-orientation task (default off = historical behavior). Root cause
    # found 2026-07-27/28 (AGENTS.md sec 3, "The ceiling is directional, not
    # just a magnitude limit"): with kp_rot=0 (required -- turning kp_rot back
    # on is unstable near the wrist_2=0 transport singularity through the
    # shared, eps-regularized Lambda-weighted wrench pipeline), orientation is
    # held only as a side effect of the nullspace-posture projector, and that
    # projector's restoring authority is itself asymmetric with wrist_2 sign
    # at the height_alpha=0.5 pose -- not a tunable-gain problem with the
    # existing architecture (a direct kp_posture/kd_posture/kd_joint sweep at
    # the exact failing case barely moved the outcome).
    #
    # This flag adds a SEPARATE joint-space PD torque term that gives
    # orientation its own dedicated authority via the wrist joints only,
    # structurally isolated from the shared Lambda-weighted wrench pipeline
    # that made kp_rot unstable (translated from two pre-torque-lane
    # kinematic controllers that split position and orientation by which
    # joints are responsible for which task -- see WRIST_ORIENTATION_MASK's
    # docstring and archive/legacy_mujoco/controller.py):
    #
    #   tau_orient_wrist = (J_rot.T @ (kp_rot_wrist * e_rot - kd_rot_wrist * omega)) * WRIST_ORIENTATION_MASK
    #
    # Reuses the same orientation_error_vec / omega already computed for the
    # existing (currently zero-gain) kp_rot/kd_rot term -- does not recompute
    # them, and does not touch kp_rot/kd_rot, which remain the historical
    # zero-gain path when this flag is off. This term is summed into the same
    # joint-space bias (tau_damping + tau_posture + ... + gravity) as
    # tau_posture, so it flows through the existing geometric-backtracking
    # and hard-clip logic identically -- no bypass, no special-cased safety
    # path.
    wrist_orientation_task: bool = False
    kp_rot_wrist: float = 0.0
    kd_rot_wrist: float = 0.0
    # Model-based joint-friction feedforward (default off = historical
    # behavior). Added 2026-07-31 to address a real sim-to-real gap: the real
    # UR5e only achieved ~55-72% of a small commanded X displacement with
    # steady-state hold-phase torque that never decayed toward zero -- a
    # friction/stiction signature the (until tonight, frictionless) sim never
    # showed. A same-night sim smoke test confirmed pure proportional gain
    # cannot close this: even with real joint friction added to the MuJoCo
    # model, closed-loop kp_x alone still recovered ~99% of target
    # displacement in sim, far short of the real shortfall -- a disturbance
    # like friction needs either feedforward cancellation or integral action
    # to fully zero out at steady state, not more gain. This term is the
    # feedforward half:
    #
    #   tau_friction_ff = friction_ff_coulomb_nm * tanh(qd / friction_ff_qd_deadband)
    #                     + friction_ff_viscous * qd
    #
    # tanh(qd/eps) is used instead of a hard sign(qd) deliberately: a sign()
    # discontinuity right at qd~=0 -- exactly the hold-phase regime this is
    # meant to fix -- would chatter/oscillate instead of settle. Defaults
    # mirror assets/ur5e_torque/ur5e_torque.xml's own frictionloss/damping
    # values (size3 joints 5.0 Nm / 0.4, size1 joints 1.0 Nm / 0.15) as a
    # starting point, independently tunable since the real robot's true
    # friction need not match the sim model exactly. Summed into the same
    # joint-space bias (tau_damping + tau_posture + ... + gravity) as
    # tau_orient_wrist, so it flows through the existing geometric-
    # backtracking and hard-clip logic identically -- no bypass.
    friction_feedforward: bool = False
    friction_ff_coulomb_nm: np.ndarray = field(
        default_factory=lambda: np.array([5.0, 5.0, 5.0, 1.0, 1.0, 1.0], dtype=np.float64)
    )
    friction_ff_viscous: np.ndarray = field(
        default_factory=lambda: np.array([0.4, 0.4, 0.4, 0.15, 0.15, 0.15], dtype=np.float64)
    )
    # Default 0.05, NOT a smaller value: a same-night sim smoke test found
    # deadband=0.01 produces a genuine closed-loop limit cycle (typical
    # hold-phase |qd| sits ~0.005-0.02 rad/s, right in the tanh term's steep
    # transition region at that setting, acting like a large local negative-
    # damping gain). 0.05 settles cleanly instead -- see
    # config/ur5e_mujoco_torque_osc_tuned_friction_ff.yaml's header for the
    # measured before/after numbers.
    friction_ff_qd_deadband: float = 0.05
    # LuGre dynamic friction feedforward -- opt-in alternative to the static
    # tanh/viscous model above (added 2026-08-01, see
    # docs/hardware/LUGRE_FRICTION_MODEL_PLAN.md for the full design and its
    # own real motivation). Real hardware the same night found the static
    # model does NOT close a stick-slip breakaway failure (guard trip within
    # ~0.2-0.4s, ~0.05-0.09% of commanded displacement achieved) -- if
    # anything the trip happened slightly *earlier* with friction_feedforward
    # active. This makes sense: tanh(qd/deadband) is a pure function of
    # instantaneous qd, so it contributes almost nothing right at the
    # breakaway moment (qd~=0) and has no memory of "how long has this joint
    # been stuck." LuGre's bristle-deflection state z is built to fill
    # exactly that gap.
    #
    # `friction_model` is nested under `friction_feedforward` (only consulted
    # when that bool is True) rather than a parallel flag, so
    # `friction_feedforward: false` unambiguously means "no friction
    # feedforward at all" -- matching every other flag in this file's own
    # pattern of one boolean gating one behavior addition. Default "static"
    # means zero behavior change for every existing config.
    friction_model: Literal["static", "lugre", "karnopp"] = "static"
    # Per-joint LuGre parameters (used only when friction_model == "lugre").
    # No authoritative UR5e-class LuGre parameter table exists (confirmed via
    # a literature pass, see the plan doc sec 1) -- these are placeholders,
    # not a calibrated fit, following the same "reasonable engineering
    # estimate, not a calibrated fit" discipline already used for the static
    # model's own friction_ff_coulomb_nm/friction_ff_viscous defaults.
    #
    # lugre_fc_nm mirrors friction_ff_coulomb_nm exactly (same physical
    # quantity: the Coulomb/kinetic friction floor). lugre_sigma2_nm_s_per_rad
    # mirrors friction_ff_viscous exactly (same role: viscous coefficient).
    # lugre_fs_nm (breakaway peak, must be > Fc) is set to 1.3x Fc per joint
    # -- a modest, explicitly-a-guess multiplier documented in
    # config/ur5e_mujoco_torque_osc_tuned_friction_ff_lugre.yaml's header.
    #
    # lugre_sigma0_nm_per_rad, lugre_sigma1_nm_s_per_rad, and lugre_vs_radps
    # have no static-model analog. Sizing them requires reading this file's
    # own _lugre_step(): note the ODE below is written exactly as specified
    # by the plan (`dz/dt = qd - |qd|*z/g(qd)`, `g` in Nm, NOT divided by
    # sigma0 the way textbook LuGre normalizes it) -- under that literal
    # form, z is bounded within +/-g(qd) (i.e. roughly +/-Fs at low speed)
    # regardless of sigma0, so sigma0 alone sets the torque scale of
    # sigma0*z. sigma0=1.0 keeps that steady-state torque comparable in
    # magnitude to Fc/Fs themselves (a sane, directly comparable anchor to
    # the static model, verified by a standalone numeric check before
    # picking this value -- sigma0=100 would produce a ~15 Nm steady-state
    # term from the same z trajectory, an unphysical jump). sigma1 (bristle
    # micro-damping, shapes breakaway-transient sharpness via the dz/dt term)
    # is set to 2x the matching sigma2/viscous value -- a modest, bounded
    # addition on top of the dominant sigma0*z term, not independently
    # identified (the plan's own sec 4 flags sigma1 as "typically needs
    # either a dedicated stick-slip test or left at a small value and tuned
    # qualitatively," unchanged here -- no such tuning pass was run).
    # vs=0.02 rad/s sits in the plan's own cited Stribeck-velocity sweep
    # range (0.01-0.2 rad/s, plan sec 4 item 2).
    lugre_sigma0_nm_per_rad: np.ndarray = field(
        default_factory=lambda: np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float64)
    )
    lugre_sigma1_nm_s_per_rad: np.ndarray = field(
        default_factory=lambda: np.array([0.8, 0.8, 0.8, 0.3, 0.3, 0.3], dtype=np.float64)
    )
    lugre_sigma2_nm_s_per_rad: np.ndarray = field(
        default_factory=lambda: np.array([0.4, 0.4, 0.4, 0.15, 0.15, 0.15], dtype=np.float64)
    )
    lugre_fc_nm: np.ndarray = field(
        default_factory=lambda: np.array([5.0, 5.0, 5.0, 1.0, 1.0, 1.0], dtype=np.float64)
    )
    lugre_fs_nm: np.ndarray = field(
        default_factory=lambda: np.array([6.5, 6.5, 6.5, 1.3, 1.3, 1.3], dtype=np.float64)
    )
    lugre_vs_radps: np.ndarray = field(
        default_factory=lambda: np.array([0.02, 0.02, 0.02, 0.02, 0.02, 0.02], dtype=np.float64)
    )
    # Karnopp stick-slip friction feedforward -- second opt-in alternative to
    # "static", nested under the same friction_feedforward flag as "lugre"
    # (added 2026-08-02, see docs/status/karnopp_stiction_friction_model_2026-08-02.md
    # for the full evidence and design rationale). Real hardware evidence found the
    # SAME night the LuGre option above landed but not analyzed until now: on a real
    # UR5e trace (config/ur5e_mujoco_torque_osc_tuned_split_base_wrist.yaml,
    # friction_feedforward=true, friction_model left at "static"), wrist_1 sat at
    # |qd| < 1.2e-4 rad/s (encoder-noise-floor stationary, q moved a total of
    # 0.000126 rad) for the ENTIRE run while qdd_residual (controller_core/
    # dynamics_residual.py) grew from ~0 to several rad/s^2 in magnitude, strongly
    # correlated with elapsed time (r=-0.80) and only weakly with qd itself
    # (r=-0.22) -- the signature of a real, growing, unmodeled static-holding
    # torque, not a velocity-dependent effect.
    #
    # This is a real structural gap in the LuGre option above, not just an
    # under-tuned-parameters gap: LuGre's own ODE, dz/dt = qd - |qd|*z/g(qd), has
    # dz/dt == 0 whenever qd == 0 EXACTLY (both terms vanish) -- proven by
    # tests/unit/test_lugre_friction.py::test_lugre_zero_velocity_leaves_z_and_torque_at_zero
    # and independently re-derived here: the ODE's relaxation time constant is
    # g(qd)/|qd|, which diverges as qd -> 0. A joint sitting at machine-precision
    # zero velocity for its entire trace (exactly this real wrist_1 case) can never
    # build meaningful bristle deflection under this ODE, regardless of sigma0/Fc/Fs
    # tuning -- already independently confirmed at a different (much higher, ODE-
    # unit-test-only) qd in docs/status/lugre_friction_feedforward_2026-08-01.md's
    # own "hundreds of seconds" relaxation-time finding. This real trace is a second,
    # independent, real-hardware confirmation of the identical structural limitation.
    #
    # Karnopp (D. Karnopp, "Computer simulation of stick-slip friction in mechanical
    # dynamic systems," ASME J. Dyn. Sys. Meas. Control, 1985) is the classical fix
    # for exactly this regime: a velocity-hysteresis switching model, not a
    # continuously-integrated bristle state. Below a "stuck" velocity band, the
    # joint is assumed not to be truly sliding at all. Above a higher "moving"
    # threshold the joint is treated as sliding and this falls back to a standard
    # kinetic Coulomb+viscous law. The gap between the two thresholds is a
    # hysteresis dead zone (holds the previous stuck/free state) -- the standard
    # fix for the chatter a single-threshold switch would otherwise produce right
    # at the boundary.
    #
    # CORRECTED 2026-08-02 (see _karnopp_step's docstring for the full derivation):
    # an earlier version of the stuck branch tried to "cancel" the net driving
    # torque already computed this cycle (task + damping + posture + orient_wrist
    # + gravity) by re-adding a clipped copy of it here -- but those terms are ALSO
    # summed into tau_bias separately by the caller, so this roughly DOUBLED the
    # commanded torque on any stuck joint (confirmed by direct measurement: full-
    # controller output came out at exactly 2x the "static" model's for an
    # identical stuck state). The stuck branch now contributes zero feedforward.
    # This does NOT close the qd~=0-forever gap LuGre also cannot close -- it only
    # removes a real, measured torque-doubling bug. A genuinely correct fix for
    # that gap (if one is needed at all -- real static friction on the physical
    # robot already self-adjusts to hold a stuck joint without any help; the
    # qdd_residual gap this feature targets reflects an incomplete DYNAMICS MODEL
    # used for prediction, which more commanded torque cannot fix) is unresolved
    # and needs its own separately-validated design, not a quick patch.
    #
    # NOTE: this is an engineering-judgment design choice made this session, not a
    # literature-verified citation pass the way this repo's other adversarially-
    # verified literature reviews were (docs/status/literature_review_dynamics_and_
    # sensor_noise_identification_2026-08-01.md) -- the Karnopp model itself is
    # textbook-standard and not in dispute, but no fresh source-verification pass
    # was run this session to back a specific numeric claim beyond the model's name
    # and year, consistent with this repo's own discipline of not fabricating
    # precise citations.
    #
    # Reuses lugre_fc_nm/lugre_fs_nm (same physical quantities: kinetic Coulomb
    # floor / static breakaway ceiling) and friction_ff_viscous (kinetic viscous
    # coefficient) rather than duplicating config fields for the sliding branch.
    # karnopp_qd_stick_enter_radps/_exit_radps are new: defaults (0.005/0.02 rad/s)
    # are grounded in this same real trace's own per-joint |qd| statistics (median/
    # p90/max), not a pure guess -- wrist_1 (truly stuck) never exceeded 1.2e-4
    # rad/s all run, while shoulder_lift (the joint actually doing the commanded
    # move, in this split_base_wrist_task config) had a median |qd| of 0.034 rad/s
    # and p90 of 0.050 rad/s -- 0.005/0.02 sits well below the "moving" joint's
    # typical speed and well above the "stuck" joint's noise floor, with room for a
    # real hysteresis band. This is still a placeholder in the sense that it has
    # not been calibrated against a dedicated breakaway-velocity measurement (no
    # such measurement exists for this arm, per the LuGre plan doc's own
    # unresolved calibration gap) -- it is grounded in real telemetry, not a
    # calibrated fit.
    karnopp_qd_stick_enter_radps: np.ndarray = field(
        default_factory=lambda: np.array([0.005, 0.005, 0.005, 0.005, 0.005, 0.005], dtype=np.float64)
    )
    karnopp_qd_stick_exit_radps: np.ndarray = field(
        default_factory=lambda: np.array([0.02, 0.02, 0.02, 0.02, 0.02, 0.02], dtype=np.float64)
    )
    # Y-axis integral action (default off = historical behavior: pure kp_y/
    # kd_y PD, ki_y implicitly 0). Added 2026-08-01 to address the -45 deg
    # base-rotation Y-drift failure (AGENTS.md sec 3): diagnosis (see
    # docs/status/neg45_y_axis_diagnosis_and_fix_2026-08-01.md) ruled out
    # geometric backtracking (task_backtrack_scale measured == 1.0 throughout
    # the failing move -- torque never gets close to tau_limit, peak ~4.7% of
    # headroom), the global cond(J) singular_scale (measured == 1.0 throughout
    # at jacobian_singular_cond_max=1e18), kd_joint (0x and 2x both fail
    # identically to baseline), and the wrench-shaping Lambda's off-diagonal
    # X->Y leak (lambda_diagonal_shaping has zero effect on the Y-drift
    # magnitude, matching the already-documented Phase-2 beam-search result).
    # A direct large-multiplier kp_y/kd_y sweep (not previously tried -- prior
    # attempts topped out around 1.6x-1.7x, both live on real hardware and in
    # the staged gain search) found the true behavior is a genuine steady-
    # state trade-off, not a simple insufficient-bandwidth problem: raising
    # kp_y/kd_y by 5x-10x does stop the Y-drift guard trip, but X tracking
    # then stalls at a *steady-state* equilibrium roughly 45-55% short of the
    # commanded X target (confirmed non-transient -- unchanged from a 1s move
    # to a 3s move + 2s hold) -- i.e. proportional/derivative gain on Y can
    # only trade X authority for Y authority at this pose, never satisfy both,
    # consistent with a persistent kinematic/dynamic X-Y coupling disturbance
    # that a P/D term structurally cannot null at steady state (the same
    # class of problem friction_feedforward already solves for joint
    # friction, just on a different axis and with an integral rather than a
    # feedforward term, since the coupling's exact model isn't known the way
    # friction's is). This flag adds a standard clamped integral term to the
    # existing Fy PD law:
    #
    #   Fy = kp_y*y_err - kd_y*v_y + ki_y*y_integral
    #   y_integral(t) = clip(y_integral(t-1) + y_err*dt, -y_integral_limit_m_s, +y_integral_limit_m_s)
    #
    # Anti-windup clamp mirrors controller_core/lqr_controller.py's existing
    # `_x_integral` pattern (accumulate, then clip every step -- never clip
    # only occasionally). y_integral resets to 0 in reset_from_state(), same
    # lifecycle as _q_rest/_quat0. Not added to _SCHEDULABLE_GAIN_FIELDS
    # (matches kp_rot_wrist/kd_rot_wrist precedent: config/gain-override only,
    # not part of the live gain-scheduling contract tested against
    # transport_metrics.GAIN_FIELDS).
    y_integral_action: bool = False
    ki_y: float = 0.0
    y_integral_limit_m_s: float = 0.02
    # X-axis integral action (default off = historical behavior: pure kp_x/
    # kd_x PD, ki_x implicitly 0). Added 2026-08-02 to address a real, directly
    # measured hold-phase undershoot: direct_torque_20260802_190759 (wrist2-
    # offset pose, split_base_wrist_task) showed shoulder_lift torque ramp to
    # a steady -6.08 Nm and x_error plateau at 0.0082m out of a 0.02m target
    # for the entire 2s hold, completely flat -- the textbook signature of
    # proportional control settling at a stable equilibrium once its own
    # error-proportional output drops below the joint's real static-friction
    # breakaway threshold (Fs), not a transient. friction_feedforward's
    # static/lugre/karnopp models are all velocity-driven and vanish exactly
    # as qd->0 approaching this equilibrium -- see karnopp_qd_stick_enter_radps's
    # docstring for why a torque-driven feedforward alternative isn't a
    # defensible fix either. Integral action is the classical, model-free
    # answer to a steady-state error under any roughly-constant disturbance:
    # as long as x_err stays nonzero, the integral term keeps growing without
    # bound until it exceeds Fs and pushes through, however small the
    # residual gap is -- no friction model or parameter accuracy required.
    #
    # Directly mirrors y_integral_action's already-validated pattern (added
    # 2026-08-01 for the -45 deg pose's Y-drift coupling -- a DIFFERENT,
    # structural disturbance, not friction; that flag's own real-hardware
    # test found "zero measurable effect at a gentle dose" for THAT problem,
    # which does not predict anything about ki_x's effectiveness here --
    # genuine stiction is the textbook integral-action use case that Y-drift
    # was not). Same anti-windup clamp shape (accumulate every step, then
    # clip -- never clip only occasionally), same reset_from_state()
    # lifecycle, same "config/gain-override only, not in the live gain-
    # scheduling contract" precedent as kp_rot_wrist/kd_rot_wrist and ki_y.
    #
    #   Fx = kp_x*x_err + kd_x*(x_vel_des - v_x) + ki_x*x_integral
    #   x_integral(t) = clip(x_integral(t-1) + x_err*dt, -x_integral_limit_m_s, +x_integral_limit_m_s)
    #
    # CORRECTED 2026-08-02, same day, direct sim testing before any real-
    # hardware attempt: accumulation only happens while |x_vel_des| is near
    # zero (i.e. during hold, not an active move) -- see _karnopp_step-
    # adjacent reasoning in compute() at the accumulation site for why. A
    # first version accumulated unconditionally and, tested in sim at the
    # exact real-evidence scenario, wound up against the large transient
    # move-phase tracking error and overshot the target during hold (x_error
    # measured crossing zero and growing negative for the rest of the hold)
    # -- a real closed-loop windup bug, not present in the original
    # motivating case (a flat HOLD-phase plateau), fixed by restricting
    # accumulation to exactly the phase this feature targets.
    x_integral_action: bool = False
    ki_x: float = 0.0
    x_integral_limit_m_s: float = 0.02
    # Y-coupling feedforward (default off = historical behavior). Added
    # 2026-08-02, a different lever on the same -45 deg pose Y-drift problem
    # y_integral_action targets above -- that flag is FEEDBACK (reacts to
    # measured y_err) and was already found to trade off against X-tracking
    # authority at any dose large enough to matter (5-10x baseline kp_y/kd_y
    # stops the guard trip but costs a 45-55% X-tracking shortfall,
    # unimproved by 3x the move duration -- see AGENTS.md sec 3's -45 deg
    # findings). This flag is FEEDFORWARD instead: it biases the Y TARGET
    # itself as a function of the COMMANDED x_des, open-loop, not reactive to
    # y_err at all. Motivation: the real-hardware trip signature was the TCP
    # moving in a near-45 deg diagonal (X and Y displacement nearly equal),
    # and a controlled sim dose-response measured Y-drift growing roughly
    # linearly with commanded X displacement (slope ~0.65-0.73, see AGENTS.md).
    # If that coupling is a repeatable, structural function of the COMMANDED
    # trajectory (not measurement noise or a stochastic disturbance), a
    # feedforward target bias lets the EXISTING baseline kp_y/kd_y do the
    # correcting against a much smaller residual error instead of the full
    # drift -- same gain, easier job, rather than more gain fighting harder.
    # NOT yet validated -- y_coupling_gain's default (0.7) is the measured
    # dose-response slope, not a value confirmed to reduce real Y-drift; the
    # sign and magnitude both need a real sim A/B before trusting this.
    #
    #   y_des = y_des - y_coupling_gain * (x_des - x0)     [applied once,
    #   after x_des/y_des are finalized by the hold/tracking branches above,
    #   before x_err/y_err are computed]
    y_coupling_feedforward: bool = False
    y_coupling_gain: float = 0.7
    # Base/wrist task split (default off = historical behavior). Added
    # 2026-08-01 after a real-hardware TCP-accel guard trip at the
    # height_alpha=0.5 pose using accel_duration_scurve at a modest
    # target_accel (0.005-0.02 m/s^2), root-caused directly from real
    # trace.jsonl: the arm sits almost exactly at wrist_2=0 (a UR5e
    # kinematic singularity present at every height_alpha,
    # hardware/poses.py::q_for_height_alpha), where the FULL 6x6 J's
    # cond() (used unconditionally above for singular_scale/adaptive-eps
    # scheduling, and for task_space_inertia_shaping's Lambda when that
    # flag is on) oscillates 5-10x cycle-to-cycle purely from being
    # parked there -- lambda_diagonal_shaping/lambda_adaptive_regularization
    # do not help (verified on real hardware, if anything slightly worse):
    # neither touches cond(J) itself, an upstream geometric property of J
    # at that exact configuration that no downstream Lambda regularization
    # fixes.
    #
    # Numeric evidence (docs/status/split_base_wrist_impedance_2026-08-01.md,
    # exact failure pose q=[0, -0.835398, -1.2, -0.985398, 0, 0] =
    # hardware/poses.py::HEIGHT_ALPHA_0_5_Q): cond(full 6x6 J) ~= 7.28e16
    # (numerically singular); cond(3x3 position-rows x base-joint-cols
    # [shoulder_pan, shoulder_lift, elbow]) ~= 7.8 (well conditioned);
    # cond(3x3 rotation-rows x wrist-joint-cols [wrist_1, wrist_2,
    # wrist_3]) == inf (exactly rank-deficient, smallest singular value
    # 0.0 -- wrist_1/wrist_3 axes align when wrist_2=0, the textbook UR
    # wrist gimbal lock). This REFUTES a naive symmetric design (base-only
    # translation impedance + wrist-only orientation impedance): the
    # wrist-only rotation sub-Jacobian is exactly as singular as the
    # shared pipeline it would replace, not better.
    #
    # Design driven by that evidence: with this flag on, the position
    # task (Fx, Fy, Fz) is mapped through a reduced Jacobian that zeroes
    # the wrist columns (J_task[:, 0:3] = J[0:3, 0:3], J_task[:, 3:6] = 0)
    # -- structurally, translation-task torque can never route through
    # wrist_2 (or any wrist joint) at all, regardless of how ill-
    # conditioned the full J is there. The rotational wrench (kp_rot/
    # kd_rot term) is dropped from this task-wrench pipeline entirely
    # (kept only in the diagnostic `wrench` output field) rather than
    # routed through the singular wrist-only submatrix. Orientation
    # instead stays held by the EXISTING, already-validated
    # nullspace_posture mechanism -- but its projector is recomputed
    # against this same reduced task Jacobian (so it sees a rank-3 task
    # with an intact nullspace for base AND wrist redundancy) instead of
    # the near-singular full 6x6 one -- and, if wrist_orientation_task is
    # also enabled, that separate joint-space PD path (no matrix
    # inversion, so it cannot itself blow up from ill-conditioning).
    #
    # Implementation note: every downstream computation that currently
    # uses the full J (wrench-shaping Lambda, the nullspace projector,
    # cond()-based singular_scale/adaptive-eps scheduling, the
    # jacobian_cond trace field) is rewritten in terms of a local
    # `J_task`/`wrench_task`/`cond_task` that equal J/wrench/cond exactly
    # when this flag is off (J_task = J is a 6x6 identity substitution),
    # so the flag-off path is unchanged arithmetic, not just
    # coincidentally equal output -- verified byte-identical in
    # tests/unit/test_split_base_wrist_task.py.
    split_base_wrist_task: bool = False
    # Acceleration feedforward (default off = historical behavior: pure PD on
    # position+velocity error, Fx = kp_x*x_err + kd_x*(x_vel_des - vx)).
    # Added 2026-08-01 (see docs/status/acceleration_feedforward_2026-08-01.md):
    # a pure PD law only ever reacts to tracking error that has already
    # appeared, which is part of why real hardware traces show tracking lag
    # and per-cycle jitter rather than smooth motion. The trajectory
    # generators for `accel_duration_triangular`/`accel_duration_scurve` (and,
    # via a closed-form second derivative, `min_jerk`/`min_jerk_move_hold`)
    # already know the reference acceleration analytically
    # (simulation/ur5e_mujoco_torque.py::x_profile_accel) -- this flag adds a
    # mass-weighted feedforward term built from that reference directly onto
    # the task wrench, alongside (not instead of) the existing PD term:
    #
    #   wrench_task[axis] += Lambda_diag[axis] * target_<axis>_accel
    #
    # where Lambda_diag is the diagonal of the SAME task-space inertia matrix
    # Lambda = (J_task M^-1 J_task^T + eps I)^-1 this file already builds for
    # task_space_inertia_shaping/nullspace_posture (reused, not recomputed --
    # this flag alone is now also a trigger for that block to run). Only the
    # X axis has a real reference today (target_x_accel, threaded through
    # RobotState/as_impedance_robot_state); target_y_accel/target_z_accel
    # default to 0.0 via a plain dict lookup so wiring them up later (no
    # trajectory generator produces them yet -- tonight's real use is X-only)
    # is a one-line addition here, not a redesign.
    #
    # Deliberately a graceful no-op, not a hard error, when the state doesn't
    # actually carry a real mass matrix (mass_matrix_provided False -- e.g. a
    # caller/dynamics_source that never supplies one): falling back to the
    # existing shaping/nullspace code's identity-matrix M fallback here would
    # silently feed a fake ~1kg effective mass into the feedforward term,
    # which is exactly the "silently produce wrong torques" failure mode this
    # flag must not have. `acceleration_feedforward_active` in the output
    # reports whether the term was actually applied this cycle so a caller
    # (or a run record) can tell a genuine no-op apart from a working
    # feedforward that happens to see zero commanded acceleration.
    acceleration_feedforward: bool = False

    @classmethod
    def from_controller_yaml_section(cls, ctrl: dict) -> "CartesianImpedanceConfig":
        gains = ctrl.get("gains", {}) or {}
        mode = str(ctrl.get("torque_limits_mode", "initial")).lower()
        lim_key = (
            "torque_limits_initial"
            if mode == "initial"
            else "torque_limits_after_stable"
        )
        lim_dict = ctrl.get(lim_key, {}) or {}
        tau_list = [float(lim_dict[name]) for name in JOINT_NAME_ORDER]
        tm = np.asarray(tau_list, dtype=np.float64)
        return cls(
            kp_x=float(gains.get("kp_x", 25.0)),
            kd_x=float(gains.get("kd_x", 8.0)),
            kp_y=float(gains.get("kp_y", 80.0)),
            kd_y=float(gains.get("kd_y", 15.0)),
            kp_z=float(gains.get("kp_z", 120.0)),
            kd_z=float(gains.get("kd_z", 20.0)),
            kp_rot=float(gains.get("kp_rot", 20.0)),
            kd_rot=float(gains.get("kd_rot", 5.0)),
            kp_posture=float(gains.get("kp_posture", 2.0)),
            kd_posture=float(gains.get("kd_posture", 0.5)),
            kd_joint=float(gains.get("kd_joint", 0.8)),
            tau_max_nm=tm,
            jacobian_singular_cond_max=float(
                ctrl.get("jacobian_singular_cond_max", 1.0e5)
            ),
            torque_headroom=float(ctrl.get("torque_headroom", 0.9)),
            task_resample_factor=float(ctrl.get("task_resample_factor", 0.5)),
            task_resample_min_scale=float(ctrl.get("task_resample_min_scale", 1.0 / 16384.0)),
            task_resample_max_iters=int(ctrl.get("task_resample_max_iters", 14)),
            task_space_inertia_shaping=bool(ctrl.get("task_space_inertia_shaping", False)),
            nullspace_posture=bool(ctrl.get("nullspace_posture", False)),
            lambda_regularization=float(ctrl.get("lambda_regularization", 1.0e-6)),
            lambda_diagonal_shaping=bool(ctrl.get("lambda_diagonal_shaping", False)),
            lambda_adaptive_regularization=bool(ctrl.get("lambda_adaptive_regularization", False)),
            lambda_regularization_far=float(ctrl.get("lambda_regularization_far", 1.0e-4)),
            lambda_cond_low=float(ctrl.get("lambda_cond_low", 1.0e4)),
            lambda_cond_high=float(ctrl.get("lambda_cond_high", 1.0e8)),
            wrench_lambda_adaptive_regularization=bool(ctrl.get("wrench_lambda_adaptive_regularization", False)),
            wrench_lambda_regularization_far=float(ctrl.get("wrench_lambda_regularization_far", 0.01)),
            posture_reanchor_on_settle=bool(ctrl.get("posture_reanchor_on_settle", False)),
            reanchor_x_tol_m=float(ctrl.get("reanchor_x_tol_m", 2.0e-3)),
            reanchor_qd_tol_radps=float(ctrl.get("reanchor_qd_tol_radps", 0.05)),
            wrist_orientation_task=bool(ctrl.get("wrist_orientation_task", False)),
            kp_rot_wrist=float(gains.get("kp_rot_wrist", 0.0)),
            kd_rot_wrist=float(gains.get("kd_rot_wrist", 0.0)),
            friction_feedforward=bool(ctrl.get("friction_feedforward", False)),
            friction_ff_coulomb_nm=(
                np.array(
                    [float(ctrl["friction_ff_coulomb_nm"][name]) for name in JOINT_NAME_ORDER],
                    dtype=np.float64,
                )
                if "friction_ff_coulomb_nm" in ctrl
                else np.array([5.0, 5.0, 5.0, 1.0, 1.0, 1.0], dtype=np.float64)
            ),
            friction_ff_viscous=(
                np.array(
                    [float(ctrl["friction_ff_viscous"][name]) for name in JOINT_NAME_ORDER],
                    dtype=np.float64,
                )
                if "friction_ff_viscous" in ctrl
                else np.array([0.4, 0.4, 0.4, 0.15, 0.15, 0.15], dtype=np.float64)
            ),
            friction_ff_qd_deadband=float(ctrl.get("friction_ff_qd_deadband", 0.05)),
            friction_model=_parse_friction_model(ctrl.get("friction_model", "static")),
            lugre_sigma0_nm_per_rad=(
                np.array(
                    [float(ctrl["lugre_sigma0_nm_per_rad"][name]) for name in JOINT_NAME_ORDER],
                    dtype=np.float64,
                )
                if "lugre_sigma0_nm_per_rad" in ctrl
                else np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float64)
            ),
            lugre_sigma1_nm_s_per_rad=(
                np.array(
                    [float(ctrl["lugre_sigma1_nm_s_per_rad"][name]) for name in JOINT_NAME_ORDER],
                    dtype=np.float64,
                )
                if "lugre_sigma1_nm_s_per_rad" in ctrl
                else np.array([0.8, 0.8, 0.8, 0.3, 0.3, 0.3], dtype=np.float64)
            ),
            lugre_sigma2_nm_s_per_rad=(
                np.array(
                    [float(ctrl["lugre_sigma2_nm_s_per_rad"][name]) for name in JOINT_NAME_ORDER],
                    dtype=np.float64,
                )
                if "lugre_sigma2_nm_s_per_rad" in ctrl
                else np.array([0.4, 0.4, 0.4, 0.15, 0.15, 0.15], dtype=np.float64)
            ),
            lugre_fc_nm=(
                np.array(
                    [float(ctrl["lugre_fc_nm"][name]) for name in JOINT_NAME_ORDER],
                    dtype=np.float64,
                )
                if "lugre_fc_nm" in ctrl
                else np.array([5.0, 5.0, 5.0, 1.0, 1.0, 1.0], dtype=np.float64)
            ),
            lugre_fs_nm=(
                np.array(
                    [float(ctrl["lugre_fs_nm"][name]) for name in JOINT_NAME_ORDER],
                    dtype=np.float64,
                )
                if "lugre_fs_nm" in ctrl
                else np.array([6.5, 6.5, 6.5, 1.3, 1.3, 1.3], dtype=np.float64)
            ),
            lugre_vs_radps=(
                np.array(
                    [float(ctrl["lugre_vs_radps"][name]) for name in JOINT_NAME_ORDER],
                    dtype=np.float64,
                )
                if "lugre_vs_radps" in ctrl
                else np.array([0.02, 0.02, 0.02, 0.02, 0.02, 0.02], dtype=np.float64)
            ),
            karnopp_qd_stick_enter_radps=(
                np.array(
                    [float(ctrl["karnopp_qd_stick_enter_radps"][name]) for name in JOINT_NAME_ORDER],
                    dtype=np.float64,
                )
                if "karnopp_qd_stick_enter_radps" in ctrl
                else np.array([0.005, 0.005, 0.005, 0.005, 0.005, 0.005], dtype=np.float64)
            ),
            karnopp_qd_stick_exit_radps=(
                np.array(
                    [float(ctrl["karnopp_qd_stick_exit_radps"][name]) for name in JOINT_NAME_ORDER],
                    dtype=np.float64,
                )
                if "karnopp_qd_stick_exit_radps" in ctrl
                else np.array([0.02, 0.02, 0.02, 0.02, 0.02, 0.02], dtype=np.float64)
            ),
            y_integral_action=bool(ctrl.get("y_integral_action", False)),
            ki_y=float(gains.get("ki_y", 0.0)),
            y_integral_limit_m_s=float(ctrl.get("y_integral_limit_m_s", 0.02)),
            x_integral_action=bool(ctrl.get("x_integral_action", False)),
            ki_x=float(gains.get("ki_x", 0.0)),
            x_integral_limit_m_s=float(ctrl.get("x_integral_limit_m_s", 0.02)),
            y_coupling_feedforward=bool(ctrl.get("y_coupling_feedforward", False)),
            y_coupling_gain=float(ctrl.get("y_coupling_gain", 0.7)),
            split_base_wrist_task=bool(ctrl.get("split_base_wrist_task", False)),
            acceleration_feedforward=bool(ctrl.get("acceleration_feedforward", False)),
        )


@dataclass
class CartesianImpedanceOutput:
    tau: np.ndarray
    tau_preclip: np.ndarray
    wrench: np.ndarray
    tau_task_nominal: np.ndarray
    tau_task: np.ndarray
    tau_damping: np.ndarray
    tau_posture: np.ndarray
    tau_orient_wrist: np.ndarray
    tau_friction_ff: np.ndarray
    tau_gravity: np.ndarray
    tau_saturated: np.ndarray
    jacobian_cond: float
    singular_scale: float
    task_backtrack_scale: float
    task_scale: float
    task_backtrack_iters: int
    task_feasible: bool
    x_error: float
    y_error: float
    z_error: float
    orientation_error_vec: np.ndarray
    orientation_error_norm: float
    inertia_shaping_active: bool = False
    lambda_diagonal_shaping_active: bool = False
    lambda_adaptive_regularization_active: bool = False
    lambda_regularization_effective: float = 0.0
    nullspace_posture_active: bool = False
    mass_matrix_provided: bool = False
    posture_reanchored: bool = False
    wrist_orientation_task_active: bool = False
    friction_feedforward_active: bool = False
    friction_model_used: str = "static"
    friction_z: np.ndarray = field(default_factory=lambda: np.zeros(6, dtype=np.float64))
    friction_karnopp_stuck: np.ndarray = field(default_factory=lambda: np.zeros(6, dtype=np.float64))
    y_integral_action_active: bool = False
    y_integral_value: float = 0.0
    x_integral_action_active: bool = False
    x_integral_value: float = 0.0
    split_base_wrist_task_active: bool = False
    acceleration_feedforward_active: bool = False
    wrench_accel_ff: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))


class XAxisCartesianImpedanceController:
    """Full 6D Cartesian impedance + posture + joint damping (+ optional gravity).

    The task-space wrench is mapped through ``J.T`` and then backtracked if the
    resulting joint torques exceed the configured headroom around the per-joint
    torque limits.
    """

    #: Gain fields a gain-scheduling policy (or any other caller) may update
    #: mid-episode via ``set_gains()``. Must stay in sync with
    #: ``transport_metrics.GAIN_FIELDS`` -- a cross-module test asserts this.
    _SCHEDULABLE_GAIN_FIELDS: tuple[str, ...] = (
        "kp_x", "kd_x", "kp_y", "kd_y", "kp_z", "kd_z",
        "kp_rot", "kd_rot", "kp_posture", "kd_posture", "kd_joint",
    )

    def __init__(self, config: CartesianImpedanceConfig) -> None:
        self.cfg = config
        self._initialized = False
        self._hold_reference_initialized = False
        self._x0 = 0.0
        self._y0 = 0.0
        self._z0 = 0.0
        self._quat0 = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        self._q_rest = np.zeros(6, dtype=np.float64)
        self._posture_reanchored = False
        self._x_des_at_anchor = 0.0
        self._y_integral = 0.0
        self._x_integral = 0.0
        # LuGre bristle-deflection state, one scalar per joint. Genuinely
        # stateful (integrated over time), same lifecycle class as
        # _q_rest/_quat0/_y_integral: zeroed here and in reset_from_state(),
        # but deliberately NOT touched by set_gains() -- a gain change must
        # never wipe accumulated stiction state mid-episode (matches
        # set_gains()'s own documented contract).
        self._friction_z = np.zeros(6, dtype=np.float64)
        # Karnopp stick-slip hysteresis latch, one bool per joint. Same lifecycle
        # class as _friction_z: persistent across compute() calls (the hysteresis
        # IS the state -- see karnopp_qd_stick_enter_radps's docstring), zeroed
        # (here: reset to "stuck") in reset_from_state(), not touched by
        # set_gains(). Initialized True (stuck): a fresh episode typically starts
        # from a held/rest pose, and the hysteresis self-corrects within a cycle
        # or two regardless of the initial guess.
        self._karnopp_stuck = np.ones(6, dtype=bool)

    def reset_from_state(self, state: dict[str, Any]) -> None:
        st = as_impedance_robot_state(state)
        ee = np.asarray(st["ee_pos"], dtype=np.float64).reshape(3)
        self._x0 = float(ee[0])
        self._y0 = float(ee[1])
        self._z0 = float(ee[2])
        self._quat0 = np.asarray(st["ee_quat"], dtype=np.float64).reshape(4).copy()
        self._q_rest = np.asarray(st["q"], dtype=np.float64).reshape(6).copy()
        self._hold_reference_initialized = False
        self._posture_reanchored = False
        self._x_des_at_anchor = float(np.asarray(st["ee_pos"], dtype=np.float64).reshape(3)[0])
        self._y_integral = 0.0
        self._x_integral = 0.0
        self._friction_z = np.zeros(6, dtype=np.float64)
        self._karnopp_stuck = np.ones(6, dtype=bool)
        self._initialized = True

    @property
    def initialized(self) -> bool:
        return self._initialized

    def set_gains(self, gains: dict[str, float]) -> None:
        """Overwrite a subset of the 11 scheduled gain fields on ``self.cfg`` in place.

        For a gain-scheduling policy that calls this every step. Does not
        touch ``tau_max_nm``, the P3 flags, or any instance state (``_q_rest``,
        posture-reanchor bookkeeping, hold reference) -- a gain change must
        never reset the posture anchor or hold reference mid-episode.
        """
        unknown = set(gains) - set(self._SCHEDULABLE_GAIN_FIELDS)
        if unknown:
            raise ValueError(f"set_gains got unknown gain field(s): {sorted(unknown)}")
        validated: dict[str, float] = {}
        for key, value in gains.items():
            value = float(value)
            if not np.isfinite(value):
                raise ValueError(f"set_gains: {key} must be finite, got {value!r}")
            validated[key] = value
        for key, value in validated.items():
            setattr(self.cfg, key, value)

    @staticmethod
    def _torque_within_limits(tau: np.ndarray, limit: np.ndarray) -> bool:
        tau = np.asarray(tau, dtype=np.float64).reshape(6)
        limit = np.asarray(limit, dtype=np.float64).reshape(6)
        return bool(np.all(np.abs(tau) <= limit + 1e-12))

    def _scheduled_lambda_regularization(self, cond: float) -> float:
        """Interpolate eps in log(cond(J)) space between the far-field value
        (used when the task is well-conditioned) and ``lambda_regularization``
        (used unchanged as the near-singularity ceiling)."""
        eps_far = max(float(self.cfg.lambda_regularization_far), 0.0)
        eps_near = max(float(self.cfg.lambda_regularization), 0.0)
        cond_low = max(float(self.cfg.lambda_cond_low), 1.0)
        cond_high = max(float(self.cfg.lambda_cond_high), cond_low * (1.0 + 1e-9))
        cond = max(float(cond), 1.0)
        log_frac = (np.log(cond) - np.log(cond_low)) / (np.log(cond_high) - np.log(cond_low))
        log_frac = float(np.clip(log_frac, 0.0, 1.0))
        return eps_far + log_frac * (eps_near - eps_far)

    def _scheduled_wrench_lambda_regularization(self, cond: float) -> float:
        """Same log(cond(J)) interpolation as _scheduled_lambda_regularization,
        applied to the wrench-shaping Lambda instead of the nullspace
        projector's -- see wrench_lambda_adaptive_regularization's docstring
        for why these are kept as two separate schedulers, not one shared
        one."""
        eps_far = max(float(self.cfg.wrench_lambda_regularization_far), 0.0)
        eps_near = max(float(self.cfg.lambda_regularization), 0.0)
        cond_low = max(float(self.cfg.lambda_cond_low), 1.0)
        cond_high = max(float(self.cfg.lambda_cond_high), cond_low * (1.0 + 1e-9))
        cond = max(float(cond), 1.0)
        log_frac = (np.log(cond) - np.log(cond_low)) / (np.log(cond_high) - np.log(cond_low))
        log_frac = float(np.clip(log_frac, 0.0, 1.0))
        return eps_far + log_frac * (eps_near - eps_far)

    def _lugre_step(self, qd: np.ndarray, dt: float) -> np.ndarray:
        """One explicit-Euler update of the LuGre bristle-deflection state.

        Implements the plan's equations exactly
        (docs/hardware/LUGRE_FRICTION_MODEL_PLAN.md sec 1/3.2):

            dz/dt = qd - |qd| * z / g(qd)
            tau_friction = sigma0 * z + sigma1 * dz/dt + sigma2 * qd
            g(qd) = Fc + (Fs - Fc) * exp(-(qd/vs)^2)

        Mutates ``self._friction_z`` in place (persistent per-joint state,
        see reset_from_state()/__init__()). Numerical-stability note (plan
        sec 3.2): explicit Euler on this ODE is only conditionally stable;
        at 500 Hz (dt=0.002s) with physically realistic qd this is expected
        to be fine, but the sim-side validation sweep must check for a
        growing, non-decaying z trace at hold, or a |qd| guard trip that
        wasn't there under the static model -- not special-cased here
        speculatively.
        """
        sigma0 = np.asarray(self.cfg.lugre_sigma0_nm_per_rad, dtype=np.float64).reshape(6)
        sigma1 = np.asarray(self.cfg.lugre_sigma1_nm_s_per_rad, dtype=np.float64).reshape(6)
        sigma2 = np.asarray(self.cfg.lugre_sigma2_nm_s_per_rad, dtype=np.float64).reshape(6)
        fc = np.asarray(self.cfg.lugre_fc_nm, dtype=np.float64).reshape(6)
        fs = np.asarray(self.cfg.lugre_fs_nm, dtype=np.float64).reshape(6)
        vs = np.maximum(np.asarray(self.cfg.lugre_vs_radps, dtype=np.float64).reshape(6), 1e-9)
        g = fc + (fs - fc) * np.exp(-((qd / vs) ** 2))
        g_safe = np.maximum(g, 1e-9)
        z_dot = qd - np.abs(qd) * self._friction_z / g_safe
        self._friction_z = self._friction_z + dt * z_dot
        return sigma0 * self._friction_z + sigma1 * z_dot + sigma2 * qd

    def _karnopp_step(self, qd: np.ndarray, driving_torque: np.ndarray) -> np.ndarray:
        """Karnopp (1985) stick-slip switching friction feedforward.

        See ``karnopp_qd_stick_enter_radps``'s docstring on ``CartesianImpedanceConfig``
        for the full derivation and real-hardware evidence motivating this. Two
        regimes, selected per joint via a velocity hysteresis latch
        (``self._karnopp_stuck``, mutated in place -- persistent state, same
        lifecycle as ``self._friction_z``):

          - Stuck (``|qd|`` below ``karnopp_qd_stick_enter_radps``, or already
            latched stuck and not yet above ``karnopp_qd_stick_exit_radps``):
            **contributes zero feedforward torque.** An earlier version of this
            method set ``tau_stuck = clip(driving_torque, -fs, fs)`` -- intended
            to "cancel" driving_torque, but ``driving_torque`` is built from
            terms (``tau_task_nominal``, ``tau_damping``, ``tau_posture``,
            ``tau_orient_wrist``, ``g``) that are ALSO summed into ``tau_bias``
            separately by the caller, so adding a second (near-)copy of them
            here roughly DOUBLED the commanded torque on any stuck joint --
            confirmed by direct measurement (2026-08-02): full-controller
            output with friction_model="karnopp" came out at exactly 2x the
            "static" model's output for an identical stuck-joint state. Beyond
            that correctness bug, the underlying premise doesn't hold up either:
            real static friction on the physical robot already self-adjusts to
            cancel whatever net torque is applied, up to its real breakaway
            limit -- that IS what "stuck" means -- so sending additional
            feedforward torque changes neither the robot's real behavior (real
            friction just absorbs more) nor the qdd_residual diagnostic gap
            that motivated this feature (that gap reflects an incomplete rigid-
            body DYNAMICS MODEL used for prediction, not insufficient commanded
            torque; fixing it needs a friction-aware model for the predictor,
            not more feedforward torque from the controller). Zero is the only
            value defensible without a fresh, separately-validated design for
            genuine breakaway assistance -- see
            docs/status/karnopp_stiction_friction_model_2026-08-02.md for the
            full trace and reasoning. With this branch at zero, "karnopp"
            reduces to "static"'s own qd==0 behavior for a stuck joint --
            no worse, but no better either; the sliding branch below is
            unaffected and still provides real value.
          - Sliding (latched free, ``|qd|`` above ``karnopp_qd_stick_exit_radps``):
            standard kinetic Coulomb ``lugre_fc_nm`` + viscous
            ``friction_ff_viscous``, same physical quantities and sign
            convention as the static model's own sliding-regime behavior.

        No time integration/ODE is needed for this model (unlike LuGre) -- the
        only state is which regime each joint is latched into, which needs no
        ``dt``. ``driving_torque`` is accepted for interface stability (the
        caller still computes and passes it) but is no longer read here.
        """
        qd = np.asarray(qd, dtype=np.float64).reshape(6)
        driving_torque = np.asarray(driving_torque, dtype=np.float64).reshape(6)
        del driving_torque  # unused -- see docstring; kept as a parameter for interface stability
        enter = np.asarray(self.cfg.karnopp_qd_stick_enter_radps, dtype=np.float64).reshape(6)
        exit_threshold = np.asarray(self.cfg.karnopp_qd_stick_exit_radps, dtype=np.float64).reshape(6)
        fc = np.asarray(self.cfg.lugre_fc_nm, dtype=np.float64).reshape(6)
        viscous = np.asarray(self.cfg.friction_ff_viscous, dtype=np.float64).reshape(6)
        abs_qd = np.abs(qd)

        # Hysteresis update: stuck->free only above exit_threshold, free->stuck
        # only below enter -- the dead zone between the two thresholds holds
        # whatever the previous latch state was, preventing chatter right at a
        # single boundary (the standard fix for a naive one-threshold switch).
        became_free = self._karnopp_stuck & (abs_qd > exit_threshold)
        became_stuck = (~self._karnopp_stuck) & (abs_qd < enter)
        stuck = np.where(became_free, False, self._karnopp_stuck)
        stuck = np.where(became_stuck, True, stuck)
        self._karnopp_stuck = stuck

        tau_stuck = np.zeros(6, dtype=np.float64)
        tau_slide = fc * np.sign(qd) + viscous * qd
        return np.where(stuck, tau_stuck, tau_slide)

    def _backtrack_task_scale(
        self,
        tau_nominal: np.ndarray,
        tau_limit: np.ndarray,
    ) -> tuple[float, np.ndarray, int, bool]:
        """Geometrically shrink the full torque candidate until it fits the limit box."""
        tau_nominal = np.asarray(tau_nominal, dtype=np.float64).reshape(6)
        tau_limit = np.asarray(tau_limit, dtype=np.float64).reshape(6)

        resample_factor = float(self.cfg.task_resample_factor)
        if not np.isfinite(resample_factor) or resample_factor <= 0.0 or resample_factor >= 1.0:
            resample_factor = 0.5

        min_scale = max(float(self.cfg.task_resample_min_scale), 0.0)
        max_iters = max(int(self.cfg.task_resample_max_iters), 0)

        task_scale = 1.0
        tau_candidate = task_scale * tau_nominal
        feasible = self._torque_within_limits(tau_candidate, tau_limit)
        iters = 0

        while (not feasible) and (iters < max_iters) and (task_scale > min_scale + 1e-12):
            next_scale = max(task_scale * resample_factor, min_scale)
            if next_scale >= task_scale - 1e-12:
                break
            next_candidate = next_scale * tau_nominal
            if np.allclose(next_candidate, tau_candidate, rtol=0.0, atol=1e-12):
                task_scale = next_scale
                tau_candidate = next_candidate
                break
            task_scale = next_scale
            tau_candidate = next_candidate
            feasible = self._torque_within_limits(tau_candidate, tau_limit)
            iters += 1

        return task_scale, tau_candidate, iters, feasible

    def compute(self, state: dict[str, Any]) -> CartesianImpedanceOutput:
        if not self._initialized:
            raise RuntimeError("Call reset_from_state() before compute().")
        st = as_impedance_robot_state(state)
        q = np.asarray(st["q"], dtype=np.float64).reshape(6)
        qd = np.asarray(st["qd"], dtype=np.float64).reshape(6)
        p = np.asarray(st["ee_pos"], dtype=np.float64).reshape(3)
        quat = np.asarray(st["ee_quat"], dtype=np.float64).reshape(4)
        v = np.asarray(st["ee_lin_vel"], dtype=np.float64).reshape(3)
        omega = np.asarray(st["ee_ang_vel"], dtype=np.float64).reshape(3)
        J = np.asarray(st["jacobian"], dtype=np.float64).reshape(6, 6)

        hold_current_pose = bool(st.get("hold_current_pose", False))
        if hold_current_pose:
            # Capture the settle reference once, then keep that reference
            # fixed so posture and gravity compensation can actually hold the
            # arm instead of re-zeroing the restoring torque every step.
            if not self._hold_reference_initialized:
                self._x0 = float(p[0])
                self._y0 = float(p[1])
                self._z0 = float(p[2])
                self._quat0 = quat.copy()
                self._q_rest = q.copy()
                self._hold_reference_initialized = True
            x_des = self._x0
            x_vel_des = 0.0
            y_des = self._y0
            z_des = self._z0
            quat_ref = self._quat0
        else:
            x_des = float(st["target_x"])
            x_vel_des = float(st.get("target_x_vel", 0.0))
            y_des = self._y0
            z_des = self._z0
            quat_ref = self._quat0

        if self.cfg.y_coupling_feedforward:
            y_des = y_des - self.cfg.y_coupling_gain * (x_des - self._x0)

        x_err = x_des - float(p[0])
        y_err = y_des - float(p[1])
        z_err = z_des - float(p[2])

        if self.cfg.posture_reanchor_on_settle:
            x_tol = float(self.cfg.reanchor_x_tol_m)
            if self._posture_reanchored and abs(float(x_des) - self._x_des_at_anchor) > x_tol:
                # The x target moved on to a new plateau: re-arm.
                self._posture_reanchored = False
            if (
                not self._posture_reanchored
                and abs(x_err) <= x_tol
                and float(np.max(np.abs(qd))) <= float(self.cfg.reanchor_qd_tol_radps)
            ):
                self._q_rest = q.copy()
                self._x_des_at_anchor = float(x_des)
                self._posture_reanchored = True

        use_y_integral = bool(self.cfg.y_integral_action)
        if use_y_integral:
            # dt_s is optional on the state contract (AGENTS.md sec 2); fall
            # back to the 500 Hz direct_torque loop period, this controller's
            # primary real-hardware cadence, if the caller doesn't supply it.
            dt = float(st.get("dt_s", 1.0 / 500.0))
            if not np.isfinite(dt) or dt <= 0.0:
                dt = 1.0 / 500.0
            y_integral_limit = max(float(self.cfg.y_integral_limit_m_s), 0.0)
            self._y_integral += y_err * dt
            if y_integral_limit > 0.0:
                self._y_integral = float(np.clip(self._y_integral, -y_integral_limit, y_integral_limit))
            else:
                self._y_integral = 0.0
        Fy_integral = self.cfg.ki_y * self._y_integral if use_y_integral else 0.0

        use_x_integral = bool(self.cfg.x_integral_action)
        if use_x_integral:
            # Accumulate ONLY while the target is holding still (|x_vel_des|
            # below a small threshold), not during an active move. Found
            # necessary by direct sim testing (2026-08-02): accumulating
            # throughout the move phase too winds the integral up against the
            # large, transient tracking error a fast move naturally has
            # (nothing to do with friction), and that windup then overshoots
            # the target once the hold phase begins -- x_error was measured
            # crossing zero and continuing to grow negative for the rest of
            # the hold, a real closed-loop failure a synthetic single-
            # integrator test could not have caught. This feature exists to
            # close a HOLD-phase steady-state gap (the real evidence is a
            # flat, non-decaying plateau during hold), not to help move-phase
            # tracking, which kp_x/kd_x already handle -- gating to hold-only
            # matches that motivating case exactly and removes the windup
            # path entirely, without needing a reset-on-move-start (frozen,
            # not reset, so a brief blip in x_vel_des doesn't wipe progress).
            x_vel_des_abs = abs(float(x_vel_des))
            if x_vel_des_abs < 1e-4:
                dt = float(st.get("dt_s", 1.0 / 500.0))
                if not np.isfinite(dt) or dt <= 0.0:
                    dt = 1.0 / 500.0
                x_integral_limit = max(float(self.cfg.x_integral_limit_m_s), 0.0)
                self._x_integral += x_err * dt
                if x_integral_limit > 0.0:
                    self._x_integral = float(np.clip(self._x_integral, -x_integral_limit, x_integral_limit))
                else:
                    self._x_integral = 0.0
        Fx_integral = self.cfg.ki_x * self._x_integral if use_x_integral else 0.0

        Fx = self.cfg.kp_x * x_err + self.cfg.kd_x * (x_vel_des - float(v[0])) + Fx_integral
        Fy = self.cfg.kp_y * y_err - self.cfg.kd_y * float(v[1]) + Fy_integral
        Fz = self.cfg.kp_z * z_err - self.cfg.kd_z * float(v[2])

        e_rot = orientation_error_vec_wxyz(quat_ref, quat)
        ori_norm = float(np.linalg.norm(e_rot))
        M = self.cfg.kp_rot * e_rot - self.cfg.kd_rot * omega

        wrench = np.array([Fx, Fy, Fz, M[0], M[1], M[2]], dtype=np.float64)

        # Jacobian conditioning: needed both for singular_scale below and,
        # when lambda_adaptive_regularization is on, to schedule eps.
        cond = float(np.linalg.cond(J))

        # Base/wrist task split (default off = historical behavior; see
        # split_base_wrist_task's docstring above for the full evidence and
        # rationale). When off, J_task/wrench_task/cond_task are exactly
        # J/wrench/cond (a 6x6 identity substitution), so every computation
        # below that uses them is unchanged arithmetic, not just
        # coincidentally-equal output.
        use_split_base_wrist = bool(self.cfg.split_base_wrist_task)
        if use_split_base_wrist:
            # Position rows only, base-joint columns only (shoulder_pan,
            # shoulder_lift, elbow -- JOINT_NAME_ORDER[0:3]); wrist columns
            # stay exactly zero so translation-task torque structurally
            # cannot route through them. The rotational wrench (M) is
            # dropped from this pipeline -- the wrist-only rotation
            # sub-Jacobian is exactly singular at the motivating pose (step-1
            # evidence above), so routing M through it would just relocate
            # the same problem, not fix it. Orientation stays with
            # nullspace_posture (recomputed below against this same reduced
            # J_task) and, if enabled, wrist_orientation_task.
            J_task = np.zeros((3, 6), dtype=np.float64)
            J_task[:, 0:3] = J[0:3, 0:3]
            wrench_task = wrench[0:3].copy()
            cond_task = float(np.linalg.cond(J[0:3, 0:3]))
        else:
            J_task = J
            wrench_task = wrench
            cond_task = cond

        # Operational-space terms (P3, flag-gated; default off).
        use_shaping = bool(self.cfg.task_space_inertia_shaping)
        use_nullspace = bool(self.cfg.nullspace_posture)
        use_adaptive_eps = bool(self.cfg.lambda_adaptive_regularization)
        use_accel_ff = bool(self.cfg.acceleration_feedforward)
        mass_matrix_provided = "mass_matrix" in st and st["mass_matrix"] is not None
        lambda_mat: np.ndarray | None = None
        lambda_mat_nullspace: np.ndarray | None = None
        m_inv: np.ndarray | None = None
        use_wrench_adaptive_eps = bool(self.cfg.wrench_lambda_adaptive_regularization)
        eps_wrench = (
            self._scheduled_wrench_lambda_regularization(cond_task)
            if use_wrench_adaptive_eps
            else max(float(self.cfg.lambda_regularization), 0.0)
        )
        eps_effective = eps_wrench
        # acceleration_feedforward alone is also a trigger for this block (not
        # just task_space_inertia_shaping/nullspace_posture): it needs the
        # same Lambda = (J_task M^-1 J_task^T + eps I)^-1 this block already
        # builds for those two flags, reused rather than recomputed.
        if use_shaping or use_nullspace or use_accel_ff:
            if mass_matrix_provided:
                m_mat = np.asarray(st["mass_matrix"], dtype=np.float64).reshape(6, 6)
            else:
                m_mat = np.eye(6, dtype=np.float64)
            m_inv = np.linalg.inv(m_mat)
            # lambda_mat (wrench shaping) uses the static, previously-validated
            # eps by default: reducing it unconditionally destabilizes the
            # shaped wrench itself (measured joint-velocity blowup at
            # cond(J)~1e3-1e4, well short of the exact singularity) -- a
            # separate failure mode from the nullspace-projector leak
            # lambda_adaptive_regularization targets. wrench_lambda_adaptive_
            # regularization (separate flag, see its own docstring) schedules
            # THIS eps by live cond_task instead, with a more conservative
            # far-field floor informed directly by that failure mode. Uses
            # J_task (== J when split_base_wrist_task is off), so a_mat is 3x3
            # in split mode and 6x6 otherwise.
            a_mat = J_task @ m_inv @ J_task.T
            eye_task = np.eye(a_mat.shape[0], dtype=np.float64)
            lambda_mat = np.linalg.inv(a_mat + eps_wrench * eye_task)
            if use_adaptive_eps:
                eps_effective = self._scheduled_lambda_regularization(cond_task)
                lambda_mat_nullspace = np.linalg.inv(a_mat + eps_effective * eye_task)
            else:
                lambda_mat_nullspace = lambda_mat

        # Acceleration feedforward: mass-weighted addition to the task wrench,
        # built from Lambda's own diagonal (the wrench-shaping lambda_mat,
        # NOT lambda_mat_nullspace -- same distinction the wrench-shaping vs.
        # nullspace-projector split above already makes). Graceful no-op
        # (accel_ff_active stays False, wrench_task unchanged) unless a real
        # mass_matrix was supplied this cycle -- see acceleration_feedforward's
        # docstring above for why an identity-matrix fallback here would be
        # the wrong kind of "silent."
        #
        # use_shaping branches which physical quantity wrench_task represents
        # at this point in the pipeline (see the wrench-shaping comment below):
        # when shaping is on, wrench_task is a desired task ACCELERATION and
        # the single lambda_for_wrench @ wrench_task step below converts the
        # combined PD+feedforward acceleration into a force exactly once: add
        # raw target_accel here, not a pre-Lambda-scaled force, or shaping
        # would apply Lambda a second time (an effective Lambda^2 scaling --
        # found and fixed 2026-08-02, confirmed numerically and live in
        # config/ur5e_mujoco_torque_osc_tuned_split_base_wrist_accel_ff.yaml,
        # which sets both flags together). When shaping is off, wrench_task
        # is used directly as a FORCE (no further Lambda multiplication), so
        # the feedforward term must be mass-weighted here to be dimensionally
        # a force contribution.
        accel_ff_active = False
        wrench_accel_ff = np.zeros(3, dtype=np.float64)
        if use_accel_ff and mass_matrix_provided and lambda_mat is not None:
            lambda_diag = np.diag(lambda_mat)
            target_x_accel = float(st.get("target_x_accel", 0.0))
            target_y_accel = float(st.get("target_y_accel", 0.0))
            target_z_accel = float(st.get("target_z_accel", 0.0))
            accel_ff_vec = np.zeros(wrench_task.shape[0], dtype=np.float64)
            if use_shaping:
                accel_ff_vec[0] = target_x_accel
                if wrench_task.shape[0] >= 2:
                    accel_ff_vec[1] = target_y_accel
                if wrench_task.shape[0] >= 3:
                    accel_ff_vec[2] = target_z_accel
            else:
                accel_ff_vec[0] = lambda_diag[0] * target_x_accel
                if wrench_task.shape[0] >= 2:
                    accel_ff_vec[1] = lambda_diag[1] * target_y_accel
                if wrench_task.shape[0] >= 3:
                    accel_ff_vec[2] = lambda_diag[2] * target_z_accel
            wrench_task = wrench_task + accel_ff_vec
            wrench_accel_ff = accel_ff_vec[:3].copy()
            accel_ff_active = True

        singular_scale = 1.0
        if cond_task > self.cfg.jacobian_singular_cond_max > 0.0:
            singular_scale = float(self.cfg.jacobian_singular_cond_max / cond_task)
        use_diagonal_shaping = bool(self.cfg.lambda_diagonal_shaping)
        if use_shaping and lambda_mat is not None:
            # Wrench is treated as a desired task acceleration; Lambda maps it
            # to a dynamically consistent task force. The nullspace projector
            # below always uses the full (undiagonalized) lambda_mat -- only
            # the wrench-shaping step is affected by lambda_diagonal_shaping.
            lambda_for_wrench = np.diag(np.diag(lambda_mat)) if use_diagonal_shaping else lambda_mat
            wrench_effective = lambda_for_wrench @ wrench_task
        else:
            wrench_effective = wrench_task
        wrench_scaled = wrench_effective * singular_scale
        tau_task_nominal = J_task.T @ wrench_scaled
        tau_damping = -self.cfg.kd_joint * qd
        tau_posture = self.cfg.kp_posture * (self._q_rest - q) - self.cfg.kd_posture * qd
        if use_nullspace and lambda_mat_nullspace is not None and m_inv is not None:
            # Dynamically consistent nullspace projector: posture torques can
            # no longer produce task-space acceleration. Uses
            # lambda_mat_nullspace (== lambda_mat unless lambda_adaptive_
            # regularization is on), not the wrench-shaping lambda_mat, and
            # J_task (== J when split_base_wrist_task is off) so the
            # projector sees the same reduced/rank-3 task the wrench-shaping
            # step used.
            j_bar = m_inv @ J_task.T @ lambda_mat_nullspace
            nullspace_proj = np.eye(6) - J_task.T @ j_bar.T
            tau_posture = nullspace_proj @ tau_posture

        use_wrist_orientation_task = bool(self.cfg.wrist_orientation_task)
        tau_orient_wrist = np.zeros(6, dtype=np.float64)
        if use_wrist_orientation_task:
            # Deliberately NOT part of the shared Lambda-weighted wrench
            # pipeline above -- that pipeline is where kp_rot was found to be
            # unstable near the wrist_2=0 singularity. This is a plain
            # joint-space PD term (computed exactly like tau_posture),
            # reusing e_rot/omega already computed for the wrench's own
            # (currently zero-gain) rotational block, masked to act mostly
            # through the wrist chain -- see WRIST_ORIENTATION_MASK.
            J_rot = J[3:6, :]
            m_wrist = self.cfg.kp_rot_wrist * e_rot - self.cfg.kd_rot_wrist * omega
            tau_orient_wrist = (J_rot.T @ m_wrist) * WRIST_ORIENTATION_MASK

        # Gravity is computed here, BEFORE the friction-feedforward block below,
        # so the karnopp branch's driving_torque can include it -- a pure
        # reordering of two mutually-independent computations (gravity doesn't
        # depend on anything the friction block computes, and the static/lugre
        # friction branches never read g), so this changes no existing output
        # (verified byte-identical for every pre-existing friction_model value).
        g = np.zeros(6, dtype=np.float64)
        if "gravity_torque" in st and st["gravity_torque"] is not None:
            g = np.asarray(st["gravity_torque"], dtype=np.float64).reshape(6)

        use_friction_feedforward = bool(self.cfg.friction_feedforward)
        friction_model_used = str(self.cfg.friction_model) if use_friction_feedforward else "static"
        tau_friction_ff = np.zeros(6, dtype=np.float64)
        if use_friction_feedforward:
            if self.cfg.friction_model == "lugre":
                # dt_s is optional on the state contract (AGENTS.md sec 2);
                # fall back to the 500 Hz direct_torque loop period, this
                # controller's primary real-hardware cadence, matching the
                # same fallback already used by y_integral_action above.
                dt = float(st.get("dt_s", 1.0 / 500.0))
                if not np.isfinite(dt) or dt <= 0.0:
                    dt = 1.0 / 500.0
                tau_friction_ff = self._lugre_step(qd, dt)
            elif self.cfg.friction_model == "karnopp":
                # driving_torque is every other joint-space bias term already
                # computed above this cycle (task torque, damping, posture,
                # wrist-orientation, gravity) -- see _karnopp_step's docstring.
                driving_torque = tau_task_nominal + tau_damping + tau_posture + tau_orient_wrist + g
                tau_friction_ff = self._karnopp_step(qd, driving_torque)
            else:
                coulomb = np.asarray(self.cfg.friction_ff_coulomb_nm, dtype=np.float64).reshape(6)
                viscous = np.asarray(self.cfg.friction_ff_viscous, dtype=np.float64).reshape(6)
                deadband = max(float(self.cfg.friction_ff_qd_deadband), 1e-9)
                tau_friction_ff = coulomb * np.tanh(qd / deadband) + viscous * qd

        tau_bias = tau_damping + tau_posture + tau_orient_wrist + tau_friction_ff + g
        tau_limit = np.asarray(self.cfg.tau_max_nm, dtype=np.float64).reshape(6)
        tau_headroom = np.clip(float(self.cfg.torque_headroom), 0.0, 1.0)
        tau_limit_headroom = tau_limit * max(tau_headroom, 0.0)
        if not np.any(tau_limit_headroom > 0.0):
            tau_limit_headroom = tau_limit.copy()

        tau_nominal = tau_task_nominal + tau_bias
        task_backtrack_scale, tau_preclip, task_backtrack_iters, task_feasible = self._backtrack_task_scale(
            tau_nominal=tau_nominal,
            tau_limit=tau_limit_headroom,
        )

        tau_task = task_backtrack_scale * tau_task_nominal
        tau_damping = task_backtrack_scale * tau_damping
        tau_posture = task_backtrack_scale * tau_posture
        tau_orient_wrist = task_backtrack_scale * tau_orient_wrist
        tau_friction_ff = task_backtrack_scale * tau_friction_ff
        g = task_backtrack_scale * g
        tau = tau_preclip

        tau_clipped = np.clip(tau, -tau_limit, +tau_limit)
        saturated = np.abs(tau - tau_clipped) > 1e-10
        task_scale = float(singular_scale * task_backtrack_scale)

        return CartesianImpedanceOutput(
            tau=tau_clipped,
            tau_preclip=tau_preclip,
            wrench=wrench,
            tau_task_nominal=tau_task_nominal,
            tau_task=tau_task,
            tau_damping=tau_damping,
            tau_posture=tau_posture,
            tau_orient_wrist=tau_orient_wrist,
            tau_friction_ff=tau_friction_ff,
            tau_gravity=g,
            tau_saturated=saturated.astype(np.float64),
            jacobian_cond=cond_task,
            singular_scale=singular_scale,
            task_backtrack_scale=float(task_backtrack_scale),
            task_scale=task_scale,
            task_backtrack_iters=int(task_backtrack_iters),
            task_feasible=bool(task_feasible),
            x_error=float(x_err),
            y_error=float(y_err),
            z_error=float(z_err),
            orientation_error_vec=e_rot,
            orientation_error_norm=ori_norm,
            inertia_shaping_active=use_shaping,
            lambda_diagonal_shaping_active=bool(use_shaping and use_diagonal_shaping),
            lambda_adaptive_regularization_active=bool((use_shaping or use_nullspace) and use_adaptive_eps),
            lambda_regularization_effective=float(eps_effective),
            nullspace_posture_active=use_nullspace,
            mass_matrix_provided=bool(mass_matrix_provided),
            posture_reanchored=bool(self._posture_reanchored),
            wrist_orientation_task_active=use_wrist_orientation_task,
            friction_feedforward_active=use_friction_feedforward,
            friction_model_used=friction_model_used,
            friction_z=self._friction_z.copy(),
            friction_karnopp_stuck=self._karnopp_stuck.astype(np.float64).copy(),
            y_integral_action_active=use_y_integral,
            y_integral_value=float(self._y_integral),
            x_integral_action_active=use_x_integral,
            x_integral_value=float(self._x_integral),
            split_base_wrist_task_active=use_split_base_wrist,
            acceleration_feedforward_active=accel_ff_active,
            wrench_accel_ff=wrench_accel_ff,
        )
