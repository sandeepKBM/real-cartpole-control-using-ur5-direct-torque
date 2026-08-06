"""
CartesianImpedanceConfig -- the gain/flag dataclass for the X-axis Cartesian
impedance controller.

Split out of the former single-file ``x_axis_cartesian_impedance.py`` module
(pure structural refactor; see the package ``__init__.py`` for the original
module docstring).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

from .constants import JOINT_NAME_ORDER
from .parsing import _parse_friction_model, _parse_y_control_mode


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
    # Per-joint posture gain override (default None = historical behavior:
    # the scalar kp_posture/kd_posture above applies uniformly to all 6
    # joints). Added 2026-08-03 after a real, visually-caught finding: with
    # reduced_task_dims dropping rx/ry (a 4D task on 6 joints, 2 genuine
    # redundant DOF), the scalar posture gain was not strong enough to stop
    # the task itself from using shoulder_pan -- the cheapest available
    # direction for correcting rz at this pose -- as its preferred solution.
    # A rendered round-trip video showed shoulder_pan swinging a full 23deg
    # (-52.7 to -29.7deg) around the -40deg start pose during a 0.06m move,
    # NOT caught by any of the scalar summary metrics (y_drift/orientation/
    # achieved_x all looked fine). This matters beyond aesthetics: -40deg was
    # chosen for real physical wall/base clearance, so a 12-13deg swing in
    # either direction can eat directly into that margin.
    #
    # When set (a length-6 array, JOINT_NAME_ORDER order), replaces the
    # scalar posture gain per-joint: tau_posture[i] = kp[i]*(q_rest[i]-q[i])
    # - kd[i]*qd[i]. Intended use: strong gain on shoulder_pan (and wrist_2,
    # to stay away from its singularity) to discourage the task from routing
    # through them, low gain on shoulder_lift/elbow (the joints that should
    # be doing the actual rail-motion work), matching the "prefer planar
    # arm-motion pattern" design goal. NOT yet validated -- this is the
    # mechanism, not an asserted-correct gain set; the caller's config
    # supplies the actual values.
    posture_kp_by_joint: np.ndarray | None = None
    posture_kd_by_joint: np.ndarray | None = None
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
    # Y control mode (default "tight" = historical behavior: continuous
    # kp_y/kd_y PD on y_err, unchanged). Added 2026-08-03. "corridor" replaces
    # that with a deadband: near zero correction while |y_err| stays inside
    # y_soft_limit_m, ramping smoothly (not a step) up to full corridor gain
    # by y_hard_limit_m. This is a DIFFERENT mechanism from y_integral_action
    # above (which adds a growing bias on top of the existing continuous PD)
    # -- corridor mode instead relaxes correction near center, on the
    # hypothesis that not fighting small, natural Y excursions at all costs
    # less X-tracking authority than continuously correcting them. Untested;
    # this flag makes the A/B possible, it does not assert an answer.
    #
    # y_hard_limit_m is the corridor's own saturation point (full
    # y_corridor_kp/kd applies beyond it), NOT a replacement for the real
    # safety termination -- that still lives in ImpedanceSafetyMonitor's
    # max_abs_y_drift_m/max_abs_orthogonal_drift_m (controller_core/safety.py),
    # untouched by this flag. Sizing y_hard_limit_m to roughly match (or sit
    # just inside) whatever safety threshold a given config uses is a
    # deliberate choice for the caller to make per-config, not enforced here.
    #
    # Smoothstep (3t^2 - 2t^3 for t in [0,1]) is used for the ramp rather than
    # a linear one: smoothstep has zero derivative at BOTH t=0 and t=1, so the
    # corridor gain is C1-continuous at both boundaries -- a linear ramp would
    # have a derivative kink at the soft-limit boundary (correction switching
    # from flat-zero to linearly-increasing), a small but real discontinuity
    # in commanded force. Mutually exclusive with y_integral_action (raises)
    # -- combining a deadbanded P-term with an always-accumulating integral
    # term on the same axis is a real interaction (the integral would keep
    # growing while inside the deadband, since y_err is nonzero there even
    # though the P-term contributes nothing) that hasn't been analyzed.
    y_control_mode: Literal["tight", "corridor"] = "tight"
    y_soft_limit_m: float = 0.015
    y_hard_limit_m: float = 0.05
    y_corridor_kp: float = 80.0
    y_corridor_kd: float = 15.0
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
    # Reduced-task dimension selection (default off = historical behavior:
    # all 6 rows active, byte-identical to today). Added 2026-08-03 as the
    # general form of the row-selection idea split_base_wrist_task already
    # uses for one specific case (position-only, base-joint-only) -- this
    # flag instead selects an ARBITRARY subset of the 6 task rows via a true
    # row-selection matrix S (J_task = S @ J, wrench_task = S @ wrench), not
    # a zeroed/masked 6x6 (that would leave A_task = J_task M^-1 J_task^T
    # singular in the dropped directions and break the Lambda inversion).
    #
    # task_dim_x/y/z select Fx/Fy/Fz (translation, always in the world frame
    # p/v are already expressed in throughout this file). task_dim_rx/ry/rz
    # select the 3 components of the WORLD-frame angular-velocity Jacobian
    # rows (jacr, i.e. M[0]/M[1]/M[2] in the wrench) -- these are NOT tool-
    # frame Euler roll/pitch/yaw; they are the axis-angle-style world-X/Y/Z
    # rotation-rate directions mj_jacSite already returns, the same
    # convention every other rotational quantity in this file already uses
    # (orientation_error_vec_wxyz, omega). Deliberately not introducing a
    # NEW tool-frame transform here: the 2026-08-02/03 frame audit spent real
    # effort ruling out exactly this class of bug (world-vs-tool-frame
    # mismatch) for the existing pipeline, and a fresh Euler/tool-frame
    # selection would reopen that risk for no established benefit yet -- if
    # a genuine tool-frame axis selection turns out to be needed (e.g. "yaw
    # about the pole axis" specifically), that is a deliberate follow-up, not
    # bundled into this flag.
    #
    # Mutually exclusive with split_base_wrist_task -- their interaction
    # (row selection AND column selection at once) is untested; compute()
    # raises ValueError if both are enabled together rather than silently
    # picking one.
    #
    # Known, explicitly NOT addressed by this flag: J_dot @ qd (the
    # Coriolis-like task-acceleration term from a moving Jacobian) is not
    # computed anywhere in this file currently -- grepped and confirmed
    # absent 2026-08-03, not just unused for this feature specifically. The
    # tau_task formula here is J_task^T @ Lambda_task @ wrench_task, exactly
    # mirroring the existing (non-reduced) wrench-shaping step's own
    # omission -- this flag does not add a new gap, it inherits the
    # existing one. Do not read reduced-task tracking error as evidence this
    # term doesn't matter; it has never been measured either way.
    reduced_task_dims: bool = False
    task_dim_x: bool = True
    task_dim_y: bool = True
    task_dim_z: bool = True
    task_dim_rx: bool = True
    task_dim_ry: bool = True
    task_dim_rz: bool = True
    # Hard per-joint task exclusion (default off = historical behavior).
    # Added 2026-08-03, a DIFFERENT and stronger mechanism than
    # posture_kp_by_joint above: that's a soft preference that competes with
    # the task (measured to cut, not eliminate, an unwanted 23deg
    # shoulder_pan swing -- down to 13deg, still real). This instead zeros
    # the excluded joint's COLUMN in J_task -- the same column-zeroing
    # split_base_wrist_task already uses for the wrist columns, generalized
    # to any joint. Since tau_task = J_task^T @ wrench_effective, a zeroed
    # column means tau_task[that joint] is exactly 0 by construction, not
    # approximately small -- no task force can reach that joint at all, only
    # gravity compensation and posture (which then has nothing to compete
    # against there, since the task genuinely cannot pull on it).
    #
    # Applied AFTER J_task/wrench_task/cond_task are established by whichever
    # of split_base_wrist_task/reduced_task_dims/the full-J default is
    # active -- composes with any of them rather than being a fourth
    # mutually-exclusive mode. cond_task is recomputed from the
    # column-locked J_task so adaptive-eps scheduling sees the real
    # (possibly worse-conditioned) matrix, not the pre-lock one.
    #
    # Real risk this does NOT eliminate: locking columns can make the
    # remaining task infeasible or push A_task toward singular if too many
    # joints are locked relative to the task's own dimensionality (e.g.
    # locking 3+ joints against a 4D task leaves only 3 free columns for 4
    # task rows) -- untested combinations should be checked before trusting
    # them, this flag doesn't guard against that itself.
    task_lock_shoulder_pan: bool = False
    task_lock_shoulder_lift: bool = False
    task_lock_elbow: bool = False
    task_lock_wrist_1: bool = False
    task_lock_wrist_2: bool = False
    task_lock_wrist_3: bool = False
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
            posture_kp_by_joint=(
                np.array([float(ctrl["posture_kp_by_joint"][name]) for name in JOINT_NAME_ORDER], dtype=np.float64)
                if "posture_kp_by_joint" in ctrl
                else None
            ),
            posture_kd_by_joint=(
                np.array([float(ctrl["posture_kd_by_joint"][name]) for name in JOINT_NAME_ORDER], dtype=np.float64)
                if "posture_kd_by_joint" in ctrl
                else None
            ),
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
            y_control_mode=_parse_y_control_mode(ctrl.get("y_control_mode", "tight")),
            y_soft_limit_m=float(ctrl.get("y_soft_limit_m", 0.015)),
            y_hard_limit_m=float(ctrl.get("y_hard_limit_m", 0.05)),
            y_corridor_kp=float(ctrl.get("y_corridor_kp", gains.get("kp_y", 80.0))),
            y_corridor_kd=float(ctrl.get("y_corridor_kd", gains.get("kd_y", 15.0))),
            x_integral_action=bool(ctrl.get("x_integral_action", False)),
            ki_x=float(gains.get("ki_x", 0.0)),
            x_integral_limit_m_s=float(ctrl.get("x_integral_limit_m_s", 0.02)),
            y_coupling_feedforward=bool(ctrl.get("y_coupling_feedforward", False)),
            y_coupling_gain=float(ctrl.get("y_coupling_gain", 0.7)),
            split_base_wrist_task=bool(ctrl.get("split_base_wrist_task", False)),
            reduced_task_dims=bool(ctrl.get("reduced_task_dims", False)),
            task_dim_x=bool(ctrl.get("task_dim_x", True)),
            task_dim_y=bool(ctrl.get("task_dim_y", True)),
            task_dim_z=bool(ctrl.get("task_dim_z", True)),
            task_dim_rx=bool(ctrl.get("task_dim_rx", True)),
            task_dim_ry=bool(ctrl.get("task_dim_ry", True)),
            task_dim_rz=bool(ctrl.get("task_dim_rz", True)),
            task_lock_shoulder_pan=bool(ctrl.get("task_lock_shoulder_pan", False)),
            task_lock_shoulder_lift=bool(ctrl.get("task_lock_shoulder_lift", False)),
            task_lock_elbow=bool(ctrl.get("task_lock_elbow", False)),
            task_lock_wrist_1=bool(ctrl.get("task_lock_wrist_1", False)),
            task_lock_wrist_2=bool(ctrl.get("task_lock_wrist_2", False)),
            task_lock_wrist_3=bool(ctrl.get("task_lock_wrist_3", False)),
            acceleration_feedforward=bool(ctrl.get("acceleration_feedforward", False)),
        )
