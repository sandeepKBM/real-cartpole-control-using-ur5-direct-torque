"""gymnasium.Env wrapping controller_core.cartesian_velocity_controller's
ik_seeded_resolution mode for gain tuning/evaluation.

Kinematic-only, same category of simulation as tools/diagnostics/
ur5e_velocity_control_kinematic_sim.py (reuses hardware/local_dynamics.py's
LocalMujocoDynamics for MuJoCo-backed forward kinematics/Jacobian at
arbitrary q -- no mj_step, no torque, no mass matrix): real speedL is
resolved to joint velocities by the robot firmware's own Jacobian-based IK,
not by rigid-body dynamics, so this is the right level of fidelity for
tuning the controller's kinematic behavior.

One episode = one scripted min-jerk move-hold transport at a fixed
(pose, target_x_delta_m, move_duration_s). The action is the controller's
gain vector; by design this environment supports BOTH usages:
  - Bandit/black-box-optimization usage (the recommended default, see
    ../optimize.py): the caller passes the SAME action every step (fixed
    gains for the whole episode) and only the cumulative/final reward
    matters -- this is what differential_evolution (or CMA-ES, Bayesian
    optimization, etc.) needs, and it structurally cannot reproduce
    rl_gain_scheduling/'s documented failure modes (deceptive sit-still
    optimum, exploration collapse) because there is no sequential temporal-
    credit-assignment problem when the action never changes within an
    episode.
  - Live per-step scheduling (available, not the recommended starting
    point): the action CAN differ every step if a caller wants to explore
    adaptive gain scheduling later -- the environment does not prevent
    this, it simply isn't what optimize.py exercises by default.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from controller_core.cartesian_velocity_controller import (
    CartesianVelocityConfig,
    CartesianVelocityController,
)
from controller_core.cartesian_velocity_controller.math_utils import _damped_pinv
from controller_core.kinematics_utils import swing_twist_axis_error
from hardware.local_dynamics import LocalMujocoDynamics
from simulation.ur5e_mujoco_torque import x_profile_target

from ..poses import POSE_SCENARIOS, PoseScenario

# Action vector order and physical ranges. log=True entries are remapped in
# log-space (span multiple orders of magnitude); log=False entries are
# remapped linearly.
#
# Widened 2026-08-06 after a first search (bounds: kp_x[0.5,6], kp_rot
# [0.2,6], ik_joint_gain[0.5,12], pinv_damping[5e-4,5e-2], qp_task_weight
# [1e2,1e6]) pushed ik_joint_gain to its exact upper bound (12.0) --
# ambiguous whether that meant "the true optimum is higher" or "the bound
# was arbitrary and doesn't matter." Evaluating that result against the
# full multi-pose grid found the SAFE-RANGE BOUNDARY unchanged at every
# pose despite the 3x gain increase (structural evidence ik_joint_gain
# doesn't move the boundary at all -- see velocity_gain_tuning/optimize.py
# module docstring's cross-reference), but that still left the boundary-hit
# itself unexplained. Widened by roughly an order of magnitude past the
# previous bounds on every field so a future boundary hit is a genuine
# "hit a real guard/limit" signal, not an artifact of an arbitrarily
# narrow search range.
ACTION_FIELDS: tuple[tuple[str, float, float, bool], ...] = (
    ("kp_x", 0.1, 30.0, False),
    ("kp_rot", 0.05, 30.0, False),
    ("ik_joint_gain", 0.2, 80.0, False),
    # pinv_damping/qp_task_weight widened further 2026-08-06: a search over
    # the FIRST widened range (pinv_damping [1e-5,1], qp_task_weight
    # [10,1e8]) found gains that both (a) substantially extended the safe
    # range at every pose (e.g. hanging_alpha_0_5's boundary moved from
    # ~0.185m to beyond 0.24m, the largest value tested) and (b) still
    # landed pinv_damping near its LOWER bound (1.26e-5) and qp_task_weight
    # near its UPPER bound (9.28e7) -- i.e. still pushing toward "less
    # regularization, closer to an exact/hard-constraint IK solve," not yet
    # converged to an interior optimum. Widened another 2 orders of
    # magnitude on both to see whether that trend continues or genuinely
    # plateaus/becomes unsafe -- this is exactly the "let the guardrails
    # tell us" search the range was widened for.
    ("pinv_damping", 1.0e-7, 1.0, True),
    ("qp_task_weight", 10.0, 1.0e10, True),
    # ik_posture_gain/ik_posture_activation_joint_dev_rad (both added
    # 2026-08-06, REMOVED the same day -- see controller_core/
    # cartesian_velocity_controller/config.py's git history) were a SOFT,
    # task_w-relative quadratic pull toward q_rest, gated on real joint
    # deviation. Fixed both known real failures at specific found values
    # (ik_posture_gain=0.816/gate=0.229 for hanging_alpha_0_5; gate=0.15
    # alone for both hanging and neg40/neg45_wrist2offset), but repeated
    # gain searches over this exact 2D sub-range -- even ones deliberately
    # seeded with forced-nonzero starting values -- almost never converged
    # to using it: the optimizer kept landing on ik_posture_gain~0 or
    # values indistinguishable from just lowering ik_joint_gain instead.
    #
    # Replaced by ik_max_joint_deviation_rad: a genuine HARD bound on the
    # null-space (redundant) part of compute_ik_seeded's solve, enforced
    # exactly via null-space-basis coordinate clipping (controller_core/
    # kinematics_utils.py::null_space_basis + modes.py's compute_ik_seeded
    # -- see that module for the full 2-version history, including a
    # first, WRONG uniform-per-joint-bound attempt that silently crippled
    # task achievement at poses where the "redundant" joints aren't the
    # same ones as at other poses). Unlike the soft pull, task achievement
    # (J_task @ dq) is PROVABLY unaffected by however aggressively this
    # clips, so there is nothing for a search to trade off against --
    # tighter should structurally never hurt tracking, only redundant
    # drift, removing the exact ambiguity that made the soft version hard
    # to find via search.
    #
    # IMPORTANT SCOPE LIMIT (see config.py's docstring for the full
    # finding): this mechanism only fixes REDUNDANT (null-space) failures
    # (confirmed for neg40/neg45_wrist2offset's wrist_2 runaway). It
    # CANNOT fix hanging_alpha_0_5's -X orientation failure -- a direct
    # linear-algebra check found that failure's orientation coupling lives
    # in the TASK (row) space itself, not the null space. Do not expect a
    # search over this field alone to recover hanging's pass rate the way
    # the old soft-pull mechanism (when it DID get used) once did.
    #
    # Range: log-scaled (spans orders of magnitude, like pinv_damping/
    # qp_task_weight above), bounds DELIBERATELY ordered (lo=2.0, hi=0.01)
    # so action=-1.0 -> 2.0 rad (loose enough that it should essentially
    # never bind for any transport-scale move, approximating "off" without
    # needing a special sentinel -- ik_max_joint_deviation_rad has no
    # natural "always inactive" numeric value the way ik_posture_gain=0
    # did) and action=+1.0 -> 0.01 rad (tight -- strongly constrains
    # redundant drift, matching the value that fixed neg40/neg45 in direct
    # validation). This ordering matters: -1.0 is the sentinel every
    # SHORTER historical action vector gets padded with for a missing
    # trailing dimension (see optimize.py's seed_from_json padding and
    # this package's tests' _SEARCH2_ACTION), and those vectors' actual
    # historical behavior (recorded before this field existed at all) was
    # genuinely unconstrained -- padding them with a value that instead
    # activates a TIGHT null-space clip would silently reinterpret their
    # real recorded guard-trip behavior into a pass, corrupting exactly
    # the kind of regression test this repo relies on (found the hard way:
    # the (0.01, 2.0) ordering flipped two real historical-regression
    # tests from fail-as-recorded to spuriously passing).
    ("ik_max_joint_deviation_rad", 2.0, 0.01, True),
)
ACTION_DIM = len(ACTION_FIELDS)
OBS_DIM = 12


def action_to_gains(action: np.ndarray) -> dict[str, float]:
    """Map an action in [-1, 1]^ACTION_DIM (ACTION_FIELDS order) to physical gain values."""
    action = np.clip(np.asarray(action, dtype=np.float64).reshape(ACTION_DIM), -1.0, 1.0)
    gains: dict[str, float] = {}
    for i, (name, lo, hi, is_log) in enumerate(ACTION_FIELDS):
        frac = (float(action[i]) + 1.0) / 2.0  # [-1,1] -> [0,1]
        if is_log:
            gains[name] = float(np.exp(np.log(lo) + frac * (np.log(hi) - np.log(lo))))
        else:
            gains[name] = float(lo + frac * (hi - lo))
    return gains


@dataclass
class VelocityTransportEnvConfig:
    rate_hz: float = 125.0
    move_duration_s: float = 1.0
    duration_s: float = 3.0
    ik_iterations: int = 6
    task_dim_rz: bool = True
    task_dim_rx: bool = False
    task_dim_ry: bool = False
    max_lin_speed_mps: float = 0.25
    max_ang_speed_radps: float = 0.5
    # orientation_priority (added 2026-08-06, default OFF = exact prior
    # behavior). Exposed as ENV CONFIG rather than as new ACTION_FIELDS
    # dimensions on purpose: every historical search_result_*.json in
    # outputs/velocity_gain_tuning/ is an action vector of the current
    # length, and widening the action space would force those vectors to be
    # padded and silently reinterpreted -- the exact corruption
    # ACTION_FIELDS' own comment on ik_max_joint_deviation_rad's bound
    # ordering documents. Keeping it here makes "baseline gains, mechanism
    # on vs. off" a genuinely apples-to-apples comparison at the SAME
    # action vector. See controller_core/cartesian_velocity_controller/
    # config.py for the mechanism itself.
    orientation_priority: bool = False
    orientation_priority_weight: float = 1.0
    orientation_priority_residual_tol_m: float = 0.0001
    orientation_priority_residual_falloff_m: float = 0.0005
    orientation_priority_falloff_power: float = 2.0
    # singularity_velocity_scaling (added 2026-08-07, default OFF = exact
    # prior behavior). Exposed as ENV CONFIG, not a new ACTION_FIELDS
    # dimension, for the identical reason orientation_priority is above --
    # every historical search_result_*.json's action vector must stay
    # interpretable, and this makes "same gains, mechanism on vs off" an
    # apples-to-apples comparison. See controller_core/cartesian_velocity_
    # controller/config.py for the mechanism and its measured evidence.
    singularity_velocity_scaling: bool = False
    singularity_sigma_min_stop: float = 0.003
    singularity_sigma_min_full_speed: float = 0.03
    singularity_scale_power: float = 2.0
    # singularity_windup_clamp_rad (added 2026-08-07, default OFF = exact
    # prior behavior, same reasoning as the two fields above): anti-windup
    # fix for singularity_velocity_scaling's own throttle-then-release
    # dynamic, see controller_core/cartesian_velocity_controller/config.py
    # and modes.py for the mechanism and its measured evidence (a 128-cell
    # grid regression this field fixes without losing the mechanism's real
    # win or introducing new regressions).
    singularity_windup_clamp_rad: float | None = None
    # Damping for the bare-pinv(jac) @ xd_cmd reconstruction step() uses to
    # ESTIMATE joint velocity for joint_velocity_guard (and to integrate
    # self._q forward -- see step()'s docstring-adjacent comment there).
    # Added 2026-08-07 fixing a confirmed false-positive: step()'s pinv is on
    # the FULL 6x6 Jacobian, purely a downstream reporting/integration
    # reconstruction -- a DIFFERENT matrix from the reduced task-space
    # Jacobian CartesianVelocityConfig.pinv_damping (default 0.005) damps
    # inside the controller's own QP, sized for a different purpose. Direct
    # trace of both documented spikes (docs/status/
    # nullspace_v2_search_results_2026-08-06.md's 18.14/161.57 rad/s cases)
    # confirmed BOTH are wrist_2=0 crossings where cond(FULL J) reaches
    # ~6500+ (sigma_min ~3.25e-4) while the controller's own reduced-task
    # Jacobian stays well-conditioned (cond~15) throughout -- the controller
    # never sees an ill-conditioned matrix; only this reconstruction does.
    #
    # Value chosen from measured data, not a guess or a reuse of
    # pinv_damping's 0.005 (that value is sized for the reduced-task matrix,
    # not this one -- confirmed NOT automatically correct here, see below):
    #   - Sampled sigma_min of the FULL Jacobian across 6063 steps of the
    #     128-cell evaluation grid's well-conditioned population (no step
    #     had cond>100): 1st/5th/25th/50th percentile = 0.038/0.046/
    #     0.065/0.072. At qd_estimate_damping=1e-3, the damped-pinv relative
    #     error in the worst (1st-percentile) direction is
    #     (1e-3/0.038)^2 ~= 7e-4 (0.07%) -- negligible, i.e. the estimate is
    #     essentially unchanged away from singularities, the same
    #     "byte-identical off the edge case" bar this session's other
    #     mechanisms were held to. Reusing pinv_damping=0.005 here instead
    #     would give ~1.7% relative error at that same population's worst
    #     percentile -- meaningfully looser, evidence the two matrices
    #     really do need different values.
    #   - At the neg45_wrist2offset tight-null-space-bound spike (wrist_2
    #     crossing 0, sigma_min~3.25e-4, guard temporarily disabled to trace
    #     the full trajectory): bare pinv gives 28.14 rad/s; damped at 1e-3
    #     gives 10.33 rad/s -- still correctly above max_joint_velocity_radps
    #     (a real kinematic hazard at this pose, the guard SHOULD still
    #     trip) but no longer the physically-implausible >100 rad/s bare
    #     numbers this session's other documented spikes showed.
    #     Heavier damping (0.005, matching pinv_damping) bounds it to 2.10
    #     rad/s -- UNDER the guard, which would silently launder a genuine
    #     near-singularity crossing into a "pass"; 1e-3 does not do this,
    #     which is why it was preferred over reusing 0.005.
    qd_estimate_damping: float = 1.0e-3
    # Safety guard thresholds -- match the sim/hardware defaults used
    # throughout this session's manual sweeps.
    max_joint_velocity_radps: float = 3.0
    max_abs_orthogonal_drift_m: float = 0.05
    max_orientation_error_rad: float = 0.25
    # Reward shaping weights. progress_weight dominates on purpose -- see
    # module docstring and the reward_components note in step()'s
    # docstring for why: rl_gain_scheduling/'s root-caused failure was
    # every dense term EXCEPT x_error being minimized by not moving at
    # all, with no offsetting signal for attempting a move. Here,
    # progress toward target is the dominant term by a wide margin, and
    # everything else is a comparatively small shaping penalty, not a
    # competing objective.
    progress_weight: float = 20.0
    orientation_weight: float = 0.5
    time_cost: float = 0.05
    terminal_success_bonus: float = 10.0
    terminal_success_tol_m: float = 0.003
    # Consecutive settled cycles (|x_error| < terminal_success_tol_m)
    # required before an episode ends early with the success bonus,
    # instead of always running to duration_s. Without this, time_cost
    # would be a CONSTANT applied over the same fixed number of steps in
    # every episode (a harmless-but-pointless offset, not a real
    # incentive) -- with it, gains that settle faster genuinely accrue
    # less accumulated time_cost and receive their terminal bonus sooner,
    # which is the actual, meaningful trade-off this term is meant to
    # express. Found and fixed during this env's own smoke test: the
    # first version ran every episode to duration_s regardless of
    # settling, making time_cost non-functional by construction.
    settle_cycles_for_early_stop: int = 10
    guard_trip_penalty: float = -20.0
    pose_scenarios: tuple[PoseScenario, ...] = field(default_factory=lambda: POSE_SCENARIOS)


class VelocityTransportEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, config: VelocityTransportEnvConfig | None = None, seed: int | None = None) -> None:
        super().__init__()
        self.cfg = config or VelocityTransportEnvConfig()
        self._dyn = LocalMujocoDynamics()
        self._rng = np.random.default_rng(seed)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(ACTION_DIM,), dtype=np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(OBS_DIM,), dtype=np.float32)

        self._controller: CartesianVelocityController | None = None
        self._q: np.ndarray = np.zeros(6)
        self._p0: np.ndarray = np.zeros(3)
        self._quat0: np.ndarray = np.array([1.0, 0.0, 0.0, 0.0])
        self._x0: float = 0.0
        self._target_x_delta_m: float = 0.0
        self._t_s: float = 0.0
        self._prev_x_error: float = 0.0
        self._dt: float = 1.0 / self.cfg.rate_hz

    def _fk_jacobian_fn(self, q: np.ndarray):
        return self._dyn.fk_and_jacobian(q)

    def _build_obs(self, p: np.ndarray, quat: np.ndarray, target_x: float) -> np.ndarray:
        x_err = float(target_x - p[0])
        y_err = float(self._p0[1] - p[1])
        z_err = float(self._p0[2] - p[2])
        yaw_err = swing_twist_axis_error(self._quat0, quat, 2)
        t_norm = float(np.clip(self._t_s / max(self.cfg.duration_s, 1.0e-9), 0.0, 1.0))
        dx_norm = float(self._target_x_delta_m)
        obs = np.concatenate(
            [
                self._q.astype(np.float64),
                [x_err, y_err, z_err, yaw_err, t_norm, dx_norm],
            ]
        ).astype(np.float32)
        return obs

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        options = options or {}

        scenario: PoseScenario = options.get("scenario") or self._rng.choice(self.cfg.pose_scenarios)
        # Default: sample dx up to 1.3x the empirically-found boundary, so
        # the optimizer/evaluator sees both comfortably-safe and
        # deliberately-past-the-edge cases, not just the easy interior.
        target_x_delta_m = options.get("target_x_delta_m")
        if target_x_delta_m is None:
            target_x_delta_m = float(self._rng.uniform(0.2 * scenario.max_dx_hint_m, 1.3 * scenario.max_dx_hint_m))
        move_duration_s = float(options.get("move_duration_s", self.cfg.move_duration_s))

        self._q = scenario.q0.copy()
        p0, quat0, _ = self._dyn.fk_and_jacobian(self._q)
        self._p0 = p0
        self._quat0 = quat0
        self._x0 = float(p0[0])
        self._target_x_delta_m = target_x_delta_m
        self._move_duration_s = move_duration_s
        self._t_s = 0.0
        self._prev_x_error = target_x_delta_m
        self._settled_cycles_count = 0
        self._scenario_name = scenario.name

        gain_cfg = CartesianVelocityConfig(
            reduced_task_dims=False,
            split_base_wrist_task=False,
            ik_seeded_resolution=True,
            ik_iterations=self.cfg.ik_iterations,
            task_dim_rx=self.cfg.task_dim_rx,
            task_dim_ry=self.cfg.task_dim_ry,
            task_dim_rz=self.cfg.task_dim_rz,
            max_lin_speed_mps=self.cfg.max_lin_speed_mps,
            max_ang_speed_radps=self.cfg.max_ang_speed_radps,
            orientation_priority=self.cfg.orientation_priority,
            orientation_priority_weight=self.cfg.orientation_priority_weight,
            orientation_priority_residual_tol_m=self.cfg.orientation_priority_residual_tol_m,
            orientation_priority_residual_falloff_m=self.cfg.orientation_priority_residual_falloff_m,
            orientation_priority_falloff_power=self.cfg.orientation_priority_falloff_power,
            singularity_velocity_scaling=self.cfg.singularity_velocity_scaling,
            singularity_sigma_min_stop=self.cfg.singularity_sigma_min_stop,
            singularity_sigma_min_full_speed=self.cfg.singularity_sigma_min_full_speed,
            singularity_scale_power=self.cfg.singularity_scale_power,
            singularity_windup_clamp_rad=self.cfg.singularity_windup_clamp_rad,
        )
        self._controller = CartesianVelocityController(gain_cfg)
        self._controller.reset_from_state(
            {
                "time": 0.0,
                "q": self._q,
                "qd": np.zeros(6),
                "ee_pos": p0,
                "ee_quat": quat0,
                "target_x": self._x0,
            }
        )

        obs = self._build_obs(p0, quat0, self._x0)
        info: dict[str, Any] = {"scenario": scenario.name, "target_x_delta_m": target_x_delta_m}
        return obs, info

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Reward = progress_weight*(reduction in |x_error| this step)
        - orientation_weight*|yaw_error| - time_cost, every step, plus a
        one-shot terminal_success_bonus if the episode ends with |x_error|
        under terminal_success_tol_m, or guard_trip_penalty (terminates
        immediately) if a safety guard trips. The progress term is
        POTENTIAL-BASED (phi(s) = -|x_error|, reward = phi(s') - phi(s)),
        so it is dense every step and gives real, immediate positive credit
        for moving toward the target -- directly the fix rl_gain_scheduling/
        diagnosed but never implemented (docs/CURRENT_STATUS.md,
        2026-07-25 entry: "potential-based progress shaping on x_error...
        for exactly this reason")."""
        assert self._controller is not None, "call reset() first"
        gains = action_to_gains(action)

        target_x, target_x_vel = x_profile_target(
            "min_jerk_move_hold",
            self._x0,
            self._target_x_delta_m,
            self._t_s,
            self.cfg.duration_s,
            move_duration_s=self._move_duration_s,
        )
        p, quat, jac = self._dyn.fk_and_jacobian(self._q)

        target_ee_pos = self._p0.copy()
        target_ee_pos[0] = float(target_x)
        target_ee_vel = np.array([target_x_vel, 0.0, 0.0], dtype=np.float64)

        self._controller.cfg.kp_x = gains["kp_x"]
        self._controller.cfg.kp_y = gains["kp_x"]
        self._controller.cfg.kp_z = gains["kp_x"]
        self._controller.cfg.kp_rot = gains["kp_rot"]
        self._controller.cfg.ik_joint_gain = gains["ik_joint_gain"]
        self._controller.cfg.pinv_damping = gains["pinv_damping"]
        self._controller.cfg.qp_task_weight = gains["qp_task_weight"]
        self._controller.cfg.ik_max_joint_deviation_rad = gains["ik_max_joint_deviation_rad"]

        robot_state = {
            "time": self._t_s,
            "q": self._q,
            "qd": np.zeros(6),
            "ee_pos": p,
            "ee_quat": quat,
            "target_x": float(target_x),
            "target_ee_pos": target_ee_pos,
            "target_ee_vel": target_ee_vel,
            "fk_jacobian_fn": self._fk_jacobian_fn,
        }
        xd_cmd = self._controller.compute(robot_state)
        # Damped, not bare, pinv -- see cfg.qd_estimate_damping's docstring
        # for why: bare np.linalg.pinv on this FULL 6x6 Jacobian blows up at
        # the wrist_2=0 kinematic singularity (a downstream reconstruction
        # artifact, confirmed independent of the controller's own internals,
        # which invert a different, well-conditioned reduced-task matrix).
        qd = _damped_pinv(jac, self.cfg.qd_estimate_damping) @ xd_cmd
        max_abs_qd = float(np.max(np.abs(qd)))

        y_drift = abs(float(p[1] - self._p0[1]))
        z_drift = abs(float(p[2] - self._p0[2]))
        orthogonal_drift = max(y_drift, z_drift)
        yaw_err = swing_twist_axis_error(self._quat0, quat, 2)
        orientation_error = float(np.linalg.norm([
            swing_twist_axis_error(self._quat0, quat, i) for i in range(3)
        ]))
        x_error = float(target_x - p[0])

        guard_reason = None
        if max_abs_qd > self.cfg.max_joint_velocity_radps:
            guard_reason = f"joint_velocity_guard: {max_abs_qd:.4f} > {self.cfg.max_joint_velocity_radps}"
        elif orthogonal_drift > self.cfg.max_abs_orthogonal_drift_m:
            guard_reason = f"orthogonal_drift_guard: {orthogonal_drift:.4f} > {self.cfg.max_abs_orthogonal_drift_m}"
        elif orientation_error > self.cfg.max_orientation_error_rad:
            guard_reason = f"orientation_guard: {orientation_error:.4f} > {self.cfg.max_orientation_error_rad}"

        progress = abs(self._prev_x_error) - abs(x_error)
        reward = self.cfg.progress_weight * progress
        reward -= self.cfg.orientation_weight * abs(yaw_err)
        reward -= self.cfg.time_cost
        self._prev_x_error = x_error

        terminated = False
        truncated = False
        if guard_reason is not None:
            reward += self.cfg.guard_trip_penalty
            terminated = True
        else:
            self._q = self._q + qd * self._dt
            self._t_s += self._dt

            if abs(x_error) < self.cfg.terminal_success_tol_m:
                self._settled_cycles_count += 1
            else:
                self._settled_cycles_count = 0
            settled_early = (
                self._t_s >= self._move_duration_s
                and self._settled_cycles_count >= self.cfg.settle_cycles_for_early_stop
            )

            if settled_early or self._t_s >= self.cfg.duration_s - 1.0e-12:
                truncated = True
                if abs(x_error) < self.cfg.terminal_success_tol_m:
                    reward += self.cfg.terminal_success_bonus

        obs = self._build_obs(p, quat, target_x)
        info = {
            "x_error": x_error,
            "orientation_error": orientation_error,
            "max_abs_qd_radps": max_abs_qd,
            "orthogonal_drift_m": orthogonal_drift,
            "guard_reason": guard_reason,
            "achieved_x_delta_m": float(p[0] - self._x0),
            "scenario": self._scenario_name,
        }
        return obs, float(reward), terminated, truncated, info
