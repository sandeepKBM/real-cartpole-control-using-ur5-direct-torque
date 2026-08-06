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
    # ik_posture_gain added 2026-08-06: posture pull toward q_rest inside
    # compute_ik_seeded's Newton-QP solve (controller_core/
    # cartesian_velocity_controller/modes.py), fixing a root-caused gap --
    # the search's own found gains push pinv_damping/qp_task_weight toward
    # near-zero-regularization/near-exact-IK, which leaves rotation axes
    # OUTSIDE the task selection (rx/ry, since task_dim_rx/ry are False by
    # default) essentially unconstrained; traced directly on a real
    # hanging_alpha_0_5 failure: rz (task-constrained) tracked to 2e-4 rad
    # while rx (unconstrained) grew to 0.25 rad and alone tripped the
    # orientation guard.
    #
    # Bound revised the SAME day after the first version ([0,50], an
    # absolute QP weight) was found to have the wrong SCALE: hand-sweeping
    # it up to 128 against that exact real failure barely moved the
    # outcome (0.2533 -> 0.2517, never passing), because that gain
    # vector's qp_task_weight (~1.8e8) makes an absolute weight in [0,50]
    # utterly negligible by comparison (ratio ~1e-7) -- confirmed directly
    # by sweeping the OLD absolute semantics up into the 1e8-1e9 range
    # against the same case, which genuinely fixed it. compute_ik_seeded
    # now interprets ik_posture_gain as a FRACTION OF qp_task_weight
    # rather than an absolute weight specifically to remove this scale
    # dependency: whatever qp_task_weight the search lands on, ik_posture_
    # gain automatically tracks it. Re-swept against the same real case
    # with this corrected scaling: a clean, monotonic pass/fail boundary
    # right around 0.5 (peak orientation error 0.2533 at 0.0 down to
    # 0.0235 at 10.0, guard-free from ~0.5 onward) -- [0,10] covers this
    # useful range with room past the boundary on both sides.
    ("ik_posture_gain", 0.0, 10.0, False),
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
        self._controller.cfg.ik_posture_gain = gains["ik_posture_gain"]

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
        qd = np.linalg.pinv(jac) @ xd_cmd
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
