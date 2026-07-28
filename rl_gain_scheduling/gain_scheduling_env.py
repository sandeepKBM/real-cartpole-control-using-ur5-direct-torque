"""Height-conditioned RL gain-scheduling Gymnasium environment.

The policy outputs the 11 Cartesian-impedance gains (transport_metrics.
GAIN_FIELDS) every control step, conditioned on live state including
end-effector height (world Z). Built to fix a finding from this session:
config/ur5e_mujoco_torque_osc_tuned.yaml's gains were tuned at one specific
pose (ee z=1.08m) and did not transfer to a different pose (ee z=0.54m) --
see AGENTS.md sec 3 and the plan this environment was built from
(/common/home/ss5772/.claude/plans/sharded-hatching-globe.md).

One env.step() == one mujoco.mj_step() at the model's native rate (500Hz).
"""

from __future__ import annotations

import dataclasses
import sys
from pathlib import Path
from typing import Any

import gymnasium as gym
import mujoco
import numpy as np
import yaml
from gymnasium import spaces

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from controller_core.kinematics_utils import orientation_error_vec_wxyz  # noqa: E402
from controller_core.logging_utils import JsonlTraceWriter, json_dumps_safe  # noqa: E402
from simulation.ur5e_mujoco_torque import (  # noqa: E402
    apply_start_q,
    build_initial_state_and_adapter,
    build_mujoco_state,
    load_model,
    x_profile_target,
)
from transport_metrics import (  # noqa: E402
    GAIN_FIELDS,
    compute_valid_move_hold_metrics,
    summarize_move_hold_trace,
)

# Both share the wrist_2=0 singularity the whole controller design leans on
# (verified: interpolating elementwise between them keeps wrist_2 exactly 0
# at every point on the segment). alpha=0 -> tall pose, alpha=1 -> low pose.
ACTIVE_ORIGIN_Q = np.array([0.0, -1.5707963267948966, 0.0, -1.5707963267948966, 0.0, 0.0], dtype=np.float64)
LOWER_B_Q = np.array([0.0, -0.1, -2.4, -0.4, 0.0, 0.0], dtype=np.float64)

OBS_DIM = 47
ACTION_DIM = len(GAIN_FIELDS)
RESIDUAL_ACTION_DIM = 6


def rescale_action_to_gains(action: np.ndarray, gain_bounds: dict[str, tuple[float, float]]) -> dict[str, float]:
    """Map an action in [-1, 1]^11 (GAIN_FIELDS order) to physical gain values."""
    action = np.clip(np.asarray(action, dtype=np.float64).reshape(ACTION_DIM), -1.0, 1.0)
    gains: dict[str, float] = {}
    for i, name in enumerate(GAIN_FIELDS):
        lo, hi = gain_bounds[name]
        frac = (action[i] + 1.0) * 0.5  # [-1,1] -> [0,1]
        gains[name] = float(lo + frac * (hi - lo))
    return gains


def gains_to_normalized_action(gains: dict[str, float], gain_bounds: dict[str, tuple[float, float]]) -> np.ndarray:
    """Inverse of rescale_action_to_gains, for building the prev_gains_action observation term."""
    out = np.zeros(ACTION_DIM, dtype=np.float32)
    for i, name in enumerate(GAIN_FIELDS):
        lo, hi = gain_bounds[name]
        span = max(hi - lo, 1e-12)
        frac = (float(gains[name]) - lo) / span
        out[i] = float(np.clip(frac * 2.0 - 1.0, -1.0, 1.0))
    return out


def rescale_action_to_residual_tau(action: np.ndarray, max_nm: np.ndarray) -> np.ndarray:
    """Map an action in [-1, 1]^6 to a signed residual torque in Nm (linear, no [0,1] remap)."""
    action = np.clip(np.asarray(action, dtype=np.float64).reshape(RESIDUAL_ACTION_DIM), -1.0, 1.0)
    max_nm = np.asarray(max_nm, dtype=np.float64).reshape(RESIDUAL_ACTION_DIM)
    return action * max_nm


class GainSchedulingEnv(gym.Env):
    """Gymnasium env for RL-trained height-conditioned gain scheduling."""

    metadata = {"render_modes": []}

    def __init__(self, config_path: str | Path = REPO_ROOT / "config" / "rl_gain_scheduling.yaml") -> None:
        super().__init__()
        cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
        self._mujoco_cfg = cfg["mujoco"]
        self._ctrl_cfg = cfg["controller"]
        self._env_cfg = cfg["env"]
        self._reward_cfg = cfg["reward"]

        decimation = int(self._env_cfg.get("control_decimation_steps", 1))
        if decimation != 1:
            raise NotImplementedError(
                "control_decimation_steps != 1 is not yet implemented; the policy "
                "runs at the model's native step rate. Change this deliberately, "
                "not by ignoring the config value."
            )

        self._gain_bounds: dict[str, tuple[float, float]] = {
            name: (float(bounds[0]), float(bounds[1]))
            for name, bounds in self._env_cfg["gain_bounds"].items()
        }
        for name in GAIN_FIELDS:
            if name not in self._gain_bounds:
                raise ValueError(f"config.env.gain_bounds missing required gain field {name!r}")

        self._height_alpha_range = tuple(float(v) for v in self._env_cfg["height_alpha_range"])
        self._target_x_delta_range = tuple(float(v) for v in self._env_cfg["target_x_delta_range_m"])
        self._move_duration_s = float(self._env_cfg["move_duration_s"])
        self._max_episode_seconds = float(self._env_cfg["max_episode_seconds"])
        self._record_lightweight_trace = bool(self._env_cfg.get("record_lightweight_trace", True))
        self._full_trace_logging = bool(self._env_cfg.get("full_trace_logging", False))

        # Domain-randomization noise (default all-zero = byte-identical to
        # before this existed). Mirrors tools/ur5e_mujoco_torque_experiments.py's
        # --q-noise-std-rad/--qd-noise-std-radps/--torque-noise-std-nm exactly:
        # q/qd noise perturbs only what the CONTROLLER sees (not the true
        # physics state used for mj_step/trace ground truth); torque noise
        # perturbs the final torque actually applied, downstream of the
        # controller's own clean diagnostic output.
        noise_cfg = self._env_cfg.get("noise", {}) or {}
        self._q_noise_std = max(float(noise_cfg.get("q_noise_std_rad", 0.0)), 0.0)
        self._qd_noise_std = max(float(noise_cfg.get("qd_noise_std_radps", 0.0)), 0.0)
        self._torque_noise_std = max(float(noise_cfg.get("torque_noise_std_nm", 0.0)), 0.0)
        self._noise_rng = np.random.default_rng(int(noise_cfg.get("noise_seed", 0)))

        # Action mode (default "gains" = today's exact behavior, byte-identical):
        # "gains" -- policy outputs the 11 Cartesian-impedance gains every step
        # (original design). "residual_torque" -- gains are fixed for the whole
        # episode (set once in reset()) and the policy instead outputs a small
        # signed 6-dim torque correction added on top of the controller's own
        # output, mirroring how tau_coriolis/tau_gravity already get added
        # externally elsewhere in this codebase.
        self._action_mode = str(self._env_cfg.get("action_mode", "gains"))
        if self._action_mode not in ("gains", "residual_torque"):
            raise ValueError(f"config.env.action_mode must be 'gains' or 'residual_torque', got {self._action_mode!r}")
        self._residual_fixed_gains: dict[str, float] | None = None
        self._residual_max_nm: np.ndarray | None = None
        if self._action_mode == "residual_torque":
            residual_cfg = self._env_cfg.get("residual_torque")
            if residual_cfg is None:
                raise ValueError("config.env.action_mode == 'residual_torque' requires config.env.residual_torque")
            fixed_gains_cfg = residual_cfg.get("fixed_gains")
            if fixed_gains_cfg is None:
                raise ValueError("config.env.residual_torque.fixed_gains is required in residual_torque mode")
            self._residual_fixed_gains = {name: float(fixed_gains_cfg[name]) for name in GAIN_FIELDS}
            # No implicit default: an unbounded residual added on top of the
            # controller's own (already shaped/clipped) output must be a
            # deliberate, explicit choice, not a silently-guessed magnitude.
            max_nm_cfg = residual_cfg.get("max_nm")
            if max_nm_cfg is None:
                raise ValueError("config.env.residual_torque.max_nm is required in residual_torque mode")
            self._residual_max_nm = np.asarray(max_nm_cfg, dtype=np.float64).reshape(RESIDUAL_ACTION_DIM)
        self._action_dim = RESIDUAL_ACTION_DIM if self._action_mode == "residual_torque" else ACTION_DIM

        scene_xml = REPO_ROOT / self._mujoco_cfg["scene_xml"]
        self.model, self.data, self.site_id, self.joint_ids, self.actuator_ids = load_model(scene_xml)
        self._max_episode_steps = max(1, round(self._max_episode_seconds / float(self.model.opt.timestep)))

        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(OBS_DIM,), dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(self._action_dim,), dtype=np.float32)

        self.adapter = None
        self._state0 = None
        self._hold_current_pose_flag = False
        self._step_count = 0
        self._prev_action = np.zeros(self._action_dim, dtype=np.float32)
        self._prev_tau = np.zeros(6, dtype=np.float64)
        self._prev_abs_x_error = 0.0
        self._alpha = 0.0
        self._target_x_delta = 0.0
        self._lightweight_trace: list[dict[str, Any]] = []
        self._trace_writer: JsonlTraceWriter | None = None
        self._trace_path: Path | None = None
        self._output_dir: Path | None = None

    # -- reset ---------------------------------------------------------

    def _sample_or_take(self, options: dict[str, Any] | None, key: str, low: float, high: float) -> float:
        if options is not None and key in options and options[key] is not None:
            return float(options[key])
        return float(self.np_random.uniform(low, high))

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)

        alpha = self._sample_or_take(options, "height_alpha", *self._height_alpha_range)
        if not (0.0 <= alpha <= 1.0):
            lo, hi = self._height_alpha_range
            if not (lo <= alpha <= hi):
                raise ValueError(f"height_alpha {alpha!r} outside configured range {self._height_alpha_range}")
            # Extrapolation opt-in: re-check joint-limit feasibility (only
            # guaranteed inside [0,1] by convexity).
            q_check = (1.0 - alpha) * ACTIVE_ORIGIN_Q + alpha * LOWER_B_Q
            for jid, qi in zip(self.joint_ids, q_check):
                qmin, qmax = float(self.model.jnt_range[jid, 0]), float(self.model.jnt_range[jid, 1])
                if qmax > qmin and not (qmin <= qi <= qmax):
                    raise ValueError(f"height_alpha={alpha} produces an infeasible q_start ({qi} outside [{qmin},{qmax}])")
        self._alpha = alpha
        q_start = (1.0 - alpha) * ACTIVE_ORIGIN_Q + alpha * LOWER_B_Q
        apply_start_q(self.model, self.data, q_start)
        # apply_start_q() resets qpos/qvel/qacc/ctrl but not data.time -- it's
        # designed for single-episode-per-process callers (the sim tools),
        # none of which reset() repeatedly. This env does, every episode, and
        # step()'s t_s = self.data.time feeds directly into the min-jerk
        # move-hold target generator as "elapsed time since episode start."
        # Without this, every episode after the first in a given env's
        # lifetime starts with a stale, already-large t_s, so the target
        # generator thinks the move phase (often the whole episode) is
        # already over before the first control step -- the policy sees an
        # already-settled target from t=0 and never learns to actually move.
        self.data.time = 0.0

        target_x_delta = self._sample_or_take(options, "target_x_delta", *self._target_x_delta_range)
        self._target_x_delta = target_x_delta

        state0, adapter = build_initial_state_and_adapter(
            self.model,
            self.data,
            self.site_id,
            self.joint_ids,
            controller_cfg=self._ctrl_cfg,
            transport_axis_index=0,
            target_x_delta=target_x_delta,
            controller_kind="impedance",
            force_hold_current_pose=False,
            gravity_mode="gravity_comp",
            gravity_source=str(self._mujoco_cfg.get("gravity_source", "mujoco_qfrc")),
            coriolis_feedforward=bool(self._mujoco_cfg.get("coriolis_feedforward", False)),
            torque_limit_scale=1.0,
        )
        self.adapter = adapter
        self._state0 = state0
        self._hold_current_pose_flag = bool(state0.hold_current_pose)
        if self._action_mode == "residual_torque":
            # Gains are static for the whole episode in this mode -- set once
            # here, never overwritten in step() (unlike "gains" mode, where
            # the policy schedules them every step).
            self.adapter.controller.set_gains(self._residual_fixed_gains)
        self._step_count = 0
        self._prev_action = np.zeros(self._action_dim, dtype=np.float32)
        self._prev_tau = np.zeros(6, dtype=np.float64)
        # True x_error at t=0 is ~0 for this min-jerk profile (target starts at
        # the current position), so 0.0 here is the correct "previous error"
        # for the progress-reward term below -- not abs(target_x_delta), which
        # would inject one artificial reward spike on every episode's first step.
        self._prev_abs_x_error = 0.0
        self._lightweight_trace = []

        self._trace_writer = None
        self._trace_path = None
        if self._full_trace_logging:
            output_dir = Path(options["output_dir"]) if options and options.get("output_dir") else (
                REPO_ROOT / "outputs" / "rl_gain_scheduling" / "eval_runs" / "unlabeled_run"
            )
            output_dir.mkdir(parents=True, exist_ok=True)
            self._output_dir = output_dir
            self._trace_path = output_dir / "trace.jsonl"
            self._trace_writer = JsonlTraceWriter(self._trace_path)
            self._trace_writer.__enter__()

        obs, info = self._build_obs_and_info(
            ee_pos=state0.ee_pos, ee_quat=state0.ee_quat, q=state0.q, qd=state0.qd,
            ee_lin_vel=state0.ee_lin_vel, ee_ang_vel=state0.ee_ang_vel,
            x_error=0.0, y_error=0.0, z_error=0.0, orientation_error_norm=0.0,
        )
        return obs, info

    # -- step ------------------------------------------------------------

    def step(self, action: np.ndarray):
        assert self.adapter is not None and self._state0 is not None, "reset() must be called before step()"
        action = np.clip(np.asarray(action, dtype=np.float64).reshape(self._action_dim), -1.0, 1.0)
        if self._action_mode == "gains":
            gains = rescale_action_to_gains(action, self._gain_bounds)
            self.adapter.controller.set_gains(gains)
        else:
            # Fixed gains, set once in reset() -- action here is a torque
            # correction, not a gain schedule.
            gains = self._residual_fixed_gains

        t_s = float(self.data.time)
        target_x_now, target_x_vel_now = x_profile_target(
            "min_jerk_move_hold",
            float(self._state0.ee_pos[0]),
            float(self._target_x_delta),
            t_s,
            self._max_episode_seconds,
            move_duration_s=self._move_duration_s,
        )
        target_ee_pos = np.array([target_x_now, self._state0.ee_pos[1], self._state0.ee_pos[2]], dtype=np.float64)
        target_ee_vel = np.array([target_x_vel_now, 0.0, 0.0], dtype=np.float64)

        pre_state = build_mujoco_state(
            self.model, self.data,
            site_id=self.site_id, joint_ids=self.joint_ids,
            time_s=t_s, dt_s=float(self.model.opt.timestep),
            target_x=target_x_now, target_x_vel=target_x_vel_now,
            target_axis=target_ee_pos[0], target_axis_vel=target_ee_vel[0],
            target_ee_pos=target_ee_pos, target_ee_vel=target_ee_vel,
            reference_quat=self._state0.reference_quat,
            hold_current_pose=self._hold_current_pose_flag,
            transport_axis_index=0,
            gravity_compensation=True,
        )
        controller_state = pre_state
        if self._q_noise_std > 0.0 or self._qd_noise_std > 0.0:
            controller_state = dataclasses.replace(
                pre_state,
                q=pre_state.q + (self._noise_rng.normal(0.0, self._q_noise_std, size=6) if self._q_noise_std > 0.0 else 0.0),
                qd=pre_state.qd + (self._noise_rng.normal(0.0, self._qd_noise_std, size=6) if self._qd_noise_std > 0.0 else 0.0),
            )
        tau, diag = self.adapter.step(state=controller_state)
        tau = np.asarray(tau, dtype=np.float64).reshape(6)
        residual_tau = None
        if self._action_mode == "residual_torque":
            residual_tau = rescale_action_to_residual_tau(action, self._residual_max_nm)
            # Re-clip to the configured joint torque limits: adapter.step()
            # already shaped/clipped its own output before returning, and an
            # unbounded residual added on top can otherwise exceed those
            # limits.
            tau = np.clip(tau + residual_tau, -self.adapter.torque_limit_nm, self.adapter.torque_limit_nm)
        if self._torque_noise_std > 0.0:
            tau = tau + self._noise_rng.normal(0.0, self._torque_noise_std, size=6)
        self.data.ctrl[:6] = tau
        mujoco.mj_step(self.model, self.data)
        self._step_count += 1

        post_state = build_mujoco_state(
            self.model, self.data,
            site_id=self.site_id, joint_ids=self.joint_ids,
            time_s=float(self.data.time), dt_s=float(self.model.opt.timestep),
            target_x=target_x_now, target_x_vel=target_x_vel_now,
            target_axis=target_ee_pos[0], target_axis_vel=target_ee_vel[0],
            target_ee_pos=target_ee_pos, target_ee_vel=target_ee_vel,
            reference_quat=self._state0.reference_quat,
            hold_current_pose=self._hold_current_pose_flag,
            transport_axis_index=0,
            gravity_compensation=True,
        )

        x_error = float(diag.get("axis_error", target_x_now - float(post_state.ee_pos[0])))
        y_error = float(self._state0.ee_pos[1] - post_state.ee_pos[1])
        z_error = float(self._state0.ee_pos[2] - post_state.ee_pos[2])
        orientation_error_norm = float(diag.get("orientation_error_norm", 0.0))
        tau_applied = np.asarray(diag.get("tau_applied", tau), dtype=np.float64).reshape(6)
        tau_controller = np.asarray(diag.get("tau_controller", tau), dtype=np.float64).reshape(6)

        safety_ok = bool(diag.get("safety_ok", True))
        terminated = not safety_ok
        truncated = self._step_count >= self._max_episode_steps

        # Phase-gating (default 1.0 = no change from before): the move phase
        # necessarily produces some transient y/z/orientation disturbance
        # while the policy is still learning good mid-move gains -- penalizing
        # that as heavily as a HOLD-phase failure (where zero disturbance is
        # the actual expectation) is exactly the deceptive-gradient problem
        # diagnosed in the prior training run: "never move" avoids move-phase
        # disturbance penalties entirely, and that avoidance can look better
        # than a still-learning attempt to move. This does NOT relax what
        # counts as success -- the terminal safety penalty (hard guard trip)
        # and terminal quality score below are unchanged and are the real
        # pass/fail signal; this only reshapes the dense per-step gradient.
        in_move_phase = t_s < self._move_duration_s
        disturbance_scale = (
            float(self._reward_cfg.get("move_phase_disturbance_scale", 1.0)) if in_move_phase else 1.0
        )

        abs_x_error = abs(x_error)
        # Potential-based progress shaping (default weight 0.0 = no change):
        # reward REDUCTION in |x_error| this step, not just its magnitude.
        # Policy-invariant (Ng et al. 1999) -- doesn't change the optimal
        # policy, only makes the "actually move toward the target" gradient
        # easier to follow than the deceptive "never move" local optimum.
        progress_weight = float(self._reward_cfg.get("progress_weight", 0.0))
        progress_reward = progress_weight * (self._prev_abs_x_error - abs_x_error)
        self._prev_abs_x_error = abs_x_error

        reward = self._reward_cfg["alive_bonus"]
        reward -= self._reward_cfg["x_hold_weight"] * abs_x_error
        reward -= disturbance_scale * self._reward_cfg["y_hold_weight"] * abs(y_error)
        reward -= disturbance_scale * self._reward_cfg["z_hold_weight"] * abs(z_error)
        reward -= disturbance_scale * self._reward_cfg["orientation_weight"] * abs(orientation_error_norm)
        reward -= self._reward_cfg["gain_smooth_weight"] * float(np.sum((action - self._prev_action) ** 2))
        reward -= self._reward_cfg["torque_smooth_weight"] * float(np.sum((tau - self._prev_tau) ** 2))
        reward += progress_reward

        if self._record_lightweight_trace or self._trace_writer is not None:
            row = {
                "step": int(self._step_count),
                "time_s": float(self.data.time),
                "ee_pos": np.asarray(post_state.ee_pos, dtype=np.float64).tolist(),
                "ee_quat": np.asarray(post_state.ee_quat, dtype=np.float64).tolist(),
                "qd": np.asarray(post_state.qd, dtype=np.float64).tolist(),
                "orientation_error_norm": orientation_error_norm,
                "x_error": x_error,
                "target_x": float(target_x_now),
                "tau_controller": tau_controller.tolist(),
                "tau_applied": tau_applied.tolist(),
                "tau": np.asarray(tau, dtype=np.float64).tolist(),
                # "tau_total" duplicates "tau" (the actually-applied sum) in
                # both modes -- kept as an explicit named field for
                # trace-schema continuity with "tau_controller" and
                # "residual_tau" in residual_torque mode.
                "tau_total": np.asarray(tau, dtype=np.float64).tolist(),
                "residual_tau": (residual_tau.tolist() if residual_tau is not None else [0.0] * 6),
                "gains": {name: float(gains[name]) for name in GAIN_FIELDS},
            }
            if self._record_lightweight_trace:
                self._lightweight_trace.append(row)
            if self._trace_writer is not None:
                self._trace_writer.write_row(row)

        info: dict[str, Any] = {
            "height_alpha": self._alpha,
            "target_x_delta": self._target_x_delta,
            "reward_components": {
                "progress_reward": progress_reward,
                "disturbance_scale": disturbance_scale,
                "in_move_phase": in_move_phase,
            },
        }
        if terminated:
            info["termination_reason"] = diag.get("safety_reason", "")
            reward += self._reward_cfg["terminal_safety_penalty"]
        elif truncated:
            quality = self._episode_end_quality_score(termination_reason="duration_complete")
            info["move_hold_quality_score"] = quality
            reward += self._reward_cfg["terminal_quality_weight"] * quality

        if terminated or truncated:
            self._finalize_trace_logging(terminated=terminated, safety_reason=diag.get("safety_reason", ""))

        self._prev_action = action.astype(np.float32).copy()
        self._prev_tau = np.asarray(tau, dtype=np.float64).copy()

        obs, _ = self._build_obs_and_info(
            ee_pos=post_state.ee_pos, ee_quat=post_state.ee_quat, q=post_state.q, qd=post_state.qd,
            ee_lin_vel=post_state.ee_lin_vel, ee_ang_vel=post_state.ee_ang_vel,
            x_error=x_error, y_error=y_error, z_error=z_error, orientation_error_norm=orientation_error_norm,
        )
        return obs, float(reward), terminated, truncated, info

    # -- helpers -----------------------------------------------------------

    def _padded_prev_action_for_obs(self) -> np.ndarray:
        """self._prev_action, zero-padded to ACTION_DIM (11) for the obs vector.

        OBS_DIM is a fixed 47 regardless of action_mode, so residual_torque
        mode's 6-dim actions occupy the first 6 slots of the same 11-wide
        "previous action" observation block gains mode fills completely --
        keeps _build_obs_and_info's shape assertion and OBS_DIM constant
        unchanged across both modes.
        """
        if self._action_dim == ACTION_DIM:
            return self._prev_action
        padded = np.zeros(ACTION_DIM, dtype=np.float32)
        padded[: self._prev_action.shape[0]] = self._prev_action
        return padded

    def _build_obs_and_info(
        self, *, ee_pos, ee_quat, q, qd, ee_lin_vel, ee_ang_vel, x_error, y_error, z_error, orientation_error_norm,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        q = np.asarray(q, dtype=np.float64).reshape(6)
        qd = np.asarray(qd, dtype=np.float64).reshape(6)
        ori_vec = orientation_error_vec_wxyz(
            np.asarray(self._state0.reference_quat if self._state0 is not None else [1.0, 0.0, 0.0, 0.0], dtype=np.float64),
            np.asarray(ee_quat, dtype=np.float64),
        )
        elapsed = float(self.data.time)
        move_phase_indicator = 1.0 if elapsed >= self._move_duration_s else float(elapsed / max(self._move_duration_s, 1e-9))
        obs = np.concatenate([
            np.sin(q), np.cos(q),
            qd / 1.5,
            np.asarray(ee_pos, dtype=np.float64).reshape(3),
            [x_error, y_error, z_error],
            ori_vec,
            np.asarray(ee_lin_vel, dtype=np.float64).reshape(3),
            np.asarray(ee_ang_vel, dtype=np.float64).reshape(3),
            [elapsed / max(self._max_episode_seconds, 1e-9)],
            [move_phase_indicator],
            [self._target_x_delta],
            self._padded_prev_action_for_obs(),
        ]).astype(np.float32)
        assert obs.shape == (OBS_DIM,), f"obs shape {obs.shape} != ({OBS_DIM},)"
        return obs, {}

    def _episode_end_quality_score(self, *, termination_reason: str) -> float:
        if not self._lightweight_trace:
            return 0.0
        run_summary = summarize_move_hold_trace(
            self._lightweight_trace,
            initial_ee_pos=self._state0.ee_pos if self._state0 is not None else None,
            move_duration_s=self._move_duration_s,
            total_duration_s=self._max_episode_seconds,
            transport_axis_index=0,
        )
        run_summary.update({
            "target_x_delta": self._target_x_delta,
            "termination_reason": termination_reason,
            "success": termination_reason == "duration_complete",
            "velocity_guard_ok": True,
            "joint_limit_guard_ok": True,
            "torque_saturation_percentage": 0.0,
            "sim_time_s": float(self.data.time),
            "duration_s": self._max_episode_seconds,
            "max_abs_qd_radps": float(np.max(np.abs([row["qd"] for row in self._lightweight_trace]))),
        })
        metrics = compute_valid_move_hold_metrics(run_summary, strict=False)
        return float(metrics.get("move_hold_quality_score", 0.0))

    def _finalize_trace_logging(self, *, terminated: bool, safety_reason: str) -> None:
        if self._trace_writer is None:
            return
        self._trace_writer.__exit__(None, None, None)
        termination_reason = safety_reason if terminated else "duration_complete"
        quality = self._episode_end_quality_score(termination_reason=termination_reason)
        run_summary = summarize_move_hold_trace(
            self._lightweight_trace,
            initial_ee_pos=self._state0.ee_pos if self._state0 is not None else None,
            move_duration_s=self._move_duration_s,
            total_duration_s=self._max_episode_seconds,
            transport_axis_index=0,
        )
        run_summary.update({
            "target_x_delta": self._target_x_delta,
            "height_alpha": self._alpha,
            "termination_reason": termination_reason,
            "success": not terminated,
            "velocity_guard_ok": not terminated,
            "joint_limit_guard_ok": True,
            "torque_saturation_percentage": 0.0,
            "sim_time_s": float(self.data.time),
            "duration_s": self._max_episode_seconds,
            "trace_path": str(self._trace_path),
        })
        if self._lightweight_trace:
            run_summary["max_abs_qd_radps"] = float(np.max(np.abs([row["qd"] for row in self._lightweight_trace])))
        # RunLogger reads these bare (non-phase-prefixed) names; derive them
        # from the phase-prefixed fields summarize_move_hold_trace already
        # computed, so RunLogger records for env-driven runs are as complete
        # as the CLI's (which populates both).
        run_summary.setdefault("achieved_x_delta_m", run_summary.get("hold_phase_achieved_x_delta_m", run_summary.get("move_phase_achieved_x_delta_m", 0.0)))
        run_summary.setdefault("final_x_error_m", run_summary.get("hold_phase_final_x_error_m", 0.0))
        run_summary.setdefault("max_abs_x_error_m", max(run_summary.get("move_phase_max_abs_x_error_m", 0.0), run_summary.get("hold_phase_max_abs_x_error_m", 0.0)))
        run_summary.setdefault("max_abs_y_drift_m", max(run_summary.get("move_phase_max_abs_y_drift_m", 0.0), run_summary.get("hold_phase_max_abs_y_drift_m", 0.0)))
        run_summary.setdefault("max_abs_z_drift_m", max(run_summary.get("move_phase_max_abs_z_drift_m", 0.0), run_summary.get("hold_phase_max_abs_z_drift_m", 0.0)))
        run_summary.setdefault("max_abs_orientation_error_rad", max(run_summary.get("move_phase_max_abs_orientation_error_rad", 0.0), run_summary.get("hold_phase_max_abs_orientation_error_rad", 0.0)))
        run_summary.setdefault("final_orientation_error_rad", run_summary.get("hold_phase_max_abs_orientation_error_rad", 0.0))
        run_summary.update(compute_valid_move_hold_metrics(run_summary, strict=False))
        if self._output_dir is not None:
            run_summary["summary_path"] = str(self._output_dir / "summary.json")
            (self._output_dir / "summary.json").write_text(json_dumps_safe(run_summary), encoding="utf-8")
        self._trace_writer = None

    def close(self) -> None:
        if self._trace_writer is not None:
            self._trace_writer.__exit__(None, None, None)
            self._trace_writer = None
