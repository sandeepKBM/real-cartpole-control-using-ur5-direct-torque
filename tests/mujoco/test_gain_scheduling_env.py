"""Tests for rl_gain_scheduling/gain_scheduling_env.py."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from rl_gain_scheduling.gain_scheduling_env import (  # noqa: E402
    ACTION_DIM,
    OBS_DIM,
    GainSchedulingEnv,
    gains_to_normalized_action,
    rescale_action_to_gains,
)
from transport_metrics import GAIN_FIELDS  # noqa: E402


def test_env_reset_and_step_shapes():
    env = GainSchedulingEnv()
    obs, info = env.reset(seed=0, options={"height_alpha": 0.0, "target_x_delta": 0.02})
    assert obs.shape == (OBS_DIM,)
    assert obs.dtype == np.float32
    assert np.all(np.isfinite(obs))

    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    assert obs.shape == (OBS_DIM,)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert isinstance(info, dict)


def test_env_height_sampling_respects_joint_limits():
    env = GainSchedulingEnv()
    for alpha in (0.0, 0.25, 0.5, 0.75, 1.0):
        obs, info = env.reset(seed=0, options={"height_alpha": alpha, "target_x_delta": 0.0})
        q = env.data.qpos[:6].copy()
        for jid, qi in zip(env.joint_ids, q):
            qmin, qmax = float(env.model.jnt_range[jid, 0]), float(env.model.jnt_range[jid, 1])
            if qmax > qmin:
                assert qmin <= qi <= qmax
        # wrist_2 (index 4) stays at the singularity throughout the family.
        assert abs(q[4]) < 1e-9


def test_env_rejects_infeasible_extrapolated_alpha():
    env = GainSchedulingEnv()
    # Way outside [0,1]: some joint should exceed its range (elbow especially,
    # since it has the tightest limits, +/-pi).
    with pytest.raises(ValueError):
        env.reset(seed=0, options={"height_alpha": 50.0, "target_x_delta": 0.0})


def test_gain_rescale_round_trip():
    env = GainSchedulingEnv()
    tuned_gains = {
        "kp_x": 400.0, "kd_x": 40.0, "kp_y": 80.0, "kd_y": 15.0, "kp_z": 120.0, "kd_z": 20.0,
        "kp_rot": 0.0, "kd_rot": 10.0, "kp_posture": 25.0, "kd_posture": 6.0, "kd_joint": 4.0,
    }
    action = gains_to_normalized_action(tuned_gains, env._gain_bounds)
    assert action.shape == (ACTION_DIM,)
    assert np.all(np.abs(action) <= 1.0 + 1e-9)
    recovered = rescale_action_to_gains(action, env._gain_bounds)
    for name in GAIN_FIELDS:
        assert recovered[name] == pytest.approx(tuned_gains[name], abs=1e-6)


def test_env_set_gains_actually_changes_torque():
    # A min-jerk ramp's s(a) ~ a^3 near a=0, so tracking error is still
    # negligible after just a handful of steps regardless of gains -- step
    # forward enough that the ramp has moved meaningfully (~150-300 steps
    # for a 1.0s move at 500Hz) before comparing.
    env = GainSchedulingEnv()
    action_low = np.full(ACTION_DIM, -0.9, dtype=np.float32)
    action_high = np.full(ACTION_DIM, 0.9, dtype=np.float32)

    env.reset(seed=0, options={"height_alpha": 0.0, "target_x_delta": 0.03})
    for _ in range(300):
        _, _, _, _, _ = env.step(action_low)
    tau_low = env._prev_tau.copy()

    env.reset(seed=0, options={"height_alpha": 0.0, "target_x_delta": 0.03})
    for _ in range(300):
        _, _, _, _, _ = env.step(action_high)
    tau_high = env._prev_tau.copy()

    assert not np.allclose(tau_low, tau_high, atol=1e-6), "different gain actions should produce different torque"


def _config_with_reward_overrides(tmp_path: Path, **reward_overrides) -> Path:
    import yaml

    repo_root = Path(__file__).resolve().parents[2]
    cfg = yaml.safe_load((repo_root / "config" / "rl_gain_scheduling.yaml").read_text(encoding="utf-8"))
    cfg["reward"].update(reward_overrides)
    out = tmp_path / "reward_override.yaml"
    out.write_text(yaml.dump(cfg), encoding="utf-8")
    return out


def test_default_config_has_no_progress_or_phase_gating_effect():
    # progress_weight/move_phase_disturbance_scale are absent from the
    # shipped default config -- .get() fallbacks (0.0, 1.0) must make this
    # byte-for-byte the same reward formula as before these fields existed.
    env = GainSchedulingEnv()
    env.reset(seed=0, options={"height_alpha": 0.5, "target_x_delta": -0.05})
    action = np.zeros(ACTION_DIM, dtype=np.float32)
    _, _, _, _, info = env.step(action)
    comps = info["reward_components"]
    assert comps["progress_reward"] == 0.0
    assert comps["disturbance_scale"] == 1.0


def test_progress_weight_rewards_error_reduction(tmp_path: Path):
    cfg_path = _config_with_reward_overrides(tmp_path, progress_weight=10.0)
    env = GainSchedulingEnv(config_path=cfg_path)
    env.reset(seed=0, options={"height_alpha": 0.5, "target_x_delta": -0.05})
    # Reasonable gains (not all-zero/random) so tracking actually improves
    # step over step rather than wandering randomly.
    tuned = {
        "kp_x": 400.0, "kd_x": 40.0, "kp_y": 80.0, "kd_y": 15.0, "kp_z": 120.0, "kd_z": 20.0,
        "kp_rot": 0.0, "kd_rot": 10.0, "kp_posture": 25.0, "kd_posture": 6.0, "kd_joint": 4.0,
    }
    action = gains_to_normalized_action(tuned, env._gain_bounds)
    seen_progress = False
    # A min-jerk ramp's error typically grows for the first portion of the
    # move (target accelerates away faster than tracking catches up) before
    # recovering -- run the full episode, not just the first few steps, to
    # actually reach the recovery phase where progress_reward > 0 is expected.
    for _ in range(env._max_episode_steps):
        _, _, terminated, truncated, info = env.step(action)
        if info["reward_components"]["progress_reward"] > 0:
            seen_progress = True
        if terminated or truncated:
            break
    assert seen_progress, "expected at least one step where tracking improved and progress_reward > 0"


def test_move_phase_disturbance_scale_only_applies_during_move(tmp_path: Path):
    cfg_path = _config_with_reward_overrides(tmp_path, move_phase_disturbance_scale=0.25)
    env = GainSchedulingEnv(config_path=cfg_path)
    move_duration_s = env._move_duration_s
    env.reset(seed=0, options={"height_alpha": 0.5, "target_x_delta": -0.05})
    action = np.zeros(ACTION_DIM, dtype=np.float32)
    saw_move_phase_scaled = False
    saw_hold_phase_unscaled = False
    for _ in range(env._max_episode_steps):
        _, _, terminated, truncated, info = env.step(action)
        comps = info["reward_components"]
        if comps["in_move_phase"]:
            assert comps["disturbance_scale"] == pytest.approx(0.25)
            saw_move_phase_scaled = True
        else:
            assert comps["disturbance_scale"] == pytest.approx(1.0)
            saw_hold_phase_unscaled = True
        if terminated or truncated:
            break
    assert saw_move_phase_scaled
    assert saw_hold_phase_unscaled, f"episode never reached the hold phase (move_duration_s={move_duration_s})"


def test_env_safety_termination_reports_reason():
    env = GainSchedulingEnv()
    env.reset(seed=0, options={"height_alpha": 0.0, "target_x_delta": 0.03})
    # Extreme, wildly oscillating actions should eventually trip a safety
    # guard (velocity/orientation/drift) well before the episode truncates.
    rng = np.random.default_rng(0)
    terminated = truncated = False
    info = {}
    for _ in range(env._max_episode_steps):
        action = rng.uniform(-1.0, 1.0, size=ACTION_DIM).astype(np.float32)
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break
    if terminated:
        assert "termination_reason" in info
        assert isinstance(info["termination_reason"], str)
        assert len(info["termination_reason"]) > 0
    # If it didn't terminate (truncated instead), that's also acceptable --
    # this test only asserts the info contract when termination happens.


def test_env_full_trace_logging_produces_valid_move_hold_metrics(tmp_path):
    env = GainSchedulingEnv()
    env._full_trace_logging = True
    tuned_gains = {
        "kp_x": 400.0, "kd_x": 40.0, "kp_y": 80.0, "kd_y": 15.0, "kp_z": 120.0, "kd_z": 20.0,
        "kp_rot": 0.0, "kd_rot": 10.0, "kp_posture": 25.0, "kd_posture": 6.0, "kd_joint": 4.0,
    }
    action = gains_to_normalized_action(tuned_gains, env._gain_bounds)
    output_dir = tmp_path / "eval_run"
    env.reset(seed=0, options={"height_alpha": 0.0, "target_x_delta": 0.02, "output_dir": str(output_dir)})
    terminated = truncated = False
    while not (terminated or truncated):
        obs, reward, terminated, truncated, info = env.step(action)
    env.close()

    assert (output_dir / "trace.jsonl").exists()
    assert (output_dir / "summary.json").exists()
    import json
    summary = json.loads((output_dir / "summary.json").read_text())
    assert "move_hold_quality_score" in summary
    assert "valid_move_and_hold" in summary
    assert summary["move_hold_quality_score"] > 0.5


if __name__ == "__main__":
    test_env_reset_and_step_shapes()
    test_env_height_sampling_respects_joint_limits()
    test_env_rejects_infeasible_extrapolated_alpha()
    test_gain_rescale_round_trip()
    test_env_set_gains_actually_changes_torque()
    test_env_safety_termination_reports_reason()
    print("gain scheduling env tests OK")
