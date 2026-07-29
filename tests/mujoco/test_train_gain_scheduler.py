"""Tests for rl_gain_scheduling/train_ppo_gain_scheduler.py."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from rl_gain_scheduling.train_ppo_gain_scheduler import (  # noqa: E402
    HeightAlphaCurriculumCallback,
    _build_ppo_kwargs,
    _build_sac_kwargs,
)

_CURRICULUM_STAGES = [
    {"alpha_range": [0.05, 0.15], "timestep_frac": 0.20},
    {"alpha_range": [0.10, 0.25], "timestep_frac": 0.20},
    {"alpha_range": [0.15, 0.35], "timestep_frac": 0.20},
    {"alpha_range": [0.25, 0.45], "timestep_frac": 0.20},
    {"alpha_range": [0.35, 0.50], "timestep_frac": 0.20},
]

_PPO_TRAINING_CFG = {
    "policy": "MlpPolicy",
    "net_arch": [128, 128],
    "learning_rate": 3.0e-4,
    "n_steps": 1024,
    "batch_size": 64,
    "n_epochs": 10,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "ent_coef": 0.01,
    "total_timesteps": 3000000,
    "n_envs": 57,
    "device": "cpu",
    "seed": 0,
}

_SAC_TRAINING_CFG = {
    "policy": "MlpPolicy",
    "net_arch": [128, 128],
    "sac": {
        "learning_rate": 3.0e-4,
        "buffer_size": 200000,
        "learning_starts": 5000,
        "batch_size": 256,
        "polyak_tau": 0.005,
        "gamma": 0.99,
        "train_freq": 1,
        "gradient_steps": 1,
        "action_noise": None,
        "ent_coef": "auto",
        "target_update_interval": 1,
        "target_entropy": "auto",
    },
}


def test_train_script_algo_ppo_default_unchanged():
    # Byte-identical to this script's original inlined PPO(...) kwargs.
    kwargs = _build_ppo_kwargs(_PPO_TRAINING_CFG)
    assert kwargs == {
        "policy": "MlpPolicy",
        "learning_rate": 3.0e-4,
        "n_steps": 1024,
        "batch_size": 64,
        "n_epochs": 10,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "ent_coef": 0.01,
        "policy_kwargs": {"net_arch": [128, 128]},
    }


def test_train_script_ppo_ent_coef_defaults_to_zero_when_absent():
    cfg = dict(_PPO_TRAINING_CFG)
    del cfg["ent_coef"]
    kwargs = _build_ppo_kwargs(cfg)
    assert kwargs["ent_coef"] == 0.0


def test_train_script_sac_kwargs_from_config():
    kwargs = _build_sac_kwargs(_SAC_TRAINING_CFG)
    # polyak_tau -> tau, the SB3 constructor kwarg name.
    assert kwargs["tau"] == pytest.approx(0.005)
    assert "polyak_tau" not in kwargs
    assert kwargs["ent_coef"] == "auto"
    assert kwargs["policy"] == "MlpPolicy"
    assert kwargs["buffer_size"] == 200000
    assert kwargs["learning_starts"] == 5000
    # PPO-only keys never leak into SAC kwargs.
    for ppo_only_key in ("n_steps", "n_epochs", "gae_lambda", "clip_range"):
        assert ppo_only_key not in kwargs


def test_train_script_sac_kwargs_ent_coef_float_supported():
    cfg = {
        "policy": "MlpPolicy",
        "net_arch": [128, 128],
        "sac": dict(_SAC_TRAINING_CFG["sac"], ent_coef=0.1),
    }
    kwargs = _build_sac_kwargs(cfg)
    assert kwargs["ent_coef"] == pytest.approx(0.1)


def test_train_script_sac_kwargs_rejects_unsupported_action_noise():
    cfg = {
        "policy": "MlpPolicy",
        "net_arch": [128, 128],
        "sac": dict(_SAC_TRAINING_CFG["sac"], action_noise="ornstein_uhlenbeck"),
    }
    with pytest.raises(NotImplementedError):
        _build_sac_kwargs(cfg)


def test_sac_smoke_train_two_steps():
    """Real training smoke test (not just import-checking): proves SAC
    actually trains against GainSchedulingEnv without crashing, matching
    this session's standing rule to trust the real test suite over a quick
    manual script."""
    from stable_baselines3 import SAC
    from stable_baselines3.common.vec_env import DummyVecEnv

    from rl_gain_scheduling.gain_scheduling_env import GainSchedulingEnv

    repo_root = Path(__file__).resolve().parents[2]
    config_path = repo_root / "config" / "rl_gain_scheduling_alpha05_bidirectional.yaml"
    env = DummyVecEnv([lambda: GainSchedulingEnv(config_path=config_path)])

    training_cfg = {
        "policy": "MlpPolicy",
        "net_arch": [16, 16],
        "sac": {
            "learning_rate": 3.0e-4,
            "buffer_size": 200,
            "learning_starts": 32,
            "batch_size": 16,
            "polyak_tau": 0.005,
            "gamma": 0.99,
            "train_freq": 1,
            "gradient_steps": 1,
            "action_noise": None,
            "ent_coef": "auto",
            "target_update_interval": 1,
            "target_entropy": "auto",
        },
    }
    kwargs = _build_sac_kwargs(training_cfg)
    policy = kwargs.pop("policy")
    model = SAC(policy, env, device="cpu", seed=0, verbose=0, **kwargs)
    model.learn(total_timesteps=64)


def test_curriculum_callback_rejects_bad_frac_sum():
    bad_stages = [
        {"alpha_range": [0.0, 0.5], "timestep_frac": 0.5},
        {"alpha_range": [0.0, 0.5], "timestep_frac": 0.2},
    ]  # sums to 0.7, not ~1.0
    with pytest.raises(ValueError):
        HeightAlphaCurriculumCallback(stages=bad_stages, total_timesteps=1000)


def test_curriculum_callback_rejects_empty_stages():
    with pytest.raises(ValueError):
        HeightAlphaCurriculumCallback(stages=[], total_timesteps=1000)


def test_curriculum_callback_stage_boundaries_are_progressive():
    """Pure-logic check of _stage_for_timestep against the real 5-stage
    config used by config/rl_gain_scheduling_sac_curriculum_alpha.yaml --
    each 20% quintile of total_timesteps should map to consecutive stage
    indices 0..4, and the very last timestep must resolve to the final
    stage exactly (boundary correctness matters most at both ends: t=0
    must be stage 0, t=total_timesteps-1 must be the last stage)."""
    total = 100_000
    cb = HeightAlphaCurriculumCallback(stages=_CURRICULUM_STAGES, total_timesteps=total)
    assert cb._stage_for_timestep(0) == 0
    assert cb._stage_for_timestep(19_999) == 0
    assert cb._stage_for_timestep(20_001) == 1
    assert cb._stage_for_timestep(39_999) == 1
    assert cb._stage_for_timestep(40_001) == 2
    assert cb._stage_for_timestep(60_001) == 3
    assert cb._stage_for_timestep(80_001) == 4
    assert cb._stage_for_timestep(total - 1) == 4
    # Monotonic non-decreasing across the whole range, no skipped-back stages.
    prev = -1
    for t in range(0, total, 1000):
        idx = cb._stage_for_timestep(t)
        assert idx >= prev
        prev = idx


def test_curriculum_callback_reaches_live_dummyvecenv_workers():
    """Integration test proving env_method genuinely reaches worker envs
    mid-training, not just that the callback's own bookkeeping is correct
    in isolation -- this is the exact mechanism the real 3M-step curriculum
    run depends on. Uses a tiny total_timesteps and tiny stage fractions so
    multiple real stage transitions happen within a fast test."""
    from stable_baselines3 import SAC
    from stable_baselines3.common.vec_env import DummyVecEnv

    from rl_gain_scheduling.gain_scheduling_env import GainSchedulingEnv

    repo_root = Path(__file__).resolve().parents[2]
    config_path = repo_root / "config" / "rl_gain_scheduling_alpha05_bidirectional.yaml"
    n_envs = 2
    env = DummyVecEnv([lambda: GainSchedulingEnv(config_path=config_path) for _ in range(n_envs)])

    tiny_stages = [
        {"alpha_range": [0.0, 0.1], "timestep_frac": 0.5},
        {"alpha_range": [0.4, 0.5], "timestep_frac": 0.5},
    ]
    total_timesteps = 64
    callback = HeightAlphaCurriculumCallback(stages=tiny_stages, total_timesteps=total_timesteps)

    training_cfg = {
        "policy": "MlpPolicy",
        "net_arch": [16, 16],
        "sac": {
            "learning_rate": 3.0e-4,
            "buffer_size": 200,
            "learning_starts": 16,
            "batch_size": 16,
            "polyak_tau": 0.005,
            "gamma": 0.99,
            "train_freq": 1,
            "gradient_steps": 1,
            "action_noise": None,
            "ent_coef": "auto",
            "target_update_interval": 1,
            "target_entropy": "auto",
        },
    }
    kwargs = _build_sac_kwargs(training_cfg)
    policy = kwargs.pop("policy")
    model = SAC(policy, env, device="cpu", seed=0, verbose=0, **kwargs)
    model.learn(total_timesteps=total_timesteps, callback=callback)

    # By the end of training every worker env's live height_alpha_range must
    # have been pushed to the final stage's range -- proves env_method
    # actually mutated the live objects, not copies (DummyVecEnv holds the
    # real env instances directly, so this is a genuine end-to-end check).
    for worker_env in env.envs:
        assert worker_env._height_alpha_range == (0.4, 0.5)
