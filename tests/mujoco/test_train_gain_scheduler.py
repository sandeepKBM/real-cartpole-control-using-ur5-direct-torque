"""Tests for rl_gain_scheduling/train_ppo_gain_scheduler.py."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from rl_gain_scheduling.train_ppo_gain_scheduler import (  # noqa: E402
    _build_ppo_kwargs,
    _build_sac_kwargs,
)

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
