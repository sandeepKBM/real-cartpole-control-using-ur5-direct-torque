#!/usr/bin/env python3
"""PPO/SAC training entrypoint for the height-conditioned gain-scheduling policy.

Simulation-only. Trains a policy conditioned on live state including
end-effector height, via GainSchedulingEnv (rl_gain_scheduling/
gain_scheduling_env.py) -- outputting either the 11 Cartesian-impedance
gains every control step (env.action_mode "gains", the original design) or
a 6-dim residual torque correction (env.action_mode "residual_torque"),
depending on the env's own config.

MuJoCo runs in-process per worker -- unlike the archived CoppeliaSim RL
lane (archive/coppelia/rl/), there is no external simulator process or
ZMQ port to manage, so vectorization is a plain SB3 VecEnv over N copies
of the same env class.

--algo {ppo,sac} selects the RL algorithm; default "ppo" reproduces this
script's original behavior exactly (byte-identical kwargs, output filenames).
SAC is off-policy and reads a separate training.sac.* config block --
PPO-only keys (n_steps, n_epochs, gae_lambda, clip_range) are never read
under --algo sac, and SAC-only keys are never read under --algo ppo.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from stable_baselines3 import PPO, SAC  # noqa: E402
from stable_baselines3.common.callbacks import CheckpointCallback  # noqa: E402
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv  # noqa: E402

from rl_gain_scheduling.gain_scheduling_env import GainSchedulingEnv  # noqa: E402

DEFAULT_CONFIG_PATH = REPO_ROOT / "config" / "rl_gain_scheduling.yaml"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "outputs" / "rl_gain_scheduling"
ALGO_CLASSES = {"ppo": PPO, "sac": SAC}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    p.add_argument("--algo", type=str, choices=["ppo", "sac"], default="ppo")
    p.add_argument("--output-root", type=Path, default=None, help="Defaults under outputs/rl_gain_scheduling/.")
    p.add_argument("--run-name", type=str, default=None, help="Defaults to a timestamp.")
    p.add_argument("--total-timesteps", type=int, default=None, help="Overrides config.training.total_timesteps.")
    p.add_argument("--n-envs", type=int, default=None, help="Overrides config.training.n_envs.")
    p.add_argument("--device", type=str, default=None, help="Overrides config.training.device ('cpu' or 'cuda').")
    p.add_argument("--seed", type=int, default=None, help="Overrides config.training.seed.")
    p.add_argument("--resume", type=Path, default=None, help="Path to a saved model .zip to resume training from.")
    p.add_argument(
        "--checkpoint-save-freq",
        type=int,
        default=10_000,
        help="Save a checkpoint every N timesteps (per-env; total = save_freq // n_envs).",
    )
    return p.parse_args()


def _make_env_factory(config_path: Path):
    def _init():
        return GainSchedulingEnv(config_path=config_path)
    return _init


def _build_ppo_kwargs(training_cfg: dict[str, Any]) -> dict[str, Any]:
    """Pure mapping from config.training to PPO(...) constructor kwargs.

    Unchanged from this script's original inlined behavior: ent_coef
    defaults to 0.0 (matching SB3's own default) if the config doesn't set
    it, so existing configs without the key are unaffected.
    """
    return {
        "policy": training_cfg["policy"],
        "learning_rate": float(training_cfg["learning_rate"]),
        "n_steps": int(training_cfg["n_steps"]),
        "batch_size": int(training_cfg["batch_size"]),
        "n_epochs": int(training_cfg["n_epochs"]),
        "gamma": float(training_cfg["gamma"]),
        "gae_lambda": float(training_cfg["gae_lambda"]),
        "clip_range": float(training_cfg["clip_range"]),
        "ent_coef": float(training_cfg.get("ent_coef", 0.0)),
        "policy_kwargs": {"net_arch": list(training_cfg["net_arch"])},
    }


def _build_sac_kwargs(training_cfg: dict[str, Any]) -> dict[str, Any]:
    """Pure mapping from config.training.sac to SAC(...) constructor kwargs.

    Naming: SB3 names the polyak-averaging coefficient `tau`, which in this
    codebase is reserved for joint torque everywhere else -- the config key
    is `polyak_tau`, mapped explicitly to `tau=` only here at the SB3 call
    boundary. ent_coef supports SB3's native "auto" string for automatic
    entropy tuning (PPO's ent_coef, by contrast, is float-only and lives
    directly under config.training, never config.training.sac).
    """
    sac_cfg = training_cfg["sac"]
    action_noise_cfg = sac_cfg.get("action_noise")
    if action_noise_cfg is not None:
        raise NotImplementedError(
            f"config.training.sac.action_noise={action_noise_cfg!r} is not supported; "
            "only null (SAC's own built-in stochastic exploration) is implemented."
        )
    ent_coef_cfg = sac_cfg.get("ent_coef", "auto")
    ent_coef = ent_coef_cfg if ent_coef_cfg == "auto" else float(ent_coef_cfg)
    return {
        "policy": training_cfg["policy"],
        "learning_rate": float(sac_cfg["learning_rate"]),
        "buffer_size": int(sac_cfg["buffer_size"]),
        "learning_starts": int(sac_cfg["learning_starts"]),
        "batch_size": int(sac_cfg["batch_size"]),
        "tau": float(sac_cfg["polyak_tau"]),
        "gamma": float(sac_cfg["gamma"]),
        "train_freq": int(sac_cfg["train_freq"]),
        "gradient_steps": int(sac_cfg["gradient_steps"]),
        "action_noise": None,
        "ent_coef": ent_coef,
        "target_update_interval": int(sac_cfg["target_update_interval"]),
        "target_entropy": sac_cfg.get("target_entropy", "auto"),
        "policy_kwargs": {"net_arch": list(training_cfg["net_arch"])},
    }


def main() -> int:
    args = parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    training_cfg: dict[str, Any] = cfg["training"]
    algo = args.algo
    model_cls = ALGO_CLASSES[algo]

    total_timesteps = int(args.total_timesteps if args.total_timesteps is not None else training_cfg["total_timesteps"])
    n_envs = int(args.n_envs if args.n_envs is not None else training_cfg["n_envs"])
    device = str(args.device if args.device is not None else training_cfg["device"])
    seed = int(args.seed if args.seed is not None else training_cfg["seed"])

    from datetime import datetime

    run_name = args.run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = (args.output_root or DEFAULT_OUTPUT_ROOT) / run_name
    checkpoint_dir = output_root / "checkpoints"
    model_dir = output_root / "models"
    tb_dir = output_root / "tb"
    for d in (checkpoint_dir, model_dir, tb_dir):
        d.mkdir(parents=True, exist_ok=True)

    env_fns = [_make_env_factory(args.config) for _ in range(n_envs)]
    vec_env = DummyVecEnv(env_fns) if n_envs == 1 else SubprocVecEnv(env_fns)

    if args.resume is not None:
        model = model_cls.load(str(args.resume), env=vec_env, tensorboard_log=str(tb_dir), device=device)
        # --total-timesteps means "how many more steps to run" regardless of
        # resume. reset_num_timesteps=False keeps the loaded step count so
        # TensorBoard/checkpoint numbering continues rather than restarting
        # at 0 (SB3's learn() defaults reset_num_timesteps=True, which would
        # otherwise silently discard the checkpoint's own step count here).
        # Do NOT add model.num_timesteps to total_timesteps here: SB3's own
        # _setup_learn() already does `total_timesteps += self.num_timesteps`
        # whenever reset_num_timesteps=False, so total_timesteps must stay
        # as the raw increment or it gets added twice.
    else:
        kwargs = _build_ppo_kwargs(training_cfg) if algo == "ppo" else _build_sac_kwargs(training_cfg)
        policy = kwargs.pop("policy")
        model = model_cls(
            policy,
            vec_env,
            tensorboard_log=str(tb_dir),
            device=device,
            seed=seed,
            verbose=1,
            **kwargs,
        )

    checkpoint_callback = CheckpointCallback(
        save_freq=max(args.checkpoint_save_freq // n_envs, 1),
        save_path=str(checkpoint_dir),
        name_prefix=f"{algo}_gain_scheduler",
    )

    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=checkpoint_callback,
            progress_bar=False,
            reset_num_timesteps=(args.resume is None),
        )
    except KeyboardInterrupt:
        print("Training interrupted -- saving current model before exit.")

    # algo="ppo" (the default) reproduces the original filename exactly.
    final_model_path = model_dir / f"{algo}_gain_scheduler_final.zip"
    model.save(str(final_model_path))
    print(f"Saved final model to {final_model_path}")
    print(f"Checkpoints under {checkpoint_dir}")
    print(f"TensorBoard logs under {tb_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
