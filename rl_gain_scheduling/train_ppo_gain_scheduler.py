#!/usr/bin/env python3
"""PPO training entrypoint for the height-conditioned gain-scheduling policy.

Simulation-only. Trains a policy that outputs the 11 Cartesian-impedance
gains every control step, conditioned on live state including end-effector
height, via GainSchedulingEnv (rl_gain_scheduling/gain_scheduling_env.py).

MuJoCo runs in-process per worker -- unlike the archived CoppeliaSim RL
lane (archive/coppelia/rl/), there is no external simulator process or
ZMQ port to manage, so vectorization is a plain SB3 VecEnv over N copies
of the same env class.
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

from stable_baselines3 import PPO  # noqa: E402
from stable_baselines3.common.callbacks import CheckpointCallback  # noqa: E402
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv  # noqa: E402

from rl_gain_scheduling.gain_scheduling_env import GainSchedulingEnv  # noqa: E402

DEFAULT_CONFIG_PATH = REPO_ROOT / "config" / "rl_gain_scheduling.yaml"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "outputs" / "rl_gain_scheduling"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
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


def main() -> int:
    args = parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    training_cfg: dict[str, Any] = cfg["training"]

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

    policy_kwargs = {"net_arch": list(training_cfg["net_arch"])}

    if args.resume is not None:
        model = PPO.load(str(args.resume), env=vec_env, tensorboard_log=str(tb_dir), device=device)
    else:
        model = PPO(
            training_cfg["policy"],
            vec_env,
            learning_rate=float(training_cfg["learning_rate"]),
            n_steps=int(training_cfg["n_steps"]),
            batch_size=int(training_cfg["batch_size"]),
            n_epochs=int(training_cfg["n_epochs"]),
            gamma=float(training_cfg["gamma"]),
            gae_lambda=float(training_cfg["gae_lambda"]),
            clip_range=float(training_cfg["clip_range"]),
            policy_kwargs=policy_kwargs,
            tensorboard_log=str(tb_dir),
            device=device,
            seed=seed,
            verbose=1,
        )

    checkpoint_callback = CheckpointCallback(
        save_freq=max(args.checkpoint_save_freq // n_envs, 1),
        save_path=str(checkpoint_dir),
        name_prefix="ppo_gain_scheduler",
    )

    try:
        model.learn(total_timesteps=total_timesteps, callback=checkpoint_callback, progress_bar=False)
    except KeyboardInterrupt:
        print("Training interrupted -- saving current model before exit.")

    final_model_path = model_dir / "ppo_gain_scheduler_final.zip"
    model.save(str(final_model_path))
    print(f"Saved final model to {final_model_path}")
    print(f"Checkpoints under {checkpoint_dir}")
    print(f"TensorBoard logs under {tb_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
