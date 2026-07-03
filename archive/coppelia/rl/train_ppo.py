#!/usr/bin/env python3
"""
Train a PPO policy for UR5 Y-axis transport in CoppeliaSim.

Usage:
    python rl/train_ppo.py --manage-sim --n-envs 2 --timesteps 50000
    python rl/train_ppo.py --config rl/config.yaml --resume outputs/rl_models/ppo_y_transport

With --manage-sim, this script starts/restarts CoppeliaSim subprocess(es) itself.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

DEFAULT_CONFIG = REPO_ROOT / "rl" / "config.yaml"


def _kill_stale_ports(base_port: int, n_envs: int, port_stride: int) -> None:
    for rank in range(n_envs):
        port = base_port + rank * port_stride
        subprocess.run(
            ["pkill", "-f", f"zmqRemoteApi.rpcPort={port}"],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )


def _build_vec_env(args, cfg: dict, train_cfg: dict):
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

    from rl.env_factory import make_coppelia_env

    n_envs = int(args.n_envs or train_cfg.get("n_envs", 1))
    base_port = int(args.port or cfg["coppeliasim"]["port"])
    port_stride = int(train_cfg.get("port_stride", 2))
    host = args.host or cfg["coppeliasim"]["host"]
    config_path = str(args.config)
    coppelia_root = args.coppelia_root or ""
    manage_sim = bool(args.manage_sim)

    if not coppelia_root and manage_sim:
        import os
        coppelia_root = os.environ.get("COPPELIA_ROOT", "")
    if manage_sim and not coppelia_root:
        raise ValueError("--manage-sim requires --coppelia-root or COPPELIA_ROOT")

    if manage_sim:
        _kill_stale_ports(base_port, n_envs, port_stride)

    env_fns = [
        make_coppelia_env(
            rank=rank,
            config_path=config_path,
            coppelia_root=str(coppelia_root),
            base_port=base_port,
            port_stride=port_stride,
            host=host,
            manage_sim=manage_sim,
        )
        for rank in range(n_envs)
    ]

    if n_envs == 1:
        return DummyVecEnv(env_fns)
    return SubprocVecEnv(env_fns, start_method="forkserver")


def _run_eval(model_path: str, args, cfg: dict, train_cfg: dict) -> None:
    import os

    from stable_baselines3 import PPO

    from rl.coppelia_y_transport_env import CoppeliaYTransportEnv

    eval_episodes = int(train_cfg.get("eval_episodes", 3))
    base_port = int(args.port or cfg["coppeliasim"]["port"])
    coppelia_root = args.coppelia_root or os.environ.get("COPPELIA_ROOT", "")

    print(f"\n=== Post-train eval ({eval_episodes} episodes) ===", flush=True)
    env = CoppeliaYTransportEnv(
        config_path=args.config,
        host=args.host,
        port=base_port,
        coppelia_root=coppelia_root,
        manage_sim=True,
        env_rank=0,
    )
    try:
        model = PPO.load(model_path, env=env)
        summaries = []
        for ep in range(eval_episodes):
            obs, reset_info = env.reset()
            initial_y = float(reset_info["initial_ee_pos"][1])
            total_reward = 0.0
            steps = 0
            terminated = truncated = False
            last_info = reset_info
            while not terminated and not truncated:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, last_info = env.step(action)
                total_reward += float(reward)
                steps += 1
            y_disp = initial_y - float(last_info.get("ee_pos", [0, initial_y, 0])[1])
            summaries.append({
                "episode": ep,
                "steps": steps,
                "total_reward": total_reward,
                "y_displacement_m": y_disp,
                "max_z_drift_m": abs(float(last_info.get("z_err", 0.0))),
                "success": not terminated,
            })
            print(
                f"  ep {ep}: steps={steps} reward={total_reward:.2f} "
                f"y_disp={summaries[-1]['y_displacement_m']:.4f}m "
                f"success={summaries[-1]['success']}",
                flush=True,
            )

        out_dir = REPO_ROOT / "outputs" / "rl_eval"
        out_dir.mkdir(parents=True, exist_ok=True)
        import json

        out_path = out_dir / "eval_summary.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(summaries, f, indent=2)
        print(f"Eval summary: {out_path}", flush=True)
    finally:
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--timesteps", type=int, default=None)
    parser.add_argument("--host", type=str, default=None)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--coppelia-root", type=str, default=None)
    parser.add_argument("--n-envs", type=int, default=None)
    parser.add_argument(
        "--manage-sim",
        action="store_true",
        help="Start/restart CoppeliaSim from Python (recommended)",
    )
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--no-eval", action="store_true")
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    train_cfg = cfg["training"]
    total_timesteps = args.timesteps or int(train_cfg.get("total_timesteps", 50000))
    tb_log = str(REPO_ROOT / train_cfg["tensorboard_log"])
    save_path = str(REPO_ROOT / train_cfg["model_save_path"])
    save_freq = int(train_cfg.get("save_freq", 10000))
    net_arch = train_cfg["net_arch"]
    n_envs = int(args.n_envs or train_cfg.get("n_envs", 1))

    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import (
        BaseCallback,
        CallbackList,
        CheckpointCallback,
    )

    class RolloutLogCallback(BaseCallback):
        def _on_step(self) -> bool:
            return True

        def _on_rollout_end(self) -> bool:
            assert self.logger is not None
            ep_rew = self.logger.name_to_value.get("rollout/ep_rew_mean")
            ep_len = self.logger.name_to_value.get("rollout/ep_len_mean")
            if ep_rew is not None:
                print(
                    f"[train] step={self.num_timesteps} "
                    f"ep_rew_mean={ep_rew:.2f} ep_len_mean={ep_len:.1f}",
                    flush=True,
                )
            return True

    print(
        f"Building vec env: n_envs={n_envs} manage_sim={args.manage_sim}",
        flush=True,
    )
    env = _build_vec_env(args, cfg, train_cfg)

    checkpoint_dir = str(Path(save_path).parent / "checkpoints")
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    Path(tb_log).mkdir(parents=True, exist_ok=True)

    checkpoint_cb = CheckpointCallback(
        save_freq=max(save_freq // max(n_envs, 1), 1),
        save_path=checkpoint_dir,
        name_prefix="ppo_y_transport",
    )
    callbacks = CallbackList([checkpoint_cb, RolloutLogCallback()])

    if args.resume:
        print(f"Resuming from {args.resume}", flush=True)
        model = PPO.load(args.resume, env=env, tensorboard_log=tb_log)
    else:
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=float(train_cfg["learning_rate"]),
            n_steps=int(train_cfg["n_steps"]),
            batch_size=int(train_cfg["batch_size"]),
            n_epochs=int(train_cfg["n_epochs"]),
            gamma=float(train_cfg["gamma"]),
            gae_lambda=float(train_cfg["gae_lambda"]),
            clip_range=float(train_cfg["clip_range"]),
            policy_kwargs=dict(net_arch=net_arch),
            verbose=1,
            tensorboard_log=tb_log,
        )

    print(f"Training PPO for {total_timesteps} timesteps ({n_envs} envs)", flush=True)
    print(f"  TensorBoard: {tb_log}", flush=True)
    print(f"  Checkpoints: {checkpoint_dir}", flush=True)
    print(f"  Final model: {save_path}", flush=True)

    try:
        try:
            import tqdm  # noqa: F401
            import rich  # noqa: F401
            use_progress = True
        except ImportError:
            use_progress = False

        model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            progress_bar=use_progress,
        )
    except KeyboardInterrupt:
        print("\nTraining interrupted by user", flush=True)
    finally:
        model.save(save_path)
        print(f"Model saved to {save_path}", flush=True)
        env.close()

    if not args.no_eval and train_cfg.get("eval_after_train", True):
        _run_eval(save_path, args, cfg, train_cfg)


if __name__ == "__main__":
    main()
