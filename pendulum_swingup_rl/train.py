#!/usr/bin/env python3
"""PPO over the swing-up energy sequence, judged against the analytic baseline.

WHAT "SUCCESS" MEANS HERE, DECIDED BEFORE TRAINING. This repo has six documented
RL failures and a standing rule that a result needing guards off is a negative
result, so the bar is set up front rather than read off a reward curve:

  BASELINE (measured, guards ON, wrist_2=-90, realrod, friction_ff config):
    the analytic sign-rule oracle at CONSTANT amplitude 0.35 reaches
    E_peak/E_top = 0.688 over a 10 s episode, positive_work_fraction = 0.940,
    no guard trip, and never gets within 0.45 rad of inverted (no capture).

  A policy is INTERESTING only if it beats 0.688 guard-clean.
  A policy is USEFUL only if it reaches the capture band (|s| <= 1.2 with
    |phi_inv| < 0.45), which the constant-amplitude oracle never does.

``positive_work_fraction`` is printed for every evaluation because it is the one
number that separates "pumping backwards" from "has not learned yet" -- both of
which look like a flat reward curve. A policy below ~0.5 is removing energy on
balance no matter what the return says.

THREADING. Per AGENTS.md 8, BLAS threads are pinned to 1 before torch/numpy are
imported: with them unset each SubprocVecEnv worker auto-detects the full core
count and spawns that many BLAS threads itself, and n_workers x n_cores threads
blows through the per-user process cap on a shared machine.
"""

from __future__ import annotations

import os

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from pendulum_swingup_rl.env import PendulumSwingupEnv  # noqa: E402

GOAL1_Q = (-2.3688, -2.1801, -1.8838, -0.7962, -1.5707963, 0.0206)
DEFAULT_ASSET = "assets/ur5e_pendulum/pendulum_attachment_realrod.xml"
DEFAULT_CONFIG = "config/ur5e_mujoco_torque_osc_tuned_friction_ff.yaml"

# Measured this session; see module docstring.
BASELINE_E_PEAK_OVER_E_TOP = 0.688


def make_env_fn(args, seed: int):
    def _f():
        env = PendulumSwingupEnv(
            pendulum_xml=args.pendulum_xml,
            arm_q=list(args.start_q_rad),
            config_path=args.config,
            controller_kind=args.controller_kind,
            transport_axis_index=args.transport_axis_index,
            a_max_mps2=args.a_max,
            episode_s=args.episode_s,
            decimation=args.decimation,
        )
        env.reset(seed=seed)
        return env
    return _f


def evaluate(model, args, n_episodes: int = 3) -> dict:
    env = make_env_fn(args, seed=12345)()
    rows = []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=12345 + ep)
        total = 0.0
        info = {}
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, r, term, trunc, info = env.step(action)
            total += float(r)
            if term or trunc:
                break
        rows.append({
            "return": total,
            "e_peak_over_e_top": info.get("e_peak_over_e_top"),
            "positive_work_fraction": info.get("positive_work_fraction"),
            "captured": info.get("captured"),
            "best_abs_s": info.get("best_abs_s"),
            "guard_fired": info.get("guard_fired"),
            "guard_reason": info.get("guard_reason"),
        })
    e_peaks = [r["e_peak_over_e_top"] for r in rows if r["e_peak_over_e_top"] is not None]
    return {
        "episodes": rows,
        "mean_return": float(np.mean([r["return"] for r in rows])),
        "mean_e_peak_over_e_top": float(np.mean(e_peaks)) if e_peaks else None,
        "mean_positive_work_fraction": float(np.mean(
            [r["positive_work_fraction"] for r in rows])),
        "any_captured": any(bool(r["captured"]) for r in rows),
        "any_guard_fired": any(bool(r["guard_fired"]) for r in rows),
        "beats_analytic_baseline": (
            bool(np.mean(e_peaks) > BASELINE_E_PEAK_OVER_E_TOP) if e_peaks else False),
        "baseline_e_peak_over_e_top": BASELINE_E_PEAK_OVER_E_TOP,
    }


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pendulum-xml", default=DEFAULT_ASSET)
    p.add_argument("--start-q-rad", type=float, nargs=6, default=list(GOAL1_Q))
    p.add_argument("--config", default=DEFAULT_CONFIG)
    p.add_argument("--controller-kind", default="impedance")
    p.add_argument("--transport-axis-index", type=int, default=0)
    p.add_argument("--a-max", type=float, default=12.0)
    p.add_argument("--episode-s", type=float, default=10.0)
    p.add_argument("--decimation", type=int, default=25)
    p.add_argument("--total-timesteps", type=int, default=300_000)
    p.add_argument("--n-envs", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--eval-every", type=int, default=50_000)
    p.add_argument("--output-dir", type=Path, default=None)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import SubprocVecEnv

    venv = SubprocVecEnv([make_env_fn(args, seed=args.seed + i) for i in range(args.n_envs)])
    model = PPO("MlpPolicy", venv, seed=args.seed, verbose=1, device="cpu",
                n_steps=256, batch_size=256, gae_lambda=0.95, gamma=0.99,
                ent_coef=0.003, learning_rate=3e-4)

    out_dir = args.output_dir
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)

    done = 0
    history = []
    while done < args.total_timesteps:
        chunk = min(args.eval_every, args.total_timesteps - done)
        model.learn(total_timesteps=chunk, reset_num_timesteps=False)
        done += chunk
        ev = evaluate(model, args)
        ev["timesteps"] = done
        history.append(ev)
        print(f"\n[eval @ {done} steps] mean_return={ev['mean_return']:.2f}  "
              f"E_peak/E_top={ev['mean_e_peak_over_e_top']}  "
              f"pos_work={ev['mean_positive_work_fraction']:.3f}  "
              f"captured={ev['any_captured']}  guard={ev['any_guard_fired']}  "
              f"beats_baseline({BASELINE_E_PEAK_OVER_E_TOP})={ev['beats_analytic_baseline']}",
              flush=True)
        if out_dir is not None:
            model.save(out_dir / "ppo_swingup")
            (out_dir / "eval_history.json").write_text(json.dumps(history, indent=2))
    venv.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
