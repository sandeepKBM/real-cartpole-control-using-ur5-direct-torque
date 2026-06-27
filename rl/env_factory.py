"""Picklable Gymnasium env factory for SubprocVecEnv."""

from __future__ import annotations

from pathlib import Path


def make_coppelia_env(
    rank: int,
    config_path: str,
    coppelia_root: str,
    base_port: int,
    port_stride: int,
    host: str,
    manage_sim: bool,
):
    """Return a thunk SB3 SubprocVecEnv can pickle."""

    def _init():
        from rl.coppelia_y_transport_env import CoppeliaYTransportEnv

        port = int(base_port) + int(rank) * int(port_stride)
        return CoppeliaYTransportEnv(
            config_path=config_path,
            host=host,
            port=port,
            coppelia_root=coppelia_root,
            manage_sim=manage_sim,
            env_rank=rank,
        )

    return _init
