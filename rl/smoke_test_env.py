#!/usr/bin/env python3
"""Quick smoke test: connect, reset, take 5 random steps, close."""
import sys
import os
from pathlib import Path

ROOT = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, ROOT)

pydeps = os.environ.get("COPPELIA_PYDEPS", "")
if pydeps and pydeps not in sys.path:
    sys.path.insert(0, pydeps)

print("[smoke] importing env...")
from rl.coppelia_y_transport_env import CoppeliaYTransportEnv
import numpy as np

print("[smoke] creating env...")
env = CoppeliaYTransportEnv(
    coppelia_root=os.environ.get("COPPELIA_ROOT", None),
)
print("[smoke] env created. resetting...")
obs, info = env.reset()
print(f"[smoke] reset OK. obs shape={obs.shape}, initial_ee={info['initial_ee_pos']}")

for i in range(5):
    action = env.action_space.sample() * 0.1
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"  step {i+1}: reward={reward:.3f} z_err={info['z_err']:.4f} y_err={info['y_err']:.4f} done={terminated or truncated}")
    if terminated or truncated:
        break

print("[smoke] resetting again (episode 2)...")
obs, info = env.reset()
print(f"[smoke] reset 2 OK. obs shape={obs.shape}")

env.close()
print("Smoke test passed!")
