# Archived CoppeliaSim Lane

Archived 2026-07-03 (branch `feature/ur5e-mujoco-torque-control`). The MuJoCo true-torque lane
is the active development target; this entire CoppeliaSim stack is historical reference.

**Nothing in here is runnable in place.** Scripts reference their old repo-relative paths
(`simulation/...`, `tools/...`, `rl/...`) and would need path fixes plus the dependencies below
to resurrect.

Contents:
- `simulation/` — the CoppeliaSim orchestrator (`run_coppeliasim_x_axis_headless.py`), Lua
  add-ons, launch scripts, ZMQ probe infra, WSL/display bring-up scripts.
- `tests/` — CoppeliaSim adapter/transport/diagnostics pytest modules.
- `tools/` — CoppeliaSim gravity probes and stale WSL-path launchers.
- `rl/` — PPO Y-transport stack (stable_baselines3/gymnasium, CoppeliaSim-backed).
- `ros2/ur5_x_axis_controller_ros/` — the CoppeliaSim controller node, bridge node, adapter,
  Jacobian provider, launch file, and `config/` (the `controller_coppelia_*.yaml` family).
- `docs/coppeliasim/` — all lane docs (pipelines, vision notes, RPC system/TODO, diagnostics).
- `config/` — `coppeliasim_movement_primitives.yaml`.

Dependencies needed to resurrect (removed from `environment.yml` at archive time):
`coppeliasim-zmqremoteapi-client`, `stable_baselines3`, `gymnasium`. The vendored simulator
runtime remains on disk at `third_party/coppelia_runtime/` (gitignored).

The full `mujoco_menagerie/` checkout (1.1GB, untracked nested git repo) was deleted at cleanup
time; the active lane uses only `vendor/mujoco_menagerie/` (tracked). To restore the full zoo:
`git clone https://github.com/google-deepmind/mujoco_menagerie && git -C mujoco_menagerie
checkout 959cabcdfb464cee47e0fbda807371f8d93a4f4c`.

The historical AGENTS.md operational lore for this lane (verified commands, RPC guardrails,
Xvfb caveats, per-date findings) was relocated to `docs/archive/AGENTS_HISTORY.md` during the
AGENTS.md rewrite.
