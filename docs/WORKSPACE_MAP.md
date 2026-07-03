# Workspace Map

Quick orientation map for `/common/users/ss5772/real_Cartpole` (post-2026-07-03 layout).

## Reality check

- The root folder **is** a git repository (branch `feature/ur5e-mujoco-torque-control`).
  `outputs/`, `reports/`, `third_party/` are gitignored.
- The folder name is historical: this is a UR5e torque-control workspace. The active lane is
  **MuJoCo true-torque simulation**; CoppeliaSim is archived.
- Root `AGENTS.md` (= `CLAUDE.md`) is the agent playbook and stays outside `docs/`.

## Active areas

| Path | Purpose |
|---|---|
| `controller_core/` | Simulator-independent controller library (impedance law, QP torque law, safety monitor, kinematics, logging utils). Numpy only. |
| `simulation/` | `ur5e_mujoco_torque.py` (MuJoCo adapter) and `workspace_guardrails.py` (viz-only lab guardrails). |
| `tools/` | Active MuJoCo entrypoints (rollout engine, gravity audit, move-hold sweep, envelope sweep, residual tuner) + hardware operator scripts; `tools/diagnostics/` for secondary analysis. |
| `observability/` | `RunLogger` — unified per-run/sweep JSON records. |
| `assets/ur5e_torque/` | The custom torque-actuated UR5e MJCF (the centerpiece model). |
| `vendor/mujoco_menagerie/` | Tracked UR5e/UR10e mesh + scene assets (the only menagerie copy on disk). |
| `config/` | `ur5e_mujoco_torque*.yaml` (lane configs), `lab_workspace_guardrails.yaml`. |
| `tests/` | `unit/` (pure numpy), `mujoco/`, `hardware/` — markers auto-applied; root `pytest.ini`. |
| `hardware/` | Real-UR5e RTDE staging lane, guardrails per AGENTS.md §4. |
| `ros2_ws/` | Hardware pipeline node + description/moveit packages (CoppeliaSim nodes archived). |
| `outputs/` | Run artifacts (gitignored). Sweep dirs carry `run_log.jsonl`/`run_log.csv`. |
| `docs/` | This documentation tree. |

## Archived areas

| Path | Contents |
|---|---|
| `archive/coppelia/` | Entire CoppeliaSim lane: orchestrator + Lua + launchers + ZMQ probes + WSL bring-up, RL PPO stack, ROS2 controller/bridge nodes, configs, tests, docs. |
| `archive/legacy_mujoco/` | Pre-torque-lane MuJoCo cartpole diagnostics. |
| `archive/superseded/` | Replaced drivers (old impedance tuner). |
| `third_party/` | Vendored CoppeliaSim runtime + pydeps (gitignored, kept on disk). |
