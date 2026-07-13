# Documentation Index

Start here. The authoritative operating guide is the root `AGENTS.md` (symlinked as
`CLAUDE.md`); this directory holds supporting docs.

- `CURRENT_STATUS.md` — active objective, latest diagnosis, what's done, what's next.
- `controller_core/`, `simulation/`, `ros2/` — subsystem notes (verify against code; some
  predate the 2026-07-03 consolidation).
- `hardware/README.md` — hardware doc index; **`hardware/HARDWARE_GUIDE.md`** is the full
  learning guide (modules, RTDE, safety, CLI, lab checklist).
- `archive/` — historical docs: `archive/AGENTS_HISTORY.md` (the full pre-2026-07-03 AGENTS.md
  operational log), `archive/PROJECT_PLAN_coppeliasim_era.md` (superseded — CoppeliaSim is
  archived and there's no current replacement plan doc; `CURRENT_STATUS.md` + `AGENTS.md`
  serve that role now), `archive/{AUDIT,BLOAT}_REPORT.md` and
  `archive/DIAGNOSTIC_real_cartpole_torque_control_questions.md` (superseded point-in-time
  reports, each banner-marked with what superseded them), `archive/hardware_v1/` (the
  pre-2026-07-07 hardware lane's docs), plus the pre-torque-lane legacy MuJoCo documentation
  set.

## Layout after the 2026-07-03 consolidation

- **Active lane: MuJoCo true-torque UR5e** — entrypoints in `tools/`, adapter in
  `simulation/ur5e_mujoco_torque.py`, model in `assets/ur5e_torque/`, meshes in
  `vendor/mujoco_menagerie/`.
- Observability: `observability/run_logger.py` — every sweep writes `run_record.json` per run
  and `run_log.jsonl`/`.csv` per sweep.
- Tests: `tests/{unit,mujoco,hardware}` with markers (`pytest -m unit`, etc.).
- Archived lanes: `archive/coppelia/` (entire CoppeliaSim stack, incl. its docs at
  `archive/coppelia/docs/coppeliasim/`), `archive/legacy_mujoco/`, `archive/superseded/`.
- **Real-UR5e hardware lane (rewritten 2026-07-07)** — `hardware/{safety,link,motion}.py` +
  `tools/ur5e_{connect,move}.py`; guardrails documented in `AGENTS.md` §4, do not weaken.
  Previous version archived to `archive/superseded/hardware_rtde_v1/`.
- **RL gain-scheduling** (`rl_gain_scheduling/`) — PPO policy that schedules the tuned OSC
  controller's gains live, vs. the fixed-gain baseline. As of 2026-07-07, unresolved: no
  trained checkpoint yet matches or beats the fixed baseline on the full comparative eval
  grid — see `CURRENT_STATUS.md`.
