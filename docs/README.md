# Documentation Index

Start here. The authoritative operating guide is the root `AGENTS.md` (symlinked as
`CLAUDE.md`); this directory holds supporting docs.

- `CURRENT_STATUS.md` — active objective, latest diagnosis, what's done, what's next.
- `controller_core/`, `simulation/`, `hardware/`, `ros2/` — subsystem notes (verify against
  code; some predate the 2026-07-03 consolidation).
- `archive/` — historical docs: `archive/AGENTS_HISTORY.md` (the full pre-2026-07-03 AGENTS.md
  operational log), `archive/PROJECT_PLAN_coppeliasim_era.md` (superseded — CoppeliaSim is
  archived and there's no current replacement plan doc; `CURRENT_STATUS.md` + `AGENTS.md`
  serve that role now), `archive/{AUDIT,BLOAT}_REPORT.md` and
  `archive/DIAGNOSTIC_real_cartpole_torque_control_questions.md` (superseded point-in-time
  reports, each banner-marked with what superseded them), plus the pre-torque-lane legacy
  MuJoCo documentation set.

## Layout after the 2026-07-03 consolidation

- **Active lane: MuJoCo true-torque UR5e** — entrypoints in `tools/`, adapter in
  `simulation/ur5e_mujoco_torque.py`, model in `assets/ur5e_torque/`, meshes in
  `vendor/mujoco_menagerie/`.
- Observability: `observability/run_logger.py` — every sweep writes `run_record.json` per run
  and `run_log.jsonl`/`.csv` per sweep.
- Tests: `tests/{unit,mujoco,hardware}` with markers (`pytest -m unit`, etc.).
- Archived lanes: `archive/coppelia/` (entire CoppeliaSim stack, incl. its docs at
  `archive/coppelia/docs/coppeliasim/`), `archive/legacy_mujoco/`, `archive/superseded/`.
- Hardware staging lane: `hardware/` + `tools/ur5e_*` operator scripts — guardrails documented
  in AGENTS.md §4; do not weaken.
