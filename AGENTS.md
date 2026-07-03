# AGENTS.md

Working playbook for agents operating in `/common/users/ss5772/real_Cartpole`.
Update rule: edit in place, keep these sections, date-stamp material changes. Do not append
chronological logs here — that pattern was retired 2026-07-03; the old log is preserved at
`docs/archive/AGENTS_HISTORY.md`.

## 1. Project reality

- The folder name is historical: this is a **UR5e torque-control workspace**, not a cartpole
  project.
- **The active lane is MuJoCo true-torque simulation** using the custom torque-actuated UR5e
  model. Everything CoppeliaSim is archived (see §6).
- The repo root **is a git repo** (branch `feature/ur5e-mujoco-torque-control`). `outputs/`,
  `reports/`, `third_party/` are gitignored — changes there are not git-recoverable.
- Robot model assets live in `vendor/mujoco_menagerie/` (tracked). The full menagerie zoo
  checkout was deleted 2026-07-03; to restore:
  `git clone https://github.com/google-deepmind/mujoco_menagerie && git -C mujoco_menagerie checkout 959cabcdfb464cee47e0fbda807371f8d93a4f4c`.
- Python env: conda `mujoco_ur5e` (py3.12) per `environment.yml`. The cluster launchers may
  hardcode `/common/users/ss5772/miniforge3/bin/python3`.

## 2. Active lane — MuJoCo true-torque UR5e

- Model: `assets/ur5e_torque/scene.xml` (includes `assets/ur5e_torque/ur5e_torque.xml`,
  meshes from `vendor/mujoco_menagerie/universal_robots_ur5e/assets`). Real per-body
  inertials; direct torque actuators. **This custom model is the centerpiece — never delete
  or silently regenerate it.**
- Configs: `config/ur5e_mujoco_torque.yaml` (base), `config/ur5e_mujoco_torque_transport.yaml`
  (transport). Gains live under `controller: gains:` and are parsed by
  `CartesianImpedanceConfig.from_controller_yaml_section`; the canonical gain field list is
  `transport_metrics.GAIN_FIELDS`.
- Entrypoints (all support `--help`):
  - `tools/ur5e_mujoco_torque_experiments.py` — single-run rollout engine (the only file with
    the per-step loop; other drivers subprocess it).
  - `tools/audit_ur5e_mujoco_gravity_torque.py` — gravity-sign / hold-quality audit.
  - `tools/ur5e_move_hold_transport.py` — move+hold sweep driver.
  - `tools/ur5e_x_frame_envelope.py` — X-frame transport envelope sweep.
  - `tools/tune_ur5e_residual_impedance_transport.py` (+ `tools/tuning_common.py`) — the
    active gain-tuning driver. Its predecessor is in `archive/superseded/`.
  - `tools/compare_ur5e_mujoco_controllers.py` — controller-family comparison.
- Verified short commands:
  - `python tools/audit_ur5e_mujoco_gravity_torque.py --poses active_origin --durations 1.0 2.0 --seed 0 --no-plot`
  - `python tools/ur5e_move_hold_transport.py --target-x-deltas 0.01 0.02 --move-durations 1.0 --hold-durations 1.0 2.0 --torque-limit-scales 1.0 --seed 0 --no-plot`
- Secondary analysis lives in `tools/diagnostics/` (guardrail trajectory check/overlay,
  torque-QP smoke). The lab workspace-guardrail workflow
  (`config/lab_workspace_guardrails.yaml`, `simulation/workspace_guardrails.py`) is
  simulation/visualization only — never wire it into real-arm e-stop logic.

### Observability (required for new experiments)

Every sweep entrypoint writes, via `observability/run_logger.py` (`RunLogger`):
- per-run `run_record.json` next to each run's `summary.json`/`trace.jsonl`;
- sweep-level `run_log.jsonl` (crash-safe incremental) + `run_log.csv` (flattened,
  per-joint dicts become `<field>__<joint>` columns).
Records carry `backend`, drift/orientation/velocity metrics with time-to-limit, per-joint
commanded-vs-clipped torque and clip counts, which safety guard fired first and when,
`gravity_hold_status`, `phase_at_failure`, `outcome`, `failure_category`. New experiment
drivers must log through `RunLogger` instead of inventing new summary schemas. Do not trust
an MP4 or a bare exit code as success evidence — read the run record.

## 3. Controller architecture

- `controller_core/` is simulator-independent (numpy only — keep it that way).
  - Law: `controller_core/x_axis_cartesian_impedance.py` — task-space PD wrench →
    `J.T @ wrench` + joint damping + posture PD + externally supplied `gravity_torque`,
    singular-value wrench scaling, geometric torque backtracking, hard clip.
    Joint order: `JOINT_NAME_ORDER` (shoulder_pan → … → wrist_3).
  - State contract: `controller_core/state_types.py` (`as_impedance_robot_state`).
  - Safety: `controller_core/safety.py` `ImpedanceSafetyMonitor` — Y/Z/orthogonal drift,
    orientation error, `|qd| > 1.5 rad/s`, axis-error growth, NaN/joint-limit e-stop. This is
    the source of `termination_reason` strings in traces.
  - Alternate torque law: `controller_core/torque_task_qp.py` (+ `box_qp.py`).
- MuJoCo adapter: `simulation/ur5e_mujoco_torque.py` — steps the sim, currently adds
  `tau_applied = tau_controller + tau_gravity` with `tau_gravity` = MuJoCo `qfrc_bias`.
- **Model-based dynamics (landed 2026-07-03, all flag-gated, defaults = legacy behavior)**:
  - `controller_core/model_dynamics.py` — `DynamicsProvider` + `PinocchioUR5eDynamics`
    (loads the active MJCF; parity vs MuJoCo <1e-8 Nm gravity, <1e-6 bias, <1e-8 mass matrix).
  - Gravity source: `mujoco.gravity_source: pinocchio` / `--gravity-source` (P1).
  - Coriolis feedforward: `mujoco.coriolis_feedforward: true` / `--coriolis-feedforward` (P2 —
    historical lane never compensated C(q,qd)qd; measured negligible below ~0.5 rad/s).
  - Operational-space (P3): `controller.task_space_inertia_shaping` (Λ(q) wrench weighting;
    gains become task-acceleration gains) + `controller.nullspace_posture` (dynamically
    consistent projection; note a full-rank 6D task has no nullspace, so this zeroes posture
    except near singularities). Needs `mass_matrix` in the state dict (`build_mujoco_state`
    supplies it). Named config: `config/ur5e_mujoco_torque_osc.yaml`.
  - P3 evidence (8-point move-hold grid, untuned gains): OSC 6/8 vs baseline 5/8; the
    dx=0.02/hold=2 orientation failure fixed (0.350→0.031 rad); dx≥0.03/hold=2 still fails —
    gain retuning for the acceleration-gain semantics is the known next step.
  - **Tuned OSC gains (landed 2026-07-03, ~250 evaluation runs)**:
    `config/ur5e_mujoco_torque_osc_tuned.yaml` — kp_x 400/kd_x 40, kp_rot 0/kd_rot 10,
    kp_posture 25/kd_posture 6, kd_joint 4, lambda_regularization 0.1,
    `posture_reanchor_on_settle: true`. Validated envelope, 0 guard trips throughout:
    canonical grid (dx 0.01–0.04 m × hold 1/2 s) 8/8, worst orientation 0.040 rad; long
    holds (dx 0.03/0.06 m, hold up to 30 s) 8/8, worst 0.067 rad; large displacements
    (dx up to 0.20 m) 16/16, worst 0.205 rad (dx=0.25 m breaks via Z-drift — a genuine
    workspace/reach limit, not a controller defect); torque-scale robustness down to 10%
    14/14. Untuned OSC on the canonical grid: 1/8 valid, 2 guard trips, |qd| 1.7 rad/s.
    Two root causes found and fixed: (1) the transport start pose sits at the UR wrist
    singularity (wrist_2=0) — orientation drifts along a task-unactuatable direction, held
    only by the joint-space posture anchor + kd_joint damping (the nullspace projector
    passes posture exactly there); (2) the task rotation PD is *unstable* at that pose —
    positive feedback through the eps-regularized Λ regardless of kp_rot magnitude, only
    slower as it shrinks (kp_rot=30 trips the guard at ~3.5 s, 5 at ~27 s, 0 never — clean
    to 30 s). Fix: kp_rot=0 (damping-only), posture re-anchoring holds orientation instead.
    Known, out-of-scope boundary: moves faster than ~0.5 s undershoot (closed-loop
    bandwidth limit at kp_x=400, not saturation — irrelevant for 1 s+ transport moves).
  - Posture re-anchoring: `controller.posture_reanchor_on_settle` (+`reanchor_x_tol_m`,
    `reanchor_qd_tol_radps`) — one-shot q_rest re-capture at settle, flag-gated, default off.

## 4. Safety & guardrails (hardware — do not weaken)

`hardware/` is the real-UR5e RTDE staging lane. All four guardrails are enforced in code and
must stay intact:
1. **Receive-only default** — `connect_receive_only()` is the default;
   `motion_opt_in: bool = False` at the dataclass level.
2. **servoJ requires explicit opt-in** — CLI flag `--i-understand-this-moves-the-robot`;
   every command still passes `MotionCommandGuard` jump/velocity/acceleration checks.
3. **Direct torque zero-only by default** — `direct_torque_zero_only: bool = True`,
   `zero_only=True` at the bridge layer.
4. **Nonzero direct torque blocked** — requires three simultaneous flags plus hard caps, and
   the probe currently cannot send nonzero torque at all regardless of flags.
The ROS2 hardware node (`ros2_ws/src/ur5_x_axis_controller_ros/.../ur5e_hardware_pipeline_node.py`)
mirrors these defaults. The e-stop latch has no un-latch path by design (restart required).

Do-not-recreate (gravity/dynamics bugs, still relevant):
- Do not tune gravity scale from single-joint probes; always test all 6 joints.
- Do not add gravity compensation twice (the QP controller adds it internally; adapters add
  it only in IK-PD/warmup paths).
- Do not conflate solver-side feasibility with a passing live summary — runtime safety and
  drift checks must pass.
- Do not let one simulator's dynamics model silently feed another's control loop (the
  archived CoppeliaSim lane defaulted `gravity_compensation_source="mujoco"` — that
  cross-lane coupling is the canonical example).

## 5. Testing

- Root `pytest.ini`; suite layout: `tests/unit/` (pure numpy controller_core),
  `tests/mujoco/` (needs mujoco), `tests/hardware/` (mocked RTDE). Markers auto-applied by
  directory: `pytest -m unit`, `-m mujoco`, `-m hardware`, `-m "not slow"`.
- Full suite: `python -m pytest -q` (94 passing as of 2026-07-03).
- Before long training/sweeps, run the tiny smoke first (`tests/mujoco/test_ur5e_mujoco_torque.py`
  covers model-load and a tiny move-hold subprocess run).

## 6. Archived lanes

- `archive/coppelia/` — the entire CoppeliaSim stack (orchestrator, Lua add-ons, launchers,
  ZMQ probes, WSL bring-up, RL PPO stack, ROS2 controller/bridge nodes, docs, configs, tests).
  Not runnable in place; resurrect notes + removed deps in `archive/coppelia/README.md`.
  The vendored simulator runtime remains at `third_party/coppelia_runtime/` (gitignored).
- `archive/legacy_mujoco/` — pre-torque-lane MuJoCo cartpole diagnostics, including four
  scripts whose names sound CoppeliaSim-related but are pure MuJoCo.
- `archive/superseded/` — replaced drivers (old impedance tuner).
- Historical operational lore and per-date findings: `docs/archive/AGENTS_HISTORY.md`.

## 7. Working rules for this repo

- Start with the simplest proof (model loads → short run → read run_record.json) before long
  sweeps or controller changes.
- Do not change training/eval logic and controller logic in the same commit; do not combine
  startup fixes with gain tuning.
- Never silently change units, control rate, action scaling, torque limits, or checkpoint
  selection; state control-rate and scaling assumptions when touching controllers.
- Preserve old configs — add new named configs instead of mutating shared ones.
- Never edit without explicit request: checkpoints, datasets, logs, `.git`, generated
  experiment artifacts under `outputs/`, large binaries.
- For every final response: list files changed, tests run, tests not run, and a rollback
  command.
