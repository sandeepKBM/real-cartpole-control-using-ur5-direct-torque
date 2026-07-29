# AGENTS.md

Working playbook for agents operating in `/common/users/ss5772/real_Cartpole`.
Update rule: edit in place, keep these sections, date-stamp material changes. Do not append
chronological logs here — that pattern was retired 2026-07-03; the old log is preserved at
`docs/archive/AGENTS_HISTORY.md`. Material hardware refresh: 2026-07-14.

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
  torque-QP smoke, `render_trace_video.py` — kinematic replay of a `trace.jsonl` to MP4 via
  `mujoco.Renderer`; needs `MUJOCO_GL=egl` on this headless host, camera defaults tuned for
  the active-origin transport pose). The lab workspace-guardrail workflow
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
  - **Two more OSC leaks found and fixed (2026-07-26), both flag-gated, neither moved the
    ~0.25–0.3 m ceiling**: `controller.lambda_diagonal_shaping` — away from the wrist_2=0
    singularity, Λ=(J M⁻¹ Jᵀ+εI)⁻¹ develops large off-diagonal terms (e.g. Λ_xz 0.32→2.0 by
    wrist_2=-0.006 rad), so the shaped wrench's Z-row picks up `Λ_xz·Fx` even with zero Z
    error; diagonalizing Λ for the wrench-shaping step only (nullspace projector keeps the
    full Λ) kills that leak. `controller.lambda_adaptive_regularization` — the SAME static
    ε=0.1 that tames Λ at the exact singularity (needed: dropping it there makes Λ's
    diagonal blow up, e.g. Λ[3,3] 9.8→670 as ε drops 0.1→0.001) also corrupts the
    nullspace-posture projector away from it (measured: at cond(J)=719, ε=0.1 leaks a real
    0.074 rad/s² task acceleration from a representative posture torque instead of nulling
    it, as theory predicts and ε=0 measures). Fix: schedule ε in log(cond(J)) space between
    a far-field value and the existing near-singularity ceiling — but ONLY for the
    nullspace projector; scheduling the wrench-shaping Λ too caused a real regression
    (previously-trivial cases failing on `|qd|>3.0`) since reducing ε also amplifies
    wrench-shaping in ways the tuned gains were never validated against. Both leaks were
    real and are now fixed (`config/ur5e_mujoco_torque_osc_tuned_adaptive_lambda.yaml`,
    zero regressions across the canonical grid), but the ceiling didn't move — good
    evidence it's structural (see next finding), not a fixable regularization defect.
  - **The ceiling is directional, not just a magnitude limit (2026-07-27/28)**: at
    height_alpha=0.5 (`hardware/poses.py::HEIGHT_ALPHA_0_5_Q`), `+0.20 m` passes cleanly
    (worst orientation 0.214 rad) but `-0.20 m` fails via orientation error at *half* the
    peak wrist_2 excursion (-0.017 rad vs +0.029 rad for the passing direction) — ruled out
    as a speed/duration or reanchor-timing artifact (identical failure at move durations
    1–5 s and with `posture_reanchor_on_settle` disabled; see the git history around this
    date for the full elimination trail). Root cause: since kp_rot=0, orientation is held
    *only* by the nullspace-projected posture term, and that projector's Frobenius norm is
    itself asymmetric with wrist_2 sign at this pose — it grows during the `+0.20 m` move
    (1.74→1.87) but shrinks monotonically during `-0.20 m` (1.74→1.59). Same kp_posture/
    kd_posture, genuinely less restoring authority available in the `-X` direction — this
    is why a `kp_posture`/`kd_posture`/`kd_joint` gain sweep at this exact case (2026-07-27)
    barely moved the outcome (quality 0.306→0.305, and `kd_joint` up made it *worse*).
    `Λ_xz` (wrench-shaping X→Z coupling) shows a related signature — it grows positive for
    `+0.20 m` but crosses zero and goes negative for `-0.20 m` — though `lambda_diagonal_shaping`
    is active in this config and already removes that specific leak from the wrench, so it
    isn't the primary driver; the nullspace-projector asymmetry is. Fixing this for real needs
    a different orientation-holding mechanism (not just retuned gains — the sweep is
    real evidence against that), which is a controller-design question, not a retune.
    Practical floor until then: **the safe symmetric range at height_alpha=0.5 is ±0.15 m**,
    not ±0.20 m.

## 4. Safety & guardrails (hardware — do not weaken)

`hardware/` is the real-UR5e RTDE lane (rewritten 2026-07-07; older lane in
`archive/superseded/hardware_rtde_v1/`). **Learning map:** `docs/hardware/README.md`.

Three control modes via `hardware/x_transport.py` (`--control-mode`):
1. **`position`** (default) — `servoL` + optional shadow OSC (`position_transport.py`);
   use on URSim / real arm to test trajectory, safety, logging without live torques.
2. **`direct_torque`** — Python OSC @ 500 Hz → `directTorque()` (`direct_torque_transport.py`);
   URSim validates the API only (no torque physics); real arm for motion.
3. **`urscript`** — OSC on PolyScope (`urscript_transport.py`); minimum-latency path.

Core modules:
- `hardware/safety.py` — `UR5eSafetyLimits`, `ConnectionHealth`, one-way `EStopLatch`,
  `CartesianMoveMonitor` (TCP drift / orientation / growth abort).
- `hardware/link.py` — `UR5eLink`: receive + optional `servoL`/`moveJ`. `read_state()`
  **raises** `RTDEStateError` (never returns stale cache — fix for the old ROS2 bug).
- `hardware/motion.py` — bounded Cartesian `servoL` move (`tools/ur5e_move.py`).
- `hardware/direct_torque_link.py` — `UR5eDirectTorqueLink` + RTDE/local J+M.
- Never add gravity torque in Python when using `directTorque()` — PolyScope adds it.

CLIs: `tools/ur5e_connect.py` (receive-only; cannot move), `tools/ur5e_move.py`,
`tools/ur5e_direct_torque_x_transport.py` (main `--control-mode` entry),
`tools/ur5e_direct_torque_height_latency_test.py`, `tools/ur5e_urscript_x_transport.py`.

Guardrails enforced in code:
1. **Receive-only default** — `UR5eLink.connect(with_control=False)` never opens control
   unless asked.
2. **Motion requires explicit opt-in** — `--i-understand-this-moves-the-robot`, checked
   before any move path runs.
3. **E-stop latch is one-way** — no `reset()`/`clear()`; tripped ⇒ new process.
4. **No reconnect mid-motion** — state-read failure aborts; reconnect only in
   `ur5e_connect.py --watch` idle loop.
5. **All three modes share `CartesianMoveMonitor` (TCP speed/accel/waypoint-jump), not just
   `position`** — fixed 2026-07-25 (see below); previously `direct_torque`/`urscript` only
   had `ImpedanceSafetyMonitor` (drift/orientation/joint-velocity/axis-growth, no Cartesian
   kinematic ceiling), i.e. the two modes capable of a torque runaway had the loosest guards.
6. **Robot-reported safety status is checked every cycle in all four loops**
   (`motion.py`, `position_transport.py`, `direct_torque_transport.py`,
   `urscript_transport.py`) via `hardware.safety.is_robot_safety_normal()` — fixed
   2026-07-25; the telemetry was already being read (`getSafetyStatusBits()`/
   `getSafetyStatus()` → `UR5eState.safety_status`) but never inspected anywhere.

**2026-07-25 audit + fixes**: full detailed writeup moved to
`docs/archive/AGENTS_HISTORY.md` (per this file's own no-chronological-logs rule). Summary of
what landed: `CartesianMoveMonitor` layered onto `direct_torque`/`urscript` (item 5 above);
robot safety-status bit checked every cycle in all four loops (item 6 above);
`urscript_transport.py`'s stop-register and NaN/Inf handling fixed; `ur5e_move.py`'s
self-referential speed guard replaced with a fixed ceiling.

**Found, not yet fixed — flagged for a deliberate decision, not silently patched:**
- **URScript (Mode 3) still omits singular-value wrench scaling; no real-hardware validation
  exists for any URScript path yet (corrected 2026-07-29).** Nullspace posture projection and
  geometric backtracking (previously listed here as missing) were fixed in commit `b24cdf4`
  (2026-07-26) and are now numerically parity-tested against `x_axis_cartesian_impedance.py`
  in `tests/hardware/test_urscript_parity.py` (415 lines) — see
  `test_nullspace_projection_matches_python` (`atol=1e-8`) and
  `test_backtracking_matches_python_under_saturation` (`atol=1e-6` under saturation); this file
  is distinct from, and newer than, `tests/hardware/test_urscript_gen.py`, which this paragraph
  previously (incorrectly) cited as having no such coverage. What's still genuinely open: the
  on-robot script still omits `cond(J)`-based singular-value wrench scaling —
  `test_gap_singular_scaling` in the same file currently asserts that gap exists (Python nulls
  task torque near the singularity; URScript doesn't), not that it's closed. The ~250-run OSC
  validation campaign, and this parity suite, are both Python-vs-Python only — no real-hardware
  validation exists yet for any of the three URScript control-law behaviors above, fixed or not.
- **`controller_core/x_axis_cartesian_impedance.py`'s global `cond(J)`-based `singular_scale`
  nulls task authority at the transport start pose.** Measured: freezes the controller for
  ~0.2s at the start of every move (`tau≈1e-11 Nm`), escaping only via numerical-noise
  perturbation of `wrist_2` off exactly zero — fragile and non-physical. It's redundant with,
  and defeats, the `lambda_regularization` already in the tuned config, which alone already
  produces a healthy X force at the singularity. Disabling it moved the speed ceiling for a
  0.15m move from ~0.4s to ~0.25s with *lower* peak velocity/torque, no regression on
  long-hold/large-displacement spot checks — but needs a full validation sweep before
  trusting, and is a controller-math change, not a hardware-lane fix.
- **RL gain-scheduling's never-move collapse has a credible root cause**, see
  `docs/CURRENT_STATUS.md` — not a hardware item, kept here only as a pointer.

**Corrected 2026-07-28** (previously listed above as "found, not yet fixed" — both were
already closed by commit `85498a0`, 2026-07-25, before that bullet list was written; this
file was never updated after that commit landed):
- `max_deadline_ms` (`UR5eSafetyLimits`) **is enforced** — `DeadlineMonitor`
  (`hardware/safety.py`) is instantiated and checked every cycle in all four motion loops.
  Real caveat found 2026-07-28 during live hardware testing: its flat 3.0ms threshold is
  calibrated for the 125Hz loops (8ms period), not `direct_torque`'s 500Hz loop (2ms
  period) — a real overrun there (up to ~2ms over) can sit under that floor by design. Not
  fixed yet; full analysis and concrete fix proposals in
  `docs/status/timing_safety_gaps_audit_2026-07-28.md`.
- **Cycle-to-cycle staleness detection during motion exists** — `StaleStateMonitor`
  (`hardware/safety.py`) is checked every cycle in all four motion loops, comparing
  `robot_timestamp_s` against the host clock; trips after 5 consecutive frozen-vs-advancing
  reads. Full trace of this mechanism in the same doc above.
- Real 2026-07-28 hardware findings (wrist-singularity divergence in `position` mode; the
  `CartesianMoveMonitor` accel estimate's own noise floor being far above its old default
  threshold, now fixed via `accel_gap_cycles`/`speed_lowpass_alpha`; two real RTDE read
  stalls, most likely the documented UR behavior of the robot controller deprioritizing
  telemetry under its own load, not a bug in this codebase): see
  `hardware_captures/2026-07-28_thinkrobot_172.16.71.77/README.md` and
  `docs/status/clock_timing_late_cycles_2026-07-28.md`.

Do-not-recreate (gravity/dynamics bugs, still relevant):
- Do not tune gravity scale from single-joint probes; always test all 6 joints.
- Do not add gravity compensation twice (the QP controller adds it internally; adapters add
  it only in IK-PD/warmup paths).
- Do not conflate solver-side feasibility with a passing live summary — runtime safety and
  drift checks must pass.
- Do not let one simulator's dynamics model silently feed another's control loop (the
  archived CoppeliaSim lane defaulted `gravity_compensation_source="mujoco"` — that
  cross-lane coupling is the canonical example).
- Do not `abs()` a signed target before comparing it to a signed achieved value in
  `transport_metrics.py` (fixed 2026-07-03 in `compute_valid_move_hold_metrics`:
  `abs(achieved - abs(target))` silently failed every negative-direction transport run).
  `abs()` is only correct where the result is used purely as a tolerance *magnitude*
  (e.g. `_move_hold_tolerances`'s own local copy), never in a signed subtraction.

## 5. Testing

- Root `pytest.ini`; suite layout: `tests/unit/` (pure numpy controller_core),
  `tests/mujoco/` (needs mujoco), `tests/hardware/` (mocked RTDE). Markers auto-applied by
  directory: `pytest -m unit`, `-m mujoco`, `-m hardware`, `-m "not slow"`.
- Full suite: `python -m pytest -q` (167 passing as of 2026-07-07; this count drifts as tests
  are added — don't treat it as a gate, just a sanity baseline).
- Before long training/sweeps, run the tiny smoke first (`tests/mujoco/test_ur5e_mujoco_torque.py`
  covers model-load and a tiny move-hold subprocess run).

## 6. Archived lanes

- `archive/coppelia/` — the entire CoppeliaSim stack (orchestrator, Lua add-ons, launchers,
  ZMQ probes, WSL bring-up, RL PPO stack, ROS2 controller/bridge nodes, docs, configs, tests).
  Not runnable in place; resurrect notes + removed deps in `archive/coppelia/README.md`.
  The vendored simulator runtime remains at `third_party/coppelia_runtime/` (gitignored).
- `archive/legacy_mujoco/` — pre-torque-lane MuJoCo cartpole diagnostics, including four
  scripts whose names sound CoppeliaSim-related but are pure MuJoCo. Design rationale and
  a per-controller reference for that lane: `docs/archive/CONTROL_DESIGN_NOTEBOOK.md`
  (implementation reference), `docs/archive/SLSQP_CONTROLLER_REFERENCE.md`
  (controller/solver/runner index), `docs/archive/FIRST_PRINCIPLES_CODE_FLOW.md`
  (onboarding walkthrough).
- `archive/superseded/` — replaced drivers (old impedance tuner); `hardware_rtde_v1/` (the
  pre-2026-07-07 real-UR5e RTDE lane: `ur5e_rtde_bridge.py`, `ur5e_control_session.py`,
  `ur5e_stages.py`, `safety_limits.py`, `ros_topics.py`, the five staged `tools/ur5e_*.py`
  CLI scripts, the ROS2 hardware pipeline node + its launch file, and their old tests —
  superseded by the current `hardware/{safety,link,motion}.py` + `tools/ur5e_{connect,move}.py`
  described in §4).
- Historical operational lore and per-date findings: `docs/archive/AGENTS_HISTORY.md`. The
  full pre-2026-07 documentation set (project origin, legacy workspace/singularity studies,
  the original CoppeliaSim-port bring-up plan) also lives under `docs/archive/` — browse it
  for anything not covered by the pointers above.
- Superseded root-level reports, now archived: `docs/archive/AUDIT_REPORT.md` (pre-archival
  bloat/dynamics audit), `docs/archive/BLOAT_REPORT.md` (bloat diagnosis, mostly executed),
  `docs/archive/DIAGNOSTIC_real_cartpole_torque_control_questions.md` (the diagnostic that
  motivated the Pinocchio P0-P3 work in §3 — read that section for the current answer).

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

## 8. Remote compute / cluster usage (added 2026-07-29)

`westeros` is a shared machine — `uptime` can show load ~100 on 72 cores from other users'
jobs with no warning. Before launching a real training/sweep run, check `uptime`/`nproc` first;
don't assume idle capacity.

Rutgers CS `ilab1`-`ilab4.cs.rutgers.edu` are viable overflow capacity (same NFS home, same
conda env, no file copying needed) but are teaching/interactive machines with real gotchas for
unattended background jobs, found the hard way running RL training there:

- **Per-process BLAS thread explosion**: with `OPENBLAS_NUM_THREADS` unset, each parallel worker
  process (e.g. `SubprocVecEnv`) auto-detects the full core count and spawns that many BLAS
  threads *itself* — `n_workers × n_cores` threads blows through the per-user `RLIMIT_NPROC`
  cap fast. Always export `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
  NUMEXPR_NUM_THREADS=1` before launching multi-process CPU workloads; the parallelism should
  come from having N processes, not from each process also being internally multi-threaded.
- **Per-user memory cgroup cap, separate from system RAM**: `free -h` showing hundreds of GB
  free is not the limit — check `cat /sys/fs/cgroup/user.slice/user-$(id -u).slice/memory.max`.
  Exceeding it triggers a silent cgroup OOM-kill: process vanishes with no Python traceback, no
  dmesg access to confirm (unprivileged users usually can't read the kernel ring buffer). If a
  background job dies cleanly mid-run with an empty log tail and no error, suspect this before
  anything else.
- **`nohup`+`disown` is not reliable on these hosts**: `systemd-logind`'s `Linger` setting for
  the account can be `no` (and can get reset back to `no` even after `loginctl enable-linger
  <user>` succeeds — cause not confirmed, possibly a periodic account-sync job). With no
  lingering, all background processes get killed the moment the last SSH session to that host
  closes, which happens naturally between one-shot SSH commands. Symptom: process dies silently,
  no OOM signature, no traceback, checkpoints just stop. **Reliable fix**: don't background
  remotely at all — run the job in the foreground of a single, continuously-open SSH connection
  (e.g. wrapped in a local `run_in_background` shell job), so the host never sees zero sessions
  for that account during the run.
- **`pkill -f <pattern>` self-match trap**: the pattern you pass is itself part of your own
  invoking shell's command line (since it arrived via an SSH command string), so a loose pattern
  matches and kills the command issuing it before it can do anything else. Prefer killing by
  explicit PID, or use the bracket trick (`grep "[m]emtest_probe"`) to exclude the invoking
  process's own literal argument text from matching.
