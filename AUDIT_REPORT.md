# real_Cartpole Repo Status Audit

Generated as a report-only diagnostic pass. No controller code, configs, or gains were changed as
part of this audit. All file:line citations below were verified by reading the actual source, not
inferred from filenames.

## 0. Framing correction — read this first

The task brief that motivated this audit described CoppeliaSim as "the controller-diagnosis
target" and cited "a recent 112-run sweep had 60 success/52 failure" as evidence of the current
controller's structural failure modes. **That sweep is MuJoCo-only data**, produced by
`tools/ur5e_x_frame_envelope.py` (output at
`outputs/ur5e_mujoco_torque_transport/x_frame_envelope_20260630_003031/`), whose own docstring
states "No RL, hardware, RTDE, URScript, or CoppeliaSim code is imported." No CoppeliaSim
equivalent of this sweep exists on disk — the largest CoppeliaSim batch artifacts are ~20
manually-named one-off `coppelia_lqr_smokeN_*` runs, not a single programmatic sweep with a CSV.

This is consistent with `docs/CURRENT_STATUS.md` (the most recently updated doc), which states
outright that MuJoCo residual-torque tuning is now the active lane, and that "CoppeliaSim
transport/torque work remains in the tree as a separate historical/reference path... but it is
not the active objective for the current branch work." This directly contradicts
`docs/README.md`/`docs/WORKSPACE_MAP.md`, which still describe CoppeliaSim as primary — those two
docs are stale relative to `CURRENT_STATUS.md` and should be updated or archived.

**The user has confirmed MuJoCo is the active lane.** The rest of this report is written with that
understood, and the new observability module (Task C, see `observability/run_logger.py`) targets
the MuJoCo entrypoints accordingly. The CoppeliaSim/MuJoCo boundary check below (§2) remains fully
in scope regardless of this pivot — it surfaced the most safety-relevant finding of the whole pass.

## 1. Structure map

One-line purpose per top-level directory, verified by reading representative files inside each
(not inferred from directory names):

| Directory | Purpose |
|---|---|
| `controller_core/` | Simulator-independent torque controller library: impedance/LQR/MPC/QP controllers, kinematics, safety monitor, filters, shared JSONL logging utility. Has its own `tests/`. |
| `simulation/` | Both CoppeliaSim launchers/runners and MuJoCo diagnostic scripts coexist here — classify by actual import, not filename (see §4). |
| `ros2_ws/` | ROS 2 workspace: `ur5_x_axis_controller_ros` (CoppeliaSim topic-based adapter + controller node), `real_cartpole_control` (real-hardware origin-hold controller), MoveIt config, plus a stubbed-out `real_cartpole_workspace_server` package (see `BLOAT_REPORT.md`). |
| `tools/` | Standalone CLI scripts: MuJoCo gravity/torque audits and sweep drivers (the active lane's real entrypoints), CoppeliaSim gravity probes, real-hardware staging scripts. |
| `config/` | YAML configs for both lanes, plus `lab_workspace_guardrails.yaml` (simulation/visualization only). |
| `docs/` | `docs/CURRENT_STATUS.md` is the authoritative current-state doc; `docs/coppeliasim/` and `docs/archive/` hold lane-specific and legacy notes respectively — see the framing correction above for doc drift. |
| `rl/` | PPO RL stack for CoppeliaSim Y-axis transport — a third, separate control lane, with its own undeclared dependencies (see `BLOAT_REPORT.md`). |
| `scripts/` | Two one-off conversion utilities (URDF→MJCF, MJCF torque-actuator patching). |
| `outputs/` | All run artifacts (829MB, ≈60% of project-owned disk usage). `control_runs/` = CoppeliaSim + misc; `ur5e_mujoco_torque*/` = MuJoCo controller-rollout/sweep outputs, including the 112-run sweep. |
| `third_party/` | Vendored CoppeliaSim runtime and its Python dependency anchor. |
| `mujoco_menagerie/` | Vendored MuJoCo UR5e/UR10e model zoo (nested git repo); a smaller duplicate subset also exists under `vendor/mujoco_menagerie/` (see `BLOAT_REPORT.md`). |
| `assets/` | Simulation-only MJCF for the MuJoCo torque lane. |
| `hardware/` | Real-UR5e RTDE staging scripts with multi-layer safety gating (see §3) — receive-only probe, zero-hold servoJ, tiny bounded motion, guarded direct-torque probe. |
| `tests/` | pytest suite covering CoppeliaSim adapter/transport/diagnostics, MuJoCo torque, and hardware session code. |
| `reports/` | Two hand-written one-off audit markdown files (not generated logs). |

## 2. CoppeliaSim vs MuJoCo boundary check

This is the audit's top safety-relevant finding: **the CoppeliaSim controller diagnostic entry
point directly depends on MuJoCo**, in ways not visible from its own docstring, which calls it the
"sole orchestrator" for CoppeliaSim.

### 2a. MuJoCo `qfrc_bias` silently used as CoppeliaSim's default gravity-compensation source

- `simulation/run_coppeliasim_x_axis_headless.py:260-273` loads a MuJoCo UR5e model purely to
  compute a gravity-bias estimate for the live CoppeliaSim run.
- `simulation/run_coppeliasim_x_axis_headless.py:275-298` (`compute_mujoco_gravity_bias`) sets a
  scratch MuJoCo `MjData`'s `qpos`/`qvel` from the **live CoppeliaSim joint state**, calls
  `mujoco.mj_forward`, and returns `data.qfrc_bias[:6]`.
- `simulation/run_coppeliasim_x_axis_headless.py:301-319` maps that into a CoppeliaSim
  joint-torque feed-forward with an empirically-tuned `sign=-1.0, scale=1.0` (the docstring notes
  this was tuned by a one-off probe on 2026-06-25).
- `simulation/run_coppeliasim_x_axis_headless.py:321-338` (`gravity_compensation_flags`): **when
  a config's `controller:` section omits `gravity_compensation_source`, the default value is
  `"mujoco"`**, not any CoppeliaSim-native computation.
- Confirmed in configs: `ros2_ws/src/ur5_x_axis_controller_ros/config/controller_coppelia_{mpc,
  reciprocating,slow,slow_seeded_probe,slow_seeded_relaxed,fast_x}.yaml` all set
  `use_gravity_compensation: true` **without** specifying `gravity_compensation_source` — i.e.
  they silently inherit the MuJoCo-derived bias. Only `controller_coppelia_y_transport_torque.yaml`
  sets it explicitly. Several other configs (the base `controller.yaml`, `_after_stable`/
  `_legacy_xz_transport*` variants) set `use_gravity_compensation: false`, so this coupling is
  config-dependent, not universal — but it is an implicit default rather than an explicit opt-in,
  and it's live in several of the actively-used bring-up configs.

**Risk**: any future MuJoCo-lane change to `mujoco_menagerie/universal_robots_ur5e/
scene_ur5e_cartpole.xml` (mass/inertia parameters) will silently change CoppeliaSim closed-loop
torque-control behavior for any config with `use_gravity_compensation: true` and no explicit
`gravity_compensation_source` override. This is exactly the kind of cross-lane conflation
`AGENTS.md` warns against, and it's currently live, not hypothetical.

### 2b. MuJoCo pole-state observer fallback in the CoppeliaSim MPC outer loop

- `simulation/run_coppeliasim_x_axis_headless.py:73-77,1853-1889` — when running the MPC
  outer-loop policy (`use_mpc_outer_policy`), the runner prefers CoppeliaSim's own pendulum sensor
  state but falls back to a MuJoCo-model-based pole-state observer
  (`controller_core.mujoco_cartpole_state`) if CoppeliaSim's pendulum handle isn't wired up.
  Limited blast radius (MPC mode only, CoppeliaSim sensor preferred when present), but still a
  MuJoCo-model dependency inside the script that calls itself the sole CoppeliaSim orchestrator.

### 2c. Shared config schema, no lane tag

- Both lanes' controller YAMLs (`config/ur5e_mujoco_torque*.yaml` for MuJoCo,
  `ros2_ws/.../config/controller_coppelia_*.yaml` for CoppeliaSim) use the **identical**
  `controller:` schema, parsed by the same `CartesianImpedanceConfig.from_controller_yaml_section`
  / `TorqueTaskQPConfig.from_controller_yaml_section` classmethods
  (`controller_core/x_axis_cartesian_impedance.py:51-83`, `controller_core/torque_task_qp.py:30-55`).
  A config file from one lane is structurally valid and would silently load in the other lane's
  runner rather than raising an error.

### 2d. No lane/backend tag in result JSON — the gap Task C's schema fixes

Both lanes independently populate a generic top-level `"success": <bool>` and
`"failure_reasons"` key in their summary JSON with **no field anywhere identifying which
simulator produced the result** (`simulation/run_coppeliasim_x_axis_headless.py:4262` vs.
`simulation/run_x_velocity_transport.py:735`, `tools/ur5e_x_frame_envelope.py`, etc. — grepped for
`"backend"`/`"simulator"`/`"engine"` tagging, zero matches in either lane). The only thing
separating CoppeliaSim results from MuJoCo results is **output-directory naming convention**
(`outputs/control_runs/coppelia_*` vs `outputs/ur5e_mujoco_torque*`), which is a filesystem
convention, not a schema guarantee — if a summary JSON were copied out of its directory (e.g. into
a report), nothing in the file itself would prevent misattributing it to the wrong lane. This is a
real mitigating factor today (directories are well-separated in practice), but it's not
schema-enforced. The new `observability/run_logger.py` module adds an explicit `backend` field to
every `RunRecord` for exactly this reason (MuJoCo-side only in this pass; the field is designed so
a future CoppeliaSim pass needs no schema break).

**Not a concern**: `transport_metrics.py` (the MuJoCo-lane scoring/ranking module) and all
MuJoCo-lane tuning tools stay MuJoCo-only — no cross-lane import found there.

## 3. Controller inventory

### Simulator-independent core (`controller_core/`, verified via its own `__init__.py` docstring: no MuJoCo/CoppeliaSim/ROS imports)

| File | What it does | Live/orphaned |
|---|---|---|
| `x_axis_cartesian_impedance.py` | Primary 6-DOF Cartesian impedance/PD law: task wrench → `J.T @ wrench` + joint damping + posture PD + externally-supplied gravity bias, geometric backtracking, hard torque clip. | **Live** — used by both CoppeliaSim and MuJoCo lanes. |
| `x_axis_controller.py` | Minimal 1-DOF Cartesian-X PD (no Jacobian, no gravity comp), earlier prototype. | **Orphaned** — exported from `__init__.py`, covered by its own test, but no runner imports it. |
| `torque_task_qp.py` | QP-based torque allocator: task Jacobian + weights → box-constrained QP → hard clip. | Live, shared/portable, selectable alternate torque law. |
| `box_qp.py` | Generic box-constrained QP solver, deliberately dependency-free (docstring states this explicitly). | Live (used by QP and MPC controllers). |
| `kinematics_utils.py` | Jacobian-transpose force→torque mapping, quaternion math, world-frame Jacobian rotation. | Live, shared/portable. |
| `joint_impedance.py` | Pure joint-space PD torque law, documented as "the simulation-only torque lane." | Live — MuJoCo lane only. |
| `transport_lqr.py` | Discrete LQR outer-loop transport controller (command-limited). | **Live** — used by both `run_coppeliasim_x_axis_headless.py` and `run_x_acceleration_transport.py` (MuJoCo). |
| `lqr_controller.py` | Cart-pole (4-state) fallback PD and LQR controllers. | **Orphaned** — exported from `__init__.py` but only referenced by its own test; superseded in practice by `transport_lqr.py`. |
| `mpc_controller.py` | Finite-horizon linear MPC on the 4-state cart-pole model, QP-solved. | Live — used by `coppelia_mpc_transport.py` (CoppeliaSim) and `run_mujoco_cartpole_mpc_smoke.py` (MuJoCo). |
| `cartpole_linear_model.py` | Linearized cart-pole plant + discrete Riccati solver backing LQR/MPC. | Live, shared/portable. |
| `controller_interfaces.py` | Interfaces + `SafetyLimits` dataclass (workspace bounds, `pole_angle_hard_cutoff_rad=0.65`, `fallback_action="brake"`). | Live, shared/portable. |
| `safety_filter.py` | `CommandGovernorSafetyFilter` — rejects on non-finite state/hard cutoffs, clamps into a dynamically computed safe interval, rate-limits command deltas. | **Live** in the CoppeliaSim LQR/MPC outer loop and MuJoCo `run_x_acceleration_transport.py`. |
| `recoverability_monitor.py` | Recoverability scoring heuristic used only inside `safety_filter.py`. | Internal helper, live. |
| `actuation_shadow.py` | Simulation-only torque-channel perturbation model (delay, slew-limit, deadzone, friction mismatch) — explicitly documented as not touching hardware-facing code. | MuJoCo-only in practice; only import site is `run_x_acceleration_transport.py`. |
| `filters.py` | `TorqueCommandFilter` — low-pass + per-joint slew-rate limiter. | Live, shared/portable — used by ROS2 `controller_node.py` and `run_coppeliasim_x_axis_headless.py`. |
| `safety.py` | `ImpedanceSafetyMonitor` — drift-from-initial-pose (Y/Z/orthogonal), orientation error, joint velocity, monotonic axis-error growth, NaN/joint-limit e-stop triggers. | **Live** — imported by both the CoppeliaSim runner and ROS2 `controller_node.py`. This is the actual source of the `termination_reason` strings seen in the 112-run sweep data (e.g. `"\|Y-Y0\| > 0.03 m"`). |
| `safety_utils.py` | A structurally different `SafetyMonitor`/`SafetyConfig`, with a docstring claiming it's "reused by... the ROS 2 controller node" — **this claim is false as currently wired**. `controller_node.py` actually imports `safety.py`'s `ImpedanceSafetyMonitor`, not this file's class. | **Effectively dead as a monitor** — the only real consumer anywhere is `hardware/safety_limits.py:11`, which imports just its `UR5_MANUFACTURER_QD_MAX_RAD_S` constant. Misleading docstring, not a functional gap (the real monitor is correctly wired elsewhere), but worth fixing the comment or removing the unused class. |
| `logging_utils.py` | `json_dumps_safe` + `JsonlTraceWriter` — the closest thing to a shared logging primitive in the repo. Near-duplicated by `hardware/logging.py` (see `BLOAT_REPORT.md`). | Live, widely used by both lanes. |

**Gravity compensation / Coriolis / mass matrix — a structural gap, not a bug**: a full-repo grep
for `coriolis|mass_matrix|inertia_matrix|mj_fullM|rnea` (case-insensitive, excluding vendored
dirs) returns **zero matches**. There is no hand-coded mass-matrix or Coriolis term anywhere.
Gravity(+Coriolis) compensation comes exclusively from MuJoCo's `data.qfrc_bias` (gravity +
Coriolis/centrifugal combined bias force), consumed in three places:
`tools/audit_ur5e_mujoco_gravity_torque.py`, `simulation/run_x_torque_transport_mujoco.py:102,201`,
and `simulation/run_coppeliasim_x_axis_headless.py:275-298` (see §2a). The impedance/QP/joint
control laws themselves never compute gravity — they only add whatever `gravity_torque` vector is
handed to them externally. This is consistent with the brief's own framing that the 112-run
sweep's failure modes (drift/orientation/settling, not raw torque saturation — the brief notes at
least one run showed 0% torque saturation with real tracking error) are structural: without a
computed mass matrix or Coriolis term, the controller has no feedforward compensation for
velocity-dependent dynamics, only for static/quasi-static gravity bias.

### Live CoppeliaSim orchestrator and its import graph

The real, single live CoppeliaSim entry point is **`simulation/run_coppeliasim_x_axis_headless.py`**
(its own docstring: "sole orchestrator"), invoked via `simulation/launch_coppeliasim_x_axis_
headless.sh` and wrapped by `simulation/run_coppelia_torque_diagnostics_smoke.py` (a smoke-test
ladder that subprocesses it per named diagnostic). Its live controller stack: `x_axis_cartesian_
impedance.py` (inner torque law) + `torque_task_qp.py` (alternate, selectable) + `transport_lqr.py`
/`mpc_controller.py` (outer transport generation, via `coppelia_lqr_transport.py`/
`coppelia_mpc_transport.py`) + `safety_filter.py`'s command governor + `safety.py`'s
`ImpedanceSafetyMonitor` + `filters.py`'s torque filter.

A **separate, parallel** CoppeliaSim path exists in the ROS2 package (`controller_node.py` +
`coppeliasim_bridge_node.py` + `coppeliasim_adapter.py`) — a topic-based node/bridge architecture
that reuses the same `controller_core` classes but is a distinct process design from the
single-process ZMQ runner above.

### Naming trap — four scripts that sound CoppeliaSim-related but are pure MuJoCo

Verified by checking for actual `mujoco.MjModel.from_xml_path`/`mj_step` usage, not filenames:
`simulation/run_fixed_z_x_transport.py`, `simulation/run_x_velocity_transport.py`,
`simulation/run_x_acceleration_transport.py`, `simulation/run_origin_stabilization.py`. All four
are pure MuJoCo diagnostics with **zero CoppeliaSim/ZMQ import**. `AGENTS.md` explicitly warns
against conflating the two lanes — these names invite exactly that mistake, and it's worth a pass
to rename them (e.g. `run_x_velocity_transport.py` → `run_mujoco_x_velocity_transport.py`) or at
minimum add a docstring banner, though that rename is out of scope for this report-only pass.

MuJoCo lane's real entrypoints (used going forward per the confirmed pivot): `tools/audit_ur5e_
mujoco_gravity_torque.py` (gravity-hold), `tools/ur5e_move_hold_transport.py` (move-hold sweep
driver — subprocesses `tools/ur5e_mujoco_torque_experiments.py` per grid point rather than running
its own step loop), `tools/ur5e_x_frame_envelope.py` (the 112-run sweep, itself subprocessing
`tools/ur5e_mujoco_torque_experiments.py`).

## 4. Hardware guardrail check

All four guardrails from the brief are **strongly enforced**, with defense in depth (opt-in flags
*and* per-cycle bound checks *and*, in one case, a code path that's currently unreachable
regardless of flags).

### (a) Receive-only / dry-run default for hardware/RTDE control paths — enforced

- `hardware/ur5e_rtde_bridge.py:101-106` — class docstring: "does not send motion by default."
  `connect_receive_only()`/`connect_control()` are separate methods; nothing calls
  `connect_control()` implicitly.
- `hardware/ur5e_control_session.py:35-49` — `UR5eHardwareSessionConfig.motion_opt_in: bool =
  False`, `allow_nonzero_direct_torque: bool = False`, `direct_torque_zero_only: bool = True` — all
  default-safe at the dataclass level.
- `ros2_ws/.../ur5e_hardware_pipeline_node.py:69-93` and `launch/run_ur5e_hardware_pipeline.
  launch.py:19-38` — ROS2 parameter and launch-file defaults mirror the library defaults
  (`stage="connection_smoke"`, all opt-ins `false`).

### (b) servoJ requires explicit opt-in — enforced

- `hardware/ur5e_control_session.py:176-184,246-254` — `request_servoj_hold`/
  `request_servoj_tiny_motion` both return a blocked result if `motion_opt_in` is false.
- `hardware/ur5e_stages.py:422-425,599-608` — `run_servoj_zero_hold`/`run_servoj_tiny_motion` both
  `raise SystemExit` unless `motion_opt_in`; the tiny-motion path additionally validates amplitude
  and joint-index bounds before any command.
- `tools/ur5e_servoj_zero_hold.py:47-51`, `tools/ur5e_servoj_tiny_motion.py` — the CLI flag
  `--i-understand-this-moves-the-robot` (default `False`) is the only way to set `motion_opt_in`.
- Every commanded `cmd_q`, even after opt-in, passes through `MotionCommandGuard.check`
  (jump/velocity/acceleration bounds) before `bridge._call_servoj` is invoked
  (`hardware/ur5e_stages.py:490-498,677-689`).

### (c) Direct torque commands zero-only by default — enforced

- `hardware/ur5e_control_session.py:42` — `direct_torque_zero_only: bool = True` (dataclass
  default).
- `hardware/ur5e_rtde_bridge.py:269-298` (`send_joint_torque`) — signature default `zero_only:
  bool = True`; blocks any nonzero torque unless explicitly overridden.
- `tools/ur5e_direct_torque_probe.py:29-41` — CLI defaults `zero_only=True`; requires
  `--enable-nonzero-torque` to flip.

### (d) Nonzero direct torque blocked — enforced, and stricter than the CLI surface implies

- `hardware/ur5e_rtde_bridge.py:290-295` and `hardware/ur5e_control_session.py:337-354` — two
  independent gates in series (`zero_only` check, then a separate `allow_nonzero` check), checked
  at both the bridge and session layers.
- `hardware/ur5e_stages.py:791-800` (`run_direct_torque_probe`) — nonzero mode additionally
  requires **three simultaneous** explicit flags (`--i-understand-direct-torque-is-dangerous`,
  `--i-am-with-trained-supervisor`, `--enable-nonzero-torque`), plus hard caps
  (`duration <= 2.0s`, `max_torque_nm <= 1.0`), else `SystemExit`.
- **Positive surprise**: `hardware/ur5e_stages.py:824-840` — even when all flags are set and
  capability is detected, the function **never actually calls `bridge.send_joint_torque`**; it
  only records `"direct torque support detected but guarded nonzero execution is not enabled in
  this patch"` and unconditionally sets `ok = False`. The standalone probe tool is currently
  **incapable of sending nonzero torque at all**, regardless of CLI flags — stricter than the flag
  surface suggests. `hardware/ur5e_rtde_bridge.py:199-212` further notes the underlying RTDE
  control library doesn't expose a real direct-torque motion API for this setup anyway, so this
  path would currently fail even if the guard were relaxed.

### Other guardrails worth noting

- `hardware/safety_limits.py:23-67,134-243` — `UR5eSafetyLimits`/`UR5eStateGuard`/
  `MotionCommandGuard` provide per-cycle state-freshness, command-jump, velocity, and acceleration
  guards, checked before every motion command across all staged hardware scripts.
- `hardware/ur5e_rtde_bridge.py:173-197` (`safe_stop`) — best-effort `servoStop → stopJ(2.0) →
  stopScript` sequence, called from every stage's `finally` block.
- `ros2_ws/.../controller_node.py:259-261,391-400,467-472,474-477` — on e-stop or NaN in
  `tau_cmd`, the node latches e-stop and publishes an all-zero torque command. **No un-latch path
  exists in this file** — once tripped, the node requires a restart. This looks like an
  intentional conservative fail-safe design; flagging for confirmation, not proposing a change.

## 5. Existing logs/traces inventory

Eight distinct schemas exist on disk (paths, formats, and field lists below). **No single existing
schema captures the full checklist** (Y/Z/orthogonal drift, orientation error, joint velocity,
commanded-vs-clipped torque per joint, saturation %, terminal/settling state) — this is exactly
the gap `observability/run_logger.py` closes for the MuJoCo lane.

| Schema | Location | Y/Z/orthog. drift | Orientation error | Joint velocity | Torque cmd vs clipped | Saturation % | Terminal state |
|---|---|---|---|---|---|---|---|
| A — MuJoCo per-step trace.jsonl | `outputs/ur5e_mujoco_torque*/**/trace.jsonl` | not per-row (derivable from `ee_pos`) | not per-row | ✅ per-joint (`qd`) | ✅ (`tau_raw`/`tau_filtered`/`tau`) | via `torque_saturation_fraction` | ✅ `termination_reason`, `safety_ok`/`safety_reason` |
| B — MuJoCo run summary.json (via `transport_metrics.py`) | sibling of each trace.jsonl above | ✅ | ✅ | ✅ scalar max only, not per-joint | ✅ controller vs applied, clip fraction | ✅ | ✅ `termination_reason`/`success` |
| C — MuJoCo sweep aggregate CSV | `outputs/ur5e_mujoco_torque_transport/*/summary.csv` | some variants missing entirely | some variants missing | ✅ | ✅ | ✅ | ✅ |
| D — CoppeliaSim per-step trace.jsonl | `outputs/control_runs/*.jsonl` | ✅ | ✅ | ✅ | partial (only in one policy's nested diagnostics) | not uniform | not per-row |
| E — CoppeliaSim run summary.json | `outputs/control_runs/*_summary.json` | ✅ | ✅ | ✅ scalar peak only | ✅ scalar | ✅ (`tau_saturation_fraction`) | ✅ |
| F — CoppeliaSim per-joint torque diagnostics | `outputs/control_runs/coppelia_torque_diagnostics/*` | ❌ not tracked at all | ❌ not tracked | n/a | ✅ true per-joint, the only one | ✅ per-joint | ✅ `pass`/`suspected_failure_reason` |
| G — plain-text `.log` | `outputs/control_runs/*.log` | n/a (unstructured stdout capture) | | | | | |
| H — workspace study JSON | `outputs/workspace_studies/*` | unrelated (Jacobian-conditioning study, not drift/torque) | | | | | |

Schema B is closest for drift/orientation/torque-fraction completeness but lacks per-joint
breakdown; Schema F has the only true per-joint torque tracking but zero Cartesian-drift
awareness. `observability/run_logger.py`'s `RunRecord` schema merges both for the MuJoCo lane.

**Caveat**: the on-disk `x_frame_envelope_20260630_003031/summary.csv` (the 112-run sweep's CSV)
does not match the current `tools/ur5e_x_frame_envelope.py` source's intended `_write_csv` field
list (which includes fields like `max_abs_y_drift_m`/`orthogonal_drift_pass` not present in the
on-disk file) — schema drift between historical output and current code. Don't assume old CSVs in
`outputs/` match what the current script would produce; re-run if you need current-schema data.

Also note: `docs/coppeliasim/RPC_CONTROLLER_TODO.md`'s Phase 6 acceptance criteria already states,
in prose, almost exactly the field checklist used to design the new logger ("Watch X error sign
and reduction. Watch Y/Z drift. Watch orientation error. Watch joint velocities. Watch torque
saturation. Save JSONL trace and summary.") — this was prior intent that was never formalized into
a schema or module until this pass.
