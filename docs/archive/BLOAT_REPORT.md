# real_Cartpole Dependency / Bloat Diagnosis

> **SUPERSEDED / HISTORICAL** — moved to `docs/archive/` 2026-07-03. Most concrete
> recommendations here were executed the same day: `mujoco_menagerie/` full checkout deleted,
> unused deps dropped from `environment.yml`, CoppeliaSim/RL scripts archived, tuner
> duplication resolved (`tools/tuning_common.py`). Size/structure claims below are stale;
> treat this as a historical diagnosis, not a current-state document.

Diagnosis only — nothing in this report has been executed. Method note: this was a grep/find-based
heuristic pass over declared manifests vs. actual `import` usage, not a full AST/import-graph
analysis. It's reliable for clear-cut cases (zero hits, single-purpose duplicate files) but can
miss dynamic imports or shell-invoked scripts that don't literally appear elsewhere. Treat
"0 references" as "strong candidate," not proof — see the risk-ordered plan in §6 before acting on
anything here.

## 1. Size summary

Total repo: **3.0GB across 20,513 files**.

| Dir | Size | Files | Notes |
|---|---|---|---|
| `mujoco_menagerie/` | 1.1G | 2,467 | vendored MuJoCo robot model zoo |
| `outputs/` | 829M | 4,289 | run artifacts — hundreds of sweep/tuning result dirs, not source |
| `third_party/` | 457M | 4,777 | vendored CoppeliaSim runtime |
| `vendor/` | 22M | 56 | a **second**, smaller copy of `mujoco_menagerie`'s UR5e/UR10e assets, alongside the full checkout at repo root — see §3 |
| `simulation/` | 1.1M | 112 | active-lane scripts (py + sh), both lanes |
| `ros2_ws/` | 517K | 66 | ROS2 workspace source |
| `controller_core/` | 539K | 63 | shared controller library |
| `tools/` | 511K | 41 | one-off diagnostic/tuning scripts |
| `docs/` | 360K | 36 | |
| `tests/` | 297K | 27 | |
| `hardware/` | 182K | 16 | real-UR5e RTDE staging lane |
| `rl/` | 71K | 10 | RL training/eval, undeclared deps (see §2) |
| `scripts/`, `config/` | 58K / 29K | 4 / 4 | |
| `reports/`, `assets/` | 23K / 12K | 2 / 2 | |
| `"controller study/"` | 1.0K | 0 | **empty directory**, note the space in the name |

Excluding `third_party/` and `mujoco_menagerie/` (1.56G vendored), the project-owned tree is
≈1.4GB, of which **829M (≈60%) is `outputs/` run artifacts**, not source. Actual
source+docs+config footprint is on the order of tens of MB.

## 2. Dependency manifests vs. actual usage

**Manifests found**: `environment.yml` (conda env `mujoco_ur5e`, Python 3.12) is the only
top-level Python dependency manifest — no `requirements*.txt`/`pyproject.toml`/`Pipfile` exists
anywhere. Several ROS2 packages under `ros2_ws/src/` have their own `setup.py`/`package.xml`.

**Declared but zero import usage found anywhere in the repo** (candidates for removal):
- `opencv-python-headless` (`cv2`) — zero references.
- `imageio` / `imageio-ffmpeg` — zero references (only mentioned in `AGENTS.md` prose about a past
  failed attempt to add them).
- `ipywidgets` — zero references.
- `filelock` — zero references.

That's 4 of ~14 declared pip/conda-forge Python packages — close to 30% of the declared
package-level surface — with no detectable usage.

**Used but undeclared anywhere** (a reproducibility risk, arguably more actionable than the
unused-package list above, since it currently breaks environment recreation for two active lanes):
- `stable_baselines3` — imported (lazily) in `rl/train_ppo.py`, `rl/eval_policy.py`, checked in
  `tools/verify_rl_deps.py`. Not in `environment.yml`.
- `gymnasium` — imported in `rl/coppelia_y_transport_env.py`, `tools/verify_rl_deps.py`. Not
  declared anywhere.
- `rtde_control` / `rtde_receive` (the `ur_rtde` package) — imported (lazily) in
  `hardware/ur5e_rtde_bridge.py`. Not declared anywhere.

Anyone recreating the conda env from `environment.yml` alone cannot run `rl/train_ppo.py` or the
real-hardware bridge without guessing these three packages.

**Broken ROS2 package manifests**: `ros2_ws/src/real_cartpole_workspace_server/` and
`ros2_ws/src/real_cartpole_workspace_msgs/` both have **1-byte stub `package.xml`/`setup.py`**
files (literally just a newline), despite containing real Python code
(`workspace_analysis_service.py`, a launch file, a `.srv` definition). Neither package can
actually be built/installed via `colcon` as-is.

## 3. Duplicate / near-duplicate implementations

- **Logging — near-identical, likely copy-pasted**: `controller_core/logging_utils.py`
  (`json_dumps_safe`, `JsonlTraceWriter`) vs. `hardware/logging.py` (`json_dumps_safe`,
  `write_json`, `JsonlWriter`). Both implement the same numpy-safe JSON serializer and the same
  "open file, append one JSON object per line" writer class, just renamed. `controller_core`'s
  version is used across both simulation lanes; `hardware/logging.py` is only used within
  `hardware/__init__.py`. Clean consolidation candidate — `hardware/` could just import from
  `controller_core`.
- **Config loading — one real abstraction, bypassed almost everywhere else**: the only dedicated
  loader is `ros2_ws/.../ur5_x_axis_controller_ros/config_loader.py`
  (`load_yaml_config`/`torque_dict_to_array`). At least 6 other files each hand-roll their own
  trivial `_load_yaml(path)` wrapper around `yaml.safe_load` instead of reusing it:
  `tools/audit_ur5e_mujoco_gravity_torque.py`, `tools/tune_ur5e_impedance_transport.py`,
  `tools/ur5e_mujoco_torque_experiments.py`, `simulation/workspace_guardrails.py`, plus inline
  `yaml.safe_load(...)` in `rl/coppelia_y_transport_env.py` and `rl/train_ppo.py`. Not a
  "duplicate system," but a scattered 2-3-line pattern that should be one shared helper.
- **Tuning-script duplication**: `tools/tune_ur5e_impedance_transport.py` (1,011 lines) and
  `tools/tune_ur5e_residual_impedance_transport.py` (1,042 lines) are near-duplicate grid-search
  tuning drivers — the "residual" variant literally imports ~13 helper functions from the first
  file but then redefines its own large gain-search grids and stage logic in parallel rather than
  parameterizing the original. Combined, ~2,050 lines of largely parallel structure for two
  similar tuning campaigns.
- **Not a duplication (verified, noted for completeness)**: `controller_core/kinematics_utils.py`'s
  `world_linear_jacobian`, `ros2_ws/.../coppeliasim_adapter.py`'s API-vs-numerical Jacobian
  cross-validation, and `jacobian_provider.py`'s topic-driven cache serve three genuinely different
  roles, not copy-paste duplication.

## 4. Dead code / orphaned script candidates

**High-confidence (zero references anywhere in the repo, including docs/shell scripts, non-test files)**:
- `scripts/patch_ur5e_mjcf_torque.py`
- `simulation/render_coppeliasim_trace_mp4.py` and `simulation/render_coppelia_trace_mujoco_mp4.py`
  (similarly-named, both unreferenced — check together, possibly a naming-drift pair)
- `simulation/step_coppeliasim_lua_video.py`
- `tools/analyze_z_drift_trace.py`, `tools/print_trace_zy.py`, `tools/summarize_trace_zy.py`,
  `tools/check_summary.py` (hardcodes a specific output path), `tools/check_mujoco_gravity.py`,
  `tools/probe_coppelia_sim_api.py` — one-off ad hoc debug scripts
- `tools/test_args.py` — a 7-line throwaway argv-parsing smoke script (not a real pytest test
  despite the filename)
- `tools/verify_rl_deps.py` — plausibly still useful as a manual sanity check (and it's what
  revealed the undeclared `stable_baselines3`/`gymnasium` deps in §2) even though nothing calls it
- **`ros2_ws/src/real_cartpole_workspace_server/`** (entire package) — near-empty stub service
  file, broken manifests (§2), launch file referenced nowhere else. Looks like unfinished
  scaffolding.
- `ros2_ws/src/real_cartpole_description/launch/view.launch.py` — weaker evidence of "dead" here,
  since ROS2 launch files are often invoked ad hoc via `ros2 launch` from the command line without
  an in-repo reference.

**Lower-confidence, needs human judgement (exactly one cross-reference, or clearly manual/interactive
tools)**: `simulation/coppelia_pendulum.py`, `simulation/probe_external_zmq_handshake.py`,
`simulation/probe_external_zmq_single_joint_torque.py`, `simulation/render_pose_orbit.py`,
`simulation/run_mujoco_cartpole_mpc_smoke.py`, `tools/check_mujoco_gravity2.py`,
`tools/diagnose_trace.py`, `tools/probe_gravity_multi_pose.py`, `tools/read_summary.py`,
`tools/ur5e_hardware_smoke_test.py`, `controller_core/actuation_shadow.py`,
`rl/coppelia_sim_manager.py`, `rl/env_factory.py`, `rl/smoke_test_env.py`. The sheer number of
similarly-named `probe_*`/`check_*`/`diagnose_*` one-offs under `tools/` and `simulation/`
suggests an accumulation pattern worth a broader cleanup pass even though each individual script
has a technical reference somewhere.

**Not orphaned (expected false positive of this heuristic)**: all `test_*.py` files under
`tests/`/`controller_core/tests/` — discovered by pytest convention, not cross-referenced.

Also worth checking: three `tools/run_*_probe.sh` gravity-probe launchers
(`tools/run_gravity_probe.sh`, `tools/run_coppelia_gravity_probe.sh`,
`tools/run_gravity_multi_pose.sh`) hardcode a Windows/WSL dev path
(`ROOT="/mnt/c/Users/sandr/Downloads/real-cartpole-control-using-ur5-direct-torque"`) that doesn't
match the current HPC/Singularity setup described in `AGENTS.md` — likely stale/non-portable
rather than dead code per se, but won't run as-is on this server copy.

## 5. Size/complexity contributors

Top project-owned Python files by line count (excluding `third_party/`, `mujoco_menagerie/`,
`outputs/`, `vendor/`):

| Lines | File |
|---|---|
| 4,361 | `simulation/run_coppeliasim_x_axis_headless.py` |
| 1,607 | `simulation/run_x_acceleration_transport.py` |
| 1,366 | `tools/audit_ur5e_mujoco_gravity_torque.py` |
| 1,173 | `simulation/run_fixed_z_x_transport.py` |
| 1,171 | `tools/ur5e_mujoco_torque_experiments.py` |
| 1,109 | `simulation/controller.py` |
| 1,042 | `tools/tune_ur5e_residual_impedance_transport.py` |
| 1,011 | `tools/tune_ur5e_impedance_transport.py` |
| 1,002 | `transport_metrics.py` (repo root) |
| 955 | `simulation/workspace_guardrails.py` |
| 868 | `hardware/ur5e_stages.py` |
| 754 | `simulation/run_x_velocity_transport.py` |
| 751 | `tools/ur5e_move_hold_transport.py` |
| 732 | `tools/ur5e_x_frame_envelope.py` |

The single largest file, `simulation/run_coppeliasim_x_axis_headless.py` at 4,361 lines, is 2.7x
the next-largest file — the strongest decomposition candidate in the repo regardless of the
dependency question. Six of the top nine files are `tools/` tuning/audit drivers, not core
`controller_core/` library code (whose files are all under ~300 lines) — most of the project's
line-count mass is in one-off diagnostic/tuning scripts, not the shared library.

**Hand-rolled vs. dependency-equivalent code**: `controller_core/box_qp.py` (42 lines) is a small
hand-rolled box-constrained QP solver, explicitly documented as "no external deps" — a deliberate
design choice given its size, not accidental bloat. No hand-rolled URDF or YAML parsing was found
(both use their respective standard libraries correctly).

`vendor/mujoco_menagerie/` (22MB, UR5e/UR10e assets only) appears to duplicate a subset of what's
already in the much larger `mujoco_menagerie/` (1.1GB) vendored checkout at repo root — flag for a
"still needed?" decision rather than assuming it's safe to delete.

## 6. Proposed cleanup plan (diagnosis only — nothing here has been executed)

Ordered by risk, for the user to review and decide on before any execution:

**Safe / low-risk**
- Remove the 4 zero-usage packages from `environment.yml`: `opencv-python-headless`, `imageio`,
  `imageio-ffmpeg`, `ipywidgets`, `filelock`.
- Delete the empty `"controller study/"` directory (odd space in the name, no content).

**Low-risk but needs a decision**
- Consolidate `hardware/logging.py` into `controller_core/logging_utils.py` (near-identical, one
  clear source of truth).
- Add `stable_baselines3`, `gymnasium`, `rtde_control`/`rtde_receive` to `environment.yml` so the
  RL and hardware lanes are actually reproducible from the manifest.
- Fix or remove the two broken ROS2 manifests (`real_cartpole_workspace_server`,
  `real_cartpole_workspace_msgs`) — either populate the stub `package.xml`/`setup.py` files or
  delete the packages if they're abandoned scaffolding.
- Fix the misleading docstring on `controller_core/safety_utils.py` (see `AUDIT_REPORT.md` §3) or
  remove the unused `SafetyMonitor` class, keeping only the constant that's actually consumed.

**Needs human review before any deletion**
- The dead-code candidate list in §4 (high-confidence list first, then the lower-confidence list).
- The tuning-script duplication (`tune_ur5e_impedance_transport.py` vs
  `tune_ur5e_residual_impedance_transport.py`) — likely worth merging into one parameterized
  driver, but that's a real refactor, not a deletion.
- `vendor/mujoco_menagerie/` vs the full `mujoco_menagerie/` checkout — confirm which consumers
  actually need the smaller copy before removing it.
- Renaming the four MuJoCo scripts that read as CoppeliaSim scripts (see `AUDIT_REPORT.md` §3,
  "naming trap") to avoid future lane confusion.

**Explicitly out of scope for this pass**: no deletions, renames, dependency-file edits, or
refactors have been executed. This section is a proposal for the user's next decision, per the
task brief's guardrail.
