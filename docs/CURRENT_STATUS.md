# Current Status

Last updated: 2026-07-03 (post Pinocchio P0-P3 + OSC gain tuning + transport_metrics sign-bug fix).

## Active Objective

**MuJoCo UR5e gravity-compensated residual-torque transport** in simulation, using a
Pinocchio-backed model-based controller (operational-space, tuned). Active lane:

```text
tools/ur5e_mujoco_torque_experiments.py    # rollout engine (owns the step loop)
tools/audit_ur5e_mujoco_gravity_torque.py  # gravity-sign / hold-quality audit
tools/ur5e_move_hold_transport.py          # move+hold sweep driver
tools/ur5e_x_frame_envelope.py             # transport envelope sweep
tools/tune_ur5e_residual_impedance_transport.py
tools/compare_ur5e_mujoco_controllers.py
simulation/ur5e_mujoco_torque.py           # MuJoCo adapter
controller_core/model_dynamics.py          # Pinocchio DynamicsProvider
```

Residual torque means `tau_applied = tau_controller + tau_gravity (+ tau_coriolis)`.
`tau_gravity` defaults to MuJoCo `qfrc_bias` but can source from Pinocchio
(`mujoco.gravity_source: pinocchio`); Coriolis feedforward is a separate opt-in flag. Raw
mode is validation/anti-cheating only.

## Current diagnosis: resolved, tuned, and validated

The 2026-07-01 diagnosis (`docs/archive/DIAGNOSTIC_real_cartpole_torque_control_questions.md`)
found the impedance controller structurally too simple — no mass matrix, no Coriolis
compensation, no inverse dynamics — combined with tight safety envelopes that tripped before
it settled. That diagnostic's recommended staged fix has since been fully implemented and its
outcome tuned well past parity:

- **P0-P2 (Pinocchio dynamics, flag-gated, default = legacy behavior)**: gravity source swap
  and Coriolis feedforward, both validated against MuJoCo to tight tolerances.
- **P3 (operational-space upgrade, flag-gated)**: task-space inertia shaping (Λ(q)) +
  dynamically consistent nullspace posture projection.
- **Tuned OSC config** (`config/ur5e_mujoco_torque_osc_tuned.yaml`): ~250+ evaluation runs
  found and fixed two root causes at the transport start pose's wrist singularity — a
  task-unactuatable drift direction (fixed by a joint-space posture anchor + re-anchoring)
  and a genuine positive-feedback instability in the task rotation PD (fixed by `kp_rot=0`,
  damping-only). Validated envelope, zero safety-guard trips throughout: canonical grid 8/8,
  long holds to 30s 8/8, displacements to 20cm 16/16, torque-scale robustness to 10% 14/14.
  Full evidence and mechanism write-up: `AGENTS.md` §3.
- **Bug found and fixed along the way**: `transport_metrics.py`'s move-hold pass-check
  `abs()`'d the target displacement before comparing it to the signed achieved value, so
  every negative-direction transport run was silently reported invalid regardless of actual
  tracking quality. Fixed 2026-07-03 (`transport_metrics.py`, see `AGENTS.md` §4).

Untested/known-out-of-scope boundaries (physical or bandwidth limits, not defects): dx=0.25m
fails via Z-drift (workspace/reach limit); moves faster than ~0.5s undershoot (closed-loop
bandwidth limit at the tuned `kp_x`, not saturation).

## Done

**2026-07-03 consolidation:**
- CoppeliaSim lane archived to `archive/coppelia/` (code, Lua, launchers, ZMQ probes, WSL
  bring-up, RL PPO stack, ROS2 controller/bridge nodes, docs, configs, tests). Legacy MuJoCo
  cartpole diagnostics archived to `archive/legacy_mujoco/`.
- Full `mujoco_menagerie/` zoo checkout removed (re-clone SHA in `archive/coppelia/README.md`
  and `AGENTS.md`); the active model's meshes come from the tracked `vendor/mujoco_menagerie/`.
- Test suite consolidated: `tests/{unit,mujoco,hardware}` with pytest markers.
- Observability layer landed: `observability/run_logger.py` writes per-run `run_record.json`
  + sweep `run_log.jsonl`/`run_log.csv` from both audit and move-hold entrypoints.
- Dependency manifest fixed (unused pkgs dropped; `ur_rtde`, `pin` added).
- AGENTS.md rewritten as a structured playbook; old log at `docs/archive/AGENTS_HISTORY.md`.

**2026-07-03 Pinocchio + OSC tuning (later the same day):**
- Pinocchio dynamics provider (P0), gravity source swap (P1), Coriolis feedforward (P2) —
  all landed, flag-gated, parity-validated against MuJoCo.
- Operational-space controller upgrade (P3) — Λ(q) inertia shaping + nullspace posture
  projection, flag-gated.
- Posture re-anchoring (`controller.posture_reanchor_on_settle`) — flag-gated controller
  feature, added specifically to fix long-hold drift.
- Tuned OSC config landed and exhaustively validated (see diagnosis section above).
- `transport_metrics.py` sign-bug fix (negative-direction transport validation).
- Test suite: 116 passing (full suite, including hardware-mocked tests).

**Documentation pass (2026-07-03, this update):**
- Archived three root-level point-in-time reports (`AUDIT_REPORT.md`, `BLOAT_REPORT.md`,
  `DIAGNOSTIC_real_cartpole_torque_control_questions.md`) to `docs/archive/` with
  superseded-banners, since their recommendations are now executed.
- Archived the CoppeliaSim-era `docs/PROJECT_PLAN.md` to
  `docs/archive/PROJECT_PLAN_coppeliasim_era.md`.
- Refreshed `docs/controller_core/`, `docs/simulation/`, `docs/ros2/` subsystem docs to match
  the current (post-archival, post-Pinocchio/OSC) code layout.

## Next

Nothing is currently blocking or in-progress on the controller/tuning side. Open items, none
urgent:
- `outputs/` still holds the pre-purge `outputs/PURGE_LIST.md` DELETE-list directories plus a
  large volume of OSC-tuning-campaign sweep output generated the same day — regenerate the
  purge list to cover them, then the user needs to run the `rm -rf` themselves (blocked from
  automated execution by the environment's destructive-command guard).
- Known, deliberately-not-chased envelope edges (0.25m Z-drift limit, sub-0.5s move bandwidth
  limit) are documented, not bugs — no action needed unless a real use case needs them.
- Gain retuning for `y`/`z`/reanchor-tolerance dimensions was evaluated and found to have no
  meaningful headroom to improve (150-4000x safety margin already) — don't re-open without a
  concrete reason.

## Historical status

The CoppeliaSim bring-up era (headless video smoke, RPC/ZMQ controller, ROS2 bridge probes,
RL Y-transport) is documented in `docs/archive/AGENTS_HISTORY.md`,
`docs/archive/PROJECT_PLAN_coppeliasim_era.md`, and `archive/coppelia/docs/`. The hardware
lane status is unchanged: receive-only staging with multi-layer guardrails; no nonzero torque
path enabled.
