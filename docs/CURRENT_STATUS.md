# Current Status

Last updated: 2026-07-03 (post-consolidation: CoppeliaSim archived, MuJoCo-first layout).

## Active Objective

**MuJoCo UR5e gravity-compensated residual-torque transport** in simulation, moving toward a
Pinocchio model-based controller upgrade. Active lane:

```text
tools/ur5e_mujoco_torque_experiments.py    # rollout engine (owns the step loop)
tools/audit_ur5e_mujoco_gravity_torque.py  # gravity-sign / hold-quality audit
tools/ur5e_move_hold_transport.py          # move+hold sweep driver
tools/ur5e_x_frame_envelope.py             # transport envelope sweep
tools/tune_ur5e_residual_impedance_transport.py
tools/compare_ur5e_mujoco_controllers.py
simulation/ur5e_mujoco_torque.py           # MuJoCo adapter
```

Residual torque means `tau_applied = tau_controller + tau_gravity` with `tau_gravity`
currently MuJoCo `qfrc_bias`. Raw mode is validation/anti-cheating only.

## Leading diagnosis (2026-07-01, unchanged)

See `../DIAGNOSTIC_real_cartpole_torque_control_questions.md`: the impedance controller is
structurally too simple — no mass matrix, no Coriolis compensation, no inverse dynamics —
combined with tight safety envelopes (`|qd| > 1.5 rad/s`, drift/axis-error guards) that trip
before it settles. Gravity-sign, unit, and joint-order bugs were ruled out. This diagnosis is
what motivates the Pinocchio dynamics-provider upgrade (staged: parity check → gravity source
swap → Coriolis feedforward → operational-space, each flag-gated, old behavior default).

## Done (2026-07-03 consolidation)

- CoppeliaSim lane archived to `archive/coppelia/` (code, Lua, launchers, ZMQ probes, WSL
  bring-up, RL PPO stack, ROS2 controller/bridge nodes, docs, configs, tests). Legacy MuJoCo
  cartpole diagnostics archived to `archive/legacy_mujoco/`.
- Full `mujoco_menagerie/` zoo checkout removed (re-clone SHA in `archive/coppelia/README.md`);
  the active model's meshes come from the tracked `vendor/mujoco_menagerie/`.
- Test suite consolidated: `tests/{unit,mujoco,hardware}` with pytest markers; 94 passing.
- Observability layer landed: `observability/run_logger.py` writes per-run `run_record.json`
  + sweep `run_log.jsonl`/`run_log.csv` from both audit and move-hold entrypoints. Validated
  on live sweeps including a genuine orientation-guard failure record.
- Dependency manifest fixed (unused pkgs dropped; `ur_rtde`, `pin` added).
- AGENTS.md rewritten as a structured playbook; old log at `docs/archive/AGENTS_HISTORY.md`.

## Next

1. Pinocchio P0: install `pin`, build the model from `assets/ur5e_torque/ur5e_torque.xml`,
   MuJoCo-vs-Pinocchio dynamics parity test.
2. P1: flag-gated gravity source swap (`dynamics: gravity_mode: pinocchio`).
3. P2: Coriolis feedforward. P3 (gated): operational-space inertia shaping + nullspace posture.

## Historical status

The CoppeliaSim bring-up era (headless video smoke, RPC/ZMQ controller, ROS2 bridge probes,
RL Y-transport) is documented in `docs/archive/AGENTS_HISTORY.md` and
`archive/coppelia/docs/`. The hardware lane status is unchanged: receive-only staging with
multi-layer guardrails; no nonzero torque path enabled.
