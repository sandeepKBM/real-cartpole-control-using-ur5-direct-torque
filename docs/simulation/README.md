# Simulation Scripts

`simulation/` now contains exactly two files: `ur5e_mujoco_torque.py` (the active MuJoCo
adapter) and `workspace_guardrails.py` (diagnostic-only, see below). Everything CoppeliaSim
(launchers, RPC runner, Lua add-ons, trace plotting) is archived to `archive/coppelia/` and
is not runnable in place — see `archive/coppelia/README.md` for resurrect notes.

## UR5e MuJoCo Torque Experiments (the active lane)

```text
tools/ur5e_mujoco_torque_experiments.py   # single-run rollout engine (owns the step loop)
tools/audit_ur5e_mujoco_gravity_torque.py # gravity-sign / hold-quality audit
tools/ur5e_move_hold_transport.py         # move+hold sweep driver
tools/ur5e_x_frame_envelope.py            # X-frame transport envelope sweep
simulation/ur5e_mujoco_torque.py          # MuJoCo adapter
assets/ur5e_torque/scene.xml              # the custom torque-actuated UR5e model
```

Full walkthrough: [`docs/simulation/ur5e_mujoco_torque_control.md`](ur5e_mujoco_torque_control.md).
Controller architecture (impedance law, Pinocchio dynamics, tuned OSC gains): `AGENTS.md` §3.

Verified short commands:
```bash
python tools/audit_ur5e_mujoco_gravity_torque.py --poses active_origin --durations 1.0 2.0 --seed 0 --no-plot
python tools/ur5e_move_hold_transport.py --target-x-deltas 0.01 0.02 --move-durations 1.0 --hold-durations 1.0 2.0 --torque-limit-scales 1.0 --seed 0 --no-plot
```

## Lab Workspace Guardrails

`simulation/workspace_guardrails.py` is a diagnostic-only workspace guardrail model, used for
offline trajectory checking and optional overlays — **never wire it into real-arm e-stop
logic** (see `AGENTS.md` §2).

```bash
python3 tools/diagnostics/check_trajectory_guardrails.py \
  --log outputs/ur5e_mujoco_torque_transport/some_run/trace.jsonl \
  --guardrail-config config/lab_workspace_guardrails.yaml \
  --output logs/guardrail_report.json
```

Optional overlay rendering:

```bash
python3 tools/diagnostics/render_guardrail_overlay.py \
  --log outputs/ur5e_mujoco_torque_transport/some_run/trace.jsonl \
  --guardrail-config config/lab_workspace_guardrails.yaml \
  --output logs/guardrail_overlay.png
```

(Both tools moved from bare `tools/` to `tools/diagnostics/` during the 2026-07-03
consolidation.)

## Archived: CoppeliaSim scripts

The CoppeliaSim lane (render smoke, controller/RPC runner, ZMQ probes, WSL bring-up) is
fully archived to `archive/coppelia/simulation/` — not runnable in place. See
`archive/coppelia/README.md` for the full file inventory and resurrect notes (dependencies to
reinstall, the `mujoco_menagerie` re-clone SHA if needed for the legacy scene files).
