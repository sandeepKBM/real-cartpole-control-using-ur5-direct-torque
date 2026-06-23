# Simulation Scripts

The active CoppeliaSim scripts live in `simulation/`.

## Known-Good Render Smoke

```bash
bash simulation/launch_coppeliasim_video_smoke.sh
```

This launches CoppeliaSim, loads the UR5 model, captures upright PNG frames, and writes:

```text
demonstration_videos/ur5e_coppeliasim/coppeliasim_ur5_video_smoke.mp4
```

Use this first when display/rendering behavior is uncertain.

## Active Controller/RPC Runner

```bash
bash simulation/launch_coppeliasim_x_axis_headless.sh
```

This launches CoppeliaSim and runs:

```text
simulation/run_coppeliasim_x_axis_headless.py
```

The controller/RPC path is still in progress. Its current weakness is startup/bootstrap, not the existence of the controller code.

## Lab Workspace Guardrails

The simulation stack now includes a diagnostic-only workspace guardrail model
extracted from the external MuJoCo visualization repo. It is used for offline
trajectory checking and optional video overlays, not for real robot safety.

```bash
python3 tools/check_trajectory_guardrails.py \
  --log outputs/control_runs/your_log.json \
  --guardrail-config config/lab_workspace_guardrails.yaml \
  --output logs/guardrail_report.json
```

Optional overlay rendering:

```bash
python3 tools/render_guardrail_overlay.py \
  --log outputs/control_runs/your_log.json \
  --guardrail-config config/lab_workspace_guardrails.yaml \
  --output logs/guardrail_overlay.png
```

The current Coppelia video runners also accept optional `--draw-guardrails`,
`--guardrail-config`, `--guardrail-margin-m`, and `--show-boundary-labels`
flags to draw the same guardrail inset on rendered frames.

## Important CoppeliaSim Files

| File | Role |
| --- | --- |
| `launch_coppeliasim_video_smoke.sh` | Known-good render-only smoke launcher. |
| `launch_coppeliasim_x_axis_headless.sh` | Active controller/RPC launcher. |
| `run_coppeliasim_x_axis_headless.py` | Direct Python runner for controller/RPC testing. |
| `ur5_video_smoke_addon.lua` | Auto-loaded Lua add-on, currently shared by smoke and RPC bootstrap. |
| `plot_coppeliasim_trace.py` | Trace plotting helper. |
