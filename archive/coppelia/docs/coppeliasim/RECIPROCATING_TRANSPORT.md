# CoppeliaSim reciprocating EE transport

Torque-controlled back-and-forth motion along world **X**: origin → `+stroke` →
`-stroke` → origin, using acceleration-limited segments.

## Quick start (cluster container)

```bash
cd /common/users/ss5772/real_Cartpole
bash simulation/launch_coppeliasim_reciprocating_transport_container.sh
```

Summary JSON is written under `outputs/control_runs/<RUN_SUFFIX>_coppeliasim_x_axis_headless_state/`.

## Motion profile

`--accel-profile reciprocating` plans six phases:

1. Move to `+stroke`
2. Hold
3. Move to `-stroke`
4. Hold
5. Return to origin
6. Hold

Implementation: `simulation/coppelia_reciprocating_transport.py`.

Run duration auto-extends when `--duration 0` (default) so the full cycle fits:
`settle + motion + 0.5 s`.

## Recommended defaults

| Knob | Default | Notes |
|------|---------|-------|
| `ACCEL_TORQUE_POLICY` | `ik_joint_pd` | Better X tracking than raw Cartesian at current gains |
| `RECIPROCATING_STROKE_M` | `0.018` | Half-stroke (18 mm each side) |
| `A_X_MAX` / `V_X_MAX` | `0.03` / `0.018` | Raise gradually after stable runs |
| `SETTLE_DURATION` | `2.0` | Hold at origin before motion |
| Config | `controller_coppelia_reciprocating.yaml` | `after_stable` torque caps, 65% headroom |

Example faster sweep (only after a clean run at defaults):

```bash
RECIPROCATING_STROKE_M=0.025 A_X_MAX=0.05 V_X_MAX=0.03 \
  ACCEL_TORQUE_POLICY=ik_joint_pd \
  bash simulation/launch_coppeliasim_reciprocating_transport_container.sh
```

Cartesian impedance (`ACCEL_TORQUE_POLICY=cartesian_impedance`) is supported; the
runner applies a 0.30 gain scale and position-only tracking (no velocity
feedforward) for stability. Prefer IK for larger strokes.

## Direct runner (CoppeliaSim already up)

```bash
python simulation/run_coppeliasim_x_axis_headless.py \
  --config ros2_ws/src/ur5_x_axis_controller_ros/config/controller_coppelia_reciprocating.yaml \
  --accel-x-transport \
  --accel-profile reciprocating \
  --accel-torque-policy ik_joint_pd \
  --reciprocating-stroke-m 0.018 \
  --a-x-max 0.03 --v-x-max 0.018 \
  --settle-duration 2.0 --duration 0 --no-video
```

## Tuning

- **Stroke / speed**: increase `RECIPROCATING_STROKE_M`, `A_X_MAX`, `V_X_MAX` in
  small steps. Watch `tau_saturation_fraction` and `safety_stop_reason` in the
  summary JSON.
- **Orthogonal drift**: X motion couples into Y/Z at the Coppelia transport
  pose. Relax `safety.max_abs_*_drift_m` in the YAML only for debugging; tighten
  again once tracking improves.
- **Gains**: edit `controller_coppelia_reciprocating.yaml` or pass
  `--impedance-gain-scale` (Cartesian path).
- **Start pose**: `--hold-transport-start-pose` seeds the Coppelia-derived `q`
  (override with env `Q_START_RAD`).

## Should we switch to MPC?

**Not yet.** For the current goal — reciprocating EE motion along X under torque limits
in CoppeliaSim — MPC is heavy engineering for a problem that is still primarily
**task-priority / constraint tuning**, not missing preview horizon.

| Approach | Fit now | Why |
|----------|---------|-----|
| **IK + joint PD** (default) | Best | Weighted IK already trades X motion vs Y/Z/orientation; cheap at 100 Hz |
| **Cartesian impedance** | Small moves | Simple, diagnostic-friendly; couples badly at larger stroke |
| **LQR outer loop** (existing) | Defer | Inner torque scaling was crushing commands; needs retuning before adding MPC |
| **MPC** | Later | Useful when you need hard torque/joint constraints *and* higher speed, or when the **cart-pole** dynamics enter the loop |

**Cheaper upgrades before MPC** (what we are doing on this branch):

1. Sync IK velocity state to the planned reciprocating profile (avoid drift at reversals).
2. Raise orthogonal/orientation weights during reciprocating.
3. Slew-limit Cartesian position references at direction changes.
4. MuJoCo gravity bias on the torque command.

**When MPC *would* make sense:**

- Full cycle is stable but you want **faster** strokes without orthogonal drift.
- You add **pole balance** or other coupled states — then preview + constraints pay off.
- You want one solver to enforce torque/joint limits instead of ad-hoc headroom backtracking.

A practical middle step is a **per-step constrained QP** on the wrench/torque (hierarchical
task stack), not full horizon MPC. That is ~10% of MPC complexity and matches the existing
`J.T` + backtracking architecture.

