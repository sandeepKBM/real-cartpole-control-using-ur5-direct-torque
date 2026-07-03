# controller_core

Simulator-independent control for the MuJoCo UR5e true-torque lane (numpy only — this
package must stay free of any simulator-specific dependency; Pinocchio is confined to
`model_dynamics.py` as a lazy optional import).

## Main entry point (Cartesian impedance)

| Module | Purpose |
|--------|---------|
| `x_axis_cartesian_impedance.py` | Full 6D `J^T wrench` + posture + damping + optional gravity; Y/Z/orientation hold. Also holds the flag-gated P3 operational-space upgrade (task-space inertia shaping, nullspace posture projection, posture re-anchoring on settle). |
| `model_dynamics.py` | `DynamicsProvider` protocol + `PinocchioUR5eDynamics` — gravity, Coriolis, mass-matrix, and bias terms computed from the same MJCF the sim uses. Parity-validated against MuJoCo (gravity <1e-8 Nm, bias <1e-6 Nm, mass matrix <1e-8). |
| `filters.py` | Torque low-pass + per-joint rate limit. |
| `safety.py` | Drift, orientation, velocity, x-error growth, NaN, joint limits. |
| `state_types.py` | `as_impedance_robot_state` (requires `jacobian` 6×6, EE velocities; optionally `mass_matrix` 6×6 for the P3 terms). |
| `logging_utils.py` | JSON helpers / JSONL writer. |

Legacy X-only PD + `safety_utils.py` remain for older MuJoCo torque demos; treat as
effectively dead unless you find a live caller.

See `AGENTS.md` §3 (root of the repo) for the current controller architecture summary,
including the tuned OSC gain set and the mechanism behind it.

## Tests

```bash
python -m pytest -q -m unit                      # pure-numpy controller_core tests
python -m pytest -q tests/unit/test_impedance_dynamics.py   # P3 operational-space terms
python -m pytest -q -m mujoco tests/mujoco/test_pinocchio_parity.py       # Pinocchio parity
python -m pytest -q -m mujoco tests/mujoco/test_coriolis_feedforward.py  # P2 Coriolis
```

(`controller_core/tests/` itself no longer holds the test files — they moved to
`tests/unit/` during the 2026-07-03 test-tree consolidation.)
