# Torque-Lane Controllers for the UR5e: Architecture and Configuration Reference

## Scope

This document describes the torque-level control stack used in the MuJoCo
true-torque simulation lane of this repository, together with its complete
configuration surface. It is intended as an orientation document for a reader
who needs to understand what controllers exist, how they are formulated, and
which parameters govern their behaviour.

Field enumerations are generated directly from the configuration dataclasses
and are complete as of 2026-08-18: 92 base parameters, 5 added by the
torque-QP layer, 23 added by the corridor-QP layer, and 11 safety parameters,
totalling 131.

---

## 1. System overview

The plant is a 6-DOF UR5e modelled with per-body inertials and direct torque
actuators (`assets/ur5e_torque/scene.xml`). Controllers are simulator-independent
and live in `controller_core/` (NumPy only); the MuJoCo adapter in
`simulation/ur5e_mujoco_torque.py` supplies state and applies torques.

Where a pendulum is attached, control is organised as a cascade:

```
high-level law  ──►  desired cart acceleration  ──►  low-level controller  ──►  joint torques
```

The high-level law observes only the pendulum; the low-level controller observes
only the arm. The interface between them is the commanded acceleration of the
end-effector along a chosen task axis, integrated into a position and velocity
reference. Switching high-level laws is therefore a change of signal source
rather than a change of controller.

### 1.1 State contract

Controllers consume a normalised state dictionary (`controller_core/state_types.py`)
containing joint positions and velocities, end-effector pose and velocity, the
Jacobian, the mass matrix, gravity/bias torques, the reference targets
(`target_x`, `target_x_vel`, `target_x_accel`), and the control interval `dt_s`.

---

## 2. Controller families

The active controller is selected by `controller_kind`, supplied at dispatch
(CLI `--controller-kind`) or by `mujoco.default_controller`. It is independent
of the configuration file: a configuration does not determine which controller
consumes it.

| `controller_kind` | Formulation | Characteristics |
|---|---|---|
| `impedance` | Cartesian impedance control (operational-space). A task-space PD wrench is mapped to joint torques by `τ = Jᵀw`, with joint damping, a posture term, and externally supplied gravity compensation. | Full 6-DOF task. No mechanism for excluding joints from the task and no constraint layer. |
| `torque_qp` | The same task expressed as a quadratic program with velocity and torque bounds. | Bounds are respected exactly rather than by clipping. |
| `hard_constraint_qp` | QP formulation with the task imposed as a hard constraint. | Used where task satisfaction must not be traded against secondary objectives. |
| `x_task_yz_corridor_qp` | Reduced task (a subset of Jacobian rows tracked) solved as a QP, with the remaining directions governed by control barrier functions rather than tracked. | Supports joint exclusion, bounded rather than tracked directions, and barrier constraints on orientation, manipulability, and joint motion. |
| `zero_torque` | Commands zero controller torque; gravity compensation only. | Baseline and counterfactual reference. |

### 2.1 Impedance formulation

The task wrench is formed from Cartesian position and orientation error,

```
w = [ Kp,trans (p_des − p) + Kd,trans (ṗ_des − ṗ) ;
      Kp,rot  e_ori        + Kd,rot  (−ω)          ]
```

and mapped to torques by `τ_task = Jᵀ w`. To this are added joint-space damping
`−Kd,joint q̇`, a posture spring toward a captured rest configuration
`Kp,posture (q_rest − q) − Kd,posture q̇`, gravity compensation, and optional
friction feedforward. A geometric backtracking step and a hard clip enforce
torque limits.

### 2.2 Reduced-task corridor QP

The corridor QP distinguishes three roles for Cartesian directions:

- **Tracked** (`task_axis_rows`): a reference is followed.
- **Bounded** (`corridor_axis_rows`): no reference; the direction is constrained
  to remain within a corridor by a barrier.
- **Free**: neither tracked nor bounded.

Joints listed in `task_excluded_joints` receive no task torque, retaining only
the non-task bias (gravity, posture, damping). This permits a task to be
executed by a subset of the kinematic chain.

---

## 3. High-level laws and gain determination

The controllers of §2 accept a commanded task acceleration. Where a pendulum is
present, that command is produced by a high-level law. Two are used.

### 3.1 Energy shaping (swing-up)

Away from the inverted equilibrium the objective is to add mechanical energy.
For a pivot accelerating along a drive axis, the hinge receives a generalised
torque `Q = c₀ cos(φ) a`, where `c₀` is the coupling of that axis to the hinge
and `φ` is measured from the hanging equilibrium. The energy rate is therefore

```
Ė = c₀ cos(φ) · a · θ̇
```

Any drive of the form `a = k · sign(c₀ cos(φ) θ̇)` gives `Ė ≥ 0`, so energy
increases monotonically — a Lyapunov construction rather than a heuristic. The
classical Åström–Furuta law takes `a = −k_e θ̇ cos(φ)(E_top − E)`, whose
`(E_top − E)` factor causes the drive to vanish as the pendulum approaches the
top. A recentring term is normally added, since energy shaping alone gives the
cart no reason to remain near its starting position.

`c₀` depends on pose, pendulum asset, and drive axis, and is measured from the
compiled model rather than assumed; its sign determines the direction of the
drive.

### 3.2 Linear-quadratic regulation (balance)

Near the inverted equilibrium the objective is stabilisation. The pendulum is
described by four states — cart position error, cart velocity, pendulum angle
from inverted, and angular rate — with the commanded cart acceleration as the
single input:

```
x = [ x − x_ref, ẋ, φ, φ̇ ]ᵀ,     u = ẍ_commanded
```

Linearising about `φ = 0` gives

```
       ⎡0  1   0        0      ⎤        ⎡ 0            ⎤
   A = ⎢0  0   0        0      ⎥ ,  B = ⎢ 1            ⎥
       ⎢0  0   0        1      ⎥        ⎢ 0            ⎥
       ⎣0  0  ω²  −b/I_pivot   ⎦        ⎣ Q_per_a/I    ⎦
```

where `ω² = mgr/I_pivot`. The positive sign on `ω²` is what makes the inverted
equilibrium unstable; the same `ω` governs small oscillation about the hanging
equilibrium. `Q_per_a` is the measured hinge torque per unit cart acceleration.
The second row encodes the assumption `ẍ = u`, i.e. that the low-level
controller realises the commanded acceleration quickly relative to the
pendulum's own timescale.

Given weighting matrices `Q ⪰ 0` and `R ≻ 0`, the infinite-horizon cost

```
J = ∫ (xᵀQx + uᵀRu) dt
```

is minimised by the state feedback `u = −Kx` with `K = R⁻¹BᵀP`, where `P`
solves the continuous algebraic Riccati equation

```
AᵀP + PA − PBR⁻¹BᵀP + Q = 0
```

**Determining the gain.** Solving the Riccati equation is immediate. What is not
determined by theory is the choice of `Q` and `R`, and the saturation limit
applied to `u`. The linear model omits the low-level controller's finite
bandwidth, actuator limits, joint friction, the constraint layer, and the safety
guards; a gain that is optimal for the model may therefore be unusable on the
plant. In this repository the weights are consequently obtained by search:
each candidate `(Q, R, a_max)` is converted to a gain by the Riccati solve, that
gain is evaluated in a full nonlinear closed-loop rollout with the actual
low-level controller and guards active, and the outcome scores the candidate.
Differential evolution or a Bayesian (TPE) backend drives the outer loop.

The computational cost is therefore dominated by the rollouts, not the algebra:
a search of a few hundred candidates, each a multi-second simulation at 500 Hz
with a QP solved per step, accounts for the runtime. `a_max` is included in the
search because it bounds what the low-level controller is asked to deliver; a
value that saturates its search bound indicates that the bound, rather than the
plant, is determining the result.

### 3.3 Handoff

Transfer from the swing-up law to the regulator is a change of which law writes
the acceleration command; the low-level controller and the integrated reference
are unaffected and are carried across unchanged. The switching criterion is
naturally expressed in the unstable mode of the linearised system,

```
s = θ̇ + ω φ
```

which is the coordinate that diverges; `s ≈ 0` describes a trajectory
approaching the equilibrium along its stable manifold. A criterion on `s` alone
is necessary but not sufficient, since it defines a manifold extending far from
the equilibrium, whereas the linearisation on which the regulator is based holds
only nearby. A bound on `|φ|` is therefore applied alongside it.

---

## 4. Configuration model

Configuration classes form an inheritance chain, so a corridor-QP configuration
also accepts every parameter of the layers beneath it:

```
CartesianImpedanceConfig            92 parameters
  └── TorqueTaskQPConfig            +5
        └── XTaskYZCorridorQPConfig +23
```

Configurations are YAML documents with a `controller:` section parsed by
`from_controller_yaml_section`, a `mujoco:` section describing the model and
integration settings, and an optional `provenance:` block (§9).

---

## 5. Base parameters (`CartesianImpedanceConfig`)

### 4.1 Task and joint gains

| Parameter | Default | Description |
|---|---|---|
| `kp_x`, `kd_x` | 25.0, 8.0 | Proportional and derivative gain on the primary task axis |
| `kp_y`, `kd_y` | 80.0, 15.0 | Second translational axis; interpretation depends on `y_control_mode` |
| `kp_z`, `kd_z` | 120.0, 20.0 | Third translational axis |
| `kp_rot`, `kd_rot` | 20.0, 5.0 | Orientation PD in task space |
| `kp_posture`, `kd_posture` | 2.0, 0.5 | Scalar joint-space posture spring and damper |
| `posture_kp_by_joint`, `posture_kd_by_joint` | None | Per-joint posture gains (honoured by the impedance controller) |
| `kd_joint`, `kd_joint_by_joint` | 0.8, None | Joint-space viscous damping, scalar or per joint |
| `tau_max_nm` | `[8,8,8,2.5,2.5,2.5]` | Per-joint torque limit |
| `torque_headroom` | 0.9 | Fraction of the limit the controller may command |

A gain is meaningful only for the combination of controller, task frame, pose,
row assignment, and role for which it was determined. Under task-space inertia
shaping (§5.2) gains become task-acceleration gains, related to impedance gains
by `k_shaped = k_impedance · Λ_axis`, where `Λ` is evaluated in the frame the
row occupies, at the configuration of interest, using that configuration's own
regularisation, and from the unreduced Jacobian — `Λ` being a property of the
mechanism rather than of which joints are permitted to act.

### 4.2 Task-space inertia shaping

`task_space_inertia_shaping` weights the task wrench by
`Λ(q) = (J M⁻¹ Jᵀ + εI)⁻¹`; `nullspace_posture` projects the posture term
through the dynamically consistent nullspace projector. A fully determined
6-DOF task has no nullspace, so the posture term is nulled except near
singularities.

Associated parameters: `lambda_regularization` (1e-6), `lambda_diagonal_shaping`,
`lambda_adaptive_regularization`, `lambda_regularization_far` (1e-4),
`lambda_cond_low` (1e4), `lambda_cond_high` (1e8),
`wrench_lambda_adaptive_regularization`, `wrench_lambda_regularization_far` (0.01),
`nullspace_inertia_adaptive_regularization`, `nullspace_inertia_eps_ratio` (0.05).

Diagonal shaping suppresses off-diagonal coupling in the wrench-shaping step
only. Adaptive regularisation schedules ε in `log cond(J)` between a far-field
value and a near-singularity ceiling; the scheduling applies to the nullspace
projector and, separately, optionally to the wrench.

### 4.3 Singularity treatment

| Parameter | Default | Description |
|---|---|---|
| `jacobian_singular_cond_max` | 1e5 | Condition number above which task authority is scaled down; a large value (1e18) disables the term |
| `svd_singularity_filtering`, `svd_sigma_threshold`, `svd_lambda_max` | off, 0.05, 0.316 | Damped-least-squares filtering of small singular values |
| `manipulability_cbf`, `_epsilon`, `_alpha1`, `_alpha2`, `_fd_step`, `_curvature_step` | off, 1e-3, 10, 10, 1e-5, 1e-4 | Barrier maintaining the Yoshikawa manipulability measure above a floor |
| `task_resample_factor`, `task_resample_min_scale`, `task_resample_max_iters` | 0.5, 6.1e-5, 14 | Geometric backtracking of the task wrench under torque saturation |

### 4.4 Friction compensation

Model-based cancellation added to the same joint-space bias as gravity:

- `friction_feedforward` enables the term; `friction_model` selects
  `static`, `lugre`, or `karnopp`.
- Static model: `friction_ff_coulomb_nm`, `friction_ff_viscous`,
  `friction_ff_qd_deadband` (0.05), applied as
  `τ = c·tanh(q̇/δ) + v·q̇`.
- LuGre: `lugre_sigma0_nm_per_rad`, `lugre_sigma1_nm_s_per_rad`,
  `lugre_sigma2_nm_s_per_rad`, `lugre_fc_nm`, `lugre_fs_nm`, `lugre_vs_radps`,
  a bristle-deflection state with a Stribeck curve.
- Karnopp: `karnopp_qd_stick_enter_radps`, `karnopp_qd_stick_exit_radps`,
  an explicit stick–slip band.

The deadband parameter interacts with closed-loop stability: an excessively
small value places hold-phase velocities in the steep region of the `tanh`.

### 4.5 Task specification and reduction

| Group | Parameters |
|---|---|
| Dimension selection | `reduced_task_dims`, `task_dim_x/y/z/rx/ry/rz` |
| Joint locking | `task_lock_shoulder_pan/shoulder_lift/elbow/wrist_1/wrist_2/wrist_3` |
| Base/wrist split | `split_base_wrist_task`, `split_base_wrist_active_joints`, `split_base_wrist_task_dims` |
| Axis assignment | `transport_axis_index` (0/1/2 = world X/Y/Z), `second_task_axis_enabled` |

### 4.6 Auxiliary terms

Integral action: `x_integral_action`, `ki_x`, `x_integral_limit_m_s`;
`y_integral_action`, `ki_y`, `y_integral_limit_m_s` — both with anti-windup limits.

Corridor-style Y handling within the impedance controller: `y_control_mode`
(`tight` or `corridor`), `y_soft_limit_m` (0.015), `y_hard_limit_m` (0.05),
`y_corridor_kp` (80), `y_corridor_kd` (15).

Others: `y_coupling_feedforward` and `y_coupling_gain`;
`acceleration_feedforward`; `posture_reanchor_on_settle` with
`reanchor_x_tol_m` and `reanchor_qd_tol_radps`, which recaptures the posture
reference once motion has settled; `wrist_orientation_task` with `kp_rot_wrist`
and `kd_rot_wrist`, a wrist-only joint-space orientation term structurally
separate from the `Λ`-weighted wrench pipeline.

---

## 6. Torque-QP layer (`TorqueTaskQPConfig`)

`max_joint_velocity_radps` (2.5), `posture_regularization` (0.35),
`enforce_velocity_torque_bounds`, `velocity_torque_coupling_kp`,
`velocity_torque_coupling_kd`.

These introduce joint-velocity limits into the optimisation and a regularisation
weight on the posture objective.

---

## 7. Corridor-QP layer (`XTaskYZCorridorQPConfig`)

### 6.1 Task structure

| Parameter | Default | Description |
|---|---|---|
| `task_excluded_joints` | `(0,)` | Joints receiving no task torque |
| `task_axis_rows` | `(0,)` | Jacobian rows tracked against a reference |
| `corridor_axis_rows` | `(1,2)` | Rows bounded by a barrier instead of tracked |
| `task_frame`, `task_rotation`, `task_frame_update` | `world`, None, `frozen` | Frame in which rows are interpreted |
| `posture_joint_weights` | None | Per-joint multiplier on the posture spring and damper |

Excluding joints reduces the dimension of the space available to the task. Where
the number of remaining joints equals the number of tracked rows the system is
exactly determined, and rank deficiency in that submatrix renders a task
direction unreachable; this is a property of the configuration and pose and is
worth evaluating before use.

### 6.2 Constraint layer

All barriers are high-order control barrier functions constructed by a single
routine, which accepts the Jacobian row mapping joint velocity to the rate of
the constrained quantity. For a Cartesian axis this is the corresponding
Jacobian row; for joint *j* it is the unit vector `e_j`.

| Constraint | Parameters | Defaults |
|---|---|---|
| Cartesian corridor | `yz_corridor_enabled`, `y_corridor_half_width_m`, `z_corridor_half_width_m`, `yz_corridor_alpha1`, `yz_corridor_alpha2` | off, 0.05, 0.05, 10, 10 |
| Orientation | `orientation_cbf`, `orientation_cbf_max_error_rad`, `orientation_cbf_alpha1`, `orientation_cbf_alpha2` | off, 0.20, 10, 10 |
| Joint motion | `joint_corridor_enabled`, `joint_corridor_joints`, `joint_corridor_half_width_rad`, `joint_corridor_alpha1`, `joint_corridor_alpha2` | off, (), 0.05, 20, 20 |
| Solver | `dual_sweeps`, `dual_root_iters` | 4, 10 |

The `alpha` pairs are the two gains of the second-order barrier; larger values
permit approach to the boundary at higher rates.

### 6.3 Restraint mechanisms compared

Three distinct mechanisms restrict joint motion, with different guarantees:

| Mechanism | Nature | Guarantee |
|---|---|---|
| Posture gain (scalar or per-joint) | Elastic restoring term | None; deviation is bounded only by the balance of torques |
| `task_excluded_joints` | Removal from the task objective | The joint receives no task torque, but is not otherwise constrained |
| `joint_corridor_*` | Inequality constraint in the QP | Torques violating the bound are infeasible |

A hard constraint may render the QP infeasible where it conflicts with the task;
this is inherent to the formulation.

---

## 8. Safety monitoring

`ImpedanceSafetyMonitor` (`controller_core/safety.py`) evaluates termination
conditions each control cycle and is the source of `termination_reason` in
recorded traces.

| Parameter | Default | Condition |
|---|---|---|
| `max_abs_y_drift_m`, `max_abs_z_drift_m` | 0.03, 0.03 | World-frame drift from the captured start position |
| `max_abs_orthogonal_drift_m` | 0.03 | Task-frame drift, per axis |
| `max_orientation_error_rad` | 0.25 | Norm of the orientation error |
| `max_joint_velocity_radps` | 1.5 | Peak joint speed |
| `max_x_error_growth_steps`, `max_axis_error_growth_steps` | 100, 100 | Consecutive steps of growing tracking error |
| `emergency_stop_on_nan`, `emergency_stop_on_joint_limit` | True, True | Latching conditions |
| `q_lower`, `q_upper` | required | Joint limits |

The monitor operates in one of two frames. When a task rotation is supplied,
drift is resolved in the task frame and each axis other than the commanded one
is compared against `max_abs_orthogonal_drift_m`; otherwise world-frame Y and Z
are compared against their respective limits. The commanded axis is exempt in
both cases, so displacement along the direction of intended motion is not
itself a termination condition. `ImpedanceSafetyMonitor.drift_vector()` returns
the quantities the monitor compares, in the frame it uses.

---

## 9. Configuration provenance

`controller_core/config_provenance.py` provides a machine-checked association
between a configuration and the conditions under which its parameters were
determined. A configuration may declare:

```yaml
provenance:
  derived_for:
    arm_q_rad: [...]                          # the six joint angles
    pendulum_xml: pendulum_attachment_realrod.xml
    controller_kind: "x_task_yz_corridor_qp"
  notes: "..."                                # how the values were obtained
```

Dispatching such a configuration at a different pose, asset, or controller
raises `ConfigPoseMismatchError`. The positional tolerance is 1e-6 rad,
expressing identity rather than proximity, since no neighbourhood of validity
has been characterised. Configurations without the block are permitted and
reported as undeclared; the environment variable
`REAL_CARTPOLE_STRICT_PROVENANCE` promotes undeclared to an error.

The check is applied by the shared pendulum dispatch helper and by the
entrypoints that construct their own context. Parameters transported through
result artefacts — notably gains passed via `--lqr-json` — are not currently
covered.

---

## 10. Model and integration settings

The `mujoco:` section of a configuration specifies `scene_xml`, `joint_order`,
`actuator_order`, `site_name`, `home_qpos`, `use_home_qpos_as_start`,
`gravity_mode`, `gravity_source` (`mujoco_qfrc` or `pinocchio`),
`coriolis_feedforward`, `default_controller`, and `output_dir`.

Gravity and bias torques may be sourced either from MuJoCo's own `qfrc_bias` or
from a Pinocchio model of the same MJCF; the two agree to within 1e-8 N·m for
gravity and 1e-6 for the full bias.

---

## 11. Parameter determination: black-box optimisation and reinforcement learning

Two distinct classes of problem arise, and they are addressed by different
machinery. Confusing them is the principal risk in this area.

### 11.1 Static parameter selection — black-box optimisation

Most quantities to be determined are a handful of **static continuous
parameters** evaluated by a **deterministic rollout** returning a **scalar
score**: swing-up law coefficients, LQR cost weights, saturation limits. There is
no state-dependent decision to make; the parameters are chosen once and held.
This is black-box optimisation.

`tools/diagnostics/pendulum_search_backends.py` provides a single
`minimize(objective, bounds, backend=..., maxiter, popsize, seed, workers,
n_trials)` returning a SciPy-shaped result, so a caller selects an optimiser by
name. Both backends receive identical objective, bounds and seed, making results
directly comparable.

**Differential evolution** (`backend="de"`, `scipy.optimize.differential_evolution`).
A population-based evolutionary method: `popsize × n_params` individuals evolve
over `maxiter` generations by mutation and crossover. It requires no gradient and
tolerates discontinuous objectives, but its evaluation count grows quickly with
dimension, and much of the budget is spent confirming a basin located early.

**Tree-structured Parzen Estimator** (`backend="optuna"`). A Bayesian method:
observed trials are used to model the densities of good and poor parameter
regions, and new trials are drawn where improvement is most probable. On smooth,
low-dimensional spaces this typically locates a comparable optimum in
substantially fewer evaluations.

*Parallelism.* Optuna's `study.optimize(n_jobs=…)` is thread-based, and these
objectives are Python-level rollout loops holding the interpreter lock, so
thread parallelism yields little speed-up. The backend therefore drives Optuna
through its ask/tell interface over a process pool: a batch of trials is
requested, evaluated in parallel processes, and reported back, which preserves
the sequential model update between batches while obtaining real parallelism
within them. Differential evolution obtains process parallelism through SciPy's
own `workers` argument.

*Callers.* `pendulum_swingup_energy_shaping.py`, `pendulum_swingup_curved.py`
and `pendulum_two_phase_swingup.py` expose `--search-backend`. Several further
tools, including `pendulum_lqr_cascade.py`, invoke `differential_evolution`
directly.

*A caution recorded in the module itself.* A faster optimiser searches a
defective objective faster. Where a result appears insensitive to a parameter,
a one-variable sweep is more informative than a larger search: a flat response
is evidence about the objective, not about the optimum.

### 11.2 State-dependent policies — reinforcement learning

Reinforcement learning is appropriate where the decision depends on the state
and unfolds over time — where what to do now is a function of what is currently
observed. Selecting a fixed parameter vector is not such a problem; it is a
bandit, and an optimiser addresses it more directly.

Two packages exist and correspond to the two sides of this distinction.

`rl_gain_scheduling/` treats the controller's gain fields as the action
(`ACTION_DIM = len(GAIN_FIELDS)`, `OBS_DIM = 47`) and contains no pendulum. It
therefore applies reinforcement learning to a parameter-selection problem.

`pendulum_swingup_rl/` defines a Gymnasium environment
(`OBS_DIM = 8`, one continuous action in `[-1, 1]`) in which the action is the
commanded pivot acceleration at each decision step, and the observation is the
pendulum and cart state together with the measured energy fraction. Here the
decision is genuinely state-dependent and sequential.

*Design considerations recorded in that environment.* The action is the drive
itself rather than a multiplier on a phase-locked term, because such a term
vanishes at the rest state and a policy scaling it could not leave that state.
The joint-space leash is part of the environment rather than the policy's
responsibility, so the policy addresses the pumping decision rather than
rediscovering drift regulation. Decision rate is decimated relative to the
500 Hz inner loop so that the horizon is commensurate with the plant's own
dynamics.

*Interpretability.* An analytic energy-shaping law satisfies `Ė ≥ 0` by
construction; a learned policy carries no such guarantee, and a policy removing
energy is indistinguishable from an untrained one by return alone. Every episode
therefore reports `positive_work_fraction`, the fraction of steps in which the
drive performs positive work on the hinge, which is approximately unity for the
analytic law. This provides a check on policy behaviour independent of the
reward.

---

## Appendix: parameter counts by layer

| Layer | Parameters | Defined in |
|---|---|---|
| `CartesianImpedanceConfig` | 92 | `controller_core/x_axis_cartesian_impedance/config.py` |
| `TorqueTaskQPConfig` | +5 | `controller_core/torque_task_qp.py` |
| `XTaskYZCorridorQPConfig` | +23 | `controller_core/x_task_yz_corridor_qp/config.py` |
| `ImpedanceSafetyConfig` | 11 | `controller_core/safety.py` |

### Note on overlapping mechanisms

Per-joint posture weighting is expressed by two distinct parameters:
`posture_kp_by_joint` / `posture_kd_by_joint` in the base configuration, read by
the impedance controller, and `posture_joint_weights` in the corridor-QP
configuration, read by that controller. The two are not interchangeable, and a
reader configuring per-joint posture behaviour should confirm which applies to
the controller in use.
