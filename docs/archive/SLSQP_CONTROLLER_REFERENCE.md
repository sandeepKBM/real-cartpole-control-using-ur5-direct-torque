# SLSQP And Controller Reference

## Purpose And Reading Guide

This file is the canonical reference for the simulation-side controllers and optimization routines in `real_Cartpole`.
It is meant to answer two recurring questions:

1. What is each controller or solver trying to do?
2. Which paths are active today versus archived, exploratory, or only kept for reference?

In this document:

- A **controller** is an online routine that runs during simulation and produces the next actuator command or joint setpoint.
- A **solver** is an offline or per-pose optimizer/search routine that finds feasible joint configurations, workspace limits, or boundary poses.
- A **runner** is a top-level script that wires scene loading, solver calls, controller calls, rendering, and JSON output into an experiment.

Scope is **simulation only**. ROS files are mentioned only when they reuse a simulation helper or act as out-of-scope scaffolding. They are not part of the active simulation controller path documented here.

One subtle but important note: not every optimization routine in this document uses SciPy SLSQP. The two origin search functions in `simulation/probe_origin_pose.py` are deterministic seed plus local-search helpers, but they are included because they still define important canonical poses and the surrounding workflow often gets discussed alongside the SLSQP tools.

## Stable Component Index

The IDs below are used throughout the rest of the document.

| ID | Component | Category | Status | Main file | Called by / used by |
| --- | --- | --- | --- | --- | --- |
| C1 | `split_forearm_origin_face_controller` | Controller | Active baseline | `simulation/controller.py` | `simulation/run_origin_stabilization.py` default |
| C2 | `differential_ik_split_controller` | Controller | Active alternate | `simulation/controller.py` | `simulation/run_origin_stabilization.py --controller differential_ik_split` |
| C3 | `differential_ik_xz_transport_controller` | Controller | Available prototype | `simulation/controller.py` | Kept for comparison; not the current default transport-runner path |
| C4 | `reverse_nested_x_servo_controller` | Controller | Archived | `simulation/legacy_reverse_nested_controller.py` | No active runner |
| S1 | `find_peak_height_origin` | Search helper | Historical reference | `simulation/probe_origin_pose.py` | `find_vertical_face_origin`, manual origin studies |
| S2 | `find_vertical_face_origin` | Search helper | Active reference pose finder | `simulation/probe_origin_pose.py` | `simulation/probe_origin_pose.py`, canonical origin analysis |
| S3 | `build_study` | SLSQP workspace study | Active analysis | `simulation/study_constrained_ee_x_workspace.py` | CLI study, reused conceptually by workspace tooling |
| S4 | `_optimize_x_at_z_one` | SLSQP boundary solve | Active analysis | `simulation/probe_workspace_xz_envelope.py` | `probe_workspace_xz_envelope.py`, `demo_ee_x_limits_at_z.py`, `run_fixed_z_x_transport.py`, exploratory studies |
| S5 | `global_max_site_z_above_floor` | SLSQP height-limit solve | Active analysis | `simulation/probe_workspace_xz_envelope.py` | `probe_workspace_xz_envelope.py`, `demo_ee_x_limits_at_z.py`, `run_fixed_z_x_transport.py`, exploratory studies |
| S6 | `solve_q_at_xz` | Continuity-aware SLSQP pose solve | Active analysis | `simulation/demo_ee_x_limits_at_z.py` | `demo_ee_x_limits_at_z.py`, `run_fixed_z_x_transport.py` |
| S7 | `solve_pose_with_restarts` | Endpoint pose selector | Active prototype | `simulation/run_fixed_z_x_transport.py` | `run_fixed_z_x_transport.py` |
| S8 | `solve_ik_fixed_xz` | Exploratory SLSQP IK multiplicity solver | Exploratory | `simulation/study_z_axis_x_freedom_singularity_report.py` | `study_z_axis_x_freedom_singularity_report.py` |
| R1 | `run_origin_stabilization.py` | Runner | Active baseline experiment | `simulation/run_origin_stabilization.py` | Origin-hold experiments |
| R2 | `demo_ee_x_limits_at_z.py` | Runner | Active analysis demo | `simulation/demo_ee_x_limits_at_z.py` | Fixed-`Z` X-sweep feasibility demo |
| R3 | `run_fixed_z_x_transport.py` | Runner | Active prototype experiment | `simulation/run_fixed_z_x_transport.py` | Continuous fixed-`Z` transport |

## Shared Concepts

### Task frames: `forearm_tip_site` versus `attachment_site`

Two different sites are used on purpose.

- `forearm_tip_site` is the **proximal origin reference**.
- `attachment_site` is the **distal tool/task frame**.

The split matters because wrist motion should be free to correct tool orientation without redefining where the project says the arm's "origin reference" lives. If the same distal site were used for both jobs, wrist motion would move the origin target and the task would become self-conflicting.

So the common pattern is:

- use `forearm_tip_site` when the task is "recover the canonical arm-origin pose"
- use `attachment_site` when the task is "hold tool orientation" or "move the end effector in world X/Z"

### MuJoCo servos versus Python controllers

The Python controllers in this repo are not direct torque controllers.
They are outer-loop routines that update `data.ctrl` for MuJoCo's built-in joint servos.

That means the active stack is:

1. Python controller computes the next desired joint command.
2. MuJoCo's actuator model applies its own low-level servo behavior.
3. Simulation integrates the resulting motion.

This distinction matters when reading the code:

- controller gain choices here shape joint setpoint motion
- they do not yet represent a final hardware torque controller
- smoothness problems can come from outer-loop task formulation even when the low-level servo is stable

### Fixed-pan assumption and the square task

The current transport task keeps `shoulder_pan_joint` fixed.
Once pan is locked, the active joints are effectively `q[1:6]`, so there are 5 free joint variables.

The fixed-`Z` transport task asks for exactly 5 constraints:

- 3 orientation constraints from the tool rotation residual
- 1 world-`X` constraint
- 1 world-`Z` constraint

That makes the online problem a **square 5-constraint / 5-joint problem**.
There is essentially no null space left for the controller to "hide" discontinuities, which is why branch continuity, conditioning, and good restarts matter so much.

### Why SLSQP is offline and controllers are online

The repo separates two jobs:

- **SLSQP/search** is used to answer feasibility questions: is there a joint vector at this pose, what is the X boundary at this Z, what is the highest Z at fixed orientation, which branch is closest to the previous pose.
- **Controllers** are used to move the simulated arm through time once a task or reference has been defined.

That split is deliberate:

- SLSQP is slow but powerful for constrained feasibility and boundary finding.
- The controllers are lightweight enough to run every simulation sub-step.

The transport work currently sits in the middle: it uses SLSQP to find good endpoints and workspace limits, then uses an online differential-IK controller to track a smooth task-space `x(t)` reference.

### Shared orientation residual

Most of the active code uses the same world-frame orientation target: `TARGET_SITE_ROTATION_WORLD`.
`tool_target_orientation_omega(...)` in `simulation/controller.py` converts the difference between the current `attachment_site` rotation and that target into a small-angle angular correction vector.

That helper is the common orientation signal for:

- C1
- C2
- C3
- C4

## Controllers

### C1 - `split_forearm_origin_face_controller`

**File:** `simulation/controller.py`

**Objective**

Recover and hold the canonical origin pose while keeping the distal tool frame pointed in the desired world orientation.

This is the current origin-hold baseline. Its task is split across the kinematic chain:

- proximal arm motion keeps `forearm_tip_site` at its target position
- distal wrist motion keeps `attachment_site` aligned to `TARGET_SITE_ROTATION_WORLD`

**Inputs and outputs**

Inputs include current joint state, target joint vector, previous control, actuator limits, joint velocity, origin position and origin Jacobian, tool rotation and rotational Jacobian, and an optional wrist posture bias.

Output is a new actuator control vector `ctrl`.

**Controlled quantities**

- `forearm_tip_site` position
- `attachment_site` orientation
- `shoulder_pan_joint` fixed to `q_target[0]`

**How it works**

The controller builds a joint increment by hand rather than solving a linear system.

It combines:

- a small posture term on joints 1 and 2
- damping on joints 1 through 5
- an origin guidance term `2.2 * J_origin^T * origin_error`
- a face-orientation guidance term `0.50 * J_rot^T * orient_omega`
- an optional wrist posture bias

Then it masks those terms so they act on different parts of the chain:

- origin guidance mostly drives shoulder lift and elbow
- face guidance mostly drives wrist joints

Finally it clips the per-step joint change and writes `ctrl[0] = q_target[0]`.

**What is fixed versus free**

- fixed: shoulder pan, target world-frame tool orientation
- effectively free: distal wrist posture except for the orientation target and optional bias

**Where it is used now**

This is the default controller in R1, `simulation/run_origin_stabilization.py`.

**Why it exists**

It is simple, readable, and good at expressing the project's original design intent: keep the forearm-origin reference stable while letting the wrist satisfy the face/orientation task.

**When not to use it**

It is not the right controller for the main fixed-`Z` transport task because that task directly constrains `attachment_site` world `X` and `Z`, not `forearm_tip_site`.

### C2 - `differential_ik_split_controller`

**File:** `simulation/controller.py`

**Objective**

Solve the same split origin-hold task as C1, but in a more explicit weighted least-squares differential-IK form.

**Inputs and outputs**

Inputs are almost the same as C1.
Output is again a new actuator control vector `ctrl`.

**Controlled quantities**

- `forearm_tip_site` position
- `attachment_site` orientation
- shoulder pan held fixed
- soft posture regularization

**How it works**

The controller works on the reduced joints `q[1:6]`.
It forms a small stacked least-squares problem with three block types:

1. origin-position block
2. tool-orientation block
3. posture-and-damping block

Conceptually it solves

```text
delta_q_red = argmin ||A_origin delta - b_origin||^2
                    + ||A_rot delta - b_rot||^2
                    + ||A_posture delta - b_posture||^2
                    + regularization
```

Then it clips the reduced increment and reinserts the locked pan joint.

Relative to C1, the main change is numerical structure:

- C1 is a masked sum of Jacobian-transpose style terms
- C2 is a weighted least-squares solve with the same task split

**Main gains and weighting idea**

- origin block uses a stronger task weight than the posture block
- orientation block is also weighted heavily, but still competes with posture and damping in one solve
- posture is intentionally soft

**What is fixed versus free**

Same high-level task partition as C1.

**Where it is used now**

R1 can switch to it with `--controller differential_ik_split`.

**Why it exists**

It is the cleaner mathematical formulation of the same origin-hold experiment and is a better base if the split origin-hold path needs to be tuned further.

**When not to use it**

It still solves the origin-hold task, not the fixed-`Z` transport task. For the current main goal, C3 is the closer match.

### C3 - `differential_ik_xz_transport_controller`

**File:** `simulation/controller.py`

**Objective**

Move `attachment_site` along world `X` while holding:

- world `Z`
- tool orientation
- shoulder pan

This is the first controller in the repo whose task definition directly matches the intended hardware motion.

**Inputs and outputs**

Inputs include current joint state, previous control, actuator limits, joint velocity, tool position, `x_target`, `z_target`, translational and rotational Jacobians, desired tool rotation, locked pan target, and an optional posture target.

Output is a new actuator control vector `ctrl`.

**Controlled quantities**

- `attachment_site` world `X`
- `attachment_site` world `Z`
- `attachment_site` world-frame orientation
- `shoulder_pan_joint`

**How it works**

Like C2, it solves a reduced 5-joint weighted least-squares step.
The task blocks are:

1. one row for world `X`
2. one row for world `Z`
3. three rows for rotation
4. a soft posture-and-damping block

Conceptually:

```text
delta_q_red = argmin ||J_x delta - k_x e_x||^2
                    + ||J_z delta - k_z e_z||^2
                    + ||J_rot delta - k_rot omega||^2
                    + ||W_posture delta - b_posture||^2
                    + regularization
```

Then it clips the reduced increment and forces `ctrl[0] = pan_target`.

**Main gains and weighting idea**

- `X` and `Z` tracking are the primary translation tasks
- orientation is also high priority
- posture is deliberately weak, because in a square problem it mostly regularizes rather than giving a real null-space objective

**What is fixed versus free**

- fixed: pan, target world orientation, target `Z`
- commanded through time: target `X`
- almost nothing is truly free because the task is square

**Where it is used now**

This controller is still present as the explicit online square-task transport formulation, but it is no longer the default path inside R3 after the planned-joint-trajectory refactor.

**Why it exists**

It is the controller that best reflects the current project goal: transport the end effector from one end of feasible `X` to the other at a chosen `Z` without changing orientation.

**When not to use it**

Do not treat it as a finished real-robot controller yet.
It remains useful as the direct online differential-IK formulation of the transport task, but the default runner now prefers precomputed joint trajectories because they are smoother and easier to bound.

### C4 - `reverse_nested_x_servo_controller`

**File:** `simulation/legacy_reverse_nested_controller.py`

**Objective**

This was an older outer-loop controller for reach-like motion with orientation correction layered in afterward.

**Inputs and outputs**

It takes current and target joint state, previous control, actuator bounds, optional site `X` target data, site rotation, and Jacobians.
It outputs a new actuator control vector.

**Controlled quantities**

- a site-`X` reaching term
- an orientation correction term
- posture and damping terms

**How it works**

The controller builds a masked Jacobian-transpose style increment.
Its notable feature is a set of priority gates that scale shoulder and elbow action based on current wrist and elbow errors, giving it a "reverse nested" feel.

It is more heuristic than the current controllers:

- `X` guidance acts mainly on joints 1 through 3
- orientation guidance is weighted more heavily toward the wrist chain
- gates change shoulder/elbow participation based on state-dependent error magnitudes

**What is fixed versus free**

Shoulder pan is still forced to the target, but the rest of the task partition does not match the current origin-hold or fixed-`Z` transport design.

**Where it is used now**

No active runner uses it.

**Why it exists**

It preserves the older design for comparison and historical reference.

**When not to use it**

Do not use it as the "current" controller. It is archived and the rest of the active codebase has moved on from its task definition and tuning assumptions.

### Behavioral comparison

| Controller | Task definition | Numerical structure | Current maturity |
| --- | --- | --- | --- |
| C1 `split_forearm_origin_face_controller` | Hold forearm-origin position and tool orientation | Hand-built masked Jacobian-transpose style update | Stable baseline for origin-hold |
| C2 `differential_ik_split_controller` | Same split task as C1 | Weighted least-squares differential IK | Active alternate / cleaner formulation |
| C3 `differential_ik_xz_transport_controller` | Hold tool orientation and `Z` while moving in `X` | Weighted least-squares differential IK on a square task | Active prototype for the main goal |
| C4 `reverse_nested_x_servo_controller` | Older reach-plus-orientation heuristic | Masked Jacobian-transpose with state-dependent gates | Archived |

## Solver And Search Routines

### Why this section mixes SLSQP and non-SLSQP routines

Most of the current pose and workspace solvers are SciPy SLSQP-based.
The exception is the origin search code in `simulation/probe_origin_pose.py`, which predates the current constrained-solver workflow and uses seeded evaluation plus coordinate-style hill climbing.

Those two routines are still documented here because they define canonical poses that the rest of the project refers to.

### S1 - `find_peak_height_origin`

**File:** `simulation/probe_origin_pose.py`

**Type**

Search helper, not SciPy SLSQP.

**Optimization variables**

All 6 actuated UR5e joints.

**Objective**

Maximize the world `Z` coordinate of `forearm_tip_site`.

In code terms, `eval_q(q)` returns `data.site_xpos[reference_site_id][2]`.

**Constraints**

- joint bounds from the loaded MuJoCo model
- after the search, joints 0 and 5 are snapped to zero if doing so keeps the same maximum height

**Seed strategy**

The routine starts from:

- all zeros
- `HOME_Q`
- one straight-ish canonical pose
- many random samples inside canonicalized joint bounds

It keeps the best seed, then runs coordinate-wise local improvements with shrinking step sizes.

**Outputs**

A single joint vector `best_q`.

**Where the result is used**

It is mainly a reference helper and seed source.
S2 uses it as one candidate seed.

**Interpretation**

This routine answers the old question "what is the highest forearm-tip origin pose" rather than the newer question "what pose best matches the current fixed face-direction convention."

### S2 - `find_vertical_face_origin`

**File:** `simulation/probe_origin_pose.py`

**Type**

Search helper, not SciPy SLSQP.

**Optimization variables**

All 6 actuated joints, with `shoulder_pan_joint` forced to zero inside the search.
Optionally the elbow can be fixed.

**Objective**

Prefer a pose that:

1. keeps `attachment_site` aligned to the fixed world-frame face orientation
2. among already aligned poses, maximizes the height of `forearm_tip_site`
3. optionally penalizes elbow bend

The score is implemented as:

```text
reference_z
- 100 * (30 - orientation_score)
- elbow_straightness_weight * |elbow|
```

where `orientation_score` is built from the dot products between the current tool axes and the desired world axes.

**Constraints**

- joint bounds
- `shoulder_pan_joint = 0`
- optional fixed elbow angle

**Seed strategy**

The search tries:

- `ACTIVE_ORIGIN_Q`
- two hand-picked canonical branches
- `HOME_Q`
- the result from S1
- random samples with pan already fixed

Then it runs local coordinate improvements with decaying step size.

**Outputs**

A single joint vector representing the chosen canonical origin pose.

**Where the result is used**

This routine is part of the origin-definition workflow in `probe_origin_pose.py`.
Its output explains why the repo moved away from the older peak-height origin and toward the current shoulder-side-face convention.

**Interpretation**

S2 is the bridge between historical origin-search experiments and the current `ACTIVE_ORIGIN_Q` convention, even though the active controller paths no longer call it every run.

### S3 - `build_study`

**File:** `simulation/study_constrained_ee_x_workspace.py`

**Type**

Multi-start SLSQP workspace study.

**Optimization variables**

Reduced joint vector `y = q[1:6]` with shoulder pan held fixed.

**Objective**

Solve two global boundary problems under fixed orientation:

- minimize world `X` of `attachment_site`
- maximize world `X` of `attachment_site`

**Equality and inequality constraints**

- equality: orientation residual `rotvec(R_target^T R(y)) = 0`
- bounds: joint limits on `q[1:6]`

There is no separate `Z` equality in this study; it computes the full global `X` span at fixed orientation and fixed pan.

**Seed strategy**

- `ACTIVE_ORIGIN_Q[1:6]`
- 24 random reduced-joint seeds

Each seed is sent through SLSQP for both the min-`X` and max-`X` objectives.

**Outputs**

The returned report includes:

- feasible min and max world `X`
- corresponding joint vectors
- full `X` span
- raw `dx/dq`
- a projected feasible `X`-motion gradient in the null space of the rotational Jacobian at the canonical pose

**Where the result is used**

It is an analysis/reporting tool.
The same constrained-workspace idea also informs later scripts that slice the workspace by `Z`.

**Interpretation**

S3 answers the coarse question: "If orientation is fixed and pan is locked, how far can the tool move in world `X` at all?"

### S4 - `_optimize_x_at_z_one`

**File:** `simulation/probe_workspace_xz_envelope.py`

**Type**

Per-slice SLSQP boundary solve.

**Optimization variables**

Reduced joint vector `y = q[1:6]`.

**Objective**

At a fixed target `Z`, solve either:

- min world `X`
- max world `X`

for `attachment_site`.

**Equality and inequality constraints**

- equality: 3 orientation residual components
- equality: `site_z - z_target = 0`
- inequality: `site_z - z_floor_world_m >= 0`
- bounds: joint limits on `q[1:6]`

So this is the core exact boundary solve for a horizontal workspace slice.

**Seed strategy**

The caller passes a multi-start seed list, usually built from:

- `ACTIVE_ORIGIN_Q[1:6]`
- random reduced-joint samples from `collect_slice_seeds(...)`

**Acceptance rule**

This function does not rely only on `res.success`.
It explicitly checks the final residuals against tolerances.
That matters because SLSQP can return messages like "singular matrix" even when the iterate is feasible enough for the study.

**Outputs**

It returns a small dictionary with:

- `ok`
- solved `x`
- corresponding world `y`
- solved `z`
- full joint vector
- solver message

**Where the result is used**

This is one of the most reused solvers in the repo.
It feeds:

- the full `X-Z` envelope study
- the fixed-`Z` X-limit demo
- the fixed-`Z` continuous transport runner
- the later exploratory singularity study

**Interpretation**

If you want the exact constrained `X` boundary at a specific `Z`, this is the central routine.

### S5 - `global_max_site_z_above_floor`

**File:** `simulation/probe_workspace_xz_envelope.py`

**Type**

SLSQP height-limit solve.

**Optimization variables**

Reduced joint vector `y = q[1:6]`.

**Objective**

Maximize the world `Z` coordinate of `attachment_site` while keeping tool orientation fixed and pan locked.

**Equality and inequality constraints**

- equality: orientation residual is zero
- inequality: `site_z >= z_floor_world_m`
- bounds: joint limits on `q[1:6]`

**Seed strategy**

Caller-provided multi-start seeds, usually the same family used for S4.

**Outputs**

One scalar `z_max` if a feasible maximum is found.

**Where the result is used**

It defines the feasible vertical range for:

- `probe_workspace_xz_envelope.py`
- `demo_ee_x_limits_at_z.py`
- `run_fixed_z_x_transport.py`
- exploratory `Z/X` studies

**Interpretation**

S5 answers the envelope question "how high can the tool go at all under the current fixed-orientation constraint set?"

### S6 - `solve_q_at_xz`

**File:** `simulation/demo_ee_x_limits_at_z.py`

**Type**

Continuity-aware SLSQP pose solve.

**Optimization variables**

Reduced joint vector `y = q[1:6]`.

**Objective**

Find a feasible joint configuration at one requested `(x_target, z_target)` while staying close to the previously accepted branch.

The current objective is a joint smoothness cost:

```text
cost(y) =
  ||W wrap(y - y_prev)||^2
  + accel_weight * ||W wrap(y - (y_prev + wrap(y_prev - y_prev_prev)))||^2
```

The first term penalizes step size away from the previous accepted reduced joint state.
The second term is optional and penalizes deviation from a constant-velocity prediction, which suppresses sharp kinks.

**Equality and inequality constraints**

- equality: 3 orientation residual components
- equality: `site_x - x_target = 0`
- equality: `site_z - z_target = 0`
- inequality: `site_z >= z_floor`
- bounds: joint limits on `q[1:6]`

**Seed strategy**

`solve_q_at_xz(...)` itself takes one seed `y0`, but the runner around it supplies many restart seeds.

**Outputs**

It returns:

- `ok`
- the solved reduced joint vector
- the smoothness objective value at the returned point

**Where the result is used**

- R2 uses it for every pose in the fixed-`Z` X sweep
- R3 reuses it indirectly through S7 to choose interior start and stop poses

**Interpretation**

This is the repo's first constrained IK routine that explicitly optimizes for branch continuity, not just feasibility.

### Continuity-aware multi-start ranking

The most important newer behavior is not just inside S6 itself, but in the way R2 uses it.

For each target `X` sample, `demo_ee_x_limits_at_z.py` builds many restart seeds:

- predicted continuation from the last two accepted states
- previous accepted state
- min-boundary solution
- max-boundary solution
- `ACTIVE_ORIGIN_Q`
- midpoint between extrema
- random restarts

Every feasible restart is scored **against the same previous accepted state**, not against its own seed.
The script then sorts all feasible candidates by continuity cost and keeps the lowest-cost branch.

That is what "continuity-aware" means in the current codebase:

- continuity is an explicit optimization objective
- all feasible branches compete under the same metric
- the chosen branch is the smoothest continuation, not the first success

This matters because a square constrained IK problem can have multiple feasible branches at the same task-space pose, especially near singular or wrist-wrap regions.

### S7 - `solve_pose_with_restarts`

**File:** `simulation/run_fixed_z_x_transport.py`

**Type**

Endpoint pose selector built on top of S6.

**Optimization variables**

Reduced joint vector `y = q[1:6]`, solved through repeated calls to S6.

**Objective**

At a single interior endpoint `(x_target, z_target)`, find the feasible pose closest to a hint branch.

Unlike R2's full sweep, this wrapper uses only the first-order joint proximity term because it solves one endpoint at a time.

**Equality and inequality constraints**

The constraints are inherited from S6:

- fixed orientation
- fixed `X`
- fixed `Z`
- floor inequality
- joint bounds

**Seed strategy**

It tries:

- `y_hint`
- `y_alt`
- `ACTIVE_ORIGIN_Q[1:6]`
- midpoint of `y_hint` and `y_alt`
- extra random seeds

Every feasible candidate is ranked by the returned continuity cost relative to `y_hint`.

**Outputs**

A single reduced joint vector for the selected endpoint.

**Where the result is used**

R3 uses it twice:

- once to get the interior start pose
- once to get the interior stop pose

**Interpretation**

S7 is the bridge between the offline boundary studies and the online transport controller.
It makes sure the continuous transport run starts from sensible interior poses rather than raw workspace-edge extrema.

### S8 - `solve_ik_fixed_xz`

**File:** `simulation/study_z_axis_x_freedom_singularity_report.py`

**Type**

Exploratory SLSQP IK solver for multiplicity and singularity analysis.

**Optimization variables**

Reduced joint vector `y = q[1:6]`.

**Objective**

Minimize `||y - y0||^2` for a given seed `y0` while satisfying the same pose constraints as S6.

**Equality and inequality constraints**

- fixed orientation
- fixed `X`
- fixed `Z`
- floor inequality
- joint bounds

**Seed strategy**

The study deliberately runs many seeds so it can discover distinct feasible branches for the same `(x, z)` task-space point.

**Outputs**

- feasibility flag
- one feasible reduced joint solution
- solver message

Later code clusters multiple successful solutions by joint-space distance and computes Jacobian conditioning metrics.

**Where the result is used**

Only in the exploratory singularity report workflow.

**Interpretation**

S8 is not part of the main execution path, but it is relevant because it confirms why the fixed-orientation transport problem is numerically delicate: there can be multiple branches and poor conditioning across the same constrained manifold.

## Runner Wiring

### R1 - `simulation/run_origin_stabilization.py`

This is the baseline origin-hold experiment.

High-level pipeline:

1. Load the UR5e scene and target joint pose, usually `ACTIVE_ORIGIN_Q`.
2. Sample a reproducible random start configuration near the target.
3. Forward-kinematically compute the target `forearm_tip_site` position and tool rotation.
4. At each simulation sub-step, compute site Jacobians and call either C1 or C2.
5. Write the returned setpoint into MuJoCo's actuators.
6. Record convergence metrics, render frames, and save a JSON summary.

The key point is that R1 is a **closed-loop stabilization experiment**, not a workspace solver.

### R2 - `simulation/demo_ee_x_limits_at_z.py`

This is a fixed-`Z` reachability and continuity demo, not a dynamic transport controller.

High-level pipeline:

1. Use S5 to find the feasible maximum `Z`.
2. Choose a demo `Z`.
3. Use S4 twice to find exact min and max constrained `X` at that `Z`.
4. Build an `X` grid between those extrema.
5. For each `X`, solve a feasible pose with S6 under continuity-aware restart ranking.
6. Render a pose-by-pose video and write JSON smoothness metrics.

The important caveat is that the video is a **kinematic pose sequence**.
It shows existence and branch continuity, not continuous servoed motion.

### R3 - `simulation/run_fixed_z_x_transport.py`

This is the current prototype continuous-motion experiment for the main task.

High-level pipeline:

1. Use S5 to find the feasible vertical ceiling.
2. At the chosen `Z`, use S4 to compute exact min and max constrained `X`.
3. Pull back from the exact boundaries by `--x-margin`.
4. Use S7 to solve good interior start and stop poses.
5. Solve a dense continuity-aware fixed-`Z` joint path across the usable `X` interval.
6. Time-parameterize that joint path against explicit reduced-joint velocity, acceleration, and jerk limits.
7. During simulation, send the planned joint references directly to MuJoCo's position servos.
8. Save a video and a JSON summary with path smoothness, timing, path-fidelity, and tracking metrics.

This runner is the closest thing in the repo today to the intended real-robot transport task.

## Known Issues And Failure Modes

### Path fidelity versus exact pose constraints

R3 now precomputes a dense continuity-aware joint path and time-parameterizes it before execution, which removes the old per-step square online IK chatter from the default transport loop.

The remaining transport-side issue is different:

- solved waypoints are exactly feasible
- the interpolated joint path between those waypoints is only approximately feasible

So the main residual errors now come from:

- waypoint density
- spline/path interpolation drift away from the exact constrained manifold
- low-level tracking error relative to that planned joint reference

### Square constrained IK is still a planning concern

Even though the default transport runner no longer solves the square constrained IK problem online every step, the fixed-pan `X + Z + orientation` formulation still controls the difficulty of the offline path solves.

That is why branch continuity and conditioning still matter in:

- S4
- S6
- S7
- S8

### Proximity to workspace boundaries

The exact constrained `X` limits are often near poorly conditioned configurations.
Trying to move too close to those boundaries increases:

- joint sensitivity
- abrupt wrist reconfiguration
- solver restart dependence

The current `--x-margin` in R3 exists specifically to stay away from these hard edges.

### Wrist wrapping and branch sensitivity

Equivalent or near-equivalent wrist configurations can differ by roughly `2*pi` in some joints.
Without continuity-aware selection, the solver can jump between branches that look similar in task space but are far apart in joint space.

This is why:

- S6 uses wrapped joint deltas
- R2 ranks all feasible restarts by a shared continuity metric
- exploratory study S8 is useful for understanding branch multiplicity

### SLSQP success flags are not always the whole story

In S4 and related workflows, the code often accepts a solution based on residual tolerances even if `res.success` is false.
That is not sloppy; it reflects observed SLSQP behavior near constrained singular points where the iterate may still satisfy the task closely enough for analysis.

### Current documentation drift

Some older top-level docs still describe historical paths as if they were current.

Examples:

- `agent.md` still says the active controller is `reverse_nested_x_servo_controller(...)`
- the code actually centers on C1, C2, the planned-joint-trajectory R3 path, and the newer continuity-aware SLSQP pose-solving stack

`CONTROL_DESIGN_NOTEBOOK.md` is closer to the current reality for origin-hold, but it does not yet fully cover the newer transport controller and continuity-aware fixed-`Z` pose solving.

### Stability and maturity snapshot

Use the following mental model:

- **Stable baseline:** R1 with C1, plus the origin-hold metrics workflow
- **Active analysis tools:** S3, S4, S5, S6 and the fixed-`Z` demos/studies
- **Active prototype:** R3 with continuity-aware SLSQP path planning plus planned joint-trajectory playback
- **Available prototype controller:** C3, kept as the explicit online differential-IK transport formulation
- **Exploratory research:** S8 and the singularity/multiplicity report workflow
- **Archived historical path:** C4 and the old reverse-nested design

## Artifacts And Example Outputs

These files are examples of the current workflows. They are useful for orientation, but the document should not be read as depending on the exact numeric values inside them.

### Origin-hold baseline example

- Video: `demonstration_videos/ur5e_cartpole/ur5e_forearm_origin_shoulder_side_face_seed7.mp4`
- JSON: `outputs/control_runs/ur5e_forearm_origin_shoulder_side_face_seed7.json`

This is the canonical example for R1 using the current forearm-origin and shoulder-side-face target convention.

### Fixed-`Z` continuous transport example

- Video: `demonstration_videos/ur5e_cartpole/fixed_z_x_transport_firstpass_z0.540_seed1.mp4`
- JSON: `outputs/control_runs/fixed_z_x_transport_firstpass_z0.540_seed1.json`

This is the current example for R3.
It is useful precisely because it shows both the promise of the transport setup and the remaining path-fidelity and tuning limitations.

### Constrained workspace / singularity report example

- Report: `outputs/workspace_studies/z_x_singularity_20260327_004222/REPORT.md`

This is a good example of the exploratory analysis side of the project and helps explain why branch continuity and conditioning need to be treated as first-class design issues.

## Practical Reading Order

If you are new to the repo and want the shortest path to understanding the live simulation stack, read in this order:

1. C1 and C2 if you want to understand the origin-hold baseline.
2. S4, S5, and S6 if you want to understand how fixed-`Z` feasible motion is being solved.
3. C3 and R3 if you want to understand the current transport prototype.
4. C4 only if you need historical context.
5. S8 and the singularity report if you need to reason about branch multiplicity or numerical fragility.

## Public Interfaces And Types

This document does not introduce any code API changes.
Its only public-facing artifact is this Markdown file.

The intended stable update pattern is:

- update the component's row in the index table
- update that component's dedicated section
- update the runner-wiring section only if call structure changed

That keeps the document maintainable even as the experiments evolve.
