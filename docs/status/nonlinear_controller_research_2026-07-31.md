# Nonlinear / higher-order representational capacity for the OSC controller — research survey and recommendation, 2026-07-31

**Scope**: read-only research task. No code changes, no runs, no commits. The user asked, open-endedly,
whether/how to make the current OSC controller (`controller_core/x_axis_cartesian_impedance.py`)
"learn more nonlinearly" or have "higher order of representation." This document surveys the real
options, assesses each against this repo's actual constraints, and ends with a ranked
recommendation.

## 0. What's already in place (the baseline any new idea has to beat)

`XAxisCartesianImpedanceController` is fundamentally task-space PD → `J.T` → joint torque, but it
has accumulated a real set of structural (model-based, not learned) nonlinear extensions, all
flag-gated, default off except where promoted:

| feature | flag | what it adds |
|---|---|---|
| Task-space inertia shaping | `task_space_inertia_shaping` | Λ(q) = (J M⁻¹ Jᵀ + εI)⁻¹ wrench weighting — a real, pose-dependent nonlinear reshaping of the PD wrench, not a constant gain |
| Nullspace-consistent posture | `nullspace_posture` | dynamically-consistent pseudoinverse projection so posture torque can't leak into task acceleration |
| Diagonal-only Λ for wrench shaping | `lambda_diagonal_shaping` | removes an off-diagonal (X→Z) leak in Λ away from the singularity |
| Adaptive regularization | `lambda_adaptive_regularization` | schedules ε in log(cond(J)) space for the nullspace projector only |
| Posture re-anchoring | `posture_reanchor_on_settle` | one-shot re-capture of q_rest at settle |
| Wrist-orientation task | `wrist_orientation_task` | separate joint-space PD term, masked to the wrist chain, structurally isolated from the (unstable-at-singularity) shared wrench pipeline — this is what actually fixed the "directional ceiling" bug (orientation error 0.22–0.25 rad → 0.06–0.07 rad, zero regressions) |
| **Friction feedforward (landed tonight)** | `friction_feedforward` | `tau_ff = coulomb·tanh(qd/deadband) + viscous·qd`, static Coulomb+viscous, opt-in |

Plus a diagnostic (not control-path) piece directly relevant to this survey:
`controller_core/dynamics_residual.py` + the async residual observer
(`hardware/residual_observer_worker.py`) already compute, every cycle, `qdd_pred` (from
`M(q)⁻¹(tau − bias(q,qd))`), `qdd_measured`, and `qdd_residual = qdd_measured − qdd_pred`. This
is currently logged to `trace_rows` only — nothing reads it into a control decision.

**Tonight's proven results, for calibration**: the friction feedforward term alone closed a real
sim-to-real regression from 50–57% pass rate back to 86–96% across a 4-category rigor sweep at
three poses (`docs/status/friction_ff_alpha_0.2_0.3_sweep_2026-07-31.md`: 73/76 = 96% combined at
α∈{0.2,0.3}; `docs/status/ur5e_sim_friction_modeling_2026-07-31.md` §3.1: 33/38 = 86.8% at
α=0.5), using nothing more exotic than a two-parameter-per-joint static model. This is the bar
any learned/higher-order approach below needs to clear to be worth the added complexity —
**and it has not yet been validated on real hardware** (flagged explicitly as "Part D, not run"
in the friction-ff docs).

## 1. Residual dynamics learning (NN / GP / basis-function regression on top of model-based torque)

**What it is**: fit a correction term `tau_res(q, qd, ...)` from real residual data, added to the
existing model-based torque. Classic real approaches: Local Gaussian Process Regression for
real-time model-based robot control (Nguyen-Tuong et al., combines LWPR-style local models with
GPR accuracy for online inverse-dynamics learning on a Barrett WAM); Deep Lagrangian Networks
and hybrid/structured NN inverse-dynamics models that keep a physics-consistent backbone and
learn only a correction; "Residual Reinforcement Learning" (Johannink et al. 2018, Silver et al.
2018) — learn an RL residual *policy* on top of a fixed controller, shown to give 3–5x data
efficiency over learning from scratch in contact-rich manipulation.

**A finding worth flagging explicitly**: this repo's residual observer already computes exactly
the raw signal a *supervised* residual-torque regression would need, and it's a cheap
transformation away from a training target. `qdd_residual = qdd_measured − qdd_pred` is a joint
acceleration residual; since `M(q)` is already computed every cycle (Pinocchio, validated to
<1e-8 Nm gravity / <1e-8 mass-matrix parity vs MuJoCo), the *effective uncompensated torque* is
simply `tau_residual = M(q) @ qdd_residual` — algebra, not new instrumentation. **This means a
supervised residual-torque model could be trained entirely from data this repo already logs
(once a real-hardware trajectory corpus exists), with no new sensing or control-loop
instrumentation required.** That's a genuinely different starting point from the RL residual
attempts already tried here (see §4) — those learn the residual *online, via trial and error,
under a reward function*; this would learn it *offline, via supervised regression, from logged
data that's already known to be safe*.

**Real-time fit at 500 Hz**: per tonight's own profiling
(`docs/status/direct_torque_controller_phase_profiling_2026-07-31.md`,
`docs/status/real_lab_session_2026-07-31.md` §4–5), the real budget is a 2 ms period; real
hardware measured `controller_mean_ms=0.615` (p95 0.677, max 0.801) for the existing controller,
and **total real-cycle cost mean 1.28 ms / p99 1.47 ms of the 2 ms budget** — i.e. roughly
**0.5–0.7 ms of real headroom** (mean vs. p99) before touching the deadline monitor. No single
matrix operation in the current controller dominates (`np.linalg.cond`'s SVD ~24 µs,
orientation-error quaternion pipeline ~20 µs, Λ-shaping block ~31 µs — all well under headroom).
A residual-model inference call needs to fit inside that same envelope, with a **bounded, data-
independent worst-case cost** (the deadline monitor doesn't care about the mean, and this repo's
own history shows a single real overrun already tripped `DeadlineMonitor` once tonight from an
unrelated ~0.15 ms diagnostic addition before it was moved off-loop).

- **Basis-function regression** (fixed set of RBF/polynomial features → linear weights): fixed
  matrix-vector product, deterministic flop count, trivially numpy-only, easily fits in headroom.
  Best fit for this constraint.
- **A small MLP** (2–3 layers, low hundreds of parameters, matmul + elementwise nonlinearity):
  also deterministic cost, also numpy-only-compatible, still comfortably inside headroom if kept
  small. Reasonable second choice, marginally more representational power than basis regression.
- **GP regression**: per-query cost scales with training-set size unless sparsified (inducing
  points / local GP as in the LGP work cited above) — an *unsparsified* GP risks a data-dependent,
  unbounded-worst-case inference cost, which is a real mismatch with a hard 2 ms deadline monitor.
  Usable only with a fixed, small, pre-selected set of basis/inducing points (which then behaves
  like basis-function regression anyway). Not recommended as the first cut here.

**Complexity**: moderate. Needs (a) a real-hardware trajectory corpus across the operating
envelope (not yet collected at any scale — tonight's session logged select move-hold runs, not a
systematic sweep), (b) an offline fitting pipeline outside `controller_core`, (c) the fitted
model's weights/features loaded into a small, deterministic-cost inference function that *is*
inside `controller_core` (numpy-only, no scipy/sklearn imports live in the sim-independent
path — those stay offline-only), (d) re-running this repo's 4-category rigor sweep before any
real-hardware exposure.

**Payoff assessment**: real but second-order. The friction feedforward term already closed the
single largest, best-characterized real disturbance (steady-state stiction) at 86–96% pass rates.
A learned residual on top would target smaller, harder-to-characterize effects (cable drag,
unmodeled joint compliance, per-unit calibration drift) — plausible value, but genuinely
data-hungry, and this repo has essentially zero real-hardware trajectory volume to fit from yet
(one lab session, a handful of runs). Premature to build before there's a real corpus; the right
sequencing is friction-ff real-hardware validation first, then (if real residuals remain
after that) start logging toward a residual-regression dataset.

## 2. Nonlinear/adaptive friction beyond tonight's static Coulomb+viscous model

**What it is**: LuGre (single internal "bristle deflection" state per joint, captures Stribeck
effect, presliding compliance, hysteresis — the most widely used dynamic friction model in
control, per the friction-compensation literature); GMS/Generalized Maxwell-Slip (multiple
parallel bristle blocks, better hysteresis fidelity than LuGre's single state, more
parameters/states to identify); online-adaptive friction estimation (RLS/Kalman-filter parameter
adaptation of a friction model's coefficients at runtime); neural friction compensators (small
MLP mapping (q, qd) → friction torque, trained from data — a special case of §1's residual
learning, scoped specifically to friction).

**Fit assessment**: LuGre adds one extra scalar ODE state per joint (bristle deflection `z`),
integrated alongside the 500 Hz loop — cheap (six extra scalars, a handful of flops/joint/cycle),
fully numpy-only, deterministic cost. This is the natural "next tier" beyond the static model, IF
real hardware shows the static model has a specific gap it can't close (e.g. presliding-regime
positioning error, or a stick-slip limit cycle the tanh-deadband tuning didn't fully kill — this
repo already found and fixed one such limit-cycle risk during tonight's static-model tuning:
`friction_ff_qd_deadband=0.05` not `0.01`, because `0.01` produced a real closed-loop limit
cycle). GMS is a heavier lift (more states, more parameters to identify) with no evidence yet
that LuGre's single-state hysteresis model would itself be insufficient — premature to reach for
GMS before LuGre is even shown necessary. Online-adaptive (RLS) friction estimation is the
riskiest of this group: an unbounded or poorly-damped adaptation law is exactly the kind of
mechanism that could destabilize torque near a real robot, and would need its own stability
argument and bounding/projection logic before being anywhere close to live torque — no evidence
in this repo's data that the static model's fixed parameters are the bottleneck yet.

**Complexity**: low–moderate for LuGre (well-understood, textbook derivation, small state
addition, still numpy-only); higher for GMS or online-adaptive (parameter ID pipeline, and for
online-adaptive, a stability proof before it's remotely safe to run near hardware).

**Payoff**: low priority right now. The static term already recovered 86–96% of the validated
sim envelope, and its real-hardware validation (the actual point of the whole exercise, given the
real hardware signature that motivated it) **hasn't happened yet**. Escalating to a dynamic
friction model before that real-hardware check is done would be optimizing a model whose real
adequacy is still unmeasured.

## 3. Koopman-operator-style basis expansion / higher-dimensional linearization

**What it is**: lift the state `(q, qd)` into a higher-dimensional space via a set of nonlinear
observable functions (RBF/polynomial/Fourier features of state), fit a (possibly large) linear
operator that approximately advances the lifted state — the resulting dynamics are linear *in the
lifted coordinates*, enabling classical linear-control-theory tools (LQR, MPC) to be applied to a
genuinely nonlinear plant. An active, growing research area (recent 2025–2026 work applies it to
legged locomotion, contact dynamics, and deformable-object/pushing manipulation — see e.g. the
Koopman-lifted contact-dynamics locomotion/manipulation work and derivative-based Koopman
operators for real-time robotic control).

**Fit assessment — the honest answer is this doesn't target a real gap in this repo's plant**.
Koopman's actual value proposition is approximating *unknown or hard-to-model* nonlinear
dynamics (contact, deformable-object interaction, hybrid/discontinuous modes) as a globally valid
linear system. This repo's plant is the opposite case: the rigid-body dynamics of a fixed-base
UR5e are already known essentially exactly — `PinocchioUR5eDynamics` matches MuJoCo to <1e-8 Nm
gravity, <1e-6 bias, <1e-8 mass matrix. There is no real "unmodeled nonlinearity" here for a
Koopman lift to usefully absorb — the actual open sim-to-real gaps found this week (friction/
stiction, the directional wrist-singularity orientation asymmetry) are exactly the kind of
effects §1 and §4's targeted approaches address directly, without needing to replace or
re-derive the whole control law. Adopting Koopman lifting would mean redesigning the controller
around a new, unvalidated linear-in-lifted-coordinates representation — a large architecture
change, in direct tension with this repo's own working norms ("do not change training/eval logic
and controller logic in the same commit," "never touch the real-time control path without
extremely explicit justification") — for a benefit that isn't motivated by any measured gap in
this specific system. Real-time cost of *evaluating* a Koopman-lifted linear controller would
likely be comparable to or cheaper than the existing OSC math (it's ultimately a matrix-vector
product), so feasibility per se isn't the blocker — motivation is.

**Complexity**: high (new controller family, full re-validation against the existing OSC's
250+-run-tuned envelope, an offline lifted-space system-identification pipeline).

**Payoff**: low expected value for this specific, already-well-characterized plant. Genuinely
interesting for a future project with real unmodeled/contact-rich dynamics; not a fit here.

## 4. Learned gain-scheduling / policy layers on top of the structured controller

**This is the one direction the repo has already spent real effort on and repeatedly failed to
beat baseline with — the report below is explicit about not re-treading that exact path.**

`rl_gain_scheduling/` (Gymnasium env wrapping this same controller, PPO/SAC) has been tried
**six** real times, targeting the height_alpha=0.5 directional-orientation-asymmetry bug:

| attempt | action space | algo | result vs. fixed-gain baseline |
|---|---|---|---|
| run1/run2 | full gains | PPO | 0/20 valid — collapsed to "never move" |
| reward_v2/v3 | full gains | PPO | 0/20, 1/20 valid — different pathological corner |
| v4/v5 (height-pinned, `data.time` bug found+fixed) | full gains | PPO | 0/4 valid both before and after the fix — "never move" collapse persists |
| residual-torque, unpenalized/penalized | bounded residual torque | PPO | 2/8, 5/8 (best result of the six) |
| 4th attempt, safety-threshold-mismatch fixed | bounded residual torque | PPO | **0/8** — fixing the mismatch made it *worse*, new failure mode (`axis_error` growth guard trips on all 8 cells) |
| SAC gains / SAC residual | both variants | SAC | 2/8, 1/8 — same or worse; SAC's residual case actively learns a *harmful* correction (mean 0.25–0.58 Nm, not near-zero) that still trips the same axis-error-growth guard |

Baseline (fixed-gain, `config/ur5e_mujoco_torque_osc_tuned*.yaml`): 7/8–8/8 on the same grid.
Documented root causes (`docs/CURRENT_STATUS.md`, 2026-07-25 audit): a deceptive reward
landscape (every dense reward term except x_error is minimized by *not moving*, and the terminal
completion bonus provides no gradient across the "sit still ↔ clean move" plateau) combined with
zero PPO exploration pressure (`ent_coef` unset → 0.0) — a genuine exploration/reward-shaping
failure, not a magnitude bug, with a concrete fix recipe proposed but never implemented. Notably,
the failure mode is now confirmed **not algorithm-specific** — PPO and SAC (on-policy vs.
off-policy, with a replay buffer and 3x the training-signal reuse) both land on essentially the
same `axis_error`-growth guard trip once the training-env safety threshold is corrected to
match the real bug. This narrows the real bottleneck to the *action space / reward shape*, not
the optimizer.

**What a "different" approach in this space would need to look like to not just be attempt #7**:
not another online RL run, but a **supervised, offline-fit** pose-conditioned gain (or bounded
correction) function — e.g. regress gains against the many already-validated, per-condition tuned
configs this repo has accumulated (the height_alpha grid, the wrist-orientation fix, the
diagonal-Λ fix), rather than discovering good gains via trial-and-error under a reward function.
This sidesteps the deceptive-reward-landscape/exploration failure mode entirely, since it trains
on data that's already known to be safe and valid rather than searching a space where "sit still"
is a local trap. Separately, Berkenkamp et al.'s "Safe Online Gain Optimization for Variable
Impedance Control" is a genuinely different *mechanism* (Bayesian optimization with explicit
safety constraints baked into the acquisition function, not policy-gradient RL, no reward
function to misdesign) worth being aware of if online adaptation is ever revisited — but it's a
heavier, more specialized undertaking than a first cut needs.

**Honest ceiling on this whole direction**: even a well-designed supervised gain-interpolator
targets the wrong layer for the specific bug that's motivated all six RL attempts. AGENTS.md is
explicit that the height_alpha=0.5 directional asymmetry is **structural**
(a nullspace-projector Frobenius-norm asymmetry with wrist_2 sign, not a gain problem — a direct
`kp_posture`/`kd_posture`/`kd_joint` sweep at the exact failing case "barely moved the outcome").
That bug already has a working *structural* fix — `wrist_orientation_task` — which closed it
cleanly (0.22–0.25 rad → 0.06–0.07 rad orientation error, zero regressions) without any learning
at all. **A gain-scheduling layer, however it's trained, is very unlikely to beat a fix that
already directly addresses the actual mechanism.**

## 5. Real-time and real-hardware safety constraints — what's compatible and what isn't

- **`controller_core/` stays pure numpy** — no simulator imports, and by established practice
  here, no heavy ML-framework imports (torch/JAX/tensorflow) either. This rules out anything that
  assumes a GPU-resident model or an autograd-framework runtime inside the control path. A fitted
  model has to be reducible to plain numpy array operations (matmuls, elementwise nonlinearities,
  fixed-size dot products) to live here — true of basis-function regression and small MLPs, not
  true of e.g. a large transformer or an unsparsified GP.
- **GPU inference is not compatible with this loop as it exists.** PCIe transfer + CUDA kernel
  launch latency is not deterministic worst-case — real driver/scheduler tail latencies can spike
  well past the ~0.5–0.7 ms of real headroom measured above, and there's no existing mechanism in
  this codebase for a dedicated pinned/real-time GPU execution path (CUDA graphs, locked clocks,
  isolated core). Any GPU-inference idea would require a fundamentally different execution
  architecture, not a drop-in addition — out of scope unless the project decides to take on that
  redesign deliberately.
- **CPU-only, small, fixed-shape models fit.** The existing controller's own profiling shows no
  single linear-algebra op dominates and the whole `compute()` call costs ~0.28–0.6 ms depending
  on machine; a small deterministic-cost addition (basis regression, small MLP forward pass) has
  real headroom to land in, provided it's kept small and its worst-case cost is a fixed flop count
  (no data-dependent loops, no unbounded kernel search as in raw GP).
- **The existing guard architecture must not be weakened, and must not be the "correctness"
  mechanism for a learned component.** `hardware/safety.py` (`CartesianMoveMonitor`,
  `DeadlineMonitor`, `StaleStateMonitor`, `EStopLatch` — one-way, no reset) and
  `controller_core/safety.py` (`ImpedanceSafetyMonitor` — drift, orientation, |qd|, axis-error
  growth, NaN/joint-limit) are the real backstop and are already this repo's own attempted
  arbiter of "did the learned thing go wrong." The RL history above (§4) is a direct
  demonstration of the failure mode to avoid: relying on the guards to catch a bad learned output
  after the fact ("guard trips") is not success, it's exactly what happened six times. Any new
  learned component needs to be shown safe *before* real deployment (offline validation against
  the same 4-category rigor sweep this repo already uses), with the guards as a backstop, not a
  design crutch.
- **Anything with live/online adaptation** (online-RLS friction, live RL, online GP updates)
  needs its own boundedness/stability argument before being anywhere near real torque — this is
  a materially higher bar than an offline-fit-then-freeze model, and nothing surveyed here has
  cleared that bar yet in this repo.

## 6. Ranked recommendation

1. **Validate `friction_feedforward` on real hardware first.** Not a new nonlinear-representation
   idea — it's finishing what already landed tonight and is unvalidated on the real arm ("Part D"
   in `docs/status/friction_ff_alpha_0.2_0.3_sweep_2026-07-31.md`, explicitly not yet run). This
   is the highest-leverage nonlinear term already built, sim-validated at 86–96%, and its real
   payoff is completely unmeasured. Nothing below is worth prioritizing over closing this loop.
2. **If a learned approach is still wanted afterward: supervised residual-torque regression
   using the existing residual-observer data pipeline** (§1), starting with basis-function
   regression (not GP, not a large NN) as the first cut. This reuses infrastructure that already
   exists (`controller_core/dynamics_residual.py`, the async observer), fits the numpy-only /
   bounded-latency constraint cleanly, and — unlike the repo's RL attempts — is a fundamentally
   different mechanism (offline supervised fit on logged, already-safe data vs. online
   trial-and-error policy search), so it doesn't inherit the RL failure mode. Real payoff is
   gated on collecting a real-hardware trajectory corpus first, which doesn't exist at meaningful
   scale yet.
3. **If gain-adaptation specifically is still desired**, a supervised pose-conditioned gain
   interpolator (§4) is worth trying once, in preference to a 7th RL attempt — but temper
   expectations going in: the actual bug motivating six prior attempts is documented as
   structural, and already has a working structural fix (`wrist_orientation_task`). Don't expect
   a smarter gain scheduler to outperform a fix that already addresses the real mechanism.
4. **LuGre dynamic friction** — only pursue if real-hardware data (from step 1) shows the static
   model has a specific, identified gap it can't close. Not justified speculatively.
5. **Koopman-operator lifting, GMS friction, online-adaptive friction, and any GPU-inference
   approach — not recommended for this repo right now.** Koopman targets a modeling gap this
   plant doesn't have (rigid-body dynamics are already known near-exactly); GMS and online
   adaptation are heavier versions of tools whose lighter forms (static friction, structural
   controller terms) haven't yet been shown insufficient; GPU inference requires an execution
   architecture this codebase doesn't have and that a hard 2 ms deadline can't safely absorb.
