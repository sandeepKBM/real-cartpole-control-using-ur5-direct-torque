# LuGre dynamic friction model — implementation plan

**Status:** planning document only. No code written, no config changed, no run executed.
**Scope:** a future implementation session, gated on the real-hardware breakaway finding below.
**Last updated:** 2026-07-31

## 0. The real gap this plan is answering

Tonight (2026-07-31) `friction_feedforward` (`controller_core/x_axis_cartesian_impedance.py`,
see §3.1 below) — a static `tau = coulomb*tanh(qd/deadband) + viscous*qd` term — was validated
extensively in sim (96% pass rate at `height_alpha` 0.2/0.3, 87% at 0.5; AGENTS.md §3) and then
run on real hardware for the first time, together with two new acceleration-driven trajectory
profiles, `accel_duration_triangular` and `accel_duration_scurve`
(`simulation/ur5e_mujoco_torque.py::x_profile_target`, lines 656–730, added the same night).

Real result, both profiles, same commanded peak accel (0.15 m/s²): both tripped the real
`CartesianMoveMonitor` TCP-acceleration guard (0.5 m/s² threshold, `hardware/safety.py`) almost
immediately (~0.37–0.38 s into a 2 s move), achieving only ~2% of target displacement, with the
**real measured** TCP acceleration spiking to ~3.3–3.5x the commanded value (0.50–0.52 m/s² vs.
0.15 m/s² commanded) at the trip moment. (As of this writing no dated `docs/status/` artifact
documents this specific real run yet — this account is taken directly from the task brief that
commissioned this plan; whoever implements this should either locate or write that status doc
before proceeding, per this repo's own evidence-over-narrative norm.)

**Why this is diagnostic, not a profile-shape bug**: the two profiles have deliberately different
jerk/smoothness characteristics — `accel_duration_triangular` is a bang-bang (bounded but
discontinuous) acceleration, `accel_duration_scurve` is jerk-continuous (`a(t) = accel *
sin(2*pi*t/T)`, chosen specifically to avoid a bang-bang accel spike per the code comment at
`simulation/ur5e_mujoco_torque.py:721–726`) — and behave very differently in sim under the
existing static friction model. Both nonetheless failed almost identically on real hardware. That
is the signature of a real stick-slip **breakaway** event: static friction holds the arm nearly
still while commanded torque builds against it, then releases suddenly with its own transient
that neither trajectory's own shaping controls, not a controller/profile defect. A smooth
tanh-based Coulomb+viscous model has no true static/breakaway threshold — it cannot represent
"stuck, then suddenly lets go." That is exactly the regime the dynamic friction model literature
(LuGre, GMS) is built for.

`docs/status/nonlinear_controller_research_2026-07-31.md` §2 surveyed this space earlier tonight
and flagged LuGre as "a cheap, deterministic next tier beyond the static model, but not justified
until real hardware shows the static model has a specific unclosed gap." That gap is now real,
specific, and measured (above). This document is the concrete, buildable follow-on.

## 1. The LuGre model

Single internal state per joint, the **bristle deflection** `z` (physically: the average
elastic deformation of the microscopic asperity contacts between the two surfaces, "bristles"
in the model's own terminology):

```
dz/dt = qd - |qd| * z / g(qd)
tau_friction = sigma0 * z + sigma1 * dz/dt + sigma2 * qd
g(qd) = Fc + (Fs - Fc) * exp(-(qd/vs)^2)          # Stribeck curve
```

Parameter meanings, all per-joint:

| symbol | physical meaning | units (rotary joint) |
|---|---|---|
| `sigma0` | bristle stiffness — how much friction torque builds per radian of pre-sliding deflection | Nm/rad |
| `sigma1` | bristle micro-damping — damps the internal `z` dynamics, controls the sharpness of the breakaway transient | Nm·s/rad |
| `sigma2` | viscous friction coefficient (same role as the existing static model's `friction_ff_viscous`) | Nm·s/rad |
| `Fc` | Coulomb / kinetic friction level (same role as the existing model's `friction_ff_coulomb_nm`, but here it is the *floor* the Stribeck curve settles to, not the whole model) | Nm |
| `Fs` | static / breakaway friction level — the peak resistance at zero velocity, `Fs > Fc` always | Nm |
| `vs` | Stribeck velocity — the velocity scale over which friction decays from `Fs` down to `Fc` | rad/s |

At `qd = 0` and held there, `z` grows toward `g(0)/1 * sign(...)` — informally, `z` integrates
up until `sigma0 * z` reaches the static torque available (`Fs`), which is exactly the "stuck,
building resistance" phase the static tanh model cannot represent (`tanh` has no memory — its
output is a pure function of instantaneous `qd`, so it can never distinguish "just started
holding still" from "held still for 2 seconds"). Once the applied torque exceeds `Fs`, `z`
"slips" and the friction torque drops toward the (lower) Coulomb/Stribeck curve — the breakaway
release. This qualitative behavior (rising resistance, then a release transient) is a plausible
match for the real trip signature: a real load builds against static friction, then breaks free,
producing a genuine, not-in-the-model acceleration transient right as motion actually starts —
consistent with the observed ~0.37–0.38 s-into-move trip timing and the 3.3–3.5x real/commanded
accel ratio.

**Literature, robot-joint-specific** (not generic sliding-surface tribology):

- Canudas de Wit, Olsson, Åström, Lischinsky, "A New Model for Control of Systems with Friction"
  (IEEE TAC, 1995) — the original LuGre paper; not robot-specific but the source of the equations
  above and the canonical notation used here.
- "Parameter Identification of LuGre Friction Model for Robot Joints" (Scientific.Net /
  ResearchGate, search result: https://www.scientific.net/AMR.479-481.1084) — robot-joint-specific
  identification.
- "A new iterative identification algorithm for estimating the LuGre friction model parameters"
  (Mechanism and Machine Theory, ScienceDirect) — iterative fitting method, general but applied to
  robot-relevant setups.
- "Friction dynamics identification based on quadratic approximation of LuGre model" (Nonlinear
  Dynamics, Springer, 2024) — notes that four static parameters (`Fc`, `Fs`, `vs`, and the
  Stribeck-curve shape) are typically identified from a measured Stribeck curve (steady-state
  torque vs. constant velocity, multiple velocities), while the two dynamic parameters (`sigma0`,
  `sigma1`) are identified separately from a stick-slip / pre-sliding response — this maps almost
  exactly onto §4's calibration proposal below. Also notes some studies use a Stribeck exponent
  other than 2 and nonlinear (not purely linear) viscous terms for industrial robot links — a
  refinement not adopted here (the simpler exponent-2 / linear-viscous form above is the
  standard starting point and matches what the existing static model already assumes for its own
  viscous term).
- "Extending a dynamic friction model with nonlinear viscous and thermal effects" (DLR,
  https://elib.dlr.de/120472/) — DLR's own extended dynamic (LuGre-family) friction model for a
  real robot joint (their FSJ joint); confirms the general architecture (Stribeck + dynamic
  bristle term + viscous, sections on "Motor Constant and Inertia Identification" and "Nonlinear
  Viscous Friction") but this document's own extraction of it could not reliably pull concrete
  numeric parameter values from the source PDF — reported honestly as a limitation, not
  fabricated. **No single authoritative numeric table for UR5e-class joints specifically was found
  in this pass**, consistent with the same honest gap already documented for the static model's
  own sourcing (`docs/status/ur5e_sim_friction_modeling_2026-07-31.md` §1: "no single table I
  could cite with confidence"). Typical published orders of magnitude for `sigma0` in
  rotary-joint LuGre fits are large (bristle deflection `z` is physically tiny, so `sigma0` must
  be large to produce Nm-scale torque from it) — this plan does not assert a specific number and
  defers to §4's real calibration procedure rather than guessing, following this repo's own
  established discipline of not fabricating precise citations.

## 2. Real-time feasibility at 500 Hz

Per `docs/status/direct_torque_controller_phase_profiling_2026-07-31.md`: the real 500 Hz loop
(2 ms period) currently spends `controller_mean_ms=0.615` (p95 0.677, max 0.801 — real hardware
numbers) inside `compute()`, with **total real-cycle cost mean 1.28 ms / p99 1.47 ms of the 2 ms
budget** — roughly 0.5–0.7 ms of real headroom. That same document's sub-operation breakdown
(measured on this machine, relative ranking transfers) shows no single linear-algebra op
dominates: `np.linalg.cond(J)` (SVD) ~24 µs, the orientation-error quaternion pipeline ~20 µs, the
whole Λ-shaping block ~31 µs — all comfortably inside headroom.

A LuGre update is, per joint, per cycle: one `g(qd)` evaluation (one `exp`, a few multiplies —
same cost class as the existing `tanh` call in `friction_feedforward`), one Euler step of the `z`
ODE (`z += dt * (qd - abs(qd)*z/g)`, a handful of scalar flops), and one friction-torque
evaluation (`sigma0*z + sigma1*dz + sigma2*qd`). Vectorized across 6 joints this is a handful of
numpy elementwise array ops on length-6 arrays — the same shape and cost class as the existing
`tau_friction_ff = coulomb * np.tanh(qd/deadband) + viscous * qd` line
(`controller_core/x_axis_cartesian_impedance.py:573`), which the profiling doc's methodology
never even singled out as separately measurable (it's part of the ~69% "everything else" bucket
of small numpy calls, not a "big matrix math" line item). **Conclusion: this is a trivial
addition computationally** — no new matrix operations, no new `O(n^3)`-class cost, deterministic
per-cycle flop count (an ODE Euler step, not an iterative solve), well inside the measured
headroom. No further profiling is needed before implementation; the real risk in this feature is
not compute cost, it is correctness/stability of the state update and its dt-dependence (§3).

## 3. Where it plugs into the existing architecture

### 3.1 Config surface — coexist with `friction_feedforward`, don't replace it

Current static model, fully read: `controller_core/x_axis_cartesian_impedance.py`:
- Config fields `friction_feedforward: bool = False`, `friction_ff_coulomb_nm`,
  `friction_ff_viscous`, `friction_ff_qd_deadband` (lines 190–204).
- YAML parsing in `from_controller_yaml_section` (lines 252–269).
- Computation inside `compute()` (lines 567–573): `tau_friction_ff = coulomb *
  np.tanh(qd/deadband) + viscous * qd`.
- Summed into the joint-space bias alongside damping/posture/gravity (line 579:
  `tau_bias = tau_damping + tau_posture + tau_orient_wrist + tau_friction_ff + g`), then scaled by
  the same `task_backtrack_scale` as everything else (line 596) and flows through the existing
  hard-clip logic identically — no bypass. `CartesianImpedanceOutput.tau_friction_ff` (line 283)
  and `.friction_feedforward_active` (line 305) carry it into telemetry.

Proposed design, following this repo's own established convention for every flag added tonight
(`wrist_orientation_task`, `lambda_diagonal_shaping`, `friction_feedforward` itself — all
opt-in, default off, additive, never replacing the prior behavior):

```python
friction_feedforward: bool = False          # unchanged — existing static model, kept working
friction_model: Literal["static", "lugre"] = "static"   # NEW — only consulted when
                                                          # friction_feedforward is True
# LuGre parameters, per-joint arrays matching JOINT_NAME_ORDER, only used when
# friction_model == "lugre":
lugre_sigma0_nm_per_rad: np.ndarray = field(default_factory=...)
lugre_sigma1_nm_s_per_rad: np.ndarray = field(default_factory=...)
lugre_sigma2_nm_s_per_rad: np.ndarray = field(default_factory=...)  # mirrors friction_ff_viscous
lugre_fc_nm: np.ndarray = field(default_factory=...)                # mirrors friction_ff_coulomb_nm
lugre_fs_nm: np.ndarray = field(default_factory=...)                # NEW: breakaway level, Fs > Fc
lugre_vs_radps: np.ndarray = field(default_factory=...)              # NEW: Stribeck velocity
```

`friction_model` is nested under the existing `friction_feedforward` boolean (not a parallel
independent flag) so that `friction_feedforward: false` unambiguously means "no friction
feedforward at all," matching every other flag in this file's own pattern of one boolean gating
one behavior addition. Inside `compute()`, the existing `if use_friction_feedforward:` block
(line 569) branches on `self.cfg.friction_model`:

```python
if use_friction_feedforward:
    if self.cfg.friction_model == "lugre":
        tau_friction_ff = self._lugre_step(qd, dt)   # new method, see 3.2 for `z` state, dt for §3.3
    else:
        coulomb = ...  # unchanged static branch
```

This keeps `tau_friction_ff` flowing through the exact same downstream lines (579, 596, 613, 635)
unchanged — the rest of `compute()` does not need to know which friction model produced the
value. New named config: `config/ur5e_mujoco_torque_osc_tuned_friction_ff_lugre.yaml` (mirroring
`config/ur5e_mujoco_torque_osc_tuned_friction_ff.yaml`'s existing naming), never modifying the
existing static-model config — same "preserve old configs" rule
(`AGENTS.md` §7) already followed by every config in this file's history.

### 3.2 New persistent per-joint state: the bristle deflection `z`

`z` is genuinely stateful — integrated over time, exactly like `_x_error_integral` was discussed
(not yet built) for a different feature, and like the state fields that already exist:
`self._q_rest`, `self._x0`/`_y0`/`_z0`/`_quat0` (all set in `__init__`, lines 328–332, and
re-captured in `reset_from_state()`, lines 336–347).

Proposed field: `self._friction_z: np.ndarray = np.zeros(6, dtype=np.float64)` added to
`__init__` alongside the other instance state (after line 332). Reset semantics: zeroed in
`reset_from_state()` (alongside line 343's `self._q_rest = ...`) — a fresh episode should start
with no accumulated bristle deflection, exactly like `_posture_reanchored`/`_x_des_at_anchor` are
reset there (lines 345–346). **Not** reset by `set_gains()` (lines 353–371) — that method's own
docstring already states the contract this must follow: "does not touch ... any instance state
... a gain change must never reset ... mid-episode." A future gain-scheduling or auto-tuning
caller must not accidentally wipe accumulated stiction state by adjusting gains mid-run.

The update itself belongs in a small private method, e.g. `_lugre_step(self, qd, dt) ->
np.ndarray`, called once per `compute()` when `friction_model == "lugre"`:

```python
def _lugre_step(self, qd: np.ndarray, dt: float) -> np.ndarray:
    sigma0 = self.cfg.lugre_sigma0_nm_per_rad
    sigma1 = self.cfg.lugre_sigma1_nm_s_per_rad
    sigma2 = self.cfg.lugre_sigma2_nm_s_per_rad
    fc = self.cfg.lugre_fc_nm
    fs = self.cfg.lugre_fs_nm
    vs = self.cfg.lugre_vs_radps
    g = fc + (fs - fc) * np.exp(-(qd / vs) ** 2)
    z_dot = qd - np.abs(qd) * self._friction_z / g
    self._friction_z = self._friction_z + dt * z_dot   # explicit Euler; see stability note below
    return sigma0 * self._friction_z + sigma1 * z_dot + sigma2 * qd
```

Numerical-stability note for whoever implements this: explicit Euler on `dz/dt = qd - |qd|*z/g`
is only conditionally stable — at large `qd*dt/g` (a large deadband-free discrete step relative to
the local relaxation time) the integration can overshoot/oscillate. At 500 Hz (`dt=0.002s`) this
is very unlikely to matter for physically realistic `qd` and any sane `sigma0`/`g` combination
(the same order-of-magnitude reasoning that made the existing `friction_ff_qd_deadband=0.05` fix
necessary for the static model's own hold-phase stability, AGENTS.md's tanh limit-cycle finding,
is the right lens to re-apply here empirically, not assume away) — flagged as a concrete thing the
sim-side validation sweep (§6) must check for (a growing, non-decaying `z` trace at hold, or a
`|qd|` guard trip that wasn't there under the static model, would be the tell), not something to
special-case in code speculatively before it's shown to be a real problem.

### 3.3 The real blocking gap: `dt_s` does not reach `compute()` today, on either path

This is stateful, dt-dependent integration — it needs a real `dt` every cycle, not an assumed
constant. Confirmed by reading the actual code, and the gap is in **two** places, not the one
named in the task brief:

1. **Hardware call site** (`hardware/direct_torque_link.py`): `compose_robot_state()` (lines
   170–199) and `build_robot_state()` (lines 201–217) build the dict passed to
   `controller.compute()` and **do not accept or include a `dt_s` key anywhere in the returned
   dict** (confirmed by reading the full body — the closest thing is `time_s`, used only to set
   `"time"`). The real call site,
   `hardware/direct_torque_transport.py:378–385` (`robot_state = link.compose_robot_state(...)`
   → `controller.compute(robot_state)` at line 390), has a real `dt_s` in scope at that exact
   point in the loop (used two call sites below, at `hardware/direct_torque_transport.py:420`,
   for `move_monitor.check(..., dt_s=dt_s)`) — it is simply never threaded into the dict the
   controller receives.
2. **`controller_core/state_types.py` itself doesn't recognize a `dt_s` key at all**, even on the
   sim path. `MujocoUR5eState.as_robot_state()` (`simulation/ur5e_mujoco_torque.py:93,116`)
   *does* put `"dt_s": float(self.dt_s)` into its raw dict — but that raw dict is filtered through
   `as_impedance_robot_state()` (`controller_core/state_types.py:191–242`) before it becomes the
   `st` dict `compute()` actually reads (line 432: `st = as_impedance_robot_state(state)`), and
   that function's explicit key whitelist (required keys at 193–203, optional keys individually
   copied at 218–241) has no `dt_s` entry anywhere — it is silently dropped. **This means even
   today's sim path does not deliver `dt_s` into `compute()`'s `st`**, not just the real-hardware
   path as the task brief suspected; the gap is one layer deeper.

**Required prerequisite fix** (small, additive, must land before any dt-dependent stateful term
can work correctly anywhere — sim or real):

- `controller_core/state_types.py`: add `dt_s: float` to the `RobotState` TypedDict (near
  `time` at line 64), and copy it through in both `as_robot_state()` (as an optional key, matching
  the pattern of `target_x_vel` etc. at lines 167–176) and `as_impedance_robot_state()` (same
  pattern, lines 218–227). Should be optional (`if "dt_s" in raw and raw["dt_s"] is not None`),
  not required — this keeps every existing caller that doesn't pass it (most of them, today)
  working unchanged; `compute()`'s `_lugre_step` caller falls back to a configured nominal dt
  (e.g. a new `lugre_fallback_dt_s: float = 0.002` config field, or simply skip the LuGre update
  entirely for that one cycle) if `dt_s` is absent, rather than crashing — a genuinely optional
  field, not a silent new required contract that breaks every other controller consumer of
  `RobotState`.
- `hardware/direct_torque_link.py`: add a `dt_s: float` parameter to `compose_robot_state()` and
  `build_robot_state()`, included in the returned dict alongside `"time"`.
- `hardware/direct_torque_transport.py:378–385`: pass `dt_s=dt_s` into the existing
  `link.compose_robot_state(...)` call (the value is already computed and in scope in this exact
  loop, per line 128: `dt_s = 1.0 / frequency_hz`, and already reused for other purposes,
  e.g. line 420).
- Sim side: `simulation/ur5e_mujoco_torque.py` already puts `dt_s` into its own
  `as_robot_state()` output (line 116) — once `state_types.py`'s whitelist recognizes it, the sim
  path starts working with **no sim-adapter code change needed**, confirming this really is one
  coherent, minimal fix rather than two separate features.
- This is a real, in-scope, additive fix to `controller_core/` and `hardware/` — narrowly
  targeted at making `dt_s` available, not touching any control-law math, gain, or safety
  threshold. It should land and be tested (existing `RobotState`/`as_impedance_robot_state` unit
  tests plus a new explicit "dt_s survives the round trip" test) **before** the LuGre term itself
  is written, as its own small, reviewable commit — per this repo's own rule (§7 of AGENTS.md,
  "do not combine startup fixes with gain tuning" generalizes cleanly to "do not combine a
  plumbing fix with a new controller term in the same commit").

### 3.4 A real interaction to be aware of, not solved here: PolyScope's own `friction_comp`

Read directly, `hardware/direct_torque_link.py:156–167`
(`direct_torque(self, tau_nm, *, friction_comp: bool = True)`) wraps UR's own
`self._control.directTorque(tau.tolist(), friction_comp)` RTDE call, and every real call site in
this repo passes `friction_comp=True` unconditionally
(`hardware/direct_torque_transport.py:433,586`, `hardware/joint_motion.py:48`,
`tools/_rtde_control_probe.py:46`). This means **UR's own built-in (PolyScope-side) friction
compensation is already active on every real run**, including tonight's failed breakaway test —
the real trip happened *despite* `friction_comp=True` already being on. Two implications for this
plan, neither resolved here:

1. It is real, positive evidence that PolyScope's own compensation does not fully cover the
   breakaway/stick-slip regime on this arm — strengthens (does not weaken) the case for adding a
   model-based term on the Python side.
2. Any LuGre feedforward this plan adds will be layered *on top of* whatever UR's own
   `friction_comp` is doing internally, not fully independent of it — the two could partially
   double-compensate in the Coulomb/viscous regime (same risk the static `friction_feedforward`
   docstring already implicitly carries, since it was validated real-hardware-adjacent only via
   sim). This plan does **not** propose disabling `friction_comp` to get a "clean" signal — that
   would be a real behavior change to an already-validated real call path, out of scope per §7's
   non-goals. Whoever runs the §4 calibration procedure should be aware the measured breakaway
   torque is *against* a system that already has some compensation active, not a bare-metal
   measurement — a caveat to carry into any published parameter values, not a blocker.

## 4. Parameter identification / calibration

There is essentially no real breakaway-specific calibration data yet — one failed automated test
tonight (not a calibration run, a validation run that happened to fail informatively) is the only
real signal. Proposed minimal procedure, matching the existing size1/size3 joint-class split
already used in `assets/ur5e_torque/ur5e_torque.xml:11–35` (`size3`: shoulder_pan, shoulder_lift,
elbow, `frictionloss=5.0, damping=0.4`; `size1`: wrist_1, wrist_2, wrist_3, `frictionloss=1.0,
damping=0.15`) and the config's own existing `friction_ff_coulomb_nm`/`friction_ff_viscous`
per-joint-class defaults:

1. **Very-low-velocity ramp test, per joint (or per representative joint per class, matching
   size1/size3 if per-joint testing isn't practical)**: command a slowly ramping torque (well
   below the joint's rated torque) against a fixed posture, starting from rest, and log
   `q`/`qd`/commanded torque at the full 500 Hz rate. The moment the joint visibly starts moving
   (a clear `qd` departure from ~0) marks the breakaway event; the commanded torque magnitude at
   that instant is a direct measurement of `Fs` for that joint. This is exactly the kind of test
   the user is already about to run manually tonight (a low-accel real test at 0.05 m/s² per the
   task brief) — that run, if logged at full rate with torque and `qd`, is a legitimate **first
   real data point** for `Fs`, not a full calibration by itself (one trajectory, one direction,
   one posture — gravity-load-dependent friction, direction-asymmetry, and temperature effects
   are all unmeasured by a single run).
2. **Steady-state Stribeck sweep** (needed for `Fc`, `vs`): hold the joint at several fixed small
   velocities (e.g. 0.01, 0.02, 0.05, 0.1, 0.2 rad/s) via a simple velocity-tracking test, log the
   steady-state torque required to sustain each velocity. `Fc` is the value the curve approaches
   at higher velocities in this sweep; `vs` is fit from how fast the curve decays from `Fs`
   (measured in step 1) down toward `Fc` — a simple least-squares fit to
   `tau(qd) = Fc + (Fs-Fc)*exp(-(qd/vs)^2)` over the swept points, no new tooling needed beyond
   `scipy.optimize.curve_fit` run offline (not in `controller_core`, which stays numpy-only per
   its own established rule).
3. **`sigma0`/`sigma1` from the pre-sliding/breakaway transient itself**: the same ramp test from
   step 1, but looking at the *shape* of the small pre-motion deflection just before breakaway
   (torque vs. the tiny resulting angular displacement while still nominally "stuck") gives
   `sigma0` (slope of that near-linear pre-sliding region); `sigma1` is harder to identify
   cleanly from a single ramp and typically needs either a dedicated stick-slip oscillation test
   or is left at a small damping value and tuned qualitatively against the sim smoke test's
   breakaway-transient shape (matching how `friction_ff_qd_deadband` was tuned empirically against
   an observed limit cycle, not derived analytically, per AGENTS.md's own account of that fix).
4. **Honesty about what's still needed beyond step 1**: a single low-accel real run gives at most
   one `(Fs, direction, posture, joint-subset)` data point — not `vs`, not a validated `sigma0`,
   not per-joint-class generalization, not repeatability/variance across runs, not
   temperature/warm-up effects (real UR arms are documented to show friction changes as motors
   warm up — outside this plan's scope to characterize). Treat step 1's result as a sanity check
   / rough magnitude anchor for `Fs`, feed it into the sim-side validation (§6) as a starting
   guess, and plan steps 2–3 as genuinely separate future hardware time, not something to rush
   into the same session.

## 5. Sim-side implications

MuJoCo's native contact/joint-friction solver does not implement LuGre dynamics for the plain
`<motor>` actuators this repo's model uses (`assets/ur5e_torque/ur5e_torque.xml:141–155` —
`gear=1`, force-limited direct-torque motors, no dcmotor electrical/friction model attached).
Two real routes were researched:

1. **MuJoCo 3.7.0+ added a native `dcmotor` actuator type with optional LuGre friction built in**
   (confirmed via the official changelog: "Added the dcmotor actuator for modeling DC motors.
   Supports optional electrical dynamics (inductance), cogging torque, thermal resistance
   variation, and LuGre friction," landed 3.7.0; this repo runs MuJoCo 3.9.0, confirmed via
   `python -c "import mujoco; print(mujoco.__version__)"`, so the feature is present in the
   installed version). This is a real, native option — but adopting it means switching the
   model's actuator class from the current pure-torque `<motor>` to `<dcmotor>`, which brings
   along back-EMF, optional inductance/cogging/thermal physics that model a literal DC motor, not
   just friction. That is a materially bigger, more invasive change to
   `assets/ur5e_torque/ur5e_torque.xml` — the file AGENTS.md explicitly calls "the centerpiece —
   never delete or silently regenerate" — than adding `frictionloss`/`damping` to a joint
   `<default>` class was (tonight's existing, already-landed, comparatively minimal friction
   asset change). Not recommended as the first move: it conflates "add LuGre friction" with "model
   the actuator as a literal DC motor," a much larger scope and validation surface than this plan
   needs.
2. **`mjcb_control`/`mjcb_passive` callbacks**: MuJoCo exposes a passive-force callback
   (`mjcb_passive`) explicitly documented as usable for velocity/position-dependent force fields
   (the docs' own example is custom fluid dynamics) — a LuGre bristle-deflection term could in
   principle be injected this way, with `z` maintained in Python between steps. This would let the
   *sim* itself carry a LuGre-friction ground truth distinct from the controller's own
   feedforward compensation of it (useful for a truly closed-loop sim validation: real dynamic
   friction present, controller trying to compensate it, exactly mirroring the real-hardware
   situation). Feasible, but real, new engineering scope (a Python-side callback wired into every
   place this repo currently steps MuJoCo directly, e.g. `simulation/ur5e_mujoco_torque.py`) that
   was not previously part of this repo's sim architecture.

**Recommendation**: do **not** build either of the above as a prerequisite. Keep sim on its
current static Coulomb+viscous friction model (`frictionloss`/`damping` in the MJCF, already
landed and validated per `docs/status/ur5e_sim_friction_modeling_2026-07-31.md`) as the *plant*
model, and make the LuGre term an opt-in **controller-side feedforward only**, validated in sim
exactly as `friction_feedforward` already is — i.e. sim validates "does the LuGre feedforward
degrade tracking/safety when the plant friction is the existing static model," not "does LuGre
feedforward perfectly cancel LuGre plant friction" (sim has no LuGre plant to cancel against).
This is an accepted, explicit sim/real asymmetry, not a hidden one: **sim and real would diverge
in how friction is modeled** (static Coulomb+viscous in sim's plant either way; static or LuGre
feedforward compensation on the controller side, controller-selectable) even though today both
already share the same `frictionloss`/`damping` MJCF values as their common starting reference
point. This mirrors this repo's own established comfort with model/reality gaps that are
explicit, characterized, and covered by real-hardware validation (e.g. `gravity_source`,
`coriolis_feedforward` — both flag-gated model refinements the controller can use independent of
what the sim plant does) rather than a new kind of risk. The real validation gap this creates
(a LuGre feedforward can only be checked "does it help/hurt against a *static*-friction plant" in
sim, never "does it correctly cancel *dynamic* friction" in sim) is real and should be stated
plainly to whoever reviews this before it ships to hardware — it is a genuine limit of what sim
can tell you here, not fully closed by this plan, only made explicit.

## 6. Validation plan

**Sim side**, adapted from the existing 4-category rigor sweep
(`tools/ur5e_pose_sweep_transport.py --categories canonical_grid long_holds large_displacements
torque_scale_robustness`, per AGENTS.md §3's own established pattern):

1. Smoke test first, matching `docs/status/ur5e_sim_friction_modeling_2026-07-31.md`'s own
   methodology: single dx=0.04m move-hold run, `friction_model: lugre` vs. `friction_model:
   static` vs. no feedforward, compare `final_x_error_m`, hold-phase torque decay shape, and
   specifically watch for a non-decaying or growing `_friction_z` trace (the numerical-stability
   risk flagged in §3.2).
2. Full 4-category sweep at the same poses already used for the static model's validation
   (`height_alpha` 0.2, 0.3, 0.5 — reuse `config/ur5e_mujoco_torque_osc_tuned_friction_ff.yaml`'s
   already-passing numbers, 96%/87%, as the bar the LuGre variant must not regress below).
3. Explicit regression check against the static model, not just an absolute pass rate — the
   static model is the working fallback (§7); LuGre only replaces it in a given config if it is
   shown to be strictly better or at minimum not worse on the existing envelope.
4. New test: an artificial "held against a wall" scenario (command a small torque insufficient to
   overcome a configured `Fs`, verify the LuGre term correctly predicts zero net motion and a
   growing-then-plateauing `z`) as a targeted unit test in `tests/unit/` (pure numpy, no MuJoCo
   needed) for the ODE step itself, independent of the full controller/sim integration.

**Real hardware**, mirroring tonight's own "start small, escalate carefully" discipline
(the exact discipline that caught tonight's static-model breakaway problem in the first place):

1. Confirm the `dt_s` prerequisite fix (§3.3) lands and is unit-tested before any real-hardware
   LuGre run — a silently-wrong or missing `dt_s` would make the `z` integration itself wrong in
   a way that is easy to not notice (no crash, just gradually incorrect friction compensation).
2. Re-run the exact same low-accel (0.05 m/s²) test the user is about to run tonight with the
   *static* model, this time with LuGre, at the smallest displacement already used tonight — a
   direct before/after comparison on the exact scenario that motivated this plan, not a new,
   larger test.
3. Only after that passes cleanly (no guard trips, real achieved-displacement fraction
   meaningfully closer to 100% than the static-model run), escalate to the
   `accel_duration_triangular`/`accel_duration_scurve` profiles that originally tripped the guard,
   still at the same small commanded accel (0.15 m/s²) before ever going larger.
4. At every step, the existing hardware safety stack (`EStopLatch`, `CartesianMoveMonitor`,
   `DeadlineMonitor`, `StaleStateMonitor`, `ImpedanceSafetyMonitor`) runs completely unmodified —
   this plan adds a feedforward term inside the torque the controller computes, not a new
   authority that can bypass or weaken what catches a bad torque command.

## 7. Explicit non-goals / scope boundaries

For whoever implements this later:

- **Must not touch `hardware/safety.py` or any guard threshold.** This is a feedforward torque
  term inside the existing controller, not a safety-architecture change.
- **Must not combine this with unrelated controller-logic changes.** No gain retuning, no other
  flag's default changed, in the same commit as the LuGre addition — per AGENTS.md §7 ("do not
  change training/eval logic and controller logic in the same commit; do not combine startup
  fixes with gain tuning") generalized to this feature. The `dt_s` plumbing fix (§3.3) is its own
  separate, prerequisite commit, not bundled with the LuGre term itself.
- **Must preserve `friction_feedforward`'s existing static model as a working fallback.** New
  named config (`config/ur5e_mujoco_torque_osc_tuned_friction_ff_lugre.yaml` or similar), never
  replacing or mutating `config/ur5e_mujoco_torque_osc_tuned_friction_ff.yaml`. `friction_model`
  defaults to `"static"` so no existing config's behavior changes by this addition landing.
- **Must not disable or change `friction_comp=True`** in any real `direct_torque()` call site
  (§3.4) — that is UR's own compensation path, already validated as part of the existing
  real-hardware call sequence; changing it is a separate, unrelated decision this plan does not
  make.
- **Must not adopt the MuJoCo native `dcmotor` actuator** as part of landing this feature (§5) —
  that is a much larger, separate scope decision about the plant model itself, not a friction
  feature.
- **Does not attempt online/adaptive identification of LuGre parameters at runtime.** Parameters
  are calibrated offline (§4) and loaded as static config values, same pattern as every other
  gain/parameter in this controller. `docs/status/nonlinear_controller_research_2026-07-31.md`
  §2 already flags online-adaptive (RLS) friction estimation as "the riskiest of this group" —
  unchanged, out of scope here.
- **Does not claim sim-side LuGre plant fidelity** (§5) — sim keeps its existing static
  Coulomb+viscous plant model; only the controller's feedforward compensation gains a LuGre
  option. This asymmetry is accepted explicitly, not silently.

## Rollback

N/A — no files were changed to produce this document beyond this plan itself. Rollback for a
future implementation session: revert whichever of the config/controller_core/state_types/
hardware files were touched, per that session's own commit boundaries (kept separate per the
non-goals above specifically so each piece has an independent, clean rollback).

## Sources consulted

- Canudas de Wit, Olsson, Åström, Lischinsky, "A New Model for Control of Systems with Friction,"
  IEEE Transactions on Automatic Control, 1995 (the original LuGre model).
- [Parameter Identification of LuGre Friction Model for Robot Joints](https://www.scientific.net/AMR.479-481.1084)
- [A new iterative identification algorithm for estimating the LuGre friction model parameters](https://www.sciencedirect.com/science/article/abs/pii/S0094114X24002246)
- [Friction dynamics identification based on quadratic approximation of LuGre model](https://link.springer.com/article/10.1007/s11071-024-09331-2)
- [Dynamic modeling and friction parameter identification of a hybrid robot considering active and passive joint frictions](https://www.sciencedirect.com/science/article/abs/pii/S0094114X25004379)
- [Extending a dynamic friction model with nonlinear viscous and thermal effects (DLR)](https://elib.dlr.de/120472/1/dynamic_friction_model.pdf)
- [MuJoCo Changelog](https://mujoco.readthedocs.io/en/latest/changelog.html) — `dcmotor` actuator
  with optional LuGre friction, landed MuJoCo 3.7.0.
- [MuJoCo DC Motor technical note](https://mujoco.readthedocs.io/en/latest/_static/dcmotor.pdf)
- [Modeling static friction/stiction — MuJoCo GitHub issue #1366](https://github.com/google-deepmind/mujoco/issues/1366)
- Local, already-cited within this document: `docs/status/nonlinear_controller_research_2026-07-31.md`,
  `docs/status/direct_torque_controller_phase_profiling_2026-07-31.md`,
  `docs/status/ur5e_sim_friction_modeling_2026-07-31.md`,
  `docs/status/friction_ff_alpha_0.2_0.3_sweep_2026-07-31.md`, `docs/hardware/AUTO_TUNING_PLAN.md`
  (style/rigor reference), `AGENTS.md` §3–§4.
