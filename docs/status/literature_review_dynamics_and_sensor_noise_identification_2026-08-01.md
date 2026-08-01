# Literature review: system-identification methods for manipulator dynamics and sensor noise/drift

**Status:** research only, no code. Produced by an adversarially-verified deep-research pass
(5 search angles, 22 sources fetched, 25 claims verified via 3-vote adversarial check: 21
confirmed, 4 refuted, 0 unverified). Saved here so findings are citable from other docs (this
repo's own discipline: no fabricated citations, verify against a real artifact before citing).
**Date:** 2026-08-01.

## Scope

Two connected questions, prioritizing system-identification/real-data-calibration methods over
general surveys: (1) modeling/identifying manipulator dynamics forces not captured by a standard
rigid-body model (Coriolis, friction — Coulomb/viscous/Stribeck/LuGre/GMS — actuator compliance,
backlash) from real trajectory data; (2) modeling real sensor noise/drift (RTDE-style position/
velocity noise, encoder quantization, timestamp jitter, slow bias drift) and injecting a matching
model into a simulator.

## Area 1 — forces/dynamics identification: strong, real results

**Most directly relevant: Clochiatti et al., "Electro-mechanical modeling and identification of
the UR5 e-series robot," Robotica, 2024.**
(https://www.cambridge.org/core/journals/robotica/article/electromechanical-modeling-and-identification-of-the-ur5-eseries-robot/1AE5BAE866D9046F79C4B159BEA2B45F,
open PDF: https://air.uniud.it/retrieve/2a8573a5-4589-4c0e-92b9-14c940095023/electro-mechanical-modeling-and-identification-of-the-ur5-e-series-robot.pdf)
UR5e joint friction modeled as viscous plus an **asymmetric smooth Coulomb term whose value
depends on the direction of mechanical power flow through the harmonic-drive reducer**
(motor-to-load vs. load-to-motor), not on velocity sign alone — qualitatively different from this
repo's current symmetric `tanh(qd/deadband)`-Coulomb+viscous `friction_feedforward` term
(`controller_core/x_axis_cartesian_impedance.py`). Because the UR5e has no joint torque sensor
(same constraint this repo has), friction coefficients and motor torque constants were identified
*jointly* (not sequentially, to avoid error accumulation) via linear least-squares/pseudo-inverse
regression on RTDE-logged motor current data across all 6 joints. Confidence: high (3-0
verification, independently re-verified via two source mirrors). **Caveat**: this paper's specific
cross-validated RMS-current-error/R² numbers were explicitly REFUTED under adversarial
verification (0-3 vote) — trust the described methodology, not those specific accuracy figures.

**Full UR5 dynamics identification: Raviola et al., RAAD 2021/Springer**
(https://www.researchgate.net/publication/351329080_Identification_of_a_UR5_Collaborative_Robot_Dynamic_Parameters).
78-parameter UR5 model (13/joint including Coulomb-as-`tanh(qd/0.001)` and separate viscous
friction, no Stribeck/LuGre term) identified via ordinary least-squares on a genetic-algorithm-
optimized 5th-order Finite Fourier Series excitation trajectory; joint torques computed from motor
current × gear ratio × torque constant (no torque sensor), logged at 125 Hz. Confirms the standard
real-data system-ID recipe on a UR5 specifically. The paper itself flags omission of static
friction as a known limitation — consistent with this repo's own finding that a purely velocity-
dependent friction model misses the real breakaway/stiction signature found on hardware tonight.
**Caveat**: a specific held-out cross-validation error number was REFUTED (0-3) — the qualitative
limitation is confirmed, the specific number is not.

**Franka Panda**: Gaz, Cognetti, Oliva, Robuffo Giordano, De Luca, IEEE RA-L 2019 — OLS fit of
linearly-parametrized dynamics, then penalty-based constrained optimization to recover physically
consistent link mass/CoM/inertia parameters (triangle-inequality-consistent). Widely cited
canonical reference for this pipeline shape; high confidence (RA-L venue, multiple independent
corroborations).

**Dynamic friction (LuGre/GMS) — real data exists, but on KUKA/ABB, not UR-series:**
- Mahmoudkhani, Gorenstein, Ahmadi, *Mechanism and Machine Theory* 2024
  (https://www.sciencedirect.com/science/article/abs/pii/S0094114X24002246): least-squares
  iterative algorithm jointly identifying LuGre parameters + joint inertial properties from
  *continuous* trajectory data (not many discrete constant-velocity experiments), validated on a
  real KUKA KR90 R3100 joint 5, ~10-iteration convergence, accurate 0.009–2.13 rad/s on held-out
  data. Directly relevant to `docs/hardware/LUGRE_FRICTION_MODEL_PLAN.md`'s own §4 calibration
  proposal (which currently plans discrete-velocity Stribeck sweeps) — this continuous-trajectory
  method may be more practical for limited real-lab time.
- Vantilborgh et al. (now IEEE Xplore, https://arxiv.org/html/2412.15756): a learned probabilistic
  state-space friction model (friction law + latent state both parameterized as neural networks,
  fit via EM + Sequential Monte Carlo) **beat** identified LuGre/GMS/Stribeck on held-out real
  KUKA KR6 R700 data (MSE 0.11 vs. 0.66/1.43/1.15), and generalized far better out-of-distribution
  (classical models' error grew 8–24× from short-horizon to full-trajectory eval, vs. ~4.6× for
  the learned model). Real evidence that even literature's best classical dynamic-friction fits
  aren't clearly superior to a data-driven alternative once honestly validated — relevant context
  for deciding LuGre vs. the residual-torque-regression direction this session also pursued.
- Bittencourt & Gunnarsson, ASME JDSMC 2012 (older than the ~10yr window, kept as foundational):
  real ABB IRB 6620 data — standstill/Coulomb friction increase ~linearly with applied load
  torque; standstill friction and Stribeck velocity increase linearly while viscous slope
  decreases exponentially with joint temperature. Adding load+temperature terms to a static LuGre-
  derived model cut mean/worst-case prediction error ~10× on held-out data. Load-dependence is
  plausible to add here (joint load torque is already computable from existing dynamics, no new
  sensing); temperature-dependence is out of scope (this sim has no thermal model).

**Newer, less certain (2025–26, mostly preprint, not fully peer-reviewed)**: hybrid analytical +
learned residual dynamics — symbolic regression/SINDy residuals on top of an analytical rigid-body
model (real Barrett WAM data; note a specific SR-vs-SINDy-vs-NN generalization comparison claim
from this same source was REFUTED, 0-3); a structured Euler-Lagrange residual decomposition with
online Bayesian adaptation; and a PINN with a current-to-torque physical prior validated on
**real UR5 and UR10e**, reporting 31–37% modeling / 40–52% tracking-error improvement (Sun et al.,
FITEE 2025, https://link.springer.com/article/10.1631/FITEE.2500254) — most relevant to this
repo's no-torque-sensor constraint, but its headline number only narrowly survived verification
(2-1 vote) — treat as promising, not settled.

**Not found in this pass**: any full LuGre/GMS identification specifically on UR-series hardware
(only KUKA/ABB); any confirmed claim on actuator compliance, cable/harness effects, or backlash
identification for collaborative arms — named in scope, zero surviving evidence.

## Area 2 — sensor noise/drift modeling: real gap, not a "doesn't exist" finding

This came back essentially empty. Nothing on RTDE-specific noise/drift characterization or
matched noise-injection-into-simulator methodology survived verification. **This is this
particular search pass's own evidence gap, not proof the literature doesn't exist** — flagged
explicitly by the research workflow itself. Two unverified leads worth a targeted follow-up if
this is revisited: `robosuite`'s documented `sensor → corrupter → delayer → filter` observable
pipeline (https://robosuite.ai/docs/modules/sensors.html) — a concrete, already-implemented
pattern for exactly this, built on MuJoCo — and general sim-to-real domain-randomization
literature. Neither got deep adversarial verification this round.

## Relevance to this repo's own open threads

- `docs/hardware/LUGRE_FRICTION_MODEL_PLAN.md` cites this same gap (no UR5e-specific LuGre table)
  and already has a full, ready-to-implement design — this review adds two things that plan didn't
  have: (a) the Mahmoudkhani continuous-trajectory identification method as a possibly more
  practical calibration procedure than the plan's current discrete-velocity-sweep proposal, and
  (b) the Vantilborgh finding that a learned friction model beat LuGre/GMS on real data, worth
  weighing before committing significant real-lab time to LuGre calibration specifically.
- The Clochiatti asymmetric-Coulomb model is a real, simpler, UR5e-specific alternative (or
  complement) to full LuGre worth prototyping in sim before deciding whether to invest in the
  fuller dynamic-friction model.
- `docs/status/nonlinear_controller_research_2026-07-31.md`'s residual-torque-regression
  recommendation is independently reinforced by the Vantilborgh result above (learned residual
  approaches winning over hand-picked classical friction models on real robot data, in a
  peer-reviewed, adversarially-verified source).

## Sources (primary/secondary quality only; full list including filtered-out ones in the raw
workflow output, not reproduced here)

- Clochiatti et al., Robotica 2024 (UR5e asymmetric Coulomb friction + joint current ID)
- Raviola et al., RAAD 2021/Springer (UR5 78-parameter dynamics ID)
- Gaz, Cognetti, Oliva, Robuffo Giordano, De Luca, IEEE RA-L 2019 (Franka Panda dynamics ID)
- Mahmoudkhani, Gorenstein, Ahmadi, Mechanism and Machine Theory 2024 (continuous-trajectory
  LuGre + inertial co-identification, real KUKA KR90)
- Vantilborgh et al., IEEE (learned probabilistic friction model vs. LuGre/GMS/Stribeck, real
  KUKA KR6)
- Bittencourt & Gunnarsson, ASME JDSMC 2012 (load/temperature-dependent static friction, real
  ABB IRB 6620)
- Sun et al., FITEE 2025 (PINN current-to-torque residual dynamics, real UR5/UR10e)
- Mower, Zong, Bou-Ammar (symbolic regression / SINDy hybrid residual dynamics, real Barrett WAM)
- robosuite sensor pipeline docs (https://robosuite.ai/docs/modules/sensors.html) — noise
  injection reference pattern, not independently verified this pass
