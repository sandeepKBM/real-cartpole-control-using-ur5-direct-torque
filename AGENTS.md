# AGENTS.md

Working playbook for agents operating in `/common/users/ss5772/real_Cartpole`.
Update rule: edit in place, keep these sections, date-stamp material changes. Do not append
chronological logs here — that pattern was retired 2026-07-03; the old log is preserved at
`docs/archive/AGENTS_HISTORY.md`. Material hardware refresh: 2026-07-14.

## 0. Current objectives — cartpole flip + hold (set 2026-08-14)

Two goals, pursued in parallel. **Both are SIM-ONLY and wall-clock cost is explicitly NOT a
constraint** (user directive) — do not trade correctness, guards, or rigor for speed here, and
do not report a per-cycle compute budget as a blocker for either goal.

**Both goals share the same architecture** — a cascade, not a monolith:

```
  swing-up law (energy shaping)  ─┐
                                  ├─► desired CART ACCELERATION ─► low-level controller ─► joint torques
  LQR (catch + hold, inverted)   ─┘
      switch when |theta - theta_inv| AND |thetadot| are inside the LQR's measured envelope
```

The high-level law never sees joints; the low-level controller never sees the pendulum. The
handoff is a source switch, not a controller swap.

### Goal 1 — flip AND hold, at a NON-singular wrist_2

- **Pose**: `ARM_Q0` with `wrist_2` moved OFF the wrist singularity. `ARM_Q0` as committed is
  `[-2.3688, -2.1801, -1.8838, -0.7962, 0.004714693, 0.0206]`, whose `wrist_2 = 0.270 deg`,
  `cond(J) = 1395.76`, `sigma_min = 1.485e-3` (measured 2026-08-14) — that IS the singularity.

  **ANSWER: `wrist_2 = -90 deg` (-1.5707963 rad), with the LOCAL-X-HINGE pendulum.**
  Measured 2026-08-14. All three constraints hold simultaneously, and this is the only region
  where they do:

  | quantity | value at `wrist_2 = -90 deg` | vs `ARM_Q0` |
  |---|---|---|
  | `cond(J)` | **7.20** | 1395.76 -> **194x better** |
  | tool Z elevation (the FACE, -> sky) | **+81.5 deg** | face points up |
  | hinge tilt off horizontal | **0.2 deg** | perfectly horizontal |
  | gravity torque retained | **100.0%** | fully live pendulum |

  It is a BROAD optimum, not a knife-edge: `wrist_2` in `[-80, -100] deg` all give `cond(J) ~7.2`,
  face elevation `>= +76.9 deg`, and `>= 98.5%` gravity torque. For reference `cond(J) = 7.20`
  is essentially the old well-conditioned pose's `6.93` — i.e. this is as well-conditioned as the
  pose that produced the only flip this repo ever achieved, but at `ARM_Q0`'s base configuration.

  **WHICH AXIS IS THE HINGE DEPENDS ON THE ASSET — this is the trap.** For
  `pendulum_attachment.xml` (and `_realfriction.xml`) the hinge is **local Z = tool Z**; for
  `pendulum_attachment_realrod.xml` (and `_longrod.xml`) it is **local X = tool X**. A hinge
  pointing skyward is a **DEAD pendulum** (rod swings in a horizontal plane, where gravity exerts
  no torque about it), so "face at the sky" and "live pendulum" can only BOTH hold when the face
  normal and the hinge are different axes — which is exactly why Goal 1 needs the local-X-hinge
  asset. Measured, both poses:

  | asset | hinge axis | `w2=0.27 deg` | `w2=-90 deg` |
  |---|---|---|---|
  | `pendulum_attachment.xml` / `_realfriction.xml` | local Z | live, 100% | **DEAD, 14.8%** |
  | `pendulum_attachment_realrod.xml` | local X | **DEAD, 12.7%** | **live, 100.0%** |

  An earlier recommendation of `wrist_2 = +18 deg` in this file was **WRONG** and has been
  removed: it assumed the hinge is tool Z, which is true only for the default asset. With the
  local-X-hinge asset the entire trade-off inverts. Always confirm which axis the chosen asset
  hinges about before reasoning about pose.

  **VALIDATED IN CLOSED LOOP 2026-08-14 — THIS CONFIGURATION FLIPS.** See the result block below.

### GOAL 1 SWING-UP: FLIP ACHIEVED 2026-08-14 (guards ON, independently reproduced)

First guard-clean flip of the 0.12 m rod in this repo. Single-axis energy shaping (NOT curved),
`controller_kind: impedance` (plain OSC — `cond(J)=7.2` here, so OSC is the correct pairing),
`config/ur5e_mujoco_torque_osc_tuned_friction_ff.yaml`,
`assets/ur5e_pendulum/pendulum_attachment_realrod.xml`, pose `ARM_Q0` with `wrist_2 = -90 deg`.

```
flipped                       = True
min_theta_dist_from_inverted  = 0.1875 rad   (10.7 deg from vertical)
guard_fired                   = False        <-- NOT a guards-off result
a_max                         = 7.1437 m/s^2 (INTERIOR to its bound, not pinned)
k_e = 277.748   k_pos = 15.797   k_vel = 8.180
kick_amplitude_m = 0.1457   kick_duration_s = 0.5952
cond(J): start 7.203 -> max 8.170  (growth 1.13x, no singularity chasing)
max_abs_x_dev_m = 0.1871
pendulum tip world z: min +0.1950 m  -> CLEARS THE FLOOR by 19.5 cm
```

Reproduced independently from the saved parameters (identical `min_theta_dist` to 4 decimals),
and tip world-z was tracked explicitly because floor penetration is SILENT here.

**Root cause of every prior null: the `a_max` SEARCH BOUND, not physics.** The bound was
`(0.3, 3.0)` m/s^2 and searches pinned at 2.990 while firing NO guard. Measured ceilings at this
pose show how wrong that bound was:

| quantity | value | limited by |
|---|---|---|
| max cart acceleration | **52.1 m/s^2** | `wrist_1` torque (280.1 N along the pump axis) |
| max pump speed | **1.023 m/s** | the `\|qd\| > 3.0 rad/s` guard |
| task inertia along pump axis | 5.376 kg | — |

Acceleration does not set energy directly (speed does, and `E ~ v^2`) — it sets **how much
displacement is burned reaching that speed**: at `a_max=3.0` the arm needs **0.174 m** to reach
1.023 m/s, which exceeds the drift budget, so it could never reach its own speed limit; at
`a_max=7.14` it needs ~0.073 m. That is the whole story. At 1.023 m/s the available kick energy
is ~1.5x `E_top`, so energy was never the shortage — phasing and displacement were.

Progression, same pose/asset, everything else equal:

| run | `a_max` bound | min dist from inverted | flipped |
|---|---|---|---|
| stock `ARM_Q0` (prior documented ceiling) | 3.0 | 2.24 rad | no |
| `wrist_2=-90`, `osc_tuned` | 3.0 | 2.187 rad | no |
| `wrist_2=-90`, friction_ff | 3.0 | 1.365 rad | no |
| **`wrist_2=-90`, friction_ff** | **12.0** | **0.1875 rad** | **YES** |

`--a-max-upper` is now a CLI flag on `pendulum_swingup_energy_shaping.py` (default 3.0 preserves
historical behavior). **If a search reports best `a_max` within ~1% of its ceiling while firing no
guard, the BOUND is the constraint — re-run wider before concluding anything about feasibility.**

Open on this result: `kick_amplitude_m = 0.1457` sits near its own `0.15` bound and may itself be
pinning, so there may be more margin available. The LQR catch/hold half is still NOT done —
"flip" is not "flip AND hold", and the 2-D (`theta`, `thetadot`) capture envelope remains
unmeasured, which is still the likeliest silent failure of the handoff.
- **Tool**: `assets/ur5e_pendulum/pendulum_attachment_realrod.xml` — 0.12 m rod, **local-X
  hinge**. Selected by physics, not by name: at `wrist_2 = -90 deg` its hinge is horizontal
  (100% gravity torque) while the tool face points skyward, which the local-Z-hinge assets
  cannot do simultaneously. This is the asset built by task #137 ("0.12 m rod + local-X hinge,
  the actually-working config").
- **High level**: LQR, emitting desired cart acceleration.
- **Low level**: OSC, **or** the reduced-task single-axis (tool-Y) controller with the CBF.
  Either is acceptable; pick on measured evidence, not preference.
- **Motion**: single axis, in that same frame, **guards UP**.
- **Done means**: the pole flips up AND is held inverted. Not one or the other.
- **Reference**: `outputs/pendulum_renders/phase_locked_diagnostic.mp4` — treat as a **method /
  motion** reference (phase-locked, adaptive-resonance single-axis pumping), **not** a pose or a
  result reference. Traced 2026-08-14 to
  `tools/diagnostics/render_phase_locked_diagnostic.py`, which runs at `ARM_Q0` as-is
  (`wrist_2 = 0.270 deg`, `cond(J) = 1395.76` — i.e. the SINGULAR pose this goal moves away
  from), with `controller_kind="impedance"` (plain OSC), the default 0.12 m
  `pendulum_attachment.xml`, and `config/ur5e_mujoco_torque_osc_hanging_pose_friction_ff_wrist_orient.yaml`.
  Frames at t~0.2 s and t~12 s of its 14 s are visually near-identical (arm barely moves, no
  visible swing) and it carries no HUD — so do **not** cite it as a flip or as a passing result
  without re-running it and reading real numbers.

### Goal 2 — 2-axis swing-up, at the SINGULAR wrist_2

- **Pose**: `ARM_Q0` as-is, wrist_2 at the singularity (`cond(J) = 1395.76`).
- **Controller**: the 2-axis controller (`controller_core/x_task_yz_corridor_qp/`), which exists
  precisely to survive this pose. **Gated on the CRITIC pass** — §5's no-exceptions rule
  applies; nothing about that controller is currently independently verified.
- **Reference behavior**: `outputs/pendulum_renders/swingup_flip_pan_lift_elbow_noguard.mp4`.
- **Pendulum**: `assets/ur5e_pendulum/pendulum_attachment.xml` (the default 0.12 m, **local-Z
  hinge**). Determined by the same physics as Goal 1's, in the opposite direction: at `ARM_Q0`'s
  singular `wrist_2 = 0.270 deg`, tool Z is horizontal, so the local-Z hinge is horizontal and
  the pendulum is fully live (**100.0%** gravity torque) — while the local-X-hinge `realrod`
  asset is **DEAD** there (12.7%). So the two goals genuinely use different attachments, each
  the live one at its own pose.
- **CRITICAL**: that reference video was produced with **guards OFF**
  (`config/_candidate_split_pan_lift_elbow_*` drives the task from joints `[0,1,2]` =
  pan/lift/elbow, `split_base_wrist_task_dims: [0,1,2]`). Per this repo's standing rule, **a
  result that needs guards disabled is a NEGATIVE result**. That video is a direction and a
  motion reference, NOT a passing result. Goal 2 must reproduce it with **guards ON**.

### Why pan/lift/elbow sidesteps the wrist singularity

That config family drives the Cartesian task using **only** joints `[0,1,2]`, never the wrist. A
wrist-2 singularity costs the *wrist* its authority; if the wrist is not carrying the task in the
first place, `cond(J6)` of the full 6x6 Jacobian overstates the problem for that joint subset.
This is the structural reason Goal 2 is viable at a pose where plain 6-DOF OSC is mismatched —
and why `cond(J6)` is the wrong number to judge that config by.

### Pose <-> controller pairing (do not mix these up)

| pose | cond(J) | correct low-level controller |
|---|---|---|
| old / well-conditioned pose | 6.93 | plain OSC is fine — the only flip this repo ever achieved |
| `ARM_Q0` (wrist_2 singular) | 1395.76 | 2-axis controller, or a pan/lift/elbow joint subset |

A real mistake made 2026-08-14, recorded so it is not repeated: a ~40-minute swing-up search ran
at `ARM_Q0` (`cond(J)=1396`) using **plain OSC** (`controller_kind: "impedance"` +
`config/ur5e_mujoco_torque_osc_tuned.yaml`) — the hardest pose paired with the controller known
to be structurally mismatched for it. A null from that pairing cannot distinguish "the swing-up
law is insufficient" from "this controller cannot hold this pose". **Check `CONTROLLER_KIND` and
the config path against the pose at dispatch time, not after.** Verifying that a search is
*running* is not the same as verifying it is running the *right thing*.

### Swing-up method: energy shaping (Astrom-Furuta). NOT RL.

```
u = clip( -k_e*thetadot*cos(phi)*(E_top - E)  -  k_pos*(x - x0)  -  k_vel*v_ref,  -a_max, +a_max )
```
`u` is a cart acceleration, integrated twice into the position target the low-level controller
tracks. `phi = theta - theta_hanging` (0 hanging, +-pi inverted);
`E = 0.5*I*thetadot^2 + M*g*r*(1-cos phi)`; `E_top = 2*M*g*r`.

Term 1 is a Lyapunov construction, not a heuristic: since `Edot = -M*r*a*thetadot*cos(phi)`,
choosing `a` proportional to `-thetadot*cos(phi)*(E_top-E)` gives
`Edot ∝ (E_top-E)*(thetadot*cos phi)^2 >= 0` — energy can only increase, and the drive
self-limits to zero at the top. Term 2 is the leash that keeps net drift inside the guard budget
(the reason this beats a single kick, which spends its whole displacement budget one-way and
still falls short of `E_top`). A seed kick is **mandatory**: at rest `thetadot = 0` makes term 1
identically zero. Reference implementation:
`tools/diagnostics/pendulum_toolY_swingup_search.py::run_energy_shaping_trial`.

### CURVED (2-D) pivot motion — measured 2026-08-14, changes the feasibility argument

The single-axis law above uses only horizontal pivot acceleration. That is leaving half the
available authority on the table. For a pivot accelerating with components `(a_par, a_z)` the
pivot-frame pseudo-force adds to gravity, so

```
I*thetaddot = -M*r*[ (g + a_z)*sin(phi) + a_par*cos(phi) ] - b*thetadot
Edot        = -M*r*thetadot*[ a_par*cos(phi) + a_z*sin(phi) ]
```

Verified against the compiled model (hinge torque vs a unit pivot acceleration, `ARM_Q0`,
`pendulum_attachment.xml`): `a_par` traces `-cos(phi)` and `a_z` traces `+sin(phi)`, with
**equal peak authority — 0.00282 vs 0.00284 Nm per 1 m/s^2, within 0.7%**:

| phi | tau from `a_par` | tau from `a_z` |
|---|---|---|
| 0 deg (hanging) | **-0.00282** | 0.00000 |
| 90 deg (horizontal) | +0.00036 (~0) | **+0.00284** |
| 180 deg (inverted) | +0.00282 | 0.00000 |

Two consequences:

1. **The two axes are complementary, not redundant.** Vertical drive is at MAXIMUM authority
   exactly where horizontal drive has NONE (rod horizontal), and vice versa at the bottom. Since
   `Edot = -M*r*thetadot*(a . n_hat)` with `n_hat = (cos phi, sin phi)` normal to the rod, the
   optimal instantaneous drive direction is PERPENDICULAR TO THE ROD — and that direction
   rotates as the rod swings, so following it traces a CURVE.

2. **A curve sidesteps the bandwidth objection, which was about REVERSAL, not speed.**
   Straight-line pumping must stop the cart dead and reverse every half-period (0.290 s for the
   0.12 m rod, outside the arm's ~0.5 s response — the tension logged in "Open" item 3). A
   curved/elliptical path never reverses: the pivot keeps moving and only TURNS, so the demand
   becomes a turning rate instead of a stop-start. **Do not cite the 0.290 s half-period against
   a curved trajectory — that budget was measured for linear reversal and is the wrong yardstick
   here.** This is untested as a swing-up strategy; it is a measured-authority result plus a
   mechanism argument, not yet a flip.

3. **Shape the loop to the floor, do not center it.** Floor clearance at `ARM_Q0` is 6.3 cm DOWN
   and effectively unlimited UP (see the clearance section above), so the loop should be an
   up-biased ellipse rather than a centered circle. This costs nothing: the `sin(phi)` term
   depends on the PHASING of vertical acceleration, not on where the loop is centered.

This is the strongest available argument for the 2-axis controller: it is not merely a way to
survive the wrist singularity, it is the machinery needed to command a 2-D curved pump at all.

**Do not use RL for this.** `rl_gain_scheduling/` is a gain-scheduling env (action = gain
multipliers, no pendulum in it) and is not reusable here. Its record on a strictly easier
problem: 0/20, 0/20, 1/20 across three reward redesigns and ~4.4M steps, versus a fixed-gain
baseline at 100%. Search with `differential_evolution`, per §7.

### Floor clearance at `ARM_Q0` is only 6.3 cm — and floor penetration is SILENT

Measured 2026-08-14 with `pendulum_attachment.xml` at `ARM_Q0`: the rod tip sits **0.0634 m**
above the floor at the HANGING equilibrium (EE at z=0.1839), and hanging is the tip's lowest
point over a full 360 deg hinge sweep. That is the pose swing-up starts from and returns to
every half cycle, so any DOWNWARD arm drift eats straight into it.

It is already being violated in a plain transport move. A 2-axis-controller run at
`dx = +0.12 m` drives the tip to **-0.0142 m at t=3.22 s — 1.4 cm BELOW the floor** (the same
run's Z drift was 0.0771 m; 0.0634 - 0.0771 = -0.0137, which is where the number comes from).
The mirrored `dx = -0.12 m` run never drops below its start (min +0.0634 m) — the same
X-direction asymmetry §7 warns about.

**Nothing detects this.** The pendulum geoms are declared `contype="0" conaffinity="0"`, so the
rod passes THROUGH the floor with no contact force, no penetration warning, and no error — it
only shows up if you render it or explicitly track the tip site's world z. Do not assume a run
that "passed" kept the hardware out of the table.

Note the near-coincidence: the Z-corridor guard trips at 0.06 m and the clearance is 0.0634 m,
so at THIS pose a Z-guard trip is roughly the floor-contact threshold — margin behind the guard
is ~3 mm, not a comfortable buffer. That is luck, not design, and it will not hold at another
pose or with a longer rod (the 0.30 m rod is 18 cm longer and would be through the floor at
rest here).

### LQR CATCH: FIXED 2026-08-14 — `frictionloss` was HIDING the instability from the linearizer

**Both halves of Goal 1 now work.** Swing-up delivers `0.1875 rad @ thetadot = 0.0102 rad/s`; the
LQR holds from that state with margin on both sides.

`tools/diagnostics/pendulum_balance_torque_lqr.py` previously FELL at every perturbation
(0.05-0.40 rad, fall times 0.18-0.56 s) at this pose/asset. It was **a wrong MODEL, not wrong
weights** — sweeping `r_weight` over 2e9 (1e6 -> 0.0005) barely moved the closed-loop eigenvalues,
which is the signature to watch for.

Root cause: Coulomb `frictionloss` (0.001 Nm on the hinge) completely dominates at the
microscopic velocities `mjd_transitionFD` perturbs with, so the linearizer saw a nearly-STUCK
hinge and reported the inverted equilibrium as almost non-divergent. Measured (dt=0.002,
`omega=10.8334 rad/s`), and note every corrected entry lands on its analytic value:

| entry | as-modelled | frictionloss zeroed for linearization | analytic |
|---|---|---|---|
| `A[thd,thd]` | 0.810829 | **0.999157** | 0.99916 |
| `A[thd,th]` | 0.023741 | **0.234904** | `w^2*dt` = 0.234725 |
| max abs eig | 1.000252 | **1.021474** | `exp(w*dt)` = 1.021903 |

0.810829 means the linearizer believed the hinge loses **19% of its velocity every 2 ms**. Viscous
damping cannot do that (`exp(-b*dt/I) = 0.99916`, 0.08%/step) — it is the Coulomb term. An LQR
designed against that plant has ~1-2 s closed-loop time constants against a pendulum whose real
divergence time constant is `1/omega = 0.092 s`, i.e. 10-20x too slow. It was correctly solving
for a plant that does not fall.

Fix: `linearize_and_design_lqr(..., zero_hinge_frictionloss_for_linearization=True)` /
CLI `--zero-frictionloss-for-linearization`. It zeroes the hinge `frictionloss` for the
`mjd_transitionFD` call ONLY and restores it in a `finally`; **the simulated plant keeps its
friction throughout**. Result at `wrist_2=-90` with `realrod`:

| perturbation | before | after |
|---|---|---|
| 0.05 rad | FELL @ 0.54 s | **SURVIVED** (final err 0.0184) |
| **0.1875 rad** (the swing-up's arrival) | FELL @ 0.30 s | **SURVIVED** (final err -0.0017) |
| 0.30 rad | FELL @ 0.22 s | **SURVIVED** (final err -0.1130) |
| 0.40 rad | FELL @ 0.18 s | **SURVIVED** (final err -0.0444) |

At `r_weight=100` every final error is `<= 0.0162 rad`. Default is **False**, preserving
previously-derived gains bit-for-bit (verified: the default path still FELL at t=0.296 s,
identical to the pre-change run).

**This is a GENERAL MuJoCo trap, not specific to this pose**: any `mjd_transitionFD`
linearization of a joint carrying `frictionloss` will understate that joint's dynamics, because
FD perturbations live entirely inside the stiction band. The long-rod LQR (task #130) was derived
through the same code path and is likely affected — it was NOT re-derived here, and the default
was deliberately left off rather than silently changing those results.

### `min_theta_dist_from_inverted` IS THE WRONG OBJECTIVE FOR A HANDOFF (found 2026-08-14)

That metric rewards REACHING inverted, so it also rewards a fast fly-THROUGH — which is useless
for a catch. Measured, same pose/asset, both flips guard-checked and independently reproduced:

| | single-axis energy shaping | curved / 2-axis |
|---|---|---|
| `min_theta_dist_from_inverted` | 0.1875 rad | **2.6e-05 rad** (looks 7000x better) |
| **`thetadot` at closest approach** | **+0.0102 rad/s** | **+1.0504 rad/s** (100x WORSE) |
| time to top | 2.206 s | 0.680 s |
| time inside the 0.4 rad ball | 0.918 s | 0.410 s |
| guard | none | `\|Y-Y0\| > 0.03 m` FIRES at 14 s |

The single-axis law "loses" on the metric while producing a far better catch state, because its
self-limiting energy term (`drive -> 0` as `E -> E_top`) genuinely PARKS the pendulum at the top.
The curved law reaches the top faster and with 32x lower `k_e`, but arrives moving.

**Any swing-up objective intended to feed the LQR must penalize `|thetadot|` at closest approach,
not just `|theta - theta_inv|`.** A run that reaches inverted at speed has not solved the problem
it appears to have solved.

**CORRECTED 2026-08-16 — the quantity above is still wrong, though in a smaller way.** Measuring
the 117-cell capture envelopes directly shows they are not a ball around the origin but an
**ANTI-DIAGONAL BAND**: capture needs `theta` and `thetadot` of OPPOSITE sign, i.e. the pendulum
must be moving TOWARD vertical. The governing quantity is the inverted pendulum's **unstable
mode**

```
s = thetadot + omega*phi          omega = sqrt(m*g*r/I) = 10.8334 rad/s
```

and the single threshold `|s| <= 1.2` classifies **94%** of the 117 cells (drift 0.03; 87% at
drift 0.06). The stable-mode coordinate `thetadot - omega*phi` has NO predictive power —
captured median 3.92 vs failed 2.84, i.e. backwards — so this is that specific mode, not merely
a better-fitting weighted combination.

Consequences:

- **Arriving at rest at the top is NOT the best arrival.** The swing-up's celebrated
  `(phi=0.1875, thetadot=0)` gives `|s| = 2.03`, **1.7x OVER threshold** — outside the band
  despite looking optimal on every metric used so far. Arriving with a small velocity of the
  correct sign is strictly better than arriving at rest.
- Penalizing `|thetadot|` alone actively drives a search toward the `thetadot = 0` column, which
  is the **worst** column of the band at any nonzero `phi`. Use `|thetadot + omega*phi|`.

### GOAL 1 COMPLETE 2026-08-16 — FLIP **AND** HOLD, guards ON, stock drift tolerance

End-to-end in ONE continuous rollout (one `MjData`, one adapter, **no teleport**):
`tools/diagnostics/pendulum_flip_catch_hold.py`, pose `ARM_Q0` with `wrist_2 = -90 deg`,
`pendulum_attachment_realrod.xml`, `config/ur5e_mujoco_torque_osc_tuned_friction_ff.yaml`
(`max_abs_*_drift_m = 0.03` — **no guard loosening anywhere**).

```
switch at t = 2.130 s   phi = -14.283 deg   thetadot = +1.8518 rad/s   |s| = 0.849
held 10.0 s, settles to final |phi| = 0.15 deg
guard_fired = False
drift  x = 0.1871   y = 0.0239   z = 0.0095 m      cond(J) 7.203 -> 8.170
tip min world z = 0.1950 m (clears floor by 19.5 cm)   arm contacts = 0
```

**K=0 counterfactual on the identical swing-up and switch point: falls to 179.98 deg and trips
the orientation guard at t=7.702 s.** The hold is control, not passive hinge friction — this
repo has retracted that exact result before, so the counterfactual is not optional.

Note the drift-0.06 balance config is **not required**: measured Y excursion is 0.0239 m, inside
the stock 0.03 guard. The widened config was only ever the LQR search's own config.

Two things the seam needed that neither half revealed alone: (1) the LQR half had only ever been
tested from a **teleport** (pendulum placed, arm at rest, reference zeroed) — the real arrival
has the arm mid-stroke, carrying cart velocity, tracking a reference running for seconds;
(2) `target_x`/`target_x_vel` must be **carried across** the switch, not re-zeroed, or the step
into the inner loop looks like a catch failure that is really a handoff artifact.

Video (HUD carries phi, thetadot, `s`, and guard state so a frame can be checked against the
claim): `tools/diagnostics/render_flip_catch_hold_video.py`. Tests:
`tests/mujoco/test_pendulum_flip_catch_hold.py` (8 passed).

Second lesson from the same comparison: the curved search's saved record says
`guard_fired=False`, but reproducing the SAME parameters at a longer `duration_s` DOES trip the
Y-drift guard. A guard-clean claim is only valid for the duration it was evaluated at — re-check
at the duration you intend to use before reporting a run as clean.

### Open — needs a human decision, do not silently pick one

1. ~~Goal 1's exact `wrist_2`~~ — **RESOLVED 2026-08-14: `-90 deg`**, with the local-X-hinge
   `realrod` asset. Rest of `ARM_Q0` unchanged. See Goal 1 above for the measured basis. Worth
   committing to `hardware/poses.py` as a named pose rather than left as a literal; note a
   "dual-constraint pose" (`pan=-86 deg`, `cond(J)=8.31`, hinge elevation `89.92 deg`) was
   validated 2026-08-13 and also never committed there — likely the same finding reached by a
   different route, worth reconciling before adding a second near-duplicate constant.
2. ~~Goal 2's pendulum asset~~ — **RESOLVED 2026-08-14: `pendulum_attachment.xml`**, by the
   hinge-orientation physics above rather than by preference.
3. **Known physical tension on the 0.12 m rod (locked by user; flagged, not re-litigated)**: its
   half-period is 0.290 s vs the arm's ~0.5 s closed-loop bandwidth, so the cart reversal it
   requires is outside what the arm can track; the 0.30 m rod's 0.449 s is inside it, which is
   why that is the one that ever flipped. If a goal stalls, **rod length is the first suspect,
   not the controller**.
4. **LQR capture envelope has never been measured in 2-D** (`theta`, `thetadot`). The existing
   0.05-0.4 rad came mostly from *position* perturbations, but a swing-up arrives *with
   velocity*. This is the handoff gate and the likeliest silent failure of either goal.

## 1. Project reality

- The folder name is historical: this is a **UR5e torque-control workspace**, not a cartpole
  project.
- **The active lane is MuJoCo true-torque simulation** using the custom torque-actuated UR5e
  model. Everything CoppeliaSim is archived (see §6).
- The repo root **is a git repo** (branch `feature/ur5e-mujoco-torque-control`). `outputs/`,
  `reports/`, `third_party/` are gitignored — changes there are not git-recoverable.
- Robot model assets live in `vendor/mujoco_menagerie/` (tracked). The full menagerie zoo
  checkout was deleted 2026-07-03; to restore:
  `git clone https://github.com/google-deepmind/mujoco_menagerie && git -C mujoco_menagerie checkout 959cabcdfb464cee47e0fbda807371f8d93a4f4c`.
- Python env: conda `mujoco_ur5e` (py3.12) per `environment.yml`. The cluster launchers may
  hardcode `/common/users/ss5772/miniforge3/bin/python3`.

## 2. Active lane — MuJoCo true-torque UR5e

- Model: `assets/ur5e_torque/scene.xml` (includes `assets/ur5e_torque/ur5e_torque.xml`,
  meshes from `vendor/mujoco_menagerie/universal_robots_ur5e/assets`). Real per-body
  inertials; direct torque actuators. **This custom model is the centerpiece — never delete
  or silently regenerate it.**
- Configs: `config/ur5e_mujoco_torque.yaml` (base), `config/ur5e_mujoco_torque_transport.yaml`
  (transport). Gains live under `controller: gains:` and are parsed by
  `CartesianImpedanceConfig.from_controller_yaml_section`; the canonical gain field list is
  `transport_metrics.GAIN_FIELDS`.
- Entrypoints (all support `--help`):
  - `tools/ur5e_mujoco_torque_experiments.py` — single-run rollout engine (the only file with
    the per-step loop; other drivers subprocess it).
  - `tools/audit_ur5e_mujoco_gravity_torque.py` — gravity-sign / hold-quality audit.
  - `tools/ur5e_move_hold_transport.py` — move+hold sweep driver.
  - `tools/ur5e_x_frame_envelope.py` — X-frame transport envelope sweep.
  - `tools/tune_ur5e_residual_impedance_transport.py` (+ `tools/tuning_common.py`) — the
    active gain-tuning driver. Its predecessor is in `archive/superseded/`.
  - `tools/compare_ur5e_mujoco_controllers.py` — controller-family comparison.
- Verified short commands:
  - `python tools/audit_ur5e_mujoco_gravity_torque.py --poses active_origin --durations 1.0 2.0 --seed 0 --no-plot`
  - `python tools/ur5e_move_hold_transport.py --target-x-deltas 0.01 0.02 --move-durations 1.0 --hold-durations 1.0 2.0 --torque-limit-scales 1.0 --seed 0 --no-plot`
- Secondary analysis lives in `tools/diagnostics/` (guardrail trajectory check/overlay,
  torque-QP smoke, `render_trace_video.py` — kinematic replay of a `trace.jsonl` to MP4 via
  `mujoco.Renderer`; needs `MUJOCO_GL=egl` on this headless host, camera defaults tuned for
  the active-origin transport pose). The lab workspace-guardrail workflow
  (`config/lab_workspace_guardrails.yaml`, `simulation/workspace_guardrails.py`) is
  simulation/visualization only — never wire it into real-arm e-stop logic.

### Observability (required for new experiments)

Every sweep entrypoint writes, via `observability/run_logger.py` (`RunLogger`):
- per-run `run_record.json` next to each run's `summary.json`/`trace.jsonl`;
- sweep-level `run_log.jsonl` (crash-safe incremental) + `run_log.csv` (flattened,
  per-joint dicts become `<field>__<joint>` columns).
Records carry `backend`, drift/orientation/velocity metrics with time-to-limit, per-joint
commanded-vs-clipped torque and clip counts, which safety guard fired first and when,
`gravity_hold_status`, `phase_at_failure`, `outcome`, `failure_category`. New experiment
drivers must log through `RunLogger` instead of inventing new summary schemas. Do not trust
an MP4 or a bare exit code as success evidence — read the run record.

## 3. Controller architecture

- `controller_core/` is simulator-independent (numpy only — keep it that way).
  - Law: `controller_core/x_axis_cartesian_impedance.py` — task-space PD wrench →
    `J.T @ wrench` + joint damping + posture PD + externally supplied `gravity_torque`,
    singular-value wrench scaling, geometric torque backtracking, hard clip.
    Joint order: `JOINT_NAME_ORDER` (shoulder_pan → … → wrist_3).
  - State contract: `controller_core/state_types.py` (`as_impedance_robot_state`).
  - Safety: `controller_core/safety.py` `ImpedanceSafetyMonitor` — Y/Z/orthogonal drift,
    orientation error, `|qd| > 1.5 rad/s`, axis-error growth, NaN/joint-limit e-stop. This is
    the source of `termination_reason` strings in traces.
  - Alternate torque law: `controller_core/torque_task_qp.py` (+ `box_qp.py`).
- MuJoCo adapter: `simulation/ur5e_mujoco_torque.py` — steps the sim, currently adds
  `tau_applied = tau_controller + tau_gravity` with `tau_gravity` = MuJoCo `qfrc_bias`.
- **Model-based dynamics (landed 2026-07-03, all flag-gated, defaults = legacy behavior)**:
  - `controller_core/model_dynamics.py` — `DynamicsProvider` + `PinocchioUR5eDynamics`
    (loads the active MJCF; parity vs MuJoCo <1e-8 Nm gravity, <1e-6 bias, <1e-8 mass matrix).
  - Gravity source: `mujoco.gravity_source: pinocchio` / `--gravity-source` (P1).
  - Coriolis feedforward: `mujoco.coriolis_feedforward: true` / `--coriolis-feedforward` (P2 —
    historical lane never compensated C(q,qd)qd; measured negligible below ~0.5 rad/s).
  - Operational-space (P3): `controller.task_space_inertia_shaping` (Λ(q) wrench weighting;
    gains become task-acceleration gains) + `controller.nullspace_posture` (dynamically
    consistent projection; note a full-rank 6D task has no nullspace, so this zeroes posture
    except near singularities). Needs `mass_matrix` in the state dict (`build_mujoco_state`
    supplies it). Named config: `config/ur5e_mujoco_torque_osc.yaml`.
  - P3 evidence (8-point move-hold grid, untuned gains): OSC 6/8 vs baseline 5/8; the
    dx=0.02/hold=2 orientation failure fixed (0.350→0.031 rad); dx≥0.03/hold=2 still fails —
    gain retuning for the acceleration-gain semantics is the known next step.
  - **Tuned OSC gains (landed 2026-07-03, ~250 evaluation runs; config additionally promoted
    2026-07-30 with a `singular_scale` fix, see §4's "Fixed and promoted to default" note —
    gains below unchanged, only `jacobian_singular_cond_max` differs)**:
    `config/ur5e_mujoco_torque_osc_tuned.yaml` — kp_x 400/kd_x 40, kp_rot 0/kd_rot 10,
    kp_posture 25/kd_posture 6, kd_joint 4, lambda_regularization 0.1,
    `posture_reanchor_on_settle: true`. Validated envelope, 0 guard trips throughout:
    canonical grid (dx 0.01–0.04 m × hold 1/2 s) 8/8, worst orientation 0.040 rad; long
    holds (dx 0.03/0.06 m, hold up to 30 s) 8/8, worst 0.067 rad; large displacements
    (dx up to 0.20 m) 16/16, worst 0.205 rad (dx=0.25 m breaks via Z-drift — a genuine
    workspace/reach limit, not a controller defect); torque-scale robustness down to 10%
    14/14. Untuned OSC on the canonical grid: 1/8 valid, 2 guard trips, |qd| 1.7 rad/s.
    Two root causes found and fixed: (1) the transport start pose sits at the UR wrist
    singularity (wrist_2=0) — orientation drifts along a task-unactuatable direction, held
    only by the joint-space posture anchor + kd_joint damping (the nullspace projector
    passes posture exactly there); (2) the task rotation PD is *unstable* at that pose —
    positive feedback through the eps-regularized Λ regardless of kp_rot magnitude, only
    slower as it shrinks (kp_rot=30 trips the guard at ~3.5 s, 5 at ~27 s, 0 never — clean
    to 30 s). Fix: kp_rot=0 (damping-only), posture re-anchoring holds orientation instead.
    Known, out-of-scope boundary: moves faster than ~0.5 s undershoot (closed-loop
    bandwidth limit at kp_x=400, not saturation — irrelevant for 1 s+ transport moves).
  - Posture re-anchoring: `controller.posture_reanchor_on_settle` (+`reanchor_x_tol_m`,
    `reanchor_qd_tol_radps`) — one-shot q_rest re-capture at settle, flag-gated, default off.
  - **Two more OSC leaks found and fixed (2026-07-26), both flag-gated, neither moved the
    ~0.25–0.3 m ceiling**: `controller.lambda_diagonal_shaping` — away from the wrist_2=0
    singularity, Λ=(J M⁻¹ Jᵀ+εI)⁻¹ develops large off-diagonal terms (e.g. Λ_xz 0.32→2.0 by
    wrist_2=-0.006 rad), so the shaped wrench's Z-row picks up `Λ_xz·Fx` even with zero Z
    error; diagonalizing Λ for the wrench-shaping step only (nullspace projector keeps the
    full Λ) kills that leak. `controller.lambda_adaptive_regularization` — the SAME static
    ε=0.1 that tames Λ at the exact singularity (needed: dropping it there makes Λ's
    diagonal blow up, e.g. Λ[3,3] 9.8→670 as ε drops 0.1→0.001) also corrupts the
    nullspace-posture projector away from it (measured: at cond(J)=719, ε=0.1 leaks a real
    0.074 rad/s² task acceleration from a representative posture torque instead of nulling
    it, as theory predicts and ε=0 measures). Fix: schedule ε in log(cond(J)) space between
    a far-field value and the existing near-singularity ceiling — but ONLY for the
    nullspace projector; scheduling the wrench-shaping Λ too caused a real regression
    (previously-trivial cases failing on `|qd|>3.0`) since reducing ε also amplifies
    wrench-shaping in ways the tuned gains were never validated against. Both leaks were
    real and are now fixed (`config/ur5e_mujoco_torque_osc_tuned_adaptive_lambda.yaml`,
    zero regressions across the canonical grid), but the ceiling didn't move — good
    evidence it's structural (see next finding), not a fixable regularization defect.
  - **The ceiling is directional, not just a magnitude limit (2026-07-27/28)**: at
    height_alpha=0.5 (`hardware/poses.py::HEIGHT_ALPHA_0_5_Q`), `+0.20 m` passes cleanly
    (worst orientation 0.214 rad) but `-0.20 m` fails via orientation error at *half* the
    peak wrist_2 excursion (-0.017 rad vs +0.029 rad for the passing direction) — ruled out
    as a speed/duration or reanchor-timing artifact (identical failure at move durations
    1–5 s and with `posture_reanchor_on_settle` disabled; see the git history around this
    date for the full elimination trail). Root cause: since kp_rot=0, orientation is held
    *only* by the nullspace-projected posture term, and that projector's Frobenius norm is
    itself asymmetric with wrist_2 sign at this pose — it grows during the `+0.20 m` move
    (1.74→1.87) but shrinks monotonically during `-0.20 m` (1.74→1.59). Same kp_posture/
    kd_posture, genuinely less restoring authority available in the `-X` direction — this
    is why a `kp_posture`/`kd_posture`/`kd_joint` gain sweep at this exact case (2026-07-27)
    barely moved the outcome (quality 0.306→0.305, and `kd_joint` up made it *worse*).
    `Λ_xz` (wrench-shaping X→Z coupling) shows a related signature — it grows positive for
    `+0.20 m` but crosses zero and goes negative for `-0.20 m` — though `lambda_diagonal_shaping`
    is active in this config and already removes that specific leak from the wrench, so it
    isn't the primary driver; the nullspace-projector asymmetry is. Fixing this for real needs
    a different orientation-holding mechanism (not just retuned gains — the sweep is
    real evidence against that), which is a controller-design question, not a retune.
    Practical floor until then: **the safe symmetric range at height_alpha=0.5 is ±0.15 m**,
    not ±0.20 m.
  - **Directional-ceiling failure, fixed at height_alpha=0.5 (2026-08-01)**:
    `config/ur5e_mujoco_torque_osc_tuned_wrist_orient_fixed.yaml` — the existing
    `wrist_orientation_task` mechanism (a dedicated wrist-only joint-space PD term,
    structurally isolated from the shared Lambda-weighted wrench pipeline; see
    `docs/status/wrist_orientation_task_2026-07-29.md`) combined with the already-promoted
    `jacobian_singular_cond_max: 1.0e18` fix — never validated together before this pass
    (`docs/status/nullspace_envelope_search_2026-08-01.md`). Fully resolves this failure in
    both directions: 8/8 vs baseline 6/8 at both `dx=+0.20m` and `dx=-0.20m` (hold 1/2s), worst-
    case orientation error roughly halved (0.2497 rad at the ceiling → 0.125–0.127 rad). Zero
    regressions anywhere tested: `canonical_grid`/`long_holds`/`torque_scale_robustness` byte-
    identical to baseline at height_alpha=0.5, and the fix generalizes with no gain retuning to
    height_alpha ∈ {0.2, 0.35}. It has **zero measurable effect** on the separate -45° pose
    failure below — a bounded Phase-2 beam search combining it with `lambda_diagonal_shaping`/
    `lambda_adaptive_regularization` (individually and together) also failed identically at
    every dx tried, direct evidence the -45° failure is a distinct task-space Y-axis phenomenon,
    not an orientation/nullspace/Lambda-coupling issue at all — the entire orientation-holding
    mechanism family this controller has targets the wrong axis for that failure. No
    `controller_core/` changes; no existing config modified. **Not yet real-hardware validated**
    and not a general replacement for the real-hardware default
    (`config/ur5e_mujoco_torque_osc_tuned.yaml`, unmodified) without a separate decision, since
    that default is still the actual real-hardware start-pose config per `hardware/poses.py`.
    (Side note correcting §4's earlier "Not yet applied" framing: a `singular_scale`-disabled
    variant of the plain `wrist_orient` config was already independently validated 2026-07-30,
    before this fix — see §4's "Fixed and promoted to default" entry below for the update.)
  - **-45° base-rotation Y-drift coupling — root-caused 2026-08-01, evidence-scoped fix landed
    for small/moderate displacement (2026-07-31 → 2026-08-01)**: `hardware/poses.py::
    HEIGHT_ALPHA_0_5_CLEARANCE_Q` (`HEIGHT_ALPHA_0_5_Q` with `shoulder_pan` overridden to
    -0.7853981633974483 rad, i.e. -45°) became the **default** real-hardware start pose for
    `direct_torque` transport this date, needed for real wall/base clearance in the physical
    lab (visually confirmed). Real hardware then reproducibly tripped
    `ImpedanceSafetyMonitor`'s `|Y-Y0| > 0.03 m` guard at dx=0.20m, 4 separate attempts, at
    almost identical magnitude every time, with the TCP moving in a near-45° diagonal (X and
    Y displacement nearly equal). Two live gain interventions (kp_y/kd_y +50%,
    `lambda_diagonal_shaping`) had **zero** effect on the trip point. Independently confirmed
    in sim the same night (`docs/status/base_rotation_neg45_retune_2026-07-31.md`): frozen-model
    4-category sweep at this pose scores 18/38 vs 36/38 un-rotated, with essentially every
    failure tripping the identical guard at the identical ~0.030 m magnitude, plus a clean
    linear dose-response in the passing canonical-grid runs extrapolating right to the failure
    onset (sim's onset is ~dx=0.05-0.06m, smaller than the real dx=0.20m trip — a real,
    unexplained duration/dynamics difference, not investigated further, that doesn't change the
    qualitative verdict). A full staged BO gain search plus targeted `kd_joint` smoke tests
    found **no candidate that fixes it** — every one failed identically to baseline, matching
    both live real-hardware attempts. Likely a structural kinematic/Jacobian effect of the
    rotated pose (same family as the directional-ceiling finding above), not a gain problem.
    **Root cause confirmed 2026-08-01** (`docs/status/neg45_y_axis_diagnosis_and_fix_2026-08-01.md`):
    new per-cycle trace instrumentation (`task_backtrack_scale`, `y_error`, raw `wrench`,
    `tau_preclip`/`tau_task`/`tau_posture`/`tau_damping`, pre-step Jacobian — added to
    `tools/ur5e_mujoco_torque_experiments.py`, purely additive) directly ruled out the two
    mechanisms most plausibly "neutralizing" gain increases: geometric backtracking and the
    global `singular_scale` term are both provably inactive throughout the failure
    (`task_backtrack_scale==1.0`, `singular_scale==1.0` for all 342 steps of the reproduction,
    torque never exceeds ~4.7% of headroom) — this also rules out saturation as the explanation
    for why the real +50% kp_y/kd_y live-hardware attempt had zero effect. Confirmed instead: a
    genuine, structural kinematic/dynamic **X-Y authority trade-off**. kp_y/kd_y at 5-10x
    baseline (well past the ~1.67x ceiling every prior gain search tried) does stop the guard
    trip, but forces X-tracking into a non-transient steady-state shortfall (45-55% of the
    0.06m target, unchanged by tripling the move duration). A new opt-in `y_integral_action`/
    `ki_y` term (`controller_core/x_axis_cartesian_impedance.py`, mirrors
    `lqr_controller.py`'s `_x_integral` anti-windup pattern) hits the identical trade-off at
    high gain and has **zero measurable effect** at a gentle, non-destructive dose (4-category
    sweep: 0/38 baseline vs 0/38 candidate, byte-identical) — ruled out as a fix, kept as a
    validated, zero-regression, default-off addition (`ki_y=0.0` by default). **Verdict: no P,
    D, or I gain in this controller architecture can hold Y without breaking X-tracking at this
    pose/displacement** — three independent investigations (gain sweep, orientation/nullspace
    mechanism family, and this P/D/I-authority + instrumentation pass) now confirm this is
    structural, not a search gap.

    A second, separate finding from the same diagnosis, reported for a human decision and
    explicitly not acted on unilaterally: with the drift guards temporarily widened for
    observation only (never committed), the natural Y excursion at dx=0.06m is a bounded,
    self-correcting **transient**, not a hazard signature — peaks 0.0423m during the move,
    decays to 0.0058m by end of hold, grows roughly linearly with dx (0.0059m at dx=0.01m), no
    oscillation (`|qd| ≤ 0.158 rad/s` throughout). `max_abs_y_drift_m`/`max_abs_z_drift_m`/
    `max_abs_orthogonal_drift_m` have been a flat, pose-independent 0.03m
    (`controller_core/safety.py`) since this repo's first commit, predating this pose,
    controller, and friction model entirely, with no commit or doc ever revisiting the value.

    **User reviewed this evidence and directed an evidence-scoped fix (2026-08-01,
    `docs/status/neg45_drift_tolerance_validation_2026-08-01.md`)**:
    `config/ur5e_mujoco_torque_osc_tuned_wrist_orient_fixed_neg45_pose.yaml` (built on the
    directional-ceiling fix above) raises `controller.safety.max_abs_orthogonal_drift_m` — the
    field `ImpedanceSafetyMonitor` actually enforces on this path, confirmed by reading the
    class directly — from the class default 0.03m to **0.05m** for this one pose only (~18%
    margin above the largest validated natural peak, 0.0423m; deliberately not raised further).
    `controller_core/safety.py`'s class default is unchanged — this is a config-only override,
    applied only when a caller also selects the -45° pose, not a silent threshold change.
    Validated (same 4-category rigor sweep, current friction-including model): **32/38** —
    `canonical_grid` 8/8, `long_holds` 8/8, `torque_scale_robustness` 14/14,
    `large_displacements` 2/8 (only `dx=0.05m` passes both hold durations; `dx=0.10-0.20m`
    deliberately still fail via `y_drift`, since the dose-response measurement this tolerance
    was sized from never validated past `dx=0.06m` — an evidence-scoped increase, not a blanket
    loosening). **Not yet real-hardware validated** — the -45° pose's real-vs-sim dose-response
    already has an unexplained gap (real trip historically at dx=0.20m vs. sim onset
    dx=0.05-0.06m), so this needs its own careful, small-first real-lab check before trust.

    **Practical floor, updated 2026-08-01**: with the plain (un-widened) default config, dx≤~0.04m
    passes cleanly in both sim and real, unchanged. With the new evidence-scoped tolerance config
    specifically (`..._neg45_pose.yaml`), small/moderate displacement (dx≤0.06m, where the
    dose-response was actually measured) now passes in sim; dx≥0.10m remains a known,
    reproducible, now root-caused failure with no fix in any PID-family controller mechanism —
    do not assume the tolerance raise covers displacement beyond what was validated.
  - **Real joint friction added to the sim model (2026-07-31)**: the same lab session found the
    real UR5e only achieves ~55-72% of a small commanded X displacement with steady-state
    hold-phase torque that never decays toward zero — a friction/stiction signature the
    (previously frictionless) sim never reproduced.
    `assets/ur5e_torque/ur5e_torque.xml` gained real `frictionloss`/`damping` values
    (`size3` joints — shoulder_pan/lift/elbow — 5.0 Nm / 0.4 Nm·s/rad; `size1` joints — wrists
    — 1.0 Nm / 0.15 Nm·s/rad; ~3-4% of each joint's rated torque, grounded against this repo's
    own torque limits since neither the upstream menagerie model — PD position actuators, no
    friction — nor the literature gave a single authoritative UR5e table). This is a real
    regression for every config that doesn't compensate for it: the plain
    `config/ur5e_mujoco_torque_osc_tuned.yaml`'s pass rate at height_alpha=0.5 roughly halved,
    36/38 → 19/38, gains unchanged (`docs/status/ur5e_sim_friction_modeling_2026-07-31.md`).
    Secondary effect: friction converts the already-known transient wrist-singularity freeze
    (see §3's earlier `jacobian_singular_cond_max` history) into a **permanent** freeze on any
    config still at the class default (1e5) — `config/ur5e_mujoco_torque_osc_tuned_wrist_orient.yaml`
    and `config/rl_gain_scheduling.yaml` are affected, unfixed as of this note.
  - **Friction feedforward, opt-in fix (2026-07-31)**:
    `CartesianImpedanceConfig.friction_feedforward` (default off) adds
    `tau_friction_ff = coulomb*tanh(qd/deadband) + viscous*qd` into the same joint-space bias
    as gravity/posture compensation — model-based cancellation, not more gain (pure
    proportional gain can't fully cancel a friction-like disturbance at steady state).
    `config/ur5e_mujoco_torque_osc_tuned_friction_ff.yaml`: same gains as the plain tuned
    config, feedforward on, `friction_ff_qd_deadband: 0.05` (validated — 0.01 causes a real
    closed-loop limit cycle, hold-phase `|qd|` sits right in the tanh term's steep transition
    zone at that setting). Sim validation at height_alpha ∈ {0.2, 0.3} (never swept before,
    `docs/status/friction_ff_alpha_0.2_0.3_sweep_2026-07-31.md`): combined 73/76 (96%) vs
    baseline 43/76 (57%), never worse than baseline in any cell, steady-state error 2.5-9x
    lower with 30% *less* commanded torque at long holds — real compensation, not a
    torque-for-accuracy tradeoff. Integral action (`ki_x`) was considered as a second-layer fix
    but judged unnecessary given how cleanly feedforward alone closes the gap. Also confirmed
    at height_alpha=0.5 itself (the pose that originally motivated this fix): 19/38 → 33/38
    (50.0% → 86.8%), most of the way back to the original frictionless 36/38 baseline. One
    residual gap, root-caused not just noted: `large_displacements` at dx=0.20m still fails
    both hold durations, unchanged before/after feedforward — an actual orientation-guard trip
    (0.25 rad ceiling), not a tracking-tolerance miss, plausibly feedforward's own added torque
    nudging an already-marginal case (this pose's directional-ceiling envelope, see above) over
    the edge. **Not yet validated on real hardware** as of this note — that is the next
    real-lab step.
  - **High-level accel/duration trajectory profiles, `accel_duration_triangular`/
    `accel_duration_scurve` (2026-07-31)**: `simulation/ur5e_mujoco_torque.py::x_profile_target()`
    (the shared trajectory function for both sim tooling and the real `direct_torque` loop)
    gained two opt-in profiles driven by (peak acceleration, duration) instead of (displacement,
    duration) — displacement becomes an analytically-computed output
    (`accel_duration_displacement()`), threaded through the existing `target_x_delta_m` plumbing
    unchanged so every downstream consumer (tolerances, scoring, `summary.json`) is unaffected.
    `accel_duration_triangular`: bang-bang ±a acceleration, real jerk discontinuity at the
    midpoint. `accel_duration_scurve`: `a(t) = accel*sin(2*pi*t/T)`, jerk-continuous, designed to
    avoid TCP-accel guard trips. New CLI flags `--trajectory-profile`/`--target-accel` on
    `tools/ur5e_direct_torque_x_transport.py` and `tools/ur5e_mujoco_torque_experiments.py`;
    `hardware/x_transport.py` wires the new profiles into `direct_torque` mode only (`position`/
    `urscript` raise explicitly rather than silently ignoring them). Sim smoke test found a real,
    quantified robustness difference before landing: at `target_accel=0.3 m/s²`/2s, triangular
    trips the orientation guard (0.25 rad) while scurve completes cleanly (98.9% of target); at
    0.15 m/s² triangular also completes cleanly (99.5%) — a genuine lower safe-accel ceiling for
    the bang-bang profile, not a bug in either. **Real-hardware tested the same night**:
    `accel_duration_scurve` clean at `accel=0.02, move_duration=4.0s`, but tripped the speed
    guard at `move_duration=6.0s` and `10.0s` despite peak commanded velocity being
    mathematically duration-independent for this profile family — a real, only-partially-
    explained finding (diagnosed as healthy X-tracking convergence plus a growing off-axis/
    orientation contribution at longer durations; connects to the directional-ceiling finding
    above). `accel_duration_triangular` has unit coverage
    (`tests/mujoco/test_accel_duration_profile.py`, 16 tests) but was never real-hardware tested.
  - **Real per-cycle `dt_s` plumbed through the `RobotState` contract (2026-07-31)**: two silent-
    drop points fixed, purely additive (verified byte-identical controller output with/without
    the key present — no consumer reads `dt_s` yet). (1) `controller_core/state_types.py`'s
    `as_robot_state()`/`as_impedance_robot_state()` didn't whitelist `dt_s`, so even though sim's
    `MujocoUR5eState.as_robot_state()` already included it, `compute()`'s internal normalization
    silently dropped it before any controller code could see it. (2)
    `hardware/direct_torque_link.py`'s `compose_robot_state()`/`build_robot_state()` never
    included it on the real-hardware path at all — now takes an optional `dt_s` kwarg (default
    `None`). `hardware/direct_torque_transport.py`'s per-cycle call now passes the real measured
    cycle interval (`cycle_start_ns - prev_cycle_start_ns`, falling back to the nominal
    `1/frequency_hz` on the first cycle) — the same `real_dt_s` formula the loop already uses for
    its residual-acceleration estimator, no new timing source. This is a real prerequisite for
    any future stateful, time-integrated controller term — see the LuGre item next.
  - **LuGre dynamic friction — motivated and planned, not yet implemented (2026-07-31)**: the
    same night's real-hardware test of the two trajectory profiles above found both
    `accel_duration_triangular` and `accel_duration_scurve` trip the real TCP-accel guard almost
    identically despite very different jerk shaping (~0.37-0.38s into a 2s move, ~2% of target
    displacement, real measured accel spiking to ~3.3-3.5x commanded) — the signature of a real
    stick-slip breakaway event, not a profile-shape bug: static friction holds the arm nearly
    still while torque builds, then releases suddenly with a transient neither trajectory's
    shaping controls, and the existing static tanh-based `friction_feedforward` model above has
    no memory of "how long has this joint been stuck" so cannot represent it.
    `docs/hardware/LUGRE_FRICTION_MODEL_PLAN.md` is the concrete, buildable follow-on: a single
    bristle-deflection state `z` per joint (`dz/dt = qd - |qd|*z/g(qd)`, Stribeck curve `g(qd)`),
    literature-grounded (Canudas de Wit, Olsson, Åström, Lischinsky, IEEE TAC 1995, plus several
    robot-joint-specific identification papers) — that plan's own honest gap: no single
    authoritative numeric parameter table for UR5e-class joints was found in this literature
    pass either (same gap already logged for the static model), so it defers to its own §4
    real-calibration procedure rather than guessing. The `dt_s` plumbing fix above was a named
    prerequisite (plan §3.3) and is now landed, unblocking this work. Planning only — no code
    written, no config changed. A follow-up, adversarially-verified literature pass
    (`docs/status/literature_review_dynamics_and_sensor_noise_identification_2026-08-01.md`)
    found real UR-series-specific precedent the original plan missed: Clochiatti et al.
    (Robotica 2024) identify UR5e joint friction as viscous plus an **asymmetric** Coulomb term
    keyed to mechanical power-flow direction through the harmonic drive (not just velocity sign,
    unlike this repo's symmetric static/LuGre forms), jointly least-squares-fit with motor torque
    constants from RTDE current logs — no torque sensor needed, same constraint this repo has.
    Also: a learned probabilistic state-space friction model beat identified LuGre/GMS/Stribeck on
    real KUKA data in held-out validation (Vantilborgh et al.) — real evidence that classical
    dynamic-friction fits aren't clearly superior to a data-driven alternative, worth weighing
    before investing real-lab calibration time in LuGre specifically over the asymmetric-Coulomb
    variant or the residual-regression direction below. Related same-night survey,
    `docs/status/nonlinear_controller_research_2026-07-31.md`, ranks next steps for higher-order
    controller representations generally; its top pick (after finishing real-hardware validation
    of `friction_feedforward`, still outstanding above) is supervised residual-torque regression
    on the existing residual-observer data pipeline (`controller_core/dynamics_residual.py`),
    **not** another RL attempt, given six documented RL gain-scheduling failures (§4's pointer to
    `docs/CURRENT_STATUS.md`). A separate same-session design doc,
    `docs/status/long_horizon_planner_design_2026-08-01.md`, evaluated a receding-horizon
    planner layered on top of the reactive controller and recommended **against** building it
    next: neither of this repo's two gain-tuning-exhausted failures (the directional ceiling,
    now fixed above; the -45° Y-drift trade-off, still open above) is a trajectory-reference
    problem a smarter planner could reach — both are torque-path/authority problems.

## 4. Safety & guardrails (hardware — do not weaken)

`hardware/` is the real-UR5e RTDE lane (rewritten 2026-07-07; older lane in
`archive/superseded/hardware_rtde_v1/`). **Learning map:** `docs/hardware/README.md`.

Three control modes via `hardware/x_transport.py` (`--control-mode`):
1. **`position`** (default) — `servoL` + optional shadow OSC (`position_transport.py`);
   use on URSim / real arm to test trajectory, safety, logging without live torques.
2. **`direct_torque`** — Python OSC @ 500 Hz → `directTorque()` (`direct_torque_transport.py`);
   URSim validates the API only (no torque physics); real arm for motion.
3. **`urscript`** — OSC on PolyScope (`urscript_transport.py`); minimum-latency path.

Core modules:
- `hardware/safety.py` — `UR5eSafetyLimits`, `ConnectionHealth`, one-way `EStopLatch`,
  `CartesianMoveMonitor` (TCP drift / orientation / growth abort).
- `hardware/link.py` — `UR5eLink`: receive + optional `servoL`/`moveJ`. `read_state()`
  **raises** `RTDEStateError` (never returns stale cache — fix for the old ROS2 bug).
- `hardware/motion.py` — bounded Cartesian `servoL` move (`tools/ur5e_move.py`).
- `hardware/direct_torque_link.py` — `UR5eDirectTorqueLink` + RTDE/local J+M.
- Never add gravity torque in Python when using `directTorque()` — PolyScope adds it.

CLIs: `tools/ur5e_connect.py` (receive-only; cannot move), `tools/ur5e_move.py`,
`tools/ur5e_direct_torque_x_transport.py` (main `--control-mode` entry),
`tools/ur5e_direct_torque_height_latency_test.py`, `tools/ur5e_urscript_x_transport.py`.

Guardrails enforced in code:
1. **Receive-only default** — `UR5eLink.connect(with_control=False)` never opens control
   unless asked.
2. **Motion requires explicit opt-in** — `--i-understand-this-moves-the-robot`, checked
   before any move path runs.
3. **E-stop latch is one-way** — no `reset()`/`clear()`; tripped ⇒ new process.
4. **No reconnect mid-motion** — state-read failure aborts; reconnect only in
   `ur5e_connect.py --watch` idle loop.
5. **All three modes share `CartesianMoveMonitor` (TCP speed/accel/waypoint-jump), not just
   `position`** — fixed 2026-07-25 (see below); previously `direct_torque`/`urscript` only
   had `ImpedanceSafetyMonitor` (drift/orientation/joint-velocity/axis-growth, no Cartesian
   kinematic ceiling), i.e. the two modes capable of a torque runaway had the loosest guards.
6. **Robot-reported safety status is checked every cycle in all four loops**
   (`motion.py`, `position_transport.py`, `direct_torque_transport.py`,
   `urscript_transport.py`) via `hardware.safety.is_robot_safety_normal()` — fixed
   2026-07-25; the telemetry was already being read (`getSafetyStatusBits()`/
   `getSafetyStatus()` → `UR5eState.safety_status`) but never inspected anywhere.

**2026-07-25 audit + fixes**: full detailed writeup moved to
`docs/archive/AGENTS_HISTORY.md` (per this file's own no-chronological-logs rule). Summary of
what landed: `CartesianMoveMonitor` layered onto `direct_torque`/`urscript` (item 5 above);
robot safety-status bit checked every cycle in all four loops (item 6 above);
`urscript_transport.py`'s stop-register and NaN/Inf handling fixed; `ur5e_move.py`'s
self-referential speed guard replaced with a fixed ceiling.

**Found, not yet fixed — flagged for a deliberate decision, not silently patched:**
- **URScript (Mode 3) now has full numerical parity with the Python controller on all three
  known behaviors; no real-hardware validation exists for any of it yet (corrected 2026-07-29,
  same night as the fix below).** Nullspace posture projection and geometric backtracking were
  fixed in commit `b24cdf4` (2026-07-26); `cond(J)`-based singular-value wrench scaling was
  fixed in commit `7406704` (2026-07-29) — URScript has no SVD, so `cond(J)` is estimated
  on-robot via a from-scratch cyclic Jacobi eigenvalue algorithm
  (`sigma_max` from `JᵀJ`'s top eigenvalue, `sigma_min` from the reciprocal of `inv(J)ᵀinv(J)`'s
  top eigenvalue — deliberately not the naive smallest-eigenvalue-of-`JᵀJ` approach, which
  squares the conditioning and loses precision past `cond(J)~1e8`). All three are now
  numerically parity-tested against `x_axis_cartesian_impedance.py` in
  `tests/hardware/test_urscript_parity.py`: `test_nullspace_projection_matches_python`
  (`atol=1e-8`), `test_backtracking_matches_python_under_saturation` (`atol=1e-6` under
  saturation), `test_gap_singular_scaling` (now asserts real parity — measured cond-estimate
  relative error ~5.7e-11, torque diff ~9.8e-12 Nm at `cond(J)=1e7`). **Two real gaps remain**:
  (1) no real-hardware/URSim execution of any of this has ever happened — only Python-vs-Python
  parity is proven; (2) the Jacobi solver's per-cycle compute cost on real PolyScope hardware
  has never been benchmarked — a from-scratch eigenvalue solve every control cycle is a real new
  computational cost on the robot's own controller that nothing here proves fits the real-time
  budget. **CLI gap closed 2026-08-01** (commit `b586f23`): `tools/ur5e_urscript_x_transport.py`
  now has the same `CartesianMoveMonitor` guard-override flags as the `direct_torque` CLI
  (`--max-tcp-accel-mps2`/`--accel-gap-cycles`/`--speed-lowpass-alpha`/
  `--accel-max-consecutive-violations`/`--accel-hard-multiple`/
  `--speed-max-consecutive-violations`/`--speed-hard-multiple`/`--noise-robust-guards`) —
  `hardware/urscript_transport.py::run_urscript_x_transport()` already accepted these overrides,
  only the CLI argparse wiring was missing, blocking reproduction of the real-hardware-validated
  `--noise-robust-guards` preset on this control mode. Purely additive (new flags default to
  `None`/off). **This does not change the two real gaps above** — URScript mode still has zero
  real-hardware or URSim execution ever; do not read this as "URScript now validated," only as
  "less blocked than before."
- **RL gain-scheduling's never-move collapse has a credible root cause**, see
  `docs/CURRENT_STATUS.md` — not a hardware item, kept here only as a pointer.

**Fixed and promoted to default, 2026-07-30**: `controller_core/x_axis_cartesian_impedance.py`'s
global `cond(J)`-based `singular_scale` nulled task authority at the transport start pose,
freezing the controller (`tau≈1e-13-1e-4 Nm`) for roughly the first HALF of every move (not
the ~0.2s earlier estimates suggested — measured on a gentle dx=0.02m/1.5s move: first real
torque at `t=0.784s` of a 1.5s move), only escaping via numerical-noise perturbation of
`wrist_2` off exactly zero, then cramming the full displacement into whatever time remained —
producing a genuine (not sensor-noise, not estimator-artifact) TCP acceleration spike 53x the
nominal min-jerk profile's theoretical peak (2.72 vs 0.05 m/s² on that same move). This was
the real root cause behind most of the real-hardware TCP-accel guard trips investigated in
`docs/status/safety_envelope_backtest_2026-07-30.md` earlier the same night — not sensor
noise, as first assumed. Fix: `jacobian_singular_cond_max: 1.0e18` (vs. the class default
1.0e5), disabling the term for any physically realizable Jacobian — it was already redundant
with, and defeating, `lambda_regularization` (already 0.1 in the tuned config), which alone
produces a healthy X force at the singularity. Full validation
(`docs/status/disable_global_singular_scale_validation_2026-07-30.md`): 4-category rigor sweep
(canonical grid, long holds, large displacements, torque-scale robustness) at
`height_alpha ∈ {0.1, 0.2, 0.3, 0.5}`, 304 runs — 152/152 pass with the fix vs. 140/152 without,
zero regressions, +12 passes recovered (every prior failure a small-displacement canonical-grid
case where the freeze ate too much of a short move window). An earlier, informal claim of this
same validation (referenced only in a commit message) was never backed by a durable artifact;
this is the first reproducible evidence. **Promoted to the default**:
`config/ur5e_mujoco_torque_osc_tuned.yaml` now has the fix baked in; the previous default
(`singular_scale` enabled) is preserved unmodified at
`config/ur5e_mujoco_torque_osc_tuned_singular_scale_enabled.yaml`. **Not yet applied** to
`config/ur5e_mujoco_torque_osc_tuned_wrist_orient.yaml` itself (the file, unmodified) — but
**since validated as the same real bug, separately** (2026-07-30, same night,
`docs/status/disable_singular_scale_wrist_orient_validation_2026-07-30.md`, not folded into
this file until the 2026-08-01 update below): freeze confirmed real in this config too (raw
peak TCP accel 2.38 → 0.060 m/s², 39.6x lower; first real torque at t=0.028s vs. t=0.744s
baseline) and fixed by the identical `jacobian_singular_cond_max: 1.0e18` override, saved as
`config/ur5e_mujoco_torque_osc_tuned_wrist_orient_no_singular_scale.yaml` — 304-run 4-category
sweep at height_alpha ∈ {0.1, 0.2, 0.3, 0.5}, zero regressions, byte-identical pass/fail
(150/152 both configs; unlike the base config's validation this one recovers no additional
passes, plausibly because `wrist_orientation_task`'s separate wrist PD path already damps
enough during the freeze window for canonical-grid moves to complete anyway — not confirmed by
a targeted ablation). Promotion into `wrist_orient.yaml` itself was left as a human decision,
not acted on. On 2026-08-01 this same fix was combined with `wrist_orientation_task` again
(functionally the same override) and validated as `config/ur5e_mujoco_torque_osc_tuned_wrist_orient_fixed.yaml`
— the config that resolves the directional-ceiling failure, see §3.
`config/ur5e_mujoco_torque_osc_tuned_adaptive_lambda.yaml` and `..._diagonal_lambda.yaml`
already had `jacobian_singular_cond_max: 1.0e18` set independently (unrelated prior work),
unaffected by this promotion.

**Corrected 2026-07-28** (previously listed above as "found, not yet fixed" — both were
already closed by commit `85498a0`, 2026-07-25, before that bullet list was written; this
file was never updated after that commit landed):
- `max_deadline_ms` (`UR5eSafetyLimits`) **is enforced** — `DeadlineMonitor`
  (`hardware/safety.py`) is instantiated and checked every cycle in all four motion loops.
  Real caveat found 2026-07-28 during live hardware testing: its flat 3.0ms threshold is
  calibrated for the 125Hz loops (8ms period), not `direct_torque`'s 500Hz loop (2ms
  period) — a real overrun there (up to ~2ms over) can sit under that floor by design. Not
  fixed yet; full analysis and concrete fix proposals in
  `docs/status/timing_safety_gaps_audit_2026-07-28.md`.
- **Cycle-to-cycle staleness detection during motion exists** — `StaleStateMonitor`
  (`hardware/safety.py`) is checked every cycle in all four motion loops, comparing
  `robot_timestamp_s` against the host clock; trips after 5 consecutive frozen-vs-advancing
  reads. Full trace of this mechanism in the same doc above.
- Real 2026-07-28 hardware findings (wrist-singularity divergence in `position` mode; the
  `CartesianMoveMonitor` accel estimate's own noise floor being far above its old default
  threshold, now fixed via `accel_gap_cycles`/`speed_lowpass_alpha`; two real RTDE read
  stalls, most likely the documented UR behavior of the robot controller deprioritizing
  telemetry under its own load, not a bug in this codebase): see
  `hardware_captures/2026-07-28_thinkrobot_172.16.71.77/README.md` and
  `docs/status/clock_timing_late_cycles_2026-07-28.md`.

**Added, 2026-07-31/2026-08-01 — automatic pre-trip diagnostic capture (`direct_torque` only)**:
`hardware/direct_torque_transport.py` now captures a `pre_trip_trend` field automatically on any
guard trip — a bounded 60-cycle deque (`PRE_TRIP_TREND_WINDOW_CYCLES`) of `qd`, TCP speed
(single-cycle position-delta/dt, matching `CartesianMoveMonitor.check()`'s own formula, not
`ee_lin_vel`), `x_error`, `tau_controller` L1 norm, `orientation_error_norm`, and (added the
same night, commit `467fe52`) `y_drift`/`z_drift`, each classified rising/falling/stable
(`_classify_trend`, a first-third-vs-last-third mean comparison) at trip time only — built
because real-hardware trip diagnosis that night required repeatedly hand-parsing `trace.jsonl`.
`summary.json`'s shape for clean runs is unchanged (`pre_trip_trend: None`).
**Only wired into `direct_torque_transport.py`** — `position_transport.py` and
`urscript_transport.py` have no equivalent capture; there is no guard/diagnostic parity across
the three control modes for this specific feature. A real bug in `_classify_trend` was found and
fixed 2026-08-01 (commit `b43a9b2`), via the Kalman-filtering investigation below: the original
relative-only deadband collapses for a signal whose mean legitimately sits near zero (e.g.
`y_drift_m`/`z_drift_m` during a clean segment) — a tiny absolute noise wiggle is then a huge
fraction of an already-tiny mean, so pure noise was misclassified rising/falling almost every
time (a sensitivity sweep found "stable" reported for only ~29-30/50 seeds even at the true-zero
null case, pre-fix). Fixed by adding a second, absolute deadband condition (2x the window's own
std) alongside the existing relative one — either being true means "stable"; verified against
the real slow `z_drift` creep from the -0.15m return-leg trace, still correctly classified
"falling".

**Investigated, 2026-08-01 — Kalman filtering for TCP-accel/drift sensor noise: negative both
times** (`docs/status/kalman_filtering_sensor_noise_2026-08-01.md`; no `hardware/safety.py`
changes made). (1) Does a per-axis constant-acceleration KF (or its steady-state
alpha-beta-gamma equivalent) beat `CartesianMoveMonitor`'s existing `accel_gap_cycles`/
`speed_lowpass_alpha`/graduated-tolerance heuristic for the TCP-accel guard's noise chain (a
genuine double finite-difference, ~1/dt² noise amplifier)? No: at a `jerk_psd` tuned to
comparable tightness, the KF's noise floor is worse (p99 0.478 vs. the shipped preset's 0.272
m/s²); tuning it tighter (p99 0.073 at `jerk_psd=0.1`) costs a real, measured ~300ms tracking
lag on a genuine fast move — a real hazard, since the one documented real divergence event in
this repo's history escalates on an 8ms/cycle timescale — and pass/fail outcomes on both
reconstructable backtest profiles were identical to the current heuristic either way. Compute
cost (77 us/cycle vs. ~1.7ms of measured loop headroom) is not the blocker. (2) A reframing
tried the same day: run a heavily-smoothed KF branch in **parallel**, purely as a
diagnostic overlay that never gates the real-time trip decision (so the ~300ms lag is
irrelevant by construction). Also no benefit, for a different, checked reason:
`y_drift`/`z_drift`/`orientation_error_norm` (the `pre_trip_trend` targets above) are direct
single-shot geometric measurements, not derivatives, so they were never noise-limited in the
first place (real noise floor ~1e-5, three to four orders of magnitude below the guards these
trends anticipate); raw `_classify_trend` already saturates at near-maximum sensitivity
(detects a true drift of a few micrometers over 60 cycles). This parallel pass is what surfaced
the `_classify_trend` deadband bug fixed above. **Recommendation: do not implement Kalman
filtering anywhere in this codebase currently** — neither as a guard replacement nor as a
parallel trend overlay. Only standalone offline diagnostic scripts were added
(`tools/diagnostics/kalman_tcp_accel_filter_prototype.py`,
`tools/diagnostics/kalman_parallel_trend_prototype.py`), no production file touched.

**Added, 2026-08-01 — SSH/rsync real-hardware log transfer**: `tools/pull_hardware_logs_ssh.sh`
(run on westeros) / `tools/push_hardware_logs_ssh.sh` (run on thinkrobot) replace an earlier,
never-configured rclone+Box-OAuth pair of scripts with a plain rsync-over-SSH transfer,
reusing SSH access already proven to work between the two machines on the same subnet — no
cloud account, no OAuth, nothing new to configure. Pulls/pushes only small text artifacts
(`run_record.json`, `summary.json`, `trace.jsonl`/`supervisor_trace.jsonl`, `run_log.jsonl`/
`.csv`), not video or other large binaries.

Do-not-recreate (gravity/dynamics bugs, still relevant):
- Do not tune gravity scale from single-joint probes; always test all 6 joints.
- Do not add gravity compensation twice (the QP controller adds it internally; adapters add
  it only in IK-PD/warmup paths).
- Do not conflate solver-side feasibility with a passing live summary — runtime safety and
  drift checks must pass.
- Do not let one simulator's dynamics model silently feed another's control loop (the
  archived CoppeliaSim lane defaulted `gravity_compensation_source="mujoco"` — that
  cross-lane coupling is the canonical example).
- Do not `abs()` a signed target before comparing it to a signed achieved value in
  `transport_metrics.py` (fixed 2026-07-03 in `compute_valid_move_hold_metrics`:
  `abs(achieved - abs(target))` silently failed every negative-direction transport run).
  `abs()` is only correct where the result is used purely as a tolerance *magnitude*
  (e.g. `_move_hold_tolerances`'s own local copy), never in a signed subtraction.

## 5. Testing

- Root `pytest.ini`; suite layout: `tests/unit/` (pure numpy controller_core),
  `tests/mujoco/` (needs mujoco), `tests/hardware/` (mocked RTDE). Markers auto-applied by
  directory: `pytest -m unit`, `-m mujoco`, `-m hardware`, `-m "not slow"`.
- Full suite: `python -m pytest -q` (167 passing as of 2026-07-07; this count drifts as tests
  are added — don't treat it as a gate, just a sanity baseline).
- Before long training/sweeps, run the tiny smoke first (`tests/mujoco/test_ur5e_mujoco_torque.py`
  covers model-load and a tiny move-hold subprocess run).
- **New modules/packages ship with pytest coverage, no exceptions (2026-08-06).** Manual/inline
  smoke checks during development are fine as a first pass but are not a substitute — every new
  package under `controller_core/`, `hardware/`, `simulation/`, or a sibling top-level package
  (e.g. `velocity_gain_tuning/`, `rl_gain_scheduling/`) needs real `tests/` coverage (unit-level
  where the logic is pure-numpy/deterministic; a `mujoco`-marked smoke test where it isn't)
  before the work is considered done, not just before it's committed. Do not skimp on this to
  save time — a documented gap ("no formal pytest coverage yet") is a standing TODO, not an
  acceptable resting state.

## 6. Archived lanes

- `archive/coppelia/` — the entire CoppeliaSim stack (orchestrator, Lua add-ons, launchers,
  ZMQ probes, WSL bring-up, RL PPO stack, ROS2 controller/bridge nodes, docs, configs, tests).
  Not runnable in place; resurrect notes + removed deps in `archive/coppelia/README.md`.
  The vendored simulator runtime remains at `third_party/coppelia_runtime/` (gitignored).
- `archive/legacy_mujoco/` — pre-torque-lane MuJoCo cartpole diagnostics, including four
  scripts whose names sound CoppeliaSim-related but are pure MuJoCo. Design rationale and
  a per-controller reference for that lane: `docs/archive/CONTROL_DESIGN_NOTEBOOK.md`
  (implementation reference), `docs/archive/SLSQP_CONTROLLER_REFERENCE.md`
  (controller/solver/runner index), `docs/archive/FIRST_PRINCIPLES_CODE_FLOW.md`
  (onboarding walkthrough).
- `archive/superseded/` — replaced drivers (old impedance tuner); `hardware_rtde_v1/` (the
  pre-2026-07-07 real-UR5e RTDE lane: `ur5e_rtde_bridge.py`, `ur5e_control_session.py`,
  `ur5e_stages.py`, `safety_limits.py`, `ros_topics.py`, the five staged `tools/ur5e_*.py`
  CLI scripts, the ROS2 hardware pipeline node + its launch file, and their old tests —
  superseded by the current `hardware/{safety,link,motion}.py` + `tools/ur5e_{connect,move}.py`
  described in §4).
- Historical operational lore and per-date findings: `docs/archive/AGENTS_HISTORY.md`. The
  full pre-2026-07 documentation set (project origin, legacy workspace/singularity studies,
  the original CoppeliaSim-port bring-up plan) also lives under `docs/archive/` — browse it
  for anything not covered by the pointers above.
- Superseded root-level reports, now archived: `docs/archive/AUDIT_REPORT.md` (pre-archival
  bloat/dynamics audit), `docs/archive/BLOAT_REPORT.md` (bloat diagnosis, mostly executed),
  `docs/archive/DIAGNOSTIC_real_cartpole_torque_control_questions.md` (the diagnostic that
  motivated the Pinocchio P0-P3 work in §3 — read that section for the current answer).

## 7. Working rules for this repo

- Start with the simplest proof (model loads → short run → read run_record.json) before long
  sweeps or controller changes.
- Do not change training/eval logic and controller logic in the same commit; do not combine
  startup fixes with gain tuning.
- Never silently change units, control rate, action scaling, torque limits, or checkpoint
  selection; state control-rate and scaling assumptions when touching controllers.
- Preserve old configs — add new named configs instead of mutating shared ones.
- **LOOK AT THE POSE BEFORE YOU RUN IT (2026-08-16). Mandatory, and it needs a HUMAN.**
  Before dispatching any swing-up or balance run, render the pose with
  `tools/diagnostics/render_pose_task_axes.py`, SHOW IT TO THE USER, and ask which axis of
  motion they want. The drive is ONE scalar axis and the entire run is worthless if it points
  somewhere the pendulum cannot feel -- a failure invisible in the config, the gains and the
  logs, and obvious in one picture. It has happened twice:
  * A tool-frame run put the drive on row 0 = tool X, which at `ARM_Q0` is 7.3 deg from
    VERTICAL. Vertical pivot acceleration exerts ZERO hinge torque at the hanging
    equilibrium, so the law drove an axis with no authority over the pole, dumped it into Z,
    and tripped the corridor in 0.134 s with the rod tip 4 mm off the floor.
  * Every world-X run at `ARM_Q0` spends ~70% of its motion ALONG the hinge.
  **TWO numbers, and the second is the one that bites.** `kappa` = fraction not wasted along
  the hinge. `kappa_hang` = authority at the HANGING start, where a swing-up must bootstrap.
  With a horizontal hinge the entire perpendicular vertical plane scores `kappa = 1.0`, so
  kappa CANNOT distinguish the vertical direction from the horizontal one -- measured at
  `ARM_Q0`, tool X and tool Y both score `kappa = 1.0000` while `kappa_hang` is 0.127 vs
  0.9919. Requiring only `kappa` is exactly the mistake that broke the run above. Both must
  be ~1.
- **A gain belongs to the case it was derived for. Never carry one across (2026-08-16).**
  A gain value is only valid for the exact `(controller, task frame, pose, row set, role)`
  it was fitted at. Changing ANY of those invalidates it, and inheriting it anyway is silent:
  the config still parses, the run still completes, and the number is simply wrong.
  When you derive a new config from an existing one, every gain you did not re-derive is a
  bug until proven otherwise — state in the config where each one came from and why it still
  applies, or re-derive it.
  Two real instances, both found the same day in one file:
  * **Wrong frame.** `kp_x = 1532.672` was fitted from `Lambda_xx = 3.8317` in the WORLD
    frame. Under `task_frame: tool` row 0 is tool X, whose `Lambda` is 3.2386 — a different
    physical axis with a different inertia, so the inherited number describes something else.
  * **Wrong ROLE, which is the nastier one.** `kp_y = 5.0` was a *corridor centering bias*.
    Moving Y into `task_axis_rows` makes it a *tracking* gain, and
    `kp_axis = (kp_x, kp_y, kp_z)` is indexed by row
    (`controller_core/x_task_yz_corridor_qp/controller.py`), so that 5.0 silently became the
    tool-Y tracking gain — **333x too small**. Nothing errors; the axis just barely tracks.
    Corollary: do not apply a tracked-axis conversion rule to a barrier/centering axis
    either. That is the same mistake pointed the other way.
  Practical rule for this controller family: `kp_QP = kp_OSC * Lambda_axis` with
  `kp_OSC/kd_OSC = 400/40`, `Lambda` measured **in the frame the row actually lives in**, at
  the pose being run, with the config's own `lambda_regularization`, and from the
  UN-excluded Jacobian (`Lambda` is a property of the mechanism, not of which joints are
  allowed to act). Verify by re-parsing the YAML and asserting each field, never by reading
  the file — see the next rule.
- **Verify a config edit by re-parsing it, not by looking at it (2026-08-16).** Twice in one
  session a config edit silently did nothing — once an indentation mismatch meant only a
  header comment landed, once the parser did not read the keys at all
  (`task_space_inertia_shaping` / `lambda_*` were dropped before the fix). Both parsed
  cleanly and reported defaults. After any config change, load it through its own
  `from_controller_yaml_section` and assert the fields you intended to set.
- Never edit without explicit request: checkpoints, datasets, logs, `.git`, generated
  experiment artifacts under `outputs/`, large binaries.
- For every final response: list files changed, tests run, tests not run, and a rollback
  command.
- **Always sweep both +X and -X when characterizing transport range/safety (2026-08-06).**
  X-direction asymmetry is a real, recurring, independently-confirmed phenomenon in this repo,
  not sampling noise — see the torque-control lane's directional-ceiling/-45° base-rotation
  findings in §3, and the velocity-control lane's `hanging_alpha_0_5` result (found this date:
  gains passing cleanly at +0.37m failed via `joint_velocity_guard` at -0.37m, with a
  non-mirrored failure-mode pattern in between, not just a smaller-magnitude version of the
  same curve). A one-directional sweep can silently report a "safe range" that is only true
  going one way. Applies to any new range/boundary characterization work, not just
  `velocity_gain_tuning/` — build the negative direction into the default sweep/search grid
  itself so it can't be silently skipped, the same way `FAST_MOVE_DURATION_S` stress-testing
  was made a structural part of the gain-search pipeline rather than a manual follow-up.

## 8. Remote compute / cluster usage (added 2026-07-29)

`westeros` is a shared machine — `uptime` can show load ~100 on 72 cores from other users'
jobs with no warning. Before launching a real training/sweep run, check `uptime`/`nproc` first;
don't assume idle capacity.

Rutgers CS `ilab1`-`ilab4.cs.rutgers.edu` are viable overflow capacity (same NFS home, same
conda env, no file copying needed) but are teaching/interactive machines with real gotchas for
unattended background jobs, found the hard way running RL training there:

- **Per-process BLAS thread explosion**: with `OPENBLAS_NUM_THREADS` unset, each parallel worker
  process (e.g. `SubprocVecEnv`) auto-detects the full core count and spawns that many BLAS
  threads *itself* — `n_workers × n_cores` threads blows through the per-user `RLIMIT_NPROC`
  cap fast. Always export `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
  NUMEXPR_NUM_THREADS=1` before launching multi-process CPU workloads; the parallelism should
  come from having N processes, not from each process also being internally multi-threaded.
- **Per-user memory cgroup cap, separate from system RAM**: `free -h` showing hundreds of GB
  free is not the limit — check `cat /sys/fs/cgroup/user.slice/user-$(id -u).slice/memory.max`.
  Exceeding it triggers a silent cgroup OOM-kill: process vanishes with no Python traceback, no
  dmesg access to confirm (unprivileged users usually can't read the kernel ring buffer). If a
  background job dies cleanly mid-run with an empty log tail and no error, suspect this before
  anything else.
- **`nohup`+`disown` is not reliable on these hosts**: `systemd-logind`'s `Linger` setting for
  the account can be `no` (and can get reset back to `no` even after `loginctl enable-linger
  <user>` succeeds — cause not confirmed, possibly a periodic account-sync job). With no
  lingering, all background processes get killed the moment the last SSH session to that host
  closes, which happens naturally between one-shot SSH commands. Symptom: process dies silently,
  no OOM signature, no traceback, checkpoints just stop. **Reliable fix**: don't background
  remotely at all — run the job in the foreground of a single, continuously-open SSH connection
  (e.g. wrapped in a local `run_in_background` shell job), so the host never sees zero sessions
  for that account during the run.
- **`pkill -f <pattern>` self-match trap**: the pattern you pass is itself part of your own
  invoking shell's command line (since it arrived via an SSH command string), so a loose pattern
  matches and kills the command issuing it before it can do anything else. Prefer killing by
  explicit PID, or use the bracket trick (`grep "[m]emtest_probe"`) to exclude the invoking
  process's own literal argument text from matching.
