# UR5e pendulum CAD -> MuJoCo model (first pass)

**Date:** 2026-08-09
**Status:** builds, compiles, physically sane in isolation and composed with the arm. **Most
pendulum-specific dimensions are labeled placeholders, not measurements** -- see below before
using this for anything beyond a first kinematic/dynamic sanity check.

## What this is

`cartpole_cad/UR5e Pendulum v1.zip` is a real SolidWorks CAD assembly for a physical apparatus:
a free-swinging pendulum mounted on the UR5e's wrist via an OnRobot tool-changer, with a
CUI/Same Sky AMT222B-8000-S single-turn absolute magnetic encoder measuring swing angle. This
is the actual hardware the workspace's historical "real_Cartpole" name refers to -- the UR5e's
existing X-transport control becomes the "cart," this apparatus is the "pole."

This session built a first MuJoCo model of it:
- `assets/ur5e_pendulum/pendulum_attachment.xml` -- standalone MJCF fragment for the pendulum
  (rod + pivot + lumped encoder/housing mass + an angle sensor).
- `simulation/ur5e_pendulum_compose.py` -- composes it onto the existing, **untouched**
  `assets/ur5e_torque/ur5e_torque.xml` (the protected centerpiece model, AGENTS.md sec 2) via
  `mujoco.MjSpec.attach()` at that model's existing `attachment_site`, in Python, at load time.
  Nothing was added to or duplicated from the centerpiece file.
- `tools/diagnostics/ur5e_pendulum_smoke_test.py` + `tests/mujoco/test_ur5e_pendulum_compose.py`
  (8 tests, all passing) -- model compiles (7 DOF: 6 arm + 1 pendulum), the centerpiece file is
  provably untouched, the pendulum released from horizontal swings down and settles under
  gravity both in isolation and composed with the arm, the angle sensor reads the joint's real
  qpos.

## Dimension provenance -- what's real vs guessed

This environment has no SolidWorks and no working reader for `.SLDPRT`/`.SLDASM` (the native
format the custom parts -- Pendulum Rod, Rod Housing, Shaft Housing, Sensor Housing, Linear
Motion Rod -- are stored in). `cadquery`/OCP was installed this session specifically to try;
it reads the archive's two open-format `.STEP` files (the OnRobot tool-changer, both sides)
successfully but cannot open the proprietary SolidWorks parts.

**Real / directly extracted:**
- OnRobot tool-changer robot-side plate bounding box: 71mm x 71mm x 16.1mm (read via cadquery
  from the STEP file).
- Tool-side plate bounding box: 75.51mm x 28.40mm x 71.00mm (same method).
- Rod/shaft diameter = 8mm -- not measured, but two independently-named parts in the archive
  reference "8mm" explicitly ("8mm Bore Clamp", "Same_Sky_AMT-Sleeve-REV-D(8 mm)"). A real,
  non-arbitrary signal, used directly in the model.
- The encoder is confirmed single-turn absolute (CUI/Same Sky AMT22 family) by part name; exact
  bit-resolution was not confirmed with confidence and is not asserted anywhere in the model.

**Placeholder / physically-reasonable but NOT measured** (every instance is also commented
in-place in `assets/ur5e_pendulum/pendulum_attachment.xml`):
- Rod length (0.30m assumed -- a common lab-scale choice, no support from the archive).
- Rod/housing material density (mild steel, 7850 kg/m^3, assumed from "Linear Motion Rod" +
  ball-bearing pivot reading as a hardened-steel shaft component -- an inference, not a spec).
- Encoder/housing lumped mass (0.06 kg, order-of-magnitude guess).
- Pivot joint axis orientation relative to `attachment_site` -- chosen only so the isolated
  pendulum hangs at 0 rad in its own local frame; NOT derived from the CAD assembly's actual
  mate definitions (which are unreadable).
- No discrete end-mass/"bob" is modeled -- the archive's part list has no separate weighted
  part beyond the rod/housings, so mass is distributed along a uniform rod plus the lumped hub
  mass. If the real assembly has a separate end weight, this is wrong.

**Do not use this model for anything beyond a first-pass sanity check until these are corrected
against the real CAD or the physical hardware.**

## A real modeling detail found and handled, not a bug

The composed (arm+pendulum) model's pendulum does NOT settle at 0 rad when released from
horizontal -- it settles at 2.786 rad in the tested neutral arm pose. This is correct, not a
bug: `attachment_site`'s fixed rotation relative to `wrist_3_link` means the pendulum joint's
local zero is not world-down once attached to the arm at a given configuration. Verified by
testing the pendulum fragment in isolation (no arm, no site rotation): there it settles cleanly
at 0 rad as expected. Both behaviors are captured as separate, correctly-scoped test cases in
`tests/mujoco/test_ur5e_pendulum_compose.py` (`test_isolated_pendulum_settles_at_true_hanging_
equilibrium` checks the true 0-rad equilibrium; `test_composed_pendulum_settles_under_gravity`
only checks convergence, since the numeric equilibrium is genuinely pose-dependent).

## What was NOT done this pass

- **No real geometry/mesh import** -- the model uses simple primitive shapes (cylinders) sized
  from the extracted bounding boxes, not the actual STEP/SLDPRT mesh geometry. A real mesh
  import (via cadquery exporting the STEP files to STL/OBJ for MuJoCo's mesh asset system)
  is a natural next step now that the STEP files are confirmed readable.
- **No coupled arm+pendulum dynamics validation** -- the smoke test holds the arm rigid via a
  per-step qpos reset (a kinematic hold, not a real actuator/equality-constraint hold). A test
  that actually drives the arm via `controller_core`'s existing velocity or torque controllers
  while the pendulum swings freely was not built.
- **No real-hardware or dimension verification** -- every placeholder value above needs
  checking against either the actual SolidWorks file (on a machine with SolidWorks) or the
  physical assembly once built.
- **No actuation/control on the pendulum joint** -- it is currently a free (unactuated),
  lightly-damped hinge, matching a real passive pendulum. This is intentional for a cartpole-
  style swing-up task (the pendulum should NOT be directly actuated) but is worth stating
  explicitly since nothing prevents adding a motor to this joint by mistake later.

## Files

| path | what |
|---|---|
| `assets/ur5e_pendulum/pendulum_attachment.xml` | pendulum MJCF fragment, full dimension provenance in its own docstring |
| `simulation/ur5e_pendulum_compose.py` | `MjSpec.attach()`-based composition helper |
| `tools/diagnostics/ur5e_pendulum_smoke_test.py` | standalone smoke-test script |
| `tests/mujoco/test_ur5e_pendulum_compose.py` | 8 pytest tests, all passing |

**Tests run:** `tests/mujoco/test_ur5e_pendulum_compose.py` (8/8 passed).
**Tests not run:** full repo suite (no existing file was modified by this work, only new files
added, so no regression path exists through untouched code).

**Rollback:**
```bash
rm -rf assets/ur5e_pendulum simulation/ur5e_pendulum_compose.py \
       tools/diagnostics/ur5e_pendulum_smoke_test.py tests/mujoco/test_ur5e_pendulum_compose.py \
       docs/status/ur5e_pendulum_cad_model_2026-08-09.md
```
Nothing else was touched; `assets/ur5e_torque/ur5e_torque.xml` is provably unmodified (see
`test_centerpiece_model_untouched`). Nothing committed.
