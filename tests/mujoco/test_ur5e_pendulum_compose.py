"""Tests for simulation/ur5e_pendulum_compose.py -- composing a physical
UR5e-mounted pendulum apparatus (assets/ur5e_pendulum/, see that directory's
own docstring for the real-vs-placeholder dimension provenance) onto the
protected centerpiece UR5e model via mujoco.MjSpec.attach().

PARAMETERIZED OVER BOTH PENDULUM ASSETS (2026-08-13). Before this date every
test here ran only against DEFAULT_PENDULUM_XML (the real 0.12 m apparatus) and
several additionally hardwired constants/poses imported from
pendulum_swingup_energy_shaping, so nothing in the suite could have caught an
asset-specific regression -- which is precisely what blocked using the
alternate 0.30 m long-rod asset.

The two assets are NOT interchangeable and the parameterization does not
pretend they are: they differ in hinge axis (local Z vs local X), rod length,
housing standoff, damping regime (overdamped vs underdamped), and -- most
importantly -- in the arm pose at which their hinge is usable at all. Those
differences live in the PendulumAssetCase records below, one per asset, so a
shared test body asserts the same PROPERTY against per-asset expectations
rather than a shared number.
"""

from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

import mujoco
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from simulation.ur5e_pendulum_compose import (  # noqa: E402
    DEFAULT_ARM_Q,
    DEFAULT_ARM_XML,
    DEFAULT_PENDULUM_XML,
    LONGROD_ARM_Q,
    LONGROD_PENDULUM_XML,
    PendulumConstants,
    arm_q_for_pendulum_xml,
    compose_ur5e_pendulum_model,
    compose_ur5e_pendulum_spec,
    derive_pendulum_constants,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclasses.dataclass(frozen=True)
class PendulumAssetCase:
    """One (asset, validated arm pose) pair plus the per-asset expectations a
    shared test body needs. Every number here is a property of the MJCF and is
    asserted against the compiled model, not assumed."""

    name: str
    xml: Path
    arm_q: np.ndarray
    hinge_axis_local: tuple[float, float, float]
    rod_length_m: float
    housing_standoff_m: float
    isolated_gravity: tuple[float, float, float]
    """Gravity to use when exercising the ISOLATED fragment's own swing
    dynamics. The fragment compiles with world gravity along -Z; if its hinge
    axis is also local Z the axis is exactly vertical there and produces zero
    torque at every angle, so there is no swing to test. Rotating gravity into
    the swing plane is exact -- the fragment's mass distribution about the
    hinge does not depend on which perpendicular direction gravity points."""
    isolated_axis_is_world_vertical: bool
    """Whether, in the isolated fragment with the default -Z gravity, the hinge
    axis is parallel to gravity (and so makes zero torque at every angle)."""
    overdamped_as_committed: bool
    """Whether the committed damping/frictionloss placeholders put this asset
    above critical damping. The 0.12 m asset is (zeta ~ 3.9, it cannot swing
    through the bottom at all); the 0.30 m one is not (zeta ~ 0.4), because
    zeta scales as b / (2*sqrt(I*m*g*r)) and the long rod has ~15x the inertia
    and ~6x the gravity torque."""


ASSET_CASES = [
    PendulumAssetCase(
        name="default_0.12m",
        xml=DEFAULT_PENDULUM_XML,
        arm_q=DEFAULT_ARM_Q,
        hinge_axis_local=(0.0, 0.0, 1.0),
        rod_length_m=0.12,
        housing_standoff_m=0.06,
        isolated_gravity=(-9.81, 0.0, 0.0),
        isolated_axis_is_world_vertical=True,
        overdamped_as_committed=True,
    ),
    PendulumAssetCase(
        name="longrod_0.30m",
        xml=LONGROD_PENDULUM_XML,
        arm_q=LONGROD_ARM_Q,
        hinge_axis_local=(1.0, 0.0, 0.0),
        rod_length_m=0.30,
        housing_standoff_m=0.12,
        isolated_gravity=(0.0, 0.0, -9.81),
        isolated_axis_is_world_vertical=False,
        overdamped_as_committed=False,
    ),
]

CASES = pytest.mark.parametrize("case", ASSET_CASES, ids=[c.name for c in ASSET_CASES])
XMLS = pytest.mark.parametrize(
    "pendulum_xml", [c.xml for c in ASSET_CASES], ids=[c.name for c in ASSET_CASES]
)

PENDULUM_HINGE = "/pendulum_hinge"


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def _hinge_addresses(model):
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, PENDULUM_HINGE)
    assert jid >= 0
    return jid, int(model.jnt_qposadr[jid]), int(model.jnt_dofadr[jid])


def _bias_at(model, arm_q, qadr, dadr, theta):
    d = mujoco.MjData(model)
    d.qpos[:6] = arm_q
    d.qpos[qadr] = float(theta)
    d.qvel[:] = 0.0
    mujoco.mj_forward(model, d)
    return float(d.qfrc_bias[dadr])


def _hanging_angle(model, arm_q, qadr, dadr, n=721):
    """The STABLE equilibrium, found over the FULL circle rather than a
    hardcoded bracket.

    The pre-2026-08-13 tests bisected [-pi, 0] on the assumption that hanging
    lands there -- true for the default asset at its own pose and for nothing
    else. qfrc_bias's zero with POSITIVE slope is the stable one: the equation
    of motion is qdd = -qfrc_bias/M, so a positive slope means the bias pushes
    back toward the zero."""
    ths = np.linspace(-np.pi, np.pi, n)
    vals = np.array([_bias_at(model, arm_q, qadr, dadr, t) for t in ths])
    for i in range(len(ths) - 1):
        if vals[i] <= 0.0 < vals[i + 1]:  # negative -> positive: positive slope
            lo, hi = ths[i], ths[i + 1]
            for _ in range(60):
                mid = (lo + hi) / 2
                if _bias_at(model, arm_q, qadr, dadr, mid) > 0:
                    hi = mid
                else:
                    lo = mid
            return (lo + hi) / 2
    raise AssertionError("no stable (positive-slope) gravity-torque zero found")


def _hold_arm_torque(model, data, arm_q):
    """Smooth gravity-compensating + stiff joint PD hold.

    NEVER pin the arm by writing qpos/qvel each step: the discontinuity injects
    momentum through the arm/pendulum mass-matrix coupling, an artifact this
    subsystem has already had to remove three times (see
    pendulum_balance_torque_lqr.py::find_inverted_angle's 2026-08-11 note)."""
    from tools.diagnostics.pendulum_balance_torque_lqr import static_gravity_torque

    return (static_gravity_torque(model, data.qpos)[:6]
            + 800.0 * (arm_q - data.qpos[:6]) - 60.0 * data.qvel[:6])


# --------------------------------------------------------------------------
# composition
# --------------------------------------------------------------------------


@XMLS
def test_centerpiece_model_untouched(pendulum_xml):
    """Composing must never modify assets/ur5e_torque/ur5e_torque.xml on
    disk -- the protected centerpiece model (AGENTS.md sec 2)."""
    before = DEFAULT_ARM_XML.read_bytes()
    compose_ur5e_pendulum_model(pendulum_xml=pendulum_xml)
    after = DEFAULT_ARM_XML.read_bytes()
    assert before == after


@XMLS
def test_compose_produces_expected_dof_count(pendulum_xml):
    model = compose_ur5e_pendulum_model(pendulum_xml=pendulum_xml)
    # 6 arm joints + 1 pendulum hinge.
    assert model.nq == 7
    assert model.nv == 7
    joint_names = {model.joint(i).name for i in range(model.njnt)}
    assert joint_names == {
        "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
        "wrist_1_joint", "wrist_2_joint", "wrist_3_joint", PENDULUM_HINGE,
    }


@XMLS
def test_compose_raises_on_missing_attachment_site(pendulum_xml):
    with pytest.raises(ValueError, match="no site named"):
        compose_ur5e_pendulum_spec(pendulum_xml=pendulum_xml,
                                   attachment_site="not_a_real_site")


def test_compose_raises_on_missing_files():
    with pytest.raises(FileNotFoundError):
        compose_ur5e_pendulum_spec(arm_xml=REPO_ROOT / "assets" / "does_not_exist.xml")
    with pytest.raises(FileNotFoundError):
        compose_ur5e_pendulum_spec(pendulum_xml=REPO_ROOT / "assets" / "does_not_exist.xml")


@XMLS
def test_compose_spec_allows_custom_worldbody_additions(pendulum_xml):
    """compose_ur5e_pendulum_spec returns an uncompiled MjSpec specifically
    so callers can add scene content (matching assets/ur5e_torque/scene.xml's
    pattern) before compiling -- confirm that actually works."""
    spec = compose_ur5e_pendulum_spec(pendulum_xml=pendulum_xml)
    geom = spec.worldbody.add_geom()
    geom.name = "custom_test_marker"
    geom.type = mujoco.mjtGeom.mjGEOM_SPHERE
    geom.size = [0.01, 0, 0]
    model = spec.compile()
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "custom_test_marker") >= 0


@XMLS
def test_pendulum_angle_sensor_matches_joint_qpos(pendulum_xml):
    model = compose_ur5e_pendulum_model(pendulum_xml=pendulum_xml)
    _, qadr, _ = _hinge_addresses(model)
    sensor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, "/pendulum_angle_sensor")
    assert sensor_id >= 0
    sensor_adr = model.sensor_adr[sensor_id]

    data = mujoco.MjData(model)
    data.qpos[qadr] = 0.37
    mujoco.mj_forward(model, data)
    assert data.sensordata[sensor_adr] == pytest.approx(0.37, abs=1e-9)


# --------------------------------------------------------------------------
# asset <-> pose registry
# --------------------------------------------------------------------------


@CASES
def test_registered_arm_pose_round_trips(case):
    np.testing.assert_allclose(arm_q_for_pendulum_xml(case.xml), case.arm_q, rtol=0, atol=0)
    # Accepts an unresolved/relative spelling of the same file too.
    np.testing.assert_allclose(arm_q_for_pendulum_xml(str(case.xml)), case.arm_q, rtol=0, atol=0)


def test_unregistered_asset_raises_rather_than_borrowing_a_pose():
    """A hinge axis is only usable at the pose it was measured at, and a
    borrowed pose fails SILENTLY (the model still compiles, still has 7 DOF,
    still reports two equilibria -- it just never swings). So an unknown asset
    must be an error, not a default."""
    with pytest.raises(KeyError, match="no validated arm pose"):
        arm_q_for_pendulum_xml(REPO_ROOT / "assets" / "ur5e_pendulum" / "not_an_asset.xml")


# --------------------------------------------------------------------------
# isolated fragment geometry / dynamics
# --------------------------------------------------------------------------


@CASES
def test_isolated_fragment_hinge_axis_and_rod_geometry(case):
    """Standalone, the fragment's hinge axis and rod geometry must be exactly
    what the MJCF claims, and the rod must be PERPENDICULAR to the hinge.

    A rod collinear with its own hinge is an offset crank that spins about its
    centreline with zero gravity torque, never a pendulum -- a real failure
    mode this apparatus has actually shipped once (see the rod comment in
    assets/ur5e_pendulum/pendulum_attachment.xml's 2026-08-12 note).

    The decomposition is asset-agnostic on purpose: the rod-to-tip vector
    splits into a component ALONG the hinge (the shaft-housing standoff, which
    buys wrist clearance and changes no dynamics -- a translation along the
    axis leaves both the gravity torque about it and the moment of inertia
    about it exactly unchanged) and a component PERPENDICULAR to it (the rod's
    own swing radius). The 0.12 m asset carries 0.06 m of standoff along local
    Z with a 0.12 m rod in the XY plane; the 0.30 m asset carries 0.12 m of
    standoff along local X with a 0.30 m rod along Z. Same property, different
    numbers."""
    model = mujoco.MjSpec.from_file(str(case.xml)).compile()
    data = mujoco.MjData(model)

    axis = np.asarray(model.jnt_axis[0], dtype=float)
    np.testing.assert_allclose(axis, np.asarray(case.hinge_axis_local, dtype=float), atol=1e-12)
    axis = axis / np.linalg.norm(axis)

    if case.isolated_axis_is_world_vertical:
        # Axis parallel to the fragment's -Z gravity: exactly zero torque at
        # every angle. Asserted exactly (== 0.0) because it is a structural
        # consequence of axis-parallel-to-gravity, not a near-cancellation.
        for theta in np.linspace(-np.pi, np.pi, 37):
            data.qpos[0] = float(theta)
            data.qvel[:] = 0.0
            mujoco.mj_forward(model, data)
            assert float(data.qfrc_bias[0]) == 0.0, (
                f"gravity torque about the hinge is nonzero ({float(data.qfrc_bias[0])}) at "
                f"theta={theta:.3f} in the ISOLATED fragment, but this asset's axis "
                f"{case.hinge_axis_local} is the mounting-face normal and therefore vertical here"
            )
    else:
        peak = 0.0
        for theta in np.linspace(-np.pi, np.pi, 37):
            data.qpos[0] = float(theta)
            data.qvel[:] = 0.0
            mujoco.mj_forward(model, data)
            peak = max(peak, abs(float(data.qfrc_bias[0])))
        assert peak > 0.01, (
            f"this asset's hinge axis {case.hinge_axis_local} lies in the mounting face, so it "
            f"is horizontal under the fragment's own -Z gravity and must make real torque; "
            f"measured peak {peak:.6f} Nm"
        )

    data.qpos[0] = 0.0
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)
    hinge = np.array(data.site_xpos[
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "pendulum_hinge_site")])
    tip = np.array(data.site_xpos[
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "pendulum_tip_site")])
    rod_vec = tip - hinge

    along = float(np.dot(rod_vec, axis))
    perp = rod_vec - along * axis
    assert abs(along) == pytest.approx(case.housing_standoff_m, abs=1e-9)
    assert float(np.linalg.norm(perp)) == pytest.approx(case.rod_length_m, abs=1e-9)
    assert abs(float(np.dot(perp, axis))) < 1e-12, (
        "the rod's swing component has an along-hinge-axis part -- a rod collinear with its "
        "own hinge is an offset crank that never swings"
    )


@CASES
def test_isolated_pendulum_settles_at_true_hanging_equilibrium(case):
    """The fragment alone (no arm, no attachment-site rotation) -- does this
    behave like a real pendulum, decoupled from any arm-pose rotation offset.

    Split into two parts because the committed hinge damping (0.02 N m s/rad)
    and frictionloss (0.01 Nm) are explicit unmeasured PLACEHOLDERS whose
    damping REGIME differs between the two assets: at the 0.12 m rod's inertia
    they are 3.9x critical (it cannot swing through the bottom at all), at the
    0.30 m rod's they are ~0.4x (it genuinely oscillates). Part 1 therefore
    tests real pendulum dynamics on a friction-free scratch model, identically
    for both; part 2 tests what each asset AS COMMITTED actually does, and the
    damping regime itself is asserted rather than assumed.

    Both assets place hanging at theta = pi in their own fragment frame (the
    rod points "up" at theta = 0 relative to the gravity direction used here),
    so the shared assertions below are genuinely shared, not coincidentally
    aligned."""
    # --- Part 0: the damping regime this asset is actually in. ---
    model = mujoco.MjSpec.from_file(str(case.xml)).compile()
    model.opt.gravity[:] = case.isolated_gravity
    constants = derive_pendulum_constants(model, arm_q=None)
    b_crit = 2.0 * np.sqrt(constants.i_pivot_kgm2 * constants.mgr_nm)
    zeta = float(model.dof_damping[0]) / b_crit
    # bool() is load-bearing: numpy comparisons return np.bool_, and
    # `np.bool_(True) is True` is False.
    assert bool(zeta > 1.0) is case.overdamped_as_committed, (
        f"damping ratio {zeta:.3f} contradicts this asset's recorded regime "
        f"(overdamped_as_committed={case.overdamped_as_committed}) -- the committed damping "
        f"placeholder or the rod geometry changed; both are real model questions, not test noise"
    )

    # --- Part 1: real pendulum dynamics, friction/damping removed. ---
    model = mujoco.MjSpec.from_file(str(case.xml)).compile()
    model.opt.gravity[:] = case.isolated_gravity
    model.dof_damping[0] = 0.0
    model.dof_frictionloss[0] = 0.0
    data = mujoco.MjData(model)
    data.qpos[0] = np.pi / 2  # released pi/2 away from hanging (which is pi)
    mujoco.mj_forward(model, data)
    hist = []
    for _ in range(6000):
        mujoco.mj_step(model, data)
        hist.append(float(data.qpos[0]))
    hist = np.array(hist)

    crossings = np.where(np.diff(np.sign(hist - np.pi)) != 0)[0]
    assert len(crossings) >= 10, (
        f"undamped fragment crossed hanging only {len(crossings)} times -- it is not "
        f"oscillating like a pendulum"
    )
    # Conservative: swings symmetrically about hanging, no energy growth.
    assert hist.min() > np.pi / 2 - 0.02 and hist.max() < 3 * np.pi / 2 + 0.02, (
        f"undamped swing range [{hist.min():.4f}, {hist.max():.4f}] is not the energy-conserving "
        f"[pi/2, 3pi/2] about hanging -- integrator or geometry problem"
    )

    # --- Part 2: the fragment exactly as committed. ---
    model = mujoco.MjSpec.from_file(str(case.xml)).compile()
    model.opt.gravity[:] = case.isolated_gravity
    data = mujoco.MjData(model)
    data.qpos[0] = np.pi / 2
    mujoco.mj_forward(model, data)
    hist = []
    for _ in range(6000):
        mujoco.mj_step(model, data)
        hist.append(float(data.qpos[0]))
    hist = np.array(hist)

    if case.overdamped_as_committed:
        assert np.all(np.diff(hist) >= -1e-9), (
            "overdamped fragment should fall monotonically toward hanging"
        )
    else:
        assert np.any(hist > np.pi), (
            "underdamped fragment should overshoot hanging (pi) at least once"
        )
    assert hist[-1] - hist[0] > 1.0, (
        f"pendulum barely moved under gravity (start {hist[0]:.4f} -> end {hist[-1]:.4f}) -- "
        f"the joint may be effectively locked"
    )
    # Coulomb frictionloss stiction-locks the fragment short of true hanging;
    # how far short is asin(frictionloss / m*g*r), which differs per asset
    # (0.29 rad for the 0.12 m rod, 0.06 rad for the 0.30 m one). Assert
    # against that measured physical bound plus margin, not a shared literal.
    lock_offset = float(np.arcsin(
        min(1.0, float(model.dof_frictionloss[0]) / constants.mgr_nm)))
    assert abs(hist[-1] - np.pi) < lock_offset + 0.1, (
        f"did not come to rest near hanging (pi), got {hist[-1]:.4f} "
        f"(stiction lock band is +-{lock_offset:.4f} rad)"
    )
    assert hist[-200:].std() < 0.01, "pendulum still moving at end of trace"


# --------------------------------------------------------------------------
# composed model dynamics
# --------------------------------------------------------------------------


@CASES
def test_composed_pendulum_settles_under_gravity(case):
    """Arm-attached case: released 90 deg off hanging, the pendulum must
    actually swing down and settle at the model's OWN hanging equilibrium.

    Runs at each asset's OWN validated arm pose. Using the wrong pose here is
    not a cosmetic mismatch: at a pose where the hinge axis maps to world -Z
    the gravity torque is identically zero at every angle, and this test would
    be asserting that a pendulum with no gravity torque settles -- vacuous.
    (That was literally the pre-2026-08-12 state of this test.)

    The arm is held by a smooth gravity-compensating + PD holding torque, never
    by per-step qpos/qvel pinning -- see _hold_arm_torque."""
    model = compose_ur5e_pendulum_model(pendulum_xml=case.xml)
    _, qadr, dadr = _hinge_addresses(model)
    hanging = _hanging_angle(model, case.arm_q, qadr, dadr)

    data = mujoco.MjData(model)
    data.qpos[:6] = case.arm_q
    data.qpos[qadr] = hanging + np.pi / 2
    mujoco.mj_forward(model, data)

    angle_hist = []
    for _ in range(8000):
        data.ctrl[:6] = _hold_arm_torque(model, data, case.arm_q)
        mujoco.mj_step(model, data)
        angle_hist.append(float(data.qpos[qadr]))
    angle_hist = np.array(angle_hist)

    assert abs(angle_hist[0] - angle_hist[-1]) > 1.0, (
        f"pendulum moved only {abs(angle_hist[0] - angle_hist[-1]):.4f} rad from a 90 deg release "
        f"-- the joint is effectively locked or the hinge axis has lost its gravity torque"
    )
    assert abs(angle_hist[-1] - hanging) < 0.2, (
        f"settled at {angle_hist[-1]:.4f} rad, {abs(angle_hist[-1] - hanging):.4f} rad from the "
        f"model's own hanging equilibrium ({hanging:.4f} rad)"
    )
    assert angle_hist[-200:].std() < 0.01, "pendulum did not converge/settle"


@CASES
def test_hinge_axis_is_usable_as_a_pendulum(case):
    """The hinge axis must actually be able to work as an X-driven pendulum at
    the pose this asset is registered for.

    Added 2026-08-12 after a proposed hinge-axis change was found to produce a
    silently INOPERATIVE pendulum: the model still compiles, still has 7 DOF,
    the sensor still reads back, and find_inverted_angle still returns two
    equilibria (it reads qfrc_bias, which ignores Coulomb friction entirely).
    The failure only shows up as swing-up runs that quietly never leave the
    hanging region.

    Three independent conditions:

    1. AUTHORITY OVER GRAVITY. Peak gravity torque about the hinge must exceed
       this joint's own Coulomb frictionloss, or the pendulum is stiction-locked
       at every angle. The bar is deliberately just >1.0x (the physical lock
       condition), not today's specific margin.
    2. HORIZONTALITY. Gravity torque scales with how far the axis is from world
       vertical.
    3. PUMPING AUTHORITY. The swing-up law drives the pendulum only through
       world-X acceleration of the pivot, whose torque about the hinge scales
       with |axis x world_X|. An axis PARALLEL to the travel direction gets zero
       coupling and is just as unusable as a vertical one -- a distinct failure
       the gravity check cannot see.

    Parameterized 2026-08-13. This test used to be named ..._at_arm_q0 and
    hardwired the default asset's pose; the long-rod asset fails every one of
    these conditions at that pose and passes all three at its own."""
    model = compose_ur5e_pendulum_model(pendulum_xml=case.xml)
    jid, qadr, dadr = _hinge_addresses(model)
    hub_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "/pendulum_hub")

    peak_gravity_torque = max(
        abs(_bias_at(model, case.arm_q, qadr, dadr, th))
        for th in np.linspace(-np.pi, np.pi, 181)
    )
    frictionloss = float(model.dof_frictionloss[dadr])
    assert peak_gravity_torque > frictionloss, (
        f"peak gravity torque about the hinge ({peak_gravity_torque:.6f} Nm) does not exceed "
        f"the joint's own frictionloss ({frictionloss:.6f} Nm) at this asset's registered pose "
        f"-- the pendulum is stiction-locked at every angle and cannot swing."
    )

    d = mujoco.MjData(model)
    d.qpos[:6] = case.arm_q
    d.qpos[qadr] = 0.0
    d.qvel[:] = 0.0
    mujoco.mj_forward(model, d)
    axis_world = d.xmat[hub_bid].reshape(3, 3) @ np.asarray(model.jnt_axis[jid], dtype=float)
    axis_world /= np.linalg.norm(axis_world)

    horizontality = float(np.sqrt(max(0.0, 1.0 - axis_world[2] ** 2)))
    assert horizontality > 0.5, (
        f"hinge axis is {np.degrees(np.arcsin(horizontality)):.1f} deg from vertical "
        f"(world {np.round(axis_world, 4)}) -- gravity torque scales with this"
    )

    pump_authority = float(np.linalg.norm(np.cross(axis_world, [1.0, 0.0, 0.0])))
    assert pump_authority > 0.5, (
        f"hinge axis is nearly parallel to the world-X transport direction "
        f"(|axis x xhat| = {pump_authority:.4f}, world {np.round(axis_world, 4)}) -- the arm's X "
        f"motion produces almost no torque about the hinge, so the swing-up law has no authority"
    )


# --------------------------------------------------------------------------
# derived constants
# --------------------------------------------------------------------------


@CASES
def test_derived_constants_match_the_model_gravity_torque(case):
    """derive_pendulum_constants must reproduce the composed model's OWN
    gravity torque and mass matrix, for every asset.

    This replaces test_swingup_constants_match_the_model_gravity_torque, which
    checked four hand-cached literals in pendulum_swingup_energy_shaping. Those
    literals are gone (they were correct only for one asset at one pose); what
    is checkable now is that the derivation itself is right, so the assertions
    move onto the helper.

    MuJoCo's qfrc_bias about this joint is exactly sinusoidal in the angle (the
    hub's COM sits ON the axis, so only the rod contributes) and its amplitude
    IS m*g*r_effective -- a direct, assumption-free ground truth needing no
    settling and no simulation."""
    model = compose_ur5e_pendulum_model(pendulum_xml=case.xml)
    _, qadr, dadr = _hinge_addresses(model)
    constants = derive_pendulum_constants(model, case.arm_q)

    thetas = np.linspace(-np.pi, np.pi, 181)  # deliberately NOT the helper's 721
    vals = np.array([_bias_at(model, case.arm_q, qadr, dadr, th) for th in thetas])
    basis = np.stack([np.sin(thetas), np.cos(thetas)], axis=1)
    coef, *_ = np.linalg.lstsq(basis, vals, rcond=None)
    mgr_model = float(np.hypot(*coef))

    # Pure sinusoid: if this fails the "amplitude == m*g*r" reading is unsafe.
    assert np.max(np.abs(vals - basis @ coef)) < 1e-9
    # And the peak of |qfrc_bias| is the same quantity by a third route. The
    # tolerance is set by GRID SAMPLING, not by model error: a 181-point sweep
    # can miss the true peak by up to half a step (0.0175 rad), which
    # undershoots a cosine by ~1.5e-4 relative. Hence 1e-3, not 1e-9 -- this
    # route is a sanity cross-check on the fit, and the fit above is the
    # precise one.
    assert float(np.max(np.abs(vals))) == pytest.approx(mgr_model, rel=1e-3)

    assert constants.mgr_nm == pytest.approx(mgr_model, rel=1e-9)
    assert constants.e_top_j == pytest.approx(2.0 * mgr_model, rel=1e-9)
    assert constants.m_total_kg * constants.g * constants.r_com_m == pytest.approx(
        mgr_model, rel=1e-9)

    d = mujoco.MjData(model)
    d.qpos[:6] = case.arm_q
    d.qpos[qadr] = 0.0
    d.qvel[:] = 0.0
    mujoco.mj_forward(model, d)
    full_m = np.zeros((model.nv, model.nv))
    mujoco.mj_fullM(model, full_m, d.qM)
    assert constants.i_pivot_kgm2 == pytest.approx(float(full_m[dadr, dadr]), rel=1e-12)

    assert constants.omega_natural_radps == pytest.approx(
        np.sqrt(constants.mgr_nm / constants.i_pivot_kgm2), rel=1e-12)
    assert constants.t_natural_s == pytest.approx(
        2.0 * np.pi / constants.omega_natural_radps, rel=1e-12)
    assert tuple(constants.arm_q) == tuple(float(v) for v in case.arm_q)


def test_derived_constants_reproduce_the_historical_hand_cached_values():
    """Regression pin for the DEFAULT asset only.

    The 2026-08-13 change replaced four literals with a derivation; this checks
    the derivation lands on the same numbers those literals held, so the change
    cannot have quietly moved the default asset's physics. Agreement is to 1
    ULP (relative ~1.2e-16) on m*g*r and E_top and exact on the other three --
    the literals were themselves qfrc_bias-fitted, over a 361-point grid rather
    than this helper's 721."""
    model = compose_ur5e_pendulum_model()
    c = derive_pendulum_constants(model, DEFAULT_ARM_Q)
    assert c.m_total_kg == pytest.approx(0.097350084474905368, rel=1e-15)
    assert c.r_com_m == pytest.approx(0.029183068431971784, rel=1e-15)
    assert c.i_pivot_kgm2 == pytest.approx(0.00023746980581744164, rel=1e-15)
    assert c.e_top_j == pytest.approx(0.05573991335449398, rel=1e-15)
    assert c.t_natural_s == pytest.approx(0.5799838761307066, rel=1e-15)


def test_derived_constants_are_pose_dependent_not_asset_properties():
    """The regression test for the whole 2026-08-13 change.

    m*g*r is the gravity torque about the hinge IN WORLD, so it depends on how
    far the arm pose tilts the hinge axis from vertical -- it is a property of
    the (asset, pose) PAIR. Using one pair's constants for another is silently
    wrong, not an error, which is exactly why the old hand-cached literals
    blocked the alternate asset."""
    longrod = compose_ur5e_pendulum_model(pendulum_xml=LONGROD_PENDULUM_XML)
    at_own = derive_pendulum_constants(longrod, LONGROD_ARM_Q)
    at_borrowed = derive_pendulum_constants(longrod, DEFAULT_ARM_Q)

    # Inertia about the hinge is pose-independent (it is the same body).
    assert at_own.i_pivot_kgm2 == pytest.approx(at_borrowed.i_pivot_kgm2, rel=1e-12)
    assert at_own.m_total_kg == pytest.approx(at_borrowed.m_total_kg, rel=1e-12)
    # Gravity torque and everything derived from it are NOT.
    assert at_own.mgr_nm / at_borrowed.mgr_nm > 5.0
    assert at_borrowed.t_natural_s / at_own.t_natural_s > 2.0

    # And the two assets differ from each other at their own poses.
    default = derive_pendulum_constants(compose_ur5e_pendulum_model(), DEFAULT_ARM_Q)
    assert at_own.mgr_nm / default.mgr_nm > 5.0
    assert at_own.i_pivot_kgm2 / default.i_pivot_kgm2 > 10.0
    assert at_own.t_natural_s > default.t_natural_s


def test_derive_pendulum_constants_rejects_a_vertical_hinge_axis():
    """At a pose that maps the hinge axis onto world vertical there is no
    pendulum, and the helper must say so rather than hand back a near-zero
    m*g*r that downstream code would divide by."""
    model = compose_ur5e_pendulum_model()
    with pytest.raises(ValueError, match="not a pure sinusoid|axis is parallel to gravity"):
        # The isolated-fragment orientation: hinge axis exactly along gravity.
        frag = mujoco.MjSpec.from_file(str(DEFAULT_PENDULUM_XML)).compile()
        derive_pendulum_constants(frag, arm_q=None)
    assert model.nq == 7  # sanity, the composed model is unaffected


@CASES
def test_pendulum_natural_period_matches_derived_constants(case):
    """The analytic small-oscillation period the phase-locked drive seeds
    itself with must match the model's real free-swing period, per asset.

    Damping and frictionloss are temporarily zeroed on a scratch model so this
    measures the undamped natural period the formula actually predicts."""
    model = compose_ur5e_pendulum_model(pendulum_xml=case.xml)
    _, qadr, dadr = _hinge_addresses(model)
    constants = derive_pendulum_constants(model, case.arm_q)
    model.dof_damping[dadr] = 0.0
    model.dof_frictionloss[dadr] = 0.0

    hanging = _hanging_angle(model, case.arm_q, qadr, dadr)

    d = mujoco.MjData(model)
    d.qpos[:6] = case.arm_q
    d.qpos[qadr] = hanging + 0.05  # small angle, so the linear period applies
    d.qvel[:] = 0.0
    mujoco.mj_forward(model, d)
    dt = float(model.opt.timestep)
    hist = []
    for _ in range(int(6.0 / dt)):
        d.ctrl[:6] = _hold_arm_torque(model, d, case.arm_q)
        mujoco.mj_step(model, d)
        hist.append(float(d.qpos[qadr]) - hanging)
    hist = np.array(hist)
    zc = np.where(np.diff(np.sign(hist)) != 0)[0]
    assert len(zc) >= 3, "pendulum did not oscillate with friction removed"
    measured_T = 2.0 * float(np.mean(np.diff(zc))) * dt
    assert measured_T == pytest.approx(constants.t_natural_s, rel=0.05), (
        f"measured free period {measured_T:.4f}s != derived t_natural_s "
        f"{constants.t_natural_s:.4f}s -- the phase-locked drive would seed itself at the "
        f"wrong frequency"
    )


def test_pendulum_constants_from_mass_and_arm_matches_derivation():
    """The scalar constructor (used to reproduce historical constant sets
    bit-for-bit in equivalence tests) must agree with the derived bundle."""
    model = compose_ur5e_pendulum_model()
    c = derive_pendulum_constants(model, DEFAULT_ARM_Q)
    rebuilt = PendulumConstants.from_mass_and_arm(
        m_total_kg=c.m_total_kg, r_com_m=c.r_com_m, i_pivot_kgm2=c.i_pivot_kgm2, g=c.g)
    for field in ("mgr_nm", "i_pivot_kgm2", "m_total_kg", "r_com_m",
                  "omega_natural_radps", "t_natural_s", "e_top_j"):
        assert getattr(rebuilt, field) == pytest.approx(getattr(c, field), rel=1e-15)
