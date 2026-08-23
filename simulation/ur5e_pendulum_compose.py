"""Compose the real UR5e torque model with the pendulum attachment fragment
(assets/ur5e_pendulum/pendulum_attachment.xml) at the arm's existing
attachment_site, via mujoco.MjSpec.attach() -- never edits or duplicates
assets/ur5e_torque/ur5e_torque.xml (the protected centerpiece model, see
AGENTS.md sec 2).

This is a composition helper, not a controller -- it hands back a compiled
mujoco.MjModel with both the arm and the pendulum in it. Model-based
dynamics/kinematics code that needs the pendulum attached should build its
model via compose_ur5e_pendulum_spec()/compose_ur5e_pendulum_model() rather
than loading assets/ur5e_torque/scene.xml directly.

See assets/ur5e_pendulum/pendulum_attachment.xml's own docstring for the
full provenance of every physical dimension used -- most of the pendulum's
own geometry is a labeled PLACEHOLDER (the real CAD is a SolidWorks
assembly this environment cannot read), only the tool-changer bounding box
and the 8mm rod/shaft diameter are extracted/inferred from the real
archive. Do not trust this model for anything beyond a first-pass
kinematic/dynamic sanity check until those placeholders are corrected.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARM_XML = REPO_ROOT / "assets" / "ur5e_torque" / "ur5e_torque.xml"
DEFAULT_PENDULUM_XML = REPO_ROOT / "assets" / "ur5e_pendulum" / "pendulum_attachment.xml"
# ALTERNATE asset (2026-08-13): the 0.30 m "long-rod" cartpole variant. Not a
# replacement for and not a revert of DEFAULT_PENDULUM_XML -- see that file's
# own header for the swing-up bandwidth argument that motivates it. Named here
# (rather than left as a bare path every caller re-spells) so the asset and the
# arm pose it is only valid at travel together; see PENDULUM_ASSET_ARM_Q below.
LONGROD_PENDULUM_XML = REPO_ROOT / "assets" / "ur5e_pendulum" / "pendulum_attachment_longrod.xml"
# THE WORKING asset (2026-08-14): real 0.12 m rod, local-X hinge (the
# long-rod's verified-usable hinge geometry), REAL measured hinge friction
# (damping 0.0001 Nms/rad, frictionloss 0.001 Nm -- NOT the other two assets'
# unvalidated placeholder friction). Paired with OLD_POSE/LONGROD_ARM_Q, the
# same well-conditioned pose the long-rod asset uses (cond(J)=6.93), since the
# hinge geometry is shared. See the asset's own header comment for the full
# rationale (why neither of the other two assets is usable here).
REALROD_PENDULUM_XML = REPO_ROOT / "assets" / "ur5e_pendulum" / "pendulum_attachment_realrod.xml"
ATTACHMENT_SITE = "attachment_site"

# ---------------------------------------------------------------------------
# Asset <-> arm-pose pairing.
#
# A pendulum asset is NOT usable at an arbitrary arm pose: the hinge axis must
# be (a) far from world-vertical, or gravity makes no torque about it, and
# (b) far from world-X, or the arm's transport motion makes no torque about it
# either. Both conditions are properties of (axis, pose) TOGETHER -- see the
# joint element comments in both MJCF files, which say so explicitly, and
# tests/mujoco/test_ur5e_pendulum_compose.py::
# test_hinge_axis_is_usable_as_a_pendulum, which asserts them per asset.
# Keeping the pairing here (next to the asset paths) is what makes
# "point a script at the other asset" a single flag rather than a silent
# mismatch: every CLI below defaults --start-q-rad from this table.
# ---------------------------------------------------------------------------

# The user's actual real-hardware UR5e configuration (wrist_2 wrapped into the
# model's valid range from a real 6.2879 rad probe). Sits essentially on the
# wrist_2=0 arm singularity (cond(J6)=1396) -- a separate, deliberately
# not-conflated arm-side concern, see the swing-up scripts' own notes.
DEFAULT_ARM_Q = np.array([-2.3688, -2.1801, -1.8838, -0.7962, 0.004714693, 0.0206])
# The "old" flip pose: cond(J)=6.93, wrist_2=+89.5 deg, i.e. well-conditioned.
# This is the pose the long-rod asset's own joint comment names as the one its
# local-X hinge was verified operable at (89.51 deg from vertical,
# |axis . xhat| = 0.0017 -- i.e. essentially full X-pumping coupling).
LONGROD_ARM_Q = np.array([
    0.0, -1.091985784398452, 2.0935362786892546,
    -2.7685637962327356, 1.5620693866337145, 0.0,
])

PENDULUM_ASSET_ARM_Q: dict[str, np.ndarray] = {
    str(DEFAULT_PENDULUM_XML.resolve()): DEFAULT_ARM_Q,
    str(LONGROD_PENDULUM_XML.resolve()): LONGROD_ARM_Q,
    # Same pose as the long-rod asset -- shares its local-X hinge geometry.
    str(REALROD_PENDULUM_XML.resolve()): LONGROD_ARM_Q,
}

GRAVITY_MPS2 = 9.81

# Joint/body names differ between the ISOLATED fragment (no prefix) and the
# COMPOSED model (MjSpec.attach() prefixes attached elements with "/").
_HINGE_JOINT_NAMES = ("/pendulum_hinge", "pendulum_hinge")
_HUB_BODY_NAMES = ("/pendulum_hub", "pendulum_hub")


def arm_q_for_pendulum_xml(pendulum_xml: str | Path = DEFAULT_PENDULUM_XML) -> np.ndarray:
    """The validated arm pose for ``pendulum_xml``.

    Raises rather than guessing for an unknown asset: silently falling back to
    some other asset's pose is exactly the failure mode this table exists to
    prevent (a hinge that is near-vertical or near-parallel to world X at the
    borrowed pose still compiles, still has 7 DOF, still reports two
    equilibria, and simply never swings)."""
    key = str(Path(pendulum_xml).resolve())
    try:
        return PENDULUM_ASSET_ARM_Q[key].copy()
    except KeyError:
        raise KeyError(
            f"no validated arm pose registered for pendulum asset {pendulum_xml!r}. "
            f"Known assets: {sorted(PENDULUM_ASSET_ARM_Q)}. Add an entry to "
            f"PENDULUM_ASSET_ARM_Q only after MEASURING the hinge axis's "
            f"horizontality and |axis x xhat| at the intended pose."
        ) from None


def _resolve_id(model: mujoco.MjModel, objtype, names: tuple[str, ...]) -> int:
    for name in names:
        found = mujoco.mj_name2id(model, objtype, name)
        if found >= 0:
            return found
    raise ValueError(f"none of {names} found in the model")


def compose_ur5e_pendulum_spec(
    *,
    arm_xml: str | Path = DEFAULT_ARM_XML,
    pendulum_xml: str | Path = DEFAULT_PENDULUM_XML,
    attachment_site: str = ATTACHMENT_SITE,
) -> mujoco.MjSpec:
    """Returns an mujoco.MjSpec with the pendulum fragment attached to the
    arm spec at ``attachment_site``, before compilation -- callers that
    want to add a floor/lighting/other worldbody content (matching
    assets/ur5e_torque/scene.xml's pattern) should do so on this spec
    before calling .compile()."""
    arm_xml = Path(arm_xml)
    pendulum_xml = Path(pendulum_xml)
    if not arm_xml.exists():
        raise FileNotFoundError(f"UR5e arm MJCF not found: {arm_xml}")
    if not pendulum_xml.exists():
        raise FileNotFoundError(f"Pendulum attachment MJCF not found: {pendulum_xml}")

    arm_spec = mujoco.MjSpec.from_file(str(arm_xml))
    pendulum_spec = mujoco.MjSpec.from_file(str(pendulum_xml))

    site = arm_spec.site(attachment_site)
    if site is None:
        raise ValueError(
            f"{arm_xml} has no site named {attachment_site!r} -- cannot attach the pendulum. "
            "See assets/ur5e_torque/ur5e_torque.xml's wrist_3_link body for the expected site."
        )
    arm_spec.attach(pendulum_spec, site=site)
    return arm_spec


def compose_ur5e_pendulum_model(
    *,
    arm_xml: str | Path = DEFAULT_ARM_XML,
    pendulum_xml: str | Path = DEFAULT_PENDULUM_XML,
    attachment_site: str = ATTACHMENT_SITE,
) -> mujoco.MjModel:
    """Compiles compose_ur5e_pendulum_spec(...) into a ready-to-simulate
    mujoco.MjModel. Adds a ground plane/light so the model is directly
    simulatable without a separate scene file, matching
    assets/ur5e_torque/scene.xml's minimal worldbody content -- this is
    NOT a copy of that file, just the same two additions applied
    programmatically to keep this module self-contained."""
    spec = compose_ur5e_pendulum_spec(
        arm_xml=arm_xml, pendulum_xml=pendulum_xml, attachment_site=attachment_site
    )
    floor = spec.worldbody.add_geom()
    floor.name = "floor"
    floor.type = mujoco.mjtGeom.mjGEOM_PLANE
    floor.size = [0, 0, 0.05]
    light = spec.worldbody.add_light()
    light.pos = [0, 0, 1.5]
    light.dir = [0, 0, -1]
    light.type = mujoco.mjtLightType.mjLIGHT_DIRECTIONAL
    return spec.compile()


@dataclass(frozen=True)
class PendulumConstants:
    """The pendulum's physical constants for ONE (asset, arm pose) pair.

    Every field is measured off a compiled ``mujoco.MjModel`` by
    :func:`derive_pendulum_constants`; none is hand-cached. This type exists
    because these numbers are NOT properties of the asset alone -- ``mgr_nm``
    is the gravity torque about the hinge *in world*, so it changes with the
    arm pose (the hinge axis tilts toward vertical and the torque shrinks with
    it). Caching them per-asset, or worse per-module, is what made pointing a
    swing-up script at a different asset silently produce confidently wrong
    energy targets and natural periods.
    """

    mgr_nm: float
    """``M*g*r_effective`` -- the amplitude of the (exactly sinusoidal) gravity
    torque about the hinge. This, not any geometric COM distance, is the
    quantity the swing-up energy law needs."""
    i_pivot_kgm2: float
    """Moment of inertia about the hinge, read from the mass matrix."""
    m_total_kg: float
    """Mass of the SWINGING subtree (hub + rod), from ``body_subtreemass``."""
    r_com_m: float
    """``mgr_nm / (m_total_kg * g)`` -- the effective moment arm. Reported for
    continuity with the older hand-derived constants; only the product
    ``mgr_nm`` enters any control law."""
    g: float
    omega_natural_radps: float
    t_natural_s: float
    e_top_j: float
    """Energy at fully inverted, referenced to hanging = 0: ``2 * mgr_nm``."""
    arm_q: tuple[float, ...]
    """The arm pose these constants were measured at (``()`` for an isolated
    fragment with no arm)."""

    @classmethod
    def from_mass_and_arm(
        cls,
        *,
        m_total_kg: float,
        r_com_m: float,
        i_pivot_kgm2: float,
        g: float = GRAVITY_MPS2,
        arm_q: tuple[float, ...] = (),
    ) -> "PendulumConstants":
        """Build the same bundle from pre-measured scalars.

        Only used to reproduce a historical hand-cached constant set bit-for-bit
        (regression/equivalence testing). Production code should call
        :func:`derive_pendulum_constants`."""
        mgr = m_total_kg * g * r_com_m
        omega = float(np.sqrt(mgr / i_pivot_kgm2))
        return cls(
            mgr_nm=mgr,
            i_pivot_kgm2=i_pivot_kgm2,
            m_total_kg=m_total_kg,
            r_com_m=r_com_m,
            g=g,
            omega_natural_radps=omega,
            t_natural_s=2.0 * np.pi / omega,
            e_top_j=2.0 * mgr,
            arm_q=tuple(float(v) for v in arm_q),
        )


def derive_pendulum_constants(
    model: mujoco.MjModel,
    arm_q=None,
    *,
    g: float = GRAVITY_MPS2,
    n_samples: int = 721,
) -> PendulumConstants:
    """Measure the pendulum's physical constants directly off ``model`` at the
    arm pose ``arm_q`` -- correct by construction for ANY asset/pose pair.

    Two independent measurements, neither of which requires simulating,
    settling, or knowing the asset's geometry:

    1. ``M*g*r`` by least-squares sine fit to the hinge DOF's gravity torque
       ``qfrc_bias`` over a full revolution. MuJoCo's ``qfrc_bias`` about this
       joint is exactly sinusoidal in the hinge angle (the hub's COM sits ON
       the axis, so only the rod contributes), so the fitted amplitude IS
       ``M*g*r_effective``. Fitting it is assumption-free: it automatically
       accounts for both corrections that hand-derivation kept getting wrong --
       that only the COM offset PERPENDICULAR to the hinge makes torque, and
       that the torque additionally scales with ``sin(angle between the hinge
       axis and world vertical)``.
    2. ``I`` about the hinge from the mass matrix's own diagonal entry for that
       DOF (``mj_fullM``), which is invariant in the hinge angle.

    ``arm_q=None`` skips posing the arm, which is what an isolated pendulum
    fragment (``nq == 1``) needs.

    Does not mutate ``model``; all work happens on a scratch ``MjData``.
    """
    hinge_jid = _resolve_id(model, mujoco.mjtObj.mjOBJ_JOINT, _HINGE_JOINT_NAMES)
    hub_bid = _resolve_id(model, mujoco.mjtObj.mjOBJ_BODY, _HUB_BODY_NAMES)
    qadr = int(model.jnt_qposadr[hinge_jid])
    dadr = int(model.jnt_dofadr[hinge_jid])

    data = mujoco.MjData(model)
    arm_q_arr = None if arm_q is None else np.asarray(arm_q, dtype=np.float64).reshape(6)

    def _pose(theta: float) -> None:
        if arm_q_arr is not None:
            data.qpos[:6] = arm_q_arr
        data.qpos[qadr] = float(theta)
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)

    thetas = np.linspace(-np.pi, np.pi, n_samples)
    taus = np.empty(n_samples)
    for i, th in enumerate(thetas):
        _pose(th)
        taus[i] = float(data.qfrc_bias[dadr])

    basis = np.stack([np.sin(thetas), np.cos(thetas)], axis=1)
    coef, *_ = np.linalg.lstsq(basis, -taus, rcond=None)
    residual = float(np.max(np.abs(-taus - basis @ coef)))
    mgr = float(np.hypot(*coef))
    if mgr <= 0.0:
        raise ValueError(
            "fitted gravity-torque amplitude about the hinge is zero -- the hinge "
            "axis is parallel to gravity at this arm pose, so the apparatus is not "
            "a pendulum here. Re-measure the axis/pose pairing before using it."
        )
    if residual > 1e-6 * max(mgr, 1e-12) + 1e-12:
        raise ValueError(
            f"gravity torque about the hinge is not a pure sinusoid "
            f"(max residual {residual:.3e} Nm against amplitude {mgr:.6f} Nm) -- "
            "the 'fitted amplitude == M*g*r' reading is unsafe for this model, "
            "most likely because a swinging body's COM no longer sits on the axis."
        )

    # I about the hinge is angle-invariant; evaluate at theta=0 for determinism.
    _pose(0.0)
    full_m = np.zeros((model.nv, model.nv))
    mujoco.mj_fullM(model, full_m, data.qM)
    i_pivot = float(full_m[dadr, dadr])

    m_total = float(model.body_subtreemass[hub_bid])
    omega = float(np.sqrt(mgr / i_pivot))
    return PendulumConstants(
        mgr_nm=mgr,
        i_pivot_kgm2=i_pivot,
        m_total_kg=m_total,
        r_com_m=mgr / (m_total * g),
        g=g,
        omega_natural_radps=omega,
        t_natural_s=2.0 * np.pi / omega,
        e_top_j=2.0 * mgr,
        arm_q=() if arm_q_arr is None else tuple(float(v) for v in arm_q_arr),
    )
