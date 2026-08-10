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

from pathlib import Path

import mujoco

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARM_XML = REPO_ROOT / "assets" / "ur5e_torque" / "ur5e_torque.xml"
DEFAULT_PENDULUM_XML = REPO_ROOT / "assets" / "ur5e_pendulum" / "pendulum_attachment.xml"
ATTACHMENT_SITE = "attachment_site"


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
