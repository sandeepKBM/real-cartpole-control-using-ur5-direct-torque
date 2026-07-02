"""Shared helpers for the UR5e MuJoCo torque-control path.

This module intentionally stays simulator-agnostic where possible.
It defines the canonical UR5e joint order, conservative torque limits,
package URI rewriting helpers, and a few XML convenience functions used by
the conversion / patching scripts and the MuJoCo experiment runner.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from typing import Any, Iterable, Sequence

import numpy as np

from controller_core.x_axis_cartesian_impedance import JOINT_NAME_ORDER as UR5E_JOINT_ORDER

UR5E_TORQUE_LIMITS_NM = {
    "shoulder_pan_joint": 150.0,
    "shoulder_lift_joint": 150.0,
    "elbow_joint": 150.0,
    "wrist_1_joint": 28.0,
    "wrist_2_joint": 28.0,
    "wrist_3_joint": 28.0,
}

UR5E_TORQUE_ACTUATOR_SPECS = (
    ("shoulder_pan_torque", "shoulder_pan_joint", 150.0),
    ("shoulder_lift_torque", "shoulder_lift_joint", 150.0),
    ("elbow_torque", "elbow_joint", 150.0),
    ("wrist_1_torque", "wrist_1_joint", 28.0),
    ("wrist_2_torque", "wrist_2_joint", 28.0),
    ("wrist_3_torque", "wrist_3_joint", 28.0),
)

_PACKAGE_URI_RE = re.compile(r"package://([^/]+)/([^\"'<> ]+)")


def _xml_local_tag(elem: ET.Element) -> str:
    return str(elem.tag).rsplit("}", 1)[-1]


def torque_limit_vector() -> np.ndarray:
    return np.asarray([UR5E_TORQUE_LIMITS_NM[name] for name in UR5E_JOINT_ORDER], dtype=np.float64)


def compute_gravity_torque(
    model: Any,
    qpos_or_data: Any,
    joint_ids: Sequence[int],
    *,
    scratch_data: Any | None = None,
) -> np.ndarray:
    """Compute the gravity-compensation torque for the current configuration.

    The returned vector is the generalized torque needed to hold the current
    joint configuration with zero velocity and zero acceleration. The helper
    uses MuJoCo inverse dynamics on a scratch ``MjData`` instance so the live
    simulation state is never left mutated.

    ``qpos_or_data`` may be either a MuJoCo ``MjData``-like object with a
    ``qpos`` attribute or a flat joint-position vector.
    """
    import mujoco

    if hasattr(qpos_or_data, "qpos"):
        qpos = np.asarray(qpos_or_data.qpos, dtype=np.float64).reshape(-1)
    else:
        qpos = np.asarray(qpos_or_data, dtype=np.float64).reshape(-1)

    scratch = scratch_data if scratch_data is not None else mujoco.MjData(model)
    scratch.qpos[:] = 0.0
    scratch.qpos[: qpos.shape[0]] = qpos
    scratch.qvel[:] = 0.0
    if hasattr(scratch, "qacc"):
        scratch.qacc[:] = 0.0
    if hasattr(scratch, "qfrc_applied"):
        scratch.qfrc_applied[:] = 0.0
    if hasattr(scratch, "xfrc_applied"):
        scratch.xfrc_applied[:] = 0.0
    if hasattr(scratch, "ctrl"):
        scratch.ctrl[:] = 0.0
    mujoco.mj_forward(model, scratch)
    mujoco.mj_inverse(model, scratch)

    tau = np.zeros(len(joint_ids), dtype=np.float64)
    for idx, jid in enumerate(joint_ids):
        dof_adr = int(model.jnt_dofadr[int(jid)])
        # MuJoCo's bias force contains the passive load (including gravity) at
        # the current configuration. For this torque-motor lane, the hold
        # torque to add is the bias force itself.
        tau[idx] = float(scratch.qfrc_bias[dof_adr])
    return tau


def parse_package_root_specs(specs: Iterable[str]) -> dict[str, Path]:
    """Parse ``name=/path`` package root mappings."""
    out: dict[str, Path] = {}
    for raw in specs:
        if "=" not in raw:
            raise ValueError(f"package root must be name=/path, got {raw!r}")
        name, value = raw.split("=", 1)
        name = name.strip()
        value = value.strip()
        if not name:
            raise ValueError(f"package root name is empty in {raw!r}")
        path = Path(value).expanduser().resolve()
        out[name] = path
    return out


def source_text(source: str) -> tuple[str, str]:
    """Return ``(text, resolved_name)`` from a path or URL."""
    parsed = urllib.parse.urlparse(source)
    if parsed.scheme in {"http", "https"}:
        try:
            with urllib.request.urlopen(source, timeout=30.0) as resp:
                data = resp.read()
        except Exception as exc:  # pragma: no cover - network failure is environment dependent
            raise RuntimeError(
                f"Failed to download {source!r}. Download the URDF manually and retry."
            ) from exc
        return data.decode("utf-8"), source

    path = Path(source).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(path)
    return path.read_text(encoding="utf-8"), str(path)


def rewrite_package_uris(xml_text: str, package_roots: dict[str, Path]) -> tuple[str, list[str]]:
    """Rewrite ``package://pkg/...`` URIs to absolute filesystem paths.

    Returns the rewritten XML and a list of package names that were resolved.
    Raises if a URI references a package without a mapping.
    """
    resolved: list[str] = []

    def _replace(match: re.Match[str]) -> str:
        pkg = match.group(1)
        suffix = match.group(2)
        if pkg not in package_roots:
            raise ValueError(
                f"Unresolved package://{pkg}/... reference; provide --package-root {pkg}=/path/to/{pkg}"
            )
        resolved.append(pkg)
        return str(package_roots[pkg] / suffix)

    rewritten = _PACKAGE_URI_RE.sub(_replace, xml_text)
    return rewritten, sorted(set(resolved))


def xml_indent(root: ET.Element) -> None:
    """Pretty-print an XML tree in-place (Python 3.9+)."""
    ET.indent(root, space="  ")


def build_torque_actuator_block() -> ET.Element:
    """Return an actuator block with conservative torque motors for the UR5e."""
    actuator = ET.Element("actuator")
    comment = ET.Comment(
        "Conservative torque-control ranges for simulation experiments only; not a hardware safety model."
    )
    actuator.append(comment)
    for actuator_name, joint_name, limit_nm in UR5E_TORQUE_ACTUATOR_SPECS:
        motor = ET.SubElement(
            actuator,
            "motor",
            {
                "name": actuator_name,
                "joint": joint_name,
                "gear": "1",
                "ctrllimited": "true",
                "ctrlrange": f"{-limit_nm:g} {limit_nm:g}",
                "forcelimited": "true",
                "forcerange": f"{-limit_nm:g} {limit_nm:g}",
            },
        )
        motor.tail = "\n"
    return actuator


def _iter_xml_source_tree(path: Path, *, visited: set[Path] | None = None) -> list[Path]:
    path = Path(path).expanduser().resolve()
    if visited is None:
        visited = set()
    if path in visited:
        return []
    visited.add(path)
    if not path.exists():
        raise FileNotFoundError(path)
    out = [path]
    root = ET.parse(path).getroot()
    for include in root.findall(".//include"):
        file_attr = include.get("file")
        if not file_attr:
            raise ValueError(f"Malformed <include> without file attribute in {path}")
        include_path = (path.parent / file_attr).expanduser().resolve()
        out.extend(_iter_xml_source_tree(include_path, visited=visited))
    return out


def validate_ur5e_torque_xml_source_tree(scene_xml: str | Path) -> dict[str, Any]:
    """Validate the XML source tree before MuJoCo compilation.

    The torque scene is expected to contain only motor actuators on the six
    UR5e joints. The validator rejects:

    - ``<position>`` and ``<velocity>`` actuators on the canonical UR5e joints
    - legacy ``<general>`` actuators on the canonical UR5e joints
    - equality constraints in the torque scene tree
    - nonzero stiffness / spring-damper on the canonical UR5e joints
    """
    scene_xml = Path(scene_xml).expanduser().resolve()
    checked_files = _iter_xml_source_tree(scene_xml)
    actuator_kinds: dict[str, int] = {}
    for xml_path in checked_files:
        root = ET.parse(xml_path).getroot()
        if root.find(".//equality") is not None:
            raise ValueError(f"Equality constraints are not allowed in the UR5e torque scene: {xml_path}")
        for joint in root.findall(".//joint"):
            joint_name = joint.get("name")
            if joint_name not in UR5E_JOINT_ORDER:
                continue
            for attr_name in ("damping", "stiffness"):
                attr_val = joint.get(attr_name)
                if attr_val is not None and abs(float(attr_val)) > 1e-9:
                    raise ValueError(
                        f"UR5e joint {joint_name!r} in {xml_path} has nonzero {attr_name}={attr_val!r}; "
                        "this would add hidden passive hold behavior."
                    )
            springdamper = joint.get("springdamper")
            if springdamper is not None:
                vals = [float(x) for x in springdamper.split()]
                if any(abs(val) > 1e-9 for val in vals):
                    raise ValueError(
                        f"UR5e joint {joint_name!r} in {xml_path} has springdamper={springdamper!r}; "
                        "this would add hidden passive hold behavior."
                    )
        for actuator_root in root.findall(".//actuator"):
            for actuator in list(actuator_root):
                kind = _xml_local_tag(actuator)
                actuator_kinds[kind] = actuator_kinds.get(kind, 0) + 1
                joint_name = actuator.get("joint")
                if joint_name not in UR5E_JOINT_ORDER:
                    continue
                if kind in {"position", "velocity", "general"}:
                    raise ValueError(
                        f"Forbidden {kind!r} actuator on UR5e joint {joint_name!r} in {xml_path}. "
                        "The torque scene must use motor actuators only."
                    )
                if kind != "motor":
                    raise ValueError(
                        f"Unsupported actuator kind {kind!r} on UR5e joint {joint_name!r} in {xml_path}"
                    )
    return {
        "scene_xml": str(scene_xml),
        "checked_files": [str(path) for path in checked_files],
        "actuator_kinds": actuator_kinds,
        "source_tree_ok": True,
    }


def validate_ur5e_xml_joint_names(root: ET.Element) -> None:
    joints = [joint.get("name") for joint in root.findall(".//joint")]
    missing = [name for name in UR5E_JOINT_ORDER if name not in joints]
    if missing:
        raise ValueError(f"UR5e MJCF is missing joints: {missing}")


def validate_ur5e_torque_actuators(root: ET.Element) -> None:
    actuators = root.findall("./actuator/motor")
    if len(actuators) != 6:
        raise ValueError(f"UR5e torque MJCF must have 6 motor actuators; found {len(actuators)}")
    expected = list(UR5E_TORQUE_ACTUATOR_SPECS)
    for elem, (expected_name, expected_joint, expected_limit) in zip(actuators, expected, strict=True):
        name = elem.get("name")
        joint = elem.get("joint")
        gear = float(elem.get("gear", "0"))
        ctrlrange = elem.get("ctrlrange", "")
        if name != expected_name:
            raise ValueError(f"Actuator name mismatch: expected {expected_name!r}, got {name!r}")
        if joint != expected_joint:
            raise ValueError(f"Actuator joint mismatch: expected {expected_joint!r}, got {joint!r}")
        if abs(gear - 1.0) > 1e-12:
            raise ValueError(f"Actuator {name!r} must use gear=1; got {gear}")
        lo_hi = [float(x) for x in ctrlrange.split()] if ctrlrange else []
        if len(lo_hi) != 2:
            raise ValueError(f"Actuator {name!r} ctrlrange missing or malformed")
        if abs(lo_hi[0] + expected_limit) > 1e-6 or abs(lo_hi[1] - expected_limit) > 1e-6:
            raise ValueError(
                f"Actuator {name!r} ctrlrange must be symmetric ±{expected_limit:g}; got {ctrlrange!r}"
            )


def get_compiled_ur5e_torque_model_diagnostics(
    model: Any,
    *,
    site_name: str = "attachment_site",
) -> dict[str, Any]:
    """Return a structured diagnostic snapshot of the compiled MuJoCo model."""
    import mujoco

    diag: dict[str, Any] = {
        "nu": int(model.nu),
        "neq": int(getattr(model, "neq", 0)),
        "gravity": np.asarray(model.opt.gravity, dtype=np.float64).tolist(),
        "actuator_biastype": np.asarray(model.actuator_biastype[: model.nu], dtype=np.int32).tolist(),
        "actuator_gaintype": np.asarray(model.actuator_gaintype[: model.nu], dtype=np.int32).tolist(),
        "actuator_dyntype": np.asarray(model.actuator_dyntype[: model.nu], dtype=np.int32).tolist(),
        "actuator_trntype": np.asarray(model.actuator_trntype[: model.nu], dtype=np.int32).tolist(),
        "actuator_ctrllimited": np.asarray(getattr(model, "actuator_ctrllimited", np.ones(model.nu, dtype=bool))[: model.nu]).astype(bool).tolist(),
        "actuator_forcelimited": np.asarray(getattr(model, "actuator_forcelimited", np.ones(model.nu, dtype=bool))[: model.nu]).astype(bool).tolist(),
        "actuator_ctrlrange": np.asarray(model.actuator_ctrlrange[: model.nu], dtype=np.float64).tolist(),
        "actuator_forcerange": np.asarray(model.actuator_forcerange[: model.nu], dtype=np.float64).tolist(),
        "jnt_type": np.asarray(model.jnt_type[: model.njnt], dtype=np.int32).tolist(),
        "jnt_range": np.asarray(model.jnt_range[: model.njnt], dtype=np.float64).tolist(),
        "jnt_stiffness": np.asarray(model.jnt_stiffness[: model.njnt], dtype=np.float64).tolist(),
        "dof_damping": np.asarray(model.dof_damping[: model.nv], dtype=np.float64).tolist(),
        "dof_armature": np.asarray(model.dof_armature[: model.nv], dtype=np.float64).tolist(),
        "site_name": site_name,
        "site_id": int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)),
    }
    return diag


def validate_compiled_ur5e_torque_model(
    model: Any,
    *,
    site_name: str = "attachment_site",
) -> tuple[list[int], list[int], int]:
    """Validate a compiled MuJoCo model and return canonical joint/actuator ids.

    This helper is intended for the *compiled* MuJoCo model, where the scene may
    have been assembled through ``<include>``. It checks that all six UR5e
    joints exist, that the torque actuators map one-to-one onto those joints, and
    that the attachment site is present.
    """
    joint_ids: list[int] = []
    actuator_ids: list[int] = []
    if int(model.nu) != 6:
        raise ValueError(f"UR5e torque model must expose 6 actuators; found {int(model.nu)}")
    import mujoco
    diag = get_compiled_ur5e_torque_model_diagnostics(model, site_name=site_name)
    gravity = np.asarray(model.opt.gravity, dtype=np.float64).reshape(3)
    if int(getattr(model, "neq", 0)) != 0:
        raise ValueError(f"UR5e torque model must not contain equality constraints; found neq={int(model.neq)}")
    if not np.all(np.isfinite(gravity)) or np.linalg.norm(gravity) < 1e-3 or gravity[2] >= -1e-3:
        raise ValueError(f"UR5e torque model must run with gravity enabled; got gravity={gravity.tolist()!r}")

    for actuator_name, joint_name, _ in UR5E_TORQUE_ACTUATOR_SPECS:
        jid = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name))
        aid = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name))
        if jid < 0 or aid < 0:
            raise ValueError(f"Missing torque mapping for {actuator_name!r} -> {joint_name!r}")
        joint_ids.append(jid)
        actuator_ids.append(aid)

    for aid, (expected_name, expected_joint, expected_limit) in zip(actuator_ids, UR5E_TORQUE_ACTUATOR_SPECS, strict=True):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, aid)
        joint = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, int(model.actuator_trnid[aid, 0]))
        gear = float(model.actuator_gear[aid, 0])
        gaintype = int(model.actuator_gaintype[aid])
        biastype = int(model.actuator_biastype[aid])
        dyntype = int(model.actuator_dyntype[aid])
        trntype = int(model.actuator_trntype[aid])
        if name != expected_name:
            raise ValueError(f"Actuator name mismatch: expected {expected_name!r}, got {name!r}")
        if joint != expected_joint:
            raise ValueError(f"Actuator joint mismatch: expected {expected_joint!r}, got {joint!r}")
        if abs(gear - 1.0) > 1e-12:
            raise ValueError(f"Actuator {name!r} must use gear=1; got {gear}")
        if gaintype != 0 or biastype != 0 or dyntype != 0 or trntype != 0:
            raise ValueError(
                f"Actuator {name!r} must be a pure torque motor (gaintype/biastype/dyntype/trntype all zero); "
                f"got {(gaintype, biastype, dyntype, trntype)!r}"
            )
        if hasattr(model, "actuator_ctrllimited") and not bool(model.actuator_ctrllimited[aid]):
            raise ValueError(f"Actuator {name!r} must be ctrl-limited")
        if hasattr(model, "actuator_forcelimited") and not bool(model.actuator_forcelimited[aid]):
            raise ValueError(f"Actuator {name!r} must be force-limited")
        ctrl_lo, ctrl_hi = map(float, model.actuator_ctrlrange[aid])
        force_lo, force_hi = map(float, model.actuator_forcerange[aid])
        if abs(ctrl_lo + expected_limit) > 1e-6 or abs(ctrl_hi - expected_limit) > 1e-6:
            raise ValueError(
                f"Actuator {name!r} ctrlrange must be symmetric ±{expected_limit:g}; got {(ctrl_lo, ctrl_hi)!r}"
            )
        if abs(force_lo + expected_limit) > 1e-6 or abs(force_hi - expected_limit) > 1e-6:
            raise ValueError(
                f"Actuator {name!r} forcerange must be symmetric ±{expected_limit:g}; got {(force_lo, force_hi)!r}"
            )
    joint_types = np.asarray(model.jnt_type[joint_ids], dtype=np.int32)
    if not np.all(joint_types == int(mujoco.mjtJoint.mjJNT_HINGE)):
        raise ValueError(f"All UR5e joints must be hinge joints; got types={joint_types.tolist()!r}")
    joint_ranges = np.asarray(model.jnt_range[joint_ids], dtype=np.float64)
    if not np.all(np.isfinite(joint_ranges)):
        raise ValueError(f"UR5e joint ranges must be finite; got {joint_ranges.tolist()!r}")
    if np.any(joint_ranges[:, 1] <= joint_ranges[:, 0] + 1e-9):
        raise ValueError(f"UR5e joint ranges must be non-degenerate; got {joint_ranges.tolist()!r}")
    dof_damping = np.asarray(model.dof_damping[:6], dtype=np.float64)
    if np.max(np.abs(dof_damping)) > 0.5:
        raise ValueError(f"UR5e dof damping is suspiciously high: {dof_damping.tolist()!r}")
    jnt_stiffness = np.asarray(model.jnt_stiffness[joint_ids], dtype=np.float64)
    if np.max(np.abs(jnt_stiffness)) > 1e-6:
        raise ValueError(f"UR5e joint stiffness is suspiciously high: {jnt_stiffness.tolist()!r}")

    site_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name))
    if site_id < 0:
        raise ValueError(f"Missing required site: {site_name!r}")
    return joint_ids, actuator_ids, site_id


def write_xml_tree(root: ET.Element, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    xml_indent(root)
    tree = ET.ElementTree(root)
    tree.write(path, encoding="utf-8", xml_declaration=False)
    return path
