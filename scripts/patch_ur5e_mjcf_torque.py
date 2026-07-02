#!/usr/bin/env python3
"""Patch a UR5e MJCF so the arm is driven by torque motors.

This utility keeps the UR5e kinematic tree, inertials, meshes, and collision
geometries intact. It only replaces the actuator block with six ``<motor>``
elements using conservative torque limits.

The resulting XML is suitable for MuJoCo torque-control experiments and does
not use position actuators.
"""

from __future__ import annotations

import argparse
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from mujoco_ur5e_tools import (  # noqa: E402
    UR5E_JOINT_ORDER,
    build_torque_actuator_block,
    parse_package_root_specs,
    source_text,
    rewrite_package_uris,
    validate_ur5e_xml_joint_names,
    validate_ur5e_torque_actuators,
    write_xml_tree,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--input",
        required=True,
        help="Base UR5e MJCF path or URL. Usually the menagerie ur5e.xml.",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "assets" / "ur5e_torque" / "ur5e_torque.xml",
        help="Torque-actuated MJCF output path.",
    )
    p.add_argument(
        "--rename-model",
        default="ur5e_torque",
        help="Model name written into the torque variant XML.",
    )
    p.add_argument(
        "--package-root",
        action="append",
        default=[],
        help="Optional package URI mapping in the form package_name=/abs/path (repeatable).",
    )
    return p.parse_args()


def patch_tree(root: ET.Element, *, model_name: str) -> ET.Element:
    validate_ur5e_xml_joint_names(root)
    actuator = root.find("./actuator")
    if actuator is None:
        raise ValueError("Base MJCF has no <actuator> block to patch")

    # Replace the original actuator block wholesale. We keep the rest of the
    # robot tree, inertials, visuals, collisions, and keyframes intact.
    parent = root
    parent.remove(actuator)
    parent.append(build_torque_actuator_block())
    root.set("model", model_name)
    validate_ur5e_torque_actuators(root)
    return root


def main() -> int:
    args = parse_args()
    package_roots = parse_package_root_specs(args.package_root)

    try:
        raw_text, resolved_source = source_text(args.input)
    except Exception as exc:
        print(f"[mjcf-torque] {exc}", file=sys.stderr)
        return 2

    if "package://" in raw_text:
        if not package_roots:
            print(
                "[mjcf-torque] package:// URIs found in the base MJCF but no --package-root mapping was provided",
                file=sys.stderr,
            )
            return 2
        try:
            raw_text, resolved_packages = rewrite_package_uris(raw_text, package_roots)
        except Exception as exc:
            print(f"[mjcf-torque] {exc}", file=sys.stderr)
            return 2

    try:
        root = ET.fromstring(raw_text)
    except Exception as exc:
        print(f"[mjcf-torque] Failed to parse base MJCF XML: {exc}", file=sys.stderr)
        return 3

    try:
        patched = patch_tree(root, model_name=str(args.rename_model))
    except Exception as exc:
        print(f"[mjcf-torque] Failed to patch MJCF: {exc}", file=sys.stderr)
        return 4

    out_path = Path(args.output)
    write_xml_tree(patched, out_path)
    summary = {
        "source": resolved_source,
        "output": str(out_path),
        "model": str(args.rename_model),
        "joint_order": list(UR5E_JOINT_ORDER),
        "actuators": [
            {
                "name": a.get("name"),
                "joint": a.get("joint"),
                "ctrlrange": a.get("ctrlrange"),
            }
            for a in patched.findall("./actuator/motor")
        ],
    }
    summary_path = out_path.with_suffix(".json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
