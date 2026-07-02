#!/usr/bin/env python3
"""Convert a UR5e URDF into a compiled MuJoCo MJCF XML.

This is a simulation-only utility. It does not touch hardware and it does not
assume CoppeliaSim semantics.

Typical usage:

    python3 scripts/convert_ur5e_urdf_to_mjcf.py \
        --input /path/to/ur5e.urdf \
        --package-root ur_description=/path/to/ur_description \
        --output outputs/ur5e_mujoco_torque/generated/ur5e_compiled.xml

If the input is an ``http(s)://`` URL, the script will try to download it
first. If the network is unavailable, it fails cleanly and tells you to
download the URDF manually.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import mujoco

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from mujoco_ur5e_tools import (  # noqa: E402
    parse_package_root_specs,
    rewrite_package_uris,
    source_text,
    validate_ur5e_xml_joint_names,
    write_xml_tree,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True, help="URDF path or URL.")
    p.add_argument(
        "--package-root",
        action="append",
        default=[],
        help="Package URI mapping in the form package_name=/abs/path (repeatable).",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Compiled MJCF output path. Default: outputs/ur5e_mujoco_torque/generated/<stem>_compiled.xml",
    )
    return p.parse_args()


def _default_output(input_label: str) -> Path:
    stem = Path(input_label).stem if not input_label.startswith(("http://", "https://")) else "ur5e"
    return REPO_ROOT / "outputs" / "ur5e_mujoco_torque" / "generated" / f"{stem}_compiled.xml"


def main() -> int:
    args = parse_args()
    out_path = Path(args.output) if args.output else _default_output(args.input)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    package_roots = parse_package_root_specs(args.package_root)

    try:
        raw_text, resolved_source = source_text(args.input)
        rewritten_text, resolved_packages = rewrite_package_uris(raw_text, package_roots)
    except Exception as exc:
        print(f"[urdf->mjcf] {exc}", file=sys.stderr)
        return 2

    rewritten_path = out_path.with_suffix(".resolved.urdf")
    rewritten_path.write_text(rewritten_text, encoding="utf-8")

    try:
        model = mujoco.MjModel.from_xml_path(str(rewritten_path))
    except Exception as exc:
        print(
            "[urdf->mjcf] MuJoCo failed to compile the rewritten URDF. "
            "Check the mesh paths and URDF structure.",
            file=sys.stderr,
        )
        print(f"[urdf->mjcf] source={resolved_source}", file=sys.stderr)
        print(f"[urdf->mjcf] rewritten={rewritten_path}", file=sys.stderr)
        print(f"[urdf->mjcf] error={exc}", file=sys.stderr)
        return 3

    try:
        mujoco.mj_saveLastXML(str(out_path), model)
    except Exception as exc:
        print(f"[urdf->mjcf] Failed to save compiled MJCF: {exc}", file=sys.stderr)
        return 4

    summary = {
        "source": resolved_source,
        "rewritten_urdf": str(rewritten_path),
        "compiled_mjcf": str(out_path),
        "resolved_packages": resolved_packages,
        "nq": int(model.nq),
        "nv": int(model.nv),
        "nu": int(model.nu),
        "njnt": int(model.njnt),
    }
    summary_path = out_path.with_suffix(".json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
