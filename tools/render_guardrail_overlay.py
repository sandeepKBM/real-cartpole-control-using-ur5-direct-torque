#!/usr/bin/env python3
"""Render a guardrail overlay snapshot from a trajectory log.

This is a diagnostic convenience wrapper around the shared guardrail overlay
helper. It never connects to hardware.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from _bootstrap import ensure_repo_root

ensure_repo_root()

import numpy as np
from PIL import Image

from check_trajectory_guardrails import _extract_points, _load_json
from workspace_guardrails import DEFAULT_GUARDRAIL_CONFIG, check_trajectory, load_guardrail_config, overlay_guardrails_on_frame


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Render a PNG snapshot with the extracted workspace guardrails overlaid.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--log", required=True, type=Path, help="JSON trajectory log.")
    p.add_argument(
        "--guardrail-config",
        type=Path,
        default=DEFAULT_GUARDRAIL_CONFIG,
        help="Guardrail YAML extracted from the external scene.",
    )
    p.add_argument("--frame", default=None, help="Input frame name (defaults to the config frame).")
    p.add_argument("--guardrail-margin-m", type=float, default=0.0, help="Additional conservative margin in meters.")
    p.add_argument("--output", required=True, type=Path, help="PNG output path.")
    return p


def main() -> int:
    args = build_arg_parser().parse_args()
    config = load_guardrail_config(args.guardrail_config)
    data = _load_json(args.log)
    points, stamps, _ = _extract_points(data)
    decision = check_trajectory(points, config, frame=args.frame or config.frame, margin_m=float(args.guardrail_margin_m), timestamp_ns=stamps)
    canvas = np.zeros((720, 1280, 3), dtype=np.uint8)
    canvas[:] = 18
    overlay = overlay_guardrails_on_frame(
        canvas,
        config,
        trajectory_xyz=points,
        current_xyz=points[-1],
        desired_xyz=points[0],
        decision=decision,
        guardrail_margin_m=float(args.guardrail_margin_m),
        show_labels=True,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(overlay).save(args.output)
    print(json.dumps(decision.as_dict(), indent=2, sort_keys=True))
    print(f"Saved overlay: {args.output}")
    return 0 if decision.state != "outside" else 1


if __name__ == "__main__":
    raise SystemExit(main())
