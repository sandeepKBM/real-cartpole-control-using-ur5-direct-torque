#!/usr/bin/env python3
"""Check a trajectory log against the extracted lab workspace guardrails.

This is diagnostic-only. It never connects to hardware and it never commands
motion. It accepts logs that contain either:

- ``samples`` with ``tcp_pose`` or ``ee_pos``
- ``trace`` rows with ``ee_pos`` / ``tool_xyz``
- a flat list of samples with ``x`` / ``y`` / ``z``
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from _bootstrap import ensure_repo_root

ensure_repo_root()

import numpy as np

from workspace_guardrails import (
    DEFAULT_GUARDRAIL_CONFIG,
    GuardrailConfig,
    GuardrailDecision,
    boundary_summary,
    check_point,
    check_trajectory,
    load_guardrail_config,
    overlay_guardrails_on_frame,
)


def _load_json(path: Path) -> Any:
    return json.loads(Path(path).read_text())


def _extract_points(data: Any) -> tuple[np.ndarray, list[int | None], str]:
    if isinstance(data, dict):
        if "samples" in data and isinstance(data["samples"], list):
            rows = data["samples"]
            pts = []
            stamps: list[int | None] = []
            for row in rows:
                if not isinstance(row, dict):
                    continue
                point = None
                if "tcp_pose" in row and row["tcp_pose"] is not None:
                    pose = np.asarray(row["tcp_pose"], dtype=np.float64).reshape(-1)
                    if pose.shape[0] >= 3:
                        point = pose[:3]
                elif "ee_pos" in row and row["ee_pos"] is not None:
                    point = np.asarray(row["ee_pos"], dtype=np.float64).reshape(-1)[:3]
                elif all(k in row for k in ("x", "y", "z")):
                    point = np.array([row["x"], row["y"], row["z"]], dtype=np.float64)
                if point is not None:
                    pts.append(point)
                    stamps.append(int(row["host_stamp_ns"]) if "host_stamp_ns" in row and row["host_stamp_ns"] is not None else None)
            if pts:
                return np.asarray(pts, dtype=np.float64), stamps, "samples"
        if "trace" in data and isinstance(data["trace"], list):
            rows = data["trace"]
            pts = []
            stamps = []
            for row in rows:
                if not isinstance(row, dict):
                    continue
                point = None
                if "tcp_pose" in row and row["tcp_pose"] is not None:
                    pose = np.asarray(row["tcp_pose"], dtype=np.float64).reshape(-1)
                    if pose.shape[0] >= 3:
                        point = pose[:3]
                elif "ee_pos" in row and row["ee_pos"] is not None:
                    point = np.asarray(row["ee_pos"], dtype=np.float64).reshape(-1)[:3]
                elif "tool_xyz" in row and row["tool_xyz"] is not None:
                    point = np.asarray(row["tool_xyz"], dtype=np.float64).reshape(-1)[:3]
                if point is not None:
                    pts.append(point)
                    stamps.append(int(round(float(row["time"]) * 1e9)) if "time" in row and row["time"] is not None else None)
            if pts:
                return np.asarray(pts, dtype=np.float64), stamps, "trace"
        if all(k in data for k in ("x", "y", "z")):
            return np.asarray([[data["x"], data["y"], data["z"]]], dtype=np.float64), [None], "point"
    if isinstance(data, list):
        pts = []
        stamps: list[int | None] = []
        for row in data:
            if not isinstance(row, dict):
                continue
            if "tcp_pose" in row and row["tcp_pose"] is not None:
                pose = np.asarray(row["tcp_pose"], dtype=np.float64).reshape(-1)
                if pose.shape[0] >= 3:
                    pts.append(pose[:3])
                    stamps.append(int(row["host_stamp_ns"]) if "host_stamp_ns" in row and row["host_stamp_ns"] is not None else None)
            elif all(k in row for k in ("x", "y", "z")):
                pts.append(np.array([row["x"], row["y"], row["z"]], dtype=np.float64))
                stamps.append(None)
        if pts:
            return np.asarray(pts, dtype=np.float64), stamps, "list"
    raise ValueError("Could not extract any 3D points from the provided log")


def _sample_report(config: GuardrailConfig, points: np.ndarray, stamps: list[int | None], *, margin_m: float, frame: str) -> dict[str, Any]:
    per_sample: list[dict[str, Any]] = []
    first_near_index: int | None = None
    first_violation_index: int | None = None
    violating_samples = 0
    worst_signed_distance: float | None = None
    worst_boundary: str | None = None
    overall = "inside"
    for idx, point in enumerate(points):
        decision = check_point(point, config, frame=frame, margin_m=margin_m, timestamp_ns=stamps[idx])
        if decision.state == "near_boundary" and first_near_index is None:
            first_near_index = idx
        if decision.state == "outside":
            violating_samples += 1
            if first_violation_index is None:
                first_violation_index = idx
            overall = "outside"
        elif decision.state == "near_boundary" and overall != "outside":
            overall = "near_boundary"
        elif decision.state == "unknown" and overall == "inside":
            overall = "unknown"
        signed_candidates = [a.signed_distance_m for a in decision.assessments if a.signed_distance_m is not None]
        if signed_candidates:
            candidate = float(min(signed_candidates, key=lambda x: abs(float(x))))
            candidate_boundary = next(
                a.name
                for a in decision.assessments
                if a.signed_distance_m is not None and float(a.signed_distance_m) == candidate
            )
            if worst_signed_distance is None or abs(candidate) < abs(worst_signed_distance):
                worst_signed_distance = candidate
                worst_boundary = candidate_boundary
        per_sample.append(
            {
                "index": idx,
                "stamp_ns": stamps[idx],
                "point_m": np.asarray(point, dtype=np.float64).tolist(),
                "decision": decision.as_dict(),
            }
        )
    summary = {
        "state": overall,
        "sample_count": int(points.shape[0]),
        "first_near_index": first_near_index,
        "first_near_stamp_ns": None if first_near_index is None else stamps[first_near_index],
        "first_violation_index": first_violation_index,
        "first_violation_stamp_ns": None if first_violation_index is None else stamps[first_violation_index],
        "violating_samples": int(violating_samples),
        "worst_signed_distance_m": worst_signed_distance,
        "worst_boundary": worst_boundary,
    }
    return {"summary": summary, "samples": per_sample}


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Check a trajectory log against the extracted lab workspace guardrails.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--log", required=True, type=Path, help="JSON log containing real trajectory points.")
    p.add_argument("--desired-log", type=Path, default=None, help="Optional JSON log containing desired trajectory points.")
    p.add_argument(
        "--guardrail-config",
        type=Path,
        default=DEFAULT_GUARDRAIL_CONFIG,
        help="Guardrail YAML extracted from the external scene.",
    )
    p.add_argument("--frame", default=None, help="Input frame name (defaults to the guardrail config frame).")
    p.add_argument("--guardrail-margin-m", type=float, default=0.0, help="Additional conservative margin in meters.")
    p.add_argument("--output", type=Path, default=Path("logs/guardrail_report.json"), help="JSON report path.")
    p.add_argument("--csv", type=Path, default=None, help="Optional CSV export path.")
    p.add_argument("--render-overlay", type=Path, default=None, help="Optional PNG snapshot showing the overlay.")
    return p


def _write_csv(path: Path, report: dict[str, Any]) -> None:
    rows = report["real"]["samples"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["index", "stamp_ns", "x_m", "y_m", "z_m", "state", "boundary", "signed_distance_m", "distance_m"])
        for row in rows:
            decision = row["decision"]
            point = row["point_m"]
            writer.writerow(
                [
                    row["index"],
                    row["stamp_ns"],
                    point[0],
                    point[1],
                    point[2],
                    decision["state"],
                    decision.get("boundary_name"),
                    decision.get("signed_distance_m"),
                    decision.get("distance_m"),
                ]
            )


def _render_overlay(path: Path, config: GuardrailConfig, points: np.ndarray, decision: GuardrailDecision, frame: str, margin_m: float) -> None:
    canvas = np.zeros((720, 1280, 3), dtype=np.uint8)
    canvas[:] = 18
    overlay = overlay_guardrails_on_frame(
        canvas,
        config,
        trajectory_xyz=points,
        current_xyz=points[-1],
        desired_xyz=points[0],
        decision=decision,
        guardrail_margin_m=margin_m,
        show_labels=True,
    )
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(overlay).save(path)


def main() -> int:
    args = build_arg_parser().parse_args()
    config = load_guardrail_config(args.guardrail_config)
    real_data = _load_json(args.log)
    real_points, real_stamps, real_source = _extract_points(real_data)
    real_report = _sample_report(
        config,
        real_points,
        real_stamps,
        margin_m=float(args.guardrail_margin_m),
        frame=args.frame or config.frame,
    )
    real_decision = check_trajectory(real_points, config, frame=args.frame or config.frame, margin_m=float(args.guardrail_margin_m), timestamp_ns=real_stamps)

    report: dict[str, Any] = {
        "guardrail_config": {"path": str(args.guardrail_config), **boundary_summary(config)},
        "real": {
            "source": real_source,
            **real_report,
            "decision": real_decision.as_dict(),
        },
    }

    if args.desired_log is not None:
        desired_data = _load_json(args.desired_log)
        desired_points, desired_stamps, desired_source = _extract_points(desired_data)
        desired_report = _sample_report(
            config,
            desired_points,
            desired_stamps,
            margin_m=float(args.guardrail_margin_m),
            frame=args.frame or config.frame,
        )
        desired_decision = check_trajectory(
            desired_points,
            config,
            frame=args.frame or config.frame,
            margin_m=float(args.guardrail_margin_m),
            timestamp_ns=desired_stamps,
        )
        report["desired"] = {
            "source": desired_source,
            **desired_report,
            "decision": desired_decision.as_dict(),
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True))

    if args.csv is not None:
        _write_csv(args.csv, report)

    if args.render_overlay is not None:
        _render_overlay(
            args.render_overlay,
            config,
            real_points,
            real_decision,
            args.frame or config.frame,
            float(args.guardrail_margin_m),
        )

    print(json.dumps(report["real"]["summary"], indent=2, sort_keys=True))
    if "desired" in report:
        print(json.dumps(report["desired"]["summary"], indent=2, sort_keys=True))
    return 0 if real_report["summary"]["state"] != "outside" else 1


if __name__ == "__main__":
    raise SystemExit(main())
