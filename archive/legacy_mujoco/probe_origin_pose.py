#!/usr/bin/env python3
"""
Report a canonical UR5e origin pose using a distal forearm reference site.

The probe searches over the 6 actuated UR5e joints and reports:

- a canonical joint vector
- the world pose of `forearm_tip_site`, which is the origin reference point
- the world pose/orientation of `attachment_site`, which is the tool frame
- each joint's world-space anchor and axis
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import mujoco
import numpy as np

from controller import (
    ACTIVE_ORIGIN_Q,
    TARGET_SITE_ROTATION_WORLD,
    TARGET_SITE_X_AXIS_WORLD,
    TARGET_SITE_Y_AXIS_WORLD,
    TARGET_SITE_Z_AXIS_WORLD,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_XML = REPO_ROOT / "mujoco_menagerie" / "universal_robots_ur5e" / "ur5e.xml"
ORIGIN_REFERENCE_SITE_NAME = "forearm_tip_site"
ATTACHMENT_SITE_NAME = "attachment_site"
ELBOW_JOINT_INDEX = 2
DEFAULT_ELBOW_STRAIGHTNESS_WEIGHT = 0.25
DEFAULT_ELBOW_TARGET_DEG = 90.0
HOME_Q = np.array(
    [-math.pi / 2, -math.pi / 2, math.pi / 2, -math.pi / 2, -math.pi / 2, 0.0],
    dtype=np.float64,
)


def matrix_columns_to_axes(mat: np.ndarray) -> dict[str, list[float]]:
    return {
        "x_in_world": mat[:, 0].tolist(),
        "y_in_world": mat[:, 1].tolist(),
        "z_in_world": mat[:, 2].tolist(),
    }


def canonical_bounds(model: mujoco.MjModel, actuated: int) -> tuple[np.ndarray, np.ndarray]:
    lower = []
    upper = []
    for joint_id in range(actuated):
        lo, hi = map(float, model.jnt_range[joint_id])
        if hi - lo > 2 * math.pi + 1e-6:
            lo, hi = -math.pi, math.pi
        lower.append(lo)
        upper.append(hi)
    return np.array(lower, dtype=np.float64), np.array(upper, dtype=np.float64)


def set_arm_qpos(model: mujoco.MjModel, data: mujoco.MjData, q: np.ndarray) -> None:
    data.qpos[:] = 0.0
    data.qvel[:] = 0.0
    data.qpos[: model.nu] = q
    mujoco.mj_forward(model, data)


def vertical_face_alignment_score(rot: np.ndarray) -> tuple[float, float, float, float]:
    tool_x = rot[:, 0]
    tool_y = rot[:, 1]
    tool_z = rot[:, 2]
    x_align = float(np.dot(tool_x, TARGET_SITE_X_AXIS_WORLD))
    y_align = float(np.dot(tool_y, TARGET_SITE_Y_AXIS_WORLD))
    z_align = float(np.dot(tool_z, TARGET_SITE_Z_AXIS_WORLD))
    score = 10.0 * (x_align + y_align + z_align)
    return score, x_align, y_align, z_align


def elbow_bend_abs_rad(q: np.ndarray) -> float:
    return float(abs(np.asarray(q, dtype=np.float64)[ELBOW_JOINT_INDEX]))


def find_peak_height_origin(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    reference_site_id: int,
    samples: int,
    seed: int,
) -> np.ndarray:
    actuated = model.nu
    lower, upper = canonical_bounds(model, actuated)

    def eval_q(q: np.ndarray) -> float:
        set_arm_qpos(model, data, q)
        return float(data.site_xpos[reference_site_id][2])

    seeds = [
        np.zeros(actuated, dtype=np.float64),
        HOME_Q.copy(),
        np.array([0.0, -math.pi / 2, math.pi / 2, -math.pi / 2, -math.pi / 2, 0.0], dtype=np.float64),
    ]
    rng = np.random.default_rng(seed)
    seeds.extend(rng.uniform(lower, upper) for _ in range(samples))

    best_q = None
    best_z = -np.inf
    for q in seeds:
        z = eval_q(q)
        if z > best_z:
            best_q = np.array(q, dtype=np.float64)
            best_z = z

    assert best_q is not None
    step = np.full(actuated, 0.5, dtype=np.float64)
    for _ in range(120):
        improved = False
        for joint_id in range(actuated):
            for sign in (1.0, -1.0):
                cand = best_q.copy()
                cand[joint_id] = np.clip(cand[joint_id] + sign * step[joint_id], lower[joint_id], upper[joint_id])
                z = eval_q(cand)
                if z > best_z + 1e-10:
                    best_q = cand
                    best_z = z
                    improved = True
        if not improved:
            step *= 0.5
            if float(np.max(step)) < 1e-5:
                break

    for joint_id in (0, 5):
        cand = best_q.copy()
        cand[joint_id] = 0.0
        z = eval_q(cand)
        if abs(z - best_z) < 1e-7:
            best_q = cand
            best_z = z

    return best_q


def find_vertical_face_origin(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    reference_site_id: int,
    attachment_site_id: int,
    samples: int,
    seed: int,
    elbow_straightness_weight: float = 0.0,
    fixed_elbow_rad: float | None = None,
) -> np.ndarray:
    actuated = model.nu
    lower, upper = canonical_bounds(model, actuated)

    def eval_q(q: np.ndarray) -> float:
        set_arm_qpos(model, data, q)
        reference_pos = data.site_xpos[reference_site_id]
        tool_rot = data.site_xmat[attachment_site_id].reshape(3, 3)
        orient_score, _, _, _ = vertical_face_alignment_score(tool_rot)
        # Keep the locked tool orientation close to perfect, but among
        # already-aligned poses let origin height dominate the choice.
        return (
            float(reference_pos[2])
            - 100.0 * (30.0 - orient_score)
            - elbow_straightness_weight * elbow_bend_abs_rad(q)
        )

    seeds = [
        ACTIVE_ORIGIN_Q.copy(),
        np.array([0.0, -math.pi / 2, 0.0, -math.pi, -math.pi / 2, 0.0], dtype=np.float64),
        np.array([0.0, -math.pi / 2, math.pi / 2, -math.pi / 2, -math.pi / 2, 0.0], dtype=np.float64),
        HOME_Q.copy(),
        find_peak_height_origin(
            model,
            data,
            reference_site_id,
            samples=min(samples, 256),
            seed=seed,
        ),
    ]
    rng = np.random.default_rng(seed)
    for _ in range(samples):
        q = rng.uniform(lower, upper)
        q[0] = 0.0
        seeds.append(q)

    if fixed_elbow_rad is not None:
        for q in seeds:
            q[ELBOW_JOINT_INDEX] = fixed_elbow_rad

    best_q = None
    best_score = -np.inf
    for q in seeds:
        q = np.array(q, dtype=np.float64)
        q[0] = 0.0
        if fixed_elbow_rad is not None:
            q[ELBOW_JOINT_INDEX] = fixed_elbow_rad
        score = eval_q(q)
        if score > best_score:
            best_q = q
            best_score = score

    assert best_q is not None
    step = np.array([0.0, 0.5, 0.5, 0.5, 0.5, 0.5], dtype=np.float64)
    if fixed_elbow_rad is not None:
        step[ELBOW_JOINT_INDEX] = 0.0
    for _ in range(160):
        improved = False
        for joint_id in range(actuated):
            if step[joint_id] == 0.0:
                continue
            for sign in (1.0, -1.0):
                cand = best_q.copy()
                cand[joint_id] = np.clip(
                    cand[joint_id] + sign * step[joint_id],
                    lower[joint_id],
                    upper[joint_id],
                )
                cand[0] = 0.0
                if fixed_elbow_rad is not None:
                    cand[ELBOW_JOINT_INDEX] = fixed_elbow_rad
                score = eval_q(cand)
                if score > best_score + 1e-10:
                    best_q = cand
                    best_score = score
                    improved = True
        if not improved:
            step *= 0.5
            if float(np.max(step)) < 1e-6:
                break

    cand = best_q.copy()
    cand[5] = 0.0
    score = eval_q(cand)
    if abs(score - best_score) < 1e-7:
        best_q = cand

    return best_q


def build_report(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    reference_site_id: int,
    attachment_site_id: int,
    q_peak: np.ndarray,
    mode: str,
    elbow_straightness_weight: float = 0.0,
    elbow_target_deg: float | None = None,
) -> dict:
    set_arm_qpos(model, data, q_peak)

    base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base")
    base_mat = data.xmat[base_id].reshape(3, 3).copy()
    tool_mat = data.site_xmat[attachment_site_id].reshape(3, 3).copy()
    origin_reference_pos = data.site_xpos[reference_site_id].copy()
    attachment_pos = data.site_xpos[attachment_site_id].copy()
    align_score, x_align, y_align, z_align = vertical_face_alignment_score(tool_mat)

    set_arm_qpos(model, data, HOME_Q)
    home_reference_pos = data.site_xpos[reference_site_id].copy()
    home_attachment_pos = data.site_xpos[attachment_site_id].copy()

    set_arm_qpos(model, data, q_peak)
    joints = []
    for joint_id in range(model.nu):
        joints.append(
            {
                "name": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id),
                "q_rad": float(q_peak[joint_id]),
                "q_deg": float(np.degrees(q_peak[joint_id])),
                "anchor_world": data.xanchor[joint_id].tolist(),
                "axis_world": data.xaxis[joint_id].tolist(),
                "range_rad": model.jnt_range[joint_id].tolist(),
            }
        )

    origin_definition = (
        "Joint vector that maximizes the world-z position of forearm_tip_site, "
        "with shoulder_pan and wrist_3 canonicalized to 0 when height is unchanged."
        if mode == "peak-height"
        else (
            "Joint vector that keeps attachment_site aligned to the blue shoulder/upper-arm side-face direction "
            "world-frame target (tool x along world -X, tool y along world -Z, tool z along world -Y) "
            + (
                f"while holding elbow_joint at {elbow_target_deg:.3f} deg and maximizing "
                "forearm_tip_site height, with shoulder_pan fixed at 0."
                if mode == "vertical-face-elbow-target"
                else (
                f"while maximizing forearm_tip_site height with an elbow-bend penalty "
                f"of {elbow_straightness_weight:.3f} m/rad, with shoulder_pan fixed at 0."
                if mode == "vertical-face-straight"
                else "while maximizing forearm_tip_site height secondarily, with shoulder_pan fixed at 0."
                )
            )
        )
    )

    return {
        "model_xml": str(MODEL_XML),
        "origin_mode": mode,
        "origin_definition": origin_definition,
        "origin_reference_site": ORIGIN_REFERENCE_SITE_NAME,
        "tool_site": ATTACHMENT_SITE_NAME,
        "world_axes": {
            "x": [1.0, 0.0, 0.0],
            "y": [0.0, 1.0, 0.0],
            "z": [0.0, 0.0, 1.0],
        },
        "target_axes_in_world": matrix_columns_to_axes(TARGET_SITE_ROTATION_WORLD),
        "base_axes_in_world": matrix_columns_to_axes(base_mat),
        "tool_axes_in_world": matrix_columns_to_axes(tool_mat),
        "origin_reference_world": origin_reference_pos.tolist(),
        "attachment_site_world": attachment_pos.tolist(),
        "peak_q_rad": q_peak.tolist(),
        "peak_q_deg": np.degrees(q_peak).tolist(),
        "home_origin_reference_world": home_reference_pos.tolist(),
        "home_attachment_site_world": home_attachment_pos.tolist(),
        "reference_height_gain_vs_home_m": float(origin_reference_pos[2] - home_reference_pos[2]),
        "vertical_face_alignment_score": float(align_score),
        "elbow_abs_rad": elbow_bend_abs_rad(q_peak),
        "elbow_abs_deg": float(np.degrees(elbow_bend_abs_rad(q_peak))),
        "elbow_straightness_weight": float(elbow_straightness_weight),
        "elbow_target_deg": None if elbow_target_deg is None else float(elbow_target_deg),
        "vertical_face_alignment_components": {
            "tool_x_vs_target_x": x_align,
            "tool_y_vs_target_y": y_align,
            "tool_z_vs_target_z": z_align,
        },
        "joints": joints,
    }


def print_report(report: dict) -> None:
    print(f"{report['origin_mode']} origin for UR5e forearm_tip_site")
    print(f"model_xml: {report['model_xml']}")
    print(
        "origin_reference_world: "
        + np.array2string(np.array(report["origin_reference_world"]), precision=6, separator=", ")
    )
    print(
        "attachment_site_world: "
        + np.array2string(np.array(report["attachment_site_world"]), precision=6, separator=", ")
    )
    print(f"peak_q_rad: {np.array2string(np.array(report['peak_q_rad']), precision=6, separator=', ')}")
    print(f"peak_q_deg: {np.array2string(np.array(report['peak_q_deg']), precision=3, separator=', ')}")
    print(
        "home_origin_reference_world: "
        + np.array2string(np.array(report["home_origin_reference_world"]), precision=6, separator=", ")
    )
    print(
        "home_attachment_site_world: "
        + np.array2string(np.array(report["home_attachment_site_world"]), precision=6, separator=", ")
    )
    print(f"reference_height_gain_vs_home_m: {report['reference_height_gain_vs_home_m']:.6f}")
    print(f"elbow_abs_rad: {report['elbow_abs_rad']:.6f}")
    print(f"elbow_abs_deg: {report['elbow_abs_deg']:.3f}")
    print(f"elbow_straightness_weight: {report['elbow_straightness_weight']:.6f}")
    if report["elbow_target_deg"] is not None:
        print(f"elbow_target_deg: {report['elbow_target_deg']:.3f}")
    print(f"vertical_face_alignment_score: {report['vertical_face_alignment_score']:.6f}")

    print("\nWorld axes:")
    for axis_name, axis_vec in report["world_axes"].items():
        print(f"  {axis_name}: {axis_vec}")

    print("\nTarget axes in world:")
    for axis_name, axis_vec in report["target_axes_in_world"].items():
        print(f"  {axis_name}: {np.array2string(np.array(axis_vec), precision=6, separator=', ')}")

    print("\nBase axes in world:")
    for axis_name, axis_vec in report["base_axes_in_world"].items():
        print(f"  {axis_name}: {np.array2string(np.array(axis_vec), precision=6, separator=', ')}")

    print("\nTool axes in world:")
    for axis_name, axis_vec in report["tool_axes_in_world"].items():
        print(f"  {axis_name}: {np.array2string(np.array(axis_vec), precision=6, separator=', ')}")

    print("\nVertical-face alignment components:")
    for axis_name, axis_vec in report["vertical_face_alignment_components"].items():
        print(f"  {axis_name}: {axis_vec:.6f}")

    print("\nJoint orientations:")
    for joint in report["joints"]:
        print(f"  {joint['name']}")
        print(f"    q_rad: {joint['q_rad']:.6f}")
        print(f"    q_deg: {joint['q_deg']:.3f}")
        print(
            "    anchor_world: "
            + np.array2string(np.array(joint["anchor_world"]), precision=6, separator=', ')
        )
        print(
            "    axis_world: "
            + np.array2string(np.array(joint["axis_world"]), precision=6, separator=', ')
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=5000, help="Random samples used before local refinement.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducible search.")
    parser.add_argument(
        "--mode",
        choices=("vertical-face", "vertical-face-straight", "vertical-face-elbow-target", "peak-height"),
        default="vertical-face",
        help="Origin definition to probe.",
    )
    parser.add_argument(
        "--elbow-straightness-weight",
        type=float,
        default=DEFAULT_ELBOW_STRAIGHTNESS_WEIGHT,
        help="Penalty weight in m/rad applied to abs(elbow_joint) in vertical-face-straight mode.",
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        default=None,
        help="Optional path for a JSON dump of the report.",
    )
    parser.add_argument(
        "--elbow-target-deg",
        type=float,
        default=DEFAULT_ELBOW_TARGET_DEG,
        help="Elbow angle in degrees for vertical-face-elbow-target mode.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = mujoco.MjModel.from_xml_path(str(MODEL_XML))
    data = mujoco.MjData(model)
    reference_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, ORIGIN_REFERENCE_SITE_NAME)
    attachment_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, ATTACHMENT_SITE_NAME)

    elbow_straightness_weight = 0.0
    elbow_target_deg = None
    if args.mode == "peak-height":
        q_peak = find_peak_height_origin(
            model,
            data,
            reference_site_id,
            samples=args.samples,
            seed=args.seed,
        )
    else:
        if args.mode == "vertical-face-straight":
            elbow_straightness_weight = float(args.elbow_straightness_weight)
        if args.mode == "vertical-face-elbow-target":
            elbow_target_deg = float(args.elbow_target_deg)
        q_peak = find_vertical_face_origin(
            model,
            data,
            reference_site_id,
            attachment_site_id,
            samples=args.samples,
            seed=args.seed,
            elbow_straightness_weight=elbow_straightness_weight,
            fixed_elbow_rad=None if elbow_target_deg is None else math.radians(elbow_target_deg),
        )
    report = build_report(
        model,
        data,
        reference_site_id,
        attachment_site_id,
        q_peak,
        mode=args.mode,
        elbow_straightness_weight=elbow_straightness_weight,
        elbow_target_deg=elbow_target_deg,
    )
    print_report(report)

    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(report, indent=2))
        print(f"\nSaved JSON report to {args.json_output}")


if __name__ == "__main__":
    main()
