#!/usr/bin/env python3
"""Check cond(J) and gravity-torque headroom for a FORWARD-EXTENDED control
point at the SAME transport origin joint configuration (q unchanged) --
i.e. the user's proposal of "same origin pose, just the end effector is
pronounced forward" (a rigid tool/bracket extension along the flange's own
forward axis), NOT a re-solved IK pose.

Two independent checks:
1. Kinematic: does a pure rigid-body offset of the control point change
   cond(J)? Computed exactly via the standard velocity-composition identity
   v_point = v_site + omega x r (no model reload needed per distance --
   J_lin_point = jacp + skew(r) @ jacr, J_ang unchanged), so this is exact,
   not a numeric-perturbation approximation.
2. Dynamic: does adding a plausible bracket/tool mass at the extended point
   change gravity-compensation torque headroom? Uses MjSpec to add a real
   point mass body at the offset and re-derives qfrc_bias (qvel=0), unlike
   check 1 this DOES depend on added mass, not just geometry.
"""

from __future__ import annotations

import sys
from pathlib import Path

import mujoco
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from hardware.poses import (  # noqa: E402
    HEIGHT_ALPHA_0_5_Q,
    HEIGHT_ALPHA_0_5_CLEARANCE_Q,
    HANGING_ORIGIN_Q,
    HANGING_LOWER_Q,
    HANGING_ALPHA_0_5_Q,
    HANGING_ALPHA_0_5_CLEARANCE_Q,
)

SCENE_XML = REPO_ROOT / "assets" / "ur5e_torque" / "scene.xml"
SITE_NAME = "attachment_site"
TORQUE_LIMIT_NM = np.array([150.0, 150.0, 150.0, 28.0, 28.0, 28.0])
OFFSETS_M = [0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40]
BRACKET_MASSES_KG = [0.0, 0.5, 1.0, 2.0]


def skew(v: np.ndarray) -> np.ndarray:
    return np.array([
        [0.0, -v[2], v[1]],
        [v[2], 0.0, -v[0]],
        [-v[1], v[0], 0.0],
    ])


def load_state(q: np.ndarray):
    model = mujoco.MjModel.from_xml_path(str(SCENE_XML))
    data = mujoco.MjData(model)
    data.qpos[:6] = q
    mujoco.mj_forward(model, data)
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, SITE_NAME)
    return model, data, site_id


def kinematic_sweep(label: str, q: np.ndarray) -> None:
    model, data, site_id = load_state(q)

    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    mujoco.mj_jacSite(model, data, jacp, jacr, site_id)
    jacp6 = jacp[:, :6]
    jacr6 = jacr[:, :6]

    site_mat = data.site_xmat[site_id].reshape(3, 3)
    site_pos = data.site_xpos[site_id].copy()
    base_pos = data.xpos[0].copy()  # world body (base)

    # Pick the local axis whose world direction points most AWAY from the
    # base -- "extend the face forward, away from the base joint" -- by
    # projecting each candidate local axis (+-X, +-Y, +-Z) onto the
    # site-to-base-outward horizontal direction and taking the best match.
    outward_world = site_pos - base_pos
    outward_world[2] = 0.0  # horizontal only -- "forward" is a floor-plane notion
    outward_world /= (np.linalg.norm(outward_world) + 1e-12)

    best_axis, best_score = None, -np.inf
    for i in range(3):
        for sign in (+1.0, -1.0):
            cand = sign * site_mat[:, i]
            score = float(np.dot(cand, outward_world))
            if score > best_score:
                best_score, best_axis = score, cand
    fwd = best_axis

    print(f"\n=== {label}  (q={np.round(q, 4).tolist()}) ===")
    print(f"attachment_site world pos={np.round(site_pos, 4)}  "
          f"chosen forward axis (world)={np.round(fwd, 3)}  "
          f"alignment-with-outward-horizontal={best_score:.3f}")

    j_base_full = np.vstack([jacp6, jacr6])
    sv_base = np.linalg.svd(j_base_full, compute_uv=False)
    base_reach = float(np.linalg.norm(site_pos - base_pos))
    print(f"  d=0.00m (flange/attachment_site itself): "
          f"cond(J)={sv_base[0] / sv_base[-1]:.4e}  sigma_min={sv_base[-1]:.6e}  "
          f"reach_from_base={base_reach:.3f}m")

    # UR5e datasheet max reach = 0.850m from the shoulder-pan axis.
    UR5E_MAX_REACH_M = 0.850
    for d in OFFSETS_M:
        if d == 0.0:
            continue
        r_world = fwd * d
        j_lin = jacp6 + skew(r_world) @ jacr6
        j_full = np.vstack([j_lin, jacr6])
        sv = np.linalg.svd(j_full, compute_uv=False)
        cond = sv[0] / sv[-1]
        point_pos = site_pos + r_world
        reach = float(np.linalg.norm(point_pos - base_pos))
        reach_frac = 100.0 * reach / UR5E_MAX_REACH_M
        flag = "  <-- EXCEEDS 850mm datasheet reach" if reach > UR5E_MAX_REACH_M else ""
        print(f"  d={d:.2f}m: cond(J)={cond:.4e}  sigma_min={sv[-1]:.6e}  sigma_max={sv[0]:.4e}  "
              f"reach_from_base={reach:.3f}m ({reach_frac:.0f}% of 850mm){flag}")


def torque_headroom_sweep(label: str, q: np.ndarray) -> None:
    print(f"\n--- gravity-torque headroom, {label} ---")
    for d in OFFSETS_M:
        for m in BRACKET_MASSES_KG:
            if d == 0.0 and m > 0.0:
                continue  # a mass literally at the flange isn't the scenario being tested
            spec = mujoco.MjSpec.from_file(str(SCENE_XML))
            if m > 0.0:
                wrist3 = spec.body("wrist_3_link")
                extra = wrist3.add_body(name="fwd_extension_test_mass", pos=[0, 0.1, 0])
                extra.add_geom(type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[0.01, 0, 0],
                                pos=[0, d, 0], mass=m, contype=0, conaffinity=0, group=4)
            model = spec.compile()
            data = mujoco.MjData(model)
            data.qpos[:6] = q
            mujoco.mj_forward(model, data)
            tau_g = np.asarray(data.qfrc_bias[:6], dtype=np.float64)
            headroom_frac = np.abs(tau_g) / TORQUE_LIMIT_NM
            worst_joint = int(np.argmax(headroom_frac))
            print(f"  d={d:.2f}m mass={m:.1f}kg: |tau_g|={np.round(tau_g, 2)}  "
                  f"worst_joint={worst_joint} used={100*headroom_frac[worst_joint]:.1f}%")


def main() -> int:
    for label, q in [
        ("HEIGHT_ALPHA_0_5_Q (unrotated, canonical)", HEIGHT_ALPHA_0_5_Q),
        ("HEIGHT_ALPHA_0_5_CLEARANCE_Q (real-hw default, -45deg base)", HEIGHT_ALPHA_0_5_CLEARANCE_Q),
        ("HANGING_ORIGIN_Q (hanging family, tall end)", HANGING_ORIGIN_Q),
        ("HANGING_LOWER_Q (hanging family, low end)", HANGING_LOWER_Q),
        ("HANGING_ALPHA_0_5_Q (hanging family, midpoint)", HANGING_ALPHA_0_5_Q),
        ("HANGING_ALPHA_0_5_CLEARANCE_Q (hanging family, -45deg base, sim-only)", HANGING_ALPHA_0_5_CLEARANCE_Q),
    ]:
        kinematic_sweep(label, q)
        torque_headroom_sweep(label, q)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
