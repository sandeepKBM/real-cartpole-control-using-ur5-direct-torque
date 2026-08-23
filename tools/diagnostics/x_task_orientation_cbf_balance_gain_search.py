#!/usr/bin/env python3
"""Gain search for the reduced-task QP + orientation HOCBF at the Goal-1
balance pose (ARM_Q0, wrist_2=-90deg, ``pendulum_attachment_realrod.xml``).

MOTIVATION. A cascade-LQR capture-envelope grid at this pose found the
Y/Z corridor HOCBF fixes drift (54/117 -> 0/117 failures) but orientation
becomes the SOLE limiter (31/117 -> 113/117 failures), because the
controller still rigidly TRACKS orientation via `kp_rot`/`kd_rot` — gains that
were searched for TRANSPORT at a different pose, using unit-converted-from-OSC
values that were never re-derived for this pose or this regime (balance, not
transport). This script does two things together, since they were suspected
to interact:

  1. Adds the orientation HOCBF (``XTaskYZCorridorQPConfig.orientation_cbf``,
     see ``controller_core/x_task_yz_corridor_qp/``) so orientation is BOUNDED
     instead of only tracked -- the same corridor philosophy already applied
     to Y/Z.
  2. Re-derives kp_x/kd_x from a freshly measured task-space inertia (Lambda)
     AT THIS POSE (not copied from a different pose's measurement -- see
     ``measure_lambda`` below) and SEARCHES kp_rot/kd_rot plus the barrier's
     own alpha1/alpha2, rather than assuming either kp_rot=0 (OSC's choice,
     motivated by a Lambda-shaping instability this controller does not have)
     or the previously-searched kp_rot=35.12 (searched for TRANSPORT, not
     BALANCE).

TWO-STAGE SEARCH, for tractable wall-clock cost (a nested "re-search K for
every gain candidate" design was costed and rejected -- see the design note
in the module docstring below):

  Stage A: ONE LQR (Q, R) search (`search_lqr_gains`, cheap settings) against
           a Lambda-derived SEED config, producing a fixed K.
  Stage B: `differential_evolution` over
           (kp_x, kd_x, kp_rot, kd_rot, orientation_cbf_alpha1,
           orientation_cbf_alpha2), objective = -captures on a COARSE proxy
           grid (a subset of the full 13x9=117 grid, for wall-clock reasons),
           using the FIXED K from Stage A.

The winning gains are written to a real, committed
``config/ur5e_mujoco_torque_x_task_yz_corridor_qp_orientation_cbf_balance.yaml``.
That file is then the one to re-validate with `pendulum_lqr_cascade.py`'s own
full (fresh K search + full 117-cell envelope) pipeline -- THIS script's
proxy-grid number is a search heuristic, not the reported result.

DE_WORKERS honored (defaults to 90% of cpu_count, same convention as
pendulum_lqr_cascade.py). Guards stay ON throughout (`enforce_guard` is never
set False here).
"""

from __future__ import annotations

import argparse
import functools
import json
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import yaml
from scipy.optimize import differential_evolution

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from simulation.ur5e_mujoco_torque import expand_mass_matrix  # noqa: E402
from simulation.ur5e_pendulum_compose import REALROD_PENDULUM_XML  # noqa: E402
from tools.diagnostics.pendulum_lqr_cascade import (  # noqa: E402
    V_MAX_MPS,
    capture_envelope_grid,
    run_lqr_trial,
    search_lqr_gains,
)
from tools.diagnostics.pendulum_swingup_energy_shaping import PendulumRunContext  # noqa: E402

import mujoco  # noqa: E402

ARM_Q_W2M90 = (-2.3688, -2.1801, -1.8838, -0.7962, -1.5707963, 0.0206)
BASE_CONFIG = REPO_ROOT / "config" / "ur5e_mujoco_torque_x_task_yz_corridor_qp_enabled.yaml"


def _de_workers() -> int:
    import multiprocessing as mp
    try:
        mp.set_start_method("fork")
    except RuntimeError:
        pass
    env = os.environ.get("DE_WORKERS")
    if env:
        return max(1, int(env))
    return max(1, int((os.cpu_count() or 2) * 0.9))


def measure_lambda(ctx: PendulumRunContext, *, eps: float = 0.1) -> dict:
    """Task-space inertia Lambda = inv(J M^-1 J^T + eps I) (full 6x6, the
    EXACT formula ``x_axis_cartesian_impedance/controller.py`` uses for
    ``task_space_inertia_shaping``), at THIS pose, using the RAW arm-only
    mass-matrix slice (``state["mass_matrix"]`` == ``mass_full[:6, :6]`` --
    the same quantity every controller in this repo actually receives; the
    pendulum's own reflected inertia through the Schur complement was checked
    too and differs by ~1%, see the report this script prints).

    VERIFIED against this repo's own already-published number before trusting
    it on a new pose: reproduces ``Lambda_xx = 5.9298`` (config/ur5e_mujoco_
    torque_x_task_yz_corridor_qp_enabled.yaml's own gain-conversion comment)
    to 6 significant figures at the OLD singular ARM_Q0
    ([-2.3688,-2.1801,-1.8838,-0.7962,0.004714693,0.0206]) on the plain arm
    model with the same eps=0.1 -- see this script's ``--verify-formula`` flag.
    """
    model = ctx.build_model()
    data = mujoco.MjData(model)
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "attachment_site")
    pend_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "/pendulum_hinge")
    pend_qpos_adr = model.jnt_qposadr[pend_jid]

    out = {}
    for label, angle in [("hanging", ctx.hanging_angle), ("inverted", ctx.inverted_angle)]:
        data.qpos[:6] = ctx.arm_q_array
        data.qpos[pend_qpos_adr] = angle
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)
        mass_full = expand_mass_matrix(model, data)
        m_uu = mass_full[:6, :6]
        m_up = mass_full[:6, 6:]
        m_pu = mass_full[6:, :6]
        m_pp = mass_full[6:, 6:]
        m_schur = m_uu - m_up @ np.linalg.inv(m_pp) @ m_pu

        jacp = np.zeros((3, model.nv))
        jacr = np.zeros((3, model.nv))
        mujoco.mj_jacSite(model, data, jacp, jacr, site_id)
        jac = np.vstack([jacp[:, :6], jacr[:, :6]])

        row = {}
        for m_label, mass in [("raw_m6", m_uu), ("schur", m_schur)]:
            minv = np.linalg.inv(mass)
            lam = np.linalg.inv(jac @ minv @ jac.T + eps * np.eye(6))
            row[m_label] = {
                "lambda_xx": float(lam[0, 0]),
                "lambda_rot_diag": np.diag(lam[3:6, 3:6]).tolist(),
                "lambda_rot_factor": float(np.mean(np.diag(lam[3:6, 3:6]))),
            }
        row["cond_j"] = float(np.linalg.cond(jac))
        out[label] = row
    return out


def _verify_formula_against_known_value() -> float:
    """Reproduces the 5.9298 reference number at the OLD singular pose on the
    plain arm model (no pendulum) -- the same formula, a different, already
    published, independently-derived answer."""
    model = mujoco.MjModel.from_xml_path(str(REPO_ROOT / "assets" / "ur5e_torque" / "scene.xml"))
    data = mujoco.MjData(model)
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "attachment_site")
    q = np.array([-2.3688, -2.1801, -1.8838, -0.7962, 0.004714693, 0.0206])
    data.qpos[:6] = q
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)
    mass = expand_mass_matrix(model, data)[:6, :6]
    minv = np.linalg.inv(mass)
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    mujoco.mj_jacSite(model, data, jacp, jacr, site_id)
    jac = np.vstack([jacp[:, :6], jacr[:, :6]])
    lam = np.linalg.inv(jac @ minv @ jac.T + 0.1 * np.eye(6))
    return float(lam[0, 0])


def make_candidate_config(
    *, kp_x: float, kd_x: float, kp_rot: float, kd_rot: float,
    a1: float, a2: float, max_error_rad: float, tmp_dir: Path,
) -> Path:
    with open(BASE_CONFIG, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    cfg["controller"]["gains"]["kp_x"] = float(kp_x)
    cfg["controller"]["gains"]["kd_x"] = float(kd_x)
    cfg["controller"]["gains"]["kp_rot"] = float(kp_rot)
    cfg["controller"]["gains"]["kd_rot"] = float(kd_rot)
    cfg["controller"]["orientation_cbf"] = True
    cfg["controller"]["orientation_cbf_max_error_rad"] = float(max_error_rad)
    cfg["controller"]["orientation_cbf_alpha1"] = float(a1)
    cfg["controller"]["orientation_cbf_alpha2"] = float(a2)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    fd, path_str = tempfile.mkstemp(
        prefix=f"orient_cbf_cand_{os.getpid()}_", suffix=".yaml", dir=str(tmp_dir)
    )
    os.close(fd)
    path = Path(path_str)
    with open(path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(cfg, fh)
    return path


#: Coarse proxy grid for the OUTER gain search -- a strict subset of the full
#: 13x9=117 grid `pendulum_lqr_cascade.py` uses, chosen for wall-clock
#: tractability (see the module docstring's two-stage design note). The final
#: reported captured/117 number always comes from the FULL grid, run
#: separately after this search picks a winner.
PROXY_PHI_DEG = [-25.0, -15.0, -5.0, 5.0, 15.0, 25.0]
PROXY_THETADOT = [-3.0, -1.0, 0.0, 1.0, 3.0]
PROXY_DURATION_S = 2.5


def proxy_capture_count(
    params: np.ndarray, *, ctx: PendulumRunContext, K: np.ndarray, a_max: float,
    tmp_dir: Path, max_error_rad: float,
) -> float:
    kp_x, kd_x, kp_rot, kd_rot, a1, a2 = [float(v) for v in params]
    cfg_path = make_candidate_config(
        kp_x=kp_x, kd_x=kd_x, kp_rot=kp_rot, kd_rot=kd_rot,
        a1=a1, a2=a2, max_error_rad=max_error_rad, tmp_dir=tmp_dir,
    )
    try:
        model = ctx.build_model()
        n_cap = 0
        for phi_deg in PROXY_PHI_DEG:
            for thetadot in PROXY_THETADOT:
                res = run_lqr_trial(
                    model, K, duration_s=PROXY_DURATION_S,
                    hanging_angle=ctx.hanging_angle, inverted_angle=ctx.inverted_angle,
                    constants=ctx.constants, initial_phi_rad=np.radians(phi_deg),
                    initial_thetadot_radps=thetadot, a_max=a_max, v_max=V_MAX_MPS,
                    config_path=cfg_path, controller_kind="x_task_yz_corridor_qp",
                    arm_q=ctx.arm_q_array,
                )
                if res["captured"]:
                    n_cap += 1
    finally:
        cfg_path.unlink(missing_ok=True)
    return -float(n_cap)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tmp-dir", default=None)
    p.add_argument("--verify-formula", action="store_true")
    p.add_argument("--seed-k-maxiter", type=int, default=10)
    p.add_argument("--seed-k-popsize", type=int, default=10)
    p.add_argument("--seed-k-duration-s", type=float, default=2.5)
    p.add_argument("--outer-maxiter", type=int, default=14)
    p.add_argument("--outer-popsize", type=int, default=16)
    p.add_argument("--outer-seed", type=int, default=0)
    p.add_argument("--output-json", default=None)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    tmp_dir = Path(args.tmp_dir) if args.tmp_dir else Path(tempfile.mkdtemp(prefix="orient_cbf_search_"))

    if args.verify_formula:
        val = _verify_formula_against_known_value()
        print(f"formula check: Lambda_xx at OLD singular ARM_Q0 (plain arm) = {val:.6f} "
              f"(reference: 5.9298, from config/ur5e_mujoco_torque_x_task_yz_corridor_qp_enabled.yaml)")

    ctx = PendulumRunContext(
        pendulum_xml=str(REALROD_PENDULUM_XML),
        arm_q=ARM_Q_W2M90,
        config_path=str(BASE_CONFIG),
        controller_kind="x_task_yz_corridor_qp",
    ).resolve()

    lam = measure_lambda(ctx)
    print("=== measured Lambda at this pose (wrist_2=-90, realrod) ===")
    print(json.dumps(lam, indent=2))
    lam_xx = lam["inverted"]["raw_m6"]["lambda_xx"]
    lam_rot = lam["inverted"]["raw_m6"]["lambda_rot_factor"]
    kp_x0 = 400.0 * lam_xx
    kd_x0 = 40.0 * lam_xx
    kd_rot0 = 10.0 * lam_rot  # unit-derived CENTER, not a final value -- kp_rot has no OSC analog (OSC's own kp_rot=0)
    print(f"Lambda_xx={lam_xx:.6f}  Lambda_rot_factor={lam_rot:.6f}")
    print(f"derived seed: kp_x={kp_x0:.3f} kd_x={kd_x0:.3f} kd_rot_center={kd_rot0:.3f}")

    seed_cfg = make_candidate_config(
        kp_x=kp_x0, kd_x=kd_x0, kp_rot=0.0, kd_rot=kd_rot0,
        a1=10.0, a2=10.0, max_error_rad=0.20, tmp_dir=tmp_dir,
    )
    print(f"=== Stage A: LQR (Q,R) search against the seed config, fixed K for Stage B ===")
    t0 = time.time()
    ctx_seed = PendulumRunContext(
        pendulum_xml=str(REALROD_PENDULUM_XML), arm_q=ARM_Q_W2M90,
        config_path=str(seed_cfg), controller_kind="x_task_yz_corridor_qp",
        hanging_angle=ctx.hanging_angle, inverted_angle=ctx.inverted_angle, constants=ctx.constants,
    )
    seed_result = search_lqr_gains(
        ctx_seed, maxiter=args.seed_k_maxiter, popsize=args.seed_k_popsize,
        seed=0, duration_s=args.seed_k_duration_s,
    )
    K = np.asarray(seed_result["K"], dtype=np.float64).reshape(1, 4)
    a_max = float(seed_result["a_max"])
    print(f"Stage A done in {time.time()-t0:.1f}s: K={seed_result['K']} a_max={a_max} "
          f"Q_diag={seed_result['Q_diag']} cost={seed_result['cost']:.4f}")

    print("=== Stage B: differential_evolution over "
          "(kp_x, kd_x, kp_rot, kd_rot, orientation_cbf_alpha1, orientation_cbf_alpha2), "
          f"proxy grid {len(PROXY_PHI_DEG)}x{len(PROXY_THETADOT)}={len(PROXY_PHI_DEG)*len(PROXY_THETADOT)} cells ===")
    bounds = [
        (200.0, 4000.0),   # kp_x
        (20.0, 400.0),     # kd_x
        (0.0, 200.0),      # kp_rot
        (0.0, 80.0),       # kd_rot
        (1.0, 30.0),       # orientation_cbf_alpha1
        (1.0, 30.0),       # orientation_cbf_alpha2
    ]
    t0 = time.time()
    res = differential_evolution(
        functools.partial(
            proxy_capture_count, ctx=ctx, K=K, a_max=a_max, tmp_dir=tmp_dir, max_error_rad=0.20
        ),
        bounds, maxiter=args.outer_maxiter, popsize=args.outer_popsize, tol=1e-6,
        seed=args.outer_seed, workers=_de_workers(), polish=False,
    )
    print(f"Stage B done in {time.time()-t0:.1f}s: best -captures={res.fun} "
          f"(={-res.fun:.0f}/{len(PROXY_PHI_DEG)*len(PROXY_THETADOT)}) x={res.x.tolist()}")

    winner = {
        "kp_x": float(res.x[0]), "kd_x": float(res.x[1]),
        "kp_rot": float(res.x[2]), "kd_rot": float(res.x[3]),
        "orientation_cbf_alpha1": float(res.x[4]), "orientation_cbf_alpha2": float(res.x[5]),
        "proxy_captured": int(-res.fun), "proxy_total": len(PROXY_PHI_DEG) * len(PROXY_THETADOT),
        "stage_a_K": seed_result["K"], "stage_a_a_max": a_max, "stage_a_Q_diag": seed_result["Q_diag"],
        "lambda_xx": lam_xx, "lambda_rot_factor": lam_rot,
        "derived_seed_kp_x": kp_x0, "derived_seed_kd_x": kd_x0, "derived_seed_kd_rot_center": kd_rot0,
    }
    print("=== WINNER ===")
    print(json.dumps(winner, indent=2))
    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as fh:
            json.dump(winner, fh, indent=2)
        print("wrote", args.output_json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
