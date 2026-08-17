"""One entrypoint, one vocabulary, for every pendulum experiment.

WHY THIS EXISTS. The lane had grown to six entrypoints with 34 distinct flags
and the same concept spelled three ways (``--lqr-json`` vs ``--lqr-gains-json``,
``--seed`` vs ``--seed-a``/``--seed-t``, ``--duration-s`` vs
``--balance-duration-s``), plus THREE separate swing-up -> LQR handoff
implementations totalling ~1100 lines. Picking the wrong one is easy and the
failure is silent.

WHAT IT DOES NOT DO: it does not reimplement anything. It dispatches to the
existing, validated backends by subprocess, so the physics has exactly one home
per stage and this file cannot drift away from it.

TWO BACKENDS, AND NEITHER SUBSUMES THE OTHER -- which is why the consolidation
is a dispatcher rather than a deletion:

  toolY   pendulum_toolY_{swingup_search,lqr,handoff}.py
          Gets the DRIVE AXIS right: rotates the state fed to an unmodified
          controller so the task axis is tool Y, a genuine 3D diagonal that the
          axis-index controller cannot express. BUT it HARDCODES pose and asset
          (ARM_Q0 + the default pendulum) -- there is no --pendulum-xml or
          --start-q-rad -- so it cannot express any other pose.
  generic pendulum_{swingup_energy_shaping,lqr_cascade,flip_catch_hold}.py
          Parameterized on pose AND asset, which is why Goal 1 at wrist_2=-90
          with the realrod asset had to use it. Drive axis is whatever the
          config's frame makes row 0.

So: toolY is correct-but-narrow, generic is general-but-easy-to-aim-wrong. This
dispatcher picks the backend that can actually express the request and REFUSES
combinations it cannot, instead of silently running a different pose than asked
for -- the single most dangerous failure available here, since every log line
still looks right.

THE AXIS CHECK IS MANDATORY (AGENTS.md 7). Every run stage prints the pumping
alignment of the chosen drive axis first, and refuses to dispatch if that axis
is dead. Two numbers, because the obvious one is not sufficient:

    kappa      fraction of the drive NOT wasted along the hinge
    kappa_hang authority at the HANGING start, where a swing-up must bootstrap

With a horizontal hinge the whole perpendicular vertical plane scores
kappa = 1.0, so kappa alone rates a vertical axis as ideal -- measured at
ARM_Q0, tool X and tool Y both score kappa = 1.0000 while kappa_hang is 0.127
vs 0.9919. A run aimed at tool X tripped the corridor guard in 0.134 s with the
rod tip 4 mm off the floor. Both numbers are checked here.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from simulation.ur5e_pendulum_compose import (  # noqa: E402
    DEFAULT_ARM_Q,
    DEFAULT_PENDULUM_XML,
    REALROD_PENDULUM_XML,
    arm_q_for_pendulum_xml,
    compose_ur5e_pendulum_model,
)

PY = sys.executable
DIAG = REPO_ROOT / "tools" / "diagnostics"

#: Named poses, so an experiment is described by a name rather than six floats
#: copy-pasted between shell commands (a real source of silent drift: the
#: registry still binds the realrod asset to an OLD pose, so --pose must be
#: explicit rather than inferred from the asset).
POSES = {
    "arm_q0": np.asarray(DEFAULT_ARM_Q, dtype=np.float64),
    "w2neg90": np.array([-2.3688, -2.1801, -1.8838, -0.7962, -1.5707963, 0.0206]),
}
ASSETS = {
    "default": Path(DEFAULT_PENDULUM_XML),   # local-Z hinge; LIVE at wrist_2 ~ 0
    "realrod": Path(REALROD_PENDULUM_XML),   # local-X hinge; LIVE at wrist_2 = -90
}


def resolve_pose(name_or_floats) -> np.ndarray:
    if isinstance(name_or_floats, (list, tuple)) and len(name_or_floats) == 6:
        return np.asarray(name_or_floats, dtype=np.float64)
    key = str(name_or_floats).lower()
    if key not in POSES:
        raise SystemExit(f"unknown --pose {name_or_floats!r}; known: {sorted(POSES)}")
    return POSES[key].copy()


def axis_alignment(pendulum_xml: Path, arm_q: np.ndarray, drive: str):
    """(kappa, kappa_hang, drive_vector_world, hinge_world) for the named axis."""
    import mujoco

    from tools.diagnostics.render_pose_task_axes import axis_report

    model = compose_ur5e_pendulum_model(pendulum_xml=str(pendulum_xml))
    data = mujoco.MjData(model)
    data.qpos[:6] = arm_q
    mujoco.mj_forward(model, data)
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "attachment_site")
    rep = axis_report(model, data, sid)
    lookup = {n.replace(" ", "-").lower(): (v, k, kh) for n, v, k, kh in rep["candidates"]}
    if drive not in lookup:
        raise SystemExit(f"unknown --drive-axis {drive!r}; known: {sorted(lookup)}")
    vec, kap, kh = lookup[drive]
    return kap, kh, vec / np.linalg.norm(vec), rep["hinge"]


def check_axis_or_die(pendulum_xml: Path, arm_q: np.ndarray, drive: str, force: bool) -> None:
    kap, kh, vec, hinge = axis_alignment(pendulum_xml, arm_q, drive)
    print(f"  drive axis   {drive}  = {np.round(vec, 4)}")
    print(f"  hinge axis            = {np.round(hinge, 4)}")
    print(f"  kappa      {kap:.4f}   (not wasted along the hinge)")
    print(f"  kappa_hang {kh:.4f}   (authority at the HANGING start)")
    if kap < 0.05:
        raise SystemExit("REFUSING: this axis lies ALONG the hinge -- the pole cannot "
                         "feel it at all. Nothing this run produces would mean anything.")
    if kh < 0.2 and not force:
        raise SystemExit(
            f"REFUSING: kappa_hang = {kh:.4f}. This axis has essentially no authority at "
            "the hanging equilibrium, so a swing-up cannot bootstrap -- driving it just "
            "dumps acceleration into the guarded axes. This exact mistake tripped the "
            "corridor in 0.134 s with the rod tip 4 mm off the floor. Pass --i-know-this-"
            "axis-is-weak to override (a HOLD task legitimately may)."
        )
    if kap < 0.95 or kh < 0.95:
        print(f"  NOTE: {100 * (1 - min(kap, kh)):.0f}% of this axis is not doing useful work.")


def check_guard_frame_or_die(config: Path | None, drive_vec, drive: str) -> None:
    """PREFLIGHT: the drift guard must resolve in the SAME basis the controller
    drives in.

    If the drive axis is not a world axis and the config does not carry a
    ``task_rotation``, the ImpedanceSafetyMonitor measures displacement along
    WORLD axes while the controller commands a diagonal -- so legitimate on-axis
    travel is counted as lateral drift and the guard fires on the motion the
    controller was told to produce.

    Measured at ARM_Q0 2026-08-16, this cost a whole day's worth of wrong
    conclusions. Drive axis [0.716 0.698 0], so 0.698 of every metre travelled
    lands in world Y; against the 0.06 m guard that caps useful travel at
    0.086 m while the validated flip needed 0.187 m. The visible symptom was a
    swing-up search that "chose" a 20x-too-weak pump gain (k_e 13.58 vs the
    277.75 that flips) -- every stronger value tripped |Y-Y0| at 0.77 s. The
    guard was selecting the gain, and nothing in the logs said so: no guard
    fired at the chosen point, and the search looked converged.

    This is NOT a licence to widen a guard. The threshold and the number of
    checked components are unchanged; only the axes they resolve onto.
    """
    v = np.asarray(drive_vec, dtype=np.float64)
    off_world = float(np.min([np.linalg.norm(v - e) for e in
                              (np.eye(3).tolist() + (-np.eye(3)).tolist())]))
    if off_world < 1e-6:
        return                      # a world axis: the world-frame guard is correct
    rot = None
    if config is not None:
        import yaml
        rot = (yaml.safe_load(config.read_text()).get("controller") or {}).get("task_rotation")
    if rot is None:
        raise SystemExit(
            f"REFUSING: --drive-axis {drive} = {np.round(v, 4)} is NOT a world axis, but "
            f"{'no --config was given' if config is None else config.name + ' sets no controller.task_rotation'}"
            ". The drift guard would resolve in WORLD axes while the controller drives a "
            "diagonal, counting on-axis travel as lateral drift and capping the run "
            "without ever reporting why. Give a config whose task_rotation puts this axis "
            "on row 0."
        )
    r0 = np.asarray(rot, dtype=np.float64).reshape(3, 3)[:, 0]
    align = abs(float(r0 @ (v / np.linalg.norm(v))))
    print(f"  guard frame  config row 0 vs drive axis: |cos| = {align:.6f}")
    if align < 0.999:
        raise SystemExit(
            f"REFUSING: the config's task_rotation row 0 is {np.round(r0, 4)} but the "
            f"axis just checked is {np.round(v, 4)} (|cos| = {align:.4f}). The guard and "
            "the drive would resolve in DIFFERENT bases, which is the same bug in a "
            "harder-to-see form."
        )


def backend_for(pose_name: str, asset_name: str, drive: str) -> str:
    """toolY where it can express the request, generic otherwise -- and say why."""
    if drive == "tool-y" and pose_name == "arm_q0" and asset_name == "default":
        return "toolY"
    return "generic"


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("stage", choices=["axes", "swingup", "balance", "handoff"],
                   help="axes = the mandatory pre-run check; the rest dispatch a run.")
    p.add_argument("--pose", default="arm_q0",
                   help=f"named pose {sorted(POSES)}, or six floats via --pose-rad")
    p.add_argument("--pose-rad", type=float, nargs=6, default=None)
    p.add_argument("--asset", default="default", choices=sorted(ASSETS))
    p.add_argument("--drive-axis", default="in-plane-horiz",
                   help="tool-x|tool-y|tool-z|world-x|world-y|world-z|in-plane-horiz")
    p.add_argument("--config", type=Path, default=None)
    p.add_argument("--duration-s", type=float, default=10.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--maxiter", type=int, default=40)
    p.add_argument("--popsize", type=int, default=16)
    p.add_argument("--out", type=Path, default=None, help="output JSON (one spelling)")
    p.add_argument("--i-know-this-axis-is-weak", action="store_true",
                   help="override the kappa_hang refusal. A HOLD task legitimately may.")
    p.add_argument("--dry-run", action="store_true",
                   help="print the backend command instead of running it")
    args = p.parse_args(argv)

    arm_q = resolve_pose(args.pose_rad if args.pose_rad is not None else args.pose)
    asset = ASSETS[args.asset]
    pose_name = args.pose.lower() if args.pose_rad is None else "custom"

    print(f"pose  {pose_name}  {list(np.round(arm_q, 6))}")
    print(f"asset {args.asset}  ({asset.name})")
    check_axis_or_die(asset, arm_q, args.drive_axis, args.i_know_this_axis_is_weak)
    _, _, drive_vec, _ = axis_alignment(asset, arm_q, args.drive_axis)
    check_guard_frame_or_die(args.config, drive_vec, args.drive_axis)
    if args.stage == "axes":
        return 0

    backend = backend_for(pose_name, args.asset, args.drive_axis)
    print(f"  backend: {backend}")
    if backend == "toolY":
        script = {"swingup": "pendulum_toolY_swingup_search.py",
                  "balance": "pendulum_toolY_lqr.py",
                  "handoff": "pendulum_toolY_handoff.py"}[args.stage]
        cmd = [PY, str(DIAG / script), "--output-json", str(args.out or "/dev/stdout")]
        if args.stage != "handoff":
            cmd += ["--duration-s", str(args.duration_s), "--seed", str(args.seed),
                    "--maxiter", str(args.maxiter), "--popsize", str(args.popsize)]
    else:
        # The generic backend cannot aim itself: the drive axis comes from the
        # CONFIG's frame. Refuse rather than run a config whose row 0 is not the
        # axis just checked -- that mismatch is exactly what makes this silent.
        if args.config is None:
            raise SystemExit(
                "the generic backend needs --config: its drive axis is whatever the "
                "config's frame makes row 0, and this tool will not guess. For "
                f"--drive-axis {args.drive_axis} use a config whose task_rotation puts "
                "that axis on row 0."
            )
        script = {"swingup": "pendulum_swingup_energy_shaping.py",
                  "balance": "pendulum_lqr_cascade.py",
                  "handoff": "pendulum_flip_catch_hold.py"}[args.stage]
        cmd = [PY, str(DIAG / script), "--pendulum-xml", str(asset),
               "--start-q-rad", *[str(v) for v in arm_q],
               "--config", str(args.config), "--duration-s", str(args.duration_s)]
        if args.stage != "handoff":
            cmd += ["--seed", str(args.seed), "--maxiter", str(args.maxiter),
                    "--popsize", str(args.popsize)]
        if args.out:
            cmd += ["--output-json", str(args.out)]

    print("  " + " ".join(cmd))
    if args.dry_run:
        return 0
    return subprocess.call(cmd)


if __name__ == "__main__":
    raise SystemExit(main())
