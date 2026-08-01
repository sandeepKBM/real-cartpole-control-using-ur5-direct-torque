"""Generate PolyScope URScript for the on-robot OSC inner loop."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TEMPLATE = REPO_ROOT / "assets" / "urscript" / "x_axis_osc_inner.script.template"
DEFAULT_CONFIG = REPO_ROOT / "config" / "ur5e_mujoco_torque_osc_tuned.yaml"

# Forum / UR defaults for direct_torque V2 friction scaling.
DEFAULT_VISCOUS = [0.9, 0.9, 0.8, 0.9, 0.9, 0.9]
DEFAULT_COULOMB = [0.8, 0.8, 0.7, 0.8, 0.8, 0.8]

# Found 2026-08-01, first-ever real URScript execution: the cond(J) estimate
# (two from-scratch 6x6 Jacobi eigendecompositions, 8 sweeps each, every
# control cycle) is real, substantial embedded-real-time computation whose
# per-cycle cost on actual PolyScope hardware had never been benchmarked
# (flagged as a known gap in AGENTS.md sec 4) -- and the first real attempt
# hung, almost certainly because the loop couldn't keep up with its own 2ms
# budget. For jacobian_singular_cond_max at or above this threshold,
# singular_scale is PROVABLY always 1.0 (cond > threshold > cond_max is
# unreachable for any physically realizable Jacobian -- see
# CartesianImpedanceConfig.jacobian_singular_cond_max's own promoted-default
# value, 1.0e18, which every config exercised so far uses). Skipping the
# whole Jacobi computation in that regime and hardcoding singular_scale=1.0
# produces bit-identical output to running it, so this is a pure performance
# fix, not a behavior change -- verified by the existing parity tests, whose
# only near-singular case (jacobian_singular_cond_max=1.0e5) stays well
# under this threshold and still exercises the real computation.
SINGULAR_SCALE_SKIP_THRESHOLD = 1.0e10

JOINT_ORDER = (
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
)


@dataclass(frozen=True)
class UrscriptOscParams:
    target_x_delta_m: float
    move_duration_s: float
    duration_s: float
    kp_x: float
    kd_x: float
    kp_y: float
    kd_y: float
    kp_z: float
    kd_z: float
    kd_rot: float
    kp_posture: float
    kd_posture: float
    kd_joint: float
    lambda_regularization: float
    use_lambda: bool
    torque_headroom: float
    reanchor_x_tol_m: float
    reanchor_qd_tol_radps: float
    tau_limits_nm: tuple[float, float, float, float, float, float]
    # Nullspace-posture projection (dynamically consistent pseudoinverse, same
    # math as controller_core.x_axis_cartesian_impedance's nullspace_posture
    # flag). Independent of use_lambda: Lambda is computed whenever EITHER is
    # set, matching XAxisCartesianImpedanceController.compute().
    use_nullspace: bool = False
    # Geometric task-scale backtracking (replaces the old per-joint clamp,
    # which distorted torque direction on saturation). Same algorithm and
    # defaults as CartesianImpedanceConfig.task_resample_*.
    task_resample_factor: float = 0.5
    task_resample_min_scale: float = 1.0 / 16384.0
    task_resample_max_iters: int = 14
    # Singular-value (cond(J)) wrench scaling (2026-07-29), mirrors
    # CartesianImpedanceConfig.jacobian_singular_cond_max exactly: when
    # cond(J) exceeds this threshold, the task wrench is scaled down by
    # jacobian_singular_cond_max / cond(J) before being mapped through J.T.
    # URScript has no built-in SVD, so cond(J) is estimated on-robot via two
    # Jacobi eigendecompositions (see the template) rather than computed
    # exactly -- see docs/status/urscript_singular_scaling_parity_2026-07-29.md
    # for the numerical-accuracy tradeoff this implies.
    jacobian_singular_cond_max: float = 1.0e5
    # Generation-only implementation constant (not sourced from the Python
    # config -- there is no equivalent there, since np.linalg.cond is exact).
    # Number of full cyclic Jacobi sweeps used for the on-robot cond(J)
    # estimate; 6 sweeps already converges a 6x6 to ~1e-10 relative error
    # empirically (measured up to cond(J)=1e7), this keeps a small margin.
    singular_scale_jacobi_sweeps: int = 8
    viscous_scale: tuple[float, float, float, float, float, float] = tuple(DEFAULT_VISCOUS)
    coulomb_scale: tuple[float, float, float, float, float, float] = tuple(DEFAULT_COULOMB)
    stop_input_int_reg: int = 18
    # Rotational stiffness. Carried through so the generator is AWARE of it, but
    # the current on-robot template implements damping-only rotation
    # (Mx/My/Mz = -kd_rot*omega, no orientation-error term), so it cannot honor a
    # nonzero kp_rot. render_urscript() raises rather than silently dropping it --
    # previously kp_rot was not a field at all, so a config with rotational
    # stiffness would have been silently ignored on the real arm. Defaulted to 0
    # so existing callers keep working (the tuned config has kp_rot=0).
    kp_rot: float = 0.0
    # Wrist-orientation task (2026-08-01), ported from
    # controller_core.x_axis_cartesian_impedance.CartesianImpedanceConfig
    # .wrist_orientation_task -- see that file's docstring and
    # docs/status/wrist_orientation_task_2026-07-29.md for the mechanism and
    # docs/status/urscript_wrist_orientation_parity_2026-08-01.md for this port.
    # Unlike kp_rot (damping-only rotation, no orientation-error math at all),
    # this term is structurally SEPARATE from the shared Lambda-weighted wrench
    # pipeline: a plain joint-space PD term computed from
    # J_rot.T @ (kp_rot_wrist*e_rot - kd_rot_wrist*omega), masked to the wrist
    # joints only via the fixed WRIST_ORIENTATION_MASK shape (baked as a literal
    # in the template -- not user-configurable, matching the Python source).
    # Porting this requires quaternion orientation-error math the template
    # previously had no need for (get_actual_tcp_pose() returns a UR rotation
    # vector, not a quaternion) -- see the template's new
    # rotvec_to_quat/orientation_error_vec helpers. Default False/0.0/0.0 =
    # historical (pre-2026-08-01) behavior; the term evaluates to the zero
    # vector whenever the flag is off (regardless of the gain values) or both
    # gains are zero, verified numerically identical to before in
    # tests/hardware/test_urscript_parity.py.
    use_wrist_orientation_task: bool = False
    kp_rot_wrist: float = 0.0
    kd_rot_wrist: float = 0.0


def load_params_from_yaml(
    config_path: Path,
    *,
    target_x_delta_m: float,
    move_duration_s: float,
    duration_s: float,
    use_lambda: bool | None = None,
) -> UrscriptOscParams:
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    ctrl = cfg.get("controller", {}) or {}
    gains = ctrl.get("gains", {}) or {}
    lim_dict = ctrl.get("torque_limits_initial", {}) or {}
    tau = tuple(float(lim_dict[name]) for name in JOINT_ORDER)
    shaping = bool(ctrl.get("task_space_inertia_shaping", False))
    nullspace = bool(ctrl.get("nullspace_posture", False))
    return UrscriptOscParams(
        target_x_delta_m=float(target_x_delta_m),
        move_duration_s=float(move_duration_s),
        duration_s=float(duration_s),
        kp_x=float(gains.get("kp_x", 400.0)),
        kd_x=float(gains.get("kd_x", 40.0)),
        kp_y=float(gains.get("kp_y", 80.0)),
        kd_y=float(gains.get("kd_y", 15.0)),
        kp_z=float(gains.get("kp_z", 120.0)),
        kd_z=float(gains.get("kd_z", 20.0)),
        kd_rot=float(gains.get("kd_rot", 10.0)),
        kp_rot=float(gains.get("kp_rot", 0.0)),
        kp_posture=float(gains.get("kp_posture", 25.0)),
        kd_posture=float(gains.get("kd_posture", 6.0)),
        kd_joint=float(gains.get("kd_joint", 4.0)),
        lambda_regularization=float(ctrl.get("lambda_regularization", 0.1)),
        use_lambda=shaping if use_lambda is None else bool(use_lambda),
        torque_headroom=float(ctrl.get("torque_headroom", 0.9)),
        reanchor_x_tol_m=float(ctrl.get("reanchor_x_tol_m", 2.0e-3)),
        reanchor_qd_tol_radps=float(ctrl.get("reanchor_qd_tol_radps", 0.05)),
        tau_limits_nm=tau,  # type: ignore[arg-type]
        use_nullspace=nullspace,
        task_resample_factor=float(ctrl.get("task_resample_factor", 0.5)),
        task_resample_min_scale=float(ctrl.get("task_resample_min_scale", 1.0 / 16384.0)),
        task_resample_max_iters=int(ctrl.get("task_resample_max_iters", 14)),
        jacobian_singular_cond_max=float(ctrl.get("jacobian_singular_cond_max", 1.0e5)),
        use_wrist_orientation_task=bool(ctrl.get("wrist_orientation_task", False)),
        kp_rot_wrist=float(gains.get("kp_rot_wrist", 0.0)),
        kd_rot_wrist=float(gains.get("kd_rot_wrist", 0.0)),
    )


def render_urscript(
    params: UrscriptOscParams,
    *,
    template_path: Path = DEFAULT_TEMPLATE,
) -> str:
    # The on-robot template implements damping-only rotation
    # (Mx/My/Mz = -kd_rot*omega); it has no orientation-error term, so it cannot
    # apply a rotational-stiffness torque. Refuse to generate a script for a
    # config with nonzero kp_rot rather than silently dropping the term on the
    # real arm (the validated tuned config uses kp_rot=0). Honoring kp_rot would
    # require adding quaternion orientation-error math to the on-robot script --
    # a control-law change that has to be validated on hardware first.
    if float(params.kp_rot) != 0.0:
        raise ValueError(
            "URScript OSC template implements damping-only rotation (no "
            f"orientation-error term); cannot honor kp_rot={params.kp_rot!r}. "
            "Set kp_rot=0 for the URScript lane, or extend the template with a "
            "validated orientation-error term before using rotational stiffness."
        )
    template = template_path.read_text(encoding="utf-8")
    repl = {
        "{{TARGET_X_DELTA}}": f"{params.target_x_delta_m:.12g}",
        "{{MOVE_DURATION_S}}": f"{params.move_duration_s:.12g}",
        "{{DURATION_S}}": f"{params.duration_s:.12g}",
        "{{KP_X}}": f"{params.kp_x:.12g}",
        "{{KD_X}}": f"{params.kd_x:.12g}",
        "{{KP_Y}}": f"{params.kp_y:.12g}",
        "{{KD_Y}}": f"{params.kd_y:.12g}",
        "{{KP_Z}}": f"{params.kp_z:.12g}",
        "{{KD_Z}}": f"{params.kd_z:.12g}",
        "{{KD_ROT}}": f"{params.kd_rot:.12g}",
        "{{KP_POSTURE}}": f"{params.kp_posture:.12g}",
        "{{KD_POSTURE}}": f"{params.kd_posture:.12g}",
        "{{KD_JOINT}}": f"{params.kd_joint:.12g}",
        "{{LAMBDA_REG}}": f"{params.lambda_regularization:.12g}",
        "{{USE_LAMBDA}}": "1" if params.use_lambda else "0",
        "{{USE_NULLSPACE}}": "1" if params.use_nullspace else "0",
        "{{USE_WRIST_ORIENT}}": "1" if params.use_wrist_orientation_task else "0",
        "{{KP_ROT_WRIST}}": f"{params.kp_rot_wrist:.12g}",
        "{{KD_ROT_WRIST}}": f"{params.kd_rot_wrist:.12g}",
        "{{TASK_RESAMPLE_FACTOR}}": f"{params.task_resample_factor:.12g}",
        "{{TASK_RESAMPLE_MIN_SCALE}}": f"{params.task_resample_min_scale:.12g}",
        "{{TASK_RESAMPLE_MAX_ITERS}}": str(int(params.task_resample_max_iters)),
        "{{JACOBIAN_SINGULAR_COND_MAX}}": f"{params.jacobian_singular_cond_max:.12g}",
        "{{SINGULAR_SCALE_JACOBI_SWEEPS}}": str(int(params.singular_scale_jacobi_sweeps)),
        "{{USE_SINGULAR_SCALE}}": (
            "1" if params.jacobian_singular_cond_max < SINGULAR_SCALE_SKIP_THRESHOLD else "0"
        ),
        "{{TORQUE_HEADROOM}}": f"{params.torque_headroom:.12g}",
        "{{REANCHOR_X_TOL}}": f"{params.reanchor_x_tol_m:.12g}",
        "{{REANCHOR_QD_TOL}}": f"{params.reanchor_qd_tol_radps:.12g}",
        "{{STOP_INPUT_INT_REG}}": str(int(params.stop_input_int_reg)),
    }
    for i, val in enumerate(params.tau_limits_nm):
        repl[f"{{{{TAU_LIMIT_{i}}}}}"] = f"{val:.12g}"
    for i, val in enumerate(params.viscous_scale):
        repl[f"{{{{VISCOUS_{i}}}}}"] = f"{val:.12g}"
    for i, val in enumerate(params.coulomb_scale):
        repl[f"{{{{COULOMB_{i}}}}}"] = f"{val:.12g}"

    out = template
    for key, val in repl.items():
        out = out.replace(key, val)
    if "{{" in out:
        raise ValueError("URScript template has unreplaced placeholders")
    return out


def write_generated_script(
    params: UrscriptOscParams,
    output_path: Path,
    *,
    template_path: Path = DEFAULT_TEMPLATE,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(render_urscript(params, template_path=template_path), encoding="utf-8")
    return output_path
