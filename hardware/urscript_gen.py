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
    viscous_scale: tuple[float, float, float, float, float, float] = tuple(DEFAULT_VISCOUS)
    coulomb_scale: tuple[float, float, float, float, float, float] = tuple(DEFAULT_COULOMB)
    stop_input_int_reg: int = 18


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
        kp_posture=float(gains.get("kp_posture", 25.0)),
        kd_posture=float(gains.get("kd_posture", 6.0)),
        kd_joint=float(gains.get("kd_joint", 4.0)),
        lambda_regularization=float(ctrl.get("lambda_regularization", 0.1)),
        use_lambda=shaping if use_lambda is None else bool(use_lambda),
        torque_headroom=float(ctrl.get("torque_headroom", 0.9)),
        reanchor_x_tol_m=float(ctrl.get("reanchor_x_tol_m", 2.0e-3)),
        reanchor_qd_tol_radps=float(ctrl.get("reanchor_qd_tol_radps", 0.05)),
        tau_limits_nm=tau,  # type: ignore[arg-type]
    )


def render_urscript(
    params: UrscriptOscParams,
    *,
    template_path: Path = DEFAULT_TEMPLATE,
) -> str:
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
