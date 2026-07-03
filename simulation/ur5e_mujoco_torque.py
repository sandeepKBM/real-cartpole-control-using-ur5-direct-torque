"""MuJoCo UR5e torque-control helpers.

This module is the simulator-side bridge between a MuJoCo UR5e model with
torque motors and the simulator-independent controller_core stack.

No CoppeliaSim, RTDE, or hardware code is imported here.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal, Sequence

import mujoco
import numpy as np

from controller_core import (
    CartesianImpedanceConfig,
    ImpedanceSafetyConfig,
    ImpedanceSafetyMonitor,
    JsonlTraceWriter,
    TorqueCommandFilter,
    TorqueTaskQPConfig,
    TorqueTaskQPController,
    XAxisCartesianImpedanceController,
)
from controller_core.kinematics_utils import orientation_error_vec_wxyz, rotmat_to_quat

from mujoco_ur5e_tools import (
    UR5E_JOINT_ORDER,
    UR5E_TORQUE_ACTUATOR_SPECS,
    compute_gravity_torque,
    validate_ur5e_torque_xml_source_tree,
    validate_compiled_ur5e_torque_model,
    torque_limit_vector,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCENE_XML = REPO_ROOT / "assets" / "ur5e_torque" / "scene.xml"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "ur5e_mujoco_torque"

ControllerKind = Literal["torque_qp", "impedance", "zero_torque"]


@dataclass
class ZeroTorqueControllerOutput:
    tau: np.ndarray

    def as_dict(self) -> dict[str, Any]:
        return {"mode": "zero_torque", "tau": np.asarray(self.tau, dtype=np.float64).reshape(6).tolist()}


class ZeroTorqueController:
    """Diagnostic controller that always returns zero torque."""

    def __init__(self) -> None:
        self._initialized = False

    def reset_from_state(self, state: dict[str, Any]) -> None:  # pragma: no cover - trivial
        del state
        self._initialized = True

    @property
    def initialized(self) -> bool:  # pragma: no cover - trivial
        return self._initialized

    def compute(self, state: dict[str, Any]) -> ZeroTorqueControllerOutput:
        del state
        if not self._initialized:
            self._initialized = True
        return ZeroTorqueControllerOutput(tau=np.zeros(6, dtype=np.float64))


@dataclass
class MujocoUR5eState:
    time_s: float
    dt_s: float
    q: np.ndarray
    qd: np.ndarray
    ee_pos: np.ndarray
    ee_quat: np.ndarray
    ee_lin_vel: np.ndarray
    ee_ang_vel: np.ndarray
    jacobian: np.ndarray
    gravity_torque: np.ndarray | None = None
    target_x: float = 0.0
    target_x_vel: float = 0.0
    target_axis: float | None = None
    target_axis_vel: float | None = None
    target_ee_pos: np.ndarray | None = None
    target_ee_vel: np.ndarray | None = None
    reference_quat: np.ndarray | None = None
    hold_current_pose: bool = False
    transport_axis_index: int = 0

    def as_robot_state(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "time": float(self.time_s),
            "dt_s": float(self.dt_s),
            "q": np.asarray(self.q, dtype=np.float64).reshape(6),
            "qd": np.asarray(self.qd, dtype=np.float64).reshape(6),
            "ee_pos": np.asarray(self.ee_pos, dtype=np.float64).reshape(3),
            "ee_quat": np.asarray(self.ee_quat, dtype=np.float64).reshape(4),
            "ee_lin_vel": np.asarray(self.ee_lin_vel, dtype=np.float64).reshape(3),
            "ee_ang_vel": np.asarray(self.ee_ang_vel, dtype=np.float64).reshape(3),
            "jacobian": np.asarray(self.jacobian, dtype=np.float64).reshape(6, 6),
            "target_x": float(self.target_x),
            "target_x_vel": float(self.target_x_vel),
            "hold_current_pose": bool(self.hold_current_pose),
            "transport_axis_index": int(self.transport_axis_index),
        }
        if self.target_axis is not None:
            out["target_axis"] = float(self.target_axis)
        if self.target_axis_vel is not None:
            out["target_axis_vel"] = float(self.target_axis_vel)
        if self.target_ee_pos is not None:
            out["target_ee_pos"] = np.asarray(self.target_ee_pos, dtype=np.float64).reshape(3)
        if self.target_ee_vel is not None:
            out["target_ee_vel"] = np.asarray(self.target_ee_vel, dtype=np.float64).reshape(3)
        if self.reference_quat is not None:
            out["reference_quat"] = np.asarray(self.reference_quat, dtype=np.float64).reshape(4)
        return out


@dataclass
class MujocoUR5eTorqueAdapterConfig:
    controller_kind: ControllerKind = "torque_qp"
    torque_limit_nm: np.ndarray = field(default_factory=torque_limit_vector)
    torque_limit_scale: float = 1.0
    rate_limit_nm_per_sec: np.ndarray = field(
        default_factory=lambda: np.array([800.0, 800.0, 800.0, 160.0, 160.0, 160.0], dtype=np.float64)
    )
    lowpass_alpha: float = 1.0
    gravity_mode: Literal["raw", "gravity_comp"] = "raw"
    gravity_compensation: bool | None = None
    # Where the gravity-compensation vector comes from when gravity_mode ==
    # "gravity_comp". "mujoco_qfrc" is the historical static qd=0 inverse-dynamics
    # bias; "pinocchio" uses controller_core.model_dynamics.PinocchioUR5eDynamics
    # (verified <1e-8 Nm parity against the MuJoCo source on this MJCF).
    gravity_source: Literal["mujoco_qfrc", "pinocchio"] = "mujoco_qfrc"
    # Adds C(q, qd) @ qd feedforward on top of gravity compensation. Historical
    # behavior never compensated velocity-product dynamics, so this defaults off.
    # Computed from the configured gravity_source's engine (MuJoCo: live bias minus
    # static bias on a scratch MjData; Pinocchio: rnea-based coriolis()).
    coriolis_feedforward: bool = False
    transport_axis_index: int = 0

    def validate(self) -> None:
        if self.controller_kind not in ("torque_qp", "impedance", "zero_torque"):
            raise ValueError(f"Unsupported controller_kind: {self.controller_kind!r}")
        self.torque_limit_nm = np.asarray(self.torque_limit_nm, dtype=np.float64).reshape(6)
        if not np.isfinite(float(self.torque_limit_scale)) or float(self.torque_limit_scale) <= 0.0:
            raise ValueError("torque_limit_scale must be a positive finite scalar")
        self.rate_limit_nm_per_sec = np.asarray(self.rate_limit_nm_per_sec, dtype=np.float64).reshape(6)
        if np.any(self.torque_limit_nm <= 0.0):
            raise ValueError("torque_limit_nm must be strictly positive")
        if np.any(self.rate_limit_nm_per_sec <= 0.0):
            raise ValueError("rate_limit_nm_per_sec must be strictly positive")
        if not np.isfinite(float(self.lowpass_alpha)):
            raise ValueError("lowpass_alpha must be finite")
        if not (0.0 < float(self.lowpass_alpha) <= 1.0):
            raise ValueError("lowpass_alpha must be in (0, 1]")
        if int(self.transport_axis_index) not in (0, 1, 2):
            raise ValueError("transport_axis_index must be 0, 1, or 2")
        if self.gravity_compensation is not None:
            legacy_mode = "gravity_comp" if bool(self.gravity_compensation) else "raw"
            if self.gravity_mode not in ("raw", "gravity_comp"):
                raise ValueError(f"gravity_mode must be 'raw' or 'gravity_comp'; got {self.gravity_mode!r}")
            if self.gravity_mode not in ("raw", legacy_mode):
                raise ValueError(
                    "gravity_mode conflicts with legacy gravity_compensation flag: "
                    f"gravity_mode={self.gravity_mode!r}, gravity_compensation={self.gravity_compensation!r}"
                )
            self.gravity_mode = legacy_mode
        if self.gravity_mode not in ("raw", "gravity_comp"):
            raise ValueError(f"gravity_mode must be 'raw' or 'gravity_comp'; got {self.gravity_mode!r}")
        if self.gravity_source not in ("mujoco_qfrc", "pinocchio"):
            raise ValueError(
                f"gravity_source must be 'mujoco_qfrc' or 'pinocchio'; got {self.gravity_source!r}"
            )


def load_model(scene_xml: str | Path) -> tuple[mujoco.MjModel, mujoco.MjData, int, list[int], list[int]]:
    """Load and validate a torque-actuated UR5e scene."""
    scene_xml = Path(scene_xml).expanduser().resolve()
    validate_ur5e_torque_xml_source_tree(scene_xml)
    model = mujoco.MjModel.from_xml_path(str(scene_xml))
    joint_ids, actuator_ids, site_id = validate_compiled_ur5e_torque_model(model, site_name="attachment_site")
    data = mujoco.MjData(model)
    return model, data, site_id, joint_ids, actuator_ids


def build_controller(kind: ControllerKind, ctrl_cfg: dict[str, Any]) -> Any:
    """Instantiate one of the reusable controller_core torque laws."""
    if kind == "torque_qp":
        return TorqueTaskQPController(TorqueTaskQPConfig.from_controller_yaml_section(ctrl_cfg))
    if kind == "impedance":
        return XAxisCartesianImpedanceController(CartesianImpedanceConfig.from_controller_yaml_section(ctrl_cfg))
    if kind == "zero_torque":
        return ZeroTorqueController()
    raise ValueError(f"Unsupported controller kind: {kind!r}")


def compute_joint_limit_proximity(model: mujoco.MjModel, q: np.ndarray, joint_ids: Sequence[int]) -> dict[str, float]:
    q = np.asarray(q, dtype=np.float64).reshape(6)
    proximity: dict[str, float] = {}
    for idx, jid in enumerate(joint_ids):
        qmin = float(model.jnt_range[jid, 0])
        qmax = float(model.jnt_range[jid, 1])
        if not np.isfinite(qmin) or not np.isfinite(qmax) or qmax <= qmin:
            continue
        dist = min(float(q[idx] - qmin), float(qmax - q[idx]))
        proximity[UR5E_JOINT_ORDER[idx]] = float(dist / max(qmax - qmin, 1e-12))
    return proximity


def build_mujoco_state(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    site_id: int,
    joint_ids: Sequence[int],
    time_s: float,
    dt_s: float,
    target_x: float,
    target_x_vel: float = 0.0,
    target_axis: float | None = None,
    target_axis_vel: float | None = None,
    target_ee_pos: np.ndarray | None = None,
    target_ee_vel: np.ndarray | None = None,
    reference_quat: np.ndarray | None = None,
    hold_current_pose: bool = False,
    transport_axis_index: int = 0,
    gravity_compensation: bool = True,
) -> MujocoUR5eState:
    q = np.zeros(6, dtype=np.float64)
    qd = np.zeros(6, dtype=np.float64)
    for idx, jid in enumerate(joint_ids):
        qadr = int(model.jnt_qposadr[jid])
        vadr = int(model.jnt_dofadr[jid])
        q[idx] = float(data.qpos[qadr])
        qd[idx] = float(data.qvel[vadr])

    mujoco.mj_forward(model, data)
    ee_pos = np.asarray(data.site_xpos[site_id], dtype=np.float64).copy()
    ee_rot = np.asarray(data.site_xmat[site_id], dtype=np.float64).reshape(3, 3).copy()
    ee_quat = rotmat_to_quat(ee_rot)
    jacp = np.zeros((3, model.nv), dtype=np.float64)
    jacr = np.zeros((3, model.nv), dtype=np.float64)
    mujoco.mj_jacSite(model, data, jacp, jacr, site_id)
    jacobian = np.vstack([jacp[:, :6], jacr[:, :6]]).astype(np.float64)
    ee_lin_vel = jacp[:, :6] @ qd
    ee_ang_vel = jacr[:, :6] @ qd
    gravity_torque = compute_gravity_torque(model, data, joint_ids) if gravity_compensation else None
    return MujocoUR5eState(
        time_s=float(time_s),
        dt_s=float(dt_s),
        q=q,
        qd=qd,
        ee_pos=ee_pos,
        ee_quat=ee_quat,
        ee_lin_vel=ee_lin_vel,
        ee_ang_vel=ee_ang_vel,
        jacobian=jacobian,
        gravity_torque=gravity_torque,
        target_x=float(target_x),
        target_x_vel=float(target_x_vel),
        target_axis=None if target_axis is None else float(target_axis),
        target_axis_vel=None if target_axis_vel is None else float(target_axis_vel),
        target_ee_pos=None if target_ee_pos is None else np.asarray(target_ee_pos, dtype=np.float64).reshape(3),
        target_ee_vel=None if target_ee_vel is None else np.asarray(target_ee_vel, dtype=np.float64).reshape(3),
        reference_quat=None if reference_quat is None else np.asarray(reference_quat, dtype=np.float64).reshape(4),
        hold_current_pose=bool(hold_current_pose),
        transport_axis_index=int(transport_axis_index),
    )


class MujocoUR5eTorqueAdapter:
    """Adapter that turns MuJoCo state into clipped torque commands.

    The adapter is intentionally thin:
    - it constructs the canonical controller_core state dict,
    - calls a reusable controller_core controller,
    - low-pass filters / slew-rate limits the raw torque,
    - saturates to the configured torque limits,
    - and computes safety diagnostics.
    """

    def __init__(
        self,
        *,
        model: mujoco.MjModel,
        site_id: int,
        joint_ids: Sequence[int],
        controller: Any,
        config: MujocoUR5eTorqueAdapterConfig,
        safety_cfg: ImpedanceSafetyConfig | None = None,
    ) -> None:
        self.model = model
        self.site_id = int(site_id)
        self.joint_ids = list(joint_ids)
        self.controller = controller
        self.cfg = config
        self.cfg.validate()
        self.torque_limit_nm = np.asarray(self.cfg.torque_limit_nm, dtype=np.float64).reshape(6)
        self.rate_limit_nm_per_sec = np.asarray(self.cfg.rate_limit_nm_per_sec, dtype=np.float64).reshape(6)
        self.torque_filter = TorqueCommandFilter(6, self.cfg.lowpass_alpha, self.rate_limit_nm_per_sec)
        self.safety_monitor = ImpedanceSafetyMonitor(
            safety_cfg
            or ImpedanceSafetyConfig(
                max_abs_y_drift_m=0.03,
                max_abs_z_drift_m=0.03,
                max_abs_orthogonal_drift_m=0.03,
                max_orientation_error_rad=0.35,
                max_joint_velocity_radps=3.0,
            )
        )
        self._gravity_scratch = mujoco.MjData(model)
        self._pin_dynamics = None
        self._initialized = False
        self._initial_pos: np.ndarray | None = None
        self._initial_quat: np.ndarray | None = None
        self._prev_tau: np.ndarray | None = None

    def reset(self, state: MujocoUR5eState) -> None:
        self.torque_filter.reset()
        robot_state = state.as_robot_state()
        if hasattr(self.controller, "reset_from_state"):
            self.controller.reset_from_state(robot_state)
        self.safety_monitor.reset()
        self.safety_monitor.set_initial_position(np.asarray(state.ee_pos, dtype=np.float64).reshape(3), self.cfg.transport_axis_index)
        self._initial_pos = np.asarray(state.ee_pos, dtype=np.float64).reshape(3).copy()
        self._initial_quat = np.asarray(state.ee_quat, dtype=np.float64).reshape(4).copy()
        self._prev_tau = np.zeros(6, dtype=np.float64)
        self._initialized = True

    @property
    def initialized(self) -> bool:
        return self._initialized

    def _controller_step(self, state: MujocoUR5eState) -> tuple[np.ndarray, dict[str, Any]]:
        robot_state = state.as_robot_state()
        if not self._initialized:
            self.reset(state)
        output = self.controller.compute(robot_state)
        tau = np.asarray(getattr(output, "tau"), dtype=np.float64).reshape(6)
        controller_diag = {
            "controller_kind": type(self.controller).__name__,
            "controller_output": output.as_dict() if hasattr(output, "as_dict") else dict(vars(output)),
        }
        return tau, controller_diag

    def shape_torque(
        self,
        tau_raw: np.ndarray,
        *,
        dt_s: float,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        tau_raw = np.asarray(tau_raw, dtype=np.float64).reshape(6)
        tau_filtered, filter_diag = self.torque_filter.apply_with_diagnostics(tau_raw, dt_s)
        tau_clipped = np.clip(tau_filtered, -self.torque_limit_nm, +self.torque_limit_nm)
        saturated = np.abs(tau_clipped - tau_filtered) > 1e-9
        diag = {
            "tau_raw": tau_raw.tolist(),
            "tau_filtered": tau_filtered.tolist(),
            "tau_clipped": tau_clipped.tolist(),
            "tau_saturated": saturated.tolist(),
            "torque_saturation_fraction": float(np.mean(np.abs(tau_clipped) / np.maximum(self.torque_limit_nm, 1e-12))),
            "torque_clip_fraction": float(np.mean(saturated.astype(np.float64))),
            "filter": filter_diag,
        }
        return tau_clipped, diag

    def _gravity_torque(self, state: MujocoUR5eState) -> np.ndarray:
        if self.cfg.gravity_mode != "gravity_comp":
            return np.zeros(6, dtype=np.float64)
        if state.gravity_torque is not None:
            return np.asarray(state.gravity_torque, dtype=np.float64).reshape(6)
        q = np.asarray(state.q, dtype=np.float64).reshape(6)
        if self.cfg.gravity_source == "pinocchio":
            return self._pinocchio_dynamics().gravity(q)
        return compute_gravity_torque(self.model, q, self.joint_ids, scratch_data=self._gravity_scratch)

    def _pinocchio_dynamics(self):
        if self._pin_dynamics is None:
            from controller_core.model_dynamics import PinocchioUR5eDynamics

            self._pin_dynamics = PinocchioUR5eDynamics()
        return self._pin_dynamics

    def _coriolis_torque(self, state: MujocoUR5eState) -> np.ndarray:
        if not self.cfg.coriolis_feedforward or self.cfg.gravity_mode != "gravity_comp":
            return np.zeros(6, dtype=np.float64)
        q = np.asarray(state.q, dtype=np.float64).reshape(6)
        qd = np.asarray(state.qd, dtype=np.float64).reshape(6)
        if self.cfg.gravity_source == "pinocchio":
            return self._pinocchio_dynamics().coriolis(q, qd)
        # MuJoCo-native: C(q,qd)qd = qfrc_bias(q,qd) - qfrc_bias(q,0), on scratch data.
        scratch = self._gravity_scratch
        scratch.qpos[:] = 0.0
        scratch.qpos[: q.shape[0]] = q
        scratch.qvel[:] = 0.0
        scratch.qvel[: qd.shape[0]] = qd
        mujoco.mj_forward(self.model, scratch)
        bias_live = np.asarray(scratch.qfrc_bias, dtype=np.float64)[:6].copy()
        scratch.qvel[:] = 0.0
        mujoco.mj_forward(self.model, scratch)
        bias_static = np.asarray(scratch.qfrc_bias, dtype=np.float64)[:6].copy()
        return bias_live - bias_static

    def apply_torque_components(
        self,
        *,
        state: MujocoUR5eState,
        tau_controller: np.ndarray,
        controller_diag: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        if not self._initialized:
            self.reset(state)
        tau_controller = np.asarray(tau_controller, dtype=np.float64).reshape(6)
        tau_controller_clipped = np.clip(tau_controller, -self.torque_limit_nm, +self.torque_limit_nm)
        controller_saturated = np.abs(tau_controller_clipped - tau_controller) > 1e-9
        tau_gravity = self._gravity_torque(state)
        tau_coriolis = self._coriolis_torque(state)
        tau_applied = tau_controller + tau_gravity + tau_coriolis
        tau, torque_diag = self.shape_torque(tau_applied, dt_s=state.dt_s)
        axis_idx = int(np.clip(self.cfg.transport_axis_index, 0, 2))
        axis_err = float(state.target_x - state.ee_pos[axis_idx]) if axis_idx == 0 else (
            float((state.target_axis if state.target_axis is not None else state.target_x) - state.ee_pos[axis_idx])
        )
        orient_err_vec = orientation_error_vec_wxyz(self._initial_quat if self._initial_quat is not None else state.ee_quat, state.ee_quat)
        safety = self.safety_monitor.check(
            state.as_robot_state(),
            axis_error=axis_err,
            orientation_error_norm=float(np.linalg.norm(orient_err_vec)),
        )
        diag = {
            **(controller_diag or {}),
            **torque_diag,
            "gravity_mode": self.cfg.gravity_mode,
            "gravity_mode_used": self.cfg.gravity_mode,
            "gravity_source": self.cfg.gravity_source,
            "gravity_compensation_active": bool(self.cfg.gravity_mode == "gravity_comp"),
            "raw_mode_used": bool(self.cfg.gravity_mode == "raw"),
            "tau_controller": tau_controller.tolist(),
            "tau_controller_clipped": tau_controller_clipped.tolist(),
            "tau_controller_saturated": controller_saturated.tolist(),
            "tau_controller_clip_fraction": float(np.mean(controller_saturated.astype(np.float64))),
            "controller_torque_clip_fraction": float(np.mean(controller_saturated.astype(np.float64))),
            "tau_gravity": tau_gravity.tolist(),
            "tau_coriolis": tau_coriolis.tolist(),
            "coriolis_feedforward_active": bool(self.cfg.coriolis_feedforward and self.cfg.gravity_mode == "gravity_comp"),
            "tau_applied": tau_applied.tolist(),
            "tau_applied_clipped": tau.tolist(),
            "tau_raw": tau_controller.tolist(),
            "tau_filtered": torque_diag["tau_filtered"],
            "tau_clipped": torque_diag["tau_clipped"],
            "tau": tau.tolist(),
            "safety_ok": bool(safety.ok),
            "safety_reason": safety.reason,
            "axis_error": float(axis_err),
            "orientation_error_norm": float(np.linalg.norm(orient_err_vec)),
            "tau_applied_clip_fraction": float(torque_diag["torque_clip_fraction"]),
            "applied_torque_clip_fraction": float(torque_diag["torque_clip_fraction"]),
        }
        self._prev_tau = tau.copy()
        return tau, diag

    def step(
        self,
        *,
        state: MujocoUR5eState,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        if not self._initialized:
            self.reset(state)
        tau_controller, controller_diag = self._controller_step(state)
        return self.apply_torque_components(state=state, tau_controller=tau_controller, controller_diag=controller_diag)


def compute_reward_terms(
    *,
    state: MujocoUR5eState,
    tau: np.ndarray,
    tau_prev: np.ndarray | None,
    reward_cfg: dict[str, Any],
    axis_idx: int = 0,
) -> dict[str, float]:
    tau = np.asarray(tau, dtype=np.float64).reshape(6)
    tau_prev = np.zeros(6, dtype=np.float64) if tau_prev is None else np.asarray(tau_prev, dtype=np.float64).reshape(6)
    ee = np.asarray(state.ee_pos, dtype=np.float64).reshape(3)
    ee_vel = np.asarray(state.ee_lin_vel, dtype=np.float64).reshape(3)
    quat = np.asarray(state.ee_quat, dtype=np.float64).reshape(4)
    quat_ref = np.asarray(state.reference_quat, dtype=np.float64).reshape(4) if state.reference_quat is not None else quat
    orient_err = float(np.linalg.norm(orientation_error_vec_wxyz(quat_ref, quat)))
    axis_idx = int(np.clip(axis_idx, 0, 2))
    target_ee = (
        np.asarray(state.target_ee_pos, dtype=np.float64).reshape(3)
        if state.target_ee_pos is not None
        else np.array(
            [
                float(state.target_x),
                float(ee[1]),
                float(ee[2]),
            ],
            dtype=np.float64,
        )
    )
    target_axis = float(target_ee[axis_idx])
    axis_err = abs(float(target_axis - ee[axis_idx]))
    reward = float(reward_cfg.get("alive_bonus", 0.0))
    reward -= float(reward_cfg.get("x_hold_weight", 0.0)) * abs(float(target_ee[0] - ee[0]))
    reward -= float(reward_cfg.get("y_hold_weight", 0.0)) * abs(float(target_ee[1] - ee[1]))
    reward -= float(reward_cfg.get("z_hold_weight", 0.0)) * abs(float(target_ee[2] - ee[2]))
    reward -= float(reward_cfg.get("orientation_weight", 0.0)) * orient_err
    reward -= float(reward_cfg.get("torque_smooth_weight", 0.0)) * float(np.sum((tau - tau_prev) ** 2))
    return {
        "reward": reward,
        "axis_error": axis_err,
        "orientation_error_norm": orient_err,
        "abs_tau_sum": float(np.sum(np.abs(tau))),
        "tau_delta_norm": float(np.linalg.norm(tau - tau_prev)),
        "ee_speed_norm": float(np.linalg.norm(ee_vel)),
    }


def write_trace_plot(trace_rows: Sequence[dict[str, Any]], output_path: str | Path) -> Path:
    """Write a simple multi-panel plot for q, qd, and torque traces."""
    import matplotlib.pyplot as plt

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not trace_rows:
        raise ValueError("trace_rows is empty")

    t = np.array([float(r["time_s"]) for r in trace_rows], dtype=np.float64)
    q = np.array([r["q"] for r in trace_rows], dtype=np.float64)
    qd = np.array([r["qd"] for r in trace_rows], dtype=np.float64)
    tau_raw = np.array([r["tau_raw"] for r in trace_rows], dtype=np.float64)
    tau = np.array([r["tau"] for r in trace_rows], dtype=np.float64)
    axis_err = np.array([float(r.get("axis_error", 0.0)) for r in trace_rows], dtype=np.float64)

    fig, axes = plt.subplots(5, 1, figsize=(12, 14), sharex=True)
    for i in range(6):
        axes[0].plot(t, q[:, i], label=UR5E_JOINT_ORDER[i])
    axes[0].set_ylabel("q [rad]")
    axes[0].legend(ncol=3, fontsize=8)

    for i in range(6):
        axes[1].plot(t, qd[:, i], label=UR5E_JOINT_ORDER[i])
    axes[1].set_ylabel("qd [rad/s]")

    for i in range(6):
        axes[2].plot(t, tau_raw[:, i], label=f"raw {UR5E_JOINT_ORDER[i]}")
    axes[2].set_ylabel("tau raw [Nm]")

    for i in range(6):
        axes[3].plot(t, tau[:, i], label=f"clipped {UR5E_JOINT_ORDER[i]}")
    axes[3].set_ylabel("tau clipped [Nm]")

    axes[4].plot(t, axis_err, color="tab:red")
    axes[4].set_ylabel("axis error [m]")
    axes[4].set_xlabel("time [s]")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path
