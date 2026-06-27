"""
CoppeliaSim ZMQ Remote API adapter for UR5 torque control.

- Resolves joint / EE handles from YAML paths; on failure prints UR5-related
  joint candidates and raises (no silent wrong mapping).
- Prefers ``sim.setJointTargetForce(handle, tau, True)`` when available;
  falls back to ``(setJointTargetVelocity, setJointMaxForce)`` convention.
- Jacobian: tries ``sim.getJacobian``; if missing or unstable, optional
  numerical Jacobian via small joint perturbations.

This module does **not** import ROS.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Sequence
import time

import numpy as np

from controller_core.kinematics_utils import (
    orientation_error_vec_wxyz,
    quat_normalize_wxyz,
)


CANONICAL_JOINT_ORDER: tuple[str, ...] = (
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
)

# Extra paths tried after ``joint_name_map`` (covers UR5.ttm vs older aliases).
JOINT_PATH_FALLBACKS: dict[str, list[str]] = {
    "shoulder_pan_joint": [
        ":/UR5/joint",
        "/UR5/UR5/joint",
        ":/UR5/UR5/joint",
        "/UR5/shoulder_pan_joint",
        ":/UR5/shoulder_pan_joint",
    ],
    "shoulder_lift_joint": [
        ":/UR5/link/joint",
        "/UR5/joint/link/joint",
        ":/UR5/joint/link/joint",
        "/UR5/UR5/link/joint",
        "/UR5/shoulder_lift_joint",
    ],
    "elbow_joint": [
        ":/UR5/link/link/joint",
        "/UR5/joint/link/joint/link/joint",
        ":/UR5/joint/link/joint/link/joint",
        "/UR5/link/joint/joint",
    ],
    "wrist_1_joint": [
        ":/UR5/link/link/link/joint",
        "/UR5/joint/link/joint/link/joint/link/joint",
        ":/UR5/joint/link/joint/link/joint/link/joint",
    ],
    "wrist_2_joint": [
        ":/UR5/link/link/link/link/joint",
        "/UR5/joint/link/joint/link/joint/link/joint/link/joint",
        ":/UR5/joint/link/joint/link/joint/link/joint/link/joint",
    ],
    "wrist_3_joint": [
        ":/UR5/link/link/link/link/link/joint",
        "/UR5/joint/link/joint/link/joint/link/joint/link/joint/link/joint",
        ":/UR5/joint/link/joint/link/joint/link/joint/link/joint/link/joint",
    ],
}


@dataclass
class CoppeliaSimConfig:
    zmq_host: str = "127.0.0.1"
    zmq_port: int = 23000
    joint_name_map: dict[str, str] = field(default_factory=dict)
    startup_joint_positions_rad: tuple[float, ...] = ()
    ee_object_name: str = "/UR5/UR5_connection"
    ee_object_name_alternates: tuple[str, ...] = ()
    task_frame_mode: str = "ee_object"  # ee_object | mujoco_attachment_dummy
    task_frame_parent_object_name: str = ""
    task_frame_parent_object_alternates: tuple[str, ...] = ()
    task_frame_attachment_offset_m: tuple[float, float, float] = (0.0, 0.0, -0.2)
    task_frame_attachment_quat_wxyz: tuple[float, float, float, float] = (
        -0.7071067811865475,
        0.7071067811865475,
        0.0,
        0.0,
    )
    task_frame_dummy_size_m: float = 0.025
    stepping: bool = False
    prefer_signed_target_force: bool = True
    fallback_large_velocity_rad_s: float = 10.0
    jacobian_source: str = "auto"  # auto | api | numerical
    numerical_epsilon: float = 1e-5
    connect_retries: int = 60
    connect_retry_delay_s: float = 0.5


@dataclass
class _JointHandle:
    name: str
    handle: int
    resolved_path: str = ""


class CoppeliaSimURAdapter:
    def __init__(self, config: CoppeliaSimConfig) -> None:
        self.config = config
        self._sim = None
        self._client = None
        self._joint_handles: list[_JointHandle] = []
        self._raw_ee_handle: int | None = None
        self._raw_ee_resolved_path: str = ""
        self._ee_handle: int | None = None
        self._ee_resolved_path: str = ""
        self._task_frame_summary: dict[str, Any] = {}
        self._log: Callable[[str], None] = print
        self._prev_joint_positions_continuous: np.ndarray | None = None
        self._last_torque_api_modes: list[str] = ["unknown"] * 6

    def last_torque_api_modes(self) -> list[str]:
        """Per-joint CoppeliaSim API used on the most recent ``apply_torque`` call."""
        return list(self._last_torque_api_modes)

    def set_logger(self, fn: Callable[[str], None]) -> None:
        self._log = fn

    def connect(self) -> None:
        try:
            from coppeliasim_zmqremoteapi_client import RemoteAPIClient
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "Install: pip install coppeliasim-zmqremoteapi-client"
            ) from exc
        retries = max(1, int(self.config.connect_retries))
        delay_s = max(0.1, float(self.config.connect_retry_delay_s))
        last_exc: Exception | None = None
        for attempt in range(1, retries + 1):
            try:
                self._client = RemoteAPIClient(self.config.zmq_host, self.config.zmq_port)
                self._sim = self._client.getObject("sim")
                self._resolve_handles_or_raise()
                self._prev_joint_positions_continuous = None
                return
            except Exception as exc:
                last_exc = exc
                self._log(
                    f"Waiting for CoppeliaSim RPC at {self.config.zmq_host}:{self.config.zmq_port} "
                    f"({attempt}/{retries}): {exc}"
                )
                time.sleep(delay_s)
        raise RuntimeError(
            f"Could not connect to CoppeliaSim RPC at "
            f"{self.config.zmq_host}:{self.config.zmq_port}"
        ) from last_exc

    def connect_with_existing_client(self, client: object) -> None:
        """Reuse an already-connected RemoteAPIClient instead of opening a new socket."""
        self._client = client
        self._sim = client.getObject("sim")  # type: ignore[union-attr]
        self._resolve_handles_or_raise()
        self._prev_joint_positions_continuous = None

    def _get_object_safe(self, path: str) -> int:
        assert self._sim is not None
        return int(self._sim.getObject(path))

    def _resolve_object_first(self, paths: list[str]) -> tuple[int, str]:
        assert self._sim is not None
        last: Exception | None = None
        seen: set[str] = set()
        for raw in paths:
            p = (raw or "").strip()
            if not p or p in seen:
                continue
            seen.add(p)
            try:
                return int(self._sim.getObject(p)), p
            except Exception as exc:
                last = exc
        msg = ", ".join(repr(p) for p in paths if p)
        raise RuntimeError(f"Could not resolve any of: [{msg}]") from last

    def _discover_joint_candidates(self) -> list[str]:
        assert self._sim is not None
        sim = self._sim
        out: list[str] = []
        try:
            base = sim.handle_scene
            handles = sim.getObjectsInTree(base, sim.object_joint_type, 0)
        except Exception:
            return out
        for h in handles:
            try:
                alias = sim.getObjectAlias(int(h), 2)
                al = str(alias).lower()
                if "ur5" in al and "joint" in al:
                    out.append(str(alias))
            except Exception:
                continue
        return sorted(set(out))

    def _resolve_handles_or_raise(self) -> None:
        assert self._sim is not None
        errors: list[str] = []
        self._joint_handles = []
        for name in CANONICAL_JOINT_ORDER:
            configured_path = self.config.joint_name_map.get(name, "")
            candidates = [configured_path, *JOINT_PATH_FALLBACKS.get(name, [])]
            try:
                h, resolved_path = self._resolve_object_first(candidates)
                self._joint_handles.append(_JointHandle(name, h, resolved_path=resolved_path))
            except Exception as exc:
                errors.append(f"{name}: candidates {candidates!r} -> {exc}")
        try:
            self._raw_ee_handle, self._raw_ee_resolved_path = self._resolve_object_first(
                [
                    self.config.ee_object_name,
                    *self.config.ee_object_name_alternates,
                ]
            )
        except Exception as exc:
            errors.append(
                "EE candidates "
                f"{[self.config.ee_object_name, *self.config.ee_object_name_alternates]!r}: {exc}"
            )

        if errors:
            self._log("--- Joint / EE resolution FAILED ---")
            for e in errors:
                self._log(f"  {e}")
            cand = self._discover_joint_candidates()
            if cand:
                self._log("Candidate joint-like aliases containing 'UR5' and 'joint':")
                for c in cand[:40]:
                    self._log(f"    {c}")
                if len(cand) > 40:
                    self._log(f"    ... ({len(cand)} total)")
            else:
                self._log("(No joint candidates found via scene tree search.)")
            raise RuntimeError("CoppeliaSim handle resolution failed; fix YAML paths.")
        self._configure_task_frame_or_raise()

    def _normalize_quat_wxyz(self, quat: Sequence[float]) -> list[float]:
        arr = np.asarray(quat, dtype=np.float64).reshape(4)
        n = float(np.linalg.norm(arr))
        if n < 1e-12:
            raise ValueError("task-frame quaternion must be nonzero")
        return (arr / n).tolist()

    def _configure_task_frame_or_raise(self) -> None:
        assert self._sim is not None and self._raw_ee_handle is not None
        sim = self._sim
        mode = str(self.config.task_frame_mode or "ee_object").strip().lower()
        if mode in ("ee_object", "raw_ee", "configured_ee"):
            self._ee_handle = self._raw_ee_handle
            self._ee_resolved_path = self._raw_ee_resolved_path
            self._task_frame_summary = {
                "mode": "ee_object",
                "handle": int(self._ee_handle),
                "resolved_path": self._ee_resolved_path,
                "parent_handle": int(self._raw_ee_handle),
                "parent_resolved_path": self._raw_ee_resolved_path,
                "mujoco_attachment_dummy": False,
            }
            return
        if mode != "mujoco_attachment_dummy":
            raise RuntimeError(
                "Unsupported Coppelia task_frame_mode "
                f"{self.config.task_frame_mode!r}; expected ee_object or mujoco_attachment_dummy."
            )

        parent_handle = self._raw_ee_handle
        parent_path = self._raw_ee_resolved_path
        parent_candidates = [
            self.config.task_frame_parent_object_name,
            *self.config.task_frame_parent_object_alternates,
        ]
        if any(str(p).strip() for p in parent_candidates):
            parent_handle, parent_path = self._resolve_object_first(parent_candidates)

        dummy = int(sim.createDummy(float(self.config.task_frame_dummy_size_m)))
        try:
            sim.setObjectAlias(dummy, "real_cartpole_mujoco_attachment_site")
        except Exception:
            pass
        quat = self._normalize_quat_wxyz(self.config.task_frame_attachment_quat_wxyz)
        offset = np.asarray(self.config.task_frame_attachment_offset_m, dtype=np.float64).reshape(3)
        pose_wxyz = [
            float(offset[0]),
            float(offset[1]),
            float(offset[2]),
            float(quat[0]),
            float(quat[1]),
            float(quat[2]),
            float(quat[3]),
        ]
        try:
            sim.setObjectParent(dummy, int(parent_handle), False)
        except Exception:
            pass
        sim.setObjectPose(dummy + sim.handleflag_wxyzquat, pose_wxyz, int(parent_handle))

        self._ee_handle = dummy
        self._ee_resolved_path = f"{parent_path}/real_cartpole_mujoco_attachment_site"
        self._task_frame_summary = {
            "mode": "mujoco_attachment_dummy",
            "handle": int(dummy),
            "resolved_path": self._ee_resolved_path,
            "parent_handle": int(parent_handle),
            "parent_resolved_path": parent_path,
            "local_offset_m": offset.tolist(),
            "local_quat_wxyz": quat,
            "mujoco_attachment_dummy": True,
        }

    def read_task_frame_summary(self) -> dict[str, Any]:
        return dict(self._task_frame_summary)

    def print_scene_summary(self) -> None:
        if self._sim is None:
            return
        sim = self._sim
        self._log("Resolved CoppeliaSim handles:")
        for jh in self._joint_handles:
            alias = sim.getObjectAlias(jh.handle, 2)
            self._log(
                f"  {jh.name:22s} handle={jh.handle:4d}  alias={alias}  path={jh.resolved_path}"
            )
        if self._ee_handle is not None:
            alias = sim.getObjectAlias(self._ee_handle, 2)
            self._log(
                "  EE object            "
                f"handle={self._ee_handle:4d}  alias={alias}  path={self._ee_resolved_path}"
            )
        if self._raw_ee_handle is not None and self._raw_ee_handle != self._ee_handle:
            alias = sim.getObjectAlias(self._raw_ee_handle, 2)
            self._log(
                "  raw EE object        "
                f"handle={self._raw_ee_handle:4d}  alias={alias}  path={self._raw_ee_resolved_path}"
            )

    def read_joint_configuration(self) -> list[dict[str, Any]]:
        """Return per-joint mode / motor readback when the API exposes it."""
        assert self._sim is not None
        sim = self._sim
        snapshot: list[dict[str, Any]] = []
        for jh in self._joint_handles:
            entry: dict[str, Any] = {
                "name": jh.name,
                "handle": int(jh.handle),
                "resolved_path": jh.resolved_path,
            }
            for param_attr, key in (
                ("jointintparam_motor_enabled", "motor_enabled"),
                ("jointintparam_ctrl_enabled", "ctrl_enabled"),
            ):
                if hasattr(sim, param_attr):
                    try:
                        entry[key] = int(sim.getObjectInt32Param(jh.handle, getattr(sim, param_attr)))
                    except Exception as exc:
                        entry[f"{key}_error"] = str(exc)
            if hasattr(sim, "getJointMode"):
                try:
                    mode_raw = sim.getJointMode(jh.handle)
                    if isinstance(mode_raw, (tuple, list, np.ndarray)):
                        mode = int(np.asarray(mode_raw).reshape(-1)[0])
                    else:
                        mode = int(mode_raw)
                    entry["joint_mode"] = mode
                    if hasattr(sim, "jointmode_dynamic"):
                        entry["joint_mode_is_dynamic"] = bool(mode == int(sim.jointmode_dynamic))
                except Exception as exc:
                    entry["joint_mode_error"] = str(exc)
            snapshot.append(entry)
        return snapshot

    def read_joint_configuration_summary(self) -> dict[str, Any]:
        """Aggregate joint mode readback into a compact verification summary."""
        joints = self.read_joint_configuration()

        def _values(key: str) -> list[Any]:
            out: list[Any] = []
            for row in joints:
                if key in row and row[key] is not None:
                    out.append(row[key])
            return out

        motor_vals = _values("motor_enabled")
        ctrl_vals = _values("ctrl_enabled")
        dyn_vals = _values("joint_mode_is_dynamic")
        mode_vals = _values("joint_mode")

        def _all_match(values: list[Any], expected: Any) -> bool | None:
            if not values:
                return None
            return all(v == expected for v in values)

        return {
            "joints": joints,
            "motor_enabled_verified": _all_match(motor_vals, 1),
            "ctrl_disabled_verified": _all_match(ctrl_vals, 0),
            "dynamic_mode_verified": _all_match(dyn_vals, True),
            "joint_mode_readback_available": bool(mode_vals),
            "motor_readback_available": bool(motor_vals),
            "ctrl_readback_available": bool(ctrl_vals),
        }

    def resolved_joint_handles(self) -> list[dict[str, Any]]:
        """Return the canonical joint name, resolved handle, and path mapping."""
        return [
            {
                "name": jh.name,
                "handle": int(jh.handle),
                "resolved_path": jh.resolved_path,
            }
            for jh in self._joint_handles
        ]

    def configure_force_torque_mode(self) -> None:
        assert self._sim is not None
        sim = self._sim
        for jh in self._joint_handles:
            sim.setJointMode(jh.handle, sim.jointmode_dynamic, 0)
            sim.setObjectInt32Param(jh.handle, sim.jointintparam_motor_enabled, 1)
            sim.setObjectInt32Param(jh.handle, sim.jointintparam_ctrl_enabled, 0)
            self._apply_torque_single(jh.handle, 0.0)

    def _set_all_joint_positions(self, q: np.ndarray) -> None:
        assert self._sim is not None
        for jh, qi in zip(self._joint_handles, q):
            try:
                self._sim.setJointPosition(jh.handle, float(qi))
            except Exception:
                self._sim.setJointPosition(jh.handle, float(qi), -1)

    def set_joint_positions(self, q: Sequence[float]) -> None:
        """Place all resolved UR joints at q before starting a stepped run."""
        q_arr = np.asarray(q, dtype=np.float64).reshape(len(self._joint_handles))
        self._set_all_joint_positions(q_arr)
        self._maybe_forward_kinematics()

    def read_joint_state(self) -> tuple[np.ndarray, np.ndarray]:
        assert self._sim is not None
        raw_q = np.array(
            [self._sim.getJointPosition(jh.handle) for jh in self._joint_handles],
            dtype=np.float64,
        )
        q = np.arctan2(np.sin(raw_q), np.cos(raw_q))
        sim_dt = 0.0
        try:
            if hasattr(self._sim, "getSimulationTimeStep"):
                sim_dt = float(self._sim.getSimulationTimeStep())
        except Exception:
            sim_dt = 0.0
        if self._prev_joint_positions_continuous is None:
            qd = np.zeros_like(q)
        else:
            delta = np.arctan2(np.sin(raw_q - self._prev_joint_positions_continuous), np.cos(raw_q - self._prev_joint_positions_continuous))
            q = self._prev_joint_positions_continuous + delta
            if sim_dt > 0.0 and np.isfinite(sim_dt):
                qd = delta / sim_dt
            else:
                qd = np.array(
                    [self._sim.getJointVelocity(jh.handle) for jh in self._joint_handles],
                    dtype=np.float64,
                )
        self._prev_joint_positions_continuous = q.copy()
        return q, qd

    def read_ee_pose_twist(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        assert self._sim is not None and self._ee_handle is not None
        sim = self._sim
        pose = sim.getObjectPose(self._ee_handle, -1)
        pos = np.array(pose[:3], dtype=np.float64)
        quat_xyzw = np.array(pose[3:], dtype=np.float64)
        quat_wxyz = np.array(
            [quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]], dtype=np.float64
        )
        lin, ang = sim.getObjectVelocity(self._ee_handle)
        return pos, quat_wxyz, np.asarray(lin, dtype=np.float64), np.asarray(ang, dtype=np.float64)

    def read_jacobian_api(self) -> tuple[np.ndarray, np.ndarray]:
        assert self._sim is not None and self._ee_handle is not None
        result = self._sim.getJacobian(self._ee_handle)
        if isinstance(result, tuple) and len(result) == 2:
            data, cols = result
            arr = np.asarray(data, dtype=np.float64)
            n = int(cols)
            mat = arr.reshape(-1, n)
        else:
            arr = np.asarray(result, dtype=np.float64)
            n = len(self._joint_handles)
            mat = arr.reshape(-1, n)
        if mat.shape[0] < 3:
            raise RuntimeError(f"getJacobian returned invalid shape {mat.shape}")
        j_pos = mat[:3, :]
        j_rot = mat[3:6, :] if mat.shape[0] >= 6 else np.zeros((3, j_pos.shape[1]))
        return j_pos, j_rot

    def read_jacobian_numerical(self, epsilon: float) -> tuple[np.ndarray, np.ndarray]:
        """Finite-difference geometric Jacobian in world frame (6x6)."""
        assert self._sim is not None
        q0, _ = self.read_joint_state()
        p0, quat0, _, _ = self.read_ee_pose_twist()
        quat0 = quat_normalize_wxyz(quat0)
        J = np.zeros((6, 6), dtype=np.float64)
        eps = max(float(epsilon), 1e-7)
        for i in range(6):
            q_p = q0.copy()
            q_p[i] += eps
            self._set_all_joint_positions(q_p)
            self._maybe_forward_kinematics()
            p1, quat1, _, _ = self.read_ee_pose_twist()
            quat1 = quat_normalize_wxyz(quat1)
            J[:3, i] = (p1 - p0) / eps
            e_rot = orientation_error_vec_wxyz(quat0, quat1)
            J[3:6, i] = e_rot / eps
        self._set_all_joint_positions(q0)
        self._maybe_forward_kinematics()
        return J[:3, :], J[3:6, :]

    def _maybe_forward_kinematics(self) -> None:
        assert self._sim is not None
        sim = self._sim
        for name in ("forwardKinematic", "handleFk", "computeJacobian"):
            if hasattr(sim, name):
                try:
                    getattr(sim, name)()
                except Exception:
                    pass
                break

    def read_jacobian(self) -> tuple[np.ndarray, np.ndarray]:
        src = (self.config.jacobian_source or "auto").lower()
        if src == "numerical":
            return self.read_jacobian_numerical(self.config.numerical_epsilon)
        if src == "api":
            return self.read_jacobian_api()
        # auto
        try:
            return self.read_jacobian_api()
        except Exception:
            return self.read_jacobian_numerical(self.config.numerical_epsilon)

    def compare_jacobians(self, epsilon: float | None = None) -> dict[str, Any]:
        """Compare API and numerical Jacobians at the current configuration."""
        eps = self.config.numerical_epsilon if epsilon is None else float(epsilon)
        api_pos, api_rot = self.read_jacobian_api()
        num_pos, num_rot = self.read_jacobian_numerical(eps)
        api = np.vstack([api_pos, api_rot])
        num = np.vstack([num_pos, num_rot])
        diff = api - num
        pos_diff = api_pos - num_pos
        rot_diff = api_rot - num_rot
        return {
            "epsilon": float(eps),
            "api": {
                "position": api_pos.tolist(),
                "rotation": api_rot.tolist(),
                "shape": list(api.shape),
                "condition_number": float(np.linalg.cond(api)),
                "frobenius_norm": float(np.linalg.norm(api)),
                "rank": int(np.linalg.matrix_rank(api)),
            },
            "numerical": {
                "position": num_pos.tolist(),
                "rotation": num_rot.tolist(),
                "shape": list(num.shape),
                "condition_number": float(np.linalg.cond(num)),
                "frobenius_norm": float(np.linalg.norm(num)),
                "rank": int(np.linalg.matrix_rank(num)),
            },
            "difference": {
                "position": pos_diff.tolist(),
                "rotation": rot_diff.tolist(),
                "shape": list(diff.shape),
                "max_abs": float(np.max(np.abs(diff))),
                "position_max_abs": float(np.max(np.abs(pos_diff))),
                "rotation_max_abs": float(np.max(np.abs(rot_diff))),
                "frobenius_norm": float(np.linalg.norm(diff)),
                "relative_frobenius_norm": float(
                    np.linalg.norm(diff) / max(np.linalg.norm(api), 1e-12)
                ),
                "all_finite": bool(np.all(np.isfinite(diff))),
            },
        }

    def read_joint_forces(self) -> np.ndarray:
        """Read per-joint applied force/torque along the joint axis (Nm or N)."""
        assert self._sim is not None
        forces = np.zeros(len(self._joint_handles), dtype=np.float64)
        for idx, jh in enumerate(self._joint_handles):
            try:
                forces[idx] = float(self._sim.getJointForce(jh.handle))
            except Exception:
                forces[idx] = float("nan")
        return forces

    def _apply_torque_single(self, handle: int, tau: float) -> str:
        """Apply torque to one joint; return the CoppeliaSim API mode used."""
        assert self._sim is not None
        sim = self._sim
        if self.config.prefer_signed_target_force and hasattr(sim, "setJointTargetForce"):
            try:
                sim.setJointTargetForce(handle, float(tau), True)
                return "setJointTargetForce"
            except Exception:
                pass
        v0 = float(self.config.fallback_large_velocity_rad_s)
        mag = float(abs(tau))
        sign = 1.0 if tau >= 0.0 else -1.0
        sim.setJointTargetVelocity(handle, sign * v0)
        sim.setJointMaxForce(handle, mag)
        return "setJointTargetVelocity+setJointMaxForce"

    def apply_torque(self, tau: Sequence[float]) -> None:
        assert self._sim is not None
        tau_arr = np.asarray(tau, dtype=np.float64).reshape(-1)
        if tau_arr.shape[0] != len(self._joint_handles):
            raise ValueError(
                f"tau length {tau_arr.shape[0]} != {len(self._joint_handles)}"
            )
        modes: list[str] = []
        for jh, t in zip(self._joint_handles, tau_arr):
            modes.append(self._apply_torque_single(jh.handle, float(t)))
        self._last_torque_api_modes = modes

    def start_simulation(self) -> None:
        assert self._sim is not None
        if self.config.stepping:
            self._sim.setStepping(True)
        self._sim.startSimulation()

    def stop_simulation(self) -> None:
        if self._sim is not None:
            self._sim.stopSimulation()

    def read_simulation_state(self) -> int | None:
        if self._sim is None:
            return None
        return int(self._sim.getSimulationState())

    def step(self) -> None:
        if self._sim is not None and self.config.stepping:
            self._sim.step()
