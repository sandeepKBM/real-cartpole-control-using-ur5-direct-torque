"""
Gymnasium environment for UR5 Y-axis transport in CoppeliaSim.

Connects to a running CoppeliaSim instance via ZMQ, loads the UR5 model,
and runs a stepped torque-control loop. The policy outputs normalized
joint torques; the env scales them to physical Nm values.

CoppeliaSim must be launched separately before creating this env.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(
    0,
    str(
        REPO_ROOT
        / "ros2_ws"
        / "src"
        / "ur5_x_axis_controller_ros"
    ),
)

from ur5_x_axis_controller_ros.coppeliasim_adapter import (
    CoppeliaSimConfig,
    CoppeliaSimURAdapter,
)
from controller_core.kinematics_utils import (
    orientation_error_vec_wxyz,
    quat_normalize_wxyz,
)
from rl.baseline_controller import BaselineConfig, TransportBaselineController

DEFAULT_CONFIG = REPO_ROOT / "rl" / "config.yaml"


def _load_config(path: Path | str | None = None) -> dict:
    p = Path(path) if path else DEFAULT_CONFIG
    with open(p) as f:
        return yaml.safe_load(f)


class CoppeliaYTransportEnv(gym.Env):
    """UR5 direct-torque Y-axis transport in CoppeliaSim."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        config_path: str | Path | None = None,
        host: str | None = None,
        port: int | None = None,
        coppelia_root: str | Path | None = None,
        manage_sim: bool = False,
        env_rank: int = 0,
    ):
        super().__init__()
        cfg = _load_config(config_path)
        env_cfg = cfg["environment"]
        cop_cfg = cfg["coppeliasim"]
        self._reward_cfg = cfg["reward"]
        self._term_cfg = cfg["termination"]

        self._host = host or cop_cfg["host"]
        self._port = port or cop_cfg["port"]
        self._sim_dt = float(env_cfg["sim_dt_s"])
        self._action_repeat = int(env_cfg["action_repeat"])
        self._max_steps = int(env_cfg["max_episode_steps"])
        self._target_dy = float(env_cfg["target_dy_m"])
        self._seed_q = np.array(env_cfg["seed_q_rad"], dtype=np.float64)
        self._init_noise_std = float(env_cfg["init_noise_std_rad"])
        self._tau_max = np.array(env_cfg["tau_max_nm"], dtype=np.float64)
        self._max_qd = float(env_cfg["max_joint_velocity_radps"])
        self._action_scale = float(env_cfg.get("action_scale", 1.0))

        bl_cfg = cfg.get("baseline", {})
        self._baseline_mode = str(bl_cfg.get("mode", "off")).strip().lower()
        self._baseline_enabled = bool(bl_cfg.get("enabled", False))
        self._residual_scale = float(bl_cfg.get("residual_scale", 0.25))
        self._reset_warmup_steps = int(bl_cfg.get("reset_warmup_steps", 0))
        self._baseline: TransportBaselineController | None = None
        if self._baseline_enabled and self._baseline_mode == "residual":
            self._baseline = TransportBaselineController(
                BaselineConfig(
                    hold_kp=float(bl_cfg.get("hold_kp", 300.0)),
                    hold_kd=float(bl_cfg.get("hold_kd", 40.0)),
                    gravity_scale=float(bl_cfg.get("gravity_scale", 1.0)),
                    cart_z_kp=float(bl_cfg.get("cart_z_kp", 200.0)),
                    cart_z_kd=float(bl_cfg.get("cart_z_kd", 40.0)),
                    y_track_kp=float(bl_cfg.get("y_track_kp", 120.0)),
                    y_track_kd=float(bl_cfg.get("y_track_kd", 20.0)),
                    enable_cart_z=bool(bl_cfg.get("enable_cart_z", False)),
                    enable_y_tracking=bool(bl_cfg.get("enable_y_tracking", False)),
                )
            )

        self._coppelia_root = Path(coppelia_root) if coppelia_root else None
        self._env_rank = int(env_rank)
        self._sim_manager = None
        if manage_sim:
            if self._coppelia_root is None:
                import os
                root = os.environ.get("COPPELIA_ROOT", "")
                if not root:
                    raise ValueError("manage_sim=True requires coppelia_root")
                self._coppelia_root = Path(root)
            from rl.coppelia_sim_manager import CoppeliaSimManager

            log_dir = REPO_ROOT / "outputs" / "control_runs"
            log_dir.mkdir(parents=True, exist_ok=True)
            log_path = log_dir / f"rl_coppelia_sim_rank{self._env_rank}_p{self._port}.log"
            self._sim_manager = CoppeliaSimManager(
                self._coppelia_root, self._port, log_path=log_path
            )
            if self._env_rank > 0:
                time.sleep(float(self._env_rank) * 3.0)
            self._sim_manager.start()
        self._ur5_model_subpath = cop_cfg["ur5_model_subpath"]
        self._ee_name = cop_cfg["ee_object_name"]
        self._ee_alts = tuple(cop_cfg.get("ee_object_alternates", ()))
        tf = cop_cfg.get("task_frame", {})
        self._tf_mode = tf.get("mode", "mujoco_attachment_dummy")
        self._tf_offset = tuple(tf.get("attachment_offset_m", (0, 0, -0.2)))
        self._tf_quat = tuple(
            tf.get(
                "attachment_quat_wxyz",
                (-0.7071067811865475, 0.7071067811865475, 0, 0),
            )
        )

        # 28-dim obs: sin(q)*6 + cos(q)*6 + qd_norm*6 + ee_pos_err*3 + ori_err*3 + ee_vel*3 + target_vy*1
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(28,), dtype=np.float32
        )
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(6,), dtype=np.float32
        )

        self._adapter: CoppeliaSimURAdapter | None = None
        self._sim: Any = None
        self._client: Any = None
        self._connected = False

        self._step_count = 0
        self._prev_action = np.zeros(6, dtype=np.float64)
        self._initial_ee_pos: np.ndarray | None = None
        self._initial_ee_quat: np.ndarray | None = None
        self._target_y: float = 0.0
        self._q_hold: np.ndarray | None = None

    def _baseline_torque(
        self,
        q: np.ndarray,
        qd: np.ndarray,
        ee_pos: np.ndarray,
        ee_lin: np.ndarray,
        *,
        j_pos: np.ndarray | None = None,
    ) -> np.ndarray:
        if self._baseline is None or self._q_hold is None or self._initial_ee_pos is None:
            return np.zeros(6, dtype=np.float64)
        return self._baseline.compute(
            q=q,
            qd=qd,
            q_hold=self._q_hold,
            ee_pos=ee_pos,
            ee_lin=ee_lin,
            z_target=float(self._initial_ee_pos[2]),
            target_y=self._target_y,
            j_pos=j_pos,
            tau_limit=self._tau_max,
        )

    def _run_reset_warmup(self) -> None:
        if self._reset_warmup_steps <= 0 or self._baseline is None:
            return
        adapter = self._adapter
        sim = self._sim
        assert adapter is not None and sim is not None

        for _ in range(self._reset_warmup_steps):
            q, qd = adapter.read_joint_state()
            ee_pos, _, ee_lin, _ = adapter.read_ee_pose_twist()
            tau = self._baseline_torque(
                q, qd, np.asarray(ee_pos), np.asarray(ee_lin)
            )
            adapter.apply_torque(tau)
            sim.step()

    def _ensure_connected(self) -> None:
        """Connect to CoppeliaSim, load UR5 if needed, and resolve handles."""
        if self._connected:
            return
        import threading
        import zmq
        from coppeliasim_zmqremoteapi_client import RemoteAPIClient

        print(f"[env] connecting to CoppeliaSim at {self._host}:{self._port}...", flush=True)

        for attempt in range(20):
            result: dict[str, Any] = {}

            def _try_connect() -> None:
                try:
                    c = RemoteAPIClient(self._host, self._port)
                    s = c.require("sim")
                    s.getSimulationState()
                    c.socket.setsockopt(zmq.RCVTIMEO, 10 * 60 * 1000)
                    result["client"] = c
                    result["sim"] = s
                except Exception as e:
                    result["error"] = e

            t = threading.Thread(target=_try_connect, daemon=True)
            t.start()
            t.join(timeout=10.0)

            if t.is_alive():
                if attempt < 5 or attempt % 5 == 0:
                    print(f"[env] attempt {attempt + 1}/20: timed out (10s)", flush=True)
                time.sleep(2.0)
                continue

            if "error" in result:
                if attempt < 5 or attempt % 5 == 0:
                    print(f"[env] attempt {attempt + 1}/20: {result['error']}", flush=True)
                time.sleep(2.0)
                continue

            self._client = result["client"]
            self._sim = result["sim"]
            print(f"[env] connected on attempt {attempt + 1}", flush=True)
            break
        else:
            raise RuntimeError(
                f"Cannot connect to CoppeliaSim at {self._host}:{self._port}"
            )

        self._load_ur5_if_needed()

        cop_adapter_cfg = CoppeliaSimConfig(
            zmq_host=self._host,
            zmq_port=self._port,
            ee_object_name=self._ee_name,
            ee_object_name_alternates=self._ee_alts,
            task_frame_mode=self._tf_mode,
            task_frame_attachment_offset_m=self._tf_offset,
            task_frame_attachment_quat_wxyz=self._tf_quat,
            stepping=True,
            prefer_signed_target_force=True,
            jacobian_source="numerical",
            numerical_epsilon=1e-5,
            connect_retries=1,
        )
        self._adapter = CoppeliaSimURAdapter(cop_adapter_cfg)
        self._adapter.connect_with_existing_client(self._client)
        print("[env] adapter connected and handles resolved", flush=True)
        self._connected = True

    def _load_ur5_if_needed(self) -> None:
        sim = self._sim
        try:
            sim.getObject("/UR5")
            print("[env] UR5 already in scene", flush=True)
        except Exception:
            root = self._coppelia_root
            if root is None:
                import os
                root = Path(os.environ.get("COPPELIA_ROOT", ""))
            model_path = root / self._ur5_model_subpath
            if not model_path.exists():
                raise FileNotFoundError(f"UR5 model not found: {model_path}")
            print(f"[env] loading UR5 from {model_path}", flush=True)
            sim.loadModel(str(model_path))
            print("[env] UR5 loaded", flush=True)

    def _disconnect(self) -> None:
        self._connected = False
        self._adapter = None
        self._sim = None
        self._client = None

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed, options=options)
        self._ensure_connected()
        return self._reset_episode_safe()

    def _recover_zmq(self, context: str) -> None:
        print(f"[env] ZMQ recovery ({context}) on port {self._port}", flush=True)
        self._disconnect()
        if self._sim_manager is not None:
            self._sim_manager.restart()
        else:
            time.sleep(2.0)
        self._ensure_connected()

    def _reset_episode_safe(self) -> tuple[np.ndarray, dict[str, Any]]:
        try:
            return self._reset_episode()
        except Exception as exc:
            if not self._is_zmq_error(exc):
                raise
            self._recover_zmq("reset")
            return self._reset_episode()

    def _reset_episode(self) -> tuple[np.ndarray, dict[str, Any]]:
        sim = self._sim
        adapter = self._adapter
        assert adapter is not None and sim is not None

        try:
            state = int(sim.getSimulationState())
            if state != sim.simulation_stopped:
                sim.stopSimulation()
                for _ in range(40):
                    time.sleep(0.02)
                    try:
                        if int(sim.getSimulationState()) == sim.simulation_stopped:
                            break
                    except Exception:
                        break
        except Exception:
            pass

        adapter._prev_joint_positions_continuous = None

        q0 = self._seed_q.copy()
        if self._init_noise_std > 0:
            q0 += self.np_random.normal(0, self._init_noise_std, size=6)

        adapter.set_joint_positions(q0)
        adapter.configure_force_torque_mode()
        self._q_hold = q0.copy()

        sim.setFloatParam(sim.floatparam_simulation_time_step, self._sim_dt)
        sim.setStepping(True)
        sim.startSimulation()

        for _ in range(2):
            sim.step()

        self._run_reset_warmup()

        q, qd = adapter.read_joint_state()
        ee_pos, ee_quat, ee_lin, _ = adapter.read_ee_pose_twist()

        self._initial_ee_pos = np.array(ee_pos, dtype=np.float64)
        self._initial_ee_quat = quat_normalize_wxyz(
            np.array(ee_quat, dtype=np.float64)
        )
        self._target_y = float(self._initial_ee_pos[1]) - self._target_dy
        self._step_count = 0
        self._prev_action = np.zeros(6, dtype=np.float64)

        obs = self._build_obs(q, qd, ee_pos, ee_quat, ee_lin)
        info: dict[str, Any] = {
            "initial_ee_pos": self._initial_ee_pos.tolist(),
            "target_y": self._target_y,
        }
        return obs, info

    def _is_zmq_error(self, exc: BaseException) -> bool:
        err = str(exc).lower()
        return "zmq" in err or "cannot be accomplished" in err or "resource temporarily unavailable" in err

    def _physics_substep(self, tau: np.ndarray) -> None:
        assert self._adapter is not None and self._sim is not None
        self._adapter.apply_torque(tau)
        self._sim.step()

    def _physics_substep_safe(self, tau: np.ndarray) -> None:
        try:
            self._physics_substep(tau)
        except Exception as exc:
            if not self._is_zmq_error(exc):
                raise
            self._recover_zmq("step")
            self._physics_substep(tau)

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        assert self._adapter is not None and self._sim is not None
        adapter = self._adapter
        sim = self._sim

        action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        action = action * self._action_scale

        for _ in range(self._action_repeat):
            q, qd = adapter.read_joint_state()
            ee_pos, _, ee_lin, _ = adapter.read_ee_pose_twist()
            ee_pos_arr = np.asarray(ee_pos, dtype=np.float64)
            ee_lin_arr = np.asarray(ee_lin, dtype=np.float64)

            if self._baseline is not None:
                tau_base = self._baseline_torque(q, qd, ee_pos_arr, ee_lin_arr)
                tau_rl = action * self._residual_scale * self._tau_max
                tau = np.clip(tau_base + tau_rl, -self._tau_max, self._tau_max)
            else:
                tau = action * self._tau_max

            self._physics_substep_safe(tau)

        q, qd = adapter.read_joint_state()
        ee_pos, ee_quat, ee_lin, _ = adapter.read_ee_pose_twist()
        ee_pos = np.array(ee_pos, dtype=np.float64)
        ee_quat = np.array(ee_quat, dtype=np.float64)
        ee_lin = np.array(ee_lin, dtype=np.float64)

        obs = self._build_obs(q, qd, ee_pos, ee_quat, ee_lin)

        x_err = float(ee_pos[0] - self._initial_ee_pos[0])
        y_err = float(ee_pos[1] - self._target_y)
        z_err = float(ee_pos[2] - self._initial_ee_pos[2])
        ori_err_vec = orientation_error_vec_wxyz(self._initial_ee_quat, ee_quat)
        ori_err_mag = float(np.linalg.norm(ori_err_vec))

        y_vel = float(ee_lin[1])
        y_direction = -1.0 if self._target_dy > 0 else 1.0
        y_vel_toward_target = y_vel * y_direction

        rc = self._reward_cfg
        reward = (
            rc["y_tracking_weight"] * y_vel_toward_target
            - rc["z_hold_weight"] * abs(z_err)
            - rc["x_hold_weight"] * abs(x_err)
            - rc["orientation_weight"] * ori_err_mag
            - rc["torque_smooth_weight"]
            * float(np.sum((action - self._prev_action) ** 2))
            + rc["alive_bonus"]
        )

        self._prev_action = action.copy()
        self._step_count += 1

        tc = self._term_cfg
        terminated = False
        truncated = False
        term_reason = ""

        if abs(z_err) > tc["max_z_error_m"]:
            terminated = True
            term_reason = "z_drift"
        elif abs(x_err) > tc["max_x_error_m"]:
            terminated = True
            term_reason = "x_drift"
        elif ori_err_mag > tc["max_orientation_error_rad"]:
            terminated = True
            term_reason = "orientation"
        elif float(np.max(np.abs(qd))) > tc["max_joint_velocity_radps"]:
            terminated = True
            term_reason = "joint_velocity"

        if terminated:
            reward += rc["terminal_penalty"]

        if self._step_count >= self._max_steps:
            truncated = True

        info: dict[str, Any] = {
            "step": self._step_count,
            "x_err": x_err,
            "y_err": y_err,
            "z_err": z_err,
            "ori_err": ori_err_mag,
            "y_vel": y_vel,
            "max_qd": float(np.max(np.abs(qd))),
            "ee_pos": ee_pos.tolist(),
        }
        if term_reason:
            info["termination_reason"] = term_reason
        if self._baseline is not None:
            info["baseline_mode"] = self._baseline_mode

        return obs, float(reward), terminated, truncated, info

    def _build_obs(
        self,
        q: np.ndarray,
        qd: np.ndarray,
        ee_pos: np.ndarray,
        ee_quat: np.ndarray,
        ee_lin: np.ndarray,
    ) -> np.ndarray:
        q = np.asarray(q, dtype=np.float64)
        qd = np.asarray(qd, dtype=np.float64)
        ee_pos = np.asarray(ee_pos, dtype=np.float64)
        ee_quat = np.asarray(ee_quat, dtype=np.float64)
        ee_lin = np.asarray(ee_lin, dtype=np.float64)

        sin_q = np.sin(q[:6])
        cos_q = np.cos(q[:6])
        qd_norm = np.clip(qd[:6] / self._max_qd, -1.0, 1.0)

        pos_err = np.zeros(3, dtype=np.float64)
        if self._initial_ee_pos is not None:
            pos_err[0] = self._initial_ee_pos[0] - ee_pos[0]
            pos_err[1] = self._target_y - ee_pos[1]
            pos_err[2] = self._initial_ee_pos[2] - ee_pos[2]

        ori_err = np.zeros(3, dtype=np.float64)
        if self._initial_ee_quat is not None:
            ori_err = orientation_error_vec_wxyz(self._initial_ee_quat, ee_quat)

        control_dt = self._sim_dt * self._action_repeat
        target_vy = np.array(
            [self._target_dy / max(self._max_steps * control_dt, 0.01)],
            dtype=np.float64,
        )

        obs = np.concatenate([
            sin_q, cos_q, qd_norm,
            pos_err, ori_err, ee_lin[:3],
            target_vy,
        ]).astype(np.float32)
        return obs

    def close(self) -> None:
        if self._adapter is not None:
            try:
                self._adapter.apply_torque(np.zeros(6))
            except Exception:
                pass
        if self._sim is not None:
            try:
                self._sim.stopSimulation()
            except Exception:
                pass
        self._disconnect()
        if self._sim_manager is not None:
            try:
                self._sim_manager.stop()
            except Exception:
                pass
            self._sim_manager = None
