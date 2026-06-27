"""
Model-based baseline torques for RL warm-start (residual policy).

Reuses the same hold PD + MuJoCo gravity + Cartesian Z feedback that
stabilized the model-based Y-transport path. RL actions become deltas
on top of this baseline instead of raw torques from a random init.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rl.gravity_utils import build_mujoco_gravity_estimator, gravity_feedforward


@dataclass
class BaselineConfig:
    hold_kp: float = 300.0
    hold_kd: float = 40.0
    gravity_scale: float = 1.0
    cart_z_kp: float = 200.0
    cart_z_kd: float = 40.0
    y_track_kp: float = 120.0
    y_track_kd: float = 20.0
    enable_y_tracking: bool = True
    enable_cart_z: bool = True


class TransportBaselineController:
    """Joint hold + gravity + optional Cartesian Z/Y tracking."""

    def __init__(self, cfg: BaselineConfig | None = None) -> None:
        self.cfg = cfg or BaselineConfig()
        self._gravity_est = build_mujoco_gravity_estimator()

    def compute(
        self,
        q: np.ndarray,
        qd: np.ndarray,
        q_hold: np.ndarray,
        ee_pos: np.ndarray,
        ee_lin: np.ndarray,
        z_target: float,
        target_y: float,
        j_pos: np.ndarray | None,
        tau_limit: np.ndarray,
    ) -> np.ndarray:
        cfg = self.cfg
        q = np.asarray(q, dtype=np.float64).reshape(6)
        qd = np.asarray(qd, dtype=np.float64).reshape(6)
        q_hold = np.asarray(q_hold, dtype=np.float64).reshape(6)
        ee_pos = np.asarray(ee_pos, dtype=np.float64).reshape(3)
        ee_lin = np.asarray(ee_lin, dtype=np.float64).reshape(3)
        tau_limit = np.asarray(tau_limit, dtype=np.float64).reshape(6)

        tau = cfg.hold_kp * (q_hold - q) - cfg.hold_kd * qd

        gravity = gravity_feedforward(
            self._gravity_est, q, qd, scale=cfg.gravity_scale
        )
        if gravity is not None:
            tau = tau + gravity

        if j_pos is not None:
            j_pos = np.asarray(j_pos, dtype=np.float64).reshape(3, 6)
            if cfg.enable_cart_z:
                z_err = float(z_target) - float(ee_pos[2])
                z_vel = float(ee_lin[2])
                f_z = cfg.cart_z_kp * z_err - cfg.cart_z_kd * z_vel
                tau = tau + j_pos[2, :] * f_z
            if cfg.enable_y_tracking:
                y_err = float(target_y) - float(ee_pos[1])
                y_vel = float(ee_lin[1])
                f_y = cfg.y_track_kp * y_err - cfg.y_track_kd * y_vel
                tau = tau + j_pos[1, :] * f_y

        return np.clip(tau, -tau_limit, tau_limit)
