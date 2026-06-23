"""Joint-space impedance torque law used by the simulation-only torque lane.

This module is intentionally small and simulator-independent. It turns a
desired joint position/velocity reference into a clipped joint torque command
with optional feedforward bias. The outer-loop motion generator stays in the
runner or a higher-level controller.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def _as_vector_or_diag(name: str, value: np.ndarray, length: int) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64)
    if arr.shape == (length,):
        arr = np.diag(arr)
    if arr.shape != (length, length):
        raise ValueError(f"{name} must have shape ({length},) or ({length}, {length}); got {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains NaN/Inf")
    return arr


def _as_vector(name: str, value: np.ndarray, length: int) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64).reshape(-1)
    if arr.shape[0] != length:
        raise ValueError(f"{name} must have length {length}; got {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains NaN/Inf")
    return arr


@dataclass
class JointImpedanceConfig:
    """Per-joint torque gains and saturation limits."""

    kp_nm_per_rad: np.ndarray = field(
        default_factory=lambda: np.array([120.0, 120.0, 100.0, 28.0, 24.0, 18.0], dtype=np.float64)
    )
    kd_nm_per_rad_s: np.ndarray = field(
        default_factory=lambda: np.array([12.0, 12.0, 10.0, 3.2, 2.8, 2.0], dtype=np.float64)
    )
    tau_max_nm: np.ndarray = field(
        default_factory=lambda: np.array([15.0, 15.0, 15.0, 4.0, 4.0, 4.0], dtype=np.float64)
    )

    def validate(self) -> None:
        _ = _as_vector_or_diag("kp_nm_per_rad", self.kp_nm_per_rad, 6)
        _ = _as_vector_or_diag("kd_nm_per_rad_s", self.kd_nm_per_rad_s, 6)
        tau_max = _as_vector("tau_max_nm", self.tau_max_nm, 6)
        if np.any(tau_max <= 0.0):
            raise ValueError("tau_max_nm must be strictly positive")
        if np.any(np.diag(_as_vector_or_diag("kp_nm_per_rad", self.kp_nm_per_rad, 6)) < 0.0):
            raise ValueError("kp_nm_per_rad must be non-negative")
        if np.any(np.diag(_as_vector_or_diag("kd_nm_per_rad_s", self.kd_nm_per_rad_s, 6)) < 0.0):
            raise ValueError("kd_nm_per_rad_s must be non-negative")


@dataclass
class JointImpedanceOutput:
    tau: np.ndarray
    tau_preclip: np.ndarray
    tau_feedback: np.ndarray
    tau_feedforward: np.ndarray
    q_error: np.ndarray
    qd_error: np.ndarray
    saturated: bool


class JointImpedanceController:
    """Joint-space PD torque law with optional feedforward bias."""

    def __init__(self, config: JointImpedanceConfig | None = None) -> None:
        self.cfg = config if config is not None else JointImpedanceConfig()
        self.cfg.validate()
        self._kp = np.diag(np.asarray(self.cfg.kp_nm_per_rad, dtype=np.float64).reshape(6))
        self._kd = np.diag(np.asarray(self.cfg.kd_nm_per_rad_s, dtype=np.float64).reshape(6))
        self._tau_max = np.asarray(self.cfg.tau_max_nm, dtype=np.float64).reshape(6)

    def compute(
        self,
        q: np.ndarray,
        qd: np.ndarray,
        q_ref: np.ndarray,
        qd_ref: np.ndarray | None = None,
        tau_feedforward: np.ndarray | None = None,
    ) -> JointImpedanceOutput:
        q = _as_vector("q", q, 6)
        qd = _as_vector("qd", qd, 6)
        q_ref = _as_vector("q_ref", q_ref, 6)
        qd_ref_arr = np.zeros(6, dtype=np.float64) if qd_ref is None else _as_vector("qd_ref", qd_ref, 6)
        tau_ff = np.zeros(6, dtype=np.float64) if tau_feedforward is None else _as_vector("tau_feedforward", tau_feedforward, 6)

        q_error = q_ref - q
        qd_error = qd_ref_arr - qd
        tau_fb = self._kp @ q_error + self._kd @ qd_error
        tau_preclip = tau_fb + tau_ff
        tau = np.clip(tau_preclip, -self._tau_max, +self._tau_max)
        saturated = bool(np.any(np.abs(tau_preclip - tau) > 1e-12))
        if not np.all(np.isfinite(tau)):
            raise RuntimeError("JointImpedanceController produced a non-finite torque command")
        return JointImpedanceOutput(
            tau=tau,
            tau_preclip=tau_preclip,
            tau_feedback=tau_fb,
            tau_feedforward=tau_ff,
            q_error=q_error,
            qd_error=qd_error,
            saturated=saturated,
        )
