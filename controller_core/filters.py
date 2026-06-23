"""Low-pass and rate limiting for joint torque commands."""

from __future__ import annotations

import numpy as np


class TorqueCommandFilter:
    """First-order low-pass + per-joint slew rate limit on torques (Nm)."""

    def __init__(
        self,
        num_joints: int,
        lowpass_alpha: float,
        rate_limit_nm_per_sec: np.ndarray,
    ) -> None:
        self._n = int(num_joints)
        self._alpha = float(np.clip(lowpass_alpha, 1e-6, 1.0))
        self._rate = np.asarray(rate_limit_nm_per_sec, dtype=np.float64).reshape(-1)
        if self._rate.shape[0] == 1:
            self._rate = np.full(self._n, float(self._rate[0]), dtype=np.float64)
        if self._rate.shape[0] != self._n:
            raise ValueError(f"rate_limit length must be 1 or {self._n}")
        self._y = np.zeros(self._n, dtype=np.float64)
        self._initialized = False

    def reset(self) -> None:
        self._y[:] = 0.0
        self._initialized = False

    def apply(self, tau_raw: np.ndarray, dt: float) -> np.ndarray:
        tau_raw = np.asarray(tau_raw, dtype=np.float64).reshape(-1)
        if tau_raw.shape[0] != self._n:
            raise ValueError(f"tau_raw length {tau_raw.shape[0]} != {self._n}")
        dt = max(float(dt), 1e-6)
        if not self._initialized:
            self._y = tau_raw.copy()
            self._initialized = True
        # Low-pass toward new command.
        x = self._alpha * tau_raw + (1.0 - self._alpha) * self._y
        # Rate limit delta from previous filtered output.
        delta_max = self._rate * dt
        dx = np.clip(x - self._y, -delta_max, +delta_max)
        self._y = self._y + dx
        return self._y.copy()
