"""
CoppeliaSim-only torque / impedance diagnostics.

Collects per-step controller and safety signals, writes JSON summaries, and
generates plots. Intended for the ZMQ headless runner; does not touch MuJoCo.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from controller_core.x_axis_cartesian_impedance import JOINT_NAME_ORDER


PASS_MAX_TAU_FRACTION = 0.50
PASS_MAX_RATE_FRACTION = 0.50


@dataclass
class CoppeliaTorqueDiagnosticsConfig:
    enable_coppelia_torque_diagnostics: bool = False
    save_controller_logs: bool = False
    save_controller_plots: bool = False
    diagnostics_output_dir: Path = Path("outputs/control_runs/coppelia_torque_diagnostics")
    impedance_gain_scale: float = 1.0
    reference_smoothing_enabled: bool = False
    max_reference_step: float = 0.02
    max_reference_velocity: float = 0.05
    diagnostics_mode: str = "live"
    sinusoid_joint_index: int = 5
    sinusoid_amplitude_rad: float = 0.02
    sinusoid_frequency_hz: float = 0.15

    @classmethod
    def from_sources(
        cls,
        yaml_section: dict[str, Any] | None,
        cli_overrides: dict[str, Any],
    ) -> "CoppeliaTorqueDiagnosticsConfig":
        data: dict[str, Any] = {}
        if yaml_section:
            data.update(yaml_section)
        data.update({k: v for k, v in cli_overrides.items() if v is not None})
        out_dir = data.get("diagnostics_output_dir", cls.diagnostics_output_dir)
        return cls(
            enable_coppelia_torque_diagnostics=bool(
                data.get("enable_coppelia_torque_diagnostics", False)
            ),
            save_controller_logs=bool(data.get("save_controller_logs", False)),
            save_controller_plots=bool(data.get("save_controller_plots", False)),
            diagnostics_output_dir=Path(str(out_dir)),
            impedance_gain_scale=float(data.get("impedance_gain_scale", 1.0)),
            reference_smoothing_enabled=bool(data.get("reference_smoothing_enabled", False)),
            max_reference_step=float(data.get("max_reference_step", 0.02)),
            max_reference_velocity=float(data.get("max_reference_velocity", 0.05)),
            diagnostics_mode=str(data.get("diagnostics_mode", "live")),
            sinusoid_joint_index=int(data.get("sinusoid_joint_index", 5)),
            sinusoid_amplitude_rad=float(data.get("sinusoid_amplitude_rad", 0.02)),
            sinusoid_frequency_hz=float(data.get("sinusoid_frequency_hz", 0.15)),
        )


def smooth_scalar_reference(
    prev: float,
    target: float,
    *,
    dt: float,
    max_step: float,
    max_velocity: float,
) -> float:
    """Clamp reference step and velocity for diagnostic smoothing."""
    dt = max(float(dt), 1e-6)
    delta = float(target) - float(prev)
    max_delta = min(abs(max_step), abs(max_velocity) * dt)
    if abs(delta) <= max_delta:
        return float(target)
    return float(prev + math.copysign(max_delta, delta))


def smooth_vector_reference(
    prev: np.ndarray,
    target: np.ndarray,
    *,
    dt: float,
    max_step: float,
    max_velocity: float,
) -> np.ndarray:
    prev = np.asarray(prev, dtype=np.float64).reshape(-1)
    target = np.asarray(target, dtype=np.float64).reshape(-1)
    out = np.zeros_like(prev)
    for i in range(prev.shape[0]):
        out[i] = smooth_scalar_reference(
            prev[i],
            target[i],
            dt=dt,
            max_step=max_step,
            max_velocity=max_velocity,
        )
    return out


@dataclass
class _RunningStats:
    rows: list[dict[str, Any]] = field(default_factory=list)
    prev_time: float | None = None
    prev_tau_final: np.ndarray | None = None
    first_torque_clip_step: int | None = None
    first_torque_clip_joint: str | None = None
    first_rate_clip_step: int | None = None
    first_rate_clip_joint: str | None = None
    num_torque_clips: np.ndarray = field(
        default_factory=lambda: np.zeros(6, dtype=np.int64)
    )
    num_rate_clips: np.ndarray = field(
        default_factory=lambda: np.zeros(6, dtype=np.int64)
    )


class CoppeliaTorqueDiagnostics:
    """Per-run recorder for CoppeliaSim torque diagnostics."""

    def __init__(
        self,
        config: CoppeliaTorqueDiagnosticsConfig,
        *,
        joint_names: Sequence[str] = JOINT_NAME_ORDER,
        tau_limit_nm: np.ndarray,
        rate_limit_nm_per_sec: np.ndarray,
        controller_mode: str,
        prefer_signed_target_force: bool,
        joint_configuration: dict[str, Any] | None = None,
        run_label: str = "run",
    ) -> None:
        self.cfg = config
        self.joint_names = list(joint_names)
        self.tau_limit = np.asarray(tau_limit_nm, dtype=np.float64).reshape(6)
        self.rate_limit = np.asarray(rate_limit_nm_per_sec, dtype=np.float64).reshape(6)
        self.controller_mode = str(controller_mode)
        self.prefer_signed_target_force = bool(prefer_signed_target_force)
        self.joint_configuration = joint_configuration or {}
        self.run_label = str(run_label)
        self._stats = _RunningStats()
        self._q_ref_smooth: np.ndarray | None = None
        self._x_ref_smooth: float | None = None

    @property
    def active(self) -> bool:
        return bool(self.cfg.enable_coppelia_torque_diagnostics)

    def smooth_joint_reference(self, q_des: np.ndarray, dt: float) -> np.ndarray:
        q_des = np.asarray(q_des, dtype=np.float64).reshape(6)
        if not self.cfg.reference_smoothing_enabled:
            return q_des
        if self._q_ref_smooth is None:
            self._q_ref_smooth = q_des.copy()
            return q_des
        self._q_ref_smooth = smooth_vector_reference(
            self._q_ref_smooth,
            q_des,
            dt=dt,
            max_step=self.cfg.max_reference_step,
            max_velocity=self.cfg.max_reference_velocity,
        )
        return self._q_ref_smooth.copy()

    def smooth_x_reference(self, x_des: float, dt: float) -> float:
        if not self.cfg.reference_smoothing_enabled:
            return float(x_des)
        if self._x_ref_smooth is None:
            self._x_ref_smooth = float(x_des)
            return float(x_des)
        self._x_ref_smooth = smooth_scalar_reference(
            self._x_ref_smooth,
            x_des,
            dt=dt,
            max_step=self.cfg.max_reference_step,
            max_velocity=self.cfg.max_reference_velocity,
        )
        return float(self._x_ref_smooth)

    def record_step(
        self,
        *,
        step_idx: int,
        timestamp: float,
        dt: float,
        q: np.ndarray,
        qd: np.ndarray,
        q_des: np.ndarray | None,
        qd_des: np.ndarray | None,
        cart_position: float | None,
        cart_velocity: float | None,
        pole_angle: float | None,
        pole_angular_velocity: float | None,
        tau_impedance_p: np.ndarray,
        tau_impedance_d: np.ndarray,
        tau_feedforward: np.ndarray | None,
        tau_gravity: np.ndarray | None,
        tau_raw_before_safety: np.ndarray,
        tau_after_saturation: np.ndarray,
        tau_final_sent: np.ndarray,
        filter_diag: dict[str, Any] | None,
        torque_api_modes: Sequence[str],
        safety_flags: dict[str, bool],
        joint_mode_snapshot: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        q = np.asarray(q, dtype=np.float64).reshape(6)
        qd = np.asarray(qd, dtype=np.float64).reshape(6)
        q_des_arr = (
            np.asarray(q_des, dtype=np.float64).reshape(6)
            if q_des is not None
            else np.full(6, np.nan)
        )
        qd_des_arr = (
            np.asarray(qd_des, dtype=np.float64).reshape(6)
            if qd_des is not None
            else np.full(6, np.nan)
        )
        tau_p = np.asarray(tau_impedance_p, dtype=np.float64).reshape(6)
        tau_d = np.asarray(tau_impedance_d, dtype=np.float64).reshape(6)
        tau_ff = (
            np.asarray(tau_feedforward, dtype=np.float64).reshape(6)
            if tau_feedforward is not None
            else np.zeros(6)
        )
        tau_g = (
            np.asarray(tau_gravity, dtype=np.float64).reshape(6)
            if tau_gravity is not None
            else np.zeros(6)
        )
        tau_raw = np.asarray(tau_raw_before_safety, dtype=np.float64).reshape(6)
        tau_sat = np.asarray(tau_after_saturation, dtype=np.float64).reshape(6)
        tau_final = np.asarray(tau_final_sent, dtype=np.float64).reshape(6)

        q_error = q_des_arr - q
        qd_error = qd_des_arr - qd
        torque_fraction = np.abs(tau_final) / np.maximum(self.tau_limit, 1e-12)
        torque_clip = np.abs(tau_raw - tau_sat) > 1e-9
        rate_fraction = np.zeros(6, dtype=np.float64)
        rate_clipped = np.zeros(6, dtype=bool)
        if filter_diag is not None:
            rate_fraction = np.asarray(
                filter_diag.get("torque_rate_fraction", [0.0] * 6), dtype=np.float64
            ).reshape(6)
            rate_clipped = np.asarray(
                filter_diag.get("torque_rate_clipped", [False] * 6), dtype=bool
            ).reshape(6)
        elif self._stats.prev_tau_final is not None and dt > 0.0:
            rate_nm_per_sec = (tau_final - self._stats.prev_tau_final) / dt
            rate_fraction = np.abs(rate_nm_per_sec) / np.maximum(self.rate_limit, 1e-12)
            rate_clipped = rate_fraction > 1.0 + 1e-9

        has_nan_inf = not (
            np.all(np.isfinite(q))
            and np.all(np.isfinite(qd))
            and np.all(np.isfinite(tau_final))
        )

        for j in range(6):
            if torque_clip[j]:
                self._stats.num_torque_clips[j] += 1
                if self._stats.first_torque_clip_step is None:
                    self._stats.first_torque_clip_step = int(step_idx)
                    self._stats.first_torque_clip_joint = self.joint_names[j]
            if rate_clipped[j]:
                self._stats.num_rate_clips[j] += 1
                if self._stats.first_rate_clip_step is None:
                    self._stats.first_rate_clip_step = int(step_idx)
                    self._stats.first_rate_clip_joint = self.joint_names[j]

        row = {
            "step_idx": int(step_idx),
            "timestamp": float(timestamp),
            "dt": float(dt),
            "controller_mode": self.controller_mode,
            "diagnostics_mode": self.cfg.diagnostics_mode,
            "q": q.tolist(),
            "qd": qd.tolist(),
            "q_des": q_des_arr.tolist(),
            "qd_des": qd_des_arr.tolist(),
            "q_error": q_error.tolist(),
            "qd_error": qd_error.tolist(),
            "cart_position": cart_position,
            "cart_velocity": cart_velocity,
            "pole_angle": pole_angle,
            "pole_angular_velocity": pole_angular_velocity,
            "tau_impedance_P": tau_p.tolist(),
            "tau_impedance_D": tau_d.tolist(),
            "tau_feedforward": tau_ff.tolist(),
            "tau_gravity": tau_g.tolist(),
            "tau_raw_before_safety": tau_raw.tolist(),
            "tau_after_saturation": tau_sat.tolist(),
            "tau_final_sent_to_coppelia": tau_final.tolist(),
            "torque_limit_nm": self.tau_limit.tolist(),
            "torque_usage_fraction": torque_fraction.tolist(),
            "torque_rate_fraction": rate_fraction.tolist(),
            "torque_clipped": torque_clip.tolist(),
            "torque_rate_clipped": rate_clipped.tolist(),
            "joint_limit_guardrail": bool(safety_flags.get("joint_limit_guardrail", False)),
            "workspace_guardrail": bool(safety_flags.get("workspace_guardrail", False)),
            "nan_or_inf": bool(has_nan_inf),
            "coppelia_api_per_joint": list(torque_api_modes),
            "prefer_signed_target_force": self.prefer_signed_target_force,
            "joint_mode_snapshot": joint_mode_snapshot,
            "filter_diagnostics": filter_diag,
            "reference_smoothing_enabled": self.cfg.reference_smoothing_enabled,
            "impedance_gain_scale": self.cfg.impedance_gain_scale,
        }
        self._stats.rows.append(row)
        self._stats.prev_time = float(timestamp)
        self._stats.prev_tau_final = tau_final.copy()
        return row

    def build_summary(self, *, duration_s: float, pass_mode: str | None = None) -> dict[str, Any]:
        rows = self._stats.rows
        if not rows:
            return {
                "run_label": self.run_label,
                "duration_s": duration_s,
                "pass": False,
                "suspected_failure_reason": "no diagnostic rows recorded",
            }

        dt_arr = np.array([r["dt"] for r in rows], dtype=np.float64)
        q_err = np.array([r["q_error"] for r in rows], dtype=np.float64)
        qd_err = np.array([r["qd_error"] for r in rows], dtype=np.float64)
        tau_raw = np.array([r["tau_raw_before_safety"] for r in rows], dtype=np.float64)
        tau_sent = np.array([r["tau_final_sent_to_coppelia"] for r in rows], dtype=np.float64)
        tau_frac = np.array([r["torque_usage_fraction"] for r in rows], dtype=np.float64)
        rate_frac = np.array([r["torque_rate_fraction"] for r in rows], dtype=np.float64)

        cart_pos = [r["cart_position"] for r in rows if r.get("cart_position") is not None]
        cart_vel = [r["cart_velocity"] for r in rows if r.get("cart_velocity") is not None]
        pole_ang = [r["pole_angle"] for r in rows if r.get("pole_angle") is not None]
        pole_ang_vel = [
            r["pole_angular_velocity"] for r in rows if r.get("pole_angular_velocity") is not None
        ]

        per_joint_summary = {}
        for j, name in enumerate(self.joint_names):
            per_joint_summary[name] = {
                "max_abs_q_error": float(np.nanmax(np.abs(q_err[:, j]))),
                "max_abs_qd_error": float(np.nanmax(np.abs(qd_err[:, j]))),
                "max_abs_tau_raw_nm": float(np.max(np.abs(tau_raw[:, j]))),
                "max_abs_tau_sent_nm": float(np.max(np.abs(tau_sent[:, j]))),
                "max_tau_fraction": float(np.max(tau_frac[:, j])),
                "max_tau_rate_fraction": float(np.max(rate_frac[:, j])),
                "num_torque_clips": int(self._stats.num_torque_clips[j]),
                "num_torque_rate_clips": int(self._stats.num_rate_clips[j]),
            }

        max_tau_fraction = float(np.max(tau_frac))
        max_rate_fraction = float(np.max(rate_frac))
        any_nan = any(bool(r.get("nan_or_inf")) for r in rows)
        any_joint_guard = any(bool(r.get("joint_limit_guardrail")) for r in rows)
        any_workspace_guard = any(bool(r.get("workspace_guardrail")) for r in rows)
        any_torque_clip = int(np.sum(self._stats.num_torque_clips)) > 0
        any_rate_clip = int(np.sum(self._stats.num_rate_clips)) > 0

        mode = pass_mode or self.cfg.diagnostics_mode
        if mode in {"passive", "zero_torque"}:
            passed = not any_nan
            failure_reason = None if passed else "NaN/Inf during passive/zero-torque run"
        elif mode == "gain_sweep":
            passed = max_tau_fraction < PASS_MAX_TAU_FRACTION and not any_nan
            failure_reason = None if passed else "gain sweep exceeded torque usage threshold"
        else:
            passed = (
                not any_torque_clip
                and max_tau_fraction < PASS_MAX_TAU_FRACTION
                and max_rate_fraction < PASS_MAX_RATE_FRACTION
                and not any_nan
                and not any_joint_guard
                and not any_workspace_guard
            )
            reasons: list[str] = []
            if any_torque_clip:
                reasons.append("torque clipping observed")
            if max_tau_fraction >= PASS_MAX_TAU_FRACTION:
                reasons.append(f"max torque usage {max_tau_fraction:.3f} >= {PASS_MAX_TAU_FRACTION}")
            if max_rate_fraction >= PASS_MAX_RATE_FRACTION:
                reasons.append(f"max rate usage {max_rate_fraction:.3f} >= {PASS_MAX_RATE_FRACTION}")
            if any_nan:
                reasons.append("NaN/Inf in state or torque")
            if any_joint_guard:
                reasons.append("joint limit guardrail triggered")
            if any_workspace_guard:
                reasons.append("workspace guardrail triggered")
            failure_reason = None if passed else "; ".join(reasons)

        api_modes = sorted({m for r in rows for m in r.get("coppelia_api_per_joint", [])})
        return {
            "run_label": self.run_label,
            "diagnostics_mode": self.cfg.diagnostics_mode,
            "controller_mode": self.controller_mode,
            "duration_s": float(duration_s),
            "mean_dt_s": float(np.mean(dt_arr)),
            "max_dt_s": float(np.max(dt_arr)),
            "controller_frequency_hz": float(1.0 / max(np.mean(dt_arr), 1e-9)),
            "per_joint": per_joint_summary,
            "first_time_torque_limit_hit": self._stats.first_torque_clip_step,
            "first_joint_torque_limit_hit": self._stats.first_torque_clip_joint,
            "first_time_rate_limit_hit": self._stats.first_rate_clip_step,
            "first_joint_rate_limit_hit": self._stats.first_rate_clip_joint,
            "max_cart_position": float(max(cart_pos)) if cart_pos else None,
            "max_cart_velocity": float(max(abs(v) for v in cart_vel)) if cart_vel else None,
            "max_pole_angle": float(max(abs(v) for v in pole_ang)) if pole_ang else None,
            "max_pole_angular_velocity": (
                float(max(abs(v) for v in pole_ang_vel)) if pole_ang_vel else None
            ),
            "max_tau_fraction_overall": max_tau_fraction,
            "max_tau_rate_fraction_overall": max_rate_fraction,
            "coppelia_torque_api_modes_observed": api_modes,
            "prefer_signed_target_force": self.prefer_signed_target_force,
            "joint_configuration": self.joint_configuration,
            "impedance_gain_scale": self.cfg.impedance_gain_scale,
            "reference_smoothing_enabled": self.cfg.reference_smoothing_enabled,
            "pass": bool(passed),
            "suspected_failure_reason": failure_reason,
            "pass_criteria": {
                "max_tau_fraction": PASS_MAX_TAU_FRACTION,
                "max_rate_fraction": PASS_MAX_RATE_FRACTION,
                "torque_clipping_allowed": mode in {"passive", "zero_torque", "gain_sweep"},
            },
        }

    def write_logs(self, trace_path: Path, summary: dict[str, Any]) -> tuple[Path, Path]:
        out_dir = self.cfg.diagnostics_output_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        trace_out = out_dir / f"{self.run_label}.jsonl"
        summary_out = out_dir / f"{self.run_label}_summary.json"
        with trace_out.open("w", encoding="utf-8") as fp:
            for row in self._stats.rows:
                fp.write(json.dumps(row, separators=(",", ":")) + "\n")
        summary_out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return trace_out, summary_out

    def generate_plots(self, plot_prefix: Path) -> list[Path]:
        if not self._stats.rows:
            return []
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            return []

        rows = self._stats.rows
        t = np.array([r["timestamp"] for r in rows], dtype=np.float64)
        dt = np.array([r["dt"] for r in rows], dtype=np.float64)
        q = np.array([r["q"] for r in rows], dtype=np.float64)
        q_des = np.array([r["q_des"] for r in rows], dtype=np.float64)
        q_err = np.array([r["q_error"] for r in rows], dtype=np.float64)
        qd = np.array([r["qd"] for r in rows], dtype=np.float64)
        qd_des = np.array([r["qd_des"] for r in rows], dtype=np.float64)
        qd_err = np.array([r["qd_error"] for r in rows], dtype=np.float64)
        tau_raw = np.array([r["tau_raw_before_safety"] for r in rows], dtype=np.float64)
        tau_final = np.array([r["tau_final_sent_to_coppelia"] for r in rows], dtype=np.float64)
        tau_lim = np.array([r["torque_limit_nm"] for r in rows], dtype=np.float64)
        tau_frac = np.array([r["torque_usage_fraction"] for r in rows], dtype=np.float64)
        rate_frac = np.array([r["torque_rate_fraction"] for r in rows], dtype=np.float64)
        tau_p = np.array([r["tau_impedance_P"] for r in rows], dtype=np.float64)
        tau_d = np.array([r["tau_impedance_D"] for r in rows], dtype=np.float64)
        cart_pos = np.array(
            [r["cart_position"] if r.get("cart_position") is not None else np.nan for r in rows],
            dtype=np.float64,
        )
        cart_vel = np.array(
            [r["cart_velocity"] if r.get("cart_velocity") is not None else np.nan for r in rows],
            dtype=np.float64,
        )
        pole_ang = np.array(
            [r["pole_angle"] if r.get("pole_angle") is not None else np.nan for r in rows],
            dtype=np.float64,
        )
        pole_ang_vel = np.array(
            [
                r["pole_angular_velocity"]
                if r.get("pole_angular_velocity") is not None
                else np.nan
                for r in rows
            ],
            dtype=np.float64,
        )
        torque_clip = np.array([r["torque_clipped"] for r in rows], dtype=bool)
        rate_clip = np.array([r["torque_rate_clipped"] for r in rows], dtype=bool)
        joint_guard = np.array([r["joint_limit_guardrail"] for r in rows], dtype=bool)
        workspace_guard = np.array([r["workspace_guardrail"] for r in rows], dtype=bool)
        nan_flag = np.array([r["nan_or_inf"] for r in rows], dtype=bool)

        plot_prefix.parent.mkdir(parents=True, exist_ok=True)
        written: list[Path] = []

        def _save(fig: Any, name: str) -> None:
            path = plot_prefix.parent / f"{plot_prefix.name}_{name}.png"
            fig.tight_layout()
            fig.savefig(path, dpi=150)
            plt.close(fig)
            written.append(path)

        fig, axes = plt.subplots(3, 2, figsize=(12, 10), sharex=True)
        for j, name in enumerate(self.joint_names):
            ax = axes[j // 2, j % 2]
            ax.plot(t, q[:, j], label="q")
            ax.plot(t, q_des[:, j], "--", label="q_des")
            ax.set_title(name)
            ax.grid(True)
            if j == 0:
                ax.legend(loc="upper right", fontsize=8)
        _save(fig, "01_q_vs_q_des")

        fig, axes = plt.subplots(3, 2, figsize=(12, 10), sharex=True)
        for j, name in enumerate(self.joint_names):
            ax = axes[j // 2, j % 2]
            ax.plot(t, q_err[:, j])
            ax.set_title(f"{name} q_error")
            ax.grid(True)
        _save(fig, "02_q_error")

        fig, axes = plt.subplots(3, 2, figsize=(12, 10), sharex=True)
        for j, name in enumerate(self.joint_names):
            ax = axes[j // 2, j % 2]
            ax.plot(t, qd[:, j], label="qd")
            ax.plot(t, qd_des[:, j], "--", label="qd_des")
            ax.set_title(name)
            ax.grid(True)
        _save(fig, "03_qd_vs_qd_des")

        fig, axes = plt.subplots(3, 2, figsize=(12, 10), sharex=True)
        for j, name in enumerate(self.joint_names):
            ax = axes[j // 2, j % 2]
            ax.plot(t, qd_err[:, j])
            ax.set_title(f"{name} qd_error")
            ax.grid(True)
        _save(fig, "04_qd_error")

        fig, axes = plt.subplots(3, 2, figsize=(12, 10), sharex=True)
        for j, name in enumerate(self.joint_names):
            ax = axes[j // 2, j % 2]
            ax.plot(t, tau_raw[:, j], label="tau_raw")
            ax.plot(t, tau_lim[:, j], "--", label="limit")
            ax.plot(t, -tau_lim[:, j], "--")
            ax.set_title(name)
            ax.grid(True)
        _save(fig, "05_tau_raw_vs_limit")

        fig, axes = plt.subplots(3, 2, figsize=(12, 10), sharex=True)
        for j, name in enumerate(self.joint_names):
            ax = axes[j // 2, j % 2]
            ax.plot(t, tau_final[:, j], label="tau_sent")
            ax.plot(t, tau_lim[:, j], "--", label="limit")
            ax.plot(t, -tau_lim[:, j], "--")
            ax.set_title(name)
            ax.grid(True)
        _save(fig, "06_tau_sent_vs_limit")

        fig, ax = plt.subplots(figsize=(10, 5))
        for j, name in enumerate(self.joint_names):
            ax.plot(t, tau_frac[:, j], label=name)
        ax.axhline(PASS_MAX_TAU_FRACTION, color="r", linestyle="--", label="pass threshold")
        ax.set_ylabel("torque usage fraction")
        ax.legend(fontsize=8, ncol=2)
        ax.grid(True)
        _save(fig, "07_torque_usage_fraction")

        fig, ax = plt.subplots(figsize=(10, 5))
        for j, name in enumerate(self.joint_names):
            ax.plot(t, rate_frac[:, j], label=name)
        ax.axhline(PASS_MAX_RATE_FRACTION, color="r", linestyle="--", label="pass threshold")
        ax.legend(fontsize=8, ncol=2)
        ax.grid(True)
        _save(fig, "08_torque_rate_usage_fraction")

        fig, axes = plt.subplots(3, 2, figsize=(12, 10), sharex=True)
        for j, name in enumerate(self.joint_names):
            ax = axes[j // 2, j % 2]
            ax.plot(t, tau_p[:, j], label="P")
            ax.plot(t, tau_d[:, j], label="D")
            ax.set_title(name)
            ax.grid(True)
        _save(fig, "09_tau_P_vs_D")

        fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
        if np.any(np.isfinite(cart_pos)):
            axes[0].plot(t, cart_pos)
            axes[0].set_ylabel("cart position (m)")
        else:
            axes[0].text(0.5, 0.5, "cart state not available", ha="center", va="center")
        if np.any(np.isfinite(cart_vel)):
            axes[1].plot(t, cart_vel)
            axes[1].set_ylabel("cart velocity (m/s)")
        else:
            axes[1].text(0.5, 0.5, "cart velocity not available", ha="center", va="center")
        axes[1].set_xlabel("time (s)")
        for ax in axes:
            ax.grid(True)
        _save(fig, "10_cart_state")

        fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
        if np.any(np.isfinite(pole_ang)):
            axes[0].plot(t, pole_ang)
            axes[0].set_ylabel("pole angle (rad)")
        else:
            axes[0].text(0.5, 0.5, "pole angle not available", ha="center", va="center")
        if np.any(np.isfinite(pole_ang_vel)):
            axes[1].plot(t, pole_ang_vel)
            axes[1].set_ylabel("pole angular velocity (rad/s)")
        else:
            axes[1].text(0.5, 0.5, "pole rate not available", ha="center", va="center")
        axes[1].set_xlabel("time (s)")
        for ax in axes:
            ax.grid(True)
        _save(fig, "11_pole_state")

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(t, torque_clip.any(axis=1).astype(float), label="torque_clip")
        ax.plot(t, rate_clip.any(axis=1).astype(float), label="rate_clip")
        ax.plot(t, joint_guard.astype(float), label="joint_guard")
        ax.plot(t, workspace_guard.astype(float), label="workspace_guard")
        ax.plot(t, nan_flag.astype(float), label="nan_inf")
        ax.set_ylim(-0.1, 1.1)
        ax.legend(loc="upper right")
        ax.grid(True)
        _save(fig, "12_safety_flags")

        fig, axes = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
        axes[0].plot(t, dt)
        axes[0].set_ylabel("dt (s)")
        axes[1].plot(t, 1.0 / np.maximum(dt, 1e-9))
        axes[1].set_ylabel("control freq (Hz)")
        axes[1].set_xlabel("time (s)")
        for ax in axes:
            ax.grid(True)
        _save(fig, "13_dt_and_frequency")

        return written
