#!/usr/bin/env python3
"""Simulation-only MuJoCo UR5e torque-control experiments.

This runner stays inside MuJoCo. It never imports hardware, RTDE, or URScript
code. The goal is to validate the torque-actuated UR5e MJCF, reuse the
simulator-independent controller_core torque laws, and write clear traces for
small experiments.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import mujoco
import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from controller_core.logging_utils import JsonlTraceWriter, json_dumps_safe  # noqa: E402
from controller_core.kinematics_utils import orientation_error_vec_wxyz, rotmat_to_quat  # noqa: E402
from mujoco_ur5e_tools import get_compiled_ur5e_torque_model_diagnostics  # noqa: E402
from mujoco_ur5e_tools import validate_ur5e_torque_xml_source_tree  # noqa: E402
from simulation.ur5e_mujoco_torque import (  # noqa: E402
    DEFAULT_OUTPUT_DIR,
    AsymmetricCoulombFrictionConfig,
    apply_start_q,
    asymmetric_coulomb_backdrive_torque,
    build_initial_state_and_adapter,
    build_mujoco_state,
    compute_joint_limit_proximity,
    accel_duration_displacement,
    compute_reward_terms,
    load_model,
    resolve_start_q,
    write_trace_plot,
    x_profile_accel,
    x_profile_target,
)
from transport_metrics import compute_valid_move_hold_metrics, controller_gain_summary, summarize_move_hold_trace, summarize_residual_torque_trace, summarize_transport_trace  # noqa: E402


MODE_CHOICES = (
    "model-load",
    "zero-gravity",
    "zero-torque-gravity",
    "gravity-comp-hold",
    "gravity-comp-hold-long",
    "single-joint-pulse",
    "constant-small-torque",
    "sinusoidal-torque",
    "impedance-hold",
    "residual-impedance-hold",
    "controller-rollout",
    "x-transport-minjerk",
    "safety-clipping",
)

_HOLD_MODES = {"gravity-comp-hold", "gravity-comp-hold-long", "impedance-hold", "residual-impedance-hold"}
# Profiles needing move_duration_s (same as min_jerk_move_hold) and, for the
# accel_duration_* pair, --target-accel instead of --target-x-delta -- see
# simulation/ur5e_mujoco_torque.py::x_profile_target's docstring comments.
_MOVE_DURATION_PROFILES = {"min_jerk_move_hold", "accel_duration_triangular", "accel_duration_scurve"}
_ACCEL_DURATION_PROFILES = {"accel_duration_triangular", "accel_duration_scurve"}
_GRAVITY_COMP_MODES = {"gravity-comp-hold", "gravity-comp-hold-long", "impedance-hold", "residual-impedance-hold", "x-transport-minjerk"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", required=True, choices=MODE_CHOICES)
    p.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "config" / "ur5e_mujoco_torque.yaml",
        help="MuJoCo torque config YAML.",
    )
    p.add_argument(
        "--scene",
        type=Path,
        default=None,
        help="Override the scene XML path from the config.",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Base output directory. Default comes from the YAML config.",
    )
    p.add_argument("--duration", type=float, default=2.0)
    p.add_argument(
        "--start-q-rad",
        nargs=6,
        type=float,
        default=None,
        metavar=("Q1", "Q2", "Q3", "Q4", "Q5", "Q6"),
        help="Optional six-joint start pose in radians. Overrides config start_q/home_qpos when provided.",
    )
    p.add_argument(
        "--controller-kind",
        choices=("torque_qp", "impedance", "hard_constraint_qp"),
        default=None,
        help="Controller used for controller-based experiments.",
    )
    p.add_argument("--target-x-delta", type=float, default=0.01)
    p.add_argument("--transport-axis-index", type=int, default=0)
    p.add_argument(
        "--trajectory-profile",
        choices=("step", "ramp", "min_jerk", "min_jerk_move_hold", "accel_duration_triangular", "accel_duration_scurve"),
        default=None,
        help=(
            "Time profile for the world-X target during controller-rollout experiments. "
            "accel_duration_triangular/accel_duration_scurve take --target-accel (peak m/s^2) "
            "instead of --target-x-delta -- displacement becomes an OUTPUT, computed from "
            "accel and --move-duration."
        ),
    )
    p.add_argument(
        "--move-duration",
        type=float,
        default=None,
        help="Move duration in seconds for min_jerk_move_hold / accel_duration_* trajectories.",
    )
    p.add_argument(
        "--target-accel",
        type=float,
        default=None,
        help="Peak acceleration in m/s^2, required when --trajectory-profile is accel_duration_*.",
    )
    p.add_argument(
        "--gravity-mode",
        choices=("raw", "gravity_comp"),
        default=None,
        help="Torque application mode for controller-rollout experiments.",
    )
    p.add_argument(
        "--gravity-source",
        choices=("mujoco_qfrc", "pinocchio"),
        default=None,
        help="Gravity-compensation source (default mujoco_qfrc; overrides mujoco.gravity_source).",
    )
    p.add_argument(
        "--coriolis-feedforward",
        action="store_true",
        help="Add C(q,qd)qd feedforward on top of gravity compensation (default off; overrides mujoco.coriolis_feedforward).",
    )
    p.add_argument(
        "--asymmetric-coulomb-friction",
        action="store_true",
        help=(
            "Enable opt-in PLANT-side extra Coulomb friction, asymmetric in the direction of "
            "mechanical power flow through the joint (Clochiatti et al. 2024, UR5e-specific; "
            "see simulation.ur5e_mujoco_torque.AsymmetricCoulombFrictionConfig). Default off; "
            "overrides mujoco.asymmetric_coulomb_friction.enabled. Can only turn this ON via "
            "CLI, never off (same pattern as --coriolis-feedforward)."
        ),
    )
    p.add_argument("--joint-index", type=int, default=0)
    p.add_argument("--torque-nm", type=float, default=1.0)
    p.add_argument("--torque-amp-nm", type=float, default=1.0)
    p.add_argument("--torque-freq-hz", type=float, default=1.0)
    p.add_argument("--pulse-duration-s", type=float, default=0.15)
    p.add_argument(
        "--torque-limit-scale",
        type=float,
        default=1.0,
        help="Scale the per-joint torque limits before clipping. Default is 1.0.",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-plot", action="store_true")
    p.add_argument(
        "--q-noise-std-rad",
        type=float,
        default=0.0,
        help=(
            "controller-rollout/x-transport-minjerk only: Gaussian noise std (rad) "
            "added to q before it reaches the controller (joint-encoder noise proxy). "
            "The true q used for physics stepping and trace logging is unaffected -- "
            "this only perturbs what the control law sees. Default 0.0 = no noise, "
            "byte-identical to before this flag existed."
        ),
    )
    p.add_argument(
        "--qd-noise-std-radps",
        type=float,
        default=0.0,
        help="Same as --q-noise-std-rad but for qd (rad/s velocity-estimate noise proxy).",
    )
    p.add_argument(
        "--torque-noise-std-nm",
        type=float,
        default=0.0,
        help=(
            "controller-rollout/x-transport-minjerk only: Gaussian noise std (Nm) added "
            "to the final torque actually applied to the physics (actuator-imprecision "
            "proxy). Reflected in the trace's own 'tau' field; 'tau_controller' in the "
            "trace still reflects the controller's clean, pre-noise output."
        ),
    )
    p.add_argument(
        "--telemetry-duplicate-prob",
        type=float,
        default=0.0,
        help=(
            "controller-rollout/x-transport-minjerk only: per-cycle probability [0,1] that "
            "the controller sees a FROZEN repeat of whatever it was delivered last cycle "
            "instead of the fresh true state (RTDE duplicate-frame proxy -- real UR5e "
            "hardware showed ~17%% isolated single-cycle duplicate tcp_pose/q reads in some "
            "2026-08-02 direct_torque runs, 0%% in others; see the guard/telemetry-gap-bridge "
            "work from that session). Independent Bernoulli trial each cycle (matches the "
            "real evidence's own 'isolated, evenly spread' pattern -- occasional back-to-back "
            "duplicates happen by chance at higher probabilities, same as real hardware, not "
            "specially modeled). Physics (mj_step) and trace ground truth are UNAFFECTED --  "
            "only what the controller is fed changes, same proxy pattern as --q-noise-std-rad. "
            "Composes with --q-noise-std-rad/--qd-noise-std-radps: duplicate-or-fresh is "
            "decided first, then sensor noise is added on top of whichever was selected, "
            "matching encoder noise applying to whatever raw sample was actually received. "
            "Default 0.0 = off, identical to today's behavior."
        ),
    )
    p.add_argument(
        "--friction-multiplier",
        type=float,
        default=1.0,
        help=(
            "controller-rollout/x-transport-minjerk only: uniform multiplier applied to "
            "every joint's dof_frictionloss AFTER the model loads (default 1.0 = today's "
            "calibrated values, byte-identical). Deliberate STRESS-TEST tool, not another "
            "calibration attempt: sample values above 1.0 (e.g. via a driver script varying "
            "this per run) to validate the controller against friction WORSE than anything "
            "measured on real hardware so far -- if it stays safe there, real hardware (which "
            "should fall inside that envelope, not outside it) has margin. Does not touch "
            "dof_damping (the viscous term) -- frictionloss is the dominant, velocity-"
            "independent term the MJCF's own class-default comment already documents as the "
            "one that reproduces the real hold-phase signature; scaling damping too would "
            "conflate two different physical effects in one knob."
        ),
    )
    p.add_argument(
        "--noise-seed",
        type=int,
        default=None,
        help="Seed for the noise RNG (separate from --seed). Defaults to --seed if unset.",
    )
    p.add_argument(
        "--enable-tcp-accel-guard",
        action="store_true",
        help=(
            "Opt-in diagnostic/guard, default OFF (zero effect on any existing run's "
            "output when unset). Ports hardware/safety.py's CartesianMoveMonitor "
            "(TCP speed/acceleration finite-difference guard -- the real-hardware safety "
            "check that has no sim-side equivalent) into this per-step loop, reusing that "
            "class via a local import (only imported when this flag is set, so this "
            "module's 'never imports hardware code' invariant holds for every "
            "existing/default run). When tripped, ends the run with "
            "termination_reason='tcp_accel_guard: <CartesianMoveMonitor reason>', exactly "
            "like any other adapter.safety_monitor trip. See "
            "docs/status/sim_tcp_accel_guard_2026-08-01.md."
        ),
    )
    p.add_argument(
        "--tcp-accel-guard-noise-robust",
        dest="tcp_accel_guard_noise_robust",
        action="store_true",
        help=(
            "Only meaningful with --enable-tcp-accel-guard. Applies "
            "hardware.safety.NOISE_ROBUST_GUARD_OVERRIDES (the exact preset validated "
            "on real hardware to avoid single-cycle finite-difference noise spikes) "
            "before any individual override flag below. Named --tcp-accel-guard-* here "
            "since this tool already has other --*-noise-* flags with a different meaning "
            "(sensor-noise injection; see --q-noise-std-rad)."
        ),
    )
    p.add_argument("--max-tcp-accel-mps2", type=float, default=None, help="Override CartesianMoveLimits.max_tcp_accel_mps2 (class default 0.5).")
    p.add_argument("--max-tcp-speed-mps", type=float, default=None, help="Override CartesianMoveLimits.max_tcp_speed_mps (class default 0.05).")
    p.add_argument("--accel-gap-cycles", type=int, default=None, help="Override CartesianMoveLimits.accel_gap_cycles (class default 1).")
    p.add_argument("--speed-lowpass-alpha", type=float, default=None, help="Override CartesianMoveLimits.speed_lowpass_alpha (class default 1.0).")
    p.add_argument("--accel-max-consecutive-violations", type=int, default=None, help="Override CartesianMoveLimits.accel_max_consecutive_violations (class default 1).")
    p.add_argument("--accel-hard-multiple", type=float, default=None, help="Override CartesianMoveLimits.accel_hard_multiple (class default 5.0).")
    p.add_argument("--speed-max-consecutive-violations", type=int, default=None, help="Override CartesianMoveLimits.speed_max_consecutive_violations (class default 1).")
    p.add_argument("--speed-hard-multiple", type=float, default=None, help="Override CartesianMoveLimits.speed_hard_multiple (class default 5.0).")
    return p.parse_args()


def _resolve_tcp_accel_guard_limits(args: argparse.Namespace):
    """Build a hardware.safety.CartesianMoveLimits from CLI overrides. Only called
    when --enable-tcp-accel-guard is set -- the local import keeps this module free of
    any hardware/RTDE dependency for every default/existing run (see that flag's help
    text). Same merge convention as tools/ur5e_direct_torque_x_transport.py's
    resolve_move_limit_overrides: --tcp-accel-guard-noise-robust's preset is applied
    first, then any explicit individual override flag wins for that field."""
    from hardware.safety import CartesianMoveLimits, NOISE_ROBUST_GUARD_OVERRIDES

    overrides: dict[str, float | int] = {}
    if bool(args.tcp_accel_guard_noise_robust):
        overrides.update(NOISE_ROBUST_GUARD_OVERRIDES)
    if args.max_tcp_accel_mps2 is not None:
        overrides["max_tcp_accel_mps2"] = float(args.max_tcp_accel_mps2)
    if args.max_tcp_speed_mps is not None:
        overrides["max_tcp_speed_mps"] = float(args.max_tcp_speed_mps)
    if args.accel_gap_cycles is not None:
        overrides["accel_gap_cycles"] = int(args.accel_gap_cycles)
    if args.speed_lowpass_alpha is not None:
        overrides["speed_lowpass_alpha"] = float(args.speed_lowpass_alpha)
    if args.accel_max_consecutive_violations is not None:
        overrides["accel_max_consecutive_violations"] = int(args.accel_max_consecutive_violations)
    if args.accel_hard_multiple is not None:
        overrides["accel_hard_multiple"] = float(args.accel_hard_multiple)
    if args.speed_max_consecutive_violations is not None:
        overrides["speed_max_consecutive_violations"] = int(args.speed_max_consecutive_violations)
    if args.speed_hard_multiple is not None:
        overrides["speed_hard_multiple"] = float(args.speed_hard_multiple)
    return CartesianMoveLimits(**overrides)


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fp:
        return yaml.safe_load(fp)


def _resolve_path(path: str | Path, *, base: Path = REPO_ROOT) -> Path:
    p = Path(path)
    return p if p.is_absolute() else (base / p).resolve()


def _make_run_dir(base_dir: Path, mode: str, controller_kind: str | None) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    ctrl = f"_{controller_kind}" if controller_kind else ""
    run_dir = base_dir / f"{mode}{ctrl}_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _choose_controller_kind(mode: str, explicit: str | None) -> str:
    if explicit is not None:
        return explicit
    if mode in ("gravity-comp-hold", "gravity-comp-hold-long"):
        return "zero_torque"
    if mode in ("impedance-hold", "residual-impedance-hold", "x-transport-minjerk"):
        return "impedance"
    return "torque_qp"


def _choose_gravity_mode(mode: str, explicit: str | None, mujoco_cfg: dict[str, Any]) -> str:
    if explicit is not None:
        return explicit
    if mode in ("gravity-comp-hold", "gravity-comp-hold-long", "impedance-hold", "residual-impedance-hold", "x-transport-minjerk"):
        return "gravity_comp"
    legacy = mujoco_cfg.get("gravity_mode")
    if legacy in ("raw", "gravity_comp"):
        return str(legacy)
    if bool(mujoco_cfg.get("use_gravity_comp", False)):
        return "gravity_comp"
    return "raw"


def _choose_gravity_source(explicit: str | None, mujoco_cfg: dict[str, Any]) -> str:
    if explicit is not None:
        return explicit
    configured = mujoco_cfg.get("gravity_source")
    if configured in ("mujoco_qfrc", "pinocchio"):
        return str(configured)
    return "mujoco_qfrc"


def _choose_asymmetric_friction_cfg(cli_flag: bool, mujoco_cfg: dict[str, Any]) -> AsymmetricCoulombFrictionConfig:
    """Same override pattern as _choose_coriolis_feedforward: the YAML
    mujoco.asymmetric_coulomb_friction block (if any) sets the baseline; the
    CLI flag can only force it ON, never off."""
    cfg = AsymmetricCoulombFrictionConfig.from_mujoco_yaml_section(mujoco_cfg)
    if cli_flag:
        cfg.enabled = True
    cfg.validate()
    return cfg


def _choose_coriolis_feedforward(cli_flag: bool, mujoco_cfg: dict[str, Any]) -> bool:
    if cli_flag:
        return True
    return bool(mujoco_cfg.get("coriolis_feedforward", False))


def _choose_trajectory_profile(mode: str, explicit: str | None, mujoco_cfg: dict[str, Any]) -> str:
    if explicit is not None:
        return explicit
    if mode in ("gravity-comp-hold", "gravity-comp-hold-long", "impedance-hold", "residual-impedance-hold"):
        return "step"
    if mode == "x-transport-minjerk":
        return "min_jerk"
    legacy = mujoco_cfg.get("trajectory_profile")
    if legacy in ("step", "ramp", "min_jerk"):
        return str(legacy)
    return "step"


def _resolve_move_duration(mujoco_cfg: dict[str, Any], explicit: float | None) -> tuple[float | None, str]:
    if explicit is not None:
        return float(explicit), "cli"
    legacy = mujoco_cfg.get("move_duration_s", mujoco_cfg.get("move_duration"))
    if legacy is not None:
        return float(legacy), "config"
    return None, "unset"


def _direct_torque_profile(mode: str, args: argparse.Namespace, t: float) -> np.ndarray:
    tau = np.zeros(6, dtype=np.float64)
    idx = int(np.clip(args.joint_index, 0, 5))
    if mode == "zero-gravity":
        return tau
    if mode == "single-joint-pulse":
        tau[idx] = float(args.torque_nm) if t <= float(args.pulse_duration_s) else 0.0
        return tau
    if mode == "constant-small-torque":
        tau[idx] = float(args.torque_nm)
        return tau
    if mode == "sinusoidal-torque":
        tau[idx] = float(args.torque_amp_nm) * math.sin(2.0 * math.pi * float(args.torque_freq_hz) * t)
        return tau
    if mode == "safety-clipping":
        tau[idx] = float(args.torque_nm)
        tau[(idx + 1) % 6] = -float(args.torque_nm)
        return tau
    raise ValueError(f"Unsupported direct torque mode: {mode!r}")


def _write_diagnostics_plot(trace_rows: list[dict[str, Any]], output_path: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    t = np.array([float(row["time_s"]) for row in trace_rows], dtype=np.float64)
    sat = np.array([float(row.get("torque_saturation_fraction", 0.0)) for row in trace_rows], dtype=np.float64)
    clip = np.array([float(row.get("torque_clip_fraction", 0.0)) for row in trace_rows], dtype=np.float64)
    prox = np.array([float(row.get("joint_limit_min_fraction", 0.0)) for row in trace_rows], dtype=np.float64)
    effort = np.array([float(row.get("control_effort_l2", 0.0)) for row in trace_rows], dtype=np.float64)
    if t.size == 0:
        raise ValueError("trace_rows is empty")

    fig, axes = plt.subplots(4, 1, figsize=(11, 10), sharex=True)
    axes[0].plot(t, sat, color="tab:orange")
    axes[0].set_ylabel("tau/limit")
    axes[0].set_title("Torque saturation")

    axes[1].plot(t, clip, color="tab:red")
    axes[1].set_ylabel("clip frac")
    axes[1].set_title("Torque clipping")

    axes[2].plot(t, prox, color="tab:green")
    axes[2].set_ylabel("min prox")
    axes[2].set_title("Joint-limit margin fraction")

    axes[3].plot(t, effort, color="tab:blue")
    axes[3].set_ylabel("L2 effort")
    axes[3].set_xlabel("time [s]")
    axes[3].set_title("Control effort proxy")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def _transport_trace_summary(
    trace_rows: list[dict[str, Any]],
    *,
    start_ee: np.ndarray,
    transport_axis_index: int,
) -> dict[str, Any]:
    return summarize_transport_trace(
        trace_rows,
        initial_ee_pos=np.asarray(start_ee, dtype=np.float64).reshape(3),
        transport_axis_index=int(transport_axis_index),
    )


def _run_zero_torque_gravity(
    *,
    args: argparse.Namespace,
    summary: dict[str, Any],
    model: mujoco.MjModel,
    data: mujoco.MjData,
    site_id: int,
    joint_ids: list[int],
    run_dir: Path,
    mujoco_cfg: dict[str, Any],
    reward_cfg: dict[str, Any],
    controller_kind: str,
) -> int:
    """Run the explicit zero-torque gravity anti-cheating probe."""
    del controller_kind, mujoco_cfg
    mujoco.mj_forward(model, data)
    dt = float(model.opt.timestep)
    steps = max(1, int(np.ceil(float(args.duration) / max(dt, 1e-9))))
    trace_rows: list[dict[str, Any]] = []
    trace_path = run_dir / "trace.jsonl"
    start_state = build_mujoco_state(
        model,
        data,
        site_id=site_id,
        joint_ids=joint_ids,
        time_s=float(data.time),
        dt_s=dt,
        target_x=float(data.site_xpos[site_id][0]),
        target_x_vel=0.0,
        target_ee_pos=np.asarray(data.site_xpos[site_id], dtype=np.float64).copy(),
        target_ee_vel=np.zeros(3, dtype=np.float64),
        reference_quat=rotmat_to_quat(np.asarray(data.site_xmat[site_id], dtype=np.float64).reshape(3, 3)),
        hold_current_pose=False,
        transport_axis_index=0,
        gravity_compensation=False,
    )
    start_q = np.asarray(start_state.q, dtype=np.float64).copy()
    start_qd = np.asarray(start_state.qd, dtype=np.float64).copy()
    start_ee = np.asarray(start_state.ee_pos, dtype=np.float64).copy()
    max_abs_q_delta = 0.0
    max_abs_qd = 0.0
    max_abs_ee_delta = 0.0
    max_abs_actuator_force = 0.0
    max_abs_qfrc_actuator = 0.0
    max_abs_qfrc_bias = 0.0
    trace_plot_path = run_dir / "trace_states.png"
    diag_plot_path = run_dir / "trace_diagnostics.png"

    with JsonlTraceWriter(trace_path) as trace_writer:
        for step_idx in range(steps):
            data.ctrl[:6] = 0.0
            mujoco.mj_step(model, data)
            post_state = build_mujoco_state(
                model,
                data,
                site_id=site_id,
                joint_ids=joint_ids,
                time_s=float(data.time),
                dt_s=dt,
                target_x=float(start_state.target_x),
                target_x_vel=0.0,
                target_ee_pos=np.asarray(start_state.target_ee_pos, dtype=np.float64).copy()
                if start_state.target_ee_pos is not None
                else np.asarray(start_ee, dtype=np.float64).copy(),
                target_ee_vel=np.zeros(3, dtype=np.float64),
                reference_quat=np.asarray(start_state.reference_quat, dtype=np.float64).copy()
                if start_state.reference_quat is not None
                else np.asarray(start_state.ee_quat, dtype=np.float64).copy(),
                hold_current_pose=False,
                transport_axis_index=0,
                gravity_compensation=False,
            )
            q_delta = np.asarray(post_state.q, dtype=np.float64) - start_q
            qd_abs = np.abs(np.asarray(post_state.qd, dtype=np.float64))
            ee_delta = np.asarray(post_state.ee_pos, dtype=np.float64) - start_ee
            actuator_force = np.asarray(data.actuator_force[:6], dtype=np.float64)
            qfrc_actuator = np.asarray(data.qfrc_actuator[:6], dtype=np.float64)
            qfrc_bias = np.asarray(data.qfrc_bias[:6], dtype=np.float64)
            max_abs_q_delta = max(max_abs_q_delta, float(np.max(np.abs(q_delta))))
            max_abs_qd = max(max_abs_qd, float(np.max(qd_abs)))
            max_abs_ee_delta = max(max_abs_ee_delta, float(np.linalg.norm(ee_delta)))
            max_abs_actuator_force = max(max_abs_actuator_force, float(np.max(np.abs(actuator_force))))
            max_abs_qfrc_actuator = max(max_abs_qfrc_actuator, float(np.max(np.abs(qfrc_actuator))))
            max_abs_qfrc_bias = max(max_abs_qfrc_bias, float(np.max(np.abs(qfrc_bias))))
            reward_terms = compute_reward_terms(
                state=post_state,
                tau=np.zeros(6, dtype=np.float64),
                tau_prev=np.zeros(6, dtype=np.float64),
                reward_cfg=reward_cfg,
                axis_idx=0,
            )
            row = {
                "step": int(step_idx),
                "time_s": float(data.time),
                "dt_s": dt,
                "q": post_state.q.tolist(),
                "qd": post_state.qd.tolist(),
                "ee_pos": post_state.ee_pos.tolist(),
                "ee_quat": post_state.ee_quat.tolist(),
                "ee_lin_vel": post_state.ee_lin_vel.tolist(),
                "ee_ang_vel": post_state.ee_ang_vel.tolist(),
                "target_x": float(start_state.target_x),
                "target_x_vel": 0.0,
                "target_ee_pos": np.asarray(start_state.target_ee_pos, dtype=np.float64).tolist()
                if start_state.target_ee_pos is not None
                else np.asarray(start_ee, dtype=np.float64).tolist(),
                "target_ee_vel": np.zeros(3, dtype=np.float64).tolist(),
                "actuator_force": actuator_force.tolist(),
                "qfrc_actuator": qfrc_actuator.tolist(),
                "qfrc_bias": qfrc_bias.tolist(),
                "tau_controller": [0.0] * 6,
                "tau_controller_clipped": [0.0] * 6,
                "tau_controller_saturated": [False] * 6,
                "tau_controller_clip_fraction": 0.0,
                "controller_torque_clip_fraction": 0.0,
                "tau_gravity": [0.0] * 6,
                "tau_applied": [0.0] * 6,
                "tau_applied_clipped": [0.0] * 6,
                "tau_raw": [0.0] * 6,
                "tau_filtered": [0.0] * 6,
                "tau": [0.0] * 6,
                "torque_saturation_fraction": 0.0,
                "torque_clip_fraction": 0.0,
                "tau_applied_clip_fraction": 0.0,
                "applied_torque_clip_fraction": 0.0,
                "joint_limit_min_fraction": float(
                    min(compute_joint_limit_proximity(model, post_state.q, joint_ids).values(), default=0.0)
                ),
                "x_error": float(start_state.target_x - post_state.ee_pos[0]),
                "orientation_error_norm": float(
                    np.linalg.norm(
                        orientation_error_vec_wxyz(
                            np.asarray(start_state.reference_quat, dtype=np.float64).copy()
                            if start_state.reference_quat is not None
                            else np.asarray(start_state.ee_quat, dtype=np.float64).copy(),
                            np.asarray(post_state.ee_quat, dtype=np.float64).copy(),
                        )
                    )
                ),
                "reward": reward_terms["reward"],
                "reward_terms": reward_terms,
                "safety_ok": True,
                "safety_reason": "",
                "controller_kind": "zero_torque_gravity",
                "termination_reason": "",
                "control_effort_l2": 0.0,
                "control_energy_proxy": 0.0,
                "q_delta_from_start": q_delta.tolist(),
                "ee_delta_from_start": ee_delta.tolist(),
                "gravity_mode": "raw",
                "gravity_mode_used": "raw",
                "gravity_compensation_active": False,
                "raw_mode_used": True,
                "trajectory_profile": "hold",
            }
            trace_rows.append(row)
            trace_writer.write_row(row)

    suspicious_reasons: list[str] = []
    if max_abs_q_delta < 1.0e-3:
        suspicious_reasons.append("max joint displacement under zero torque stayed below 1e-3 rad")
    if max_abs_ee_delta < 1.0e-3:
        suspicious_reasons.append("end-effector displacement under zero torque stayed below 1e-3 m")
    if max_abs_qd < 1.0e-3:
        suspicious_reasons.append("joint speeds under zero torque stayed below 1e-3 rad/s")

    suspicious = bool(suspicious_reasons)
    termination_reason = "suspicious_zero_torque_hold" if suspicious else "gravity_motion_observed"
    if suspicious:
        summary["failure_reason"] = "; ".join(
            [
                "suspicious_zero_torque_hold",
                "possible hidden position servo, gravity compensation, excessive damping, frozen joints, equality constraints, disabled gravity, or a controller still active",
            ]
        )
    if trace_rows and not args.no_plot:
        write_trace_plot(trace_rows, trace_plot_path)
        _write_diagnostics_plot(trace_rows, diag_plot_path)

    if trace_rows:
        summary.update(
            {
                "success": not suspicious,
                "termination_reason": termination_reason,
                "steps": int(len(trace_rows)),
                "sim_time_s": float(data.time),
                "dt_s": dt,
                "initial_q": start_q.tolist(),
                "initial_qd": start_qd.tolist(),
                "initial_ee_pos": start_ee.tolist(),
                "final_q": trace_rows[-1]["q"],
                "final_qd": trace_rows[-1]["qd"],
                "final_ee_pos": trace_rows[-1]["ee_pos"],
                "final_ee_quat": trace_rows[-1]["ee_quat"],
                "max_abs_q_delta_from_start": max_abs_q_delta,
                "max_abs_qd": max_abs_qd,
                "max_abs_ee_delta_from_start": max_abs_ee_delta,
                "max_abs_actuator_force": max_abs_actuator_force,
                "max_abs_qfrc_actuator": max_abs_qfrc_actuator,
                "max_abs_qfrc_bias": max_abs_qfrc_bias,
                "max_abs_q_rad": float(max_abs_q_delta + float(np.max(np.abs(start_q)))),
                "max_abs_qd_radps": max_abs_qd,
                "max_abs_tau_controller_nm": 0.0,
                "max_abs_tau_gravity_nm": 0.0,
                "max_abs_tau_applied_nm": 0.0,
                "mean_abs_tau_controller_nm": 0.0,
                "mean_abs_tau_applied_nm": 0.0,
                "max_abs_x_error_m": float(max((abs(float(row.get("x_error", 0.0))) for row in trace_rows), default=0.0)),
                "max_abs_orientation_error_rad": float(max((row.get("orientation_error_norm", 0.0) for row in trace_rows), default=0.0)),
                "achieved_x_delta_m": float(trace_rows[-1]["ee_pos"][0] - start_ee[0]),
                "suspicious_zero_torque_hold": suspicious,
                "suspicious_zero_torque_reasons": suspicious_reasons,
                "gravity_mode": "raw",
                "trajectory_profile": "hold",
                "trace_path": str(trace_path),
                "plot_path": str(trace_plot_path if trace_plot_path.exists() else ""),
                "diagnostics_plot_path": str(diag_plot_path if diag_plot_path.exists() else ""),
                "source_validation": summary.get("source_validation", {}),
                "compiled_model_diagnostics": summary.get("compiled_model_diagnostics", {}),
                "torque_limit_scale": 1.0,
                "max_abs_tau_nm": 0.0,
                "mean_abs_tau_nm": 0.0,
                "torque_saturation_percentage": 0.0,
                "clipping_count": 0,
            }
        )
        q_all = np.asarray([r.get("q", [0.0] * 6) for r in trace_rows], dtype=np.float64)
        qd_all = np.asarray([r.get("qd", [0.0] * 6) for r in trace_rows], dtype=np.float64)
        joint_limit_min_fraction = float(min((r.get("joint_limit_min_fraction", 1.0) for r in trace_rows), default=1.0))
        transport_summary = _transport_trace_summary(trace_rows, start_ee=start_ee, transport_axis_index=0)
        final_ee = np.asarray(transport_summary.get("final_ee_pos", trace_rows[-1]["ee_pos"]), dtype=np.float64).reshape(3)
        summary.update(
            {
                **transport_summary,
                "final_ee_error_norm_m": float(np.linalg.norm(final_ee - start_ee)),
                "max_abs_q_rad": float(np.max(np.abs(q_all))),
                "max_abs_qd_radps": float(np.max(np.abs(qd_all))),
                "velocity_guard_ok": bool(np.max(np.abs(qd_all)) <= 3.0 + 1e-9),
                "joint_limit_min_fraction": joint_limit_min_fraction,
                "joint_limit_guard_ok": bool(joint_limit_min_fraction > 0.0),
                "target_x_delta": 0.0,
                "duration_s": float(args.duration),
                "transport_axis_index": 0,
            }
        )
        summary.update(summarize_residual_torque_trace(trace_rows))

    summary_path = run_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0 if summary["success"] else 1


def run() -> int:
    args = parse_args()
    np.random.seed(int(args.seed))
    noise_seed = int(args.noise_seed) if args.noise_seed is not None else int(args.seed)
    noise_rng = np.random.default_rng(noise_seed)
    q_noise_std = max(float(args.q_noise_std_rad), 0.0)
    qd_noise_std = max(float(args.qd_noise_std_radps), 0.0)
    torque_noise_std = max(float(args.torque_noise_std_nm), 0.0)
    telemetry_duplicate_prob = min(max(float(args.telemetry_duplicate_prob), 0.0), 1.0)
    last_delivered_controller_state = None

    cfg = _load_yaml(args.config)
    mujoco_cfg = cfg["mujoco"]
    ctrl_cfg = cfg["controller"]
    reward_cfg = cfg.get("reward", {})
    scene_xml = _resolve_path(args.scene if args.scene is not None else mujoco_cfg["scene_xml"])
    output_dir = _resolve_path(args.output_dir if args.output_dir is not None else mujoco_cfg["output_dir"])
    controller_kind = _choose_controller_kind(args.mode, args.controller_kind)
    gravity_mode = _choose_gravity_mode(args.mode, args.gravity_mode, mujoco_cfg)
    gravity_source = _choose_gravity_source(args.gravity_source, mujoco_cfg)
    coriolis_feedforward = _choose_coriolis_feedforward(bool(args.coriolis_feedforward), mujoco_cfg)
    asym_friction_cfg = _choose_asymmetric_friction_cfg(bool(args.asymmetric_coulomb_friction), mujoco_cfg)
    trajectory_profile = _choose_trajectory_profile(args.mode, args.trajectory_profile, mujoco_cfg)
    move_duration_s, move_duration_source = _resolve_move_duration(mujoco_cfg, args.move_duration)
    run_dir = _make_run_dir(output_dir, args.mode, controller_kind if args.mode != "model-load" else None)

    summary: dict[str, Any] = {
        "mode": args.mode,
        "controller_kind": controller_kind if args.mode != "model-load" else None,
        "scene_xml": str(scene_xml),
        "config_path": str(args.config),
        "output_dir": str(run_dir),
        "gravity_mode": gravity_mode,
        "gravity_source": gravity_source,
        "coriolis_feedforward": coriolis_feedforward,
        "trajectory_profile": trajectory_profile,
        "move_duration_s": float(move_duration_s) if move_duration_s is not None else None,
        "move_duration_source": move_duration_source,
        "hold_duration_s": None,
        "success": False,
        "failure_reason": "",
    }

    if trajectory_profile in _MOVE_DURATION_PROFILES:
        if move_duration_s is None:
            summary["failure_reason"] = f"move_duration_required_for_{trajectory_profile}"
            summary_path = run_dir / "summary.json"
            summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
            print(json.dumps(summary, indent=2))
            return 2
        if float(move_duration_s) <= 0.0 or float(move_duration_s) >= float(args.duration):
            summary["failure_reason"] = "move_duration_must_be_positive_and_less_than_duration"
            summary["hold_duration_s"] = float(args.duration) - float(move_duration_s)
            summary["total_duration_s"] = float(args.duration)
            summary_path = run_dir / "summary.json"
            summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
            print(json.dumps(summary, indent=2))
            return 2
    if trajectory_profile in _ACCEL_DURATION_PROFILES and args.target_accel is None:
        summary["failure_reason"] = f"target_accel_required_for_{trajectory_profile}"
        summary_path = run_dir / "summary.json"
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps(summary, indent=2))
        return 2

    try:
        model, data, site_id, joint_ids, actuator_ids = load_model(scene_xml)
        friction_multiplier = max(float(args.friction_multiplier), 0.0)
        if friction_multiplier != 1.0:
            # Stress-test proxy (default 1.0 = no-op) -- see
            # --friction-multiplier's help text. Scales dof_frictionloss for
            # every joint uniformly, applied once after load, before any
            # rollout step -- affects real physics (mj_step), unlike the
            # controller-facing noise/duplicate proxies above, since the
            # whole point is testing against a worse PLANT, not a worse
            # sensor.
            model.dof_frictionloss[:] = model.dof_frictionloss * friction_multiplier
        # Reused by every build_mujoco_state()/build_initial_state_and_adapter()
        # call below with gravity_compensation active, instead of letting
        # compute_gravity_torque allocate a brand new mujoco.MjData (plus a
        # mj_forward/mj_inverse pass on it) every single call -- this is the
        # single-run engine every sweep driver subprocesses, so it runs once
        # per simulated step across the whole sweep infrastructure. Measured
        # ~11-12x per-call speedup; see docs/status/performance_audit_2026-07-29.md.
        gravity_scratch = mujoco.MjData(model)
    except Exception as exc:
        summary["failure_reason"] = f"model_load_failed: {exc}"
        summary_path = run_dir / "summary.json"
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps(summary, indent=2))
        return 2

    summary.update(
        {
            "model_nq": int(model.nq),
            "model_nv": int(model.nv),
            "model_nu": int(model.nu),
            "joint_ids": list(joint_ids),
            "actuator_ids": list(actuator_ids),
            "site_id": int(site_id),
            "joint_names": [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid) for jid in joint_ids],
            "actuator_names": [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, aid) for aid in actuator_ids],
        }
    )
    summary["source_validation"] = validate_ur5e_torque_xml_source_tree(scene_xml)
    summary["compiled_model_diagnostics"] = get_compiled_ur5e_torque_model_diagnostics(model, site_name="attachment_site")
    summary["true_torque_verified"] = True
    summary.update(controller_gain_summary(ctrl_cfg))
    start_q, start_q_source = resolve_start_q(mujoco_cfg, args.start_q_rad)
    if start_q is not None:
        apply_start_q(model, data, start_q)
    summary["start_q_source"] = start_q_source
    summary["start_q_rad"] = start_q.tolist() if start_q is not None else None

    if args.mode == "model-load":
        summary["success"] = True
        summary_path = run_dir / "summary.json"
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps(summary, indent=2))
        return 0
    if args.mode in ("zero-gravity", "zero-torque-gravity"):
        summary["controller_kind"] = "zero_torque_gravity"
        return _run_zero_torque_gravity(
            args=args,
            summary=summary,
            model=model,
            data=data,
            site_id=site_id,
            joint_ids=joint_ids,
            run_dir=run_dir,
            mujoco_cfg=mujoco_cfg,
            reward_cfg=reward_cfg,
            controller_kind=controller_kind,
        )

    if args.mode in _HOLD_MODES:
        target_x_delta = 0.0
    elif trajectory_profile in _ACCEL_DURATION_PROFILES:
        # Displacement is an OUTPUT for these profiles, not an input --
        # computed once upfront via the same closed form x_profile_target()
        # uses internally, so downstream scoring/logging (which all expect
        # target_x_delta as a known quantity) is unaffected.
        target_x_delta = accel_duration_displacement(trajectory_profile, float(args.target_accel), float(move_duration_s))
        summary["target_accel_mps2"] = float(args.target_accel)
    else:
        target_x_delta = float(args.target_x_delta)
    summary["target_x_delta"] = float(target_x_delta)
    summary["duration_s"] = float(args.duration)
    summary["total_duration_s"] = float(args.duration)
    summary["transport_axis_index"] = int(args.transport_axis_index)
    if trajectory_profile in _MOVE_DURATION_PROFILES:
        summary["hold_duration_s"] = float(args.duration) - float(move_duration_s if move_duration_s is not None else 0.0)
    state0, adapter = build_initial_state_and_adapter(
        model,
        data,
        site_id,
        joint_ids,
        controller_cfg=ctrl_cfg,
        transport_axis_index=int(args.transport_axis_index),
        target_x_delta=target_x_delta,
        controller_kind=controller_kind,
        force_hold_current_pose=bool(args.mode in _HOLD_MODES),
        gravity_mode=gravity_mode,
        gravity_source=gravity_source,
        coriolis_feedforward=coriolis_feedforward,
        torque_limit_scale=float(args.torque_limit_scale),
        gravity_scratch_data=gravity_scratch,
    )
    dt = float(model.opt.timestep)
    steps = max(1, int(np.ceil(float(args.duration) / max(dt, 1e-9))))
    trace_rows: list[dict[str, Any]] = []
    trace_path = run_dir / "trace.jsonl"
    termination_reason = "duration_complete"
    safety_violation = False
    clip_observed = False

    # Opt-in TCP-accel/speed guard (see --enable-tcp-accel-guard's help text).
    # move_monitor stays None -- and every downstream branch below is a no-op --
    # unless the flag is set, so this whole block has zero effect on any
    # existing/default run (no import, no new summary/trace keys, identical
    # control flow).
    move_monitor = None
    if bool(args.enable_tcp_accel_guard):
        from hardware.safety import CartesianMoveMonitor

        move_limits = _resolve_tcp_accel_guard_limits(args)
        move_monitor = CartesianMoveMonitor(move_limits)
        tcp_pose0 = np.concatenate(
            [np.asarray(state0.ee_pos, dtype=np.float64).reshape(3), np.zeros(3, dtype=np.float64)]
        )
        move_monitor.set_start(tcp_pose0, move_axis_index=int(args.transport_axis_index))
        summary["tcp_accel_guard_enabled"] = True
        summary["tcp_accel_guard_limits"] = dataclasses.asdict(move_limits)
    total_effort = 0.0
    total_energy_proxy = 0.0
    max_abs_tau_controller = 0.0
    max_abs_tau_gravity = 0.0
    max_abs_tau_applied = 0.0
    mean_abs_tau_controller_accum = 0.0
    mean_abs_tau_applied_accum = 0.0
    max_abs_x_error = 0.0
    max_abs_orientation_error = 0.0
    failure_time_s: float | None = None
    target_ee_pos = (
        np.asarray(state0.target_ee_pos, dtype=np.float64).reshape(3)
        if state0.target_ee_pos is not None
        else np.array([state0.target_x, state0.ee_pos[1], state0.ee_pos[2]], dtype=np.float64)
    )

    with JsonlTraceWriter(trace_path) as trace_writer:
        for step_idx in range(steps):
            target_x_now, target_x_vel_now = x_profile_target(
                trajectory_profile,
                float(state0.ee_pos[0]),
                float(target_x_delta),
                float(data.time),
                float(args.duration),
                move_duration_s=move_duration_s if trajectory_profile in _MOVE_DURATION_PROFILES else None,
                target_accel_mps2=float(args.target_accel) if trajectory_profile in _ACCEL_DURATION_PROFILES else None,
            )
            target_x_accel_now = x_profile_accel(
                trajectory_profile,
                float(target_x_delta),
                float(data.time),
                float(args.duration),
                move_duration_s=move_duration_s if trajectory_profile in _MOVE_DURATION_PROFILES else None,
                target_accel_mps2=float(args.target_accel) if trajectory_profile in _ACCEL_DURATION_PROFILES else None,
            )
            target_ee_pos = np.array([target_x_now, state0.ee_pos[1], state0.ee_pos[2]], dtype=np.float64)
            target_ee_vel = np.array([target_x_vel_now, 0.0, 0.0], dtype=np.float64)
            prev_tau = (
                np.asarray(adapter._prev_tau, dtype=np.float64).reshape(6).copy()
                if getattr(adapter, "_prev_tau", None) is not None
                else np.zeros(6, dtype=np.float64)
            )
            pre_state = build_mujoco_state(
                model,
                data,
                site_id=site_id,
                joint_ids=joint_ids,
                time_s=float(data.time),
                dt_s=dt,
                target_x=float(target_x_now),
                target_x_vel=float(target_x_vel_now),
                target_x_accel=float(target_x_accel_now),
                target_axis=float(target_ee_pos[int(args.transport_axis_index)]),
                target_axis_vel=float(target_ee_vel[int(args.transport_axis_index)]),
                target_ee_pos=target_ee_pos,
                target_ee_vel=target_ee_vel,
                reference_quat=state0.reference_quat,
                hold_current_pose=state0.hold_current_pose,
                transport_axis_index=int(args.transport_axis_index),
                gravity_compensation=bool(gravity_mode == "gravity_comp"),
                gravity_scratch_data=gravity_scratch,
            )

            # Sensor-noise proxy (default off = identical to pre_state): the
            # controller sees q/qd perturbed by Gaussian noise, but the TRUE
            # pre_state (used for mj_step physics and trace ground truth) is
            # untouched. Only the joint-space terms that read st["q"]/st["qd"]
            # directly (tau_damping, tau_posture, and J@qd-derived
            # ee_lin_vel/ee_ang_vel) are affected -- ee_pos/ee_quat here still
            # come from the true state, so the task-space x/y/z/orientation
            # error terms are not perturbed by this flag. See --q-noise-std-rad
            # / --qd-noise-std-radps help text.
            # Telemetry-duplicate proxy (default off = identical to pre_state): with
            # probability telemetry_duplicate_prob, the controller is fed whatever it
            # was delivered LAST cycle (fresh or itself already a duplicate -- matches
            # real RTDE just replaying its last buffered sample) instead of this
            # cycle's true fresh state. Physics (mj_step) and trace ground truth are
            # UNAFFECTED, same proxy pattern as the sensor-noise block below -- only
            # what the controller is fed changes. See --telemetry-duplicate-prob help
            # text. Decided BEFORE sensor noise so noise applies to whichever raw
            # sample (fresh or duplicate) was actually "received" this cycle, matching
            # real encoder noise applying on top of whatever RTDE actually returned.
            if telemetry_duplicate_prob > 0.0 and last_delivered_controller_state is not None:
                is_duplicate = bool(noise_rng.random() < telemetry_duplicate_prob)
            else:
                is_duplicate = False
            controller_state = last_delivered_controller_state if is_duplicate else pre_state

            # Sensor-noise proxy (default off = identical to pre_state): the
            # controller sees q/qd perturbed by Gaussian noise, but the TRUE
            # pre_state (used for mj_step physics and trace ground truth) is
            # untouched. Only the joint-space terms that read st["q"]/st["qd"]
            # directly (tau_damping, tau_posture, and J@qd-derived
            # ee_lin_vel/ee_ang_vel) are affected -- ee_pos/ee_quat here still
            # come from the true state, so the task-space x/y/z/orientation
            # error terms are not perturbed by this flag. See --q-noise-std-rad
            # / --qd-noise-std-radps help text.
            if q_noise_std > 0.0 or qd_noise_std > 0.0:
                controller_state = dataclasses.replace(
                    controller_state,
                    q=controller_state.q + (noise_rng.normal(0.0, q_noise_std, size=6) if q_noise_std > 0.0 else 0.0),
                    qd=controller_state.qd + (noise_rng.normal(0.0, qd_noise_std, size=6) if qd_noise_std > 0.0 else 0.0),
                )
            last_delivered_controller_state = controller_state

            if args.mode in ("single-joint-pulse", "constant-small-torque", "sinusoidal-torque", "safety-clipping"):
                tau_raw = _direct_torque_profile(args.mode, args, float(data.time))
                tau, diag = adapter.apply_torque_components(
                    state=controller_state,
                    tau_controller=tau_raw,
                    controller_diag={"controller_kind": "direct_torque_profile", "controller_output": {"mode": args.mode, "tau_raw": tau_raw.tolist()}},
                )
            elif args.mode == "gravity-comp-hold":
                tau, diag = adapter.step(state=controller_state)
            else:
                tau, diag = adapter.step(state=controller_state)

            # Actuator-noise proxy (default off = tau unchanged): perturbs the
            # torque actually applied to the physics below. diag's own
            # tau_controller/tau_applied breakdown still reflects the
            # controller's clean, pre-noise output -- only this local `tau`
            # (which becomes data.ctrl and the trace's own "tau" field) is
            # affected, so the trace can show both the clean controller
            # decision and the noisy applied result.
            if torque_noise_std > 0.0:
                tau = np.asarray(tau, dtype=np.float64).reshape(6) + noise_rng.normal(0.0, torque_noise_std, size=6)

            # PLANT-side extra friction (opt-in, default zeros -- see
            # simulation.ur5e_mujoco_torque.AsymmetricCoulombFrictionConfig):
            # injected as an extra generalized force via qfrc_applied, i.e. it
            # affects the physics regardless of what the controller commands,
            # exactly like the MJCF's own frictionloss/damping already do --
            # this is a plant realism addition, not a controller term. Set
            # every step (even when disabled, where it is all-zeros) since
            # nothing else in this script's step loop touches qfrc_applied.
            extra_friction_tau = asymmetric_coulomb_backdrive_torque(pre_state.qd, tau, asym_friction_cfg)
            data.qfrc_applied[:6] = extra_friction_tau

            clip_observed = clip_observed or bool(np.any(np.asarray(diag.get("tau_saturated", []), dtype=bool)))
            if not bool(diag.get("safety_ok", True)):
                safety_violation = True
                termination_reason = str(diag.get("safety_reason", "safety_violation")) or "safety_violation"
                data.ctrl[:6] = np.asarray(tau, dtype=np.float64).reshape(6)
                mujoco.mj_step(model, data)
                post_state = build_mujoco_state(
                    model,
                    data,
                    site_id=site_id,
                    joint_ids=joint_ids,
                    time_s=float(data.time),
                    dt_s=dt,
                    target_x=float(target_x_now),
                    target_x_vel=float(target_x_vel_now),
                    target_x_accel=float(target_x_accel_now),
                    target_axis=float(target_ee_pos[int(args.transport_axis_index)]),
                    target_axis_vel=float(target_ee_vel[int(args.transport_axis_index)]),
                    target_ee_pos=target_ee_pos,
                    target_ee_vel=target_ee_vel,
                    reference_quat=state0.reference_quat,
                    hold_current_pose=state0.hold_current_pose,
                    transport_axis_index=int(args.transport_axis_index),
                    gravity_compensation=bool(gravity_mode == "gravity_comp"),
                    gravity_scratch_data=gravity_scratch,
                )
                joint_prox = compute_joint_limit_proximity(model, post_state.q, joint_ids)
                joint_limit_min_fraction = float(min(joint_prox.values())) if joint_prox else 0.0
                reward_terms = compute_reward_terms(
                    state=post_state,
                    tau=tau,
                    tau_prev=prev_tau,
                    reward_cfg=reward_cfg,
                    axis_idx=int(args.transport_axis_index),
                )
                control_effort = float(np.linalg.norm(tau))
                energy_proxy = float(np.sum(np.abs(tau * post_state.qd)))
                tau_controller = np.asarray(diag.get("tau_controller", diag.get("tau_raw", tau)), dtype=np.float64).reshape(6)
                tau_gravity = np.asarray(diag.get("tau_gravity", np.zeros(6, dtype=np.float64)), dtype=np.float64).reshape(6)
                tau_applied = np.asarray(diag.get("tau_applied", tau_controller + tau_gravity), dtype=np.float64).reshape(6)
                # Controller-internal diagnostics already computed inside
                # controller.compute() (see CartesianImpedanceOutput) but
                # previously discarded every cycle here -- mirrors the fix
                # applied to hardware/direct_torque_transport.py. Absent
                # (None) for controller kinds that don't compute them (e.g.
                # zero_torque, direct_torque_profile).
                controller_output = diag.get("controller_output") or {}
                total_effort += control_effort * dt
                total_energy_proxy += energy_proxy * dt
                row = {
                    "step": int(step_idx),
                    "time_s": float(data.time),
                    "dt_s": dt,
                    "q": post_state.q.tolist(),
                    "qd": post_state.qd.tolist(),
                    "ee_pos": post_state.ee_pos.tolist(),
                    "ee_quat": post_state.ee_quat.tolist(),
                    "ee_lin_vel": post_state.ee_lin_vel.tolist(),
                    "ee_ang_vel": post_state.ee_ang_vel.tolist(),
                    "target_x": float(target_x_now),
                    "target_x_vel": float(target_x_vel_now),
                    "target_x_accel": float(target_x_accel_now),
                    "target_ee_pos": target_ee_pos.tolist(),
                    "target_ee_vel": target_ee_vel.tolist(),
                    "actuator_force": np.asarray(data.actuator_force[:6], dtype=np.float64).tolist(),
                    "qfrc_actuator": np.asarray(data.qfrc_actuator[:6], dtype=np.float64).tolist(),
                    "qfrc_bias": np.asarray(data.qfrc_bias[:6], dtype=np.float64).tolist(),
                    "tau_controller": tau_controller.tolist(),
                    "tau_controller_clipped": np.asarray(diag.get("tau_controller_clipped", tau_controller), dtype=np.float64).tolist(),
                    "tau_controller_saturated": np.asarray(diag.get("tau_controller_saturated", [False] * 6), dtype=bool).tolist(),
                    "tau_controller_clip_fraction": float(diag.get("tau_controller_clip_fraction", diag.get("controller_torque_clip_fraction", 0.0))),
                    "controller_torque_clip_fraction": float(diag.get("controller_torque_clip_fraction", diag.get("tau_controller_clip_fraction", 0.0))),
                    "tau_gravity": tau_gravity.tolist(),
                    "tau_applied": tau_applied.tolist(),
                    "tau_applied_clipped": np.asarray(diag.get("tau_applied_clipped", tau), dtype=np.float64).tolist(),
                    "tau_raw": np.asarray(diag.get("tau_raw", tau), dtype=np.float64).tolist(),
                    "tau_filtered": np.asarray(diag.get("tau_filtered", tau), dtype=np.float64).tolist(),
                    "tau": np.asarray(tau, dtype=np.float64).tolist(),
                    "torque_saturation_fraction": float(diag.get("torque_saturation_fraction", 0.0)),
                    "torque_clip_fraction": float(diag.get("torque_clip_fraction", 0.0)),
                    "tau_applied_clip_fraction": float(diag.get("tau_applied_clip_fraction", diag.get("torque_clip_fraction", 0.0))),
                    "applied_torque_clip_fraction": float(diag.get("applied_torque_clip_fraction", diag.get("torque_clip_fraction", 0.0))),
                    "joint_limit_min_fraction": joint_limit_min_fraction,
                    "x_error": float(diag.get("axis_error", 0.0)),
                    "orientation_error_norm": float(diag.get("orientation_error_norm", 0.0)),
                    "reward": reward_terms["reward"],
                    "reward_terms": reward_terms,
                    "safety_ok": bool(diag.get("safety_ok", True)),
                    "safety_reason": str(diag.get("safety_reason", "")),
                    "controller_kind": str(diag.get("controller_kind", "")),
                    "jacobian_cond": controller_output.get("jacobian_cond"),
                    "singular_scale": controller_output.get("singular_scale"),
                    "task_scale": controller_output.get("task_scale"),
                    "task_backtrack_scale": controller_output.get("task_backtrack_scale"),
                    "task_backtrack_iters": controller_output.get("task_backtrack_iters"),
                    "task_feasible": controller_output.get("task_feasible"),
                    "y_error": controller_output.get("y_error"),
                    "z_error": controller_output.get("z_error"),
                    "wrench": controller_output.get("wrench"),
                    "tau_preclip": controller_output.get("tau_preclip"),
                    "tau_task": controller_output.get("tau_task"),
                    "tau_posture": controller_output.get("tau_posture"),
                    "tau_damping": controller_output.get("tau_damping"),
                    "tau_friction_ff": controller_output.get("tau_friction_ff"),
                    "friction_z": controller_output.get("friction_z"),
                    "wrench_accel_ff": controller_output.get("wrench_accel_ff"),
                    "acceleration_feedforward_active": controller_output.get("acceleration_feedforward_active"),
                    "jacobian_pre_step": pre_state.jacobian.tolist(),
                    "termination_reason": termination_reason,
                    "control_effort_l2": control_effort,
                    "control_energy_proxy": energy_proxy,
                    "gravity_mode": str(diag.get("gravity_mode", gravity_mode)),
                    "gravity_mode_used": str(diag.get("gravity_mode_used", diag.get("gravity_mode", gravity_mode))),
                    "gravity_compensation_active": bool(diag.get("gravity_compensation_active", gravity_mode == "gravity_comp")),
                    "raw_mode_used": bool(diag.get("raw_mode_used", gravity_mode == "raw")),
                    "trajectory_profile": trajectory_profile,
                }
                trace_rows.append(row)
                trace_writer.write_row(row)
                break

            data.ctrl[:6] = np.asarray(tau, dtype=np.float64).reshape(6)
            mujoco.mj_step(model, data)
            post_state = build_mujoco_state(
                model,
                data,
                site_id=site_id,
                joint_ids=joint_ids,
                time_s=float(data.time),
                dt_s=dt,
                target_x=float(target_x_now),
                target_x_vel=float(target_x_vel_now),
                target_x_accel=float(target_x_accel_now),
                target_axis=float(target_ee_pos[int(args.transport_axis_index)]),
                target_axis_vel=float(target_ee_vel[int(args.transport_axis_index)]),
                target_ee_pos=target_ee_pos,
                target_ee_vel=target_ee_vel,
                reference_quat=state0.reference_quat,
                hold_current_pose=state0.hold_current_pose,
                transport_axis_index=int(args.transport_axis_index),
                gravity_compensation=bool(gravity_mode == "gravity_comp"),
                gravity_scratch_data=gravity_scratch,
            )
            post_orient_ref = (
                np.asarray(post_state.reference_quat, dtype=np.float64).reshape(4)
                if post_state.reference_quat is not None
                else np.asarray(post_state.ee_quat, dtype=np.float64).reshape(4)
            )
            post_orient_err = float(
                np.linalg.norm(
                    orientation_error_vec_wxyz(post_orient_ref, np.asarray(post_state.ee_quat, dtype=np.float64).reshape(4))
                )
            )
            post_axis_err = float(
                (post_state.target_x - post_state.ee_pos[0])
                if int(args.transport_axis_index) == 0
                else (
                    (post_state.target_axis if post_state.target_axis is not None else post_state.target_x)
                    - post_state.ee_pos[int(args.transport_axis_index)]
                )
            )
            post_axis_target_vel = float(
                post_state.target_x_vel
                if int(args.transport_axis_index) == 0
                else (post_state.target_axis_vel if post_state.target_axis_vel is not None else post_state.target_x_vel)
            )
            post_safety = adapter.safety_monitor.check(
                post_state.as_robot_state(),
                axis_error=post_axis_err,
                orientation_error_norm=post_orient_err,
                axis_target_moving=bool(abs(post_axis_target_vel) > 1e-9),
            )
            # Opt-in TCP-accel/speed guard (see move_monitor's construction above).
            # move_monitor is None unless --enable-tcp-accel-guard was passed, in
            # which case move_guard_decision stays None and row_safety_ok/
            # row_safety_reason below reduce to exactly bool(post_safety.ok)/
            # str(post_safety.reason) -- byte-identical to before this flag existed.
            move_guard_decision = None
            if move_monitor is not None:
                tcp_pose_now = np.concatenate(
                    [np.asarray(post_state.ee_pos, dtype=np.float64).reshape(3), np.zeros(3, dtype=np.float64)]
                )
                target_tcp_pose_now = np.concatenate(
                    [np.asarray(target_ee_pos, dtype=np.float64).reshape(3), np.zeros(3, dtype=np.float64)]
                )
                move_guard_decision = move_monitor.check(
                    q=post_state.q,
                    qd=post_state.qd,
                    tcp_pose=tcp_pose_now,
                    target_tcp_pose=target_tcp_pose_now,
                    orientation_error_rad=post_orient_err,
                    axis_target_moving=bool(abs(post_axis_target_vel) > 1e-9),
                    dt_s=dt,
                )
            row_safety_ok = bool(post_safety.ok)
            row_safety_reason = str(post_safety.reason)
            if move_guard_decision is not None and not move_guard_decision.ok:
                guard_reason = f"tcp_accel_guard: {move_guard_decision.reason}"
                row_safety_ok = False
                row_safety_reason = f"{row_safety_reason}; {guard_reason}" if row_safety_reason else guard_reason
            joint_prox = compute_joint_limit_proximity(model, post_state.q, joint_ids)
            joint_limit_min_fraction = float(min(joint_prox.values())) if joint_prox else 0.0
            reward_terms = compute_reward_terms(
                state=post_state,
                tau=tau,
                tau_prev=prev_tau,
                reward_cfg=reward_cfg,
                axis_idx=int(args.transport_axis_index),
            )
            control_effort = float(np.linalg.norm(tau))
            energy_proxy = float(np.sum(np.abs(tau * post_state.qd)))
            tau_controller = np.asarray(diag.get("tau_controller", diag.get("tau_raw", tau)), dtype=np.float64).reshape(6)
            tau_gravity = np.asarray(diag.get("tau_gravity", np.zeros(6, dtype=np.float64)), dtype=np.float64).reshape(6)
            tau_applied = np.asarray(diag.get("tau_applied", tau_controller + tau_gravity), dtype=np.float64).reshape(6)
            # Controller-internal diagnostics already computed inside
            # controller.compute() (see CartesianImpedanceOutput) but
            # previously discarded every cycle here -- mirrors the fix
            # applied to hardware/direct_torque_transport.py. Absent (None)
            # for controller kinds that don't compute them (e.g.
            # zero_torque, direct_torque_profile).
            controller_output = diag.get("controller_output") or {}
            total_effort += control_effort * dt
            total_energy_proxy += energy_proxy * dt
            row = {
                "step": int(step_idx),
                "time_s": float(data.time),
                "dt_s": dt,
                "q": post_state.q.tolist(),
                "qd": post_state.qd.tolist(),
                "ee_pos": post_state.ee_pos.tolist(),
                "ee_quat": post_state.ee_quat.tolist(),
                "ee_lin_vel": post_state.ee_lin_vel.tolist(),
                "ee_ang_vel": post_state.ee_ang_vel.tolist(),
                "target_x": float(target_x_now),
                "target_x_vel": float(target_x_vel_now),
                "target_x_accel": float(target_x_accel_now),
                "target_ee_pos": target_ee_pos.tolist(),
                "target_ee_vel": target_ee_vel.tolist(),
                "actuator_force": np.asarray(data.actuator_force[:6], dtype=np.float64).tolist(),
                "qfrc_actuator": np.asarray(data.qfrc_actuator[:6], dtype=np.float64).tolist(),
                "qfrc_bias": np.asarray(data.qfrc_bias[:6], dtype=np.float64).tolist(),
                "tau_controller": tau_controller.tolist(),
                "tau_controller_clipped": np.asarray(diag.get("tau_controller_clipped", tau_controller), dtype=np.float64).tolist(),
                "tau_controller_saturated": np.asarray(diag.get("tau_controller_saturated", [False] * 6), dtype=bool).tolist(),
                "tau_controller_clip_fraction": float(diag.get("tau_controller_clip_fraction", diag.get("controller_torque_clip_fraction", 0.0))),
                "controller_torque_clip_fraction": float(diag.get("controller_torque_clip_fraction", diag.get("tau_controller_clip_fraction", 0.0))),
                "tau_gravity": tau_gravity.tolist(),
                "tau_applied": tau_applied.tolist(),
                "tau_applied_clipped": np.asarray(diag.get("tau_applied_clipped", tau), dtype=np.float64).tolist(),
                "tau_raw": np.asarray(diag.get("tau_raw", tau), dtype=np.float64).tolist(),
                "tau_filtered": np.asarray(diag.get("tau_filtered", tau), dtype=np.float64).tolist(),
                "tau": np.asarray(tau, dtype=np.float64).tolist(),
                "torque_saturation_fraction": float(diag.get("torque_saturation_fraction", 0.0)),
                "torque_clip_fraction": float(diag.get("torque_clip_fraction", 0.0)),
                "tau_applied_clip_fraction": float(diag.get("tau_applied_clip_fraction", diag.get("torque_clip_fraction", 0.0))),
                "applied_torque_clip_fraction": float(diag.get("applied_torque_clip_fraction", diag.get("torque_clip_fraction", 0.0))),
                "joint_limit_min_fraction": joint_limit_min_fraction,
                "x_error": float(diag.get("axis_error", 0.0)),
                "orientation_error_norm": float(diag.get("orientation_error_norm", 0.0)),
                "reward": reward_terms["reward"],
                "reward_terms": reward_terms,
                "safety_ok": row_safety_ok,
                "safety_reason": row_safety_reason,
                "controller_kind": str(diag.get("controller_kind", "")),
                "jacobian_cond": controller_output.get("jacobian_cond"),
                "singular_scale": controller_output.get("singular_scale"),
                "task_scale": controller_output.get("task_scale"),
                "task_backtrack_scale": controller_output.get("task_backtrack_scale"),
                "task_backtrack_iters": controller_output.get("task_backtrack_iters"),
                "task_feasible": controller_output.get("task_feasible"),
                "y_error": controller_output.get("y_error"),
                "z_error": controller_output.get("z_error"),
                "wrench": controller_output.get("wrench"),
                "tau_preclip": controller_output.get("tau_preclip"),
                "tau_task": controller_output.get("tau_task"),
                "tau_posture": controller_output.get("tau_posture"),
                "tau_damping": controller_output.get("tau_damping"),
                "tau_friction_ff": controller_output.get("tau_friction_ff"),
                "friction_z": controller_output.get("friction_z"),
                "wrench_accel_ff": controller_output.get("wrench_accel_ff"),
                "acceleration_feedforward_active": controller_output.get("acceleration_feedforward_active"),
                "jacobian_pre_step": pre_state.jacobian.tolist(),
                "termination_reason": "",
                "control_effort_l2": control_effort,
                "control_energy_proxy": energy_proxy,
                "gravity_mode": str(diag.get("gravity_mode", gravity_mode)),
                "gravity_mode_used": str(diag.get("gravity_mode_used", diag.get("gravity_mode", gravity_mode))),
                "gravity_compensation_active": bool(diag.get("gravity_compensation_active", gravity_mode == "gravity_comp")),
                "raw_mode_used": bool(diag.get("raw_mode_used", gravity_mode == "raw")),
                "trajectory_profile": trajectory_profile,
            }
            if move_monitor is not None:
                row["tcp_accel_guard_ok"] = bool(move_guard_decision.ok) if move_guard_decision is not None else True
                row["tcp_accel_guard_reason"] = str(move_guard_decision.reason) if move_guard_decision is not None else ""
            trace_rows.append(row)
            trace_writer.write_row(row)
            if not row_safety_ok:
                safety_violation = True
                termination_reason = row_safety_reason or "safety_violation"
                failure_time_s = float(data.time)
                row["termination_reason"] = termination_reason
                trace_rows[-1] = row
                break

    plot_path = run_dir / "trace_states.png"
    diag_plot_path = run_dir / "trace_diagnostics.png"
    if trace_rows and not args.no_plot:
        write_trace_plot(trace_rows, plot_path)
        _write_diagnostics_plot(trace_rows, diag_plot_path)

    final_state = trace_rows[-1] if trace_rows else {}
    summary.update(
        {
            "success": not safety_violation,
            "termination_reason": termination_reason,
            "steps": int(len(trace_rows)),
            "timestep_count": int(len(trace_rows)),
            "sim_time_s": float(data.time),
            "dt_s": dt,
            "initial_q": state0.q.tolist(),
            "initial_qd": state0.qd.tolist(),
            "initial_ee_pos": state0.ee_pos.tolist(),
            "final_q": final_state.get("q"),
            "final_qd": final_state.get("qd"),
            "final_ee_pos": final_state.get("ee_pos"),
            "final_ee_quat": final_state.get("ee_quat"),
            "gravity_mode": gravity_mode,
            "trajectory_profile": trajectory_profile,
            "failure_time_s": failure_time_s,
            "max_torque_saturation_fraction": float(max((r.get("torque_saturation_fraction", 0.0) for r in trace_rows), default=0.0)),
            "max_torque_clip_fraction": float(max((r.get("torque_clip_fraction", 0.0) for r in trace_rows), default=0.0)),
            "min_joint_limit_fraction": float(min((r.get("joint_limit_min_fraction", 1.0) for r in trace_rows), default=1.0)),
            "mean_control_effort_l2": float(np.mean([r.get("control_effort_l2", 0.0) for r in trace_rows])) if trace_rows else 0.0,
            "mean_control_energy_proxy": float(np.mean([r.get("control_energy_proxy", 0.0) for r in trace_rows])) if trace_rows else 0.0,
            "trace_path": str(trace_path),
            "plot_path": str(plot_path if plot_path.exists() else ""),
            "diagnostics_plot_path": str(diag_plot_path if diag_plot_path.exists() else ""),
            "clip_observed": bool(clip_observed),
            "torque_limit_scale": float(args.torque_limit_scale),
        }
    )
    if trace_rows:
        tau_controller_all = np.asarray([r.get("tau_controller", r.get("tau_raw", [0.0] * 6)) for r in trace_rows], dtype=np.float64)
        tau_gravity_all = np.asarray([r.get("tau_gravity", [0.0] * 6) for r in trace_rows], dtype=np.float64)
        tau_applied_all = np.asarray([r.get("tau_applied", r.get("tau", [0.0] * 6)) for r in trace_rows], dtype=np.float64)
        tau_all = np.asarray([r.get("tau", [0.0] * 6) for r in trace_rows], dtype=np.float64)
        q_all = np.asarray([r.get("q", [0.0] * 6) for r in trace_rows], dtype=np.float64)
        qd_all = np.asarray([r.get("qd", [0.0] * 6) for r in trace_rows], dtype=np.float64)
        x_error_all = np.asarray([float(r.get("x_error", 0.0)) for r in trace_rows], dtype=np.float64)
        orientation_error_all = np.asarray([float(r.get("orientation_error_norm", 0.0)) for r in trace_rows], dtype=np.float64)
        final_ee = np.asarray(final_state.get("ee_pos", state0.ee_pos.tolist()), dtype=np.float64).reshape(3)
        final_target_x = float(final_state.get("target_x", target_ee_pos[0] if len(target_ee_pos) else state0.ee_pos[0]))
        target_ee = np.asarray(
            final_state.get("target_ee_pos", target_ee_pos.tolist() if isinstance(target_ee_pos, np.ndarray) else target_ee_pos),
            dtype=np.float64,
        ).reshape(3)
        transport_summary = _transport_trace_summary(
            trace_rows,
            start_ee=np.asarray(state0.ee_pos, dtype=np.float64).reshape(3),
            transport_axis_index=int(args.transport_axis_index),
        )
        summary.update(
            {
                **transport_summary,
                "final_x_error_m": float(final_target_x - final_ee[0]),
                "final_ee_error_norm_m": float(np.linalg.norm(target_ee - final_ee)),
                "final_x_displacement_m": float(final_ee[0] - float(state0.ee_pos[0])),
                "achieved_x_delta_m": float(final_ee[0] - float(state0.ee_pos[0])),
                "max_abs_x_error_m": float(np.max(np.abs(x_error_all))) if x_error_all.size else 0.0,
                "max_abs_orientation_error_rad": float(np.max(np.abs(orientation_error_all))) if orientation_error_all.size else 0.0,
                "final_orientation_error_rad": float(final_state.get("orientation_error_norm", 0.0)),
                "max_abs_tau_nm": float(np.max(np.abs(tau_all))),
                "mean_abs_tau_nm": float(np.mean(np.abs(tau_all))),
                "torque_saturation_percentage": float(100.0 * np.mean([r.get("torque_clip_fraction", 0.0) for r in trace_rows])),
                "clipping_count": int(sum(1 for r in trace_rows if float(r.get("torque_clip_fraction", 0.0)) > 1.0e-12)),
                "max_abs_q_rad": float(np.max(np.abs(q_all))),
                "max_abs_qd_radps": float(np.max(np.abs(qd_all))),
                "velocity_guard_ok": bool(np.max(np.abs(qd_all)) <= float(ctrl_cfg.get("safety", {}).get("max_joint_velocity_radps", 3.0)) + 1e-9),
                "velocity_guard_margin_radps": float(float(ctrl_cfg.get("safety", {}).get("max_joint_velocity_radps", 3.0)) - float(np.max(np.abs(qd_all)))),
                "joint_limit_guard_ok": bool(float(summary.get("min_joint_limit_fraction", 0.0)) > 0.0),
                "joint_limit_margin_fraction": float(summary.get("min_joint_limit_fraction", 0.0)),
                "max_abs_tau_controller_nm": float(np.max(np.abs(tau_controller_all))),
                "max_abs_tau_gravity_nm": float(np.max(np.abs(tau_gravity_all))),
                "max_abs_tau_applied_nm": float(np.max(np.abs(tau_applied_all))),
                "mean_abs_tau_controller_nm": float(np.mean(np.abs(tau_controller_all))),
                "mean_abs_tau_applied_nm": float(np.mean(np.abs(tau_applied_all))),
                "torque_limit_nm": adapter.torque_limit_nm.tolist(),
                "target_x_delta": float(target_x_delta),
                "duration_s": float(args.duration),
                "transport_axis_index": int(args.transport_axis_index),
            }
        )
        summary.update(summarize_residual_torque_trace(trace_rows, torque_limit_nm=adapter.torque_limit_nm))
        if trajectory_profile in _MOVE_DURATION_PROFILES:
            summary.update(
                summarize_move_hold_trace(
                    trace_rows,
                    initial_ee_pos=np.asarray(state0.ee_pos, dtype=np.float64).reshape(3),
                    move_duration_s=float(move_duration_s if move_duration_s is not None else 0.0),
                    total_duration_s=float(args.duration),
                    transport_axis_index=int(args.transport_axis_index),
                )
            )
            summary.update(compute_valid_move_hold_metrics(summary))
        if failure_time_s is None and not summary["success"]:
            summary["failure_time_s"] = float(trace_rows[-1]["time_s"])
    if not summary["success"]:
        summary["failure_reason"] = termination_reason
    if move_monitor is not None:
        summary["tcp_accel_guard_tripped"] = bool(termination_reason.startswith("tcp_accel_guard:"))
    if args.mode == "safety-clipping":
        summary["success"] = bool(clip_observed) and not safety_violation

    summary_path = run_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0 if summary["success"] else 1


if __name__ == "__main__":
    raise SystemExit(run())
