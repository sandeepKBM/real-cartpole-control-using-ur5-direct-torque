"""Tests for the MuJoCo UR5e torque-control path."""

import json
import subprocess
import sys
from pathlib import Path

import mujoco
import numpy as np
import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from controller_core.kinematics_utils import rotmat_to_quat  # noqa: E402
from mujoco_ur5e_tools import UR5E_JOINT_ORDER, UR5E_TORQUE_ACTUATOR_SPECS  # noqa: E402
from mujoco_ur5e_tools import compute_gravity_torque  # noqa: E402
from mujoco_ur5e_tools import get_compiled_ur5e_torque_model_diagnostics  # noqa: E402
from mujoco_ur5e_tools import validate_ur5e_torque_xml_source_tree  # noqa: E402
from simulation.ur5e_mujoco_torque import (  # noqa: E402
    MujocoUR5eTorqueAdapter,
    MujocoUR5eTorqueAdapterConfig,
    build_controller,
    build_mujoco_state,
    compute_joint_limit_proximity,
    load_model,
)
from tools.audit_ur5e_mujoco_gravity_torque import (  # noqa: E402
    GRAVITY_SIGN_VARIANTS,
    HOLD_VARIANTS,
    audit_actuator_mapping_case,
    compute_bias_torque_live,
    compute_bias_torque_static,
    compute_gravity_torque_static,
)
from tools.ur5e_mujoco_torque_experiments import _x_profile_target  # noqa: E402
from transport_metrics import compute_valid_move_hold_metrics, compute_valid_transport_metrics, move_hold_ranking_key, summarize_move_hold_trace  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "config" / "ur5e_mujoco_torque.yaml"
TRANSPORT_CONFIG_PATH = REPO_ROOT / "config" / "ur5e_mujoco_torque_transport.yaml"
SCENE_PATH = REPO_ROOT / "assets" / "ur5e_torque" / "scene.xml"
LOW_Z_TRANSPORT_START_Q = np.array(
    [
        0.0,
        -0.1133064268431449,
        -0.664621645801302,
        4.921777393344012,
        -6.283185307179586,
        5.280928640069786,
    ],
    dtype=np.float64,
)
TRANSPORT_CONFIG_START_Q = np.array(
    [0.0, -1.5707963267948966, 0.0, -1.5707963267948966, 0.0, 0.0],
    dtype=np.float64,
)


def _load_controller_cfg() -> dict:
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))["controller"]


def test_torque_scene_loads_and_maps_joints() -> None:
    model, data, site_id, joint_ids, actuator_ids = load_model(SCENE_PATH)
    assert int(model.nu) == 6
    assert site_id >= 0
    assert len(joint_ids) == 6
    assert len(actuator_ids) == 6
    joint_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid) for jid in joint_ids]
    actuator_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, aid) for aid in actuator_ids]
    assert joint_names == list(UR5E_JOINT_ORDER)
    assert [spec[0] for spec in UR5E_TORQUE_ACTUATOR_SPECS] == actuator_names
    assert data is not None


def test_true_torque_source_tree_validation_passes() -> None:
    report = validate_ur5e_torque_xml_source_tree(SCENE_PATH)
    assert report["source_tree_ok"] is True
    assert any(path.endswith("assets/ur5e_torque/ur5e_torque.xml") for path in report["checked_files"])
    assert report["actuator_kinds"].get("motor", 0) == 6
    assert report["actuator_kinds"].get("position", 0) == 0
    assert report["actuator_kinds"].get("velocity", 0) == 0


def test_compiled_torque_model_is_true_torque() -> None:
    model, _, site_id, joint_ids, actuator_ids = load_model(SCENE_PATH)
    diag = get_compiled_ur5e_torque_model_diagnostics(model, site_name="attachment_site")
    assert diag["site_id"] == site_id
    assert int(diag["nu"]) == 6
    assert int(diag["neq"]) == 0
    assert np.asarray(diag["gravity"], dtype=np.float64)[2] < -1e-3
    assert np.all(np.asarray(diag["actuator_biastype"], dtype=np.int32) == 0)
    assert np.all(np.asarray(diag["actuator_gaintype"], dtype=np.int32) == 0)
    assert np.all(np.asarray(diag["actuator_dyntype"], dtype=np.int32) == 0)
    assert np.all(np.asarray(diag["actuator_trntype"], dtype=np.int32) == 0)
    assert np.all(np.asarray(diag["actuator_ctrllimited"], dtype=bool))
    assert np.all(np.asarray(diag["actuator_forcelimited"], dtype=bool))
    assert np.max(np.abs(np.asarray(diag["jnt_stiffness"], dtype=np.float64))) == 0.0
    assert np.max(np.abs(np.asarray(diag["dof_damping"], dtype=np.float64))) <= 0.5
    assert len(joint_ids) == 6
    assert len(actuator_ids) == 6


def test_gravity_torque_utility_returns_finite_vector() -> None:
    model, data, _, joint_ids, _ = load_model(SCENE_PATH)
    mujoco.mj_forward(model, data)
    tau_g = compute_gravity_torque(model, data, joint_ids)
    tau_bias_static = compute_bias_torque_static(model, data.qpos[:6], joint_ids, scratch_data=mujoco.MjData(model))
    assert tau_g.shape == (6,)
    assert np.all(np.isfinite(tau_g))
    assert np.linalg.norm(tau_g) > 0.0
    np.testing.assert_allclose(tau_g, tau_bias_static, atol=1e-9)


def test_adapter_hold_step_returns_finite_torque() -> None:
    model, data, site_id, joint_ids, _ = load_model(SCENE_PATH)
    mujoco.mj_forward(model, data)
    ee_pos = np.asarray(data.site_xpos[site_id], dtype=np.float64).copy()
    ee_rot = np.asarray(data.site_xmat[site_id], dtype=np.float64).reshape(3, 3).copy()
    quat = rotmat_to_quat(ee_rot)
    ctrl_cfg = _load_controller_cfg()
    controller = build_controller("torque_qp", ctrl_cfg)
    adapter = MujocoUR5eTorqueAdapter(
        model=model,
        site_id=site_id,
        joint_ids=joint_ids,
        controller=controller,
        config=MujocoUR5eTorqueAdapterConfig(
            controller_kind="torque_qp",
            gravity_compensation=True,
            transport_axis_index=0,
        ),
    )
    state = build_mujoco_state(
        model,
        data,
        site_id=site_id,
        joint_ids=joint_ids,
        time_s=float(data.time),
        dt_s=float(model.opt.timestep),
        target_x=float(ee_pos[0]),
        target_ee_pos=ee_pos.copy(),
        reference_quat=quat.copy(),
        hold_current_pose=True,
        transport_axis_index=0,
        gravity_compensation=True,
    )
    tau, diag = adapter.step(state=state)
    assert np.all(np.isfinite(tau))
    assert diag["safety_ok"] is True
    assert len(diag["tau_clipped"]) == 6
    prox = compute_joint_limit_proximity(model, state.q, joint_ids)
    assert len(prox) == 6


def test_shape_torque_clips_and_reports_saturation() -> None:
    model, data, site_id, joint_ids, _ = load_model(SCENE_PATH)
    ctrl_cfg = _load_controller_cfg()
    controller = build_controller("torque_qp", ctrl_cfg)
    adapter = MujocoUR5eTorqueAdapter(
        model=model,
        site_id=site_id,
        joint_ids=joint_ids,
        controller=controller,
        config=MujocoUR5eTorqueAdapterConfig(
            controller_kind="torque_qp",
            gravity_compensation=True,
            transport_axis_index=0,
        ),
    )
    tau_raw = np.full(6, 1.0e6, dtype=np.float64)
    tau_clipped, diag = adapter.shape_torque(tau_raw, dt_s=float(model.opt.timestep))
    assert np.all(np.isfinite(tau_clipped))
    assert np.max(np.abs(tau_clipped)) <= float(np.max(adapter.torque_limit_nm)) + 1e-9
    assert any(bool(v) for v in diag["tau_saturated"])
    assert float(diag["torque_clip_fraction"]) > 0.0


def _latest_run_dir(root: Path) -> Path:
    candidates = [p for p in root.rglob("summary.json")]
    assert candidates, f"no summary.json found under {root}"
    return max((p.parent for p in candidates), key=lambda p: p.stat().st_mtime)


def _run_experiment_cli(tmp_path: Path, *args: str) -> tuple[dict, Path, subprocess.CompletedProcess[str]]:
    out_root = tmp_path / "run"
    completed = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "tools" / "ur5e_mujoco_torque_experiments.py"),
            *args,
            "--output-dir",
            str(out_root),
        ],
        cwd=str(REPO_ROOT),
        check=False,
        text=True,
        capture_output=True,
    )
    assert completed.returncode == 0, completed.stderr
    run_dir = _latest_run_dir(out_root)
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    return summary, run_dir, completed


def _run_experiment_cli_raw(tmp_path: Path, *args: str) -> tuple[subprocess.CompletedProcess[str], Path]:
    out_root = tmp_path / "run"
    completed = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "tools" / "ur5e_mujoco_torque_experiments.py"),
            *args,
            "--output-dir",
            str(out_root),
        ],
        cwd=str(REPO_ROOT),
        check=False,
        text=True,
        capture_output=True,
    )
    run_dir = _latest_run_dir(out_root)
    return completed, run_dir


def _assert_start_pose(summary: dict, expected_q: np.ndarray, *, source: str) -> None:
    assert summary["start_q_source"] == source
    np.testing.assert_allclose(np.asarray(summary["start_q_rad"], dtype=np.float64), expected_q)
    np.testing.assert_allclose(np.asarray(summary["initial_q"], dtype=np.float64), expected_q)


def test_valid_transport_metric_rejects_overshoot_and_accepts_clean_tracking() -> None:
    overshoot = {
        "success": True,
        "termination_reason": "duration_complete",
        "target_x_delta": 0.005,
        "final_x_error_m": 0.018,
        "achieved_x_delta_m": 0.053,
        "max_abs_y_drift_m": 0.01,
        "max_abs_z_drift_m": 0.01,
        "max_abs_orthogonal_drift_m": 0.01,
        "max_abs_orientation_error_rad": 0.05,
        "final_orientation_error_rad": 0.01,
        "velocity_guard_ok": True,
        "joint_limit_guard_ok": True,
        "max_abs_qd_radps": 0.5,
        "torque_saturation_percentage": 0.0,
    }
    overshoot_metrics = compute_valid_transport_metrics(overshoot)
    assert overshoot_metrics["valid_transport"] is False
    assert overshoot_metrics["x_tracking_pass"] is False

    clean = {
        "success": True,
        "termination_reason": "duration_complete",
        "target_x_delta": 0.02,
        "final_x_error_m": 0.001,
        "achieved_x_delta_m": 0.0195,
        "max_abs_y_drift_m": 0.01,
        "max_abs_z_drift_m": 0.01,
        "max_abs_orthogonal_drift_m": 0.01,
        "max_abs_orientation_error_rad": 0.08,
        "final_orientation_error_rad": 0.04,
        "velocity_guard_ok": True,
        "joint_limit_guard_ok": True,
        "max_abs_qd_radps": 0.3,
        "torque_saturation_percentage": 0.0,
    }
    clean_metrics = compute_valid_transport_metrics(clean)
    assert clean_metrics["valid_transport"] is True
    assert clean_metrics["strict_valid_transport"] is True
    assert clean_metrics["transport_quality_score"] <= clean_metrics["tracking_score"]


def _make_move_hold_trace(
    *,
    target_x_delta: float,
    move_duration_s: float,
    hold_positions: list[float],
    move_positions: list[float] | None = None,
    hold_orientations: list[float] | None = None,
) -> list[dict[str, object]]:
    move_positions = move_positions or [0.0, target_x_delta]
    hold_orientations = hold_orientations or [0.0 for _ in hold_positions]
    trace_rows: list[dict[str, object]] = []
    move_times = [0.0, move_duration_s]
    hold_times = [move_duration_s + 0.05 * (idx + 1) for idx in range(len(hold_positions))]
    for idx, (time_s, x_pos, orient) in enumerate(zip(move_times, move_positions, [0.0, 0.0], strict=True)):
        trace_rows.append(
            {
                "time_s": float(time_s),
                "ee_pos": [float(x_pos), 0.0, 0.0],
                "ee_quat": [1.0, 0.0, 0.0, 0.0],
                "qd": [0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                "x_error": float(target_x_delta - x_pos),
                "target_x": float(target_x_delta if idx > 0 else 0.0),
                "target_ee_pos": [float(target_x_delta if idx > 0 else 0.0), 0.0, 0.0],
                "orientation_error_norm": float(orient),
                "tau_controller": [1.0, 0.5, 0.0, 0.0, 0.0, 0.0],
                "tau_applied": [1.2, 0.6, 0.0, 0.0, 0.0, 0.0],
            }
        )
    for idx, (time_s, x_pos, orient) in enumerate(zip(hold_times, hold_positions, hold_orientations, strict=True)):
        trace_rows.append(
            {
                "time_s": float(time_s),
                "ee_pos": [float(x_pos), 0.0, 0.0],
                "ee_quat": [1.0, 0.0, 0.0, 0.0],
                "qd": [0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                "x_error": float(target_x_delta - x_pos),
                "target_x": float(target_x_delta),
                "target_ee_pos": [float(target_x_delta), 0.0, 0.0],
                "orientation_error_norm": float(orient),
                "tau_controller": [1.0, 0.5, 0.0, 0.0, 0.0, 0.0],
                "tau_applied": [1.2, 0.6, 0.0, 0.0, 0.0, 0.0],
            }
        )
    return trace_rows


def test_min_jerk_move_hold_profile_reaches_target_and_holds() -> None:
    x_at_move_end, vel_at_move_end = _x_profile_target("min_jerk_move_hold", 0.0, 0.02, 0.05, 0.1, move_duration_s=0.05)
    x_after_hold, vel_after_hold = _x_profile_target("min_jerk_move_hold", 0.0, 0.02, 0.075, 0.1, move_duration_s=0.05)
    assert x_at_move_end == pytest.approx(0.02, abs=1e-9)
    assert vel_at_move_end == pytest.approx(0.0, abs=1e-9)
    assert x_after_hold == pytest.approx(0.02, abs=1e-9)
    assert vel_after_hold == pytest.approx(0.0, abs=1e-9)


def test_move_hold_summary_and_metric_reject_overshoot_and_accept_clean() -> None:
    overshoot_rows = _make_move_hold_trace(
        target_x_delta=0.005,
        move_duration_s=0.05,
        move_positions=[0.0, 0.005],
        hold_positions=[0.020, 0.026, 0.025],
        hold_orientations=[0.05, 0.05, 0.05],
    )
    overshoot_summary = summarize_move_hold_trace(
        overshoot_rows,
        initial_ee_pos=[0.0, 0.0, 0.0],
        move_duration_s=0.05,
        total_duration_s=0.15,
        transport_axis_index=0,
    )
    overshoot_summary.update(
        {
            "success": True,
            "termination_reason": "duration_complete",
            "target_x_delta": 0.005,
            "velocity_guard_ok": True,
            "joint_limit_guard_ok": True,
            "torque_saturation_percentage": 0.0,
        }
    )
    overshoot_metrics = compute_valid_move_hold_metrics(overshoot_summary)
    assert overshoot_metrics["valid_move_phase"] is True
    assert overshoot_metrics["valid_hold_phase"] is False
    assert overshoot_metrics["valid_move_and_hold"] is False
    assert overshoot_metrics["hold_phase_x_drift_from_hold_start_m"] > 0.003

    clean_rows = _make_move_hold_trace(
        target_x_delta=0.005,
        move_duration_s=0.05,
        move_positions=[0.0, 0.005],
        hold_positions=[0.0051, 0.0050, 0.0052],
        hold_orientations=[0.01, 0.01, 0.01],
    )
    clean_summary = summarize_move_hold_trace(
        clean_rows,
        initial_ee_pos=[0.0, 0.0, 0.0],
        move_duration_s=0.05,
        total_duration_s=0.15,
        transport_axis_index=0,
    )
    clean_summary.update(
        {
            "success": True,
            "termination_reason": "duration_complete",
            "target_x_delta": 0.005,
            "velocity_guard_ok": True,
            "joint_limit_guard_ok": True,
            "torque_saturation_percentage": 0.0,
        }
    )
    clean_metrics = compute_valid_move_hold_metrics(clean_summary)
    assert clean_metrics["valid_move_phase"] is True
    assert clean_metrics["valid_hold_phase"] is True
    assert clean_metrics["valid_move_and_hold"] is True
    assert clean_metrics["hold_phase_x_drift_from_hold_start_m"] < 0.003
    assert move_hold_ranking_key(clean_summary) > move_hold_ranking_key(overshoot_summary)


def test_min_jerk_move_hold_duration_guard_fails_clearly(tmp_path: Path) -> None:
    completed, run_dir = _run_experiment_cli_raw(
        tmp_path,
        "--mode",
        "controller-rollout",
        "--controller-kind",
        "impedance",
        "--gravity-mode",
        "gravity_comp",
        "--trajectory-profile",
        "min_jerk_move_hold",
        "--move-duration",
        "0.1",
        "--duration",
        "0.05",
        "--target-x-delta",
        "0.005",
        "--config",
        str(TRANSPORT_CONFIG_PATH),
        "--no-plot",
    )
    assert completed.returncode == 2
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["failure_reason"] == "move_duration_must_be_positive_and_less_than_duration"


def test_move_hold_runner_tiny_smoke(tmp_path: Path) -> None:
    out_root = tmp_path / "move_hold"
    completed = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "tools" / "ur5e_move_hold_transport.py"),
            "--config",
            str(TRANSPORT_CONFIG_PATH),
            "--target-x-deltas",
            "0.005",
            "--move-durations",
            "0.05",
            "--hold-durations",
            "0.05",
            "--torque-limit-scales",
            "0.5",
            "--no-plot",
            "--output-root",
            str(out_root),
        ],
        cwd=str(REPO_ROOT),
        check=False,
        text=True,
        capture_output=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert (out_root / "summary.csv").exists()
    assert (out_root / "summary.json").exists()
    assert (out_root / "best_settings.json").exists()
    summary = json.loads((out_root / "summary.json").read_text(encoding="utf-8"))
    best_settings = json.loads((out_root / "best_settings.json").read_text(encoding="utf-8"))
    assert summary["num_runs"] == 1
    assert summary["rows"][0]["stage"] == "baseline"
    assert summary["rows"][0]["controller_kind"] == "impedance"
    assert summary["rows"][0]["gravity_mode"] == "gravity_comp"
    assert summary["rows"][0]["trajectory_profile"] == "min_jerk_move_hold"
    assert "valid_move_and_hold" in summary["rows"][0]
    assert "move_phase_final_x_error_m" in summary["rows"][0]
    assert "hold_phase_x_drift_from_hold_start_m" in summary["rows"][0]
    assert Path(summary["rows"][0]["trace_path"]).exists()
    assert "best_valid_move_and_hold_overall" in best_settings
    assert "best_valid_move_and_hold_by_total_duration" in best_settings
    assert "largest_valid_target_x_delta_by_hold_duration" in best_settings
    assert "largest_valid_target_x_delta_by_total_duration" in best_settings
    assert "dominant_failure_phase" in best_settings
    assert "common_move_failure_reasons" in best_settings
    assert "common_hold_failure_reasons" in best_settings
    assert "residual_torque_ratio_summary" in best_settings


def test_zero_torque_gravity_probe_reports_motion(tmp_path: Path) -> None:
    summary, _, _ = _run_experiment_cli(
        tmp_path,
        "--mode",
        "zero-torque-gravity",
        "--duration",
        "0.1",
        "--no-plot",
    )
    assert summary["true_torque_verified"] is True
    assert summary["success"] is True
    assert summary["suspicious_zero_torque_hold"] is False
    assert float(summary["max_abs_q_delta_from_start"]) > 1e-4
    assert float(summary["max_abs_ee_delta_from_start"]) > 1e-4
    assert float(summary["max_abs_qd_radps"]) > 1e-4


def test_gravity_comp_hold_reports_applied_gravity_torque(tmp_path: Path) -> None:
    summary, run_dir, _ = _run_experiment_cli(
        tmp_path,
        "--mode",
        "gravity-comp-hold",
        "--duration",
        "0.05",
        "--no-plot",
    )
    assert summary["true_torque_verified"] is True
    assert summary["gravity_mode"] == "gravity_comp"
    assert summary["gravity_mode_used"] == "gravity_comp"
    assert summary["gravity_compensation_active"] is True
    assert summary["controller_kind"] == "zero_torque"
    assert summary["success"] is True
    assert float(summary["max_abs_tau_controller_nm"]) == 0.0
    assert float(summary["max_abs_tau_gravity_nm"]) > 0.0
    assert float(summary["max_abs_tau_applied_nm"]) > 0.0
    assert float(summary["mean_abs_tau_gravity_nm"]) > 0.0
    assert float(summary["gravity_torque_fraction"]) > 0.0
    assert float(summary["controller_torque_fraction"]) == 0.0
    trace_path = Path(summary["trace_path"])
    assert trace_path.exists()
    first_row = json.loads(trace_path.read_text(encoding="utf-8").splitlines()[0])
    assert "tau_controller" in first_row
    assert "tau_gravity" in first_row
    assert "tau_applied" in first_row
    assert "tau_controller_clipped" in first_row
    assert "tau_applied_clipped" in first_row
    assert "gravity_compensation_active" in first_row
    assert run_dir.exists()


@pytest.mark.parametrize(
    "mode",
    ["gravity-comp-hold-long", "residual-impedance-hold"],
)
def test_residual_diagnostic_mode_aliases_run(tmp_path: Path, mode: str) -> None:
    summary, _, _ = _run_experiment_cli(
        tmp_path,
        "--mode",
        mode,
        "--duration",
        "0.05",
        "--no-plot",
    )
    assert summary["success"] is True
    assert summary["gravity_mode"] == "gravity_comp"
    assert summary["gravity_mode_used"] == "gravity_comp"
    assert summary["gravity_compensation_active"] is True
    assert float(summary["mean_abs_tau_gravity_nm"]) > 0.0
    assert "controller_torque_fraction" in summary
    assert "gravity_torque_fraction" in summary


def test_gravity_torque_audit_helpers_are_consistent() -> None:
    model, data, _, joint_ids, _ = load_model(SCENE_PATH)
    mujoco.mj_forward(model, data)
    tau_static = compute_gravity_torque_static(model, data.qpos[:6], joint_ids, scratch_data=mujoco.MjData(model))
    tau_bias_static = compute_bias_torque_static(model, data.qpos[:6], joint_ids, scratch_data=mujoco.MjData(model))
    tau_bias_live = compute_bias_torque_live(model, data, joint_ids)
    assert tau_static.shape == (6,)
    assert tau_bias_static.shape == (6,)
    assert tau_bias_live.shape == (6,)
    assert np.all(np.isfinite(tau_static))
    assert np.all(np.isfinite(tau_bias_static))
    assert np.all(np.isfinite(tau_bias_live))
    np.testing.assert_allclose(tau_bias_static, tau_static, atol=1e-9)
    assert set(GRAVITY_SIGN_VARIANTS) == {
        "raw_zero",
        "plus_gravity_comp",
        "minus_gravity_comp",
        "direct_bias_force",
        "inverse_dynamics_hold",
    }
    assert set(HOLD_VARIANTS) == {"gravity_comp_hold", "residual_impedance_hold"}


def test_actuator_mapping_audit_reports_six_dimensional_vectors() -> None:
    model, data, _, joint_ids, _ = load_model(SCENE_PATH)
    mujoco.mj_forward(model, data)
    tau_cmd = np.array([0.2, -0.1, 0.05, 0.0, 0.1, -0.05], dtype=np.float64)
    result = audit_actuator_mapping_case(model, data, joint_ids=joint_ids, tau_cmd=tau_cmd)
    assert len(result["tau_cmd"]) == 6
    assert len(result["qfrc_actuator_joint"]) == 6
    assert len(result["actuator_force_joint"]) == 6
    assert len(result["qfrc_bias_joint"]) == 6
    assert len(result["joint_dof_ids"]) == 6
    assert result["max_abs_mapping_error"] < 1e-8
    assert result["mean_abs_mapping_error"] < 1e-8
    assert result["max_abs_actuator_force_error"] < 1e-8
    assert result["mean_abs_actuator_force_error"] < 1e-8


def test_gravity_torque_audit_cli_smoke(tmp_path: Path) -> None:
    out_root = tmp_path / "gravity_audit"
    completed = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "tools" / "audit_ur5e_mujoco_gravity_torque.py"),
            "--config",
            str(TRANSPORT_CONFIG_PATH),
            "--poses",
            "active_origin",
            "--durations",
            "0.05",
            "--no-plot",
            "--output-root",
            str(out_root),
        ],
        cwd=str(REPO_ROOT),
        check=False,
        text=True,
        capture_output=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert (out_root / "summary.csv").exists()
    assert (out_root / "summary.json").exists()
    assert (out_root / "gravity_sign_audit.json").exists()
    assert (out_root / "actuator_mapping_audit.json").exists()
    assert (out_root / "clamping_audit.json").exists()
    assert (out_root / "hold_quality_audit.json").exists()
    summary = json.loads((out_root / "summary.json").read_text(encoding="utf-8"))
    gravity_sign = json.loads((out_root / "gravity_sign_audit.json").read_text(encoding="utf-8"))
    mapping = json.loads((out_root / "actuator_mapping_audit.json").read_text(encoding="utf-8"))
    clamping = json.loads((out_root / "clamping_audit.json").read_text(encoding="utf-8"))
    hold_quality = json.loads((out_root / "hold_quality_audit.json").read_text(encoding="utf-8"))
    assert summary["num_runs"] >= 1
    assert summary["gravity_compensation_used_static_state"] is True
    assert "gravity_compensation_sign_correct" in summary
    assert "gravity_compensation_uses_live_qd_terms" in summary
    assert gravity_sign["variants"] == list(GRAVITY_SIGN_VARIANTS)
    assert hold_quality["variants"] == list(HOLD_VARIANTS)
    assert len(mapping["cases"]) >= 1
    first_mapping_case = mapping["cases"][0]
    assert len(first_mapping_case["qfrc_actuator_joint"]) == 6
    assert len(first_mapping_case["actuator_force_joint"]) == 6
    assert len(first_mapping_case["qfrc_bias_joint"]) == 6
    assert mapping["max_abs_mapping_error"] < 1e-8
    assert "controller_level_clip_appears" in clamping
    assert "mujo_co_clip_appears" in clamping
    assert summary["actuator_mapping_audit"]["one_to_one"] is True


def test_explicit_start_q_override_applies(tmp_path: Path) -> None:
    summary, _, _ = _run_experiment_cli(
        tmp_path,
        "--mode",
        "gravity-comp-hold",
        "--start-q-rad",
        *[str(v) for v in LOW_Z_TRANSPORT_START_Q],
        "--duration",
        "0.05",
        "--no-plot",
    )
    assert summary["success"] is True
    _assert_start_pose(summary, LOW_Z_TRANSPORT_START_Q, source="cli")


@pytest.mark.parametrize(
    ("mode", "extra_args"),
    [
        ("single-joint-pulse", ["--duration", "0.05", "--joint-index", "0", "--torque-nm", "0.1"]),
        ("constant-small-torque", ["--duration", "0.05", "--joint-index", "0", "--torque-nm", "0.05"]),
        ("sinusoidal-torque", ["--duration", "0.05", "--joint-index", "0", "--torque-amp-nm", "0.05", "--torque-freq-hz", "2.0"]),
    ],
)
def test_true_torque_sanity_modes_log_actuation(tmp_path: Path, mode: str, extra_args: list[str]) -> None:
    summary, run_dir, _ = _run_experiment_cli(tmp_path, "--mode", mode, *extra_args, "--no-plot")
    assert summary["success"] is True
    assert float(summary["max_abs_tau_nm"]) > 0.0
    trace_path = Path(summary["trace_path"])
    assert trace_path.exists()
    first_row = json.loads(trace_path.read_text(encoding="utf-8").splitlines()[0])
    assert "actuator_force" in first_row
    assert "qfrc_actuator" in first_row
    assert "qfrc_bias" in first_row
    assert run_dir.exists()


def test_comparison_runner_tiny_sweep(tmp_path: Path) -> None:
    out_root = tmp_path / "comparison"
    completed = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "tools" / "compare_ur5e_mujoco_controllers.py"),
            "--controllers",
            "torque_qp",
            "--target-x-deltas",
            "0.0025",
            "--durations",
            "0.05",
            "--torque-limit-scales",
            "0.5",
            "--start-q-rad",
            *[str(v) for v in LOW_Z_TRANSPORT_START_Q],
            "--no-run-plots",
            "--output-root",
            str(out_root),
        ],
        cwd=str(REPO_ROOT),
        check=False,
        text=True,
        capture_output=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert (out_root / "summary.csv").exists()
    assert (out_root / "summary.json").exists()
    summary = json.loads((out_root / "summary.json").read_text(encoding="utf-8"))
    assert summary["num_runs"] == 1
    assert summary["runs"][0]["controller_kind"] == "torque_qp"
    assert summary["runs"][0]["success"] in (True, False)
    assert Path(summary["runs"][0]["trace_path"]).exists()
    child_summary = json.loads(Path(summary["runs"][0]["summary_path"]).read_text(encoding="utf-8"))
    _assert_start_pose(child_summary, LOW_Z_TRANSPORT_START_Q, source="cli")


def test_x_frame_envelope_runner_tiny_sweep(tmp_path: Path) -> None:
    out_root = tmp_path / "envelope"
    completed = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "tools" / "ur5e_x_frame_envelope.py"),
            "--controllers",
            "impedance",
            "--gravity-modes",
            "gravity_comp",
            "--profiles",
            "min_jerk",
            "--target-x-deltas",
            "0.005",
            "--durations",
            "0.05",
            "--torque-limit-scales",
            "0.5",
            "--config",
            str(TRANSPORT_CONFIG_PATH),
            "--no-plot",
            "--output-root",
            str(out_root),
        ],
        cwd=str(REPO_ROOT),
        check=False,
        text=True,
        capture_output=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert (out_root / "summary.csv").exists()
    assert (out_root / "summary.json").exists()
    assert (out_root / "best_settings.json").exists()
    summary = json.loads((out_root / "summary.json").read_text(encoding="utf-8"))
    best_settings = json.loads((out_root / "best_settings.json").read_text(encoding="utf-8"))
    assert summary["num_runs"] == 1
    assert summary["runs"][0]["controller_kind"] == "impedance"
    assert summary["runs"][0]["gravity_mode"] == "gravity_comp"
    assert summary["runs"][0]["trajectory_profile"] == "min_jerk"
    assert "valid_transport" in summary["runs"][0]
    assert "tracking_score" in summary["runs"][0]
    assert Path(summary["runs"][0]["trace_path"]).exists()
    child_summary = json.loads(Path(summary["runs"][0]["summary_path"]).read_text(encoding="utf-8"))
    _assert_start_pose(child_summary, TRANSPORT_CONFIG_START_Q, source="config:home_qpos")
    assert "best_overall_valid_transport" in best_settings
    assert "best_valid_transport_by_duration" in best_settings
    assert "largest_valid_target_x_delta_by_duration" in best_settings
    assert "largest_valid_achieved_x_delta_by_duration" in best_settings
    assert "best_raw_motion_result" in best_settings
    assert "best_tracking_result" in best_settings
    assert "best_overall_achieved" in best_settings
    assert "best_by_duration_achieved" in best_settings


def test_impedance_tuning_runner_smoke_single_gain(tmp_path: Path) -> None:
    out_root = tmp_path / "tuning"
    gain_overrides = json.dumps(
        {
            "kp_x": 25.0,
            "kd_x": 8.0,
            "kp_y": 80.0,
            "kd_y": 15.0,
            "kp_z": 120.0,
            "kd_z": 20.0,
            "kp_rot": 20.0,
            "kd_rot": 5.0,
            "kp_posture": 2.0,
            "kd_posture": 0.5,
            "kd_joint": 0.8,
        }
    )
    completed = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "tools" / "tune_ur5e_residual_impedance_transport.py"),
            "--config",
            str(TRANSPORT_CONFIG_PATH),
            "--gain-overrides-json",
            gain_overrides,
            "--target-x-deltas",
            "0.005",
            "--durations",
            "0.1",
            "--torque-limit-scales",
            "0.5",
            "--no-plot",
            "--output-root",
            str(out_root),
        ],
        cwd=str(REPO_ROOT),
        check=False,
        text=True,
        capture_output=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert (out_root / "summary.csv").exists()
    assert (out_root / "summary.json").exists()
    assert (out_root / "best_settings.json").exists()
    summary = json.loads((out_root / "summary.json").read_text(encoding="utf-8"))
    best_settings = json.loads((out_root / "best_settings.json").read_text(encoding="utf-8"))
    assert summary["num_runs"] == 1
    assert summary["rows"][0]["stage"] == "single"
    assert summary["rows"][0]["mode"] == "x-transport-minjerk"
    assert summary["rows"][0]["controller_kind"] == "impedance"
    assert summary["rows"][0]["gravity_mode"] == "gravity_comp"
    assert summary["rows"][0]["trajectory_profile"] == "min_jerk"
    assert "valid_transport" in summary["rows"][0]
    assert float(summary["rows"][0]["mean_abs_tau_gravity_nm"]) > 0.0
    assert summary["rows"][0]["gravity_compensation_active"] is True
    assert Path(summary["rows"][0]["trace_path"]).exists()
    assert "best_overall_valid_transport" in best_settings
    assert "best_hold_stability_gain_set" in best_settings
    assert "best_1s_transport_gain_set" in best_settings
    assert "best_3s_transport_gain_set" in best_settings
    assert "best_raw_motion_result" in best_settings
    assert "best_tracking_result" in best_settings
