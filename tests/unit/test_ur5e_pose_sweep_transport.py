"""Unit tests for tools/ur5e_pose_sweep_transport.py.

Pure numpy / no subprocess: covers pose computation, gain extraction, and
command construction only. The actual multi-category sweep (subprocessing
tools/ur5e_move_hold_transport.py once per alpha x category) is exercised
manually for the alpha=0.2/0.3 validation, not here -- that's real simulation
compute, out of scope for a fast unit test.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from hardware.poses import HEIGHT_ALPHA_0_5_Q, q_for_height_alpha  # noqa: E402
from tools.ur5e_pose_sweep_transport import (  # noqa: E402
    CATEGORY_GRIDS,
    CATEGORY_ORDER,
    build_move_hold_command,
    gains_from_config,
    parse_pass_count,
    pose_for_alpha,
)


def test_pose_for_alpha_matches_hardware_poses_directly() -> None:
    for alpha in (0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0):
        np.testing.assert_allclose(pose_for_alpha(alpha), q_for_height_alpha(alpha))


def test_pose_for_alpha_0_5_matches_named_constant() -> None:
    np.testing.assert_allclose(pose_for_alpha(0.5), HEIGHT_ALPHA_0_5_Q)


def test_pose_for_alpha_rejects_out_of_range() -> None:
    with pytest.raises(ValueError):
        pose_for_alpha(1.5)


def test_category_grids_match_documented_shapes() -> None:
    # AGENTS.md sec 3 / docs/status/wrist_orientation_task_2026-07-29.md sec 4:
    # canonical grid 8, long holds 8, large displacements 8 (16 combined with
    # canonical), torque-scale robustness 14.
    assert set(CATEGORY_GRIDS.keys()) == set(CATEGORY_ORDER)

    def _row_count(name: str) -> int:
        grid = CATEGORY_GRIDS[name]
        return (
            len(grid["target_x_deltas"])
            * len(grid["move_durations"])
            * len(grid["hold_durations"])
            * len(grid["torque_limit_scales"])
        )

    assert _row_count("canonical_grid") == 8
    assert _row_count("long_holds") == 8
    assert _row_count("large_displacements") == 8
    assert _row_count("torque_scale_robustness") == 14


def test_gains_from_config_extracts_only_gain_fields(tmp_path: Path) -> None:
    cfg = {
        "controller": {
            "gains": {
                "kp_x": 400.0,
                "kd_x": 40.0,
                "kp_y": 80.0,
                "kd_y": 15.0,
                "kp_z": 120.0,
                "kd_z": 20.0,
                "kp_rot": 0.0,
                "kd_rot": 10.0,
                "kp_posture": 25.0,
                "kd_posture": 6.0,
                "kd_joint": 4.0,
                "kp_rot_wrist": 0.0,
                "kd_rot_wrist": 10.0,
            },
            "wrist_orientation_task": True,
        }
    }
    config_path = tmp_path / "candidate.yaml"
    config_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    gains = gains_from_config(config_path)

    assert gains["kp_x"] == 400.0
    assert gains["kp_rot"] == 0.0
    assert gains["kd_posture"] == 6.0
    # kp_rot_wrist/kd_rot_wrist are not GAIN_FIELDS -- must not be forwarded via
    # --gain-overrides-json (the child driver would silently drop them anyway,
    # but this asserts the wrapper doesn't even try).
    assert "kp_rot_wrist" not in gains
    assert "kd_rot_wrist" not in gains


def test_gains_from_config_handles_missing_gains_section(tmp_path: Path) -> None:
    config_path = tmp_path / "empty.yaml"
    config_path.write_text(yaml.safe_dump({"controller": {}}), encoding="utf-8")
    assert gains_from_config(config_path) == {}


def test_build_move_hold_command_includes_start_q_and_gain_overrides(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("controller: {}\n", encoding="utf-8")
    output_root = tmp_path / "out"
    start_q = pose_for_alpha(0.2)
    gain_overrides = {"kp_x": 400.0, "kd_x": 40.0}

    cmd = build_move_hold_command(
        config=config_path,
        output_root=output_root,
        seed=7,
        gain_overrides=gain_overrides,
        start_q_rad=start_q,
        category_params=CATEGORY_GRIDS["canonical_grid"],
        no_plot=True,
    )

    assert cmd[0] == sys.executable
    assert cmd[1].endswith("ur5e_move_hold_transport.py")
    assert "--config" in cmd and str(config_path) in cmd
    assert "--output-root" in cmd and str(output_root) in cmd
    assert "--seed" in cmd and "7" in cmd
    assert "--no-plot" in cmd

    # start-q-rad forwarded as six separate float tokens, in order.
    idx = cmd.index("--start-q-rad")
    forwarded_q = [float(v) for v in cmd[idx + 1 : idx + 7]]
    np.testing.assert_allclose(forwarded_q, start_q)

    # gain overrides forwarded as a single JSON blob.
    idx = cmd.index("--gain-overrides-json")
    forwarded_gains = json.loads(cmd[idx + 1])
    assert forwarded_gains == gain_overrides

    # category grid values forwarded verbatim.
    idx = cmd.index("--target-x-deltas")
    n = len(CATEGORY_GRIDS["canonical_grid"]["target_x_deltas"])
    forwarded_deltas = [float(v) for v in cmd[idx + 1 : idx + 1 + n]]
    assert forwarded_deltas == CATEGORY_GRIDS["canonical_grid"]["target_x_deltas"]


def test_build_move_hold_command_omits_gain_overrides_flag_when_none(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("controller: {}\n", encoding="utf-8")
    cmd = build_move_hold_command(
        config=config_path,
        output_root=tmp_path / "out",
        seed=0,
        gain_overrides=None,
        start_q_rad=pose_for_alpha(0.0),
        category_params=CATEGORY_GRIDS["canonical_grid"],
    )
    assert "--gain-overrides-json" not in cmd


def test_build_move_hold_command_plot_flag_toggles_no_plot(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("controller: {}\n", encoding="utf-8")
    cmd_no_plot = build_move_hold_command(
        config=config_path,
        output_root=tmp_path / "out",
        seed=0,
        gain_overrides=None,
        start_q_rad=pose_for_alpha(0.0),
        category_params=CATEGORY_GRIDS["canonical_grid"],
        no_plot=True,
    )
    cmd_plot = build_move_hold_command(
        config=config_path,
        output_root=tmp_path / "out",
        seed=0,
        gain_overrides=None,
        start_q_rad=pose_for_alpha(0.0),
        category_params=CATEGORY_GRIDS["canonical_grid"],
        no_plot=False,
    )
    assert "--no-plot" in cmd_no_plot
    assert "--no-plot" not in cmd_plot


def test_parse_pass_count_reads_summary_fields() -> None:
    summary = {"num_valid_move_and_hold": 6, "num_runs": 8, "other": "ignored"}
    assert parse_pass_count(summary) == (6, 8)


def test_parse_pass_count_defaults_to_zero_when_missing() -> None:
    assert parse_pass_count({}) == (0, 0)
