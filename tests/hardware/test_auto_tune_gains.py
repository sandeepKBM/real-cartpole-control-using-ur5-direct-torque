"""Tests for tools/ur5e_auto_tune_gains.py -- the batch, sim-gated real-hardware
candidate proposer.

Pure local-file + Optuna + mocked-subprocess logic; no RTDE/robot involved at
all (matches tests/hardware/test_suggest_gains.py's own precedent for why
this lives under tests/hardware/ despite never opening a socket)."""

from __future__ import annotations

import ast
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "tools"))

import ur5e_auto_tune_gains as atg  # noqa: E402
import ur5e_suggest_gains as sg  # noqa: E402


def _write_summary(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _fake_move_hold_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "num_valid_move_and_hold": sum(1 for r in rows if r.get("valid_move_and_hold")),
        "num_runs": len(rows),
        "rows": rows,
    }


# --------------------------------------------------------------------------- #
# Reuse: this module must import ur5e_suggest_gains's functions, not
# reimplement them.
# --------------------------------------------------------------------------- #
def test_reuses_suggest_gains_functions_by_identity() -> None:
    assert atg._load_trial_summaries is sg._load_trial_summaries
    assert atg._parse_search_space is sg._parse_search_space
    assert atg.build_study is sg.build_study
    assert atg.score_trial is sg.score_trial


def test_suggest_gains_own_tests_unaffected() -> None:
    # ur5e_suggest_gains.py's single-candidate CLI must still behave exactly
    # as before -- nothing in this module patches/monkeypatches it globally.
    assert sg.score_trial({"valid_move_and_hold": True, "move_hold_quality_score": 0.5}) == pytest.approx(0.5)


# --------------------------------------------------------------------------- #
# Safety: never import anything hardware-/real-motion-capable.
# --------------------------------------------------------------------------- #
def test_never_imports_hardware_modules() -> None:
    source = Path(atg.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
    forbidden = {name for name in imported if name == "hardware" or name.startswith("hardware.")}
    assert not forbidden, f"tools/ur5e_auto_tune_gains.py must never import hardware.*, found: {forbidden}"


def test_module_never_calls_subprocess_against_hardware_scripts() -> None:
    # The only script this module is allowed to subprocess is the sim gate --
    # find every subprocess.run(...) call site via AST and confirm each one's
    # first positional argument is built from MOVE_HOLD_SCRIPT, never a
    # literal path to a real-motion-capable script like
    # ur5e_direct_torque_x_transport.py (which appears elsewhere in this file
    # only as a printed string for a human to copy, never as a subprocess
    # target).
    source = Path(atg.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    call_sites = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "run"
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "subprocess"
    ]
    assert len(call_sites) == 1, f"expected exactly one subprocess.run(...) call site, found {len(call_sites)}"
    first_arg = call_sites[0].args[0]
    assert isinstance(first_arg, ast.Name) and first_arg.id == "cmd", (
        "the sole subprocess.run(...) call must take a locally-built cmd list "
        "(from build_sim_gate_command), not a literal command"
    )
    assert atg.MOVE_HOLD_SCRIPT.name == "ur5e_move_hold_transport.py"


# --------------------------------------------------------------------------- #
# Batch diversity: study.ask() without study.tell() must not degenerate.
# --------------------------------------------------------------------------- #
def test_batch_ask_without_tell_produces_distinct_candidates_cold_start():
    search_space = {"kp_x": (200.0, 800.0), "kd_x": (20.0, 80.0), "kd_joint": (2.0, 10.0)}
    study, _ = sg.build_study(search_space, [], seed=0)
    candidates = atg.propose_batch(study, search_space, batch_size=6)
    assert len(candidates) == 6
    distinct = {tuple(sorted(c.items())) for c in candidates}
    assert len(distinct) == 6, "cold-start batch ask() produced duplicate candidates"


def test_batch_ask_without_tell_produces_distinct_candidates_warm_start():
    # >= TPESampler's default n_startup_trials (10) prior COMPLETE trials, so
    # the TPE model is already fit before any ask() in this batch runs.
    search_space = {"kp_x": (200.0, 800.0), "kd_x": (20.0, 80.0), "kd_joint": (2.0, 10.0)}
    summaries = []
    for i in range(15):
        frac = i / 14.0
        summaries.append(
            {
                "gain_overrides": {
                    "kp_x": 200.0 + frac * 600.0,
                    "kd_x": 20.0 + frac * 60.0,
                    "kd_joint": 2.0 + frac * 8.0,
                },
                "valid_move_and_hold": True,
                "move_hold_quality_score": frac,
            }
        )
    study, _ = sg.build_study(search_space, summaries, seed=0)
    assert len(study.trials) == 15
    candidates = atg.propose_batch(study, search_space, batch_size=6)
    distinct = {tuple(sorted(c.items())) for c in candidates}
    assert len(distinct) == 6, "warm-start batch ask() produced duplicate candidates"


# --------------------------------------------------------------------------- #
# validate_sim_target_x_deltas
# --------------------------------------------------------------------------- #
def test_validate_sim_target_x_deltas_accepts_proven_range():
    atg.validate_sim_target_x_deltas([0.10, 0.15, 0.20])


def test_validate_sim_target_x_deltas_rejects_out_of_range():
    with pytest.raises(ValueError, match="proven-safe range"):
        atg.validate_sim_target_x_deltas([0.05])
    with pytest.raises(ValueError, match="proven-safe range"):
        atg.validate_sim_target_x_deltas([0.25])


# --------------------------------------------------------------------------- #
# build_sim_gate_command
# --------------------------------------------------------------------------- #
def test_build_sim_gate_command_shape(tmp_path: Path):
    config_path = tmp_path / "config.yaml"
    output_root = tmp_path / "out"
    gains = {"kp_x": 400.0, "kd_x": 40.0}
    cmd = atg.build_sim_gate_command(
        config=config_path,
        output_root=output_root,
        seed=3,
        gain_overrides=gains,
        target_x_deltas=[0.10, 0.20],
        move_duration=1.0,
        hold_duration=2.0,
        torque_limit_scale=1.0,
    )
    assert cmd[0] == sys.executable
    assert cmd[1] == str(atg.MOVE_HOLD_SCRIPT)
    assert "--config" in cmd and str(config_path) in cmd
    assert "--output-root" in cmd and str(output_root) in cmd
    assert "--seed" in cmd and "3" in cmd
    assert "--no-plot" in cmd

    idx = cmd.index("--target-x-deltas")
    assert cmd[idx + 1 : idx + 3] == ["0.1", "0.2"]

    idx = cmd.index("--gain-overrides-json")
    assert json.loads(cmd[idx + 1]) == gains


def test_build_sim_gate_command_forwards_start_q(tmp_path: Path):
    cmd = atg.build_sim_gate_command(
        config=tmp_path / "config.yaml",
        output_root=tmp_path / "out",
        seed=0,
        gain_overrides={"kp_x": 400.0},
        target_x_deltas=[0.15],
        move_duration=1.0,
        hold_duration=2.0,
        torque_limit_scale=1.0,
        start_q_rad=[0.0, -1.4, -0.24, -1.45, 0.0, 0.0],
    )
    idx = cmd.index("--start-q-rad")
    forwarded = [float(v) for v in cmd[idx + 1 : idx + 7]]
    assert forwarded == pytest.approx([0.0, -1.4, -0.24, -1.45, 0.0, 0.0])


# --------------------------------------------------------------------------- #
# run_sim_gate: mocked subprocess.run
# --------------------------------------------------------------------------- #
def test_run_sim_gate_pass(tmp_path: Path, monkeypatch):
    output_root = tmp_path / "candidate_01"

    def fake_run(cmd, cwd=None, check=False, text=True, capture_output=True):
        rows = [
            {"valid_move_and_hold": True, "move_hold_quality_score": 0.8},
            {"valid_move_and_hold": True, "move_hold_quality_score": 0.7},
        ]
        _write_summary(output_root / "summary.json", _fake_move_hold_summary(rows))
        return subprocess.CompletedProcess(cmd, returncode=0, stdout="", stderr="")

    monkeypatch.setattr(atg.subprocess, "run", fake_run)
    result = atg.run_sim_gate(
        candidate_gains={"kp_x": 400.0},
        candidate_output_root=output_root,
        config=Path("config/ur5e_mujoco_torque_osc_tuned.yaml"),
        seed=0,
        target_x_deltas=[0.10, 0.20],
        move_duration=1.0,
        hold_duration=2.0,
        torque_limit_scale=1.0,
    )
    assert result["sim_pass"] is True
    assert result["num_valid_move_and_hold"] == 2
    assert result["num_runs"] == 2
    assert result["gate_score"] == pytest.approx(0.75)


def test_run_sim_gate_fail_when_any_dx_point_invalid(tmp_path: Path, monkeypatch):
    output_root = tmp_path / "candidate_02"

    def fake_run(cmd, cwd=None, check=False, text=True, capture_output=True):
        rows = [
            {"valid_move_and_hold": True, "move_hold_quality_score": 0.9},
            {"valid_move_and_hold": False, "move_hold_quality_score": 0.99},  # guard tripped
        ]
        _write_summary(output_root / "summary.json", _fake_move_hold_summary(rows))
        return subprocess.CompletedProcess(cmd, returncode=0, stdout="", stderr="")

    monkeypatch.setattr(atg.subprocess, "run", fake_run)
    result = atg.run_sim_gate(
        candidate_gains={"kp_x": 400.0},
        candidate_output_root=output_root,
        config=Path("config/ur5e_mujoco_torque_osc_tuned.yaml"),
        seed=0,
        target_x_deltas=[0.10, 0.20],
        move_duration=1.0,
        hold_duration=2.0,
        torque_limit_scale=1.0,
    )
    assert result["sim_pass"] is False
    assert result["num_valid_move_and_hold"] == 1
    assert result["num_runs"] == 2


def test_run_sim_gate_fail_on_subprocess_error(tmp_path: Path, monkeypatch):
    output_root = tmp_path / "candidate_03"

    def fake_run(cmd, cwd=None, check=False, text=True, capture_output=True):
        return subprocess.CompletedProcess(cmd, returncode=1, stdout="boom", stderr="traceback")

    monkeypatch.setattr(atg.subprocess, "run", fake_run)
    result = atg.run_sim_gate(
        candidate_gains={"kp_x": 400.0},
        candidate_output_root=output_root,
        config=Path("config/ur5e_mujoco_torque_osc_tuned.yaml"),
        seed=0,
        target_x_deltas=[0.10],
        move_duration=1.0,
        hold_duration=2.0,
        torque_limit_scale=1.0,
    )
    assert result["sim_pass"] is False
    assert result["gate_score"] == -1.0
    assert "error" in result


# --------------------------------------------------------------------------- #
# End-to-end main(): mocked subprocess.run, batch with mixed pass/fail.
# --------------------------------------------------------------------------- #
def test_main_only_prints_sim_passing_candidates(tmp_path: Path, monkeypatch, capsys):
    trials_dir = tmp_path / "trials"
    sim_output_root = tmp_path / "sim_out"

    call_count = {"n": 0}

    def fake_run(cmd, cwd=None, check=False, text=True, capture_output=True):
        call_count["n"] += 1
        # Even-numbered candidates pass, odd fail -- forces a mixed batch.
        idx = call_count["n"]
        output_root_arg = Path(cmd[cmd.index("--output-root") + 1])
        valid = (idx % 2) == 0
        rows = [{"valid_move_and_hold": valid, "move_hold_quality_score": 0.6 if valid else 0.1}]
        _write_summary(output_root_arg / "summary.json", _fake_move_hold_summary(rows))
        return subprocess.CompletedProcess(cmd, returncode=0, stdout="", stderr="")

    monkeypatch.setattr(atg.subprocess, "run", fake_run)

    argv = [
        "--trials-dir", str(trials_dir),
        "--search-space-json", '{"kp_x": [200, 800], "kd_x": [20, 80]}',
        "--seed", "0",
        "--batch-size", "4",
        "--sim-output-root", str(sim_output_root),
        "--sim-target-x-deltas", "0.15",
        "--yes",
    ]
    rc = atg.main(argv)
    assert rc == 0
    out = capsys.readouterr().out
    assert call_count["n"] == 4
    assert "Sim gate results: 2/4 candidates passed." in out
    assert "Approved-for-real-hardware batch" in out
    assert "Ready-to-copy real-hardware commands" in out


def test_main_zero_sim_passes_exits_cleanly_without_confirmation_prompt(tmp_path: Path, monkeypatch, capsys):
    trials_dir = tmp_path / "trials"
    sim_output_root = tmp_path / "sim_out"

    def fake_run(cmd, cwd=None, check=False, text=True, capture_output=True):
        output_root_arg = Path(cmd[cmd.index("--output-root") + 1])
        rows = [{"valid_move_and_hold": False, "move_hold_quality_score": 0.9}]
        _write_summary(output_root_arg / "summary.json", _fake_move_hold_summary(rows))
        return subprocess.CompletedProcess(cmd, returncode=0, stdout="", stderr="")

    monkeypatch.setattr(atg.subprocess, "run", fake_run)

    def fail_input(prompt=""):
        raise AssertionError("must not prompt for confirmation when zero candidates pass the sim gate")

    monkeypatch.setattr("builtins.input", fail_input)

    argv = [
        "--trials-dir", str(trials_dir),
        "--search-space-json", '{"kp_x": [200, 800]}',
        "--seed", "0",
        "--batch-size", "3",
        "--sim-output-root", str(sim_output_root),
        "--sim-target-x-deltas", "0.15",
    ]
    rc = atg.main(argv)
    assert rc == 0
    out = capsys.readouterr().out
    assert "No candidates in this batch passed the sim gate" in out
    assert "Sim gate results: 0/3 candidates passed." in out


def test_main_declined_confirmation_prints_nothing_real(tmp_path: Path, monkeypatch, capsys):
    trials_dir = tmp_path / "trials"
    sim_output_root = tmp_path / "sim_out"

    def fake_run(cmd, cwd=None, check=False, text=True, capture_output=True):
        output_root_arg = Path(cmd[cmd.index("--output-root") + 1])
        rows = [{"valid_move_and_hold": True, "move_hold_quality_score": 0.5}]
        _write_summary(output_root_arg / "summary.json", _fake_move_hold_summary(rows))
        return subprocess.CompletedProcess(cmd, returncode=0, stdout="", stderr="")

    monkeypatch.setattr(atg.subprocess, "run", fake_run)
    monkeypatch.setattr("builtins.input", lambda prompt="": "nope")

    argv = [
        "--trials-dir", str(trials_dir),
        "--search-space-json", '{"kp_x": [200, 800]}',
        "--seed", "0",
        "--batch-size", "2",
        "--sim-output-root", str(sim_output_root),
        "--sim-target-x-deltas", "0.15",
        "--robot-ip", "10.0.0.5",
    ]
    rc = atg.main(argv)
    assert rc == 2
    out = capsys.readouterr().out
    assert "Ready-to-copy real-hardware commands" not in out
    assert "10.0.0.5" not in out


def test_main_prints_real_command_with_robot_ip(tmp_path: Path, monkeypatch, capsys):
    trials_dir = tmp_path / "trials"
    sim_output_root = tmp_path / "sim_out"

    def fake_run(cmd, cwd=None, check=False, text=True, capture_output=True):
        output_root_arg = Path(cmd[cmd.index("--output-root") + 1])
        rows = [{"valid_move_and_hold": True, "move_hold_quality_score": 0.5}]
        _write_summary(output_root_arg / "summary.json", _fake_move_hold_summary(rows))
        return subprocess.CompletedProcess(cmd, returncode=0, stdout="", stderr="")

    monkeypatch.setattr(atg.subprocess, "run", fake_run)

    argv = [
        "--trials-dir", str(trials_dir),
        "--search-space-json", '{"kp_x": [200, 800]}',
        "--seed", "0",
        "--batch-size", "1",
        "--sim-output-root", str(sim_output_root),
        "--sim-target-x-deltas", "0.15",
        "--robot-ip", "10.0.0.5",
        "--yes",
    ]
    rc = atg.main(argv)
    assert rc == 0
    out = capsys.readouterr().out
    assert "tools/ur5e_direct_torque_x_transport.py" in out
    assert "--robot-ip 10.0.0.5" in out
    assert "--i-understand-this-moves-the-robot" in out
    assert "--yes" not in out.split("Ready-to-copy")[1]  # never auto-skips the real script's own MOVE prompt
