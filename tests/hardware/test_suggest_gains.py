"""Tests for tools/ur5e_suggest_gains.py -- the real-hardware retune helper.

Pure local-file + Optuna logic, no RTDE/robot involved at all, but lives
under tests/hardware/ since it's part of the hardware retuning workflow
(matching the existing test_urscript_gen.py precedent for tools/ scripts that
don't literally open a socket)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "tools"))

import ur5e_suggest_gains as sg  # noqa: E402


def _write_summary(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


# --------------------------------------------------------------------------- #
# score_trial
# --------------------------------------------------------------------------- #
def test_score_trial_uses_quality_score_when_valid():
    assert sg.score_trial({"valid_move_and_hold": True, "move_hold_quality_score": 0.72}) == pytest.approx(0.72)


def test_score_trial_penalizes_invalid_regardless_of_quality_score():
    # Even a high raw quality_score must not outrank a completed trial if the
    # run itself didn't validly finish (guard tripped mid-run, etc.).
    assert sg.score_trial({"valid_move_and_hold": False, "move_hold_quality_score": 0.99}) == -1.0


def test_score_trial_defaults_to_invalid_when_field_missing():
    assert sg.score_trial({}) == -1.0


# --------------------------------------------------------------------------- #
# _parse_search_space
# --------------------------------------------------------------------------- #
def test_parse_search_space_inline_json():
    space = sg._parse_search_space('{"kp_x": [200, 800], "kd_joint": [2, 10]}')
    assert space == {"kp_x": (200.0, 800.0), "kd_joint": (2.0, 10.0)}


def test_parse_search_space_from_file(tmp_path: Path):
    f = tmp_path / "space.json"
    f.write_text('{"kd_x": [10, 90]}', encoding="utf-8")
    assert sg._parse_search_space(str(f)) == {"kd_x": (10.0, 90.0)}


def test_parse_search_space_rejects_unknown_gain():
    with pytest.raises(ValueError, match="not a schedulable gain field"):
        sg._parse_search_space('{"not_a_real_gain": [0, 1]}')


def test_parse_search_space_rejects_bad_bounds():
    with pytest.raises(ValueError, match="low < high"):
        sg._parse_search_space('{"kp_x": [800, 200]}')


def test_parse_search_space_rejects_empty():
    with pytest.raises(ValueError, match="non-empty"):
        sg._parse_search_space("{}")


# --------------------------------------------------------------------------- #
# _load_trial_summaries
# --------------------------------------------------------------------------- #
def test_load_trial_summaries_finds_nested_files(tmp_path: Path):
    _write_summary(tmp_path / "a" / "summary.json", {"x": 1})
    _write_summary(tmp_path / "b" / "c" / "summary.json", {"x": 2})
    summaries = sg._load_trial_summaries(tmp_path)
    assert len(summaries) == 2


def test_load_trial_summaries_skips_unreadable(tmp_path: Path):
    good = tmp_path / "good"
    good.mkdir()
    (good / "summary.json").write_text('{"x": 1}', encoding="utf-8")
    bad = tmp_path / "bad"
    bad.mkdir()
    (bad / "summary.json").write_text("not valid json{{{", encoding="utf-8")
    summaries = sg._load_trial_summaries(tmp_path)
    assert len(summaries) == 1


def test_load_trial_summaries_empty_dir(tmp_path: Path):
    assert sg._load_trial_summaries(tmp_path) == []


# --------------------------------------------------------------------------- #
# build_study
# --------------------------------------------------------------------------- #
def test_build_study_skips_trials_missing_search_space_gains():
    search_space = {"kp_x": (200.0, 800.0), "kd_x": (20.0, 80.0)}
    summaries = [
        {"gain_overrides": {"kp_x": 400.0}, "valid_move_and_hold": True, "move_hold_quality_score": 0.5},  # missing kd_x
        {"gain_overrides": {"kp_x": 400.0, "kd_x": 40.0}, "valid_move_and_hold": True, "move_hold_quality_score": 0.6},
    ]
    study, skipped = sg.build_study(search_space, summaries, seed=0)
    assert len(study.trials) == 1
    assert skipped == 1


def test_build_study_skips_out_of_range_trials():
    search_space = {"kp_x": (200.0, 800.0)}
    summaries = [
        {"gain_overrides": {"kp_x": 1000.0}, "valid_move_and_hold": True, "move_hold_quality_score": 0.9},  # out of bounds
    ]
    study, skipped = sg.build_study(search_space, summaries, seed=0)
    assert len(study.trials) == 0
    assert skipped == 1


def test_build_study_best_trial_matches_highest_score():
    search_space = {"kp_x": (200.0, 800.0)}
    summaries = [
        {"gain_overrides": {"kp_x": 400.0}, "valid_move_and_hold": True, "move_hold_quality_score": 0.5},
        {"gain_overrides": {"kp_x": 600.0}, "valid_move_and_hold": True, "move_hold_quality_score": 0.9},
        {"gain_overrides": {"kp_x": 750.0}, "valid_move_and_hold": False, "move_hold_quality_score": 0.99},
    ]
    study, _ = sg.build_study(search_space, summaries, seed=0)
    assert len(study.trials) == 3
    assert study.best_trial.params == {"kp_x": 600.0}
    assert study.best_value == pytest.approx(0.9)


# --------------------------------------------------------------------------- #
# End-to-end: main() prints an in-bounds suggestion
# --------------------------------------------------------------------------- #
def test_main_suggests_in_bounds_candidate(tmp_path: Path, capsys):
    _write_summary(
        tmp_path / "trial1" / "summary.json",
        {"gain_overrides": {"kp_x": 400.0, "kd_joint": 4.0}, "valid_move_and_hold": True, "move_hold_quality_score": 0.7},
    )
    argv = sys.argv
    sys.argv = [
        "ur5e_suggest_gains.py",
        "--trials-dir", str(tmp_path),
        "--search-space-json", '{"kp_x": [200, 800], "kd_joint": [2, 10]}',
        "--seed", "0",
    ]
    try:
        rc = sg.main()
    finally:
        sys.argv = argv
    assert rc == 0
    out = capsys.readouterr().out
    assert "Next candidate to try:" in out
    line = [l for l in out.splitlines() if "--gain-overrides-json" in l][0]
    payload = json.loads(line.split("--gain-overrides-json ", 1)[1].strip().strip("'"))
    assert 200.0 <= payload["kp_x"] <= 800.0
    assert 2.0 <= payload["kd_joint"] <= 10.0


def test_main_with_no_prior_trials_still_suggests(tmp_path: Path, capsys):
    argv = sys.argv
    sys.argv = [
        "ur5e_suggest_gains.py",
        "--trials-dir", str(tmp_path),
        "--search-space-json", '{"kp_x": [200, 800]}',
        "--seed", "1",
    ]
    try:
        rc = sg.main()
    finally:
        sys.argv = argv
    assert rc == 0
    out = capsys.readouterr().out
    assert "Loaded 0 trial summaries" in out
    assert "Next candidate to try:" in out
