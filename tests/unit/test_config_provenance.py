"""Tests for the machine-enforced config <-> pose pairing.

The centrepiece is test_the_real_regression_*: it reproduces the ACTUAL bad
dispatch that cost a ~20 h differential-evolution search plus five
capture-envelope grids, using the REAL shipped config file and the REAL pose
those runs were launched with. Per AGENTS.md sec.7 ("A TEST THAT NEVER EXERCISES
THE PATH IS NOT COVERAGE"), the point is that this test fails if anyone weakens
the guard, deletes the provenance block, or edits the declared pose to make the
mismatch disappear -- not merely that the helper works on synthetic input.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from controller_core.config_provenance import (
    ConfigPoseMismatchError,
    ConfigProvenance,
    STRICT_ENV_VAR,
    check_config_pose,
    describe_provenance,
    parse_provenance,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "config"

# The pose every balance config declares: ARM_Q0 with wrist_2 = -90 deg. Same
# constant as x_task_orientation_cbf_balance_gain_search.py::ARM_Q_W2M90 (the
# script that measured those configs' Lambda) and
# pendulum_flip_catch_hold.py::ARM_Q_W2NEG90.
DECLARED_Q = [-2.3688, -2.1801, -1.8838, -0.7962, -1.5707963, 0.0206]
# The pose the killed runs actually dispatched at: the SINGULAR ARM_Q0.
DISPATCHED_Q = [-2.3688, -2.1801, -1.8838, -0.7962, 0.004714693, 0.0206]

BALANCE_CONFIGS = [
    "ur5e_mujoco_torque_x_task_yz_corridor_qp_orientation_cbf_balance.yaml",
    "ur5e_mujoco_torque_x_task_yz_corridor_qp_balance_kprot0.yaml",
    "ur5e_mujoco_torque_x_task_yz_corridor_qp_orientcbf_panfree.yaml",
]


def load_yaml(name: str) -> dict:
    return yaml.safe_load((CONFIG_DIR / name).read_text())


# ---------------------------------------------------------------------------
# The real regression.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", BALANCE_CONFIGS)
def test_the_real_regression_singular_armq0_is_refused(name):
    """The exact (config, pose, asset) triple of the killed 20 h search."""
    with pytest.raises(ConfigPoseMismatchError) as exc:
        check_config_pose(
            load_yaml(name),
            DISPATCHED_Q,
            "pendulum_attachment.xml",
            config_name=name,
        )
    msg = str(exc.value)
    # The message must name the offending joint and quantify it -- a bare
    # "mismatch" would not have told anyone which of six joints was wrong.
    assert "wrist_2" in msg
    assert "90.27" in msg          # degrees, so the size is obvious at a glance
    assert "ASSET MISMATCH" in msg  # both problems reported, not just the first


@pytest.mark.parametrize("name", BALANCE_CONFIGS)
def test_balance_configs_declare_the_pose_their_gains_were_measured_at(name):
    prov = parse_provenance(load_yaml(name), source=name)
    assert prov.declared, f"{name} lost its provenance block"
    assert np.allclose(prov.arm_q_rad, DECLARED_Q)
    assert prov.pendulum_xml == "pendulum_attachment_realrod.xml"
    assert prov.notes, f"{name} declares a pose but no derivation note"


@pytest.mark.parametrize("name", BALANCE_CONFIGS)
def test_correct_dispatch_is_not_blocked(name):
    """The guard must not false-positive on the case it was written to protect."""
    prov = check_config_pose(
        load_yaml(name),
        DECLARED_Q,
        "pendulum_attachment_realrod.xml",
        config_name=name,
    )
    assert prov.declared
    assert prov.mismatches == ()


@pytest.mark.parametrize("name", BALANCE_CONFIGS)
def test_configs_still_parse_as_controller_configs(name):
    """The provenance block must not disturb what the rest of the stack reads."""
    cfg = load_yaml(name)
    assert {"mujoco", "controller", "reward"} <= set(cfg)
    assert cfg["controller"]["gains"]["kp_x"] > 0


# ---------------------------------------------------------------------------
# Guard semantics.
# ---------------------------------------------------------------------------

def _declared_cfg(**overrides) -> dict:
    derived = {
        "arm_q_rad": list(DECLARED_Q),
        "pendulum_xml": "pendulum_attachment_realrod.xml",
    }
    derived.update(overrides)
    return {"provenance": {"derived_for": derived, "notes": "test"}}


def test_asset_mismatch_alone_raises_even_at_the_right_pose():
    with pytest.raises(ConfigPoseMismatchError, match="ASSET MISMATCH"):
        check_config_pose(_declared_cfg(), DECLARED_Q, "pendulum_attachment.xml")


def test_pose_mismatch_alone_raises_even_with_the_right_asset():
    with pytest.raises(ConfigPoseMismatchError, match="POSE MISMATCH"):
        check_config_pose(
            _declared_cfg(), DISPATCHED_Q, "pendulum_attachment_realrod.xml"
        )


def test_tolerance_is_tight_enough_to_catch_a_small_pose_drift():
    """1e-6 rad means "the same pose", not "a nearby pose"."""
    nudged = list(DECLARED_Q)
    nudged[1] += 1e-4
    with pytest.raises(ConfigPoseMismatchError):
        check_config_pose(
            _declared_cfg(), nudged, "pendulum_attachment_realrod.xml"
        )


def test_allow_mismatch_returns_the_mismatch_instead_of_raising():
    """Off-provenance runs must stay distinguishable in their own output."""
    prov = check_config_pose(
        _declared_cfg(),
        DISPATCHED_Q,
        "pendulum_attachment.xml",
        allow_mismatch=True,
    )
    assert prov.mismatches, "an allowed mismatch must still be recorded"
    assert any("POSE MISMATCH" in m for m in prov.mismatches)
    assert any("ASSET MISMATCH" in m for m in prov.mismatches)
    assert "OFF-PROVENANCE" in describe_provenance(prov)
    assert prov.as_dict()["mismatches"], "must survive into the run's JSON"


def test_allow_mismatch_cannot_be_granted_by_the_config_itself():
    """The failure guarded against is a config being trusted about its own
    applicability, so no config field may unlock the guard."""
    cfg = _declared_cfg()
    cfg["provenance"]["allow_mismatch"] = True
    cfg["provenance"]["derived_for"]["allow_mismatch"] = True
    with pytest.raises(ConfigPoseMismatchError):
        check_config_pose(cfg, DISPATCHED_Q, "pendulum_attachment_realrod.xml")


def test_asset_check_compares_basenames_not_full_paths():
    check_config_pose(
        _declared_cfg(),
        DECLARED_Q,
        "/somewhere/else/assets/ur5e_pendulum/pendulum_attachment_realrod.xml",
    )


def test_pose_only_provenance_ignores_the_asset():
    cfg = {"provenance": {"derived_for": {"arm_q_rad": list(DECLARED_Q)}}}
    prov = check_config_pose(cfg, DECLARED_Q, "anything_at_all.xml")
    assert prov.declared


# ---------------------------------------------------------------------------
# Undeclared configs: permitted by default, hard error under strict mode.
# ---------------------------------------------------------------------------

def test_undeclared_config_is_permitted_by_default():
    prov = check_config_pose({"controller": {}}, DECLARED_Q, "whatever.xml")
    assert not prov.declared
    assert "UNDECLARED" in describe_provenance(prov)


def test_undeclared_config_is_an_error_under_strict_mode(monkeypatch):
    monkeypatch.setenv(STRICT_ENV_VAR, "1")
    with pytest.raises(ConfigPoseMismatchError, match="declares no"):
        check_config_pose({"controller": {}}, DECLARED_Q, "whatever.xml")


def test_strict_mode_off_values_are_not_treated_as_on(monkeypatch):
    for value in ("", "0", "false"):
        monkeypatch.setenv(STRICT_ENV_VAR, value)
        assert not check_config_pose({}, DECLARED_Q).declared


def test_explicit_strict_argument_overrides_the_environment(monkeypatch):
    monkeypatch.setenv(STRICT_ENV_VAR, "1")
    assert not check_config_pose({}, DECLARED_Q, strict_undeclared=False).declared


# ---------------------------------------------------------------------------
# Malformed input fails loudly rather than silently disabling the guard.
# ---------------------------------------------------------------------------

def test_non_mapping_provenance_is_rejected():
    with pytest.raises(ConfigPoseMismatchError, match="must be a mapping"):
        parse_provenance({"provenance": ["not", "a", "mapping"]})


def test_non_mapping_derived_for_is_rejected():
    with pytest.raises(ConfigPoseMismatchError, match="derived_for"):
        parse_provenance({"provenance": {"derived_for": "ARM_Q0"}})


def test_wrong_length_declared_pose_is_rejected():
    with pytest.raises(ConfigPoseMismatchError, match="6"):
        parse_provenance({"provenance": {"derived_for": {"arm_q_rad": [0.0, 1.0]}}})


def test_wrong_length_actual_pose_is_rejected():
    with pytest.raises(ConfigPoseMismatchError, match="6 entries"):
        check_config_pose(_declared_cfg(), [0.0, 1.0, 2.0])


def test_empty_config_is_undeclared_not_a_crash():
    assert not parse_provenance({}).declared
    assert not parse_provenance(None).declared


def test_provenance_is_serialisable_for_run_records():
    prov = ConfigProvenance(
        arm_q_rad=np.asarray(DECLARED_Q), pendulum_xml="a.xml", notes="n"
    )
    d = prov.as_dict()
    assert d["arm_q_rad"] == pytest.approx(DECLARED_Q)
    assert isinstance(d["arm_q_rad"], list)  # not an ndarray -> json-safe
