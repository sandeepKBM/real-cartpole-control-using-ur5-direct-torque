"""Regression coverage for the BASELINE_GAINS silent-substitution footgun found during
the height_alpha=0.2/0.3 validation sweep (docs/status/bug_audit_2026-07-29.md): the
driver used to silently discard a named --config's own gains and substitute a hardcoded
constant unless every field was re-specified via --gain-overrides-json."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "tools"))

from ur5e_move_hold_transport import BASELINE_GAINS, _resolve_candidate_gains  # noqa: E402

CONFIG_GAINS = {
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
}


def test_default_uses_the_named_configs_own_gains_untouched() -> None:
    resolved = _resolve_candidate_gains(
        CONFIG_GAINS, use_legacy_baseline_gains=False, gain_overrides_json=None
    )
    assert resolved == CONFIG_GAINS
    # Specifically confirm the config's real kp_x/kp_rot survive rather than being
    # silently replaced by BASELINE_GAINS' kp_x=80.0/kp_rot=30.0.
    assert resolved["kp_x"] == 400.0
    assert resolved["kp_rot"] == 0.0


def test_legacy_flag_reproduces_old_baseline_gains_substitution() -> None:
    resolved = _resolve_candidate_gains(
        CONFIG_GAINS, use_legacy_baseline_gains=True, gain_overrides_json=None
    )
    assert resolved == BASELINE_GAINS


def test_gain_overrides_json_applies_on_top_of_the_configs_own_gains() -> None:
    resolved = _resolve_candidate_gains(
        CONFIG_GAINS,
        use_legacy_baseline_gains=False,
        gain_overrides_json='{"kp_x": 999.0}',
    )
    assert resolved["kp_x"] == 999.0
    # Everything else still comes from the config, not BASELINE_GAINS.
    assert resolved["kp_rot"] == 0.0
    assert resolved["kp_posture"] == 25.0


def test_gain_overrides_json_applies_on_top_of_legacy_baseline_gains() -> None:
    resolved = _resolve_candidate_gains(
        CONFIG_GAINS,
        use_legacy_baseline_gains=True,
        gain_overrides_json='{"kp_x": 999.0}',
    )
    assert resolved["kp_x"] == 999.0
    assert resolved["kp_rot"] == BASELINE_GAINS["kp_rot"]
