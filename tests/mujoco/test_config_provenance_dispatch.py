"""The provenance guard, exercised through the PRODUCTION dispatch path.

tests/unit/test_config_provenance.py covers check_config_pose directly. That is
not enough on its own: the bug this guards was a CLI dispatch, and a guard that
works in isolation but is never reached from the command line would be exactly
the "test that never exercises the path" AGENTS.md sec.7 warns about. These
tests therefore go through build_parser() -> context_from_args(), the same two
calls tools/diagnostics/pendulum_lqr_cascade.py::main makes.

Marked mujoco because importing the pendulum tooling pulls the simulator in,
not because the check itself needs a model.
"""

from __future__ import annotations

import numpy as np
import pytest

from controller_core.config_provenance import ConfigPoseMismatchError
from tools.diagnostics.pendulum_lqr_cascade import build_parser
from tools.diagnostics.pendulum_swingup_energy_shaping import (
    context_from_args,
    describe_context,
)

BALANCE_CONFIG = (
    "config/ur5e_mujoco_torque_x_task_yz_corridor_qp_orientation_cbf_balance.yaml"
)
REALROD = "assets/ur5e_pendulum/pendulum_attachment_realrod.xml"
DEFAULT_ASSET = "assets/ur5e_pendulum/pendulum_attachment.xml"
W2_POSTURE_CONFIG = (
    "config/ur5e_mujoco_torque_x_task_yz_corridor_qp_wrist2_posture.yaml"
)

DECLARED_Q = ["-2.3688", "-2.1801", "-1.8838", "-0.7962", "-1.5707963", "0.0206"]
SINGULAR_Q = ["-2.3688", "-2.1801", "-1.8838", "-0.7962", "0.004714693", "0.0206"]


def argv_for(pose, asset, *extra):
    return [
        "--pendulum-xml", asset,
        "--start-q-rad", *pose,
        "--controller-kind", "x_task_yz_corridor_qp",
        "--config", BALANCE_CONFIG,
        *extra,
    ]


def test_the_killed_runs_command_line_is_now_refused():
    """Verbatim the dispatch of the ~20 h search and the five envelope grids."""
    args = build_parser().parse_args(argv_for(SINGULAR_Q, DEFAULT_ASSET))
    with pytest.raises(ConfigPoseMismatchError) as exc:
        context_from_args(args)
    assert "wrist_2" in str(exc.value)


def test_the_declared_case_still_dispatches():
    args = build_parser().parse_args(argv_for(DECLARED_Q, REALROD))
    ctx = context_from_args(args)
    assert ctx.provenance is not None
    assert ctx.provenance.declared
    assert ctx.provenance.mismatches == ()
    assert np.allclose(ctx.arm_q_array, [float(v) for v in DECLARED_Q])


def test_allow_pose_mismatch_flag_permits_and_records_the_mismatch():
    args = build_parser().parse_args(
        argv_for(SINGULAR_Q, DEFAULT_ASSET, "--allow-pose-mismatch")
    )
    ctx = context_from_args(args)
    assert ctx.provenance.mismatches, "an allowed mismatch must still be recorded"
    # and it must be visible in the run header, not silently swallowed
    assert "OFF-PROVENANCE" in describe_context(ctx.resolve())


def test_allow_pose_mismatch_defaults_to_off():
    args = build_parser().parse_args(argv_for(SINGULAR_Q, DEFAULT_ASSET))
    assert args.allow_pose_mismatch is False


def test_undeclared_config_still_dispatches_unchanged():
    """The guard must not have broken the configs that declare nothing --
    that is most of config/, including every swing-up run."""
    args = build_parser().parse_args([
        "--pendulum-xml", REALROD,
        "--start-q-rad", *DECLARED_Q,
        "--config", "config/ur5e_mujoco_torque_osc_tuned_friction_ff.yaml",
    ])
    ctx = context_from_args(args)
    assert ctx.provenance is not None
    assert not ctx.provenance.declared


def test_curved_tool_does_not_bypass_the_guard_with_its_own_context_builder():
    """tools/diagnostics/pendulum_swingup_curved.py defines a LOCAL
    context_from_args that shadows the shared one. A script that builds its own
    context silently opts out of every guard the shared path added, which is
    exactly how a config reaches the wrong pose -- so the local builder must
    perform the same check."""
    from tools.diagnostics import pendulum_swingup_curved as curved

    parser = curved.build_parser()
    args = parser.parse_args([
        "--pendulum-xml", DEFAULT_ASSET,
        "--start-q-rad", *SINGULAR_Q,
        "--config", BALANCE_CONFIG,
    ])
    with pytest.raises(ConfigPoseMismatchError):
        curved.context_from_args(args)


def test_curved_tool_default_context_is_constructible():
    """Regression: default_context() referenced a bare `controller_kind` that
    exists nowhere in that scope, so it raised NameError on every call and took
    six tests down with it."""
    from tools.diagnostics import pendulum_swingup_curved as curved

    ctx = curved.default_context()
    assert ctx.controller_kind == curved.CONTROLLER_KIND
    assert ctx.controller_kind


def test_flip_catch_hold_refuses_a_config_from_another_pose(tmp_path):
    """The one entrypoint that drives BOTH halves of Goal 1 predates the shared
    context helper, so it carries its own check. Verified through main(), not by
    reading the source -- the guard has to actually be reached."""
    import json

    from tools.diagnostics import pendulum_flip_catch_hold as fch

    swing = tmp_path / "swing.json"
    swing.write_text(json.dumps({"best_params": {
        "k_e": 1.0, "a_max": 1.0, "k_pos": 1.0, "k_vel": 1.0,
        "kick_amplitude_m": 0.01, "kick_duration_s": 0.1,
    }}))
    lqr = tmp_path / "lqr.json"
    lqr.write_text(json.dumps({"lqr": {"K": [0.0, 0.0, 0.0, 0.0], "a_max": 1.0}}))

    argv = [
        "--swingup-json", str(swing),
        "--lqr-json", str(lqr),
        "--config", BALANCE_CONFIG,
        "--pendulum-xml", DEFAULT_ASSET,
        "--start-q-rad", *SINGULAR_Q,
    ]
    with pytest.raises(ConfigPoseMismatchError):
        fch.main(argv)


def test_run_header_reports_provenance_for_declared_configs():
    args = build_parser().parse_args(argv_for(DECLARED_Q, REALROD))
    header = describe_context(context_from_args(args).resolve())
    assert "provenance: declared" in header
    assert "pendulum_attachment_realrod.xml" in header


# --- the video renderer -------------------------------------------------
#
# render_two_phase_handoff_video.py was the last dispatch point with no
# provenance check, and it emits the artifact a reviewer is most likely to
# accept on sight. A video rendered from a config derived at another pose is
# the same defect as a mis-dispatched run, only harder to notice -- so the
# guard has to be reachable from THIS command line too, not just from the
# search tools'.

def _render_argv(pose, asset, kind="x_task_yz_corridor_qp"):
    return [
        "--pendulum-xml", asset,
        "--start-q-rad", *pose,
        "--config", W2_POSTURE_CONFIG,
        "--controller-kind", kind,
        "--schedule", "1", "2", "0.5", "0.1", "0.05", "0.02", "1.0",
        "--lqr-json", "unused.json",
        "--out", "unused.mp4",
    ]


def _render_check(argv):
    """The exact three calls the renderer's main() makes before it builds the model."""
    from pathlib import Path

    from controller_core.config_provenance import check_config_pose
    from tools.diagnostics.pendulum_swingup_energy_shaping import load_config
    from tools.diagnostics.render_two_phase_handoff_video import build_parser as rp

    args = rp().parse_args(argv)
    return check_config_pose(
        load_config(Path(args.config)),
        np.asarray(args.start_q_rad, dtype=np.float64),
        args.pendulum_xml,
        controller_kind=str(args.controller_kind),
        config_name=Path(args.config).name,
        allow_mismatch=bool(args.allow_pose_mismatch),
    )


def test_renderer_refuses_a_config_from_another_pose():
    with pytest.raises(ConfigPoseMismatchError, match="POSE MISMATCH"):
        _render_check(_render_argv(SINGULAR_Q, REALROD))


def test_renderer_refuses_a_config_from_another_asset():
    with pytest.raises(ConfigPoseMismatchError, match="ASSET MISMATCH"):
        _render_check(_render_argv(DECLARED_Q, DEFAULT_ASSET))


def test_renderer_refuses_a_config_from_another_controller():
    # The failure this catches is concrete: plain OSC has no joint-exclusion
    # mechanism, so rendering this config as `impedance` frees shoulder_pan.
    with pytest.raises(ConfigPoseMismatchError, match="CONTROLLER MISMATCH"):
        _render_check(_render_argv(DECLARED_Q, REALROD, kind="impedance"))


def test_renderer_accepts_the_declared_case():
    prov = _render_check(_render_argv(DECLARED_Q, REALROD))
    assert prov.declared
    assert prov.mismatches == ()


def test_renderer_allow_mismatch_records_rather_than_raises():
    # Only reachable with the explicit flag, and the mismatch is retained so
    # the frames can be stamped OFF-PROVENANCE.
    prov = _render_check(_render_argv(SINGULAR_Q, REALROD) + ["--allow-pose-mismatch"])
    assert prov.mismatches
    assert any("POSE MISMATCH" in m for m in prov.mismatches)


def test_renderer_allow_mismatch_defaults_off():
    from tools.diagnostics.render_two_phase_handoff_video import build_parser as rp

    assert rp().parse_args(_render_argv(DECLARED_Q, REALROD)).allow_pose_mismatch is False
