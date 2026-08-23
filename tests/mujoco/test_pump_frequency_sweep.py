"""Coverage for the energy-per-guard-budget frequency sweep.

The two properties worth protecting are the ones that make the sweep MEAN
anything: that every row really does spend the same displacement budget
(A = excursion*w^2/2, so A must scale as f^2), and that the script cannot be
run without a deliberate drive-axis choice. The second is not pedantry -- a pump
along the hinge exerts no torque, and AGENTS.md records that failure happening
twice in a form invisible to the config, the gains and the logs.
"""

from __future__ import annotations

import numpy as np
import pytest

from simulation.ur5e_pendulum_compose import (
    compose_ur5e_pendulum_model,
    derive_pendulum_constants,
)
from tools.diagnostics.pendulum_swingup_energy_shaping import resolve_equilibria
from tools.diagnostics.pump_frequency_sweep import (
    AXIS_NAMES,
    build_parser,
    main,
    measure_axis_couplings,
    run_pump_trial,
)

ARMQ0 = np.array([-2.3688, -2.1801, -1.8838, -0.7962, 0.004714693, 0.0206])
ASSET = "assets/ur5e_pendulum/pendulum_attachment.xml"
CONFIG = "config/ur5e_mujoco_torque_x_task_yz_corridor_qp_orientcbf_armq0.yaml"


@pytest.fixture(scope="module")
def rig():
    model = compose_ur5e_pendulum_model(pendulum_xml=ASSET)
    hanging, _ = resolve_equilibria(model, ARMQ0)
    constants = derive_pendulum_constants(model, ARMQ0)
    couplings = measure_axis_couplings(model, ARMQ0, hanging)
    return model, hanging, constants, couplings


def test_vertical_drive_has_no_authority_at_hanging(rig):
    """Physics check on the coupling measurement: pivot acceleration couples as
    sin(phi), so world Z must be ~0 AT HANGING (and maximal at horizontal).
    A nonzero value here would mean the measurement, not the pose, is wrong."""
    _, _, _, c = rig
    assert abs(c["world_Z"]) < 1e-9
    assert abs(c["world_X"]) > 1e-4
    assert abs(c["world_Y"]) > 1e-4


def test_no_single_world_axis_captures_full_authority(rig):
    """The hinge sits at ~45 deg in XY here, so the best world axis is well
    short of the optimal in-plane direction. This is the measured basis for
    preferring a rotated task frame over a transport_axis_index."""
    _, _, _, c = rig
    best = max(abs(c["world_X"]), abs(c["world_Y"]))
    optimal = float(np.hypot(c["world_X"], c["world_Y"]))
    assert best < 0.85 * optimal, "expected a real off-axis penalty at this pose"


def test_amplitude_scales_as_frequency_squared(rig):
    """The whole point of the sweep: constant displacement budget across rows."""
    model, hanging, constants, c = rig
    kw = dict(
        arm_q=ARMQ0, excursion_m=0.02, duration_s=0.3, config_path=CONFIG,
        controller_kind="x_task_yz_corridor_qp", hanging_angle=hanging,
        constants=constants, coupling_c0=c["world_X"], transport_axis_index=0,
    )
    lo = run_pump_trial(model, freq_hz=1.0, **kw)
    hi = run_pump_trial(model, freq_hz=2.0, **kw)
    assert hi["amplitude_mps2"] == pytest.approx(4.0 * lo["amplitude_mps2"], rel=1e-9)
    assert lo["excursion_m"] == hi["excursion_m"]


def test_trial_reports_energy_and_guard_utilisation(rig):
    model, hanging, constants, c = rig
    r = run_pump_trial(
        model, freq_hz=1.724, arm_q=ARMQ0, excursion_m=0.02, duration_s=0.5,
        config_path=CONFIG, controller_kind="x_task_yz_corridor_qp",
        hanging_angle=hanging, constants=constants, coupling_c0=c["world_X"],
        transport_axis_index=0,
    )
    assert 0.0 <= r["positive_work_fraction"] <= 1.0
    assert np.isfinite(r["delta_e_j"])
    assert r["guard_utilisation"] > 0.0, "a driven run must consume some budget"
    assert r["e_peak_over_e_top"] is not None


def test_refuses_to_run_without_a_deliberate_axis_choice(capsys):
    """No axis flag => exit 2 and no rollout. Guessing here is the documented
    way to burn an entire experiment invisibly."""
    rc = main([
        "--pendulum-xml", ASSET,
        "--start-q-rad", *map(str, ARMQ0),
        "--config", CONFIG,
        "--controller-kind", "x_task_yz_corridor_qp",
        "--freqs-hz", "1.0",
    ])
    assert rc == 2
    assert "REFUSING TO GUESS" in capsys.readouterr().out


def test_auto_axis_picks_the_largest_coupling_and_says_so(rig, capsys):
    _, _, _, c = rig
    rc = main([
        "--pendulum-xml", ASSET,
        "--start-q-rad", *map(str, ARMQ0),
        "--config", CONFIG,
        "--controller-kind", "x_task_yz_corridor_qp",
        "--freqs-hz", "1.0", "--duration-s", "0.3",
        "--excursion-m", "0.02", "--auto-axis",
    ])
    assert rc == 0
    out = capsys.readouterr().out
    expected = AXIS_NAMES[int(np.argmax([abs(c[n]) for n in AXIS_NAMES]))]
    assert f"--auto-axis -> using {expected}" in out


def test_parser_defaults_bracket_the_natural_frequency():
    """A sweep that never reaches resonance cannot answer the question it is
    asked; the 0.12 m rod's natural frequency is 1.724 Hz."""
    args = build_parser().parse_args([
        "--pendulum-xml", ASSET, "--start-q-rad", *map(str, ARMQ0), "--config", CONFIG,
    ])
    assert min(args.freqs_hz) < 1.724 < max(args.freqs_hz)
    assert args.excursion_m < 0.06, "default demand must sit inside the drift guard"
