"""Closed-loop MuJoCo validation of ``transport_axis_index`` = 1 (Y) and 2 (Z).

The axis-generic transport work was validated with (a) pure-numpy unit tests
against a synthetic point-mass plant (``tests/unit/test_transport_axis_index_
controller.py``) and (b) mocked-RTDE plumbing tests (``tests/hardware/test_
transport_axis_index.py``). Neither can see anything that depends on the real
UR5e: the Jacobian's per-axis force transmission, the wrist singularity, joint
friction, or gravity acting along world Z but not X/Y. This file drives the
SAME adapter pipeline every sim tool in this repo uses --
``simulation/ur5e_mujoco_torque.py``'s ``build_initial_state_and_adapter`` /
``build_mujoco_state`` / ``adapter.step()`` / ``mujoco.mj_step`` on
``assets/ur5e_torque/scene.xml`` -- with the transport axis set to Y and Z, at
poses already defined in ``hardware/poses.py``.

The per-step loop lives in ``tools/diagnostics/axis_generic_transport_sim_check.py``
(also runnable standalone with argparse for one-off exploration); this file is
the reproducible assertion layer over it.

What this file locks down
-------------------------
1. At WELL-CONDITIONED poses (``MEGA_SEARCH_WINNER_Q``, ``HANGING_ALPHA_0_5_Q``),
   Y and Z transport really converge, hold the other two axes, and trip no guard
   -- in BOTH directions (AGENTS.md sec 7: never characterize one direction only).
2. The measured quality at axis 1/2 is comparable to the same pose's axis-0
   baseline, measured in the same run rather than quoted from a doc.
3. The KNOWN LIMIT: at the ``height_alpha=0.5`` pose family, Y/Z transport is
   dramatically worse than X, and it is NOT the wrist singularity -- the
   non-singular ``HEIGHT_ALPHA_0_5_WRIST2_OFFSET_Q`` variant fails identically.
   The cause is a purely kinematic force-transmission asymmetry (test below
   measures it directly from the Jacobian and the model's own frictionloss).
4. Regression guard for the axis-target-generation bug this validation found:
   ``tools/ur5e_mujoco_torque_experiments.py --transport-axis-index 1`` used to
   command a constant hold (the profile was seeded from world X and the delta
   written into pose component 0) and report ``achieved_x_delta_m == 0.0`` with
   ``success: True``.
5. Sim-vs-hardware reference consistency: the mocked-RTDE ``position`` transport
   loop and the sim side generate the SAME reference trajectory for the same
   axis/target/pose.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import mujoco
import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hardware.link import UR5eLink  # noqa: E402
from hardware.poses import (  # noqa: E402
    HEIGHT_ALPHA_0_5_Q,
    HEIGHT_ALPHA_0_5_WRIST2_OFFSET_Q,
    MEGA_SEARCH_WINNER_Q,
)
from hardware.position_transport import run_x_transport_position  # noqa: E402
from simulation.ur5e_mujoco_torque import load_model, x_profile_target  # noqa: E402

SCENE_PATH = REPO_ROOT / "assets" / "ur5e_torque" / "scene.xml"
POSITION_CONFIG = REPO_ROOT / "config" / "ur5e_mujoco_torque_osc_tuned.yaml"


def _load_axis_check_module():
    """Import the diagnostic script as a module (same pattern as
    tests/hardware/test_direct_torque_gain_overrides.py)."""
    path = REPO_ROOT / "tools" / "diagnostics" / "axis_generic_transport_sim_check.py"
    spec = importlib.util.spec_from_file_location("axis_generic_transport_sim_check", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # Registering before exec is required: the module defines a @dataclass, and
    # dataclasses resolves annotations via sys.modules[cls.__module__].
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


AXIS_CHECK = _load_axis_check_module()

# Benign displacement: the small end of this repo's canonical move-hold grid
# (dx 0.01-0.04 m), well inside every pose's validated X envelope, so a failure
# here is an axis-genericity problem and not an envelope problem.
BENIGN_DELTA_M = 0.02
MOVE_DURATION_S = 1.0
HOLD_DURATION_S = 1.0

#: Poses whose own X-axis characterization in this repo is clean, paired with
#: the config that characterization used. Both are well-conditioned
#: (cond(J) ~7 and ~9) and both configs enable friction feedforward, which the
#: model's real frictionloss makes a prerequisite for decent tracking on ANY axis.
WELL_CONDITIONED_POSES = ("mega_search_winner", "hanging_alpha_0_5")

#: Below the ImpedanceSafetyMonitor default orthogonal-drift guard (0.03 m) with
#: real margin -- a held axis creeping past this at a 0.02 m move would be a
#: genuine problem, not a tolerance quibble.
HELD_AXIS_DRIFT_BOUND_M = 0.025

#: Achieved/commanded floor. Measured across the well-conditioned poses and all
#: three axes at this displacement: 0.745-0.903. This bound is deliberately
#: loose enough to be about "did the axis actually track" rather than a
#: gain-tuning regression detector.
TRACKING_FRACTION_FLOOR = 0.60


def _trial(pose: str, axis: int, delta_m: float, *, config: str | None = None):
    q_start, default_cfg = AXIS_CHECK.POSES[pose]
    return AXIS_CHECK.run_axis_transport_trial(
        q_start,
        config_path=config or default_cfg,
        axis=axis,
        target_delta_m=delta_m,
        move_duration_s=MOVE_DURATION_S,
        hold_duration_s=HOLD_DURATION_S,
        pose_label=pose,
    )


_TRIAL_CACHE: dict[tuple[str, int, float, str | None], object] = {}


def trial(pose: str, axis: int, delta_m: float, *, config: str | None = None):
    """Memoized trial -- each (pose, axis, delta) rollout costs ~1.7 s, and
    several tests compare the same rollouts against each other."""
    key = (pose, int(axis), float(delta_m), config)
    if key not in _TRIAL_CACHE:
        _TRIAL_CACHE[key] = _trial(pose, axis, delta_m, config=config)
    return _TRIAL_CACHE[key]


# --------------------------------------------------------------------------- #
# 1. Y and Z transport really work at well-conditioned poses, both directions.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("pose", WELL_CONDITIONED_POSES)
@pytest.mark.parametrize("axis", [1, 2])
@pytest.mark.parametrize("sign", [+1.0, -1.0])
def test_off_x_axis_transport_converges_and_holds_the_other_axes(pose: str, axis: int, sign: float) -> None:
    res = trial(pose, axis, sign * BENIGN_DELTA_M)

    assert not res.guard_tripped, (
        f"{pose} axis={axis} dx={sign * BENIGN_DELTA_M:+.3f}: safety guard tripped at "
        f"t={res.guard_time_s}s ({res.guard_reason})"
    )
    # The commanded axis moved, in the commanded direction, by roughly the
    # commanded amount.
    assert np.sign(res.achieved_delta_m) == np.sign(sign), (
        f"{pose} axis={axis}: moved the wrong way "
        f"(achieved {res.achieved_delta_m:+.5f} for dx={sign * BENIGN_DELTA_M:+.3f})"
    )
    assert res.tracking_fraction >= TRACKING_FRACTION_FLOOR, (
        f"{pose} axis={axis} dx={sign * BENIGN_DELTA_M:+.3f}: only tracked "
        f"{100 * res.tracking_fraction:.1f}% of the commanded displacement"
    )
    # The two NON-transport axes stayed held.
    assert res.max_abs_orthogonal_drift_m < HELD_AXIS_DRIFT_BOUND_M, (
        f"{pose} axis={axis}: held-axis drift {res.max_abs_held_drift_m} exceeded "
        f"{HELD_AXIS_DRIFT_BOUND_M} m"
    )
    # A well-conditioned pose must stay well-conditioned while transporting
    # off-X; a Y/Z move that walked the arm into a singularity would show here.
    assert res.max_cond_j < 50.0, f"{pose} axis={axis}: cond(J) reached {res.max_cond_j}"
    assert res.max_abs_qd_radps < 1.5, f"{pose} axis={axis}: |qd| reached {res.max_abs_qd_radps} rad/s"


@pytest.mark.parametrize("pose", WELL_CONDITIONED_POSES)
@pytest.mark.parametrize("axis", [1, 2])
def test_off_x_axis_transport_is_direction_symmetric(pose: str, axis: int) -> None:
    """AGENTS.md sec 7: X-direction asymmetry is a real, recurring phenomenon in
    this repo. Check the same thing for Y/Z rather than assuming symmetry."""
    pos = trial(pose, axis, +BENIGN_DELTA_M)
    neg = trial(pose, axis, -BENIGN_DELTA_M)
    assert abs(pos.tracking_fraction - neg.tracking_fraction) < 0.15, (
        f"{pose} axis={axis}: strongly direction-asymmetric tracking "
        f"(+{100 * pos.tracking_fraction:.1f}% vs -{100 * neg.tracking_fraction:.1f}%)"
    )


# --------------------------------------------------------------------------- #
# 2. Comparison against the SAME pose's axis-0 baseline, measured here.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("pose", WELL_CONDITIONED_POSES)
@pytest.mark.parametrize("axis", [1, 2])
@pytest.mark.parametrize("sign", [+1.0, -1.0])
def test_off_x_axis_quality_is_comparable_to_the_axis0_baseline(pose: str, axis: int, sign: float) -> None:
    baseline = trial(pose, 0, sign * BENIGN_DELTA_M)
    candidate = trial(pose, axis, sign * BENIGN_DELTA_M)
    assert not baseline.guard_tripped, "axis-0 baseline itself tripped a guard; comparison is meaningless"
    assert candidate.tracking_fraction >= 0.75 * baseline.tracking_fraction, (
        f"{pose} axis={axis} dx={sign * BENIGN_DELTA_M:+.3f}: tracking "
        f"{100 * candidate.tracking_fraction:.1f}% is far below the axis-0 baseline's "
        f"{100 * baseline.tracking_fraction:.1f}%"
    )


# --------------------------------------------------------------------------- #
# 3. The known limit: the height_alpha=0.5 pose family, and why.
# --------------------------------------------------------------------------- #
def _min_friction_breakaway_force_n(q: np.ndarray) -> dict[str, float]:
    """Smallest pure-axis TCP force that puts SOME joint's torque over its own
    ``frictionloss``, i.e. the force floor below which the arm cannot move at
    all along that axis.

    ``tau = J_pos[axis, :].T * F`` for a unit force along one world axis, so the
    per-joint breakaway force is ``frictionloss_i / |J_pos[axis, i]|`` and the
    axis' floor is the smallest of those. Pure kinematics + the model's own
    friction numbers -- no controller, no gains, no rollout.
    """
    model, data, site_id, joint_ids, _ = load_model(SCENE_PATH)
    friction = np.array([float(model.dof_frictionloss[int(model.jnt_dofadr[j])]) for j in joint_ids])
    data.qpos[:] = 0.0
    data.qvel[:] = 0.0
    for idx, jid in enumerate(joint_ids):
        data.qpos[int(model.jnt_qposadr[jid])] = float(q[idx])
    mujoco.mj_forward(model, data)
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    mujoco.mj_jacSite(model, data, jacp, jacr, site_id)
    jp = jacp[:, :6]
    out: dict[str, float] = {}
    for axis, name in enumerate("XYZ"):
        ratios = np.where(np.abs(jp[axis]) > 1e-9, friction / np.maximum(np.abs(jp[axis]), 1e-12), np.inf)
        out[name] = float(np.min(ratios))
    return out


def test_well_conditioned_poses_transmit_force_comparably_on_all_three_axes() -> None:
    """Why Y/Z transport works at these poses: no axis needs much more force to
    break static friction than any other."""
    for pose in WELL_CONDITIONED_POSES:
        q, _cfg = AXIS_CHECK.POSES[pose]
        forces = _min_friction_breakaway_force_n(q)
        worst = max(forces.values())
        best = min(forces.values())
        assert worst / best < 2.0, f"{pose}: force-transmission spread {forces} is larger than expected"


@pytest.mark.parametrize("pose", ["height_alpha_0_5", "height_alpha_0_5_wrist2_offset"])
def test_height_alpha_0_5_family_is_force_transmission_limited_off_x(pose: str) -> None:
    """The height_alpha=0.5 family needs 3-6x more TCP force to move along Y
    than along X, purely kinematically.

    This is the mechanism behind that family's poor Y/Z closed-loop tracking,
    and it is NOT the wrist singularity: the non-singular
    HEIGHT_ALPHA_0_5_WRIST2_OFFSET_Q variant (cond(J) ~29 vs ~7e16) shows the
    same ratio. It is a property of the arm at that pose, so it will hold on
    real hardware too -- see this module's docstring.
    """
    q, _cfg = AXIS_CHECK.POSES[pose]
    forces = _min_friction_breakaway_force_n(q)
    assert forces["Y"] > 3.0 * forces["X"], f"{pose}: expected a large Y/X force penalty, got {forces}"
    assert forces["Z"] > 2.0 * forces["X"], f"{pose}: expected a large Z/X force penalty, got {forces}"


@pytest.mark.parametrize("pose", ["height_alpha_0_5", "height_alpha_0_5_wrist2_offset"])
def test_height_alpha_0_5_family_y_transport_underperforms_x_in_closed_loop(pose: str) -> None:
    """Pins the negative closed-loop result that goes with the kinematic finding
    above, so a future controller change that fixes it is visible as a failure
    here rather than passing silently."""
    x_run = trial(pose, 0, BENIGN_DELTA_M)
    y_run = trial(pose, 1, BENIGN_DELTA_M)
    assert x_run.tracking_fraction > 0.8, (
        f"{pose}: the axis-0 reference run regressed ({100 * x_run.tracking_fraction:.1f}%); "
        "the comparison below is only meaningful while X itself tracks well"
    )
    assert y_run.tracking_fraction < 0.5, (
        f"{pose}: Y transport now tracks {100 * y_run.tracking_fraction:.1f}%, better than the "
        "<50% this pose family was measured at. If a real fix landed, update this test and "
        "AGENTS.md rather than deleting the check."
    )


# --------------------------------------------------------------------------- #
# 4. Regression guard for the target-generation bug found by this validation.
# --------------------------------------------------------------------------- #
def _run_experiments_tool(tmp_path: Path, *, axis: int, delta: float) -> dict:
    out_root = tmp_path / f"axis{axis}"
    completed = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "tools" / "ur5e_mujoco_torque_experiments.py"),
            "--mode", "controller-rollout",
            "--controller-kind", "impedance",
            "--config", str(REPO_ROOT / "config" / "ur5e_mujoco_torque_osc_mega_search_winner.yaml"),
            "--trajectory-profile", "min_jerk_move_hold",
            "--move-duration", str(MOVE_DURATION_S),
            "--duration", str(MOVE_DURATION_S + HOLD_DURATION_S),
            "--target-x-delta", str(delta),
            "--transport-axis-index", str(axis),
            "--start-q-rad", *[repr(float(v)) for v in MEGA_SEARCH_WINNER_Q],
            "--seed", "0",
            "--no-plot",
            "--output-dir", str(out_root),
        ],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr[-4000:]
    summaries = list(out_root.rglob("summary.json"))
    assert summaries, f"no summary.json under {out_root}"
    with max(summaries, key=lambda p: p.stat().st_mtime).open() as fp:
        return json.load(fp)


@pytest.mark.slow
@pytest.mark.parametrize("axis", [1, 2])
def test_experiments_tool_actually_transports_the_selected_axis(tmp_path: Path, axis: int) -> None:
    """``--transport-axis-index 1/2`` used to command a constant hold: the
    min-jerk reference was seeded from ``state0.ee_pos[0]`` and written into
    ``target_ee_pos[0]``, while ``target_axis`` read back the untouched Y/Z
    component. The run completed with ``success: True`` and
    ``achieved_x_delta_m == 0.0``. Every individual piece of plumbing was
    correct, which is why only a closed-loop rollout could catch it.
    """
    summary = _run_experiments_tool(tmp_path, axis=axis, delta=BENIGN_DELTA_M)
    assert summary["transport_axis_index"] == axis
    achieved = float(summary["achieved_x_delta_m"])
    assert achieved > 0.5 * BENIGN_DELTA_M, (
        f"axis={axis}: achieved_x_delta_m={achieved} -- the selected axis barely moved"
    )
    start = np.asarray(summary["initial_ee_pos"], dtype=np.float64)
    final = np.asarray(summary["final_ee_pos"], dtype=np.float64)
    displacement = np.abs(final - start)
    assert int(np.argmax(displacement)) == axis, (
        f"axis={axis}: the largest displacement was on axis {int(np.argmax(displacement))} "
        f"(per-axis |delta| = {displacement.tolist()})"
    )


@pytest.mark.slow
def test_experiments_tool_axis0_still_reports_x() -> None:
    """The axis-generic target generation must not have changed the default
    path. Cross-checks the tool's axis-0 result against this file's independent
    rollout implementation of the same trajectory."""
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        summary = _run_experiments_tool(Path(tmp), axis=0, delta=BENIGN_DELTA_M)
    reference = trial("mega_search_winner", 0, BENIGN_DELTA_M)
    assert summary["transport_axis_index"] == 0
    assert float(summary["achieved_x_delta_m"]) == pytest.approx(reference.achieved_delta_m, abs=2e-4)


# --------------------------------------------------------------------------- #
# 5. Sim-vs-hardware reference-trajectory consistency (mocked RTDE, no robot).
# --------------------------------------------------------------------------- #
class _FakeReceive:
    def __init__(self, start_xyz: np.ndarray) -> None:
        self.q = list(map(float, MEGA_SEARCH_WINNER_Q))
        self.qd = [0.0] * 6
        self.pos = [float(v) for v in start_xyz]
        self.rot = [0.0, 3.14, 0.0]
        self._ts = 0.0

    def getActualQ(self):
        return list(self.q)

    def getActualQd(self):
        return list(self.qd)

    def getActualTCPPose(self):
        return list(self.pos) + list(self.rot)

    def getTimestamp(self):
        self._ts += 0.008
        return self._ts

    def getSafetyStatusBits(self):
        return 1  # IS_NORMAL_MODE

    def disconnect(self):
        pass


class _FakeControl:
    """Records every commanded waypoint and tracks it with a bounded step, so
    the loop's own guards see plausible motion."""

    _MAX_TCP_STEP_M = 0.002

    def __init__(self, receive: _FakeReceive, axis: int) -> None:
        self._receive = receive
        self._axis = int(axis)
        self.waypoints: list[list[float]] = []

    def servoL(self, pose, speed, acceleration, time, lookahead_time, gain):
        self.waypoints.append([float(v) for v in pose])
        delta = float(pose[self._axis]) - self._receive.pos[self._axis]
        step = min(abs(delta), self._MAX_TCP_STEP_M)
        if step > 0.0:
            self._receive.pos[self._axis] += step if delta > 0.0 else -step

    def servoStop(self):
        pass

    def stopScript(self):
        pass

    def disconnect(self):
        pass


def _mega_search_winner_ee_pos() -> np.ndarray:
    """The REAL MuJoCo site position at MEGA_SEARCH_WINNER_Q -- the same start
    point the sim rollout uses, fed to the mocked robot as its TCP."""
    model, data, site_id, joint_ids, _ = load_model(SCENE_PATH)
    data.qpos[:] = 0.0
    data.qvel[:] = 0.0
    for idx, jid in enumerate(joint_ids):
        data.qpos[int(model.jnt_qposadr[jid])] = float(MEGA_SEARCH_WINNER_Q[idx])
    mujoco.mj_forward(model, data)
    return np.asarray(data.site_xpos[site_id], dtype=np.float64).copy()


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_position_transport_waypoints_match_the_sim_side_reference(tmp_path: Path, axis: int) -> None:
    """Sim-vs-hardware plumbing consistency for a non-zero transport axis.

    The existing mocked-RTDE tests prove the axis threads through the hardware
    loop; this one proves the NUMBERS agree with what the sim side commands for
    the same axis/target/start pose. Both lanes drive
    ``simulation.ur5e_mujoco_torque.x_profile_target`` with the same
    ``min_jerk_move_hold`` shape, and both must seed it from the SELECTED axis'
    start value and write the result into the SELECTED pose component -- an
    axis-0-seeded reference dropped into component 1 (the sim-side bug this
    validation found) would show up here as a large mismatch, not a rounding
    difference. Exact equality is the right assertion because both lanes
    accumulate ``t`` the same way from the same rate.
    """
    start_xyz = _mega_search_winner_ee_pos()
    rate_hz = 50.0
    move_s = 0.20
    total_s = 0.40
    delta = 0.01

    receive = _FakeReceive(start_xyz)
    control = _FakeControl(receive, axis)
    link = UR5eLink(
        "127.0.0.1",
        rate_hz,
        receive_factory=lambda ip, freq: receive,
        control_factory=lambda ip, freq: control,
    )
    result = run_x_transport_position(
        link,
        config_path=POSITION_CONFIG,
        target_x_delta_m=delta,
        move_duration_s=move_s,
        duration_s=total_s,
        output_dir=tmp_path,
        motion_opt_in=True,
        rate_hz=rate_hz,
        shadow_osc=False,
        transport_axis_index=axis,
    )
    assert result.summary["transport_axis_index"] == axis
    assert control.waypoints, "servoL was never called"

    # Independently regenerate the reference the sim side would use.
    dt = 1.0 / rate_hz
    expected: list[float] = []
    t_s = 0.0
    while t_s < total_s - 1e-12:
        target, _vel = x_profile_target(
            "min_jerk_move_hold",
            float(start_xyz[axis]),
            delta,
            t_s,
            total_s,
            move_duration_s=move_s,
        )
        expected.append(target)
        t_s += dt

    assert len(control.waypoints) == len(expected)
    for step, (waypoint, target) in enumerate(zip(control.waypoints, expected)):
        assert waypoint[axis] == target, (
            f"axis={axis} step={step}: hardware commanded {waypoint[axis]!r} but the sim-side "
            f"reference for the same axis/target/start pose is {target!r}"
        )
        for other in (0, 1, 2):
            if other == axis:
                continue
            assert waypoint[other] == pytest.approx(float(start_xyz[other])), (
                f"axis={axis} step={step}: non-transport component {other} was not pinned to the start pose"
            )
    # And the reference really is a displacement along the selected axis.
    assert expected[-1] - expected[0] == pytest.approx(delta, abs=1e-12)
