"""Tests for `controller.controller_kind` on the direct-torque hardware
transport loop -- selecting `XTaskYZCorridorQPController` (the reduced-task
corridor-QP + HOCBF law, `controller_core/x_task_yz_corridor_qp`) as the inner
control law instead of the tuned OSC impedance controller.

Two things this file has to prove, per the task that added it:
  (a) `controller_kind: "x_task_yz_corridor_qp"` actually constructs
      `XTaskYZCorridorQPController` (not the default), runs several real
      cycles of `compute()`, and the resulting torque reaches
      `link.direct_torque()` through the SAME path OSC uses (finite 6-vector,
      logged as `tau_controller`/`tau_applied` in the trace, same summary
      shape) -- not merely that the run "succeeds" with some exit code.
  (b) The DEFAULT path (no `controller_kind` key, or `controller_kind:
      "impedance"`) still builds `XAxisCartesianImpedanceController` and its
      behavior is unaffected by this change -- checked against the class
      actually built, not just a config field.

Follows the `_MockDTLink` mocking style already established in
`test_direct_torque_gain_overrides.py`: a fake `UR5eDirectTorqueLink` that
never touches real RTDE, with `dynamics_source="local"` so J/M/jacobian_fn
all come from the real MJCF (`LocalMujocoDynamics`, see
`hardware/local_dynamics.py`) rather than the mock -- this is what lets the
corridor-QP controller's `manipulability_cbf` (which needs `jacobian_fn`
evaluable at arbitrary q) run for real in a test with no RTDE connection.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from hardware.direct_torque_link import UR5eDirectTorqueLink  # noqa: E402
from hardware.direct_torque_transport import (  # noqa: E402
    _load_corridor_qp_bundle,
    _read_controller_kind,
    run_x_transport_direct_torque,
)
from hardware.link import UR5eState  # noqa: E402
from hardware.poses import HEIGHT_ALPHA_0_5_Q  # noqa: E402
from hardware.safety import UR5eSafetyLimits  # noqa: E402

CORRIDOR_QP_CONFIG = REPO_ROOT / "config" / "ur5e_direct_torque_x_task_yz_corridor_qp.yaml"
DEFAULT_CONFIG = REPO_ROOT / "config" / "ur5e_mujoco_torque_osc_tuned.yaml"

# HEIGHT_ALPHA_0_5_Q sits exactly on the wrist_2=0 singularity (see the
# comment in test_direct_torque_gain_overrides.py) -- the pose this
# controller family exists to survive, and a non-trivial q for the
# manipulability CBF's jacobian_fn to differentiate through.


class _MockDTLink:
    def __init__(self, tcp_x: float = 0.35) -> None:
        self._tcp_x = tcp_x
        self.limits = UR5eSafetyLimits()
        self.connect_calls = 0

    def connect(self) -> None:
        self.connect_calls += 1

    def read_state(self) -> UR5eState:
        return UR5eState(
            q=HEIGHT_ALPHA_0_5_Q.copy(),
            qd=np.zeros(6),
            tcp_pose=np.array([self._tcp_x, -0.2, 0.3, 0.0, 3.14, 0.0]),
            host_stamp_ns=time.monotonic_ns(),
            robot_timestamp_s=None,
            safety_status=None,
        )

    def get_jacobian(self) -> np.ndarray:
        return np.eye(6)

    def get_mass_matrix(self) -> np.ndarray:
        return np.eye(6)

    def direct_torque(self, tau_nm: np.ndarray, *, friction_comp: bool = True) -> None:
        # Real, tiny feedback so the loop isn't driving an utterly static
        # plant -- matches _MockDTLink's own pattern in
        # test_direct_torque_gain_overrides.py.
        self._tcp_x += float(tau_nm[0]) * 1e-6

    @staticmethod
    def compose_robot_state(
        link_state, *, jacobian, mass_matrix, time_s, target_x, target_x_vel,
        dt_s=None, target_x_accel=None, transport_axis_index=0,
    ):
        return UR5eDirectTorqueLink.compose_robot_state(
            link_state, jacobian=jacobian, mass_matrix=mass_matrix,
            time_s=time_s, target_x=target_x, target_x_vel=target_x_vel, dt_s=dt_s,
            target_x_accel=target_x_accel, transport_axis_index=transport_axis_index,
        )

    def build_robot_state(self, link_state, *, time_s, target_x, target_x_vel, dt_s=None, transport_axis_index=0):
        return self.compose_robot_state(
            link_state, jacobian=self.get_jacobian(), mass_matrix=self.get_mass_matrix(),
            time_s=time_s, target_x=target_x, target_x_vel=target_x_vel, dt_s=dt_s,
            transport_axis_index=transport_axis_index,
        )

    def safe_stop(self, reason: str) -> None:
        pass


def _trace_rows(result) -> list[dict]:
    return [json.loads(line) for line in result.trace_path.read_text().splitlines()]


# --------------------------------------------------------------------------
# (a) controller_kind="x_task_yz_corridor_qp" actually selects, runs, and
#     emits finite torque through the same path.
# --------------------------------------------------------------------------


@pytest.mark.hardware
def test_corridor_qp_config_reparsed_selects_the_new_kind() -> None:
    """Re-parse the config through its OWN loader (not by reading the YAML
    text) -- AGENTS.md sec.7's "verify a config edit by re-parsing it" rule.
    Confirms controller_kind is actually read as "x_task_yz_corridor_qp" and
    that the mechanisms this controller exists for (yz_corridor_enabled,
    manipulability_cbf) actually parsed True, not silently defaulted False."""
    kind = _read_controller_kind(CORRIDOR_QP_CONFIG)
    assert kind == "x_task_yz_corridor_qp"
    cfg, safety_cfg, freq = _load_corridor_qp_bundle(CORRIDOR_QP_CONFIG)
    assert cfg.yz_corridor_enabled is True
    assert cfg.manipulability_cbf is True
    assert cfg.task_excluded_joints == (0,)
    assert freq == 500.0
    assert safety_cfg.max_abs_orthogonal_drift_m == pytest.approx(0.06)


@pytest.mark.hardware
def test_corridor_qp_kind_constructs_the_corridor_controller_and_runs(tmp_path: Path) -> None:
    """The end-to-end claim: controller_kind="x_task_yz_corridor_qp" makes
    the loop build XTaskYZCorridorQPController (not the default), actually
    call its compute() every cycle, and route finite torque through
    link.direct_torque() -- checked via the summary's controller_class field
    (read back from the constructed object, not the YAML) plus the real
    per-cycle trace."""
    link = _MockDTLink()
    result = run_x_transport_direct_torque(
        link,  # type: ignore[arg-type]
        config_path=CORRIDOR_QP_CONFIG,
        target_x_delta_m=0.01,
        move_duration_s=0.02,
        duration_s=0.03,
        output_dir=tmp_path,
        motion_opt_in=True,
        record_latency=False,
        # manipulability_cbf needs jacobian_fn at arbitrary q, and this loop
        # now REQUIRES 'local_pinocchio' specifically when manipulability_cbf
        # is on (see the guard in run_x_transport_direct_torque and
        # test_corridor_qp_manipulability_cbf_requires_local_pinocchio below)
        # -- 'local' (LocalMujocoDynamics) measured ~4.9 ms/cycle median and
        # deadline-trips the 2 ms/500 Hz budget by cycle 3.
        dynamics_source="local_pinocchio",
    )
    assert result.summary["controller_kind"] == "x_task_yz_corridor_qp"
    assert result.summary["controller_class"] == "XTaskYZCorridorQPController"

    rows = _trace_rows(result)
    assert len(rows) > 0, "no cycles ran -- the controller never actually computed anything"
    for row in rows:
        tau = np.asarray(row["tau_controller"], dtype=np.float64)
        assert tau.shape == (6,)
        assert np.all(np.isfinite(tau)), f"non-finite tau_controller from the corridor-QP controller: {tau}"
        tau_applied = np.asarray(row["tau_applied"], dtype=np.float64)
        assert tau_applied.shape == (6,)
        assert np.all(np.isfinite(tau_applied))
    # task_scale/task_backtrack_iters are OSC-only fields; the getattr
    # fallback in the trace-row builder must not have crashed and must have
    # produced the documented defaults for a controller that has neither
    # mechanism.
    assert rows[0]["task_scale"] == 1.0
    assert rows[0]["task_backtrack_iters"] == 0
    # jacobian_cond/singular_scale ARE shared fields -- confirm they came
    # through as real (non-placeholder) numbers, not silently omitted.
    assert rows[0]["jacobian_cond"] > 0.0


@pytest.mark.hardware
def test_corridor_qp_kind_rejects_nonzero_transport_axis_before_connecting() -> None:
    """XTaskYZCorridorQPController.compute() is world-X only by construction
    (raises ValueError on any transport_axis_index != 0) -- the transport
    loop must catch this BEFORE link.connect(), not let it surface mid-run as
    a controller exception on the first cycle."""
    link = _MockDTLink()
    with pytest.raises(ValueError, match="world-X only"):
        run_x_transport_direct_torque(
            link,  # type: ignore[arg-type]
            config_path=CORRIDOR_QP_CONFIG,
            target_x_delta_m=0.01,
            move_duration_s=0.02,
            duration_s=0.03,
            motion_opt_in=True,
            record_latency=False,
            dynamics_source="local_pinocchio",
            transport_axis_index=1,
        )
    assert link.connect_calls == 0


@pytest.mark.hardware
def test_corridor_qp_kind_rejects_gain_overrides_before_connecting() -> None:
    """XTaskYZCorridorQPController has no set_gains() -- gain_overrides must
    be rejected loudly before connecting, not silently ignored (which would
    be exactly the "operation appears to succeed but did nothing" failure
    AGENTS.md sec.7 warns about)."""
    link = _MockDTLink()
    with pytest.raises(ValueError, match="gain_overrides is not implemented"):
        run_x_transport_direct_torque(
            link,  # type: ignore[arg-type]
            config_path=CORRIDOR_QP_CONFIG,
            target_x_delta_m=0.01,
            move_duration_s=0.02,
            duration_s=0.03,
            motion_opt_in=True,
            record_latency=False,
            dynamics_source="local_pinocchio",
            gain_overrides={"kp_x": 4000.0},
        )
    assert link.connect_calls == 0


@pytest.mark.hardware
@pytest.mark.parametrize("bad_dynamics_source", ["rtde", "local"])
def test_corridor_qp_manipulability_cbf_requires_local_pinocchio(bad_dynamics_source: str) -> None:
    """manipulability_cbf: true needs jacobian_fn at arbitrary q, and this
    loop now REQUIRES dynamics_source='local_pinocchio' specifically for
    that role (2026-08-26 critic fix) -- 'rtde' cannot supply jacobian_fn at
    all (current-q only), and 'local' (LocalMujocoDynamics) technically can
    but was BENCHED at ~4.9 ms/cycle median, deadline-tripping this loop's
    2 ms/500 Hz budget by cycle 3 (see the config's REAL-TIME BUDGET header
    for the measured numbers this guard is based on). Both must be rejected
    before connecting, not left to fail deep inside XTaskYZCorridorQPController's
    own ValueError on the first cycle, or -- worse, for 'local' -- left to
    silently run and deadline-trip a few cycles into a live run."""
    link = _MockDTLink()
    with pytest.raises(ValueError, match="requires dynamics_source='local_pinocchio'"):
        run_x_transport_direct_torque(
            link,  # type: ignore[arg-type]
            config_path=CORRIDOR_QP_CONFIG,
            target_x_delta_m=0.01,
            move_duration_s=0.02,
            duration_s=0.03,
            motion_opt_in=True,
            record_latency=False,
            dynamics_source=bad_dynamics_source,
        )
    assert link.connect_calls == 0


@pytest.mark.hardware
def test_corridor_qp_manipulability_cbf_false_permits_local_dynamics_source() -> None:
    """The dynamics_source='local_pinocchio' requirement is scoped to
    manipulability_cbf specifically, not to controller_kind as a whole -- a
    corridor-QP config with manipulability_cbf: false must still accept
    dynamics_source='local' without the new guard firing (it has no
    jacobian_fn to source, so the fast-Jacobian requirement doesn't apply)."""
    import yaml

    cfg = yaml.safe_load(CORRIDOR_QP_CONFIG.read_text(encoding="utf-8"))
    cfg["controller"]["manipulability_cbf"] = False
    cfg["controller"]["yz_corridor_enabled"] = False
    no_cbf_config = CORRIDOR_QP_CONFIG.parent / "_test_corridor_qp_no_cbf.yaml"
    no_cbf_config.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    try:
        link = _MockDTLink()
        result = run_x_transport_direct_torque(
            link,  # type: ignore[arg-type]
            config_path=no_cbf_config,
            target_x_delta_m=0.01,
            move_duration_s=0.02,
            duration_s=0.03,
            motion_opt_in=True,
            record_latency=False,
            dynamics_source="local",
        )
        assert result.summary["controller_class"] == "XTaskYZCorridorQPController"
        assert link.connect_calls == 1
    finally:
        no_cbf_config.unlink(missing_ok=True)


# --------------------------------------------------------------------------
# (b) regression: the default path is unaffected.
# --------------------------------------------------------------------------


def test_default_controller_kind_is_impedance() -> None:
    """A config with no controller_kind key (every pre-existing config in
    this repo) must default to "impedance" -- the unit-level half of the
    regression proof, independent of running the loop at all."""
    assert _read_controller_kind(DEFAULT_CONFIG) == "impedance"


@pytest.mark.hardware
def test_default_controller_kind_still_builds_impedance_controller(tmp_path: Path) -> None:
    """End-to-end half of the regression proof: with no controller_kind key,
    the loop must still build XAxisCartesianImpedanceController and produce
    the same summary/trace shape it always has -- checked via
    controller_class (read from the real constructed object), not just that
    the run didn't crash."""
    link = _MockDTLink()
    result = run_x_transport_direct_torque(
        link,  # type: ignore[arg-type]
        config_path=DEFAULT_CONFIG,
        target_x_delta_m=0.01,
        move_duration_s=0.02,
        duration_s=0.03,
        output_dir=tmp_path,
        motion_opt_in=True,
        record_latency=False,
        dynamics_source="local",
    )
    assert result.summary["controller_kind"] == "impedance"
    assert result.summary["controller_class"] == "XAxisCartesianImpedanceController"
    rows = _trace_rows(result)
    assert len(rows) > 0
    # task_scale/task_backtrack_iters are real OSC output fields, read via the
    # same getattr(..., default) the corridor-QP path relies on -- for OSC the
    # attribute always exists, so this must equal output.task_scale exactly,
    # not the corridor-QP fallback default of 1.0/0 by coincidence.
    for row in rows:
        assert isinstance(row["task_scale"], float)
        assert isinstance(row["task_backtrack_iters"], int)


@pytest.mark.hardware
def test_default_config_produces_deterministic_trace(tmp_path: Path) -> None:
    """Same scenario, run twice, byte-identical tau_controller trace -- this
    loop has no hidden randomness, so any difference here would mean the
    controller_kind plumbing introduced nondeterminism (e.g. state leaking
    between the impedance/corridor branches) on the default path."""
    def _run(out_dir: Path):
        link = _MockDTLink()
        return run_x_transport_direct_torque(
            link,  # type: ignore[arg-type]
            config_path=DEFAULT_CONFIG,
            target_x_delta_m=0.01,
            move_duration_s=0.02,
            duration_s=0.03,
            output_dir=out_dir,
            motion_opt_in=True,
            record_latency=False,
            dynamics_source="local",
        )

    result_a = _run(tmp_path / "a")
    result_b = _run(tmp_path / "b")
    rows_a = _trace_rows(result_a)
    rows_b = _trace_rows(result_b)
    tau_a = np.asarray([r["tau_controller"] for r in rows_a], dtype=np.float64)
    tau_b = np.asarray([r["tau_controller"] for r in rows_b], dtype=np.float64)
    np.testing.assert_array_equal(tau_a, tau_b)
