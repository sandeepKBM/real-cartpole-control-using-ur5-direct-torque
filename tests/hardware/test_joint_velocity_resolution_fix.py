"""Regression coverage for the double-resolution bug found by review
(2026-08-26) in hardware/joint_velocity_transport.py.

Bug: config/ur5e_velocity_control.yaml's default reduced_task_dims=true
makes CartesianVelocityController resolve the Cartesian task to a joint
velocity INTERNALLY (qd_internal = qd_primary + N @ qd_posture) and return
xd_cmd = J @ qd_internal -- an already-resolved joint velocity re-expressed
as a Cartesian one. Feeding that into damped_least_squares_qd(J, xd_cmd)
double-resolves: DLS(J, J @ qd_internal). At a well-conditioned (invertible)
J with negligible damping this exactly reconstructs qd_internal (DLS becomes
a silent no-op); at the ARM_Q0 singularity this mode exists to handle, it
stacks DLS's own damping on top of the controller's internal
pinv_damping-regularized reduced-task resolution -- a real, uncharacterized
double-damped result.

Fix: config/ur5e_speedj_joint_velocity.yaml sets reduced_task_dims (and the
other two resolution-mode flags) to false, which routes
CartesianVelocityController through modes.py::compute_full_hold -- literally
`return xd_full`, no Jacobian, no redundancy resolution -- so DLS is the
SOLE Cartesian-to-joint resolver. hardware/joint_velocity_transport.py also
fails fast if a loaded config has any of the three flags on, as defense in
depth.

These tests prove, numerically and structurally, that the fix actually
changed the resolution path (not just added a config file nobody uses):
  1. the OLD (buggy) double-resolution pipeline is reproduced and shown to
     be a near-exact no-op at a well-conditioned pose, and a real ~1e-2
     perturbation at ARM_Q0 -- documenting exactly what was wrong;
  2. the NEW config's controller.compute() never reads a Jacobian at all
     (proven by omitting "jacobian" from the state dict entirely and
     confirming no error) -- structural proof there is no internal
     resolution step left to stack with DLS;
  3. with the new config, DLS(J, xd_cmd) differs MEANINGFULLY from the old
     pipeline's reconstructed qd_internal -- DLS is doing real, different
     work, not coincidentally reproducing the old answer;
  4. run_x_transport_joint_velocity() itself refuses to run against the old
     (double-resolving) config.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import yaml  # noqa: E402

from controller_core.cartesian_velocity_controller import (  # noqa: E402
    CartesianVelocityConfig,
    CartesianVelocityController,
)
from controller_core.cartesian_velocity_controller.math_utils import _damped_pinv  # noqa: E402
from controller_core.damped_least_squares import damped_least_squares_qd  # noqa: E402
from controller_core.kinematics_utils import swing_twist_axis_error  # noqa: E402
from hardware.joint_velocity_transport import (  # noqa: E402
    DEFAULT_DAMPING_LAMBDA_MAX,
    DEFAULT_DAMPING_SIGMA0,
    run_x_transport_joint_velocity,
)
from hardware.link import UR5eLink  # noqa: E402
from hardware.local_dynamics import LocalMujocoDynamics  # noqa: E402

OLD_CONFIG = REPO_ROOT / "config" / "ur5e_velocity_control.yaml"
NEW_CONFIG = REPO_ROOT / "config" / "ur5e_speedj_joint_velocity.yaml"

# Well-conditioned pose reused from tests/hardware/test_joint_velocity_transport.py.
WELL_CONDITIONED_Q = np.array([0.0, -0.835, -1.2, -0.985, 0.2, 0.0], dtype=np.float64)
# The documented singular pose (AGENTS.md), sigma_min=1.485e-3, cond(J)=1395.76.
ARM_Q0_SINGULAR = np.array(
    [-2.3688, -2.1801, -1.8838, -0.7962, 0.004714693, 0.0206], dtype=np.float64
)


def _load_velocity_config(path: Path) -> CartesianVelocityConfig:
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    return CartesianVelocityConfig.from_controller_yaml_section(cfg.get("controller", {}) or {})


def _state_for(q: np.ndarray, dyn: LocalMujocoDynamics, *, include_jacobian: bool) -> tuple[dict, np.ndarray]:
    jacobian = dyn.jacobian(q)
    ee_pos, ee_quat, _ = dyn.fk_and_jacobian(q)
    p_des = ee_pos.copy()
    p_des[0] += 0.02  # a small +X target displacement, everything else held
    state = {
        "time": 0.0,
        "q": q,
        "qd": np.zeros(6),
        "ee_pos": ee_pos,
        "ee_quat": ee_quat,
        "target_x": float(p_des[0]),
        "target_ee_pos": p_des,
        "target_ee_vel": np.array([0.05, 0.0, 0.0], dtype=np.float64),
    }
    if include_jacobian:
        state["jacobian"] = jacobian
    return state, jacobian


def _internal_qd_reduced_task_dims(
    cfg: CartesianVelocityConfig, q: np.ndarray, dyn: LocalMujocoDynamics, xd_full: np.ndarray
) -> np.ndarray:
    """Exact replica of modes.py::compute_reduced_task_dims's internal
    ``qd = qd_primary + nullspace_proj @ qd_secondary`` -- the TRUE
    qd_internal CartesianVelocityController computed before re-expressing it
    as ``xd_cmd = J @ qd_internal``.

    Deliberately NOT reconstructed by inverting J (``np.linalg.solve``/
    ``np.linalg.pinv``) from the returned ``xd_cmd``: at a near-singular J
    (ARM_Q0) that inversion is itself lossy (pinv only recovers the
    row-space component of qd_internal, discarding whatever the nullspace
    posture term contributed outside it), which would conflate genuine
    "double damping" with plain reconstruction error. Replicating the exact
    formula instead gives an honest ground truth. q_rest is offset from q by
    a small fixed amount so the nullspace posture term (qd_secondary) is
    genuinely nonzero and representative of steady-state posture pull, not
    the trivially-zero value it would be exactly at reset.
    """
    J = dyn.jacobian(q)
    rot_flags = [cfg.task_dim_rx, cfg.task_dim_ry, cfg.task_dim_rz]
    selected = [0, 1, 2] + [3 + i for i, on in enumerate(rot_flags) if on]
    j_task = J[selected, :]
    xd_task = xd_full[selected]
    qd_primary = _damped_pinv(j_task, cfg.pinv_damping) @ xd_task
    nullspace_proj = np.eye(6) - np.linalg.pinv(j_task) @ j_task
    q_rest = q + 0.05
    qd_secondary = cfg.kp_posture * (q_rest - q)
    return qd_primary + nullspace_proj @ qd_secondary


def _xd_full_for(q: np.ndarray, dyn: LocalMujocoDynamics, cfg: CartesianVelocityConfig) -> np.ndarray:
    """Reproduces controller.compute()'s pre-dispatch xd_full construction
    (position P+feedforward, swing-twist orientation P) directly, so it can
    be fed to both the real internal resolution (for ground truth) and
    DLS (for the fix), independent of which resolution mode a given cfg
    selects."""
    ee_pos, ee_quat, _ = dyn.fk_and_jacobian(q)
    p_des = ee_pos.copy()
    p_des[0] += 0.02
    v_ff = np.array([0.05, 0.0, 0.0])
    pos_err = p_des - ee_pos
    kp_lin = np.array([cfg.kp_x, cfg.kp_y, cfg.kp_z])
    v_cmd = v_ff + kp_lin * pos_err
    w_cmd = cfg.kp_rot * np.array(
        [swing_twist_axis_error(ee_quat, ee_quat, i) for i in range(3)]
    )
    return np.concatenate([v_cmd, w_cmd]).astype(np.float64)


def _reset_state(q: np.ndarray, dyn: LocalMujocoDynamics) -> dict:
    ee_pos, ee_quat, _ = dyn.fk_and_jacobian(q)
    return {
        "time": 0.0,
        "q": q,
        "qd": np.zeros(6),
        "ee_pos": ee_pos,
        "ee_quat": ee_quat,
        "target_x": float(ee_pos[0]),
    }


@pytest.fixture(scope="module")
def dyn() -> LocalMujocoDynamics:
    return LocalMujocoDynamics()


class TestOldConfigDoubleResolvesAndIsBuggy:
    """Reproduces the exact bug the critic found, so this file also serves
    as a durable record of what was wrong (not just that it's fixed)."""

    def test_well_conditioned_pose_old_pipeline_is_a_near_exact_noop(self, dyn: LocalMujocoDynamics) -> None:
        old_cfg = _load_velocity_config(OLD_CONFIG)
        assert old_cfg.reduced_task_dims is True  # sanity: this IS the buggy config

        J = dyn.jacobian(WELL_CONDITIONED_Q)
        xd_full = _xd_full_for(WELL_CONDITIONED_Q, dyn, old_cfg)
        qd_internal = _internal_qd_reduced_task_dims(old_cfg, WELL_CONDITIONED_Q, dyn, xd_full)
        xd_cmd_old = J @ qd_internal  # exactly what controller.compute() would return

        # The bug: feeding that already-resolved xd_cmd back into DLS.
        double_resolved = damped_least_squares_qd(
            J, xd_cmd_old, lambda_max=DEFAULT_DAMPING_LAMBDA_MAX, sigma0=DEFAULT_DAMPING_SIGMA0
        ).qd
        residual = float(np.linalg.norm(double_resolved - qd_internal))
        assert residual < 1e-9, f"expected near-exact no-op reconstruction, got residual={residual}"

    def test_arm_q0_old_pipeline_double_damps_by_a_real_amount(self, dyn: LocalMujocoDynamics) -> None:
        old_cfg = _load_velocity_config(OLD_CONFIG)

        J = dyn.jacobian(ARM_Q0_SINGULAR)
        xd_full = _xd_full_for(ARM_Q0_SINGULAR, dyn, old_cfg)
        qd_internal = _internal_qd_reduced_task_dims(old_cfg, ARM_Q0_SINGULAR, dyn, xd_full)
        xd_cmd_old = J @ qd_internal

        double_resolved = damped_least_squares_qd(
            J, xd_cmd_old, lambda_max=DEFAULT_DAMPING_LAMBDA_MAX, sigma0=DEFAULT_DAMPING_SIGMA0
        ).qd
        residual = float(np.linalg.norm(double_resolved - qd_internal))
        # A real, non-trivial perturbation at the singularity -- this is the
        # "uncharacterized double-damped resolution" the review flagged
        # (measured ~1.8e-2 with this exact scenario).
        assert residual > 1e-3, f"expected a real double-damping perturbation, got residual={residual}"


class TestNewConfigIsSingleResolution:
    def test_new_config_disables_all_resolution_modes(self) -> None:
        new_cfg = _load_velocity_config(NEW_CONFIG)
        assert new_cfg.reduced_task_dims is False
        assert new_cfg.split_base_wrist_task is False
        assert new_cfg.ik_seeded_resolution is False

    def test_compute_never_touches_a_jacobian(self, dyn: LocalMujocoDynamics) -> None:
        """Structural proof there is no internal resolution step left: the
        controller.compute() call succeeds even with NO 'jacobian' key at
        all in the state dict (as_robot_state() only requires it when a
        resolution mode is enabled -- see controller_core/state_types.py)."""
        new_cfg = _load_velocity_config(NEW_CONFIG)
        controller = CartesianVelocityController(new_cfg)
        controller.reset_from_state(_reset_state(WELL_CONDITIONED_Q, dyn))

        state, _J = _state_for(WELL_CONDITIONED_Q, dyn, include_jacobian=False)
        assert "jacobian" not in state
        xd_cmd = controller.compute(state)  # must not raise
        assert xd_cmd.shape == (6,)

    def test_full_hold_returns_the_raw_task_velocity_unchanged(self, dyn: LocalMujocoDynamics) -> None:
        new_cfg = _load_velocity_config(NEW_CONFIG)
        controller = CartesianVelocityController(new_cfg)
        controller.reset_from_state(_reset_state(WELL_CONDITIONED_Q, dyn))
        state, _J = _state_for(WELL_CONDITIONED_Q, dyn, include_jacobian=False)
        xd_cmd = controller.compute(state)
        # compute_full_hold is `return xd_full` (before the shared speed
        # clamp) -- at these small commanded speeds the clamp is inactive,
        # so xd_cmd should equal the raw P+feedforward/hold law directly.
        v_ff = state["target_ee_vel"]
        pos_err = state["target_ee_pos"] - state["ee_pos"]
        kp = np.array([new_cfg.kp_x, new_cfg.kp_y, new_cfg.kp_z])
        expected_v = v_ff + kp * pos_err
        np.testing.assert_allclose(xd_cmd[:3], expected_v, atol=1e-9)

    def test_dls_meaningfully_differs_from_old_pipelines_reconstruction(self, dyn: LocalMujocoDynamics) -> None:
        """The core "not a no-op" proof: DLS fed the NEW config's raw
        xd_full produces a genuinely different joint velocity than the OLD
        pipeline's internal qd -- it is not coincidentally reproducing the
        old (buggy) answer."""
        old_cfg = _load_velocity_config(OLD_CONFIG)
        J = dyn.jacobian(WELL_CONDITIONED_Q)
        xd_full = _xd_full_for(WELL_CONDITIONED_Q, dyn, old_cfg)  # same for both configs: neither cfg's kp_x/y/z/rot differ
        qd_internal_old = _internal_qd_reduced_task_dims(old_cfg, WELL_CONDITIONED_Q, dyn, xd_full)

        new_cfg = _load_velocity_config(NEW_CONFIG)
        assert new_cfg.reduced_task_dims is False
        # compute_full_hold: xd_cmd_new == xd_full exactly (no Jacobian, no resolution).
        qd_new = damped_least_squares_qd(
            J, xd_full, lambda_max=DEFAULT_DAMPING_LAMBDA_MAX, sigma0=DEFAULT_DAMPING_SIGMA0
        ).qd

        residual = float(np.linalg.norm(qd_new - qd_internal_old))
        # The old (buggy) double-resolution reconstructed qd_internal to
        # <1e-9 (see TestOldConfigDoubleResolvesAndIsBuggy above) -- the new
        # single-resolution path must differ by many orders of magnitude
        # more than that, proving DLS is now doing real, independent work.
        assert residual > 1e-3, f"expected a meaningful difference, got residual={residual}"


class TestTransportRefusesTheDoubleResolvingConfig:
    def test_run_x_transport_joint_velocity_rejects_old_config(self, tmp_path: Path) -> None:
        class _FakeReceive:
            def getActualQ(self):
                return list(WELL_CONDITIONED_Q)

            def getActualQd(self):
                return [0.0] * 6

            def getActualTCPPose(self):
                return [0.4, -0.2, 0.3, 0.0, 3.14, 0.0]

            def getTimestamp(self):
                return 1.0

            def getSafetyStatusBits(self):
                return 1

            def disconnect(self):
                pass

        class _FakeControl:
            def speedJ(self, qd, acceleration, time):
                raise AssertionError("speedJ must never be called against the old config")

            def speedStop(self, acceleration=10.0):
                pass

            def servoL(self, pose, speed, acceleration, time, lookahead_time, gain):
                pass

            def servoStop(self):
                pass

            def stopScript(self):
                pass

            def disconnect(self):
                pass

        receive = _FakeReceive()
        control = _FakeControl()
        link = UR5eLink(
            "127.0.0.1",
            125.0,
            receive_factory=lambda ip, freq: receive,
            control_factory=lambda ip, freq: control,
        )
        with pytest.raises(ValueError, match="reduced_task_dims"):
            run_x_transport_joint_velocity(
                link,
                config_path=OLD_CONFIG,
                target_x_delta_m=0.01,
                move_duration_s=0.04,
                duration_s=0.08,
                output_dir=tmp_path,
                motion_opt_in=True,
                rate_hz=50.0,
            )

    def test_run_x_transport_joint_velocity_accepts_new_config(self, tmp_path: Path) -> None:
        class _FakeReceive:
            def __init__(self):
                self._tcp_x = 0.4

            def getActualQ(self):
                return list(WELL_CONDITIONED_Q)

            def getActualQd(self):
                return [0.0] * 6

            def getActualTCPPose(self):
                return [self._tcp_x, -0.2, 0.3, 0.0, 3.14, 0.0]

            def getTimestamp(self):
                return 1.0

            def getSafetyStatusBits(self):
                return 1

            def disconnect(self):
                pass

        class _FakeControl:
            def __init__(self, receive):
                self._receive = receive
                self.speed_j_calls = 0

            def speedJ(self, qd, acceleration, time):
                self.speed_j_calls += 1
                step = min(abs(float(qd[0])) * 0.02, 0.002)
                if step > 0.0:
                    self._receive._tcp_x += step if qd[0] > 0 else -step

            def speedStop(self, acceleration=10.0):
                pass

            def servoL(self, pose, speed, acceleration, time, lookahead_time, gain):
                pass

            def servoStop(self):
                pass

            def stopScript(self):
                pass

            def disconnect(self):
                pass

        receive = _FakeReceive()
        control = _FakeControl(receive)
        link = UR5eLink(
            "127.0.0.1",
            125.0,
            receive_factory=lambda ip, freq: receive,
            control_factory=lambda ip, freq: control,
        )
        result = run_x_transport_joint_velocity(
            link,
            config_path=NEW_CONFIG,
            target_x_delta_m=0.01,
            move_duration_s=0.04,
            duration_s=0.08,
            output_dir=tmp_path,
            motion_opt_in=True,
            rate_hz=50.0,
        )
        assert control.speed_j_calls >= 2
        assert result.summary["control_mode"] == "joint_velocity"
        assert result.summary["config_path"] == str(NEW_CONFIG)
