from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "ros2_ws" / "src"))
sys.path.insert(0, str(REPO_ROOT / "ros2_ws" / "src" / "ur5_x_axis_controller_ros"))

from ur5_x_axis_controller_ros.coppeliasim_adapter import (
    CoppeliaSimConfig,
    CoppeliaSimURAdapter,
    _JointHandle,
)
from ur5_x_axis_controller_ros.coppeliasim_bridge_node import (
    _startup_joint_positions_rad,
    seed_startup_joint_positions,
)


class _FakeSim:
    jointintparam_motor_enabled = 11
    jointintparam_ctrl_enabled = 12
    jointmode_dynamic = 3
    handleflag_wxyzquat = 10_000

    def __init__(self) -> None:
        self._int_params: dict[tuple[int, int], int] = {}
        self._joint_modes: dict[int, int] = {}

    def getObjectInt32Param(self, handle: int, param: int) -> int:
        return self._int_params[(handle, param)]

    def getJointMode(self, handle: int) -> int:
        return self._joint_modes[handle]


class _TaskFrameFakeSim(_FakeSim):
    def __init__(self) -> None:
        super().__init__()
        self.created_dummy: int | None = None
        self.aliases: dict[int, str] = {}
        self.parents: list[tuple[int, int, bool]] = []
        self.poses: list[tuple[int, list[float], int]] = []

    def createDummy(self, size: float) -> int:
        assert size > 0.0
        self.created_dummy = 42
        return 42

    def setObjectAlias(self, handle: int, alias: str) -> None:
        self.aliases[int(handle)] = alias

    def setObjectParent(self, child: int, parent: int, keep_in_place: bool) -> None:
        self.parents.append((int(child), int(parent), bool(keep_in_place)))

    def setObjectPose(self, handle: int, pose: list[float], relative_to: int) -> None:
        self.poses.append((int(handle), list(pose), int(relative_to)))


class _SeedFakeAdapter:
    def __init__(self) -> None:
        self.seeded: list[list[float]] = []
        self._q = np.zeros(6, dtype=np.float64)

    def set_joint_positions(self, q: np.ndarray) -> None:
        arr = np.asarray(q, dtype=np.float64).reshape(-1)
        self.seeded.append(arr.tolist())
        self._q = arr.copy()

    def read_joint_state(self) -> tuple[np.ndarray, np.ndarray]:
        return self._q.copy(), np.zeros_like(self._q)


class _WrappedSeedFakeAdapter(_SeedFakeAdapter):
    def read_joint_state(self) -> tuple[np.ndarray, np.ndarray]:
        q = self._q.copy()
        q[5] += 2.0 * np.pi
        return q, np.zeros_like(q)


def _make_adapter() -> CoppeliaSimURAdapter:
    adapter = CoppeliaSimURAdapter(CoppeliaSimConfig())
    adapter._joint_handles = [
        _JointHandle("shoulder_pan_joint", 1, "/UR5/joint"),
        _JointHandle("shoulder_lift_joint", 2, "/UR5/link/joint"),
    ]
    return adapter


def test_joint_configuration_summary_reports_verified_modes() -> None:
    adapter = _make_adapter()
    fake = _FakeSim()
    fake._int_params[(1, fake.jointintparam_motor_enabled)] = 1
    fake._int_params[(1, fake.jointintparam_ctrl_enabled)] = 0
    fake._int_params[(2, fake.jointintparam_motor_enabled)] = 1
    fake._int_params[(2, fake.jointintparam_ctrl_enabled)] = 0
    fake._joint_modes[1] = fake.jointmode_dynamic
    fake._joint_modes[2] = fake.jointmode_dynamic
    adapter._sim = fake

    summary = adapter.read_joint_configuration_summary()

    assert summary["motor_enabled_verified"] is True
    assert summary["ctrl_disabled_verified"] is True
    assert summary["dynamic_mode_verified"] is True
    assert summary["joint_mode_readback_available"] is True
    assert summary["motor_readback_available"] is True
    assert summary["ctrl_readback_available"] is True


def test_compare_jacobians_reports_matrix_deltas() -> None:
    adapter = _make_adapter()
    api_pos = np.eye(3, 6, dtype=np.float64)
    api_rot = np.zeros((3, 6), dtype=np.float64)
    num_pos = api_pos.copy()
    num_pos[0, 0] += 0.5
    num_rot = np.zeros((3, 6), dtype=np.float64)
    num_rot[2, 5] = -0.25

    adapter.read_jacobian_api = lambda: (api_pos, api_rot)  # type: ignore[assignment]
    adapter.read_jacobian_numerical = lambda epsilon: (num_pos, num_rot)  # type: ignore[assignment]

    summary = adapter.compare_jacobians(1.0e-5)

    assert summary["difference"]["all_finite"] is True
    assert np.isclose(summary["difference"]["max_abs"], 0.5)
    assert np.isclose(summary["difference"]["position_max_abs"], 0.5)
    assert np.isclose(summary["difference"]["rotation_max_abs"], 0.25)
    assert summary["api"]["shape"] == [6, 6]
    assert summary["numerical"]["shape"] == [6, 6]


def test_mujoco_attachment_task_frame_creates_parented_dummy() -> None:
    adapter = CoppeliaSimURAdapter(
        CoppeliaSimConfig(task_frame_mode="mujoco_attachment_dummy")
    )
    fake = _TaskFrameFakeSim()
    adapter._sim = fake
    adapter._raw_ee_handle = 7
    adapter._raw_ee_resolved_path = "/UR5/UR5_connection"

    adapter._configure_task_frame_or_raise()
    summary = adapter.read_task_frame_summary()

    assert summary["mode"] == "mujoco_attachment_dummy"
    assert summary["handle"] == 42
    assert summary["parent_handle"] == 7
    assert summary["local_offset_m"] == [0.0, 0.0, -0.2]
    assert np.isclose(np.linalg.norm(summary["local_quat_wxyz"]), 1.0)
    assert fake.parents == [(42, 7, False)]
    assert fake.poses[0][0] == 42 + fake.handleflag_wxyzquat


def test_startup_joint_positions_seed_helper_applies_and_verifies() -> None:
    seed = _startup_joint_positions_rad([0.2, -1.15, 1.55, -1.8, -1.45, 0.35])
    fake = _SeedFakeAdapter()

    seed_startup_joint_positions(fake, seed, log_fn=lambda _: None)

    assert fake.seeded == [list(seed)]
    np.testing.assert_allclose(fake.read_joint_state()[0], np.asarray(seed))


def test_startup_joint_positions_seed_helper_accepts_wrapped_revolute_angles() -> None:
    seed = _startup_joint_positions_rad([0.2, -1.15, 1.55, -1.8, -1.45, 0.35])
    fake = _WrappedSeedFakeAdapter()

    seed_startup_joint_positions(fake, seed, log_fn=lambda _: None)

    assert fake.seeded == [list(seed)]
