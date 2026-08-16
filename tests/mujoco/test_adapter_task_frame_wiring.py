"""Adapter-level wiring of the task-frame drift guard.

``ImpedanceSafetyMonitor`` grew ``task_rotation``/``tracked_axes`` support, but
nothing could reach it: ``MujocoUR5eTorqueAdapter.reset()`` called
``set_initial_position()`` positionally with world axes and ``check()`` was
never given a rotation. Harnesses worked around that by rotating the *state*
before handing it over, which silently mixes frames -- the monitor then stores
an origin rotated under R(0) and compares it against a position rotated under
R(t).

The tests are ordered by what breaks worst if wrong:
  1. The default path is untouched (this guard is shared with real hardware).
  2. A live frame reaches ``check()`` every cycle, which is the whole point.
  3. Misuse is rejected loudly.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import mujoco  # noqa: E402
import yaml  # noqa: E402

from simulation.ur5e_mujoco_torque import (  # noqa: E402
    build_initial_state_and_adapter,
)

SCENE = REPO_ROOT / "assets/ur5e_torque/scene.xml"
# The real-hardware default config -- this wiring must work against the
# controller actually in use, not a synthetic stub.
CONFIG = REPO_ROOT / "config/ur5e_mujoco_torque_osc_tuned.yaml"
ARM_Q0 = np.array([-2.3688, -2.1801, -1.8838, -0.7962, 0.004714693, 0.0206])

# A proper right-handed basis, deliberately NOT axis-aligned so a dropped
# rotation cannot coincidentally pass.
_RAW = np.array([[0.7103, -0.0941, -0.6976],
                 [0.6924, -0.0851, 0.7164],
                 [0.1268, 0.9919, -0.0047]])


def _basis() -> np.ndarray:
    q, r = np.linalg.qr(_RAW)
    return q * np.sign(np.diag(r))


@pytest.fixture(scope="module")
def scene():
    model = mujoco.MjModel.from_xml_path(str(SCENE))
    data = mujoco.MjData(model)
    data.qpos[:6] = ARM_Q0
    mujoco.mj_forward(model, data)
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "attachment_site")
    joint_ids = list(range(6))
    return model, data, site_id, joint_ids


def _controller_cfg() -> dict:
    with open(CONFIG) as f:
        return yaml.safe_load(f)["controller"]


def _build(scene):
    model, data, site_id, joint_ids = scene
    return build_initial_state_and_adapter(
        model, data, site_id, joint_ids,
        controller_cfg=_controller_cfg(),
        transport_axis_index=0,
        target_x_delta=0.0,
        controller_kind="impedance",
        force_hold_current_pose=True,
        gravity_mode="gravity_comp",
        torque_limit_scale=1.0,
    )


# ===================== 1. DEFAULT PATH UNTOUCHED =========================

def test_default_adapter_leaves_guard_in_world_frame(scene):
    """No configure_task_frame() call -> historical world-axis behavior."""
    state, adapter = _build(scene)
    adapter.reset(state)
    assert adapter.safety_monitor.task_rotation is None


def test_configure_none_is_still_world_frame(scene):
    """Calling it with all-None must not switch frames by accident."""
    state, adapter = _build(scene)
    adapter.configure_task_frame()
    adapter.reset(state)
    assert adapter.safety_monitor.task_rotation is None


# ===================== 2. THE FRAME ACTUALLY REACHES THE GUARD ============

def test_fixed_rotation_reaches_the_monitor(scene):
    state, adapter = _build(scene)
    R = _basis()
    adapter.configure_task_frame(task_rotation=R, tracked_axes=[0, 1])
    adapter.reset(state)
    assert adapter.safety_monitor.task_rotation is not None
    np.testing.assert_allclose(adapter.safety_monitor.task_rotation, R, atol=1e-12)


def test_live_rotation_fn_is_called_every_cycle(scene):
    """The live frame is the reason this wiring exists: a fixed R captured at
    reset goes stale as the tool rotates (~12 deg over a 0.22 m move)."""
    model, data, site_id, joint_ids = scene
    state, adapter = _build(scene)

    calls: list[int] = []

    def live_R(_state) -> np.ndarray:
        calls.append(1)
        return _basis()

    adapter.configure_task_frame(task_rotation_fn=live_R, tracked_axes=[0, 1])
    adapter.reset(state)
    assert len(calls) == 1, "reset() must seed the origin with a live frame too"

    adapter.step(state=state)
    assert len(calls) >= 2, "check() must receive a freshly evaluated rotation"


def test_rotating_the_frame_does_not_move_a_stationary_tool(scene):
    """Origin stays in WORLD and the DISPLACEMENT is rotated, so spinning the
    frame on a stationary tool must report exactly zero drift -- the property
    that makes a live frame sound in the first place."""
    state, adapter = _build(scene)
    adapter.configure_task_frame(task_rotation=np.eye(3), tracked_axes=[0, 1])
    adapter.reset(state)

    ee = np.asarray(state.ee_pos, dtype=np.float64).reshape(3)
    for R in (np.eye(3), _basis()):
        d = adapter.safety_monitor.drift_vector(ee, task_rotation=R)
        np.testing.assert_allclose(d, np.zeros(3), atol=1e-12)


# ===================== 3. MISUSE REJECTED ================================

def test_fixed_and_live_rotation_are_mutually_exclusive(scene):
    _, adapter = _build(scene)
    with pytest.raises(ValueError, match="not both"):
        adapter.configure_task_frame(
            task_rotation=_basis(), task_rotation_fn=lambda s: _basis()
        )


def test_bad_rotation_rejected_at_configure_time(scene):
    _, adapter = _build(scene)
    with pytest.raises(ValueError, match="orthonormal"):
        adapter.configure_task_frame(task_rotation=np.diag([1.0, 1.0, 2.0]))


def test_tracked_axes_cannot_disable_the_guard_via_adapter(scene):
    """The all-three-axes escape hatch must stay closed through this path."""
    state, adapter = _build(scene)
    adapter.configure_task_frame(task_rotation=_basis(), tracked_axes=[0, 1, 2])
    with pytest.raises(ValueError, match="silently disables the guard"):
        adapter.reset(state)
