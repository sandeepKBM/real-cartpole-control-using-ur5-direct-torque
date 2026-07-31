"""Tests for the opt-in async residual observer (2026-07-31) -- see
docs/status/direct_torque_residual_observer_async_2026-07-31.md.

Covers:
- sync mode (residual_observer_async=False, the default) is unchanged --
  verified against a golden trace captured from the code as it existed
  immediately before this change landed (fixtures/
  direct_torque_sync_pre_async_baseline_trace.json), under a fully
  deterministic fake clock (see _install_fake_clock below) so the
  comparison is genuinely byte-for-byte, not "close enough" -- real
  wall-clock jitter would otherwise make even identical code produce
  slightly different cycle_work_ms/lateness_ms/qdd_* (the latter because
  qdd estimation divides by real elapsed time) across separate runs.
- async mode produces the same computed diagnostic values as sync mode for
  every cycle whose result makes it back before shutdown.
- the producer side (ResidualObserverWorkerHandle.submit) never blocks, even
  when flooded far faster than any consumer could drain.
- the worker process is fully gone (no zombie) after both a normal exit and
  an exception propagating out of the control loop.
"""

from __future__ import annotations

import json
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np
import pytest

import hardware.direct_torque_transport as dtt
from hardware.direct_torque_link import UR5eDirectTorqueLink
from hardware.link import RTDEStateError, UR5eState
from hardware.poses import HEIGHT_ALPHA_0_5_Q
from hardware.residual_observer_worker import ResidualObserverWorkerHandle, start_residual_observer_worker
from hardware.safety import UR5eSafetyLimits

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG = REPO_ROOT / "config" / "ur5e_mujoco_torque_osc_tuned.yaml"
GOLDEN_TRACE_PATH = Path(__file__).resolve().parent / "fixtures" / "direct_torque_sync_pre_async_baseline_trace.json"


class _MockDTLink:
    def __init__(self) -> None:
        self._tcp_x = 0.35
        self.limits = UR5eSafetyLimits()
        self.connect_calls = 0
        self.n_reads = 0
        self.raise_after: int | None = None

    def connect(self) -> None:
        self.connect_calls += 1

    def read_state(self) -> UR5eState:
        self.n_reads += 1
        if self.raise_after is not None and self.n_reads > self.raise_after:
            raise RuntimeError("synthetic mid-run failure")
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
        self._tcp_x += float(tau_nm[0]) * 1e-6

    @staticmethod
    def compose_robot_state(link_state, *, jacobian, mass_matrix, time_s, target_x, target_x_vel):
        return UR5eDirectTorqueLink.compose_robot_state(
            link_state, jacobian=jacobian, mass_matrix=mass_matrix,
            time_s=time_s, target_x=target_x, target_x_vel=target_x_vel,
        )

    def build_robot_state(self, link_state, *, time_s, target_x, target_x_vel):
        return self.compose_robot_state(
            link_state, jacobian=self.get_jacobian(), mass_matrix=self.get_mass_matrix(),
            time_s=time_s, target_x=target_x, target_x_vel=target_x_vel,
        )

    def safe_stop(self, reason: str) -> None:
        pass


class _FakeClock:
    """Deterministic stand-in for time.monotonic_ns()/time.sleep() so a run's
    cycle timing (and therefore interval_ns/real_dt_s, which feed the
    residual observer's finite-difference qdd estimate) is 100% reproducible
    across separate process invocations."""

    def __init__(self) -> None:
        self.ns = 0

    def monotonic_ns(self) -> int:
        self.ns += 50_000  # 0.05 ms per poll -- arbitrary but fixed
        return self.ns

    def sleep(self, seconds: float) -> None:
        self.ns += int(seconds * 1e9)


def _install_fake_clock(monkeypatch: pytest.MonkeyPatch) -> _FakeClock:
    clock = _FakeClock()
    monkeypatch.setattr(dtt, "monotonic_ns", clock.monotonic_ns)
    monkeypatch.setattr(dtt.time, "sleep", clock.sleep)
    return clock


def _run_scenario(*, output_dir: Path, async_mode: bool | None = None, **extra_kwargs):
    link = _MockDTLink()
    kwargs = dict(
        config_path=CONFIG,
        target_x_delta_m=0.01,
        move_duration_s=0.02,
        duration_s=0.05,
        output_dir=output_dir,
        motion_opt_in=True,
        record_latency=False,
        dynamics_source="local",
        residual_qdd_gap_cycles=2,
    )
    if async_mode is not None:
        kwargs["residual_observer_async"] = async_mode
    kwargs.update(extra_kwargs)
    return dtt.run_x_transport_direct_torque(link, **kwargs)


def _load_trace(trace_path: Path) -> list[dict]:
    return [json.loads(line) for line in trace_path.read_text().splitlines()]


@pytest.mark.hardware
def test_sync_mode_default_matches_pre_async_change_golden_trace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """residual_observer_async defaults to False; under a deterministic
    clock, today's sync path must reproduce EXACTLY the trace captured from
    the code as it existed immediately before this change (git commit
    f1748c1) -- including cycle_work_ms/lateness_ms, which are only
    deterministic here because the clock is fake."""
    _install_fake_clock(monkeypatch)
    result = _run_scenario(output_dir=tmp_path)  # async_mode omitted -> default False
    rows = _load_trace(result.trace_path)
    golden = json.loads(GOLDEN_TRACE_PATH.read_text())
    assert rows == golden


@pytest.mark.hardware
def test_sync_mode_explicit_false_matches_default(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Passing residual_observer_async=False explicitly must be identical to
    omitting it."""
    _install_fake_clock(monkeypatch)
    result_default = _run_scenario(output_dir=tmp_path / "default")
    _install_fake_clock(monkeypatch)  # fresh clock instance, same deterministic sequence
    result_explicit = _run_scenario(output_dir=tmp_path / "explicit", async_mode=False)
    assert _load_trace(result_default.trace_path) == _load_trace(result_explicit.trace_path)


@pytest.mark.hardware
def test_async_mode_produces_equivalent_diagnostic_values(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Async mode's merged qdd_pred/qdd_measured/qdd_residual values must
    exactly match the sync path's for every step that gets merged back (this
    short scenario -- a handful of cycles -- completes well within the
    default shutdown timeout, so every step should merge)."""
    _install_fake_clock(monkeypatch)
    sync_result = _run_scenario(output_dir=tmp_path / "sync", async_mode=False)
    _install_fake_clock(monkeypatch)
    async_result = _run_scenario(output_dir=tmp_path / "async", async_mode=True)

    sync_rows = _load_trace(sync_result.trace_path)
    async_rows = _load_trace(async_result.trace_path)
    assert len(sync_rows) == len(async_rows) >= 3

    lifecycle = async_result.summary["residual_observer_async"]
    assert lifecycle["enabled"] is True
    assert lifecycle["dropped_request_count"] == 0
    assert lifecycle["unmerged_step_count"] == 0
    assert lifecycle["worker_exitcode"] == 0
    assert lifecycle["worker_terminated_forcefully"] is False

    for sync_row, async_row in zip(sync_rows, async_rows):
        for field in ("qdd_pred", "qdd_measured", "qdd_residual", "qdd_residual_norm"):
            sv, av = sync_row[field], async_row[field]
            if sv is None or av is None:
                assert sv is None and av is None, f"{field}: sync={sv!r} async={av!r}"
            else:
                np.testing.assert_allclose(sv, av, atol=1e-12)


@pytest.mark.hardware
def test_producer_never_blocks_under_backpressure() -> None:
    """Flood ResidualObserverWorkerHandle.submit() far faster than a
    deliberately-absent consumer can drain a tiny queue -- confirms submit()
    never raises and never blocks, only drops and counts."""
    ctx = mp.get_context("spawn")
    request_queue = ctx.Queue(maxsize=2)
    result_queue = ctx.Queue(maxsize=2)
    stop_event = ctx.Event()

    class _NeverStartedProcess:
        """Stand-in for mp.Process -- no real consumer ever reads
        request_queue, so it fills up almost immediately and every
        subsequent submit() must be dropped, not blocked."""

        def is_alive(self) -> bool:
            return False

        def join(self, timeout: float | None = None) -> None:
            return None

        def terminate(self) -> None:
            return None

        def kill(self) -> None:
            return None

        exitcode = 0

    handle = ResidualObserverWorkerHandle(
        process=_NeverStartedProcess(),  # type: ignore[arg-type]
        request_queue=request_queue,
        result_queue=result_queue,
        stop_event=stop_event,
    )

    q = np.zeros(6)
    tau = np.ones(6)
    mm = np.eye(6)
    n = 2000
    t0 = time.monotonic()
    for i in range(n):
        handle.submit(step=i, q=q, qd=q, tau=tau, mass_matrix=mm, real_dt_s=0.002)
    elapsed_s = time.monotonic() - t0

    # 2000 non-blocking calls against a maxsize=2 queue with no consumer:
    # if any of these ever blocked, this would take a very long time.
    # Generous bound (this is a shared cluster host per AGENTS.md SS8).
    assert elapsed_s < 2.0, f"submit() calls took {elapsed_s:.3f}s -- looks like it blocked"
    assert handle.dropped_request_count >= n - 2

    handle.request_queue.cancel_join_thread()
    handle.result_queue.cancel_join_thread()
    handle.request_queue.close()
    handle.result_queue.close()


@pytest.mark.hardware
def test_clean_process_lifecycle_normal_exit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """After a normal (non-erroring) async run completes, no
    residual-observer-worker process may remain among this process's
    children."""
    _install_fake_clock(monkeypatch)
    before = {p.name for p in mp.active_children()}
    result = _run_scenario(output_dir=tmp_path, async_mode=True)
    assert result.summary["residual_observer_async"]["worker_exitcode"] == 0
    after = {p.name for p in mp.active_children()}
    assert "residual-observer-worker" not in after
    assert after - before == set()


@pytest.mark.hardware
def test_clean_process_lifecycle_exception_mid_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """If an exception (not RTDEStateError, which is caught) propagates out
    of the control loop mid-run, the worker process must still be fully torn
    down -- no zombie left behind."""
    _install_fake_clock(monkeypatch)
    link = _MockDTLink()
    link.raise_after = 2  # let a couple of cycles run, then blow up

    with pytest.raises(RuntimeError, match="synthetic mid-run failure"):
        dtt.run_x_transport_direct_torque(
            link,
            config_path=CONFIG,
            target_x_delta_m=0.01,
            move_duration_s=0.02,
            duration_s=0.05,
            output_dir=tmp_path,
            motion_opt_in=True,
            record_latency=False,
            dynamics_source="local",
            residual_qdd_gap_cycles=2,
            residual_observer_async=True,
        )

    after = {p.name for p in mp.active_children()}
    assert "residual-observer-worker" not in after


@pytest.mark.hardware
def test_rtde_state_error_mid_run_also_tears_down_worker(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """RTDEStateError is caught inline (turned into an estop) rather than
    propagating -- confirm the worker is still cleanly torn down on this
    path too, and the function returns normally."""
    _install_fake_clock(monkeypatch)

    class _RaisingLink(_MockDTLink):
        def read_state(self):
            self.n_reads += 1
            if self.n_reads > 2:
                raise RTDEStateError("synthetic RTDE stall")
            return super().read_state()

    link = _RaisingLink()
    result = dtt.run_x_transport_direct_torque(
        link,
        config_path=CONFIG,
        target_x_delta_m=0.01,
        move_duration_s=0.02,
        duration_s=0.05,
        output_dir=tmp_path,
        motion_opt_in=True,
        record_latency=False,
        dynamics_source="local",
        residual_qdd_gap_cycles=2,
        residual_observer_async=True,
    )
    assert result.summary["termination_reason"].startswith("rtde_state_error")
    after = {p.name for p in mp.active_children()}
    assert "residual-observer-worker" not in after


@pytest.mark.hardware
@pytest.mark.slow
def test_residual_observer_async_phase_cost_is_much_lower_than_sync(tmp_path: Path) -> None:
    """Real wall-clock timing evidence (no fake clock here -- this is the
    genuine per-cycle cost, matching
    docs/status/direct_torque_residual_observer_async_2026-07-31.md's
    reported numbers): a longer run (500 real cycles) comparing the
    latency_phases['residual_observer'] phase, sync vs async."""

    class _SlowMoveLink(_MockDTLink):
        def direct_torque(self, tau_nm: np.ndarray, *, friction_comp: bool = True) -> None:
            self._tcp_x += float(tau_nm[0]) * 1e-9  # tiny gain: never trips the move guard

    def run(async_mode: bool) -> dict:
        link = _SlowMoveLink()
        result = dtt.run_x_transport_direct_torque(
            link,
            config_path=CONFIG,
            target_x_delta_m=0.0001,
            move_duration_s=0.3,
            duration_s=1.0,
            output_dir=None,
            motion_opt_in=True,
            record_latency=True,
            dynamics_source="local",
            residual_qdd_gap_cycles=2,
            residual_observer_async=async_mode,
        )
        return result.summary

    sync_summary = run(async_mode=False)
    async_summary = run(async_mode=True)

    assert sync_summary["termination_reason"] == "duration_complete"
    assert async_summary["termination_reason"] == "duration_complete"

    sync_phase = sync_summary["latency_phases"]["residual_observer"]
    async_phase = async_summary["latency_phases"]["residual_observer"]

    # Generous, non-flaky margin (shared cluster host, AGENTS.md SS8) -- the
    # real measured ratio was ~0.20-0.25x (async/sync mean); assert well
    # short of that so this doesn't flake under host load.
    assert async_phase["mean_ms"] < sync_phase["mean_ms"] * 0.6, (
        f"async mean {async_phase['mean_ms']:.5f}ms not meaningfully lower than "
        f"sync mean {sync_phase['mean_ms']:.5f}ms"
    )

    async_lifecycle = async_summary["residual_observer_async"]
    assert async_lifecycle["worker_exitcode"] == 0
    assert async_lifecycle["worker_terminated_forcefully"] is False
    assert async_lifecycle["unmerged_step_count"] == 0
