"""Off-critical-path worker process for the diagnostic-only dynamics residual
observer (2026-07-31).

Real hardware tonight measured the inline residual-observer phase
(``hardware/direct_torque_transport.py``'s per-cycle ``coriolis()`` +
``predict_joint_acceleration`` + ``JointAccelEstimator.update`` block) at
~0.15ms mean but with real, large occasional spikes; disabling it entirely
(``--disable-residual-observer``) eliminated a real deadline-overrun trip on
the 500 Hz loop. This module moves that computation into a separate
``multiprocessing.Process`` so its cost -- and especially its tail-latency
spikes -- leaves the control loop's timing budget entirely, while still
capturing the same diagnostic data with a real, explicitly-acceptable delay
(a few cycles is fine; this is never read by any safety guard).

Contract with the 500 Hz caller (``hardware/direct_torque_transport.py``):

- The worker process is started ONCE, before the per-cycle loop begins, via
  :func:`start_residual_observer_worker`.
- Every cycle, the control loop calls :meth:`ResidualObserverWorkerHandle.submit`
  -- a strictly non-blocking enqueue (``put_nowait`` under the hood). If the
  bounded request queue is full, that cycle's sample is DROPPED and a counter
  incremented; ``submit`` never raises and never blocks, under any
  circumstance. This is the only method of this module the hot loop may call.
- The worker owns its OWN ``PinocchioUR5eDynamics`` + ``JointAccelEstimator``
  instances, constructed fresh inside the child process (these objects wrap
  C++ state via pybind11 and cannot be pickled across a process boundary).
- Results come back tagged by step index via a second bounded queue, so they
  can be merged into ``trace_rows`` after the fact
  (:meth:`ResidualObserverWorkerHandle.shutdown_and_collect`, called once
  after the per-cycle loop exits -- normal completion, e-stop, or exception).
  Cycles whose request was dropped, or whose result arrives after the worker
  is torn down, are never computed -- the corresponding trace row keeps
  ``qdd_*=None``, the same convention already used today for "gap window
  still filling."

See ``docs/status/direct_torque_residual_observer_async_2026-07-31.md`` for
the timing evidence and validation this module was built against.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import queue
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from controller_core.dynamics_residual import joint_acceleration_residual, predict_joint_acceleration
from controller_core.model_dynamics import PinocchioUR5eDynamics

from .joint_accel_estimator import JointAccelEstimator

# "A few seconds of buffering at 500 Hz" (task spec) -- 2000 slots is 4s of
# headroom at 500 Hz before the producer side starts dropping samples. Small
# per-item payload (six 6-vectors/one 6x6 matrix worth of float64 + a couple
# scalars), so the worst-case buffered memory is a few MB, not a concern.
DEFAULT_REQUEST_QUEUE_MAXSIZE = 2000
DEFAULT_RESULT_QUEUE_MAXSIZE = 2000
DEFAULT_SHUTDOWN_JOIN_TIMEOUT_S = 2.0
# Real finding (2026-07-31, see
# docs/status/direct_torque_residual_observer_async_2026-07-31.md): AGENTS.md
# S8 documents that an unconstrained worker process auto-detects the full
# core count and spawns that many BLAS threads for itself the moment it
# loads numpy/Pinocchio -- on this 72-core shared host, that produced a real,
# measured deadline-overrun trip in the PARENT's 500 Hz loop during the
# child's one-time PinocchioUR5eDynamics() construction (competing for real
# CPU cycles with the main loop's own real-time thread even though the
# child's *steady-state* per-cycle work is cheap). Pinned to 1 here for
# exactly the reason AGENTS.md S8 gives: parallelism should come from having
# a second PROCESS, not from that process also being internally
# multi-threaded.
_WORKER_THREAD_ENV_VARS = (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)
# How many consecutive empty reads (each capped at _WORKER_GET_TIMEOUT_S) the
# worker tolerates after stop_event is set before concluding the request
# queue is truly drained. Guards against multiprocessing.Queue's put()/empty()
# feeder-thread propagation race (an item can be put() by the main process
# slightly before a worker-side get()/empty() call observes it).
_WORKER_GET_TIMEOUT_S = 0.05
_WORKER_STOP_GRACE_EMPTY_READS = 3


@dataclass
class ResidualObserverRequest:
    """Per-cycle payload sent to the worker. Mirrors exactly the inputs the
    inline computation in ``direct_torque_transport.py`` uses today:
    ``residual_dynamics.coriolis(q, qd)``, then
    ``predict_joint_acceleration(mass_matrix, tau, coriolis_term)``, then
    ``JointAccelEstimator.update(qd, real_dt_s)``."""

    step: int
    q: np.ndarray
    qd: np.ndarray
    tau: np.ndarray
    mass_matrix: np.ndarray
    real_dt_s: float


@dataclass
class ResidualAsyncSummary:
    """Returned once, at the end of a run, by
    :meth:`ResidualObserverWorkerHandle.shutdown_and_collect`."""

    results: dict[int, dict[str, Any]]
    dropped_request_count: int
    # None if the worker never reported (e.g. it was forcefully killed before
    # it could send its final-stats message) -- best-effort, not a hard
    # guarantee, since the worker's own result queue can itself be under
    # backpressure right at shutdown.
    dropped_result_count: int | None
    worker_init_error: str | None
    exitcode: int | None
    terminated_forcefully: bool


def _residual_observer_worker_main(
    request_queue: "mp.Queue",
    result_queue: "mp.Queue",
    stop_event: Any,
    *,
    gap_cycles: int,
    lowpass_alpha: float,
    qd0: np.ndarray,
) -> None:
    """Worker process entry point. Must be a module-level function (not a
    closure/lambda) so it is picklable under the ``spawn`` start method.

    Builds its own ``PinocchioUR5eDynamics`` + ``JointAccelEstimator`` -- see
    the module docstring for why these can't be constructed in the parent and
    passed across the process boundary. Never raises out of this function:
    an init failure or a per-cycle compute error is reported back via
    ``result_queue`` (best-effort) rather than crashing the process in a way
    that could confuse the parent's lifecycle bookkeeping.
    """
    try:
        dynamics = PinocchioUR5eDynamics()
        estimator = JointAccelEstimator(gap_cycles=gap_cycles, lowpass_alpha=lowpass_alpha)
        estimator.reset(np.asarray(qd0, dtype=np.float64))
    except Exception as exc:  # noqa: BLE001 - diagnostic-only, must never hang/crash silently
        try:
            result_queue.put_nowait({"init_error": f"{type(exc).__name__}: {exc}"})
        except queue.Full:
            pass
        return

    dropped_result_count = 0
    consecutive_empty_after_stop = 0
    while True:
        stopping = bool(stop_event.is_set())
        try:
            request = request_queue.get(timeout=_WORKER_GET_TIMEOUT_S)
        except queue.Empty:
            if stopping:
                consecutive_empty_after_stop += 1
                if consecutive_empty_after_stop >= _WORKER_STOP_GRACE_EMPTY_READS:
                    break
            continue
        consecutive_empty_after_stop = 0

        try:
            coriolis_term = dynamics.coriolis(request.q, request.qd)
            qdd_pred = predict_joint_acceleration(request.mass_matrix, request.tau, coriolis_term)
            qdd_measured = estimator.update(request.qd, request.real_dt_s)
            qdd_residual = (
                None if qdd_measured is None else joint_acceleration_residual(qdd_measured, qdd_pred)
            )
            payload: dict[str, Any] = {
                "step": int(request.step),
                "qdd_pred": qdd_pred.tolist(),
                "qdd_measured": None if qdd_measured is None else qdd_measured.tolist(),
                "qdd_residual": None if qdd_residual is None else qdd_residual.tolist(),
                "qdd_residual_norm": (
                    None if qdd_residual is None else float(np.linalg.norm(qdd_residual))
                ),
            }
        except Exception as exc:  # noqa: BLE001 - one bad cycle must not kill the worker
            payload = {"step": int(request.step), "error": f"{type(exc).__name__}: {exc}"}

        try:
            result_queue.put_nowait(payload)
        except queue.Full:
            # Symmetric with the producer side's own backpressure handling:
            # the worker must not block on a full result queue either (that
            # would eventually stall it from ever noticing stop_event).
            dropped_result_count += 1

    try:
        result_queue.put_nowait({"final_stats": True, "dropped_result_count": dropped_result_count})
    except queue.Full:
        pass  # best-effort; the count is a diagnostic nicety, not required for correctness


class ResidualObserverWorkerHandle:
    """Owns the worker process and its two queues.

    ``submit()`` is the only method the 500 Hz control loop may call -- it
    never blocks and never raises. ``shutdown_and_collect()`` is called
    exactly once, after the per-cycle loop has exited (any exit path), and
    guarantees the worker process is gone (terminated/killed if it doesn't
    exit cleanly within the timeout) before returning.
    """

    def __init__(
        self,
        *,
        process: "mp.process.BaseProcess",
        request_queue: "mp.Queue",
        result_queue: "mp.Queue",
        stop_event: Any,
    ) -> None:
        self.process = process
        self.request_queue = request_queue
        self.result_queue = result_queue
        self.stop_event = stop_event
        self.dropped_request_count = 0
        self._shutdown_called = False

    def submit(
        self,
        *,
        step: int,
        q: np.ndarray,
        qd: np.ndarray,
        tau: np.ndarray,
        mass_matrix: np.ndarray,
        real_dt_s: float,
    ) -> bool:
        """Non-blocking enqueue. Returns True if accepted, False if dropped
        (queue full). NEVER raises, NEVER blocks -- safe to call every cycle
        from the 500 Hz control loop."""
        request = ResidualObserverRequest(
            step=int(step),
            q=np.asarray(q, dtype=np.float64).copy(),
            qd=np.asarray(qd, dtype=np.float64).copy(),
            tau=np.asarray(tau, dtype=np.float64).copy(),
            mass_matrix=np.asarray(mass_matrix, dtype=np.float64).copy(),
            real_dt_s=float(real_dt_s),
        )
        try:
            self.request_queue.put_nowait(request)
            return True
        except queue.Full:
            self.dropped_request_count += 1
            return False

    def shutdown_and_collect(
        self, *, join_timeout_s: float = DEFAULT_SHUTDOWN_JOIN_TIMEOUT_S
    ) -> ResidualAsyncSummary:
        """Signal the worker to stop, wait for a clean exit (escalating to
        ``terminate()``/``kill()`` if it doesn't exit within
        ``join_timeout_s``), then drain every result still sitting in
        ``result_queue`` and merge-ready them by step index.

        Must be called exactly once, after the per-cycle loop has exited via
        ANY path (normal completion, e-stop, or an exception propagating out
        of the loop) -- callers should invoke this from a ``finally`` block
        so no zombie process can survive a run under any exit path.

        Real finding (2026-07-31, see
        docs/status/direct_torque_residual_observer_async_2026-07-31.md):
        draining MUST happen concurrently with waiting for the process to
        exit, not strictly after ``process.join()`` returns. This is a
        documented ``multiprocessing.Queue`` gotcha, not an optimization --
        each ``put_nowait()`` in the worker only enqueues to an internal
        deque; a background *feeder thread* in the worker process actually
        pushes those bytes through the real OS pipe, whose buffer is much
        smaller than this queue's logical ``maxsize``. If nobody reads the
        other end while the worker tries to exit, that feeder thread blocks
        trying to flush its backlog, which blocks the whole process from
        exiting -- so ``process.join()`` times out and the worker gets
        force-killed with a large, entirely real backlog of already-computed
        results still stuck in the pipe, never delivered. Measured before
        this fix: a 2000-item burst finished its OWN compute loop in ~130ms
        internally, but the process would not actually exit within an 11s
        wait, and only ~35% of its results were ever recovered before being
        force-killed.
        """
        if self._shutdown_called:
            raise RuntimeError("shutdown_and_collect() must only be called once")
        self._shutdown_called = True

        results: dict[int, dict[str, Any]] = {}
        dropped_result_count: int | None = None
        worker_init_error: str | None = None

        def _drain_available() -> None:
            nonlocal dropped_result_count, worker_init_error
            while True:
                try:
                    payload = self.result_queue.get_nowait()
                except queue.Empty:
                    return
                if payload.get("init_error") is not None:
                    worker_init_error = payload["init_error"]
                    continue
                if payload.get("final_stats"):
                    dropped_result_count = int(payload.get("dropped_result_count", 0))
                    continue
                results[int(payload["step"])] = payload

        self.stop_event.set()
        deadline = time.monotonic() + join_timeout_s
        while time.monotonic() < deadline:
            _drain_available()
            if not self.process.is_alive():
                break
            self.process.join(0.02)
        _drain_available()

        terminated_forcefully = False
        if self.process.is_alive():
            self.process.terminate()
            self.process.join(1.0)
            terminated_forcefully = True
            _drain_available()
        if self.process.is_alive():
            self.process.kill()
            self.process.join(1.0)
            terminated_forcefully = True
        _drain_available()

        # Avoid the well-known multiprocessing.Queue shutdown hang (a feeder
        # thread trying to flush any still-buffered items at interpreter
        # exit/GC) -- the worker side is already gone, so nothing will ever
        # consume from request_queue again regardless.
        self.request_queue.cancel_join_thread()
        self.result_queue.cancel_join_thread()
        try:
            self.request_queue.close()
        except Exception:  # noqa: BLE001 - best-effort cleanup only
            pass
        try:
            self.result_queue.close()
        except Exception:  # noqa: BLE001 - best-effort cleanup only
            pass

        return ResidualAsyncSummary(
            results=results,
            dropped_request_count=self.dropped_request_count,
            dropped_result_count=dropped_result_count,
            worker_init_error=worker_init_error,
            exitcode=self.process.exitcode,
            terminated_forcefully=terminated_forcefully,
        )


def start_residual_observer_worker(
    *,
    qd0: np.ndarray,
    gap_cycles: int,
    lowpass_alpha: float,
    request_maxsize: int = DEFAULT_REQUEST_QUEUE_MAXSIZE,
    result_maxsize: int = DEFAULT_RESULT_QUEUE_MAXSIZE,
) -> ResidualObserverWorkerHandle:
    """Start the worker process once, before the per-cycle loop begins.

    Uses the ``spawn`` start method (not the Linux-default ``fork``) so the
    child gets a fresh interpreter and does not inherit any of the parent's
    already-loaded C-extension state (Pinocchio/Eigen/BLAS thread pools) --
    safer than forking a process that has already loaded those libraries.
    Paid once at startup, off the 500 Hz loop's timing budget entirely.
    """
    ctx = mp.get_context("spawn")
    request_queue: "mp.Queue" = ctx.Queue(maxsize=int(request_maxsize))
    result_queue: "mp.Queue" = ctx.Queue(maxsize=int(result_maxsize))
    stop_event = ctx.Event()
    process = ctx.Process(
        target=_residual_observer_worker_main,
        args=(request_queue, result_queue, stop_event),
        kwargs={
            "gap_cycles": int(gap_cycles),
            "lowpass_alpha": float(lowpass_alpha),
            "qd0": np.asarray(qd0, dtype=np.float64).copy(),
        },
        daemon=True,
        name="residual-observer-worker",
    )
    # Temporarily pin BLAS/OMP thread counts to 1 in the environment the
    # spawned child inherits at exec-time (see _WORKER_THREAD_ENV_VARS above)
    # -- the parent's own already-initialized numpy/BLAS thread pool is
    # unaffected by this (those libraries only read the env var once, at
    # their own first initialization, which already happened earlier in this
    # process), and the child gets its own independent environ snapshot at
    # exec time, so restoring the parent's values immediately after start()
    # cannot retroactively change what the child already inherited.
    _prev_env = {name: os.environ.get(name) for name in _WORKER_THREAD_ENV_VARS}
    try:
        for name in _WORKER_THREAD_ENV_VARS:
            os.environ[name] = "1"
        process.start()
    finally:
        for name, value in _prev_env.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
    return ResidualObserverWorkerHandle(
        process=process,
        request_queue=request_queue,
        result_queue=result_queue,
        stop_event=stop_event,
    )
