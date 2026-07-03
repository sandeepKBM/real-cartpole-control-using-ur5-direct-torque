#!/usr/bin/env python3
"""HPC / ZMQ attach-only diagnostic for the live external controller lane.

This script does **not** start simulation, enable stepping, advance steps, or
command torques. It only checks whether the remote API can be attached cleanly
from Python in the current environment.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import time
import traceback
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_COPPELIA_ROOT = (
    REPO_ROOT
    / "third_party"
    / "coppelia_runtime"
    / "CoppeliaSim_Edu_V4_10_0_rev0_Ubuntu24_04"
)
DEFAULT_COPPELIA_PYDEPS = REPO_ROOT / "third_party" / "coppelia_pydeps"
DEFAULT_SUMMARY = (
    REPO_ROOT / "outputs" / "control_runs" / "hpc_zmq_attach" / "hpc_zmq_attach_summary.json"
)

_bootstrap_root = Path(os.environ.get("COPPELIA_ROOT", str(DEFAULT_COPPELIA_ROOT)))
_bootstrap_pydeps = Path(os.environ.get("COPPELIA_PYDEPS", str(DEFAULT_COPPELIA_PYDEPS)))
for candidate in (
    _bootstrap_root / "programming" / "zmqRemoteApi" / "clients" / "python" / "src",
    _bootstrap_pydeps,
):
    if candidate.exists():
        sys.path.insert(0, str(candidate))


def _selected_env() -> dict[str, Any]:
    keys = (
        "HOSTNAME",
        "SLURM_JOB_ID",
        "SLURM_JOB_NODELIST",
        "SLURM_NODEID",
        "SLURMD_NODENAME",
        "DISPLAY",
        "QT_QPA_PLATFORM",
        "LD_LIBRARY_PATH",
        "COPPELIASIM_ROOT",
        "COPPELIA_ROOT",
        "PYTHONPATH",
    )
    return {key: os.environ.get(key) for key in keys}


def _tail_text(path: Path, max_lines: int = 80) -> list[str]:
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except Exception:
        return []
    if max_lines <= 0 or len(lines) <= max_lines:
        return lines
    return lines[-max_lines:]


def _tcp_port_open(host: str, port: int, timeout_s: float = 0.25) -> bool:
    try:
        with socket.create_connection((host, port), timeout=max(float(timeout_s), 0.05)):
            return True
    except OSError:
        return False


def _hostname_reference() -> str:
    candidates = [
        os.environ.get("HOSTNAME", ""),
        os.environ.get("SLURMD_NODENAME", ""),
        socket.getfqdn(),
        socket.gethostname(),
    ]
    for value in candidates:
        value = (value or "").strip()
        if value:
            return value
    return ""


def _short_hostname(value: str) -> str:
    value = (value or "").strip().lower()
    if not value:
        return ""
    return value.split(".", 1)[0]


def _record_summary(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _close_client(client: Any) -> None:
    try:
        import zmq

        client.socket.setsockopt(zmq.LINGER, 0)
        client.socket.close()
        client.context.term()
    except Exception:
        pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--rpc-port", type=int, default=23000)
    parser.add_argument("--cnt-port", type=int, default=None)
    parser.add_argument("--timeout-s", type=float, default=60.0)
    parser.add_argument("--poll-interval-s", type=float, default=0.5)
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=DEFAULT_SUMMARY,
    )
    parser.add_argument(
        "--coppeliasim-log",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--require-same-hostname",
        action="store_true",
        help="Fail if the local Python hostname does not match the scheduler hostname reference.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cnt_port = int(args.cnt_port) if args.cnt_port is not None else int(args.rpc_port) + 1
    start = time.monotonic()
    python_hostname = socket.gethostname()
    hostname_reference = _hostname_reference()
    summary: dict[str, Any] = {
        "success": False,
        "probe_family": "hpc_zmq_attach_only",
        "host": str(args.host),
        "rpc_port": int(args.rpc_port),
        "cnt_port": int(cnt_port),
        "python_hostname": python_hostname,
        "python_pid": os.getpid(),
        "cwd": str(Path.cwd()),
        "python_executable": sys.executable,
        "sys_path_head": sys.path[:8],
        "env": _selected_env(),
        "rpc_port_open": False,
        "cnt_port_open": None,
        "client_import_ok": False,
        "remoteapi_client_created": False,
        "require_sim_ok": False,
        "get_simulation_state_ok": False,
        "simulation_state": None,
        "simulation_time_s": None,
        "elapsed_s": 0.0,
        "error": None,
    }

    if args.require_same_hostname:
        if not hostname_reference:
            summary["error"] = (
                "--require-same-hostname was set, but no hostname reference was available "
                "from HOSTNAME, SLURMD_NODENAME, or the local node identity."
            )
            summary["elapsed_s"] = round(time.monotonic() - start, 6)
            _record_summary(args.summary_json, summary)
            print(json.dumps(summary, indent=2))
            return 1
        if _short_hostname(python_hostname) != _short_hostname(hostname_reference):
            summary["error"] = (
                "hostname mismatch: "
                f"python_hostname={python_hostname!r}, reference={hostname_reference!r}"
            )
            summary["elapsed_s"] = round(time.monotonic() - start, 6)
            _record_summary(args.summary_json, summary)
            print(json.dumps(summary, indent=2))
            return 1

    try:
        from coppeliasim_zmqremoteapi_client import RemoteAPIClient
        import zmq

        summary["client_import_ok"] = True

        deadline = start + max(float(args.timeout_s), 0.0)
        poll_interval = max(float(args.poll_interval_s), 0.05)
        last_exc: Exception | None = None
        attempt = 0
        while time.monotonic() < deadline:
            attempt += 1
            summary["rpc_port_open"] = _tcp_port_open(args.host, int(args.rpc_port))
            try:
                summary["cnt_port_open"] = _tcp_port_open(args.host, cnt_port)
            except Exception:
                summary["cnt_port_open"] = None

            print(
                f"[hpc-zmq-attach] attempt {attempt}: "
                f"rpc_open={summary['rpc_port_open']} cnt_open={summary['cnt_port_open']}",
                flush=True,
            )

            client = None
            try:
                client = RemoteAPIClient(host=args.host, port=int(args.rpc_port))
                summary["remoteapi_client_created"] = True
                first_timeout_ms = int(os.environ.get("REAL_CARTPOLE_RPC_FIRST_RCVTIMEO_MS", "20000") or 20000)
                client.socket.setsockopt(zmq.RCVTIMEO, first_timeout_ms)
                sim = client.require("sim")
                summary["require_sim_ok"] = True
                state = sim.getSimulationState()
                summary["get_simulation_state_ok"] = True
                summary["simulation_state"] = state
                summary["simulation_time_s"] = sim.getSimulationTime()
                summary["success"] = True
                summary["error"] = None
                break
            except Exception as exc:
                last_exc = exc
                print(
                    f"[hpc-zmq-attach] attach failure on attempt {attempt}: "
                    f"{type(exc).__name__}: {exc}",
                    flush=True,
                )
                if client is not None:
                    _close_client(client)
                remaining = deadline - time.monotonic()
                if remaining > 0.0:
                    time.sleep(min(poll_interval, remaining))

        if not summary["success"]:
            if last_exc is not None:
                summary["error"] = f"{type(last_exc).__name__}: {last_exc}"
            else:
                summary["error"] = "timeout waiting for usable remote API attach"
    except Exception as exc:
        summary["error"] = f"{type(exc).__name__}: {exc}"
        summary["client_import_ok"] = summary["client_import_ok"] or False
        summary["remoteapi_client_created"] = summary["remoteapi_client_created"] or False
        traceback.print_exc()
    finally:
        summary["elapsed_s"] = round(time.monotonic() - start, 6)
        try:
            _record_summary(args.summary_json, summary)
        except Exception as exc:
            print(f"[hpc-zmq-attach] failed to write summary: {exc}", file=sys.stderr, flush=True)

    print("[hpc-zmq-attach] diagnosis:", flush=True)
    print(json.dumps(summary, indent=2), flush=True)
    if args.coppeliasim_log is not None and args.coppeliasim_log.exists() and not summary["success"]:
        print(f"[hpc-zmq-attach] tail of CoppeliaSim log: {args.coppeliasim_log}", flush=True)
        for line in _tail_text(args.coppeliasim_log, max_lines=60):
            print(f"[coppelia-log] {line}", flush=True)
    return 0 if summary["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
