"""Start, stop, and restart a CoppeliaSim ZMQ server subprocess."""

from __future__ import annotations

import os
import socket
import subprocess
import time
from pathlib import Path


class CoppeliaSimManager:
    """Own one CoppeliaSim process bound to a single RPC port."""

    def __init__(
        self,
        coppelia_root: str | Path,
        port: int,
        log_path: str | Path | None = None,
    ) -> None:
        self.coppelia_root = Path(coppelia_root)
        self.port = int(port)
        self.cnt_port = self.port + 1
        self.log_path = Path(log_path) if log_path else None
        self._proc: subprocess.Popen | None = None

    def _port_listening(self) -> bool:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(0.5)
            return sock.connect_ex(("127.0.0.1", self.port)) == 0

    def _kill_stale(self) -> None:
        os.system(
            f"pkill -f 'zmqRemoteApi.rpcPort={self.port}' >/dev/null 2>&1"
        )

    def start(self, timeout_s: float = 90.0) -> None:
        self._kill_stale()
        time.sleep(1.0)

        if self.log_path is not None:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            log_handle = open(self.log_path, "w", encoding="utf-8")
        else:
            log_handle = open(os.devnull, "w")

        env = os.environ.copy()
        env.pop("QT_QPA_PLATFORM", None)
        env["REAL_CARTPOLE_ENABLE_VIDEO_SMOKE"] = "0"

        launcher = self.coppelia_root / "coppeliaSim.sh"
        if not launcher.exists():
            raise FileNotFoundError(f"Missing CoppeliaSim launcher: {launcher}")

        self._proc = subprocess.Popen(
            [
                str(launcher),
                f"-GzmqRemoteApi.rpcPort={self.port}",
                f"-GzmqRemoteApi.cntPort={self.cnt_port}",
            ],
            cwd=str(self.coppelia_root),
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            env=env,
        )

        deadline = time.time() + timeout_s
        while time.time() < deadline:
            if self._proc.poll() is not None:
                raise RuntimeError(
                    f"CoppeliaSim exited before port {self.port} opened "
                    f"(code={self._proc.returncode})"
                )
            if self._port_listening():
                time.sleep(3.0)
                if self._port_listening() and self._proc.poll() is None:
                    print(
                        f"[sim-mgr] CoppeliaSim ready on port {self.port} "
                        f"(pid={self._proc.pid})",
                        flush=True,
                    )
                    return
            time.sleep(1.0)

        self.stop()
        raise RuntimeError(f"CoppeliaSim port {self.port} not ready in {timeout_s}s")

    def stop(self) -> None:
        if self._proc is not None:
            try:
                self._proc.terminate()
                self._proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._proc.kill()
                self._proc.wait(timeout=5)
            except Exception:
                pass
            self._proc = None
        self._kill_stale()

    def restart(self) -> None:
        print(f"[sim-mgr] restarting CoppeliaSim on port {self.port}", flush=True)
        self.stop()
        time.sleep(2.0)
        self.start()
