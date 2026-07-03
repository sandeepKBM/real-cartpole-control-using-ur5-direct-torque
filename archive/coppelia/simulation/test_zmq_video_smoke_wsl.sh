#!/usr/bin/env bash
set -euo pipefail
ROOT="/mnt/c/Users/sandr/Downloads/real-cartpole-control-using-ur5-direct-torque"
source "${ROOT}/simulation/env_wsl_local.sh"
PORT="${PORT:-23097}"
export DISPLAY="${DISPLAY:-:0}"
unset QT_QPA_PLATFORM

pkill -f "zmqRemoteApi.rpcPort=${PORT}" 2>/dev/null || true
pkill -f coppeliaSim 2>/dev/null || true
sleep 1

cd "${COPPELIA_ROOT}"
./coppeliaSim.sh -GzmqRemoteApi.rpcPort="${PORT}" -GzmqRemoteApi.cntPort="$((PORT+1))" \
  >/tmp/coppelia_vsmoke.log 2>&1 &
PID=$!
trap 'kill "${PID}" 2>/dev/null || true' EXIT

for _ in $(seq 1 60); do
  ss -ltn "sport = :${PORT}" 2>/dev/null | grep -q LISTEN && break
  sleep 1
done

cd "${ROOT}"
python3 simulation/run_coppeliasim_video_smoke.py \
  --coppelia-root "${COPPELIA_ROOT}" \
  --port "${PORT}" \
  --duration 1 --fps 10 \
  --video-name zmq_smoke_test.mp4 \
  --summary-name zmq_smoke_test_summary.json

python3 - <<'PY'
import json
import subprocess
from pathlib import Path
s = json.loads(Path("outputs/control_runs/zmq_smoke_test_summary.json").read_text())
v = Path("demonstration_videos/ur5e_coppeliasim/zmq_smoke_test.mp4")
print("frames_written=", s.get("frames_written"))
if v.exists():
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height", "-of", "csv=p=0", str(v)],
        capture_output=True, text=True,
    )
    print("video=", v, "probe=", (r.stdout or r.stderr).strip())
PY
