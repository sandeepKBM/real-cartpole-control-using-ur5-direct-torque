#!/usr/bin/env bash
# Run CoppeliaSim controller variant matrix on WSL (ZMQ external + MPC).
set -euo pipefail
ROOT="/mnt/c/Users/sandr/Downloads/real-cartpole-control-using-ur5-direct-torque"
# shellcheck disable=SC1091
source "${ROOT}/simulation/env_wsl_local.sh"

REPORT="${ROOT}/outputs/control_runs/wsl_controller_matrix_report.json"
RUNNER="${ROOT}/simulation/run_coppeliasim_x_axis_headless.py"
MPC_CONFIG="${ROOT}/ros2_ws/src/ur5_x_axis_controller_ros/config/controller_coppelia_mpc.yaml"
DEFAULT_CONFIG="${ROOT}/ros2_ws/src/ur5_x_axis_controller_ros/config/controller.yaml"
PORT=23000
PLATFORM="${COPPELIA_QT_PLATFORM:-xcb}"

mkdir -p "${ROOT}/outputs/control_runs"

python3 - <<'PY' > "${REPORT}"
import json
print(json.dumps({"tests": [], "started": True}))
PY

log_result() {
  local name="$1"
  local exit_code="$2"
  local summary="$3"
  python3 - <<PY
import json
from pathlib import Path
p = Path("${REPORT}")
data = json.loads(p.read_text())
entry = {
    "name": """${name}""",
    "exit_code": int(${exit_code}),
    "summary_path": """${summary}""",
}
sp = Path("""${summary}""")
if sp.is_file():
    try:
        s = json.loads(sp.read_text())
        entry["success"] = s.get("success")
        entry["probe_passed"] = s.get("probe_passed")
        entry["failure_reasons"] = s.get("failure_reasons")
        entry["safety_stop_reason"] = s.get("safety_stop_reason")
        entry["x_net_displacement_m"] = s.get("x_net_displacement_m")
        entry["outer_transport_controller"] = s.get("outer_transport_controller")
        entry["accel_profile"] = s.get("accel_profile")
        entry["mpc_pole_observer_has_pendulum"] = s.get("mpc_pole_observer_has_pendulum")
        entry["coppelia_pendulum_spawned"] = s.get("coppelia_pendulum_spawned")
        entry["first_frame_std_rgb"] = s.get("first_frame_std_rgb")
    except Exception as e:
        entry["summary_parse_error"] = str(e)
p.write_text(json.dumps(data, indent=2))
data["tests"].append(entry)
p.write_text(json.dumps(data, indent=2))
PY
}

start_coppelia() {
  pkill -f "zmqRemoteApi.rpcPort=${PORT}" 2>/dev/null || true
  pkill -f coppeliaSim 2>/dev/null || true
  sleep 2
  export LD_LIBRARY_PATH="${COPPELIA_ROOT}:${LD_LIBRARY_PATH:-}"
  export PYTHONPATH="${COPPELIA_ROOT}/programming/zmqRemoteApi/clients/python/src:${COPPELIA_PYDEPS}:${PYTHONPATH:-}"
  export DISPLAY="${DISPLAY:-:0}"
  cd "${COPPELIA_ROOT}"
  if [[ "${PLATFORM}" == "xcb" ]]; then
    unset QT_QPA_PLATFORM
  else
    export QT_QPA_PLATFORM="${PLATFORM}"
  fi
  ./coppeliaSim.sh -GzmqRemoteApi.rpcPort="${PORT}" -GzmqRemoteApi.cntPort="$((PORT+1))" \
    >"${ROOT}/outputs/control_runs/matrix_coppelia.log" 2>&1 &
  COP_PID=$!
  for _ in $(seq 1 90); do
    if ss -ltn "sport = :${PORT}" 2>/dev/null | grep -q LISTEN && pgrep -f coppeliaSim >/dev/null; then
      sleep 2
      return 0
    fi
    if ! kill -0 "${COP_PID}" 2>/dev/null; then
      tail -20 "${ROOT}/outputs/control_runs/matrix_coppelia.log" >&2
      return 1
    fi
    sleep 1
  done
  return 1
}

stop_coppelia() {
  pkill -f "zmqRemoteApi.rpcPort=${PORT}" 2>/dev/null || true
  pkill -f coppeliaSim 2>/dev/null || true
  sleep 1
}

run_case() {
  local name="$1"
  shift
  local summary="${ROOT}/outputs/control_runs/matrix_${name// /_}.json"
  echo ""
  echo "========== ${name} =========="
  if ! start_coppelia; then
    echo "FAIL: CoppeliaSim did not start"
    log_result "${name}" 99 "${summary}"
    return
  fi
  cd "${ROOT}"
  set +e
  "${PYTHON_BIN}" "${RUNNER}" \
    --coppelia-root "${COPPELIA_ROOT}" \
    --port "${PORT}" \
    --summary-name "$(basename "${summary}")" \
    --no-video \
    "$@" 2>&1 | tee "${ROOT}/outputs/control_runs/matrix_${name// /_}.log"
  local ec=${PIPESTATUS[0]}
  set -e
  stop_coppelia
  log_result "${name}" "${ec}" "${summary}"
  echo "exit=${ec} summary=${summary}"
}

echo "=== Unit tests: MPC core ==="
cd "${ROOT}"
python3 -m pytest controller_core/tests/test_mpc_controller.py -q 2>&1 | tee "${ROOT}/outputs/control_runs/matrix_mpc_unit_test.log" || true

run_case "zmq_probe" \
  --probe-only \
  --task-frame-mode mujoco_attachment_dummy

run_case "zmq_ik_joint_pd_small" \
  --config "${DEFAULT_CONFIG}" \
  --accel-x-transport \
  --accel-torque-policy ik_joint_pd \
  --accel-profile point_to_point \
  --target-dx 0.003 \
  --duration 4 \
  --settle-duration 1 \
  --task-frame-mode mujoco_attachment_dummy

run_case "zmq_mpc_spawn_pendulum" \
  --config "${MPC_CONFIG}" \
  --accel-x-transport \
  --accel-profile mpc \
  --accel-torque-policy ik_joint_pd \
  --transport-axis x \
  --target-dx 0.01 \
  --duration 6 \
  --settle-duration 1.5 \
  --a-x-max 0.06 \
  --v-x-max 0.03 \
  --mpc-horizon 20 \
  --spawn-coppelia-pendulum \
  --pendulum-pole-length-m 0.4 \
  --task-frame-mode mujoco_attachment_dummy

run_case "zmq_mpc_tiny_dx" \
  --config "${MPC_CONFIG}" \
  --accel-x-transport \
  --accel-profile mpc \
  --accel-torque-policy ik_joint_pd \
  --transport-axis x \
  --target-dx 0.005 \
  --duration 5 \
  --settle-duration 1 \
  --a-x-max 0.04 \
  --v-x-max 0.02 \
  --mpc-horizon 15 \
  --mpc-q-theta 50 \
  --spawn-coppelia-pendulum \
  --task-frame-mode mujoco_attachment_dummy

run_case "zmq_cartesian_impedance_small" \
  --config "${DEFAULT_CONFIG}" \
  --accel-x-transport \
  --accel-torque-policy cartesian_impedance \
  --accel-profile point_to_point \
  --target-dx 0.003 \
  --duration 4 \
  --settle-duration 1 \
  --task-frame-mode mujoco_attachment_dummy

echo ""
echo "=== Matrix complete ==="
python3 - <<PY
import json
from pathlib import Path
p = Path("${REPORT}")
data = json.loads(p.read_text())
print(json.dumps(data, indent=2))
print()
print("SUMMARY TABLE:")
for t in data.get("tests", []):
    ok = t.get("success", t.get("probe_passed"))
    print(f"  {t['name']:30s}  success={ok}  x_net={t.get('x_net_displacement_m')}  reasons={t.get('failure_reasons')}")
PY
