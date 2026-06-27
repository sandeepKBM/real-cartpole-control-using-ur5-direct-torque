#!/usr/bin/env bash
# External ZMQ direct-torque Y transport with cart-pole MPC outer loop.
# Movement primitives locked in config/coppeliasim_movement_primitives.yaml.
set -euo pipefail
ROOT="/mnt/c/Users/sandr/Downloads/real-cartpole-control-using-ur5-direct-torque"
# shellcheck disable=SC1091
source "${ROOT}/simulation/env_wsl_local.sh"

COPPELIA="${COPPELIA_ROOT}"
PORT="${PORT:-23000}"
CONFIG="${CONFIG:-${ROOT}/ros2_ws/src/ur5_x_axis_controller_ros/config/controller_coppelia_y_transport_torque.yaml}"
RUNNER="${ROOT}/simulation/run_coppeliasim_x_axis_headless.py"
SIM_LOG="${ROOT}/outputs/control_runs/coppelia_torque_y_mpc_transport_sim.log"
RUNNER_LOG="${ROOT}/outputs/control_runs/coppelia_torque_y_mpc_transport_runner.log"
VIDEO_OUT="${ROOT}/demonstration_videos/ur5e_coppeliasim/coppeliasim_ur5_wsl_y_mpc_transport.mp4"
SUMMARY_OUT="${ROOT}/outputs/control_runs/coppeliasim_ur5_wsl_y_mpc_transport_summary.json"

# Locked primitives (override via env for tuning iterations).
TRANSPORT_AXIS="${TRANSPORT_AXIS:-y}"
Q_START_RAD="${Q_START_RAD:-0,0.0443244063,-1.67570517,5.09435844,-6.28345754,5.96178335}"
SHOULDER_PAN_LOCKED_RAD="${SHOULDER_PAN_LOCKED_RAD:-0}"
TARGET_DY="${TARGET_DY:-0.04}"
SETTLE_DURATION="${SETTLE_DURATION:-0}"
MOTION_HOLD_WARMUP="${MOTION_HOLD_WARMUP:-0.5}"
DURATION="${DURATION:-20}"
A_AXIS_MAX="${A_AXIS_MAX:-0.008}"
V_AXIS_MAX="${V_AXIS_MAX:-0.008}"
GRAVITY_COMP_SOURCE="${GRAVITY_COMP_SOURCE:-mujoco}"
GRAVITY_SCALE="${GRAVITY_SCALE:-1.0}"
IK_JOINT_KP="${IK_JOINT_KP:-140}"
IK_JOINT_KD="${IK_JOINT_KD:-25}"
IK_TORQUE_HEADROOM="${IK_TORQUE_HEADROOM:-0.90}"
CART_Z_KP="${CART_Z_KP:-200}"
CART_Z_KD="${CART_Z_KD:-40}"
CART_Z_KI="${CART_Z_KI:-50}"
MPC_HORIZON="${MPC_HORIZON:-20}"
MPC_Q_THETA="${MPC_Q_THETA:-180}"
SPAWN_PENDULUM="${SPAWN_PENDULUM:-0}"
ACCEL_PROFILE="${ACCEL_PROFILE:-point_to_point}"
# Green-on-green (world Y) needs Jacobian-row transport; qp_torque Cartesian PD still couples Z.
INNER_TORQUE_POLICY="${INNER_TORQUE_POLICY:-ik_joint_pd}"
NO_VIDEO="${NO_VIDEO:-0}"

mkdir -p "${ROOT}/outputs/control_runs" "${ROOT}/demonstration_videos/ur5e_coppeliasim"

if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "ERROR: ffmpeg required" >&2
  exit 1
fi

pkill -f "zmqRemoteApi.rpcPort=${PORT}" 2>/dev/null || true
pkill -f coppeliaSim 2>/dev/null || true
sleep 2

export COPPELIA_ROOT COPPELIA_PYDEPS REAL_CARTPOLE_ROOT
export REAL_CARTPOLE_ENABLE_VIDEO_SMOKE=0
export Q_START_RAD
export LD_LIBRARY_PATH="${COPPELIA}:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${COPPELIA}/programming/zmqRemoteApi/clients/python/src:${COPPELIA_PYDEPS}:${PYTHONPATH:-}"
export DISPLAY="${DISPLAY:-:0}"
export WAYLAND_DISPLAY="${WAYLAND_DISPLAY:-wayland-0}"
if [[ -z "${XDG_RUNTIME_DIR:-}" && -d "/run/user/$(id -u)" ]]; then
  export XDG_RUNTIME_DIR="/run/user/$(id -u)"
fi

# xcb + WSLg renders vision sensors; offscreen Qt yields black ZMQ MP4 buffers.
_PLATFORM="${COPPELIA_QT_PLATFORM:-xcb}"
if [[ "${_PLATFORM}" == "xcb" ]]; then
  unset QT_QPA_PLATFORM
else
  export QT_QPA_PLATFORM="${_PLATFORM}"
fi

echo "=== CoppeliaSim MPC + direct-torque Y transport (locked primitives) ==="
echo "  display=DISPLAY=${DISPLAY} QT=${QT_QPA_PLATFORM:-xcb(default-unset)}"
echo "  outer_loop=${ACCEL_PROFILE}  inner_torque=${INNER_TORQUE_POLICY}"
echo "  transport_axis=${TRANSPORT_AXIS} (green-on-green / world Y)"
echo "  shoulder_pan_locked_rad=${SHOULDER_PAN_LOCKED_RAD}"
echo "  q_start_rad=${Q_START_RAD}"
echo "  target_displacement_m=${TARGET_DY}  mpc_horizon=${MPC_HORIZON}"
echo "  motion_hold_warmup_s=${MOTION_HOLD_WARMUP}  gravity_source=${GRAVITY_COMP_SOURCE}  gravity_scale=${GRAVITY_SCALE}"
echo "  config=${CONFIG}"
echo "  video=${VIDEO_OUT}"

cd "${COPPELIA}"
./coppeliaSim.sh \
  -GzmqRemoteApi.rpcPort="${PORT}" -GzmqRemoteApi.cntPort="$((PORT + 1))" \
  >"${SIM_LOG}" 2>&1 &
COPPELIA_PID=$!

cleanup() {
  kill "${COPPELIA_PID}" 2>/dev/null || true
  wait "${COPPELIA_PID}" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

echo "Waiting for ZMQ port ${PORT}..."
for _ in $(seq 1 90); do
  if ss -ltn "sport = :${PORT}" 2>/dev/null | grep -q LISTEN; then
    sleep 2
    if ss -ltn "sport = :${PORT}" 2>/dev/null | grep -q LISTEN && pgrep -f coppeliaSim >/dev/null 2>&1; then
      echo "RPC ready."
      break
    fi
  fi
  if ! kill -0 "${COPPELIA_PID}" 2>/dev/null; then
    echo "CoppeliaSim exited early:" >&2
    tail -40 "${SIM_LOG}" >&2 || true
    exit 1
  fi
  sleep 1
done

if ! ss -ltn "sport = :${PORT}" 2>/dev/null | grep -q LISTEN; then
  echo "Timed out waiting for port ${PORT}" >&2
  exit 1
fi

PENDULUM_ARGS=()
if [[ "${SPAWN_PENDULUM}" == "1" ]]; then
  PENDULUM_ARGS+=(--spawn-coppelia-pendulum --pendulum-pole-length-m 0.4)
else
  PENDULUM_ARGS+=(--no-spawn-coppelia-pendulum)
fi

VIDEO_ARGS=(--video-camera smoke --video-name "$(basename "${VIDEO_OUT}")" --fps 20)
if [[ "${NO_VIDEO}" == "1" ]]; then
  VIDEO_ARGS=(--no-video)
fi

cd "${ROOT}"
set +e
"${PYTHON_BIN}" "${RUNNER}" \
  --coppelia-root "${COPPELIA}" \
  --port "${PORT}" \
  --config "${CONFIG}" \
  --accel-x-transport \
  --accel-profile "${ACCEL_PROFILE}" \
  --accel-torque-policy "${INNER_TORQUE_POLICY}" \
  --transport-axis "${TRANSPORT_AXIS}" \
  --hold-transport-start-pose \
  --task-frame-mode mujoco_attachment_dummy \
  --target-dx "${TARGET_DY}" \
  --duration "${DURATION}" \
  --settle-duration "${SETTLE_DURATION}" \
  --motion-hold-warmup "${MOTION_HOLD_WARMUP}" \
  --gravity-compensation-source "${GRAVITY_COMP_SOURCE}" \
  --gravity-scale "${GRAVITY_SCALE}" \
  --a-x-max "${A_AXIS_MAX}" \
  --v-x-max "${V_AXIS_MAX}" \
  --ik-joint-kp "${IK_JOINT_KP}" \
  --ik-joint-kd "${IK_JOINT_KD}" \
  --ik-torque-headroom "${IK_TORQUE_HEADROOM}" \
  --cartesian-z-kp "${CART_Z_KP}" \
  --cartesian-z-kd "${CART_Z_KD}" \
  --cartesian-z-ki "${CART_Z_KI}" \
  --mpc-horizon "${MPC_HORIZON}" \
  --mpc-q-theta "${MPC_Q_THETA}" \
  "${VIDEO_ARGS[@]}" \
  --summary-name "$(basename "${SUMMARY_OUT}")" \
  "${PENDULUM_ARGS[@]}" \
  2>&1 | tee "${RUNNER_LOG}"
RUN_EXIT=${PIPESTATUS[0]}
set -e

echo ""
echo "=== MPC torque transport summary ==="
if [[ -f "${SUMMARY_OUT}" ]]; then
  python3 - <<PY
import json
s = json.load(open("${SUMMARY_OUT}"))
keys = [
    "success", "uses_direct_torque_control", "outer_transport_controller",
    "accel_torque_policy", "accel_profile", "transport_axis",
    "mpc_horizon", "mpc_target_x_m", "coppelia_pendulum_spawned",
    "failure_reasons",
    "transport_axis_net_displacement_m", "axis_net_displacement_m",
    "max_abs_fixed_axis_1_drift_m", "max_abs_fixed_axis_2_drift_m",
    "initial_ee_world_m", "final_ee_world_m",
    "shoulder_pan_locked_rad", "q_start_rad", "q_final_rad",
    "first_frame_std_rgb",
]
for k in keys:
    if k in s:
        print(f"{k}=", s[k])
q0 = s.get("q_start_rad") or [None]*6
qf = s.get("q_final_rad") or [None]*6
if q0[0] is not None and qf[0] is not None:
    print("shoulder_pan_drift_rad=", float(qf[0]) - float(q0[0]))
PY
else
  echo "Summary missing: ${SUMMARY_OUT}"
fi
if [[ -f "${VIDEO_OUT}" ]]; then
  ls -lh "${VIDEO_OUT}"
else
  echo "Video missing: ${VIDEO_OUT}"
fi
echo "Runner exit code: ${RUN_EXIT}"
exit "${RUN_EXIT}"
