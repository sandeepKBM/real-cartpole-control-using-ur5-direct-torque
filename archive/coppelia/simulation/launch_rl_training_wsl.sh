#!/usr/bin/env bash
# Launch PPO training with managed CoppeliaSim (single or parallel envs).
#
# Usage:
#   bash simulation/launch_rl_training_wsl.sh
#   N_ENVS=2 TIMESTEPS=50000 bash simulation/launch_rl_training_wsl.sh
#   bash simulation/launch_rl_training_wsl.sh --resume outputs/rl_models/ppo_y_transport
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

source "${SCRIPT_DIR}/env_wsl_local.sh"

PORT="${PORT:-23000}"
TIMESTEPS="${TIMESTEPS:-50000}"
N_ENVS="${N_ENVS:-2}"
CONFIG="${CONFIG:-${ROOT}/rl/config.yaml}"
RESUME_ARGS=""

for arg in "$@"; do
  case "${arg}" in
    --resume=*) RESUME_ARGS="--resume ${arg#*=}" ;;
    --resume)   shift; RESUME_ARGS="--resume $1" ;;
  esac
done

rm -f "${COPPELIA_ROOT}/addOns/ur5_video_smoke_addon.lua"
export REAL_CARTPOLE_ENABLE_VIDEO_SMOKE=0
unset QT_QPA_PLATFORM 2>/dev/null || true

echo "=== RL PPO training (managed CoppeliaSim) ==="
echo "  coppelia_root=${COPPELIA_ROOT}"
echo "  base_port=${PORT}"
echo "  n_envs=${N_ENVS}"
echo "  timesteps=${TIMESTEPS}"
echo "  config=${CONFIG}"

cd "${ROOT}"
PYTHONPATH="${ROOT}:${COPPELIA_PYDEPS}:${PYTHONPATH:-}" \
PYTHONUNBUFFERED=1 \
  python3 -u "${ROOT}/rl/train_ppo.py" \
    --config "${CONFIG}" \
    --timesteps "${TIMESTEPS}" \
    --port "${PORT}" \
    --n-envs "${N_ENVS}" \
    --manage-sim \
    --coppelia-root "${COPPELIA_ROOT}" \
    ${RESUME_ARGS}
