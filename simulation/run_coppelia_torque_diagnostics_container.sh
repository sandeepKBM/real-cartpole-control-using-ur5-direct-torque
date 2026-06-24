#!/usr/bin/env bash
set -euo pipefail

ROOT="/common/users/ss5772/real_Cartpole"
SIF="${REAL_CARTPOLE_SIF:-/common/users/ss5772/containers/aha_u2404.sif}"
XVFB_BIN="${XVFB_RUN_BIN:-/common/home/ss5772/.tmp/tmp.pyUiwLzWVT/root/usr/bin/xvfb-run}"
PORT="${PORT:-23260}"
DURATION="${DURATION:-2.0}"
INNER_SCRIPT="${ROOT}/outputs/control_runs/coppelia_torque_diagnostics/_container_smoke_inner.sh"

mkdir -p "$(dirname "${INNER_SCRIPT}")"
cat >"${INNER_SCRIPT}" <<EOF
#!/usr/bin/env bash
set -euo pipefail
export PATH=$(dirname "${XVFB_BIN}"):/common/users/ss5772/miniforge3/bin:/usr/bin:/bin
export PYTHON_BIN=/common/users/ss5772/miniforge3/bin/python3
export XVFB_RUN_BIN=${XVFB_BIN}
export LD_LIBRARY_PATH=/common/home/ss5772/.tmp/container_bind_libs:${ROOT}/third_party/coppelia_runtime/CoppeliaSim_Edu_V4_10_0_rev0_Ubuntu24_04
export PORT=${PORT}
export XDG_RUNTIME_DIR=/common/home/ss5772/.tmp/xdg-runtime-\$\$
mkdir -p "\${XDG_RUNTIME_DIR}"
cd ${ROOT}
if [[ "\$#" -gt 0 ]]; then
  exec python simulation/run_coppelia_torque_diagnostics_smoke.py \\
    --host 127.0.0.1 \\
    --port ${PORT} \\
    --duration ${DURATION} \\
    --use-launcher \\
    --tests "\$@"
else
  exec python simulation/run_coppelia_torque_diagnostics_smoke.py \\
    --host 127.0.0.1 \\
    --port ${PORT} \\
    --duration ${DURATION} \\
    --use-launcher
fi
EOF
chmod +x "${INNER_SCRIPT}"

if [[ ! -f "${SIF}" ]]; then
  echo "Missing Singularity image: ${SIF}" >&2
  exit 1
fi

exec singularity exec \
  --bind "${ROOT}:${ROOT}" \
  --bind /common/home/ss5772/.tmp:/common/home/ss5772/.tmp \
  --bind /common/users/ss5772/miniforge3:/common/users/ss5772/miniforge3 \
  --bind /usr/bin/xkbcomp:/usr/bin/xkbcomp \
  --bind /usr/share/X11/xkb:/usr/lib/X11/xkb \
  --bind /usr/share/X11/xkb:/usr/share/X11/xkb \
  "${SIF}" \
  bash "${INNER_SCRIPT}" "$@"
