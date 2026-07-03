#!/usr/bin/env bash
#SBATCH --job-name=ur5_controller_probe
#SBATCH --partition=unlimited
#SBATCH --nodelist=rlab3
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --output=/common/users/ss5772/real_Cartpole/outputs/control_runs/slurm_controller_probe_%j.out
#SBATCH --error=/common/users/ss5772/real_Cartpole/outputs/control_runs/slurm_controller_probe_%j.err

set -euo pipefail

ROOT="/common/users/ss5772/real_Cartpole"

echo "=== Job ${SLURM_JOB_ID} on $(hostname) ==="
echo "GPU:"
nvidia-smi -L 2>&1 || echo "(no nvidia-smi)"
echo "---"

# Kill any orphaned CoppeliaSim from prior runs.
pkill -u "$(whoami)" -f coppeliaSim 2>/dev/null || true
sleep 1

# Check port 23000.
if ss -tlnp 2>/dev/null | grep -q ':23000 '; then
  echo "WARNING: port 23000 already in use"
  ss -tlnp 2>/dev/null | grep ':23000 '
fi

echo "--- Starting controller probe ---"
cd "${ROOT}"
bash simulation/launch_coppeliasim_x_axis_headless.sh --probe-only --no-video

echo "--- Probe done ---"
