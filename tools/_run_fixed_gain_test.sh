#!/usr/bin/env bash
set -euo pipefail
source ~/ur5e_repo/.venv/bin/activate
REPO="/mnt/c/Users/sandr/Downloads/real-cartpole-control-using-ur5-direct-torque"
cd "$REPO"

python -c "import mujoco; import pinocchio; print('mujoco', mujoco.__version__)"

echo "=== Test 1: active-origin pose, dx=0.10m, move 1s + hold 2s ==="
python tools/ur5e_mujoco_torque_experiments.py \
  --mode controller-rollout \
  --controller-kind impedance \
  --config config/ur5e_mujoco_torque_osc_tuned.yaml \
  --trajectory-profile min_jerk_move_hold \
  --move-duration 1.0 \
  --duration 3.0 \
  --target-x-delta 0.10 \
  --seed 0 \
  --no-plot \
  --output-dir outputs/ur5e_mujoco_torque_transport/fixed_gain_test_origin

echo "=== Test 2: height_alpha=0.5 pose, dx=0.10m, move 1s + hold 2s ==="
# Interpolated start q at alpha=0.5 (mid-height)
python tools/ur5e_mujoco_torque_experiments.py \
  --mode controller-rollout \
  --controller-kind impedance \
  --config config/ur5e_mujoco_torque_osc_tuned.yaml \
  --trajectory-profile min_jerk_move_hold \
  --move-duration 1.0 \
  --duration 3.0 \
  --target-x-delta 0.10 \
  --start-q-rad 0.0 -0.835098168 -1.2 -0.985398163 0.0 0.0 \
  --seed 0 \
  --no-plot \
  --output-dir outputs/ur5e_mujoco_torque_transport/fixed_gain_test_height0.5

echo "=== Summaries ==="
python - <<'PY'
import json
from pathlib import Path

repo = Path("/mnt/c/Users/sandr/Downloads/real-cartpole-control-using-ur5-direct-torque")
for label, sub in [
    ("origin", "fixed_gain_test_origin"),
    ("height0.5", "fixed_gain_test_height0.5"),
]:
    root = repo / "outputs/ur5e_mujoco_torque_transport" / sub
    summary_paths = list(root.rglob("summary.json"))
    if not summary_paths:
        print(f"{label}: NO SUMMARY")
        continue
    s = json.loads(max(summary_paths, key=lambda p: p.stat().st_mtime).read_text())
    keys = [
        "termination_reason",
        "safety_pass",
        "valid_move_and_hold",
        "achieved_x_delta_m",
        "final_x_error_m",
        "max_abs_orientation_error_rad",
        "max_abs_qd_radps",
        "move_hold_quality_score",
    ]
    print(f"\n{label}:")
    for k in keys:
        print(f"  {k}: {s.get(k)}")
PY
