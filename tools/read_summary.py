#!/usr/bin/env python3
import json, sys
path = sys.argv[1] if len(sys.argv) > 1 else "outputs/control_runs/coppeliasim_ur5_wsl_y_mpc_transport_summary.json"
s = json.load(open(path))
keys = [
    "success", "failure_reasons", "safety_stop_reason",
    "transport_started", "transport_arm_reason",
    "transport_axis_net_displacement_m",
    "max_abs_fixed_axis_1_drift_m", "max_abs_fixed_axis_2_drift_m",
    "max_orientation_error_deg",
    "sim_dt_s", "elapsed_sim_s", "total_steps",
    "initial_ee_world_m", "final_ee_world_m",
    "q_start_rad", "q_final_rad",
    "accel_torque_policy", "gravity_compensation_source",
    "gravity_compensation_scale",
]
for k in keys:
    if k in s:
        print(f"{k} = {s[k]}")
