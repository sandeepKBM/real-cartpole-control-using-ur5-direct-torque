#!/usr/bin/env python3
import json
s = json.load(open("outputs/control_runs/coppeliasim_ur5_wsl_y_mpc_transport_summary.json"))
for k in ["success", "safety_stop_reason", "transport_arm_reason",
           "transport_start_time_s", "max_abs_fixed_axis_1_drift_m",
           "max_abs_fixed_axis_2_drift_m", "max_orientation_error_deg",
           "max_abs_transport_axis_drift_m", "transport_axis_net_displacement_m",
           "peak_joint_speed_rad_s", "duration_s", "failure_reasons"]:
    print(f"{k}: {s.get(k)}")
