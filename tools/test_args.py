#!/usr/bin/env python3
import sys
sys.argv = ['test', '--cartesian-z-kp', '200', '--cartesian-z-kd', '40', '--cartesian-z-ki', '50', '--gravity-scale', '1.0']
sys.path.insert(0, '.')
from simulation.run_coppeliasim_x_axis_headless import parse_args
args = parse_args()
print(f"kp={args.cartesian_z_kp} kd={args.cartesian_z_kd} ki={args.cartesian_z_ki} gs={args.gravity_scale}")
