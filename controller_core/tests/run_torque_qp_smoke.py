#!/usr/bin/env python3
from controller_core.tests.test_torque_task_qp import (
    test_box_qp_respects_bounds,
    test_qp_controller_returns_finite_torque,
    test_velocity_bounds_tighten_torque_box,
)

if __name__ == "__main__":
    test_box_qp_respects_bounds()
    test_velocity_bounds_tighten_torque_box()
    test_qp_controller_returns_finite_torque()
    print("torque_task_qp smoke tests passed")
