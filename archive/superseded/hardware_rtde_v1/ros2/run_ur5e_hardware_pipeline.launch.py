"""Launch the staged UR5e hardware pipeline node."""

from __future__ import annotations

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    args = [
        DeclareLaunchArgument("robot_ip", default_value="", description="UR5e robot IP address."),
        DeclareLaunchArgument(
            "frequency_hz",
            default_value="500.0",
            description="Target RTDE / actuator loop frequency in Hz.",
        ),
        DeclareLaunchArgument(
            "stage",
            default_value="connection_smoke",
            description="Pipeline stage: connection_smoke | basic_servoj_hold | basic_servoj_tiny | direct_torque_probe.",
        ),
        DeclareLaunchArgument(
            "motion_opt_in",
            default_value="false",
            description="Explicit opt-in for servoJ motion stages.",
        ),
        DeclareLaunchArgument(
            "allow_nonzero_direct_torque",
            default_value="false",
            description="Explicit opt-in for nonzero direct torque requests.",
        ),
        DeclareLaunchArgument(
            "direct_torque_zero_only",
            default_value="true",
            description="Keep direct torque probe zero-only by default.",
        ),
        DeclareLaunchArgument("joint_index", default_value="0", description="Joint index for tiny servoJ."),
        DeclareLaunchArgument("amplitude_rad", default_value="0.005", description="Tiny servoJ amplitude in radians."),
        DeclareLaunchArgument(
            "max_amplitude_rad",
            default_value="0.01",
            description="Maximum allowed tiny servoJ amplitude in radians.",
        ),
        DeclareLaunchArgument("gain", default_value="100.0", description="servoJ gain."),
        DeclareLaunchArgument(
            "lookahead_time",
            default_value="0.1",
            description="servoJ lookahead time in seconds.",
        ),
        DeclareLaunchArgument("velocity", default_value="0.05", description="servoJ velocity cap."),
        DeclareLaunchArgument("acceleration", default_value="0.05", description="servoJ acceleration cap."),
        DeclareLaunchArgument(
            "output_path",
            default_value="outputs/control_runs/ur5e_hardware_pipeline_summary.json",
            description="JSON summary path written on shutdown.",
        ),
        DeclareLaunchArgument(
            "publish_status_hz",
            default_value="5.0",
            description="Status publication frequency in Hz.",
        ),
        DeclareLaunchArgument(
            "direct_torque_topic",
            default_value="/ur5e/direct_torque_command",
            description="Topic used for direct torque command input.",
        ),
    ]

    node = Node(
        package="ur5_x_axis_controller_ros",
        executable="ur5e_hardware_pipeline_node",
        name="ur5e_hardware_pipeline",
        output="screen",
        parameters=[
            {
                "robot_ip": LaunchConfiguration("robot_ip"),
                "frequency_hz": LaunchConfiguration("frequency_hz"),
                "stage": LaunchConfiguration("stage"),
                "motion_opt_in": LaunchConfiguration("motion_opt_in"),
                "allow_nonzero_direct_torque": LaunchConfiguration("allow_nonzero_direct_torque"),
                "direct_torque_zero_only": LaunchConfiguration("direct_torque_zero_only"),
                "joint_index": LaunchConfiguration("joint_index"),
                "amplitude_rad": LaunchConfiguration("amplitude_rad"),
                "max_amplitude_rad": LaunchConfiguration("max_amplitude_rad"),
                "gain": LaunchConfiguration("gain"),
                "lookahead_time": LaunchConfiguration("lookahead_time"),
                "velocity": LaunchConfiguration("velocity"),
                "acceleration": LaunchConfiguration("acceleration"),
                "output_path": LaunchConfiguration("output_path"),
                "publish_status_hz": LaunchConfiguration("publish_status_hz"),
                "direct_torque_topic": LaunchConfiguration("direct_torque_topic"),
            }
        ],
    )

    return LaunchDescription([*args, node])

