from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    share = Path(get_package_share_directory("real_cartpole_control"))
    config_file = share / "config" / "origin_hold.yaml"

    publish_commands = LaunchConfiguration("publish_commands")
    trajectory_topic = LaunchConfiguration("trajectory_topic")
    joint_state_topic = LaunchConfiguration("joint_state_topic")

    controller_node = Node(
        package="real_cartpole_control",
        executable="origin_hold_controller",
        output="screen",
        parameters=[
            str(config_file),
            {
                "publish_commands": publish_commands,
                "trajectory_topic": trajectory_topic,
                "joint_state_topic": joint_state_topic,
            },
        ],
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument("publish_commands", default_value="false"),
            DeclareLaunchArgument(
                "trajectory_topic",
                default_value="/scaled_joint_trajectory_controller/joint_trajectory",
            ),
            DeclareLaunchArgument("joint_state_topic", default_value="/joint_states"),
            controller_node,
        ]
    )
