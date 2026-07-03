"""Launch the CoppeliaSim bridge plus the selected UR5 controller node."""

from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg_share = get_package_share_directory("ur5_x_axis_controller_ros")
    default_cfg = str(Path(pkg_share) / "config" / "controller.yaml")

    cfg_arg = DeclareLaunchArgument(
        "config_path",
        default_value=default_cfg,
        description="Path to controller.yaml (nested controller/coppeliasim/safety/topics).",
    )
    run_bridge_arg = DeclareLaunchArgument(
        "run_bridge",
        default_value="true",
        description="If true, start coppeliasim_bridge_node.",
    )

    bridge = Node(
        package="ur5_x_axis_controller_ros",
        executable="coppeliasim_bridge_node",
        name="coppeliasim_ur5_bridge",
        output="screen",
        parameters=[{"config_path": LaunchConfiguration("config_path")}],
        condition=IfCondition(LaunchConfiguration("run_bridge")),
    )

    controller = Node(
        package="ur5_x_axis_controller_ros",
        executable="controller_node",
        name="ur5_x_axis_controller",
        output="screen",
        parameters=[{"config_path": LaunchConfiguration("config_path")}],
    )

    return LaunchDescription([cfg_arg, run_bridge_arg, bridge, controller])
