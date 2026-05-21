from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription(
        [
            Node(
                package="bookarm_control_py",
                executable="grasp_book",
                output="screen",
            )
        ]
    )
