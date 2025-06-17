from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import LaunchConfiguration
from launch.conditions import IfCondition
from launch_ros.actions import Node, SetParameter
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    rviz_config = os.path.join(get_package_share_directory('dataproc'), 'config', 'rviz', 'helhest.rviz')

    ld = LaunchDescription([
        # Declare launch arguments
        DeclareLaunchArgument('rviz', default_value='false', description='Launch RVIZ2 or not'),
        DeclareLaunchArgument('bag_path', description='Full path to the ROS 2 bag file to play'),

        SetParameter(name='use_sim_time', value=True),

        # Start playing the bag file
        ExecuteProcess(
            cmd=['ros2', 'bag', 'play', LaunchConfiguration('bag_path'), '--clock'],
            output='screen'
        ),

        # Publish a static transform between base_link and camera_front
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='static_tf_publisher',
            arguments=['0.170', '0.0', '0.0', '0.0', '-0.06028860567433052', '0.0', '0.9981811451441096', 'base_link', 'camera_front'],
        ),

        # RVIZ2 visualization
        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            arguments=['-d', rviz_config],
            condition=IfCondition(LaunchConfiguration('rviz'))
        )
    ])

    return ld
